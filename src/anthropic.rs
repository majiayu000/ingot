//! Anthropic Messages API (`POST /v1/messages`) on the same engine worker as
//! the OpenAI endpoints. Tools are rendered with the model's own tool template.
use crate::openai::{ApiError, Prompt, Request, bad};
use crate::sampling::Sampling;
use crate::server::{Event, Events, Shared, respond, write_head};
use crate::thinking::Thinking;
use crate::tokenizer::{Message, Role};
use crate::tools::ToolCall;
use serde_json::{Map, Value, json};
use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;

/// Accepted fields. Without `temperature` decoding is greedy (Anthropic's
/// default is 1.0), which keeps tool calls deterministic and allows MTP.
const FIELDS: [&str; 14] = [
    "model",
    "thinking",
    "max_tokens",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "metadata",
    "output_format",
];
/// Claude Code fields accepted and ignored: server-side context editing, effort
/// and safeguards do not apply. Only `output_config.format` (structured output)
/// is used.
const IGNORED: [&str; 3] = ["context_management", "output_config", "safeguards"];

fn error_body(e: &ApiError) -> Value {
    let kind = match e.status {
        400 | 411 | 413 | 431 => "invalid_request_error",
        404 => "not_found_error",
        _ => "api_error",
    };
    json!({"type": "error", "error": {"type": kind, "message": e.message}})
}

/// Text of a string or an array of text blocks.
fn text(value: &Value, what: &str) -> Result<String, ApiError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(
                |b| match (b.get("type").and_then(Value::as_str), b.get("text")) {
                    (Some("text"), Some(Value::String(t))) => Ok(t.as_str()),
                    _ => Err(bad(format!("{what} supports only text blocks"), None)),
                },
            )
            .collect(),
        _ => Err(bad(format!("{what} must be a string or text blocks"), None)),
    }
}

fn parse_messages(
    items: &[Value],
    images: &mut Vec<crate::media::ImageSource>,
) -> Result<Vec<Message>, ApiError> {
    let mut messages = vec![];
    for (i, item) in items.iter().enumerate() {
        let role = item.get("role").and_then(Value::as_str);
        let content = item
            .get("content")
            .ok_or_else(|| bad(format!("messages[{i}].content is required"), None))?;
        let blocks = match content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(blocks) => blocks.clone(),
            _ => {
                return Err(bad(
                    format!("messages[{i}].content must be a string or blocks"),
                    None,
                ));
            }
        };
        let mut message = Message::new(
            match role {
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                // The template has system text only at the start; later system
                // messages (Claude Code context) become user turns.
                Some("system") => Role::User,
                _ => {
                    return Err(bad(
                        format!("messages[{i}].role must be user, assistant or system"),
                        None,
                    ));
                }
            },
            "",
        );
        for block in &blocks {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            match (message.role, kind) {
                (_, "text") => message.content += &text(&json!([block]), "text block")?,
                (Role::User, "image") => {
                    let what = format!("messages[{i}]");
                    message.content += &crate::media::anthropic_image(block, &what, images)?;
                }
                (Role::User, "tool_result") => {
                    let content = block.get("content").map_or(Ok(String::new()), |c| {
                        text(c, &format!("messages[{i}] tool_result content"))
                    })?;
                    messages.push(Message::new(Role::Tool, content));
                }
                (Role::Assistant, "tool_use") => {
                    let name = block.get("name").and_then(Value::as_str);
                    let input = block.get("input").and_then(Value::as_object);
                    let (Some(name), Some(input)) = (name, input) else {
                        return Err(bad(
                            format!("messages[{i}] tool_use needs a name and object input"),
                            None,
                        ));
                    };
                    message.tool_calls.push(ToolCall {
                        name: name.to_owned(),
                        arguments: input.clone(),
                    });
                }
                // Earlier reasoning goes back into the template's think block.
                (Role::Assistant, "thinking") => {
                    if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                        message.reasoning.get_or_insert_default().push_str(t);
                    }
                }
                (Role::Assistant, "redacted_thinking") => {}
                _ => {
                    return Err(bad(
                        format!("messages[{i}]: content block type '{kind}' is not supported"),
                        None,
                    ));
                }
            }
        }
        if message.role == Role::Assistant || !message.content.is_empty() {
            messages.push(message);
        }
    }
    Ok(messages)
}

/// Anthropic tools as the OpenAI function objects the chat template expects.
fn parse_tools(tools: Option<&Value>) -> Result<Vec<Value>, ApiError> {
    let Some(tools) = tools else {
        return Ok(vec![]);
    };
    let items = tools
        .as_array()
        .ok_or_else(|| bad("tools must be an array", None))?;
    items
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let name = t.get("name").and_then(Value::as_str);
            let schema = t.get("input_schema");
            match (t.get("type").and_then(Value::as_str), name, schema) {
                (None | Some("custom"), Some(name), Some(schema)) => {
                    let mut function = Map::new();
                    function.insert("name".into(), json!(name));
                    if let Some(d) = t.get("description") {
                        function.insert("description".into(), d.clone());
                    }
                    function.insert("parameters".into(), schema.clone());
                    Ok(json!({"type": "function", "function": function}))
                }
                _ => Err(bad(
                    format!(
                        "tools[{i}]: only custom tools with name and input_schema are supported"
                    ),
                    None,
                )),
            }
        })
        .collect()
}

/// `thinking`: disabled is off; enabled maps `budget_tokens` to a level
/// (under 4096 low, under 16384 medium, else high); adaptive is medium.
fn parse_thinking(value: Option<&Value>) -> Result<Option<Thinking>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(Some(Thinking::Off)),
        Some("adaptive") => Ok(Some(Thinking::Medium)),
        Some("enabled") => match value.get("budget_tokens").and_then(Value::as_u64) {
            Some(n) if n > 0 && n < 4096 => Ok(Some(Thinking::Low)),
            Some(n) if n > 0 && n < 16384 => Ok(Some(Thinking::Medium)),
            Some(n) if n > 0 => Ok(Some(Thinking::High)),
            _ => Err(bad(
                "thinking.budget_tokens must be a positive integer",
                None,
            )),
        },
        _ => Err(bad(
            "thinking.type must be enabled, disabled or adaptive",
            None,
        )),
    }
}

pub(crate) fn parse_request(body: &[u8]) -> Result<(Request, String), ApiError> {
    let value: Value =
        serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}"), None))?;
    let Value::Object(fields) = value else {
        return Err(bad("request body must be a JSON object", None));
    };
    let fields: Map<String, Value> = fields.into_iter().filter(|(_, v)| !v.is_null()).collect();
    if let Some(key) = fields
        .keys()
        .find(|k| !FIELDS.contains(&k.as_str()) && !IGNORED.contains(&k.as_str()))
    {
        return Err(bad(format!("unsupported parameter: {key}"), None));
    }
    let model = fields
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let max_tokens = fields
        .get("max_tokens")
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("max_tokens must be a positive integer", None))?;
    let mut messages = match fields.get("system") {
        Some(system) => vec![Message::new(Role::System, text(system, "system")?)],
        None => vec![],
    };
    let items = fields
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| bad("messages must be a non-empty array", None))?;
    let mut images = vec![];
    messages.extend(parse_messages(items, &mut images)?);
    let mut tools = parse_tools(fields.get("tools"))?;
    let mut parallel_tool_calls = true;
    if let Some(choice) = fields.get("tool_choice") {
        match choice.get("type").and_then(Value::as_str) {
            Some("auto") => {}
            Some("none") => tools.clear(),
            _ => return Err(bad("tool_choice supports only auto and none", None)),
        }
        parallel_tool_calls = !choice
            .get("disable_parallel_tool_use")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    }
    let number = |key: &str| fields.get(key).and_then(Value::as_f64);
    let d = Sampling::default();
    let sampling = Sampling {
        temperature: number("temperature").map_or(d.temperature, |v| v as f32),
        top_p: number("top_p").map_or(d.top_p, |v| v as f32),
        top_k: fields
            .get("top_k")
            .and_then(Value::as_u64)
            .map_or(d.top_k, |v| v as usize),
        seed: crate::sampling::random_seed(),
    };
    sampling.validate().map_err(|e| bad(e.to_string(), None))?;
    let stop = match fields.get("stop_sequences") {
        None => vec![],
        Some(v) => v
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .map(|s| s.as_str().filter(|s| !s.is_empty()).map(str::to_owned))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| bad("stop_sequences must be non-empty strings", None))?,
    };
    let request = Request {
        prompt: Prompt::Chat {
            messages,
            developer: false,
            images,
        },
        max_tokens: Some(max_tokens as usize),
        sampling,
        stop,
        stream: fields
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        include_usage: false,
        tools,
        parallel_tool_calls,
        clamp_max_tokens: true,
        format: crate::structured::anthropic(&fields)?,
        thinking: parse_thinking(fields.get("thinking"))?,
    };
    Ok((request, model))
}

fn sse(stream: &mut TcpStream, event: &str, data: Value) -> std::io::Result<()> {
    stream.write_all(format!("event: {event}\ndata: {data}\n\n").as_bytes())
}

/// Content blocks and stop fields of a finished message.
fn finish_fields(finish: &str, stop_sequence: Option<String>) -> (&'static str, Value) {
    let reason = match (finish, &stop_sequence) {
        ("tool_calls", _) => "tool_use",
        ("length", _) => "max_tokens",
        (_, Some(_)) => "stop_sequence",
        _ => "end_turn",
    };
    (reason, json!(stop_sequence))
}

fn tool_block(id: &str, i: usize, call: &ToolCall) -> Value {
    json!({"type": "tool_use", "id": format!("toolu_{id}_{i}"), "name": call.name,
        "input": Value::Object(call.arguments.clone())})
}

pub(crate) fn handle(mut stream: TcpStream, body: &[u8], shared: &Shared) {
    let fail = |stream: &mut TcpStream, e: ApiError| respond(stream, e.status, &error_body(&e));
    let (request, model) = match parse_request(body) {
        Ok(v) => v,
        Err(e) => return fail(&mut stream, e),
    };
    let streaming = request.stream;
    let served = match shared.served.anthropic(&model) {
        Ok(id) => id,
        Err(e) => return fail(&mut stream, e),
    };
    let mut events = match Events::submit(shared, &served, true, request) {
        Ok(events) => events,
        Err(e) => return fail(&mut stream, e),
    };
    let id = format!("msg_{}", shared.next_id.fetch_add(1, Ordering::Relaxed));
    // Streaming: index and kind ("text" or "thinking") of the open block.
    let mut open: Option<(usize, bool)> = None;
    let mut blocks = 0;
    // Headers are written once a streaming job starts; later errors are SSE events.
    let mut started = false;
    let (mut content, mut thinking) = (String::new(), String::new());
    while let Some(event) = events.next(&stream) {
        let written = match event {
            Event::Error(e) if started => sse(&mut stream, "error", error_body(&e)),
            Event::Error(e) => return fail(&mut stream, e),
            Event::Started { prompt_tokens } if streaming => {
                started = true;
                let message = json!({"id": id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}});
                write_head(&mut stream, 200, "text/event-stream", None).and_then(|_| {
                    sse(
                        &mut stream,
                        "message_start",
                        json!({"type": "message_start", "message": message}),
                    )
                })
            }
            Event::Started { .. } => Ok(()),
            Event::Delta(d) if streaming => (|| {
                if open.is_some_and(|(_, r)| r != d.reasoning) {
                    let (index, _) = open.take().unwrap();
                    sse(
                        &mut stream,
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": index}),
                    )?;
                }
                let (block, delta) = if d.reasoning {
                    (
                        json!({"type": "thinking", "thinking": "", "signature": ""}),
                        json!({"type": "thinking_delta", "thinking": d.text}),
                    )
                } else {
                    (
                        json!({"type": "text", "text": ""}),
                        json!({"type": "text_delta", "text": d.text}),
                    )
                };
                let index = match open {
                    Some((index, _)) => index,
                    None => {
                        let index = blocks;
                        blocks += 1;
                        open = Some((index, d.reasoning));
                        sse(
                            &mut stream,
                            "content_block_start",
                            json!({"type": "content_block_start",
                            "index": index, "content_block": block}),
                        )?;
                        index
                    }
                };
                sse(
                    &mut stream,
                    "content_block_delta",
                    json!({"type": "content_block_delta",
                    "index": index, "delta": delta}),
                )
            })(),
            Event::Delta(d) => {
                if d.reasoning {
                    thinking += &d.text;
                } else {
                    content += &d.text;
                }
                Ok(())
            }
            Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
                tool_calls,
                stop_sequence,
            } => {
                let (reason, stop_sequence) = finish_fields(finish, stop_sequence);
                if streaming {
                    let _ = (|| {
                        if let Some((index, _)) = open.take() {
                            sse(
                                &mut stream,
                                "content_block_stop",
                                json!({"type": "content_block_stop", "index": index}),
                            )?;
                        }
                        for (i, call) in tool_calls.iter().enumerate() {
                            let index = blocks + i;
                            let mut block = tool_block(&id, i, call);
                            let input = std::mem::replace(&mut block["input"], json!({}));
                            sse(
                                &mut stream,
                                "content_block_start",
                                json!({"type": "content_block_start",
                                "index": index, "content_block": block}),
                            )?;
                            sse(
                                &mut stream,
                                "content_block_delta",
                                json!({"type": "content_block_delta",
                                "index": index, "delta": {"type": "input_json_delta", "partial_json": input.to_string()}}),
                            )?;
                            sse(
                                &mut stream,
                                "content_block_stop",
                                json!({"type": "content_block_stop", "index": index}),
                            )?;
                        }
                        sse(
                            &mut stream,
                            "message_delta",
                            json!({"type": "message_delta",
                            "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                            "usage": {"output_tokens": completion_tokens}}),
                        )?;
                        sse(&mut stream, "message_stop", json!({"type": "message_stop"}))
                    })();
                } else {
                    let mut blocks = vec![];
                    if !thinking.is_empty() {
                        blocks.push(
                            json!({"type": "thinking", "thinking": thinking, "signature": ""}),
                        );
                    }
                    if !content.trim().is_empty() || tool_calls.is_empty() {
                        blocks.push(json!({"type": "text", "text": content}));
                    }
                    blocks.extend(
                        tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, c)| tool_block(&id, i, c)),
                    );
                    let message = json!({"id": id, "type": "message", "role": "assistant", "model": model,
                        "content": blocks, "stop_reason": reason, "stop_sequence": stop_sequence,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}});
                    respond(&mut stream, 200, &message);
                }
                return;
            }
        };
        if written.is_err() {
            events.cancel.cancel();
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converts_messages_tools_and_options() {
        let (r, model) = parse_request(
            br#"{"model":"claude-x","max_tokens":64000,"stream":true,"metadata":{"user_id":"u"},
            "thinking":{"type":"enabled","budget_tokens":10},
            "system":[{"type":"text","text":"Sys","cache_control":{"type":"ephemeral"}}],
            "tools":[{"name":"Read","description":"read a file","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"auto","disable_parallel_tool_use":true},
            "messages":[{"role":"user","content":"hi"},
              {"role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"ok"},
                {"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"a"}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1",
                "content":[{"type":"text","text":"data"}]},{"type":"text","text":"next"}]}]}"#,
        )
        .unwrap();
        assert_eq!(model, "claude-x");
        assert!(r.stream && r.clamp_max_tokens && !r.parallel_tool_calls);
        assert!(r.sampling.is_greedy());
        assert_eq!(r.thinking, Some(Thinking::Low));
        assert_eq!(
            r.tools[0],
            json!({"type": "function", "function": {"name": "Read", "description": "read a file",
                "parameters": {"type": "object"}}})
        );
        let Prompt::Chat { messages, .. } = &r.prompt else {
            panic!("chat prompt")
        };
        let roles: Vec<_> = messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::User
            ]
        );
        assert_eq!(messages[0].content, "Sys");
        assert_eq!(messages[2].content, "ok");
        assert_eq!(messages[2].reasoning.as_deref(), Some("t"));
        assert_eq!(messages[2].tool_calls[0].arguments["path"], "a");
        assert_eq!(
            (messages[3].content.as_str(), messages[4].content.as_str()),
            ("data", "next")
        );
        for body in [
            r#"{"messages":[{"role":"user","content":"x"}]}"#,
            r#"{"max_tokens":1,"messages":[]}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":[{"type":"image"}]}]}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"any"}}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search_20250305","name":"w"}]}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"service_tier":"auto"}"#,
            r#"{"max_tokens":1,"messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"on"}}"#,
            r#"{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}}"#,
        ] {
            let e = parse_request(body.as_bytes()).unwrap_err();
            assert_eq!(
                error_body(&e)["error"]["type"],
                "invalid_request_error",
                "{body}"
            );
        }
    }
    #[test]
    fn maps_thinking_budgets_to_levels() {
        let level = |thinking: &str| {
            let body = format!(
                r#"{{"max_tokens":1,"messages":[{{"role":"user","content":"x"}}],"thinking":{thinking}}}"#
            );
            parse_request(body.as_bytes()).unwrap().0.thinking
        };
        assert_eq!(level(r#"{"type":"disabled"}"#), Some(Thinking::Off));
        assert_eq!(level(r#"{"type":"adaptive"}"#), Some(Thinking::Medium));
        assert_eq!(
            level(r#"{"type":"enabled","budget_tokens":1024}"#),
            Some(Thinking::Low)
        );
        assert_eq!(
            level(r#"{"type":"enabled","budget_tokens":8000}"#),
            Some(Thinking::Medium)
        );
        assert_eq!(
            level(r#"{"type":"enabled","budget_tokens":31999}"#),
            Some(Thinking::High)
        );
        assert_eq!(level("null"), None);
    }
    #[test]
    fn maps_stop_reasons() {
        assert_eq!(finish_fields("tool_calls", None).0, "tool_use");
        assert_eq!(finish_fields("length", None).0, "max_tokens");
        assert_eq!(
            finish_fields("stop", Some("X".into())),
            ("stop_sequence", json!("X"))
        );
        assert_eq!(finish_fields("stop", None), ("end_turn", Value::Null));
    }
}
