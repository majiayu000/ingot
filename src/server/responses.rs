//! OpenAI Responses API (`POST /v1/responses`), stateless: each request carries
//! the whole conversation. Items become chat messages for the same worker.
use super::{Event, Events, Shared, respond, write_head};
use crate::openai::{ApiError, parse_request};
use crate::tools::ToolCall;
use serde_json::{Value, json};
use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

fn text_of(content: &Value) -> Value {
    let Some(parts) = content.as_array() else {
        return content.clone();
    };
    let parts: Vec<_> = parts
        .iter()
        .filter_map(|part| match part["type"].as_str()? {
            "input_text" | "output_text" => Some(json!({"type": "text", "text": part["text"]})),
            "input_image" => {
                Some(json!({"type": "image_url", "image_url": {"url": part["image_url"]}}))
            }
            _ => None,
        })
        .collect();
    json!(parts)
}

/// Conversation items as chat messages. Calls in a row share one message.
fn messages(input: &Value) -> Vec<Value> {
    let Some(items) = input.as_array() else {
        return vec![json!({"role": "user", "content": input})];
    };
    let mut list: Vec<Value> = Vec::new();
    for item in items {
        match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let role = match item["role"].as_str() {
                    Some("developer") => "system",
                    Some(role) => role,
                    None => "user",
                };
                list.push(json!({"role": role, "content": text_of(&item["content"])}));
            }
            "function_call" => {
                let call = json!({"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}});
                match list.last_mut() {
                    Some(last) if last["tool_calls"].is_array() => {
                        if let Some(calls) = last["tool_calls"].as_array_mut() {
                            calls.push(call);
                        }
                    }
                    _ => list.push(json!({"role": "assistant", "content": "",
                        "tool_calls": [call]})),
                }
            }
            "function_call_output" => list.push(json!({"role": "tool",
                "tool_call_id": item["call_id"], "content": item["output"]})),
            _ => {}
        }
    }
    list
}

/// The chat completions body for a Responses request.
fn translate(body: &Value) -> Value {
    let mut list = messages(&body["input"]);
    if let Some(instructions) = body["instructions"].as_str() {
        list.insert(0, json!({"role": "system", "content": instructions}));
    }
    let mut out = json!({"messages": list});
    if let Some(model) = body.get("model") {
        out["model"] = model.clone();
    }
    for key in ["temperature", "top_p", "tool_choice", "parallel_tool_calls"] {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            out[key] = value.clone();
        }
    }
    if let Some(limit) = body.get("max_output_tokens").filter(|v| !v.is_null()) {
        out["max_tokens"] = limit.clone();
    }
    if let Some(effort) = body["reasoning"]["effort"].as_str() {
        out["reasoning_effort"] = json!(effort);
    }
    if let Some(tools) = body["tools"].as_array() {
        // Hosted tools such as web search have nothing to run them here.
        let tools: Vec<_> = tools
            .iter()
            .filter(|tool| tool["type"] == "function")
            .map(|tool| {
                json!({"type": "function", "function": {"name": tool["name"],
                    "description": tool["description"], "parameters": tool["parameters"]}})
            })
            .collect();
        if !tools.is_empty() {
            out["tools"] = json!(tools);
        }
    }
    out
}

fn function_items(calls: &[ToolCall], id: &str) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            json!({"type": "function_call", "id": format!("fc_{id}_{index}"),
                "call_id": format!("call_{id}_{index}"), "name": call.name, "status": "completed",
                "arguments": Value::Object(call.arguments.clone()).to_string()})
        })
        .collect()
}

fn send(stream: &mut TcpStream, kind: &str, mut value: Value) -> std::io::Result<()> {
    value["type"] = json!(kind);
    stream.write_all(format!("event: {kind}\ndata: {value}\n\n").as_bytes())
}

pub(super) fn handle(mut stream: TcpStream, body: &[u8], shared: &Shared) {
    let fail = |stream: &mut TcpStream, e: ApiError| respond(stream, e.status, &e.body());
    let body: Value = match serde_json::from_slice(body) {
        Ok(body) => body,
        Err(e) => {
            return fail(
                &mut stream,
                ApiError::new(400, format!("invalid JSON: {e}")),
            );
        }
    };
    let streaming = body["stream"].as_bool().unwrap_or(false);
    let request = translate(&body).to_string();
    let parsed = shared
        .served
        .customized(request.as_bytes())
        .and_then(|bytes| {
            let model = shared.served.openai(&bytes)?;
            Ok((parse_request(&bytes, true, &model)?, model))
        });
    let (parsed, model) = match parsed {
        Ok(v) => v,
        Err(e) => return fail(&mut stream, e),
    };
    let mut events = match Events::submit(shared, &model, true, parsed) {
        Ok(events) => events,
        Err(e) => return fail(&mut stream, e),
    };
    let n = shared.next_id.fetch_add(1, Ordering::Relaxed);
    let id = format!("resp_{n}");
    let message = format!("msg_{n}");
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let envelope = |status: &str, output: Value, usage: Value| {
        json!({"id": id, "object": "response", "created_at": created, "status": status,
            "model": model, "output": output, "usage": usage})
    };
    let mut content = String::new();
    let mut opened = false;
    while let Some(event) = events.next(&stream) {
        let written = match event {
            Event::Error(e) if opened => {
                let _ = send(&mut stream, "error", json!({"message": e.message}));
                return;
            }
            Event::Error(e) => return fail(&mut stream, e),
            Event::Started { .. } if streaming => {
                opened = true;
                write_head(&mut stream, 200, "text/event-stream", None).and_then(|_| {
                    let started = envelope("in_progress", json!([]), Value::Null);
                    send(
                        &mut stream,
                        "response.created",
                        json!({"response": started}),
                    )
                })
            }
            Event::Started { .. } => Ok(()),
            Event::Delta(d) if d.reasoning => Ok(()),
            Event::Delta(d) => {
                let first = content.is_empty();
                content += &d.text;
                if !streaming {
                    Ok(())
                } else if first {
                    let item = json!({"type": "message", "id": message, "role": "assistant",
                        "status": "in_progress", "content": []});
                    send(
                        &mut stream,
                        "response.output_item.added",
                        json!({"output_index": 0, "item": item}),
                    )
                    .and_then(|_| {
                        send(
                            &mut stream,
                            "response.content_part.added",
                            json!({"item_id": message, "output_index": 0, "content_index": 0,
                                "part": {"type": "output_text", "text": ""}}),
                        )
                    })
                    .and_then(|_| {
                        send(
                            &mut stream,
                            "response.output_text.delta",
                            json!({"item_id": message, "output_index": 0, "content_index": 0,
                                "delta": d.text}),
                        )
                    })
                } else {
                    send(
                        &mut stream,
                        "response.output_text.delta",
                        json!({"item_id": message, "output_index": 0, "content_index": 0,
                            "delta": d.text}),
                    )
                }
            }
            Event::Done {
                prompt_tokens,
                completion_tokens,
                tool_calls,
                ..
            } => {
                let usage = json!({"input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens});
                let mut output = Vec::new();
                if !content.is_empty() {
                    output.push(
                        json!({"type": "message", "id": message, "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                            "text": content, "annotations": []}]}),
                    );
                }
                output.extend(function_items(&tool_calls, &n.to_string()));
                if !streaming {
                    return respond(
                        &mut stream,
                        200,
                        &envelope("completed", json!(output), usage),
                    );
                }
                let start = usize::from(!content.is_empty());
                let mut result = Ok(());
                if start == 1 {
                    result = send(
                        &mut stream,
                        "response.output_text.done",
                        json!({"item_id": message, "output_index": 0, "content_index": 0,
                            "text": content}),
                    );
                }
                for (index, item) in output.iter().enumerate() {
                    if index >= start {
                        result = result.and_then(|_| {
                            send(
                                &mut stream,
                                "response.output_item.added",
                                json!({"output_index": index, "item": item}),
                            )
                        });
                    }
                    result = result.and_then(|_| {
                        send(
                            &mut stream,
                            "response.output_item.done",
                            json!({"output_index": index, "item": item}),
                        )
                    });
                }
                let done = envelope("completed", json!(output), usage);
                let _ = result.and_then(|_| {
                    send(&mut stream, "response.completed", json!({"response": done}))
                });
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
    fn items_become_chat_messages() {
        let body = json!({"model": "m", "instructions": "be brief", "max_output_tokens": 9,
            "reasoning": {"effort": "low"}, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "1"},
            {"type": "function_call_output", "call_id": "b", "output": "2"},
            {"type": "reasoning", "summary": []},
        ], "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}},
            {"type": "web_search"}]});
        let out = translate(&body);
        let list = out["messages"].as_array().unwrap();
        assert_eq!(list[0]["role"], "system");
        assert_eq!(list[1]["content"][0]["text"], "hi");
        assert_eq!(list[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(list[3]["tool_call_id"], "a");
        assert_eq!(out["max_tokens"], 9);
        assert_eq!(out["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            translate(&json!({"input": "hi"}))["messages"][0]["content"],
            "hi"
        );
    }
}
