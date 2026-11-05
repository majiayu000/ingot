//! OpenAI request validation and generated-text streaming, independent of HTTP.
use crate::sampling::{Sampling, random_seed};
use crate::thinking::Thinking;
use crate::tokenizer::{Message, Role};
use crate::tools::{self, ToolCall, holdback};
use serde_json::{Map, Value, json};
use std::time::Duration;

const COMPLETIONS_DEFAULT_MAX_TOKENS: usize = 16;
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

#[derive(Debug, PartialEq)]
pub(crate) struct ApiError {
    pub(crate) status: u16,
    pub(crate) message: String,
    pub(crate) param: Option<&'static str>,
}
pub(crate) fn bad(message: impl Into<String>, param: Option<&'static str>) -> ApiError {
    ApiError {
        status: 400,
        message: message.into(),
        param,
    }
}
impl ApiError {
    pub(crate) fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            param: None,
        }
    }
    pub(crate) fn body(&self) -> Value {
        let kind = match self.status {
            400 | 411 | 413 | 431 => "invalid_request_error",
            404 => "not_found_error",
            _ => "server_error",
        };
        json!({"error": {"message": self.message, "type": kind, "param": self.param, "code": null}})
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum Prompt {
    Chat {
        messages: Vec<Message>,
        /// The `developer` role was used (mapped to system; Qwen template only).
        developer: bool,
        /// Images in prompt order; message text holds a marker for each.
        images: Vec<crate::media::ImageSource>,
    },
    Text(String),
}
#[derive(Debug, PartialEq)]
pub(crate) struct Request {
    pub(crate) prompt: Prompt,
    /// None: fill the remaining context.
    pub(crate) max_tokens: Option<usize>,
    pub(crate) sampling: Sampling,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    /// OpenAI function tools rendered into the prompt; empty disables tool calls.
    pub(crate) tools: Vec<Value>,
    pub(crate) parallel_tool_calls: bool,
    /// Lower `max_tokens` to the remaining context instead of rejecting it.
    pub(crate) clamp_max_tokens: bool,
    pub(crate) format: crate::structured::Format,
    /// Requested thinking level; None is the server default, else the model's.
    pub(crate) thinking: Option<Thinking>,
    /// `num_ctx`: load the model with at least this context.
    pub(crate) context: Option<usize>,
    /// `keep_alive`: idle time before unloading; `Some(None)` keeps it loaded.
    pub(crate) keep_alive: Option<Option<Duration>>,
}

const COMMON_FIELDS: [&str; 18] = [
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "n",
    "user",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    // Not in OpenAI's API: Ollama options, also set by custom models.
    "min_p",
    "repeat_penalty",
    "num_ctx",
    "keep_alive",
];

/// Validate an OpenAI request body. Every accepted field has an effect, or is
/// accepted only at the value that matches what the engine does; anything else
/// is a 400 naming the parameter.
pub(crate) fn parse_request(body: &[u8], chat: bool, model_id: &str) -> Result<Request, ApiError> {
    let value: Value =
        serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}"), None))?;
    let Value::Object(fields) = value else {
        return Err(bad("request body must be a JSON object", None));
    };
    // Null is treated as absent; OpenAI clients often send explicit nulls.
    let fields: Map<String, Value> = fields.into_iter().filter(|(_, v)| !v.is_null()).collect();
    let extra: &[&str] = if chat {
        &[
            "messages",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "reasoning_effort",
            "chat_template_kwargs",
        ]
    } else {
        &["prompt", "echo"]
    };
    if let Some(key) = fields
        .keys()
        .find(|k| !COMMON_FIELDS.contains(&k.as_str()) && !extra.contains(&k.as_str()))
    {
        return Err(bad(format!("unsupported parameter: {key}"), None));
    }
    let number = |key: &'static str| -> Result<Option<f64>, ApiError> {
        fields
            .get(key)
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| bad(format!("{key} must be a number"), Some(key)))
            })
            .transpose()
    };
    let integer = |key: &'static str| -> Result<Option<u64>, ApiError> {
        fields
            .get(key)
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| bad(format!("{key} must be a non-negative integer"), Some(key)))
            })
            .transpose()
    };
    let boolean = |key: &'static str| -> Result<Option<bool>, ApiError> {
        fields
            .get(key)
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| bad(format!("{key} must be a boolean"), Some(key)))
            })
            .transpose()
    };
    if let Some(model) = fields.get("model") {
        let model = model
            .as_str()
            .ok_or_else(|| bad("model must be a string", Some("model")))?;
        if model != model_id {
            return Err(ApiError {
                status: 404,
                message: format!("model '{model}' not found; this server serves '{model_id}'"),
                param: Some("model"),
            });
        }
    }
    if integer("n")?.is_some_and(|n| n != 1) {
        return Err(bad("only n=1 is supported", Some("n")));
    }
    for key in ["presence_penalty", "frequency_penalty"] {
        if number(key)?.is_some_and(|v| v != 0.) {
            return Err(bad(format!("{key} is not supported (only 0)"), Some(key)));
        }
    }
    if fields
        .get("logprobs")
        .is_some_and(|v| v != &Value::Bool(false))
    {
        return Err(bad("logprobs are not supported", Some("logprobs")));
    }
    if boolean("echo")? == Some(true) {
        return Err(bad("echo is not supported", Some("echo")));
    }
    if fields.get("user").is_some_and(|v| !v.is_string()) {
        return Err(bad("user must be a string", Some("user")));
    }
    let max_tokens = match (integer("max_tokens")?, integer("max_completion_tokens")?) {
        (Some(_), Some(_)) => {
            return Err(bad(
                "give only one of max_tokens and max_completion_tokens",
                Some("max_tokens"),
            ));
        }
        (Some(0), _) | (_, Some(0)) => {
            return Err(bad("max_tokens must be at least 1", Some("max_tokens")));
        }
        (a, b) => a.or(b).map(|v| v as usize),
    };
    let max_tokens = if chat {
        max_tokens
    } else {
        Some(max_tokens.unwrap_or(COMPLETIONS_DEFAULT_MAX_TOKENS))
    };
    let d = Sampling::default();
    let sampling = Sampling {
        temperature: number("temperature")?.map_or(d.temperature, |v| v as f32),
        top_p: number("top_p")?.map_or(d.top_p, |v| v as f32),
        top_k: integer("top_k")?.map_or(d.top_k, |v| v as usize),
        min_p: number("min_p")?.map_or(d.min_p, |v| v as f32),
        repeat_penalty: number("repeat_penalty")?.map_or(d.repeat_penalty, |v| v as f32),
        seed: match fields.get("seed") {
            None => random_seed(),
            Some(v) => v
                .as_u64()
                .or_else(|| v.as_i64().map(|i| i as u64))
                .ok_or_else(|| bad("seed must be an integer", Some("seed")))?,
        },
    };
    if let Err(e) = sampling.validate() {
        let message = e.to_string();
        let param = ["top_p", "min_p", "repeat_penalty"]
            .into_iter()
            .find(|p| message.starts_with(p))
            .unwrap_or("temperature");
        return Err(bad(message, Some(param)));
    }
    let stop_error = || bad("stop must be a string or an array of strings", Some("stop"));
    let stop = match fields.get("stop") {
        None => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(stop_error)?,
        Some(_) => return Err(stop_error()),
    };
    if stop.len() > 4 || stop.iter().any(String::is_empty) {
        return Err(bad("stop accepts up to 4 non-empty strings", Some("stop")));
    }
    let stream = boolean("stream")?.unwrap_or(false);
    let p = Some("stream_options");
    let include_usage = match fields.get("stream_options") {
        None => false,
        Some(_) if !stream => return Err(bad("stream_options requires stream=true", p)),
        Some(Value::Object(o)) => {
            if let Some(key) = o.keys().find(|k| *k != "include_usage") {
                return Err(bad(format!("unsupported stream option: {key}"), p));
            }
            match o.get("include_usage") {
                None | Some(Value::Null) => false,
                Some(Value::Bool(b)) => *b,
                Some(_) => return Err(bad("include_usage must be a boolean", p)),
            }
        }
        Some(_) => return Err(bad("stream_options must be an object", p)),
    };
    let parallel_tool_calls = boolean("parallel_tool_calls")?.unwrap_or(true);
    let tools = parse_tools(fields.get("tools"), fields.get("tool_choice"))?;
    let prompt = if chat {
        parse_messages(fields.get("messages"))?
    } else {
        let p = Some("prompt");
        match fields.get("prompt") {
            Some(Value::String(s)) if !s.is_empty() => Prompt::Text(s.clone()),
            Some(Value::String(_)) => return Err(bad("prompt must not be empty", p)),
            Some(_) => return Err(bad("only a single string prompt is supported", p)),
            None => return Err(bad("prompt is required", p)),
        }
    };
    Ok(Request {
        prompt,
        max_tokens,
        sampling,
        stop,
        stream,
        include_usage,
        tools,
        parallel_tool_calls,
        clamp_max_tokens: false,
        format: crate::structured::openai(&fields)?,
        thinking: parse_thinking(&fields)?,
        context: integer("num_ctx")?.map(|n| n as usize),
        keep_alive: fields.get("keep_alive").map(keep_alive).transpose()?,
    })
}

/// `keep_alive`: seconds or a duration such as `5m`; negative keeps the model
/// loaded. The outer `Option` of the field is "not given", the inner "forever".
pub(crate) fn keep_alive(value: &Value) -> Result<Option<Duration>, ApiError> {
    let p = Some("keep_alive");
    let text = match value {
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => return Err(bad("keep_alive must be a number or a duration", p)),
    };
    if text.starts_with('-') {
        return Ok(None);
    }
    crate::server::worker::duration(&text)
        .map(Some)
        .map_err(|e| bad(e.to_string(), p))
}

/// `reasoning_effort` (none, minimal, low, medium, high, xhigh) and
/// `chat_template_kwargs.enable_thinking`; enabling without an effort is high.
fn parse_thinking(fields: &Map<String, Value>) -> Result<Option<Thinking>, ApiError> {
    let p = Some("reasoning_effort");
    let effort = match fields.get("reasoning_effort") {
        None => None,
        Some(Value::String(s)) => Some(s.parse::<Thinking>().map_err(|_| {
            bad(
                "reasoning_effort must be none, minimal, low, medium, high or xhigh",
                p,
            )
        })?),
        Some(_) => return Err(bad("reasoning_effort must be a string", p)),
    };
    let k = Some("chat_template_kwargs");
    let enable = match fields.get("chat_template_kwargs") {
        None => None,
        Some(Value::Object(o)) => {
            if let Some(key) = o.keys().find(|key| *key != "enable_thinking") {
                return Err(bad(
                    format!("chat_template_kwargs.{key} is not supported (only enable_thinking)"),
                    k,
                ));
            }
            match o.get("enable_thinking") {
                None | Some(Value::Null) => None,
                Some(Value::Bool(b)) => Some(*b),
                Some(_) => return Err(bad("enable_thinking must be a boolean", k)),
            }
        }
        Some(_) => return Err(bad("chat_template_kwargs must be an object", k)),
    };
    match (enable, effort) {
        (Some(false), Some(e)) if e != Thinking::Off => Err(bad(
            "enable_thinking=false conflicts with reasoning_effort",
            p,
        )),
        (Some(true), Some(Thinking::Off)) => Err(bad(
            "enable_thinking=true conflicts with reasoning_effort=none",
            p,
        )),
        (Some(false), _) => Ok(Some(Thinking::Off)),
        (Some(true), None) => Ok(Some(Thinking::High)),
        (_, effort) => Ok(effort),
    }
}

/// `tools` with `tool_choice` auto (default) or none; forced calls are rejected.
fn parse_tools(tools: Option<&Value>, choice: Option<&Value>) -> Result<Vec<Value>, ApiError> {
    let p = Some("tools");
    let tools = match tools {
        None => vec![],
        Some(Value::Array(items)) => items.clone(),
        Some(_) => return Err(bad("tools must be an array", p)),
    };
    for (i, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function")
            || tool
                .pointer("/function/name")
                .and_then(Value::as_str)
                .is_none()
        {
            return Err(bad(
                format!(
                    "tools[{i}] must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
                ),
                p,
            ));
        }
    }
    match choice.map(|c| c.as_str()) {
        None | Some(Some("auto")) => Ok(tools),
        Some(Some("none")) => Ok(vec![]),
        Some(_) => Err(bad(
            "tool_choice supports only \"auto\" and \"none\"",
            Some("tool_choice"),
        )),
    }
}

/// Assistant `tool_calls`; arguments may be a JSON object or a JSON string of one.
fn parse_tool_calls(value: Option<&Value>, i: usize) -> Result<Vec<ToolCall>, ApiError> {
    let error = || {
        bad(
            format!(
                "messages[{i}].tool_calls must be function calls with a name and object arguments"
            ),
            Some("messages"),
        )
    };
    let Some(value) = value else {
        return Ok(vec![]);
    };
    value
        .as_array()
        .ok_or_else(error)?
        .iter()
        .map(|call| {
            let function = call.get("function").ok_or_else(error)?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(error)?;
            let arguments = match function.get("arguments") {
                None | Some(Value::Null) => Value::Object(Default::default()),
                Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Default::default()),
                Some(Value::String(s)) => serde_json::from_str(s).map_err(|_| error())?,
                Some(v) => v.clone(),
            };
            match arguments {
                Value::Object(arguments) => Ok(ToolCall {
                    name: name.to_owned(),
                    arguments,
                }),
                _ => Err(error()),
            }
        })
        .collect()
}

fn parse_messages(value: Option<&Value>) -> Result<Prompt, ApiError> {
    let p = Some("messages");
    let items = match value {
        Some(Value::Array(items)) if !items.is_empty() => items,
        Some(Value::Array(_)) => return Err(bad("messages must not be empty", p)),
        Some(_) => return Err(bad("messages must be an array", p)),
        None => return Err(bad("messages is required", p)),
    };
    let mut developer = false;
    let mut messages = Vec::with_capacity(items.len());
    let mut images = vec![];
    for (i, item) in items.iter().enumerate() {
        let Value::Object(m) = item else {
            return Err(bad(format!("messages[{i}] must be an object"), p));
        };
        if let Some((key, _)) = m.iter().find(|(k, v)| {
            !v.is_null()
                && !matches!(
                    k.as_str(),
                    "role" | "content" | "reasoning_content" | "tool_calls" | "tool_call_id"
                )
        }) {
            return Err(bad(format!("messages[{i}].{key} is not supported"), p));
        }
        let role = match m.get("role").and_then(Value::as_str) {
            Some("system") => Role::System,
            Some("developer") => {
                developer = true;
                Role::System
            }
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            Some(other) => {
                return Err(bad(
                    format!("messages[{i}].role '{other}' is not supported"),
                    p,
                ));
            }
            None => return Err(bad(format!("messages[{i}].role is required"), p)),
        };
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                crate::media::openai_content(parts, i, role == Role::User, &mut images)?
            }
            None | Some(Value::Null) if role == Role::Assistant => String::new(),
            None | Some(Value::Null) => {
                return Err(bad(format!("messages[{i}].content is required"), p));
            }
            Some(_) => {
                return Err(bad(
                    format!("messages[{i}].content must be a string or text parts"),
                    p,
                ));
            }
        };
        let reasoning = match m.get("reasoning_content") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if role == Role::Assistant => Some(s.clone()),
            Some(_) => {
                return Err(bad(
                    format!(
                        "messages[{i}].reasoning_content must be a string on an assistant message"
                    ),
                    p,
                ));
            }
        };
        if role != Role::Assistant && m.get("tool_calls").is_some_and(|v| !v.is_null()) {
            return Err(bad(
                format!("messages[{i}].tool_calls is only valid on an assistant message"),
                p,
            ));
        }
        messages.push(Message {
            role,
            content,
            reasoning,
            tool_calls: parse_tool_calls(m.get("tool_calls").filter(|v| !v.is_null()), i)?,
        });
    }
    Ok(Prompt::Chat {
        messages,
        developer,
        images,
    })
}

#[derive(Debug, PartialEq)]
pub(crate) struct Delta {
    pub(crate) reasoning: bool,
    pub(crate) text: String,
}
/// Turns generated bytes into UTF-8 text deltas. Reasoning (a prompt that opens
/// the think block, or a reply that starts with `<think>`) is routed until
/// `</think>`; stop strings apply to the answer text. Text that may be the start
/// of a marker or stop string is held back until resolved.
pub(crate) struct TextStream {
    bytes: Vec<u8>,
    held: String,
    reasoning: bool,
    /// The reply may open its own think block (`detect_think`).
    leading: bool,
    /// Drop whitespace at the start of reasoning and of the answer after it.
    trim: bool,
    stops: Vec<String>,
    pub(crate) stopped: bool,
    /// The stop string that ended the answer.
    pub(crate) matched: Option<String>,
    /// Watch for tool calls; text from there on is captured, not emitted.
    tools: bool,
    pub(crate) tool_text: String,
    /// Answer text was emitted, so a bare JSON tool call can no longer start.
    answered: bool,
}
impl TextStream {
    pub(crate) fn new(reasoning: bool, stops: Vec<String>, tools: bool) -> Self {
        Self {
            bytes: vec![],
            held: String::new(),
            reasoning,
            leading: false,
            trim: reasoning,
            stops,
            stopped: false,
            matched: None,
            tools,
            tool_text: String::new(),
            answered: false,
        }
    }
    /// Also route reasoning when the reply itself starts with `<think>`.
    pub(crate) fn detect_think(mut self) -> Self {
        self.leading = !self.reasoning;
        self
    }
    fn emit(&mut self, out: &mut Vec<Delta>, mut text: &str) {
        if self.trim {
            text = text.trim_start();
            self.trim = text.is_empty();
        }
        self.answered |= !self.reasoning && !text.trim().is_empty();
        if !text.is_empty() {
            out.push(Delta {
                reasoning: self.reasoning,
                text: text.to_owned(),
            });
        }
    }
    fn decode(&mut self) {
        loop {
            match std::str::from_utf8(&self.bytes) {
                Ok(s) => {
                    self.held += s;
                    self.bytes.clear();
                    return;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    self.held += std::str::from_utf8(&self.bytes[..valid]).unwrap();
                    let Some(n) = e.error_len() else {
                        // Incomplete trailing sequence: wait for the next token.
                        self.bytes.drain(..valid);
                        return;
                    };
                    self.held.push(char::REPLACEMENT_CHARACTER);
                    self.bytes.drain(..valid + n);
                }
            }
        }
    }
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<Delta> {
        let mut out = vec![];
        if self.stopped {
            return out;
        }
        self.bytes.extend_from_slice(bytes);
        self.decode();
        let held = std::mem::take(&mut self.held);
        if !self.tool_text.is_empty() {
            self.tool_text += &held;
            return out;
        }
        let mut rest = held.as_str();
        if self.leading {
            let start = rest.trim_start();
            if let Some(after) = start.strip_prefix(THINK_START) {
                (self.leading, self.reasoning, self.trim) = (false, true, true);
                rest = after;
            } else if THINK_START.starts_with(start) {
                self.held = held;
                return out;
            } else {
                self.leading = false;
            }
        }
        if self.reasoning {
            match rest.find(THINK_END) {
                Some(i) => {
                    self.emit(&mut out, &rest[..i]);
                    (self.reasoning, self.trim) = (false, true);
                    rest = &rest[i + THINK_END.len()..];
                }
                None => {
                    let keep = holdback(rest, THINK_END);
                    self.emit(&mut out, &rest[..rest.len() - keep]);
                    self.held = rest[rest.len() - keep..].to_owned();
                    return out;
                }
            }
        }
        let stop = self
            .stops
            .iter()
            .filter_map(|s| rest.find(s.as_str()).map(|i| (i, s.clone())))
            .min_by_key(|(i, _)| *i);
        let fresh = !self.answered;
        let call = self.tools.then(|| tools::call_start(rest, fresh)).flatten();
        match (stop, call) {
            (Some((i, s)), c) if c.is_none_or(|c| i < c) => {
                self.emit(&mut out, &rest[..i]);
                self.stopped = true;
                self.matched = Some(s);
                return out;
            }
            (_, Some(c)) => {
                self.emit(&mut out, &rest[..c]);
                self.tool_text = rest[c..].to_owned();
                return out;
            }
            _ => {}
        }
        let markers = self.stops.iter().map(|s| holdback(rest, s));
        let keep = markers
            .chain(self.tools.then(|| tools::call_holdback(rest, fresh)))
            .max()
            .unwrap_or(0);
        self.emit(&mut out, &rest[..rest.len() - keep]);
        self.held = rest[rest.len() - keep..].to_owned();
        out
    }
    /// Flush held text at the end of generation (incomplete UTF-8 becomes U+FFFD).
    pub(crate) fn finish(&mut self) -> Vec<Delta> {
        let mut out = vec![];
        if !self.stopped && self.tool_text.is_empty() {
            let mut text = std::mem::take(&mut self.held);
            text += &String::from_utf8_lossy(&std::mem::take(&mut self.bytes));
            self.emit(&mut out, &text);
        }
        out
    }
}

#[cfg(test)]
#[path = "openai_tests.rs"]
mod tests;
