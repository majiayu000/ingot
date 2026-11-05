//! Ollama-compatible API under `/api/`, so apps that offer an Ollama provider
//! work with Ingot, plus the endpoints the app's own pages use. Requests become
//! OpenAI requests for the same engine worker; options the engine does not have
//! are accepted and ignored, because clients send them by default.
use super::http::HttpRequest;
use super::monitor;
use super::{Event, Events, Served, Shared, Work, discover, library, respond, write_head};
use crate::cli::DEFAULT_CONTEXT;
use crate::models::{self, Manifest};
use crate::openai::{ApiError, parse_request};
use crate::tools::ToolCall;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Ollama API level implemented; clients check it before they connect.
const API_VERSION: &str = "0.40.0";

fn error(e: &ApiError) -> Value {
    json!({"error": e.message})
}

pub(super) fn handle(mut stream: TcpStream, request: &HttpRequest, shared: &Shared) {
    let (method, path) = (request.method.as_str(), request.path.as_str());
    let body = request.body.as_slice();
    let result = match (method, path) {
        ("GET", "/api/version") => Ok(json!({"version": API_VERSION,
            "ingot": env!("CARGO_PKG_VERSION")})),
        ("GET", "/api/tags") => tags(shared),
        ("GET", "/api/ps") => running(shared),
        ("GET", "/api/catalog") => library::catalog(shared),
        ("GET", "/api/model") => library::details(request, shared),
        ("GET", "/api/apps") => Ok(discover::apps()),
        ("GET", "/api/update") => discover::update(),
        ("GET", "/api/search") => discover::search(&request.parameter("q").unwrap_or_default()),
        ("GET", "/api/search/files") => shared.served.local().and_then(|local| {
            discover::files(&request.parameter("repo").unwrap_or_default(), local.root())
        }),
        ("POST", "/api/extract") => discover::extract(body),
        ("POST", "/api/create") => library::create(body, shared),
        ("POST", "/api/copy") => library::copy(body, shared),
        ("GET" | "PUT", "/api/settings" | "/api/chats") => {
            if request.names_loopback() && stream.peer_addr().is_ok_and(|a| a.ip().is_loopback()) {
                discover::private(method, path, body)
            } else {
                Err(ApiError::new(403, "available only on this Mac"))
            }
        }
        ("POST", "/api/show") => show(body, shared),
        ("DELETE", "/api/delete") => library::delete(body, shared),
        ("POST", "/api/pull") => return library::pull(stream, body, shared),
        ("POST", "/api/chat") => return generate(stream, body, shared, true),
        ("POST", "/api/generate") => return generate(stream, body, shared, false),
        (
            method,
            "/api/version" | "/api/tags" | "/api/ps" | "/api/catalog" | "/api/show" | "/api/delete"
            | "/api/pull" | "/api/chat" | "/api/generate" | "/api/create" | "/api/copy"
            | "/api/settings" | "/api/chats" | "/api/extract" | "/api/search" | "/api/model",
        ) => Err(ApiError::new(405, format!("method {method} not allowed"))),
        (_, path) => Err(ApiError::new(404, format!("unknown endpoint {path}"))),
    };
    match result {
        Ok(value) => respond(&mut stream, 200, &value),
        Err(e) => respond(&mut stream, e.status, &error(&e)),
    }
}

/// RFC 3339 UTC time, which Ollama clients parse.
fn timestamp(time: SystemTime) -> String {
    let seconds = time.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rest) = ((seconds / 86400) as i64, seconds % 86400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest / 60 % 60,
        rest % 60
    )
}

fn digest(id: &str) -> String {
    format!("{:x}", Sha256::digest(id.as_bytes()))
}

fn details() -> Value {
    json!({"format": "gguf", "family": "", "families": null, "parameter_size": "",
        "quantization_level": ""})
}

fn sizes(shared: &Shared, id: &str) -> u64 {
    match &shared.served {
        Served::Local(local) => models::local_bytes(local.manifest(), id, local.root()),
        Served::Fixed(_) => 0,
    }
}

fn tags(shared: &Shared) -> Result<Value, ApiError> {
    let now = timestamp(SystemTime::now());
    let list: Vec<_> = shared
        .served
        .list()?
        .iter()
        .map(|id| {
            json!({"name": id, "model": id, "modified_at": now, "size": sizes(shared, id),
                "digest": digest(id), "details": details()})
        })
        .collect();
    Ok(json!({"models": list}))
}

fn running(shared: &Shared) -> Result<Value, ApiError> {
    let loaded = match &shared.served {
        Served::Fixed(id) => Some(id.clone()),
        Served::Local(local) => super::catalog::lock(&local.loaded).clone(),
    };
    let list: Vec<_> = loaded
        .iter()
        .map(|id| {
            let bytes = monitor::metal().0;
            json!({"name": id, "model": id, "size": bytes, "size_vram": bytes,
                "digest": digest(id), "details": details(),
                "context_length": shared.monitor.status()["context"],
                "expires_at": timestamp(SystemTime::now())})
        })
        .collect();
    Ok(json!({"models": list}))
}

fn context_of(id: &str) -> usize {
    let fitted = Manifest::builtin().ok().and_then(|manifest| {
        let (_, working_set) = models::working_set().ok()?;
        manifest
            .get(id)
            .ok()?
            .fitting_context(models::budget_cap_gib(working_set))
    });
    fitted.unwrap_or(DEFAULT_CONTEXT)
}

fn show(body: &[u8], shared: &Shared) -> Result<Value, ApiError> {
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))?;
    let name = library::named(&body)?;
    let id = shared
        .served
        .openai(json!({"model": name}).to_string().as_bytes())?;
    let base = match &shared.served {
        Served::Local(local) => local.base(&id),
        Served::Fixed(_) => id.clone(),
    };
    let mut capabilities = vec!["completion"];
    if !base.to_ascii_lowercase().contains("glm") {
        capabilities.push("tools");
    }
    if shared.served.vision(&id, shared.fixed_vision) {
        capabilities.push("vision");
    }
    Ok(json!({
        "details": details(), "template": "", "parameters": "",
        "model_info": {"general.architecture": "ingot", "ingot.context_length": context_of(&base)},
        "capabilities": capabilities,
    }))
}

/// The MIME type a base64 image starts with; PNG and JPEG are supported.
fn image_part(data: &str) -> Value {
    let kind = if data.starts_with("/9j/") {
        "jpeg"
    } else {
        "png"
    };
    json!({"type": "image_url", "image_url": {"url": format!("data:image/{kind};base64,{data}")}})
}

/// An Ollama message as an OpenAI one: images become content parts and tool
/// arguments, which Ollama sends as objects, become JSON strings.
fn message(item: &Value) -> Value {
    let mut out = Map::new();
    for key in ["role", "tool_call_id"] {
        if let Some(value) = item.get(key) {
            out.insert(key.into(), value.clone());
        }
    }
    if let Some(thinking) = item.get("thinking").filter(|t| t.is_string()) {
        out.insert("reasoning_content".into(), thinking.clone());
    }
    let text = item.get("content").cloned().unwrap_or_else(|| json!(""));
    let images = item.get("images").and_then(Value::as_array);
    let content = match images.filter(|list| !list.is_empty()) {
        Some(list) => {
            let mut parts: Vec<_> = list
                .iter()
                .filter_map(Value::as_str)
                .map(image_part)
                .collect();
            parts.push(json!({"type": "text", "text": text}));
            json!(parts)
        }
        None => text,
    };
    out.insert("content".into(), content);
    if let Some(calls) = item.get("tool_calls").and_then(Value::as_array) {
        let calls: Vec<_> = calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let function = call.get("function");
                let arguments = function.and_then(|f| f.get("arguments"));
                let arguments = match arguments {
                    Some(Value::String(text)) => text.clone(),
                    Some(value) => value.to_string(),
                    None => "{}".into(),
                };
                json!({"id": format!("call_{index}"), "type": "function", "function": {
                    "name": function.and_then(|f| f.get("name")), "arguments": arguments}})
            })
            .collect();
        out.insert("tool_calls".into(), json!(calls));
    }
    Value::Object(out)
}

/// The OpenAI body for an Ollama request. `chat` tells whether it uses the
/// model's chat template: `/api/generate` does too, unless `raw` is set.
fn translate(body: &Value, endpoint_chat: bool) -> (Value, bool) {
    let mut out = Map::new();
    let raw = !endpoint_chat && body.get("raw").and_then(Value::as_bool) == Some(true);
    if raw {
        out.insert("prompt".into(), body["prompt"].clone());
    } else if endpoint_chat {
        let items = body.get("messages").and_then(Value::as_array);
        let list: Vec<_> = items.into_iter().flatten().map(message).collect();
        out.insert("messages".into(), json!(list));
    } else {
        let mut list = vec![];
        let system = body.get("system").and_then(Value::as_str);
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            list.push(json!({"role": "system", "content": system}));
        }
        let mut user = json!({"role": "user", "content": body["prompt"]});
        if let Some(images) = body.get("images") {
            user["images"] = images.clone();
        }
        list.push(message(&user));
        out.insert("messages".into(), json!(list));
    }
    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        out.insert("tools".into(), tools.clone());
    }
    match body.get("format") {
        Some(Value::String(format)) if format == "json" => {
            out.insert("response_format".into(), json!({"type": "json_object"}));
        }
        Some(schema @ Value::Object(_)) => {
            let format = json!({"type": "json_schema", "json_schema": {"schema": schema}});
            out.insert("response_format".into(), format);
        }
        _ => {}
    }
    match body.get("think") {
        Some(Value::Bool(true)) => {
            out.insert("reasoning_effort".into(), json!("high"));
        }
        Some(Value::Bool(false)) => {
            out.insert("reasoning_effort".into(), json!("none"));
        }
        Some(level @ Value::String(_)) => {
            out.insert("reasoning_effort".into(), level.clone());
        }
        _ => {}
    }
    if let Some(options) = body.get("options").and_then(Value::as_object) {
        let keys = [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "seed",
            "stop",
            "num_ctx",
        ];
        for key in keys {
            if let Some(value) = options.get(key) {
                out.insert(key.into(), value.clone());
            }
        }
        if let Some(limit) = options
            .get("num_predict")
            .and_then(Value::as_i64)
            .filter(|n| *n > 0)
        {
            out.insert("max_tokens".into(), json!(limit));
        }
    }
    if let Some(keep) = body.get("keep_alive").filter(|k| !k.is_null()) {
        out.insert("keep_alive".into(), keep.clone());
    }
    (Value::Object(out), !raw)
}

/// Runs `work` on the engine worker and waits for it.
fn control<T>(
    shared: &Shared,
    work: impl FnOnce(mpsc::SyncSender<T>) -> Work,
) -> Result<T, ApiError> {
    let (done, finished) = mpsc::sync_channel(1);
    shared
        .jobs
        .try_send(work(done))
        .map_err(|_| ApiError::new(503, "server busy; retry later"))?;
    finished
        .recv()
        .map_err(|_| ApiError::new(500, "engine worker stopped"))
}

/// A request without input: `keep_alive` 0 unloads the model, anything else
/// loads it, so `ollama stop` and preloading work.
fn load_or_unload(body: &Value, name: &str, shared: &Shared) -> Result<&'static str, ApiError> {
    let (mut request, _) = translate(body, true);
    request["model"] = json!(name);
    let keep_alive = request
        .get("keep_alive")
        .map(crate::openai::keep_alive)
        .transpose()?;
    let bytes = shared.served.customized(request.to_string().as_bytes())?;
    let model = shared.served.openai(&bytes)?;
    if keep_alive == Some(Some(Duration::ZERO)) {
        control(shared, |done| Work::Unload { model, done })?;
        return Ok("unload");
    }
    let customized: Value = serde_json::from_slice(&bytes).unwrap_or_default();
    let context = customized["num_ctx"].as_u64().map(|n| n as usize);
    control(shared, |done| Work::Load {
        model,
        context,
        keep_alive,
        done,
    })??;
    Ok("load")
}

/// A request without input only controls whether the model stays loaded.
fn load_only(body: &Value, chat: bool) -> bool {
    if chat {
        body.get("messages")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
    } else {
        body.get("prompt")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    }
}

fn calls_json(calls: &[ToolCall]) -> Value {
    let calls: Vec<_> = calls
        .iter()
        .map(|c| json!({"function": {"name": c.name, "arguments": c.arguments}}))
        .collect();
    json!(calls)
}

/// `/api/chat` and `/api/generate`, streamed as NDJSON unless `stream` is false.
fn generate(mut stream: TcpStream, body: &[u8], shared: &Shared, chat: bool) {
    let started = Instant::now();
    let fail = |stream: &mut TcpStream, e: ApiError| respond(stream, e.status, &error(&e));
    let body: Value = match serde_json::from_slice(body) {
        Ok(body) => body,
        Err(e) => {
            return fail(
                &mut stream,
                ApiError::new(400, format!("invalid JSON: {e}")),
            );
        }
    };
    let name = match library::named(&body) {
        Ok(name) => name.to_owned(),
        Err(e) => return fail(&mut stream, e),
    };
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let now = || timestamp(SystemTime::now());
    let piece = |text: &str, thinking: Option<&str>| {
        let mut value = json!({"model": name, "created_at": now(), "done": false});
        if chat {
            let mut message = json!({"role": "assistant", "content": text});
            if let Some(thinking) = thinking {
                message["thinking"] = json!(thinking);
            }
            value["message"] = message;
        } else {
            value["response"] = json!(text);
            if let Some(thinking) = thinking {
                value["thinking"] = json!(thinking);
            }
        }
        value
    };
    if load_only(&body, chat) {
        return match load_or_unload(&body, &name, shared) {
            Ok(reason) => {
                let mut value = piece("", None);
                value["done"] = json!(true);
                value["done_reason"] = json!(reason);
                respond(&mut stream, 200, &value);
            }
            Err(e) => fail(&mut stream, e),
        };
    }
    let (mut request, template) = translate(&body, chat);
    request["model"] = json!(name);
    let parsed = shared
        .served
        .customized(request.to_string().as_bytes())
        .and_then(|bytes| {
            let mut request: Value =
                serde_json::from_slice(&bytes).map_err(|e| ApiError::new(500, e.to_string()))?;
            let id = shared.served.openai(&bytes)?;
            if matches!(shared.served, Served::Local(_)) {
                request["model"] = json!(id);
            }
            let parsed = parse_request(request.to_string().as_bytes(), template, &id)?;
            Ok((id, parsed))
        });
    let (id, parsed) = match parsed {
        Ok(v) => v,
        Err(e) => return fail(&mut stream, e),
    };
    let mut events = match Events::submit(shared, &id, template, parsed) {
        Ok(events) => events,
        Err(e) => return fail(&mut stream, e),
    };
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut head_sent = false;
    let write =
        |stream: &mut TcpStream, value: Value| stream.write_all(format!("{value}\n").as_bytes());
    while let Some(event) = events.next(&stream) {
        let written = match event {
            Event::Error(e) if head_sent => {
                let _ = write(&mut stream, error(&e));
                return;
            }
            Event::Error(e) => return fail(&mut stream, e),
            Event::Started { .. } if streaming => {
                head_sent = true;
                write_head(&mut stream, 200, "application/x-ndjson", None)
            }
            Event::Started { .. } => Ok(()),
            Event::Delta(d) if streaming => {
                let line = if d.reasoning {
                    piece("", Some(&d.text))
                } else {
                    piece(&d.text, None)
                };
                write(&mut stream, line)
            }
            Event::Delta(d) => {
                if d.reasoning {
                    reasoning += &d.text;
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
                ..
            } => {
                let thinking = (!reasoning.is_empty()).then_some(reasoning.as_str());
                let mut last = piece(&content, thinking);
                last["done"] = json!(true);
                last["done_reason"] = json!(if finish == "tool_calls" {
                    "stop"
                } else {
                    finish
                });
                last["total_duration"] = json!(started.elapsed().as_nanos() as u64);
                last["prompt_eval_count"] = json!(prompt_tokens);
                last["eval_count"] = json!(completion_tokens);
                if chat && !tool_calls.is_empty() {
                    last["message"]["tool_calls"] = calls_json(&tool_calls);
                }
                if streaming {
                    let _ = write(&mut stream, last);
                } else {
                    respond(&mut stream, 200, &last);
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
    fn times_are_rfc3339() {
        let at = |s| UNIX_EPOCH + std::time::Duration::from_secs(s);
        assert_eq!(timestamp(at(0)), "1970-01-01T00:00:00Z");
        assert_eq!(timestamp(at(1_709_210_096)), "2024-02-29T12:34:56Z");
    }
    #[test]
    fn messages_become_openai_messages() {
        let (body, template) = translate(
            &json!({"messages": [
                {"role": "user", "content": "look", "images": ["/9j/abcd"]},
                {"role": "assistant", "content": "", "tool_calls":
                    [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                {"role": "tool", "content": "2", "tool_name": "f"},
            ], "options": {"temperature": 0.5, "num_predict": 9, "num_ctx": 4096,
                "repeat_penalty": 1.1}, "keep_alive": "5m", "think": false, "format": "json"}),
            true,
        );
        assert!(template);
        let m = &body["messages"];
        assert_eq!(
            m[0]["content"][0]["image_url"]["url"],
            "data:image/jpeg;base64,/9j/abcd"
        );
        assert_eq!(m[0]["content"][1]["text"], "look");
        assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], r#"{"a":1}"#);
        assert!(m[2].get("tool_name").is_none());
        assert_eq!(body["max_tokens"], 9);
        assert_eq!(body["num_ctx"], 4096);
        assert_eq!(body["keep_alive"], "5m");
        assert!(body.get("repeat_last_n").is_none());
        assert_eq!(body["reasoning_effort"], "none");
        assert_eq!(body["response_format"]["type"], "json_object");
        let parsed = parse_request(body.to_string().as_bytes(), true, "m");
        assert_eq!(parsed.err().map(|e| e.message), None);
    }
    #[test]
    fn generate_uses_the_template_unless_raw() {
        let (body, template) = translate(&json!({"prompt": "hi", "system": "be brief"}), false);
        assert!(template);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "hi");
        let (body, template) = translate(&json!({"prompt": "hi", "raw": true}), false);
        assert!(!template);
        assert_eq!(body["prompt"], "hi");
        assert!(parse_request(body.to_string().as_bytes(), false, "m").is_ok());
        // Ollama's client sends every field, empty ones too.
        let (body, _) = translate(&json!({"prompt": "hi", "system": ""}), false);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn the_model_name_may_be_in_either_field() {
        let name = |body: Value| library::named(&body).map(str::to_owned).ok();
        assert_eq!(name(json!({"model": "", "name": "m"})).unwrap(), "m");
        assert_eq!(name(json!({"model": "m:latest", "name": ""})).unwrap(), "m");
        assert_eq!(name(json!({"model": "", "name": ""})), None);
    }
    #[test]
    fn empty_requests_only_load() {
        assert!(load_only(&json!({"model": "m"}), true));
        assert!(load_only(&json!({"prompt": ""}), false));
        assert!(!load_only(&json!({"prompt": "x"}), false));
    }
}
