//! OpenAI-compatible HTTP server. The main thread owns the engine and runs
//! requests one at a time in arrival order; connection threads parse requests
//! and exchange events with it over channels. HTTP/1.1 subset: one request per
//! connection, Content-Length bodies, `Connection: close` responses.
mod auth;
mod catalog;
mod discover;
mod http;
mod library;
pub(crate) mod monitor;
mod ollama;
mod responses;
pub(crate) mod worker;

use crate::engine::{Cancellation, Engine, Session, Stop};
use crate::openai::{ApiError, Delta, Prompt, Request, TextStream, bad, parse_request};
use crate::session_pool::{self, Pool};
use crate::thinking::{self, Thinking};
use crate::tools::{self, ToolCall};
use crate::{Result, anthropic, invalid};
pub(crate) use catalog::Served;
use http::{disconnected, read_request, sse};
pub(crate) use http::{respond, write_head};
use monitor::{Monitor, Record};
use serde_json::{Value, json};
use std::ffi::OsString;
use std::io::{ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const USAGE: &str = "usage: ingot serve MODEL [CONTEXT [MEMORY_GIB]] [--host 127.0.0.1] [--port 8080] [--sessions N] \
[--keep-alive DURATION] [--thinking LEVEL] [--api-key KEY] [--insecure] \
[--no-draft | --draft-model MODEL [--draft-tokens N] [--draft-vocab IDS]] [--prefix-cache] \
[--mmproj PATH] [--allow-remote-images]
       ingot serve [--default MODEL] [--context N] [--memory-gib N] [--keep-alive DURATION] [--host H] [--port P] \
[--sessions N] [--thinking LEVEL] [--api-key KEY] [--insecure] [--no-draft] [--prefix-cache]

Without MODEL every complete local model is listed at /v1/models and the model a
request names is loaded on demand, one at a time. Anthropic requests and requests
without a model use the loaded model, else --default, else the first listed one.
--keep-alive unloads a model after that long without requests (30s, 5m, 1h).
--thinking off|low|medium|high is the level for requests that set none (default:
the model's own: off for Qwen, Max effort for GLM).
--api-key KEY (or INGOT_API_KEY) requires `Authorization: Bearer KEY` or
`x-api-key: KEY` on the API, /metrics and /v1/status. A host other than loopback
needs a key unless --insecure is given.
Images: a vision projector (--mmproj, or one mmproj*.gguf beside the model) enables
image input; --allow-remote-images also fetches http(s) image URLs.";
const MAX_CONNECTIONS: usize = 16;
const MAX_PENDING: usize = 8;
const MAX_EVENTS: usize = 8;
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// The app's page and its parts, served as they are.
const WEB: [(&str, &str, &str); 9] = [
    (
        "/",
        "text/html; charset=utf-8",
        include_str!("web/index.html"),
    ),
    (
        "/style.css",
        "text/css; charset=utf-8",
        include_str!("web/style.css"),
    ),
    (
        "/markdown.js",
        "text/javascript; charset=utf-8",
        include_str!("web/markdown.js"),
    ),
    (
        "/chat.js",
        "text/javascript; charset=utf-8",
        include_str!("web/chat.js"),
    ),
    (
        "/monitor.js",
        "text/javascript; charset=utf-8",
        include_str!("web/monitor.js"),
    ),
    (
        "/models.js",
        "text/javascript; charset=utf-8",
        include_str!("web/models.js"),
    ),
    (
        "/settings.js",
        "text/javascript; charset=utf-8",
        include_str!("web/settings.js"),
    ),
    (
        "/shell.js",
        "text/javascript; charset=utf-8",
        include_str!("web/shell.js"),
    ),
    ("/favicon.ico", "image/svg+xml", ""),
];
/// True for a path of the app's page, which needs no API key.
pub(crate) fn is_web(path: &str) -> bool {
    WEB.iter().any(|(web, ..)| *web == path)
}

pub(crate) enum Event {
    Started {
        prompt_tokens: usize,
    },
    Delta(Delta),
    Done {
        /// OpenAI finish reason: stop, length or tool_calls.
        finish: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
        tool_calls: Vec<ToolCall>,
        /// The stop string that ended the answer.
        stop_sequence: Option<String>,
    },
    Error(ApiError),
}
/// What the engine worker does next, in arrival order.
pub(crate) enum Work {
    Run(Job),
    /// Load a model before its first request.
    Load {
        model: String,
        context: Option<usize>,
        keep_alive: Option<Option<Duration>>,
        done: SyncSender<std::result::Result<(), ApiError>>,
    },
    /// Unload a model if it is the resident one.
    Unload {
        model: String,
        done: SyncSender<()>,
    },
}
pub(crate) struct Job {
    /// Served id of the model to run on.
    model: String,
    chat: bool,
    request: Request,
    cancel: Cancellation,
    events: SyncSender<Event>,
}

/// Encode the prompt and check it against the context.
fn prepare(engine: &mut Engine, job: &Job) -> std::result::Result<(Vec<u32>, usize), ApiError> {
    let r = &job.request;
    let ids = match &r.prompt {
        Prompt::Chat {
            developer: true, ..
        } if engine.tokenizer.is_glm() => {
            return Err(bad(
                "the developer role is not part of this model's chat template",
                Some("messages"),
            ));
        }
        Prompt::Chat { messages, .. } => engine
            .tokenizer
            .chat_thinking(messages, &r.tools, r.thinking),
        Prompt::Text(text) => engine.tokenizer.encode(text, true),
    }
    .map_err(|e| bad(e.to_string(), Some("messages")))?;
    let images = crate::media::bytes(&r.prompt)?;
    let ids = engine
        .insert_images(ids, &images)
        .map_err(|e| match e.kind() {
            ErrorKind::InvalidData => bad(e.to_string(), Some("messages")),
            _ => ApiError::new(500, format!("image encoding failed: {e}")),
        })?;
    let context = engine.context();
    let remaining = context.saturating_sub(ids.len());
    let max_tokens = match r.max_tokens {
        Some(n) if r.clamp_max_tokens => n.min(remaining),
        Some(n) => n,
        None => remaining,
    };
    if ids.is_empty() || max_tokens == 0 || ids.len() + max_tokens > context {
        return Err(bad(
            format!(
                "this server's context length is {context} tokens; the request needs {} \
                 ({} prompt + {} completion)",
                ids.len() + max_tokens.max(1),
                ids.len(),
                max_tokens.max(1)
            ),
            Some(if job.chat { "messages" } else { "prompt" }),
        ));
    }
    Ok((ids, max_tokens))
}

/// Run one request on the engine thread; events go back to its connection.
fn run_job(engine: &mut Engine, pool: &mut Pool<Session>, job: Job, monitor: &Monitor) {
    let send = |e| job.events.send(e).is_ok();
    if job.cancel.is_cancelled() {
        return monitor.outcome("cancelled");
    }
    let (ids, max_tokens) = match prepare(engine, &job) {
        Ok(v) => v,
        Err(e) => {
            send(Event::Error(e));
            return monitor.outcome("error");
        }
    };
    if !send(Event::Started {
        prompt_tokens: ids.len(),
    }) {
        return monitor.outcome("cancelled");
    }
    let r = &job.request;
    let tail = engine.tokenizer.decode(&ids[ids.len().saturating_sub(4)..]);
    let mut text = TextStream::new(
        job.chat && tail.is_ok_and(|t| thinking::opens_think(&t)),
        r.stop.clone(),
        !r.tools.is_empty(),
    );
    // GGUF templates with enable_thinking let the model open the block itself.
    if job.chat && r.thinking.is_some_and(|t| t != Thinking::Off) {
        text = text.detect_think();
    }
    let cancel = &job.cancel;
    let (mut kept, pick) = pool.take(&ids);
    let result = engine.generate_request(
        &ids,
        max_tokens,
        cancel.clone(),
        r,
        Some(&mut kept),
        |_, bytes| {
            monitor.token();
            for delta in text.push(bytes) {
                if !send(Event::Delta(delta)) {
                    cancel.cancel();
                }
            }
            if text.stopped {
                cancel.cancel();
            }
            Ok(())
        },
    );
    let session = kept.map(|s| pool.put(pick.reused(), s));
    let g = match result {
        Ok(g) => g,
        Err(e) => {
            send(Event::Error(ApiError::new(
                500,
                format!("generation failed: {e}"),
            )));
            return monitor.outcome("error");
        }
    };
    let mut finish = match (text.stopped, g.stop) {
        (true, _) | (false, Stop::Eos) => "stop",
        (false, Stop::Length) => "length",
        (false, Stop::Cancelled) => {
            eprintln!("ingot serve: request cancelled (client disconnected)");
            return monitor.outcome("cancelled");
        }
    };
    for delta in text.finish() {
        send(Event::Delta(delta));
    }
    let mut tool_calls = vec![];
    if !text.tool_text.is_empty() {
        match tools::parse_calls(&text.tool_text, &r.tools) {
            Some(calls) => {
                tool_calls = calls;
                if !r.parallel_tool_calls {
                    tool_calls.truncate(1);
                }
                if finish == "stop" {
                    finish = "tool_calls";
                }
            }
            None => {
                eprintln!(
                    "ingot serve: generated tool call is incomplete or malformed; returned as text"
                );
                send(Event::Delta(Delta {
                    reasoning: false,
                    text: std::mem::take(&mut text.tool_text),
                }));
            }
        }
    }
    let s = &g.statistics;
    let decode = s.total.saturating_sub(s.prefill).as_secs_f64();
    let record = Record::new(
        &job.model,
        job.chat,
        ids.len(),
        s.prefix_reused_tokens,
        g.tokens.len(),
        s.prefill.as_secs_f64(),
        decode,
        finish,
        s.mtp_enabled,
    );
    eprintln!(
        "ingot serve: {} prompt={} reused={} session={} completion={} finish={finish} \
         prefill={:.2}s decode={:.1}tok/s mtp={}",
        if job.chat { "chat" } else { "completion" },
        ids.len(),
        s.prefix_reused_tokens,
        session_pool::describe(&pick, session, pool.len()),
        g.tokens.len(),
        s.prefill.as_secs_f64(),
        record.decode_rate(),
        s.mtp_enabled,
    );
    monitor.finished(record, pool.len());
    send(Event::Done {
        finish,
        prompt_tokens: ids.len(),
        completion_tokens: g.tokens.len(),
        tool_calls,
        stop_sequence: text.matched.take(),
    });
}

pub(crate) struct Shared {
    jobs: SyncSender<Work>,
    pub(crate) served: Served,
    pub(crate) next_id: AtomicU64,
    api_key: Option<String>,
    /// `--thinking`: the level for requests that give none.
    thinking: Option<Thinking>,
    monitor: Arc<Monitor>,
    /// Download http(s) image URLs (`--allow-remote-images`).
    remote_images: bool,
    /// `serve MODEL` was given a vision projector.
    fixed_vision: bool,
}
impl Shared {
    fn new(jobs: SyncSender<Work>, served: Served) -> Self {
        Self {
            jobs,
            served,
            next_id: AtomicU64::new(0),
            api_key: None,
            thinking: None,
            monitor: Arc::default(),
            remote_images: false,
            fixed_vision: false,
        }
    }
}

/// Events of one submitted job, with client disconnect detection.
pub(crate) struct Events {
    receiver: Receiver<Event>,
    pub(crate) cancel: Cancellation,
    probed: Instant,
}
impl Events {
    pub(crate) fn submit(
        shared: &Shared,
        model: &str,
        chat: bool,
        mut request: Request,
    ) -> std::result::Result<Self, ApiError> {
        request.thinking = request.thinking.or(shared.thinking);
        crate::media::resolve(&mut request.prompt, shared.remote_images)?;
        let cancel = Cancellation::default();
        let (events, receiver) = mpsc::sync_channel(MAX_EVENTS);
        let job = Job {
            model: model.to_owned(),
            chat,
            request,
            cancel: cancel.clone(),
            events,
        };
        shared.jobs.try_send(Work::Run(job)).map_err(|e| match e {
            TrySendError::Full(_) => ApiError::new(503, "server busy; retry later"),
            TrySendError::Disconnected(_) => ApiError::new(500, "engine worker stopped"),
        })?;
        shared.monitor.submitted();
        Ok(Self {
            receiver,
            cancel,
            probed: Instant::now(),
        })
    }
    /// Next event, or None once the client disconnected or the job ended
    /// without a result (it was cancelled).
    pub(crate) fn next(&mut self, stream: &TcpStream) -> Option<Event> {
        // Probe the socket on a fixed cadence: tokens can arrive faster than the
        // receive timeout, and a non-streaming response writes nothing until the end.
        let probe = Duration::from_millis(100);
        loop {
            let event = self.receiver.recv_timeout(probe);
            if self.probed.elapsed() >= probe {
                self.probed = Instant::now();
                if disconnected(stream) {
                    self.cancel.cancel();
                    return None;
                }
            }
            match event {
                Ok(e) => return Some(e),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
    }
}

/// OpenAI `tool_calls` entries; arguments are JSON strings.
fn openai_calls(calls: &[ToolCall], id: &str, index: bool) -> Value {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut v = json!({"id": format!("call_{id}_{i}"), "type": "function",
                "function": {"name": c.name, "arguments": Value::Object(c.arguments.clone()).to_string()}});
            if index {
                v["index"] = json!(i);
            }
            v
        })
        .collect()
}
fn model_object(id: &str, shared: &Shared) -> Value {
    json!({"id": id, "object": "model", "created": 0, "owned_by": "ingot",
        "vision": shared.served.vision(id, shared.fixed_vision), "thinking": shared.served.thinking(id)})
}
fn handle(mut stream: TcpStream, shared: Arc<Shared>) {
    if stream.set_read_timeout(Some(IO_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(IO_TIMEOUT)).is_err()
    {
        return;
    }
    let request = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => return respond(&mut stream, e.status, &e.body()),
    };
    if request.foreign(shared.api_key.is_some()) {
        let e = ApiError::new(403, "requests from another site are not allowed");
        return respond(&mut stream, e.status, &e.body());
    }
    let (method, path) = (request.method.as_str(), request.path.as_str());
    if let Some(key) = &shared.api_key
        && !auth::public(method, path)
        && !request.credentials.allows(key)
    {
        return respond(&mut stream, 401, &auth::rejection(path));
    }
    if path.starts_with("/api/") {
        return ollama::handle(stream, &request, &shared);
    }
    let page = |stream: &mut TcpStream, kind: &str, body: &str| {
        let _ = write_head(stream, 200, kind, Some(body.len()))
            .and_then(|_| stream.write_all(body.as_bytes()));
    };
    if method == "GET"
        && let Some((_, kind, body)) = WEB.iter().find(|(web, ..)| *web == path)
    {
        return page(&mut stream, kind, body);
    }
    let chat = match (method, path) {
        // Ollama clients check the server is up this way before every command.
        ("HEAD", "/") => {
            let _ = write_head(&mut stream, 200, "text/plain", Some(0));
            return;
        }
        ("GET", "/metrics") => {
            let text = shared.monitor.metrics();
            return page(&mut stream, "text/plain; version=0.0.4", &text);
        }
        ("GET", "/v1/status") => return respond(&mut stream, 200, &shared.monitor.status()),
        ("GET", "/v1/models") => {
            let ids = match shared.served.list() {
                Ok(ids) => ids,
                Err(e) => return respond(&mut stream, e.status, &e.body()),
            };
            let data: Vec<_> = ids.iter().map(|id| model_object(id, &shared)).collect();
            return respond(&mut stream, 200, &json!({"object": "list", "data": data}));
        }
        ("GET", path) if path.starts_with("/v1/models/") => {
            let id = http::decode(&path["/v1/models/".len()..]);
            return match shared.served.list() {
                Ok(ids) if ids.contains(&id) => {
                    respond(&mut stream, 200, &model_object(&id, &shared));
                }
                Ok(_) => {
                    let e = ApiError::new(404, format!("model '{id}' not found"));
                    respond(&mut stream, e.status, &e.body());
                }
                Err(e) => respond(&mut stream, e.status, &e.body()),
            };
        }
        ("POST", "/v1/responses") => return responses::handle(stream, &request.body, &shared),
        ("POST", "/v1/chat/completions") => true,
        ("POST", "/v1/completions") => false,
        ("POST", "/v1/messages") => return anthropic::handle(stream, &request.body, &shared),
        (
            method,
            "/metrics"
            | "/v1/status"
            | "/v1/models"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/messages",
        ) => {
            let e = ApiError::new(405, format!("method {method} not allowed"));
            return respond(&mut stream, e.status, &e.body());
        }
        (method, path) if is_web(path) => {
            let e = ApiError::new(405, format!("method {method} not allowed"));
            return respond(&mut stream, e.status, &e.body());
        }
        (_, path) => {
            let e = ApiError::new(404, format!("unknown endpoint {path}"));
            return respond(&mut stream, e.status, &e.body());
        }
    };
    let parsed = shared.served.customized(&request.body).and_then(|body| {
        let model = shared.served.openai(&body)?;
        Ok((parse_request(&body, chat, &model)?, model))
    });
    let (parsed, model) = match parsed {
        Ok(r) => r,
        Err(e) => return respond(&mut stream, e.status, &e.body()),
    };
    let (streaming, include_usage) = (parsed.stream, parsed.include_usage);
    let mut events = match Events::submit(&shared, &model, chat, parsed) {
        Ok(events) => events,
        Err(e) => return respond(&mut stream, e.status, &e.body()),
    };
    let n = shared.next_id.fetch_add(1, Ordering::Relaxed);
    let id = format!("{}-{n}", if chat { "chatcmpl" } else { "cmpl" });
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let object = match (chat, streaming) {
        (true, true) => "chat.completion.chunk",
        (true, false) => "chat.completion",
        (false, _) => "text_completion",
    };
    let envelope = |choices: Value, usage: Option<Value>| {
        let mut v = json!({"id": id, "object": object, "created": created, "model": model, "choices": choices});
        if let Some(u) = usage {
            v["usage"] = u;
        }
        v
    };
    let chunk = |delta: Option<&Delta>, finish: Option<&str>| -> Value {
        let choice = if chat {
            let delta = match delta {
                Some(d) if d.reasoning => json!({"reasoning_content": d.text}),
                Some(d) => json!({"content": d.text}),
                None => json!({}),
            };
            json!({"index": 0, "delta": delta, "finish_reason": finish})
        } else {
            let text = delta.map_or("", |d| d.text.as_str());
            json!({"index": 0, "text": text, "logprobs": null, "finish_reason": finish})
        };
        envelope(json!([choice]), None)
    };
    let (mut content, mut reasoning) = (String::new(), String::new());
    // Once SSE has started its HTTP status cannot change.
    let mut headers_sent = false;
    while let Some(event) = events.next(&stream) {
        let written = match event {
            Event::Error(e) => {
                if headers_sent {
                    // The error is a final SSE event, not a successful completion.
                    let _ = sse(&mut stream, &e.body());
                } else {
                    respond(&mut stream, e.status, &e.body());
                }
                return;
            }
            Event::Started { .. } if streaming => {
                write_head(&mut stream, 200, "text/event-stream", None).and_then(|_| {
                    headers_sent = true;
                    if chat {
                        let role = json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]);
                        sse(&mut stream, &envelope(role, None))
                    } else {
                        Ok(())
                    }
                })
            }
            Event::Started { .. } => Ok(()),
            Event::Delta(d) if streaming => sse(&mut stream, &chunk(Some(&d), None)),
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
                let usage = json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                });
                if streaming {
                    let mut last = chunk(None, Some(finish));
                    last["usage"] = usage.clone();
                    let calls = (!tool_calls.is_empty()).then(|| {
                        let delta = json!({"tool_calls": openai_calls(&tool_calls, &id, true)});
                        envelope(json!([{"index": 0, "delta": delta, "finish_reason": null}]), None)
                    });
                    let _ = calls
                        .map_or(Ok(()), |c| sse(&mut stream, &c))
                        .and_then(|_| sse(&mut stream, &last))
                        .and_then(|_| {
                            if include_usage {
                                sse(&mut stream, &envelope(json!([]), Some(usage)))
                            } else {
                                Ok(())
                            }
                        })
                        .and_then(|_| stream.write_all(b"data: [DONE]\n\n"));
                } else {
                    let choice = if chat {
                        let mut message = json!({"role": "assistant", "content": content});
                        if !reasoning.is_empty() {
                            message["reasoning_content"] = json!(reasoning);
                        }
                        if !tool_calls.is_empty() {
                            if content.trim().is_empty() {
                                message["content"] = Value::Null;
                            }
                            message["tool_calls"] = openai_calls(&tool_calls, &id, false);
                        }
                        json!({"index": 0, "message": message, "finish_reason": finish})
                    } else {
                        json!({"index": 0, "text": content, "logprobs": null, "finish_reason": finish})
                    };
                    respond(&mut stream, 200, &envelope(json!([choice]), Some(usage)));
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

fn accept_connection(
    mut stream: TcpStream,
    shared: &Arc<Shared>,
    connections: &mut Vec<JoinHandle<()>>,
) {
    connections.retain(|thread| !thread.is_finished());
    if connections.len() >= MAX_CONNECTIONS {
        // Keep the accept loop responsive even if an excess client never reads.
        if stream
            .set_write_timeout(Some(Duration::from_millis(100)))
            .is_ok()
        {
            let e = ApiError::new(503, "connection limit reached; retry later");
            respond(&mut stream, e.status, &e.body());
        }
        return;
    }
    let shared = shared.clone();
    match std::thread::Builder::new().spawn(move || handle(stream, shared)) {
        Ok(thread) => connections.push(thread),
        Err(e) => eprintln!("ingot serve: cannot start connection handler: {e}"),
    }
}

pub fn main(args: &[OsString]) -> Result<()> {
    let args = worker::Args::split(args)?;
    let host = args.string("--host").unwrap_or("127.0.0.1").to_owned();
    let port: u16 = args.get("--port")?.unwrap_or(8080);
    let keep_alive = args
        .string("--keep-alive")
        .map(worker::duration)
        .transpose()?;
    let api_key = auth::key(args.string("--api-key"))?;
    auth::check_bind(&host, api_key.is_some(), args.insecure)?;
    let default_thinking = args
        .string("--thinking")
        .map(str::parse::<Thinking>)
        .transpose()?;
    let source = args.source()?;
    let fixed_vision = matches!(&source, worker::Source::Fixed(o) if o.mmproj.is_some());
    let served = match &source {
        worker::Source::Fixed(options) => Served::Fixed(catalog::file_id(&options.model)),
        worker::Source::Local(_) => Served::Local(catalog::Local::new(
            crate::models::default_dir()?,
            args.string("--default"),
        )?),
    };
    let loaded = match &served {
        Served::Local(local) => local.loaded.clone(),
        Served::Fixed(_) => Arc::new(Mutex::new(None)),
    };
    // Bind before loading so a busy port fails fast.
    let listener = TcpListener::bind((host.as_str(), port))
        .map_err(|e| invalid(format!("cannot listen on {host}:{port}: {e}")))?;
    let monitor = Arc::new(Monitor::new(loaded.clone(), default_thinking));
    let mut worker = worker::Worker::new(
        source,
        args.get("--sessions")?,
        keep_alive,
        loaded,
        monitor.clone(),
    );
    match &served {
        Served::Fixed(id) => worker.ensure(id, None)?,
        Served::Local(local) => {
            let ids = local.list()?;
            eprintln!(
                "ingot serve: {} local model(s) in {}, loaded on demand: {}",
                ids.len(),
                local.root().display(),
                ids.join(", ")
            );
        }
    }
    if let Some(idle) = keep_alive {
        eprintln!(
            "ingot serve: an idle model is unloaded after {}s",
            idle.as_secs()
        );
    }
    eprintln!("ingot serve: API http://{host}:{port}/v1, chat page http://{host}:{port}/");
    eprintln!(
        "ingot serve: monitor http://{host}:{port}/metrics and /v1/status; API key {}; \
         thinking default {}",
        if api_key.is_some() {
            "required"
        } else {
            "not required"
        },
        default_thinking.map_or("from the model", |t| t.name()),
    );
    let (jobs, queue) = mpsc::sync_channel::<Work>(MAX_PENDING);
    let mut shared = Shared::new(jobs, served);
    shared.api_key = api_key;
    shared.thinking = default_thinking;
    shared.monitor = monitor;
    shared.remote_images = args.remote_images();
    shared.fixed_vision = fixed_vision;
    let shared = Arc::new(shared);
    std::thread::spawn(move || {
        let mut connections = Vec::new();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => accept_connection(stream, &shared, &mut connections),
                Err(e) => eprintln!("ingot serve: accept failed: {e}"),
            }
        }
    });
    worker.run(queue);
    Ok(())
}

#[cfg(test)]
mod tests;
