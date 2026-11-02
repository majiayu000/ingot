//! Everyday model commands: `list`, `ps`, `show`, `stop`, `rm`, `cp`, `create`
//! and `launch`. Commands about the running server talk to it over HTTP.
use crate::models::custom::{self, Custom};
use crate::models::{self, Manifest};
use crate::{Result, invalid, settings};
use serde_json::{Value, json};
use std::ffi::OsString;
use std::io;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;

/// Context `launch` loads for coding tools, whose instructions are long.
const CODING_CONTEXT: usize = 65536;

pub const USAGE: &str = "usage: ingot list
       ingot ps
       ingot show MODEL
       ingot stop MODEL
       ingot rm MODEL
       ingot cp SOURCE DESTINATION
       ingot create NAME -f MODELFILE
       ingot launch TOOL [--model MODEL] [--context N] [--dry-run]

ps, stop, cp, create and launch need a running server: the Mac app or `ingot serve`.
launch sets up and starts a coding tool: claude, codex or opencode.";

pub struct Tool {
    pub id: &'static str,
    pub name: &'static str,
    pub binary: &'static str,
    /// `ingot launch` starts it.
    pub launch: bool,
    /// Manual setup for tools that take a base URL.
    pub steps: &'static str,
}

pub const TOOLS: [Tool; 6] = [
    Tool {
        id: "claude",
        name: "Claude Code",
        binary: "claude",
        launch: true,
        steps: "",
    },
    Tool {
        id: "codex",
        name: "Codex",
        binary: "codex",
        launch: true,
        steps: "",
    },
    Tool {
        id: "opencode",
        name: "OpenCode",
        binary: "opencode",
        launch: true,
        steps: "",
    },
    Tool {
        id: "vscode",
        name: "VS Code",
        binary: "code",
        launch: false,
        steps: "In your AI extension, add an Ollama or OpenAI-compatible provider with this address.",
    },
    Tool {
        id: "cursor",
        name: "Cursor",
        binary: "cursor",
        launch: false,
        steps: "In Settings > Models, turn on Override OpenAI Base URL and enter this address with /v1.",
    },
    Tool {
        id: "openwebui",
        name: "Open WebUI",
        binary: "open-webui",
        launch: false,
        steps: "In Admin Panel > Settings > Connections, add an OpenAI connection with this address and /v1.",
    },
];

fn api_key() -> Option<String> {
    std::env::var("INGOT_API_KEY").ok().or_else(|| {
        let settings = settings::load().ok()?;
        settings["api_key"]
            .as_str()
            .filter(|k| !k.is_empty())
            .map(str::to_owned)
    })
}

/// The server to talk to: `INGOT_HOST`, the app's port, or `serve`'s default.
fn server() -> Option<String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(1)))
        .build()
        .into();
    let saved = settings::load().ok().and_then(|s| s["port"].as_u64());
    let mut candidates = Vec::new();
    if let Ok(host) = std::env::var("INGOT_HOST") {
        candidates.push(if host.contains("://") {
            host
        } else {
            format!("http://{host}")
        });
    }
    for port in saved
        .into_iter()
        .chain([u64::from(settings::APP_PORT), 8080])
    {
        candidates.push(format!("http://127.0.0.1:{port}"));
    }
    candidates.into_iter().find(|url| {
        agent
            .get(format!("{url}/api/version"))
            .call()
            .ok()
            .and_then(|r| r.into_body().read_to_string().ok())
            .is_some_and(|text| text.contains("\"ingot\""))
    })
}

fn required_server() -> Result<String> {
    server().ok_or_else(|| invalid("no Ingot server is running; open the app or run `ingot serve`"))
}

fn call(method: &str, url: &str, body: Option<&Value>) -> Result<Value> {
    let builder = ureq::http::Request::builder().method(method).uri(url);
    let builder = match api_key() {
        Some(key) => builder.header("Authorization", format!("Bearer {key}")),
        None => builder,
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .allow_non_standard_methods(true)
        .timeout_recv_response(Some(Duration::from_secs(600)))
        .build()
        .into();
    let text = body.map_or(String::new(), Value::to_string);
    let request = builder.body(text).map_err(io::Error::other)?;
    let response = agent.run(request).map_err(io::Error::other)?;
    let status = response.status().as_u16();
    let text = response
        .into_body()
        .read_to_string()
        .map_err(io::Error::other)?;
    let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status >= 400 {
        return Err(invalid(value["error"].as_str().unwrap_or(&text)));
    }
    Ok(value)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn list() -> Result<()> {
    let manifest = Manifest::builtin()?;
    let root = models::default_dir()?;
    let local = models::local_main(&manifest, &root)?;
    println!("{:<44} {:>10}  TITLE", "NAME", "SIZE");
    for id in &local {
        let title = manifest.get(id).map_or("", |m| m.title.as_str());
        let size = gib(models::local_bytes(&manifest, id, &root));
        println!("{id:<44} {size:>10}  {title}");
    }
    let customs = custom::list(&root)?;
    for model in &customs {
        println!("{:<44} {:>10}  based on {}", model.name, "-", model.base);
    }
    if local.is_empty() && customs.is_empty() {
        println!("No models yet. Run `ingot` to choose one.");
    }
    Ok(())
}

fn ps() -> Result<()> {
    let Some(url) = server() else {
        println!("No Ingot server is running.");
        return Ok(());
    };
    let running = call("GET", &format!("{url}/api/ps"), None)?;
    println!("{:<44} {:>10} {:>9}", "NAME", "MEMORY", "CONTEXT");
    for model in running["models"].as_array().into_iter().flatten() {
        println!(
            "{:<44} {:>10} {:>9}",
            model["name"].as_str().unwrap_or(""),
            gib(model["size"].as_u64().unwrap_or(0)),
            model["context_length"]
        );
    }
    Ok(())
}

fn show(name: &str) -> Result<()> {
    let manifest = Manifest::builtin()?;
    let root = models::default_dir()?;
    if let Some(model) = custom::load(&root, name)? {
        println!("Custom model {}\nBased on: {}", model.name, model.base);
        if !model.system.is_empty() {
            println!("System prompt:\n{}", model.system);
        }
        for (key, value) in &model.parameters {
            println!("{key}: {value}");
        }
        return Ok(());
    }
    if let Ok(model) = manifest.get(name) {
        println!(
            "{}\n{}\nDownload: {}",
            model.title,
            model.summary,
            gib(model.download_bytes())
        );
        for (context, memory) in &model.memory {
            println!("Memory at context {context}: {memory} GiB");
        }
        if let Some(draft) = &model.draft {
            println!("Faster decoding with: {draft}");
        }
        println!("Source: https://huggingface.co/{}", model.files[0].repo);
        return Ok(());
    }
    match models::hf::Spec::parse(name) {
        Some(spec) => {
            let spec = spec?;
            println!("{name}\nSource: https://huggingface.co/{}", spec.repo);
            Ok(())
        }
        None => Err(invalid(format!("unknown model {name}"))),
    }
}

fn stop(name: &str) -> Result<()> {
    let url = required_server()?;
    let body = json!({"model": name, "keep_alive": 0});
    call("POST", &format!("{url}/api/generate"), Some(&body))?;
    Ok(())
}

fn remove(name: &str) -> Result<()> {
    if let Some(url) = server() {
        let body = json!({"model": name});
        call("DELETE", &format!("{url}/api/delete"), Some(&body))?;
    } else {
        let root = models::default_dir()?;
        if !custom::remove(&root, name)? {
            models::remove(&Manifest::builtin()?, name, &root)?;
        }
    }
    println!("Deleted {name}");
    Ok(())
}

fn copy(source: &str, destination: &str) -> Result<()> {
    let url = required_server()?;
    let body = json!({"source": source, "destination": destination});
    call("POST", &format!("{url}/api/copy"), Some(&body))?;
    println!("Copied {source} to {destination}");
    Ok(())
}

fn create(name: &str, file: &str) -> Result<()> {
    let text =
        std::fs::read_to_string(file).map_err(|e| invalid(format!("cannot read {file}: {e}")))?;
    let parsed: Custom = custom::parse_modelfile(&text)?;
    custom::check_name(name)?;
    let url = required_server()?;
    let body = json!({"model": name, "from": parsed.base, "system": parsed.system,
        "parameters": parsed.parameters});
    call("POST", &format!("{url}/api/create"), Some(&body))?;
    println!("Created {name}");
    Ok(())
}

/// The model `launch` uses: the given one, the loaded one, the default, or the
/// first local one.
fn choose_model(url: &str, given: Option<&str>) -> Result<String> {
    if let Some(name) = given {
        return Ok(name.to_owned());
    }
    let running = call("GET", &format!("{url}/api/ps"), None)?;
    if let Some(name) = running["models"][0]["name"].as_str() {
        return Ok(name.to_owned());
    }
    let settings = settings::load()?;
    if let Some(name) = settings["default_model"].as_str().filter(|n| !n.is_empty()) {
        return Ok(name.to_owned());
    }
    let tags = call("GET", &format!("{url}/api/tags"), None)?;
    tags["models"][0]["name"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("no local models yet; run `ingot` to download one"))
}

fn tool_command(tool: &str, url: &str, model: &str) -> Result<Command> {
    let key = api_key().unwrap_or_else(|| "local".into());
    let command = match tool {
        "claude" => {
            let mut command = Command::new("claude");
            command
                .env("ANTHROPIC_BASE_URL", url)
                .env("ANTHROPIC_API_KEY", key)
                .env("ANTHROPIC_MODEL", model)
                .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");
            command
        }
        "codex" => {
            let mut command = Command::new("codex");
            command.env("INGOT_API_KEY", key).args([
                "-c".to_owned(),
                "model_provider=\"ingot\"".to_owned(),
                "-c".to_owned(),
                format!(
                    "model_providers.ingot={{name=\"Ingot\",base_url=\"{url}/v1\",\
                     env_key=\"INGOT_API_KEY\",wire_api=\"responses\"}}"
                ),
                "-m".to_owned(),
                model.to_owned(),
            ]);
            command
        }
        "opencode" => {
            let config = json!({"provider": {"ingot": {"npm": "@ai-sdk/openai-compatible",
                "name": "Ingot", "options": {"baseURL": format!("{url}/v1"), "apiKey": key},
                "models": {model: {"name": model}}}}, "model": format!("ingot/{model}")});
            let mut command = Command::new("opencode");
            command.env("OPENCODE_CONFIG_CONTENT", config.to_string());
            command
        }
        other => {
            return Err(invalid(format!(
                "unknown tool {other}; use claude, codex or opencode"
            )));
        }
    };
    Ok(command)
}

fn launch(args: &[String]) -> Result<()> {
    let tool = args.first().ok_or_else(|| invalid(USAGE))?;
    let (mut model, mut context, mut dry_run) = (None, CODING_CONTEXT, false);
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--model" => model = rest.next().map(String::as_str),
            "--context" => {
                context = rest
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| invalid("--context needs a number"))?;
            }
            _ => return Err(invalid(USAGE)),
        }
    }
    let url = required_server()?;
    let model = choose_model(&url, model)?;
    let mut command = tool_command(tool, &url, &model)?;
    if dry_run {
        println!("{command:?}");
        return Ok(());
    }
    eprintln!("Loading {model} with a context of {context} tokens…");
    let body = json!({"model": model, "keep_alive": -1, "options": {"num_ctx": context}});
    call("POST", &format!("{url}/api/generate"), Some(&body))?;
    Err(command.exec())
}

pub fn main(command: &str, args: &[OsString]) -> Result<()> {
    let args: Vec<String> = args
        .iter()
        .map(|a| a.to_str().map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| invalid("arguments must be UTF-8"))?;
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match (command, words.as_slice()) {
        ("list", []) => list(),
        ("ps", []) => ps(),
        ("show", [name]) => show(name),
        ("stop", [name]) => stop(name),
        ("rm", [name]) => remove(name),
        ("cp", [source, destination]) => copy(source, destination),
        ("create", [name, "-f", file]) => create(name, file),
        ("launch", _) => launch(&args),
        _ => Err(invalid(USAGE)),
    }
}
