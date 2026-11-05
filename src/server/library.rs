//! The model library behind the Models tab and `/api/pull`, `/api/delete`:
//! what this Mac can run, downloads with progress, and deletion.
use super::catalog::{lock, scan_error};
use super::http::HttpRequest;
use super::{Shared, respond, write_head};
use crate::models::custom::{self, Custom};
use crate::models::{self, Kind, ModelFile, hf};
use crate::onboarding::{self, Fit, Mac};
use crate::openai::ApiError;
use serde_json::{Value, json};
use std::io::Write;
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Least time between two progress lines of one file.
const PROGRESS_EVERY: Duration = Duration::from_millis(250);

fn io_error(e: std::io::Error) -> ApiError {
    ApiError::new(500, e.to_string())
}

/// `GET /api/catalog`: this Mac and every model with how it would run here.
pub(super) fn catalog(shared: &Shared) -> Result<Value, ApiError> {
    let local = shared.served.local()?;
    let (manifest, root) = (local.manifest(), local.root());
    let mac = Mac::probe(root).map_err(io_error)?;
    let choices = manifest
        .models
        .iter()
        .filter(|m| m.kind == Kind::Main)
        .map(|m| Ok((m, onboarding::installed(manifest, m, root)?)))
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(io_error)?;
    let best = onboarding::recommended(&choices, &mac);
    let loaded = lock(&local.loaded).clone();
    let pulling = lock(&local.pulling).clone();
    let state = |id: &str| (loaded.as_deref() == Some(id), pulling.contains(id));
    let mut list: Vec<Value> = choices
        .iter()
        .enumerate()
        .map(|(index, (model, present))| {
            let fit = onboarding::fit(model, *present, &mac);
            let (fit_name, context) = match fit {
                Fit::Memory(context) => ("memory", Some(context)),
                Fit::Streaming => ("streaming", None),
                Fit::TooLarge => ("too_large", None),
                Fit::NoDisk => ("no_disk", None),
            };
            let memory = context
                .and_then(|c| model.memory.iter().find(|(known, _)| *known == c))
                .map(|(_, gib)| *gib);
            let (active, downloading) = state(&model.name);
            json!({
                "name": model.name, "title": model.title, "summary": model.summary,
                "download_bytes": model.download_bytes(), "present": present,
                "fit": fit_name, "context": context, "memory_gib": memory,
                "recommended": index == best, "loaded": active, "downloading": downloading,
            })
        })
        .collect();
    // Models pulled from Hugging Face: not rated, but they run and can be deleted.
    for id in local.list().map_err(scan_error)? {
        if id.starts_with(models::hf::PREFIX) {
            let (active, downloading) = state(&id);
            list.push(json!({
                "name": id, "title": super::catalog::file_id(id.as_ref()),
                "summary": "Downloaded from Hugging Face",
                "download_bytes": models::local_bytes(manifest, &id, root), "present": true,
                "fit": "unknown", "context": null, "memory_gib": null,
                "recommended": false, "loaded": active, "downloading": downloading,
            }));
        }
    }
    for model in custom::list(root).map_err(io_error)? {
        let (active, downloading) = state(&model.base);
        list.push(json!({
            "name": model.name, "title": model.name,
            "summary": format!("Custom model based on {}", model.base),
            "download_bytes": 0, "present": true, "fit": "custom", "context": null,
            "memory_gib": null, "recommended": false, "loaded": active,
            "downloading": downloading, "custom": true, "base": model.base,
        }));
    }
    Ok(json!({
        "mac": {"device": mac.device, "memory_gib": mac.cap_gib, "free_bytes": mac.free_bytes},
        "directory": root.display().to_string(),
        "models": list,
    }))
}

/// Name a pull body asks for: `model`, or the older `name`.
pub(super) fn named(body: &Value) -> Result<&str, ApiError> {
    // Ollama's client sends both fields, one of them empty.
    ["model", "name"]
        .iter()
        .filter_map(|key| body.get(key)?.as_str())
        .find(|name| !name.is_empty())
        .map(|name| name.strip_suffix(":latest").unwrap_or(name))
        .ok_or_else(|| ApiError::new(400, "model is required"))
}

/// Stops refusing the model when the pull ends, however it ends.
struct Pulling<'a>(&'a super::catalog::Local, String);
impl Drop for Pulling<'_> {
    fn drop(&mut self) {
        lock(&self.0.pulling).remove(&self.1);
    }
}

/// Checks a pull can start and reserves the model name.
fn begin<'a>(shared: &'a Shared, name: &str) -> Result<Pulling<'a>, ApiError> {
    let local = shared.served.local()?;
    let is_hf = models::hf::Spec::parse(name)
        .transpose()
        .map_err(|e| ApiError::new(400, e.to_string()))?
        .is_some();
    if !is_hf {
        let model = local
            .manifest()
            .get(name)
            .map_err(|e| ApiError::new(404, e.to_string()))?;
        let present = onboarding::installed(local.manifest(), model, local.root());
        let mac = Mac::probe(local.root()).map_err(io_error)?;
        if onboarding::fit(model, present.map_err(io_error)?, &mac) == Fit::NoDisk {
            return Err(ApiError::new(
                400,
                format!(
                    "{name} needs {:.1} GiB but only {:.1} GiB is free",
                    model.download_bytes() as f64 / (1u64 << 30) as f64,
                    mac.free_bytes as f64 / (1u64 << 30) as f64
                ),
            ));
        }
    }
    if !lock(&local.pulling).insert(name.to_owned()) {
        return Err(ApiError::new(409, format!("{name} is already downloading")));
    }
    Ok(Pulling(local, name.to_owned()))
}

/// `POST /api/pull`: NDJSON progress. Closing the connection stops the
/// download, and pulling again resumes it.
pub(super) fn pull(mut stream: TcpStream, body: &[u8], shared: &Shared) {
    let parsed = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))
        .and_then(|body: Value| {
            let name = named(&body)?.to_owned();
            let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(true);
            let guard = begin(shared, &name)?;
            Ok((name, streaming, guard))
        });
    let (name, streaming, _guard) = match parsed {
        Ok(v) => v,
        Err(e) => return respond(&mut stream, e.status, &json!({"error": e.message})),
    };
    let Ok(local) = shared.served.local() else {
        return;
    };
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| models::DEFAULT_ENDPOINT.into());
    if streaming && write_head(&mut stream, 200, "application/x-ndjson", None).is_err() {
        return;
    }
    let line = |stream: &mut TcpStream, value: Value| -> std::io::Result<()> {
        if streaming {
            stream.write_all(format!("{value}\n").as_bytes())?;
        }
        Ok(())
    };
    let manifest = local.manifest();
    let mut last: Option<(String, Instant)> = None;
    let mut report = |file: &str, done: u64, total: u64| {
        let same = last.as_ref().is_some_and(|(name, at)| {
            name == file && at.elapsed() < PROGRESS_EVERY && done < total
        });
        if same {
            return Ok(());
        }
        last = Some((file.to_owned(), Instant::now()));
        let value = json!({"status": format!("pulling {file}"), "digest": file,
            "total": total, "completed": done});
        line(&mut stream, value)
    };
    // The draft that speeds up decoding is part of the model for app users.
    let draft = manifest.get(&name).ok().and_then(|m| m.draft.clone());
    let root = local.root();
    let mut result = models::pull_with(manifest, &name, root, &endpoint, &mut report);
    if let (Ok(_), Some(draft)) = (&result, draft) {
        result = models::pull_with(manifest, &draft, root, &endpoint, &mut report);
    }
    match result {
        Ok(_) if streaming => {
            let _ = stream.write_all(b"{\"status\":\"success\"}\n");
        }
        Ok(_) => respond(&mut stream, 200, &json!({"status": "success"})),
        Err(e) if streaming => {
            let _ = stream.write_all(format!("{}\n", json!({"error": e.to_string()})).as_bytes());
        }
        Err(e) => respond(&mut stream, 500, &json!({"error": e.to_string()})),
    }
}

/// `GET /api/model?name=`: everything the model page shows.
pub(super) fn details(request: &HttpRequest, shared: &Shared) -> Result<Value, ApiError> {
    let local = shared.served.local()?;
    let (manifest, root) = (local.manifest(), local.root());
    let name = request.parameter("name").unwrap_or_default();
    let found = custom::load(root, &name).map_err(io_error)?;
    let id = found.as_ref().map_or(name.clone(), |c| c.base.clone());
    let mut value = json!({"name": name, "title": name, "base": id,
        "loaded": lock(&local.loaded).as_deref() == Some(id.as_str()),
        "vision": shared.served.vision(&id, shared.fixed_vision),
        "tools": !id.to_ascii_lowercase().contains("glm")});
    if let Some(custom) = &found {
        value["custom"] = custom.to_json();
    }
    let sized = |files: &[ModelFile]| -> Vec<Value> {
        files
            .iter()
            .map(|f| json!({"name": f.name(), "bytes": f.size}))
            .collect()
    };
    if let Ok(model) = manifest.get(&id) {
        let mac = Mac::probe(root).map_err(io_error)?;
        let present = onboarding::installed(manifest, model, root).map_err(io_error)?;
        let (fit, context) = match onboarding::fit(model, present, &mac) {
            Fit::Memory(context) => ("memory", Some(context)),
            Fit::Streaming => ("streaming", None),
            Fit::TooLarge => ("too_large", None),
            Fit::NoDisk => ("no_disk", None),
        };
        let memory = context
            .and_then(|c| model.memory.iter().find(|(known, _)| *known == c))
            .map(|(_, gib)| *gib);
        let mut files = sized(&model.files);
        if let Some(draft) = model.draft.as_deref().and_then(|d| manifest.get(d).ok()) {
            files.extend(sized(&draft.files));
        }
        if found.is_none() {
            value["title"] = json!(model.title);
        }
        value["summary"] = json!(model.summary);
        value["present"] = json!(present);
        value["fit"] = json!(fit);
        value["context"] = json!(context);
        value["memory_gib"] = json!(memory);
        value["cap_gib"] = json!(mac.cap_gib);
        value["download_bytes"] = json!(model.download_bytes());
        value["draft"] = json!(model.draft);
        value["files"] = json!(files);
        value["source"] = json!(format!("https://huggingface.co/{}", model.files[0].repo));
    } else if let Some(Ok(spec)) = hf::Spec::parse(&id) {
        let files: Vec<_> = spec
            .files(root)
            .iter()
            .map(|path| {
                let bytes = std::fs::metadata(path).map_or(0, |m| m.len());
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
                json!({"name": name, "bytes": bytes})
            })
            .collect();
        value["title"] = json!(super::catalog::file_id(id.as_ref()));
        value["summary"] = json!("Downloaded from Hugging Face");
        value["present"] = json!(true);
        value["fit"] = json!("unknown");
        value["files"] = json!(files);
        value["source"] = json!(format!("https://huggingface.co/{}", spec.repo));
    } else {
        return Err(ApiError::new(404, format!("model '{name}' not found")));
    }
    Ok(value)
}

/// A custom model named `name` from `from`; a custom base is flattened, so
/// models never depend on each other.
fn derive(
    local: &super::catalog::Local,
    name: &str,
    from: &str,
    system: &str,
    parameters: &serde_json::Map<String, Value>,
) -> Result<Custom, ApiError> {
    if local.manifest().get(name).is_ok() {
        return Err(ApiError::new(
            409,
            format!("{name} is a built-in model name"),
        ));
    }
    let root = local.root();
    let from = from.strip_suffix(":latest").unwrap_or(from);
    let parent = custom::load(root, from).map_err(io_error)?;
    let Some(id) = local.find(from)? else {
        return Err(ApiError::new(404, format!("model '{from}' not found")));
    };
    let mut model = Custom {
        name: name.to_owned(),
        base: id,
        system: system.to_owned(),
        parameters: parameters.clone(),
    };
    if let Some(parent) = parent {
        model.base = parent.base;
        if model.system.is_empty() {
            model.system = parent.system;
        }
        for (key, value) in parent.parameters {
            model.parameters.entry(key).or_insert(value);
        }
    }
    custom::save(root, &model).map_err(|e| ApiError::new(400, e.to_string()))?;
    Ok(model)
}

/// `POST /api/create`: `from`, `system`, `parameters`, or a `modelfile`.
pub(super) fn create(body: &[u8], shared: &Shared) -> Result<Value, ApiError> {
    let local = shared.served.local()?;
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))?;
    let name = named(&body)?;
    let parsed = body["modelfile"]
        .as_str()
        .map(custom::parse_modelfile)
        .transpose()
        .map_err(|e| ApiError::new(400, e.to_string()))?;
    let model = match parsed {
        Some(file) => derive(local, name, &file.base, &file.system, &file.parameters)?,
        None => {
            let from = body["from"].as_str().unwrap_or_default();
            let system = body["system"].as_str().unwrap_or_default();
            let parameters = body["parameters"].as_object().cloned().unwrap_or_default();
            derive(local, name, from, system, &parameters)?
        }
    };
    eprintln!("ingot serve: created {} from {}", model.name, model.base);
    Ok(json!({"status": "success"}))
}

/// `POST /api/copy`: a custom model of the destination name.
pub(super) fn copy(body: &[u8], shared: &Shared) -> Result<Value, ApiError> {
    let local = shared.served.local()?;
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))?;
    let text = |key: &str| {
        body[key]
            .as_str()
            .map(|n| n.strip_suffix(":latest").unwrap_or(n))
            .filter(|n| !n.is_empty())
            .ok_or_else(|| ApiError::new(400, format!("{key} is required")))
    };
    derive(
        local,
        text("destination")?,
        text("source")?,
        "",
        &serde_json::Map::new(),
    )?;
    Ok(json!({"status": "success"}))
}

/// `DELETE /api/delete`: removes a local model that is not loaded.
pub(super) fn delete(body: &[u8], shared: &Shared) -> Result<Value, ApiError> {
    let local = shared.served.local()?;
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))?;
    let name = named(&body)?;
    if custom::remove(local.root(), name).map_err(io_error)? {
        return Ok(json!({"status": "success", "freed_bytes": 0}));
    }
    let Some(id) = local.find(name)? else {
        return Err(ApiError::new(404, format!("model '{name}' not found")));
    };
    if lock(&local.loaded).as_deref() == Some(id.as_str()) {
        return Err(ApiError::new(
            409,
            format!("{id} is loaded; it can be deleted once unloaded"),
        ));
    }
    if lock(&local.pulling).contains(&id) {
        return Err(ApiError::new(409, format!("{id} is downloading")));
    }
    let freed = models::remove(local.manifest(), &id, local.root()).map_err(io_error)?;
    eprintln!("ingot serve: deleted {id}, {freed} bytes freed");
    Ok(json!({"status": "success", "freed_bytes": freed}))
}
