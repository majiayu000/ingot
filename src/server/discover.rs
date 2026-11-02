//! Finding models and apps from the Models and Connect pages: Hugging Face
//! search, one repository's files, installed tools and updates.
use crate::models::{self, DEFAULT_ENDPOINT, hf};
use crate::onboarding::Mac;
use crate::openai::ApiError;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

const RELEASES: &str = "https://api.github.com/repos/majiayu000/ingot/releases/latest";
/// Results shown for one search.
const SEARCH_LIMIT: usize = 20;

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into())
}

fn remote(e: std::io::Error) -> ApiError {
    ApiError::new(502, e.to_string())
}

fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// `GET /api/search?q=`: GGUF repositories, most downloaded first.
pub(super) fn search(query: &str) -> Result<Value, ApiError> {
    let url = format!(
        "{}/api/models?search={}&filter=gguf&limit={SEARCH_LIMIT}&sort=downloads\
         &expand%5B%5D=gguf&expand%5B%5D=downloads",
        endpoint().trim_end_matches('/'),
        encode(query)
    );
    let found = hf::get_json(&url).map_err(remote)?;
    let list = found.as_array().map(Vec::as_slice).unwrap_or_default();
    let results: Vec<_> = list
        .iter()
        .map(|model| {
            let architecture = model["gguf"]["architecture"].as_str().unwrap_or("");
            json!({"id": model["id"], "downloads": model["downloads"],
                "architecture": architecture,
                "supported": models::supported_architecture(architecture)})
        })
        .collect();
    Ok(json!({"results": results}))
}

/// The first shard's path for a shard, or the path itself.
fn first_shard(path: &str) -> String {
    let stem = path.strip_suffix(".gguf").unwrap_or(path);
    match stem.rsplit_once("-of-") {
        Some((head, count)) if head.ends_with(|c: char| c.is_ascii_digit()) => {
            let digits = head.bytes().rev().take_while(u8::is_ascii_digit).count();
            let base = &head[..head.len() - digits];
            format!("{base}{:0digits$}-of-{count}.gguf", 1)
        }
        _ => path.to_owned(),
    }
}

/// `GET /api/search/files?repo=`: its GGUF models, shards counted as one, with
/// what each needs here. Memory is estimated from the file size, because only
/// listed models have measured numbers.
pub(super) fn files(repo: &str, root: &Path) -> Result<Value, ApiError> {
    let valid = repo.split('/').count() == 2
        && repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.'));
    if !valid || repo.contains("..") {
        return Err(ApiError::new(400, "repo must be OWNER/NAME"));
    }
    let url = format!(
        "{}/api/models/{repo}/tree/main?recursive=true",
        endpoint().trim_end_matches('/')
    );
    let tree = hf::get_json(&url).map_err(remote)?;
    let mut sizes: BTreeMap<String, u64> = BTreeMap::new();
    for entry in tree.as_array().into_iter().flatten() {
        let path = entry["path"].as_str().unwrap_or("");
        if path.ends_with(".gguf") && !path.to_ascii_lowercase().contains("mmproj") {
            let size = entry["lfs"]["size"].as_u64().or(entry["size"].as_u64());
            *sizes.entry(first_shard(path)).or_default() += size.unwrap_or(0);
        }
    }
    let mac = Mac::probe(root).map_err(|e| ApiError::new(500, e.to_string()))?;
    let list: Vec<_> = sizes
        .iter()
        .map(|(path, bytes)| {
            let name = format!("{}{repo}/{path}", hf::PREFIX);
            let present = hf::Spec::parse(&name)
                .and_then(Result::ok)
                .is_some_and(|spec| spec.resolve(root).is_ok());
            let memory = *bytes as f64 * 1.15 / (1u64 << 30) as f64;
            let fit = if !present && *bytes > mac.free_bytes {
                "no_disk"
            } else if memory <= mac.cap_gib as f64 {
                "memory"
            } else {
                "too_large"
            };
            json!({"name": name, "file": path, "bytes": bytes, "present": present,
                "fit": fit, "memory_gib": (memory * 10.).round() / 10.})
        })
        .collect();
    Ok(json!({"files": list}))
}

/// `GET /api/apps`: tools `ingot launch` knows and whether they are installed.
pub(super) fn apps() -> Value {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut directories: Vec<String> = std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .map(str::to_owned)
        .collect();
    // Apps opened from the Finder have a short PATH.
    for extra in ["/opt/homebrew/bin", "/usr/local/bin"] {
        directories.push(extra.into());
    }
    for extra in [".local/bin", ".bun/bin", ".npm-global/bin", ".cargo/bin"] {
        directories.push(format!("{home}/{extra}"));
    }
    let installed = |name: &str| {
        directories
            .iter()
            .any(|dir| Path::new(dir).join(name).is_file())
    };
    let list: Vec<_> = crate::commands::TOOLS
        .iter()
        .map(|tool| {
            json!({"id": tool.id, "name": tool.name, "installed": installed(tool.binary),
                "launch": tool.launch, "steps": tool.steps})
        })
        .collect();
    json!({"apps": list})
}

/// `GET /api/update`: the latest published release, if any.
pub(super) fn update() -> Result<Value, ApiError> {
    let current = env!("CARGO_PKG_VERSION");
    match hf::get_json(RELEASES) {
        Ok(release) => {
            let tag = release["tag_name"].as_str().unwrap_or("");
            let latest = tag.strip_prefix('v').unwrap_or(tag);
            Ok(json!({"current": current, "latest": latest,
                "available": !latest.is_empty() && latest != current,
                "url": release["html_url"]}))
        }
        // No release is published yet: GitHub answers 404.
        Err(e) if e.to_string().contains("404") => {
            Ok(json!({"current": current, "latest": null, "available": false}))
        }
        Err(e) => Err(remote(e)),
    }
}

/// `POST /api/extract`: text of a PDF sent as the request body.
pub(super) fn extract(body: &[u8]) -> Result<Value, ApiError> {
    let (text, pages) = crate::pdf::extract(body).map_err(|e| ApiError::new(400, e.to_string()))?;
    Ok(json!({"text": text, "pages": pages}))
}

/// `GET /api/settings`, `PUT /api/settings`, `GET /api/chats`, `PUT /api/chats`.
pub(super) fn private(method: &str, path: &str, body: &[u8]) -> Result<Value, ApiError> {
    use crate::settings;
    let failed = |e: std::io::Error| ApiError::new(400, e.to_string());
    let parse = |bytes: &[u8]| -> Result<Value, ApiError> {
        serde_json::from_slice(bytes).map_err(|e| ApiError::new(400, format!("invalid JSON: {e}")))
    };
    match (method, path) {
        ("GET", "/api/settings") => settings::load().map_err(failed),
        ("PUT", "/api/settings") => settings::update(&parse(body)?).map_err(failed),
        ("GET", "/api/chats") => parse(settings::load_chats().map_err(failed)?.as_bytes()),
        _ => {
            settings::save_chats(body).map_err(failed)?;
            Ok(json!({"status": "success"}))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shards_are_named_by_the_first_one() {
        assert_eq!(
            first_shard("q/M-Q4-00002-of-00003.gguf"),
            "q/M-Q4-00001-of-00003.gguf"
        );
        assert_eq!(first_shard("M-Q4.gguf"), "M-Q4.gguf");
    }
    #[test]
    fn search_text_is_encoded() {
        assert_eq!(encode("qwen 3&x=1"), "qwen%203%26x%3D1");
    }
}
