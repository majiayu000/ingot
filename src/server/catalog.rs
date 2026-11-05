//! Model names a server answers to: the one `serve MODEL` loaded, or every
//! complete local main model when `serve` runs without MODEL.
use crate::models::{self, Manifest, custom};
use crate::openai::ApiError;
use crate::{Result, invalid};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) enum Served {
    /// `serve MODEL`: OpenAI requests naming another model are 404.
    Fixed(String),
    Local(Local),
}

pub(crate) struct Local {
    manifest: Manifest,
    root: PathBuf,
    default: Option<String>,
    /// Ids from the last directory scan.
    ids: Mutex<Vec<String>>,
    /// Id of the resident model; the engine worker updates it.
    pub(crate) loaded: Arc<Mutex<Option<String>>>,
    /// Models being downloaded.
    pub(crate) pulling: Mutex<HashSet<String>>,
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Served name of a model file: its stem without a `-00001-of-0000N` suffix.
pub(crate) fn file_id(path: &std::path::Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    match stem.rsplit_once("-00001-of-") {
        Some((base, count)) if count.bytes().all(|b| b.is_ascii_digit()) => base.to_owned(),
        _ => stem.to_owned(),
    }
}

/// `name` is `id` itself or, case-insensitively, a pulled file's stem.
fn matches(id: &str, name: &str) -> bool {
    id == name || id.starts_with("hf:") && file_id(id.as_ref()).eq_ignore_ascii_case(name)
}

impl Local {
    /// Scans `root`; `default` must name a local model.
    pub(crate) fn new(root: PathBuf, default: Option<&str>) -> Result<Self> {
        let mut local = Self {
            manifest: Manifest::builtin()?,
            root,
            default: None,
            ids: Mutex::new(Vec::new()),
            loaded: Arc::default(),
            pulling: Mutex::default(),
        };
        let ids = local.scan()?;
        if let Some(name) = default {
            let id = ids.iter().find(|id| matches(id, name)).ok_or_else(|| {
                invalid(format!(
                    "--default {name} is not a complete local model in {}; local models: {}",
                    local.root.display(),
                    describe(&ids)
                ))
            })?;
            local.default = Some(id.clone());
        }
        Ok(local)
    }
    pub(crate) fn root(&self) -> &std::path::Path {
        &self.root
    }
    pub(crate) fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn scan(&self) -> Result<Vec<String>> {
        let mut ids = models::local_main(&self.manifest, &self.root)?;
        ids.extend(custom::list(&self.root)?.into_iter().map(|c| c.name));
        *lock(&self.ids) = ids.clone();
        Ok(ids)
    }
    /// Local ids, rescanned: the resident model first, then the default.
    pub(crate) fn list(&self) -> Result<Vec<String>> {
        let mut ids = self.scan()?;
        let loaded = lock(&self.loaded).clone();
        for first in [self.default.as_ref(), loaded.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Some(i) = ids.iter().position(|id| id == first) {
                let id = ids.remove(i);
                ids.insert(0, id);
            }
        }
        Ok(ids)
    }
    /// The local id `name` refers to; rescans once when it is not known yet.
    pub(crate) fn find(&self, name: &str) -> std::result::Result<Option<String>, ApiError> {
        let known = lock(&self.ids).iter().find(|id| matches(id, name)).cloned();
        if known.is_some() {
            return Ok(known);
        }
        let ids = self.scan().map_err(scan_error)?;
        Ok(ids.into_iter().find(|id| matches(id, name)))
    }
    /// The weights behind `id`: a custom model's base, else `id`.
    pub(crate) fn base(&self, id: &str) -> String {
        match custom::load(&self.root, id) {
            Ok(Some(custom)) => custom.base,
            _ => id.to_owned(),
        }
    }
    /// Model for a request that names none or an unknown one: the resident
    /// model, then `--default`, then the first local model.
    fn fallback(&self, name: &str) -> std::result::Result<String, ApiError> {
        let resident = lock(&self.loaded).clone();
        let id = match resident.or_else(|| self.default.clone()) {
            Some(id) => id,
            None => self
                .list()
                .map_err(scan_error)?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ApiError::new(
                        404,
                        format!(
                            "no complete local models in {}; run `ingot pull NAME` first",
                            self.root.display()
                        ),
                    )
                })?,
        };
        if !name.is_empty() {
            eprintln!("ingot serve: model '{name}' is not a local model; using {id}");
        }
        Ok(self.base(&id))
    }
}

pub(crate) fn scan_error(e: std::io::Error) -> ApiError {
    ApiError::new(500, format!("cannot list local models: {e}"))
}

fn describe(ids: &[String]) -> String {
    if ids.is_empty() {
        "none".into()
    } else {
        ids.join(", ")
    }
}

impl Served {
    /// The local model library; a server for one fixed model has none.
    pub(crate) fn local(&self) -> std::result::Result<&Local, ApiError> {
        match self {
            Self::Local(local) => Ok(local),
            Self::Fixed(_) => Err(ApiError::new(
                400,
                "managing models needs `ingot serve` without a MODEL argument",
            )),
        }
    }
    /// A vision projector is configured (`serve MODEL --mmproj`) or, for local
    /// models, an `mmproj*.gguf` lies beside the model file.
    pub(crate) fn vision(&self, id: &str, fixed: bool) -> bool {
        match self {
            Self::Fixed(_) => fixed,
            Self::Local(local) => models::resolve(&local.manifest, &local.base(id), &local.root)
                .ok()
                .and_then(|r| crate::engine::projector_for(&r.model, None).ok().flatten())
                .is_some(),
        }
    }
    pub(crate) fn list(&self) -> std::result::Result<Vec<String>, ApiError> {
        match self {
            Self::Fixed(id) => Ok(vec![id.clone()]),
            Self::Local(local) => local.list().map_err(scan_error),
        }
    }
    /// Model id for an OpenAI body. A fixed server leaves the name check to
    /// request parsing; a local server loads any local model and answers 404
    /// for others. Without a `model` string the fallback model is used.
    pub(crate) fn openai(&self, body: &[u8]) -> std::result::Result<String, ApiError> {
        let local = match self {
            Self::Fixed(id) => return Ok(id.clone()),
            Self::Local(local) => local,
        };
        let value: Option<Value> = serde_json::from_slice(body).ok();
        let Some(name) = value.as_ref().and_then(|v| v.get("model")?.as_str()) else {
            return local.fallback("");
        };
        match local.find(name)? {
            Some(id) => Ok(id),
            None => {
                let ids = local.list().map_err(scan_error)?;
                Err(ApiError {
                    status: 404,
                    message: format!(
                        "model '{name}' not found in {}; local models: {}",
                        local.root.display(),
                        describe(&ids)
                    ),
                    param: Some("model"),
                })
            }
        }
    }
    /// Model id for an Anthropic request. Names that are not local models,
    /// such as Claude Code's `claude-*`, use the fallback model.
    pub(crate) fn anthropic(&self, name: &str) -> std::result::Result<String, ApiError> {
        match self {
            Self::Fixed(id) => Ok(id.clone()),
            Self::Local(local) => match local.find(name)? {
                Some(id) => Ok(local.base(&id)),
                None => local.fallback(name),
            },
        }
    }
    /// An OpenAI body with the custom model it names, if any, applied.
    pub(crate) fn customized(&self, body: &[u8]) -> std::result::Result<Vec<u8>, ApiError> {
        let Self::Local(local) = self else {
            return Ok(body.to_vec());
        };
        let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
            return Ok(body.to_vec());
        };
        let name = value.get("model").and_then(Value::as_str).unwrap_or("");
        let found =
            custom::load(&local.root, name).map_err(|e| ApiError::new(500, e.to_string()))?;
        match found {
            Some(custom) => {
                custom::apply(&custom, &mut value);
                Ok(value.to_string().into_bytes())
            }
            None => Ok(body.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_models_are_found_named_and_used_as_fallback() {
        let root = std::env::temp_dir().join(format!("ingot-catalog-{}", std::process::id()));
        let repo = root.join("hf/Owner/Repo");
        std::fs::create_dir_all(repo.join("sub")).unwrap();
        for file in [
            "Small-Q8_0.gguf",
            "sub/Big-00001-of-00002.gguf",
            "Partial.gguf.part",
        ] {
            std::fs::write(repo.join(file), b"x").unwrap();
        }
        assert!(Local::new(root.clone(), Some("missing")).is_err());
        let served = Served::Local(Local::new(root.clone(), None).unwrap());
        let small = "hf:Owner/Repo/Small-Q8_0.gguf";
        assert_eq!(served.list().unwrap(), [small]);
        let open = |body: &str| served.openai(body.as_bytes());
        assert_eq!(open(&format!(r#"{{"model":"{small}"}}"#)).unwrap(), small);
        assert_eq!(open(r#"{"model":"small-q8_0"}"#).unwrap(), small);
        assert_eq!(open(r#"{"prompt":"no model"}"#).unwrap(), small);
        let e = open(r#"{"model":"Big"}"#).unwrap_err();
        assert_eq!(e.status, 404);
        assert!(e.message.contains(small), "{}", e.message);
        assert_eq!(served.anthropic("claude-sonnet").unwrap(), small);
        // A model pulled after startup is found without a restart.
        std::fs::write(repo.join("sub/Big-00002-of-00002.gguf"), b"x").unwrap();
        let big = "hf:Owner/Repo/sub/Big-00001-of-00002.gguf";
        assert_eq!(open(r#"{"model":"Big"}"#).unwrap(), big);
        let Served::Local(local) = &served else {
            unreachable!()
        };
        *lock(&local.loaded) = Some(big.into());
        assert_eq!(served.list().unwrap(), [big, small]);
        assert_eq!(served.anthropic("claude-haiku").unwrap(), big);
        assert_eq!(Served::Fixed("m".into()).anthropic("other").unwrap(), "m");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
