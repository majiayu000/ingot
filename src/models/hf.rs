//! `hf:OWNER/REPO/PATH.gguf[@REVISION]`: any GGUF on Hugging Face, with size
//! and SHA-256 taken from the repository's LFS metadata.
use super::ModelFile;
use super::download::{agent, fetch};
use crate::{Result, invalid};
use serde_json::Value;
use std::io;
use std::path::{Path, PathBuf};

pub const PREFIX: &str = "hf:";
const DEFAULT_REVISION: &str = "main";

#[derive(Debug, PartialEq, Eq)]
pub struct Spec {
    pub repo: String,
    pub revision: String,
    /// File path inside the repository.
    pub path: String,
}

impl Spec {
    /// `None` when `arg` is not an `hf:` spec.
    pub fn parse(arg: &str) -> Option<Result<Self>> {
        let rest = arg.strip_prefix(PREFIX)?;
        let usage = || {
            invalid(format!(
                "{arg}: expected hf:OWNER/REPO/PATH.gguf[@REVISION]"
            ))
        };
        let (rest, revision) = match rest.rsplit_once('@') {
            Some((rest, revision)) if !revision.is_empty() => (rest, revision),
            Some(_) => return Some(Err(usage())),
            None => (rest, DEFAULT_REVISION),
        };
        let parts: Vec<_> = rest.split('/').collect();
        let valid = parts.len() >= 3
            && parts
                .iter()
                .all(|p| !p.is_empty() && *p != "." && *p != "..")
            && rest.ends_with(".gguf");
        if !valid {
            return Some(Err(usage()));
        }
        Some(Ok(Self {
            repo: parts[..2].join("/"),
            revision: revision.into(),
            path: parts[2..].join("/"),
        }))
    }

    /// Repository paths of every shard, the first shard first.
    pub fn shard_paths(&self) -> Vec<String> {
        let (dir, name) = match self.path.rsplit_once('/') {
            Some((dir, name)) => (format!("{dir}/"), name),
            None => (String::new(), self.path.as_str()),
        };
        let split = name
            .strip_suffix(".gguf")
            .and_then(|stem| stem.rsplit_once("-of-"))
            .and_then(|(left, count)| Some((left.rsplit_once('-')?, count)))
            .filter(|((_, index), count)| {
                [index, count]
                    .iter()
                    .all(|s| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))
            });
        match split {
            Some(((base, _), count)) => {
                let n: u32 = count.parse().unwrap_or(0);
                (1..=n)
                    .map(|i| format!("{dir}{base}-{i:05}-of-{count}.gguf"))
                    .collect()
            }
            None => vec![self.path.clone()],
        }
    }

    /// Local directory holding the shards: `ROOT/hf/OWNER/REPO/<dir of PATH>`.
    fn local(&self, root: &Path, path: &str) -> PathBuf {
        root.join("hf").join(&self.repo).join(path)
    }

    /// Local first shard when every shard is present.
    pub fn resolve(&self, root: &Path) -> Result<PathBuf> {
        let paths = self.shard_paths();
        for p in &paths {
            if !self.local(root, p).is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "{} is not downloaded; run `ingot pull {PREFIX}{}/{}` or set INGOT_MODELS",
                        self.local(root, p).display(),
                        self.repo,
                        self.path
                    ),
                ));
            }
        }
        Ok(self.local(root, &paths[0]))
    }
}

/// Specs of every complete pulled GGUF below `ROOT/hf`, one per model (its
/// first shard), sorted. Partial `.part` downloads and later shards are skipped.
pub fn downloaded(root: &Path) -> Result<Vec<String>> {
    let base = root.join("hf");
    let mut specs = Vec::new();
    let mut pending = vec![base.clone()];
    while let Some(dir) = pending.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound && dir == base => break,
            Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", dir.display()))),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if entry.file_type()?.is_dir() {
                pending.push(path);
                continue;
            }
            let Some(relative) = path.strip_prefix(&base).ok().and_then(|p| p.to_str()) else {
                continue;
            };
            let arg = format!("{PREFIX}{relative}");
            let Some(Ok(spec)) = Spec::parse(&arg) else {
                continue;
            };
            if spec.shard_paths().first() == Some(&spec.path) && spec.resolve(root).is_ok() {
                specs.push(arg);
            }
        }
    }
    specs.sort();
    Ok(specs)
}

/// Lists the files with their LFS size and SHA-256 at `spec.revision`.
fn lookup(spec: &Spec, endpoint: &str) -> Result<Vec<ModelFile>> {
    let paths = spec.shard_paths();
    let revision = spec.revision.replace('/', "%2F");
    let mut url = format!(
        "{}/api/models/{}/tree/{revision}",
        endpoint.trim_end_matches('/'),
        spec.repo
    );
    if let Some((dir, _)) = spec.path.rsplit_once('/') {
        url = format!("{url}/{dir}");
    }
    let response = agent().get(&url).call().map_err(|e| match e {
        ureq::Error::StatusCode(401 | 403) => invalid(format!(
            "{url}: HTTP access denied; {} is private or gated",
            spec.repo
        )),
        ureq::Error::StatusCode(404) => invalid(format!(
            "{url}: HTTP 404; check the repository {} and revision {}",
            spec.repo, spec.revision
        )),
        e => io::Error::other(format!("cannot list {url}: {e}")),
    })?;
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| io::Error::other(format!("cannot read {url}: {e}")))?;
    let entries: Vec<Value> = serde_json::from_str(&body)
        .map_err(|e| invalid(format!("{url}: unexpected response ({e})")))?;
    paths
        .iter()
        .map(|path| {
            let entry = entries
                .iter()
                .find(|e| e["type"] == "file" && e["path"] == path.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("{path} not found in {} at {}", spec.repo, spec.revision),
                    )
                })?;
            let lfs = &entry["lfs"];
            let (Some(sha256), Some(size)) = (lfs["oid"].as_str(), lfs["size"].as_u64()) else {
                return Err(invalid(format!(
                    "{path} in {} has no LFS SHA-256 to verify against",
                    spec.repo
                )));
            };
            if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) || size == 0 {
                return Err(invalid(format!("{path}: invalid LFS metadata")));
            }
            Ok(ModelFile {
                repo: spec.repo.clone(),
                revision: revision.clone(),
                path: path.clone(),
                size,
                sha256: sha256.to_ascii_lowercase(),
            })
        })
        .collect()
}

/// Downloads or verifies every shard of `spec`; returns the first shard path.
pub fn pull(spec: &Spec, root: &Path, endpoint: &str) -> Result<PathBuf> {
    let files = lookup(spec, endpoint)?;
    for f in &files {
        let target = spec.local(root, &f.path);
        if let Some(dir) = target.parent() {
            std::fs::create_dir_all(dir)?;
        }
        fetch(f, &target, endpoint, || Ok(false))?;
    }
    Ok(spec.local(root, &files[0].path))
}
