//! Hugging Face downloads with resume, SHA-256 verification and hard-link reuse.
use super::{Found, Manifest, ModelFile, colocated, file_len, gib, hf, locate};
use crate::{Result, invalid};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Downloads or verifies `name` below `root`; returns the primary file path.
/// `name` may also be an `hf:` spec for any Hugging Face GGUF.
pub fn pull(manifest: &Manifest, name: &str, root: &Path, endpoint: &str) -> Result<PathBuf> {
    if let Some(spec) = hf::Spec::parse(name) {
        return hf::pull(&spec?, root, endpoint);
    }
    let model = manifest.get(name)?;
    let found = locate(root, &manifest.file_names())?;
    // Keep an existing layout: shards must stay next to the primary file.
    let local = colocated(&found, model);
    let dir = local
        .first()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join(&model.name));
    fs::create_dir_all(&dir)?;
    for f in &model.files {
        let target = dir.join(f.name());
        fetch(f, &target, endpoint, || reuse(manifest, f, &found, &target))?;
    }
    Ok(dir.join(model.files[0].name()))
}

/// Verifies an existing `target`, or tries `reuse` and then downloads it.
pub(super) fn fetch(
    file: &ModelFile,
    target: &Path,
    endpoint: &str,
    reuse: impl FnOnce() -> Result<bool>,
) -> Result<()> {
    if target.exists() {
        let len = fs::metadata(target)?.len();
        if len != file.size {
            return Err(invalid(format!(
                "{} has {len} bytes, expected {}; move it away and pull again",
                target.display(),
                file.size
            )));
        }
        verify(target, file)?;
        eprintln!("{}: verified, skipping", file.name());
    } else if !reuse()? {
        download(file, target, endpoint)?;
    }
    Ok(())
}

pub(super) fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into()
}

/// Hard-links an identical verified file (same SHA-256) found below the root.
fn reuse(manifest: &Manifest, file: &ModelFile, found: &Found, target: &Path) -> Result<bool> {
    let mut seen = HashSet::new();
    for other in manifest.models.iter().flat_map(|m| &m.files) {
        if other.sha256 != file.sha256 || other.size != file.size {
            continue;
        }
        for source in found.get(other.name()).into_iter().flatten() {
            if !seen.insert(source) || file_len(source) != Some(file.size) {
                continue;
            }
            if let Err(e) = verify(source, file) {
                eprintln!("not reusing {}: {e}", source.display());
                continue;
            }
            match fs::hard_link(source, target) {
                Ok(()) => {
                    eprintln!(
                        "{}: hard-linked identical {}",
                        file.name(),
                        source.display()
                    );
                    return Ok(true);
                }
                Err(e) => eprintln!(
                    "cannot hard-link {} ({e}); downloading instead",
                    source.display()
                ),
            }
        }
    }
    Ok(false)
}

fn verify(path: &Path, file: &ModelFile) -> Result<()> {
    let mut hasher = Sha256::new();
    hash_prefix(
        path,
        file.size,
        &mut hasher,
        &format!("verify {}", file.name()),
    )?;
    check_digest(hasher, file, path)
}

fn check_digest(hasher: Sha256, file: &ModelFile, path: &Path) -> Result<()> {
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if digest != file.sha256 {
        return Err(invalid(format!(
            "SHA-256 mismatch for {}: expected {}, got {digest}",
            path.display(),
            file.sha256
        )));
    }
    Ok(())
}

fn hash_prefix(path: &Path, len: u64, hasher: &mut Sha256, label: &str) -> Result<()> {
    let mut input = fs::File::open(path)?.take(len);
    let mut progress = Progress::new(label, len);
    let mut buffer = vec![0; 8 << 20];
    let mut done = 0;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        done += n as u64;
        progress.update(done);
    }
    progress.finish(done);
    if done != len {
        return Err(invalid(format!(
            "{} ended at {done} of {len} bytes",
            path.display()
        )));
    }
    Ok(())
}

/// Downloads into `NAME.part`, resuming from its length, then renames once the
/// SHA-256 matches. A mismatching download is removed.
fn download(file: &ModelFile, target: &Path, endpoint: &str) -> Result<()> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        file.repo,
        file.revision,
        file.path
    );
    let resume = |e: &dyn std::fmt::Display| format!("download {url}: {e}; rerun pull to resume");
    let part = target.with_file_name(format!("{}.part", file.name()));
    let mut have = match fs::metadata(&part) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
        Err(e) => return Err(e),
    };
    if have > file.size {
        eprintln!("{} is larger than expected; restarting", part.display());
        fs::remove_file(&part)?;
        have = 0;
    }
    let mut hasher = Sha256::new();
    if have > 0 {
        hash_prefix(&part, have, &mut hasher, &format!("resume {}", file.name()))?;
    }
    if have < file.size {
        let mut request = agent().get(&url);
        if have > 0 {
            request = request.header("Range", format!("bytes={have}-"));
        }
        let response = request.call().map_err(|e| io::Error::other(resume(&e)))?;
        let mut output = OpenOptions::new().create(true).append(true).open(&part)?;
        match response.status().as_u16() {
            206 => {
                let range = response
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                if !range.starts_with(&format!("bytes {have}-")) {
                    return Err(invalid(format!(
                        "download {url}: unexpected Content-Range {range:?} for offset {have}"
                    )));
                }
            }
            200 if have > 0 => {
                eprintln!(
                    "{}: server ignored the resume range; restarting",
                    file.name()
                );
                output.set_len(0)?;
                hasher = Sha256::new();
                have = 0;
            }
            200 => {}
            status => return Err(invalid(format!("download {url}: HTTP status {status}"))),
        }
        let mut body = response.into_body().into_reader();
        let mut progress = Progress::new(&format!("download {}", file.name()), file.size);
        let mut buffer = vec![0; 1 << 20];
        loop {
            let n = body
                .read(&mut buffer)
                .map_err(|e| io::Error::new(e.kind(), resume(&e)))?;
            if n == 0 {
                break;
            }
            have += n as u64;
            if have > file.size {
                return Err(invalid(format!(
                    "download {url}: more than {} bytes",
                    file.size
                )));
            }
            output.write_all(&buffer[..n])?;
            hasher.update(&buffer[..n]);
            progress.update(have);
        }
        progress.finish(have);
        output.sync_all()?;
        if have != file.size {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                resume(&format!("stopped at {have} of {} bytes", file.size)),
            ));
        }
    }
    if let Err(e) = check_digest(hasher, file, &part) {
        fs::remove_file(&part)?;
        return Err(invalid(format!("{e}; the partial download was removed")));
    }
    fs::rename(&part, target)?;
    eprintln!("{}: downloaded and verified", file.name());
    Ok(())
}

struct Progress {
    label: String,
    total: u64,
    start: Instant,
    last: Instant,
}
impl Progress {
    fn new(label: &str, total: u64) -> Self {
        let now = Instant::now();
        Self {
            label: label.into(),
            total,
            start: now,
            last: now,
        }
    }
    fn update(&mut self, done: u64) {
        if self.last.elapsed() >= Duration::from_millis(500) {
            self.last = Instant::now();
            self.print(done);
        }
    }
    fn print(&self, done: u64) {
        let seconds = self.start.elapsed().as_secs_f64().max(1e-3);
        eprint!(
            "\r{}: {} / {} ({:.1}%) {:.1} MiB/s   ",
            self.label,
            gib(done),
            gib(self.total),
            done as f64 * 100.0 / self.total.max(1) as f64,
            done as f64 / seconds / (1 << 20) as f64
        );
    }
    fn finish(&self, done: u64) {
        // Short operations print nothing; longer ones end their progress line.
        if self.start.elapsed() >= Duration::from_millis(500) {
            self.print(done);
            eprintln!();
        }
    }
}
