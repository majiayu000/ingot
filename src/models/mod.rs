//! Known model manifest, local discovery and memory-budget recommendations.
mod download;
mod hf;
#[cfg(test)]
mod tests;

pub use download::pull;

use crate::gguf::Gguf;
use crate::{Result, invalid};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MANIFEST: &str = include_str!("manifest.txt");
const GIB: u64 = 1 << 30;
/// Reserve above resident weights for session state, workspace and host overhead.
const HEADROOM: u64 = 4 * GIB;
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// Directory levels searched below the model root for existing files.
const SEARCH_DEPTH: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Main,
    Draft,
}

#[derive(Debug)]
pub struct ModelFile {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
impl ModelFile {
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub kind: Kind,
    pub summary: String,
    pub resident_bytes: u64,
    pub draft: Option<String>,
    /// (context, GiB of engine memory) measured for chat.
    pub memory: Vec<(usize, f64)>,
    /// Smallest recorded budget that streams experts from SSD.
    pub stream_gib: Option<f64>,
    pub files: Vec<ModelFile>,
}
impl Model {
    pub fn download_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Longest chat context that fits `cap_gib` of engine memory.
    pub fn fitting_context(&self, cap_gib: u64) -> Option<usize> {
        self.memory
            .iter()
            .filter(|(_, gib)| *gib <= cap_gib as f64)
            .map(|(context, _)| *context)
            .max()
    }

    /// True when the engine can stream this model's experts within `cap_gib`.
    pub fn streams_in(&self, cap_gib: u64) -> bool {
        self.stream_gib.is_some_and(|gib| gib <= cap_gib as f64)
    }
}

#[derive(Debug)]
pub struct Manifest {
    pub models: Vec<Model>,
}
impl Manifest {
    pub fn builtin() -> Result<Self> {
        Self::parse(MANIFEST)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut models: Vec<Model> = Vec::new();
        let mut keys: HashSet<&str> = HashSet::new();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            let error = |m: &str| invalid(format!("model manifest line {}: {m}", n + 1));
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                if let Some(last) = models.last() {
                    check_model(last, &keys)?;
                }
                if name.is_empty() || models.iter().any(|m| m.name == name) {
                    return Err(error("empty or duplicate model name"));
                }
                models.push(Model {
                    name: name.into(),
                    kind: Kind::Main,
                    summary: String::new(),
                    resident_bytes: 0,
                    draft: None,
                    memory: Vec::new(),
                    stream_gib: None,
                    files: Vec::new(),
                });
                keys.clear();
                continue;
            }
            let model = models
                .last_mut()
                .ok_or_else(|| error("entry before the first [model]"))?;
            let (key, value) = line
                .split_once('=')
                .map(|(k, v)| (k.trim(), v.trim()))
                .ok_or_else(|| error("expected key = value"))?;
            if key != "file" && !keys.insert(key) {
                return Err(error(&format!("duplicate key {key}")));
            }
            match key {
                "kind" => {
                    model.kind = match value {
                        "main" => Kind::Main,
                        "draft" => Kind::Draft,
                        _ => return Err(error("kind must be main or draft")),
                    }
                }
                "summary" => model.summary = value.into(),
                "resident_bytes" => {
                    model.resident_bytes = value.parse().map_err(|_| error("resident_bytes"))?
                }
                "draft" => model.draft = Some(value.into()),
                "memory" => {
                    model.memory = value
                        .split_whitespace()
                        .map(|entry| {
                            let (context, gib) = entry.split_once(':')?;
                            Some((context.parse().ok()?, gib.parse().ok()?))
                        })
                        .collect::<Option<_>>()
                        .ok_or_else(|| error("memory needs CONTEXT:GIB entries"))?
                }
                "stream_gib" => {
                    model.stream_gib = Some(value.parse().map_err(|_| error("stream_gib"))?)
                }
                "file" => {
                    let fields: Vec<_> = value.split_whitespace().collect();
                    let [repo, revision, path, size, sha256] = fields[..] else {
                        return Err(error("file needs REPO REVISION PATH SIZE SHA256"));
                    };
                    let size: u64 = size.parse().map_err(|_| error("file size"))?;
                    if size == 0
                        || !repo.contains('/')
                        || revision.len() != 40
                        || sha256.len() != 64
                        || !sha256
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    {
                        return Err(error("invalid file entry"));
                    }
                    let file = ModelFile {
                        repo: repo.into(),
                        revision: revision.into(),
                        path: path.into(),
                        size,
                        sha256: sha256.into(),
                    };
                    // Shards are opened from one directory, so names must be unique.
                    if model.files.iter().any(|f| f.name() == file.name()) {
                        return Err(error("duplicate file name in model"));
                    }
                    model.files.push(file);
                }
                _ => return Err(error(&format!("unknown key {key}"))),
            }
        }
        let last = models
            .last()
            .ok_or_else(|| invalid("model manifest has no models"))?;
        check_model(last, &keys)?;
        for m in &models {
            if let Some(draft) = &m.draft
                && !models
                    .iter()
                    .any(|d| &d.name == draft && d.kind == Kind::Draft)
            {
                return Err(invalid(format!(
                    "model {}: draft {draft} is not a draft model",
                    m.name
                )));
            }
        }
        Ok(Self { models })
    }

    pub fn get(&self, name: &str) -> Result<&Model> {
        self.models.iter().find(|m| m.name == name).ok_or_else(|| {
            let names: Vec<_> = self.models.iter().map(|m| m.name.as_str()).collect();
            invalid(format!(
                "unknown model {name}; known models: {}",
                names.join(", ")
            ))
        })
    }

    fn file_names(&self) -> HashSet<&str> {
        self.models
            .iter()
            .flat_map(|m| &m.files)
            .map(ModelFile::name)
            .collect()
    }
}

fn check_model(m: &Model, keys: &HashSet<&str>) -> Result<()> {
    for key in ["kind", "summary", "resident_bytes"] {
        if !keys.contains(key) {
            return Err(invalid(format!("model {}: missing {key}", m.name)));
        }
    }
    if m.files.is_empty() || m.resident_bytes == 0 {
        return Err(invalid(format!(
            "model {}: needs files and resident_bytes",
            m.name
        )));
    }
    Ok(())
}

/// `INGOT_MODELS`, otherwise `~/.cache/ingot/models`.
pub fn default_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("INGOT_MODELS") {
        return Ok(dir.into());
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| invalid("set INGOT_MODELS or HOME, or pass --dir"))?;
    Ok(Path::new(&home).join(".cache/ingot/models"))
}

type Found = HashMap<String, Vec<PathBuf>>;

/// Finds files named in the manifest anywhere below `root`, skipping hidden
/// directories and `.ranges` sidecars. A missing root has no files.
fn locate(root: &Path, names: &HashSet<&str>) -> Result<Found> {
    let mut found = Found::new();
    let mut pending = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = pending.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound && depth == 0 => break,
            Err(e) => {
                return Err(io::Error::new(e.kind(), format!("{}: {e}", dir.display())));
            }
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if entry.file_type()?.is_dir() {
                if depth < SEARCH_DEPTH && !name.ends_with(".ranges") {
                    pending.push((entry.path(), depth + 1));
                }
            } else if names.contains(name.as_ref()) {
                found
                    .entry(name.into_owned())
                    .or_default()
                    .push(entry.path());
            }
        }
    }
    for paths in found.values_mut() {
        paths.sort();
    }
    Ok(found)
}

fn file_len(path: &Path) -> Option<u64> {
    fs::metadata(path).ok().map(|m| m.len())
}

/// First located copy of `file` with the expected size.
fn sized<'a>(found: &'a Found, file: &ModelFile) -> Option<&'a PathBuf> {
    found
        .get(file.name())?
        .iter()
        .find(|p| file_len(p) == Some(file.size))
}

/// Matching shards from one directory, in manifest order. Prefer a complete
/// copy; otherwise keep the fullest partial layout for listing and `pull`.
fn colocated(found: &Found, model: &Model) -> Vec<PathBuf> {
    let directories: BTreeSet<_> = model
        .files
        .iter()
        .filter_map(|file| found.get(file.name()))
        .flatten()
        .filter_map(|path| path.parent())
        .collect();
    let mut best = Vec::new();
    for dir in directories {
        let files: Vec<_> = model
            .files
            .iter()
            .filter_map(|file| {
                let path = dir.join(file.name());
                (file_len(&path) == Some(file.size)).then_some(path)
            })
            .collect();
        if files.len() > best.len() {
            best = files;
        }
        if best.len() == model.files.len() {
            break;
        }
    }
    best
}

/// Bytes the engine keeps resident with every expert loaded. Qwen reads the
/// PLE table by row through mmap and only loads blk.48 as an MTP head.
pub fn resident_weight_bytes(g: &Gguf) -> Result<u64> {
    let qwen = g.architecture == "qwen4exp";
    let mut total = 0u64;
    for (name, t) in &g.tensors {
        if qwen && (name.starts_with("blk.48.") || name == "per_layer_token_embd.weight") {
            continue;
        }
        total = total
            .checked_add(t.mapped_bytes()? as u64)
            .ok_or_else(|| invalid("weight bytes overflow"))?;
    }
    Ok(total)
}

/// Metal device name and `recommendedMaxWorkingSetSize`.
pub fn working_set() -> Result<(String, u64)> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| io::Error::other("Metal device unavailable"))?;
    let bytes = device.recommendedMaxWorkingSetSize();
    if bytes == 0 {
        return Err(io::Error::other(
            "Metal reported no recommended working set",
        ));
    }
    Ok((device.name().to_string(), bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub gib: usize,
    /// False when the budget cannot hold every expert and the engine streams
    /// the remaining experts from SSD through fixed slots.
    pub resident: bool,
}

/// Engine memory available when MEMORY_GIB is omitted.
pub fn budget_cap_gib(working_set: u64) -> u64 {
    working_set.saturating_sub(HEADROOM) / GIB
}

/// Free bytes on the disk that holds `path`, which need not exist yet.
pub fn free_bytes(path: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let dir = path
        .ancestors()
        .find(|p| p.exists())
        .unwrap_or(Path::new("."));
    let c_path = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| invalid("path contains a NUL byte"))?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_path is NUL-terminated and stat is valid for writes.
    if unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: statvfs returned 0, so it initialized stat.
    let stat = unsafe { stat.assume_init() };
    Ok(stat.f_bavail as u64 * stat.f_frsize)
}

/// The training context declared by a GGUF file.
pub fn context_limit(model: &Path) -> Result<Option<usize>> {
    let g = Gguf::open(model)?;
    Ok(g.optional_string("general.architecture")
        .and_then(|a| g.optional_integer(&format!("{a}.context_length"))))
}

pub fn recommend(resident_bytes: u64, working_set: u64) -> Budget {
    let full = (resident_bytes + HEADROOM).div_ceil(GIB);
    // Leave the same headroom for the OS and file cache below the Metal limit.
    let cap = budget_cap_gib(working_set);
    Budget {
        gib: full.min(cap) as usize,
        resident: full <= cap,
    }
}

impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.resident {
            write!(f, "{} GiB (all experts resident)", self.gib)
        } else {
            write!(f, "{} GiB (SSD expert streaming)", self.gib)
        }
    }
}

/// Budget when MEMORY_GIB is omitted: the Metal working set minus headroom.
/// The engine allocates only what the weights, context state and workspace
/// need within it, so a long context does not push experts to SSD streaming.
pub fn auto_budget_gib(model: &Path, draft: Option<&Path>) -> Result<usize> {
    let mut resident = resident_weight_bytes(&Gguf::open(model)?)?;
    if let Some(draft) = draft {
        resident += fs::metadata(draft)?.len();
    }
    let (device, working_set) = working_set()?;
    let budget = recommend(resident, working_set);
    let cap = budget_cap_gib(working_set);
    eprintln!(
        "memory budget: up to {cap} GiB ({device} working set {} minus headroom); weights {}{}",
        gib(working_set),
        gib(resident),
        if budget.resident {
            ""
        } else {
            ", experts streamed from SSD"
        }
    );
    Ok(cap as usize)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// Primary shard and locally available MTP draft for a CLI MODEL argument.
#[derive(Debug)]
pub struct Resolved {
    pub model: PathBuf,
    pub draft: Option<PathBuf>,
}

/// MODEL is an existing file, a main model name from the manifest found
/// below `root`, or a pulled `hf:` spec. A known model's draft is returned
/// when it is fully present.
pub fn resolve(manifest: &Manifest, arg: &str, root: &Path) -> Result<Resolved> {
    if let Some(spec) = hf::Spec::parse(arg) {
        let model = spec?.resolve(root)?;
        return Ok(Resolved { model, draft: None });
    }
    let found = locate(root, &manifest.file_names())?;
    let primary = |m: &Model| {
        let files = colocated(&found, m);
        if files.len() == m.files.len() {
            files.into_iter().next()
        } else {
            None
        }
    };
    let path = Path::new(arg);
    let (model, known) =
        if path.is_file() {
            let name = path.file_name().and_then(|n| n.to_str());
            let known = manifest
                .models
                .iter()
                .find(|m| m.kind == Kind::Main && Some(m.files[0].name()) == name);
            (path.to_path_buf(), known)
        } else {
            let m = manifest
                .get(arg)
                .map_err(|e| invalid(format!("{arg} is not a file or a known model name ({e})")))?;
            if m.kind != Kind::Main {
                return Err(invalid(format!("{arg} is a draft model, not a main model")));
            }
            let primary = primary(m).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!(
                "model {arg} is not complete in {}; run `ingot pull {arg}` or set INGOT_MODELS",
                root.display()
            ))
        })?;
            (primary, Some(m))
        };
    let draft = match known.and_then(|m| m.draft.as_deref()) {
        Some(name) => {
            let d = manifest.get(name)?;
            primary(d)
        }
        None => None,
    };
    Ok(Resolved { model, draft })
}

/// Complete local main models a server can load: manifest names in manifest
/// order, then pulled `hf:` specs that are not manifest files.
pub fn local_main(manifest: &Manifest, root: &Path) -> Result<Vec<String>> {
    let names = manifest.file_names();
    let found = locate(root, &names)?;
    let mut local: Vec<String> = manifest
        .models
        .iter()
        .filter(|m| m.kind == Kind::Main && colocated(&found, m).len() == m.files.len())
        .map(|m| m.name.clone())
        .collect();
    local.extend(
        hf::downloaded(root)?
            .into_iter()
            .filter(|spec| !names.contains(spec.rsplit('/').next().unwrap_or(spec))),
    );
    Ok(local)
}

/// Text table for `ingot models`.
pub fn list(manifest: &Manifest, root: &Path, device: &str, working_set: u64) -> Result<String> {
    let found = locate(root, &manifest.file_names())?;
    let mut out = format!(
        "Model directory: {}\nMetal working set: {} ({device} recommendedMaxWorkingSetSize)\n\n\
{:<26} {:<5} {:>10} {:>10}  {:<12} RECOMMENDED BUDGET\n",
        root.display(),
        gib(working_set),
        "NAME",
        "KIND",
        "DOWNLOAD",
        "RESIDENT",
        "STATUS"
    );
    let mut paths = Vec::new();
    let mut wrong = Vec::new();
    for m in &manifest.models {
        let present = colocated(&found, m);
        for f in &m.files {
            if sized(&found, f).is_none()
                && let Some(p) = found.get(f.name()).and_then(|p| p.first())
            {
                wrong.push(format!(
                    "  {}: {} bytes, expected {}",
                    p.display(),
                    file_len(p).unwrap_or(0),
                    f.size
                ));
            }
        }
        let status = if present.len() == m.files.len() {
            paths.push(format!("  {}: {}", m.name, present[0].display()));
            "present".to_string()
        } else if present.is_empty() {
            "missing".to_string()
        } else {
            format!("partial {}/{}", present.len(), m.files.len())
        };
        let (kind, budget) = match m.kind {
            Kind::Main => {
                let cap = budget_cap_gib(working_set);
                let budget = match (m.fitting_context(cap), m.stream_gib) {
                    (Some(context), None) => {
                        let (_, need) = m.memory.iter().find(|(c, _)| *c == context).unwrap();
                        format!("{need:.1} GiB (context {context})")
                    }
                    (None, _) if !m.memory.is_empty() && !m.streams_in(cap) => {
                        "too large for this Mac".to_string()
                    }
                    _ => recommend(m.resident_bytes, working_set).to_string(),
                };
                ("main", budget)
            }
            Kind::Draft => (
                "draft",
                format!("+{}, used by its main model", gib(m.resident_bytes)),
            ),
        };
        out += &format!(
            "{:<26} {:<5} {:>10} {:>10}  {:<12} {budget}\n",
            m.name,
            kind,
            gib(m.download_bytes()),
            gib(m.resident_bytes),
            status
        );
    }
    if !paths.is_empty() {
        out += &format!("\nPrimary files:\n{}\n", paths.join("\n"));
    }
    if !wrong.is_empty() {
        out += &format!(
            "\nFiles with unexpected size (incomplete or another revision):\n{}\n",
            wrong.join("\n")
        );
    }
    out += "\nPresence requires matching names and sizes in one directory; `ingot pull NAME` verifies SHA-256.\n\
RESIDENT excludes Qwen's per_layer_token_embd table, which is read by row through mmap.\n\
chat, serve and generate use the working set minus 4 GiB when MEMORY_GIB is omitted (QWEN_MEMORY_GIB /\n\
GLM_MEMORY_GIB for real-weight tests). Below the full value the engine keeps dense\n\
weights resident and streams routed experts from SSD through fixed slots.\n";
    Ok(out)
}

/// `ingot models [--dir PATH]` and `ingot pull NAME [--dir PATH]`.
pub fn cli(args: &[OsString]) -> Result<()> {
    let usage = "usage: ingot models [--dir PATH] | ingot pull NAME|hf:OWNER/REPO/FILE.gguf[@REVISION] [--dir PATH]";
    if args
        .get(1)
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!("{usage}");
        return Ok(());
    }
    let mut dir = None;
    let mut positional = Vec::new();
    let mut rest = args.iter().skip(1);
    while let Some(a) = rest.next() {
        if a == "--dir" {
            dir = Some(PathBuf::from(rest.next().ok_or_else(|| invalid(usage))?));
        } else {
            positional.push(a.to_str().ok_or_else(|| invalid(usage))?);
        }
    }
    let root = match dir {
        Some(dir) => dir,
        None => default_dir()?,
    };
    let manifest = Manifest::builtin()?;
    match (args.first().and_then(|a| a.to_str()), &positional[..]) {
        (Some("models"), []) => {
            let (device, working_set) = working_set()?;
            print!("{}", list(&manifest, &root, &device, working_set)?);
        }
        (Some("pull"), [name]) => {
            let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into());
            let path = pull(&manifest, name, &root, &endpoint)?;
            println!("{}", path.display());
            if name.starts_with(hf::PREFIX) {
                eprintln!("run: ingot chat {name}");
                return Ok(());
            }
            let model = manifest.get(name)?;
            if model.kind == Kind::Main {
                eprintln!("run: ingot chat {name}");
                if let Some(draft) = &model.draft {
                    eprintln!("faster decoding with the MTP draft: ingot pull {draft}");
                }
            }
        }
        _ => return Err(invalid(usage)),
    }
    Ok(())
}
