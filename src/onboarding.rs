//! Select a model, confirm its download, then enter the existing chat loop.
use crate::models::{self, Kind, Manifest, Model};
use crate::{Result, invalid};
use std::ffi::OsString;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::Path;

pub const USAGE: &str = "usage: ingot run [MODEL] [--yes]

Choose a model, download it if needed, and start chatting.
MODEL is a name from `ingot models` or the path to a local GGUF file.
Without MODEL, an interactive menu recommends the best model for this Mac and
prefers one that is already downloaded.
Downloads require confirmation; --yes explicitly accepts the download size.
For chat settings, use `ingot chat --help` after downloading a model.";

pub fn main(args: &[OsString]) -> Result<()> {
    let mut name = None;
    let mut yes = false;
    for arg in args {
        match arg.to_str().ok_or_else(|| invalid(USAGE))? {
            "--yes" => yes = true,
            value if value.starts_with('-') || name.is_some() => return Err(invalid(USAGE)),
            value => name = Some(value),
        }
    }
    let manifest = Manifest::builtin()?;
    let root = models::default_dir()?;
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    // Arguments are checked first: a failed probe only stops the menu.
    let mac = Mac::probe(&root);
    let selected = {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let input = stdin
            .is_terminal()
            .then_some(&mut reader as &mut dyn BufRead);
        prepare(
            &manifest,
            &root,
            name,
            yes,
            input,
            &mut io::stderr(),
            &mac,
            &endpoint,
        )?
    };
    if let Some(selected) = selected {
        let mut args = vec![OsString::from(&selected)];
        // Keep the context the recommendation was based on.
        if let (Ok(model), Ok(mac)) = (manifest.get(&selected), &mac)
            && let Some(context) = model.fitting_context(mac.cap_gib)
        {
            args.push(context.to_string().into());
        }
        crate::chat::main(&args)?;
    }
    Ok(())
}

/// What this Mac offers a model.
pub(crate) struct Mac {
    pub(crate) device: String,
    /// Engine memory available by default.
    pub(crate) cap_gib: u64,
    /// Free space on the model disk.
    pub(crate) free_bytes: u64,
}

impl Mac {
    pub(crate) fn probe(root: &Path) -> Result<Self> {
        let (device, working_set) = models::working_set()?;
        Ok(Self {
            device,
            cap_gib: models::budget_cap_gib(working_set),
            free_bytes: models::free_bytes(root)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Runs in memory at this context.
    Memory(usize),
    /// Runs with experts streamed from SSD.
    Streaming,
    TooLarge,
    NoDisk,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

pub(crate) fn fit(model: &Model, present: bool, mac: &Mac) -> Fit {
    if !present && model.download_bytes() > mac.free_bytes {
        Fit::NoDisk
    } else if let Some(context) = model.fitting_context(mac.cap_gib) {
        Fit::Memory(context)
    } else if model.streams_in(mac.cap_gib) {
        Fit::Streaming
    } else {
        Fit::TooLarge
    }
}

/// Best choice: a downloaded model that runs in memory, then a new one, then
/// slower options. Longer context wins, then manifest order; when nothing can
/// run, the smallest download.
pub(crate) fn recommended(choices: &[(&Model, bool)], mac: &Mac) -> usize {
    (0..choices.len())
        .min_by_key(|&index| {
            let (model, present) = choices[index];
            let (tier, context) = match fit(model, present, mac) {
                Fit::Memory(context) => (0, context),
                Fit::Streaming => (2, 0),
                Fit::TooLarge | Fit::NoDisk => (4, 0),
            };
            let tier = tier + !present as u8;
            let size = if tier >= 4 { model.download_bytes() } else { 0 };
            (tier, std::cmp::Reverse(context), size, index)
        })
        .unwrap_or(0)
}

pub(crate) fn installed(manifest: &Manifest, model: &Model, root: &Path) -> Result<bool> {
    match models::resolve(manifest, &model.name, root) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    manifest: &Manifest,
    root: &Path,
    name: Option<&str>,
    yes: bool,
    mut input: Option<&mut dyn BufRead>,
    output: &mut dyn Write,
    mac: &Result<Mac>,
    endpoint: &str,
) -> Result<Option<String>> {
    let (model, present) = if let Some(name) = name {
        if Path::new(name).is_file() {
            return Ok(Some(name.to_owned()));
        }
        let model = manifest.get(name)?;
        if model.kind != Kind::Main {
            return Err(invalid(format!(
                "{name} is an optional draft, not a chat model. Run `ingot run` to choose a main model."
            )));
        }
        (model, installed(manifest, model, root)?)
    } else {
        let reader = input.as_deref_mut().ok_or_else(|| {
            invalid(
                "choose a model in a terminal with `ingot run`, or specify `ingot run MODEL --yes`",
            )
        })?;
        let choices = manifest
            .models
            .iter()
            .filter(|m| m.kind == Kind::Main)
            .map(|m| Ok((m, installed(manifest, m, root)?)))
            .collect::<Result<Vec<_>>>()?;
        let mac = mac
            .as_ref()
            .map_err(|error| io::Error::new(error.kind(), error.to_string()))?;
        let default = recommended(&choices, mac);
        writeln!(
            output,
            "Welcome to Ingot! {}: {} GiB of memory for models, {:.0} GiB free on disk.",
            mac.device,
            mac.cap_gib,
            gib(mac.free_bytes)
        )?;
        let (best, best_present) = choices[default];
        match fit(best, best_present, mac) {
            Fit::Memory(_) => writeln!(output, "Recommended for this Mac: {}.\n", best.name)?,
            Fit::Streaming => writeln!(
                output,
                "Best option for this Mac: {}, streamed from SSD, so it runs slowly.\n",
                best.name
            )?,
            Fit::TooLarge | Fit::NoDisk => writeln!(
                output,
                "No listed model fits this Mac. Free disk space or use a smaller GGUF file.\n"
            )?,
        }
        for (index, (model, present)) in choices.iter().enumerate() {
            writeln!(
                output,
                "  {}. {} — {:.1} GiB{}{}{}\n     {}",
                index + 1,
                model.name,
                gib(model.download_bytes()),
                if *present {
                    ", already downloaded"
                } else {
                    " download"
                },
                match fit(model, *present, mac) {
                    Fit::Memory(_) => "",
                    Fit::Streaming => ", slow on this Mac (streams from SSD)",
                    Fit::TooLarge => ", too large for this Mac",
                    Fit::NoDisk => ", not enough disk space",
                },
                if index == default { " (default)" } else { "" },
                model.summary,
            )?;
        }
        loop {
            write!(
                output,
                "\nModel number or name [{}], q to cancel: ",
                default + 1
            )?;
            output.flush()?;
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || matches!(line.trim(), "q" | "quit") {
                writeln!(output, "Cancelled.")?;
                return Ok(None);
            }
            let value = line.trim();
            let index = if value.is_empty() {
                Some(default)
            } else {
                value
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| n.checked_sub(1))
                    .filter(|n| *n < choices.len())
                    .or_else(|| choices.iter().position(|(m, _)| m.name == value))
            };
            if let Some(index) = index {
                break choices[index];
            }
            writeln!(
                output,
                "Please enter a number from 1 to {}, a model name, or q.",
                choices.len()
            )?;
        }
    };
    if present {
        writeln!(output, "Using downloaded model {}.", model.name)?;
        return Ok(Some(model.name.clone()));
    }
    writeln!(
        output,
        "\n{} needs {:.1} GiB of model files. Existing files and partial downloads are reused.\n\
Destination: {}",
        model.name,
        gib(model.download_bytes()),
        root.display(),
    )?;
    if let Ok(mac) = mac {
        writeln!(output, "Free disk space: {:.0} GiB", gib(mac.free_bytes))?;
        match fit(model, present, mac) {
            Fit::Memory(context) => writeln!(
                output,
                "It runs in memory on this Mac, with a context of {context} tokens."
            )?,
            Fit::Streaming => writeln!(
                output,
                "Warning: it does not fit in memory here. Experts stream from SSD, so it runs slowly."
            )?,
            Fit::TooLarge => writeln!(
                output,
                "Warning: this Mac has {} GiB of memory for models, too little to run it.",
                mac.cap_gib
            )?,
            Fit::NoDisk => writeln!(output, "Warning: there may not be enough free disk space.")?,
        }
    }
    writeln!(output, "Optional draft weights will not be downloaded.")?;
    if !yes {
        let reader = input.ok_or_else(|| {
            invalid(format!(
                "download not started. Run in a terminal to confirm, or use `ingot run {} --yes`.",
                model.name
            ))
        })?;
        write!(output, "Download and start chatting? [y/N] ")?;
        output.flush()?;
        let mut answer = String::new();
        reader.read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            writeln!(output, "Cancelled. No download started.")?;
            return Ok(None);
        }
    }
    models::pull(manifest, &model.name, root, endpoint).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "{error}\nTo retry and resume a partial download, run `ingot run {}`.",
                model.name
            ),
        )
    })?;
    Ok(Some(model.name.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelFile;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::io::{Cursor, Read};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const NO_SERVER: &str = "http://127.0.0.1:9";
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "ingot-onboarding-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            models: ["small", "large", "draft"]
                .iter()
                .map(|name| Model {
                    name: (*name).into(),
                    kind: if *name == "draft" {
                        Kind::Draft
                    } else {
                        Kind::Main
                    },
                    title: String::new(),
                    summary: format!("{name} test model"),
                    resident_bytes: 4,
                    draft: None,
                    memory: vec![(2048, 2.0), (8192, 4.0)],
                    stream_gib: None,
                    files: vec![ModelFile {
                        repo: "test/model".into(),
                        revision: "a".repeat(40),
                        path: format!("{name}.gguf"),
                        size: 4,
                        sha256: format!("{:x}", Sha256::digest(b"test")),
                    }],
                })
                .collect(),
        }
    }

    fn mac() -> Mac {
        Mac {
            device: "Test Mac".into(),
            cap_gib: 64,
            free_bytes: 1 << 40,
        }
    }

    fn sized(name: &str, memory: Vec<(usize, f64)>, stream_gib: Option<f64>, gib: u64) -> Model {
        Model {
            name: name.into(),
            kind: Kind::Main,
            title: String::new(),
            summary: String::new(),
            resident_bytes: 4,
            draft: None,
            memory,
            stream_gib,
            files: vec![ModelFile {
                repo: "test/model".into(),
                revision: "a".repeat(40),
                path: format!("{name}.gguf"),
                size: gib << 30,
                sha256: "0".repeat(64),
            }],
        }
    }

    fn pick(models: &[(&Model, bool)], cap_gib: u64, free_gib: u64) -> usize {
        recommended(
            models,
            &Mac {
                device: String::new(),
                cap_gib,
                free_bytes: free_gib << 30,
            },
        )
    }

    #[test]
    fn without_metal_arguments_are_still_checked_and_the_menu_fails_clearly() {
        let dir = Temp::new();
        let no_metal = || Err(io::Error::other("Metal device unavailable"));
        let error = prepare(
            &manifest(),
            &dir.0,
            Some("small"),
            false,
            None,
            &mut Vec::new(),
            &no_metal(),
            NO_SERVER,
        )
        .unwrap_err();
        assert!(error.to_string().contains("--yes"));
        let error = prepare(
            &manifest(),
            &dir.0,
            None,
            false,
            None,
            &mut Vec::new(),
            &no_metal(),
            NO_SERVER,
        )
        .unwrap_err();
        assert!(error.to_string().contains("specify"));
        let mut input = Cursor::new("");
        let error = prepare(
            &manifest(),
            &dir.0,
            None,
            false,
            Some(&mut input),
            &mut Vec::new(),
            &no_metal(),
            NO_SERVER,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Metal device unavailable"));
    }

    #[test]
    fn recommendation_follows_memory_disk_and_downloads() {
        let huge = sized("huge", vec![(8192, 41.0)], Some(8.0), 63);
        let mid = sized("mid", vec![(2048, 5.8), (8192, 7.5)], None, 5);
        let tiny = sized("tiny", vec![(8192, 5.0)], None, 2);
        let free = |h, m, t| [(&huge, h), (&mid, m), (&tiny, t)];
        // Best listed model when it fits.
        assert_eq!(pick(&free(false, false, false), 64, 500), 0);
        // Memory rules out the first and shortens the second's context.
        assert_eq!(pick(&free(false, false, false), 8, 500), 1);
        assert_eq!(pick(&free(false, false, false), 6, 500), 2);
        // Not enough disk.
        assert_eq!(pick(&free(false, false, false), 64, 40), 1);
        // A downloaded model that fits beats a better one to download.
        assert_eq!(pick(&free(false, false, true), 64, 500), 2);
        // A downloaded model too large for memory does not beat a fitting one.
        assert_eq!(pick(&free(true, false, false), 6, 500), 2);
        // In-memory beats streaming; streaming beats nothing.
        assert_eq!(pick(&[(&huge, false), (&mid, false)], 8, 500), 1);
        let large = sized("large", vec![(8192, 30.0)], None, 30);
        assert_eq!(pick(&[(&large, false), (&huge, false)], 8, 500), 1);
        // With nothing runnable, offer the smallest download.
        assert_eq!(pick(&free(false, false, false), 1, 500), 2);
    }

    #[test]
    fn builtin_models_can_be_rated_and_small_macs_get_small_models() {
        let manifest = Manifest::builtin().unwrap();
        let mains: Vec<_> = manifest
            .models
            .iter()
            .filter(|m| m.kind == Kind::Main)
            .map(|m| (m, false))
            .collect();
        assert!(mains.iter().all(|(m, _)| !m.memory.is_empty()));
        let name = |cap| mains[pick(&mains, cap, 1000)].0.name.as_str();
        assert_eq!(name(73), "qwen3.8-iq2_xs");
        assert_eq!(name(20), "qwen3-30b-a3b-q4_k_m");
        assert_eq!(name(12), "qwen3-8b-q4_k_m");
        assert_eq!(name(6), "gemma3-4b-it-qat-q4_0");
        let phi = manifest.get("phi3-mini-4k-instruct-q4").unwrap();
        assert_eq!(phi.fitting_context(73), Some(4096));
    }

    #[test]
    fn cancel_or_eof_never_creates_a_download_directory() {
        for answer in ["", "\n", "n\n", "perhaps\n"] {
            let dir = Temp::new();
            let mut input = Cursor::new(answer);
            let mut out = Vec::new();
            assert_eq!(
                prepare(
                    &manifest(),
                    &dir.0,
                    Some("small"),
                    false,
                    Some(&mut input),
                    &mut out,
                    &Ok(mac()),
                    NO_SERVER
                )
                .unwrap(),
                None
            );
            assert!(!dir.0.exists());
            let out = String::from_utf8(out).unwrap();
            assert!(
                out.contains("GiB") && out.contains("Destination:") && out.contains("Cancelled")
            );
        }
    }

    #[test]
    fn menu_prefers_installed_main_and_recovers_from_invalid_input() {
        let dir = Temp::new();
        fs::create_dir(&dir.0).unwrap();
        fs::write(dir.0.join("large.gguf"), b"test").unwrap();
        let mut input = Cursor::new("0\n999\nunknown\n\n");
        let mut out = Vec::new();
        assert_eq!(
            prepare(
                &manifest(),
                &dir.0,
                None,
                false,
                Some(&mut input),
                &mut out,
                &Ok(mac()),
                NO_SERVER
            )
            .unwrap(),
            Some("large".into())
        );
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("already downloaded (default)"));
        assert!(out.contains("[2]"));
        assert!(!out.contains("draft"));
    }

    #[test]
    fn menu_can_cancel_and_select_by_name() {
        for answer in ["", "q\n", "quit\n", "small\nn\n", "\nn\n"] {
            let dir = Temp::new();
            let mut input = Cursor::new(answer);
            assert_eq!(
                prepare(
                    &manifest(),
                    &dir.0,
                    None,
                    false,
                    Some(&mut input),
                    &mut Vec::new(),
                    &Ok(mac()),
                    NO_SERVER
                )
                .unwrap(),
                None
            );
            assert!(!dir.0.exists());
        }
    }

    #[test]
    fn noninteractive_requires_explicit_model_and_download_consent() {
        let dir = Temp::new();
        for (name, yes, expected) in [(None, true, "specify"), (Some("small"), false, "--yes")] {
            let error = prepare(
                &manifest(),
                &dir.0,
                name,
                yes,
                None,
                &mut Vec::new(),
                &Ok(mac()),
                NO_SERVER,
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected));
            assert!(!dir.0.exists());
        }
    }

    #[test]
    fn existing_model_and_explicit_path_need_no_input_or_network() {
        let dir = Temp::new();
        fs::create_dir(&dir.0).unwrap();
        let path = dir.0.join("small.gguf");
        fs::write(&path, b"test").unwrap();
        for name in ["small", path.to_str().unwrap()] {
            assert_eq!(
                prepare(
                    &manifest(),
                    &dir.0,
                    Some(name),
                    false,
                    None,
                    &mut Vec::new(),
                    &Ok(mac()),
                    NO_SERVER
                )
                .unwrap()
                .as_deref(),
                Some(name)
            );
        }
    }

    #[test]
    fn invalid_models_and_unreadable_root_are_not_treated_as_missing_downloads() {
        let dir = Temp::new();
        for name in ["draft", "unknown"] {
            assert!(
                prepare(
                    &manifest(),
                    &dir.0,
                    Some(name),
                    true,
                    None,
                    &mut Vec::new(),
                    &Ok(mac()),
                    NO_SERVER
                )
                .is_err()
            );
            assert!(!dir.0.exists());
        }
        fs::create_dir(&dir.0).unwrap();
        let root = dir.0.join("not-a-directory");
        fs::write(&root, b"keep").unwrap();
        let error = prepare(
            &manifest(),
            &root,
            Some("small"),
            true,
            None,
            &mut Vec::new(),
            &Ok(mac()),
            NO_SERVER,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotADirectory);
        assert_eq!(fs::read(root).unwrap(), b"keep");
    }

    #[test]
    fn confirmed_download_is_verified_before_chat_handoff() {
        for (answer, data) in [("yes\n", "test"), ("y\n", "oops")] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            listener.set_nonblocking(true).unwrap();
            let server = std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < deadline,
                                "download never started"
                            );
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(error) => panic!("{error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut head = Vec::new();
                while !head.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    head.push(byte[0]);
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n{data}"
                )
                .unwrap();
                String::from_utf8(head).unwrap()
            });
            let dir = Temp::new();
            let mut input = Cursor::new(answer);
            let result = prepare(
                &manifest(),
                &dir.0,
                Some("small"),
                false,
                Some(&mut input),
                &mut Vec::new(),
                &Ok(mac()),
                &endpoint,
            );
            assert!(
                server
                    .join()
                    .unwrap()
                    .starts_with("GET /test/model/resolve/")
            );
            if data == "test" {
                assert_eq!(result.unwrap(), Some("small".into()));
                assert_eq!(fs::read(dir.0.join("small/small.gguf")).unwrap(), b"test");
                assert!(!dir.0.join("draft").exists());
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("SHA-256") && error.contains("ingot run small"));
                assert!(!dir.0.join("small/small.gguf").exists());
            }
        }
    }
}
