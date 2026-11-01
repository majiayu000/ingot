//! Engine worker: keeps at most one model resident, loads the model each job
//! names (after the jobs before it finished) and unloads it after
//! `--keep-alive` without requests.
use super::catalog::lock;
use super::monitor::{self, Monitor};
use super::{Event, Job, USAGE, run_job};
use crate::cli::{DEFAULT_CONTEXT, Options};
use crate::engine::{Engine, Session};
use crate::openai::ApiError;
use crate::session_pool::{self, Pool};
use crate::{Result, invalid};
use std::ffi::OsString;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Flags `serve` handles itself, each followed by a value.
const SERVE_FLAGS: [&str; 9] = [
    "--host",
    "--thinking",
    "--api-key",
    "--port",
    "--sessions",
    "--keep-alive",
    "--default",
    "--context",
    "--memory-gib",
];
/// Only meaningful without MODEL.
const LOCAL_FLAGS: [&str; 3] = ["--default", "--context", "--memory-gib"];

/// `serve` arguments: serve-level flags plus the model arguments, which
/// `Options::parse` checks exactly as `serve MODEL` always has.
pub(crate) struct Args {
    values: Vec<(String, String)>,
    model: Vec<OsString>,
    /// `--insecure`: serve a non-loopback host without an API key.
    pub(crate) insecure: bool,
    remote_images: bool,
}
impl Args {
    pub(crate) fn split(args: &[OsString]) -> Result<Self> {
        let (mut values, mut model) = (Vec::new(), Vec::new());
        let mut insecure = false;
        let mut remote_images = false;
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            if arg == "--insecure" {
                insecure = true;
                continue;
            }
            if arg == "--allow-remote-images" {
                remote_images = true;
                continue;
            }
            let Some(flag) = arg.to_str().filter(|a| SERVE_FLAGS.contains(a)) else {
                model.push(arg.clone());
                continue;
            };
            let value = rest
                .next()
                .and_then(|v| v.to_str())
                .ok_or_else(|| invalid(format!("{flag} requires a value")))?;
            if values.iter().any(|(f, _)| f == flag) {
                return Err(invalid(format!("{flag} given more than once")));
            }
            values.push((flag.to_owned(), value.to_owned()));
        }
        Ok(Self {
            values,
            model,
            insecure,
            remote_images,
        })
    }
    pub(crate) fn remote_images(&self) -> bool {
        self.remote_images
    }
    pub(crate) fn string(&self, flag: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(f, _)| f == flag)
            .map(|(_, v)| v.as_str())
    }
    pub(crate) fn get<T: std::str::FromStr>(&self, flag: &str) -> Result<Option<T>> {
        self.string(flag)
            .map(|v| {
                v.parse()
                    .map_err(|_| invalid(format!("invalid value for {flag}: {v}")))
            })
            .transpose()
    }
    /// True without MODEL: only `--no-draft` and `--prefix-cache` remain.
    pub(crate) fn local(&self) -> bool {
        self.model
            .iter()
            .all(|a| a == "--no-draft" || a == "--prefix-cache")
    }
    /// The loading source: `serve MODEL ...` resolves MODEL once now.
    pub(crate) fn source(&self) -> Result<Source> {
        if !self.local() {
            if let Some(flag) = LOCAL_FLAGS.iter().find(|f| self.string(f).is_some()) {
                return Err(invalid(format!(
                    "{flag} applies only to `serve` without MODEL; give CONTEXT and MEMORY_GIB after MODEL"
                )));
            }
            return Ok(Source::Fixed(Options::parse(&self.model, &[], 0, USAGE)?));
        }
        let context: Option<usize> = self.get("--context")?;
        let gib: Option<usize> = self.get("--memory-gib")?;
        let mut tail = Vec::new();
        if context.is_some() || gib.is_some() {
            tail.push(OsString::from(
                context.unwrap_or(DEFAULT_CONTEXT).to_string(),
            ));
        }
        if let Some(gib) = gib {
            tail.push(gib.to_string().into());
        }
        tail.extend(self.model.iter().cloned());
        Ok(Source::Local(tail))
    }
}

pub(crate) enum Source {
    Fixed(Options),
    /// Arguments after MODEL for every load.
    Local(Vec<OsString>),
}

/// `30`, `30s`, `5m` or `1h`.
pub(crate) fn duration(text: &str) -> Result<Duration> {
    let (number, unit) = match text.char_indices().find(|(_, c)| !c.is_ascii_digit()) {
        Some((i, _)) => text.split_at(i),
        None => (text, "s"),
    };
    let seconds = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => 0,
    };
    number
        .parse::<u64>()
        .ok()
        .filter(|_| seconds > 0)
        .and_then(|n| n.checked_mul(seconds))
        .map(Duration::from_secs)
        .ok_or_else(|| {
            invalid(format!(
                "invalid --keep-alive {text}; use seconds or a number with s, m or h (e.g. 5m)"
            ))
        })
}

/// GiB this process has allocated on the default Metal device.
fn metal_bytes() -> f64 {
    monitor::metal().0 as f64 / (1u64 << 30) as f64
}

struct Resident {
    id: String,
    engine: Engine,
    pool: Pool<Session>,
}

pub(crate) struct Worker {
    source: Source,
    sessions: Option<usize>,
    keep_alive: Option<Duration>,
    resident: Option<Resident>,
    /// Published resident id for request routing.
    loaded: Arc<Mutex<Option<String>>>,
    monitor: Arc<Monitor>,
}
impl Worker {
    pub(crate) fn new(
        source: Source,
        sessions: Option<usize>,
        keep_alive: Option<Duration>,
        loaded: Arc<Mutex<Option<String>>>,
        monitor: Arc<Monitor>,
    ) -> Self {
        Self {
            source,
            sessions,
            keep_alive,
            resident: None,
            loaded,
            monitor,
        }
    }
    /// Make `id` the resident model, unloading another one first.
    pub(crate) fn ensure(&mut self, id: &str) -> Result<()> {
        if self.resident.as_ref().is_some_and(|r| r.id == id) {
            return Ok(());
        }
        self.unload(&format!("switching to {id}"));
        let (start, before) = (Instant::now(), metal_bytes());
        eprintln!("ingot serve: loading {id}");
        let parsed;
        let options = match &self.source {
            Source::Fixed(options) => options,
            Source::Local(tail) => {
                let mut args = vec![OsString::from(id)];
                args.extend(tail.iter().cloned());
                parsed = Options::parse(&args, &[], 0, USAGE)?;
                &parsed
            }
        };
        let mut engine = options.load()?;
        let (sessions, state) =
            session_pool::configure(&mut engine, self.sessions, options.budget)?;
        eprintln!(
            "ingot serve: loaded {id} in {:.1}s (context {}, {sessions} kept session(s) of \
             {:.1} GiB, MTP {}; Metal allocated {before:.2} -> {:.2} GiB)",
            start.elapsed().as_secs_f64(),
            options.context,
            state as f64 / (1u64 << 30) as f64,
            if engine.has_draft() {
                "on for temperature 0"
            } else {
                "off"
            },
            metal_bytes(),
        );
        // Sessions are kept between requests so a conversation that grows by
        // appending messages only computes the new suffix, even when unrelated
        // requests arrive in between.
        let pool = Pool::new(sessions).partial(engine.truncatable());
        self.resident = Some(Resident {
            id: id.to_owned(),
            engine,
            pool,
        });
        *lock(&self.loaded) = Some(id.to_owned());
        self.monitor.sessions(sessions);
        Ok(())
    }
    fn unload(&mut self, reason: &str) {
        let Some(resident) = self.resident.take() else {
            return;
        };
        *lock(&self.loaded) = None;
        self.monitor.sessions(0);
        let (start, before) = (Instant::now(), metal_bytes());
        let id = resident.id.clone();
        objc2::rc::autoreleasepool(|_| drop(resident));
        eprintln!(
            "ingot serve: unloaded {id} ({reason}) in {:.2}s; Metal allocated {before:.2} -> {:.2} GiB",
            start.elapsed().as_secs_f64(),
            metal_bytes(),
        );
    }
    /// Run jobs in arrival order until every sender is gone.
    pub(crate) fn run(mut self, queue: Receiver<Job>) {
        loop {
            let job = match (self.keep_alive, &self.resident) {
                (Some(idle), Some(_)) => match queue.recv_timeout(idle) {
                    Ok(job) => job,
                    Err(RecvTimeoutError::Timeout) => {
                        self.unload(&format!("idle for {}s", idle.as_secs()));
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                },
                _ => match queue.recv() {
                    Ok(job) => job,
                    Err(_) => return,
                },
            };
            self.monitor.begin(&job.model);
            self.run_one(job);
            self.monitor.end();
        }
    }
}

impl Worker {
    fn run_one(&mut self, job: Job) {
        if job.cancel.is_cancelled() {
            return self.monitor.outcome("cancelled");
        }
        if let Err(e) = objc2::rc::autoreleasepool(|_| self.ensure(&job.model)) {
            eprintln!("ingot serve: cannot load {}: {e}", job.model);
            let message = format!("cannot load model {}: {e}", job.model);
            let _ = job.events.send(Event::Error(ApiError::new(500, message)));
            return self.monitor.outcome("error");
        }
        if let Some(r) = self.resident.as_mut() {
            let monitor = &self.monitor;
            objc2::rc::autoreleasepool(|_| run_job(&mut r.engine, &mut r.pool, job, monitor));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(list: &[&str]) -> Args {
        Args::split(&list.iter().map(OsString::from).collect::<Vec<_>>()).unwrap()
    }
    #[test]
    fn keep_alive_durations() {
        assert_eq!(duration("30").unwrap(), Duration::from_secs(30));
        assert_eq!(duration("45s").unwrap(), Duration::from_secs(45));
        assert_eq!(duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(duration("0").unwrap(), Duration::ZERO);
        for bad in ["", "m", "5d", "-1", "1.5m", "5mm"] {
            assert!(duration(bad).is_err(), "{bad}");
        }
    }
    #[test]
    fn serve_flags_are_split_from_model_arguments() {
        let a = args(&[
            "--port",
            "9",
            "--no-draft",
            "--keep-alive",
            "5m",
            "--insecure",
        ]);
        assert!(a.local() && a.insecure);
        assert_eq!(a.get::<u16>("--port").unwrap(), Some(9));
        assert_eq!(a.string("--keep-alive"), Some("5m"));
        let Source::Local(tail) =
            args(&["--context", "4096", "--memory-gib", "20", "--prefix-cache"])
                .source()
                .unwrap()
        else {
            panic!("expected a local source");
        };
        assert_eq!(tail, ["4096", "20", "--prefix-cache"]);
        assert!(!args(&["model.gguf", "--host", "::1"]).local());
        let e = args(&["model.gguf", "--context", "4096"])
            .source()
            .err()
            .unwrap();
        assert!(
            e.to_string().contains("only to `serve` without MODEL"),
            "{e}"
        );
        assert!(Args::split(&["--port".into()]).is_err());
        assert!(Args::split(&["--port".into(), "1".into(), "--port".into(), "2".into()]).is_err());
    }
}
