//! Live server statistics for `GET /metrics` (Prometheus text) and
//! `GET /v1/status` (JSON, polled by the web page's Monitor tab). The engine
//! worker records requests; connection threads only read.
use super::catalog::lock;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Finished requests kept for the Monitor tab.
const RECENT: usize = 20;

/// Bytes the Metal device has allocated for this process, and its
/// recommended working set.
pub(crate) fn metal() -> (u64, u64) {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    MTLCreateSystemDefaultDevice().map_or((0, 0), |d| {
        (
            d.currentAllocatedSize() as u64,
            d.recommendedMaxWorkingSetSize(),
        )
    })
}

/// One finished generation.
pub(crate) struct Record {
    pub(crate) model: String,
    pub(crate) chat: bool,
    pub(crate) prompt: usize,
    pub(crate) reused: usize,
    pub(crate) completion: usize,
    pub(crate) prefill_seconds: f64,
    pub(crate) decode_seconds: f64,
    pub(crate) finish: &'static str,
    pub(crate) mtp: bool,
    finished: u64,
}
impl Record {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: &str,
        chat: bool,
        prompt: usize,
        reused: usize,
        completion: usize,
        prefill_seconds: f64,
        decode_seconds: f64,
        finish: &'static str,
        mtp: bool,
    ) -> Self {
        Self {
            model: model.to_owned(),
            chat,
            prompt,
            reused,
            completion,
            prefill_seconds,
            decode_seconds,
            finish,
            mtp,
            finished: unix_now(),
        }
    }
    /// Prompt tokens computed (not reused) per second.
    fn prefill_rate(&self) -> f64 {
        rate(
            self.prompt.saturating_sub(self.reused),
            self.prefill_seconds,
        )
    }
    /// Tokens after the first per second of decoding.
    pub(crate) fn decode_rate(&self) -> f64 {
        rate(self.completion.saturating_sub(1), self.decode_seconds)
    }
    fn json(&self) -> Value {
        json!({"finished": self.finished, "model": self.model,
            "endpoint": if self.chat { "chat" } else { "completion" },
            "prompt_tokens": self.prompt, "reused_tokens": self.reused,
            "completion_tokens": self.completion, "prefill_seconds": self.prefill_seconds,
            "prefill_tokens_per_second": self.prefill_rate(),
            "decode_tokens_per_second": self.decode_rate(), "finish": self.finish, "mtp": self.mtp})
    }
}
fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0. {
        tokens as f64 / seconds
    } else {
        0.
    }
}
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

struct Active {
    model: String,
    started: Instant,
}
#[derive(Default)]
struct State {
    active: Option<Active>,
    /// Finished requests by outcome (finish reason, error or cancelled).
    outcomes: BTreeMap<&'static str, u64>,
    prompt_tokens: u64,
    reused_tokens: u64,
    completion_tokens: u64,
    recent: VecDeque<Record>,
    sessions: (usize, usize),
    metal_peak: u64,
}

pub(crate) struct Monitor {
    started: Instant,
    /// Resident model id, published by the worker.
    loaded: Arc<Mutex<Option<String>>>,
    /// Submitted jobs the worker has not taken yet.
    queued: AtomicUsize,
    /// Tokens generated so far by the active request.
    generated: AtomicUsize,
    thinking: &'static str,
    state: Mutex<State>,
}
impl Default for Monitor {
    fn default() -> Self {
        Self::new(Arc::default(), None)
    }
}
impl Monitor {
    pub(crate) fn new(
        loaded: Arc<Mutex<Option<String>>>,
        thinking: Option<crate::thinking::Thinking>,
    ) -> Self {
        Self {
            started: Instant::now(),
            loaded,
            queued: AtomicUsize::new(0),
            generated: AtomicUsize::new(0),
            thinking: thinking.map_or("model default", |t| t.name()),
            state: Mutex::default(),
        }
    }
    pub(crate) fn submitted(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
    }
    /// The worker took a job off the queue.
    pub(crate) fn begin(&self, model: &str) {
        // Only the worker decrements, so this cannot underflow; jobs queued
        // without `submitted` (tests) leave it at zero.
        if self.queued.load(Ordering::Relaxed) > 0 {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        }
        self.generated.store(0, Ordering::Relaxed);
        lock(&self.state).active = Some(Active {
            model: model.to_owned(),
            started: Instant::now(),
        });
    }
    pub(crate) fn token(&self) {
        self.generated.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn end(&self) {
        let mut s = lock(&self.state);
        s.active = None;
        s.metal_peak = s.metal_peak.max(metal().0);
    }
    /// A request that ended without a generation result.
    pub(crate) fn outcome(&self, outcome: &'static str) {
        *lock(&self.state).outcomes.entry(outcome).or_default() += 1;
    }
    pub(crate) fn finished(&self, r: Record, sessions: usize) {
        let mut s = lock(&self.state);
        *s.outcomes.entry(r.finish).or_default() += 1;
        s.prompt_tokens += r.prompt as u64;
        s.reused_tokens += r.reused as u64;
        s.completion_tokens += r.completion as u64;
        s.sessions.0 = sessions;
        if s.recent.len() == RECENT {
            s.recent.pop_front();
        }
        s.recent.push_back(r);
    }
    /// A model was loaded with room for `capacity` kept sessions (0: unloaded).
    pub(crate) fn sessions(&self, capacity: usize) {
        let mut s = lock(&self.state);
        s.sessions = (0, capacity);
        s.metal_peak = s.metal_peak.max(metal().0);
    }
    fn snapshot(&self) -> (std::sync::MutexGuard<'_, State>, Option<String>, u64, u64) {
        let (allocated, working_set) = metal();
        let loaded = lock(&self.loaded).clone();
        let mut s = lock(&self.state);
        s.metal_peak = s.metal_peak.max(allocated);
        (s, loaded, allocated, working_set)
    }
    pub(crate) fn status(&self) -> Value {
        let (s, loaded, allocated, working_set) = self.snapshot();
        let active = s.active.as_ref().map(|a| {
            json!({"model": a.model, "seconds": a.started.elapsed().as_secs_f64(),
                "generated_tokens": self.generated.load(Ordering::Relaxed)})
        });
        json!({
            "model": loaded,
            "uptime_seconds": self.started.elapsed().as_secs(),
            "thinking_default": self.thinking,
            "active": active,
            "queued": self.queued.load(Ordering::Relaxed),
            "memory": {"metal_allocated_bytes": allocated, "metal_peak_bytes": s.metal_peak,
                "recommended_working_set_bytes": working_set},
            "sessions": {"kept": s.sessions.0, "capacity": s.sessions.1},
            "totals": {"requests": s.outcomes.values().sum::<u64>(), "outcomes": s.outcomes,
                "prompt_tokens": s.prompt_tokens, "reused_tokens": s.reused_tokens,
                "completion_tokens": s.completion_tokens},
            "recent": s.recent.iter().rev().map(Record::json).collect::<Vec<_>>(),
        })
    }
    pub(crate) fn metrics(&self) -> String {
        let (s, loaded, allocated, working_set) = self.snapshot();
        let mut out = String::new();
        let mut metric = |name: &str, kind: &str, help: &str, samples: &[(String, f64)]| {
            let _ = writeln!(
                out,
                "# HELP ingot_{name} {help}\n# TYPE ingot_{name} {kind}"
            );
            for (labels, value) in samples {
                let _ = writeln!(out, "ingot_{name}{labels} {value}");
            }
        };
        let one = |v: f64| [(String::new(), v)];
        let outcomes: Vec<_> = s
            .outcomes
            .iter()
            .map(|(k, v)| (format!("{{outcome=\"{k}\"}}"), *v as f64))
            .collect();
        metric(
            "requests_total",
            "counter",
            "Generation requests by outcome.",
            &outcomes,
        );
        for (name, help, v) in [
            ("prompt_tokens_total", "Prompt tokens.", s.prompt_tokens),
            (
                "prompt_reused_tokens_total",
                "Prompt tokens served from kept sessions.",
                s.reused_tokens,
            ),
            (
                "completion_tokens_total",
                "Generated tokens.",
                s.completion_tokens,
            ),
        ] {
            metric(name, "counter", help, &one(v as f64));
        }
        let last = s.recent.back();
        metric(
            "prefill_tokens_per_second",
            "gauge",
            "Prefill speed of the last request.",
            &one(last.map_or(0., Record::prefill_rate)),
        );
        metric(
            "decode_tokens_per_second",
            "gauge",
            "Decode speed of the last request.",
            &one(last.map_or(0., Record::decode_rate)),
        );
        metric(
            "queue_depth",
            "gauge",
            "Requests waiting for the engine.",
            &one(self.queued.load(Ordering::Relaxed) as f64),
        );
        metric(
            "requests_active",
            "gauge",
            "Requests running on the engine.",
            &one(s.active.is_some() as u8 as f64),
        );
        let model: Vec<_> = loaded
            .iter()
            .map(|m| (format!("{{model=\"{}\"}}", label(m)), 1.))
            .collect();
        metric("model_loaded", "gauge", "The resident model.", &model);
        for (name, help, v) in [
            (
                "metal_allocated_bytes",
                "Metal memory allocated by this process.",
                allocated,
            ),
            (
                "metal_peak_allocated_bytes",
                "Highest sampled Metal allocation.",
                s.metal_peak,
            ),
            (
                "metal_recommended_working_set_bytes",
                "Metal recommended working set.",
                working_set,
            ),
        ] {
            metric(name, "gauge", help, &one(v as f64));
        }
        metric(
            "kept_sessions",
            "gauge",
            "Sessions kept for prefix reuse.",
            &one(s.sessions.0 as f64),
        );
        metric(
            "kept_session_capacity",
            "gauge",
            "Sessions the loaded model can keep.",
            &one(s.sessions.1 as f64),
        );
        metric(
            "uptime_seconds",
            "gauge",
            "Seconds since the server started.",
            &one(self.started.elapsed().as_secs() as f64),
        );
        out
    }
}
/// Prometheus label value escaping.
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn records_requests_for_metrics_and_status() {
        let loaded = Arc::new(Mutex::new(Some("m\"1".to_owned())));
        let m = Monitor::new(loaded, Some(crate::thinking::Thinking::Low));
        m.submitted();
        m.submitted();
        m.begin("m");
        m.token();
        m.finished(
            Record::new("m", true, 100, 40, 11, 0.5, 2., "stop", true),
            1,
        );
        m.end();
        m.begin("m");
        m.outcome("error");
        m.end();
        let text = m.metrics();
        for line in [
            "ingot_requests_total{outcome=\"stop\"} 1",
            "ingot_requests_total{outcome=\"error\"} 1",
            "ingot_prompt_tokens_total 100",
            "ingot_prefill_tokens_per_second 120",
            "ingot_decode_tokens_per_second 5",
            "ingot_queue_depth 0",
            "ingot_model_loaded{model=\"m\\\"1\"} 1",
            "ingot_kept_sessions 1",
        ] {
            assert!(text.lines().any(|l| l == line), "{line}\n{text}");
        }
        let status = m.status();
        assert_eq!(status["model"], "m\"1");
        assert_eq!(status["thinking_default"], "low");
        assert_eq!(status["totals"]["requests"], 2);
        assert_eq!(status["recent"][0]["decode_tokens_per_second"], 5.);
        assert!(status["active"].is_null());
    }
}
