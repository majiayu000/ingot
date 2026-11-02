//! Request ownership and generation. A failed/cancelled session cannot be reused.
use crate::model::{Model, State};
use crate::sampling::{Sampler, Sampling};
use crate::{Result, gguf::Gguf, glm::Glm, invalid, qwen::Qwen, tokenizer::Tokenizer};
use std::cell::Cell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
#[path = "engine_vision.rs"]
mod vision;
pub use vision::{ACTIVATION_RESERVE, projector_for};
// Search only the committed past, after the trained head supplies the first guess.
// The last two consumed tokens plus current/guess form an exact four-token key.
pub(crate) fn history_continuation(
    history: &[u32],
    current: u32,
    neural: u32,
    count: usize,
) -> &[u32] {
    if history.len() < 5 || count == 0 {
        return &[];
    }
    let key = [
        history[history.len() - 2],
        history[history.len() - 1],
        current,
        neural,
    ];
    history
        .windows(4)
        .enumerate()
        .rev()
        .find(|(i, window)| *window == key && i + 4 < history.len())
        .map(|(i, _)| &history[i + 4..(i + 4 + count).min(history.len())])
        .unwrap_or(&[])
}

#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stop {
    Eos,
    Length,
    Cancelled,
}
#[derive(Debug, Clone)]
pub struct DraftConfig {
    pub model: PathBuf,
    pub max_tokens: usize,
    pub vocabulary: Option<PathBuf>,
}
#[derive(Debug, Default)]
pub struct MtpRound {
    pub proposed: usize,
    pub accepted: usize,
    pub history_proposed: usize,
    pub history_accepted: usize,
    pub committed: usize,
    pub verification_tokens: usize,
    pub replayed_tokens: usize,
    pub snapshot_bytes: usize,
    pub head: Duration,
    pub verify: Duration,
    pub snapshot: Duration,
    /// Host encoding only; GPU checkpoint copies are included in verify time.
    pub snapshot_encode: Duration,
    pub replay: Duration,
    pub catch_up: Duration,
    pub total: Duration,
}
#[cfg(test)]
pub(crate) fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite logits"));
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.partial_cmp(b).unwrap().then_with(|| ib.cmp(ia)))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| invalid("empty logits"))
}
#[derive(Debug, Default)]
pub struct Statistics {
    pub prefill_block_tokens: usize,
    pub load: Duration,
    pub prefill: Duration,
    pub decode: Vec<Duration>,
    pub first_token: Duration,
    /// Time until the first nonempty token piece is passed to the output callback.
    pub first_visible: Option<Duration>,
    pub total: Duration,
    pub reserved_bytes: usize,
    pub metal_bytes: usize,
    /// High water mark since model load, sampled after each Metal buffer allocation.
    pub metal_peak_bytes: usize,
    pub file_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub read_time: Duration,
    /// PLE row-gather wall time; excludes row index calculation and GPU upload.
    pub ple_read_time: Duration,
    pub ple_prefill_read_time: Duration,
    pub prefix_reused_tokens: usize,
    pub prefix_saved_tokens: usize,
    pub prefix_restore: Duration,
    pub prefix_save: Duration,
    pub workspace_allocations: u64,
    pub workspace_reuses: u64,
    pub workspace_bytes: usize,
    pub workspace_peak_bytes: usize,
    pub mtp_enabled: bool,
    pub mtp_rounds: Vec<MtpRound>,
}
pub struct Session {
    pub(crate) state: State,
    consumed: Vec<u32>,
    output: Vec<u32>,
    pending: Option<u32>,
    valid: bool,
    cancel: Cancellation,
    live: Rc<Cell<usize>>,
}
impl Session {
    pub fn consumed(&self) -> &[u32] {
        &self.consumed
    }
    pub fn output(&self) -> &[u32] {
        &self.output
    }
    pub fn pending(&self) -> Option<u32> {
        self.pending
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
/// Chooses each generated token from the full logits, e.g. to keep the output
/// inside a grammar. Implementations may change the logits.
pub trait Constraint {
    fn choose(&mut self, logits: &mut [f32], sampler: &mut Sampler) -> Result<u32>;
}
fn select(
    sampler: &mut Sampler,
    constraint: &mut Option<&mut dyn Constraint>,
    mut logits: Vec<f32>,
) -> Result<u32> {
    match constraint {
        Some(c) => c.choose(&mut logits, sampler),
        None => sampler.sample(&logits),
    }
}
#[derive(Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub text: String,
    pub stop: Stop,
    pub statistics: Statistics,
}
struct Prefix {
    // Engine ownership binds the model, draft, precision and fixed block semantics.
    state: State,
    tokens: Vec<u32>,
    next: u32,
}
pub struct Engine {
    prefix_enabled: bool,
    prefix: Option<Prefix>,
    pub tokenizer: Tokenizer,
    model: Model,
    load: Duration,
    context: usize,
    /// Sessions alive at once (each owns a full CONTEXT state) and their limit.
    live: Rc<Cell<usize>>,
    max_sessions: usize,
    vision: Option<crate::vision::Encoder>,
}
impl Engine {
    pub fn load(
        path: &Path,
        budget: usize,
        context: usize,
        draft: Option<DraftConfig>,
        prefix_cache: bool,
    ) -> Result<Self> {
        let start = Instant::now();
        let g = Gguf::open(path)?;
        let architecture = g.architecture.clone();
        if !matches!(architecture.as_str(), "qwen4exp" | "glm5-next")
            && !crate::generic::ARCHITECTURES.contains(&architecture.as_str())
        {
            return Err(invalid(format!(
                "unsupported model architecture: {architecture}; supported: qwen4exp, glm5-next, {}",
                crate::generic::ARCHITECTURES.join(", ")
            )));
        }
        let tokenizer = Tokenizer::from_gguf(&g)?;
        let model = if crate::generic::ARCHITECTURES.contains(&architecture.as_str()) {
            Model::Generic(Box::new(crate::generic::Generic::load(
                g,
                budget,
                context,
                draft.as_ref(),
                prefix_cache,
            )?))
        } else if architecture == "glm5-next" {
            Model::Glm(Box::new(Glm::load(
                g,
                budget,
                context,
                draft.as_ref(),
                prefix_cache,
            )?))
        } else {
            Model::Qwen(Box::new(Qwen::load(
                g,
                budget,
                context,
                draft.as_ref(),
                prefix_cache,
            )?))
        };
        Ok(Self {
            prefix_enabled: prefix_cache,
            prefix: None,
            tokenizer,
            model,
            load: start.elapsed(),
            context,
            live: Rc::new(Cell::new(0)),
            max_sessions: 1,
            vision: None,
        })
    }
    #[doc(hidden)]
    pub fn set_validation_trace(&mut self, position: usize, path: &Path) -> Result<()> {
        self.prefix = None;
        self.prefix_enabled = false;
        self.model.set_trace(position, path)
    }
    /// Metal bytes of one session's state, measured by allocating one.
    pub fn session_bytes(&self) -> Result<usize> {
        let before = self.model.metal_bytes();
        let state = self.model.state(self.context)?;
        let bytes = self.model.metal_bytes().saturating_sub(before);
        drop(state);
        Ok(bytes)
    }
    /// Allow `n` live sessions. Only one runs at a time (`&mut self`); the
    /// others are idle kept states the caller has budgeted memory for.
    pub fn set_max_sessions(&mut self, n: usize) {
        self.max_sessions = n.max(1);
    }
    pub fn session(&self, cancel: Cancellation) -> Result<Session> {
        if self.live.get() >= self.max_sessions {
            return Err(invalid(format!(
                "only {} live session(s) are supported",
                self.max_sessions
            )));
        }
        let state = self.model.state(self.context)?;
        self.live.set(self.live.get() + 1);
        Ok(Session {
            state,
            consumed: vec![],
            output: vec![],
            pending: None,
            valid: true,
            cancel,
            live: self.live.clone(),
        })
    }
    pub fn consume(&mut self, s: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        self.model.set_profile_phase("prefill");
        self.consume_blocks(s, tokens, Model::forward_block)
    }
    fn consume_next(&mut self, s: &mut Session, tokens: &[u32]) -> Result<u32> {
        self.consume_blocks(s, tokens, Model::forward_next)
    }
    fn consume_logits(&mut self, s: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        self.consume_blocks(s, tokens, Model::forward_logits)
    }
    /// True when kept sessions can be cut back to a shared prefix.
    pub fn truncatable(&self) -> bool {
        self.model.truncatable()
    }
    /// True when the model was loaded with an MTP draft head.
    pub fn has_draft(&self) -> bool {
        match &self.model {
            Model::Qwen(m) => m.draft_tokens().is_some(),
            Model::Glm(m) => m.draft_tokens().is_some(),
            Model::Generic(_) => false,
        }
    }
    pub fn context(&self) -> usize {
        self.context
    }
    /// Bytes the model reserved within its budget, including one session state.
    pub fn reserved_bytes(&self) -> usize {
        self.model.reserved_bytes()
    }
    fn consume_blocks<T>(
        &mut self,
        s: &mut Session,
        tokens: &[u32],
        mut forward: impl FnMut(&mut Model, &mut State, &[u32], &[u32], &Cancellation) -> Result<T>,
    ) -> Result<T> {
        if !Rc::ptr_eq(&s.live, &self.live)
            || !s.valid
            || tokens.is_empty()
            || s.consumed
                .len()
                .checked_add(tokens.len())
                .is_none_or(|n| n > self.context)
        {
            return Err(invalid("invalid session or context limit"));
        }
        let mut result = None;
        let block_size = self.model.prefill_block_size();
        for block in tokens.chunks(block_size) {
            if s.cancel.is_cancelled() {
                s.valid = false;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            match forward(&mut self.model, &mut s.state, &s.consumed, block, &s.cancel) {
                Ok(v) => {
                    result = Some(v);
                    s.consumed.extend_from_slice(block);
                    s.pending = None;
                }
                Err(e) => {
                    s.valid = false;
                    return Err(e);
                }
            }
        }
        Ok(result.unwrap())
    }
    fn prefill(&mut self, s: &mut Session, prompt: &[u32], stats: &mut Statistics) -> Result<u32> {
        self.model.set_profile_phase("prefill");
        if !self.prefix_enabled {
            return self.consume_next(s, prompt);
        }
        if s.cancel.is_cancelled() {
            s.valid = false;
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "generation cancelled",
            ));
        }
        let block = self.model.prefix_checkpoint_tokens();
        let hit = self.prefix.as_ref().filter(|p| {
            prompt.starts_with(&p.tokens)
                && (p.tokens.len() == block || p.tokens.len() == prompt.len())
        });
        let (consumed, next) = if let Some(prefix) = hit {
            let t = Instant::now();
            self.model.copy_prefix_state(&prefix.state, &mut s.state)?;
            stats.prefix_restore = t.elapsed();
            stats.prefix_reused_tokens = prefix.tokens.len();
            s.consumed.extend_from_slice(&prefix.tokens);
            (prefix.tokens.len(), prefix.next)
        } else {
            // Release the old checkpoint before allocating its replacement.
            self.prefix = None;
            let n = prompt.len().min(block);
            let next = self.consume_next(s, &prompt[..n])?;
            let t = Instant::now();
            let state = self.model.snapshot_prefix(&s.state, n)?;
            self.prefix = Some(Prefix {
                state,
                tokens: prompt[..n].to_vec(),
                next,
            });
            stats.prefix_saved_tokens = n;
            stats.prefix_save = t.elapsed();
            (n, next)
        };
        if consumed == prompt.len() {
            Ok(next)
        } else {
            self.consume_next(s, &prompt[consumed..])
        }
    }
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        cancel: Cancellation,
        output: impl FnMut(u32, &[u8]) -> Result<()>,
    ) -> Result<Generation> {
        self.generate_with(
            prompt,
            max_tokens,
            cancel,
            &Sampling::default(),
            None,
            output,
        )
    }
    /// Generate with explicit sampling. `keep` carries a live session between calls:
    /// when its consumed tokens are a strict prefix of `prompt`, only the new suffix
    /// is computed; otherwise it is dropped and a fresh session is used. A session
    /// that is still valid afterwards is stored back for the next call.
    /// MTP verification is greedy, so it is disabled for temperature > 0.
    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        cancel: Cancellation,
        sampling: &Sampling,
        keep: Option<&mut Option<Session>>,
        output: impl FnMut(u32, &[u8]) -> Result<()>,
    ) -> Result<Generation> {
        self.generate_constrained(prompt, max_tokens, cancel, sampling, keep, None, output)
    }
    /// `generate_with` where a constraint picks every token from the logits.
    /// Tokens then come from CPU logits even when greedy, and MTP is off.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_constrained(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        cancel: Cancellation,
        sampling: &Sampling,
        mut keep: Option<&mut Option<Session>>,
        mut constraint: Option<&mut dyn Constraint>,
        mut output: impl FnMut(u32, &[u8]) -> Result<()>,
    ) -> Result<Generation> {
        if max_tokens == 0
            || prompt.is_empty()
            || prompt
                .len()
                .checked_add(max_tokens)
                .is_none_or(|n| n > self.context)
        {
            return Err(invalid(
                "prompt plus output exceeds context, or empty request",
            ));
        }
        let start = Instant::now();
        let file_start = self.model.file_bytes();
        let ple_start = self.model.ple_read_time();
        let cache_start = self.model.cache_statistics();
        let workspace_start = self.model.workspace_statistics();
        let mut sampler = if sampling.needs_no_sampler() && constraint.is_none() {
            None
        } else {
            let mut sampler = Sampler::new(*sampling)?;
            sampler.remember(prompt);
            Some(sampler)
        };
        // Draft heads are trained on text positions only.
        let mtp = self.has_draft() && sampler.is_none() && !vision::has_images(prompt);
        if self.has_draft() && constraint.is_some() {
            eprintln!(
                "ingot: MTP disabled for this request: constrained decoding (structured output) \
                 selects every token from the full logits"
            );
        } else if self.has_draft() && !mtp {
            eprintln!(
                "ingot: MTP disabled for this request: speculative verification is greedy-only, \
                 so temperature {} sampling runs autoregressively",
                sampling.temperature
            );
        }
        // A kept session is reused when it holds a strict prefix of this prompt,
        // or, for truncatable (attention-only) state, cut back to the common
        // prefix; otherwise it is dropped here, before allocating a new one.
        let truncatable = self.model.truncatable();
        let reused = keep.as_mut().and_then(|k| k.take()).and_then(|mut s| {
            if !Rc::ptr_eq(&s.live, &self.live) || !s.valid {
                return None;
            }
            if s.consumed.len() < prompt.len() && prompt.starts_with(&s.consumed) {
                return Some(s);
            }
            let common = s
                .consumed
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            let common = common.min(prompt.len() - 1);
            (truncatable && common > 0).then(|| {
                s.consumed.truncate(common);
                s
            })
        });
        let mut s = match reused {
            Some(mut s) => {
                s.cancel = cancel;
                s.output.clear();
                s.pending = None;
                s
            }
            None => self.session(cancel)?,
        };
        let reused = s.consumed.len();
        let mut stats = Statistics {
            prefill_block_tokens: self.model.prefill_block_size(),
            mtp_enabled: mtp,
            load: self.load,
            reserved_bytes: self.model.reserved_bytes(),
            ..Default::default()
        };
        stats.prefix_reused_tokens = reused;
        let first = (|| {
            let (head, last) = prompt[reused..].split_at(prompt.len() - reused - 1);
            let Some(sampler) = sampler.as_mut() else {
                return if reused == 0 {
                    self.prefill(&mut s, prompt, &mut stats)
                } else {
                    self.consume_next(&mut s, &prompt[reused..])
                };
            };
            if !head.is_empty() {
                if reused == 0 {
                    self.prefill(&mut s, head, &mut stats)?;
                } else {
                    self.consume_next(&mut s, head)?;
                }
            }
            let logits = self.consume_logits(&mut s, last)?;
            select(sampler, &mut constraint, logits)
        })();
        let mut next = match first {
            Ok(l) => l,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                stats.prefill = start.elapsed();
                stats.total = start.elapsed();
                stats.metal_bytes = self.model.metal_bytes();
                stats.metal_peak_bytes = self.model.metal_peak_bytes();
                let w = self.model.workspace_statistics();
                stats.workspace_allocations = w.0 - workspace_start.0;
                stats.workspace_reuses = w.1 - workspace_start.1;
                stats.workspace_bytes = w.2;
                stats.workspace_peak_bytes = w.3;
                stats.file_bytes = self.model.file_bytes() - file_start;
                let cache = self.model.cache_statistics();
                stats.cache_hits = cache.0 - cache_start.0;
                stats.cache_misses = cache.1 - cache_start.1;
                stats.read_time = cache.2 - cache_start.2;
                stats.ple_read_time = self.model.ple_read_time() - ple_start;
                stats.ple_prefill_read_time = stats.ple_read_time;
                return Ok(Generation {
                    tokens: vec![],
                    text: String::new(),
                    stop: Stop::Cancelled,
                    statistics: stats,
                });
            }
            Err(e) => return Err(e),
        };
        self.model.set_profile_phase("decode");
        stats.prefill = start.elapsed();
        stats.ple_prefill_read_time = self.model.ple_read_time() - ple_start;
        let mut stop = Stop::Length;
        let mut ready = VecDeque::new();
        ready.push_back(next);
        // The configured draft length is a maximum. After a rejected suffix,
        // propose only one position beyond the last accepted prefix next time.
        // A fully accepted round grows it again, up to the configured maximum.
        let mut qwen_draft_limit = usize::MAX;
        for i in 0..max_tokens {
            if s.cancel.is_cancelled() {
                stop = Stop::Cancelled;
                break;
            }
            let id = ready
                .pop_front()
                .ok_or_else(|| invalid("missing verified token"))?;
            if i == 0 {
                stats.first_token = start.elapsed();
            }
            if self.tokenizer.is_eos(id) {
                stop = Stop::Eos;
                break;
            }
            s.output.push(id);
            s.pending = if ready.is_empty() { Some(id) } else { None };
            let piece = self.tokenizer.token_bytes(id)?;
            if !piece.is_empty() && stats.first_visible.is_none() {
                stats.first_visible = Some(start.elapsed());
            }
            output(id, piece)?;
            if i + 1 < max_tokens && ready.is_empty() {
                if s.cancel.is_cancelled() {
                    stop = Stop::Cancelled;
                    break;
                }
                if mtp {
                    self.model.set_profile_phase("decode");
                    let result = match (&mut self.model, &mut s.state) {
                        (Model::Qwen(m), State::Qwen(state)) => m.speculate(
                            state,
                            &mut s.consumed,
                            id,
                            (max_tokens - i - 2).min(qwen_draft_limit),
                            &s.cancel,
                            |token| self.tokenizer.is_eos(token),
                        ),
                        (Model::Glm(m), State::Glm(state)) => m.speculate(
                            state,
                            &mut s.consumed,
                            id,
                            max_tokens - i - 2,
                            &s.cancel,
                            |token| self.tokenizer.is_eos(token),
                        ),
                        _ => Err(invalid("MTP model/state mismatch")),
                    };
                    match result {
                        Ok((tokens, round)) => {
                            qwen_draft_limit = round.accepted + 1;
                            ready.extend(tokens);
                            s.pending = None;
                            stats.mtp_rounds.push(round);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                            stop = Stop::Cancelled;
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    let t = Instant::now();
                    let step = match sampler.as_mut() {
                        None => self.consume_next(&mut s, &[id]),
                        Some(sampler) => self
                            .consume_logits(&mut s, &[id])
                            .and_then(|logits| select(sampler, &mut constraint, logits)),
                    };
                    next = match step {
                        Ok(l) => l,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                            stop = Stop::Cancelled;
                            break;
                        }
                        Err(e) => return Err(e),
                    };
                    stats.decode.push(t.elapsed());
                    ready.push_back(next);
                }
            }
        }

        stats.total = start.elapsed();
        stats.metal_bytes = self.model.metal_bytes();
        stats.metal_peak_bytes = self.model.metal_peak_bytes();
        let w = self.model.workspace_statistics();
        stats.workspace_allocations = w.0 - workspace_start.0;
        stats.workspace_reuses = w.1 - workspace_start.1;
        stats.workspace_bytes = w.2;
        stats.workspace_peak_bytes = w.3;
        stats.file_bytes = self.model.file_bytes() - file_start;
        let cache = self.model.cache_statistics();
        stats.cache_hits = cache.0 - cache_start.0;
        stats.cache_misses = cache.1 - cache_start.1;
        stats.read_time = cache.2 - cache_start.2;
        stats.ple_read_time = self.model.ple_read_time() - ple_start;
        let text = self.tokenizer.decode(&s.output)?;
        let tokens = std::mem::take(&mut s.output);
        if let Some(keep) = keep
            && s.valid
        {
            *keep = Some(s);
        }
        Ok(Generation {
            tokens,
            text,
            stop,
            statistics: stats,
        })
    }
}
