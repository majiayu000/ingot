//! Per-architecture dispatch for the engine: every model implements the same
//! state, forward and prefix-snapshot contract.
use crate::engine::Cancellation;
use crate::{Result, generic::Generic, glm::Glm, invalid, qwen::Qwen};
use std::path::Path;
use std::time::Duration;

pub(crate) enum Model {
    Qwen(Box<Qwen>),
    Glm(Box<Glm>),
    Generic(Box<Generic>),
}
pub(crate) enum State {
    Qwen(crate::qwen::State),
    Glm(crate::glm::State),
    Generic(crate::generic::State),
}
impl Model {
    /// True when session state can be cut back to an earlier position
    /// (attention-only KV caches; recurrent DeltaNet/KDA state cannot).
    pub(crate) fn truncatable(&self) -> bool {
        matches!(self, Self::Generic(_))
    }
    pub(crate) fn set_profile_phase(&self, phase: &'static str) {
        if let Self::Qwen(model) = self {
            model.set_profile_phase(phase);
        }
    }

    pub(crate) fn prefix_checkpoint_tokens(&self) -> usize {
        match self {
            Self::Qwen(m) => m.prefill_block_size(),
            // Save one useful, bounded prompt segment independently of the
            // internal GLM prefill block size.
            Self::Glm(_) => 1024,
            Self::Generic(m) => m.prefill_block_size(),
        }
    }
    pub(crate) fn snapshot_prefix(&self, source: &State, position: usize) -> Result<State> {
        match (self, source) {
            (Self::Qwen(m), State::Qwen(s)) => m.snapshot_prefix(s, position).map(State::Qwen),
            (Self::Glm(m), State::Glm(s)) => m.snapshot_prefix(s, position).map(State::Glm),
            (Self::Generic(m), State::Generic(s)) => {
                m.snapshot_prefix(s, position).map(State::Generic)
            }
            _ => Err(invalid("prefix model/state mismatch")),
        }
    }
    pub(crate) fn copy_prefix_state(&self, source: &State, destination: &mut State) -> Result<()> {
        match (self, source, destination) {
            (Self::Qwen(m), State::Qwen(a), State::Qwen(b)) => m.copy_prefix_state(a, b),
            (Self::Glm(m), State::Glm(a), State::Glm(b)) => m.copy_prefix_state(a, b),
            (Self::Generic(m), State::Generic(a), State::Generic(b)) => m.copy_prefix_state(a, b),
            _ => Err(invalid("prefix model/state mismatch")),
        }
    }
    pub(crate) fn state(&self, context: usize) -> Result<State> {
        match self {
            Self::Qwen(m) => m.state(context).map(State::Qwen),
            Self::Glm(m) => m.state(context).map(State::Glm),
            Self::Generic(m) => m.state(context).map(State::Generic),
        }
    }
    pub(crate) fn forward_block(
        &mut self,
        state: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<Vec<f32>> {
        match (self, state) {
            (Self::Qwen(m), State::Qwen(s)) => m.forward_block(s, history, tokens, cancel),
            (Self::Glm(m), State::Glm(s)) => m.forward_block(s, history, tokens, cancel),
            (Self::Generic(m), State::Generic(s)) => m.forward_block(s, history, tokens, cancel),
            _ => Err(invalid("model and session state mismatch")),
        }
    }
    pub(crate) fn forward_next(
        &mut self,
        state: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<u32> {
        match (self, state) {
            (Self::Qwen(m), State::Qwen(s)) => m.forward_next(s, history, tokens, cancel),
            (Self::Glm(m), State::Glm(s)) => m.forward_next(s, history, tokens, cancel),
            (Self::Generic(m), State::Generic(s)) => m.forward_next(s, history, tokens, cancel),
            _ => Err(invalid("model and session state mismatch")),
        }
    }
    pub(crate) fn forward_logits(
        &mut self,
        state: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<Vec<f32>> {
        match (self, state) {
            (Self::Qwen(m), State::Qwen(s)) => m.forward_logits(s, history, tokens, cancel),
            // GLM's block forward is already batched and projects only the last row.
            (Self::Glm(m), State::Glm(s)) => m.forward_block(s, history, tokens, cancel),
            (Self::Generic(m), State::Generic(s)) => m.forward_block(s, history, tokens, cancel),
            _ => Err(invalid("model and session state mismatch")),
        }
    }
    pub(crate) fn set_trace(&mut self, position: usize, path: &Path) -> Result<()> {
        match self {
            Self::Qwen(m) => m.set_trace(position, path),
            Self::Glm(m) => m.set_trace(position, path),
            Self::Generic(_) => Err(invalid(
                "validation traces are not available for this architecture",
            )),
        }
    }
    pub(crate) fn reserved_bytes(&self) -> usize {
        match self {
            Self::Qwen(m) => m.reserved_bytes(),
            Self::Glm(m) => m.reserved_bytes(),
            Self::Generic(m) => m.reserved_bytes(),
        }
    }
    pub(crate) fn metal_bytes(&self) -> usize {
        match self {
            Self::Qwen(m) => m.metal_bytes(),
            Self::Glm(m) => m.metal_bytes(),
            Self::Generic(m) => m.metal_bytes(),
        }
    }
    pub(crate) fn ple_read_time(&self) -> Duration {
        match self {
            Self::Qwen(m) => m.ple_read_time(),
            Self::Glm(_) | Self::Generic(_) => Duration::ZERO,
        }
    }
    pub(crate) fn file_bytes(&self) -> u64 {
        match self {
            Self::Qwen(m) => m.file_bytes(),
            Self::Glm(m) => m.file_bytes(),
            Self::Generic(m) => m.file_bytes(),
        }
    }
    pub(crate) fn metal_peak_bytes(&self) -> usize {
        match self {
            Self::Qwen(m) => m.metal_peak_bytes(),
            Self::Glm(m) => m.metal_peak_bytes(),
            Self::Generic(m) => m.metal_peak_bytes(),
        }
    }
    pub(crate) fn prefill_block_size(&self) -> usize {
        match self {
            Self::Qwen(m) => m.prefill_block_size(),
            Self::Glm(_) => crate::glm::PREFILL_BLOCK,
            Self::Generic(m) => m.prefill_block_size(),
        }
    }
    pub(crate) fn workspace_statistics(&self) -> (u64, u64, usize, usize) {
        match self {
            Self::Qwen(m) => m.workspace_statistics(),
            Self::Glm(m) => m.workspace_statistics(),
            Self::Generic(m) => m.workspace_statistics(),
        }
    }
    pub(crate) fn cache_statistics(&self) -> (u64, u64, Duration) {
        match self {
            Self::Qwen(m) => m.cache_statistics(),
            Self::Glm(m) => m.cache_statistics(),
            Self::Generic(m) => m.cache_statistics(),
        }
    }
}
