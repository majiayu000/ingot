//! Qwen's single trained NextN layer. Target and draft share operators, not weights.
use super::*;
use crate::engine::{Cancellation, DraftConfig, MtpRound, history_continuation};
use std::time::Instant;

pub(super) struct Draft {
    layer: Layer,
    embedding: Weight,
    output: Weight,
    eh: Weight,
    enorm: Weight,
    hnorm: Weight,
    head: Hc,
    pub(super) max_tokens: usize,
    vocabulary_ids: Option<Buffer>,
}
pub(super) struct DraftState {
    mixer: MixerState,
    // Immutable output of the last committed target position, never a draft prediction.
    target_hidden: Buffer,
}
// QSA writes key/value/index at pos and pooled at floor(pos/ratio), only when
// that pool becomes full. It never overwrites a committed full pool. Restoring
// the logical position and rewriting the rejected suffix is therefore enough.
// GDN and PLE do overwrite history, so every such buffer is copied below.
#[derive(Default)]
pub(super) struct Checkpoint {
    pub(super) mutable: Vec<(Buffer, Buffer, usize)>,
    pub(super) bytes: usize,
}
impl Draft {
    pub(super) fn open(
        target: &Gguf,
        c: &Config,
        config: &DraftConfig,
    ) -> Result<(Gguf, Vec<u32>)> {
        if !(1..=3).contains(&config.max_tokens) {
            return Err(invalid("Qwen draft-tokens must be 1, 2 or 3"));
        }
        let g = Gguf::open(&config.model)?;
        if g.architecture != target.architecture
            || g.integer("qwen4exp.block_count")? != c.layers + 1
            || g.integer("qwen4exp.nextn_predict_layers")? != 1
        {
            return Err(invalid("incompatible Qwen NextN model"));
        }
        // Exact tokenizer and numerical model metadata matching, excluding the
        // one extra block and its compression ratio. Names/quantization may differ.
        for (name, value) in &target.metadata {
            if (name.starts_with("tokenizer.ggml.") || name.starts_with("qwen4exp."))
                && !name.starts_with("qwen4exp.ple.")
                && name != "qwen4exp.embedding_length_per_layer_input"
                && !matches!(
                    name.as_str(),
                    "qwen4exp.block_count"
                        | "qwen4exp.nextn_predict_layers"
                        | "qwen4exp.attention.compress_ratios"
                )
                && g.metadata.get(name) != Some(value)
            {
                return Err(invalid(format!("draft metadata mismatch: {name}")));
            }
        }
        let ratios = g.integers("qwen4exp.attention.compress_ratios")?;
        if ratios.len() != c.layers + 1 || ratios[..c.layers] != c.ratios || ratios[c.layers] == 0 {
            return Err(invalid("incompatible draft attention compression"));
        }
        let embedded = if g.metadata.contains_key("strata.draft_vocab_ids") {
            g.integers("strata.draft_vocab_ids")?
                .into_iter()
                .map(|i| u32::try_from(i).map_err(invalid))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let vocabulary = if let Some(path) = &config.vocabulary {
            let raw = std::fs::read(path)?;
            if raw.is_empty() || raw.len() % 4 != 0 {
                return Err(invalid(
                    "draft-vocab requires nonempty little-endian i32 IDs",
                ));
            }
            let ids = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if !embedded.is_empty() && embedded != ids {
                return Err(invalid("draft-vocab differs from compact head mapping"));
            }
            ids
        } else {
            embedded
        };
        let vocab = target.array("tokenizer.ggml.tokens")?.len();
        let mut seen = vec![false; vocab];
        for &id in &vocabulary {
            if id as usize >= vocab || seen[id as usize] {
                return Err(invalid("invalid or duplicate draft vocabulary ID"));
            }
            seen[id as usize] = true;
        }
        if vocabulary.len() >= vocab {
            return Err(invalid(
                "draft vocabulary must be smaller than the full vocabulary",
            ));
        }
        let rows = g
            .tensor("output.weight")?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| invalid("draft output shape"))?;
        if rows != vocab && (vocabulary.is_empty() || rows != vocabulary.len()) {
            return Err(invalid(
                "compact draft head requires its complete global ID mapping",
            ));
        }
        Ok((g, vocabulary))
    }
    pub(super) fn reserved_bytes(
        g: &Gguf,
        vocabulary: &[u32],
        c: &Config,
        context: usize,
    ) -> Result<usize> {
        let weights = g.tensors.values().try_fold(0usize, |sum, t| {
            sum.checked_add(t.mapped_bytes()?)
                .ok_or_else(|| invalid("draft weight size overflow"))
        })?;
        let full_output = g.tensor("output.weight")?;
        let saved = if vocabulary.is_empty() {
            0
        } else {
            (full_output.dims[1] - vocabulary.len()) * full_output.stride
        };
        let weights = weights - saved + vocabulary.len() * 4;
        let state = Self::state_bytes(g, c, context)?;
        weights
            .checked_add(state + Self::working_bytes(c))
            .ok_or_else(|| invalid("draft memory overflow"))
    }
    pub(super) fn state_bytes(g: &Gguf, c: &Config, context: usize) -> Result<usize> {
        Ok((2 * c.kv * c.head * context
            + c.idx * context
            + c.idx
                * context.div_ceil(
                    g.integers("qwen4exp.attention.compress_ratios")?[c.layers] as usize,
                ))
            * 4
            + c.width * c.hc * 4)
    }
    pub(super) fn working_bytes(c: &Config) -> usize {
        let snapshot: usize = (0..c.layers)
            .filter(|&i| c.ratios[i] == 0)
            .map(|_| (c.nv * c.d * c.d + (c.conv - 1) * (2 * c.nk + c.nv) * c.d) * 4)
            .sum::<usize>()
            + c.ple_layers.len() * (c.ple_taps - 1) * c.ngram * c.width * c.hc * 4;
        3 * snapshot + 64 * 1024 * 1024
    }
    pub(super) fn load(
        (mut g, vocabulary): (Gguf, Vec<u32>),
        gpu: &Gpu,
        c: &Config,
        config: &DraftConfig,
    ) -> Result<Self> {
        let vocab = g.array("tokenizer.ggml.tokens")?.len();
        let embedding = Weight::load(&mut g, gpu, "token_embd.weight", &[c.width, vocab])?;
        let tensor = g.tensor("output.weight")?.clone();
        let output = if !vocabulary.is_empty() && tensor.dims == [c.width, vocab] {
            let mut t = tensor.clone();
            t.dims[1] = vocabulary.len();
            t.bytes = t.stride * vocabulary.len();
            let buffer = gpu.empty_bytes(t.bytes)?;
            // Coalesce adjacent source rows without expanding the complete head on the host.
            let mut row = 0;
            while row < vocabulary.len() {
                let mut end = row + 1;
                while end < vocabulary.len() && vocabulary[end] == vocabulary[end - 1] + 1 {
                    end += 1;
                }
                gpu.load_range(
                    &mut g,
                    &tensor,
                    vocabulary[row] as usize * t.stride,
                    &buffer,
                    row * t.stride,
                    (end - row) * t.stride,
                )?;
                row = end;
            }
            Weight { buffer, t }
        } else {
            Weight::load(
                &mut g,
                gpu,
                "output.weight",
                &[
                    c.width,
                    if vocabulary.is_empty() {
                        vocab
                    } else {
                        vocabulary.len()
                    },
                ],
            )?
        };
        let attn = Qwen::hc_load(&mut g, gpu, "blk.48.hc_attn", c, true)?;
        let ffn = Qwen::hc_load(&mut g, gpu, "blk.48.hc_ffn", c, true)?;
        let head = Qwen::hc_load(&mut g, gpu, "blk.48.nextn.hc_head", c, false)?;
        let ratio = g.integers("qwen4exp.attention.compress_ratios")?[c.layers] as usize;
        let mut w = |name: &str, shape: &[usize]| {
            Weight::load(&mut g, gpu, &format!("blk.48.{name}"), shape)
        };
        let attention = Attention {
            q: w("attn_q.weight", &[c.width, 2 * c.head * c.heads])?,
            k: w("attn_k.weight", &[c.width, c.head * c.kv])?,
            v: w("attn_v.weight", &[c.width, c.head * c.kv])?,
            out: w("attn_output.weight", &[c.head * c.heads, c.width])?,
            qn: w("attn_q_norm.weight", &[c.head])?,
            kn: w("attn_k_norm.weight", &[c.head])?,
            iq: w("indexer.q_proj.weight", &[c.width, c.idx * c.ih])?,
            ik: w("indexer.k_proj.weight", &[c.width, c.idx])?,
            iqn: w("indexer.q_norm.weight", &[c.idx])?,
            ikn: w("indexer.k_norm.weight", &[c.idx])?,
            ratio,
        };
        for norm in [&attention.qn, &attention.kn, &attention.iqn, &attention.ikn] {
            norm.require_f32()?;
        }
        let experts = Ffn {
            router: w("ffn_gate_inp.weight", &[c.width, c.experts])?,
            experts: ExpertWeights::Resident([
                w("ffn_gate_exps.weight", &[c.width, c.ff, c.experts])?,
                w("ffn_up_exps.weight", &[c.width, c.ff, c.experts])?,
                w("ffn_down_exps.weight", &[c.ff, c.width, c.experts])?,
            ]),
            sgate: w("ffn_gate_shexp.weight", &[c.width, c.sff])?,
            sup: w("ffn_up_shexp.weight", &[c.width, c.sff])?,
            sdown: w("ffn_down_shexp.weight", &[c.sff, c.width])?,
            shared: w("ffn_gate_inp_shexp.weight", &[c.width])?,
        };
        let eh = w("nextn.eh_proj.weight", &[2 * c.width, c.width])?;
        let enorm = w("nextn.enorm.weight", &[c.width])?;
        let hnorm = w("nextn.hnorm.weight", &[c.width * c.hc])?;
        enorm.require_f32()?;
        hnorm.require_f32()?;
        Ok(Self {
            layer: Layer {
                attn,
                ffn,
                mixer: Mixer::Attention(Box::new(attention)),
                experts,
                ple: None,
            },
            embedding,
            output,
            eh,
            enorm,
            hnorm,
            head,
            max_tokens: config.max_tokens,
            vocabulary_ids: if vocabulary.is_empty() {
                None
            } else {
                Some(
                    gpu.bytes(
                        &vocabulary
                            .iter()
                            .flat_map(|id| id.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?,
                )
            },
        })
    }
    pub(super) fn state(&self, gpu: &Gpu, c: &Config, context: usize) -> Result<DraftState> {
        let Mixer::Attention(a) = &self.layer.mixer else {
            unreachable!()
        };
        Ok(DraftState {
            mixer: MixerState::Attention {
                key: gpu.zeros(context * c.kv * c.head)?,
                value: gpu.zeros(context * c.kv * c.head)?,
                index: gpu.zeros(context * c.idx)?,
                pooled: gpu.zeros(context.div_ceil(a.ratio) * c.idx)?,
            },
            target_hidden: gpu.zeros(c.width * c.hc)?,
        })
    }
}
impl Qwen {
    fn draft_step(
        &self,
        state: &DraftState,
        pos: usize,
        token: u32,
        hidden: &Buffer,
        output: bool,
        cancel: &Cancellation,
    ) -> Result<Option<(u32, Buffer)>> {
        let _phase = self.gpu.profile_phase("draft");
        if cancel.is_cancelled() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "generation cancelled",
            ));
        }
        objc2::rc::autoreleasepool(|_| {
            let d = self
                .draft
                .as_ref()
                .ok_or_else(|| invalid("missing draft"))?;
            let c = &self.c;
            self.trace_active
                .set(self.trace.as_ref().is_some_and(|(p, _)| *p == pos));
            let cmd = self.gpu.command()?;
            let emb = self.gpu.output(c.width)?;
            self.point(
                &cmd,
                "embedding",
                &[&d.embedding.buffer, &emb],
                &[
                    c.width as u32,
                    1,
                    d.embedding.t.stride as u32,
                    d.embedding.t.kind,
                    token,
                ],
                c.width,
            )?;
            self.record("mtp_tok_embd", &emb);
            self.record("mtp_h_input", hidden);
            let en = self.norm(&cmd, &emb, &d.enorm, c.width, 1, 0, c.width, false)?;
            let hn = self.norm(&cmd, hidden, &d.hnorm, c.width, c.hc, 0, c.width, false)?;
            self.record("mtp_enorm", &en);
            self.record("mtp_hnorm", &hn);
            let combined = self.gpu.output(2 * c.width * c.hc)?;
            self.point(
                &cmd,
                "mtp_eh_input",
                &[&en, &hn, &combined],
                &[c.width as u32, c.hc as u32, 1],
                2 * c.width * c.hc,
            )?;
            let mut h = self.gpu.matmul(
                &cmd,
                &d.eh.buffer,
                d.eh.t.dims[0],
                d.eh.t.dims[1],
                d.eh.t.stride,
                d.eh.t.kind,
                0,
                &combined,
                c.hc,
            )?;
            self.record("mtp_eh_proj", &h);
            let (mixed, inject) = self.hc_mix(&cmd, &d.layer.attn, &h)?;
            let Mixer::Attention(a) = &d.layer.mixer else {
                unreachable!()
            };
            let attended = self.attention(&cmd, a, &state.mixer, &mixed, pos)?;
            // Only the attention cache is persistent in this layer. Catch-up does
            // not need its FFN, resulting hidden, or the full vocabulary output.
            if !output && !self.trace_active.get() {
                self.gpu.complete(cmd)?;
                self.flush_trace(c.layers)?;
                return Ok(None);
            }
            h = self.hc_add(
                &cmd,
                &h,
                &attended,
                inject
                    .as_ref()
                    .ok_or_else(|| invalid("missing draft injection"))?,
            )?;
            let (mixed, inject) = self.hc_mix(&cmd, &d.layer.ffn, &h)?;
            let (ffn, cmd) = self.ffn(cmd, &d.layer.experts, &mixed, false, cancel)?;
            h = self.hc_add(
                &cmd,
                &h,
                &ffn,
                inject
                    .as_ref()
                    .ok_or_else(|| invalid("missing draft FFN injection"))?,
            )?;
            self.record("mtp_hidden", &h);
            let (mixed, _) = self.hc_mix(&cmd, &d.head, &h)?;
            self.record("mtp_result_norm", &mixed);
            let logits = d.output.project(&self.gpu, &cmd, &mixed)?;
            self.record("mtp_logits", &logits);
            let id = self.gpu.finish_greedy(
                cmd,
                &logits,
                1,
                d.output.t.dims[1],
                d.vocabulary_ids.as_ref(),
            )?[0];
            self.flush_trace(c.layers)?;
            Ok(Some((id, h)))
        })
    }
    pub(super) fn catch_up(
        &self,
        state: &mut DraftState,
        pos: usize,
        tokens: &[u32],
        hidden: &[Buffer],
        cancel: &Cancellation,
    ) -> Result<()> {
        let _phase = self.gpu.profile_phase("draft");
        // Full scalar traces remain diagnostic; the hot path only updates persistent state.
        if self.trace.is_some() {
            for (i, &token) in tokens.iter().enumerate() {
                self.draft_step(state, pos + i, token, &state.target_hidden, false, cancel)?;
                state.target_hidden = hidden[i].clone();
            }
            return Ok(());
        }
        let d = self
            .draft
            .as_ref()
            .ok_or_else(|| invalid("missing draft"))?;
        let c = &self.c;
        for (chunk, tokens) in tokens.chunks(128).enumerate() {
            if cancel.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            let begin = chunk * 128;
            let count = tokens.len();
            objc2::rc::autoreleasepool(|_| -> Result<()> {
                let cmd = self.gpu.command()?;
                let mut previous = vec![state.target_hidden.clone()];
                previous.extend(hidden[begin..begin + count - 1].iter().cloned());
                let previous = self.pack(&cmd, &previous, c.width * c.hc)?;
                let ids = self.gpu.bytes(
                    &tokens
                        .iter()
                        .flat_map(|id| id.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let emb = self.gpu.output(c.width * count)?;
                self.point(
                    &cmd,
                    "embedding_batch",
                    &[&d.embedding.buffer, &ids, &emb],
                    &[
                        c.width as u32,
                        d.embedding.t.stride as u32,
                        d.embedding.t.kind,
                        count as u32,
                    ],
                    c.width * count,
                )?;
                let en = self.norm(&cmd, &emb, &d.enorm, c.width, count, 0, c.width, false)?;
                let hn = self.norm(
                    &cmd,
                    &previous,
                    &d.hnorm,
                    c.width,
                    c.hc * count,
                    0,
                    c.width,
                    false,
                )?;
                let combined = self.gpu.output(2 * c.width * c.hc * count)?;
                self.point(
                    &cmd,
                    "mtp_eh_input",
                    &[&en, &hn, &combined],
                    &[c.width as u32, c.hc as u32, count as u32],
                    2 * c.width * c.hc * count,
                )?;
                let h = if count > 4 {
                    self.gpu.matmul(
                        &cmd,
                        &d.eh.buffer,
                        d.eh.t.dims[0],
                        d.eh.t.dims[1],
                        d.eh.t.stride,
                        d.eh.t.kind,
                        0,
                        &combined,
                        c.hc * count,
                    )?
                } else {
                    let h = self.gpu.output(c.width * c.hc * count)?;
                    // Keep the scalar F32 lane order even when there are >= 8 HC rows.
                    self.gpu.op(
                        &cmd,
                        "quant_matmul",
                        &[&d.eh.buffer, &combined, &h],
                        &[
                            d.eh.t.dims[0] as u32,
                            d.eh.t.dims[1] as u32,
                            d.eh.t.stride as u32,
                            d.eh.t.kind,
                            0,
                            (c.hc * count) as u32,
                        ],
                        c.width,
                        (c.hc * count).div_ceil(4),
                    )?;
                    h
                };
                let (mixed, _) = self.hc_packed(&cmd, &d.layer.attn, &h, count)?;
                let Mixer::Attention(a) = &d.layer.mixer else {
                    unreachable!()
                };
                self.attention_cache_packed(&cmd, a, &state.mixer, &mixed, pos + begin, count)?;
                self.gpu.complete(cmd)
            })?;
            state.target_hidden = hidden[begin + count - 1].clone();
        }
        Ok(())
    }
    pub(super) fn save_buffer(
        &self,
        cmd: &Command,
        buffer: &Buffer,
        saved: &mut Checkpoint,
    ) -> Result<()> {
        let copy = self.copy(cmd, buffer, buffer.length() / 4, 0)?;
        saved.bytes += buffer.length();
        saved.mutable.push((buffer.clone(), copy, 0));
        Ok(())
    }
    fn restore(&self, checkpoint: &Checkpoint) -> Result<()> {
        let cmd = self.gpu.command()?;
        for (destination, source, offset) in &checkpoint.mutable {
            self.point(
                &cmd,
                "copy_range",
                &[source, destination],
                &[(destination.length() / 4) as u32, *offset as u32],
                destination.length() / 4,
            )?;
        }
        self.gpu.complete(cmd)
    }
    pub(crate) fn snapshot_prefix(&self, source: &State, position: usize) -> Result<State> {
        let mut saved = self.state(position)?;
        self.copy_prefix_state(source, &mut saved)?;
        Ok(saved)
    }
    pub(crate) fn copy_prefix_state(&self, source: &State, destination: &mut State) -> Result<()> {
        let cmd = self.gpu.command()?;
        let copy = |a: &Buffer, b: &Buffer| {
            let n = a.length().min(b.length()) / 4;
            self.point(&cmd, "copy_range", &[a, b], &[n as u32, 0], n)
        };
        let mixer = |a: &MixerState, b: &MixerState| -> Result<()> {
            match (a, b) {
                (
                    MixerState::Linear {
                        history: ah,
                        state: as_,
                    },
                    MixerState::Linear {
                        history: bh,
                        state: bs,
                    },
                ) => {
                    copy(ah, bh)?;
                    copy(as_, bs)?;
                }
                (
                    MixerState::Attention {
                        key: ak,
                        value: av,
                        index: ai,
                        pooled: ap,
                    },
                    MixerState::Attention {
                        key: bk,
                        value: bv,
                        index: bi,
                        pooled: bp,
                    },
                ) => {
                    copy(ak, bk)?;
                    copy(av, bv)?;
                    copy(ai, bi)?;
                    copy(ap, bp)?;
                }
                _ => unreachable!("prefix states belong to the same immutable model"),
            }
            Ok(())
        };
        for (a, b) in source.layers.iter().zip(&destination.layers) {
            mixer(&a.mixer, &b.mixer)?;
            if let (Some(a), Some(b)) = (&a.ple, &b.ple) {
                copy(a, b)?;
            }
        }
        if let (Some(a), Some(b)) = (&source.draft, &destination.draft) {
            mixer(&a.mixer, &b.mixer)?;
            copy(&a.target_hidden, &b.target_hidden)?;
        }
        self.gpu.complete(cmd)
    }
    pub(crate) fn draft_tokens(&self) -> Option<usize> {
        self.draft.as_ref().map(|d| d.max_tokens)
    }
    pub(crate) fn speculate(
        &mut self,
        s: &mut State,
        history: &mut Vec<u32>,
        current: u32,
        max_draft: usize,
        cancel: &Cancellation,
        is_eos: impl Fn(u32) -> bool,
    ) -> Result<(Vec<u32>, MtpRound)> {
        let start = Instant::now();
        let mut stats = MtpRound::default();
        let count = self
            .draft_tokens()
            .ok_or_else(|| invalid("missing draft"))?
            .min(max_draft);
        let mut tokens = vec![current];
        let d = s
            .draft
            .as_ref()
            .ok_or_else(|| invalid("missing draft state"))?;
        let mut h = d.target_hidden.clone();
        let head_start = Instant::now();
        for i in 0..count {
            let (token, hidden) = self
                .draft_step(d, history.len() + i, tokens[i], &h, true, cancel)?
                .ok_or_else(|| invalid("missing draft result"))?;
            tokens.push(token);
            h = hidden;
            if is_eos(token) {
                break;
            }
            if i == 0 {
                for &past in history_continuation(history, current, token, count - 1) {
                    tokens.push(past);
                    stats.history_proposed += 1;
                    if is_eos(past) {
                        break;
                    }
                }
                if stats.history_proposed != 0 {
                    break;
                }
            }
        }
        stats.head = head_start.elapsed();
        let count = tokens.len() - 1;
        stats.proposed = count;
        let t = Instant::now();
        self.gpu.set_profile_phase("verify");
        let verified = self.target_block(s, history, &tokens, cancel, true)?;
        stats.verify = t.elapsed();
        stats.snapshot_encode = verified.snapshot_encode;
        stats.snapshot_bytes = verified.snapshots.iter().map(|c| c.bytes).sum();
        stats.verification_tokens = tokens.len();
        let choices = verified.choices(&self.gpu)?;
        let accepted = tokens[1..]
            .iter()
            .zip(&choices)
            .take_while(|(a, b)| a == b)
            .count();
        stats.accepted = accepted;
        stats.history_accepted = accepted.saturating_sub(1).min(stats.history_proposed);
        let committed = 1 + tokens[1..=accepted]
            .iter()
            .take_while(|&&token| !is_eos(token))
            .count();
        let output = choices[..committed].to_vec();
        if committed < tokens.len() {
            let t = Instant::now();
            let saved = &verified.snapshots[committed - 1];
            self.restore(saved)?;
            stats.snapshot = t.elapsed();
            stats.snapshot_bytes += saved.bytes;
        }
        let t = Instant::now();
        self.catch_up(
            s.draft
                .as_mut()
                .ok_or_else(|| invalid("missing draft state"))?,
            history.len(),
            &tokens[..committed],
            &verified.hidden[..committed],
            cancel,
        )?;
        stats.catch_up = t.elapsed();
        history.extend_from_slice(&tokens[..committed]);
        stats.committed = committed;
        stats.total = start.elapsed();
        Ok((output, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires QWEN_MODEL and Apple Silicon; validates real recurrent and QSA rollback"]
    fn every_verification_position_and_rollback_prefix_matches_sequential() {
        let positions = if std::env::var_os("QWEN_8K_MTP").is_some() {
            vec![8191, 8192, 8195]
        } else if std::env::var_os("QWEN_LONG_MTP").is_some() {
            vec![2047, 2052, 2055, 2112]
        } else {
            vec![3, 7, 8, 15]
        };
        check_verification_positions(positions, 44 * (1usize << 30), false, None);
    }

    #[test]
    #[ignore = "requires QWEN_MODEL and Apple Silicon; fixed64GiB 4096-block8K logits/hidden/rollback"]
    fn every_8k_verification_position_and_rollback_prefix_matches_sequential_64g() {
        check_verification_positions(vec![8191, 8192, 8195], 64 * (1usize << 30), true, None);
    }

    #[test]
    #[ignore = "requires QWEN_MODEL and Apple Silicon; fixed8GiB cached logits/hidden/rollback"]
    fn cached_8g_short_prefix_all_positions_and_rollback() {
        let draft = DraftConfig {
            model: PathBuf::from(
                std::env::var_os("QWEN_DRAFT")
                    .expect("QWEN_DRAFT must name the matching real Q4 MTP model"),
            ),
            max_tokens: 3,
            vocabulary: None,
        };
        check_verification_positions(vec![3, 63, 64, 65], 8 * (1usize << 30), false, Some(draft));
    }

    fn check_verification_positions(
        positions: Vec<usize>,
        budget: usize,
        large_block: bool,
        draft: Option<DraftConfig>,
    ) {
        let path = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
        let g = Gguf::open(&path).unwrap();
        let context = positions.iter().max().unwrap() + 16;
        let mut q = Qwen::load(g, budget, context, draft.as_ref(), false).unwrap();
        if large_block {
            println!(
                "STRICT_8K budget_bytes={budget} context={context} prefill_block_tokens={}",
                q.prefill_block_size()
            );
            assert_eq!(q.prefill_block_size(), 4096);
        }
        let cancel = Cancellation::default();
        let tail = [109266, 248044, 96748, 1];
        for pos in positions {
            let mut state = q.state(context).unwrap();
            let mut prefix = Vec::new();
            let input: Vec<_> = [1, 109266, 96748, 198]
                .into_iter()
                .cycle()
                .take(pos)
                .collect();
            for block in input.chunks(q.prefill_block_size()) {
                q.target_block(&mut state, &prefix, block, &cancel, false)
                    .unwrap();
                prefix.extend_from_slice(block);
            }
            let cmd = q.gpu.command().unwrap();
            let mut before = Checkpoint::default();
            for l in &state.layers {
                if let MixerState::Linear { history, state } = &l.mixer {
                    q.save_buffer(&cmd, history, &mut before).unwrap();
                    q.save_buffer(&cmd, state, &mut before).unwrap();
                }
                if let Some(p) = &l.ple {
                    q.save_buffer(&cmd, p, &mut before).unwrap();
                }
            }
            q.gpu.complete(cmd).unwrap();
            let mut history = prefix.clone();
            let mut expected = Vec::new();
            let mut hidden = Vec::new();
            for token in tail {
                let f = q
                    .target_block(&mut state, &history, &[token], &cancel, true)
                    .unwrap();
                expected.push(f.read_logits(&q.gpu).unwrap()[0].clone());
                hidden.push(q.gpu.read_f32(&f.hidden[0]).unwrap());
                history.push(token);
            }
            q.restore(&before).unwrap();
            drop(before);
            let f = q
                .target_block(&mut state, &prefix, &tail, &cancel, true)
                .unwrap();
            assert_eq!(
                f.read_logits(&q.gpu).unwrap(),
                expected,
                "all logits at prefix {pos}"
            );
            for (actual, expected) in f.hidden.iter().zip(&hidden) {
                assert_eq!(q.gpu.read_f32(actual).unwrap(), *expected);
            }
            // Accepted draft counts 0,1,2: current token is always verified.
            for committed in 1..tail.len() {
                q.restore(&f.snapshots[committed - 1]).unwrap();
                let mut history = prefix.clone();
                history.extend_from_slice(&tail[..committed]);
                for j in committed..tail.len() {
                    let next = q
                        .target_block(&mut state, &history, &[tail[j]], &cancel, false)
                        .unwrap();
                    assert_eq!(
                        next.read_logits(&q.gpu).unwrap()[0],
                        expected[j],
                        "rollback prefix={pos}, committed={committed}, future={j}"
                    );
                    assert_eq!(q.gpu.read_f32(&next.hidden[0]).unwrap(), hidden[j]);
                    history.push(tail[j]);
                }
            }
            println!(
                "PASS all-position logits/hidden and rollback 0/1/2/3 accepted at prefix {pos}"
            );
        }
    }
}

#[cfg(test)]
mod history_tests {
    use super::history_continuation;
    #[test]
    fn uses_only_committed_context_after_the_neural_guess() {
        let past = [7, 8, 9, 10, 11, 12, 7, 8];
        assert_eq!(history_continuation(&past, 9, 10, 2), [11, 12]);
        assert_eq!(history_continuation(&past, 9, 10, 1), [11]);
        assert!(history_continuation(&past, 9, 100, 2).is_empty());
        assert!(history_continuation(&past, 9, 10, 0).is_empty());
        assert!(history_continuation(&[7, 8], 9, 10, 2).is_empty());
        let latest = [7, 8, 9, 10, 11, 7, 8, 9, 10, 15, 7, 8];
        assert_eq!(history_continuation(&latest, 9, 10, 2), [15, 7]);
        let no_future = [1, 2, 7, 8, 7, 8];
        assert!(history_continuation(&no_future, 7, 8, 2).is_empty());
    }
}

#[cfg(test)]
mod catch_up_tests {
    use super::*;
    // Larger F32 matrix blocks change the reduction order, not the cache formula.
    // Compare every value and the next proposal; target hidden remains an exact copy.
    fn close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
        let error: f64 = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let energy: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
        let relative = (error / energy.max(1e-30)).sqrt();
        assert!(relative <= 2e-5, "{label}: relative L2 {relative}");
        println!("CATCH_UP {label} relative_l2={relative}");
    }
    #[test]
    #[ignore = "requires QWEN_MODEL, QWEN_DRAFT and Apple Silicon"]
    fn packed_catch_up_keeps_cache_and_next_proposal_equivalent() {
        check_catch_up(&[3, 1, 4, 2, 7, 15, 128], 256, 160);
    }
    #[test]
    #[ignore = "requires QWEN_MODEL, QWEN_DRAFT and Apple Silicon"]
    fn packed_catch_up_crosses_128_token_chunks() {
        check_catch_up(&[129, 257], 512, 386);
    }
    fn check_catch_up(counts: &[usize], context: usize, token_len: usize) {
        let config = DraftConfig {
            model: PathBuf::from(std::env::var_os("QWEN_DRAFT").unwrap()),
            max_tokens: 3,
            vocabulary: std::env::var_os("DRAFT_VOCAB").map(Into::into),
        };
        let model = PathBuf::from(std::env::var_os("QWEN_MODEL").unwrap());
        let mut q = Qwen::load(
            Gguf::open(&model).unwrap(),
            44 << 30,
            context,
            Some(&config),
            false,
        )
        .unwrap();
        let cancel = Cancellation::default();
        let tokens: Vec<_> = [1, 109266, 96748, 198, 248044, 42, 13]
            .into_iter()
            .cycle()
            .take(token_len)
            .collect();
        let mut target = q.state(context).unwrap();
        let mut history = Vec::new();
        let mut hidden = Vec::new();
        for block in tokens.chunks(q.prefill_block_size()) {
            let forward = q
                .target_block(&mut target, &history, block, &cancel, false)
                .unwrap();
            hidden.extend(forward.hidden);
            history.extend_from_slice(block);
        }
        let d = q.draft.as_ref().unwrap();
        let mut scalar = d.state(&q.gpu, &q.c, context).unwrap();
        let mut packed = d.state(&q.gpu, &q.c, context).unwrap();
        let mut pos = 0;
        for &count in counts {
            for (i, &token) in tokens.iter().enumerate().skip(pos).take(count) {
                q.draft_step(&scalar, i, token, &scalar.target_hidden, false, &cancel)
                    .unwrap();
                scalar.target_hidden = hidden[i].clone();
            }
            q.catch_up(
                &mut packed,
                pos,
                &tokens[pos..pos + count],
                &hidden[pos..pos + count],
                &cancel,
            )
            .unwrap();
            let buffers = |s: &DraftState| {
                let MixerState::Attention {
                    key,
                    value,
                    index,
                    pooled,
                } = &s.mixer
                else {
                    unreachable!()
                };
                vec![
                    key.clone(),
                    value.clone(),
                    index.clone(),
                    pooled.clone(),
                    s.target_hidden.clone(),
                ]
            };
            for (i, (a, b)) in buffers(&scalar).iter().zip(buffers(&packed)).enumerate() {
                let av = q.gpu.read_f32(a).unwrap();
                let bv = q.gpu.read_f32(&b).unwrap();
                if i == 4 {
                    assert_eq!(av, bv, "target hidden must remain canonical");
                } else {
                    close(&av, &bv, &format!("cache={i} position={pos} count={count}"));
                }
            }
            pos += count;
            let a = q
                .draft_step(&scalar, pos, 42, &scalar.target_hidden, true, &cancel)
                .unwrap()
                .unwrap();
            let b = q
                .draft_step(&packed, pos, 42, &packed.target_hidden, true, &cancel)
                .unwrap()
                .unwrap();
            assert_eq!(a.0, b.0);
            close(
                &q.gpu.read_f32(&a.1).unwrap(),
                &q.gpu.read_f32(&b.1).unwrap(),
                "next draft hidden",
            );
        }
    }
}
