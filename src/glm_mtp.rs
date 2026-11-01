//! GLM5-Next's trained NextN layer: post-norm target hidden, ordinary residuals,
//! its own DSA pool, and no mHC streams. Reference revision is in kernels/NOTICE.
use super::*;
use crate::engine::{MtpRound, history_continuation};

pub(super) struct Draft {
    embedding: Weight,
    output: Weight,
    enorm: Weight,
    hnorm: Weight,
    eh: Weight,
    attention_norm: Weight,
    ffn_norm: Weight,
    head_norm: Weight,
    attention: Dsa,
    moe: Moe,
    max_tokens: usize,
    vocabulary_ids: Option<Buffer>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires GLM_MODEL, GLM_TF_INDEX, GLM_TF_OUT, Apple Silicon and 32 GiB"]
    fn glm_score_frozen_corpus() {
        use std::fs::{self, OpenOptions};
        let model = PathBuf::from(std::env::var_os("GLM_MODEL").unwrap());
        let index = fs::read_to_string(std::env::var_os("GLM_TF_INDEX").unwrap()).unwrap();
        let out = PathBuf::from(std::env::var_os("GLM_TF_OUT").unwrap());
        fs::create_dir(&out).unwrap();
        let mut scores = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("scores.jsonl"))
            .unwrap();
        let budget = 32 * (1usize << 30);
        let glm = Glm::load(Gguf::open(&model).unwrap(), budget, 4096, None, false).unwrap();
        let cancel = Cancellation::default();
        for line in index.lines() {
            let f: Vec<_> = line.split('\t').collect();
            assert_eq!(f.len(), 4);
            let raw = fs::read(f[2]).unwrap();
            assert_eq!(raw.len() % 4, 0);
            let tokens: Vec<_> = raw
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert!(tokens.len() >= 2 && tokens.len() <= 4096);
            let samples: Vec<usize> = f[3].split(',').map(|v| v.parse().unwrap()).collect();
            let mut full = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(out.join(format!("{}-{}.f32", f[0], f[1])))
                    .unwrap(),
            );
            let mut state = glm.state(4096).unwrap();
            let mut nll = 0.0_f64;
            let mut position = 0;
            let start = std::time::Instant::now();
            for block in tokens[..tokens.len() - 1].chunks(PREFILL_BLOCK) {
                let hidden = glm
                    .prefill_hidden(&mut state, position, block, &cancel, &mut [])
                    .unwrap();
                for h in hidden {
                    let cmd = glm.gpu.command().unwrap();
                    let mean = glm.gpu.output(glm.c.width).unwrap();
                    glm.point(
                        &cmd,
                        "glm_hc_mean",
                        &[&h, &mean],
                        &[glm.c.width as u32, glm.c.hc as u32],
                        glm.c.width,
                    )
                    .unwrap();
                    let normalized = glm
                        .norm(
                            &cmd,
                            &mean,
                            &glm.norm.buffer,
                            glm.c.width,
                            1,
                            glm.c.eps,
                            false,
                        )
                        .unwrap();
                    let logits = glm.output.project(&glm.gpu, &cmd, &normalized).unwrap();
                    let logits = glm.gpu.finish(cmd, &logits).unwrap();
                    assert!(logits.iter().all(|v| v.is_finite()));
                    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    nll += maximum
                        + logits
                            .iter()
                            .map(|v| (f64::from(*v) - maximum).exp())
                            .sum::<f64>()
                            .ln()
                        - f64::from(logits[tokens[position + 1] as usize]);
                    if samples.contains(&position) {
                        for v in &logits {
                            full.write_all(&v.to_le_bytes()).unwrap();
                        }
                    }
                    position += 1;
                }
            }
            full.flush().unwrap();
            assert!(glm.metal_peak_bytes() <= budget);
            writeln!(scores, "{{\"category\":\"{}\",\"id\":\"{}\",\"tokens\":{},\"nll_sum\":{},\"seconds\":{},\"metal_peak_bytes\":{},\"prefill_block_tokens\":{}}}",
                f[0], f[1], position, nll, start.elapsed().as_secs_f64(), glm.metal_peak_bytes(), PREFILL_BLOCK).unwrap();
            scores.flush().unwrap();
            println!(
                "GLM TF {}/{} tokens={} nll={} seconds={}",
                f[0],
                f[1],
                position,
                nll,
                start.elapsed().as_secs_f64()
            );
        }
    }
    #[test]
    #[ignore = "requires GLM_MODEL, GLM_DRAFT, GLM_MTP_TRACE_TOKENS, GLM_MTP_TRACE_OUT and 32 GiB"]
    fn glm_dump_canonical_head_inputs_and_intermediates() {
        let model = PathBuf::from(std::env::var_os("GLM_MODEL").unwrap());
        let draft = DraftConfig {
            model: std::env::var_os("GLM_DRAFT").unwrap().into(),
            max_tokens: 3,
            vocabulary: None,
        };
        let raw = std::fs::read(std::env::var_os("GLM_MTP_TRACE_TOKENS").unwrap()).unwrap();
        assert_eq!(raw.len() % 4, 0);
        let tokens: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(!tokens.is_empty() && tokens.len() < 256);
        let out = PathBuf::from(std::env::var_os("GLM_MTP_TRACE_OUT").unwrap());
        let mut glm = Glm::load(
            Gguf::open(&model).unwrap(),
            32 * (1usize << 30),
            256,
            Some(&draft),
            false,
        )
        .unwrap();
        let mut state = glm.state(256).unwrap();
        let cancel = Cancellation::default();
        for (pos, &token) in tokens.iter().enumerate() {
            glm.set_trace(pos, &out.join(pos.to_string())).unwrap();
            let d = state.draft.as_ref().unwrap();
            let (id, _) = glm
                .draft_step(d, pos, token, &d.target_hidden, true, &cancel)
                .unwrap()
                .unwrap();
            let target = glm
                .target_block(&mut state, pos, &[token], &cancel, false)
                .unwrap();
            state.draft.as_mut().unwrap().target_hidden = target.hidden[0].clone();
            println!("position={pos} draft_argmax={id}");
        }
    }
    #[test]
    #[ignore = "requires GLM_MODEL, Apple Silicon and 24 GiB"]
    fn glm_every_verification_position_and_rollback_matches_sequential() {
        let model = PathBuf::from(std::env::var_os("GLM_MODEL").unwrap());
        let positions = if std::env::var_os("GLM_LONG_MTP").is_some() {
            vec![2047, 2048, 2051, 8191, 8192, 8195]
        } else {
            vec![3, 63, 64, 65]
        };
        let context = positions.iter().max().unwrap() + 16;
        let mut glm = Glm::load(
            Gguf::open(&model).unwrap(),
            22 * (1usize << 30),
            context,
            None,
            false,
        )
        .unwrap();
        // This internal AR diagnostic requests snapshots that production AR
        // never allocates. Reserve at most 2 GiB scratch within the 24 GiB test budget.
        glm.gpu.configure_workspace(2 << 30);
        let cancel = Cancellation::default();
        let logits = |glm: &Glm, h: &Buffer| {
            let cmd = glm.gpu.command().unwrap();
            let out = glm.output.project(&glm.gpu, &cmd, h).unwrap();
            glm.gpu.finish(cmd, &out).unwrap()
        };
        for position in positions {
            let input: Vec<_> = [1, 234, 1000, 42]
                .into_iter()
                .cycle()
                .take(position)
                .collect();
            let tail = [198, 42, 234, 1];
            let mut state = glm.state(context).unwrap();
            for (i, block) in input.chunks(PREFILL_BLOCK).enumerate() {
                glm.target_block(&mut state, i * PREFILL_BLOCK, block, &cancel, false)
                    .unwrap();
            }
            let mut before = Checkpoint::default();
            let cmd = glm.gpu.command().unwrap();
            for layer in &state.layers {
                glm.save_layer(&cmd, layer, &mut before).unwrap();
            }
            glm.gpu.complete(cmd).unwrap();
            let mut expected = Vec::new();
            let mut hidden = Vec::new();
            for (i, &token) in tail.iter().enumerate() {
                let frame = glm
                    .target_block(&mut state, position + i, &[token], &cancel, false)
                    .unwrap();
                expected.push(logits(&glm, &frame.hidden[0]));
                hidden.push(glm.gpu.read_f32(&frame.hidden[0]).unwrap());
            }
            glm.restore(&before).unwrap();
            let frame = glm
                .target_block(&mut state, position, &tail, &cancel, true)
                .unwrap();
            for (i, h) in frame.hidden.iter().enumerate() {
                assert_eq!(
                    glm.gpu.read_f32(h).unwrap(),
                    hidden[i],
                    "GLM hidden position={position} row={i}"
                );
                assert_eq!(logits(&glm, h), expected[i]);
            }
            let selected = glm.select_hidden(&frame.hidden).unwrap();
            let reference: Vec<_> = expected
                .iter()
                .map(|v| crate::engine::greedy(v).unwrap())
                .collect();
            assert_eq!(selected, reference);
            for committed in 1..tail.len() {
                glm.restore(&frame.snapshots[committed - 1]).unwrap();
                for i in committed..tail.len() {
                    let future = glm
                        .target_block(&mut state, position + i, &tail[i..i + 1], &cancel, false)
                        .unwrap();
                    assert_eq!(
                        glm.gpu.read_f32(&future.hidden[0]).unwrap(),
                        hidden[i],
                        "GLM rollback position={position} committed={committed} row={i}"
                    );
                    assert_eq!(logits(&glm, &future.hidden[0]), expected[i]);
                }
            }
            println!(
                "PASS GLM every-position full logits/hidden, GPU choices, and rollback accepted=0/1/2/3 prefix={position}"
            );
        }
    }
}
pub(super) struct DraftState {
    pub(super) mixer: LayerState,
    pub(super) target_hidden: Buffer,
}
#[derive(Default)]
pub(super) struct Checkpoint {
    mutable: Vec<(Buffer, Buffer)>,
    bytes: usize,
    encode: Duration,
}
pub(super) struct Verified {
    pub(super) hidden: Vec<Buffer>,
    snapshots: Vec<Checkpoint>,
    snapshot_encode: Duration,
}
impl Draft {
    pub(super) fn open(
        target: &Gguf,
        c: &Config,
        config: &DraftConfig,
    ) -> Result<(Gguf, Vec<u32>)> {
        if !(1..=3).contains(&config.max_tokens) {
            return Err(invalid("GLM draft-tokens must be 1, 2 or 3"));
        }
        let g = Gguf::open(&config.model)?;
        if g.architecture != target.architecture
            || g.integer("glm5-next.block_count")? != c.layers + 1
            || g.integer("glm5-next.nextn_predict_layers")? != 1
            || g.metadata.get("general.name") != target.metadata.get("general.name")
        {
            return Err(invalid("incompatible GLM5-Next draft model"));
        }
        for (name, value) in &target.metadata {
            if (name.starts_with("tokenizer.") || name.starts_with("glm5-next."))
                && !matches!(
                    name.as_str(),
                    "glm5-next.block_count"
                        | "glm5-next.nextn_predict_layers"
                        | "glm5-next.attention.head_count_kv"
                        | "glm5-next.swiglu_clamp_exp"
                        | "glm5-next.swiglu_clamp_shexp"
                )
                && g.metadata.get(name) != Some(value)
            {
                return Err(invalid(format!("GLM draft metadata mismatch: {name}")));
            }
        }
        let sparse = g.integers("glm5-next.attention.head_count_kv")?;
        if sparse.len() != c.layers + 1 || sparse[..c.layers] != c.sparse || sparse[c.layers] != 1 {
            return Err(invalid("GLM draft must have its own DSA layer"));
        }
        for name in ["glm5-next.swiglu_clamp_exp", "glm5-next.swiglu_clamp_shexp"] {
            let a = g.array(name)?;
            if a.len() != c.layers + 1
                || a.iter().any(|v| !matches!(v, Value::Float(x) if *x==10.0))
            {
                return Err(invalid("unsupported GLM draft SwiGLU limit"));
            }
        }
        let embedded = if g.metadata.contains_key("strata.draft_vocab_ids") {
            g.integers("strata.draft_vocab_ids")?
                .into_iter()
                .map(|v| u32::try_from(v).map_err(invalid))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let vocabulary = if let Some(path) = &config.vocabulary {
            let raw = std::fs::read(path)?;
            if raw.is_empty() || raw.len() % 4 != 0 {
                return Err(invalid(
                    "GLM draft-vocab requires nonempty little-endian i32 IDs",
                ));
            }
            let ids = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if !embedded.is_empty() && embedded != ids {
                return Err(invalid("GLM draft-vocab differs from compact head mapping"));
            }
            ids
        } else {
            embedded
        };
        let vocab = target.array("tokenizer.ggml.tokens")?.len();
        let mut seen = vec![false; vocab];
        for &id in &vocabulary {
            if id as usize >= vocab || seen[id as usize] {
                return Err(invalid("invalid or duplicate GLM draft vocabulary ID"));
            }
            seen[id as usize] = true;
        }
        if vocabulary.len() >= vocab {
            return Err(invalid(
                "GLM draft vocabulary must be smaller than full vocabulary",
            ));
        }
        let tensor = g.tensor("output.weight")?;
        if tensor.dims.len() != 2
            || tensor.dims[0] != c.width
            || (tensor.dims[1] != vocab
                && (vocabulary.is_empty() || tensor.dims[1] != vocabulary.len()))
        {
            return Err(invalid(
                "GLM compact draft head requires its complete global ID mapping",
            ));
        }
        Ok((g, vocabulary))
    }
    pub(super) fn reserved_bytes(
        (g, vocabulary): &(Gguf, Vec<u32>),
        c: &Config,
        context: usize,
    ) -> Result<usize> {
        let mut weights = g.tensors.values().try_fold(0usize, |sum, t| {
            sum.checked_add(t.mapped_bytes()?)
                .ok_or_else(|| invalid("GLM draft weight bytes overflow"))
        })?;
        if !vocabulary.is_empty() {
            let full = g.tensor("output.weight")?;
            weights -= full.bytes.div_ceil(16 * 1024) * 16 * 1024;
            weights += (full.stride * vocabulary.len()).div_ceil(16 * 1024) * 16 * 1024
                + (vocabulary.len() * 4).div_ceil(16 * 1024) * 16 * 1024;
        }
        let state =
            (c.latent * context + 2 * c.idx * context + c.idx * context.div_ceil(c.pool) + c.width)
                * 4;
        weights
            .checked_add(state + Self::working_bytes(c))
            .ok_or_else(|| invalid("GLM draft memory overflow"))
    }
    pub(super) fn working_bytes(c: &Config) -> usize {
        let mutable = c.sparse.iter().filter(|&&v| v == 0).count()
            * (c.heads * c.d * c.d + 3 * (c.conv - 1) * c.heads * c.d)
            * 4;
        3 * mutable + 64 * 1024 * 1024
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
        let vocabulary_ids = if vocabulary.is_empty() {
            None
        } else {
            Some(
                gpu.bytes(
                    &vocabulary
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
            )
        };
        let p = format!("blk.{}.", c.layers);
        let mut w =
            |name: &str, shape: &[usize]| Weight::load(&mut g, gpu, &format!("{p}{name}"), shape);
        let enorm = w("nextn.enorm.weight", &[c.width])?;
        let hnorm = w("nextn.hnorm.weight", &[c.width])?;
        let eh = w("nextn.eh_proj.weight", &[2 * c.width, c.width])?;
        let attention_norm = w("attn_norm.weight", &[c.width])?;
        let ffn_norm = w("ffn_norm.weight", &[c.width])?;
        let head_norm = w("nextn.shared_head_norm.weight", &[c.width])?;
        for weight in [&enorm, &hnorm, &attention_norm, &ffn_norm, &head_norm] {
            weight.require_f32()?;
        }
        let attention = Glm::dsa_load(&mut g, gpu, &p, c)?;
        let moe = Glm::moe_load(&mut g, gpu, &p, c, c.experts)?;
        Ok(Self {
            embedding,
            output,
            enorm,
            hnorm,
            eh,
            attention_norm,
            ffn_norm,
            head_norm,
            attention,
            moe,
            max_tokens: config.max_tokens,
            vocabulary_ids,
        })
    }
}
impl Glm {
    pub(super) fn dsa_state(&self, context: usize) -> Result<LayerState> {
        let c = &self.c;
        Ok(LayerState::Dsa {
            latent: self.gpu.zeros(c.latent * context)?,
            key: self.gpu.zeros(c.idx * context)?,
            gate: self.gpu.zeros(c.idx * context)?,
            pooled: self.gpu.zeros(c.idx * context.div_ceil(c.pool))?,
        })
    }
    fn add(&self, cmd: &Command, a: &Buffer, b: &Buffer) -> Result<Buffer> {
        let y = self.gpu.output(self.c.width)?;
        self.point(
            cmd,
            "glm_add",
            &[a, b, &y],
            &[self.c.width as u32],
            self.c.width,
        )?;
        Ok(y)
    }
    pub(super) fn save_layer(
        &self,
        cmd: &Command,
        state: &LayerState,
        saved: &mut Checkpoint,
    ) -> Result<()> {
        let start = Instant::now();
        // DSA writes only at the speculative suffix. Completed earlier pools
        // are immutable; rejected suffix rows are overwritten before being read.
        if let LayerState::Kda { recurrent, conv } = state {
            for buffer in std::iter::once(recurrent).chain(conv.iter()) {
                let copy = self.copy(cmd, buffer, buffer.length() / 4, 0)?;
                saved.bytes += buffer.length();
                saved.mutable.push((buffer.clone(), copy));
            }
        }
        saved.encode += start.elapsed();
        Ok(())
    }
    fn restore(&self, saved: &Checkpoint) -> Result<()> {
        let cmd = self.gpu.command()?;
        for (target, source) in &saved.mutable {
            self.point(
                &cmd,
                "copy_range",
                &[source, target],
                &[(target.length() / 4) as u32, 0],
                target.length() / 4,
            )?;
        }
        self.gpu.complete(cmd)
    }
    pub(super) fn target_block(
        &mut self,
        state: &mut State,
        position: usize,
        tokens: &[u32],
        cancel: &Cancellation,
        snapshot: bool,
    ) -> Result<Verified> {
        objc2::rc::autoreleasepool(|_| {
            self.target_block_inner(state, position, tokens, cancel, snapshot)
        })
    }
    fn target_block_inner(
        &mut self,
        state: &mut State,
        position: usize,
        tokens: &[u32],
        cancel: &Cancellation,
        snapshot: bool,
    ) -> Result<Verified> {
        if tokens.is_empty() {
            return Err(invalid("empty GLM token block"));
        }
        let mut snapshots: Vec<_> = (0..if snapshot { tokens.len() - 1 } else { 0 })
            .map(|_| Checkpoint::default())
            .collect();
        let hidden = if tokens.len() == 1 || self.trace.is_some() {
            let mut hidden = Vec::with_capacity(tokens.len());
            for (i, &token) in tokens.iter().enumerate() {
                hidden.push(self.forward_token(state, position + i, token, cancel)?);
                if let Some(saved) = snapshots.get_mut(i) {
                    let cmd = self.gpu.command()?;
                    for layer in &state.layers {
                        self.save_layer(&cmd, layer, saved)?;
                    }
                    self.gpu.complete(cmd)?;
                }
            }
            hidden
        } else {
            self.prefill_hidden(state, position, tokens, cancel, &mut snapshots)?
        };
        let cmd = self.gpu.command()?;
        let skip = if snapshot || self.draft.is_some() {
            0
        } else {
            hidden.len() - 1
        };
        let mut normalized = Vec::with_capacity(hidden.len() - skip);
        for h in hidden.into_iter().skip(skip) {
            let mean = self.gpu.output(self.c.width)?;
            self.point(
                &cmd,
                "glm_hc_mean",
                &[&h, &mean],
                &[self.c.width as u32, self.c.hc as u32],
                self.c.width,
            )?;
            normalized.push(self.norm(
                &cmd,
                &mean,
                &self.norm.buffer,
                self.c.width,
                1,
                self.c.eps,
                false,
            )?);
        }
        self.gpu.complete(cmd)?;
        if let Some(h) = normalized.last() {
            self.record("target_result_norm", h);
            self.flush_trace(self.c.layers)?;
        }
        let snapshot_encode = snapshots.iter().map(|s| s.encode).sum();
        Ok(Verified {
            hidden: normalized,
            snapshots,
            snapshot_encode,
        })
    }
    pub(super) fn select_hidden(&self, hidden: &[Buffer]) -> Result<Vec<u32>> {
        let cmd = self.gpu.command()?;
        let packed = self.pack(&cmd, hidden, self.c.width)?;
        // Verification uses at most four rows and the same F32 lane order as AR.
        let logits = self.gpu.matmul(
            &cmd,
            &self.output.buffer,
            self.c.width,
            self.output.t.dims[1],
            self.output.t.stride,
            self.output.t.kind,
            0,
            &packed,
            hidden.len(),
        )?;
        self.gpu
            .finish_greedy(cmd, &logits, hidden.len(), self.output.t.dims[1], None)
    }
    fn draft_step(
        &self,
        state: &DraftState,
        position: usize,
        token: u32,
        hidden: &Buffer,
        output: bool,
        cancel: &Cancellation,
    ) -> Result<Option<(u32, Buffer)>> {
        objc2::rc::autoreleasepool(|_| {
            self.draft_step_inner(state, position, token, hidden, output, cancel)
        })
    }
    fn draft_step_inner(
        &self,
        state: &DraftState,
        position: usize,
        token: u32,
        hidden: &Buffer,
        output: bool,
        cancel: &Cancellation,
    ) -> Result<Option<(u32, Buffer)>> {
        if cancel.is_cancelled() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "generation cancelled",
            ));
        }
        let d = self
            .draft
            .as_ref()
            .ok_or_else(|| invalid("missing GLM draft"))?;
        let c = &self.c;
        self.trace_active
            .set(self.trace.as_ref().is_some_and(|(p, _)| *p == position));
        self.trace_events.borrow_mut().clear();
        let cmd = self.gpu.command()?;
        let emb = self.gpu.output(c.width)?;
        self.point(
            &cmd,
            "embedding",
            &[&d.embedding.buffer, &emb],
            &[
                c.width as u32,
                0,
                d.embedding.t.stride as u32,
                d.embedding.t.kind,
                token,
            ],
            c.width,
        )?;
        self.record("mtp_tok_embd", &emb);
        self.record("mtp_h_input", hidden);
        let en = self.norm(&cmd, &emb, &d.enorm.buffer, c.width, 1, c.eps, false)?;
        let hn = self.norm(&cmd, hidden, &d.hnorm.buffer, c.width, 1, c.eps, false)?;
        self.record("mtp_enorm", &en);
        self.record("mtp_hnorm", &hn);
        let combined = self.pack(&cmd, &[en, hn], c.width)?;
        let h = d.eh.project(&self.gpu, &cmd, &combined)?;
        self.record("mtp_eh_proj", &h);
        let x = self.norm(&cmd, &h, &d.attention_norm.buffer, c.width, 1, c.eps, false)?;
        self.record("mtp_attn_norm", &x);
        if !output {
            self.dsa_cache(&cmd, &d.attention, &state.mixer, &x, position)?;
            self.gpu.complete(cmd)?;
            return Ok(None);
        }
        let attention = self.dsa(&cmd, &d.attention, &state.mixer, &x, position)?;
        self.record("mtp_attn_out", &attention);
        let residual = self.add(&cmd, &h, &attention)?;
        self.record("mtp_ffn_inp", &residual);
        let x = self.norm(
            &cmd,
            &residual,
            &d.ffn_norm.buffer,
            c.width,
            1,
            c.eps,
            false,
        )?;
        self.record("mtp_ffn_norm", &x);
        let (ffn, cmd) = self.moe(cmd, &d.moe, &x, cancel)?;
        self.record("mtp_ffn_out", &ffn);
        let h = self.add(&cmd, &residual, &ffn)?;
        self.record("mtp_post_ffn", &h);
        let h = self.norm(&cmd, &h, &d.head_norm.buffer, c.width, 1, c.eps, false)?;
        self.record("mtp_shared_head_norm", &h);
        let logits = d.output.project(&self.gpu, &cmd, &h)?;
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
    }
    pub(super) fn catch_up_committed(
        &self,
        state: &mut State,
        position: usize,
        tokens: &[u32],
        hidden: &[Buffer],
        cancel: &Cancellation,
    ) -> Result<()> {
        if let Some(draft) = &mut state.draft {
            for (i, &token) in tokens.iter().enumerate() {
                self.draft_step(
                    draft,
                    position + i,
                    token,
                    &draft.target_hidden,
                    false,
                    cancel,
                )?;
                draft.target_hidden = hidden[i].clone();
            }
        }
        Ok(())
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
        let layer = |a: &LayerState, b: &LayerState| -> Result<()> {
            match (a, b) {
                (
                    LayerState::Kda {
                        recurrent: ar,
                        conv: ac,
                    },
                    LayerState::Kda {
                        recurrent: br,
                        conv: bc,
                    },
                ) => {
                    copy(ar, br)?;
                    for (a, b) in ac.iter().zip(bc) {
                        copy(a, b)?;
                    }
                }
                (
                    LayerState::Dsa {
                        latent: al,
                        key: ak,
                        gate: ag,
                        pooled: ap,
                    },
                    LayerState::Dsa {
                        latent: bl,
                        key: bk,
                        gate: bg,
                        pooled: bp,
                    },
                ) => {
                    copy(al, bl)?;
                    copy(ak, bk)?;
                    copy(ag, bg)?;
                    copy(ap, bp)?;
                }
                _ => return Err(invalid("GLM prefix layer state mismatch")),
            }
            Ok(())
        };
        for (a, b) in source.layers.iter().zip(&destination.layers) {
            layer(a, b)?;
        }
        if let (Some(a), Some(b)) = (&source.draft, &destination.draft) {
            layer(&a.mixer, &b.mixer)?;
            copy(&a.target_hidden, &b.target_hidden)?;
        }
        self.gpu.complete(cmd)
    }
    pub(crate) fn draft_tokens(&self) -> Option<usize> {
        self.draft.as_ref().map(|d| d.max_tokens)
    }
    pub(crate) fn speculate(
        &mut self,
        state: &mut State,
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
            .ok_or_else(|| invalid("missing GLM draft"))?
            .min(max_draft);
        let draft = state
            .draft
            .as_ref()
            .ok_or_else(|| invalid("missing GLM draft state"))?;
        let mut h = draft.target_hidden.clone();
        let mut tokens = vec![current];
        let head = Instant::now();
        for i in 0..count {
            let (id, hidden) = self
                .draft_step(draft, history.len() + i, tokens[i], &h, true, cancel)?
                .ok_or_else(|| invalid("missing GLM draft result"))?;
            tokens.push(id);
            h = hidden;
            if is_eos(id) {
                break;
            }
            if i == 0 {
                for &past in history_continuation(history, current, id, count - 1) {
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
        stats.head = head.elapsed();
        stats.proposed = tokens.len() - 1;
        let verify = Instant::now();
        let frame = self.target_block(state, history.len(), &tokens, cancel, true)?;
        let choices = self.select_hidden(&frame.hidden)?;
        stats.verify = verify.elapsed();
        stats.verification_tokens = tokens.len();
        stats.snapshot_encode = frame.snapshot_encode;
        stats.snapshot_bytes = frame.snapshots.iter().map(|s| s.bytes).sum();
        let accepted = tokens[1..]
            .iter()
            .zip(&choices)
            .take_while(|(a, b)| a == b)
            .count();
        let committed = 1 + tokens[1..=accepted]
            .iter()
            .take_while(|&&id| !is_eos(id))
            .count();
        let output = choices[..committed].to_vec();
        if committed < tokens.len() {
            let restore = Instant::now();
            self.restore(&frame.snapshots[committed - 1])?;
            stats.snapshot = restore.elapsed();
            stats.snapshot_bytes += frame.snapshots[committed - 1].bytes;
        }
        let catch_up = Instant::now();
        self.catch_up_committed(
            state,
            history.len(),
            &tokens[..committed],
            &frame.hidden[..committed],
            cancel,
        )?;
        stats.catch_up = catch_up.elapsed();
        history.extend_from_slice(&tokens[..committed]);
        stats.accepted = accepted;
        stats.history_accepted = accepted.saturating_sub(1).min(stats.history_proposed);
        stats.committed = committed;
        stats.total = start.elapsed();
        Ok((output, stats))
    }
}
