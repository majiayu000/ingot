//! Generic decoder-only transformers in GGUF: Llama (and Mistral/Mixtral),
//! Qwen2, Qwen3, Qwen3-MoE, Gemma 3/4 and Phi-3 (configuration and weights in
//! `generic_arch.rs`). GQA attention with RoPE over an F32 KV cache, optional
//! sliding windows, then a SwiGLU/GeGLU FFN or a top-k softmax MoE. All weights
//! stay resident and run in their stored quantized formats through the shared
//! projection kernels.
use crate::engine::Cancellation;
use crate::generic_arch::{Config, Family, Ffn, Geometry, Layer, Loader, Qkv};
use crate::gguf::Gguf;
use crate::metal::{Buffer, Command, Gpu};
use crate::weights::Weight;
use crate::{Result, invalid};
use std::time::Duration;

pub(crate) use crate::generic_arch::ARCHITECTURES;
/// Tokens per forward block: every block reads all weights (for MoE, nearly
/// every expert) once. Without a flash kernel for every head size, dense
/// attention scores (block × heads × context floats) bound it to 128.
const PREFILL_BLOCK: [usize; 2] = [128, 512];
/// Head dimensions with a flash prefill kernel.
fn flash_head(head: usize) -> bool {
    matches!(head, 64 | 96 | 128 | 256 | 512)
}
/// Flash head dimensions without a tail kernel: their last partial tile of
/// fewer than 64 query rows uses dense attention.
const NO_FLASH_TAIL: usize = 512;

pub(crate) struct State {
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    rows: usize,
}
pub(crate) struct Generic {
    gpu: Gpu,
    c: Config,
    embedding: Weight,
    /// None when the output projection is tied to the embedding.
    output: Option<Weight>,
    output_norm: Weight,
    /// One-element placeholder for layers without RoPE factors.
    no_factors: Buffer,
    /// Gemma 4 normalizes values without a weight.
    value_norm: bool,
    block: usize,
    /// Zero shared-expert output and gate for `combine_packed` (no shared expert).
    no_shared: Option<(Buffer, Buffer)>,
    layers: Vec<Layer>,
    bytes: usize,
    file_bytes: u64,
}

fn cancelled() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Interrupted, "generation cancelled")
}

impl Generic {
    pub(crate) fn load(
        mut g: Gguf,
        budget: usize,
        context: usize,
        draft: Option<&crate::engine::DraftConfig>,
        prefix_cache: bool,
    ) -> Result<Self> {
        if draft.is_some() {
            return Err(invalid(format!(
                "{} models have no MTP draft head",
                g.architecture
            )));
        }
        let c = Config::load(&g)?;
        let a = g.architecture.clone();
        if context == 0
            || g.optional_integer(&format!("{a}.context_length"))
                .is_some_and(|n| context > n)
        {
            return Err(invalid("invalid context capacity"));
        }
        let weights = g
            .tensors
            .values()
            .map(|t| t.mapped_bytes())
            .sum::<Result<usize>>()?;
        if c.geometry.iter().any(|l| l.head > 512) {
            return Err(invalid(format!("{a}: head dimension above 512")));
        }
        // Prefill attention is tiled; head sizes without a flash kernel (and
        // decode blocks of sizes without a decode kernel) keep full score rows
        // (scores and probabilities).
        let flash = c.geometry.iter().all(|l| flash_head(l.head));
        let block_size = PREFILL_BLOCK[flash as usize];
        let block = block_size.min(context);
        let dense = if !flash {
            block
        } else if c.geometry.iter().any(|l| l.head == NO_FLASH_TAIL) {
            64.min(block)
        } else {
            4
        };
        // Two layers in flight, each up to 12 width-sized and 3 FFN-sized rows
        // per token (fused QKV and gate/up, residuals and norms).
        let workspace = 256 * 1024 * 1024
            + 2 * block * 4 * (12 * c.width.max(c.max_q()) + 3 * c.ff.max(c.top * c.ff))
            + 2 * dense * c.max_heads() * context * 4;
        let prefix = if prefix_cache {
            c.state_bytes(block)
        } else {
            0
        };
        let required = weights + c.state_bytes(context) + workspace + prefix;
        if required > budget {
            return Err(invalid(format!(
                "{a} with context {context} needs {:.1} GiB, budget is {:.1} GiB",
                required as f64 / (1u64 << 30) as f64,
                budget as f64 / (1u64 << 30) as f64
            )));
        }
        let gpu = Gpu::new()?;
        gpu.configure_workspace(workspace);
        let w = c.width;
        let vocab = g.array("tokenizer.ggml.tokens")?.len();
        let mut loader = Loader::new(&mut g, &gpu);
        let embedding = loader.load("token_embd.weight", &[w, vocab])?;
        let output_norm = loader.load("output_norm.weight", &[w])?;
        output_norm.require_f32()?;
        let layers = loader.layers(&c)?;
        let output = loader.optional("output.weight", &[w, vocab])?;
        loader.check_all_used()?;
        let no_shared = if c.experts > 0 {
            Some((gpu.zeros(c.width * block)?, gpu.zeros(block)?))
        } else {
            None
        };
        Ok(Self {
            no_factors: gpu.floats(&[1.0])?,
            value_norm: c.family == Family::Gemma4,
            block: block_size,
            gpu,
            c,
            embedding,
            output,
            output_norm,
            no_shared,
            layers,
            bytes: required,
            file_bytes: g.bytes_read,
        })
    }

    pub(crate) fn state(&self, context: usize) -> Result<State> {
        let cache = || -> Result<Vec<Buffer>> {
            self.c
                .geometry
                .iter()
                .map(|l| self.gpu.zeros(context * l.kv * l.head))
                .collect()
        };
        Ok(State {
            keys: cache()?,
            values: cache()?,
            rows: context,
        })
    }
    fn copy_rows(&self, source: &State, destination: &State, rows: usize) -> Result<()> {
        if rows > source.rows || rows > destination.rows {
            return Err(invalid("prefix snapshot exceeds the session context"));
        }
        if rows == 0 {
            return Ok(());
        }
        let cmd = self.gpu.command()?;
        for (from, to) in [
            (&source.keys, &destination.keys),
            (&source.values, &destination.values),
        ] {
            for ((a, b), l) in from.iter().zip(to).zip(&self.c.geometry) {
                let n = rows * l.kv * l.head;
                self.point(&cmd, "copy_range", &[a, b], &[n as u32, 0], n)?;
            }
        }
        self.gpu.complete(cmd)
    }
    pub(crate) fn snapshot_prefix(&self, source: &State, position: usize) -> Result<State> {
        let snapshot = self.state(position)?;
        self.copy_rows(source, &snapshot, position)?;
        Ok(snapshot)
    }
    pub(crate) fn copy_prefix_state(&self, source: &State, destination: &mut State) -> Result<()> {
        self.copy_rows(source, destination, source.rows)
    }

    fn point(&self, cmd: &Command, name: &str, b: &[&Buffer], p: &[u32], n: usize) -> Result<()> {
        self.gpu.op(cmd, name, b, p, n.div_ceil(32), 1)
    }
    fn binary(
        &self,
        cmd: &Command,
        name: &str,
        a: &Buffer,
        b: &Buffer,
        n: usize,
        period: usize,
    ) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(cmd, name, &[a, b, &y], &[n as u32, period as u32], n)?;
        Ok(y)
    }
    /// Element-wise `name` (scale_rows, softcap_rows) with one f32 parameter.
    fn unary(&self, cmd: &Command, name: &str, x: &Buffer, value: f32, n: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(cmd, name, &[x, &y], &[n as u32, value.to_bits()], n)?;
        Ok(y)
    }
    /// Columns [offset, offset + width) of `count` rows of `stride` floats.
    fn columns(
        &self,
        cmd: &Command,
        x: &Buffer,
        count: usize,
        width: usize,
        offset: usize,
        stride: usize,
    ) -> Result<Buffer> {
        let n = width * count;
        let y = self.gpu.output(n)?;
        let p = [n as u32, width as u32, offset as u32, stride as u32];
        self.point(cmd, "copy_columns", &[x, &y], &p, n)?;
        Ok(y)
    }
    fn project(&self, cmd: &Command, w: &Weight, x: &Buffer, count: usize) -> Result<Buffer> {
        if let Some(y) = self.gpu.generic_project(cmd, w, x, count)? {
            return Ok(y);
        }
        self.gpu.matmul(
            cmd,
            &w.buffer,
            w.t.dims[0],
            w.t.dims[1],
            w.t.stride,
            w.t.kind,
            0,
            x,
            count,
        )
    }
    /// QK-norm (when present), RoPE and query scale of `q` and `k`, writing
    /// `k` and `v` (normed for Gemma 4) into the caches at `position`;
    /// returns the rotated `q`.
    #[allow(clippy::too_many_arguments)]
    fn rope_cache(
        &self,
        cmd: &Command,
        l: &Layer,
        g: &Geometry,
        [q, k, v]: [&Buffer; 3],
        cache: [&Buffer; 2],
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        let factors = l.factors.as_ref();
        let (qn, kn) = match &l.qk_norm {
            Some([qn, kn]) => (&qn.buffer, &kn.buffer),
            None => (&self.no_factors, &self.no_factors),
        };
        let y = self.gpu.output(g.heads * g.head * count)?;
        let params = [
            g.head as u32,
            g.heads as u32,
            g.kv as u32,
            g.rot as u32,
            position as u32,
            g.base.to_bits(),
            self.c.neox as u32,
            factors.is_some() as u32,
            l.qk_norm.is_some() as u32,
            self.c.eps.to_bits(),
            self.value_norm as u32,
            g.q_scale.to_bits(),
        ];
        let factors = factors.unwrap_or(&self.no_factors);
        self.gpu.dispatch(
            cmd,
            "qk_rope_cache",
            &[q, k, v, qn, kn, factors, &y, cache[0], cache[1]],
            &params,
            [g.heads + 2 * g.kv, count],
            32,
        )?;
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        cmd: &Command,
        l: &Layer,
        g: &Geometry,
        cache: [&Buffer; 2],
        x: &Buffer,
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        let (h, kvd) = (g.heads * g.head, g.kv * g.head);
        let (q, k, v) = match &l.qkv {
            Qkv::Split([wq, wk, wv]) => (
                self.project(cmd, wq, x, count)?,
                self.project(cmd, wk, x, count)?,
                self.project(cmd, wv, x, count)?,
            ),
            Qkv::Fused(w) => {
                let all = self.project(cmd, w, x, count)?;
                let stride = h + 2 * kvd;
                (
                    self.columns(cmd, &all, count, h, 0, stride)?,
                    self.columns(cmd, &all, count, kvd, h, stride)?,
                    self.columns(cmd, &all, count, kvd, h + kvd, stride)?,
                )
            }
            Qkv::KeyAsValue([wq, wk]) => {
                let k = self.project(cmd, wk, x, count)?;
                (self.project(cmd, wq, x, count)?, k.clone(), k)
            }
        };
        let (q, k, v) = match &l.bias {
            Some([bq, bk, bv]) => (
                self.binary(cmd, "bias_rows", &q, &bq.buffer, h * count, h)?,
                self.binary(cmd, "bias_rows", &k, &bk.buffer, kvd * count, kvd)?,
                self.binary(cmd, "bias_rows", &v, &bv.buffer, kvd * count, kvd)?,
            ),
            None => (q, k, v),
        };
        let q = self.rope_cache(cmd, l, g, [&q, &k, &v], cache, position, count)?;
        let y = if count > 4 && flash_head(g.head) {
            self.flash(cmd, &q, cache, g, position, count)?
        } else if count > 4 || !matches!(g.head, 64 | 96 | 128 | 256 | 512) {
            self.dense_attention(cmd, &q, cache, g, position, count)?
        } else {
            let y = self.gpu.output(h * count)?;
            let params = [g.head, g.heads, g.kv, position, g.window].map(|v| v as u32);
            self.gpu.dispatch(
                cmd,
                &format!("attention_decode_{}", g.head),
                &[&q, cache[0], cache[1], &y],
                &params,
                [g.heads, count],
                256,
            )?;
            y
        };
        self.project(cmd, &l.out, &y, count)
    }

    /// Causal attention for decode-sized blocks: full score rows per query.
    fn dense_attention(
        &self,
        cmd: &Command,
        q: &Buffer,
        cache: [&Buffer; 2],
        c: &Geometry,
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        let y = self.gpu.output(c.heads * count * c.head)?;
        self.dense_into(cmd, q, cache, c, position, count, &y)?;
        Ok(y)
    }
    #[allow(clippy::too_many_arguments)]
    fn dense_into(
        &self,
        cmd: &Command,
        q: &Buffer,
        cache: [&Buffer; 2],
        c: &Geometry,
        position: usize,
        count: usize,
        y: &Buffer,
    ) -> Result<()> {
        let length = position + count;
        let rows = c.heads * count;
        let params = [
            c.head as u32,
            c.heads as u32,
            c.kv as u32,
            length as u32,
            position as u32,
            c.window as u32,
        ];
        let scores = self.gpu.output(rows * length)?;
        let (name, n) = if c.window > 0 {
            ("attention_scores_window", 6)
        } else {
            ("attention_scores_dense_batch", 5)
        };
        self.gpu.op(
            cmd,
            name,
            &[q, cache[0], &scores],
            &params[..n],
            length,
            rows,
        )?;
        let probabilities = self.gpu.output(rows * length)?;
        self.gpu.op(
            cmd,
            "softmax_dense_batch",
            &[&scores, &probabilities],
            &[length as u32, c.heads as u32, position as u32],
            rows,
            1,
        )?;
        self.gpu.op(
            cmd,
            "attention_values_dense_batch",
            &[&probabilities, cache[1], y],
            &params[..5],
            c.head,
            rows,
        )
    }
    /// Flash-style causal prefill attention. Rows whose keys end inside the
    /// last whole 64-key tile use the bounds-free kernel; the rest the tail one.
    fn flash(
        &self,
        cmd: &Command,
        q: &Buffer,
        cache: [&Buffer; 2],
        c: &Geometry,
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        let width = c.heads * c.head;
        let out = self.gpu.output(width * count)?;
        let length = position + count;
        let full = (length / 64 * 64).saturating_sub(position).min(count);
        for (begin, rows, tail) in [(0, full, false), (full, count - full, true)] {
            if rows == 0 {
                continue;
            }
            if tail && c.head == NO_FLASH_TAIL {
                let (q, y) = (
                    q.view_f32(begin * width, rows * width)?,
                    out.view_f32(begin * width, rows * width)?,
                );
                self.dense_into(cmd, &q, cache, c, position + begin, rows, &y)?;
                continue;
            }
            let name = format!(
                "flash_prefill_{}{}",
                c.head,
                if tail { "_tail" } else { "" }
            );
            self.gpu.op(
                cmd,
                &name,
                &[
                    &q.view_f32(begin * width, rows * width)?,
                    cache[0],
                    cache[1],
                    &out.view_f32(begin * width, rows * width)?,
                ],
                &[
                    c.head as u32,
                    c.heads as u32,
                    c.kv as u32,
                    length as u32,
                    (position + begin) as u32,
                    rows as u32,
                    c.window as u32,
                ],
                rows.div_ceil(8) * 4,
                c.heads,
            )?;
        }
        Ok(out)
    }

    fn ffn(&self, cmd: &Command, ffn: &Ffn, x: &Buffer, count: usize) -> Result<Buffer> {
        let c = &self.c;
        let (router, [gate, up, down]) = match ffn {
            Ffn::Dense([gate, up, down]) => {
                let fused = if count <= 4 {
                    self.gpu
                        .generic_gate_up(cmd, [gate, up], x, count, c.gelu)?
                } else {
                    None
                };
                let act = match fused {
                    Some(act) => act,
                    None => {
                        let g = self.project(cmd, gate, x, count)?;
                        let u = self.project(cmd, up, x, count)?;
                        if c.gelu {
                            self.binary(cmd, "gelu_mul", &g, &u, c.ff * count, 0)?
                        } else {
                            self.gpu.activation(cmd, &g, &u, c.ff * count)?
                        }
                    }
                };
                return self.project(cmd, down, &act, count);
            }
            Ffn::Fused([gate_up, down]) => {
                let gu = self.project(cmd, gate_up, x, count)?;
                let n = c.ff * count;
                let act = self.gpu.output(n)?;
                let p = [n as u32, c.ff as u32];
                self.point(cmd, "swiglu_fused", &[&gu, &act], &p, n)?;
                return self.project(cmd, down, &act, count);
            }
            Ffn::Moe { router, experts } => (router, experts),
        };
        let pairs = count * c.top;
        let logits = self.project(cmd, router, x, count)?;
        let ids = self.gpu.output(pairs)?;
        let weights = self.gpu.output(pairs)?;
        let route = [&logits, &ids, &weights];
        let params = [c.experts as u32, c.top as u32];
        if c.experts <= 512 {
            self.gpu
                .dispatch(cmd, "route_topk", &route, &params, [count, 1], 32)?;
        } else {
            self.gpu
                .op(cmd, "route_gpu_batch", &route, &params, count, 1)?;
        }
        if count > 4 {
            return self.moe_packed(cmd, [gate, up, down], x, &ids, &weights, count);
        }
        let routed = |w: &Weight, input: &Buffer, per_pair: bool| -> Result<Buffer> {
            let rows = w.t.dims[1];
            let y = self.gpu.output(rows * pairs)?;
            // Q4_K/Q6_K/Q4_0/F16/F32 experts use the vector kernels offset per pair.
            if self
                .gpu
                .generic_routed(cmd, w, [input, &ids, &y], per_pair, c.top, pairs)?
            {
                return Ok(y);
            }
            self.gpu.op(
                cmd,
                "project_routed_batch",
                &[&w.buffer, input, &ids, &y],
                &[
                    w.t.dims[0] as u32,
                    rows as u32,
                    w.t.stride as u32,
                    w.t.kind,
                    c.top as u32,
                    per_pair as u32,
                ],
                rows,
                pairs,
            )?;
            Ok(y)
        };
        let fused = self
            .gpu
            .generic_routed_gate_up(cmd, [gate, up], [x, &ids], c.top, pairs)?;
        let act = match fused {
            Some(act) => act,
            None => {
                let g = routed(gate, x, false)?;
                let u = routed(up, x, false)?;
                self.gpu.activation(cmd, &g, &u, c.ff * pairs)?
            }
        };
        let d = routed(down, &act, true)?;
        let y = self.gpu.output(c.width * count)?;
        self.point(
            cmd,
            "moe_sum",
            &[&d, &weights, &y],
            &[c.width as u32, c.top as u32, count as u32],
            c.width * count,
        )?;
        Ok(y)
    }

    /// Prefill MoE: the GPU groups routed pairs by expert (kernels/prefill.metal)
    /// and runs one tiled GEMM per expert, as in the Qwen3.8 resident path.
    fn moe_packed(
        &self,
        cmd: &Command,
        w: [&Weight; 3],
        x: &Buffer,
        ids: &Buffer,
        weights: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let c = &self.c;
        if c.experts > 512 {
            return Err(invalid("GPU expert grouping supports at most 512 experts"));
        }
        let total = c.top * count;
        let blocks = total.div_ceil(32) + c.experts;
        let layout = self.gpu.output(2 * c.experts)?;
        let tiles = self.gpu.output(4 * blocks)?;
        let pairs = self.gpu.output(total)?;
        let inverse = self.gpu.output(total)?;
        let params = [total as u32, c.experts as u32, blocks as u32];
        self.gpu
            .op(cmd, "expert_layout", &[ids, &layout, &tiles], &params, 1, 1)?;
        self.gpu.op(
            cmd,
            "expert_pairs",
            &[ids, &layout, &pairs, &inverse],
            &params,
            c.experts,
            1,
        )?;
        let geometry = |w: &Weight| {
            [
                w.t.dims[0] as u32,
                w.t.dims[1] as u32,
                w.t.stride as u32,
                w.t.kind,
                c.top as u32,
            ]
        };
        let project = |name: &str, w: &Weight, x: &Buffer| -> Result<Buffer> {
            let out = self.gpu.output(w.t.dims[1] * total)?;
            self.gpu.op(
                cmd,
                name,
                &[&w.buffer, x, &tiles, &out, &pairs],
                &geometry(w),
                w.t.dims[1],
                blocks,
            )?;
            Ok(out)
        };
        let [gate, up, down] = w;
        let act = if gate.t.kind == up.t.kind && gate.t.stride == up.t.stride {
            let act = self.gpu.output(c.ff * total)?;
            self.gpu.op(
                cmd,
                "quant_expert_gate_up",
                &[&gate.buffer, &up.buffer, x, &tiles, &act, &pairs],
                &geometry(gate),
                c.ff,
                blocks,
            )?;
            act
        } else {
            let g = project("quant_expert_gather", gate, x)?;
            let u = project("quant_expert_gather", up, x)?;
            self.gpu.activation(cmd, &g, &u, c.ff * total)?
        };
        let routed = project("quant_expert_packed", down, &act)?;
        let (zeros, gate_zeros) = self
            .no_shared
            .as_ref()
            .ok_or_else(|| invalid("MoE buffers missing"))?;
        let out = self.gpu.output(c.width * count)?;
        self.point(
            cmd,
            "combine_packed",
            &[&routed, zeros, weights, gate_zeros, &inverse, &out],
            &[c.width as u32, c.top as u32, count as u32],
            c.width * count,
        )?;
        Ok(out)
    }

    /// Encode `tokens` at `position`; returns the final normed rows (count ×
    /// width) and the command that will produce them.
    fn hidden(
        &self,
        s: &mut State,
        position: usize,
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<(Buffer, Command)> {
        let c = &self.c;
        let count = tokens.len();
        if count == 0 || position + count > s.rows {
            return Err(invalid("block exceeds the session context"));
        }
        let vocab = self.embedding.t.dims[1];
        if tokens.iter().any(|&t| t as usize >= vocab) {
            return Err(invalid("token outside vocabulary"));
        }
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let ids = self.gpu.bytes(&ids)?;
        let cmd = self.gpu.command()?;
        let mut x = self.gpu.output(c.width * count)?;
        let e = &self.embedding.t;
        self.point(
            &cmd,
            "embedding_batch",
            &[&self.embedding.buffer, &ids, &x],
            &[c.width as u32, e.stride as u32, e.kind, count as u32],
            c.width * count,
        )?;
        if let Some(scale) = c.embed_scale {
            x = self.unary(&cmd, "scale_rows", &x, scale, c.width * count)?;
        }
        let w = c.width;
        let rms = |cmd: &Command, a: &Buffer, b: Option<&Buffer>, norm: &Weight| {
            self.gpu.add_rms(cmd, a, b, norm, w, count, c.eps)
        };
        // `n` is the normed input of the next layer, then of the output head.
        let (mut x, mut n) = rms(&cmd, &x, None, &self.layers[0].attn_norm)?;
        // Decode-sized blocks encode every layer into one command buffer;
        // prefill blocks submit one per layer, waiting for the layer before so
        // at most two layers of workspace are in flight.
        let single = count <= 4;
        let mut cmd = cmd;
        let mut pending = None;
        for (i, (l, g)) in self.layers.iter().zip(&c.geometry).enumerate() {
            if cancel.is_cancelled() {
                return Err(cancelled());
            }
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.output_norm, |l| &l.attn_norm);
            (x, n) = objc2::rc::autoreleasepool(|_| {
                self.gpu.serial_scope(&cmd, || {
                    let cache = [&s.keys[i], &s.values[i]];
                    let mut a = self.attention(&cmd, l, g, cache, &n, position, count)?;
                    if let Some([post, _]) = &l.post_norms {
                        a = rms(&cmd, &a, None, post)?.1;
                    }
                    let (x, n) = rms(&cmd, &x, Some(&a), &l.ffn_norm)?;
                    let mut f = self.ffn(&cmd, &l.ffn, &n, count)?;
                    if let Some([_, post]) = &l.post_norms {
                        f = rms(&cmd, &f, None, post)?.1;
                    }
                    match l.out_scale {
                        Some(scale) => {
                            let x = self.binary(&cmd, "add_rows", &x, &f, w * count, 0)?;
                            let x = self.unary(&cmd, "scale_rows", &x, scale, w * count)?;
                            rms(&cmd, &x, None, next)
                        }
                        None => rms(&cmd, &x, Some(&f), next),
                    }
                })
            })?;
            if !single {
                self.gpu.submit(&cmd);
                if let Some(previous) = pending.replace(cmd) {
                    self.gpu.wait(previous)?;
                }
                cmd = self.gpu.command()?;
            }
        }
        if let Some(previous) = pending {
            self.gpu.wait(previous)?;
        }
        Ok((n, cmd))
    }
    /// Logits of the last of `count` normed rows.
    fn last_logits(&self, cmd: &Command, n: &Buffer, count: usize) -> Result<Buffer> {
        let w = self.c.width;
        let n = n.view_f32((count - 1) * w, w)?;
        let output = self.output.as_ref().unwrap_or(&self.embedding);
        let logits = self.project(cmd, output, &n, 1)?;
        match self.c.softcap {
            Some(cap) => {
                let vocab = self.embedding.t.dims[1];
                self.unary(cmd, "softcap_rows", &logits, cap, vocab)
            }
            None => Ok(logits),
        }
    }
    pub(crate) fn forward_block(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<Vec<f32>> {
        let (x, cmd) = self.hidden(s, history.len(), tokens, cancel)?;
        let logits = self.last_logits(&cmd, &x, tokens.len())?;
        self.gpu.finish(cmd, &logits)
    }
    pub(crate) fn forward_next(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<u32> {
        let (x, cmd) = self.hidden(s, history.len(), tokens, cancel)?;
        let logits = self.last_logits(&cmd, &x, tokens.len())?;
        let vocab = self.embedding.t.dims[1];
        Ok(self.gpu.finish_greedy(cmd, &logits, 1, vocab, None)?[0])
    }

    pub(crate) fn prefill_block_size(&self) -> usize {
        self.block
    }
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn metal_bytes(&self) -> usize {
        self.gpu.allocated_bytes()
    }
    pub(crate) fn metal_peak_bytes(&self) -> usize {
        self.gpu.peak_allocated_bytes()
    }
    pub(crate) fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub(crate) fn workspace_statistics(&self) -> (u64, u64, usize, usize) {
        self.gpu.workspace_statistics()
    }
    pub(crate) fn cache_statistics(&self) -> (u64, u64, Duration) {
        (0, 0, Duration::ZERO)
    }
}
