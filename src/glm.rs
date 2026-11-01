//! Independent GLM5-Next trunk: mHC, KDA, compressed NoPE MLA, pooled DSA and MoE.
use crate::{
    Result,
    engine::{Cancellation, DraftConfig},
    gguf::{Gguf, Tensor, Value},
    invalid,
    metal::{Buffer, Command, Gpu},
    weights::{ExpertCache, ExpertWeights, Weight},
};
use std::cell::{Cell, RefCell};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[path = "glm_mtp.rs"]
mod mtp;
use mtp::{Checkpoint, Draft, DraftState};

// A real matrix block for dense projections and grouped experts. Verification
// still uses only the one-to-four target positions requested by the MTP loop.
pub(crate) const PREFILL_BLOCK: usize = 64;

struct Config {
    width: usize,
    layers: usize,
    heads: usize,
    d: usize,
    conv: usize,
    hc: usize,
    sinkhorn: usize,
    hc_eps: f32,
    eps: f32,
    norm_eps: f32,
    dense: usize,
    ff: usize,
    eff: usize,
    experts: usize,
    top: usize,
    scale: f32,
    qr: usize,
    latent: usize,
    qk: usize,
    value: usize,
    idx: usize,
    ih: usize,
    pool: usize,
    topk: usize,
    lower: f32,
    sparse: Vec<u64>,
}
impl Config {
    fn load(g: &Gguf) -> Result<Self> {
        let n = |s: &str| g.integer(&format!("glm5-next.{s}"));
        let f = |s: &str| g.float(&format!("glm5-next.{s}"));
        let indexers = g.integers("glm5-next.attention.indexer.types")?;
        let c = Self {
            width: n("embedding_length")?,
            layers: n("block_count")?,
            heads: n("attention.head_count")?,
            d: n("kda.head_dim")?,
            conv: n("ssm.conv_kernel")?,
            hc: n("hyper_connection.count")?,
            sinkhorn: n("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            norm_eps: f("attention.layer_norm_epsilon")?,
            dense: n("leading_dense_block_count")?,
            ff: n("feed_forward_length")?,
            eff: n("expert_feed_forward_length")?,
            experts: n("expert_count")?,
            top: n("expert_used_count")?,
            scale: f("expert_weights_scale")?,
            qr: n("attention.q_lora_rank")?,
            latent: n("attention.kv_lora_rank")?,
            qk: n("attention.key_length_mla")?,
            value: n("attention.value_length_mla")?,
            idx: n("attention.indexer.key_length")?,
            ih: n("attention.indexer.head_count")?,
            pool: n("attention.indexer.kpool")?,
            topk: n("attention.indexer.top_k")?,
            lower: f("kda.gate_lower_bound")?,
            sparse: g.integers("glm5-next.attention.head_count_kv")?,
        };
        // This binding covers the verified 320B text checkpoint, not arbitrary GLM variants.
        if (
            c.width, c.layers, c.heads, c.d, c.conv, c.hc, c.dense, c.ff, c.eff, c.experts, c.top,
        ) != (4096, 45, 64, 128, 4, 4, 3, 12288, 2048, 288, 8)
            || (c.qr, c.latent, c.qk, c.value, c.idx, c.ih, c.pool, c.topk)
                != (1536, 512, 256, 256, 128, 32, 4, 2048)
            || c.sparse.len() != c.layers
            || c.sparse
                .iter()
                .enumerate()
                .any(|(i, &v)| v != u64::from(i % 4 == 3))
            || n("rope.dimension_count")? != 0
            || n("expert_shared_count")? != 1
            || n("expert_group_count")? != 1
            || n("expert_group_used_count")? != 1
            || n("expert_gating_func")? != 2
            || n("expert_weights_norm")? != 1
            || n("attention.indexer.kpool_select_tail")? != 1
            || indexers.len() != c.layers
            || indexers.iter().any(|&v| v != 1)
            || c.sinkhorn != 20
            || c.lower != -5.0
            || c.scale != 2.5
            || [c.eps, c.norm_eps, c.hc_eps]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid("unsupported or malformed glm5-next configuration"));
        }
        for key in ["glm5-next.swiglu_clamp_exp", "glm5-next.swiglu_clamp_shexp"] {
            let values = g.array(key)?;
            if values.len() != c.layers
                || values
                    .iter()
                    .any(|v| !matches!(v, Value::Float(x) if *x==10.0))
            {
                return Err(invalid("unsupported GLM SwiGLU limits"));
            }
        }
        Ok(c)
    }
    fn state_bytes(&self, context: usize) -> usize {
        self.sparse
            .iter()
            .map(|&s| {
                if s == 0 {
                    (self.heads * self.d * self.d + 3 * (self.conv - 1) * self.heads * self.d) * 4
                } else {
                    (self.latent * context
                        + 2 * self.idx * context
                        + self.idx * context.div_ceil(self.pool))
                        * 4
                }
            })
            .sum()
    }
}
struct Hc {
    function: Weight,
    base: Weight,
    scale: Weight,
    norm: Weight,
}
struct Kda {
    q: Weight,
    k: Weight,
    v: Weight,
    conv: [Weight; 3],
    fa: Weight,
    fb: Weight,
    beta: Weight,
    dt: Weight,
    a: Weight,
    ga: Weight,
    gb: Weight,
    norm: Weight,
    out: Weight,
}
struct Dsa {
    qa: Weight,
    qb: Weight,
    qn: Weight,
    kva: Weight,
    kvn: Weight,
    kb: Weight,
    vb: Weight,
    out: Weight,
    iq: Weight,
    ik: Weight,
    ikn: Weight,
    ikb: Weight,
    iw: Weight,
    gate: Weight,
    ape: Weight,
}
enum Mixer {
    Kda(Box<Kda>),
    Dsa(Box<Dsa>),
}
struct Dense {
    gate: Weight,
    up: Weight,
    down: Weight,
}
struct Moe {
    router: Weight,
    bias: Weight,
    experts: ExpertWeights,
    shared: Dense,
}
enum Ffn {
    Dense(Box<Dense>),
    Moe(Box<Moe>),
}
struct Layer {
    attention: Hc,
    ffn: Hc,
    mixer: Mixer,
    feed_forward: Ffn,
}
enum LayerState {
    Kda {
        recurrent: Buffer,
        conv: [Buffer; 3],
    },
    Dsa {
        latent: Buffer,
        key: Buffer,
        gate: Buffer,
        pooled: Buffer,
    },
}
pub(crate) struct State {
    layers: Vec<LayerState>,
    draft: Option<DraftState>,
}
pub(crate) struct Glm {
    g: RefCell<Gguf>,
    gpu: Gpu,
    c: Config,
    layers: Vec<Layer>,
    embedding: Weight,
    output: Weight,
    norm: Weight,
    ones: Buffer,
    head_ids: Buffer,
    bytes: usize,
    draft: Option<Draft>,
    trace: Option<(usize, PathBuf)>,
    trace_active: Cell<bool>,
    trace_events: RefCell<Vec<(String, Buffer)>>,
}
impl Glm {
    pub(crate) fn load(
        mut g: Gguf,
        budget: usize,
        context: usize,
        draft: Option<&DraftConfig>,
        prefix_cache: bool,
    ) -> Result<Self> {
        let c = Config::load(&g)?;
        if context == 0 || context > g.integer("glm5-next.context_length")? {
            return Err(invalid("invalid context capacity"));
        }
        let draft_model = draft.map(|d| Draft::open(&g, &c, d)).transpose()?;
        let draft_bytes = draft_model
            .as_ref()
            .map(|g| Draft::reserved_bytes(g, &c, context))
            .transpose()?
            .unwrap_or(0);
        let (mut dense, mut expert) = (0usize, 0usize);
        let mut mapping_padding = 0usize;
        for (name, t) in &g.tensors {
            mapping_padding = mapping_padding
                .checked_add(t.mapped_bytes()? - t.bytes)
                .ok_or_else(|| invalid("mapping padding overflow"))?;
            let sum = if name.ends_with("_exps.weight") {
                &mut expert
            } else {
                &mut dense
            };
            *sum = sum
                .checked_add(t.bytes)
                .ok_or_else(|| invalid("weight bytes overflow"))?;
        }
        // Bound all in-flight per-token DSA masks/probabilities and activation
        // buffers for a full prefill block, including device-page rounding.
        let workspace = 512 * 1024 * 1024 + 8 * PREFILL_BLOCK * context;
        let prefix_bytes = if prefix_cache {
            let n = context.min(1024);
            c.state_bytes(n)
                + if draft.is_some() {
                    (c.latent * n + 2 * c.idx * n + c.idx * n.div_ceil(c.pool) + c.width) * 4
                } else {
                    0
                }
        } else {
            0
        };
        let fixed = dense
            .checked_add(c.state_bytes(context))
            .and_then(|n| n.checked_add(mapping_padding))
            .and_then(|v| v.checked_add(workspace + 256 * 1024 * 1024))
            .and_then(|v| v.checked_add(draft_bytes + prefix_bytes))
            .ok_or_else(|| invalid("memory requirement overflow"))?;
        let per_slot = expert / c.experts;
        if per_slot == 0 {
            return Err(invalid("invalid GLM expert geometry"));
        }
        let slots = (budget.saturating_sub(fixed) / per_slot).min(c.experts);
        if slots == 0 {
            return Err(invalid(format!(
                "memory budget {budget} cannot hold required {} bytes",
                fixed + per_slot
            )));
        }
        let gpu = Gpu::new()?;
        gpu.configure_workspace(
            workspace
                + if draft.is_some() {
                    Draft::working_bytes(&c)
                } else {
                    0
                },
        );
        let vocab = g.array("tokenizer.ggml.tokens")?.len();
        let embedding = Weight::load(&mut g, &gpu, "token_embd.weight", &[c.width, vocab])?;
        let output = Weight::load(&mut g, &gpu, "output.weight", &[c.width, vocab])?;
        let norm = Weight::load(&mut g, &gpu, "output_norm.weight", &[c.width])?;
        norm.require_f32()?;
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("blk.{i}.");
            let attention = Self::hc_load(&mut g, &gpu, &p, "attn", &c)?;
            let ffn = Self::hc_load(&mut g, &gpu, &p, "ffn", &c)?;
            let mut w = |name: &str, dims: &[usize]| {
                Weight::load(&mut g, &gpu, &format!("{p}{name}"), dims)
            };
            let inner = c.heads * c.d;
            let mixer = if c.sparse[i] == 0 {
                let conv = [
                    w("ssm_conv1d_q.weight", &[c.conv, 1, inner])?,
                    w("ssm_conv1d_k.weight", &[c.conv, 1, inner])?,
                    w("ssm_conv1d_v.weight", &[c.conv, 1, inner])?,
                ];
                let l = Kda {
                    q: w("attn_q.weight", &[c.width, inner])?,
                    k: w("attn_k.weight", &[c.width, inner])?,
                    v: w("attn_v.weight", &[c.width, inner])?,
                    conv,
                    fa: w("ssm_f_a.weight", &[c.width, c.d])?,
                    fb: w("ssm_f_b.weight", &[c.d, inner])?,
                    beta: w("ssm_beta.weight", &[c.width, c.heads])?,
                    dt: w("ssm_dt.bias", &[inner])?,
                    a: w("ssm_a", &[c.heads])?,
                    ga: w("ssm_g_a.weight", &[c.width, c.d])?,
                    gb: w("ssm_g_b.weight", &[c.d, inner])?,
                    norm: w("ssm_norm.weight", &[c.d])?,
                    out: w("attn_output.weight", &[inner, c.width])?,
                };
                for t in [&l.dt, &l.a, &l.norm] {
                    t.require_f32()?;
                }
                Mixer::Kda(Box::new(l))
            } else {
                let l = Self::dsa_load(&mut g, &gpu, &p, &c)?;
                Mixer::Dsa(Box::new(l))
            };
            let feed_forward = if i < c.dense {
                Ffn::Dense(Box::new(Self::dense_load(
                    &mut g, &gpu, &p, "", c.width, c.ff,
                )?))
            } else {
                Ffn::Moe(Box::new(Self::moe_load(&mut g, &gpu, &p, &c, slots)?))
            };
            layers.push(Layer {
                attention,
                ffn,
                mixer,
                feed_forward,
            });
        }
        let ones = gpu.floats(&vec![1.0; c.width * c.hc])?;
        let head_ids = gpu.bytes(
            &(0..c.heads as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let draft = draft_model
            .zip(draft)
            .map(|(g, cfg)| Draft::load(g, &gpu, &c, cfg))
            .transpose()?;
        Ok(Self {
            g: RefCell::new(g),
            gpu,
            c,
            layers,
            embedding,
            output,
            norm,
            ones,
            head_ids,
            bytes: fixed + slots * per_slot,
            draft,
            trace: None,
            trace_active: Cell::new(false),
            trace_events: RefCell::new(Vec::new()),
        })
    }
    fn dsa_load(g: &mut Gguf, gpu: &Gpu, p: &str, c: &Config) -> Result<Dsa> {
        let mut w = |name: &str, dims: &[usize]| Weight::load(g, gpu, &format!("{p}{name}"), dims);
        let l = Dsa {
            qa: w("attn_q_a.weight", &[c.width, c.qr])?,
            qb: w("attn_q_b.weight", &[c.qr, c.heads * c.qk])?,
            qn: w("attn_q_a_norm.weight", &[c.qr])?,
            kva: w("attn_kv_a_mqa.weight", &[c.width, c.latent])?,
            kvn: w("attn_kv_a_norm.weight", &[c.latent])?,
            kb: w("attn_k_b.weight", &[c.qk, c.latent, c.heads])?,
            vb: w("attn_v_b.weight", &[c.latent, c.value, c.heads])?,
            out: w("attn_output.weight", &[c.heads * c.value, c.width])?,
            iq: w("indexer.attn_q_b.weight", &[c.qr, c.ih * c.idx])?,
            ik: w("indexer.attn_k.weight", &[c.width, c.idx])?,
            ikn: w("indexer.k_norm.weight", &[c.idx])?,
            ikb: w("indexer.k_norm.bias", &[c.idx])?,
            iw: w("indexer.proj.weight", &[c.width, c.ih])?,
            gate: w("indexer_compressor_gate.weight", &[c.width, c.idx])?,
            ape: w("indexer_compressor_ape.weight", &[c.idx, c.pool])?,
        };
        for t in [&l.qn, &l.kvn, &l.ikn, &l.ikb, &l.ape] {
            t.require_f32()?;
        }
        Ok(l)
    }
    fn moe_load(g: &mut Gguf, gpu: &Gpu, p: &str, c: &Config, slots: usize) -> Result<Moe> {
        let router = Weight::load(
            g,
            gpu,
            &format!("{p}ffn_gate_inp.weight"),
            &[c.width, c.experts],
        )?;
        let bias = Weight::load(g, gpu, &format!("{p}exp_probs_b.bias"), &[c.experts])?;
        bias.require_f32()?;
        let tensors = [
            ("gate", [c.width, c.eff, c.experts]),
            ("up", [c.width, c.eff, c.experts]),
            ("down", [c.eff, c.width, c.experts]),
        ]
        .into_iter()
        .map(|(role, dims)| {
            let name = format!("{p}ffn_{role}_exps.weight");
            let t = g.tensor(&name)?.clone();
            if t.dims != dims {
                return Err(invalid(format!("invalid GLM expert shape: {name}")));
            }
            Ok(t)
        })
        .collect::<Result<Vec<_>>>()?;
        let tensors: [Tensor; 3] = tensors
            .try_into()
            .map_err(|_| invalid("GLM expert bindings"))?;
        let experts = if slots == c.experts {
            let loaded = tensors
                .iter()
                .map(|t| {
                    Ok(Weight {
                        buffer: gpu.load_weight(g, t)?,
                        t: t.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ExpertWeights::Resident(
                loaded
                    .try_into()
                    .map_err(|_| invalid("GLM expert bindings"))?,
            )
        } else {
            let buffers = tensors
                .iter()
                .map(|t| gpu.empty_bytes(t.bytes / c.experts * slots))
                .collect::<Result<Vec<_>>>()?;
            ExpertWeights::Cached(RefCell::new(ExpertCache {
                tensors,
                buffers: buffers
                    .try_into()
                    .map_err(|_| invalid("GLM slot bindings"))?,
                slots: vec![None; slots],
                clock: 0,
                hits: 0,
                misses: 0,
                read: Duration::ZERO,
            }))
        };
        let shared = Self::dense_load(g, gpu, p, "_shexp", c.width, c.eff)?;
        Ok(Moe {
            router,
            bias,
            experts,
            shared,
        })
    }
    fn hc_load(g: &mut Gguf, gpu: &Gpu, p: &str, role: &str, c: &Config) -> Result<Hc> {
        let w = |g: &mut Gguf, name: &str, dims: &[usize]| {
            Weight::load(g, gpu, &format!("{p}{name}"), dims)
        };
        let h = Hc {
            function: w(
                g,
                &format!("hc_{role}_fn.weight"),
                &[c.width * c.hc, (2 + c.hc) * c.hc],
            )?,
            base: w(g, &format!("hc_{role}_base.weight"), &[(2 + c.hc) * c.hc])?,
            scale: w(g, &format!("hc_{role}_scale.weight"), &[3])?,
            norm: w(g, &format!("{role}_norm.weight"), &[c.width])?,
        };
        for t in [&h.base, &h.scale, &h.norm] {
            t.require_f32()?;
        }
        Ok(h)
    }
    fn dense_load(
        g: &mut Gguf,
        gpu: &Gpu,
        p: &str,
        suffix: &str,
        width: usize,
        ff: usize,
    ) -> Result<Dense> {
        Ok(Dense {
            gate: Weight::load(g, gpu, &format!("{p}ffn_gate{suffix}.weight"), &[width, ff])?,
            up: Weight::load(g, gpu, &format!("{p}ffn_up{suffix}.weight"), &[width, ff])?,
            down: Weight::load(g, gpu, &format!("{p}ffn_down{suffix}.weight"), &[ff, width])?,
        })
    }
    pub(crate) fn state(&self, context: usize) -> Result<State> {
        let c = &self.c;
        let layers = c
            .sparse
            .iter()
            .map(|&s| {
                if s == 0 {
                    Ok(LayerState::Kda {
                        recurrent: self.gpu.zeros(c.heads * c.d * c.d)?,
                        conv: [
                            self.gpu.zeros((c.conv - 1) * c.heads * c.d)?,
                            self.gpu.zeros((c.conv - 1) * c.heads * c.d)?,
                            self.gpu.zeros((c.conv - 1) * c.heads * c.d)?,
                        ],
                    })
                } else {
                    Ok(LayerState::Dsa {
                        latent: self.gpu.zeros(c.latent * context)?,
                        key: self.gpu.zeros(c.idx * context)?,
                        gate: self.gpu.zeros(c.idx * context)?,
                        pooled: self.gpu.zeros(c.idx * context.div_ceil(c.pool))?,
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let draft = self
            .draft
            .as_ref()
            .map(|_| -> Result<DraftState> {
                Ok(DraftState {
                    mixer: self.dsa_state(context)?,
                    target_hidden: self.gpu.zeros(c.width)?,
                })
            })
            .transpose()?;
        Ok(State { layers, draft })
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
        self.g.borrow().bytes_read
    }
    pub(crate) fn workspace_statistics(&self) -> (u64, u64, usize, usize) {
        self.gpu.workspace_statistics()
    }
    pub(crate) fn cache_statistics(&self) -> (u64, u64, Duration) {
        self.layers
            .iter()
            .fold((0, 0, Duration::ZERO), |(h, m, t), l| {
                if let Ffn::Moe(moe) = &l.feed_forward
                    && let ExpertWeights::Cached(cache) = &moe.experts
                {
                    let cache = cache.borrow();
                    return (h + cache.hits, m + cache.misses, t + cache.read);
                }
                (h, m, t)
            })
    }
    pub(crate) fn set_trace(&mut self, position: usize, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        self.trace = Some((position, path.to_path_buf()));
        Ok(())
    }
    fn record(&self, name: &str, b: &Buffer) {
        if self.trace_active.get() {
            self.trace_events
                .borrow_mut()
                .push((name.into(), b.clone()));
        }
    }
    fn flush_trace(&self, layer: usize) -> Result<()> {
        if let Some((_, path)) = &self.trace {
            for (name, b) in self.trace_events.borrow_mut().drain(..) {
                let mut out = BufWriter::new(
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path.join(format!("{layer}.{name}.f32")))?,
                );
                for x in self.gpu.read_u32(&b, b.length() / 4)? {
                    out.write_all(&x.to_le_bytes())?;
                }
                out.flush()?;
            }
        }
        Ok(())
    }
    fn point(&self, cmd: &Command, name: &str, b: &[&Buffer], p: &[u32], n: usize) -> Result<()> {
        self.gpu.op(cmd, name, b, p, n.div_ceil(32), 1)
    }
    #[allow(clippy::too_many_arguments)] // Explicit RMS/L2 kernel shape and precision.
    fn norm(
        &self,
        cmd: &Command,
        x: &Buffer,
        w: &Buffer,
        width: usize,
        rows: usize,
        eps: f32,
        l2: bool,
    ) -> Result<Buffer> {
        let y = self.gpu.output(width * rows)?;
        self.gpu.op(
            cmd,
            "rms",
            &[x, w, &y],
            &[
                width as u32,
                (w.length() / 4) as u32,
                0,
                width as u32,
                eps.to_bits(),
                l2 as u32,
            ],
            rows,
            1,
        )?;
        Ok(y)
    }
    fn hc_pre(&self, cmd: &Command, h: &Hc, x: &Buffer) -> Result<(Buffer, Buffer)> {
        let c = &self.c;
        let flat = self.norm(cmd, x, &self.ones, c.width * c.hc, 1, c.eps, false)?;
        let mixes = h.function.project(&self.gpu, cmd, &flat)?;
        let gates = self.gpu.output((2 + c.hc) * c.hc)?;
        self.point(
            cmd,
            "glm_hc_gates",
            &[&mixes, &h.base.buffer, &h.scale.buffer, &gates],
            &[c.hc as u32, c.sinkhorn as u32, c.hc_eps.to_bits()],
            1,
        )?;
        let y = self.gpu.output(c.width)?;
        self.point(
            cmd,
            "glm_hc_pre",
            &[x, &gates, &y],
            &[c.width as u32, c.hc as u32],
            c.width,
        )?;
        let normed = self.norm(cmd, &y, &h.norm.buffer, c.width, 1, c.eps, false)?;
        Ok((normed, gates))
    }
    fn hc_post(
        &self,
        cmd: &Command,
        residual: &Buffer,
        x: &Buffer,
        gates: &Buffer,
    ) -> Result<Buffer> {
        let c = &self.c;
        let y = self.gpu.output(c.width * c.hc)?;
        self.point(
            cmd,
            "glm_hc_post",
            &[residual, x, gates, &y],
            &[c.width as u32, c.hc as u32],
            c.width * c.hc,
        )?;
        Ok(y)
    }
    fn kda(&self, cmd: &Command, l: &Kda, s: &LayerState, x: &Buffer) -> Result<Buffer> {
        let q = l.q.project(&self.gpu, cmd, x)?;
        let k = l.k.project(&self.gpu, cmd, x)?;
        let v = l.v.project(&self.gpu, cmd, x)?;
        let f = l.fa.project(&self.gpu, cmd, x)?;
        let f = l.fb.project(&self.gpu, cmd, &f)?;
        let beta = l.beta.project(&self.gpu, cmd, x)?;
        let gate = l.ga.project(&self.gpu, cmd, x)?;
        let gate = l.gb.project(&self.gpu, cmd, &gate)?;
        let y = self.kda_projected(cmd, l, s, &[q, k, v, f, beta, gate])?;
        l.out.project(&self.gpu, cmd, &y)
    }
    fn kda_projected(
        &self,
        cmd: &Command,
        l: &Kda,
        s: &LayerState,
        inputs: &[Buffer; 6],
    ) -> Result<Buffer> {
        let LayerState::Kda { recurrent, conv } = s else {
            return Err(invalid("GLM KDA state mismatch"));
        };
        let c = &self.c;
        let inner = c.heads * c.d;
        let mut qkv = Vec::with_capacity(3);
        for (i, projected) in inputs[..3].iter().enumerate() {
            let y = self.gpu.output(inner)?;
            let cw = &l.conv[i];
            self.point(
                cmd,
                "convolution",
                &[projected, &cw.buffer, &conv[i], &y],
                &[
                    inner as u32,
                    c.conv as u32,
                    1,
                    cw.t.kind,
                    cw.t.stride as u32,
                ],
                inner,
            )?;
            qkv.push(if i < 2 {
                self.norm(cmd, &y, &self.ones, c.d, c.heads, 1e-6, true)?
            } else {
                y
            });
        }
        let f = &inputs[3];
        let beta = &inputs[4];
        let out = self.gpu.output(inner)?;
        self.gpu.op(
            cmd,
            "glm_kda",
            &[
                &qkv[0],
                &qkv[1],
                &qkv[2],
                f,
                beta,
                &l.dt.buffer,
                &l.a.buffer,
                recurrent,
                &out,
            ],
            &[c.d as u32, c.lower.to_bits()],
            c.d,
            c.heads,
        )?;
        self.record("kda_scan_out", &out);
        let normed = self.norm(cmd, &out, &l.norm.buffer, c.d, c.heads, c.eps, false)?;
        let gate = &inputs[5];
        let y = self.gpu.output(inner)?;
        self.point(
            cmd,
            "mul_gate",
            &[&normed, gate, &y],
            &[inner as u32],
            inner,
        )?;
        Ok(y)
    }
    fn heads(&self, cmd: &Command, w: &Weight, x: &Buffer) -> Result<Buffer> {
        let rows = w.t.dims[1];
        let y = self.gpu.output(rows * self.c.heads)?;
        self.gpu.op(
            cmd,
            "project_experts",
            &[&w.buffer, x, &self.head_ids, &y],
            &[
                w.t.dims[0] as u32,
                rows as u32,
                w.t.stride as u32,
                w.t.kind,
                w.t.dims[0] as u32,
            ],
            rows,
            self.c.heads,
        )?;
        Ok(y)
    }
    fn dsa_cache(
        &self,
        cmd: &Command,
        l: &Dsa,
        s: &LayerState,
        x: &Buffer,
        position: usize,
    ) -> Result<Buffer> {
        let LayerState::Dsa {
            latent,
            key,
            gate,
            pooled,
        } = s
        else {
            return Err(invalid("GLM MLA state mismatch"));
        };
        let c = &self.c;
        let length = position + 1;
        let kv = l.kva.project(&self.gpu, cmd, x)?;
        let kv = self.norm(cmd, &kv, &l.kvn.buffer, c.latent, 1, c.eps, false)?;
        self.point(
            cmd,
            "cache_write",
            &[&kv, latent],
            &[c.latent as u32, position as u32],
            c.latent,
        )?;
        let ik = l.ik.project(&self.gpu, cmd, x)?;
        let ikn = self.gpu.output(c.idx)?;
        self.gpu.op(
            cmd,
            "glm_layer_norm",
            &[&ik, &l.ikn.buffer, &l.ikb.buffer, &ikn],
            &[c.idx as u32, c.norm_eps.to_bits()],
            1,
            1,
        )?;
        let ig = l.gate.project(&self.gpu, cmd, x)?;
        for (v, cache) in [(&ikn, key), (&ig, gate)] {
            self.point(
                cmd,
                "cache_write",
                &[v, cache],
                &[c.idx as u32, position as u32],
                c.idx,
            )?;
        }
        if length.is_multiple_of(c.pool) {
            self.point(
                cmd,
                "glm_pool",
                &[key, gate, &l.ape.buffer, pooled],
                &[c.idx as u32, c.pool as u32, (length / c.pool - 1) as u32],
                c.idx,
            )?;
        }
        Ok(kv)
    }
    fn dsa(
        &self,
        cmd: &Command,
        l: &Dsa,
        s: &LayerState,
        x: &Buffer,
        position: usize,
    ) -> Result<Buffer> {
        let LayerState::Dsa { latent, pooled, .. } = s else {
            return Err(invalid("GLM MLA state mismatch"));
        };
        let c = &self.c;
        let length = position + 1;
        let qr = l.qa.project(&self.gpu, cmd, x)?;
        let qr = self.norm(cmd, &qr, &l.qn.buffer, c.qr, 1, c.eps, false)?;
        let q = l.qb.project(&self.gpu, cmd, &qr)?;
        let absorbed = self.heads(cmd, &l.kb, &q)?;
        let kv = self.dsa_cache(cmd, l, s, x, position)?;
        let iq = l.iq.project(&self.gpu, cmd, &qr)?;
        let iw = l.iw.project(&self.gpu, cmd, x)?;
        let blocks = length / c.pool;
        let score = self.gpu.output(blocks.max(1))?;
        if blocks > 0 {
            self.gpu.op(
                cmd,
                "glm_index_score",
                &[&iq, pooled, &iw, &score],
                &[c.idx as u32, c.ih as u32],
                blocks,
                1,
            )?;
        }
        let mask = self.gpu.output(length)?;
        if blocks > c.topk / c.pool && blocks <= 4096 {
            self.gpu.op(
                cmd,
                "glm_index_mask_sorted",
                &[&score, &mask],
                &[length as u32, c.pool as u32, c.topk as u32],
                1,
                1,
            )?;
        } else {
            self.point(
                cmd,
                "glm_index_mask",
                &[&score, &mask],
                &[length as u32, c.pool as u32, c.topk as u32],
                blocks.max(1),
            )?;
        }
        let sparse = blocks > c.topk / c.pool;
        let capacity = if sparse {
            (c.topk / c.pool) * c.pool + length % c.pool
        } else {
            length
        };
        // GLM retains whole selected pools plus the unfinished tail. Only their
        // visit order changes here; the mask and selected set remain unchanged.
        let indices = self.gpu.output(capacity)?;
        let offsets = self.gpu.output(33)?;
        if sparse {
            self.gpu.op(
                cmd,
                "compact_mask_ordered",
                &[&mask, &indices, &offsets],
                &[length as u32, position as u32, capacity as u32],
                1,
                1,
            )?;
        }
        let parts = capacity.div_ceil(64);
        let partial = self.gpu.output(c.heads * parts * (c.latent + 2))?;
        self.gpu.op(
            cmd,
            "glm_mla_online_parts",
            &[&absorbed, latent, &indices, &offsets, &partial],
            &[length as u32, parts as u32, u32::from(sparse)],
            parts,
            c.heads,
        )?;
        let values = self.gpu.output(c.latent * c.heads)?;
        self.gpu.op(
            cmd,
            "glm_mla_online_merge",
            &[&partial, &values],
            &[parts as u32],
            c.heads,
            1,
        )?;
        let values = self.heads(cmd, &l.vb, &values)?;
        self.record("kv_cmpr", &kv);
        self.record("q_absorbed", &absorbed);
        self.record("indexer_sel", &mask);
        l.out.project(&self.gpu, cmd, &values)
    }
    fn activation(&self, cmd: &Command, gate: &Buffer, up: &Buffer, n: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(
            cmd,
            "glm_swiglu",
            &[gate, up, &y],
            &[n as u32, 10f32.to_bits()],
            n,
        )?;
        Ok(y)
    }
    fn dense(&self, cmd: &Command, l: &Dense, x: &Buffer) -> Result<Buffer> {
        let g = l.gate.project(&self.gpu, cmd, x)?;
        let u = l.up.project(&self.gpu, cmd, x)?;
        let y = self.activation(cmd, &g, &u, l.gate.t.dims[1])?;
        l.down.project(&self.gpu, cmd, &y)
    }
    fn merge(
        &self,
        cmd: &Command,
        routed: &Buffer,
        shared: &Buffer,
        weights: &Buffer,
    ) -> Result<Buffer> {
        let c = &self.c;
        let y = self.gpu.output(c.width)?;
        self.point(
            cmd,
            "glm_merge",
            &[routed, shared, weights, &y],
            &[c.width as u32, c.top as u32],
            c.width,
        )?;
        Ok(y)
    }
    fn moe(
        &self,
        cmd: Command,
        l: &Moe,
        x: &Buffer,
        cancel: &Cancellation,
    ) -> Result<(Buffer, Command)> {
        let c = &self.c;
        let logits = l.router.project(&self.gpu, &cmd, x)?;
        let ids = self.gpu.output(c.top)?;
        let weights = self.gpu.output(c.top)?;
        self.point(
            &cmd,
            "glm_route",
            &[&logits, &l.bias.buffer, &ids, &weights],
            &[c.experts as u32, c.top as u32, c.scale.to_bits()],
            1,
        )?;
        self.record("ffn_moe_topk", &ids);
        self.record("ffn_moe_logits", &logits);
        match &l.experts {
            ExpertWeights::Resident(m) => {
                let project = |w: &Weight, x: &Buffer, per: bool| -> Result<Buffer> {
                    let rows = w.t.dims[1];
                    let y = self.gpu.output(rows * c.top)?;
                    self.gpu.op(
                        &cmd,
                        "project_experts",
                        &[&w.buffer, x, &ids, &y],
                        &[
                            w.t.dims[0] as u32,
                            rows as u32,
                            w.t.stride as u32,
                            w.t.kind,
                            if per { w.t.dims[0] as u32 } else { 0 },
                        ],
                        rows,
                        c.top,
                    )?;
                    Ok(y)
                };
                let g = project(&m[0], x, false)?;
                let u = project(&m[1], x, false)?;
                let a = self.activation(&cmd, &g, &u, c.eff * c.top)?;
                let routed = project(&m[2], &a, true)?;
                let shared = self.dense(&cmd, &l.shared, x)?;
                Ok((self.merge(&cmd, &routed, &shared, &weights)?, cmd))
            }
            ExpertWeights::Cached(cache) => {
                self.gpu.complete(cmd)?;
                let ids = self.gpu.read_u32(&ids, c.top)?;
                let mut cache = cache.borrow_mut();
                let mut slot = cache.slot(
                    &self.gpu,
                    &mut self.g.borrow_mut(),
                    c.experts,
                    ids[0] as usize,
                    &[],
                )?;
                let routed = self.gpu.output(c.width * c.top)?;
                let first = self.gpu.command()?;
                let shared = self.dense(&first, &l.shared, x)?;
                let mut current = Some(first);
                let mut i = 0;
                while i < ids.len() {
                    if cancel.is_cancelled() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "generation cancelled",
                        ));
                    }
                    let cmd = match current.take() {
                        Some(cmd) => cmd,
                        None => self.gpu.command()?,
                    };
                    let mut slots = vec![slot];
                    while i + slots.len() < ids.len() {
                        let Some(hit) = cache.hit(ids[i + slots.len()] as usize) else {
                            break;
                        };
                        slots.push(hit);
                    }
                    let count = slots.len();
                    let slot_ids = self.gpu.bytes(
                        &slots
                            .iter()
                            .flat_map(|&s| (s as u32).to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let project = |index: usize, x: &Buffer, per: bool| -> Result<Buffer> {
                        let t = &cache.tensors[index];
                        let y = self.gpu.output(t.dims[1] * count)?;
                        self.gpu.op(
                            &cmd,
                            "project_experts",
                            &[&cache.buffers[index], x, &slot_ids, &y],
                            &[
                                t.dims[0] as u32,
                                t.dims[1] as u32,
                                t.stride as u32,
                                t.kind,
                                if per { t.dims[0] as u32 } else { 0 },
                            ],
                            t.dims[1],
                            count,
                        )?;
                        Ok(y)
                    };
                    let g = project(0, x, false)?;
                    let u = project(1, x, false)?;
                    let a = self.activation(&cmd, &g, &u, c.eff * count)?;
                    let out = project(2, &a, true)?;
                    self.point(
                        &cmd,
                        "copy_into",
                        &[&out, &routed],
                        &[(c.width * count) as u32, (i * c.width) as u32],
                        c.width * count,
                    )?;
                    i += count;
                    self.gpu.submit(&cmd);
                    let next = if i < ids.len() && cache.slots.len() > slots.len() {
                        Some(cache.slot(
                            &self.gpu,
                            &mut self.g.borrow_mut(),
                            c.experts,
                            ids[i] as usize,
                            &slots,
                        ))
                    } else {
                        None
                    };
                    self.gpu.wait(cmd)?;
                    if let Some(next) = next {
                        slot = next?;
                    } else if i < ids.len() {
                        slot = cache.slot(
                            &self.gpu,
                            &mut self.g.borrow_mut(),
                            c.experts,
                            ids[i] as usize,
                            &[],
                        )?;
                    }
                }
                let cmd = self.gpu.command()?;
                Ok((self.merge(&cmd, &routed, &shared, &weights)?, cmd))
            }
        }
    }
    fn copy(&self, cmd: &Command, x: &Buffer, n: usize, offset: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(cmd, "copy_range", &[x, &y], &[n as u32, offset as u32], n)?;
        Ok(y)
    }
    fn pack(&self, cmd: &Command, inputs: &[Buffer], width: usize) -> Result<Buffer> {
        let x = self.gpu.output(width * inputs.len())?;
        for (i, input) in inputs.iter().enumerate() {
            self.point(
                cmd,
                "copy_into",
                &[input, &x],
                &[width as u32, (i * width) as u32],
                width,
            )?;
        }
        Ok(x)
    }
    fn project_many(&self, cmd: &Command, w: &Weight, inputs: &[Buffer]) -> Result<Vec<Buffer>> {
        let x = self.pack(cmd, inputs, w.t.dims[0])?;
        let rows = w.t.dims.get(1).copied().unwrap_or(1);
        let y = self.gpu.matmul(
            cmd,
            &w.buffer,
            w.t.dims[0],
            rows,
            w.t.stride,
            w.t.kind,
            0,
            &x,
            inputs.len(),
        )?;
        (0..inputs.len())
            .map(|i| self.copy(cmd, &y, rows, i * rows))
            .collect()
    }
    fn dense_many(&self, cmd: &Command, l: &Dense, x: &[Buffer]) -> Result<Vec<Buffer>> {
        let gate = self.project_many(cmd, &l.gate, x)?;
        let up = self.project_many(cmd, &l.up, x)?;
        let hidden = l.gate.t.dims[1];
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| self.activation(cmd, g, u, hidden))
            .collect::<Result<Vec<_>>>()?;
        self.project_many(cmd, &l.down, &activated)
    }
    fn moe_many(
        &self,
        cmd: Command,
        f: &Moe,
        inputs: &[Buffer],
        cancel: &crate::engine::Cancellation,
    ) -> Result<(Vec<Buffer>, Command)> {
        let c = &self.c;
        let scores = self.project_many(&cmd, &f.router, inputs)?;
        let mut ids = Vec::with_capacity(inputs.len());
        let mut weights = Vec::with_capacity(inputs.len());
        for score in &scores {
            let id = self.gpu.output(c.top)?;
            let weight = self.gpu.output(c.top)?;
            self.point(
                &cmd,
                "glm_route",
                &[score, &f.bias.buffer, &id, &weight],
                &[c.experts as u32, c.top as u32, c.scale.to_bits()],
                1,
            )?;
            ids.push(id);
            weights.push(weight);
        }
        self.gpu.complete(cmd)?;
        // Only the small [token, top-k] ID array crosses to the host. Activations,
        // routing probabilities and expert results stay on the GPU.
        let mut groups = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (token, id) in ids.iter().enumerate() {
            for (rank, expert) in self.gpu.read_u32(id, c.top)?.into_iter().enumerate() {
                groups
                    .entry(expert as usize)
                    .or_default()
                    .push((token, rank));
            }
        }
        let groups: Vec<_> = groups.into_iter().collect();
        let routed = (0..inputs.len())
            .map(|_| self.gpu.output(c.width * c.top))
            .collect::<Result<Vec<_>>>()?;
        let mut cmd = self.gpu.command()?;
        let shared = self.dense_many(&cmd, &f.shared, inputs)?;
        let mut cache = match &f.experts {
            ExpertWeights::Cached(c) => Some(c.borrow_mut()),
            _ => None,
        };
        let mut slot = if let Some(cache) = cache.as_mut() {
            cache.slot(
                &self.gpu,
                &mut self.g.borrow_mut(),
                c.experts,
                groups[0].0,
                &[],
            )?
        } else {
            groups[0].0
        };
        let mut pinned = Vec::new();
        for (group, &(expert, ref assignments)) in groups.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            let (buffers, tensors): ([&Buffer; 3], [&Tensor; 3]) = match (&f.experts, &cache) {
                (ExpertWeights::Resident(w), _) => (
                    [&w[0].buffer, &w[1].buffer, &w[2].buffer],
                    [&w[0].t, &w[1].t, &w[2].t],
                ),
                (_, Some(cache)) => (
                    [&cache.buffers[0], &cache.buffers[1], &cache.buffers[2]],
                    [&cache.tensors[0], &cache.tensors[1], &cache.tensors[2]],
                ),
                _ => unreachable!(),
            };
            let expert_slot = if cache.is_some() { slot } else { expert };
            let xs: Vec<_> = assignments
                .iter()
                .map(|&(token, _)| inputs[token].clone())
                .collect();
            let x = self.pack(&cmd, &xs, c.width)?;
            let project = |index: usize, x: &Buffer| {
                let t = tensors[index];
                self.gpu.matmul(
                    &cmd,
                    buffers[index],
                    t.dims[0],
                    t.dims[1],
                    t.stride,
                    t.kind,
                    expert_slot,
                    x,
                    assignments.len(),
                )
            };
            let g = project(0, &x)?;
            let u = project(1, &x)?;
            let act = self.activation(&cmd, &g, &u, c.eff * assignments.len())?;
            let y = project(2, &act)?;
            for (j, &(token, rank)) in assignments.iter().enumerate() {
                let result = self.copy(&cmd, &y, c.width, j * c.width)?;
                self.point(
                    &cmd,
                    "copy_into",
                    &[&result, &routed[token]],
                    &[c.width as u32, (rank * c.width) as u32],
                    c.width,
                )?;
            }
            if let Some(cache) = cache.as_mut() {
                pinned.push(slot);
                if group + 1 < groups.len()
                    && let Some(hit) = cache.hit(groups[group + 1].0)
                {
                    slot = hit;
                    continue;
                }
                self.gpu.submit(&cmd);
                let next = if group + 1 < groups.len() && cache.slots.len() > pinned.len() {
                    Some(cache.slot(
                        &self.gpu,
                        &mut self.g.borrow_mut(),
                        c.experts,
                        groups[group + 1].0,
                        &pinned,
                    ))
                } else {
                    None
                };
                // Even a failed read cannot release resources still read by the GPU.
                self.gpu.wait(cmd)?;
                pinned.clear();
                slot = if let Some(next) = next {
                    next?
                } else if group + 1 < groups.len() {
                    cache.slot(
                        &self.gpu,
                        &mut self.g.borrow_mut(),
                        c.experts,
                        groups[group + 1].0,
                        &[],
                    )?
                } else {
                    slot
                };
                cmd = self.gpu.command()?;
            }
        }
        let out = (0..inputs.len())
            .map(|i| self.merge(&cmd, &routed[i], &shared[i], &weights[i]))
            .collect::<Result<Vec<_>>>()?;
        Ok((out, cmd))
    }
    pub(crate) fn forward_block(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<Vec<f32>> {
        let frame = self.target_block(s, history.len(), tokens, cancel, false)?;
        self.catch_up_committed(s, history.len(), tokens, &frame.hidden, cancel)?;
        let cmd = self.gpu.command()?;
        let logits = self.output.project(
            &self.gpu,
            &cmd,
            frame
                .hidden
                .last()
                .ok_or_else(|| invalid("empty GLM block"))?,
        )?;
        self.gpu.finish(cmd, &logits)
    }
    pub(crate) fn forward_next(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &Cancellation,
    ) -> Result<u32> {
        let frame = self.target_block(s, history.len(), tokens, cancel, false)?;
        self.catch_up_committed(s, history.len(), tokens, &frame.hidden, cancel)?;
        self.select_hidden(&frame.hidden[frame.hidden.len() - 1..])
            .map(|x| x[0])
    }
    fn prefill_hidden(
        &self,
        s: &mut State,
        position: usize,
        tokens: &[u32],
        cancel: &Cancellation,
        snapshots: &mut [Checkpoint],
    ) -> Result<Vec<Buffer>> {
        self.trace_active.set(false);
        self.trace_events.borrow_mut().clear();
        let c = &self.c;
        let cmd = self.gpu.command()?;
        let mut hidden = Vec::with_capacity(tokens.len());
        for &token in tokens {
            if token as usize >= self.embedding.t.dims[1] {
                return Err(invalid("token outside vocabulary"));
            }
            let emb = self.gpu.output(c.width)?;
            self.point(
                &cmd,
                "embedding",
                &[&self.embedding.buffer, &emb],
                &[
                    c.width as u32,
                    0,
                    self.embedding.t.stride as u32,
                    self.embedding.t.kind,
                    token,
                ],
                c.width,
            )?;
            let h = self.gpu.output(c.width * c.hc)?;
            self.point(
                &cmd,
                "repeat_emb",
                &[&emb, &h],
                &[c.width as u32, c.hc as u32],
                c.width * c.hc,
            )?;
            hidden.push(h);
        }
        self.gpu.complete(cmd)?;
        for (layer, l) in self.layers.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            hidden = objc2::rc::autoreleasepool(|_| -> Result<Vec<Buffer>> {
                let cmd = self.gpu.command()?;
                let mut attention_inputs = Vec::with_capacity(tokens.len());
                let mut attention_gates = Vec::with_capacity(tokens.len());
                for h in &hidden {
                    let (x, g) = self.hc_pre(&cmd, &l.attention, h)?;
                    attention_inputs.push(x);
                    attention_gates.push(g);
                }
                let attention = match &l.mixer {
                    Mixer::Kda(k) => {
                        let q = self.project_many(&cmd, &k.q, &attention_inputs)?;
                        let keys = self.project_many(&cmd, &k.k, &attention_inputs)?;
                        let v = self.project_many(&cmd, &k.v, &attention_inputs)?;
                        let f = self.project_many(&cmd, &k.fa, &attention_inputs)?;
                        let f = self.project_many(&cmd, &k.fb, &f)?;
                        let beta = self.project_many(&cmd, &k.beta, &attention_inputs)?;
                        let gate = self.project_many(&cmd, &k.ga, &attention_inputs)?;
                        let gate = self.project_many(&cmd, &k.gb, &gate)?;
                        let mut states = Vec::with_capacity(tokens.len());
                        for i in 0..tokens.len() {
                            states.push(self.kda_projected(
                                &cmd,
                                k,
                                &s.layers[layer],
                                &[
                                    q[i].clone(),
                                    keys[i].clone(),
                                    v[i].clone(),
                                    f[i].clone(),
                                    beta[i].clone(),
                                    gate[i].clone(),
                                ],
                            )?);
                            if let Some(saved) = snapshots.get_mut(i) {
                                self.save_layer(&cmd, &s.layers[layer], saved)?;
                            }
                        }
                        self.project_many(&cmd, &k.out, &states)?
                    }
                    Mixer::Dsa(d) => attention_inputs
                        .iter()
                        .enumerate()
                        .map(|(i, x)| self.dsa(&cmd, d, &s.layers[layer], x, position + i))
                        .collect::<Result<Vec<_>>>()?,
                };
                let mut residuals = Vec::with_capacity(tokens.len());
                let mut inputs = Vec::with_capacity(tokens.len());
                let mut post_gates = Vec::with_capacity(tokens.len());
                for ((h, attn), gates) in hidden.iter().zip(&attention).zip(&attention_gates) {
                    let residual = self.hc_post(&cmd, h, attn, gates)?;
                    let (x, gates) = self.hc_pre(&cmd, &l.ffn, &residual)?;
                    residuals.push(residual);
                    inputs.push(x);
                    post_gates.push(gates);
                }
                let (out, cmd) = match &l.feed_forward {
                    Ffn::Dense(d) => (self.dense_many(&cmd, d, &inputs)?, cmd),
                    Ffn::Moe(m) => self.moe_many(cmd, m, &inputs, cancel)?,
                };
                let h = residuals
                    .iter()
                    .zip(&out)
                    .zip(&post_gates)
                    .map(|((h, out), gates)| self.hc_post(&cmd, h, out, gates))
                    .collect::<Result<Vec<_>>>()?;
                self.gpu.complete(cmd)?;
                Ok(h)
            })?;
        }
        Ok(hidden)
    }
    fn forward_token(
        &mut self,
        s: &mut State,
        position: usize,
        token: u32,
        cancel: &Cancellation,
    ) -> Result<Buffer> {
        if token as usize >= self.embedding.t.dims[1] {
            return Err(invalid("token outside vocabulary"));
        }
        self.trace_active
            .set(self.trace.as_ref().is_some_and(|(p, _)| *p == position));
        self.trace_events.borrow_mut().clear();
        let c = &self.c;
        let cmd = self.gpu.command()?;
        let emb = self.gpu.output(c.width)?;
        self.point(
            &cmd,
            "embedding",
            &[&self.embedding.buffer, &emb],
            &[
                c.width as u32,
                0,
                self.embedding.t.stride as u32,
                self.embedding.t.kind,
                token,
            ],
            c.width,
        )?;
        let mut h = self.gpu.output(c.width * c.hc)?;
        self.point(
            &cmd,
            "repeat_emb",
            &[&emb, &h],
            &[c.width as u32, c.hc as u32],
            c.width * c.hc,
        )?;
        self.gpu.complete(cmd)?;
        for (i, l) in self.layers.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            h = objc2::rc::autoreleasepool(|_| -> Result<Buffer> {
                let cmd = self.gpu.command()?;
                let (x, gates) = self.hc_pre(&cmd, &l.attention, &h)?;
                self.record("attn_norm", &x);
                let attn = match &l.mixer {
                    Mixer::Kda(l) => self.kda(&cmd, l, &s.layers[i], &x)?,
                    Mixer::Dsa(l) => self.dsa(&cmd, l, &s.layers[i], &x, position)?,
                };
                self.record("attn_out", &attn);
                let residual = self.hc_post(&cmd, &h, &attn, &gates)?;
                let (x, gates) = self.hc_pre(&cmd, &l.ffn, &residual)?;
                self.record("ffn_norm", &x);
                let (out, cmd) = match &l.feed_forward {
                    Ffn::Dense(l) => (self.dense(&cmd, l, &x)?, cmd),
                    Ffn::Moe(l) => self.moe(cmd, l, &x, cancel)?,
                };
                self.record("ffn_out", &out);
                let h = self.hc_post(&cmd, &residual, &out, &gates)?;
                self.record("l_out", &h);
                self.gpu.complete(cmd)?;
                if self.trace_active.get() {
                    self.flush_trace(i)?;
                }
                Ok(h)
            })?;
        }
        Ok(h)
    }
}
