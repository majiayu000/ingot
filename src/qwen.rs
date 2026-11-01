//! The qwen4exp trunk: HC, Gated DeltaNet, QSA, PLE and routed/shared FFN.
#[path = "qwen_hc.rs"]
mod hc;
#[path = "qwen_media.rs"]
pub(crate) mod media;
#[path = "qwen_prefill.rs"]
mod packed;
use crate::{
    Result,
    gguf::{Gguf, Tensor},
    invalid,
    metal::{Buffer, Command, Gpu},
    weights::{ExpertCache, ExpertWeights, Weight},
};
#[path = "qwen_mtp.rs"]
mod mtp;
use mtp::{Draft, DraftState};

pub(crate) const PREFILL_BLOCK: usize = 4096;

/// IQ2_S gate/up experts of one shape share a single pass that also applies SwiGLU.
fn fused_gate_up(gate: &Tensor, up: &Tensor) -> bool {
    gate.kind == 22
        && up.kind == 22
        && gate.dims == up.dims
        && gate.stride == up.stride
        && gate.dims[0].is_multiple_of(256)
}

struct Forward {
    logits: Buffer,
    rows: usize,
    vocabulary: usize,
    hidden: Vec<Buffer>,
    snapshots: Vec<mtp::Checkpoint>,
    snapshot_encode: Duration,
}
impl Forward {
    fn read_logits(&self, gpu: &Gpu) -> Result<Vec<Vec<f32>>> {
        Ok(gpu
            .read_f32(&self.logits)?
            .chunks_exact(self.vocabulary)
            .map(<[f32]>::to_vec)
            .collect())
    }
    fn choices(&self, gpu: &Gpu) -> Result<Vec<u32>> {
        gpu.finish_greedy(
            gpu.command()?,
            &self.logits,
            self.rows,
            self.vocabulary,
            None,
        )
    }
}
use std::cell::{Cell, RefCell};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
struct Hc {
    norm: Weight,
    down: Weight,
    up: Weight,
    inject: Option<Weight>,
}
struct Ffn {
    router: Weight,
    experts: ExpertWeights,
    sgate: Weight,
    sup: Weight,
    sdown: Weight,
    shared: Weight,
}
struct Linear {
    qkv: Weight,
    z: Weight,
    conv: Weight,
    dt: Weight,
    a: Weight,
    beta: Weight,
    alpha: Weight,
    norm: Weight,
    out: Weight,
}
struct Attention {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    qn: Weight,
    kn: Weight,
    iq: Weight,
    ik: Weight,
    iqn: Weight,
    ikn: Weight,
    ratio: usize,
}
enum Mixer {
    Linear(Box<Linear>),
    Attention(Box<Attention>),
}
struct Ple {
    key: Weight,
    value: Weight,
    kn: Weight,
    qn: Weight,
    cn: Weight,
    conv: Weight,
}
struct Layer {
    attn: Hc,
    ffn: Hc,
    mixer: Mixer,
    experts: Ffn,
    ple: Option<Ple>,
}
struct Config {
    width: usize,
    hc: usize,
    layers: usize,
    lr: usize,
    experts: usize,
    top: usize,
    ff: usize,
    sff: usize,
    d: usize,
    nk: usize,
    nv: usize,
    conv: usize,
    heads: usize,
    kv: usize,
    head: usize,
    idx: usize,
    ih: usize,
    topk: usize,
    rot: usize,
    rope: f32,
    eps: f32,
    ple_layers: Vec<u64>,
    ngram: usize,
    pergram: usize,
    ple_taps: usize,
    ple_dim: usize,
    ple_eos: u32,
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    ratios: Vec<u64>,
}
impl Config {
    fn load(g: &Gguf) -> Result<Self> {
        let n = |s: &str| g.integer(&format!("qwen4exp.{s}"));
        let f = |s: &str| g.float(&format!("qwen4exp.{s}"));
        let a = |s: &str| g.integers(&format!("qwen4exp.{s}"));
        let c = Self {
            width: n("embedding_length")?,
            hc: n("hyper_connection.count")?,
            layers: n("block_count")?,
            lr: n("hyper_connection.low_rank")?,
            experts: n("expert_count")?,
            top: n("expert_used_count")?,
            ff: n("expert_feed_forward_length")?,
            sff: n("expert_shared_feed_forward_length")?,
            d: n("ssm.state_size")?,
            nk: n("ssm.group_count")?,
            nv: n("ssm.time_step_rank")?,
            conv: n("ssm.conv_kernel")?,
            heads: n("attention.head_count")?,
            kv: n("attention.head_count_kv")?,
            head: n("attention.key_length")?,
            idx: n("attention.indexer.key_length")?,
            ih: n("attention.indexer.head_count")?,
            topk: n("attention.indexer.top_k")?,
            rot: n("rope.dimension_count")?,
            rope: f("rope.freq_base")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            ple_layers: a("ple.layers")?,
            ngram: n("ple.ngram_size")?,
            pergram: n("ple.heads_per_ngram")?,
            ple_taps: n("ple.conv_kernel")?,
            ple_dim: n("embedding_length_per_layer_input")?,
            ple_eos: n("ple.eos_token_id")? as u32,
            multipliers: a("ple.layer_multipliers")?,
            offsets: a("ple.head_offsets")?,
            sizes: a("ple.head_vocab_sizes")?,
            ratios: a("attention.compress_ratios")?,
        };
        if c.layers != 48
            || c.width != 2560
            || c.hc != 4
            || c.lr != 320
            || c.d != 128
            || c.nk != 16
            || c.nv != 48
            || c.head != 256
            || c.heads != 24
            || c.kv != 2
            || c.idx != 128
            || c.ih != 4
            || c.top == 0
            || c.top > c.experts
            || c.conv == 0
            || c.conv > u32::MAX as usize
            || c.ple_taps == 0
            || c.ple_taps > u32::MAX as usize
            || c.ngram < 2
            || c.ngram > c.multipliers.len()
            || !c.rot.is_multiple_of(2)
            || c.rot > c.idx
            || c.ratios.len() != c.layers
            || (c.ngram - 1).checked_mul(c.pergram) != Some(c.sizes.len())
            || c.offsets.len() != c.sizes.len()
            || c.sizes.contains(&0)
            || c.ple_layers.iter().any(|&l| l >= c.layers as u64)
            || c.eps <= 0.0
            || c.rope <= 0.0
        {
            return Err(invalid("unsupported or malformed qwen4exp configuration"));
        }
        Ok(c)
    }
    fn state_bytes(&self, context: usize) -> usize {
        (0..self.layers)
            .map(|i| {
                if self.ratios[i] == 0 {
                    (self.nv * self.d * self.d + (self.conv - 1) * (2 * self.nk + self.nv) * self.d)
                        * 4
                } else {
                    (2 * self.kv * self.head * context
                        + self.idx * context
                        + self.idx * context.div_ceil(self.ratios[i] as usize))
                        * 4
                }
            })
            .sum::<usize>()
            + self.ple_layers.len() * (self.ple_taps - 1) * self.ngram * self.width * self.hc * 4
    }
}
enum MixerState {
    Linear {
        history: Buffer,
        state: Buffer,
    },
    Attention {
        key: Buffer,
        value: Buffer,
        index: Buffer,
        pooled: Buffer,
    },
}
struct LayerState {
    mixer: MixerState,
    ple: Option<Buffer>,
}
pub struct State {
    layers: Vec<LayerState>,
    draft: Option<DraftState>,
}
pub struct Qwen {
    g: RefCell<Gguf>,
    gpu: Gpu,
    c: Config,
    layers: Vec<Layer>,
    embedding: Weight,
    head: Hc,
    output: Weight,
    bytes: usize,
    prefill_block: usize,
    draft: Option<Draft>,
    trace: Option<(usize, PathBuf)>,
    trace_active: Cell<bool>,
    trace_events: RefCell<Vec<(String, Buffer)>>,
    media: RefCell<media::Media>,
}
impl Qwen {
    pub fn load(
        mut g: Gguf,
        budget: usize,
        context: usize,
        draft_config: Option<&crate::engine::DraftConfig>,
        prefix_cache: bool,
    ) -> Result<Self> {
        let c = Config::load(&g)?;
        if context == 0 || context > g.integer("qwen4exp.context_length")? {
            return Err(invalid("invalid context capacity"));
        }
        let draft_file = draft_config
            .map(|config| Draft::open(&g, &c, config))
            .transpose()?;
        let draft_bytes = draft_file
            .as_ref()
            .map(|(g, v)| Draft::reserved_bytes(g, v, &c, context))
            .transpose()?
            .unwrap_or(0);
        let prefix_bytes = |block: usize| -> Result<usize> {
            if !prefix_cache {
                return Ok(0);
            }
            let n = context.min(block);
            let draft = draft_file
                .as_ref()
                .map(|(g, _)| Draft::state_bytes(g, &c, n))
                .transpose()?
                .unwrap_or(0);
            Ok(c.state_bytes(n) + draft + n * 4 + 4)
        };
        let mut dense = 0usize;
        let mut expert = 0usize;
        let mut mapping_padding = 0usize;
        for (name, t) in &g.tensors {
            if name.starts_with("blk.48.") || name == "per_layer_token_embd.weight" {
                continue;
            }
            // Reserve all possible mapping padding even when experts are streamed;
            // per-slot geometry below must continue to use unpadded tensor bytes.
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
        // Use the large block only when it also leaves every expert resident.
        // A small budget keeps a bounded block and gives the rest to expert slots.
        // Compact QSA stores scores/probabilities only for selected positions.
        // Masks retain context stride; compact IDs and offsets have their own bounds.
        let selected =
            context.min(c.topk + c.ratios.iter().copied().max().unwrap_or(1) as usize - 1);
        let attention_per_token = c.heads * selected * 8 + context * 4 + selected * 4 + 33 * 4;
        let workspace_for = |block: usize| block * (2 * 1024 * 1024 + attention_per_token);
        let resident_weights = dense
            .checked_add(expert)
            .and_then(|n| n.checked_add(mapping_padding))
            .and_then(|n| n.checked_add(c.state_bytes(context)))
            .and_then(|n| n.checked_add(draft_bytes + 256 * 1024 * 1024))
            .ok_or_else(|| invalid("memory requirement overflow"))?;
        let mut prefill_block = 128;
        for block in [PREFILL_BLOCK, 1024] {
            if block > context {
                continue;
            }
            let candidate_prefix = prefix_bytes(block)?;
            let required = resident_weights
                .checked_add(workspace_for(block))
                .and_then(|n| n.checked_add(candidate_prefix))
                .ok_or_else(|| invalid("memory requirement overflow"))?;
            if required <= budget {
                prefill_block = block;
                break;
            }
        }
        let workspace = workspace_for(prefill_block);
        let prefix = prefix_bytes(prefill_block)?;
        let fixed = dense
            .checked_add(c.state_bytes(context))
            .and_then(|n| n.checked_add(mapping_padding))
            .and_then(|n| n.checked_add(workspace + prefix + 256 * 1024 * 1024))
            .and_then(|n| n.checked_add(draft_bytes))
            .ok_or_else(|| invalid("memory requirement overflow"))?;
        let per_slot = expert / c.experts;
        if per_slot == 0 {
            return Err(invalid("invalid expert weight geometry"));
        }
        let slots = budget.saturating_sub(fixed) / per_slot;
        if slots == 0 {
            return Err(invalid(format!(
                "memory budget {budget} cannot hold required {} bytes",
                fixed + per_slot
            )));
        }
        let slots = slots.min(c.experts);
        let bytes = fixed + slots * per_slot;
        let table = g.tensor("per_layer_token_embd.weight")?;
        if table.dims.len() != 2
            || table.dims[0] != c.ple_dim
            || c.ple_dim.checked_mul(c.sizes.len()) != Some(c.width)
            || c.offsets.iter().zip(&c.sizes).any(|(&o, &s)| {
                o.checked_add(s)
                    .is_none_or(|end| end > table.dims[1] as u64)
            })
        {
            return Err(invalid("invalid PLE table/head ranges"));
        }
        let gpu = Gpu::new()?;
        gpu.configure_workspace(
            workspace
                + if draft_config.is_some() {
                    Draft::working_bytes(&c)
                } else {
                    0
                },
        );
        let draft = draft_file
            .map(|g| Draft::load(g, &gpu, &c, draft_config.expect("draft configuration")))
            .transpose()?;
        let vocab = g.array("tokenizer.ggml.tokens")?.len();
        let embedding = Weight::load(&mut g, &gpu, "token_embd.weight", &[c.width, vocab])?;
        let vocab = embedding.t.dims[1];
        let output = Weight::load(&mut g, &gpu, "output.weight", &[c.width, vocab])?;
        let head = Self::hc_load(&mut g, &gpu, "output_hc", &c, false)?;
        let mut layers = Vec::new();
        for i in 0..c.layers {
            let p = format!("blk.{i}.");
            let attn = Self::hc_load(&mut g, &gpu, &format!("{p}hc_attn"), &c, true)?;
            let ffn = Self::hc_load(&mut g, &gpu, &format!("{p}hc_ffn"), &c, true)?;
            let mut w =
                |s: &str, shape: &[usize]| Weight::load(&mut g, &gpu, &format!("{p}{s}"), shape);
            let mixer = if c.ratios[i] == 0 {
                let v = c.nv * c.d;
                let channels = (2 * c.nk + c.nv) * c.d;
                Mixer::Linear(Box::new(Linear {
                    qkv: w("attn_qkv.weight", &[c.width, channels])?,
                    z: w("attn_gate.weight", &[c.width, v])?,
                    conv: w("ssm_conv1d.weight", &[c.conv, channels])?,
                    dt: w("ssm_dt.bias", &[c.nv])?,
                    a: w("ssm_a", &[c.nv])?,
                    beta: w("ssm_beta.weight", &[c.width, c.nv])?,
                    alpha: w("ssm_alpha.weight", &[c.width, c.nv])?,
                    norm: w("ssm_norm.weight", &[c.d])?,
                    out: w("ssm_out.weight", &[v, c.width])?,
                }))
            } else {
                Mixer::Attention(Box::new(Attention {
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
                    ratio: c.ratios[i] as usize,
                }))
            };
            let float_parameters: &[&Weight] = match &mixer {
                Mixer::Linear(l) => &[&l.dt, &l.a, &l.norm],
                Mixer::Attention(a) => &[&a.qn, &a.kn, &a.iqn, &a.ikn],
            };
            for parameter in float_parameters {
                parameter.require_f32()?;
            }
            let tensors = [
                ("gate", vec![c.width, c.ff, c.experts]),
                ("up", vec![c.width, c.ff, c.experts]),
                ("down", vec![c.ff, c.width, c.experts]),
            ]
            .into_iter()
            .map(|(role, shape)| {
                let name = format!("{p}ffn_{role}_exps.weight");
                let t = g.tensor(&name)?.clone();
                if t.dims != shape {
                    return Err(invalid(format!("invalid expert shape: {name}")));
                }
                Ok(t)
            })
            .collect::<Result<Vec<_>>>()?;
            let tensors: [Tensor; 3] =
                tensors.try_into().map_err(|_| invalid("expert bindings"))?;
            let experts = if slots == c.experts {
                ExpertWeights::Resident(
                    tensors
                        .map(|t| {
                            let buffer = gpu.load_weight(&mut g, &t)?;
                            Ok(Weight { buffer, t })
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| invalid("expert weights"))?,
                )
            } else {
                let buffers = tensors
                    .each_ref()
                    .map(|t| gpu.empty_bytes(t.bytes / c.experts * slots))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| invalid("expert buffers"))?;
                ExpertWeights::Cached(RefCell::new(ExpertCache {
                    tensors,
                    buffers,
                    slots: vec![None; slots],
                    clock: 0,
                    hits: 0,
                    misses: 0,
                    read: Duration::ZERO,
                }))
            };
            let mut w =
                |s: &str, shape: &[usize]| Weight::load(&mut g, &gpu, &format!("{p}{s}"), shape);
            let experts = Ffn {
                router: w("ffn_gate_inp.weight", &[c.width, c.experts])?,
                experts,
                sgate: w("ffn_gate_shexp.weight", &[c.width, c.sff])?,
                sup: w("ffn_up_shexp.weight", &[c.width, c.sff])?,
                sdown: w("ffn_down_shexp.weight", &[c.sff, c.width])?,
                shared: w("ffn_gate_inp_shexp.weight", &[c.width])?,
            };
            let ple = if c.ple_layers.contains(&(i as u64)) {
                let h = c.width * c.hc;
                Some(Ple {
                    key: w("ple_key.weight", &[c.width, h])?,
                    value: w("ple_value.weight", &[c.width, c.width])?,
                    kn: w("ple_norm_key.weight", &[h])?,
                    qn: w("ple_norm_query.weight", &[h])?,
                    cn: w("ple_norm_conv.weight", &[h])?,
                    conv: w("ple_conv1d.weight", &[c.ple_taps, h])?,
                })
            } else {
                None
            };
            if let Some(p) = &ple {
                for norm in [&p.kn, &p.qn, &p.cn] {
                    norm.require_f32()?;
                }
            }
            layers.push(Layer {
                attn,
                ffn,
                mixer,
                experts,
                ple,
            });
        }
        let media = RefCell::new(media::Media::new(&g)?);
        Ok(Self {
            g: RefCell::new(g),
            gpu,
            c,
            layers,
            embedding,
            head,
            output,
            bytes,
            prefill_block,
            draft,
            trace: None,
            trace_active: Cell::new(false),
            trace_events: RefCell::new(Vec::new()),
            media,
        })
    }
    fn hc_load(g: &mut Gguf, gpu: &Gpu, p: &str, c: &Config, inject: bool) -> Result<Hc> {
        let h = c.width * c.hc;
        let mut w = |s: &str, d: &[usize]| Weight::load(g, gpu, &format!("{p}_{s}.weight"), d);
        let hc = Hc {
            norm: w("norm", &[h])?,
            down: w("down", &[h, c.lr])?,
            up: w("up", &[c.lr, h])?,
            inject: if inject {
                Some(w("inject", &[h, c.hc])?)
            } else {
                None
            },
        };
        hc.norm.require_f32()?;
        Ok(hc)
    }
    pub fn state(&self, context: usize) -> Result<State> {
        let c = &self.c;
        let mut layers = Vec::new();
        for l in &self.layers {
            let mixer = match &l.mixer {
                Mixer::Linear(_) => MixerState::Linear {
                    history: self.gpu.zeros((c.conv - 1) * (2 * c.nk + c.nv) * c.d)?,
                    state: self.gpu.zeros(c.nv * c.d * c.d)?,
                },
                Mixer::Attention(a) => MixerState::Attention {
                    key: self.gpu.zeros(context * c.kv * c.head)?,
                    value: self.gpu.zeros(context * c.kv * c.head)?,
                    index: self.gpu.zeros(context * c.idx)?,
                    pooled: self.gpu.zeros(context.div_ceil(a.ratio) * c.idx)?,
                },
            };
            layers.push(LayerState {
                mixer,
                ple: if l.ple.is_some() {
                    Some(
                        self.gpu
                            .zeros((c.ple_taps - 1) * c.ngram * c.width * c.hc)?,
                    )
                } else {
                    None
                },
            });
        }
        let draft = self
            .draft
            .as_ref()
            .map(|d| d.state(&self.gpu, &self.c, context))
            .transpose()?;
        Ok(State { layers, draft })
    }
    pub fn set_trace(&mut self, position: usize, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        self.trace = Some((position, path.to_owned()));
        Ok(())
    }
    fn record(&self, name: &str, buffer: &Buffer) {
        if self.trace_active.get() {
            self.trace_events
                .borrow_mut()
                .push((name.to_owned(), buffer.clone()));
        }
    }
    fn flush_trace(&self, layer: usize) -> Result<()> {
        if let Some((_, dir)) = &self.trace {
            for (name, buffer) in self.trace_events.borrow_mut().drain(..) {
                let bits = self.gpu.read_u32(&buffer, buffer.length() / 4)?;
                let mut out = BufWriter::new(std::fs::File::create(
                    dir.join(format!("{name}-{layer}.bin")),
                )?);
                for v in bits {
                    out.write_all(&v.to_le_bytes())?;
                }
                out.flush()?;
            }
        }
        Ok(())
    }
    pub(crate) fn workspace_statistics(&self) -> (u64, u64, usize, usize) {
        self.gpu.workspace_statistics()
    }
    pub fn reserved_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn ple_read_time(&self) -> Duration {
        self.g.borrow().row_read_time
    }
    pub fn file_bytes(&self) -> u64 {
        self.g.borrow().bytes_read
    }
    fn point(&self, cmd: &Command, name: &str, b: &[&Buffer], p: &[u32], n: usize) -> Result<()> {
        self.gpu.op(cmd, name, b, p, n.div_ceil(32), 1)
    }
    #[allow(clippy::too_many_arguments)] // A strided view and the normalization contract.
    fn norm(
        &self,
        cmd: &Command,
        x: &Buffer,
        w: &Weight,
        n: usize,
        rows: usize,
        offset: usize,
        stride: usize,
        l2: bool,
    ) -> Result<Buffer> {
        let y = self.gpu.output(n * rows)?;
        self.gpu.op(
            cmd,
            "rms",
            &[x, &w.buffer, &y],
            &[
                n as u32,
                (w.t.bytes / 4) as u32,
                offset as u32,
                stride as u32,
                self.c.eps.to_bits(),
                l2 as u32,
            ],
            rows,
            1,
        )?;
        Ok(y)
    }
    fn unary(&self, cmd: &Command, x: &Buffer, n: usize, kind: u32, scale: f32) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(
            cmd,
            "unary",
            &[x, &y],
            &[n as u32, kind, scale.to_bits()],
            n,
        )?;
        Ok(y)
    }

    // Small BF16 HC batches share count-one reductions, with fused epilogues.
    fn hc_fused(
        &self,
        cmd: &Command,
        hc: &Hc,
        x: &Buffer,
        count: usize,
    ) -> Result<Option<(Buffer, Option<Buffer>)>> {
        if !self.hc_fusable(hc, count) {
            return Ok(None);
        }
        let c = &self.c;
        let norm = self.norm(cmd, x, &hc.norm, c.width, c.hc * count, 0, c.width, false)?;
        self.hc_project(cmd, hc, &norm, count).map(Some)
    }
    fn hc_mix(&self, cmd: &Command, h: &Hc, x: &Buffer) -> Result<(Buffer, Option<Buffer>)> {
        if let Some(result) = self.hc_fused(cmd, h, x, 1)? {
            return Ok(result);
        }
        let c = &self.c;
        let n = c.width * c.hc;
        let xn = self.norm(cmd, x, &h.norm, c.width, c.hc, 0, c.width, false)?;
        let lo = h.down.project(&self.gpu, cmd, &xn)?;
        let lo = self.unary(cmd, &lo, c.lr, 1, 1.0 / c.hc as f32)?;
        let g = h.up.project(&self.gpu, cmd, &lo)?;
        let g = self.unary(cmd, &g, n, 0, 1.0)?;
        let y = self.gpu.output(c.width)?;
        self.point(
            cmd,
            "hc_collapse",
            &[&xn, &g, &y],
            &[c.width as u32, c.hc as u32],
            c.width,
        )?;
        let inject = h
            .inject
            .as_ref()
            .map(|w| w.project(&self.gpu, cmd, &xn))
            .transpose()?;
        Ok((y, inject))
    }
    fn hc_add(&self, cmd: &Command, x: &Buffer, b: &Buffer, g: &Buffer) -> Result<Buffer> {
        let c = &self.c;
        let y = self.gpu.output(c.width * c.hc)?;
        self.point(
            cmd,
            "hc_add",
            &[x, b, g, &y],
            &[c.width as u32, c.hc as u32],
            c.width * c.hc,
        )?;
        Ok(y)
    }
    fn copy(&self, cmd: &Command, x: &Buffer, n: usize, offset: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(cmd, "copy_range", &[x, &y], &[n as u32, offset as u32], n)?;
        Ok(y)
    }
    fn pack(&self, cmd: &Command, inputs: &[Buffer], width: usize) -> Result<Buffer> {
        if let Some(packed) = Buffer::contiguous_f32(inputs, width) {
            return Ok(packed);
        }
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
    fn hc_many(
        &self,
        cmd: &Command,
        hc: &Hc,
        inputs: &[Buffer],
    ) -> Result<(Vec<Buffer>, Vec<Buffer>)> {
        let c = &self.c;
        let norm = inputs
            .iter()
            .map(|x| self.norm(cmd, x, &hc.norm, c.width, c.hc, 0, c.width, false))
            .collect::<Result<Vec<_>>>()?;
        let lo = self.project_many(cmd, &hc.down, &norm)?;
        let lo = lo
            .iter()
            .map(|x| self.unary(cmd, x, c.lr, 1, 1.0 / c.hc as f32))
            .collect::<Result<Vec<_>>>()?;
        let gates = self.project_many(cmd, &hc.up, &lo)?;
        let mut mixed = Vec::with_capacity(inputs.len());
        for (x, g) in norm.iter().zip(gates) {
            let g = self.unary(cmd, &g, c.width * c.hc, 0, 1.0)?;
            let y = self.gpu.output(c.width)?;
            self.point(
                cmd,
                "hc_collapse",
                &[x, &g, &y],
                &[c.width as u32, c.hc as u32],
                c.width,
            )?;
            mixed.push(y);
        }
        let inject = self.project_many(
            cmd,
            hc.inject
                .as_ref()
                .ok_or_else(|| invalid("missing HC injection"))?,
            &norm,
        )?;
        Ok((mixed, inject))
    }
    fn gated(&self, cmd: &Command, x: &Buffer, g: &Buffer, n: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        self.point(cmd, "mul_gate", &[x, g, &y], &[n as u32], n)?;
        Ok(y)
    }
    fn linear(&self, cmd: &Command, l: &Linear, s: &MixerState, x: &Buffer) -> Result<Buffer> {
        let projected = self.gpu.independent_projections(cmd, || {
            Ok([
                l.qkv.project(&self.gpu, cmd, x)?,
                l.z.project(&self.gpu, cmd, x)?,
                l.alpha.project(&self.gpu, cmd, x)?,
                l.beta.project(&self.gpu, cmd, x)?,
            ])
        })?;
        let out = self.linear_projected(cmd, l, s, &projected)?;
        l.out.project(&self.gpu, cmd, &out)
    }
    fn linear_projected(
        &self,
        cmd: &Command,
        l: &Linear,
        s: &MixerState,
        projected: &[Buffer; 4],
    ) -> Result<Buffer> {
        let c = &self.c;
        let MixerState::Linear { history, state } = s else {
            return Err(invalid("linear state mismatch"));
        };
        let channels = (2 * c.nk + c.nv) * c.d;
        let [qkv, z, a, b] = projected;
        let conv = self.gpu.output(channels)?;
        self.point(
            cmd,
            "convolution",
            &[qkv, &l.conv.buffer, history, &conv],
            &[
                channels as u32,
                c.conv as u32,
                1,
                l.conv.t.kind,
                l.conv.t.stride as u32,
            ],
            channels,
        )?;
        let q = self.norm(cmd, &conv, &l.norm, c.d, c.nk, 0, c.d, true)?;
        let k = self.norm(cmd, &conv, &l.norm, c.d, c.nk, c.nk * c.d, c.d, true)?;
        let v = self.copy(cmd, &conv, c.nv * c.d, 2 * c.nk * c.d)?;
        let delta = self.gpu.output(c.nv * c.d)?;
        self.gpu.op(
            cmd,
            "delta_net",
            &[&q, &k, &v, a, b, &l.dt.buffer, &l.a.buffer, state, &delta],
            &[c.d as u32, c.nk as u32],
            c.d,
            c.nv,
        )?;
        let out = self.norm(cmd, &delta, &l.norm, c.d, c.nv, 0, c.d, false)?;
        self.gated(cmd, &out, z, c.nv * c.d)
    }
    fn rope(
        &self,
        cmd: &Command,
        x: &Buffer,
        dim: usize,
        heads: usize,
        pos: usize,
    ) -> Result<Buffer> {
        let pos = match self.rope_plan(pos, 1, 1)? {
            media::Rope::Linear(pos, _) => pos,
            media::Rope::Rows(rows) => return self.rope_rows(cmd, x, dim, heads, &rows, 1),
        };
        let y = self.gpu.output(dim * heads)?;
        self.point(
            cmd,
            "rope",
            &[x, &y],
            &[
                dim as u32,
                heads as u32,
                self.c.rot as u32,
                pos as u32,
                self.c.rope.to_bits(),
            ],
            dim * heads,
        )?;
        Ok(y)
    }
    fn attention(
        &self,
        cmd: &Command,
        a: &Attention,
        s: &MixerState,
        x: &Buffer,
        pos: usize,
    ) -> Result<Buffer> {
        let projected = [
            a.q.project(&self.gpu, cmd, x)?,
            a.k.project(&self.gpu, cmd, x)?,
            a.v.project(&self.gpu, cmd, x)?,
            a.ik.project(&self.gpu, cmd, x)?,
            a.iq.project(&self.gpu, cmd, x)?,
        ];
        let out = self.attention_projected(cmd, a, s, &projected, pos)?;
        a.out.project(&self.gpu, cmd, &out)
    }
    fn attention_projected(
        &self,
        cmd: &Command,
        a: &Attention,
        s: &MixerState,
        projected: &[Buffer; 5],
        pos: usize,
    ) -> Result<Buffer> {
        let c = &self.c;
        let [qr, kr, v, ik, iq] = projected;
        let MixerState::Attention {
            key,
            value,
            index,
            pooled,
        } = s
        else {
            return Err(invalid("attention state mismatch"));
        };
        let length = pos + 1;
        let (q, k, gate) = self.gpu.independent_projections(cmd, || {
            let q = self.norm(cmd, qr, &a.qn, c.head, c.heads, 0, 2 * c.head, false)?;
            let k = self.norm(cmd, kr, &a.kn, c.head, c.kv, 0, c.head, false)?;
            let gate = self.gpu.output(c.head * c.heads)?;
            self.point(
                cmd,
                "extract_gate",
                &[qr, &gate],
                &[c.head as u32, c.heads as u32],
                c.head * c.heads,
            )?;
            Ok((q, k, gate))
        })?;
        let (q, k) = self.gpu.independent_projections(cmd, || {
            Ok((
                self.rope(cmd, &q, c.head, c.heads, pos)?,
                self.rope(cmd, &k, c.head, c.kv, pos)?,
            ))
        })?;
        self.gpu.independent_projections(cmd, || {
            self.point(
                cmd,
                "cache_write",
                &[&k, key],
                &[(c.head * c.kv) as u32, pos as u32],
                c.head * c.kv,
            )?;
            self.point(
                cmd,
                "cache_write",
                &[v, value],
                &[(c.head * c.kv) as u32, pos as u32],
                c.head * c.kv,
            )?;
            self.point(
                cmd,
                "cache_write",
                &[ik, index],
                &[c.idx as u32, pos as u32],
                c.idx,
            )?;
            Ok(())
        })?;
        if length.is_multiple_of(a.ratio) {
            let block = length / a.ratio - 1;
            let raw = self.gpu.output(c.idx)?;
            self.point(
                cmd,
                "pool_index",
                &[index, &raw],
                &[c.idx as u32, a.ratio as u32, block as u32],
                c.idx,
            )?;
            let norm = self.norm(cmd, &raw, &a.ikn, c.idx, 1, 0, c.idx, false)?;
            let rotated = self.rope(cmd, &norm, c.idx, 1, block * a.ratio)?;
            self.point(
                cmd,
                "cache_write",
                &[&rotated, pooled],
                &[c.idx as u32, block as u32],
                c.idx,
            )?;
        }
        if length < c.topk + a.ratio {
            let out = self.dense_attention(cmd, &q, key, value, pos, 1)?;
            return self.gated(cmd, &out, &gate, c.heads * c.head);
        }
        let iq = self.norm(cmd, iq, &a.iqn, c.idx, c.ih, 0, c.idx, false)?;
        let iq = self.rope(cmd, &iq, c.idx, c.ih, pos)?;
        let blocks = length / a.ratio;
        let score = self.gpu.output(blocks)?;
        self.gpu.op(
            cmd,
            "index_score",
            &[&iq, pooled, &score],
            &[c.idx as u32, c.ih as u32],
            blocks,
            1,
        )?;
        let mask = self.gpu.output(length)?;
        self.point(
            cmd,
            "index_mask",
            &[&score, &mask],
            &[length as u32, a.ratio as u32, c.topk as u32],
            blocks,
        )?;
        let out = self.compact_attention(cmd, &q, key, value, &mask, pos, 1, a.ratio)?;
        self.gated(cmd, &out, &gate, c.heads * c.head)
    }
    pub fn cache_statistics(&self) -> (u64, u64, Duration) {
        self.layers
            .iter()
            .fold((0, 0, Duration::ZERO), |(h, m, t), l| {
                match &l.experts.experts {
                    ExpertWeights::Cached(c) => {
                        let c = c.borrow();
                        (h + c.hits, m + c.misses, t + c.read)
                    }
                    _ => (h, m, t),
                }
            })
    }
    fn shared(&self, cmd: &Command, l: &Ffn, x: &Buffer) -> Result<(Buffer, Buffer)> {
        let g = l.sgate.project(&self.gpu, cmd, x)?;
        let u = l.sup.project(&self.gpu, cmd, x)?;
        let act = self.gpu.activation(cmd, &g, &u, self.c.sff)?;
        Ok((
            l.sdown.project(&self.gpu, cmd, &act)?,
            l.shared.project(&self.gpu, cmd, x)?,
        ))
    }
    fn cache_slot(&self, cache: &mut ExpertCache, id: usize, pinned: &[usize]) -> Result<usize> {
        cache.slot(
            &self.gpu,
            &mut self.g.borrow_mut(),
            self.c.experts,
            id,
            pinned,
        )
    }
    fn ffn(
        &self,
        cmd: Command,
        l: &Ffn,
        x: &Buffer,
        canonical_resident: bool,
        cancellation: &crate::engine::Cancellation,
    ) -> Result<(Buffer, Command)> {
        let c = &self.c;
        let scores = l.router.project(&self.gpu, &cmd, x)?;
        let ids = self.gpu.output(c.top)?;
        let weights = self.gpu.output(c.top)?;
        self.point(
            &cmd,
            "route_gpu",
            &[&scores, &ids, &weights],
            &[c.experts as u32, c.top as u32],
            1,
        )?;
        self.record("ffn_moe_topk", &ids);
        self.record("ffn_moe_logits", &scores);
        match &l.experts {
            ExpertWeights::Resident(m) => {
                let project = |w: &Weight, x: &Buffer, per: bool| -> Result<Buffer> {
                    let rows = w.t.dims[1];
                    let y = self.gpu.output(rows * c.top)?;
                    if canonical_resident
                        && ((!per && matches!(w.t.kind, 16 | 17 | 18 | 21 | 29))
                            || (per && matches!(w.t.kind, 20 | 42)))
                    {
                        self.gpu.op(
                            &cmd,
                            "project_routed_batch",
                            &[&w.buffer, x, &ids, &y],
                            &[
                                w.t.dims[0] as u32,
                                rows as u32,
                                w.t.stride as u32,
                                w.t.kind,
                                c.top as u32,
                                u32::from(per),
                            ],
                            rows,
                            c.top,
                        )?;
                    } else {
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
                    }
                    Ok(y)
                };
                let (routed_first, routed_up, g, u, gate) =
                    self.gpu.independent_projections(&cmd, || {
                        let (first, up) = if canonical_resident && fused_gate_up(&m[0].t, &m[1].t) {
                            let act = self.gpu.output(m[0].t.dims[1] * c.top)?;
                            self.gpu.op(
                                &cmd,
                                "routed_gate_up_iq2s",
                                &[&m[0].buffer, &m[1].buffer, x, &ids, &act],
                                &[
                                    m[0].t.dims[0] as u32,
                                    m[0].t.dims[1] as u32,
                                    m[0].t.stride as u32,
                                    m[0].t.kind,
                                    c.top as u32,
                                    0,
                                ],
                                m[0].t.dims[1],
                                c.top,
                            )?;
                            (act, None)
                        } else {
                            let g = project(&m[0], x, false)?;
                            let u = project(&m[1], x, false)?;
                            (g, Some(u))
                        };
                        Ok((
                            first,
                            up,
                            l.sgate.project(&self.gpu, &cmd, x)?,
                            l.sup.project(&self.gpu, &cmd, x)?,
                            l.shared.project(&self.gpu, &cmd, x)?,
                        ))
                    })?;
                let act = match routed_up {
                    Some(u) => self.gpu.activation(&cmd, &routed_first, &u, c.ff * c.top)?,
                    None => routed_first,
                };
                let routed = project(&m[2], &act, true)?;
                let act = self.gpu.activation(&cmd, &g, &u, c.sff)?;
                let shared = l.sdown.project(&self.gpu, &cmd, &act)?;
                Ok((
                    self.gpu
                        .merge(&cmd, &routed, &shared, &weights, &gate, c.width, c.top)?,
                    cmd,
                ))
            }
            ExpertWeights::Cached(cache) => {
                self.gpu.complete(cmd)?;
                let ids = self.gpu.read_u32(&ids, c.top)?;
                let mut cache = cache.borrow_mut();
                let mut slot = self.cache_slot(&mut cache, ids[0] as usize, &[])?;
                let routed = self.gpu.output(c.width * c.top)?;
                let first = self.gpu.command()?;
                let (shared, gate) = self.shared(&first, l, x)?;
                let mut current = Some(first);
                let mut i = 0;
                while i < ids.len() {
                    if cancellation.is_cancelled() {
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
                    let act = self.gpu.activation(&cmd, &g, &u, c.ff * count)?;
                    let out = project(2, &act, true)?;
                    self.point(
                        &cmd,
                        "copy_into",
                        &[&out, &routed],
                        &[(c.width * count) as u32, (i * c.width) as u32],
                        c.width * count,
                    )?;
                    i += count;
                    self.gpu.submit(&cmd);
                    // Pin every slot read by this batch while one bounded read overlaps it.
                    let next = if i < ids.len() && cache.slots.len() > slots.len() {
                        Some(self.cache_slot(&mut cache, ids[i] as usize, &slots))
                    } else {
                        None
                    };
                    // Drain the GPU before propagating a read failure or reusing a slot.
                    self.gpu.wait(cmd)?;
                    if let Some(next) = next {
                        slot = next?;
                    } else if i < ids.len() {
                        slot = self.cache_slot(&mut cache, ids[i] as usize, &[])?;
                    }
                }
                let cmd = self.gpu.command()?;
                Ok((
                    self.gpu
                        .merge(&cmd, &routed, &shared, &weights, &gate, c.width, c.top)?,
                    cmd,
                ))
            }
        }
    }
    fn ffn_many(
        &self,
        cmd: Command,
        f: &Ffn,
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
                "route_gpu",
                &[score, &id, &weight],
                &[c.experts as u32, c.top as u32],
                1,
            )?;
            ids.push(id);
            weights.push(weight);
        }
        if let ExpertWeights::Resident(w) = &f.experts
            && inputs.len() <= 4
        {
            let x = self.pack(&cmd, inputs, c.width)?;
            let routed_ids = self.pack(&cmd, &ids, c.top)?;
            let project = |w: &Weight, x: &Buffer, per: bool| -> Result<Buffer> {
                let y = self.gpu.output(w.t.dims[1] * c.top * inputs.len())?;
                self.gpu.op(
                    &cmd,
                    "project_routed_batch",
                    &[&w.buffer, x, &routed_ids, &y],
                    &[
                        w.t.dims[0] as u32,
                        w.t.dims[1] as u32,
                        w.t.stride as u32,
                        w.t.kind,
                        c.top as u32,
                        per as u32,
                    ],
                    w.t.dims[1],
                    c.top * inputs.len(),
                )?;
                Ok(y)
            };
            let a = if fused_gate_up(&w[0].t, &w[1].t) {
                let a = self.gpu.output(w[0].t.dims[1] * c.top * inputs.len())?;
                self.gpu.op(
                    &cmd,
                    "routed_gate_up_iq2s",
                    &[&w[0].buffer, &w[1].buffer, &x, &routed_ids, &a],
                    &[
                        w[0].t.dims[0] as u32,
                        w[0].t.dims[1] as u32,
                        w[0].t.stride as u32,
                        w[0].t.kind,
                        c.top as u32,
                        0,
                    ],
                    w[0].t.dims[1],
                    c.top * inputs.len(),
                )?;
                a
            } else {
                let g = project(&w[0], &x, false)?;
                let u = project(&w[1], &x, false)?;
                self.gpu
                    .activation(&cmd, &g, &u, c.ff * c.top * inputs.len())?
            };
            let routed = project(&w[2], &a, true)?;
            let sg = self.project_many(&cmd, &f.sgate, inputs)?;
            let su = self.project_many(&cmd, &f.sup, inputs)?;
            let activation = sg
                .iter()
                .zip(&su)
                .map(|(g, u)| self.gpu.activation(&cmd, g, u, c.sff))
                .collect::<Result<Vec<_>>>()?;
            let shared = self.project_many(&cmd, &f.sdown, &activation)?;
            let shared_gate = self.project_many(&cmd, &f.shared, inputs)?;
            let out = (0..inputs.len())
                .map(|i| {
                    let routed = self.copy(&cmd, &routed, c.width * c.top, i * c.width * c.top)?;
                    self.gpu.merge(
                        &cmd,
                        &routed,
                        &shared[i],
                        &weights[i],
                        &shared_gate[i],
                        c.width,
                        c.top,
                    )
                })
                .collect::<Result<_>>()?;
            return Ok((out, cmd));
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
        let sg = self.project_many(&cmd, &f.sgate, inputs)?;
        let su = self.project_many(&cmd, &f.sup, inputs)?;
        let activation = sg
            .iter()
            .zip(&su)
            .map(|(g, u)| self.gpu.activation(&cmd, g, u, c.sff))
            .collect::<Result<Vec<_>>>()?;
        let shared = self.project_many(&cmd, &f.sdown, &activation)?;
        let shared_gate = self.project_many(&cmd, &f.shared, inputs)?;
        let mut cache = match &f.experts {
            ExpertWeights::Cached(c) => Some(c.borrow_mut()),
            _ => None,
        };
        let mut slot = if let Some(cache) = cache.as_mut() {
            self.cache_slot(cache, groups[0].0, &[])?
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
            let act = self
                .gpu
                .activation(&cmd, &g, &u, c.ff * assignments.len())?;
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
                    Some(self.cache_slot(cache, groups[group + 1].0, &pinned))
                } else {
                    None
                };
                // Even a failed read cannot release resources still read by the GPU.
                self.gpu.wait(cmd)?;
                pinned.clear();
                slot = if let Some(next) = next {
                    next?
                } else if group + 1 < groups.len() {
                    self.cache_slot(cache, groups[group + 1].0, &[])?
                } else {
                    slot
                };
                cmd = self.gpu.command()?;
            }
        }
        let out = (0..inputs.len())
            .map(|i| {
                self.gpu.merge(
                    &cmd,
                    &routed[i],
                    &shared[i],
                    &weights[i],
                    &shared_gate[i],
                    c.width,
                    c.top,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((out, cmd))
    }
    fn ple_indices(&self, history: &[u32], token: u32) -> Vec<usize> {
        let c = &self.c;
        let mut ctx = vec![token as u64];
        let mut cut = false;
        for j in 1..c.ngram {
            let t = history.len().checked_sub(j).map(|i| history[i]);
            cut |= t.is_none() || t == Some(c.ple_eos);
            ctx.push(if cut {
                c.ple_eos as u64
            } else {
                t.unwrap_or(c.ple_eos) as u64
            });
        }
        let mut rows = Vec::new();
        for n in 2..=c.ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&c.multipliers)
                .fold(0, |v, (&t, &m)| v ^ t.wrapping_mul(m));
            for h in (n - 2) * c.pergram..(n - 1) * c.pergram {
                rows.push((mixed % c.sizes[h] + c.offsets[h]) as usize);
            }
        }
        rows
    }
    fn ple_rows(&mut self, history: &[u32], token: u32) -> Result<Buffer> {
        let rows = self.ple_indices(history, token);
        let c = &self.c;
        let m = self.gpu.read_rows(
            &mut self.g.borrow_mut(),
            "per_layer_token_embd.weight",
            &rows,
        )?;
        let w = self.gpu.bytes(&m.data)?;
        let y = self.gpu.output(c.width)?;
        let cmd = self.gpu.command()?;
        self.point(
            &cmd,
            "read_rows",
            &[&w, &y],
            &[m.cols as u32, m.rows as u32, m.stride as u32, m.kind],
            c.width,
        )?;
        self.gpu.complete(cmd)?;
        Ok(y)
    }
    fn ple(
        &self,
        cmd: &Command,
        p: &Ple,
        history: &Buffer,
        emb: &Buffer,
        x: &Buffer,
    ) -> Result<Buffer> {
        let k = p.key.project(&self.gpu, cmd, emb)?;
        let v = p.value.project(&self.gpu, cmd, emb)?;
        self.ple_projected(cmd, p, history, x, &k, &v)
    }
    fn ple_projected(
        &self,
        cmd: &Command,
        p: &Ple,
        history: &Buffer,
        x: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<Buffer> {
        let c = &self.c;
        let n = c.width * c.hc;
        let k = self.norm(cmd, k, &p.kn, c.width, c.hc, 0, c.width, false)?;
        let q = self.norm(cmd, x, &p.qn, c.width, c.hc, 0, c.width, false)?;
        let gated = self.gpu.output(n)?;
        self.gpu.op(
            cmd,
            "ple_gate",
            &[&k, &q, v, &gated],
            &[c.width as u32],
            c.hc,
            1,
        )?;
        let norm = self.norm(cmd, &gated, &p.cn, c.width, c.hc, 0, c.width, false)?;
        let conv = self.gpu.output(n)?;
        self.point(
            cmd,
            "convolution",
            &[&norm, &p.conv.buffer, history, &conv],
            &[
                n as u32,
                c.ple_taps as u32,
                c.ngram as u32,
                p.conv.t.kind,
                p.conv.t.stride as u32,
            ],
            n,
        )?;
        let out = self.gpu.output(n)?;
        self.point(cmd, "add_three", &[x, &gated, &conv, &out], &[n as u32], n)?;
        Ok(out)
    }
    pub fn metal_bytes(&self) -> usize {
        self.gpu.allocated_bytes()
    }
    pub(crate) fn width(&self) -> usize {
        self.c.width
    }
    pub(crate) fn prefill_block_size(&self) -> usize {
        self.prefill_block
    }
    pub(crate) fn set_profile_phase(&self, phase: &'static str) {
        self.gpu.set_profile_phase(phase);
    }
    pub fn metal_peak_bytes(&self) -> usize {
        self.gpu.peak_allocated_bytes()
    }
    pub(crate) fn forward_block(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
    ) -> Result<Vec<f32>> {
        let (history, tokens) = self.enter(history, tokens)?;
        let (history, tokens) = (&history[..], &tokens[..]);
        self.check_block(tokens)?;
        let mut history = history.to_vec();
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let frame = objc2::rc::autoreleasepool(|_| {
                self.forward_token(s, &history, token, cancellation)
            })?;
            if let Some(draft_state) = &mut s.draft {
                self.catch_up(
                    draft_state,
                    history.len(),
                    std::slice::from_ref(&token),
                    &frame.hidden,
                    cancellation,
                )?;
            }
            history.push(token);
            if i + 1 == tokens.len() {
                logits = frame.read_logits(&self.gpu)?.pop().unwrap();
            }
        }
        Ok(logits)
    }
    pub(crate) fn forward_next(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
    ) -> Result<u32> {
        let (history, tokens) = self.enter(history, tokens)?;
        let (history, tokens) = (&history[..], &tokens[..]);
        let f = self.forward(s, history, tokens, cancellation)?;
        Ok(*f.choices(&self.gpu)?.last().unwrap())
    }
    /// Batched forward that returns the final position's logits for sampling.
    pub(crate) fn forward_logits(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
    ) -> Result<Vec<f32>> {
        let (history, tokens) = self.enter(history, tokens)?;
        let (history, tokens) = (&history[..], &tokens[..]);
        let f = self.forward(s, history, tokens, cancellation)?;
        f.read_logits(&self.gpu)?
            .pop()
            .ok_or_else(|| invalid("missing logits"))
    }
    fn forward(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
    ) -> Result<Forward> {
        let result = self.target_block(s, history, tokens, cancellation, false)?;
        if let Some(draft_state) = &mut s.draft {
            self.catch_up(
                draft_state,
                history.len(),
                tokens,
                &result.hidden,
                cancellation,
            )?;
        }
        Ok(result)
    }
    fn check_block(&self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.prefill_block
            || tokens
                .iter()
                .any(|&t| t as usize >= self.embedding.t.dims[1])
        {
            return Err(invalid("invalid token block"));
        }
        Ok(())
    }
    fn target_block(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
        all: bool,
    ) -> Result<Forward> {
        self.check_block(tokens)?;
        objc2::rc::autoreleasepool(|_| {
            if tokens.len() == 1 {
                self.forward_token(s, history, tokens[0], cancellation)
            } else if tokens.len() >= 2 {
                self.packed_prefill(s, history, tokens, cancellation, all, all)
            } else {
                self.prefill_block(s, history, tokens, cancellation, all)
            }
        })
    }
    fn prefill_block(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancellation: &crate::engine::Cancellation,
        all: bool,
    ) -> Result<Forward> {
        self.trace_active.set(false);
        let mut snapshots: Vec<mtp::Checkpoint> = if all {
            (0..tokens.len() - 1)
                .map(|_| mtp::Checkpoint::default())
                .collect()
        } else {
            Vec::new()
        };
        let mut snapshot_encode = Duration::ZERO;
        let mut history = history.to_vec();
        let position = history.len();
        let mut embeddings = Vec::with_capacity(tokens.len());
        for &token in tokens {
            if cancellation.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            if !self.c.ple_layers.is_empty() {
                embeddings.push(self.ple_rows(&history, token)?);
            }
            history.push(token);
        }
        let cmd = self.gpu.command()?;
        let mut h = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let e = self.gpu.output(self.c.width)?;
            self.point(
                &cmd,
                "embedding",
                &[&self.embedding.buffer, &e],
                &[
                    self.c.width as u32,
                    1,
                    self.embedding.t.stride as u32,
                    self.embedding.t.kind,
                    token,
                ],
                self.c.width,
            )?;
            let x = self.gpu.output(self.c.width * self.c.hc)?;
            self.point(
                &cmd,
                "repeat_emb",
                &[&e, &x],
                &[self.c.width as u32, self.c.hc as u32],
                self.c.width * self.c.hc,
            )?;
            h.push(x);
        }
        self.gpu.complete(cmd)?;
        for (layer, l) in self.layers.iter().enumerate() {
            if cancellation.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            h = objc2::rc::autoreleasepool(|_| -> Result<Vec<Buffer>> {
                let mut h = h.clone();
                let cmd = self.gpu.command()?;
                if let Some(p) = &l.ple {
                    let keys = self.project_many(&cmd, &p.key, &embeddings)?;
                    let values = self.project_many(&cmd, &p.value, &embeddings)?;
                    for j in 0..tokens.len() {
                        h[j] = self.ple_projected(
                            &cmd,
                            p,
                            s.layers[layer]
                                .ple
                                .as_ref()
                                .ok_or_else(|| invalid("missing PLE state"))?,
                            &h[j],
                            &keys[j],
                            &values[j],
                        )?;
                        if let Some(snapshot) = snapshots.get_mut(j) {
                            let start = std::time::Instant::now();
                            self.save_buffer(
                                &cmd,
                                s.layers[layer].ple.as_ref().expect("PLE state"),
                                snapshot,
                            )?;
                            snapshot_encode += start.elapsed();
                        }
                    }
                }
                let (mixed, inject) = self.hc_many(&cmd, &l.attn, &h)?;
                let out = match &l.mixer {
                    Mixer::Linear(l) => {
                        let qkv = self.project_many(&cmd, &l.qkv, &mixed)?;
                        let z = self.project_many(&cmd, &l.z, &mixed)?;
                        let a = self.project_many(&cmd, &l.alpha, &mixed)?;
                        let b = self.project_many(&cmd, &l.beta, &mixed)?;
                        let mut out = Vec::with_capacity(tokens.len());
                        for j in 0..tokens.len() {
                            out.push(self.linear_projected(
                                &cmd,
                                l,
                                &s.layers[layer].mixer,
                                &[qkv[j].clone(), z[j].clone(), a[j].clone(), b[j].clone()],
                            )?);
                            if let Some(snapshot) = snapshots.get_mut(j) {
                                let start = std::time::Instant::now();
                                if let MixerState::Linear { history, state } =
                                    &s.layers[layer].mixer
                                {
                                    self.save_buffer(&cmd, history, snapshot)?;
                                    self.save_buffer(&cmd, state, snapshot)?;
                                }
                                snapshot_encode += start.elapsed();
                            }
                        }
                        self.project_many(&cmd, &l.out, &out)?
                    }
                    Mixer::Attention(a) => {
                        let q = self.project_many(&cmd, &a.q, &mixed)?;
                        let k = self.project_many(&cmd, &a.k, &mixed)?;
                        let v = self.project_many(&cmd, &a.v, &mixed)?;
                        let ik = self.project_many(&cmd, &a.ik, &mixed)?;
                        let iq = self.project_many(&cmd, &a.iq, &mixed)?;
                        let mut out = Vec::with_capacity(tokens.len());
                        for j in 0..tokens.len() {
                            out.push(self.attention_projected(
                                &cmd,
                                a,
                                &s.layers[layer].mixer,
                                &[
                                    q[j].clone(),
                                    k[j].clone(),
                                    v[j].clone(),
                                    ik[j].clone(),
                                    iq[j].clone(),
                                ],
                                position + j,
                            )?);
                        }
                        self.project_many(&cmd, &a.out, &out)?
                    }
                };
                h = h
                    .iter()
                    .zip(&out)
                    .zip(&inject)
                    .map(|((x, out), g)| self.hc_add(&cmd, x, out, g))
                    .collect::<Result<_>>()?;
                let (mixed, inject) = self.hc_many(&cmd, &l.ffn, &h)?;
                let (out, cmd) = self.ffn_many(cmd, &l.experts, &mixed, cancellation)?;
                h = h
                    .iter()
                    .zip(&out)
                    .zip(&inject)
                    .map(|((x, out), g)| self.hc_add(&cmd, x, out, g))
                    .collect::<Result<_>>()?;
                self.gpu.complete(cmd)?;
                Ok(h)
            })?;
        }
        let cmd = self.gpu.command()?;
        let selected = if all { &h[..] } else { &h[h.len() - 1..] };
        let mut outputs = Vec::with_capacity(selected.len());
        for hidden in selected {
            let (mixed, _) = self.hc_mix(&cmd, &self.head, hidden)?;
            outputs.push(self.output.project(&self.gpu, &cmd, &mixed)?);
        }
        let logits = self.pack(&cmd, &outputs, self.output.t.dims[1])?;
        self.gpu.complete(cmd)?;
        Ok(Forward {
            logits,
            rows: outputs.len(),
            vocabulary: self.output.t.dims[1],
            hidden: h,
            snapshots,
            snapshot_encode,
        })
    }
    fn forward_token(
        &mut self,
        s: &mut State,
        history: &[u32],
        token: u32,
        cancellation: &crate::engine::Cancellation,
    ) -> Result<Forward> {
        self.trace_active.set(
            self.trace
                .as_ref()
                .is_some_and(|(pos, _)| *pos == history.len()),
        );
        let emb = if self.c.ple_layers.is_empty() {
            None
        } else {
            Some(self.ple_rows(history, token)?)
        };
        let mut cmd = self.gpu.command()?;
        let e = self.gpu.output(self.c.width)?;
        self.point(
            &cmd,
            "embedding",
            &[&self.embedding.buffer, &e],
            &[
                self.c.width as u32,
                1,
                self.embedding.t.stride as u32,
                self.embedding.t.kind,
                token,
            ],
            self.c.width,
        )?;
        self.embed_images(&cmd, &e, history.len(), 1)?;
        let mut h = self.gpu.output(self.c.width * self.c.hc)?;
        self.point(
            &cmd,
            "repeat_emb",
            &[&e, &h],
            &[self.c.width as u32, self.c.hc as u32],
            self.c.width * self.c.hc,
        )?;
        // Keep at most two eight-layer chunks alive; queue ordering retains dependencies.
        let mut pending = std::collections::VecDeque::new();
        // The next block's HC mix, already encoded with the preceding residual add.
        let mut carried = None;
        for (i, l) in self.layers.iter().enumerate() {
            if cancellation.is_cancelled() {
                for cmd in pending.drain(..) {
                    self.gpu.wait(cmd)?;
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            (h, cmd) = objc2::rc::autoreleasepool(|_| -> Result<(Buffer, Command)> {
                let mut h = h.clone();
                if let Some(p) = &l.ple {
                    h = self.ple(
                        &cmd,
                        p,
                        s.layers[i]
                            .ple
                            .as_ref()
                            .ok_or_else(|| invalid("missing PLE state"))?,
                        emb.as_ref()
                            .ok_or_else(|| invalid("missing PLE embedding"))?,
                        &h,
                    )?;
                }
                let (mixed, inject) = match carried.take() {
                    Some(mixed) => mixed,
                    None => self.hc_mix(&cmd, &l.attn, &h)?,
                };
                let out = match &l.mixer {
                    Mixer::Linear(l) => self.linear(&cmd, l, &s.layers[i].mixer, &mixed)?,
                    Mixer::Attention(a) => {
                        self.attention(&cmd, a, &s.layers[i].mixer, &mixed, history.len())?
                    }
                };
                self.record(
                    if matches!(l.mixer, Mixer::Linear(_)) {
                        "linear_attn_out"
                    } else {
                        "attn_output"
                    },
                    &out,
                );
                let inject = inject.ok_or_else(|| invalid("missing HC injection"))?;
                let (residual, mixed, inject) = self.hc_add_mix(&cmd, &h, &out, &inject, &l.ffn)?;
                h = residual;
                self.record("hc_mixed", &mixed);
                let (out, cmd) = self.ffn(cmd, &l.experts, &mixed, true, cancellation)?;
                self.record("ffn_out", &out);
                let inject = inject.ok_or_else(|| invalid("missing FFN injection"))?;
                // A following PLE block consumes the raw residual, so it cannot reuse the norm.
                let next = (self.layers.get(i + 1))
                    .map_or(Some(&self.head), |n| n.ple.is_none().then_some(&n.attn));
                if let Some(next) = next {
                    let (residual, mixed, gates) =
                        self.hc_add_mix(&cmd, &h, &out, &inject, next)?;
                    (h, carried) = (residual, Some((mixed, gates)));
                } else {
                    h = self.hc_add(&cmd, &h, &out, &inject)?;
                }
                self.record("l_last", &h);
                // Bound retained activations while avoiding one host wait per layer.
                let cmd = if self.trace_active.get() {
                    self.gpu.complete(cmd)?;
                    self.flush_trace(i)?;
                    self.gpu.command()?
                } else if (i + 1) % 8 == 0 {
                    self.gpu.submit(&cmd);
                    pending.push_back(cmd);
                    if pending.len() > 1 {
                        self.gpu.wait(pending.pop_front().unwrap())?;
                    }
                    self.gpu.command()?
                } else {
                    cmd
                };
                Ok((h, cmd))
            })?;
        }
        self.record("h_nextn", &h);
        let (mixed, _) = match carried.take() {
            Some(mixed) => mixed,
            None => self.hc_mix(&cmd, &self.head, &h)?,
        };
        let logits = self.output.project(&self.gpu, &cmd, &mixed)?;
        self.gpu.complete(cmd)?;
        for cmd in pending {
            self.gpu.wait(cmd)?;
        }
        self.flush_trace(self.c.layers)?;
        Ok(Forward {
            logits,
            rows: 1,
            vocabulary: self.output.t.dims[1],
            hidden: vec![h],
            snapshots: Vec::new(),
            snapshot_encode: Duration::ZERO,
        })
    }
}
