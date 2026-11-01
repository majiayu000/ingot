//! Configuration and weights of the generic decoder path per architecture:
//! Llama (also Mistral and Mixtral GGUFs), Qwen2/3(-MoE), Gemma 3, Gemma 4
//! and Phi-3. Features outside these are rejected by name.
use crate::gguf::Gguf;
use crate::metal::{Buffer, Gpu};
use crate::weights::Weight;
use crate::{Result, invalid};
use std::collections::HashSet;

pub(crate) const ARCHITECTURES: [&str; 7] = [
    "llama", "qwen2", "qwen3", "qwen3moe", "gemma3", "gemma4", "phi3",
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Llama,
    Qwen,
    Gemma3,
    Gemma4,
    Phi3,
}

/// Attention geometry of one layer.
pub(crate) struct Geometry {
    pub(crate) heads: usize,
    pub(crate) kv: usize,
    pub(crate) head: usize,
    pub(crate) rot: usize,
    pub(crate) base: f32,
    /// Sliding-window size in keys; 0 is full causal attention.
    pub(crate) window: usize,
    /// Query multiplier so the kernels' 1/sqrt(head) gives the model's scale.
    pub(crate) q_scale: f32,
    /// Global (non-sliding) layer: takes `rope_freqs` and linear RoPE scaling.
    pub(crate) global: bool,
}

pub(crate) struct Config {
    pub(crate) family: Family,
    pub(crate) layers: usize,
    pub(crate) width: usize,
    pub(crate) ff: usize,
    pub(crate) eps: f32,
    pub(crate) neox: bool,
    pub(crate) experts: usize,
    pub(crate) top: usize,
    pub(crate) geometry: Vec<Geometry>,
    /// Linear RoPE scaling factor of global layers (1 when absent).
    pub(crate) rope_scale: f32,
    /// Embedding multiplier (Gemma: sqrt(width)).
    pub(crate) embed_scale: Option<f32>,
    /// Final logit soft-capping value.
    pub(crate) softcap: Option<f32>,
    /// GeGLU (tanh GeLU) instead of SwiGLU.
    pub(crate) gelu: bool,
}

impl Config {
    pub(crate) fn load(g: &Gguf) -> Result<Self> {
        let a = g.architecture.as_str();
        let key = |k: &str| format!("{a}.{k}");
        let unsupported = |what: &str| invalid(format!("unsupported {a} feature: {what}"));
        // Present but non-integer (for example per-layer arrays) is unsupported.
        let optional = |k: &str| -> Result<Option<usize>> {
            match (
                g.optional_integer(&key(k)),
                g.metadata.contains_key(&key(k)),
            ) {
                (Some(v), _) => Ok(Some(v)),
                (None, false) => Ok(None),
                (None, true) => Err(invalid(format!(
                    "unsupported {}: not a single integer",
                    key(k)
                ))),
            }
        };
        let family = match a {
            "llama" => Family::Llama,
            "qwen2" | "qwen3" | "qwen3moe" => Family::Qwen,
            "gemma3" => Family::Gemma3,
            "gemma4" => Family::Gemma4,
            "phi3" => Family::Phi3,
            _ => return Err(invalid(format!("unsupported architecture {a}"))),
        };
        let gemma = matches!(family, Family::Gemma3 | Family::Gemma4);
        let mut rope_scale = 1.0;
        if let Some(kind) = g.optional_string(&key("rope.scaling.type")) {
            match kind {
                "none" | "llama3" => {}
                "linear" => {
                    rope_scale = g
                        .optional_float(&key("rope.scaling.factor"))
                        .filter(|f| *f > 0.0)
                        .unwrap_or(1.0)
                }
                _ => return Err(invalid(format!("unsupported RoPE scaling: {kind}"))),
            }
        }
        if g.tensors.contains_key("rope_factors_long.weight") {
            return Err(unsupported("LongRoPE (rope_factors_long)"));
        }
        if family == Family::Gemma4 {
            for (k, what) in [
                ("attention.shared_kv_layers", "shared KV layers"),
                ("embedding_length_per_layer_input", "per-layer embeddings"),
                ("expert_count", "mixture of experts"),
            ] {
                if g.optional_integer(&key(k)).is_some_and(|n| n > 0) {
                    return Err(unsupported(what));
                }
            }
        }
        let width = g.integer(&key("embedding_length"))?;
        let heads = g.integer(&key("attention.head_count"))?;
        let layers = g.integer(&key("block_count"))?;
        let per_layer = |k: &str, default: usize| -> Result<Vec<usize>> {
            match g.metadata.get(&key(k)) {
                None => Ok(vec![default; layers]),
                Some(crate::gguf::Value::Array(_)) => {
                    let v = g.integers(&key(k))?;
                    if v.len() != layers {
                        return Err(invalid(format!("{}: expected {layers} values", key(k))));
                    }
                    Ok(v.into_iter().map(|x| x as usize).collect())
                }
                Some(_) => Ok(vec![g.integer(&key(k))?; layers]),
            }
        };
        let kv = per_layer("attention.head_count_kv", heads)?;
        let head = optional("attention.key_length")?.unwrap_or(width / heads.max(1));
        let rot = optional("rope.dimension_count")?.unwrap_or(head);
        let base = g.optional_float(&key("rope.freq_base")).unwrap_or(10000.0);
        // Sliding windows: Gemma 3 every sixth layer is global unless a
        // pattern is given; Gemma 4 declares the pattern; Phi-3's declared
        // window is ignored, as in llama.cpp.
        let window = match family {
            Family::Phi3 | Family::Qwen => 0,
            _ => optional("attention.sliding_window")?.unwrap_or(0),
        };
        let pattern_key = key("attention.sliding_window_pattern");
        let swa: Vec<bool> = match g.metadata.get(&pattern_key) {
            _ if window == 0 => vec![false; layers],
            Some(crate::gguf::Value::Array(_)) => {
                let v = g.integers(&pattern_key)?;
                if v.len() != layers {
                    return Err(invalid(format!("{pattern_key}: expected {layers} values")));
                }
                v.into_iter().map(|b| b != 0).collect()
            }
            _ if family == Family::Gemma3 => {
                let n = g.optional_integer(&pattern_key).unwrap_or(6);
                (0..layers).map(|i| n == 0 || i % n < n - 1).collect()
            }
            Some(_) => return Err(unsupported("integer sliding-window pattern")),
            None => vec![true; layers],
        };
        let base_swa = g
            .optional_float(&key("rope.freq_base_swa"))
            .unwrap_or(10000.0);
        let head_swa = optional("attention.key_length_swa")?.unwrap_or(head);
        let rot_swa = optional("rope.dimension_count_swa")?.unwrap_or(head_swa);
        let geometry = (0..layers)
            .map(|i| {
                let sliding = swa[i];
                let (head, rot) = if family == Family::Gemma4 && sliding {
                    (head_swa, rot_swa)
                } else {
                    (head, rot)
                };
                let scale = match family {
                    Family::Gemma4 => 1.0,
                    // gemma_pytorch: 27B scales by width/heads, the rest by head.
                    Family::Gemma3 if layers == 62 => 1.0 / ((width / heads) as f32).sqrt(),
                    _ => 1.0 / (head as f32).sqrt(),
                };
                Geometry {
                    heads,
                    kv: kv[i],
                    head,
                    rot,
                    base: if sliding && gemma { base_swa } else { base },
                    window: if sliding { window } else { 0 },
                    q_scale: scale * (head as f32).sqrt(),
                    global: !sliding,
                }
            })
            .collect::<Vec<_>>();
        let experts = optional("expert_count")?.unwrap_or(0);
        let c = Self {
            family,
            layers,
            width,
            ff: if experts > 0 {
                g.integer(&key("expert_feed_forward_length"))?
            } else {
                g.integer(&key("feed_forward_length"))?
            },
            eps: g.float(&key("attention.layer_norm_rms_epsilon"))?,
            neox: family != Family::Llama,
            experts,
            top: optional("expert_used_count")?.unwrap_or(0),
            geometry,
            rope_scale,
            embed_scale: gemma.then(|| (width as f32).sqrt()),
            softcap: g
                .optional_float(&key("final_logit_softcapping"))
                .filter(|&c| c > 0.0 && gemma),
            gelu: gemma,
        };
        if layers == 0
            || (experts > 0) != (c.top > 0)
            || c.top > experts
            || optional("attention.value_length")?.is_some_and(|v| v != head)
            || optional("attention.value_length_swa")?.is_some_and(|v| v != head_swa)
            || c.geometry.iter().any(|l| {
                l.heads == 0
                    || l.kv == 0
                    || !l.heads.is_multiple_of(l.kv)
                    || l.rot > l.head
                    || !l.rot.is_multiple_of(2)
            })
        {
            return Err(invalid(format!(
                "unsupported or malformed {a} configuration"
            )));
        }
        Ok(c)
    }
    pub(crate) fn state_bytes(&self, context: usize) -> usize {
        self.geometry
            .iter()
            .map(|l| 2 * context * l.kv * l.head * 4)
            .sum()
    }
    pub(crate) fn max_q(&self) -> usize {
        self.geometry
            .iter()
            .map(|l| l.heads * l.head)
            .max()
            .unwrap_or(0)
    }
    pub(crate) fn max_heads(&self) -> usize {
        self.geometry.iter().map(|l| l.heads).max().unwrap_or(0)
    }
}

pub(crate) enum Qkv {
    Split([Weight; 3]),
    /// Phi-3 `attn_qkv`: rows q | k | v.
    Fused(Weight),
    /// Gemma 4 global layers without `attn_v`: values are the key projection.
    KeyAsValue([Weight; 2]),
}
pub(crate) enum Ffn {
    Dense([Weight; 3]),
    /// Phi-3 `ffn_up` with rows gate | up, and `ffn_down`.
    Fused([Weight; 2]),
    Moe {
        router: Weight,
        experts: [Weight; 3],
    },
}
pub(crate) struct Layer {
    pub(crate) attn_norm: Weight,
    pub(crate) qkv: Qkv,
    pub(crate) bias: Option<[Weight; 3]>,
    pub(crate) qk_norm: Option<[Weight; 2]>,
    pub(crate) out: Weight,
    pub(crate) ffn_norm: Weight,
    pub(crate) ffn: Ffn,
    /// Gemma post-attention and post-FFN norms.
    pub(crate) post_norms: Option<[Weight; 2]>,
    /// Gemma 4 per-layer output scalar.
    pub(crate) out_scale: Option<f32>,
    /// RoPE frequency divisors of this layer, if any.
    pub(crate) factors: Option<Buffer>,
}

/// Loads tensors and records their names so unconsumed tensors (an
/// unsupported feature) are rejected.
pub(crate) struct Loader<'a> {
    pub(crate) g: &'a mut Gguf,
    gpu: &'a Gpu,
    used: HashSet<String>,
}
impl<'a> Loader<'a> {
    pub(crate) fn new(g: &'a mut Gguf, gpu: &'a Gpu) -> Self {
        Self {
            g,
            gpu,
            used: HashSet::new(),
        }
    }
    pub(crate) fn has(&self, name: &str) -> bool {
        self.g.tensors.contains_key(name)
    }
    pub(crate) fn load(&mut self, name: &str, dims: &[usize]) -> Result<Weight> {
        self.used.insert(name.to_owned());
        Weight::load(self.g, self.gpu, name, dims)
    }
    pub(crate) fn optional(&mut self, name: &str, dims: &[usize]) -> Result<Option<Weight>> {
        if self.has(name) {
            self.load(name, dims).map(Some)
        } else {
            Ok(None)
        }
    }
    fn scalar(&mut self, name: &str) -> Result<Option<f32>> {
        if !self.has(name) {
            return Ok(None);
        }
        self.used.insert(name.to_owned());
        let t = self.g.tensor(name)?.clone();
        if t.kind != 0 || t.dims != [1] {
            return Err(invalid(format!("{name}: expected one F32 value")));
        }
        let mut bytes = [0u8; 4];
        self.g.read_range(&t, 0, &mut bytes)?;
        Ok(Some(f32::from_le_bytes(bytes)))
    }
    /// Error naming the first tensor no part of the model consumed.
    pub(crate) fn check_all_used(&self) -> Result<()> {
        let mut unused: Vec<_> = self
            .g
            .tensors
            .keys()
            .filter(|n| !self.used.contains(*n))
            .collect();
        unused.sort();
        match unused.first() {
            Some(name) => Err(invalid(format!(
                "unsupported {} tensor {name} ({} unused tensors)",
                self.g.architecture,
                unused.len()
            ))),
            None => Ok(()),
        }
    }

    pub(crate) fn layers(&mut self, c: &Config) -> Result<Vec<Layer>> {
        let (w, f) = (c.width, c.ff);
        let gemma = matches!(c.family, Family::Gemma3 | Family::Gemma4);
        // Llama 3.1 / Gemma 4 `rope_freqs` (global layers) and linear scaling.
        let rot = c.geometry.iter().find(|l| l.global).map_or(0, |l| l.rot);
        let freqs = self.optional("rope_freqs.weight", &[rot / 2])?;
        if let Some(f) = &freqs {
            f.require_f32()?;
        }
        let factors = match (freqs, c.rope_scale != 1.0) {
            (Some(_), true) => return Err(invalid("rope_freqs with linear RoPE scaling")),
            (Some(f), false) => Some(f.buffer),
            (None, true) => Some(self.gpu.floats(&vec![c.rope_scale; rot / 2])?),
            (None, false) => None,
        };
        let mut layers = Vec::with_capacity(c.layers);
        for (i, l) in c.geometry.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let (h, kvd) = (l.heads * l.head, l.kv * l.head);
            let qkv = if self.has(&p("attn_qkv.weight")) {
                Qkv::Fused(self.load(&p("attn_qkv.weight"), &[w, h + 2 * kvd])?)
            } else {
                let q = self.load(&p("attn_q.weight"), &[w, h])?;
                let k = self.load(&p("attn_k.weight"), &[w, kvd])?;
                match self.optional(&p("attn_v.weight"), &[w, kvd])? {
                    Some(v) => Qkv::Split([q, k, v]),
                    None if c.family == Family::Gemma4 => Qkv::KeyAsValue([q, k]),
                    None => return Err(invalid(format!("missing {}", p("attn_v.weight")))),
                }
            };
            let ffn = if c.experts > 0 {
                Ffn::Moe {
                    router: self.load(&p("ffn_gate_inp.weight"), &[w, c.experts])?,
                    experts: [
                        self.load(&p("ffn_gate_exps.weight"), &[w, f, c.experts])?,
                        self.load(&p("ffn_up_exps.weight"), &[w, f, c.experts])?,
                        self.load(&p("ffn_down_exps.weight"), &[f, w, c.experts])?,
                    ],
                }
            } else if !self.has(&p("ffn_gate.weight")) && c.family == Family::Phi3 {
                Ffn::Fused([
                    self.load(&p("ffn_up.weight"), &[w, 2 * f])?,
                    self.load(&p("ffn_down.weight"), &[f, w])?,
                ])
            } else {
                Ffn::Dense([
                    self.load(&p("ffn_gate.weight"), &[w, f])?,
                    self.load(&p("ffn_up.weight"), &[w, f])?,
                    self.load(&p("ffn_down.weight"), &[f, w])?,
                ])
            };
            let bias = if self.has(&p("attn_q.bias")) {
                Some([
                    self.load(&p("attn_q.bias"), &[h])?,
                    self.load(&p("attn_k.bias"), &[kvd])?,
                    self.load(&p("attn_v.bias"), &[kvd])?,
                ])
            } else {
                None
            };
            let qk_norm = if self.has(&p("attn_q_norm.weight")) {
                Some([
                    self.load(&p("attn_q_norm.weight"), &[l.head])?,
                    self.load(&p("attn_k_norm.weight"), &[l.head])?,
                ])
            } else {
                None
            };
            let post_norms = if gemma {
                Some([
                    self.load(&p("post_attention_norm.weight"), &[w])?,
                    self.load(&p("post_ffw_norm.weight"), &[w])?,
                ])
            } else {
                None
            };
            let layer = Layer {
                attn_norm: self.load(&p("attn_norm.weight"), &[w])?,
                qkv,
                bias,
                qk_norm,
                out: self.load(&p("attn_output.weight"), &[h, w])?,
                ffn_norm: self.load(&p("ffn_norm.weight"), &[w])?,
                ffn,
                post_norms,
                out_scale: self.scalar(&p("layer_output_scale.weight"))?,
                factors: if l.global || c.family == Family::Llama {
                    factors.clone()
                } else {
                    None
                },
            };
            for weight in std::iter::once(&layer.attn_norm)
                .chain([&layer.ffn_norm])
                .chain(layer.qk_norm.iter().flatten())
                .chain(layer.bias.iter().flatten())
                .chain(layer.post_norms.iter().flatten())
            {
                weight.require_f32()?;
            }
            layers.push(layer);
        }
        Ok(layers)
    }
}
