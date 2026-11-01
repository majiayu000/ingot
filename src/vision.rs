//! Vision encoder for GGUF multimodal projectors (`mmproj`) of type
//! `qwen3vl_merger`: a ViT with learned, bilinearly resized position
//! embeddings and 2D RoPE, then a 2x2 patch merger MLP into the text model's
//! embedding width. It owns its own Metal queue and workspace, so encoding an
//! image never competes with the text model's bounded buffers.
use crate::gguf::{Gguf, Tensor};
use crate::image::{self, Rgb};
use crate::metal::{Buffer, Command, Gpu};
use crate::weights::Weight;
use crate::{Result, invalid};
use std::path::Path;

/// Chat-template text for one image; the placeholder token becomes the image's tokens.
pub const IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// Default token limits of llama.cpp's Qwen-VL preprocessing.
const MIN_TOKENS: usize = 8;
const MAX_TOKENS: usize = 4096;

struct Block {
    norm1: [Weight; 2],
    qkv: [Weight; 2],
    out: [Weight; 2],
    norm2: [Weight; 2],
    up: [Weight; 2],
    down: [Weight; 2],
}

pub struct Encoder {
    gpu: Gpu,
    width: usize,
    heads: usize,
    patch: usize,
    merge: usize,
    /// Text embedding width of the merged tokens.
    pub output: usize,
    eps: f32,
    mean: [f32; 3],
    std: [f32; 3],
    /// Side of the square learned position grid.
    grid: usize,
    /// Learned position embeddings, `grid * grid` rows of `width`.
    positions: Vec<f32>,
    patch_bias: Vec<f32>,
    /// Sum of the two temporal patch kernels: a still image is two equal frames.
    embed: Weight,
    blocks: Vec<Block>,
    post: [Weight; 2],
    merger: [[Weight; 2]; 2],
    bytes: usize,
}

/// Merged image tokens, `width * height` rows of the text embedding width in
/// raster order.
pub struct Encoded {
    pub embeddings: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

fn floats(w: &Weight) -> Result<Vec<f32>> {
    w.require_f32()?;
    // SAFETY: a mapped, read-only F32 tensor of `t.bytes` bytes.
    let data = unsafe {
        std::slice::from_raw_parts(w.buffer.contents().as_ptr().cast::<f32>(), w.t.bytes / 4)
    };
    Ok(data.to_vec())
}

impl Encoder {
    pub fn load(path: &Path) -> Result<Self> {
        let mut g = Gguf::open(path)?;
        if g.architecture != "clip" || g.optional_integer("clip.has_vision_encoder") != Some(1) {
            return Err(invalid(format!(
                "{} is not a vision projector (mmproj) GGUF",
                path.display()
            )));
        }
        let kind = g.optional_string("clip.projector_type").unwrap_or("");
        if kind != "qwen3vl_merger" {
            return Err(invalid(format!(
                "unsupported vision projector type {kind:?}; supported: qwen3vl_merger"
            )));
        }
        if g.integers("clip.vision.is_deepstack_layers")
            .is_ok_and(|layers| layers.iter().any(|&d| d != 0))
        {
            return Err(invalid("deepstack vision projectors are not supported"));
        }
        let n = |k: &str| g.integer(&format!("clip.vision.{k}"));
        let (width, heads, ff, layers) = (
            n("embedding_length")?,
            n("attention.head_count")?,
            n("feed_forward_length")?,
            n("block_count")?,
        );
        let (patch, output) = (n("patch_size")?, n("projection_dim")?);
        let merge = g
            .optional_integer("clip.vision.spatial_merge_size")
            .unwrap_or(2);
        let eps = g.float("clip.vision.attention.layer_norm_epsilon")?;
        let three = |k: &str| -> Result<[f32; 3]> {
            match g.array(&format!("clip.vision.{k}"))? {
                [a, b, c] => {
                    let f = |v: &crate::gguf::Value| match v {
                        crate::gguf::Value::Float(x) => Ok(*x as f32),
                        _ => Err(invalid(format!("clip.vision.{k}: not a float"))),
                    };
                    Ok([f(a)?, f(b)?, f(c)?])
                }
                _ => Err(invalid(format!("clip.vision.{k}: expected three values"))),
            }
        };
        let (mean, std) = (three("image_mean")?, three("image_std")?);
        if heads == 0 || width != heads * 72 || merge != 2 || patch == 0 || layers == 0 {
            return Err(invalid(
                "unsupported vision geometry: the encoder needs 72-dim heads and 2x2 merging",
            ));
        }
        let gpu = Gpu::new()?;
        let mut load = |name: &str, dims: &[usize]| Weight::load(&mut g, &gpu, name, dims);
        let pair = |load: &mut dyn FnMut(&str, &[usize]) -> Result<Weight>,
                    p: &str,
                    w: &[usize],
                    b: usize|
         -> Result<[Weight; 2]> {
            Ok([
                load(&format!("{p}.weight"), w)?,
                load(&format!("{p}.bias"), &[b])?,
            ])
        };
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(Block {
                norm1: pair(&mut load, &p("ln1"), &[width], width)?,
                qkv: pair(&mut load, &p("attn_qkv"), &[width, 3 * width], 3 * width)?,
                out: pair(&mut load, &p("attn_out"), &[width, width], width)?,
                norm2: pair(&mut load, &p("ln2"), &[width], width)?,
                up: pair(&mut load, &p("ffn_up"), &[width, ff], ff)?,
                down: pair(&mut load, &p("ffn_down"), &[ff, width], width)?,
            });
        }
        let post = pair(&mut load, "v.post_ln", &[width], width)?;
        let wide = width * merge * merge;
        let merger = [
            pair(&mut load, "mm.0", &[wide, wide], wide)?,
            pair(&mut load, "mm.2", &[wide, output], output)?,
        ];
        let kernel = [patch, patch, 3, width];
        let first = floats(&load("v.patch_embd.weight", &kernel)?)?;
        let second = floats(&load("v.patch_embd.weight.1", &kernel)?)?;
        let summed: Vec<f32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
        let cols = 3 * patch * patch;
        let embed = Weight {
            buffer: gpu.floats(&summed)?,
            t: Tensor {
                dims: vec![cols, width],
                kind: 0,
                offset: 0,
                shard: 0,
                stride: cols * 4,
                bytes: summed.len() * 4,
            },
        };
        let patch_bias = floats(&load("v.patch_embd.bias", &[width])?)?;
        let position = g.tensor("v.position_embd.weight")?.dims.clone();
        let grid = position.get(1).map_or(0, |&n| n.isqrt());
        if position.len() != 2 || position[0] != width || grid * grid != position[1] {
            return Err(invalid("vision position embeddings must be a square grid"));
        }
        let positions = floats(&Weight::load(
            &mut g,
            &gpu,
            "v.position_embd.weight",
            &position,
        )?)?;
        for w in blocks
            .iter()
            .flat_map(|b| [&b.norm1, &b.norm2])
            .flatten()
            .chain(&post)
            .chain(
                blocks
                    .iter()
                    .flat_map(|b| [&b.qkv[1], &b.out[1], &b.up[1], &b.down[1]]),
            )
            .chain(merger.iter().map(|m| &m[1]))
        {
            w.require_f32()?;
        }
        let bytes = g.tensors.values().map(|t| t.bytes).sum();
        Ok(Self {
            gpu,
            width,
            heads,
            patch,
            merge,
            output,
            eps,
            mean,
            std,
            grid,
            positions,
            patch_bias,
            embed,
            blocks,
            post,
            merger,
            bytes,
        })
    }

    /// Bytes of projector weights; the transient activations are on top.
    pub fn weight_bytes(&self) -> usize {
        self.bytes
    }

    /// Merged-token grid (columns, rows) for an image of `width` x `height`.
    pub fn grid(&self, width: usize, height: usize) -> (usize, usize) {
        let cell = self.patch * self.merge;
        let (w, h) = image::smart_size(
            width,
            height,
            cell,
            MIN_TOKENS * cell * cell,
            MAX_TOKENS * cell * cell,
        );
        (w / cell, h / cell)
    }

    pub fn encode(&self, rgb: &Rgb) -> Result<Encoded> {
        let (cols, rows) = self.grid(rgb.width, rgb.height);
        let cell = self.patch * self.merge;
        let resized = image::resize(rgb, cols * cell, rows * cell);
        let pixels = image::patches(&resized, self.patch, self.mean, self.std);
        let (gw, gh) = (cols * self.merge, rows * self.merge);
        let n = gw * gh;
        let (additive, pos) = self.position_inputs(gw, gh);
        objc2::rc::autoreleasepool(|_| self.forward(&pixels, &additive, &pos, n)).map(
            |embeddings| Encoded {
                embeddings,
                width: cols,
                height: rows,
            },
        )
    }

    /// Resized position embeddings plus the patch bias, and each patch's
    /// (row, column), both in merge order.
    fn position_inputs(&self, gw: usize, gh: usize) -> (Vec<f32>, Vec<u32>) {
        let (d, side) = (self.width, self.grid);
        // ggml bilinear interpolation with aligned corners.
        let axis = |i: usize, out: usize| -> (usize, usize, f32) {
            let scale = if out > 1 && side > 1 {
                (out - 1) as f32 / (side - 1) as f32
            } else {
                out as f32 / side as f32
            };
            let x = i as f32 / scale;
            let x0 = (x.floor() as usize).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            (x0, x1, (x - x0 as f32).clamp(0.0, 1.0))
        };
        let mut additive = Vec::with_capacity(gw * gh * d);
        let mut pos = Vec::with_capacity(gw * gh * 2);
        for y in (0..gh).step_by(2) {
            for x in (0..gw).step_by(2) {
                for (py, px) in [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                    pos.extend([py as u32, px as u32]);
                    let row = |r: usize, c: usize| &self.positions[(r * side + c) * d..][..d];
                    if gw == side && gh == side {
                        let p = row(py, px);
                        additive.extend(p.iter().zip(&self.patch_bias).map(|(a, b)| a + b));
                        continue;
                    }
                    let (y0, y1, dy) = axis(py, gh);
                    let (x0, x1, dx) = axis(px, gw);
                    let (a, b, c, e) = (row(y0, x0), row(y0, x1), row(y1, x0), row(y1, x1));
                    for j in 0..d {
                        let v = a[j] * (1.0 - dx) * (1.0 - dy)
                            + b[j] * dx * (1.0 - dy)
                            + c[j] * (1.0 - dx) * dy
                            + e[j] * dx * dy;
                        additive.push(v + self.patch_bias[j]);
                    }
                }
            }
        }
        (additive, pos)
    }

    fn project(&self, cmd: &Command, w: &Weight, x: &Buffer, count: usize) -> Result<Buffer> {
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
    fn point(&self, cmd: &Command, name: &str, b: &[&Buffer], p: &[u32], n: usize) -> Result<()> {
        self.gpu.op(cmd, name, b, p, n.div_ceil(32), 1)
    }
    fn norm(&self, cmd: &Command, x: &Buffer, w: &[Weight; 2], rows: usize) -> Result<Buffer> {
        let y = self.gpu.output(rows * self.width)?;
        self.gpu.op(
            cmd,
            "vit_norm",
            &[x, &w[0].buffer, &w[1].buffer, &y],
            &[self.width as u32, self.eps.to_bits()],
            rows,
            1,
        )?;
        Ok(y)
    }
    /// `r + x + bias`, or `x + bias` without a residual.
    fn bias_add(
        &self,
        cmd: &Command,
        r: Option<&Buffer>,
        x: &Buffer,
        b: &Weight,
        n: usize,
    ) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        let period = (b.t.bytes / 4) as u32;
        match r {
            Some(r) => self.point(
                cmd,
                "vit_bias_add",
                &[r, x, &b.buffer, &y],
                &[n as u32, period],
                n,
            )?,
            None => self.point(
                cmd,
                "bias_rows",
                &[x, &b.buffer, &y],
                &[n as u32, period],
                n,
            )?,
        }
        Ok(y)
    }
    fn bias_gelu(&self, cmd: &Command, x: &Buffer, b: &Weight, n: usize) -> Result<Buffer> {
        let y = self.gpu.output(n)?;
        let period = (b.t.bytes / 4) as u32;
        self.point(
            cmd,
            "vit_bias_gelu",
            &[x, &b.buffer, &y],
            &[n as u32, period],
            n,
        )?;
        Ok(y)
    }

    fn forward(&self, pixels: &[f32], additive: &[f32], pos: &[u32], n: usize) -> Result<Vec<f32>> {
        let (d, gpu) = (self.width, &self.gpu);
        let padded = n.div_ceil(8) * 8;
        let pos_bytes: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pos = gpu.bytes(&pos_bytes)?;
        let cmd = gpu.command()?;
        let x = gpu.floats(pixels)?;
        let embedded = self.project(&cmd, &self.embed, &x, n)?;
        let additive = gpu.floats(additive)?;
        let mut h = gpu.output(n * d)?;
        self.point(
            &cmd,
            "add_rows",
            &[&embedded, &additive, &h],
            &[(n * d) as u32],
            n * d,
        )?;
        gpu.complete(cmd)?;
        let scale = (1.0 / (d / self.heads) as f32).sqrt();
        for block in &self.blocks {
            let cmd = gpu.command()?;
            let normed = self.norm(&cmd, &h, &block.norm1, n)?;
            let qkv = self.project(&cmd, &block.qkv[0], &normed, n)?;
            let [q, k, v] = [(); 3].map(|_| gpu.output(padded * d));
            let (q, k, v) = (q?, k?, v?);
            self.point(
                &cmd,
                "vit_qkv",
                &[&qkv, &block.qkv[1].buffer, &pos, &q, &k, &v],
                &[
                    d as u32,
                    n as u32,
                    padded as u32,
                    (d / self.heads) as u32,
                    10000f32.to_bits(),
                ],
                padded * d,
            )?;
            let attended = gpu.output(n * d)?;
            gpu.op(
                &cmd,
                "vit_attention",
                &[&q, &k, &v, &attended],
                &[n as u32, self.heads as u32, d as u32, scale.to_bits()],
                padded / 8,
                self.heads,
            )?;
            let o = self.project(&cmd, &block.out[0], &attended, n)?;
            h = self.bias_add(&cmd, Some(&h), &o, &block.out[1], n * d)?;
            let normed = self.norm(&cmd, &h, &block.norm2, n)?;
            let up = self.project(&cmd, &block.up[0], &normed, n)?;
            let ff = block.up[0].t.dims[1];
            let act = self.bias_gelu(&cmd, &up, &block.up[1], n * ff)?;
            let down = self.project(&cmd, &block.down[0], &act, n)?;
            h = self.bias_add(&cmd, Some(&h), &down, &block.down[1], n * d)?;
            gpu.complete(cmd)?;
        }
        let cmd = gpu.command()?;
        let normed = self.norm(&cmd, &h, &self.post, n)?;
        let merged = n / (self.merge * self.merge);
        let wide = self.merger[0][0].t.dims[1];
        let hidden = self.project(&cmd, &self.merger[0][0], &normed, merged)?;
        let act = self.bias_gelu(&cmd, &hidden, &self.merger[0][1], merged * wide)?;
        let out = self.project(&cmd, &self.merger[1][0], &act, merged)?;
        let out = self.bias_add(&cmd, None, &out, &self.merger[1][1], merged * self.output)?;
        gpu.complete(cmd)?;
        gpu.read_f32(&out)
    }
}
