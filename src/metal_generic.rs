//! Generic-path kernels (kernels/generic.metal) dispatched with an explicit
//! threadgroup size.
use super::{Buffer, Command, Gpu};
use crate::weights::Weight;
use crate::{Result, invalid};
use objc2_metal::*;
use std::io;

pub(super) const OPERATIONS: [&str; 42] = [
    "routed_q4k",
    "routed_q6k",
    "flash_prefill_64",
    "flash_prefill_64_tail",
    "flash_prefill_96",
    "flash_prefill_96_tail",
    "flash_prefill_128",
    "flash_prefill_128_tail",
    "flash_prefill_256",
    "flash_prefill_256_tail",
    "mv_q4k_2",
    "mv_q6k_2",
    "mv_q4_0_4",
    "routed_mv_q4k_2",
    "routed_mv_q6k_2",
    "routed_mv_q4_0_4",
    "q4_0_vector",
    "mv_f16_4",
    "routed_mv_f16_4",
    "gemm_q4k",
    "gemm_q6k",
    "gemm_q4_0",
    "add_rms",
    "qk_rope_cache",
    "attention_decode_64",
    "attention_decode_96",
    "attention_decode_128",
    "attention_decode_256",
    "attention_decode_512",
    "mv_f32_4",
    "routed_mv_f32_4",
    "route_topk",
    "gate_up_mv_q4k_2",
    "gate_up_mv_q6k_2",
    "gate_up_mv_q4_0_4",
    "gate_up_mv_f16_4",
    "gate_up_mv_f32_4",
    "routed_gate_up_mv_q4k_2",
    "routed_gate_up_mv_q6k_2",
    "routed_gate_up_mv_q4_0_4",
    "routed_gate_up_mv_f16_4",
    "routed_gate_up_mv_f32_4",
];

/// The vector kernel shared by a gate and an up weight of the same shape.
fn paired_kernel(gate: &Weight, up: &Weight) -> Option<(&'static str, usize)> {
    let same = gate.t.kind == up.t.kind && gate.t.stride == up.t.stride && gate.t.dims == up.t.dims;
    vector_kernel(gate).filter(|_| same)
}

impl Gpu {
    /// Encode every dispatch of `body` into one serial compute encoder: the
    /// same ordering as one encoder per dispatch, without the per-encoder cost.
    /// Kernel profiling keeps one encoder per dispatch for per-kernel timings.
    pub(crate) fn serial_scope<T>(
        &self,
        cmd: &Command,
        body: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if cmd.profile.borrow().is_some() {
            return body();
        }
        if cmd.independent_encoder.borrow().is_some() {
            return Err(invalid("nested encoder scope"));
        }
        let encoder = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or_else(|| io::Error::other("Metal compute encoder unavailable"))?;
        *cmd.independent_encoder.borrow_mut() = Some(encoder);
        struct EndScope<'a>(&'a Command);
        impl Drop for EndScope<'_> {
            fn drop(&mut self) {
                if let Some(encoder) = self.0.independent_encoder.borrow_mut().take() {
                    encoder.endEncoding();
                }
            }
        }
        let _end = EndScope(cmd);
        body()
    }
    /// `rows` rows of `a + b` (or `a`) and their RMS norm scaled by `w`:
    /// returns (sum, normed).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_rms(
        &self,
        cmd: &Command,
        a: &Buffer,
        b: Option<&Buffer>,
        w: &Weight,
        width: usize,
        rows: usize,
        eps: f32,
    ) -> Result<(Buffer, Buffer)> {
        if !width.is_multiple_of(4) || w.t.bytes != 4 * width {
            return Err(invalid(
                "RMS norm width must be a multiple of 4 F32 weights",
            ));
        }
        let y = self.output(width * rows)?;
        let s = if b.is_some() {
            self.output(width * rows)?
        } else {
            a.clone()
        };
        let params = [width as u32, eps.to_bits(), b.is_some() as u32];
        self.dispatch(
            cmd,
            "add_rms",
            &[a, b.unwrap_or(a), &w.buffer, &s, &y],
            &params,
            [rows, 1],
            256,
        )?;
        Ok((s, y))
    }
    /// Dispatch `groups` threadgroups of `threads` (32, 64, 128 or 256) threads.
    pub(crate) fn dispatch(
        &self,
        cmd: &Command,
        name: &str,
        buffers: &[&Buffer],
        params: &[u32],
        groups: [usize; 2],
        threads: usize,
    ) -> Result<()> {
        // `encode` maps its row tile to the threadgroup width.
        let tile = match threads {
            32 => 1,
            64 => 8,
            128 => 4,
            256 => 32,
            _ => return Err(invalid("unsupported threadgroup size")),
        };
        let pipeline = self
            .operations
            .get(name)
            .ok_or_else(|| invalid(format!("unknown Metal operation {name}")))?;
        let groups = MTLSize {
            width: groups[0] * tile,
            height: groups[1],
            depth: 1,
        };
        self.encode(cmd, pipeline, buffers, params, groups, tile)
    }
}

/// Tokens per prefill GEMM tile (GEMM_TOKENS in kernels/generic.metal).
const GEMM_TOKENS: usize = 64;
/// SIMD groups per threadgroup for the matrix-vector kernels.
const VECTOR_SIMD_GROUPS: usize = 2;
/// Matrix-vector kernel and rows per threadgroup for a weight.
fn vector_kernel(w: &Weight) -> Option<(&'static str, usize)> {
    match w.t.kind {
        12 => Some(("mv_q4k_2", 2 * VECTOR_SIMD_GROUPS)),
        14 => Some(("mv_q6k_2", 2 * VECTOR_SIMD_GROUPS)),
        2 => Some(("mv_q4_0_4", 4 * VECTOR_SIMD_GROUPS)),
        1 if w.t.dims[0].is_multiple_of(4) => Some(("mv_f16_4", 4 * VECTOR_SIMD_GROUPS)),
        0 if w.t.dims[0].is_multiple_of(4) => Some(("mv_f32_4", 4 * VECTOR_SIMD_GROUPS)),
        _ => None,
    }
}

impl Gpu {
    /// `w` times `count` rows of `x`; None for types without a kernel.
    pub(crate) fn generic_project(
        &self,
        cmd: &Command,
        w: &Weight,
        x: &Buffer,
        count: usize,
    ) -> Result<Option<Buffer>> {
        if count > 4 {
            let name = match w.t.kind {
                12 => "gemm_q4k",
                14 => "gemm_q6k",
                2 => "gemm_q4_0",
                _ => return Ok(None),
            };
            let rows = w.t.dims[1];
            let y = self.output(rows * count)?;
            self.dispatch(
                cmd,
                name,
                &[&w.buffer, x, &y],
                &[
                    w.t.dims[0] as u32,
                    rows as u32,
                    w.t.stride as u32,
                    w.t.kind,
                    0,
                    count as u32,
                ],
                [count.div_ceil(GEMM_TOKENS), rows.div_ceil(64)],
                128,
            )?;
            return Ok(Some(y));
        }
        let Some((name, rows_per_group)) = vector_kernel(w) else {
            return Ok(None);
        };
        let rows = w.t.dims[1];
        let y = self.output(rows * count)?;
        self.dispatch(
            cmd,
            name,
            &[&w.buffer, x, &y],
            &[
                w.t.dims[0] as u32,
                rows as u32,
                w.t.stride as u32,
                w.t.kind,
                0,
            ],
            [rows.div_ceil(rows_per_group), count],
            32 * VECTOR_SIMD_GROUPS,
        )?;
        Ok(Some(y))
    }
    /// silu(gate x) * up x (gelu with `gelu`) for one to four rows of `x`;
    /// None when the two weights differ in type or shape or have no kernel.
    pub(crate) fn generic_gate_up(
        &self,
        cmd: &Command,
        [gate, up]: [&Weight; 2],
        x: &Buffer,
        count: usize,
        gelu: bool,
    ) -> Result<Option<Buffer>> {
        let Some((name, rows_per_group)) = paired_kernel(gate, up) else {
            return Ok(None);
        };
        let rows = gate.t.dims[1];
        let y = self.output(rows * count)?;
        let t = &gate.t;
        self.dispatch(
            cmd,
            &format!("gate_up_{name}"),
            &[&gate.buffer, &up.buffer, x, &y],
            &[
                t.dims[0] as u32,
                rows as u32,
                t.stride as u32,
                t.kind,
                0,
                gelu as u32,
            ],
            [rows.div_ceil(rows_per_group), count],
            32 * VECTOR_SIMD_GROUPS,
        )?;
        Ok(Some(y))
    }
    /// Routed silu(gate x) * up x of `pairs` (per-token input, expert ids);
    /// None when no fused kernel applies.
    pub(crate) fn generic_routed_gate_up(
        &self,
        cmd: &Command,
        [gate, up]: [&Weight; 2],
        [x, ids]: [&Buffer; 2],
        top: usize,
        pairs: usize,
    ) -> Result<Option<Buffer>> {
        let Some((name, rows_per_group)) = paired_kernel(gate, up) else {
            return Ok(None);
        };
        let t = &gate.t;
        let y = self.output(t.dims[1] * pairs)?;
        let params = [
            t.dims[0] as u32,
            t.dims[1] as u32,
            t.stride as u32,
            t.kind,
            0,
            0,
            top as u32,
        ];
        self.dispatch(
            cmd,
            &format!("routed_gate_up_{name}"),
            &[&gate.buffer, &up.buffer, x, ids, &y],
            &params,
            [t.dims[1].div_ceil(rows_per_group), pairs],
            32 * VECTOR_SIMD_GROUPS,
        )?;
        Ok(Some(y))
    }
    /// Routed expert projection of `pairs` (input, expert ids, output) into
    /// `y`; false for types without a kernel.
    pub(crate) fn generic_routed(
        &self,
        cmd: &Command,
        w: &Weight,
        [x, ids, y]: [&Buffer; 3],
        per_pair: bool,
        top: usize,
        pairs: usize,
    ) -> Result<bool> {
        let Some((name, rows_per_group)) = vector_kernel(w) else {
            return Ok(false);
        };
        let rows = w.t.dims[1];
        let params = [
            w.t.dims[0] as u32,
            rows as u32,
            w.t.stride as u32,
            w.t.kind,
            0,
            per_pair as u32,
            top as u32,
        ];
        self.dispatch(
            cmd,
            &format!("routed_{name}"),
            &[&w.buffer, x, ids, y],
            &params,
            [rows.div_ceil(rows_per_group), pairs],
            32 * VECTOR_SIMD_GROUPS,
        )?;
        Ok(true)
    }
}
