//! Small-batch BF16 HC blocks and the count-one residual add fused with the
//! next block's RMS norm.
use super::{Hc, Qwen};
use crate::{
    Result,
    metal::{Buffer, Command},
};

impl Qwen {
    /// Shapes and formats served by the fused BF16 HC kernels.
    pub(super) fn hc_fusable(&self, hc: &Hc, count: usize) -> bool {
        let c = &self.c;
        count <= 4
            && c.width == 2560
            && c.hc == 4
            && c.lr == 320
            && hc.down.t.kind == 30
            && hc.up.t.kind == 30
            && hc.inject.as_ref().is_none_or(|w| w.t.kind == 30)
    }

    /// Down/inject and up/collapse projections of an already normalized HC input.
    pub(super) fn hc_project(
        &self,
        cmd: &Command,
        hc: &Hc,
        norm: &Buffer,
        count: usize,
    ) -> Result<(Buffer, Option<Buffer>)> {
        let c = &self.c;
        let wide = c.width * c.hc;
        let lo = self.gpu.output(c.lr * count)?;
        let inject = hc
            .inject
            .as_ref()
            .map(|_| self.gpu.output(c.hc * count))
            .transpose()?;
        let iw = hc.inject.as_ref().unwrap_or(&hc.down);
        // Every small batch uses the scalar four-SIMD reduction per down row.
        let down_groups = c.lr;
        self.gpu.op(
            cmd,
            "hc_down_inject",
            &[
                &hc.down.buffer,
                &iw.buffer,
                norm,
                &lo,
                inject.as_ref().unwrap_or(&lo),
            ],
            &[
                wide as u32,
                c.lr as u32,
                hc.down.t.stride as u32,
                iw.t.stride as u32,
                if inject.is_some() { c.hc as u32 } else { 0 },
                count as u32,
                (1.0 / c.hc as f32).to_bits(),
            ],
            down_groups + if inject.is_some() { c.hc } else { 0 },
            count,
        )?;
        let out = self.gpu.output(c.width * count)?;
        // Keep the count-one F32 dot order for every token, with up and collapse fused.
        self.gpu.op(
            cmd,
            "hc_up_collapse",
            &[&hc.up.buffer, &lo, norm, &out],
            &[
                c.width as u32,
                c.hc as u32,
                c.lr as u32,
                hc.up.t.stride as u32,
                count as u32,
            ],
            c.width.div_ceil(4),
            count,
        )?;
        Ok((out, inject))
    }

    /// `hc_add` then `hc_mix(next, ..)` for one token, returning the new residual
    /// with the next block's mixed input and injection gates.
    pub(super) fn hc_add_mix(
        &self,
        cmd: &Command,
        x: &Buffer,
        b: &Buffer,
        g: &Buffer,
        next: &Hc,
    ) -> Result<(Buffer, Buffer, Option<Buffer>)> {
        if !self.hc_fusable(next, 1) {
            let y = self.hc_add(cmd, x, b, g)?;
            let (mixed, inject) = self.hc_mix(cmd, next, &y)?;
            return Ok((y, mixed, inject));
        }
        let c = &self.c;
        let y = self.gpu.output(c.width * c.hc)?;
        let norm = self.gpu.output(c.width * c.hc)?;
        self.gpu.op(
            cmd,
            "hc_add_rms",
            &[x, b, g, &next.norm.buffer, &y, &norm],
            &[
                c.width as u32,
                c.hc as u32,
                (next.norm.t.bytes / 4) as u32,
                self.c.eps.to_bits(),
            ],
            c.hc.div_ceil(2),
            1,
        )?;
        let (mixed, inject) = self.hc_project(cmd, next, &norm, 1)?;
        Ok((y, mixed, inject))
    }
}
