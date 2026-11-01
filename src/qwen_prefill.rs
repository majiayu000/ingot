//! Packed Qwen prefill. Bounded matrices; recurrent transitions remain causal.
use super::*;
/// Tokens per DeltaNet prefill dispatch (about 3 ms of recurrence per threadgroup).
const DELTA_SEGMENT: usize = 512;

impl Qwen {
    // A single allocation holds every rollback position for this mutable buffer.
    // The recurrence kernels write each position directly, without a replay/copy pass.
    fn recurrence_checkpoints(
        &self,
        buffer: &Buffer,
        snapshots: &mut [mtp::Checkpoint],
    ) -> Result<Buffer> {
        if snapshots.is_empty() {
            return Ok(buffer.clone());
        }
        let length = buffer.length() / 4;
        let saved = self.gpu.output(length * snapshots.len())?;
        for (i, checkpoint) in snapshots.iter_mut().enumerate() {
            checkpoint
                .mutable
                .push((buffer.clone(), saved.clone(), i * length));
            checkpoint.bytes += buffer.length();
        }
        Ok(saved)
    }
    #[allow(clippy::too_many_arguments)] // Explicit tensor bindings, geometry and rollback destinations.
    /// Resident experts: the GPU groups routed pairs by expert and lays out the
    /// tiles (kernels/prefill.metal), so no host wait separates routing and GEMMs.
    fn resident_ffn_packed(
        &self,
        cmd: &Command,
        f: &Ffn,
        w: &[Weight; 3],
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
        let geometry = |t: &Tensor| {
            [
                t.dims[0] as u32,
                t.dims[1] as u32,
                t.stride as u32,
                t.kind,
                c.top as u32,
            ]
        };
        // Expert outputs stay in packed expert order; combine_packed maps them back.
        let project = |name: &str, w: &Weight, x: &Buffer| -> Result<Buffer> {
            let t = &w.t;
            let out = self.gpu.output(t.dims[1] * total)?;
            self.gpu.op(
                cmd,
                name,
                &[&w.buffer, x, &tiles, &out, &pairs],
                &geometry(t),
                t.dims[1],
                blocks,
            )?;
            Ok(out)
        };
        let (gt, ut) = (&w[0].t, &w[1].t);
        let act = if gt.kind == ut.kind && gt.dims == ut.dims && gt.stride == ut.stride {
            let act = self.gpu.output(gt.dims[1] * total)?;
            self.gpu.op(
                cmd,
                "quant_expert_gate_up",
                &[&w[0].buffer, &w[1].buffer, x, &tiles, &act, &pairs],
                &geometry(gt),
                gt.dims[1],
                blocks,
            )?;
            act
        } else {
            let g = project("quant_expert_gather", &w[0], x)?;
            let u = project("quant_expert_gather", &w[1], x)?;
            self.gpu.activation(cmd, &g, &u, c.ff * total)?
        };
        let routed = project("quant_expert_packed", &w[2], &act)?;
        let sg = self.matrix(cmd, &f.sgate, x, count)?;
        let su = self.matrix(cmd, &f.sup, x, count)?;
        let act = self.gpu.activation(cmd, &sg, &su, c.sff * count)?;
        let shared = self.matrix(cmd, &f.sdown, &act, count)?;
        let gate = self.matrix(cmd, &f.shared, x, count)?;
        let out = self.gpu.output(c.width * count)?;
        self.point(
            cmd,
            "combine_packed",
            &[&routed, &shared, weights, &gate, &inverse, &out],
            &[c.width as u32, c.top as u32, count as u32],
            c.width * count,
        )?;
        Ok(out)
    }

    fn matrix(&self, cmd: &Command, w: &Weight, x: &Buffer, count: usize) -> Result<Buffer> {
        let cols = w.t.dims[0];
        let rows = w.t.dims.get(1).copied().unwrap_or(1);
        if count >= 4096
            && matches!(w.t.kind, 23 | 30)
            && matches!(cols, 2560 | 6144)
            && (512..=12288).contains(&rows)
            && cols * rows <= 2560 * 12288
        {
            return self
                .gpu
                .dense_mps(cmd, &w.buffer, cols, rows, w.t.stride, w.t.kind, x, count);
        }
        self.gpu.matmul(
            cmd,
            &w.buffer,
            w.t.dims[0],
            w.t.dims.get(1).copied().unwrap_or(1),
            w.t.stride,
            w.t.kind,
            0,
            x,
            count,
        )
    }
    pub(super) fn hc_packed(
        &self,
        cmd: &Command,
        hc: &Hc,
        x: &Buffer,
        count: usize,
    ) -> Result<(Buffer, Option<Buffer>)> {
        if let Some(result) = self.hc_fused(cmd, hc, x, count)? {
            return Ok(result);
        }
        let c = &self.c;
        let wide = c.width * c.hc;
        let norm = self.norm(cmd, x, &hc.norm, c.width, c.hc * count, 0, c.width, false)?;
        let lo = self.matrix(cmd, &hc.down, &norm, count)?;
        let lo = self.unary(cmd, &lo, c.lr * count, 1, 1.0 / c.hc as f32)?;
        let gates = self.matrix(cmd, &hc.up, &lo, count)?;
        let gates = self.unary(cmd, &gates, wide * count, 0, 1.0)?;
        let out = self.gpu.output(c.width * count)?;
        self.point(
            cmd,
            "hc_collapse_batch",
            &[&norm, &gates, &out],
            &[c.width as u32, c.hc as u32, count as u32],
            c.width * count,
        )?;
        let inject = hc
            .inject
            .as_ref()
            .map(|w| self.matrix(cmd, w, &norm, count))
            .transpose()?;
        Ok((out, inject))
    }
    fn hc_add_packed(
        &self,
        cmd: &Command,
        h: &Buffer,
        x: &Buffer,
        inject: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let c = &self.c;
        let n = c.width * c.hc * count;
        let out = self.gpu.output(n)?;
        self.point(
            cmd,
            "hc_add_batch",
            &[h, x, inject, &out],
            &[c.width as u32, c.hc as u32, count as u32],
            n,
        )?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)] // Explicit tensor bindings, geometry and rollback destinations.
    fn convolution_packed(
        &self,
        cmd: &Command,
        x: &Buffer,
        w: &Weight,
        history: &Buffer,
        channels: usize,
        dilation: usize,
        count: usize,
        snapshots: &mut [mtp::Checkpoint],
    ) -> Result<Buffer> {
        let out = self.gpu.output(channels * count)?;
        let saved = self.recurrence_checkpoints(history, snapshots)?;
        self.point(
            cmd,
            if count >= 16 {
                "convolution_parallel"
            } else {
                "convolution_batch"
            },
            &[x, &w.buffer, history, &out, &saved],
            &[
                channels as u32,
                w.t.dims[0] as u32,
                dilation as u32,
                w.t.kind,
                w.t.stride as u32,
                count as u32,
                snapshots.len() as u32,
            ],
            if count >= 16 {
                channels * count
            } else {
                channels
            },
        )?;
        if count >= 16 {
            self.point(
                cmd,
                "convolution_history",
                &[x, history],
                &[
                    channels as u32,
                    w.t.dims[0] as u32,
                    dilation as u32,
                    count as u32,
                ],
                channels,
            )?;
        }
        Ok(out)
    }
    fn linear_packed(
        &self,
        cmd: &Command,
        l: &Linear,
        state: &MixerState,
        x: &Buffer,
        count: usize,
        snapshots: &mut [mtp::Checkpoint],
    ) -> Result<Buffer> {
        let c = &self.c;
        let channels = (2 * c.nk + c.nv) * c.d;
        let MixerState::Linear { history, state } = state else {
            return Err(invalid("GDN state mismatch"));
        };
        let qkv = self.matrix(cmd, &l.qkv, x, count)?;
        let z = self.matrix(cmd, &l.z, x, count)?;
        let a = self.matrix(cmd, &l.alpha, x, count)?;
        let b = self.matrix(cmd, &l.beta, x, count)?;
        let conv =
            self.convolution_packed(cmd, &qkv, &l.conv, history, channels, 1, count, snapshots)?;
        let q = self.gpu.output(c.d * c.nk * count)?;
        let k = self.gpu.output(c.d * c.nk * count)?;
        for (out, offset) in [(&q, 0), (&k, c.nk * c.d)] {
            self.gpu.op(
                cmd,
                "rms_tokens",
                &[&conv, &l.norm.buffer, out],
                &[
                    c.d as u32,
                    (l.norm.t.bytes / 4) as u32,
                    offset as u32,
                    c.d as u32,
                    channels as u32,
                    c.nk as u32,
                    c.eps.to_bits(),
                    1,
                ],
                c.nk * count,
                1,
            )?;
        }
        let gates = self.gpu.output(c.nv * count * 2)?;
        self.point(
            cmd,
            "delta_gates_batch",
            &[&a, &b, &l.dt.buffer, &l.a.buffer, &gates],
            &[c.nv as u32, count as u32],
            c.nv * count,
        )?;
        let delta = self.gpu.output(c.nv * c.d * count)?;
        let saved = self.recurrence_checkpoints(state, snapshots)?;
        // Long blocks run the recurrence in segments that carry the F32 state through
        // memory, keeping each threadgroup short enough for the GPU to be preempted
        // (one 4K-token threadgroup can trip macOS's interactivity watchdog).
        let segment = if snapshots.is_empty() {
            DELTA_SEGMENT
        } else {
            count
        };
        for t0 in (0..count).step_by(segment) {
            let n = segment.min(count - t0);
            let at = |b: &Buffer, width: usize| b.view_f32(t0 * width, n * width);
            self.gpu.op(
                cmd,
                "delta_net_batch",
                &[
                    &at(&q, c.nk * c.d)?,
                    &at(&k, c.nk * c.d)?,
                    &at(&conv, channels)?,
                    &at(&gates, c.nv * 2)?,
                    state,
                    &at(&delta, c.nv * c.d)?,
                    &saved,
                ],
                &[
                    c.d as u32,
                    c.nk as u32,
                    c.nv as u32,
                    n as u32,
                    channels as u32,
                    snapshots.len() as u32,
                ],
                c.d / 4,
                c.nv,
            )?;
        }
        let norm = self.norm(cmd, &delta, &l.norm, c.d, c.nv * count, 0, c.d, false)?;
        let gated = self.gated(cmd, &norm, &z, c.nv * c.d * count)?;
        self.matrix(cmd, &l.out, &gated, count)
    }
    #[allow(clippy::too_many_arguments)] // Tensor bindings and the causal query geometry.
    pub(super) fn compact_attention(
        &self,
        cmd: &Command,
        q: &Buffer,
        key: &Buffer,
        value: &Buffer,
        mask: &Buffer,
        position: usize,
        count: usize,
        ratio: usize,
    ) -> Result<Buffer> {
        let c = &self.c;
        let length = position + count;
        if count >= 8 {
            return self.matrix_attention(cmd, q, key, value, mask, position, count, true);
        }
        let capacity = length.min(c.topk + ratio - 1);
        let indices = self.gpu.output(capacity * count)?;
        let offsets = self.gpu.output(33 * count)?;
        self.gpu.op(
            cmd,
            "compact_mask_ordered",
            &[mask, &indices, &offsets],
            &[length as u32, position as u32, capacity as u32],
            count,
            1,
        )?;
        self.online_attention(
            cmd,
            q,
            key,
            value,
            position,
            count,
            capacity,
            Some((&indices, &offsets)),
        )
    }
    #[allow(clippy::too_many_arguments)] // Exact QSA mask and cache geometry.
    fn matrix_attention(
        &self,
        cmd: &Command,
        q: &Buffer,
        key: &Buffer,
        value: &Buffer,
        mask: &Buffer,
        position: usize,
        count: usize,
        masked: bool,
    ) -> Result<Buffer> {
        let c = &self.c;
        let out = self.gpu.output(c.heads * c.head * count)?;
        let length = position + count;
        let full = (length / 64 * 64).saturating_sub(position).min(count);
        let width = c.heads * c.head;
        for (begin, rows, tail) in [(0, full, false), (full, count - full, true)] {
            if rows == 0 {
                continue;
            }
            let query = q.view_f32(begin * width, rows * width)?;
            let output = out.view_f32(begin * width, rows * width)?;
            let selected = if masked {
                mask.view_f32(begin * length, rows * length)?
            } else {
                q.clone()
            };
            self.gpu.op(
                cmd,
                if tail {
                    "attention_prefill_tail"
                } else {
                    "attention_prefill_tiles"
                },
                &[&query, key, value, &selected, &output],
                &[
                    c.head as u32,
                    c.heads as u32,
                    c.kv as u32,
                    length as u32,
                    (position + begin) as u32,
                    rows as u32,
                    u32::from(masked),
                ],
                rows.div_ceil(8) * 4,
                c.heads,
            )?;
        }
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)] // Tensor geometry and optional selected positions.
    fn online_attention(
        &self,
        cmd: &Command,
        q: &Buffer,
        key: &Buffer,
        value: &Buffer,
        position: usize,
        count: usize,
        capacity: usize,
        selected: Option<(&Buffer, &Buffer)>,
    ) -> Result<Buffer> {
        let c = &self.c;
        let parts = capacity.div_ceil(64);
        let compact = u32::from(selected.is_some());
        // Dense queries never dereference the index bindings; reuse a live buffer.
        let (indices, offsets) = selected.unwrap_or((q, q));
        let partial = self.gpu.output(c.heads * count * parts * (c.head + 2))?;
        self.gpu.op(
            cmd,
            "attention_online_parts",
            &[q, key, value, indices, offsets, &partial],
            &[
                c.head as u32,
                c.heads as u32,
                c.kv as u32,
                capacity as u32,
                position as u32,
                compact,
                parts as u32,
            ],
            parts,
            c.heads * count,
        )?;
        let out = self.gpu.output(c.heads * c.head * count)?;
        self.gpu.op(
            cmd,
            "attention_online_merge",
            &[&partial, &out],
            &[parts as u32],
            c.heads * count,
            1,
        )?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)] // The model's RoPE dimensions and position step.
    fn rope_packed(
        &self,
        cmd: &Command,
        x: &Buffer,
        dim: usize,
        heads: usize,
        position: usize,
        count: usize,
        step: usize,
    ) -> Result<Buffer> {
        let (position, step) = match self.rope_plan(position, count, step)? {
            super::media::Rope::Linear(position, step) => (position, step),
            super::media::Rope::Rows(rows) => {
                return self.rope_rows(cmd, x, dim, heads, &rows, count);
            }
        };
        let out = self.gpu.output(dim * heads * count)?;
        self.point(
            cmd,
            "rope_batch",
            &[x, &out],
            &[
                dim as u32,
                heads as u32,
                self.c.rot as u32,
                position as u32,
                self.c.rope.to_bits(),
                count as u32,
                step as u32,
            ],
            dim * heads * count,
        )?;
        Ok(out)
    }
    pub(super) fn dense_attention(
        &self,
        cmd: &Command,
        q: &Buffer,
        key: &Buffer,
        value: &Buffer,
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        if count >= 8 {
            return self.matrix_attention(cmd, q, key, value, q, position, count, false);
        }
        self.online_attention(cmd, q, key, value, position, count, position + count, None)
    }
    pub(super) fn attention_cache_packed(
        &self,
        cmd: &Command,
        a: &Attention,
        state: &MixerState,
        x: &Buffer,
        position: usize,
        count: usize,
    ) -> Result<()> {
        let kr = self.matrix(cmd, &a.k, x, count)?;
        let v = self.matrix(cmd, &a.v, x, count)?;
        let ik = self.matrix(cmd, &a.ik, x, count)?;

        self.write_attention_cache_packed(cmd, a, state, [&kr, &v, &ik], position, count)
    }
    fn write_attention_cache_packed(
        &self,
        cmd: &Command,
        a: &Attention,
        state: &MixerState,
        projected: [&Buffer; 3],
        position: usize,
        count: usize,
    ) -> Result<()> {
        let c = &self.c;
        let MixerState::Attention {
            key,
            value,
            index,
            pooled,
        } = state
        else {
            return Err(invalid("QSA state mismatch"));
        };
        let length = position + count;
        let [kr, v, ik] = projected;
        let k = self.norm(cmd, kr, &a.kn, c.head, c.kv * count, 0, c.head, false)?;
        let k = self.rope_packed(cmd, &k, c.head, c.kv, position, count, 1)?;
        for (source, destination, width) in [
            (&k, key, c.head * c.kv),
            (v, value, c.head * c.kv),
            (ik, index, c.idx),
        ] {
            self.point(
                cmd,
                "copy_into",
                &[source, destination],
                &[(width * count) as u32, (position * width) as u32],
                width * count,
            )?;
        }
        let first = position / a.ratio;
        let blocks = length / a.ratio - first;
        if blocks > 0 {
            let raw = self.gpu.output(c.idx * blocks)?;
            self.point(
                cmd,
                "pool_index_batch",
                &[index, &raw],
                &[c.idx as u32, a.ratio as u32, first as u32, blocks as u32],
                c.idx * blocks,
            )?;
            let norm = self.norm(cmd, &raw, &a.ikn, c.idx, blocks, 0, c.idx, false)?;
            let rotated =
                self.rope_packed(cmd, &norm, c.idx, 1, first * a.ratio, blocks, a.ratio)?;
            self.point(
                cmd,
                "copy_into",
                &[&rotated, pooled],
                &[(c.idx * blocks) as u32, (first * c.idx) as u32],
                c.idx * blocks,
            )?;
        }
        Ok(())
    }
    fn attention_packed(
        &self,
        cmd: &Command,
        a: &Attention,
        state: &MixerState,
        x: &Buffer,
        position: usize,
        count: usize,
    ) -> Result<Buffer> {
        let c = &self.c;
        let MixerState::Attention {
            key,
            value,
            index: _,
            pooled,
        } = state
        else {
            return Err(invalid("QSA state mismatch"));
        };
        let length = position + count;
        let qr = self.matrix(cmd, &a.q, x, count)?;
        let kr = self.matrix(cmd, &a.k, x, count)?;
        let v = self.matrix(cmd, &a.v, x, count)?;
        let ik = self.matrix(cmd, &a.ik, x, count)?;

        let q = self.norm(
            cmd,
            &qr,
            &a.qn,
            c.head,
            c.heads * count,
            0,
            2 * c.head,
            false,
        )?;
        let q = self.rope_packed(cmd, &q, c.head, c.heads, position, count, 1)?;
        let gate = self.gpu.output(c.head * c.heads * count)?;
        self.point(
            cmd,
            "extract_gate_batch",
            &[&qr, &gate],
            &[c.head as u32, c.heads as u32, count as u32],
            c.head * c.heads * count,
        )?;
        self.write_attention_cache_packed(cmd, a, state, [&kr, &v, &ik], position, count)?;
        let out = if length < c.topk + a.ratio {
            self.dense_attention(cmd, &q, key, value, position, count)?
        } else {
            let iq = self.matrix(cmd, &a.iq, x, count)?;
            let iq = self.norm(cmd, &iq, &a.iqn, c.idx, c.ih * count, 0, c.idx, false)?;
            let iq = self.rope_packed(cmd, &iq, c.idx, c.ih, position, count, 1)?;
            let max_blocks = (length / a.ratio).max(1);
            let score = self.gpu.output(max_blocks * count)?;
            self.gpu.op(
                cmd,
                "index_score_batch",
                &[&iq, pooled, &score],
                &[
                    c.idx as u32,
                    c.ih as u32,
                    max_blocks as u32,
                    position as u32,
                    a.ratio as u32,
                ],
                max_blocks,
                count,
            )?;
            let mask = self.gpu.output(length * count)?;
            let params = [
                length as u32,
                a.ratio as u32,
                c.topk as u32,
                position as u32,
                count as u32,
                max_blocks as u32,
            ];
            if max_blocks <= 4096 {
                self.gpu.op(
                    cmd,
                    "index_mask_sorted",
                    &[&score, &mask],
                    &params,
                    count,
                    1,
                )?;
            } else {
                self.point(
                    cmd,
                    "index_mask_batch",
                    &[&score, &mask],
                    &params,
                    max_blocks * count,
                )?;
            }
            self.compact_attention(cmd, &q, key, value, &mask, position, count, a.ratio)?
        };
        let gated = self.gated(cmd, &out, &gate, c.head * c.heads * count)?;
        self.matrix(cmd, &a.out, &gated, count)
    }
    #[allow(clippy::too_many_arguments)] // Explicit tensor bindings, geometry and rollback destinations.
    fn ple_packed(
        &self,
        cmd: &Command,
        p: &Ple,
        history: &Buffer,
        emb: &Buffer,
        x: &Buffer,
        count: usize,
        snapshots: &mut [mtp::Checkpoint],
    ) -> Result<Buffer> {
        let c = &self.c;
        let n = c.width * c.hc * count;
        let k = self.matrix(cmd, &p.key, emb, count)?;
        let v = self.matrix(cmd, &p.value, emb, count)?;
        let k = self.norm(cmd, &k, &p.kn, c.width, c.hc * count, 0, c.width, false)?;
        let q = self.norm(cmd, x, &p.qn, c.width, c.hc * count, 0, c.width, false)?;
        let gated = self.gpu.output(n)?;
        self.gpu.op(
            cmd,
            "ple_gate_batch",
            &[&k, &q, &v, &gated],
            &[c.width as u32, c.hc as u32],
            c.hc * count,
            1,
        )?;
        let norm = self.norm(cmd, &gated, &p.cn, c.width, c.hc * count, 0, c.width, false)?;
        let conv = self.convolution_packed(
            cmd,
            &norm,
            &p.conv,
            history,
            c.width * c.hc,
            c.ngram,
            count,
            snapshots,
        )?;
        let out = self.gpu.output(n)?;
        self.point(cmd, "add_three", &[x, &gated, &conv, &out], &[n as u32], n)?;
        Ok(out)
    }
    fn ffn_packed(
        &self,
        cmd: Command,
        f: &Ffn,
        x: &Buffer,
        count: usize,
        cancel: &crate::engine::Cancellation,
    ) -> Result<(Buffer, Command)> {
        let c = &self.c;
        let scores = self.matrix(&cmd, &f.router, x, count)?;
        let ids = self.gpu.output(c.top * count)?;
        let weights = self.gpu.output(c.top * count)?;
        self.gpu.op(
            &cmd,
            "route_gpu_batch",
            &[&scores, &ids, &weights],
            &[c.experts as u32, c.top as u32],
            count,
            1,
        )?;
        if count <= 4
            && let ExpertWeights::Resident(w) = &f.experts
        {
            // Small verification blocks keep routing IDs on the GPU. Grouping
            // experts on the CPU would introduce a wait at every layer.
            let project = |w: &Weight, input: &Buffer, per: bool| -> Result<Buffer> {
                let t = &w.t;
                let out = self.gpu.output(t.dims[1] * c.top * count)?;
                self.gpu.op(
                    &cmd,
                    "project_routed_batch",
                    &[&w.buffer, input, &ids, &out],
                    &[
                        t.dims[0] as u32,
                        t.dims[1] as u32,
                        t.stride as u32,
                        t.kind,
                        c.top as u32,
                        u32::from(per),
                    ],
                    t.dims[1],
                    c.top * count,
                )?;
                Ok(out)
            };
            let act = if super::fused_gate_up(&w[0].t, &w[1].t) {
                let act = self.gpu.output(w[0].t.dims[1] * c.top * count)?;
                self.gpu.op(
                    &cmd,
                    "routed_gate_up_iq2s",
                    &[&w[0].buffer, &w[1].buffer, x, &ids, &act],
                    &[
                        w[0].t.dims[0] as u32,
                        w[0].t.dims[1] as u32,
                        w[0].t.stride as u32,
                        w[0].t.kind,
                        c.top as u32,
                        0,
                    ],
                    w[0].t.dims[1],
                    c.top * count,
                )?;
                act
            } else {
                let g = project(&w[0], x, false)?;
                let u = project(&w[1], x, false)?;
                self.gpu.activation(&cmd, &g, &u, c.ff * c.top * count)?
            };
            let routed = project(&w[2], &act, true)?;
            let sg = self.matrix(&cmd, &f.sgate, x, count)?;
            let su = self.matrix(&cmd, &f.sup, x, count)?;
            let act = self.gpu.activation(&cmd, &sg, &su, c.sff * count)?;
            let shared = self.matrix(&cmd, &f.sdown, &act, count)?;
            let gate = self.matrix(&cmd, &f.shared, x, count)?;
            let out = self.gpu.output(c.width * count)?;
            self.point(
                &cmd,
                "combine_batch",
                &[&routed, &shared, &weights, &gate, &out],
                &[c.width as u32, c.top as u32, count as u32],
                c.width * count,
            )?;
            return Ok((out, cmd));
        }
        if let ExpertWeights::Resident(w) = &f.experts {
            let out = self.resident_ffn_packed(&cmd, f, w, x, &ids, &weights, count)?;
            return Ok((out, cmd));
        }
        self.gpu.complete(cmd)?;
        let mut groups = std::collections::BTreeMap::<usize, Vec<u32>>::new();
        for (pair, expert) in self
            .gpu
            .read_u32(&ids, c.top * count)?
            .into_iter()
            .enumerate()
        {
            groups.entry(expert as usize).or_default().push(pair as u32);
        }
        let groups: Vec<_> = groups.into_iter().collect();

        let routed = self.gpu.output(c.width * c.top * count)?;
        let mut cmd = self.gpu.command()?;
        let sg = self.matrix(&cmd, &f.sgate, x, count)?;
        let su = self.matrix(&cmd, &f.sup, x, count)?;
        let act = self.gpu.activation(&cmd, &sg, &su, c.sff * count)?;
        let shared = self.matrix(&cmd, &f.sdown, &act, count)?;
        let shared_gate = self.matrix(&cmd, &f.shared, x, count)?;
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
                (_, Some(c)) => (
                    [&c.buffers[0], &c.buffers[1], &c.buffers[2]],
                    [&c.tensors[0], &c.tensors[1], &c.tensors[2]],
                ),
                _ => unreachable!(),
            };
            let expert_slot = if cache.is_some() { slot } else { expert };
            let pairs = self.gpu.bytes(
                &assignments
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let input = self.gpu.output(c.width * assignments.len())?;
            self.point(
                &cmd,
                "gather_rows",
                &[x, &pairs, &input],
                &[c.width as u32, assignments.len() as u32, c.top as u32],
                c.width * assignments.len(),
            )?;
            let slot_ids = if count <= 4 && cache.is_some() {
                Some(
                    self.gpu.bytes(
                        &(0..assignments.len())
                            .flat_map(|_| (expert_slot as u32).to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?,
                )
            } else {
                None
            };
            let project = |index: usize, x: &Buffer| {
                let t = tensors[index];
                if let Some(slot_ids) = &slot_ids {
                    let out = self.gpu.output(t.dims[1] * assignments.len())?;
                    self.gpu.op(
                        &cmd,
                        "project_experts",
                        &[buffers[index], x, slot_ids, &out],
                        &[
                            t.dims[0] as u32,
                            t.dims[1] as u32,
                            t.stride as u32,
                            t.kind,
                            t.dims[0] as u32,
                        ],
                        t.dims[1],
                        assignments.len(),
                    )?;
                    Ok(out)
                } else {
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
                }
            };
            let g = project(0, &input)?;
            let u = project(1, &input)?;
            let a = self
                .gpu
                .activation(&cmd, &g, &u, c.ff * assignments.len())?;
            let y = project(2, &a)?;
            self.point(
                &cmd,
                "scatter_rows",
                &[&y, &pairs, &routed],
                &[c.width as u32, assignments.len() as u32],
                c.width * assignments.len(),
            )?;
            if let Some(cache) = cache.as_mut() {
                pinned.push(slot);
                if group + 1 < groups.len() {
                    if let Some(hit) = cache.hit(groups[group + 1].0) {
                        slot = hit;
                        continue;
                    }
                    // Fill a bounded number of distinct slots before submission.
                    // Keep at least half the cache available for an overlapping read.
                    if pinned.len() < cache.slots.len().div_ceil(2).min(8) {
                        slot = self.cache_slot(cache, groups[group + 1].0, &pinned)?;
                        continue;
                    }
                }
                self.gpu.submit(&cmd);
                let next = if group + 1 < groups.len() && cache.slots.len() > pinned.len() {
                    Some(self.cache_slot(cache, groups[group + 1].0, &pinned))
                } else {
                    None
                };
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
        let out = self.gpu.output(c.width * count)?;
        self.point(
            &cmd,
            "combine_batch",
            &[&routed, &shared, &weights, &shared_gate, &out],
            &[c.width as u32, c.top as u32, count as u32],
            c.width * count,
        )?;
        Ok((out, cmd))
    }
    pub(super) fn packed_prefill(
        &mut self,
        s: &mut State,
        history: &[u32],
        tokens: &[u32],
        cancel: &crate::engine::Cancellation,
        all: bool,
        checkpoint: bool,
    ) -> Result<Forward> {
        self.trace_active.set(false);
        let count = tokens.len();
        let mut snapshots: Vec<mtp::Checkpoint> = if checkpoint {
            (0..count - 1).map(|_| mtp::Checkpoint::default()).collect()
        } else {
            Vec::new()
        };
        let position = history.len();
        let mut history = history.to_vec();
        let mut ple_rows = Vec::new();
        for &token in tokens {
            if cancel.is_cancelled() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "generation cancelled",
                ));
            }
            if !self.c.ple_layers.is_empty() {
                ple_rows.extend(self.ple_indices(&history, token));
            }
            history.push(token);
        }
        let c = &self.c;
        let mut cmd = self.gpu.command()?;
        let ple = if ple_rows.is_empty() {
            None
        } else {
            let m = self.gpu.read_rows(
                &mut self.g.borrow_mut(),
                "per_layer_token_embd.weight",
                &ple_rows,
            )?;
            let w = self.gpu.bytes(&m.data)?;
            let y = self.gpu.output(c.width * count)?;
            self.point(
                &cmd,
                "read_rows",
                &[&w, &y],
                &[m.cols as u32, m.rows as u32, m.stride as u32, m.kind],
                c.width * count,
            )?;
            Some(y)
        };
        let ids = self.gpu.bytes(
            &tokens
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let e = self.gpu.output(c.width * count)?;
        self.point(
            &cmd,
            "embedding_batch",
            &[&self.embedding.buffer, &ids, &e],
            &[
                c.width as u32,
                self.embedding.t.stride as u32,
                self.embedding.t.kind,
                count as u32,
            ],
            c.width * count,
        )?;
        self.embed_images(&cmd, &e, position, count)?;
        let mut h = self.gpu.output(c.width * c.hc * count)?;
        self.point(
            &cmd,
            "repeat_emb_batch",
            &[&e, &h],
            &[c.width as u32, c.hc as u32, count as u32],
            c.width * c.hc * count,
        )?;
        let mut pending = std::collections::VecDeque::new();
        for (i, l) in self.layers.iter().enumerate() {
            if cancel.is_cancelled() {
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
                    h = self.ple_packed(
                        &cmd,
                        p,
                        s.layers[i]
                            .ple
                            .as_ref()
                            .ok_or_else(|| invalid("PLE state"))?,
                        ple.as_ref().ok_or_else(|| invalid("PLE embeddings"))?,
                        &h,
                        count,
                        &mut snapshots,
                    )?;
                }
                let (mixed, inject) = self.hc_packed(&cmd, &l.attn, &h, count)?;
                let out = match &l.mixer {
                    Mixer::Linear(l) => self.linear_packed(
                        &cmd,
                        l,
                        &s.layers[i].mixer,
                        &mixed,
                        count,
                        &mut snapshots,
                    )?,
                    Mixer::Attention(a) => {
                        self.attention_packed(&cmd, a, &s.layers[i].mixer, &mixed, position, count)?
                    }
                };
                h = self.hc_add_packed(
                    &cmd,
                    &h,
                    &out,
                    inject.as_ref().ok_or_else(|| invalid("HC injection"))?,
                    count,
                )?;
                let (mixed, inject) = self.hc_packed(&cmd, &l.ffn, &h, count)?;
                let (out, cmd) = self.ffn_packed(cmd, &l.experts, &mixed, count, cancel)?;
                h = self.hc_add_packed(
                    &cmd,
                    &h,
                    &out,
                    inject.as_ref().ok_or_else(|| invalid("FFN injection"))?,
                    count,
                )?;
                let cmd = if count > 4 {
                    self.gpu.complete(cmd)?;
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
        let hidden = if self.draft.is_some() || checkpoint {
            (0..count)
                .map(|i| h.view_f32(i * c.width * c.hc, c.width * c.hc))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let selected = if all {
            h
        } else {
            h.view_f32((count - 1) * c.width * c.hc, c.width * c.hc)?
        };
        let rows = if all { count } else { 1 };
        let (mixed, _) = self.hc_packed(&cmd, &self.head, &selected, rows)?;
        let output = self.matrix(&cmd, &self.output, &mixed, rows)?;
        self.gpu.complete(cmd)?;
        for cmd in pending {
            self.gpu.wait(cmd)?;
        }
        Ok(Forward {
            logits: output,
            rows,
            vocabulary: self.output.t.dims[1],
            hidden,
            snapshots,
            snapshot_encode: Duration::ZERO,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires QWEN_MODEL, QWEN_FIXTURE and Apple Silicon"]
    fn packed_matches_original_schedule_and_future_state() {
        let model = PathBuf::from(std::env::var_os("QWEN_MODEL").unwrap());
        let bytes = std::fs::read(std::env::var_os("QWEN_FIXTURE").unwrap()).unwrap();
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let mut q = Qwen::load(
            Gguf::open(&model).unwrap(),
            std::env::var("QWEN_MEMORY_GIB")
                .unwrap_or("44".into())
                .parse::<usize>()
                .unwrap()
                * (1usize << 30),
            256,
            None,
            false,
        )
        .unwrap();
        let cancel = crate::engine::Cancellation::default();
        for n in [7, 8, 17, 128] {
            let mut a = q.state(256).unwrap();
            let mut b = q.state(256).unwrap();
            // Both schedules use identical GEMM thresholds, separating packing/state
            // correctness from the separately gated matrix reduction change.
            let expected = q
                .prefill_block(&mut a, &[], &tokens[..n], &cancel, false)
                .unwrap();
            let actual = q
                .packed_prefill(&mut b, &[], &tokens[..n], &cancel, false, false)
                .unwrap();
            let error = expected.read_logits(&q.gpu).unwrap()[0]
                .iter()
                .zip(&actual.read_logits(&q.gpu).unwrap()[0])
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert_eq!(
                actual.read_logits(&q.gpu).unwrap(),
                expected.read_logits(&q.gpu).unwrap(),
                "packed n={n}, max_abs={error}"
            );
            for (i, (a, b)) in a.layers.iter().zip(&b.layers).enumerate() {
                if let (
                    MixerState::Linear {
                        history: ah,
                        state: as_,
                    },
                    MixerState::Linear {
                        history: bh,
                        state: bs,
                    },
                ) = (&a.mixer, &b.mixer)
                {
                    assert_eq!(
                        q.gpu.read_f32(ah).unwrap(),
                        q.gpu.read_f32(bh).unwrap(),
                        "history layer {i} n={n}"
                    );
                    assert_eq!(
                        q.gpu.read_f32(as_).unwrap(),
                        q.gpu.read_f32(bs).unwrap(),
                        "GDN layer {i} n={n}"
                    );
                }
                if let (Some(a), Some(b)) = (&a.ple, &b.ple) {
                    assert_eq!(
                        q.gpu.read_f32(a).unwrap(),
                        q.gpu.read_f32(b).unwrap(),
                        "PLE layer {i} n={n}"
                    );
                }
            }
            for p in n..n + 3 {
                let expected = q
                    .target_block(&mut a, &tokens[..p], &tokens[p..p + 1], &cancel, false)
                    .unwrap();
                let actual = q
                    .target_block(&mut b, &tokens[..p], &tokens[p..p + 1], &cancel, false)
                    .unwrap();
                assert_eq!(
                    actual.read_logits(&q.gpu).unwrap(),
                    expected.read_logits(&q.gpu).unwrap(),
                    "continuation p={p}"
                );
            }
            println!(
                "PASS packed schedule count={n}: logits, GDN, PLE and 3 continuation tokens bitwise exact"
            );
        }
    }
}
#[cfg(test)]
mod corpus_validation {
    use super::*;
    use std::fs::{self, OpenOptions};
    #[test]
    #[ignore = "frozen corpus validation; requires QWEN_MODEL, QWEN_TF_INDEX, QWEN_TF_OUT"]
    fn score_frozen_corpus() {
        score_corpus(44 * (1usize << 30), false);
    }

    #[test]
    #[ignore = "frozen mixed-small corpus validation; requires QWEN_MODEL, QWEN_TF_INDEX, QWEN_TF_OUT"]
    fn score_frozen_corpus_mixed_small() {
        score_corpus(64 * (1usize << 30), true);
    }

    fn score_corpus(budget: usize, mixed_small: bool) {
        let path = PathBuf::from(std::env::var_os("QWEN_MODEL").unwrap());
        let index = fs::read_to_string(std::env::var_os("QWEN_TF_INDEX").unwrap()).unwrap();
        let out = PathBuf::from(std::env::var_os("QWEN_TF_OUT").unwrap());
        fs::create_dir(&out).unwrap();
        let mut metrics = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out.join("scores.jsonl"))
            .unwrap();
        let mut q = Qwen::load(Gguf::open(&path).unwrap(), budget, 2048, None, false).unwrap();
        if mixed_small {
            println!(
                "TF_MIXED budget_bytes={budget} context=2048 prefill_block_tokens={} schedule=[1,2,3,4]",
                q.prefill_block_size()
            );
        }
        let cancel = crate::engine::Cancellation::default();
        for line in index.lines() {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(fields.len(), 4);
            let raw = fs::read(fields[2]).unwrap();
            assert_eq!(raw.len() % 4, 0);
            let tokens = raw
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let samples = fields[3]
                .split(',')
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            let mut sample_file = std::io::BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(out.join(format!("{}-{}.f32", fields[0], fields[1])))
                    .unwrap(),
            );
            let mut state = q.state(2048).unwrap();
            let mut total = 0f64;
            let start = std::time::Instant::now();
            let mut position = 0;
            let mut step = 0;
            let mut counts = [0usize; 4];
            while position < tokens.len() - 1 {
                let count = if mixed_small {
                    [1usize, 2, 3, 4][step % 4]
                } else {
                    PREFILL_BLOCK
                }
                .min(tokens.len() - 1 - position);
                let block = &tokens[position..position + count];
                let f = if mixed_small {
                    counts[count - 1] += 1;
                    q.target_block(&mut state, &tokens[..position], block, &cancel, true)
                } else {
                    q.packed_prefill(&mut state, &tokens[..position], block, &cancel, true, false)
                }
                .unwrap();
                let logits = f.read_logits(&q.gpu).unwrap();
                assert_eq!(logits.len(), block.len());
                for l in &logits {
                    assert!(l.iter().all(|x| x.is_finite()));
                    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    total += m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
                        - l[tokens[position + 1] as usize] as f64;
                    if samples.contains(&position) {
                        for v in l {
                            sample_file.write_all(&v.to_le_bytes()).unwrap();
                        }
                    }
                    position += 1;
                }
                step += 1;
            }
            if mixed_small {
                println!(
                    "TF_MIXED_COUNTS {}/{} count1_to4={counts:?}",
                    fields[0], fields[1]
                );
            }
            writeln!(metrics,"{{\"category\":\"{}\",\"id\":\"{}\",\"tokens\":{},\"nll_sum\":{},\"seconds\":{},\"metal_peak_bytes\":{}}}",fields[0],fields[1],position,total,start.elapsed().as_secs_f64(),q.gpu.peak_allocated_bytes()).unwrap();
            metrics.flush().unwrap();
            println!(
                "TF {}/{} tokens={} nll={} seconds={}",
                fields[0],
                fields[1],
                position,
                total,
                start.elapsed().as_secs_f64()
            );
        }
    }
}
