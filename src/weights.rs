//! Shared quantized weights and fixed expert slots; models own routing semantics.
use crate::{
    Result,
    gguf::{Gguf, Tensor},
    invalid,
    metal::{Buffer, Command, Gpu},
};
use std::cell::RefCell;
use std::time::{Duration, Instant};
pub(crate) struct Weight {
    pub(crate) buffer: Buffer,
    pub(crate) t: Tensor,
}
impl Weight {
    pub(crate) fn require_f32(&self) -> Result<()> {
        if self.t.kind != 0 {
            return Err(invalid(
                "normalization/state parameters require F32 weights",
            ));
        }
        Ok(())
    }
    pub(crate) fn load(g: &mut Gguf, gpu: &Gpu, name: &str, shape: &[usize]) -> Result<Self> {
        let t = g.tensor(name)?.clone();
        if t.dims != shape {
            return Err(invalid(format!(
                "{name}: expected {shape:?}, found {:?}",
                t.dims
            )));
        }
        let buffer = gpu.load_weight(g, &t)?;
        Ok(Self { buffer, t })
    }
    pub(crate) fn project(&self, gpu: &Gpu, c: &Command, x: &Buffer) -> Result<Buffer> {
        gpu.matvec(
            c,
            &self.buffer,
            self.t.dims[0],
            self.t.dims.get(1).copied().unwrap_or(1),
            self.t.stride,
            self.t.kind,
            x,
        )
    }
}
pub(crate) struct ExpertCache {
    pub(crate) tensors: [Tensor; 3],
    pub(crate) buffers: [Buffer; 3],
    pub(crate) slots: Vec<Option<(usize, u64)>>,
    pub(crate) clock: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) read: Duration,
}
pub(crate) enum ExpertWeights {
    Resident([Weight; 3]),
    Cached(RefCell<ExpertCache>),
}
impl ExpertCache {
    // Metadata only: a hit never mutates weight storage read by an in-flight command.
    pub(crate) fn hit(&mut self, id: usize) -> Option<usize> {
        let slot = self
            .slots
            .iter()
            .position(|s| s.is_some_and(|(e, _)| e == id))?;
        self.clock += 1;
        self.slots[slot] = Some((id, self.clock));
        self.hits += 1;
        Some(slot)
    }
    pub(crate) fn slot(
        &mut self,
        gpu: &Gpu,
        g: &mut Gguf,
        count: usize,
        id: usize,
        pinned: &[usize],
    ) -> Result<usize> {
        if let Some(slot) = self.hit(id) {
            return Ok(slot);
        }
        self.clock += 1;
        let clock = self.clock;
        let slot = (0..self.slots.len())
            .filter(|s| !pinned.contains(s))
            .min_by_key(|&i| self.slots[i].map(|(_, age)| age).unwrap_or(0))
            .ok_or_else(|| invalid("no reusable expert slot"))?;
        self.slots[slot] = None;
        let start = Instant::now();
        gpu.load_expert(g, &self.tensors, &self.buffers, count, id, slot)?;
        self.read += start.elapsed();
        self.slots[slot] = Some((id, clock));
        self.misses += 1;
        Ok(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_slots_remain_pinned_during_prefetch_and_failed_reads_stay_invalid() {
        let path = std::env::temp_dir().join(format!("ingot-cache-{}.gguf", std::process::id()));
        let mut data = Vec::new();
        let text = |out: &mut Vec<u8>, value: &str| {
            out.extend((value.len() as u64).to_le_bytes());
            out.extend(value.as_bytes());
        };
        data.extend(0x4655_4747u32.to_le_bytes());
        data.extend(3u32.to_le_bytes());
        data.extend(1u64.to_le_bytes());
        data.extend(1u64.to_le_bytes());
        text(&mut data, "general.architecture");
        data.extend(8u32.to_le_bytes());
        text(&mut data, "fixture");
        text(&mut data, "experts");
        data.extend(3u32.to_le_bytes());
        for dim in [2u64, 2, 4] {
            data.extend(dim.to_le_bytes());
        }
        data.extend(0u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.resize(data.len().div_ceil(32) * 32, 0);
        let payload = data.len();
        for expert in 0..4 {
            for value in 1..=4 {
                data.extend(((expert * 10 + value) as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, data).unwrap();
        let mut model = Gguf::open(&path).unwrap();
        let tensor = model.tensor("experts").unwrap().clone();
        let gpu = Gpu::new().unwrap();
        let mut cache = ExpertCache {
            tensors: [tensor.clone(), tensor.clone(), tensor],
            buffers: std::array::from_fn(|_| gpu.empty_bytes(3 * 16).unwrap()),
            slots: vec![None; 3],
            clock: 0,
            hits: 0,
            misses: 0,
            read: Duration::ZERO,
        };
        for expert in 0..3 {
            assert_eq!(
                cache.slot(&gpu, &mut model, 4, expert, &[]).unwrap(),
                expert
            );
        }
        assert_eq!(cache.hit(0), Some(0));
        assert_eq!(cache.hit(1), Some(1));
        let ids = gpu
            .bytes(
                &[0u32, 1]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = gpu.floats(&[1., 1.]).unwrap();
        let out = gpu.output(4).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "project_experts",
            &[&cache.buffers[0], &x, &ids, &out],
            &[2, 2, 8, 0, 0],
            2,
            2,
        )
        .unwrap();
        gpu.submit(&cmd);
        let next = cache.slot(&gpu, &mut model, 4, 3, &[0, 1]);
        gpu.wait(cmd).unwrap();
        assert_eq!(next.unwrap(), 2);
        let actual: Vec<_> = gpu
            .read_u32(&out, 4)
            .unwrap()
            .into_iter()
            .map(f32::from_bits)
            .collect();
        assert_eq!(actual, [3., 7., 23., 27.]);
        let before = cache.slots.clone();
        assert_eq!(
            cache
                .slot(&gpu, &mut model, 4, 2, &[0, 1, 2])
                .unwrap_err()
                .to_string(),
            "no reusable expert slot"
        );
        assert_eq!(cache.slots, before);
        assert_eq!(cache.slot(&gpu, &mut model, 4, 0, &[0, 1, 2]).unwrap(), 0);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(payload as u64 + 8)
            .unwrap();
        assert_eq!(
            cache
                .slot(&gpu, &mut model, 4, 2, &[0, 1])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        assert_eq!(cache.slots[2], None);
        assert_eq!(cache.hit(2), None);
        assert_eq!(cache.hit(3), None);
        assert_eq!((cache.hits, cache.misses), (3, 4));
        std::fs::remove_file(path).unwrap();
    }
}
