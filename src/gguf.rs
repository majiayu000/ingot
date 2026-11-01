//! Validated GGUF v3 directories and positional reads. No inference runtime dependency.
use crate::{Result, invalid};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Integer(u64),
    Float(f64),
    Text(String),
    Array(Vec<Value>),
}
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub kind: u32,
    pub offset: u64,
    pub shard: usize,
    pub stride: usize,
    pub bytes: usize,
}
impl Tensor {
    // A multiple of both macOS page sizes (4 KiB / 16 KiB). Metal requires
    // page-aligned address and length; tensor bindings retain their byte offset.
    pub(crate) fn mapping_layout(&self) -> Result<(u64, usize, usize)> {
        const PAGE: usize = 16 * 1024;
        let offset = (self.offset % PAGE as u64) as usize;
        let length = offset
            .checked_add(self.bytes)
            .and_then(|n| n.checked_add(PAGE - 1))
            .map(|n| n / PAGE * PAGE)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("mapped tensor size overflow"))?;
        Ok((self.offset - offset as u64, offset, length))
    }
    pub(crate) fn mapped_bytes(&self) -> Result<usize> {
        Ok(self.mapping_layout()?.2)
    }
}
pub struct Gguf {
    files: Vec<File>,
    pub tensors: HashMap<String, Tensor>,
    pub architecture: String,
    pub metadata: HashMap<String, Value>,
    pub bytes_read: u64,
    /// Wall time spent gathering requested rows, including planning and worker joins.
    pub row_read_time: Duration,
}
struct Directory {
    tensors: HashMap<String, Tensor>,
    metadata: HashMap<String, Value>,
}
struct Reader {
    input: BufReader<File>,
    size: u64,
}
impl Reader {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let pos = self.input.stream_position()?;
        if (n as u64) > self.size.saturating_sub(pos) {
            return Err(invalid("truncated GGUF value"));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(n).map_err(io::Error::other)?;
        data.resize(n, 0);
        self.input.read_exact(&mut data)?;
        Ok(data)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().map_err(|_| invalid("u32"))?,
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().map_err(|_| invalid("u64"))?,
        ))
    }
    fn string(&mut self) -> Result<String> {
        let n = usize::try_from(self.u64()?).map_err(invalid)?;
        String::from_utf8(self.bytes(n)?).map_err(invalid)
    }
    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            8 => Value::Text(self.string()?),
            9 => {
                let element = self.u32()?;
                if element == 9 {
                    return Err(invalid("nested GGUF arrays"));
                }
                let n = usize::try_from(self.u64()?).map_err(invalid)?;
                if n as u64 > self.size.saturating_sub(self.input.stream_position()?) {
                    return Err(invalid("truncated GGUF array"));
                }
                let mut a = Vec::new();
                a.try_reserve_exact(n).map_err(io::Error::other)?;
                for _ in 0..n {
                    a.push(self.value(element)?);
                }
                Value::Array(a)
            }
            0 | 1 | 7 => Value::Integer(u64::from(self.bytes(1)?[0])),
            2 | 3 => Value::Integer(u64::from(u16::from_le_bytes(
                self.bytes(2)?.try_into().map_err(|_| invalid("u16"))?,
            ))),
            4 | 5 => Value::Integer(u64::from(self.u32()?)),
            10 | 11 => Value::Integer(self.u64()?),
            6 => Value::Float(f32::from_bits(self.u32()?) as f64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return Err(invalid(format!("unknown GGUF metadata type {kind}"))),
        })
    }
    fn directory(mut self, shard: usize) -> Result<Directory> {
        if self.u32()? != 0x4655_4747 || self.u32()? != 3 {
            return Err(invalid("expected little-endian GGUF v3"));
        }
        let nt = self.u64()?;
        let nm = self.u64()?;
        let mut metadata = HashMap::new();
        for _ in 0..nm {
            let key = self.string()?;
            let kind = self.u32()?;
            if metadata.insert(key, self.value(kind)?).is_some() {
                return Err(invalid("duplicate GGUF metadata"));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Integer(n)) => *n,
            None => 32,
            _ => return Err(invalid("invalid alignment metadata")),
        };
        if !alignment.is_power_of_two() {
            return Err(invalid("invalid GGUF alignment"));
        }
        let mut tensors = HashMap::new();
        for _ in 0..nt {
            let name = self.string()?;
            let rank = self.u32()?;
            if !(1..=4).contains(&rank) {
                return Err(invalid("invalid tensor rank"));
            }
            let mut dims = Vec::new();
            for _ in 0..rank {
                dims.push(usize::try_from(self.u64()?).map_err(invalid)?);
            }
            let kind = self.u32()?;
            let offset = self.u64()?;
            let (block, size) = geometry(kind)?;
            if dims.contains(&0) || dims[0] % block != 0 || offset % alignment != 0 {
                return Err(invalid(format!("invalid shape/alignment for {name}")));
            }
            let stride = (dims[0] / block)
                .checked_mul(size)
                .ok_or_else(|| invalid("row overflow"))?;
            let bytes = dims[1..].iter().try_fold(stride, |n, &d| {
                n.checked_mul(d).ok_or_else(|| invalid("tensor overflow"))
            })?;
            if tensors
                .insert(
                    name,
                    Tensor {
                        dims,
                        kind,
                        offset,
                        shard,
                        stride,
                        bytes,
                    },
                )
                .is_some()
            {
                return Err(invalid("duplicate tensor"));
            }
        }
        let pos = self.input.stream_position()?;
        let start = pos
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("header overflow"))?
            & !(alignment - 1);
        let mut ranges = Vec::new();
        for (name, t) in &mut tensors {
            t.offset = t
                .offset
                .checked_add(start)
                .ok_or_else(|| invalid("offset overflow"))?;
            let end = t
                .offset
                .checked_add(t.bytes as u64)
                .filter(|&n| n <= self.size)
                .ok_or_else(|| invalid(format!("{name}: extends beyond file")))?;
            ranges.push((t.offset, end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(invalid("overlapping tensor ranges"));
        }
        Ok(Directory { tensors, metadata })
    }
}
impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let dir = Reader {
            size: file.metadata()?.len(),
            input: BufReader::new(file.try_clone()?),
        }
        .directory(0)?;
        let architecture = match dir.metadata.get("general.architecture") {
            Some(Value::Text(s)) => s.clone(),
            _ => return Err(invalid("missing architecture")),
        };
        let mut this = Self {
            files: vec![file],
            tensors: dir.tensors,
            architecture,
            metadata: dir.metadata,
            bytes_read: 0,
            row_read_time: Duration::ZERO,
        };
        let count = this.optional_integer("split.count").unwrap_or(1);
        if count > 1 {
            if this.integer("split.no")? != 0 {
                return Err(invalid("pass the primary GGUF shard"));
            }
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| invalid("invalid shard name"))?;
            let marker = format!("-00001-of-{count:05}.gguf");
            let prefix = name
                .strip_suffix(&marker)
                .ok_or_else(|| invalid("invalid split filename"))?;
            for shard in 1..count {
                let next =
                    path.with_file_name(format!("{prefix}-{:05}-of-{count:05}.gguf", shard + 1));
                let f = File::open(next)?;
                let d = Reader {
                    size: f.metadata()?.len(),
                    input: BufReader::new(f.try_clone()?),
                }
                .directory(shard)?;
                if d.metadata.get("split.no") != Some(&Value::Integer(shard as u64))
                    || d.metadata.get("split.count") != Some(&Value::Integer(count as u64))
                    || d.metadata.get("split.tensors.count")
                        != this.metadata.get("split.tensors.count")
                {
                    return Err(invalid("inconsistent GGUF shards"));
                }
                for (k, v) in &d.metadata {
                    if !k.starts_with("split.") && this.metadata.get(k).is_some_and(|old| old != v)
                    {
                        return Err(invalid(format!("inconsistent shard metadata: {k}")));
                    }
                }
                for (name, t) in d.tensors {
                    if this.tensors.insert(name, t).is_some() {
                        return Err(invalid("duplicate tensor across shards"));
                    }
                }
                this.files.push(f);
            }
            if this.integer("split.tensors.count")? != this.tensors.len() {
                return Err(invalid("split tensor count mismatch"));
            }
        }
        Ok(this)
    }
    pub fn optional_integer(&self, key: &str) -> Option<usize> {
        match self.metadata.get(key) {
            Some(Value::Integer(n)) => usize::try_from(*n).ok(),
            _ => None,
        }
    }
    pub fn integer(&self, key: &str) -> Result<usize> {
        self.optional_integer(key)
            .ok_or_else(|| invalid(format!("missing integer: {key}")))
    }
    pub fn float(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(Value::Float(n)) if (*n as f32).is_finite() => Ok(*n as f32),
            _ => Err(invalid(format!("missing float: {key}"))),
        }
    }
    pub fn optional_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(Value::Text(s)) => Some(s),
            _ => None,
        }
    }
    pub fn optional_float(&self, key: &str) -> Option<f32> {
        self.float(key).ok()
    }
    pub fn string(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(Value::Text(s)) => Ok(s),
            _ => Err(invalid(format!("missing string: {key}"))),
        }
    }
    pub fn array(&self, key: &str) -> Result<&[Value]> {
        match self.metadata.get(key) {
            Some(Value::Array(a)) => Ok(a),
            _ => Err(invalid(format!("missing array: {key}"))),
        }
    }
    pub fn integers(&self, key: &str) -> Result<Vec<u64>> {
        self.array(key)?
            .iter()
            .map(|v| match v {
                Value::Integer(n) => Ok(*n),
                _ => Err(invalid(key)),
            })
            .collect()
    }
    pub fn strings(&self, key: &str) -> Result<Vec<String>> {
        self.array(key)?
            .iter()
            .map(|v| match v {
                Value::Text(s) => Ok(s.clone()),
                _ => Err(invalid(key)),
            })
            .collect()
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor: {name}")))
    }
    fn read_location(&self, t: &Tensor, offset: usize, length: usize) -> Result<(&File, u64)> {
        if offset.checked_add(length).is_none_or(|n| n > t.bytes) {
            return Err(invalid("tensor read out of bounds"));
        }
        let file = self
            .files
            .get(t.shard)
            .ok_or_else(|| invalid("invalid tensor shard"))?;
        let start = t
            .offset
            .checked_add(offset as u64)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        Ok((file, start))
    }
    pub fn read_range(&mut self, t: &Tensor, offset: usize, data: &mut [u8]) -> Result<()> {
        let (file, start) = self.read_location(t, offset, data.len())?;
        file.read_exact_at(data, start)?;
        self.bytes_read += data.len() as u64;
        Ok(())
    }
    pub(crate) fn map_tensor(&self, t: &Tensor) -> Result<memmap2::Mmap> {
        let (file, start) = self.read_location(t, 0, t.bytes)?;
        let file_length = file.metadata()?.len();
        if start
            .checked_add(t.bytes as u64)
            .is_none_or(|end| end > file_length)
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "mapped tensor was truncated",
            ));
        }
        let (start, _, length) = t.mapping_layout()?;
        // SAFETY: model shards must remain immutable while the engine is loaded.
        // Only validated tensor bytes are read, including the final partial file
        // page; page padding is never exposed as logical tensor storage. Mmap owns
        // the mapping independently of this File, and no CPU/GPU path writes it.
        unsafe {
            memmap2::MmapOptions::new()
                .offset(start)
                .len(length)
                .map(file)
        }
    }
    pub(crate) fn read_ranges(
        &mut self,
        pool: &rayon::ThreadPool,
        ranges: Vec<(&Tensor, usize, &mut [u8])>,
    ) -> Result<()> {
        use rayon::prelude::*;
        let jobs = ranges
            .into_iter()
            .map(|(t, offset, bytes)| {
                let (file, start) = self.read_location(t, offset, bytes.len())?;
                Ok((file, start, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let results = pool.install(|| {
            jobs.into_par_iter()
                .map(|(file, start, bytes)| {
                    file.read_exact_at(bytes, start)
                        .map(|()| bytes.len() as u64)
                })
                .collect::<Vec<_>>()
        });
        // Drain every read before propagating failure. Successful reads remain observable.
        self.bytes_read += results.iter().filter_map(|x| x.as_ref().ok()).sum::<u64>();
        for result in results {
            result?;
        }
        Ok(())
    }
    pub fn rows(&mut self, pool: &rayon::ThreadPool, name: &str, rows: &[usize]) -> Result<Matrix> {
        let t = self.tensor(name)?.clone();
        let n = t.bytes / t.stride;
        if rows.is_empty() || rows.iter().any(|&r| r >= n) {
            return Err(invalid("row out of range"));
        }
        let len = t
            .stride
            .checked_mul(rows.len())
            .ok_or_else(|| invalid("rows overflow"))?;
        let started = Instant::now();
        let mut data = vec![0; len];
        // Each job owns a disjoint output row; repeated indices keep caller order.
        let ranges = rows
            .iter()
            .zip(data.chunks_exact_mut(t.stride))
            .map(|(&row, out)| (&t, row * t.stride, out))
            .collect();
        let result = self.read_ranges(pool, ranges);
        self.row_read_time += started.elapsed();
        result?;
        Ok(Matrix {
            data,
            cols: t.dims[0],
            rows: rows.len(),
            count: 1,
            kind: t.kind,
            stride: t.stride,
        })
    }
    pub fn matrix(&mut self, name: &str, shape: &[usize], experts: &[usize]) -> Result<Matrix> {
        let t = self.tensor(name)?.clone();
        if t.dims != shape {
            return Err(invalid(format!(
                "{name}: expected {shape:?}, found {:?}",
                t.dims
            )));
        }
        let rows = shape.get(1).copied().unwrap_or(1);
        let count = shape.get(2).copied().unwrap_or(1);
        if experts.is_empty() || experts.iter().any(|&e| e >= count) {
            return Err(invalid("expert out of range"));
        }
        let size = t.stride * rows;
        let mut data = vec![0; size * experts.len()];
        for (i, &e) in experts.iter().enumerate() {
            self.read_range(&t, e * size, &mut data[i * size..(i + 1) * size])?;
        }
        Ok(Matrix {
            data,
            cols: shape[0],
            rows,
            count: experts.len(),
            stride: t.stride,
            kind: t.kind,
        })
    }
}
pub struct Matrix {
    pub data: Vec<u8>,
    pub cols: usize,
    pub rows: usize,
    pub count: usize,
    pub kind: u32,
    pub stride: usize,
}
pub fn geometry(kind: u32) -> Result<(usize, usize)> {
    match kind {
        0 => Ok((1, 4)),
        1 | 30 => Ok((1, 2)),
        2 => Ok((32, 18)),
        8 => Ok((32, 34)),
        10 => Ok((256, 84)),
        12 => Ok((256, 144)),
        13 => Ok((256, 176)),
        14 => Ok((256, 210)),
        16 => Ok((256, 66)),
        17 => Ok((256, 74)),
        18 => Ok((256, 98)),
        20 => Ok((32, 18)),
        21 => Ok((256, 110)),
        22 => Ok((256, 82)),
        23 => Ok((256, 136)),
        29 => Ok((256, 56)),
        42 => Ok((64, 18)),
        _ => Err(invalid(format!("GGML type {kind} is unsupported"))),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncated_header() {
        let p = std::env::temp_dir().join(format!("ingot-{}.gguf", std::process::id()));
        std::fs::write(&p, b"GGUF\x03\0\0\0").unwrap();
        assert!(Gguf::open(&p).is_err());
        std::fs::remove_file(p).unwrap();
        assert!(geometry(19).is_err());
    }
    #[test]
    fn positional_read_reports_truncation_after_directory_validation() {
        use std::io::Write;
        let p = std::env::temp_dir().join(format!("ingot-short-read-{}.gguf", std::process::id()));
        let mut header = Vec::new();
        let text = |out: &mut Vec<u8>, value: &str| {
            out.extend((value.len() as u64).to_le_bytes());
            out.extend(value.as_bytes());
        };
        header.extend(0x4655_4747u32.to_le_bytes());
        header.extend(3u32.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        text(&mut header, "general.architecture");
        header.extend(8u32.to_le_bytes());
        text(&mut header, "fixture");
        text(&mut header, "weight");
        header.extend(2u32.to_le_bytes());
        header.extend(2u64.to_le_bytes());
        header.extend(2u64.to_le_bytes());
        header.extend(0u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.resize(header.len().div_ceil(32) * 32, 0);
        let payload = header.len();
        for v in [1f32, 2., 3., 4.] {
            header.extend(v.to_le_bytes());
        }
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&p)
            .unwrap();
        f.write_all(&header).unwrap();
        let mut g = Gguf::open(&p).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let t = g.tensor("weight").unwrap();
        let mapped = g.map_tensor(t).unwrap();
        let (_, offset, length) = t.mapping_layout().unwrap();
        assert_eq!(mapped.len(), length);
        assert_eq!(&mapped[offset..offset + t.bytes], &header[payload..]);
        drop(mapped); // Never truncate a file with live mappings.
        assert_eq!(
            g.rows(&pool, "weight", &[1]).unwrap().data,
            [3f32, 4.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let before = g.bytes_read;
        assert_eq!(
            g.rows(&pool, "weight", &[1, 0, 1, 0]).unwrap().data,
            [3f32, 4., 1., 2., 3., 4., 1., 2.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(g.bytes_read - before, 32);
        assert!(g.row_read_time > Duration::ZERO);
        let before = g.bytes_read;
        assert!(g.rows(&pool, "weight", &[]).is_err());
        assert!(g.rows(&pool, "weight", &[0, 2]).is_err());
        assert_eq!(g.bytes_read, before);
        f.set_len(payload as u64 + 10).unwrap();
        assert_eq!(
            g.map_tensor(g.tensor("weight").unwrap())
                .err()
                .unwrap()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            g.rows(&pool, "weight", &[1]).err().unwrap().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        let before = g.bytes_read;
        let elapsed = g.row_read_time;
        assert_eq!(
            g.rows(&pool, "weight", &[0, 1, 0, 1]).err().unwrap().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        // All workers finish before returning the error; both valid rows are counted.
        assert_eq!(g.bytes_read - before, 16);
        assert!(g.row_read_time > elapsed);
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    #[ignore = "real PLE file rows: bytes and accounting, not a cache-controlled speed ranking"]
    fn real_ple_parallel_rows_match_serial_bytes() {
        let mut g = Gguf::open(Path::new(&std::env::var("QWEN_MODEL").unwrap())).unwrap();
        let t = g.tensor("per_layer_token_embd.weight").unwrap().clone();
        let n = t.bytes / t.stride;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        eprintln!(
            "PLE_TABLE dims={:?} stride={} bytes={}",
            t.dims, t.stride, t.bytes
        );
        for count in [16usize, 64, 65_536] {
            let mut seed = 0x123456789abcdefu64;
            let mut rows: Vec<usize> = (0..count)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as usize % n
                })
                .collect();
            rows[..4].copy_from_slice(&[n - 1, 0, n - 1, 0]);
            let mut expected = vec![0; count * t.stride];
            for (&row, out) in rows.iter().zip(expected.chunks_exact_mut(t.stride)) {
                g.read_range(&t, row * t.stride, out).unwrap();
            }
            let before = g.bytes_read;
            let actual = g.rows(&pool, "per_layer_token_embd.weight", &rows).unwrap();
            assert_eq!(actual.data, expected);
            assert_eq!(g.bytes_read - before, (count * t.stride) as u64);
            eprintln!(
                "PLE_BYTES count={} equal=true bytes={}",
                count,
                actual.data.len()
            );
        }
    }

    #[test]
    #[ignore = "controlled PLE cache state in complete requests; exclusive GPU use"]
    fn ple_cache_controlled_model_requests() {
        use crate::engine::{DraftConfig, Engine};
        use std::ffi::{c_char, c_int, c_void};
        unsafe extern "C" {
            fn mincore(addr: *const c_void, length: usize, vector: *mut c_char) -> c_int;
            fn msync(addr: *mut c_void, length: usize, flags: c_int) -> c_int;
            fn getpagesize() -> c_int;
        }
        let model = std::env::var("QWEN_MODEL").unwrap();
        let g = Gguf::open(Path::new(&model)).unwrap();
        let t = g.tensor("per_layer_token_embd.weight").unwrap().clone();
        let mapped = g.map_tensor(&t).unwrap();
        let (_, offset, _) = t.mapping_layout().unwrap();
        // SAFETY: getpagesize takes no arguments.
        let page = unsafe { getpagesize() } as usize;
        let raw = std::fs::read(std::env::var("PLE_BENCH_INPUT").unwrap()).unwrap();
        let prompt: Vec<u32> = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(raw.len(), prompt.len() * 4);
        // Reconstruct only the requested page list for cache-state observation.
        let ngram = g.integer("qwen4exp.ple.ngram_size").unwrap();
        let pergram = g.integer("qwen4exp.ple.heads_per_ngram").unwrap();
        let eos = g.integer("qwen4exp.ple.eos_token_id").unwrap() as u32;
        let multipliers = g.integers("qwen4exp.ple.layer_multipliers").unwrap();
        let offsets = g.integers("qwen4exp.ple.head_offsets").unwrap();
        let sizes = g.integers("qwen4exp.ple.head_vocab_sizes").unwrap();
        let mut pages = Vec::new();
        for (position, &token) in prompt.iter().enumerate() {
            let mut ctx = vec![token as u64];
            let mut cut = false;
            for j in 1..ngram {
                let previous = position.checked_sub(j).map(|i| prompt[i]);
                cut |= previous.is_none() || previous == Some(eos);
                ctx.push(if cut {
                    eos as u64
                } else {
                    previous.unwrap() as u64
                });
            }
            for n in 2..=ngram {
                let mixed = ctx[..n]
                    .iter()
                    .zip(&multipliers)
                    .fold(0, |v, (&a, &b)| v ^ a.wrapping_mul(b));
                for h in (n - 2) * pergram..(n - 1) * pergram {
                    let row = (mixed % sizes[h] + offsets[h]) as usize;
                    let at = offset + row * t.stride;
                    pages.extend([at / page, (at + t.stride - 1) / page]);
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        let resident = || {
            let mut bits = vec![0u8; mapped.len().div_ceil(page)];
            // SAFETY: live aligned read-only mapping and one-byte result per page.
            assert_eq!(
                unsafe {
                    mincore(
                        mapped.as_ptr().cast(),
                        mapped.len(),
                        bits.as_mut_ptr().cast(),
                    )
                },
                0,
                "{}",
                std::io::Error::last_os_error()
            );
            pages.iter().filter(|&&i| bits[i] & 1 != 0).count()
        };
        let context: usize = std::env::var("PLE_BENCH_CONTEXT").unwrap().parse().unwrap();
        let mut engine = Engine::load(
            Path::new(&model),
            64 * (1 << 30),
            context,
            Some(DraftConfig {
                model: std::env::var("QWEN_DRAFT").unwrap().into(),
                max_tokens: 3,
                vocabulary: None,
            }),
            false,
        )
        .unwrap();
        let reference = engine
            .generate(&prompt, 128, Default::default(), |_, _| Ok(()))
            .unwrap();
        assert_eq!(reference.tokens.len(), 128);
        for cycle in 0..2 {
            // Evict only whole pages fully inside PLE. Boundary pages may also
            // contain adjacent GPU-resident tensors, so leave them alone.
            let start = offset.div_ceil(page) * page;
            let end = (offset + t.bytes) / page * page;
            // SAFETY: immutable, clean mapped file bytes; previous GPU request ended.
            assert_eq!(
                unsafe {
                    msync(
                        mapped.as_ptr().add(start).cast_mut().cast(),
                        end - start,
                        0x0010 | 0x0002,
                    )
                },
                0,
                "{}",
                std::io::Error::last_os_error()
            );
            for cold in [true, false] {
                let before = resident();
                if cold {
                    assert_eq!(before, 0, "requested PLE pages not all cold");
                }
                let result = engine
                    .generate(&prompt, 128, Default::default(), |_, _| Ok(()))
                    .unwrap();
                let after = resident();
                assert_eq!(result.tokens, reference.tokens);
                assert_eq!(result.stop, reference.stop);
                let s = result.statistics;
                let state = if cold {
                    "cold_verified"
                } else if before == pages.len() {
                    "warm_verified"
                } else {
                    "repeat_partial_residency"
                };
                let decode = 127.0 / (s.total - s.first_token).as_secs_f64();
                eprintln!(
                    "MODEL_PLE_CACHE {{\"cycle\":{cycle},\"cache_state\":\"{state}\",\"resident_before\":{before},\"resident_after\":{after},\"requested_pages\":{},\"total_s\":{},\"prefill_s\":{},\"decode_tokens_s\":{decode},\"ple_read_s\":{},\"ple_prefill_read_s\":{},\"ple_generation_read_s\":{},\"file_bytes\":{},\"metal_peak_since_load\":{},\"stop\":\"{:?}\",\"tokens\":{:?}}}",
                    pages.len(),
                    s.total.as_secs_f64(),
                    s.prefill.as_secs_f64(),
                    s.ple_read_time.as_secs_f64(),
                    s.ple_prefill_read_time.as_secs_f64(),
                    (s.ple_read_time - s.ple_prefill_read_time).as_secs_f64(),
                    s.file_bytes,
                    s.metal_peak_bytes,
                    result.stop,
                    result.tokens
                );
            }
        }
    }
}
