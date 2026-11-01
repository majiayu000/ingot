//! Dump checkpoint logits after grouped prefill, for old/new and oracle comparison.
use ingot::{Result, engine::Engine, invalid};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err(invalid(
            "validate MODEL BUDGET_GIB CONTEXT TOKENS CHECKPOINTS OUTPUT",
        ));
    }
    let budget = a[1]
        .parse::<usize>()
        .map_err(invalid)?
        .checked_mul(1 << 30)
        .ok_or_else(|| invalid("budget overflow"))?;
    let context = a[2].parse().map_err(invalid)?;
    let raw = fs::read(&a[3])?;
    if raw.len() % 4 != 0 {
        return Err(invalid("partial token"));
    }
    let tokens: Vec<_> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let checkpoints = a[4]
        .split(',')
        .map(|s| s.parse::<usize>().map_err(invalid))
        .collect::<Result<Vec<_>>>()?;
    let mut previous = 0;
    for &end in &checkpoints {
        if end <= previous || end > tokens.len() || end > context {
            return Err(invalid("invalid checkpoints"));
        }
        previous = end;
    }
    let mut out = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&a[5])?,
    );
    let mut engine = Engine::load(Path::new(&a[0]), budget, context, None, false)?;
    let mut session = engine.session(Default::default())?;
    let mut previous = 0;
    let start = Instant::now();
    for end in checkpoints {
        let logits = engine.consume(&mut session, &tokens[previous..end])?;
        for v in &logits {
            out.write_all(&v.to_le_bytes())?;
        }
        out.flush()?;
        let top = logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0;
        println!("position={end} top={top} elapsed={:?}", start.elapsed());
        previous = end;
    }
    Ok(())
}
