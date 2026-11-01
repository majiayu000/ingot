//! CPU-only vocabulary selection using the engine's verified model tokenizer.
//! Keep training-observed IDs, all Han, byte tokens and control/user-defined IDs.
use ingot::{Result, gguf::Gguf, invalid, tokenizer::Tokenizer};
use std::{fs, io::Write, path::Path};
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 3 {
        return Err(invalid("draft_vocab MODEL OUTPUT_IDS CORPUS..."));
    }
    let g = Gguf::open(Path::new(&args[0]))?;
    let tokenizer = Tokenizer::from_gguf(&g)?;
    let types = g.integers("tokenizer.ggml.token_type")?;
    let mut counts = vec![0u64; types.len()];
    for path in &args[2..] {
        for id in tokenizer.encode(&fs::read_to_string(path)?, false)? {
            counts[id as usize] += 1;
        }
    }
    let mut selected = Vec::new();
    for (id, (&count, &kind)) in counts.iter().zip(&types).enumerate() {
        let piece = tokenizer.token_bytes(id as u32)?;
        let han = std::str::from_utf8(piece).is_ok_and(|s| s.chars().any(|c| {
            matches!(c as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2fa1f)
        }));
        if count != 0 || piece.len() == 1 || matches!(kind, 3 | 4) || han {
            selected.push(id as u32);
        }
    }
    if selected.is_empty() || selected.len() >= types.len() {
        return Err(invalid(
            "selection must be nonempty and smaller than vocabulary",
        ));
    }
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    for id in &selected {
        out.write_all(&id.to_le_bytes())?;
    }
    out.flush()?;
    println!(
        "selected={} vocabulary={} training_occurrences={} training_distinct={}",
        selected.len(),
        types.len(),
        counts.iter().sum::<u64>(),
        counts.iter().filter(|&&x| x > 0).count()
    );
    Ok(())
}
