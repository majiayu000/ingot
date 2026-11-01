//! Validation only: per-position native target/MTP tensors for a fixed oracle.
use ingot::{
    Result,
    engine::{DraftConfig, Engine},
    invalid,
};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 4 {
        return Err(invalid("mtp_trace MODEL DRAFT TOKENS_U32 OUTPUT_DIR"));
    }
    let raw = std::fs::read(&a[2])?;
    if raw.is_empty() || raw.len() % 4 != 0 {
        return Err(invalid("token file"));
    }
    let ids = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut e = Engine::load(
        Path::new(&a[0]),
        44 * (1usize << 30),
        ids.len() + 8,
        Some(DraftConfig {
            model: a[1].clone().into(),
            max_tokens: 3,
            vocabulary: None,
        }),
        false,
    )?;
    let mut session = e.session(Default::default())?;
    for (pos, &id) in ids.iter().enumerate() {
        e.set_validation_trace(pos, &Path::new(&a[3]).join(pos.to_string()))?;
        e.consume(&mut session, &[id])?;
        println!("traced position={pos}");
    }
    Ok(())
}
