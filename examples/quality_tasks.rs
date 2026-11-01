//! Run the frozen official-answer quality tasks through the public generation API.
use ingot::{
    Result,
    engine::{DraftConfig, Engine},
    invalid,
};
use std::{fs, io::Write, path::Path};
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(invalid(
            "quality_tasks MODEL BUDGET_GIB CONTEXT INDEX_TSV OUTPUT_JSONL",
        ));
    }
    let budget = args[1]
        .parse::<usize>()
        .map_err(invalid)?
        .checked_mul(1 << 30)
        .ok_or_else(|| invalid("budget overflow"))?;
    let context = args[2].parse().map_err(invalid)?;
    let index = fs::read_to_string(&args[3])?;
    let mut out = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[4])?;
    let draft = std::env::var_os("QWEN_DRAFT")
        .map(|model| -> Result<DraftConfig> {
            Ok(DraftConfig {
                model: model.into(),
                max_tokens: std::env::var("DRAFT_TOKENS")
                    .unwrap_or("3".into())
                    .parse()
                    .map_err(invalid)?,
                vocabulary: std::env::var_os("DRAFT_VOCAB").map(Into::into),
            })
        })
        .transpose()?;
    let mut engine = Engine::load(Path::new(&args[0]), budget, context, draft, false)?;
    for line in index.lines() {
        let f = line.split('\t').collect::<Vec<_>>();
        if f.len() != 4 {
            return Err(invalid("quality index fields"));
        }
        let raw = fs::read(f[2])?;
        if raw.len() % 4 != 0 {
            return Err(invalid("partial token"));
        }
        let tokens = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let result = engine.generate(
            &tokens,
            f[3].parse().map_err(invalid)?,
            Default::default(),
            |_, _| Ok(()),
        )?;
        writeln!(
            out,
            "{{\"category\":\"{}\",\"id\":\"{}\",\"stop\":\"{:?}\",\"tokens\":{:?},\"total_s\":{},\"prefill_s\":{},\"metal_peak_bytes\":{},\"prefill_block_tokens\":{},\"mtp\":{}}}",
            f[0],
            f[1],
            result.stop,
            result.tokens,
            result.statistics.total.as_secs_f64(),
            result.statistics.prefill.as_secs_f64(),
            result.statistics.metal_peak_bytes,
            result.statistics.prefill_block_tokens,
            result.statistics.mtp_enabled
        )?;
        out.flush()?;
        println!(
            "TASK {}/{} {:?} output_tokens={} seconds={}",
            f[0],
            f[1],
            result.stop,
            result.tokens.len(),
            result.statistics.total.as_secs_f64()
        );
    }
    Ok(())
}
