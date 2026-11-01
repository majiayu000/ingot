//! Persistent-engine benchmark: round zero is retained warmup, later rounds measured.
use ingot::{
    Result,
    engine::{DraftConfig, Engine},
    invalid,
};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err(invalid(
            "bench MODEL BUDGET_GIB CONTEXT MAX_TOKENS REPEATS PROMPT",
        ));
    }
    let budget = a[1]
        .parse::<usize>()
        .map_err(invalid)?
        .checked_mul(1 << 30)
        .ok_or_else(|| invalid("budget overflow"))?;
    let context = a[2].parse().map_err(invalid)?;
    let max_tokens = a[3].parse().map_err(invalid)?;
    let repeats = a[4].parse::<usize>().map_err(invalid)?;
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
    let mut engine = Engine::load(Path::new(&a[0]), budget, context, draft, false)?;
    let prompt = engine.tokenizer.chat(&a[5])?;
    let mut reference = None;
    for round in 0..=repeats {
        let result = engine.generate(&prompt, max_tokens, Default::default(), |_, _| Ok(()))?;
        if let Some((tokens, stop)) = &reference {
            if &result.tokens != tokens || &result.stop != stop {
                return Err(invalid("repeated output differs"));
            }
        } else {
            reference = Some((result.tokens.clone(), result.stop));
        }
        println!(
            "round={round} stop={:?} token_ids={:?}\nstatistics={:?}",
            result.stop, result.tokens, result.statistics
        );
    }
    Ok(())
}
