//! Four fixed conversation requests after a distinct warmup. Measurement only.
use ingot::{
    Result,
    engine::{DraftConfig, Engine},
    invalid,
};
use std::{fs, path::Path};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 10 {
        return Err(invalid(
            "bench_sequence MODEL BUDGET_GIB CONTEXT CAP PREFIX WARMUP ROUND1 ROUND2 ROUND3 ROUND4",
        ));
    }
    let budget = a[1]
        .parse::<usize>()
        .map_err(invalid)?
        .checked_mul(1 << 30)
        .ok_or_else(|| invalid("budget overflow"))?;
    let context = a[2].parse().map_err(invalid)?;
    let cap = a[3].parse().map_err(invalid)?;
    let prefix = match a[4].as_str() {
        "on" => true,
        "off" => false,
        _ => return Err(invalid("PREFIX must be on or off")),
    };
    let prompts = a[5..]
        .iter()
        .map(|path| -> Result<Vec<u32>> {
            let raw = fs::read(path)?;
            if raw.len() % 4 != 0 {
                return Err(invalid("partial token"));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;
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
    let mut engine = Engine::load(Path::new(&a[0]), budget, context, draft, prefix)?;
    for (round, prompt) in prompts.iter().enumerate() {
        let result = engine.generate(prompt, cap, Default::default(), |_, _| Ok(()))?;
        let s = result.statistics;
        let seconds = s.total.saturating_sub(s.first_token).as_secs_f64();
        let decode = if result.tokens.len() > 1 && seconds > 0. {
            ((result.tokens.len() - 1) as f64 / seconds).to_string()
        } else {
            "null".into()
        };
        let visible = s
            .first_visible
            .map(|v| v.as_secs_f64().to_string())
            .unwrap_or("null".into());
        println!("{{\"round\":{round},\"input_tokens\":{},\"load_s\":{},\"prefill_s\":{},\"first_token_s\":{},\"first_visible_s\":{visible},\"total_s\":{},\"decode_tokens_s\":{decode},\"reserved_bytes\":{},\"metal_bytes_at_end\":{},\"metal_peak_since_load\":{},\"file_bytes\":{},\"cache_hits\":{},\"cache_misses\":{},\"read_s\":{},\"prefix_reused_tokens\":{},\"prefix_saved_tokens\":{},\"prefix_restore_s\":{},\"prefix_save_s\":{},\"workspace_allocations\":{},\"workspace_reuses\":{},\"workspace_bytes\":{},\"workspace_peak_bytes\":{},\"prefill_block_tokens\":{},\"mtp\":{},\"stop\":\"{:?}\",\"tokens\":{:?},\"mtp_rounds\":[{}]}}",
            prompt.len(), s.load.as_secs_f64(), s.prefill.as_secs_f64(), s.first_token.as_secs_f64(), s.total.as_secs_f64(),
            s.reserved_bytes, s.metal_bytes, s.metal_peak_bytes, s.file_bytes, s.cache_hits, s.cache_misses, s.read_time.as_secs_f64(), s.prefix_reused_tokens, s.prefix_saved_tokens, s.prefix_restore.as_secs_f64(), s.prefix_save.as_secs_f64(), s.workspace_allocations, s.workspace_reuses, s.workspace_bytes, s.workspace_peak_bytes, s.prefill_block_tokens, s.mtp_enabled,
            result.stop, result.tokens, s.mtp_rounds.iter().map(|r| format!(
                "{{\"proposed\":{},\"accepted\":{},\"history_proposed\":{},\"history_accepted\":{},\"committed\":{},\"verification_tokens\":{},\"replayed_tokens\":{},\"snapshot_bytes\":{},\"head_s\":{},\"verify_s\":{},\"snapshot_s\":{},\"snapshot_encode_s\":{},\"replay_s\":{},\"catch_up_s\":{},\"total_s\":{}}}",
                r.proposed,r.accepted,r.history_proposed,r.history_accepted,r.committed,r.verification_tokens,r.replayed_tokens,r.snapshot_bytes,
                r.head.as_secs_f64(),r.verify.as_secs_f64(),r.snapshot.as_secs_f64(),r.snapshot_encode.as_secs_f64(),r.replay.as_secs_f64(),r.catch_up.as_secs_f64(),r.total.as_secs_f64()
            )).collect::<Vec<_>>().join(",")
        );
    }
    Ok(())
}
