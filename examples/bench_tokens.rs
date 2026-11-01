//! Frozen-token, persistent-engine measurement. Round zero is warmup.
use ingot::{
    Result,
    engine::{DraftConfig, Engine},
    invalid,
};
use std::{fs, path::Path};

fn cache_snapshot(round: usize, stage: &str) -> Result<()> {
    let output = std::process::Command::new("vm_stat").output()?;
    if !output.status.success() {
        return Err(invalid("vm_stat failed"));
    }
    eprintln!(
        "OS_CACHE_SNAPSHOT round={round} stage={stage}\n{}OS_CACHE_END",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err(invalid(
            "bench_tokens MODEL BUDGET_GIB CONTEXT CAP REPEATS TOKENS_U32",
        ));
    }
    let budget = a[1]
        .parse::<usize>()
        .map_err(invalid)?
        .checked_mul(1 << 30)
        .ok_or_else(|| invalid("budget overflow"))?;
    let context = a[2].parse().map_err(invalid)?;
    let cap = a[3].parse().map_err(invalid)?;
    let repeats = a[4].parse::<usize>().map_err(invalid)?;
    let raw = fs::read(&a[5])?;
    if raw.len() % 4 != 0 {
        return Err(invalid("partial token"));
    }
    let prompt: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
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
    let mut engine = Engine::load(
        Path::new(&a[0]),
        budget,
        context,
        draft,
        std::env::var_os("PREFIX_CACHE").is_some(),
    )?;
    let mut reference = None;
    for round in 0..=repeats {
        cache_snapshot(round, "before")?;
        let result = engine.generate(&prompt, cap, Default::default(), |_, _| Ok(()))?;
        cache_snapshot(round, "after")?;
        if let Some((tokens, stop)) = &reference {
            if &result.tokens != tokens || &result.stop != stop {
                return Err(invalid("repeated output differs"));
            }
        } else {
            reference = Some((result.tokens.clone(), result.stop));
        }
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
        println!("{{\"round\":{round},\"input_tokens\":{},\"load_s\":{},\"prefill_s\":{},\"first_token_s\":{},\"first_visible_s\":{visible},\"total_s\":{},\"decode_tokens_s\":{decode},\"reserved_bytes\":{},\"metal_bytes_at_end\":{},\"metal_peak_since_load\":{},\"file_bytes\":{},\"cache_hits\":{},\"cache_misses\":{},\"read_s\":{},\"ple_read_s\":{},\"ple_prefill_read_s\":{},\"ple_generation_read_s\":{},\"prefix_reused_tokens\":{},\"prefix_saved_tokens\":{},\"prefix_restore_s\":{},\"prefix_save_s\":{},\"workspace_allocations\":{},\"workspace_reuses\":{},\"workspace_bytes\":{},\"workspace_peak_bytes\":{},\"prefill_block_tokens\":{},\"mtp\":{},\"stop\":\"{:?}\",\"tokens\":{:?},\"mtp_rounds\":[{}]}}",
            prompt.len(), s.load.as_secs_f64(), s.prefill.as_secs_f64(), s.first_token.as_secs_f64(), s.total.as_secs_f64(),
            s.reserved_bytes, s.metal_bytes, s.metal_peak_bytes, s.file_bytes, s.cache_hits, s.cache_misses, s.read_time.as_secs_f64(), s.ple_read_time.as_secs_f64(), s.ple_prefill_read_time.as_secs_f64(), (s.ple_read_time - s.ple_prefill_read_time).as_secs_f64(), s.prefix_reused_tokens, s.prefix_saved_tokens, s.prefix_restore.as_secs_f64(), s.prefix_save.as_secs_f64(), s.workspace_allocations, s.workspace_reuses, s.workspace_bytes, s.workspace_peak_bytes, s.prefill_block_tokens, s.mtp_enabled,
            result.stop, result.tokens, s.mtp_rounds.iter().map(|r| format!(
                "{{\"proposed\":{},\"accepted\":{},\"history_proposed\":{},\"history_accepted\":{},\"committed\":{},\"verification_tokens\":{},\"replayed_tokens\":{},\"snapshot_bytes\":{},\"head_s\":{},\"verify_s\":{},\"snapshot_s\":{},\"snapshot_encode_s\":{},\"replay_s\":{},\"catch_up_s\":{},\"total_s\":{}}}",
                r.proposed,r.accepted,r.history_proposed,r.history_accepted,r.committed,r.verification_tokens,r.replayed_tokens,r.snapshot_bytes,
                r.head.as_secs_f64(),r.verify.as_secs_f64(),r.snapshot.as_secs_f64(),r.snapshot_encode.as_secs_f64(),r.replay.as_secs_f64(),r.catch_up.as_secs_f64(),r.total.as_secs_f64()
            )).collect::<Vec<_>>().join(",")
        );
    }
    Ok(())
}
