//! Explicit real-Qwen tests; no model downloads and no GPU use in ordinary CI.
use ingot::engine::{Cancellation, DraftConfig, Engine, Stop};
use std::path::PathBuf;
#[test]
#[ignore = "requires QWEN_MODEL, QWEN_DRAFT, Apple Silicon and 44 GiB"]
fn real_draft_matches_autoregressive_and_reuses_clean_sessions() {
    let model = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
    let draft = PathBuf::from(std::env::var_os("QWEN_DRAFT").expect("QWEN_DRAFT"));
    let k = std::env::var("DRAFT_TOKENS")
        .unwrap_or("3".into())
        .parse()
        .unwrap();
    let budget = std::env::var("QWEN_MEMORY_GIB")
        .unwrap_or("44".into())
        .parse::<usize>()
        .unwrap()
        * (1usize << 30);
    let requests = [
        ("只回答一个数字：1+1等于几？", 24),
        ("写一个 Python 函数，返回两个整数的和。只输出代码。", 48),
        ("请按顺序列出一到十的中文数字，不作解释。", 48),
        ("用一句中文介绍你自己。", 48),
        ("原样复制：𐍈ǅ 你好 🦊 x≤y ∑α。", 48),
        ("只回答一个数字：1+1等于几？", 1),
        ("只回答一个数字：1+1等于几？", 2),
        ("只回答一个数字：1+1等于几？", 3),
    ];
    let mut reference = Vec::new();
    {
        let mut e = Engine::load(&model, budget, 256, None, false).unwrap();
        for (prompt, cap) in requests {
            let tokens = e.tokenizer.chat(prompt).unwrap();
            let generated = e
                .generate(&tokens, cap, Default::default(), |_, _| Ok(()))
                .unwrap();
            println!(
                "AR prompt={prompt:?} cap={cap} ids={:?} stop={:?}",
                generated.tokens, generated.stop
            );
            reference.push((tokens, cap, generated));
        }
    }
    let config = DraftConfig {
        model: draft,
        max_tokens: k,
        vocabulary: std::env::var_os("DRAFT_VOCAB").map(PathBuf::from),
    };
    assert!(Engine::load(&model, 1 << 30, 256, Some(config.clone()), false).is_err());
    let mut e = Engine::load(&model, budget, 256, Some(config), false).unwrap();
    let mut proposed = 0;
    let mut accepted = 0;
    for (ids, cap, expected) in &reference {
        let mut callbacks = Vec::new();
        let actual = e
            .generate(ids, *cap, Default::default(), |id, _| {
                callbacks.push(id);
                Ok(())
            })
            .unwrap();
        println!(
            "MTP k={k} cap={cap} ids={:?} stop={:?} stats={:?}",
            actual.tokens, actual.stop, actual.statistics
        );
        assert_eq!(actual.tokens, expected.tokens);
        assert_eq!(actual.stop, expected.stop);
        assert_eq!(callbacks, actual.tokens);
        assert!(actual.statistics.mtp_enabled);
        assert!(actual.statistics.reserved_bytes <= budget);
        assert!(actual.statistics.metal_bytes <= budget);
        assert!(actual.statistics.metal_peak_bytes <= budget);
        proposed += actual
            .statistics
            .mtp_rounds
            .iter()
            .map(|r| r.proposed)
            .sum::<usize>();
        accepted += actual
            .statistics
            .mtp_rounds
            .iter()
            .map(|r| r.accepted)
            .sum::<usize>();
    }
    assert!(
        proposed > 0 && accepted > 0,
        "the real draft must be executed and accepted"
    );
    for (ids, cap, expected) in reference.iter().take(2).rev() {
        let again = e
            .generate(ids, *cap, Default::default(), |_, _| Ok(()))
            .unwrap();
        assert_eq!(again.tokens, expected.tokens);
        assert_eq!(again.stop, expected.stop);
    }
    let ids = &reference[2].0;
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        e.generate(ids, 24, cancel, |_, _| panic!("cancelled callback"))
            .unwrap()
            .stop,
        Stop::Cancelled
    );
    let signal = Cancellation::default();
    let copy = signal.clone();
    let mut count = 0;
    let cancelled = e
        .generate(ids, 24, signal, |_, _| {
            count += 1;
            if count == 3 {
                copy.cancel();
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(cancelled.stop, Stop::Cancelled);
    assert_eq!(cancelled.tokens.len(), 3);
    let error = e
        .generate(ids, 24, Default::default(), |_, _| {
            Err(std::io::Error::other("callback failed"))
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "callback failed");
    let clean = e
        .generate(ids, reference[2].1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(clean.tokens, reference[2].2.tokens);
    println!(
        "PASS real MTP k={k}: proposed={proposed} accepted={accepted}; exact IDs, EOS, caps, callbacks, cancellation, repeated/alternating sessions"
    );
}

#[test]
#[ignore = "requires QWEN_MODEL, optional QWEN_DRAFT and Apple Silicon"]
fn confirmed_prefix_matches_four_uncached_conversation_rounds() {
    let model = PathBuf::from(std::env::var_os("QWEN_MODEL").unwrap());
    let tokenizer =
        ingot::tokenizer::Tokenizer::from_gguf(&ingot::gguf::Gguf::open(&model).unwrap()).unwrap();
    // A deterministic token prefix, independent of any external evidence file.
    // This checks cache/state equivalence, not language quality.
    let text = "推理回归需要核对计算结果、缓存状态、内存预算与请求隔离。\n".repeat(100);
    let mut first = tokenizer.chat(&text).unwrap();
    first.truncate(1024);
    assert_eq!(first.len(), 1024);
    let budget = std::env::var("QWEN_MEMORY_GIB")
        .unwrap_or("44".into())
        .parse::<usize>()
        .unwrap()
        << 30;
    let draft = std::env::var_os("QWEN_DRAFT").map(|model| DraftConfig {
        model: model.into(),
        max_tokens: 3,
        vocabulary: std::env::var_os("DRAFT_VOCAB").map(Into::into),
    });
    let mut references = Vec::new();
    {
        let mut e = Engine::load(&model, budget, 1792, draft.clone(), false).unwrap();
        let mut prompt = first;
        for question in [
            "再用一句话说明原因。",
            "给出一个具体例子。",
            "指出一个限制。",
            "最后给出一句结论。",
        ] {
            let result = e
                .generate(&prompt, 24, Default::default(), |_, _| Ok(()))
                .unwrap();
            references.push((prompt.clone(), result.tokens.clone(), result.stop));
            prompt.extend(result.tokens);
            prompt.extend(e.tokenizer.encode(&format!("<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), true).unwrap());
        }
    }
    let mut e = Engine::load(&model, budget, 1792, draft, true).unwrap();
    let mut checkpoint = 0;
    for (round, (prompt, tokens, stop)) in references.iter().enumerate() {
        let actual = e
            .generate(prompt, 24, Default::default(), |_, _| Ok(()))
            .unwrap();
        assert_eq!(&actual.tokens, tokens, "round={round}");
        assert_eq!(&actual.stop, stop);
        if round == 0 {
            checkpoint = actual.statistics.prefix_saved_tokens;
            assert!(checkpoint > 0);
        }
        assert_eq!(
            actual.statistics.prefix_reused_tokens,
            if round == 0 { 0 } else { checkpoint }
        );
        assert!(actual.statistics.metal_peak_bytes <= budget);
        println!(
            "PREFIX round={round} input={} reused={} saved={} total_s={} restore_s={} save_s={} peak={}",
            prompt.len(),
            actual.statistics.prefix_reused_tokens,
            actual.statistics.prefix_saved_tokens,
            actual.statistics.total.as_secs_f64(),
            actual.statistics.prefix_restore.as_secs_f64(),
            actual.statistics.prefix_save.as_secs_f64(),
            actual.statistics.metal_peak_bytes
        );
    }
    let short = e.tokenizer.chat("只回答数字：1+1等于几？").unwrap();
    let miss = e
        .generate(&short, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(miss.statistics.prefix_reused_tokens, 0);
    assert_eq!(miss.statistics.prefix_saved_tokens, short.len());
    let mut extended = short.clone();
    extended.push(198);
    let extend = e
        .generate(&extended, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(
        extend.statistics.prefix_reused_tokens, 0,
        "a partial block cannot be reused inside a larger GEMM"
    );
    let hit = e
        .generate(&extended, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(hit.tokens, extend.tokens);
    assert_eq!(hit.statistics.prefix_reused_tokens, extended.len());
    let cancel = Cancellation::default();
    cancel.cancel();
    let cancelled = e
        .generate(&extended, 1, cancel, |_, _| panic!("cancelled callback"))
        .unwrap();
    assert_eq!(cancelled.stop, Stop::Cancelled);
    assert_eq!(cancelled.statistics.prefix_reused_tokens, 0);
    let error = e
        .generate(&extended, 1, Default::default(), |_, _| {
            Err(std::io::Error::other("prefix callback failed"))
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "prefix callback failed");
    let clean = e
        .generate(&extended, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(clean.tokens, hit.tokens);
    assert_eq!(clean.statistics.prefix_reused_tokens, extended.len());
}
