//! Explicit real-GLM tests; no model downloads and no GPU use in ordinary CI.
use ingot::engine::{Cancellation, DraftConfig, Engine, Stop};
use std::path::PathBuf;
#[test]
#[ignore = "requires GLM_MODEL, GLM_DRAFT, Apple Silicon and 32 GiB"]
fn real_draft_matches_autoregressive_and_reuses_clean_sessions() {
    let model = PathBuf::from(std::env::var_os("GLM_MODEL").expect("GLM_MODEL"));
    let draft = PathBuf::from(std::env::var_os("GLM_DRAFT").expect("GLM_DRAFT"));
    let k = std::env::var("DRAFT_TOKENS")
        .unwrap_or("3".into())
        .parse()
        .unwrap();
    let budget = std::env::var("GLM_MEMORY_GIB")
        .unwrap_or("32".into())
        .parse::<usize>()
        .unwrap()
        * (1usize << 30);
    let requests = [
        ("只回答一个数字：1+1等于几？", 128),
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
    assert!(
        reference
            .iter()
            .any(|(_, _, result)| result.stop == Stop::Eos),
        "real GLM fixture must reach EOS; truncated requests do not prove EOS handling"
    );
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
#[ignore = "requires GLM_MODEL, optional GLM_DRAFT and an explicit Apple GPU budget"]
fn glm_one_prefix_preserves_four_rounds_and_request_isolation() {
    use ingot::{gguf::Gguf, tokenizer::Tokenizer};
    let model = PathBuf::from(std::env::var_os("GLM_MODEL").unwrap());
    let budget = std::env::var("GLM_MEMORY_GIB")
        .unwrap_or("32".into())
        .parse::<usize>()
        .unwrap()
        << 30;
    let draft = std::env::var_os("GLM_DRAFT").map(|model| DraftConfig {
        model: model.into(),
        max_tokens: 3,
        vocabulary: std::env::var_os("DRAFT_VOCAB").map(Into::into),
    });
    let tok = Tokenizer::from_gguf(&Gguf::open(&model).unwrap()).unwrap();
    let text =
        "材料：推理优化需要同时验证计算结果、缓存状态、内存预算和实际请求时间。\n".repeat(70);
    let mut prompt = tok
        .chat(&format!("{text}\n请用一句话概括这段材料。"))
        .unwrap();
    assert!(prompt.len() > 1024 && prompt.len() < 3000);
    let mut reference = Vec::new();
    {
        let mut engine = Engine::load(&model, budget, 4096, draft.clone(), false).unwrap();
        for question in [
            "再说明一个原因。",
            "举一个具体例子。",
            "指出一个限制。",
            "用一句话总结。",
        ] {
            let result = engine
                .generate(&prompt, 24, Default::default(), |_, _| Ok(()))
                .unwrap();
            reference.push((prompt.clone(), result.tokens.clone(), result.stop));
            prompt.extend(result.tokens);
            prompt.extend(
                tok.encode(&format!("<|user|>{question}<|assistant|><think>"), true)
                    .unwrap(),
            );
        }
    }
    let mut engine = Engine::load(&model, budget, 4096, draft, true).unwrap();
    for (round, (ids, tokens, stop)) in reference.iter().enumerate() {
        let actual = engine
            .generate(ids, 24, Default::default(), |_, _| Ok(()))
            .unwrap();
        assert_eq!(&actual.tokens, tokens, "round={round}");
        assert_eq!(&actual.stop, stop);
        assert_eq!(
            actual.statistics.prefix_reused_tokens,
            if round == 0 { 0 } else { 1024 }
        );
        assert!(actual.statistics.metal_peak_bytes <= budget);
        assert!(actual.statistics.reserved_bytes <= budget);
        println!(
            "GLM PREFIX round={round} reused={} total_s={} peak={}",
            actual.statistics.prefix_reused_tokens,
            actual.statistics.total.as_secs_f64(),
            actual.statistics.metal_peak_bytes
        );
    }
    let (ids, tokens, stop) = &reference[0];
    let signal = Cancellation::default();
    signal.cancel();
    assert_eq!(
        engine
            .generate(ids, 24, signal, |_, _| panic!("cancelled callback"))
            .unwrap()
            .stop,
        Stop::Cancelled
    );
    let signal = Cancellation::default();
    let copy = signal.clone();
    let mut outputs = 0;
    let cancelled = engine
        .generate(ids, 24, signal, |_, _| {
            outputs += 1;
            if outputs == 3 {
                copy.cancel();
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(cancelled.stop, Stop::Cancelled);
    assert_eq!(cancelled.tokens.len(), 3);
    assert!(
        engine
            .generate(ids, 24, Default::default(), |_, _| Err(
                std::io::Error::other("intentional callback failure")
            ))
            .is_err()
    );
    let clean = engine
        .generate(ids, 24, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(&clean.tokens, tokens);
    assert_eq!(&clean.stop, stop);
    let short = tok.chat("只回答数字：1+1等于几？").unwrap();
    let miss = engine
        .generate(&short, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(miss.statistics.prefix_reused_tokens, 0);
    let mut longer = short.clone();
    longer.push(198);
    let miss = engine
        .generate(&longer, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(miss.statistics.prefix_reused_tokens, 0);
    let hit = engine
        .generate(&longer, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(hit.tokens, miss.tokens);
    assert_eq!(hit.statistics.prefix_reused_tokens, longer.len());
}

#[test]
#[ignore = "requires GLM_MODEL, GLM_DRAFT and 32 GiB; records actual history proposals"]
fn glm_confirmed_history_proposals_still_match_target_generation() {
    let model = PathBuf::from(std::env::var_os("GLM_MODEL").unwrap());
    let draft = DraftConfig {
        model: std::env::var_os("GLM_DRAFT").unwrap().into(),
        max_tokens: 3,
        vocabulary: None,
    };
    let text = "不要解释，不需要推导。原样复制下面的 Python 代码，保留所有行：\n```python\ndef add_one(x):\n    return x + 1\n\ndef add_two(x):\n    return x + 2\n\ndef add_three(x):\n    return x + 3\n```";
    let (ids, expected) = {
        let mut engine = Engine::load(&model, 32 << 30, 1024, None, false).unwrap();
        let ids = engine.tokenizer.chat(text).unwrap();
        let output = engine
            .generate(&ids, 192, Default::default(), |_, _| Ok(()))
            .unwrap();
        (ids, output)
    };
    let mut engine = Engine::load(&model, 32 << 30, 1024, Some(draft), false).unwrap();
    let actual = engine
        .generate(&ids, 192, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(actual.tokens, expected.tokens);
    assert_eq!(actual.stop, expected.stop);
    let proposed = actual
        .statistics
        .mtp_rounds
        .iter()
        .map(|r| r.history_proposed)
        .sum::<usize>();
    let accepted = actual
        .statistics
        .mtp_rounds
        .iter()
        .map(|r| r.history_accepted)
        .sum::<usize>();
    println!(
        "GLM HISTORY proposed={proposed} accepted={accepted} output={} exact=true",
        actual.tokens.len()
    );
    assert!(
        proposed > 0,
        "history integration path not exercised by this probe"
    );
}
