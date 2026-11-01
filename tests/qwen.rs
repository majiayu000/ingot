//! Real-weight integration. Run explicitly; ordinary unit tests need no model files.
use ingot::engine::{Cancellation, Engine, Stop};
use std::path::PathBuf;
use std::time::{Duration, Instant};
#[test]
#[ignore = "requires QWEN_MODEL and Apple Silicon; loads an 8 GiB expert cache"]
fn state_cancellation_memory_and_reuse() {
    let path = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
    assert!(Engine::load(&path, 1024 * 1024 * 1024, 64, None, false).is_err());
    let mut e = Engine::load(&path, 8 * 1024 * 1024 * 1024, 64, None, false).unwrap();
    assert_eq!(
        e.tokenizer.encode("\0\u{1}\u{7f}", false).unwrap(),
        [188, 189, 221]
    );
    assert_eq!(
        e.tokenizer.decode(&[188, 189, 221]).unwrap(),
        "\0\u{1}\u{7f}"
    );
    assert_eq!(
        e.tokenizer.chat(" 你好 ").unwrap(),
        [
            248045, 846, 198, 109266, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
        ]
    );
    let ids = [1, 109266, 248044, 96748];
    let mut s = e.session(Default::default()).unwrap();
    assert!(e.session(Default::default()).is_err());
    let first = e.consume(&mut s, &ids[..2]).unwrap();
    let last = e.consume(&mut s, &ids[2..]).unwrap();
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    let again = e.consume(&mut s, &ids).unwrap();
    assert_eq!(last, again);
    assert_eq!(s.consumed(), ids);
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    assert!(e.consume(&mut s, &vec![1; 65]).is_err());
    assert!(s.consumed().is_empty());
    assert!(e.consume(&mut s, &[u32::MAX]).is_err());
    assert!(e.consume(&mut s, &[1]).is_err());
    drop(s);
    let cancel = Cancellation::default();
    let mut s = e.session(cancel.clone()).unwrap();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(5));
        cancel.cancel();
    });
    let start = Instant::now();
    assert_eq!(
        e.consume(&mut s, &[1]).unwrap_err().kind(),
        std::io::ErrorKind::Interrupted
    );
    handle.join().unwrap();
    println!("in-flight cancellation exited after {:?}", start.elapsed());
    assert!(e.consume(&mut s, &[1]).is_err());
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    assert_eq!(e.consume(&mut s, &ids[..2]).unwrap(), first);
    drop(s);
    let block = [1, 109266, 248044, 96748, 1, 109266, 248044, 96748, 1];
    let mut s = e.session(Default::default()).unwrap();
    let mut sequential = vec![];
    for token in block {
        sequential = e.consume(&mut s, &[token]).unwrap();
    }
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    assert_eq!(e.consume(&mut s, &block).unwrap(), sequential);
    drop(s);
    let cancel = Cancellation::default();
    let signal = cancel.clone();
    let result = e
        .generate(&ids, 3, cancel, move |_, _| {
            signal.cancel();
            Ok(())
        })
        .unwrap();
    assert_eq!(result.stop, Stop::Cancelled);
    assert_eq!(result.tokens.len(), 1);
    assert!(result.statistics.reserved_bytes <= 8 * 1024 * 1024 * 1024);
    let error = e
        .generate(&ids, 1, Default::default(), |_, _| {
            Err(std::io::Error::other("output failed"))
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "output failed");
    let result = e
        .generate(&ids, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert_eq!(result.stop, Stop::Length);
    assert_eq!(result.tokens.len(), 1);
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert_eq!(
        e.generate(&ids, 1, cancelled, |_, _| panic!("cancelled output"))
            .unwrap()
            .stop,
        Stop::Cancelled
    );
    println!(
        "state/chunking/cancellation/reuse/budget/error contracts passed; statistics={:?}",
        result.statistics
    );
}

#[test]
#[ignore = "requires QWEN_MODEL; complete 8192-token, 8 GiB context smoke test"]
fn full_8192_context() {
    let path = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
    let mut e = Engine::load(&path, 8 * 1024 * 1024 * 1024, 8192, None, false).unwrap();
    let phrase = e
        .tokenizer
        .encode(
            "Rust 引擎按顺序处理输入，Metal 负责矩阵运算。记录每个位置的状态与数值。\n",
            false,
        )
        .unwrap();
    let ids: Vec<_> = phrase.iter().copied().cycle().take(8192).collect();
    let mut session = e.session(Default::default()).unwrap();
    let start = Instant::now();
    for block in ids.chunks(512) {
        let logits = e.consume(&mut session, block).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        println!(
            "long-context position={} elapsed={:?} top={}",
            session.consumed().len(),
            start.elapsed(),
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        );
    }
    assert_eq!(session.consumed(), ids);
    assert!(e.consume(&mut session, &[1]).is_err());
    assert_eq!(session.consumed().len(), 8192);
}

#[test]
#[ignore = "requires QWEN_MODEL; resident prefill comparison with alternating order"]
fn prefill_comparison() {
    let path = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
    let mut e = Engine::load(&path, 64 * 1024 * 1024 * 1024, 8192, None, false).unwrap();
    let ids = e.tokenizer.chat("用一句中文介绍你自己。").unwrap();
    let mut reference = None;
    for (sample, batched) in [true, false, false, true, true, false]
        .into_iter()
        .enumerate()
    {
        let mut s = e.session(Default::default()).unwrap();
        let start = Instant::now();
        let logits = if batched {
            e.consume(&mut s, &ids).unwrap()
        } else {
            let mut logits = vec![];
            for token in &ids {
                logits = e.consume(&mut s, &[*token]).unwrap();
            }
            logits
        };
        if let Some(expected) = &reference {
            assert_eq!(&logits, expected);
        } else {
            reference = Some(logits);
        }
        println!(
            "prefill sample={sample} batched={batched} tokens={} elapsed={:?}",
            ids.len(),
            start.elapsed()
        );
    }
}
