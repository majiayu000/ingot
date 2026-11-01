//! Real GLM5-Next weights; explicitly enabled and run one model at a time.
use ingot::engine::{Cancellation, Engine, Stop};
use std::path::PathBuf;
use std::time::{Duration, Instant};
#[test]
#[ignore = "requires GLM_MODEL metadata; no GPU or tensor payload needed"]
fn tokenizer_templates() {
    let path = PathBuf::from(std::env::var_os("GLM_MODEL").expect("GLM_MODEL"));
    let g = ingot::gguf::Gguf::open(&path).unwrap();
    let tk = ingot::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    // Golden IDs come from the published tokenizer.json and rendered GGUF Jinja template.
    let cases: &[(&str, &[u32])] = &[
        (
            "",
            &[
                154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 154828, 154841,
            ],
        ),
        (
            "你好，Rust！123456789",
            &[
                154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 109377, 3837, 49,
                590, 6313, 108714, 100461, 21, 100928, 24, 154828, 154841,
            ],
        ),
        (
            "  保留首尾空白  ",
            &[
                154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 220, 220, 103201,
                98861, 100453, 108052, 256, 154828, 154841,
            ],
        ),
        (
            "<|assistant|>测试",
            &[
                154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 154828, 100838,
                154828, 154841,
            ],
        ),
        (
            "写一段Python代码\n\tprint(1)",
            &[
                154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 99032, 101688,
                30163, 103461, 198, 6900, 7, 16, 8, 154828, 154841,
            ],
        ),
    ];
    for &(text, ids) in cases {
        assert_eq!(tk.chat(text).unwrap(), ids);
        assert_eq!(
            tk.decode(ids).unwrap(),
            format!(
                "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{text}<|assistant|><think>"
            )
        );
    }
    // GLM4 honors a full pre-token already present in the vocabulary.
    for (text, expected) in &[
        (" 截至", &[118450][..]),
        (" 形式", &[64721, 95, 98522][..]),
        (" 饰", &[105545][..]),
        (" 约", &[118835][..]),
        (" 地位", &[105106, 98476][..]),
        ("  \r\n", &[10414][..]),
    ] {
        assert_eq!(tk.encode(text, false).unwrap(), *expected);
        assert_eq!(tk.decode(expected).unwrap(), *text);
    }
    assert_eq!(tk.encode("\0\u{1}\u{7f}", false).unwrap(), [188, 189, 221]);
    assert_eq!(tk.decode(&[188, 189, 221]).unwrap(), "\0\u{1}\u{7f}");
    for id in [154820, 154827, 154829] {
        assert!(tk.is_eos(id));
    }
    assert!(!tk.is_eos(154828));
}
#[test]
#[ignore = "requires GLM_MODEL and Apple Silicon; 16 GiB total memory budget"]
fn state_cancellation_memory_and_reuse() {
    let path = PathBuf::from(std::env::var_os("GLM_MODEL").expect("GLM_MODEL"));
    assert!(Engine::load(&path, 8 * 1024 * 1024 * 1024, 64, None, false).is_err());
    let mut e = Engine::load(&path, 16 * 1024 * 1024 * 1024, 64, None, false).unwrap();
    for id in [154820, 154827, 154829] {
        assert!(e.tokenizer.is_eos(id));
    }
    assert!(!e.tokenizer.is_eos(154828));
    let ids = e.tokenizer.encode("你好，Rust！123456789", false).unwrap();
    let ids = &ids[..ids.len().min(8)];
    let mut s = e.session(Default::default()).unwrap();
    assert!(e.session(Default::default()).is_err());
    let mut sequential = Vec::new();
    for id in ids {
        sequential = e.consume(&mut s, &[*id]).unwrap();
    }
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    let cut = ids.len() / 2;
    e.consume(&mut s, &ids[..cut]).unwrap();
    let chunked = e.consume(&mut s, &ids[cut..]).unwrap();
    assert!(
        chunked == sequential,
        "four-row chunking changed GLM logits"
    );
    drop(s);
    // Eight rows select GEMM, while one/four rows use GEMV. Keep state
    // isolation comparisons on the same arithmetic path. Cross-path model
    // quality belongs to the independent numerical/task validation protocol.
    let mut s = e.session(Default::default()).unwrap();
    let clean_batch = e.consume(&mut s, ids).unwrap();
    assert!(clean_batch.iter().chain(&sequential).all(|v| v.is_finite()));
    let max_abs = clean_batch
        .iter()
        .zip(&sequential)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0_f64, f64::max);
    let diff_l2: f64 = clean_batch
        .iter()
        .zip(&sequential)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum();
    let ref_l2: f64 = sequential.iter().map(|v| f64::from(*v).powi(2)).sum();
    let top1 = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0
    };
    assert_eq!(top1(&clean_batch), top1(&sequential));
    println!(
        "GLM GEMM/GEMV diagnostic max_abs={max_abs:.9} rel_l2={:.9}",
        (diff_l2 / ref_l2).sqrt()
    );
    assert_eq!(s.consumed(), ids);
    drop(s);
    let mut s = e.session(Default::default()).unwrap();
    assert!(e.consume(&mut s, &vec![1; 65]).is_err());
    assert!(s.consumed().is_empty());
    assert!(e.consume(&mut s, &[u32::MAX]).is_err());
    assert!(e.consume(&mut s, &[1]).is_err());
    drop(s);
    for token_count in [1, 8] {
        let cancel = Cancellation::default();
        let mut s = e.session(cancel.clone()).unwrap();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            cancel.cancel();
        });
        let start = Instant::now();
        assert_eq!(
            e.consume(&mut s, &vec![1; token_count]).unwrap_err().kind(),
            std::io::ErrorKind::Interrupted
        );
        handle.join().unwrap();
        println!(
            "GLM in-flight cancellation tokens={token_count} elapsed={:?}",
            start.elapsed()
        );
        assert!(e.consume(&mut s, &[1]).is_err());
        drop(s);
    }
    let mut s = e.session(Default::default()).unwrap();
    assert!(
        e.consume(&mut s, ids).unwrap() == clean_batch,
        "cancelled session contaminated a fresh GLM session"
    );
    drop(s);
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert_eq!(
        e.generate(ids, 1, cancelled, |_, _| panic!("cancelled output"))
            .unwrap()
            .stop,
        Stop::Cancelled
    );
    let result = e
        .generate(ids, 1, Default::default(), |_, _| Ok(()))
        .unwrap();
    assert!(matches!(result.stop, Stop::Length | Stop::Eos));
    assert!(result.statistics.reserved_bytes <= 16 * 1024 * 1024 * 1024);
    println!(
        "GLM state/chunking/cancellation/reuse/budget passed; statistics={:?}",
        result.statistics
    );
    drop(e);
    let mut larger = Engine::load(&path, 24 * 1024 * 1024 * 1024, 64, None, false).unwrap();
    let mut s = larger.session(Default::default()).unwrap();
    assert!(
        larger.consume(&mut s, ids).unwrap() == clean_batch,
        "24 GiB cache changed GLM logits"
    );
    drop(s);
    drop(larger);
    // The recorded Q2_K_S artifact has fewer slots than its top-8 at 11 GiB.
    let mut smaller = Engine::load(&path, 11 * 1024 * 1024 * 1024, 64, None, false).unwrap();
    let mut s = smaller.session(Default::default()).unwrap();
    assert!(
        smaller.consume(&mut s, ids).unwrap() == clean_batch,
        "11 GiB cache changed GLM logits"
    );
    println!("GLM 11/16/24 GiB total memory budgets produced identical logits");
}
