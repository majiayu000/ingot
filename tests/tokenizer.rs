//! Real vocabulary stop-token contract; CPU-only, no model weights loaded.
use ingot::{gguf::Gguf, tokenizer::Tokenizer};
use std::path::PathBuf;

#[test]
#[ignore = "requires QWEN_MODEL metadata; no GPU commands"]
fn qwen_generation_terminators_match_reference_vocabulary() {
    let path = PathBuf::from(std::env::var_os("QWEN_MODEL").expect("QWEN_MODEL"));
    let g = Gguf::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let eog: Vec<_> = (0..g.strings("tokenizer.ggml.tokens").unwrap().len() as u32)
        .filter(|&id| tok.is_eos(id))
        .collect();
    // Independently recorded from the frozen reference runtime for this model.
    assert_eq!(eog, [248044, 248046, 248063, 248064, 248065]);
    assert!(!tok.is_eos(248062)); // fim_suffix must remain an ordinary input delimiter.
    assert_eq!(tok.encode("<|endoftext|>", true).unwrap(), [248044]);
    assert_eq!(
        tok.chat(" 你好 ").unwrap(),
        [
            248045, 846, 198, 109266, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
        ]
    );
}

#[test]
#[ignore = "requires GLM_MODEL metadata; no GPU commands"]
fn glm_generation_terminators_match_reference_vocabulary() {
    let path = PathBuf::from(std::env::var_os("GLM_MODEL").expect("GLM_MODEL"));
    let g = Gguf::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let eog: Vec<_> = (0..g.strings("tokenizer.ggml.tokens").unwrap().len() as u32)
        .filter(|&id| tok.is_eos(id))
        .collect();
    assert_eq!(eog, [154820, 154827, 154829]);
    assert!(!tok.is_eos(154840)); // GLM code_suffix is not an EOG marker either.
}

/// Multi-turn rendering stays token-aligned with single-turn `chat()` and with
/// the incremental continuation the chat CLI uses to reuse its session.
#[test]
#[ignore = "requires QWEN_MODEL and GLM_MODEL metadata; no GPU commands"]
fn multi_turn_tokens_extend_single_turn() {
    use ingot::tokenizer::{Message, Role};
    for var in ["QWEN_MODEL", "GLM_MODEL"] {
        let path = PathBuf::from(std::env::var_os(var).expect(var));
        let tok = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
        let reply = if tok.is_glm() {
            "先想一想。</think>你好！有什么可以帮你？"
        } else {
            "你好！有什么可以帮你？"
        };
        let first = tok.chat("你好").unwrap();
        let all = tok
            .chat_messages(&[
                Message::user("你好"),
                Message::new(Role::Assistant, reply),
                Message::user("Tell me a joke."),
            ])
            .unwrap();
        let incremental = [
            first.as_slice(),
            &tok.encode_continuation(
                &(reply.to_owned() + &tok.chat_continuation("Tell me a joke.").unwrap()),
            )
            .unwrap(),
        ]
        .concat();
        assert_eq!(all, incremental, "{var}");
    }
}
