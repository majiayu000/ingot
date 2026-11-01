use super::*;
use serde_json::json;

/// A tiny vocabulary: structural pieces, multi-byte and merged tokens.
const PIECES: [&[u8]; 16] = [
    b"<eos>",
    b"{",
    b"}",
    b"\"",
    b"name",
    b"\":",
    b" \"",
    b"Bob",
    b"\"}",
    b"```json",
    b"\xe6\x97",
    b"\xa5",
    b"age",
    b"1",
    b"0",
    b"\n",
];
const EOS: u32 = 0;

fn vocab() -> Vocabulary {
    let mut special = vec![false; PIECES.len()];
    special[EOS as usize] = true;
    Vocabulary::build(&PIECES, special, vec![EOS])
}
fn constraint(schema: Value) -> JsonConstraint {
    JsonConstraint::with(vocab(), Arc::new(Grammar::schema(&schema).unwrap()), false)
}
fn allowed(c: &mut JsonConstraint) -> Vec<u32> {
    c.fill(PIECES.len());
    (0..PIECES.len() as u32)
        .filter(|&i| c.allow[i as usize])
        .collect()
}
fn greedy() -> Sampler {
    Sampler::new(crate::sampling::Sampling::default()).unwrap()
}
/// Logits that prefer `ids` in order, then everything else.
fn prefer(ids: &[u32]) -> Vec<f32> {
    let mut l = vec![0.; PIECES.len()];
    for (rank, &id) in ids.iter().enumerate() {
        l[id as usize] = 10. - rank as f32;
    }
    l
}

#[test]
fn masks_follow_the_grammar() {
    let mut c = constraint(
        json!({"type": "object", "properties": {"name": {"type": "string"}},
        "required": ["name"], "additionalProperties": false}),
    );
    // Start: only `{` (no whitespace before the root).
    assert_eq!(allowed(&mut c), [1]);
    let mut s = greedy();
    // The model wants a code fence; the mask turns it into `{`.
    assert_eq!(c.choose(&mut prefer(&[9, 1]), &mut s).unwrap(), 1);
    assert_eq!(c.masks, 1);
    assert_eq!(allowed(&mut c), [3, 6, 15]);
    for id in [3, 4, 5, 6] {
        assert_eq!(c.choose(&mut prefer(&[id]), &mut s).unwrap(), id);
    }
    // Inside the string: plain text, split UTF-8 and the closing pieces; no EOS.
    assert_eq!(allowed(&mut c), [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]);
    assert_eq!(c.choose(&mut prefer(&[10]), &mut s).unwrap(), 10);
    // Mid-character only the continuation byte fits.
    assert_eq!(allowed(&mut c), [11]);
    for id in [11, 8] {
        assert_eq!(c.choose(&mut prefer(&[EOS, id]), &mut s).unwrap(), id);
    }
    // Complete: only EOS.
    assert_eq!(allowed(&mut c), [EOS]);
    assert_eq!(c.choose(&mut prefer(&[15]), &mut s).unwrap(), EOS);
}

#[test]
fn sampling_never_leaves_the_grammar() {
    let schema = json!({"type": "object", "properties": {"age": {"type": "integer"},
        "name": {"type": "string"}}, "required": ["age"]});
    // Byte-level BPE vocabularies hold every single byte, so no state is a dead end.
    let singles: Vec<[u8; 1]> = (0x20u8..0x7f).map(|b| [b]).collect();
    let mut pieces = PIECES.to_vec();
    pieces.extend(singles.iter().map(|b| b.as_slice()));
    let n = pieces.len();
    for seed in 0..50 {
        let mut special = vec![false; n];
        special[EOS as usize] = true;
        let v = Vocabulary::build(&pieces, special, vec![EOS]);
        let grammar = Arc::new(Grammar::schema(&schema).unwrap());
        let mut c = JsonConstraint::with(v, grammar, false);
        let config = crate::sampling::Sampling {
            temperature: 1.5,
            seed,
            ..Default::default()
        };
        let mut s = Sampler::new(config).unwrap();
        let mut text = vec![];
        for _ in 0..200 {
            let id = c.choose(&mut vec![0.; n], &mut s).unwrap();
            if id == EOS {
                break;
            }
            text.extend_from_slice(pieces[id as usize]);
        }
        assert!(c.steps < 200 || !c.matcher.complete());
        if c.matcher.complete() {
            let v: Value = serde_json::from_slice(&text).unwrap();
            // "-0" parses as a float but is integer-valued.
            assert!(v["age"].as_f64().is_some_and(|x| x.fract() == 0.), "{v}");
        }
    }
}

#[test]
fn reasoning_runs_free_until_the_marker() {
    let pieces: [&[u8]; 4] = [b"<eos>", b"think", b"</think>", b"{}"];
    let v = Vocabulary::build(&pieces, vec![true, false, false, false], vec![0]);
    let mut c = JsonConstraint::with(v, Arc::new(Grammar::object()), true);
    let mut s = greedy();
    // EOS is not allowed before the answer; free text is.
    assert_eq!(c.choose(&mut vec![9., 8., 0., 0.], &mut s).unwrap(), 1);
    assert_eq!(c.choose(&mut vec![0., 0., 9., 0.], &mut s).unwrap(), 2);
    assert_eq!(c.choose(&mut vec![0., 9., 0., 1.], &mut s).unwrap(), 3);
    assert_eq!(c.choose(&mut vec![0., 9., 0., 1.], &mut s).unwrap(), 0);
}

#[test]
fn parses_request_formats() {
    let fields = |v: Value| v.as_object().unwrap().clone();
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    assert!(openai(&fields(json!({}))).unwrap().is_none());
    assert!(
        openai(&fields(json!({"response_format": {"type": "text"}})))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        openai(&fields(json!({"response_format": {"type": "json_object"}}))).unwrap(),
        Some(Arc::new(Grammar::object()))
    );
    let rf = json!({"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": schema}});
    assert!(
        openai(&fields(json!({"response_format": rf})))
            .unwrap()
            .is_some()
    );
    for (body, expected) in [
        (
            json!({"response_format": {"type": "xml"}}),
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {}}}),
            "needs a schema",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "string", "minLength": 1}}}}),
            "'minLength'",
        ),
        (
            json!({"response_format": {"type": "json_object"}, "tools": [{}]}),
            "tools",
        ),
    ] {
        let e = openai(&fields(body.clone())).unwrap_err();
        assert_eq!(
            (e.status, e.param),
            (400, Some("response_format")),
            "{body}"
        );
        assert!(e.message.contains(expected), "{body}: {}", e.message);
    }
    let ga = json!({"output_config": {"effort": "high", "format": {"type": "json_schema", "schema": schema}}});
    assert!(anthropic(&fields(ga)).unwrap().is_some());
    let beta = json!({"output_format": {"type": "json_schema", "schema": schema}});
    assert!(anthropic(&fields(beta)).unwrap().is_some());
    assert!(
        anthropic(&fields(json!({"output_config": {"effort": "high"}})))
            .unwrap()
            .is_none()
    );
    let e = anthropic(&fields(json!({"output_format": {"type": "json_object"}}))).unwrap_err();
    assert!(e.message.contains("json_schema"));
}
