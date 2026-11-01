use super::*;
use serde_json::json;

fn grammar(schema: Value) -> Arc<Grammar> {
    Arc::new(Grammar::schema(&schema).unwrap())
}
/// Whole text accepted and complete.
fn full(g: &Arc<Grammar>, text: &str) -> bool {
    let mut m = Matcher::new(g.clone());
    text.bytes().all(|b| m.step(b)) && m.complete()
}
/// Accepted as a prefix (possibly incomplete).
fn prefix(g: &Arc<Grammar>, text: &str) -> bool {
    Matcher::new(g.clone()).accepts(text.as_bytes())
}

#[test]
fn json_mode_accepts_objects_only() {
    let g = Arc::new(Grammar::object());
    for ok in [
        "{}",
        r#"{"a":1}"#,
        "{ \"a\" : [1, 2.5e-3, -0, true, false, null, \"x\\n\\u00e9\"] }",
        r#"{"nested":{"deep":[[{}],[]]},"s":"日本語 ✓"}"#,
        "{\n  \"a\": {\"b\": \"\\\"q\\\"\"}\n}",
    ] {
        assert!(full(&g, ok), "{ok}");
        assert!(serde_json::from_str::<Value>(ok).is_ok());
    }
    for bad in [
        "[]",
        "1",
        "\"s\"",
        "{a:1}",
        "{\"a\":01}",
        "{\"a\":1,}",
        "{\"a\":[1,]}",
        "{\"a\":tru}",
        "{\"a\":\"\\x\"}",
        "{\"a\":\"\x01\"}",
        "{\"a\":1.}",
        "{\"a\":1e}",
        "{} ",
        "{}{}",
    ] {
        assert!(!full(&g, bad), "{bad}");
    }
    // Incomplete but valid prefixes are accepted, not complete.
    assert!(prefix(&g, "{\"a\": [1, {\"b\": \"tex"));
    assert!(!full(&g, "{\"a\": [1"));
}

#[test]
fn schema_object_properties_and_required() {
    let g = grammar(json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "ok": {"type": "boolean"}
        },
        "required": ["name", "age"],
        "additionalProperties": false
    }));
    assert!(full(&g, r#"{"name":"a","age":3}"#));
    assert!(full(
        &g,
        r#"{"age":3,"name":"a","ok":true,"tags":["x","y"]}"#
    ));
    assert!(full(&g, "{\n  \"name\": \"a\",\n  \"age\": -12\n}"));
    for bad in [
        r#"{"name":"a"}"#,
        r#"{"name":"a","age":3.5}"#,
        r#"{"name":"a","age":3,"extra":1}"#,
        r#"{"name":"a","name":"b","age":1}"#,
        r#"{"name":1,"age":1}"#,
        r#"{"name":"a","age":1,"tags":["x","y","z"]}"#,
        r#"{"name":"a","age":1,"ok":null}"#,
        r#"{"nam":"a","age":1}"#,
    ] {
        assert!(!full(&g, bad), "{bad}");
    }
    // Once every property is used, only the closing brace may follow.
    let all = r#"{"name":"a","age":3,"ok":true,"tags":[]"#;
    assert!(prefix(&g, &format!("{all}}}")));
    assert!(!prefix(&g, &format!("{all},")));
}

#[test]
fn enums_unions_refs_and_numbers() {
    let g = grammar(json!({
        "type": "object",
        "properties": {
            "color": {"enum": ["red", "green", "re"]},
            "n": {"type": ["number", "null"]},
            "c": {"const": 12},
            "child": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}
        },
        "required": ["color", "n", "c", "child"],
        "$defs": {"node": {
            "type": "object",
            "properties": {"v": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
            "required": ["v", "next"]
        }}
    }));
    assert!(full(
        &g,
        r#"{"color":"re","n":null,"c":12,"child":{"v":1,"next":{"v":2,"next":null}}}"#
    ));
    assert!(full(
        &g,
        r#"{"color":"green","n":-1.5E+3,"c":12,"child":null}"#
    ));
    for bad in [
        r#"{"color":"blue","n":1,"c":12,"child":null}"#,
        r#"{"color":"red","n":1,"c":123,"child":null}"#,
        r#"{"color":"red","n":"1","c":12,"child":null}"#,
        r#"{"color":"red","n":1,"c":12,"child":{"v":1}}"#,
    ] {
        assert!(!full(&g, bad), "{bad}");
    }
    // Top-level scalars: complete numbers may still grow.
    let n = grammar(json!({"type": "number"}));
    let mut m = Matcher::new(n);
    assert!(m.step(b'4') && m.complete() && m.step(b'2') && m.complete());
    assert!(m.step(b'.') && !m.complete());
    let digits = "1".repeat(MAX_NUMBER as usize + 1);
    assert!(!full(&grammar(json!({"type": "integer"})), &digits));
}

#[test]
fn maps_arrays_bounds_whitespace_and_depth() {
    let g = grammar(json!({"type": "object", "additionalProperties": {"type": "integer"}}));
    assert!(full(&g, r#"{"any key":1,"other":2}"#));
    assert!(!full(&g, r#"{"a":"x"}"#));
    let g = grammar(
        json!({"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2}),
    );
    assert!(full(&g, "[true]") && full(&g, "[true, false]"));
    assert!(!full(&g, "[]") && !full(&g, "[true,true,true]"));
    let g = Arc::new(Grammar::object());
    let spaces = " ".repeat(MAX_SPACE as usize);
    assert!(prefix(&g, &format!("{{\n{spaces}")));
    assert!(!prefix(&g, &format!("{{\n{spaces} ")));
    // No whitespace before the root, no blank lines, tabs or CR, one space inline.
    for bad in [" {", "\n{", "{\n\n", "{ \n", "{\t", "{\r\n", "{  "] {
        assert!(!prefix(&g, bad), "{bad:?}");
    }
    assert!(full(&g, "{\n  \"a\": [\n    1, 2\n  ]\n}"));
    let deep = format!("{{\"a\":{}", "[".repeat(MAX_DEPTH - 1));
    assert!(prefix(&g, &deep));
    assert!(!prefix(&g, &format!("{deep}[")));
}

#[test]
fn rejects_unsupported_schemas() {
    for (schema, expected) in [
        (
            json!({"type": "string", "pattern": "a+"}),
            "keyword 'pattern'",
        ),
        (
            json!({"type": "object", "properties": {"a": {"type": "string", "format": "date"}}}),
            "#.properties.a: unsupported JSON Schema keyword 'format'",
        ),
        (json!({"type": "tuple"}), "unsupported type"),
        (json!({"$ref": "https://x/y"}), "$ref"),
        (
            json!({"anyOf": [{"type": "string"}, {"enum": ["a"]}]}),
            "start differently",
        ),
        (
            json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
            "not in properties",
        ),
        (
            json!({"type": "array", "minItems": 3, "maxItems": 1}),
            "minItems",
        ),
        (json!({"enum": []}), "enum"),
        (json!({"oneOf": []}), "oneOf"),
        (json!(false), "must be an object"),
    ] {
        let e = Grammar::schema(&schema).unwrap_err();
        assert!(e.contains(expected), "{schema}: {e}");
    }
    // Annotations and the OpenAI strict shape are accepted.
    assert!(
        Grammar::schema(&json!({"title": "t", "description": "d", "type": "object",
        "properties": {}, "additionalProperties": false}))
        .is_ok()
    );
}

#[test]
fn multibyte_utf8_is_validated() {
    let g = grammar(json!({"type": "string"}));
    let text = "\"é日😀\"";
    assert!(full(&g, text));
    let bytes = text.as_bytes();
    // A token boundary inside a character is fine; an invalid sequence is not.
    let mut m = Matcher::new(g.clone());
    assert!(bytes[..2].iter().all(|&b| m.step(b)) && !m.plain_string());
    assert!(!Matcher::new(g.clone()).accepts(b"\"\x80"));
    assert!(!Matcher::new(g.clone()).accepts(b"\"\xc3\x41"));
    assert!(!Matcher::new(g.clone()).accepts(b"\"\xff"));
    assert!(full(&g, "\"\\u00e9\\uFFFD\\ucafe\""));
    assert!(!prefix(&g, "\"\\ud83d"));
}
