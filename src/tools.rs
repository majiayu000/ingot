//! Qwen tool calling: the GGUF chat template's tool prompt, rendering of
//! assistant tool calls in history, and parsing of generated `<tool_call>`
//! blocks; also Llama 3.x JSON calls (a bare `{"name": ..., "parameters": ...}`
//! reply or one after `<|python_tag|>`).
use serde_json::{Map, Value};

pub const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
/// Llama 3.1 marks tool-call replies with this token.
pub const PYTHON_TAG: &str = "<|python_tag|>";
const INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// Jinja `tojson` (Python `json.dumps` defaults with `ensure_ascii=False`):
/// `", "` and `": "` separators, keys in their original order.
pub fn tojson(v: &Value) -> String {
    match v {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(tojson).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), tojson(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// System turn when tools are given; `system` is the merged system text.
pub fn system_prompt(tools: &[Value], system: &str) -> String {
    let mut out =
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
            .to_owned();
    for tool in tools {
        out += "\n";
        out += &tojson(tool);
    }
    out += "\n</tools>";
    out += INSTRUCTIONS;
    if !system.is_empty() {
        out += "\n\n";
        out += system;
    }
    out + "<|im_end|>\n"
}

/// Tool calls appended to an assistant turn whose trimmed content is `content`.
pub fn render_calls(content: &str, calls: &[ToolCall]) -> String {
    let mut out = String::new();
    for (i, call) in calls.iter().enumerate() {
        if i > 0 {
            out += "\n";
        } else if !content.is_empty() {
            out += "\n\n";
        }
        out += &format!("{CALL_START}\n<function={}>\n", call.name);
        for (name, value) in &call.arguments {
            let value = match value {
                Value::String(s) => s.clone(),
                v => tojson(v),
            };
            out += &format!("<parameter={name}>\n{value}\n</parameter>\n");
        }
        out += &format!("</function>\n{CALL_END}");
    }
    out
}

/// JSON schema type of a tool parameter, if declared.
fn parameter_type<'a>(tools: &'a [Value], function: &str, parameter: &str) -> Option<&'a str> {
    tools
        .iter()
        .map(|t| t.get("function").unwrap_or(t))
        .find(|f| f.get("name").and_then(Value::as_str) == Some(function))?
        .pointer(&format!("/parameters/properties/{parameter}/type"))
        .and_then(Value::as_str)
}

/// Hermes-style JSON call (Qwen2.5/Qwen3): `{"name": ..., "arguments": {...}}`;
/// arguments may also be a JSON string of an object, or named `parameters`.
fn json_call(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_owned();
    let arguments = match v.get("arguments").or_else(|| v.get("parameters")) {
        None => Map::new(),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok()?.as_object()?.clone(),
        Some(a) => a.as_object()?.clone(),
    };
    Some(ToolCall { name, arguments })
}

/// Start of a tool call in answer text: the first `<tool_call>` or
/// `<|python_tag|>`, or, when `fresh` (no answer text before `text`), a
/// leading bare JSON object whose first key is `"name"` (Llama 3.x).
pub fn call_start(text: &str, fresh: bool) -> Option<usize> {
    let lead = text.len() - text.trim_start().len();
    let bare = (fresh && bare_call(&text[lead..]) == Some(true)).then_some(0);
    [text.find(CALL_START), text.find(PYTHON_TAG), bare]
        .into_iter()
        .flatten()
        .min()
}
/// Whether text that is all answer so far may still become a bare JSON call.
pub fn may_start_bare_call(text: &str) -> bool {
    bare_call(text.trim_start()).is_none()
}
/// Length of the longest suffix of `text` that is a proper prefix of `marker`.
pub fn holdback(text: &str, marker: &str) -> usize {
    (1..marker.len().min(text.len() + 1))
        .rev()
        .find(|&k| {
            text.is_char_boundary(text.len() - k) && marker.starts_with(&text[text.len() - k..])
        })
        .unwrap_or(0)
}
/// Bytes at the end of answer text to hold back while they may start a call.
pub fn call_holdback(text: &str, fresh: bool) -> usize {
    if fresh && may_start_bare_call(text) {
        return text.len();
    }
    holdback(text, CALL_START).max(holdback(text, PYTHON_TAG))
}
/// Some(true) for `{"name"...`, None while `text` is a prefix of that, else Some(false).
fn bare_call(text: &str) -> Option<bool> {
    let Some(rest) = text.strip_prefix('{') else {
        return if text.is_empty() { None } else { Some(false) };
    };
    let rest = rest.trim_start();
    if rest.starts_with("\"name\"") {
        Some(true)
    } else if "\"name\"".starts_with(rest) {
        None
    } else {
        Some(false)
    }
}

/// Llama 3.x JSON calls: objects separated by whitespace or `;`.
fn json_calls(body: &str) -> Option<Vec<ToolCall>> {
    let mut calls = vec![];
    for value in serde_json::Deserializer::from_str(&body.replace(';', " ")).into_iter::<Value>() {
        calls.push(json_call(&value.ok()?.to_string())?);
    }
    (!calls.is_empty()).then_some(calls)
}

/// Parse generated text that starts at the first `<tool_call>`. Parameter
/// values are strings unless the tool schema declares a non-string type and
/// the value parses as JSON. Returns None if any block is malformed.
pub fn parse_calls(text: &str, tools: &[Value]) -> Option<Vec<ToolCall>> {
    let text = text.trim();
    if let Some(body) = text.strip_prefix(PYTHON_TAG) {
        return json_calls(body);
    }
    if text.starts_with('{') {
        return json_calls(text);
    }
    let mut calls = vec![];
    let mut rest = text;
    while !rest.is_empty() {
        let body = rest.strip_prefix(CALL_START)?;
        let (body, after) = body.split_once(CALL_END)?;
        rest = after.trim_start();
        let body = body.trim();
        if body.starts_with('{') {
            calls.push(json_call(body)?);
            continue;
        }
        let body = body.strip_prefix("<function=")?;
        let (name, body) = body.split_once('>')?;
        let mut body = body.trim_start().strip_suffix("</function>")?.trim_end();
        let mut arguments = Map::new();
        while !body.is_empty() {
            let (key, after) = body.strip_prefix("<parameter=")?.split_once('>')?;
            let (value, after) = after.split_once("</parameter>")?;
            body = after.trim_start();
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            let parsed = match parameter_type(tools, name, key) {
                Some("string") | None => None,
                Some(_) => serde_json::from_str(value).ok(),
            };
            arguments.insert(
                key.to_owned(),
                parsed.unwrap_or_else(|| Value::String(value.to_owned())),
            );
        }
        calls.push(ToolCall {
            name: name.to_owned(),
            arguments,
        });
    }
    (!calls.is_empty()).then_some(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn tojson_matches_python_separators_and_order() {
        let v: Value =
            serde_json::from_str(r#"{"z":1,"a":[true,null,"é\n"],"m":{"k":1.5}}"#).unwrap();
        assert_eq!(
            tojson(&v),
            r#"{"z": 1, "a": [true, null, "é\n"], "m": {"k": 1.5}}"#
        );
    }
    #[test]
    fn parses_and_renders_calls() {
        let tools = [
            json!({"type": "function", "function": {"name": "get", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
            "opts": {"type": "object"}}}}}),
        ];
        let text = "<tool_call>\n<function=get>\n<parameter=city>\n42\n</parameter>\n\
            <parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"a\": 1}\n</parameter>\n\
            </function>\n</tool_call>\n<tool_call>\n<function=other>\n<parameter=x>\nmulti\nline\n\
            </parameter>\n</function>\n</tool_call>";
        let calls = parse_calls(text, &tools).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get");
        assert_eq!(
            Value::Object(calls[0].arguments.clone()),
            json!({"city": "42", "days": 3, "opts": {"a": 1}})
        );
        assert_eq!(calls[1].arguments["x"], "multi\nline");
        // Rendering the parsed calls reproduces what the model generated.
        assert_eq!(render_calls("", &calls), text);
        assert!(render_calls("hi", &calls[..1]).starts_with("\n\n<tool_call>"));
        let hermes =
            "<tool_call>\n{\"name\": \"get\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        assert_eq!(
            parse_calls(hermes, &tools).unwrap()[0].arguments["city"],
            "Paris"
        );
        assert!(parse_calls("<tool_call>{\"name\": 1}</tool_call>", &tools).is_none());
        // Llama 3.x: bare JSON reply, or after <|python_tag|>, `;`-separated.
        let llama = parse_calls(
            " {\"name\": \"get\", \"parameters\": {\"city\": \"Oslo\", \"days\": 2}}",
            &tools,
        )
        .unwrap();
        assert_eq!(llama[0].name, "get");
        assert_eq!(
            Value::Object(llama[0].arguments.clone()),
            json!({"city": "Oslo", "days": 2})
        );
        let tagged = "<|python_tag|>{\"name\": \"a\", \"parameters\": {}}; {\"name\": \"b\", \"parameters\": {\"x\": 1}}";
        let calls = parse_calls(tagged, &tools).unwrap();
        assert_eq!((calls[0].name.as_str(), calls[1].name.as_str()), ("a", "b"));
        assert!(parse_calls("{\"name\": \"a\"} trailing", &tools).is_none());
        assert!(parse_calls("<|python_tag|>brave_search.call(query=\"x\")", &tools).is_none());
        assert_eq!(call_start("  {\"name\": \"a\"}", true), Some(0));
        assert_eq!(call_start("{\"name\": \"a\"}", false), None);
        assert_eq!(call_start("{\"title\": 1}", true), None);
        assert_eq!(call_start("ok <|python_tag|>{", false), Some(3));
        assert!(may_start_bare_call(" {\"na") && may_start_bare_call(""));
        assert!(!may_start_bare_call("Hello") && !may_start_bare_call("{\"x\""));
        // Streaming: a reply that opens with a JSON `name` object is captured.
        use crate::openai::TextStream;
        let mut s = TextStream::new(false, vec![], true);
        assert!(s.push(b" {").is_empty() && s.push(b"\"name\": \"f\"}").is_empty());
        assert_eq!(s.tool_text, " {\"name\": \"f\"}");
        let mut s = TextStream::new(false, vec![], true);
        assert_eq!(s.push(b"Hi {\"name\"")[0].text, "Hi {\"name\"");
        for bad in [
            "<tool_call>\n<function=x>\n</function>",
            "<tool_call><function=x></tool_call>",
            "text",
            "",
        ] {
            assert!(parse_calls(bad, &tools).is_none(), "{bad}");
        }
    }
}
