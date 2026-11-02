//! Generic chat templates: the GGUF `tokenizer.chat_template` rendered with
//! minijinja the way Hugging Face transformers renders it (trim/lstrip blocks,
//! Python string methods, `tojson` with Python separators).
use crate::thinking::Thinking;
use crate::tokenizer::{Message, Role};
use crate::tools::tojson;
use crate::{Result, invalid};
use minijinja::{Environment, Error, ErrorKind, Value as Jinja};
use serde_json::{Value, json};

pub struct ChatTemplate {
    source: String,
    bos: String,
    eos: String,
}

fn message_value(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut v = json!({"role": role, "content": m.content});
    if let Some(r) = &m.reasoning {
        v["reasoning_content"] = json!(r);
    }
    if !m.tool_calls.is_empty() {
        v["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| json!({"type": "function", "function": {"name": c.name, "arguments": Value::Object(c.arguments.clone())}}))
            .collect();
    }
    v
}

/// Some templates, such as Phi-3's, have no system role and skip those
/// messages without a word. Their text goes into the first user message.
fn merge_system(messages: &mut Vec<Value>) {
    let mut system = Vec::new();
    messages.retain(|m| {
        let found = m["role"] == "system";
        if found {
            system.push(m["content"].as_str().unwrap_or_default().to_owned());
        }
        !found
    });
    if system.is_empty() {
        return;
    }
    let text = system.join("\n\n");
    match messages.iter_mut().find(|m| m["role"] == "user") {
        Some(user) => {
            user["content"] = json!(format!(
                "{text}\n\n{}",
                user["content"].as_str().unwrap_or("")
            ));
        }
        None => messages.insert(0, json!({"role": "user", "content": text})),
    }
}

impl ChatTemplate {
    pub fn new(source: String, bos: String, eos: String) -> Self {
        Self { source, bos, eos }
    }
    pub fn bos(&self) -> &str {
        &self.bos
    }
    /// Render with `add_generation_prompt`. Thinking is disabled unless a level
    /// is given; a level sets `enable_thinking` and `reasoning_effort`.
    pub fn render(
        &self,
        messages: &[Message],
        tools: &[Value],
        thinking: Option<Thinking>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(invalid("no messages provided"));
        }
        if !tools.is_empty() && !self.source.contains("tools") {
            return Err(invalid("this model's chat template does not support tools"));
        }
        let mut env = Environment::new();
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| invalid(format!("chat template syntax: {e}")))?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<Jinja, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        // Keyword arguments (indent, ensure_ascii) are accepted; output always
        // matches json.dumps(ensure_ascii=False) with default separators.
        env.add_filter(
            "tojson",
            |v: Jinja, _: minijinja::value::Kwargs| -> std::result::Result<Jinja, Error> {
                let v: Value = serde_json::to_value(&v)
                    .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
                Ok(Jinja::from_safe_string(tojson(&v)))
            },
        );
        env.add_template("chat", &self.source)
            .map_err(|e| invalid(format!("chat template: {e}")))?;
        let mut messages: Vec<Value> = messages.iter().map(message_value).collect();
        if !self.source.contains("system") {
            merge_system(&mut messages);
        }
        let level = thinking.unwrap_or(Thinking::Off);
        let mut context = json!({
            "messages": messages,
            "tools": if tools.is_empty() { Value::Null } else { json!(tools) },
            "add_generation_prompt": true,
            "enable_thinking": level != Thinking::Off,
            "bos_token": self.bos,
            "eos_token": self.eos,
        });
        if level != Thinking::Off {
            context["reasoning_effort"] = json!(level.name());
        }
        env.get_template("chat")
            .and_then(|t| t.render(minijinja::value::Serde(&context)))
            .map_err(|e| invalid(format!("chat template: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCall;
    #[test]
    fn templates_without_a_system_role_keep_its_text() {
        let source = "{% for m in messages %}{% if m.role == 'user' %}<u>{{ m.content }}{% endif %}{% endfor %}";
        let template = ChatTemplate::new(source.into(), String::new(), String::new());
        let messages = [
            Message::new(Role::System, "Be brief."),
            Message::new(Role::User, "Hi"),
        ];
        let text = template.render(&messages, &[], None).unwrap();
        assert_eq!(text, "<u>Be brief.\n\nHi");
    }
    // A ChatML template in the style of Qwen2.5/Qwen3 (abridged), exercising
    // tools, tool calls, tool results and string methods.
    const CHATML: &str = r##"{%- if tools %}{{- '<|im_start|>system\n' }}{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}{{- "# Tools\n\n<tools>" }}{%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}{{- "\n</tools><|im_end|>\n" }}{%- elif messages[0].role == 'system' %}{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}{%- endif %}
{%- for message in messages %}{%- if message.role == "user" or (message.role == "system" and not loop.first) %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- elif message.role == "assistant" %}{{- '<|im_start|>' + message.role + '\n' + message.content.strip() }}{%- for tool_call in message.tool_calls %}{%- set tool_call = tool_call.function %}{{- '\n<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\n' }}{%- elif message.role == "tool" %}{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}{%- endif %}{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}"##;
    #[test]
    fn renders_chatml_with_tools() {
        let t = ChatTemplate::new(CHATML.into(), String::new(), "<|im_end|>".into());
        let mut a = Message::new(Role::Assistant, " ok ");
        a.tool_calls = vec![ToolCall {
            name: "f".into(),
            arguments: json!({"x": 1}).as_object().unwrap().clone(),
        }];
        let tools = [json!({"type": "function", "function": {"name": "f"}})];
        let out = t
            .render(
                &[
                    Message::new(Role::System, "S"),
                    Message::user("q"),
                    a,
                    Message::new(Role::Tool, "r"),
                ],
                &tools,
                None,
            )
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nS\n\n# Tools\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"f\"}}\n</tools><|im_end|>\n\
             <|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\nok\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\nr\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let on = t
            .render(&[Message::user("q")], &[], Some(Thinking::Low))
            .unwrap();
        assert!(on.ends_with("<|im_start|>assistant\n"), "{on}");
        let bad = ChatTemplate::new(
            "{{ raise_exception('nope') }}".into(),
            String::new(),
            String::new(),
        );
        assert!(
            bad.render(&[Message::user("q")], &[], None)
                .unwrap_err()
                .to_string()
                .contains("nope")
        );
    }
}
