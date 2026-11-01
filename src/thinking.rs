//! Thinking levels, mapped onto each chat template's own mechanism:
//! Qwen3.8 reasoning-effort instructions and an open think block, GLM's
//! `Reasoning Effort` header, and `enable_thinking`/`reasoning_effort` for
//! GGUF templates. A request without a level uses the model's default:
//! thinking off for Qwen and GGUF templates, Max effort for GLM.
use crate::invalid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    Off,
    Low,
    Medium,
    High,
}
impl Thinking {
    /// The name `--thinking` and `reasoning_effort` accept (`none` is off).
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}
impl std::str::FromStr for Thinking {
    type Err = std::io::Error;
    /// OpenAI `reasoning_effort` values; `minimal` is low and `xhigh` high.
    fn from_str(s: &str) -> std::io::Result<Self> {
        match s {
            "off" | "none" => Ok(Self::Off),
            "minimal" | "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" | "xhigh" => Ok(Self::High),
            _ => Err(invalid(format!(
                "unknown thinking level {s}; use off, low, medium or high"
            ))),
        }
    }
}

/// The level a request runs at: its own, else the model default.
pub(crate) fn resolve(thinking: Option<Thinking>, glm: bool) -> Thinking {
    thinking.unwrap_or(if glm { Thinking::High } else { Thinking::Off })
}

/// Qwen3.8 GGUF template: the system instruction for `reasoning_effort`
/// (xhigh for high; medium has none).
pub(crate) fn qwen_instruction(t: Thinking) -> Option<&'static str> {
    match t {
        Thinking::High => Some(
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key \
             assumptions, consider plausible alternatives, and prioritize correctness, consistency, \
             and clarity in the final answer.",
        ),
        Thinking::Low => Some(
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly \
             to the conclusion without unnecessary elaboration.",
        ),
        Thinking::Off | Thinking::Medium => None,
    }
}

/// Qwen generation prompt: an empty think block (`enable_thinking=false`) or an open one.
pub(crate) fn qwen_generation(t: Thinking) -> &'static str {
    match t {
        Thinking::Off => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        _ => "<|im_start|>assistant\n<think>\n",
    }
}

/// GLM header: the template's efforts are Low, High and Max (its default).
/// Off keeps the default header; the reply then starts with an empty think block.
pub(crate) fn glm_header(t: Thinking) -> &'static str {
    match t {
        Thinking::Low => "[gMASK]<sop><|system|>Reasoning Effort: Low",
        Thinking::Medium => "[gMASK]<sop><|system|>Reasoning Effort: High",
        Thinking::Off | Thinking::High => "[gMASK]<sop><|system|>Reasoning Effort: Max",
    }
}

/// GLM generation prompt; off closes the think block the way the template
/// renders an assistant turn without reasoning.
pub(crate) fn glm_generation(t: Thinking) -> &'static str {
    match t {
        Thinking::Off => "<|assistant|><think></think>",
        _ => "<|assistant|><think>",
    }
}

/// True when the rendered prompt ends inside an open think block, so the
/// reply starts with reasoning.
pub(crate) fn opens_think(prompt_tail: &str) -> bool {
    prompt_tail.trim_end().ends_with("<think>")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_levels_and_resolves_model_defaults() {
        for (text, level) in [
            ("none", Thinking::Off),
            ("off", Thinking::Off),
            ("minimal", Thinking::Low),
            ("low", Thinking::Low),
            ("medium", Thinking::Medium),
            ("high", Thinking::High),
            ("xhigh", Thinking::High),
        ] {
            assert_eq!(text.parse::<Thinking>().unwrap(), level, "{text}");
        }
        assert!("max".parse::<Thinking>().is_err());
        assert_eq!(resolve(None, false), Thinking::Off);
        assert_eq!(resolve(None, true), Thinking::High);
        assert_eq!(resolve(Some(Thinking::Low), true), Thinking::Low);
        assert!(opens_think("<|im_start|>assistant\n<think>\n"));
        assert!(!opens_think("<think>\n\n</think>\n\n"));
    }
    #[test]
    fn levels_match_gguf_templates() {
        // Rendered from the Qwen3.8 and GLM-5.3 GGUF chat templates with jinja2
        // (trim/lstrip blocks, add_generation_prompt; Qwen without a level uses
        // enable_thinking=false, otherwise reasoning_effort).
        use crate::tokenizer::{Message, Role, render_chat_with};
        use serde_json::{Value, json};
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../tests/fixtures/thinking_templates.json"))
                .unwrap();
        let tool = json!({"type": "function", "function": {"name": "get_weather",
            "description": "Weather for a city", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}, "required": ["city"]}}});
        let mut reply = Message::new(Role::Assistant, "Hi!");
        reply.reasoning = Some("plan".into());
        for case in &cases {
            let mut messages = vec![];
            if case["system"] == true {
                messages.push(Message::new(Role::System, "Be brief."));
            }
            messages.extend([
                Message::user("Hello"),
                reply.clone(),
                Message::user("Again"),
            ]);
            let tools = if case["tools"] == true {
                vec![tool.clone()]
            } else {
                vec![]
            };
            let level = case["effort"].as_str().map(|e| e.parse().unwrap());
            let glm = case["model"] == "glm";
            assert_eq!(
                render_chat_with(glm, &messages, &tools, level).unwrap(),
                case["text"].as_str().unwrap(),
                "{case}"
            );
        }
        let off = render_chat_with(true, &[Message::user("q")], &[], Some(Thinking::Off)).unwrap();
        assert!(off.ends_with("<|user|>q<|assistant|><think></think>"));
    }
}
