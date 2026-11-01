//! GGUF GPT-2 BPE with the verified qwen35 and glm4 pre-tokenizers, plus
//! SentencePiece-style vocabularies (`spm.rs`).
use crate::gguf::Gguf;
use crate::template::ChatTemplate;
use crate::thinking::{self, Thinking};
use crate::tools::{self, ToolCall};
use crate::{Result, invalid};
use serde_json::Value;
use std::collections::HashMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::{
    PreTokenizerWrapper,
    byte_level::ByteLevel,
    sequence::Sequence,
    split::{Split, SplitPattern},
};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer as Core};
const PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
// Also the llama-bpe (Llama 3) pattern.
const GLM_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
enum Backend {
    Bpe {
        plain: Box<Core>,
        special: Box<Core>,
    },
    Spm(Box<crate::spm::Vocab>),
}
pub struct Tokenizer {
    backend: Backend,
    pieces: Vec<Vec<u8>>,
    /// Control and user-defined tokens (chat markers), never plain text.
    special_ids: Vec<bool>,
    pub eos: u32,
    bos: Option<u32>,
    glm: bool,
    eos_tokens: Vec<u32>,
    /// Models other than qwen35/glm4 render chat with their GGUF template.
    generic: bool,
    template: Option<ChatTemplate>,
}
impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        if matches!(g.string("tokenizer.ggml.model")?, "llama" | "gemma4") {
            return Self::from_spm(g);
        }
        let pre = g.string("tokenizer.ggml.pre")?;
        // (pattern, ignore_merges) as llama.cpp configures each pre-tokenizer.
        let (pattern, ignore_merges) = match pre {
            "qwen35" => (PATTERN, false),
            "glm4" | "llama-bpe" => (GLM_PATTERN, true),
            "dbrx" => (GLM_PATTERN, false),
            "qwen2" => (QWEN2_PATTERN, false),
            _ => (PATTERN, false),
        };
        if g.string("tokenizer.ggml.model")? != "gpt2"
            || !matches!(pre, "qwen35" | "glm4" | "qwen2" | "llama-bpe" | "dbrx")
        {
            return Err(invalid(format!(
                "unsupported tokenizer {}/{pre}; supported: GPT-2 BPE with qwen35, glm4, qwen2, llama-bpe or dbrx pre-tokenizers, SentencePiece (llama) or gemma4",
                g.string("tokenizer.ggml.model")?
            )));
        }
        let glm = pre == "glm4";
        let tokens = g.strings("tokenizer.ggml.tokens")?;
        let types = g.integers("tokenizer.ggml.token_type")?;
        if types.len() != tokens.len() {
            return Err(invalid("token type count mismatch"));
        }
        let vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect::<ahash::AHashMap<_, _>>();
        let merges = g
            .strings("tokenizer.ggml.merges")?
            .iter()
            .map(|s| {
                s.split_once(' ')
                    .map(|(a, b)| (a.to_owned(), b.to_owned()))
                    .ok_or_else(|| invalid("invalid BPE merge"))
            })
            .collect::<Result<Vec<_>>>()?;
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(ignore_merges)
            .build()
            .map_err(invalid)?;
        let mut plain = Core::new(bpe);
        let split = Split::new(
            SplitPattern::Regex(pattern.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(invalid)?;
        plain.with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
        let always = tokens
            .iter()
            .zip(&types)
            .filter(|(_, t)| **t == 4)
            .map(|(s, _)| AddedToken::from(s.clone(), true).normalized(false))
            .collect::<Vec<_>>();
        plain.add_special_tokens(&always);
        let mut special = plain.clone();
        special.add_special_tokens(
            &tokens
                .iter()
                .zip(&types)
                .filter(|(_, t)| **t == 3)
                .map(|(s, _)| AddedToken::from(s.clone(), true).normalized(false))
                .collect::<Vec<_>>(),
        );
        let mut alphabet = HashMap::new();
        let mut extra = 256;
        for b in 0u16..=255 {
            let c = if (33..=126).contains(&b)
                || (161..=172).contains(&b)
                || (174..=255).contains(&b)
            {
                char::from_u32(b as u32)
            } else {
                let c = char::from_u32(extra);
                extra += 1;
                c
            }
            .ok_or_else(|| invalid("byte alphabet"))?;
            alphabet.insert(c, b as u8);
        }
        let pieces = tokens
            .iter()
            .zip(&types)
            .map(|(s, &t)| {
                if t == 3 || t == 4 {
                    Ok(s.as_bytes().to_vec())
                } else {
                    s.chars()
                        .map(|c| {
                            alphabet
                                .get(&c)
                                .copied()
                                .ok_or_else(|| invalid("vocabulary byte alphabet"))
                        })
                        .collect()
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let bos = if g.optional_integer("tokenizer.ggml.add_bos_token") == Some(1) {
            Some(g.integer("tokenizer.ggml.bos_token_id")? as u32)
        } else {
            None
        };
        let eos = g.integer("tokenizer.ggml.eos_token_id")? as u32;
        let mut eos_tokens = vec![eos];
        if glm {
            for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
                eos_tokens.push(g.integer(key)? as u32);
            }
        } else {
            // Qwen GGUF stores only im_end as eos, but these vocabulary
            // markers also terminate generation (including code/FIM output).
            // Resolve their IDs from this model; fim_suffix is an input delimiter.
            // Llama 3 ends turns with eot/eom.
            for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
                if let Some(id) = g.optional_integer(key) {
                    eos_tokens.push(id as u32);
                }
            }
            eos_tokens.extend(tokens.iter().enumerate().filter_map(|(id, token)| {
                matches!(
                    token.as_str(),
                    "<|endoftext|>"
                        | "<|im_end|>"
                        | "<|fim_pad|>"
                        | "<|repo_name|>"
                        | "<|file_sep|>"
                        | "<|eot_id|>"
                        | "<|eom_id|>"
                )
                .then_some(id as u32)
            }));
            eos_tokens.sort_unstable();
            eos_tokens.dedup();
        }
        // qwen35 and glm4 use the verified built-in templates below.
        let template_needed = !matches!(pre, "qwen35" | "glm4");
        let template = if !template_needed {
            None
        } else {
            gguf_template(g, &tokens, eos)
        };
        Ok(Self {
            backend: Backend::Bpe {
                plain: Box::new(plain),
                special: Box::new(special),
            },
            pieces,
            special_ids: types.iter().map(|&t| t == 3 || t == 4).collect(),
            eos,
            bos,
            glm,
            eos_tokens,
            generic: template_needed,
            template,
        })
    }
    /// SentencePiece (`llama`) and `gemma4` vocabularies; chat always uses
    /// the GGUF template.
    fn from_spm(g: &Gguf) -> Result<Self> {
        let vocab = crate::spm::Vocab::from_gguf(g)?;
        let tokens = g.strings("tokenizer.ggml.tokens")?;
        let types = g.integers("tokenizer.ggml.token_type")?;
        let eos = g.integer("tokenizer.ggml.eos_token_id")? as u32;
        Ok(Self {
            pieces: vocab.pieces(),
            special_ids: types.iter().map(|&t| t == 3 || t == 4).collect(),
            eos,
            bos: None,
            glm: false,
            eos_tokens: vocab.eog.clone(),
            generic: true,
            template: gguf_template(g, &tokens, eos),
            backend: Backend::Spm(Box::new(vocab)),
        })
    }
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let (plain, special) = match &self.backend {
            Backend::Bpe { plain, special } => (plain, special),
            Backend::Spm(v) => return v.encode(text, true, parse_special),
        };
        let core = if parse_special { special } else { plain };
        let mut ids = core
            .encode(text, false)
            .map_err(invalid)?
            .get_ids()
            .to_vec();
        if let Some(bos) = self.bos {
            ids.insert(0, bos);
        }
        Ok(ids)
    }
    pub fn token_bytes(&self, id: u32) -> Result<&[u8]> {
        self.pieces
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("token outside vocabulary"))
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.token_bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.get(id as usize).copied().unwrap_or(true)
    }
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_tokens.contains(&id)
    }
    pub fn is_glm(&self) -> bool {
        self.glm
    }
    /// Encode a template continuation: special markers are parsed and no BOS is added.
    pub fn encode_continuation(&self, text: &str) -> Result<Vec<u32>> {
        let special = match &self.backend {
            Backend::Bpe { special, .. } => special,
            Backend::Spm(v) => return v.encode(text, false, true),
        };
        Ok(special
            .encode(text, false)
            .map_err(invalid)?
            .get_ids()
            .to_vec())
    }
    /// Verified single-user text templates: Qwen without thinking, GLM default Max effort.
    pub fn chat(&self, text: &str) -> Result<Vec<u32>> {
        self.chat_messages(&[Message::user(text)])
    }
    /// Multi-turn conversation rendered with the model's GGUF chat template.
    pub fn chat_messages(&self, messages: &[Message]) -> Result<Vec<u32>> {
        self.chat_with_tools(messages, &[])
    }
    /// Conversation with tool definitions (OpenAI `tools` objects; Qwen only).
    pub fn chat_with_tools(&self, messages: &[Message], tools: &[Value]) -> Result<Vec<u32>> {
        self.chat_thinking(messages, tools, None)
    }
    /// Conversation at a thinking level; None is the model's default.
    pub fn chat_thinking(
        &self,
        messages: &[Message],
        tools: &[Value],
        thinking: Option<Thinking>,
    ) -> Result<Vec<u32>> {
        if self.uses_gguf_template() {
            let t = self
                .template
                .as_ref()
                .ok_or_else(|| invalid("this model has no chat template (base model?)"))?;
            let text = t.render(messages, tools, thinking)?;
            // Templates that emit the BOS text already carry it.
            return if !t.bos().is_empty() && text.starts_with(t.bos()) {
                self.encode_continuation(&text)
            } else {
                self.encode(&text, true)
            };
        }
        self.encode(
            &render_chat_with(self.glm, messages, tools, thinking)?,
            true,
        )
    }
    /// True when chat is rendered from the GGUF template (generic models).
    fn uses_gguf_template(&self) -> bool {
        self.generic
    }
    /// Template text that follows a generated assistant reply and opens the
    /// next turn; None for GGUF templates, which are re-rendered instead.
    pub fn chat_continuation(&self, user: &str) -> Option<String> {
        self.chat_continuation_with(user, None)
    }
    pub fn chat_continuation_with(&self, user: &str, thinking: Option<Thinking>) -> Option<String> {
        let t = thinking::resolve(thinking, self.glm);
        (!self.generic).then(|| continuation(self.glm, user, t))
    }
}

/// The GGUF chat template with the BOS/EOS token texts it may reference.
fn gguf_template(g: &Gguf, tokens: &[String], eos: u32) -> Option<ChatTemplate> {
    let name = |id: Option<usize>| id.and_then(|i| tokens.get(i)).cloned().unwrap_or_default();
    g.optional_string("tokenizer.chat_template").map(|source| {
        ChatTemplate::new(
            source.to_owned(),
            name(g.optional_integer("tokenizer.ggml.bos_token_id")),
            name(Some(eos as usize)),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool result, answering the preceding assistant's tool calls.
    Tool,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Assistant reasoning (`reasoning_content`); rendered inside the think block.
    pub reasoning: Option<String>,
    /// Assistant tool calls, rendered after the content.
    pub tool_calls: Vec<ToolCall>,
}
impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning: None,
            tool_calls: vec![],
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
}
fn qwen_user(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text.trim())
}
fn continuation(glm: bool, user: &str, t: Thinking) -> String {
    if glm {
        format!("<|user|>{user}{}", thinking::glm_generation(t))
    } else {
        format!(
            "<|im_end|>\n{}{}",
            qwen_user(user),
            thinking::qwen_generation(t)
        )
    }
}
/// Render `tokenizer.chat_template` with `add_generation_prompt`. Qwen uses
/// `enable_thinking=false`; GLM uses the default Max reasoning effort. Both
/// keep assistant reasoning in history (template defaults). Tools are Qwen only.
pub fn render_chat(glm: bool, messages: &[Message], tools: &[Value]) -> Result<String> {
    render_chat_with(glm, messages, tools, None)
}
/// `render_chat` at a thinking level (see `thinking`); None is the default above.
pub fn render_chat_with(
    glm: bool,
    messages: &[Message],
    tools: &[Value],
    thinking: Option<Thinking>,
) -> Result<String> {
    let t = thinking::resolve(thinking, glm);
    if messages.is_empty() {
        return Err(invalid("no messages provided"));
    }
    let uses_tools = !tools.is_empty()
        || messages
            .iter()
            .any(|m| m.role == Role::Tool || !m.tool_calls.is_empty());
    if glm && uses_tools {
        return Err(invalid("tool calling is supported only for Qwen models"));
    }
    if glm {
        let mut out = thinking::glm_header(t).to_owned();
        for m in messages {
            match m.role {
                Role::System => out += &format!("<|system|>{}", m.content),
                Role::User => out += &format!("<|user|>{}", m.content),
                Role::Tool => unreachable!("rejected above"),
                Role::Assistant => {
                    let (reasoning, content) = match (&m.reasoning, m.content.find("</think>")) {
                        (Some(r), _) => (Some(r.as_str()), m.content.as_str()),
                        (None, Some(_)) => {
                            let before = m.content.split("</think>").next().unwrap_or("");
                            let reasoning = before.rsplit("<think>").next().unwrap_or("");
                            let content = m.content.rsplit("</think>").next().unwrap_or("");
                            (Some(reasoning), content)
                        }
                        (None, None) => (None, m.content.as_str()),
                    };
                    out += &format!("<|assistant|><think>{}</think>", reasoning.unwrap_or(""));
                    out += content.trim();
                }
            }
        }
        return Ok(out + thinking::glm_generation(t));
    }
    let system_count = messages
        .iter()
        .take_while(|m| m.role == Role::System)
        .count();
    let system = messages[..system_count]
        .iter()
        .map(|m| m.content.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let instruction = thinking::qwen_instruction(t);
    let mut out = String::new();
    if !tools.is_empty() {
        out += &tools::system_prompt(tools, &system);
        if let Some(i) = instruction {
            out.insert_str("<|im_start|>system\n".len(), &format!("{i}\n\n"));
        }
    } else {
        let system = match instruction {
            Some(i) if system.is_empty() => i.to_owned(),
            Some(i) => format!("{i}\n\n{system}"),
            None => system,
        };
        if !system.is_empty() {
            out += &format!("<|im_start|>system\n{system}<|im_end|>\n");
        }
    }
    let rest = &messages[system_count..];
    for (i, m) in rest.iter().enumerate() {
        match m.role {
            Role::System => return Err(invalid("system message must be at the beginning")),
            Role::User => out += &qwen_user(&m.content),
            Role::Assistant => {
                let content = m.content.trim();
                out += &format!(
                    "<|im_start|>assistant\n<think>\n{}\n</think>\n\n{content}{}<|im_end|>\n",
                    m.reasoning.as_deref().unwrap_or("").trim(),
                    tools::render_calls(content, &m.tool_calls)
                );
            }
            // Consecutive tool results share one user turn.
            Role::Tool => {
                if i == 0 || rest[i - 1].role != Role::Tool {
                    out += "<|im_start|>user";
                }
                out += &format!("\n<tool_response>\n{}\n</tool_response>", m.content.trim());
                if rest.get(i + 1).is_none_or(|n| n.role != Role::Tool) {
                    out += "<|im_end|>\n";
                }
            }
        }
    }
    Ok(out + thinking::qwen_generation(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn m(role: Role, content: &str) -> Message {
        Message::new(role, content)
    }
    #[test]
    fn single_turn_matches_previous_chat_format() {
        // The exact strings the single-user `chat()` produced before multi-turn support.
        let text = " 你好 ";
        assert_eq!(
            render_chat(false, &[Message::user(text)], &[]).unwrap(),
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                text.trim()
            )
        );
        assert_eq!(
            render_chat(true, &[Message::user(text)], &[]).unwrap(),
            format!(
                "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{text}<|assistant|><think>"
            )
        );
    }
    #[test]
    fn qwen_multi_turn_matches_gguf_template() {
        let mut a = m(Role::Assistant, " Hi! ");
        a.reasoning = Some(" plan ".into());
        let messages = [
            m(Role::System, " Be brief. "),
            m(Role::System, ""),
            m(Role::System, "Use English."),
            m(Role::User, "Hello"),
            a,
            m(Role::User, "Again"),
        ];
        assert_eq!(
            render_chat(false, &messages, &[]).unwrap(),
            "<|im_start|>system\nBe brief.\nUse English.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n<think>\nplan\n</think>\n\nHi!<|im_end|>\n\
             <|im_start|>user\nAgain<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(render_chat(false, &[m(Role::User, "a"), m(Role::System, "b")], &[]).is_err());
        assert!(render_chat(false, &[], &[]).is_err());
        assert!(render_chat(true, &[], &[]).is_err());
    }
    #[test]
    fn glm_multi_turn_matches_gguf_template() {
        let messages = [
            m(Role::System, "Be brief."),
            m(Role::User, "Hello"),
            m(Role::Assistant, "<think>greet</think>\n Hi! "),
            m(Role::User, "Again"),
            m(Role::Assistant, "plain"),
            m(Role::User, "Third"),
        ];
        assert_eq!(
            render_chat(true, &messages, &[]).unwrap(),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>Be brief.<|user|>Hello\
             <|assistant|><think>greet</think>Hi!<|user|>Again\
             <|assistant|><think></think>plain<|user|>Third<|assistant|><think>"
        );
        let mut explicit = m(Role::Assistant, "x</think>y");
        explicit.reasoning = Some("r".into());
        assert!(
            render_chat(true, &[m(Role::User, "q"), explicit], &[])
                .unwrap()
                .ends_with("<|assistant|><think>r</think>x</think>y<|assistant|><think>")
        );
    }
    #[test]
    fn qwen_tools_match_gguf_template() {
        // Rendered from the GGUF's chat template with jinja2 (trim/lstrip
        // blocks, enable_thinking=false, add_generation_prompt).
        let tools = [
            serde_json::json!({"type":"function","function":{"name":"get_weather",
            "description":"Weather for a city","parameters":{"type":"object","properties":
            {"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}),
        ];
        let call = |args: Value| ToolCall {
            name: "get_weather".into(),
            arguments: args.as_object().unwrap().clone(),
        };
        let mut first = m(Role::Assistant, "Checking.");
        first.tool_calls = vec![
            call(serde_json::json!({"city": "Paris", "days": 2})),
            call(serde_json::json!({"city": "Rome"})),
        ];
        let mut second = m(Role::Assistant, "");
        second.tool_calls = vec![call(serde_json::json!({"city": "Oslo"}))];
        let messages = [
            m(Role::System, "Be brief."),
            m(Role::User, "Weather in Paris and Rome?"),
            first,
            m(Role::Tool, "sunny"),
            m(Role::Tool, " rain "),
            second,
            m(Role::Tool, "snow"),
        ];
        assert_eq!(
            render_chat(false, &messages, &tools).unwrap(),
            include_str!("../tests/fixtures/qwen_tools_template.txt")
        );
        assert!(render_chat(true, &messages[..2], &tools).is_err());
    }
    #[test]
    fn continuation_extends_the_rendered_history() {
        for glm in [false, true] {
            let reply = if glm { "think</think>Answer" } else { "Answer" };
            let first = [m(Role::System, "S"), m(Role::User, "Q1")];
            let mut all = first.to_vec();
            all.push(m(Role::Assistant, reply));
            all.push(m(Role::User, "Q2"));
            assert_eq!(
                render_chat(glm, &first, &[]).unwrap()
                    + reply
                    + &continuation(glm, "Q2", thinking::resolve(None, glm)),
                render_chat(glm, &all, &[]).unwrap()
            );
        }
    }
}
