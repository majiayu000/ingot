//! Structured outputs: OpenAI `response_format` (json_object / json_schema)
//! and Anthropic `output_config.format` / `output_format` (json_schema),
//! enforced by masking every token whose bytes cannot continue a valid text.
use crate::engine::{Cancellation, Constraint, Engine, Generation, Session};
use crate::json_grammar::{Grammar, Matcher};
use crate::openai::{ApiError, Request, bad};
use crate::sampling::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Result, invalid};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Requested output grammar; None is free text.
pub(crate) type Format = Option<Arc<Grammar>>;

fn compile(
    schema: Option<&Value>,
    param: Option<&'static str>,
) -> std::result::Result<Grammar, ApiError> {
    let schema = schema.ok_or_else(|| bad("a json_schema format needs a schema", param))?;
    Grammar::schema(schema).map_err(|e| bad(format!("unsupported output schema: {e}"), param))
}
fn without_tools(
    fields: &Map<String, Value>,
    grammar: Grammar,
    param: Option<&'static str>,
) -> std::result::Result<Format, ApiError> {
    if fields
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty())
    {
        return Err(bad(
            "structured output cannot be combined with tools",
            param,
        ));
    }
    Ok(Some(Arc::new(grammar)))
}

/// OpenAI `response_format`.
pub(crate) fn openai(fields: &Map<String, Value>) -> std::result::Result<Format, ApiError> {
    let p = Some("response_format");
    let Some(format) = fields.get("response_format") else {
        return Ok(None);
    };
    let grammar = match format.get("type").and_then(Value::as_str) {
        Some("text") => return Ok(None),
        Some("json_object") => Grammar::object(),
        Some("json_schema") => compile(format.get("json_schema").and_then(|j| j.get("schema")), p)?,
        _ => {
            return Err(bad(
                "response_format.type must be text, json_object or json_schema",
                p,
            ));
        }
    };
    without_tools(fields, grammar, p)
}

/// Anthropic `output_config.format`, or the beta `output_format`.
pub(crate) fn anthropic(fields: &Map<String, Value>) -> std::result::Result<Format, ApiError> {
    let format = fields.get("output_format").or_else(|| {
        fields
            .get("output_config")
            .and_then(|c| c.get("format"))
            .filter(|f| !f.is_null())
    });
    let Some(format) = format else {
        return Ok(None);
    };
    if format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return Err(bad("output format type must be json_schema", None));
    }
    let grammar = compile(format.get("schema"), None)?;
    without_tools(fields, grammar, None)
}

/// Token byte strings, indexed for masking.
struct Vocabulary {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
    /// Non-special tokens by first byte, then by second byte (256: none), so
    /// a mask steps each two-byte prefix once instead of every token.
    groups: Vec<Vec<(u16, Vec<u32>)>>,
    /// Tokens valid anywhere inside a string (see `Matcher::plain_string`).
    plain: Vec<u32>,
    eos: Vec<u32>,
    special: Vec<bool>,
}
impl Vocabulary {
    fn new(t: &Tokenizer) -> Result<Self> {
        let ids = 0..t.vocab_size() as u32;
        let pieces = ids
            .clone()
            .map(|i| t.token_bytes(i))
            .collect::<Result<Vec<_>>>()?;
        let special = ids.clone().map(|i| t.is_special(i)).collect();
        Ok(Self::build(
            &pieces,
            special,
            ids.filter(|&i| t.is_eos(i)).collect(),
        ))
    }
    fn build(pieces: &[&[u8]], special: Vec<bool>, eos: Vec<u32>) -> Self {
        let mut v = Self {
            bytes: vec![],
            offsets: vec![0],
            groups: vec![vec![]; 256],
            plain: vec![],
            eos,
            special,
        };
        // Position of each (first, second) group within its first-byte list.
        let mut slots = vec![usize::MAX; 256 * 257];
        for (id, &piece) in (0..).zip(pieces) {
            v.bytes.extend_from_slice(piece);
            v.offsets.push(v.bytes.len());
            if v.special[id as usize] || piece.is_empty() {
                continue;
            }
            let second = piece.get(1).map_or(256, |&b| b as u16);
            let group = &mut v.groups[piece[0] as usize];
            let slot = &mut slots[piece[0] as usize * 257 + second as usize];
            if *slot == usize::MAX {
                *slot = group.len();
                group.push((second, vec![]));
            }
            group[*slot].1.push(id);
            if plain(piece) {
                v.plain.push(id);
            }
        }
        v
    }
    fn piece(&self, id: u32) -> &[u8] {
        &self.bytes[self.offsets[id as usize]..self.offsets[id as usize + 1]]
    }
}
/// String content with no quote, backslash or control byte, and well-formed
/// UTF-8 (a character may continue in the next token).
fn plain(piece: &[u8]) -> bool {
    let mut pending = 0;
    piece.iter().all(|&b| match (pending, b) {
        (0, b'"' | b'\\') => false,
        (0, 0x20..=0x7f) => true,
        (0, 0xc2..=0xdf) => {
            pending = 1;
            true
        }
        (0, 0xe0..=0xef) => {
            pending = 2;
            true
        }
        (0, 0xf0..=0xf4) => {
            pending = 3;
            true
        }
        (0, _) => false,
        (_, 0x80..=0xbf) => {
            pending -= 1;
            true
        }
        _ => false,
    })
}

/// Keeps generation inside a grammar. The sampler's own choice is taken when
/// valid (one cheap check); otherwise all invalid tokens are masked and it
/// samples again, which is equivalent to sampling the masked distribution.
pub(crate) struct JsonConstraint {
    vocab: Vocabulary,
    matcher: Matcher,
    /// Reasoning models (GLM) think first: constrain only after this marker.
    gate: Option<&'static [u8]>,
    recent: Vec<u8>,
    allow: Vec<bool>,
    pub(crate) steps: usize,
    pub(crate) masks: usize,
    pub(crate) mask_time: Duration,
    /// Constraint work only; the sampler itself is excluded.
    pub(crate) time: Duration,
}
impl JsonConstraint {
    pub(crate) fn new(t: &Tokenizer, grammar: Arc<Grammar>, prompt: &[u32]) -> Result<Self> {
        let tail = t.decode(&prompt[prompt.len().saturating_sub(4)..])?;
        let thinking = tail.trim_end().ends_with("<think>");
        Ok(Self::with(Vocabulary::new(t)?, grammar, thinking))
    }
    fn with(vocab: Vocabulary, grammar: Arc<Grammar>, thinking: bool) -> Self {
        Self {
            vocab,
            matcher: Matcher::new(grammar),
            gate: thinking.then_some(b"</think>".as_slice()),
            recent: vec![],
            allow: vec![],
            steps: 0,
            masks: 0,
            mask_time: Duration::ZERO,
            time: Duration::ZERO,
        }
    }
    fn allows(&self, id: u32) -> bool {
        if self.vocab.eos.contains(&id) {
            return self.matcher.complete();
        }
        let i = id as usize;
        i < self.vocab.special.len()
            && !self.vocab.special[i]
            && self.matcher.accepts(self.vocab.piece(id))
    }
    /// Mark every token that may follow the current state.
    fn fill(&mut self, n: usize) {
        self.allow.clear();
        self.allow.resize(n, false);
        let allow = &mut self.allow;
        if self.matcher.complete() {
            for &e in &self.vocab.eos {
                allow[e as usize] = true;
            }
        }
        if self.matcher.plain_string() {
            for &t in &self.vocab.plain {
                allow[t as usize] = true;
            }
        }
        for (first, group) in (0..=255u8).zip(&self.vocab.groups) {
            let mut m = self.matcher.clone();
            if group.is_empty() || !m.step(first) {
                continue;
            }
            for (second, ids) in group {
                if ids.iter().all(|&t| allow[t as usize]) {
                    continue;
                }
                let mut m = m.clone();
                if *second < 256 && !m.step(*second as u8) {
                    continue;
                }
                for &t in ids {
                    let rest = self.vocab.piece(t).get(2..).unwrap_or(&[]);
                    if !allow[t as usize] && m.accepts(rest) {
                        allow[t as usize] = true;
                    }
                }
            }
        }
    }
    fn constrained(&mut self, logits: &mut [f32], sampler: &mut Sampler) -> Result<u32> {
        let first = sampler.sample(logits)?;
        let start = Instant::now();
        let valid = self.allows(first);
        self.time += start.elapsed();
        if valid {
            return Ok(first);
        }
        self.masks += 1;
        let start = Instant::now();
        self.fill(logits.len());
        let mut any = false;
        for (l, &a) in logits.iter_mut().zip(&self.allow) {
            if a {
                any = true;
            } else {
                *l = f32::MIN;
            }
        }
        self.mask_time += start.elapsed();
        self.time += start.elapsed();
        if !any {
            return Err(invalid("structured output: no token can continue the JSON"));
        }
        sampler.sample(logits)
    }
}
impl Constraint for JsonConstraint {
    fn choose(&mut self, logits: &mut [f32], sampler: &mut Sampler) -> Result<u32> {
        let id = match self.gate {
            Some(marker) => {
                // The answer must follow the reasoning, so it may not end here.
                for &e in &self.vocab.eos {
                    if let Some(l) = logits.get_mut(e as usize) {
                        *l = f32::MIN;
                    }
                }
                let id = sampler.sample(logits)?;
                if (id as usize) < self.vocab.special.len() {
                    self.recent.extend_from_slice(self.vocab.piece(id));
                }
                if self.recent.ends_with(marker) {
                    self.gate = None;
                }
                let keep = self.recent.len().saturating_sub(marker.len());
                self.recent.drain(..keep);
                id
            }
            None => {
                let id = self.constrained(logits, sampler)?;
                let start = Instant::now();
                if !self.vocab.eos.contains(&id) {
                    for &b in self.vocab.piece(id) {
                        if !self.matcher.step(b) {
                            return Err(invalid("structured output: masked token was sampled"));
                        }
                    }
                }
                self.time += start.elapsed();
                id
            }
        };
        self.steps += 1;
        Ok(id)
    }
}

impl Engine {
    /// Generate a server request, under its output grammar when it has one.
    pub(crate) fn generate_request(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        cancel: Cancellation,
        r: &Request,
        keep: Option<&mut Option<Session>>,
        output: impl FnMut(u32, &[u8]) -> Result<()>,
    ) -> Result<Generation> {
        let Some(grammar) = &r.format else {
            return self.generate_with(prompt, max_tokens, cancel, &r.sampling, keep, output);
        };
        let start = Instant::now();
        let mut c = JsonConstraint::new(&self.tokenizer, grammar.clone(), prompt)?;
        let setup = start.elapsed();
        let g = self.generate_constrained(
            prompt,
            max_tokens,
            cancel,
            &r.sampling,
            keep,
            Some(&mut c),
            output,
        )?;
        eprintln!(
            "ingot serve: structured output: setup {:.1} ms, constraint {:.1} us/token over \
             {} tokens, {} full mask(s) of {:.2} ms",
            setup.as_secs_f64() * 1e3,
            c.time.as_secs_f64() * 1e6 / c.steps.max(1) as f64,
            c.steps,
            c.masks,
            c.mask_time.as_secs_f64() * 1e3 / c.masks.max(1) as f64
        );
        Ok(g)
    }
}

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;
