//! SentencePiece-style GGUF tokenizers, ported from llama.cpp's llama-vocab:
//! `llama` (SPM: score-ordered merges with `<0xXX>` byte fallback, used by
//! Llama 2, Mistral, Phi-3 and Gemma 3) and `gemma4` (BPE merges over raw
//! UTF-8 with `▁` for spaces). Special tokens are split out first exactly as
//! llama.cpp's `tokenizer_st_partition` does.
use crate::gguf::Gguf;
use crate::{Result, invalid};
use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const UNKNOWN: u32 = 1;
const UNUSED: u32 = 1 << 1;
const NORMAL: u32 = 1 << 2;
const CONTROL: u32 = 1 << 3;
const USER_DEFINED: u32 = 1 << 4;
const BYTE: u32 = 1 << 5;
const LSTRIP: u32 = 1 << 7;
const RSTRIP: u32 = 1 << 8;
const SPACE: &str = "\u{2581}";
/// Token texts llama.cpp treats as end-of-generation (and forces to control).
const EOG_TEXTS: [&str; 14] = [
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "<|end_of_text|>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<|return|>",
    "<|call|>",
];

pub(crate) struct Vocab {
    texts: Vec<String>,
    scores: Vec<f32>,
    attrs: Vec<u32>,
    /// Text to id; a later duplicate wins, as in llama.cpp's `token_to_id`.
    ids: AHashMap<String, u32>,
    /// Control, user-defined and unknown tokens, longest text first.
    special: Vec<u32>,
    /// `gemma4`: "left right" merge string to rank (first occurrence wins).
    ranks: Option<AHashMap<String, u32>>,
    add_space_prefix: bool,
    pub(crate) bos: Option<u32>,
    pub(crate) eog: Vec<u32>,
}

impl Vocab {
    pub(crate) fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.string("tokenizer.ggml.model")?;
        let bpe = match model {
            "llama" => false,
            "gemma4" => true,
            _ => return Err(invalid(format!("unsupported SentencePiece model {model}"))),
        };
        if g.optional_integer("tokenizer.ggml.add_eos_token") == Some(1) {
            return Err(invalid(
                "unsupported tokenizer feature: tokenizer.ggml.add_eos_token",
            ));
        }
        let mut texts = g.strings("tokenizer.ggml.tokens")?;
        let types = g.integers("tokenizer.ggml.token_type")?;
        let scores = match g.metadata.get("tokenizer.ggml.scores") {
            Some(_) => g
                .array("tokenizer.ggml.scores")?
                .iter()
                .map(|v| match v {
                    crate::gguf::Value::Float(f) => Ok(*f as f32),
                    _ => Err(invalid("tokenizer scores must be floats")),
                })
                .collect::<Result<Vec<_>>>()?,
            None if bpe => vec![0.0; texts.len()],
            None => return Err(invalid("SentencePiece vocabulary has no scores")),
        };
        if types.len() != texts.len() || scores.len() != texts.len() {
            return Err(invalid("token type or score count mismatch"));
        }
        let mut ids = AHashMap::with_capacity(texts.len());
        for (i, t) in texts.iter_mut().enumerate() {
            if t.is_empty() {
                *t = format!("[EMPTY_{i}]");
            }
            ids.insert(t.clone(), i as u32);
        }
        let mut attrs: Vec<u32> = types
            .iter()
            .map(|t| match t {
                1 => NORMAL,
                2 => UNKNOWN,
                3 => CONTROL,
                4 => USER_DEFINED,
                5 => UNUSED,
                6 => BYTE,
                _ => 0,
            })
            .collect();
        let has = |s: &str| ids.contains_key(s);
        // gemma4 has a normal `</s>` (llama.cpp detects it by `<|tool_response>`).
        let normal_s = has("<|tool_response>");
        let mut eog = Vec::new();
        for text in EOG_TEXTS {
            if let Some(&id) = ids.get(text) {
                if text == "</s>" && normal_s {
                    attrs[id as usize] = NORMAL;
                    continue;
                }
                attrs[id as usize] |= CONTROL;
                eog.push(id);
            }
        }
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if let Some(id) = g.optional_integer(key) {
                eog.push(id as u32);
            }
        }
        eog.sort_unstable();
        eog.dedup();
        let mut special: Vec<u32> = (0..texts.len() as u32)
            .filter(|&i| attrs[i as usize] & (CONTROL | USER_DEFINED | UNKNOWN) != 0)
            .collect();
        special.sort_by_key(|&i| std::cmp::Reverse(texts[i as usize].len()));
        // llama.cpp strips whitespace right of Phi-3 special tokens.
        let name = g
            .optional_string("general.name")
            .unwrap_or("")
            .to_lowercase();
        if name.contains("phi-3") || name.contains("phi3") {
            for &id in &special {
                attrs[id as usize] |= RSTRIP;
            }
            for (text, on) in [
                ("</s>", true),
                ("<unk>", false),
                ("<s>", false),
                ("<|endoftext|>", false),
            ] {
                if let Some(&id) = ids.get(text) {
                    let a = &mut attrs[id as usize];
                    *a = if on { *a | RSTRIP } else { *a & !RSTRIP };
                }
            }
        }
        let ranks = if bpe {
            let merges = g.strings("tokenizer.ggml.merges")?;
            let mut ranks = AHashMap::with_capacity(merges.len());
            for (i, m) in merges.into_iter().enumerate() {
                // llama.cpp splits at the first space after the first byte.
                if m.bytes().skip(1).any(|b| b == b' ') {
                    ranks.entry(m).or_insert(i as u32);
                }
            }
            Some(ranks)
        } else {
            None
        };
        let add_bos = bpe
            || g.optional_integer("tokenizer.ggml.add_bos_token")
                .is_none_or(|v| v == 1);
        let bos = if add_bos {
            Some(g.integer("tokenizer.ggml.bos_token_id")? as u32)
        } else {
            None
        };
        let add_space_prefix = !bpe
            && g.optional_integer("tokenizer.ggml.add_space_prefix")
                .is_none_or(|v| v == 1);
        Ok(Self {
            texts,
            scores,
            attrs,
            ids,
            special,
            ranks,
            add_space_prefix,
            bos,
            eog,
        })
    }

    /// Decoded bytes of every token (llama.cpp `token_to_piece` with specials).
    pub(crate) fn pieces(&self) -> Vec<Vec<u8>> {
        self.texts
            .iter()
            .zip(&self.attrs)
            .map(|(text, &attr)| {
                if attr & (UNKNOWN | CONTROL | USER_DEFINED) != 0 {
                    text.as_bytes().to_vec()
                } else if attr & NORMAL != 0 {
                    text.replace(SPACE, " ").into_bytes()
                } else if attr & BYTE != 0 {
                    text.get(3..5)
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                        .map(|b| vec![b])
                        .unwrap_or_default()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    pub(crate) fn encode(
        &self,
        text: &str,
        add_bos: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        if add_bos && let Some(bos) = self.bos {
            out.push(bos);
        }
        let mut prev_special = true;
        for fragment in self.partition(text, parse_special) {
            match fragment {
                Fragment::Token(id) => {
                    out.push(id);
                    prev_special = true;
                }
                Fragment::Text(start, end) => {
                    let raw = &text[start..end];
                    if self.ranks.is_some() {
                        self.bpe(&raw.replace(' ', SPACE), &mut out)?;
                    } else {
                        let prefix = if self.add_space_prefix && prev_special {
                            " "
                        } else {
                            ""
                        };
                        self.spm(&format!("{prefix}{raw}").replace(' ', SPACE), &mut out)?;
                    }
                    prev_special = false;
                }
            }
        }
        Ok(out)
    }

    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        // llama.cpp starts with no fragment at all for empty text.
        if text.is_empty() {
            return Vec::new();
        }
        let mut fragments = vec![Fragment::Text(0, text.len())];
        for &id in &self.special {
            let attr = self.attrs[id as usize];
            if !parse_special && attr & (CONTROL | UNKNOWN) != 0 {
                continue;
            }
            let needle = self.texts[id as usize].as_str();
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                let Fragment::Text(mut start, end) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(found) = text[start..end].find(needle) {
                    let at = start + found;
                    let mut left = at;
                    if attr & LSTRIP != 0 {
                        while left > start && is_space(text.as_bytes()[left - 1]) {
                            left -= 1;
                        }
                    }
                    if left > start {
                        next.push(Fragment::Text(start, left));
                    }
                    next.push(Fragment::Token(id));
                    start = at + needle.len();
                    if attr & RSTRIP != 0 {
                        while start < end && is_space(text.as_bytes()[start]) {
                            start += 1;
                        }
                    }
                }
                if start < end {
                    next.push(Fragment::Text(start, end));
                }
            }
            fragments = next;
        }
        fragments
    }

    fn byte_token(&self, b: u8) -> Option<u32> {
        self.ids.get(&format!("<0x{b:02X}>")).copied()
    }

    /// llama.cpp `llm_tokenizer_spm_session::tokenize`.
    fn spm(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        let mut symbols = chars(text);
        let mut queue = BinaryHeap::new();
        let push = |queue: &mut BinaryHeap<SpmBigram>, symbols: &[Symbol], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (a, b) = (&symbols[l as usize], &symbols[r as usize]);
            let piece = &text[a.start..a.start + a.n + b.n];
            if let Some(&id) = self.ids.get(piece) {
                queue.push(SpmBigram {
                    score: self.scores[id as usize],
                    left: l as usize,
                    right: r as usize,
                    size: piece.len(),
                });
            }
        };
        for i in 1..symbols.len() as isize {
            push(&mut queue, &symbols, i - 1, i);
        }
        while let Some(b) = queue.pop() {
            let (ln, rn) = (symbols[b.left].n, symbols[b.right].n);
            if ln == 0 || rn == 0 || ln + rn != b.size {
                continue;
            }
            symbols[b.left].n += rn;
            symbols[b.right].n = 0;
            let next = symbols[b.right].next;
            symbols[b.left].next = next;
            if next >= 0 {
                symbols[next as usize].prev = b.left as isize;
            }
            let prev = symbols[b.left].prev;
            push(&mut queue, &symbols, prev, b.left as isize);
            push(&mut queue, &symbols, b.left as isize, next);
        }
        let mut i = if symbols.is_empty() { -1 } else { 0 };
        while i >= 0 {
            let s = &symbols[i as usize];
            let piece = &text[s.start..s.start + s.n];
            if let Some(&id) = self.ids.get(piece) {
                out.push(id);
            } else {
                for &b in piece.as_bytes() {
                    let id = self
                        .byte_token(b)
                        .or_else(|| {
                            std::str::from_utf8(&[b])
                                .ok()
                                .and_then(|c| self.ids.get(c).copied())
                        })
                        .ok_or_else(|| invalid(format!("no byte fallback token for {b:#04x}")))?;
                    out.push(id);
                }
            }
            i = s.next;
        }
        Ok(())
    }

    /// llama.cpp `llm_tokenizer_bpe_session::tokenize` with the gemma4
    /// pre-tokenizer: runs of newlines and runs of other text.
    fn bpe(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        let ranks = self
            .ranks
            .as_ref()
            .ok_or_else(|| invalid("no BPE merges"))?;
        let bytes = text.as_bytes();
        let mut begin = 0;
        while begin < bytes.len() {
            let newline = bytes[begin] == b'\n';
            let mut end = begin;
            while end < bytes.len() && (bytes[end] == b'\n') == newline {
                end += 1;
            }
            let word = &text[begin..end];
            begin = end;
            let mut symbols = if newline && self.ids.contains_key(word) {
                vec![Symbol {
                    start: 0,
                    n: word.len(),
                    prev: -1,
                    next: -1,
                }]
            } else {
                chars(word)
            };
            let mut queue = BinaryHeap::new();
            let push =
                |queue: &mut BinaryHeap<BpeBigram>, symbols: &[Symbol], l: isize, r: isize| {
                    if l < 0 || r < 0 {
                        return;
                    }
                    let (a, b) = (&symbols[l as usize], &symbols[r as usize]);
                    let left = &word[a.start..a.start + a.n];
                    let right = &word[b.start..b.start + b.n];
                    if let Some(&rank) = ranks.get(&format!("{left} {right}")) {
                        queue.push(BpeBigram {
                            rank,
                            left: l as usize,
                            right: r as usize,
                            size: left.len() + right.len(),
                        });
                    }
                };
            for i in 1..symbols.len() as isize {
                push(&mut queue, &symbols, i - 1, i);
            }
            while let Some(b) = queue.pop() {
                let (ln, rn) = (symbols[b.left].n, symbols[b.right].n);
                // Symbols are contiguous, so a matching length is the same text.
                if ln == 0 || rn == 0 || ln + rn != b.size {
                    continue;
                }
                symbols[b.left].n += rn;
                symbols[b.right].n = 0;
                let next = symbols[b.right].next;
                symbols[b.left].next = next;
                if next >= 0 {
                    symbols[next as usize].prev = b.left as isize;
                }
                let prev = symbols[b.left].prev;
                push(&mut queue, &symbols, prev, b.left as isize);
                push(&mut queue, &symbols, b.left as isize, next);
            }
            for s in symbols.iter().filter(|s| s.n > 0) {
                let piece = &word[s.start..s.start + s.n];
                match self.ids.get(piece) {
                    Some(&id) => out.push(id),
                    None => out.extend(piece.bytes().filter_map(|b| self.byte_token(b))),
                }
            }
        }
        Ok(())
    }
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

enum Fragment {
    Text(usize, usize),
    Token(u32),
}

struct Symbol {
    start: usize,
    n: usize,
    prev: isize,
    next: isize,
}

fn chars(text: &str) -> Vec<Symbol> {
    let count = text.chars().count();
    text.char_indices()
        .enumerate()
        .map(|(i, (start, c))| Symbol {
            start,
            n: c.len_utf8(),
            prev: i as isize - 1,
            next: if i + 1 == count { -1 } else { i as isize + 1 },
        })
        .collect()
}

/// Highest score first; ties go to the leftmost pair.
struct SpmBigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}
impl Ord for SpmBigram {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .partial_cmp(&o.score)
            .unwrap_or(Ordering::Equal)
            .then(o.left.cmp(&self.left))
    }
}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for SpmBigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for SpmBigram {}

/// Lowest rank first; ties go to the leftmost pair.
struct BpeBigram {
    rank: u32,
    left: usize,
    right: usize,
    size: usize,
}
impl Ord for BpeBigram {
    fn cmp(&self, o: &Self) -> Ordering {
        o.rank.cmp(&self.rank).then(o.left.cmp(&self.left))
    }
}
impl PartialOrd for BpeBigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for BpeBigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for BpeBigram {}
