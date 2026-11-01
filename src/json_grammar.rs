//! JSON grammar for structured outputs: a JSON Schema subset compiled into
//! nodes, and a byte-at-a-time matcher that accepts exactly the JSON texts the
//! schema allows (compact or indented; whitespace is limited so a masked
//! model cannot drift into blank lines). Token masking on top of
//! it lives in `structured.rs`.
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Container nesting limit; deeper output is masked.
const MAX_DEPTH: usize = 32;
/// Indentation after a newline. Between two tokens at most one newline and,
/// without one, a single space are allowed; nothing precedes the root value.
const MAX_SPACE: u8 = 32;
const MAX_NUMBER: u8 = 32;
/// Object properties and enum values are tracked in 64-bit masks.
const MAX_MEMBERS: usize = 64;

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// `props`: JSON-escaped names (without quotes) and value nodes. Without
    /// declared properties, `extra` is the node of any-key values (a map).
    Object {
        props: Vec<(Vec<u8>, u32)>,
        required: u64,
        extra: Option<u32>,
    },
    Array {
        items: u32,
        min: u64,
        max: u64,
    },
    String,
    Number {
        integer: bool,
    },
    /// Exact compact JSON texts (enum, const, booleans, null).
    Literals(Vec<Vec<u8>>),
    /// Alternatives told apart by their first byte.
    Union(Vec<u32>),
}

/// Compiled schema. Nodes 0..=5 describe any JSON value.
#[derive(Debug, PartialEq)]
pub(crate) struct Grammar {
    nodes: Vec<Node>,
    /// Bytes that can start a value of each node.
    first: Vec<[bool; 256]>,
    root: u32,
}
const ANY: u32 = 0;
const ANY_OBJECT: u32 = 1;

const ANNOTATIONS: [&str; 8] = [
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
];
const KEYWORDS: [&str; 13] = [
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "$ref",
    "$defs",
    "definitions",
];

struct Compiler<'a> {
    root: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, u32>,
}

fn literal(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).expect("JSON value serializes")
}

impl Grammar {
    fn base() -> Vec<Node> {
        vec![
            Node::Union(vec![1, 2, 3, 4, 5]),
            Node::Object {
                props: vec![],
                required: 0,
                extra: Some(ANY),
            },
            Node::Array {
                items: ANY,
                min: 0,
                max: u64::MAX,
            },
            Node::Literals(vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()]),
            Node::String,
            Node::Number { integer: false },
        ]
    }
    /// JSON mode: any JSON object.
    pub(crate) fn object() -> Self {
        Self::finish(Self::base(), ANY_OBJECT).expect("base grammar is valid")
    }
    /// Compile a JSON Schema; unsupported keywords are errors naming their path.
    pub(crate) fn schema(schema: &Value) -> Result<Self, String> {
        let mut c = Compiler {
            root: schema,
            nodes: Self::base(),
            refs: HashMap::new(),
        };
        let root = c.compile(schema, "#")?;
        Self::finish(c.nodes, root)
    }
    fn finish(nodes: Vec<Node>, root: u32) -> Result<Self, String> {
        // Fixed point: unions may refer to nodes compiled later (recursion).
        let mut first = vec![[false; 256]; nodes.len()];
        loop {
            let mut changed = false;
            for (i, node) in nodes.iter().enumerate() {
                let mut set = [false; 256];
                match node {
                    Node::Object { .. } => set[b'{' as usize] = true,
                    Node::Array { .. } => set[b'[' as usize] = true,
                    Node::String => set[b'"' as usize] = true,
                    Node::Number { .. } => {
                        for b in b"-0123456789" {
                            set[*b as usize] = true;
                        }
                    }
                    Node::Literals(list) => list.iter().for_each(|l| set[l[0] as usize] = true),
                    Node::Union(branches) => {
                        for &b in branches {
                            for (s, f) in set.iter_mut().zip(&first[b as usize]) {
                                *s |= *f;
                            }
                        }
                    }
                }
                if set != first[i] {
                    first[i] = set;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for node in &nodes {
            if let Node::Union(branches) = node {
                for (i, &a) in branches.iter().enumerate() {
                    for &b in &branches[i + 1..] {
                        if (0..256).any(|x| first[a as usize][x] && first[b as usize][x]) {
                            return Err("anyOf / type alternatives must start differently \
                                 (e.g. object, array, string, number, boolean, null)"
                                .into());
                        }
                    }
                }
            }
        }
        if first[root as usize].iter().all(|f| !f) {
            return Err("schema matches no JSON value".into());
        }
        Ok(Self { nodes, first, root })
    }
}

impl Compiler<'_> {
    fn push(&mut self, node: Node) -> u32 {
        self.nodes.push(node);
        self.nodes.len() as u32 - 1
    }
    fn compile(&mut self, schema: &Value, path: &str) -> Result<u32, String> {
        let s = match schema {
            Value::Bool(true) => return Ok(ANY),
            Value::Object(s) => s,
            _ => return Err(format!("{path}: schema must be an object")),
        };
        if let Some(k) = s
            .keys()
            .find(|k| !KEYWORDS.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()))
        {
            return Err(format!("{path}: unsupported JSON Schema keyword '{k}'"));
        }
        if let Some(r) = s.get("$ref") {
            return self.reference(r, path);
        }
        if let Some(values) = s.get("enum") {
            let values = values
                .as_array()
                .filter(|v| !v.is_empty() && v.len() <= MAX_MEMBERS)
                .ok_or_else(|| format!("{path}.enum must be an array of 1 to 64 values"))?;
            return Ok(self.push(Node::Literals(values.iter().map(literal).collect())));
        }
        if let Some(v) = s.get("const") {
            return Ok(self.push(Node::Literals(vec![literal(v)])));
        }
        if let Some(branches) = s.get("anyOf") {
            let branches = branches
                .as_array()
                .filter(|b| !b.is_empty())
                .ok_or_else(|| format!("{path}.anyOf must be a non-empty array"))?;
            let ids = branches
                .iter()
                .enumerate()
                .map(|(i, b)| self.compile(b, &format!("{path}.anyOf[{i}]")))
                .collect::<Result<_, _>>()?;
            return Ok(self.push(Node::Union(ids)));
        }
        let types = match s.get("type") {
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => ts
                .iter()
                .map(|t| {
                    t.as_str()
                        .ok_or_else(|| format!("{path}.type must hold strings"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(format!("{path}.type must be a string or array")),
            None if s.contains_key("properties") => vec!["object"],
            None if s.contains_key("items") => vec!["array"],
            None => return Ok(ANY),
        };
        let ids = types
            .iter()
            .map(|t| self.typed(t, s, path))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match ids[..] {
            [one] => one,
            _ => self.push(Node::Union(ids)),
        })
    }
    fn reference(&mut self, r: &Value, path: &str) -> Result<u32, String> {
        let r = r.as_str().unwrap_or("");
        if let Some(&id) = self.refs.get(r) {
            return Ok(id);
        }
        let target = ["#/$defs/", "#/definitions/"]
            .iter()
            .find_map(|prefix| {
                let name = r.strip_prefix(prefix)?;
                self.root.get(&prefix[2..prefix.len() - 1])?.get(name)
            })
            .ok_or_else(|| format!("{path}: unsupported or unknown $ref '{r}'"))?;
        // Reserve the node first so recursive references resolve to it.
        let id = self.push(Node::Union(vec![]));
        self.refs.insert(r.to_owned(), id);
        let body = self.compile(target, r)?;
        self.nodes[id as usize] = Node::Union(vec![body]);
        Ok(id)
    }
    fn typed(&mut self, t: &str, s: &Map<String, Value>, path: &str) -> Result<u32, String> {
        Ok(match t {
            "string" => self.push(Node::String),
            "number" => self.push(Node::Number { integer: false }),
            "integer" => self.push(Node::Number { integer: true }),
            "boolean" => self.push(Node::Literals(vec![b"true".to_vec(), b"false".to_vec()])),
            "null" => self.push(Node::Literals(vec![b"null".to_vec()])),
            "array" => {
                let items = match s.get("items") {
                    None => ANY,
                    Some(i) => self.compile(i, &format!("{path}.items"))?,
                };
                let bound = |k: &str| {
                    s.get(k)
                        .map(|v| v.as_u64().ok_or(format!("{path}.{k} must be an integer")))
                        .transpose()
                };
                let (min, max) = (bound("minItems")?, bound("maxItems")?);
                let (min, max) = (min.unwrap_or(0), max.unwrap_or(u64::MAX));
                if min > max {
                    return Err(format!("{path}: minItems exceeds maxItems"));
                }
                self.push(Node::Array { items, min, max })
            }
            "object" => self.object(s, path)?,
            _ => return Err(format!("{path}: unsupported type '{t}'")),
        })
    }
    fn object(&mut self, s: &Map<String, Value>, path: &str) -> Result<u32, String> {
        let empty = Map::new();
        let props = match s.get("properties") {
            None => &empty,
            Some(Value::Object(p)) if p.len() <= MAX_MEMBERS => p,
            Some(_) => return Err(format!("{path}.properties must be an object of at most 64")),
        };
        let mut compiled = vec![];
        for (name, schema) in props {
            let key = literal(&Value::String(name.clone()));
            let node = self.compile(schema, &format!("{path}.properties.{name}"))?;
            compiled.push((key[1..key.len() - 1].to_vec(), node));
        }
        let mut required = 0u64;
        for name in s
            .get("required")
            .and_then(Value::as_array)
            .unwrap_or(&vec![])
        {
            let i = name
                .as_str()
                .and_then(|n| props.keys().position(|k| k == n))
                .ok_or_else(|| format!("{path}.required: {name} is not in properties"))?;
            required |= 1 << i;
        }
        // Only declared properties are generated, which conforms whether or
        // not additional ones are allowed. Without declared properties a
        // schema-valued additionalProperties describes a map.
        let extra = match (props.is_empty(), s.get("additionalProperties")) {
            (true, None | Some(Value::Bool(true))) => Some(ANY),
            (true, Some(Value::Object(v))) => Some(self.compile(
                &Value::Object(v.clone()),
                &format!("{path}.additionalProperties"),
            )?),
            (false, Some(Value::Object(_))) => {
                return Err(format!(
                    "{path}: additionalProperties schemas are supported only without properties"
                ));
            }
            _ => None,
        };
        Ok(self.push(Node::Object {
            props: compiled,
            required,
            extra,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// After `{` or `[`.
    Open,
    /// After `,`: a key or item is required.
    Next,
    /// After an object key.
    Colon,
    /// After a member value.
    After,
}
#[derive(Debug, Clone, Copy)]
struct Frame {
    node: u32,
    phase: Phase,
    /// Objects: used property bits. Arrays: item count.
    used: u64,
    /// Objects: node of the value after the colon.
    value: u32,
}
// Number phases; the accepting ones can end the number.
const MINUS: u8 = 0;
const ZERO: u8 = 1;
const INT: u8 = 2;
const DOT: u8 = 3;
const FRAC: u8 = 4;
const EXP: u8 = 5;
const EXP_SIGN: u8 = 6;
const EXP_DIGITS: u8 = 7;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Lex {
    /// Between values: the top frame's phase decides.
    Struct,
    Value(u32),
    /// `escape`: 1 after a backslash, 2..=5 while reading \u hex digits
    /// (6: second digit after a leading D).
    /// `utf8`: continuation bytes still expected.
    Str {
        key: bool,
        escape: u8,
        utf8: u8,
    },
    /// Declared property name; `cand` holds the still matching properties.
    Key {
        cand: u64,
        pos: u16,
    },
    Num {
        integer: bool,
        phase: u8,
        len: u8,
    },
    Lit {
        node: u32,
        cand: u64,
        pos: u16,
    },
    Done,
}

/// Matching state over a grammar; `step` feeds one byte.
#[derive(Debug, Clone)]
pub(crate) struct Matcher {
    g: Arc<Grammar>,
    frames: Vec<Frame>,
    lex: Lex,
    space: u8,
    newline: bool,
}

impl Matcher {
    pub(crate) fn new(g: Arc<Grammar>) -> Self {
        let root = g.root;
        Self {
            g,
            frames: Vec::with_capacity(8),
            lex: Lex::Value(root),
            space: 0,
            newline: false,
        }
    }
    /// The bytes so far form a complete value; only the end may follow
    /// (or, for a top-level number, more digits).
    pub(crate) fn complete(&self) -> bool {
        match self.lex {
            Lex::Done => true,
            Lex::Num { phase, .. } => self.frames.is_empty() && accepting(phase),
            Lex::Lit { node, cand, pos } => self.frames.is_empty() && self.exact(node, cand, pos),
            _ => false,
        }
    }
    /// Inside a string with no escape or multi-byte character pending, where
    /// any plain text (no quote, backslash or control byte) is valid.
    pub(crate) fn plain_string(&self) -> bool {
        matches!(
            self.lex,
            Lex::Str {
                escape: 0,
                utf8: 0,
                ..
            }
        )
    }
    pub(crate) fn accepts(&self, bytes: &[u8]) -> bool {
        let mut m = self.clone();
        bytes.iter().all(|&b| m.step(b))
    }
    fn exact(&self, node: u32, cand: u64, pos: u16) -> bool {
        let Node::Literals(list) = &self.g.nodes[node as usize] else {
            return false;
        };
        bits(cand).any(|i| list[i].len() == pos as usize)
    }
    fn top(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("frame for structural state")
    }
    fn value_done(&mut self) {
        self.space = 0;
        self.newline = false;
        match self.frames.last_mut() {
            Some(f) => {
                f.phase = Phase::After;
                self.lex = Lex::Struct;
            }
            None => self.lex = Lex::Done,
        }
    }
    /// Feed one byte; false when it cannot continue a valid text (the state
    /// is then unspecified).
    pub(crate) fn step(&mut self, b: u8) -> bool {
        let white = matches!(b, b' ' | b'\n' | b'\r' | b'\t');
        match self.lex {
            Lex::Str { key, escape, utf8 } => self.string(b, key, escape, utf8),
            Lex::Key { cand, pos } => self.key(b, cand, pos),
            Lex::Num {
                integer,
                phase,
                len,
            } => match number(phase, b, integer) {
                Some(phase) if len < MAX_NUMBER => {
                    self.lex = Lex::Num {
                        integer,
                        phase,
                        len: len + 1,
                    };
                    true
                }
                Some(_) => false,
                None if accepting(phase) => {
                    self.value_done();
                    self.step(b)
                }
                None => false,
            },
            Lex::Lit { node, cand, pos } => {
                let Node::Literals(list) = &self.g.nodes[node as usize] else {
                    return false;
                };
                let next = bits(cand)
                    .filter(|&i| list[i].get(pos as usize) == Some(&b))
                    .fold(0, |m, i| m | 1 << i);
                if next != 0 {
                    self.lex = Lex::Lit {
                        node,
                        cand: next,
                        pos: pos + 1,
                    };
                    true
                } else if self.exact(node, cand, pos) {
                    self.value_done();
                    self.step(b)
                } else {
                    false
                }
            }
            Lex::Done => false,
            Lex::Value(_) if white && self.frames.is_empty() => false,
            _ if white => self.whitespace(b),
            Lex::Value(node) => {
                self.space = 0;
                self.newline = false;
                self.start(node, b)
            }
            Lex::Struct => {
                self.space = 0;
                self.newline = false;
                self.structural(b)
            }
        }
    }
    fn whitespace(&mut self, b: u8) -> bool {
        match b {
            b'\n' if !self.newline && self.space == 0 => self.newline = true,
            b' ' if self.space < if self.newline { MAX_SPACE } else { 1 } => self.space += 1,
            _ => return false,
        }
        true
    }
    fn start(&mut self, node: u32, b: u8) -> bool {
        let g = self.g.clone();
        match &g.nodes[node as usize] {
            Node::Union(branches) => {
                match branches.iter().find(|&&n| g.first[n as usize][b as usize]) {
                    Some(&n) => self.start(n, b),
                    None => false,
                }
            }
            Node::Object { .. } => self.open(node, b == b'{'),
            Node::Array { .. } => self.open(node, b == b'['),
            Node::String => {
                self.lex = Lex::Str {
                    key: false,
                    escape: 0,
                    utf8: 0,
                };
                b == b'"'
            }
            Node::Number { integer } => {
                let phase = match b {
                    b'-' => MINUS,
                    b'0' => ZERO,
                    b'1'..=b'9' => INT,
                    _ => return false,
                };
                self.lex = Lex::Num {
                    integer: *integer,
                    phase,
                    len: 1,
                };
                true
            }
            Node::Literals(list) => {
                let cand = (0..list.len())
                    .filter(|&i| list[i][0] == b)
                    .fold(0, |m, i| m | 1 << i);
                self.lex = Lex::Lit { node, cand, pos: 1 };
                cand != 0
            }
        }
    }
    fn structural(&mut self, b: u8) -> bool {
        let g = self.g.clone();
        let f = *self.top();
        match &g.nodes[f.node as usize] {
            Node::Object {
                props,
                required,
                extra,
            } => {
                let all = if props.len() == 64 {
                    u64::MAX
                } else {
                    (1u64 << props.len()) - 1
                };
                let closable = f.used & required == *required;
                match (f.phase, b) {
                    (Phase::Open, b'}') | (Phase::After, b'}') if closable => self.close(),
                    (Phase::Open | Phase::Next, b'"') if !props.is_empty() => {
                        let cand = all & !f.used;
                        self.lex = Lex::Key { cand, pos: 0 };
                        cand != 0
                    }
                    (Phase::Open | Phase::Next, b'"') if extra.is_some() => {
                        self.lex = Lex::Str {
                            key: true,
                            escape: 0,
                            utf8: 0,
                        };
                        true
                    }
                    (Phase::Colon, b':') => {
                        self.lex = Lex::Value(f.value);
                        true
                    }
                    (Phase::After, b',') if extra.is_some() || all & !f.used != 0 => {
                        self.top().phase = Phase::Next;
                        true
                    }
                    _ => false,
                }
            }
            Node::Array { items, min, max } => match (f.phase, b) {
                (Phase::Open, b']') if *min == 0 => self.close(),
                (Phase::After, b']') if f.used >= *min => self.close(),
                (Phase::After, b',') if f.used < *max => {
                    self.top().phase = Phase::Next;
                    true
                }
                (Phase::Open | Phase::Next, _) if f.used < *max => {
                    self.top().used += 1;
                    self.lex = Lex::Value(*items);
                    self.start(*items, b)
                }
                _ => false,
            },
            _ => false,
        }
    }
    fn open(&mut self, node: u32, ok: bool) -> bool {
        if !ok || self.frames.len() >= MAX_DEPTH {
            return false;
        }
        self.frames.push(Frame {
            node,
            phase: Phase::Open,
            used: 0,
            value: 0,
        });
        self.lex = Lex::Struct;
        true
    }
    fn close(&mut self) -> bool {
        self.frames.pop();
        self.value_done();
        true
    }
    fn key(&mut self, b: u8, cand: u64, pos: u16) -> bool {
        let g = self.g.clone();
        let Node::Object { props, .. } = &g.nodes[self.top().node as usize] else {
            return false;
        };
        if b == b'"' {
            let Some(i) = bits(cand).find(|&i| props[i].0.len() == pos as usize) else {
                return false;
            };
            let f = self.top();
            f.used |= 1 << i;
            f.value = props[i].1;
            f.phase = Phase::Colon;
            self.lex = Lex::Struct;
            return true;
        }
        let next = bits(cand)
            .filter(|&i| props[i].0.get(pos as usize) == Some(&b))
            .fold(0, |m, i| m | 1 << i);
        self.lex = Lex::Key {
            cand: next,
            pos: pos + 1,
        };
        next != 0
    }
    fn string(&mut self, b: u8, key: bool, escape: u8, utf8: u8) -> bool {
        let (escape, utf8) = match (escape, utf8) {
            (0, 0) => match b {
                b'"' if key => {
                    let node = self.top().node as usize;
                    let Node::Object { extra, .. } = self.g.nodes[node] else {
                        return false;
                    };
                    let f = self.top();
                    f.value = extra.unwrap_or(ANY);
                    f.phase = Phase::Colon;
                    self.lex = Lex::Struct;
                    return true;
                }
                b'"' => {
                    self.value_done();
                    return true;
                }
                b'\\' => (1, 0),
                0x20..=0x7f => (0, 0),
                0xc2..=0xdf => (0, 1),
                0xe0..=0xef => (0, 2),
                0xf0..=0xf4 => (0, 3),
                _ => return false,
            },
            (0, n) if (0x80..=0xbf).contains(&b) => (0, n - 1),
            (0, _) => return false,
            (1, _) => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => (0, 0),
                b'u' => (5, 0),
                _ => return false,
            },
            // Surrogate escapes (\uD800-\uDFFF) are refused: a lone one is
            // rejected by many JSON parsers.
            (5, _) if matches!(b, b'd' | b'D') => (6, 0),
            (6, _) => match b {
                b'0'..=b'7' => (3, 0),
                _ => return false,
            },
            (e, _) if b.is_ascii_hexdigit() => (if e == 2 { 0 } else { e - 1 }, 0),
            _ => return false,
        };
        self.lex = Lex::Str { key, escape, utf8 };
        true
    }
}

fn bits(mask: u64) -> impl Iterator<Item = usize> {
    (0..64).filter(move |i| mask >> i & 1 == 1)
}
fn accepting(phase: u8) -> bool {
    matches!(phase, ZERO | INT | FRAC | EXP_DIGITS)
}
fn number(phase: u8, b: u8, integer: bool) -> Option<u8> {
    let digit = b.is_ascii_digit();
    Some(match (phase, b) {
        (MINUS, b'0') => ZERO,
        (MINUS, b'1'..=b'9') | (INT, b'0'..=b'9') => INT,
        (ZERO | INT, b'.') if !integer => DOT,
        (ZERO | INT | FRAC, b'e' | b'E') if !integer => EXP,
        (DOT | FRAC, _) if digit => FRAC,
        (EXP, b'+' | b'-') => EXP_SIGN,
        (EXP | EXP_SIGN | EXP_DIGITS, _) if digit => EXP_DIGITS,
        _ => return None,
    })
}

#[cfg(test)]
#[path = "json_grammar_tests.rs"]
mod tests;
