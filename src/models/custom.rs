//! Custom models: a saved system prompt and parameters on top of a local model.
//! Each is `ROOT/custom/NAME.json`; no weights are copied.
use crate::{Result, invalid};
use serde_json::{Map, Value, json};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct Custom {
    pub name: String,
    pub base: String,
    pub system: String,
    pub parameters: Map<String, Value>,
}

const NUMBERS: [&str; 7] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "num_ctx",
    "num_predict",
];

fn path(root: &Path, name: &str) -> PathBuf {
    root.join("custom").join(format!("{name}.json"))
}

pub fn check_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'));
    if valid {
        Ok(())
    } else {
        Err(invalid(format!(
            "invalid model name {name:?}; use letters, digits, dots, dashes and underscores"
        )))
    }
}

fn check_parameter(key: &str, value: &Value) -> Result<()> {
    let valid = match key {
        "stop" => value
            .as_array()
            .is_some_and(|list| list.len() <= 4 && list.iter().all(Value::is_string)),
        "seed" => value.is_i64() || value.is_u64(),
        key if NUMBERS.contains(&key) => value.is_number(),
        _ => return Err(invalid(format!("unsupported parameter {key}"))),
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(format!("invalid value for {key}")))
    }
}

impl Custom {
    pub fn from_json(value: &Value) -> Result<Self> {
        let text = |key: &str| value[key].as_str().unwrap_or_default().to_owned();
        let custom = Self {
            name: text("name"),
            base: text("base"),
            system: text("system"),
            parameters: value["parameters"].as_object().cloned().unwrap_or_default(),
        };
        custom.check()?;
        Ok(custom)
    }
    pub fn check(&self) -> Result<()> {
        check_name(&self.name)?;
        if self.base.is_empty() {
            return Err(invalid("a custom model needs a base model"));
        }
        self.parameters
            .iter()
            .try_for_each(|(key, value)| check_parameter(key, value))
    }
    pub fn to_json(&self) -> Value {
        json!({"name": self.name, "base": self.base, "system": self.system,
            "parameters": self.parameters})
    }
}

pub fn list(root: &Path) -> Result<Vec<Custom>> {
    let entries = match fs::read_dir(root.join("custom")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut models = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let value: Value = serde_json::from_slice(&fs::read(&path)?)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            models.push(Custom::from_json(&value)?);
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

pub fn load(root: &Path, name: &str) -> Result<Option<Custom>> {
    if check_name(name).is_err() {
        return Ok(None);
    }
    match fs::read(path(root, name)) {
        Ok(bytes) => {
            let value: Value = serde_json::from_slice(&bytes)
                .map_err(|e| invalid(format!("custom model {name}: {e}")))?;
            Custom::from_json(&value).map(Some)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn save(root: &Path, custom: &Custom) -> Result<()> {
    custom.check()?;
    let target = path(root, &custom.name);
    if let Some(dir) = target.parent() {
        fs::create_dir_all(dir)?;
    }
    let bytes = serde_json::to_vec_pretty(&custom.to_json()).map_err(io::Error::other)?;
    fs::write(target, bytes)
}

/// True when a custom model of that name existed.
pub fn remove(root: &Path, name: &str) -> Result<bool> {
    check_name(name)?;
    match fs::remove_file(path(root, name)) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// A Modelfile as a custom model without a name. `FROM`, `SYSTEM` and
/// `PARAMETER` take effect and `LICENSE` is ignored; anything else is an error
/// rather than being dropped.
pub fn parse_modelfile(text: &str) -> Result<Custom> {
    let mut custom = Custom {
        name: String::new(),
        base: String::new(),
        system: String::new(),
        parameters: Map::new(),
    };
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        // `"""` values may span lines.
        let mut value = rest.to_owned();
        if let Some(open) = rest.strip_prefix("\"\"\"") {
            value = open.to_owned();
            while !value.ends_with("\"\"\"") || (value.len() < 3) {
                let next = lines
                    .next()
                    .ok_or_else(|| invalid("unterminated \"\"\" in Modelfile"))?;
                value.push('\n');
                value.push_str(next);
            }
            value.truncate(value.len() - 3);
        } else if let Some(inner) = rest.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
            value = inner.to_owned();
        }
        match word.to_ascii_uppercase().as_str() {
            "FROM" => custom.base = value.strip_suffix(":latest").unwrap_or(&value).into(),
            "SYSTEM" => custom.system = value,
            "LICENSE" => {}
            "PARAMETER" => {
                let (key, raw) = rest
                    .split_once(char::is_whitespace)
                    .ok_or_else(|| invalid(format!("PARAMETER needs a value: {line}")))?;
                let raw = raw.trim().trim_matches('"');
                if key == "stop" {
                    let list = custom.parameters.entry("stop").or_insert_with(|| json!([]));
                    if let Value::Array(list) = list {
                        list.push(json!(raw));
                    }
                } else {
                    let number: Value = serde_json::from_str(raw)
                        .map_err(|_| invalid(format!("{key} must be a number")))?;
                    check_parameter(key, &number)?;
                    custom.parameters.insert(key.into(), number);
                }
            }
            other => {
                return Err(invalid(format!("Modelfile {other} is not supported")));
            }
        }
    }
    if custom.base.is_empty() {
        return Err(invalid("Modelfile needs a FROM line"));
    }
    Ok(custom)
}

/// Applies a custom model to an OpenAI request: its system prompt unless the
/// request has one, and its parameters where the request sets none.
pub fn apply(custom: &Custom, request: &mut Value) {
    request["model"] = json!(custom.base);
    let has_system = request["messages"]
        .as_array()
        .is_some_and(|m| m.iter().any(|m| m["role"] == "system"));
    if !custom.system.is_empty()
        && !has_system
        && let Some(messages) = request["messages"].as_array_mut()
    {
        messages.insert(0, json!({"role": "system", "content": custom.system}));
    }
    for (key, value) in &custom.parameters {
        let key = if key == "num_predict" {
            "max_tokens"
        } else {
            key
        };
        // Sampling with temperature 0 is greedy, so an explicit value wins.
        if request.get(key).is_none_or(Value::is_null) {
            request[key] = value.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modelfiles_are_parsed() {
        let c = parse_modelfile(
            "# review\nFROM qwen3-8b-q4_k_m:latest\nSYSTEM \"\"\"Be brief.\nReview Rust.\"\"\"\n\
             PARAMETER temperature 0.3\nparameter stop \"END\"\nPARAMETER stop STOP\nLICENSE x\n",
        )
        .unwrap();
        assert_eq!(c.base, "qwen3-8b-q4_k_m");
        assert_eq!(c.system, "Be brief.\nReview Rust.");
        assert_eq!(c.parameters["temperature"], 0.3);
        assert_eq!(c.parameters["stop"], json!(["END", "STOP"]));
        assert!(parse_modelfile("SYSTEM x").is_err());
        assert!(parse_modelfile("FROM m\nTEMPLATE x").is_err());
        assert!(parse_modelfile("FROM m\nPARAMETER nope 1").is_err());
    }
    #[test]
    fn requests_take_defaults_only_where_unset() {
        let custom = Custom {
            name: "c".into(),
            base: "m".into(),
            system: "S".into(),
            parameters: json!({"temperature": 0.3, "num_predict": 5, "num_ctx": 4096})
                .as_object()
                .unwrap()
                .clone(),
        };
        let mut request = json!({"model": "c", "temperature": 0.9,
            "messages": [{"role": "user", "content": "hi"}]});
        apply(&custom, &mut request);
        assert_eq!(request["model"], "m");
        assert_eq!(request["temperature"], 0.9);
        assert_eq!(request["max_tokens"], 5);
        assert_eq!(request["num_ctx"], 4096);
        assert_eq!(request["messages"][0]["content"], "S");
        let mut request = json!({"messages": [{"role": "system", "content": "mine"}]});
        apply(&custom, &mut request);
        assert_eq!(request["messages"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn names_cannot_leave_the_folder() {
        assert!(check_name("rust-reviewer.v2").is_ok());
        for bad in ["", "../x", "a/b", ".hidden", "a b"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }
}
