//! App settings and chat history under `~/Library/Application Support/Ingot`
//! (`INGOT_SUPPORT` overrides it). The Mac app reads the settings when it
//! starts the engine; the file holds an API key, so only its owner can read it.
use crate::{Result, invalid};
use serde_json::{Map, Value, json};
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const SETTINGS: &str = "settings.json";
const CHATS: &str = "chats.json";
/// Port the Mac app tries first.
pub const APP_PORT: u16 = 11435;

pub fn support_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("INGOT_SUPPORT") {
        return Ok(dir.into());
    }
    let home = std::env::var_os("HOME").ok_or_else(|| invalid("HOME is not set"))?;
    Ok(Path::new(&home).join("Library/Application Support/Ingot"))
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)
}

fn defaults() -> Value {
    json!({
        "default_model": "",
        "context": 0,
        "keep_alive": "10m",
        "models_dir": "",
        "port": APP_PORT,
        "allow_network": false,
        "api_key": "",
        "check_updates": true,
        "menu_bar": true,
    })
}

fn check(key: &str, value: &Value) -> Result<()> {
    let valid = match key {
        "default_model" | "models_dir" | "api_key" => value.is_string(),
        "context" => value.as_u64().is_some_and(|n| n <= 1 << 20),
        "port" => value.as_u64().is_some_and(|n| (1024..=65535).contains(&n)),
        "allow_network" | "check_updates" | "menu_bar" => value.is_boolean(),
        "keep_alive" => value
            .as_str()
            .is_some_and(|text| text == "-1" || crate::server::worker::duration(text).is_ok()),
        _ => return Err(invalid(format!("unknown setting {key}"))),
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(format!("invalid value for {key}")))
    }
}

/// Stored settings over the defaults.
pub fn load() -> Result<Value> {
    let mut settings = defaults();
    let path = support_dir()?.join(SETTINGS);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(settings),
        Err(e) => return Err(e),
    };
    let stored: Map<String, Value> =
        serde_json::from_str(&text).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    for (key, value) in stored {
        check(&key, &value).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        settings[key] = value;
    }
    Ok(settings)
}

/// Merges `changes` into the stored settings and returns the result.
pub fn update(changes: &Value) -> Result<Value> {
    let changes = changes
        .as_object()
        .ok_or_else(|| invalid("settings must be a JSON object"))?;
    let mut settings = load()?;
    for (key, value) in changes {
        check(key, value)?;
        settings[key] = value.clone();
    }
    if settings["allow_network"] == true && settings["api_key"] == "" {
        return Err(invalid("allowing other devices needs an API key"));
    }
    let bytes = serde_json::to_vec_pretty(&settings).map_err(io::Error::other)?;
    write_private(&support_dir()?.join(SETTINGS), &bytes)?;
    Ok(settings)
}

/// The model folder setting, when one is stored.
pub fn models_dir() -> Option<PathBuf> {
    let settings = load().ok()?;
    let dir = settings["models_dir"].as_str()?;
    (!dir.is_empty()).then(|| dir.into())
}

/// Stored chat history as JSON text; an empty list when there is none.
pub fn load_chats() -> Result<String> {
    match fs::read_to_string(support_dir()?.join(CHATS)) {
        Ok(text) => Ok(text),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok("[]".into()),
        Err(e) => Err(e),
    }
}

pub fn save_chats(bytes: &[u8]) -> Result<()> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|e| invalid(format!("chats must be JSON: {e}")))?;
    if !value.is_array() {
        return Err(invalid("chats must be a JSON array"));
    }
    write_private(&support_dir()?.join(CHATS), bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn values_are_checked() {
        assert!(check("port", &json!(11435)).is_ok());
        assert!(check("port", &json!(80)).is_err());
        assert!(check("keep_alive", &json!("10m")).is_ok());
        assert!(check("keep_alive", &json!("-1")).is_ok());
        assert!(check("keep_alive", &json!("soon")).is_err());
        assert!(check("context", &json!(8192)).is_ok());
        assert!(check("unknown", &json!(1)).is_err());
    }
}
