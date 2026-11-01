//! `--api-key`: API routes require `Authorization: Bearer KEY` or
//! `x-api-key: KEY`. The chat page itself stays public so it can ask for the key.
use crate::{Result, invalid};
use serde_json::{Value, json};
use std::net::IpAddr;

/// Credentials a request carried.
#[derive(Default)]
pub(crate) struct Credentials {
    pub(crate) bearer: Option<String>,
    pub(crate) x_api_key: Option<String>,
}
impl Credentials {
    /// Record one request header; returns false for unrelated headers.
    pub(crate) fn header(&mut self, name: &str, value: &str) -> bool {
        match name {
            "authorization" => {
                self.bearer = value
                    .split_once(' ')
                    .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                    .map(|(_, key)| key.trim().to_owned());
            }
            "x-api-key" => self.x_api_key = Some(value.to_owned()),
            _ => return false,
        }
        true
    }
    pub(crate) fn allows(&self, key: &str) -> bool {
        // Evaluate both so timing does not reveal which header matched.
        let bearer = self.bearer.as_deref().is_some_and(|k| same(k, key));
        let x = self.x_api_key.as_deref().is_some_and(|k| same(k, key));
        bearer | x
    }
}

/// Constant-time comparison (only the lengths can be learned).
fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
        diff |= std::hint::black_box(x) as usize;
    }
    diff == 0
}

/// Paths served without a key: the chat page and its script.
pub(crate) fn public(method: &str, path: &str) -> bool {
    method == "GET" && matches!(path, "/" | "/monitor.js")
}

/// 401 body in the shape of the API the path belongs to.
pub(crate) fn rejection(path: &str) -> Value {
    let message = "missing or invalid API key; send Authorization: Bearer KEY or x-api-key: KEY";
    if path == "/v1/messages" {
        json!({"type": "error", "error": {"type": "authentication_error", "message": message}})
    } else {
        json!({"error": {"message": message, "type": "invalid_request_error", "param": null,
            "code": "invalid_api_key"}})
    }
}

/// `--api-key`, else `INGOT_API_KEY`; an empty key is an error.
pub(crate) fn key(flag: Option<&str>) -> Result<Option<String>> {
    let key = match flag {
        Some(k) => Some(k.to_owned()),
        None => std::env::var("INGOT_API_KEY").ok(),
    };
    match key {
        Some(k) if k.trim().is_empty() => Err(invalid("the API key must not be empty")),
        Some(k) if k.trim() != k => {
            Err(invalid("the API key must not start or end with whitespace"))
        }
        k => Ok(k),
    }
}

/// A non-loopback address needs a key unless `--insecure` is given.
pub(crate) fn check_bind(host: &str, key: bool, insecure: bool) -> Result<()> {
    let loopback = host == "localhost" || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if loopback || key {
        return Ok(());
    }
    if insecure {
        eprintln!(
            "ingot serve: warning: listening on {host} with --insecure; the API has no authentication"
        );
        return Ok(());
    }
    Err(invalid(format!(
        "refusing to listen on {host} without authentication; give --api-key KEY \
         (or INGOT_API_KEY), or --insecure to serve the API to the network without one"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_either_header_with_the_exact_key() {
        let mut c = Credentials::default();
        assert!(!c.allows("k1"));
        assert!(c.header("authorization", "Bearer k1"));
        assert!(c.allows("k1") && !c.allows("k") && !c.allows("k12"));
        c.header("authorization", "Basic k1");
        assert!(!c.allows("k1"));
        c.header("x-api-key", "k1");
        assert!(c.allows("k1"));
        assert!(!c.header("content-type", "x"));
        assert!(same("", "") && !same("a", "") && !same("ab", "ac"));
    }
    #[test]
    fn non_loopback_hosts_need_a_key_or_insecure() {
        for host in ["127.0.0.1", "::1", "localhost", "127.0.0.2"] {
            assert!(check_bind(host, false, false).is_ok(), "{host}");
        }
        assert!(check_bind("0.0.0.0", false, false).is_err());
        assert!(check_bind("0.0.0.0", true, false).is_ok());
        assert!(check_bind("192.168.1.2", false, true).is_ok());
        assert!(key(Some("")).is_err() && key(Some(" k")).is_err());
        assert_eq!(key(Some("k")).unwrap().as_deref(), Some("k"));
        assert_eq!(
            rejection("/v1/messages")["error"]["type"],
            "authentication_error"
        );
        assert_eq!(rejection("/v1/models")["error"]["code"], "invalid_api_key");
    }
}
