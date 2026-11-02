//! HTTP/1.1 subset: request parsing and response writing.
use super::auth;
use crate::openai::ApiError;
use serde_json::Value;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

const MAX_HEADER: usize = 64 * 1024;
const MAX_BODY: usize = 16 * 1024 * 1024;

pub(super) struct HttpRequest {
    pub(super) method: String,
    pub(super) path: String,
    /// The target after `?`, still percent-encoded.
    pub(super) query: String,
    pub(super) body: Vec<u8>,
    pub(super) credentials: auth::Credentials,
    origin: Option<String>,
    host: Option<String>,
}
impl HttpRequest {
    /// True when the request names this machine. Chats and settings are served
    /// only then, which keeps a hostile DNS name away from them.
    pub(super) fn names_loopback(&self) -> bool {
        loopback(self.host.as_deref().unwrap_or(""))
    }
    /// Value of one query parameter, decoded.
    pub(super) fn parameter(&self, name: &str) -> Option<String> {
        self.query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| *key == name)
            .map(|(_, value)| decode(value))
    }
    /// True for a state-changing request from a page of another site, which
    /// a browser sends without asking. Models can be downloaded and deleted
    /// here, so this also stops a hostile DNS name that resolves to loopback.
    pub(super) fn foreign(&self, keyed: bool) -> bool {
        let Some(origin) = self.origin.as_deref().filter(|_| self.method != "GET") else {
            return false;
        };
        let host = self.host.as_deref().unwrap_or("");
        let site = origin.split_once("://").map_or("", |(_, site)| site);
        !site.eq_ignore_ascii_case(host) || !keyed && !loopback(host)
    }
}
pub(super) fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(bytes.len()), 0);
    while i < bytes.len() {
        let hex = (bytes[i] == b'%')
            .then(|| text.get(i + 1..i + 3))
            .flatten()
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match (hex, bytes[i]) {
            (Some(byte), _) => {
                out.push(byte);
                i += 2;
            }
            (None, b'+') => out.push(b' '),
            (None, byte) => out.push(byte),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
fn loopback(host: &str) -> bool {
    let name = match host.rsplit_once(':') {
        Some((name, port)) if port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => host,
    };
    matches!(name, "localhost" | "127.0.0.1" | "[::1]")
}
pub(super) fn read_request(stream: &mut TcpStream) -> std::result::Result<HttpRequest, ApiError> {
    let mut buf = vec![];
    let mut chunk = [0u8; 8192];
    let end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        if buf.len() > MAX_HEADER {
            return Err(ApiError::new(431, "request headers too large"));
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return Err(ApiError::new(400, "incomplete request")),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = std::str::from_utf8(&buf[..end])
        .map_err(|_| ApiError::new(400, "headers must be UTF-8"))?;
    let mut lines = head.split("\r\n");
    let line: Vec<_> = lines.next().unwrap_or("").split(' ').collect();
    let [method, target, version] = line[..] else {
        return Err(ApiError::new(400, "malformed request line"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(ApiError::new(400, "only HTTP/1.x is supported"));
    }
    let mut length = None;
    let mut expect_continue = false;
    let mut credentials = auth::Credentials::default();
    let (mut origin, mut host) = (None, None);
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ApiError::new(400, "malformed header"))?;
        let value = value.trim();
        let name = name.trim().to_ascii_lowercase();
        if credentials.header(&name, value) {
            continue;
        }
        match name.as_str() {
            "origin" => origin = Some(value.to_owned()),
            "host" => host = Some(value.to_owned()),
            "content-length" => {
                length = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| ApiError::new(400, "invalid Content-Length"))?,
                )
            }
            "transfer-encoding" => {
                return Err(ApiError::new(
                    501,
                    "Transfer-Encoding is not supported; send Content-Length",
                ));
            }
            "expect" => expect_continue = value.eq_ignore_ascii_case("100-continue"),
            _ => {}
        }
    }
    let length = match (method, length) {
        ("POST", None) => return Err(ApiError::new(411, "Content-Length is required")),
        (_, l) => l.unwrap_or(0),
    };
    if length > MAX_BODY {
        return Err(ApiError::new(413, "request body too large"));
    }
    let mut body = buf[end + 4..].to_vec();
    if expect_continue && body.len() < length {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|_| ApiError::new(400, "connection closed"))?;
    }
    while body.len() < length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return Err(ApiError::new(400, "incomplete request body")),
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }
    body.truncate(length);
    Ok(HttpRequest {
        method: method.to_owned(),
        path: target.split('?').next().unwrap_or("").to_owned(),
        query: target.split_once('?').map_or("", |(_, q)| q).to_owned(),
        body,
        credentials,
        origin,
        host,
    })
}
pub(crate) fn write_head(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    length: Option<usize>,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nConnection: close\r\n"
    );
    match length {
        Some(n) => head += &format!("Content-Length: {n}\r\n"),
        None => head += "Cache-Control: no-cache\r\n",
    }
    stream.write_all((head + "\r\n").as_bytes())
}
pub(crate) fn respond(stream: &mut TcpStream, status: u16, body: &Value) {
    let body = body.to_string();
    // The client may already be gone; there is nobody left to report to.
    let _ = write_head(stream, status, "application/json", Some(body.len()))
        .and_then(|_| stream.write_all(body.as_bytes()));
}
pub(super) fn sse(stream: &mut TcpStream, value: &Value) -> std::io::Result<()> {
    stream.write_all(format!("data: {value}\n\n").as_bytes())
}
/// True once the peer has closed its side (the whole request was already read).
pub(super) fn disconnected(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return true;
    }
    let gone = match stream.peek(&mut [0u8]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != ErrorKind::WouldBlock,
    };
    gone || stream.set_nonblocking(false).is_err()
}
