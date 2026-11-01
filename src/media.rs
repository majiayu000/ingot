//! Images in API requests: OpenAI `image_url` content parts and Anthropic
//! `image` blocks. Each image becomes the chat template's image marker in the
//! message text, and its bytes are collected in prompt order. Inline data
//! (`data:` URLs, base64) is always accepted; remote URLs only when the server
//! was started with `--allow-remote-images`.
use crate::openai::{ApiError, Prompt, bad};
use crate::vision::IMAGE_MARKER;
use base64::Engine as _;
use serde_json::Value;
use std::io::Read;
use std::time::Duration;

/// Largest accepted encoded image, inline or downloaded.
pub(crate) const MAX_IMAGE_BYTES: usize = 20 << 20;
const MEDIA_TYPES: [&str; 3] = ["image/png", "image/jpeg", "image/jpg"];

#[derive(Debug, PartialEq)]
pub(crate) enum ImageSource {
    Bytes(Vec<u8>),
    Url(String),
}

fn decode_base64(data: &str, what: &str) -> Result<Vec<u8>, ApiError> {
    let compact: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(compact)
        .map_err(|e| bad(format!("{what}: invalid base64 image data: {e}"), None))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(bad(
            format!("{what}: image is larger than {MAX_IMAGE_BYTES} bytes"),
            None,
        ));
    }
    Ok(bytes)
}

fn media_type(kind: &str, what: &str) -> Result<(), ApiError> {
    if MEDIA_TYPES.contains(&kind.to_ascii_lowercase().as_str()) {
        Ok(())
    } else {
        Err(bad(
            format!("{what}: unsupported image type {kind:?}; send PNG or JPEG"),
            None,
        ))
    }
}

fn source(url: &str, what: &str) -> Result<ImageSource, ApiError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) = rest
            .split_once(',')
            .ok_or_else(|| bad(format!("{what}: malformed data URL"), None))?;
        let kind = header
            .strip_suffix(";base64")
            .ok_or_else(|| bad(format!("{what}: data URLs must be base64"), None))?;
        media_type(kind, what)?;
        return decode_base64(data, what).map(ImageSource::Bytes);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(ImageSource::Url(url.to_owned()));
    }
    Err(bad(
        format!("{what}: image URL must be a data: URL or http(s) URL"),
        None,
    ))
}

/// Text of one OpenAI content part; an `image_url` part adds its image.
pub(crate) fn openai_part(
    part: &Value,
    what: &str,
    images: &mut Vec<ImageSource>,
) -> Result<String, ApiError> {
    match (part.get("type").and_then(Value::as_str), part.get("text")) {
        (Some("text"), Some(Value::String(t))) => Ok(t.clone()),
        (Some("image_url"), _) => {
            let url = match part.get("image_url") {
                Some(Value::String(url)) => Some(url.as_str()),
                Some(Value::Object(o)) => o.get("url").and_then(Value::as_str),
                _ => None,
            }
            .ok_or_else(|| bad(format!("{what}: image_url needs a url"), None))?;
            images.push(source(url, what)?);
            Ok(IMAGE_MARKER.to_owned())
        }
        _ => Err(bad(
            format!("{what} supports only text and image_url parts"),
            None,
        )),
    }
}

/// Text of an OpenAI content array; images are accepted in user messages.
pub(crate) fn openai_content(
    parts: &[Value],
    i: usize,
    user: bool,
    images: &mut Vec<ImageSource>,
) -> Result<String, ApiError> {
    let (what, before) = (format!("messages[{i}].content"), images.len());
    let text = (parts.iter())
        .map(|part| openai_part(part, &what, images))
        .collect::<Result<String, _>>()?;
    if !user && images.len() > before {
        return Err(bad(
            format!("{what}: images need a user message"),
            Some("messages"),
        ));
    }
    Ok(text)
}

/// An Anthropic `image` block: base64 data, or a URL source.
pub(crate) fn anthropic_image(
    block: &Value,
    what: &str,
    images: &mut Vec<ImageSource>,
) -> Result<String, ApiError> {
    let s = block
        .get("source")
        .ok_or_else(|| bad(format!("{what}: image block needs a source"), None))?;
    let image = match s.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let kind = s.get("media_type").and_then(Value::as_str).unwrap_or("");
            media_type(kind, what)?;
            let data = s
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| bad(format!("{what}: base64 image source needs data"), None))?;
            ImageSource::Bytes(decode_base64(data, what)?)
        }
        Some("url") => source(
            s.get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| bad(format!("{what}: url image source needs a url"), None))?,
            what,
        )?,
        _ => {
            return Err(bad(
                format!("{what}: image source type must be base64 or url"),
                None,
            ));
        }
    };
    images.push(image);
    Ok(IMAGE_MARKER.to_owned())
}

/// Download remote images of a chat prompt, or refuse them without opt-in.
pub(crate) fn resolve(prompt: &mut Prompt, allow_remote: bool) -> Result<(), ApiError> {
    let Prompt::Chat { images, .. } = prompt else {
        return Ok(());
    };
    for image in images.iter_mut() {
        let ImageSource::Url(url) = image else {
            continue;
        };
        if !allow_remote {
            return Err(bad(
                "remote image URLs are disabled; send the image inline as a data: URL \
                 (or start the server with --allow-remote-images)",
                Some("messages"),
            ));
        }
        *image = ImageSource::Bytes(fetch(url)?);
    }
    Ok(())
}

fn fetch(url: &str) -> Result<Vec<u8>, ApiError> {
    let failed = |e: &dyn std::fmt::Display| bad(format!("cannot fetch image {url}: {e}"), None);
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .new_agent();
    let response = agent.get(url).call().map_err(|e| failed(&e))?;
    let mut bytes = Vec::new();
    response
        .into_body()
        .into_reader()
        .take(MAX_IMAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| failed(&e))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(failed(&format!("larger than {MAX_IMAGE_BYTES} bytes")));
    }
    Ok(bytes)
}

/// Inline image bytes of a resolved chat prompt, in prompt order.
pub(crate) fn bytes(prompt: &Prompt) -> Result<Vec<&[u8]>, ApiError> {
    let Prompt::Chat { images, .. } = prompt else {
        return Ok(vec![]);
    };
    images
        .iter()
        .map(|i| match i {
            ImageSource::Bytes(b) => Ok(b.as_slice()),
            ImageSource::Url(_) => Err(bad("remote image was not fetched", Some("messages"))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PNG: &str = "iVBORw0KGgo=";

    #[test]
    fn parses_openai_and_anthropic_images() {
        let mut images = vec![];
        let part = json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG}")}});
        assert_eq!(openai_part(&part, "m", &mut images).unwrap(), IMAGE_MARKER);
        let part = json!({"type": "image_url", "image_url": "https://example.com/a.jpg"});
        openai_part(&part, "m", &mut images).unwrap();
        let block = json!({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": PNG}});
        anthropic_image(&block, "m", &mut images).unwrap();
        assert_eq!(images[0], ImageSource::Bytes(b"\x89PNG\r\n\x1a\n".to_vec()));
        assert_eq!(
            images[1],
            ImageSource::Url("https://example.com/a.jpg".into())
        );
        for (bad_url, expected) in [
            ("data:image/gif;base64,AAAA", "unsupported image type"),
            ("data:image/png,raw", "must be base64"),
            ("data:image/png;base64,***", "invalid base64"),
            ("file:///etc/passwd", "data: URL or http(s)"),
        ] {
            let part = json!({"type": "image_url", "image_url": {"url": bad_url}});
            let e = openai_part(&part, "m", &mut images).unwrap_err();
            assert!(e.message.contains(expected), "{bad_url}: {}", e.message);
        }
        let block = json!({"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": PNG}});
        assert!(anthropic_image(&block, "m", &mut images).is_err());
    }

    #[test]
    fn remote_images_need_opt_in() {
        let mut prompt = Prompt::Chat {
            messages: vec![],
            developer: false,
            images: vec![ImageSource::Url("https://example.com/a.png".into())],
        };
        let e = resolve(&mut prompt, false).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("--allow-remote-images"));
        assert!(bytes(&prompt).is_err());
    }
}
