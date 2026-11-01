//! Image input for the engine: a vision projector attached to a model whose
//! token stream can carry image embeddings (native Qwen3.8 today).
use super::Engine;
use crate::model::Model;
use crate::vision::Encoder;
use crate::{Result, invalid};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Bytes kept free for encoder activations of the largest accepted image.
pub const ACTIVATION_RESERVE: usize = 2 << 30;

/// The projector for `model`: `explicit`, else one `mmproj*.gguf` beside it.
pub fn projector_for(model: &Path, explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        return Ok(Some(path.to_owned()));
    }
    let Some(dir) = model.parent().filter(|d| !d.as_os_str().is_empty()) else {
        return Ok(None);
    };
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
        })
        .collect();
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.pop()),
        _ => Err(invalid(format!(
            "several mmproj*.gguf files in {}; choose one with --mmproj",
            dir.display()
        ))),
    }
}

impl Engine {
    /// Load the vision projector at `path` for this model.
    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        let Model::Qwen(qwen) = &self.model else {
            return Err(invalid(
                "image input is supported only for Qwen3.8 (qwen4exp) models",
            ));
        };
        let encoder = Encoder::load(path)?;
        if encoder.output != qwen.width() {
            return Err(invalid(format!(
                "{} projects to width {}, the model's embedding width is {}",
                path.display(),
                encoder.output,
                qwen.width()
            )));
        }
        self.vision = Some(encoder);
        Ok(())
    }
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }
    /// The model's image tokens for an encoded PNG or JPEG. A repeated image
    /// reuses its earlier embeddings (and any kept session state that holds it).
    pub fn image_tokens(&mut self, bytes: &[u8]) -> Result<Vec<u32>> {
        let (Some(encoder), Model::Qwen(qwen)) = (&self.vision, &self.model) else {
            return Err(invalid(
                "this model has no vision projector, so it cannot read images \
                 (start it with --mmproj PATH)",
            ));
        };
        let key: [u8; 32] = Sha256::digest(bytes).into();
        if let Some(tokens) = qwen.image_tokens(&key) {
            return Ok(tokens);
        }
        let start = Instant::now();
        let rgb = crate::image::decode(bytes)?;
        let encoded = encoder.encode(&rgb)?;
        eprintln!(
            "ingot: image {}x{} -> {}x{} tokens, encoded in {:.2}s",
            rgb.width,
            rgb.height,
            encoded.width,
            encoded.height,
            start.elapsed().as_secs_f64()
        );
        qwen.register_image(key, &encoded.embeddings, encoded.width, encoded.height)
    }
    /// Replace each image placeholder token of `ids`, in order, with the
    /// tokens of the matching image. Without images `ids` is returned as is.
    pub fn insert_images(&mut self, ids: Vec<u32>, images: &[&[u8]]) -> Result<Vec<u32>> {
        if images.is_empty() {
            return Ok(ids);
        }
        let pad = match self
            .tokenizer
            .encode_continuation(crate::vision::IMAGE_PAD)?[..]
        {
            [id] => id,
            _ => return Err(invalid("this model's tokenizer has no image placeholder")),
        };
        let found = ids.iter().filter(|&&t| t == pad).count();
        if found != images.len() {
            return Err(invalid(format!(
                "the prompt has {found} image placeholder(s) for {} image(s)",
                images.len()
            )));
        }
        let mut out = Vec::with_capacity(ids.len());
        let mut next = images.iter();
        for t in ids {
            if t != pad {
                out.push(t);
                continue;
            }
            let image = next
                .next()
                .ok_or_else(|| invalid("image placeholder without an image"))?;
            out.extend(self.image_tokens(image)?);
        }
        Ok(out)
    }
}

pub(crate) fn has_images(tokens: &[u32]) -> bool {
    tokens.iter().any(|&t| crate::qwen::media::is_image(t))
}
