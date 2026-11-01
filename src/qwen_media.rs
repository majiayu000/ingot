//! Image tokens in the qwen4exp token stream. Each merged image token is a
//! virtual id (`IMAGE_TOKEN | serial << 16 | index`) outside the vocabulary,
//! so equal prompts mean equal images for session and prefix reuse. Before a
//! forward, virtual ids become the PLE image token id; their embeddings come
//! from the registered image and their RoPE positions follow M-RoPE: every
//! token of an image shares the temporal position of its first token, rows and
//! columns advance the other two axes, and the image occupies
//! `max(columns, rows)` positions of the text that follows.
use super::Qwen;
use crate::metal::{Buffer, Command};
use crate::{Result, invalid};
use std::borrow::Cow;
use std::collections::HashMap;

pub(crate) const IMAGE_TOKEN: u32 = 0x8000_0000;
/// History and block tokens as the text model sees them.
pub(super) type Streams<'a> = (Cow<'a, [u32]>, Cow<'a, [u32]>);
const SERIALS: u32 = 1 << 15;
/// Registered images kept for reuse by later turns, least recently used first out.
const KEPT_IMAGES: usize = 64;

pub(crate) fn is_image(token: u32) -> bool {
    token & IMAGE_TOKEN != 0
}
fn split(token: u32) -> (u32, usize) {
    ((token >> 16) & (SERIALS - 1), (token & 0xffff) as usize)
}

struct Image {
    key: [u8; 32],
    embeddings: Buffer,
    cols: usize,
    rows: usize,
    used: u64,
}
struct Span {
    start: usize,
    len: usize,
    cols: usize,
    serial: u32,
    /// Temporal position shared by the image's tokens.
    t: usize,
}
pub(super) struct Media {
    /// PLE n-gram id that stands in for image tokens.
    token: u32,
    /// Interleaved M-RoPE (t, h, w) sections, when the model has them.
    sections: Option<[u32; 3]>,
    images: HashMap<u32, Image>,
    next: u32,
    clock: u64,
    /// Image spans of the stream of the forward in progress; empty for text.
    spans: Vec<Span>,
}
pub(super) enum Rope {
    /// Equal positions on all axes, `start + i * step` for row i.
    Linear(usize, usize),
    /// Explicit (t, h, w) per row.
    Rows(Buffer),
}

impl Media {
    pub(super) fn new(g: &crate::gguf::Gguf) -> Result<Self> {
        let token = g
            .optional_integer("qwen4exp.ple.image_token_id")
            .map_or_else(|| g.integer("qwen4exp.ple.eos_token_id"), Ok)? as u32;
        let sections = match g.integers("qwen4exp.rope.dimension_sections") {
            Ok(s) if s.len() >= 3 => Some([s[0] as u32, s[1] as u32, s[2] as u32]),
            _ => None,
        };
        Ok(Self::with(token, sections))
    }
    fn with(token: u32, sections: Option<[u32; 3]>) -> Self {
        Self {
            token,
            sections,
            images: HashMap::new(),
            next: 0,
            clock: 0,
            spans: Vec::new(),
        }
    }
    fn position(&self, index: usize) -> [u32; 3] {
        let mut shift = 0;
        for s in &self.spans {
            if index < s.start {
                break;
            }
            if index < s.start + s.len {
                let k = index - s.start;
                return [s.t, s.t + k / s.cols, s.t + k % s.cols].map(|v| v as u32);
            }
            shift += s.len - s.cols.max(s.len / s.cols);
        }
        [(index - shift) as u32; 3]
    }
}

impl Qwen {
    /// Virtual tokens of a registered image with this content key, if kept.
    pub(crate) fn image_tokens(&self, key: &[u8; 32]) -> Option<Vec<u32>> {
        let mut media = self.media.borrow_mut();
        media.clock += 1;
        let clock = media.clock;
        let (&serial, image) = media.images.iter_mut().find(|(_, i)| &i.key == key)?;
        image.used = clock;
        Some(tokens(serial, image.cols * image.rows))
    }
    /// Register merged image embeddings (`cols * rows` rows of the model width).
    pub(crate) fn register_image(
        &self,
        key: [u8; 32],
        embeddings: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<u32>> {
        let n = cols * rows;
        if self.media.borrow().sections.is_none() {
            return Err(invalid(
                "this model has no M-RoPE sections for image positions",
            ));
        }
        if n == 0 || n > 0x10000 || embeddings.len() != n * self.c.width {
            return Err(invalid(format!(
                "image embeddings must be {cols}x{rows} rows of {}",
                self.c.width
            )));
        }
        let buffer = self.gpu.floats(embeddings)?;
        let mut media = self.media.borrow_mut();
        if media.images.len() >= KEPT_IMAGES
            && let Some(&oldest) = media
                .images
                .iter()
                .min_by_key(|(_, i)| i.used)
                .map(|(s, _)| s)
        {
            media.images.remove(&oldest);
        }
        let serial = media.next;
        media.next = (serial + 1) % SERIALS;
        media.clock += 1;
        let used = media.clock;
        media.images.insert(
            serial,
            Image {
                key,
                embeddings: buffer,
                cols,
                rows,
                used,
            },
        );
        Ok(tokens(serial, n))
    }

    /// Lay out the image spans of `history + tokens` for this forward and map
    /// image tokens to the PLE image id. Text-only streams are borrowed as is.
    pub(super) fn enter<'a>(&self, history: &'a [u32], tokens: &'a [u32]) -> Result<Streams<'a>> {
        let mut media = self.media.borrow_mut();
        media.spans.clear();
        if !history.iter().chain(tokens).any(|&t| is_image(t)) {
            return Ok((Cow::Borrowed(history), Cow::Borrowed(tokens)));
        }
        let stream = || history.iter().chain(tokens).copied();
        let total = history.len() + tokens.len();
        let mut spans: Vec<Span> = Vec::new();
        let mut shift = 0;
        for (i, t) in stream().enumerate() {
            if !is_image(t) {
                continue;
            }
            let (serial, index) = split(t);
            let continues = spans
                .last()
                .is_some_and(|s| s.serial == serial && s.start + index == i && index < s.len);
            if continues {
                continue;
            }
            let image = media
                .images
                .get(&serial)
                .ok_or_else(|| invalid("image is no longer cached; send it again"))?;
            let len = image.cols * image.rows;
            if index != 0 || spans.last().is_some_and(|s| s.start + s.len > i) {
                return Err(invalid("incomplete image token run"));
            }
            if let Some(s) = spans.last() {
                shift += s.len - s.cols.max(s.len / s.cols);
            }
            spans.push(Span {
                start: i,
                len,
                cols: image.cols,
                serial,
                t: i - shift,
            });
        }
        // Every image run is complete, except one cut by the end of the stream.
        for s in &spans {
            let end = (s.start + s.len).min(total);
            if stream()
                .skip(s.start)
                .take(end - s.start)
                .enumerate()
                .any(|(k, t)| t != image_id(s.serial, k))
            {
                return Err(invalid("incomplete image token run"));
            }
        }
        media.spans = spans;
        let token = media.token;
        let map = |s: &[u32]| -> Vec<u32> {
            s.iter()
                .map(|&t| if is_image(t) { token } else { t })
                .collect()
        };
        Ok((Cow::Owned(map(history)), Cow::Owned(map(tokens))))
    }

    /// Positions for rows `first + i * step` of the current stream.
    pub(super) fn rope_plan(&self, first: usize, count: usize, step: usize) -> Result<Rope> {
        let media = self.media.borrow();
        if media.spans.is_empty() {
            return Ok(Rope::Linear(first, step));
        }
        let rows: Vec<[u32; 3]> = (0..count)
            .map(|i| media.position(first + i * step))
            .collect();
        let flat = rows.iter().all(|p| p[0] == p[1] && p[1] == p[2]);
        let delta = if count > 1 {
            rows[1][0].wrapping_sub(rows[0][0])
        } else {
            0
        };
        if flat
            && rows
                .windows(2)
                .all(|w| w[1][0].wrapping_sub(w[0][0]) == delta)
        {
            return Ok(Rope::Linear(rows[0][0] as usize, delta as usize));
        }
        let bytes: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Ok(Rope::Rows(self.gpu.bytes(&bytes)?))
    }
    /// RoPE with explicit per-row positions over `count` rows of `heads` heads.
    pub(super) fn rope_rows(
        &self,
        cmd: &Command,
        x: &Buffer,
        dim: usize,
        heads: usize,
        positions: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let n = dim * heads * count;
        let y = self.gpu.output(n)?;
        let s = self
            .media
            .borrow()
            .sections
            .ok_or_else(|| invalid("missing M-RoPE sections"))?;
        self.point(
            cmd,
            "rope_rows",
            &[x, positions, &y],
            &[
                dim as u32,
                heads as u32,
                self.c.rot as u32,
                self.c.rope.to_bits(),
                count as u32,
                s[0],
                s[1],
                s[2],
            ],
            n,
        )?;
        Ok(y)
    }
    /// Overwrite the embedding rows of image tokens among stream rows
    /// `first..first + count` of `e`.
    pub(super) fn embed_images(
        &self,
        cmd: &Command,
        e: &Buffer,
        first: usize,
        count: usize,
    ) -> Result<()> {
        let media = self.media.borrow();
        let width = self.c.width;
        for s in &media.spans {
            let (lo, hi) = (s.start.max(first), (s.start + s.len).min(first + count));
            if lo >= hi {
                continue;
            }
            let image = media
                .images
                .get(&s.serial)
                .ok_or_else(|| invalid("image is no longer cached; send it again"))?;
            let n = (hi - lo) * width;
            let rows = image.embeddings.view_f32((lo - s.start) * width, n)?;
            self.point(
                cmd,
                "copy_into",
                &[&rows, e],
                &[n as u32, ((lo - first) * width) as u32],
                n,
            )?;
        }
        Ok(())
    }
}

fn image_id(serial: u32, index: usize) -> u32 {
    IMAGE_TOKEN | serial << 16 | index as u32
}
fn tokens(serial: u32, n: usize) -> Vec<u32> {
    (0..n).map(|i| image_id(serial, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_follow_mrope_layout() {
        let mut media = Media::with(7, Some([11, 11, 10]));
        // Text 0..3, a 3x2 image at 3..9, then text.
        media.spans.push(Span {
            start: 3,
            len: 6,
            cols: 3,
            serial: 0,
            t: 3,
        });
        assert_eq!(media.position(2), [2, 2, 2]);
        assert_eq!(media.position(3), [3, 3, 3]);
        assert_eq!(media.position(5), [3, 3, 5]);
        assert_eq!(media.position(7), [3, 4, 4]);
        // The image advances max(3, 2) = 3 positions.
        assert_eq!(media.position(9), [6, 6, 6]);
        assert_eq!(media.position(10), [7, 7, 7]);
        assert!(is_image(image_id(5, 9)) && split(image_id(5, 9)) == (5, 9));
    }
}
