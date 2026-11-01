//! Image input: PNG/JPEG decoding to 8-bit RGB and the Qwen-VL preprocessing
//! (aspect-preserving "smart resize" to a multiple of the merged patch size,
//! Pillow-compatible bicubic resampling, mean/std normalization).
use crate::{Result, invalid};

/// Decoded images larger than this are refused before any resampling.
const MAX_PIXELS: usize = 64 * 1024 * 1024;

pub struct Rgb {
    pub width: usize,
    pub height: usize,
    /// Row-major, three bytes per pixel.
    pub data: Vec<u8>,
}

/// Decode PNG or JPEG bytes. Alpha is dropped and grayscale is expanded, as
/// llama.cpp's three-channel stb_image load does.
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        decode_jpeg(bytes)
    } else {
        Err(invalid("unsupported image format: expected PNG or JPEG"))
    }
}

fn checked(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_PIXELS {
        return Err(invalid(format!(
            "image size {width}x{height} is empty or above {MAX_PIXELS} pixels"
        )));
    }
    Ok(())
}

fn decode_png(bytes: &[u8]) -> Result<Rgb> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| invalid(format!("PNG: {e}")))?;
    let (width, height) = {
        let info = reader.info();
        (info.width as usize, info.height as usize)
    };
    checked(width, height)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| invalid("PNG: image too large"))?;
    let mut buffer = vec![0; size];
    let frame = reader
        .next_frame(&mut buffer)
        .map_err(|e| invalid(format!("PNG: {e}")))?;
    let channels = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(invalid("PNG: palette was not expanded")),
    };
    if frame.bit_depth != png::BitDepth::Eight {
        return Err(invalid("PNG: unsupported bit depth"));
    }
    let mut data = Vec::with_capacity(width * height * 3);
    for row in buffer[..frame.buffer_size()].chunks_exact(frame.line_size) {
        for pixel in row[..width * channels].chunks_exact(channels) {
            match channels {
                1 | 2 => data.extend([pixel[0]; 3]),
                _ => data.extend(&pixel[..3]),
            }
        }
    }
    Ok(Rgb {
        width,
        height,
        data,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb> {
    use zune_jpeg::zune_core::{
        bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions,
    };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(1 << 16)
        .set_max_height(1 << 16);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|e| invalid(format!("JPEG: {e:?}")))?;
    let (width, height) = decoder
        .dimensions()
        .ok_or_else(|| invalid("JPEG: missing dimensions"))?;
    checked(width, height)?;
    let data = decoder
        .decode()
        .map_err(|e| invalid(format!("JPEG: {e:?}")))?;
    if data.len() != width * height * 3 {
        return Err(invalid("JPEG: unexpected decoded size"));
    }
    Ok(Rgb {
        width,
        height,
        data,
    })
}

/// Qwen2-VL `smart_resize`: round each edge to a multiple of `align`, then
/// scale into [min_pixels, max_pixels] keeping the aspect ratio. Rounding is
/// Python's half-to-even, in f32 as llama.cpp computes it.
pub fn smart_size(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = align as f32;
    let round = |x: f32| ((x / f).round_ties_even() as usize) * align;
    let ceil = |x: f32| ((x / f).ceil() as usize) * align;
    let floor = |x: f32| ((x / f).floor() as usize) * align;
    let (w, h) = (width as f32, height as f32);
    let mut wb = align.max(round(w));
    let mut hb = align.max(round(h));
    if hb * wb > max_pixels {
        let beta = (h * w / max_pixels as f32).sqrt();
        hb = align.max(floor(h / beta));
        wb = align.max(floor(w / beta));
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f32 / (h * w)).sqrt();
        hb = ceil(h * beta);
        wb = ceil(w * beta);
    }
    (wb, hb)
}

const PRECISION: u32 = 22;

/// Pillow's bicubic (a = -0.5) filter taps for one axis in 22-bit fixed point.
fn taps(input: usize, output: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        let a = -0.5;
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(output);
    let mut weights = vec![0i32; output * ksize];
    for xx in 0..output {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let count = xmax - xmin;
        let pre: Vec<f64> = (0..count)
            .map(|x| filter((x as f64 + xmin as f64 - center + 0.5) / filterscale))
            .collect();
        let total: f64 = pre.iter().sum();
        for (x, &w) in pre.iter().enumerate() {
            let w = if total != 0.0 { w / total } else { w };
            let fixed = w * (1u64 << PRECISION) as f64 + if w < 0.0 { -0.5 } else { 0.5 };
            weights[xx * ksize + x] = fixed as i32;
        }
        bounds.push((xmin, count));
    }
    (ksize, bounds, weights)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION).clamp(0, 255) as u8
}

/// Pillow-compatible separable bicubic resize, horizontal pass first.
pub fn resize(src: &Rgb, width: usize, height: usize) -> Rgb {
    let mut data = src.data.clone();
    let mut w = src.width;
    if width != src.width {
        let (k, bounds, weights) = taps(src.width, width);
        let mut out = vec![0u8; width * src.height * 3];
        for y in 0..src.height {
            let row = &data[y * w * 3..(y + 1) * w * 3];
            for (xx, &(xmin, count)) in bounds.iter().enumerate() {
                let mut acc = [1i32 << (PRECISION - 1); 3];
                for x in 0..count {
                    let p = &row[(xmin + x) * 3..];
                    for c in 0..3 {
                        acc[c] += p[c] as i32 * weights[xx * k + x];
                    }
                }
                for c in 0..3 {
                    out[(y * width + xx) * 3 + c] = clip8(acc[c]);
                }
            }
        }
        data = out;
        w = width;
    }
    if height != src.height {
        let (k, bounds, weights) = taps(src.height, height);
        let row = w * 3;
        let mut out = vec![0u8; row * height];
        let mut acc = vec![0i32; row];
        for (yy, &(ymin, count)) in bounds.iter().enumerate() {
            acc.fill(1 << (PRECISION - 1));
            for y in 0..count {
                let weight = weights[yy * k + y];
                for (a, &p) in acc
                    .iter_mut()
                    .zip(&data[(ymin + y) * row..(ymin + y + 1) * row])
                {
                    *a += p as i32 * weight;
                }
            }
            for (o, &a) in out[yy * row..(yy + 1) * row].iter_mut().zip(&acc) {
                *o = clip8(a);
            }
        }
        data = out;
    }
    Rgb {
        width,
        height: if height != src.height {
            height
        } else {
            src.height
        },
        data,
    }
}

/// Normalized patches of a resized image in 2x2-merge order: for each merged
/// cell (raster order), its four patches (row-major), each flattened as
/// channel, row, column to match the patch-embedding convolution weights.
pub fn patches(image: &Rgb, patch: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let (gw, gh) = (image.width / patch, image.height / patch);
    let size = 3 * patch * patch;
    let mut out = Vec::with_capacity(gw * gh * size);
    for y in (0..gh).step_by(2) {
        for x in (0..gw).step_by(2) {
            for (py, px) in [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                for c in 0..3 {
                    for ky in 0..patch {
                        let row = (py * patch + ky) * image.width;
                        for kx in 0..patch {
                            let v = image.data[(row + px * patch + kx) * 3 + c] as f32 / 255.0;
                            out.push((v - mean[c]) / std[c]);
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_size_rounds_and_bounds_pixels() {
        // 32-pixel cells, 8..4096 cells.
        let (lo, hi) = (8 * 1024, 4096 * 1024);
        assert_eq!(smart_size(640, 480, 32, lo, hi), (640, 480));
        assert_eq!(smart_size(100, 50, 32, lo, hi), (128, 64));
        // Half-to-even: 48 / 32 = 1.5 rounds to 2, 80 / 32 = 2.5 rounds to 2.
        assert_eq!(smart_size(48, 80, 32, 0, hi), (64, 64));
        let (w, h) = smart_size(4000, 3000, 32, lo, hi);
        assert!(w * h <= hi && w % 32 == 0 && h % 32 == 0);
        let (w, h) = smart_size(10, 10, 32, lo, hi);
        assert!(w * h >= lo);
    }

    #[test]
    fn resize_keeps_flat_color_and_identity() {
        let src = Rgb {
            width: 7,
            height: 5,
            data: [10u8, 200, 30].repeat(35),
        };
        let up = resize(&src, 64, 32);
        assert_eq!((up.width, up.height), (64, 32));
        assert!(up.data.chunks(3).all(|p| p == [10, 200, 30]));
        let same = resize(&src, 7, 5);
        assert_eq!(same.data, src.data);
    }

    #[test]
    fn patches_follow_merge_order() {
        // 4x2 patches of 1 pixel; value encodes the patch column and row.
        let mut data = Vec::new();
        for y in 0..2u8 {
            for x in 0..4u8 {
                data.extend([x * 10 + y, 0, 0]);
            }
        }
        let image = Rgb {
            width: 4,
            height: 2,
            data,
        };
        let p = patches(&image, 1, [0.0; 3], [1.0; 3]);
        let red: Vec<f32> = p.chunks(3).map(|c| (c[0] * 255.0).round()).collect();
        assert_eq!(red, [0.0, 10.0, 1.0, 11.0, 20.0, 30.0, 21.0, 31.0]);
    }

    #[test]
    fn rejects_unknown_formats() {
        assert!(decode(b"GIF89a").is_err());
    }
}
