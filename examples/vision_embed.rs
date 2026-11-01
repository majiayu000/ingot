//! Encode one image with a vision projector and write the merged-token
//! embeddings as little-endian f32 rows (validation against llama.cpp mtmd).
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, mmproj, image, out] = args.as_slice() else {
        eprintln!("usage: vision_embed MMPROJ IMAGE OUT.f32");
        std::process::exit(2);
    };
    let encoder = ingot::vision::Encoder::load(mmproj.as_ref())?;
    let rgb = ingot::image::decode(&std::fs::read(image)?)?;
    let start = Instant::now();
    let encoded = encoder.encode(&rgb)?;
    let elapsed = start.elapsed();
    let bytes: Vec<u8> = encoded
        .embeddings
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out, bytes)?;
    println!(
        "image {}x{} -> {}x{} tokens of {} in {:.3}s",
        rgb.width,
        rgb.height,
        encoded.width,
        encoded.height,
        encoder.output,
        elapsed.as_secs_f64()
    );
    Ok(())
}
