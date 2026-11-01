//! Greedy chat completion with images (validation against llama.cpp mtmd).
//! Writes the rendered prompt with `<__media__>` per image for the oracle and
//! prints the prompt layout and the generated token ids.
use ingot::tokenizer::{Message, render_chat};
use ingot::vision::IMAGE_MARKER;
use std::ffi::OsString;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("usage: vision_generate MODEL MMPROJ PROMPT MAX_TOKENS ORACLE_PROMPT IMAGE...");
        std::process::exit(2);
    }
    let (model, mmproj, text, max) = (&args[1], &args[2], &args[3], args[4].parse().unwrap());
    let cli: Vec<OsString> = [model.as_str(), "8192", "--mmproj", mmproj, "--no-draft"]
        .map(OsString::from)
        .to_vec();
    let mut engine = ingot::cli::Options::parse(&cli, &[], 0, "vision_generate")?.load()?;
    let images: Vec<Vec<u8>> = args[6..]
        .iter()
        .map(std::fs::read)
        .collect::<Result<_, _>>()?;
    let message = Message::user(IMAGE_MARKER.repeat(images.len()) + text);
    let rendered = render_chat(false, std::slice::from_ref(&message), &[])?;
    std::fs::write(&args[5], rendered.replace(IMAGE_MARKER, "<__media__>"))?;
    let ids = engine.tokenizer.chat_messages(&[message])?;
    let refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let prompt = engine.insert_images(ids, &refs)?;
    let image_tokens = prompt.iter().filter(|&&t| t >= 0x8000_0000).count();
    println!("prompt tokens={} image tokens={image_tokens}", prompt.len());
    let g = engine.generate(&prompt, max, Default::default(), |_, _| Ok(()))?;
    let ids: Vec<String> = g.tokens.iter().map(u32::to_string).collect();
    println!("generated {}", ids.join(" "));
    println!("text {:?}", g.text);
    let s = &g.statistics;
    println!(
        "prefill {:.2}s decode {:.2}s",
        s.prefill.as_secs_f64(),
        s.total.saturating_sub(s.prefill).as_secs_f64()
    );
    Ok(())
}
