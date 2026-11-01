//! ingot command line: model management, chat, serving, generation and
//! numerical validation commands.
use ingot::{Result, ffn_weights, gguf, invalid, metal, route};

use std::ffi::OsString;
use std::fs;
use std::io::{BufWriter, IsTerminal, Write};
use std::path::Path;
use std::process::ExitCode;

use gguf::Gguf;
use metal::Gpu;

const HELP: &str = "ingot: run AI models locally on your Apple Silicon Mac.

Start here:
  ingot                                  choose a model, download and chat
  ingot run                              the same guided start
  ingot run qwen3.8-iq2_xs                download if needed, then chat

Downloads range from 2 to 103 GiB. You will see the size and confirm first.
Downloaded models are reused. Memory settings are chosen automatically.

More commands:
  ingot models                           list models, local status and memory budgets
  ingot pull NAME|hf:OWNER/REPO/FILE.gguf download or verify a model without chatting
  ingot chat MODEL [options]             chat with an existing model; advanced settings
  ingot serve MODEL [options]            OpenAI-compatible API at 127.0.0.1:8080
  ingot serve [options]                  the same, loading any local model on demand
  ingot service install|status|logs      keep `serve` running in the background (launchd)
  ingot generate MODEL PROMPT MAX_TOKENS  one completion
  ingot dev                              validation and debugging commands
  ingot --version                        show the installed version

Use `ingot COMMAND --help` for details.
MODEL is a listed model name, a pulled hf:OWNER/REPO/FILE.gguf spec or the path
to the first GGUF shard.
Model files default to ~/.cache/ingot/models (override with INGOT_MODELS).";

const GENERATE_USAGE: &str = "usage: ingot generate MODEL PROMPT MAX_TOKENS [CONTEXT [MEMORY_GIB]] \
[--no-draft | --draft-model MODEL [--draft-tokens N] [--draft-vocab IDS]] [--prefix-cache]";

const MODEL_OPTIONS_HELP: &str = "
MODEL is a name from `ingot models` or the path to the first GGUF shard.
CONTEXT defaults to 8192 tokens; MEMORY_GIB defaults to an automatic budget.
An already downloaded MTP draft is used automatically for greedy decoding.
--no-draft disables it; --draft-model PATH selects a draft explicitly.
--draft-tokens accepts 1, 2 or 3; --draft-vocab accepts a token-ID file.
--prefix-cache reuses the KV state of a shared prompt prefix.";

const DEV_HELP: &str =
    "ingot dev: validation and debugging commands. MODEL is the first GGUF shard.

  ingot dev tokenize MODEL TEXT [--special]
  ingot dev logits MODEL TOKENS_FILE OUTPUT CONTEXT [MEMORY_GIB]
  ingot dev trace MODEL TOKENS_FILE OUTPUT CONTEXT POSITION TRACE_DIR [MEMORY_GIB]
  ingot dev dequant MODEL TENSOR ROW OUTPUT
  ingot dev moe-layer MODEL LAYER INPUT.f32 OUTPUT.f32

Token files contain little-endian u32 IDs; numerical output is F32.
MEMORY_GIB defaults to 32. Output files must not exist.";

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.first().and_then(|a| a.to_str()) {
        None if std::io::stdin().is_terminal() => ingot::onboarding::main(&[]),
        None | Some("--help" | "-h" | "help") => {
            println!("{HELP}");
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("ingot {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(command @ ("run" | "chat" | "serve" | "generate"))
            if args.get(1).is_some_and(|a| a == "--help" || a == "-h") =>
        {
            println!(
                "{}",
                match command {
                    "run" => ingot::onboarding::USAGE,
                    "chat" => ingot::chat::USAGE,
                    "serve" => ingot::server::USAGE,
                    _ => GENERATE_USAGE,
                }
            );
            if command != "run" {
                println!("{MODEL_OPTIONS_HELP}");
            }
            Ok(())
        }
        Some("run") => ingot::onboarding::main(&args[1..]),
        Some("serve") => ingot::server::main(&args[1..]),
        Some("service") => ingot::service::main(&args[1..]),
        Some("chat") => ingot::chat::main(&args[1..]),
        Some("models" | "pull") => ingot::models::cli(&args),
        Some("generate") => generate(&args[1..]),
        Some("dev") => dev(&args[1..]),
        Some(other) => Err(invalid(format!(
            "unknown command {other}; run `ingot --help`"
        ))),
    }
}

fn generate(args: &[OsString]) -> Result<()> {
    let options = ingot::cli::Options::parse(args, &[], 2, GENERATE_USAGE)?;
    let max = options.extra[1]
        .parse()
        .map_err(|_| invalid("MAX_TOKENS must be an integer"))?;
    let mut engine = options.load()?;
    let prompt = engine.tokenizer.chat(&options.extra[0])?;
    let result = engine.generate(&prompt, max, Default::default(), |_, bytes| {
        std::io::stdout().write_all(bytes)?;
        std::io::stdout().flush()
    })?;
    eprintln!(
        "\nstop={:?} tokens={} statistics={:?}\ntoken_ids={:?}",
        result.stop,
        result.tokens.len(),
        result.statistics,
        result.tokens
    );
    Ok(())
}

fn dev(args: &[OsString]) -> Result<()> {
    match args.first().and_then(|a| a.to_str()) {
        Some("tokenize") => tokenize(args),
        Some("dequant") => dequant(args),
        Some("logits" | "trace") => logits(args),
        Some("moe-layer") => moe_layer(args),
        None | Some("--help" | "-h" | "help") => {
            println!("{DEV_HELP}");
            Ok(())
        }
        Some(other) => Err(invalid(format!(
            "unknown dev command {other}; run `ingot dev`"
        ))),
    }
}

fn dequant(args: &[OsString]) -> Result<()> {
    if args.len() != 5 {
        return Err(invalid("usage: ingot dev dequant MODEL TENSOR ROW OUTPUT"));
    }
    let mut model = Gguf::open(Path::new(&args[1]))?;
    let row: usize = args[3]
        .to_str()
        .ok_or_else(|| invalid("row"))?
        .parse()
        .map_err(invalid)?;
    let gpu = Gpu::new()?;
    let matrix = gpu.read_rows(
        &mut model,
        args[2].to_str().ok_or_else(|| invalid("tensor"))?,
        &[row],
    )?;
    let values = gpu.dequant(&matrix)?;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn tokenize(args: &[OsString]) -> Result<()> {
    if args.len() < 3 || args.len() > 4 || args.get(3).is_some_and(|a| a != "--special") {
        return Err(invalid("usage: ingot dev tokenize MODEL TEXT [--special]"));
    }
    let model = Gguf::open(Path::new(&args[1]))?;
    let tokenizer = ingot::tokenizer::Tokenizer::from_gguf(&model)?;
    let text = args[2]
        .to_str()
        .ok_or_else(|| invalid("text is not UTF-8"))?;
    let ids = tokenizer.encode(text, args.get(3).is_some_and(|a| a == "--special"))?;
    println!("{ids:?}");
    Ok(())
}

fn logits(args: &[OsString]) -> Result<()> {
    let trace = args[0] == "trace";
    if trace && args.len() != 7 && args.len() != 8 || !trace && args.len() != 5 && args.len() != 6 {
        return Err(invalid(
            "usage: ingot dev logits MODEL TOKENS_FILE OUTPUT CONTEXT [MEMORY_GIB] | dev trace MODEL TOKENS_FILE OUTPUT CONTEXT POSITION TRACE_DIR [MEMORY_GIB]",
        ));
    }
    let context = args[4]
        .to_str()
        .ok_or_else(|| invalid("context"))?
        .parse()
        .map_err(invalid)?;
    let gib: usize = match args.get(if trace { 7 } else { 5 }) {
        Some(a) => a
            .to_str()
            .ok_or_else(|| invalid("memory budget"))?
            .parse()
            .map_err(invalid)?,
        None => 32,
    };
    let budget = gib
        .checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| invalid("memory budget overflow"))?;
    let mut engine =
        ingot::engine::Engine::load(Path::new(&args[1]), budget, context, None, false)?;
    if trace {
        engine.set_validation_trace(
            args[5]
                .to_str()
                .ok_or_else(|| invalid("trace position"))?
                .parse()
                .map_err(invalid)?,
            Path::new(&args[6]),
        )?;
    }
    let raw = fs::read(&args[2])?;
    if raw.len() % 4 != 0 {
        return Err(invalid("token file must contain little-endian u32 values"));
    }
    let tokens: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let mut output = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?,
    );
    let mut session = engine.session(Default::default())?;
    for token in tokens {
        for v in engine.consume(&mut session, &[token])? {
            output.write_all(&v.to_le_bytes())?;
        }
    }
    output.flush()?;
    Ok(())
}

fn moe_layer(args: &[OsString]) -> Result<()> {
    if args.len() != 5 {
        return Err(invalid(
            "usage: ingot dev moe-layer MODEL.gguf LAYER INPUT.f32 OUTPUT.f32",
        ));
    }
    let layer: usize = args[2]
        .to_str()
        .ok_or_else(|| invalid("invalid layer"))?
        .parse()
        .map_err(invalid)?;
    let mut model = Gguf::open(Path::new(&args[1]))?;
    if model.architecture != "qwen4exp" {
        return Err(invalid("only qwen4exp is supported"));
    }
    if layer >= model.integer("qwen4exp.block_count")? {
        return Err(invalid("layer out of range"));
    }
    let width = model.integer("qwen4exp.embedding_length")?;
    let count = model.integer("qwen4exp.expert_count")?;
    let top = model.integer("qwen4exp.expert_used_count")?;
    let hidden = model.integer("qwen4exp.expert_feed_forward_length")?;
    let shared_hidden = model.integer("qwen4exp.expert_shared_feed_forward_length")?;
    let raw = fs::read(&args[3])?;
    if raw.len()
        != width
            .checked_mul(4)
            .ok_or_else(|| invalid("embedding size overflow"))?
    {
        return Err(invalid(format!(
            "expected {width} little-endian F32 input values"
        )));
    }
    let input: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if input.iter().any(|x| !x.is_finite()) {
        return Err(invalid("input contains non-finite values"));
    }
    let router = model.matrix(
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        &[width, count],
        &[0],
    )?;
    let gpu = Gpu::new()?;
    let (ids, weights) = route(&gpu.project(&router, &input)?, top)?;
    let routed = ffn_weights(&mut model, layer, width, hidden, count, &ids, false)?;
    let shared = ffn_weights(&mut model, layer, width, shared_hidden, 1, &[0], true)?;
    let shared_gate = model.matrix(
        &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        &[width],
        &[0],
    )?;
    let output = gpu.moe(&routed, &shared, &shared_gate, &input, &weights)?;
    // Do not replace an existing artifact, model or input by accident.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    for value in output {
        file.write_all(&value.to_le_bytes())?;
    }
    println!(
        "device={}\nselected_experts={ids:?}\nrouting_weights={weights:?}\noutput_values={width}",
        gpu.name()
    );
    Ok(())
}

fn main() -> ExitCode {
    objc2::rc::autoreleasepool(|_| match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ingot: {error}");
            ExitCode::FAILURE
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn router_handles_large_logits_and_ties() {
        let (ids, w) = route(&[1000., -1000., 1000., 999.], 3).unwrap();
        assert_eq!(ids, [0, 2, 3]);
        assert!((w.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert!((w[0] / w[2] - std::f32::consts::E).abs() < 1e-6);
        assert!(route(&[f32::NAN], 1).is_err());
        assert!(route(&[0.], 0).is_err());
        assert!(route(&[0.], 2).is_err());
    }
}
