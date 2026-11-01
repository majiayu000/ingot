//! Interactive multi-turn chat. The live session is kept between turns, so each
//! turn only computes the new user message; history is the exact token stream
//! the model produced, followed by the chat template's next-turn markers.
use crate::cli::Options;
use crate::engine::{Cancellation, Stop};
use crate::openai::TextStream;
use crate::sampling::random_seed;
use crate::thinking::{self, Thinking};
use crate::tokenizer::{Message, Role};
use crate::{Result, invalid};
use std::ffi::OsString;
use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const USAGE: &str =
    "usage: ingot chat MODEL [CONTEXT [MEMORY_GIB]] [--system TEXT] [--max-tokens N] \
[--temperature T] [--top-p P] [--top-k K] [--seed S] [--thinking off|low|medium|high] \
[--no-draft | --draft-model MODEL [--draft-tokens N] [--draft-vocab IDS]] [--prefix-cache] \
[--mmproj PATH] [--image PATH]

--image attaches a PNG or JPEG to the first message; /image PATH attaches one to the next.";

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
extern "C" fn on_interrupt(_: i32) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}
unsafe extern "C" {
    fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
}
const SIGINT: i32 = 2;
const SIG_ERR: usize = usize::MAX;

/// Ctrl-C cancels the generation in progress instead of terminating the process.
fn install_interrupt(current: Arc<Mutex<Option<Cancellation>>>) -> Result<()> {
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    if unsafe { signal(SIGINT, on_interrupt) } == SIG_ERR {
        return Err(std::io::Error::last_os_error());
    }
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(50));
            if INTERRUPTED.swap(false, Ordering::SeqCst) {
                match current.lock().unwrap().as_ref() {
                    Some(cancel) => cancel.cancel(),
                    None => eprint!("\n(type /exit or press Ctrl-D to quit)\n> "),
                }
            }
        }
    });
    Ok(())
}

pub fn main(args: &[OsString]) -> Result<()> {
    let options = Options::parse(
        args,
        &[
            "--system",
            "--max-tokens",
            "--temperature",
            "--top-p",
            "--top-k",
            "--seed",
            "--thinking",
            "--image",
        ],
        0,
        USAGE,
    )?;
    let (mut sampling, seed) = options.sampling()?;
    let max_tokens: Option<usize> = options.get("--max-tokens")?;
    if max_tokens == Some(0) {
        return Err(invalid("--max-tokens must be at least 1"));
    }
    let system = options.string("--system").map(str::to_owned);
    let level = options
        .string("--thinking")
        .map(str::parse::<Thinking>)
        .transpose()?;
    // Reasoning is shown dimmed on a terminal.
    let (dim, plain) = if std::io::stdout().is_terminal() {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };
    eprintln!("Loading model… This can take a few minutes on the first run.");
    let mut engine = options.load()?;
    let context = engine.context();
    let current = Arc::new(Mutex::new(None::<Cancellation>));
    install_interrupt(current.clone())?;
    eprintln!(
        "Ready! Type a message and press Enter. /help shows commands. Ctrl-C stops a reply.\n\
ingot chat: context {context} tokens, {}{}.",
        if sampling.is_greedy() {
            "greedy".to_owned()
        } else {
            format!(
                "temperature {} top_p {} top_k {}",
                sampling.temperature, sampling.top_p, sampling.top_k
            )
        },
        if engine.has_draft() && sampling.is_greedy() {
            ", MTP on"
        } else {
            ""
        }
    );
    let fresh = || -> Vec<Message> {
        system
            .iter()
            .map(|s| Message::new(Role::System, s.clone()))
            .collect()
    };
    let mut messages = fresh();
    // Full token transcript (prompt + reply) of the previous turn.
    let mut transcript: Option<Vec<u32>> = None;
    let mut kept = None;
    // Images for the next message, and every image of the conversation.
    let mut pending: Vec<Vec<u8>> = Vec::new();
    let mut images: Vec<Vec<u8>> = Vec::new();
    if let Some(path) = options.string("--image") {
        if !engine.has_vision() {
            return Err(invalid(
                "--image needs a vision projector: pass --mmproj PATH or put mmproj*.gguf beside the model",
            ));
        }
        pending.push(std::fs::read(path)?);
    }
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let input = line.trim();
        match input {
            "" => continue,
            "/help" => {
                eprintln!(
                    "/reset starts a new conversation; /image PATH attaches a picture to the next \
                     message; /exit quits. Ctrl-C stops a reply; Ctrl-D quits."
                );
                continue;
            }
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                messages = fresh();
                transcript = None;
                kept = None;
                pending.clear();
                images.clear();
                eprintln!("(conversation cleared)");
                continue;
            }
            _ if input.starts_with("/image ") => {
                let path = input["/image ".len()..].trim();
                match std::fs::read(path) {
                    Ok(_) if !engine.has_vision() => {
                        eprintln!("this model has no vision projector (start with --mmproj PATH)")
                    }
                    Ok(bytes) => {
                        pending.push(bytes);
                        eprintln!("(attached {path} to the next message)");
                    }
                    Err(e) => eprintln!("cannot read {path}: {e}"),
                }
                continue;
            }
            _ if input.starts_with('/') => {
                eprintln!("unknown command {input}; type /help for commands");
                continue;
            }
            _ => {}
        }
        let input = crate::vision::IMAGE_MARKER.repeat(pending.len()) + input;
        messages.push(Message::user(&input));
        let next = engine.tokenizer.chat_continuation_with(&input, level);
        let prompt = match (&transcript, next) {
            (Some(previous), Some(next)) => engine
                .tokenizer
                .encode_continuation(&next)
                .map(|suffix| [previous.as_slice(), &suffix].concat())
                .and_then(|p| engine.insert_images(p, &refs(&pending))),
            _ => engine
                .tokenizer
                .chat_thinking(&messages, &[], level)
                .and_then(|p| {
                    let all: Vec<&[u8]> = refs(&images).into_iter().chain(refs(&pending)).collect();
                    engine.insert_images(p, &all)
                }),
        };
        let prompt = match prompt {
            Ok(p) if p.len() < context => p,
            Ok(p) => {
                eprintln!(
                    "conversation needs {} tokens but the context is {context}; use /reset",
                    p.len() + 1
                );
                messages.pop();
                continue;
            }
            Err(e) => {
                eprintln!("error: {e}");
                if !pending.is_empty() {
                    eprintln!("(attached images dropped)");
                    pending.clear();
                }
                messages.pop();
                continue;
            }
        };
        let limit = context - prompt.len();
        let tail = engine
            .tokenizer
            .decode(&prompt[prompt.len().saturating_sub(4)..]);
        let mut text =
            TextStream::new(tail.is_ok_and(|t| thinking::opens_think(&t)), vec![], false);
        if level.is_some_and(|t| t != Thinking::Off) {
            text = text.detect_think();
        }
        let (mut reasoning, mut answer, mut shown) = (String::new(), String::new(), false);
        let mut print = |deltas: Vec<crate::openai::Delta>| -> std::io::Result<()> {
            let mut out = std::io::stdout();
            for d in deltas {
                if d.reasoning != shown {
                    out.write_all(if d.reasoning { dim } else { plain }.as_bytes())?;
                    if !d.reasoning && !reasoning.is_empty() {
                        out.write_all(b"\n\n")?;
                    }
                    shown = d.reasoning;
                }
                out.write_all(d.text.as_bytes())?;
                if d.reasoning {
                    reasoning += &d.text;
                } else {
                    answer += &d.text;
                }
            }
            out.flush()
        };
        sampling.seed = seed.unwrap_or_else(random_seed);
        let cancel = Cancellation::default();
        *current.lock().unwrap() = Some(cancel.clone());
        let result = engine.generate_with(
            &prompt,
            max_tokens.map_or(limit, |m| m.min(limit)),
            cancel,
            &sampling,
            Some(&mut kept),
            |_, bytes| print(text.push(bytes)),
        );
        let flushed = print(text.finish());
        drop(print);
        if shown {
            print!("{plain}");
        }
        *current.lock().unwrap() = None;
        flushed?;
        match result {
            Ok(g) => {
                let s = &g.statistics;
                let decode = s.total.saturating_sub(s.prefill).as_secs_f64();
                println!();
                eprintln!(
                    "[{}{} tokens, prompt {} (+{} new), {:.1} tok/s]",
                    match g.stop {
                        Stop::Cancelled => "interrupted, ",
                        Stop::Length => "length limit, ",
                        Stop::Eos => "",
                    },
                    g.tokens.len(),
                    prompt.len(),
                    prompt.len() - s.prefix_reused_tokens,
                    if decode > 0. {
                        g.tokens.len().saturating_sub(1) as f64 / decode
                    } else {
                        0.
                    }
                );
                let mut reply = Message::new(Role::Assistant, answer);
                reply.reasoning = (!reasoning.is_empty()).then_some(reasoning);
                messages.push(reply);
                transcript = Some([prompt, g.tokens].concat());
                images.append(&mut pending);
            }
            Err(e) => {
                eprintln!("\nerror: {e}");
                messages.pop();
            }
        }
    }
}

fn refs(images: &[Vec<u8>]) -> Vec<&[u8]> {
    images.iter().map(Vec::as_slice).collect()
}
