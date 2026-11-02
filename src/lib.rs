mod anthropic;
pub mod chat;
pub mod cli;
pub mod commands;
pub mod engine;
#[cfg(test)]
mod expert_bench;
mod generic;
mod generic_arch;
pub mod gguf;
mod glm;
pub mod image;
mod json_grammar;
mod media;
pub mod metal;
mod model;
pub mod models;
pub mod onboarding;
mod openai;
pub mod pdf;
mod qwen;
pub mod sampling;
pub mod server;
pub mod service;
mod session_pool;
pub mod settings;
mod spm;
mod structured;
pub mod template;
pub mod thinking;
pub mod tokenizer;
pub mod tools;
pub mod vision;
mod weights;

use gguf::{Gguf, Matrix};
use std::io;
pub type Result<T> = io::Result<T>;
pub fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub fn route(logits: &[f32], top: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    if top == 0 || top > logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid router scores or expert count"));
    }
    let mut ids: Vec<_> = (0..logits.len()).collect();
    // Stable sort gives lower expert IDs priority on exact ties.
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    ids.truncate(top);
    // Softmax followed by renormalization of the selected experts is equivalent
    // to softmax over their logits. No bias or route scaling in this checkpoint.
    let max = logits[ids[0]];
    let mut weights: Vec<_> = ids.iter().map(|&i| (logits[i] - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    Ok((ids, weights))
}

pub fn ffn_weights(
    model: &mut Gguf,
    layer: usize,
    width: usize,
    hidden: usize,
    count: usize,
    ids: &[usize],
    shared: bool,
) -> Result<[Matrix; 3]> {
    let suffix = if shared { "shexp" } else { "exps" };
    let load = |model: &mut Gguf, name: &str, cols, rows| -> Result<Matrix> {
        let mut shape = vec![cols, rows];
        if !shared {
            shape.push(count);
        }
        model.matrix(
            &format!("blk.{layer}.ffn_{name}_{suffix}.weight"),
            &shape,
            ids,
        )
    };
    Ok([
        load(model, "gate", width, hidden)?,
        load(model, "up", width, hidden)?,
        load(model, "down", hidden, width)?,
    ])
}
