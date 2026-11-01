//! Argument parsing shared by the `generate`, `serve` and `chat` subcommands.
use crate::engine::{DraftConfig, Engine};
use crate::sampling::Sampling;
use crate::{Result, invalid, models};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;

const DRAFT_FLAGS: [&str; 3] = ["--draft-model", "--draft-tokens", "--draft-vocab"];
/// Session capacity in tokens when CONTEXT is omitted.
pub const DEFAULT_CONTEXT: usize = 8192;

pub struct Options {
    pub model: PathBuf,
    /// Required positionals between MODEL and CONTEXT, such as generate's PROMPT.
    pub extra: Vec<String>,
    pub context: usize,
    pub budget: usize,
    pub draft: Option<DraftConfig>,
    pub prefix_cache: bool,
    /// Vision projector: `--mmproj PATH`, else one `mmproj*.gguf` beside MODEL.
    pub mmproj: Option<PathBuf>,
    values: HashMap<String, String>,
}
impl Options {
    /// Parse `MODEL EXTRA... [CONTEXT [MEMORY_GIB]]` plus `--flag value` options.
    /// Only the draft flags, `--no-draft`, `--prefix-cache` and the caller's
    /// `flags` are accepted. MODEL may be a manifest name; its MTP draft is used
    /// when present unless `--no-draft` or `--draft-model` is given.
    pub fn parse(args: &[OsString], flags: &[&str], extra: usize, usage: &str) -> Result<Self> {
        let mut positional = vec![];
        let mut values = HashMap::new();
        let mut prefix_cache = false;
        let mut no_draft = false;
        let mut i = 0;
        while i < args.len() {
            let arg = args[i]
                .to_str()
                .ok_or_else(|| invalid("arguments must be UTF-8"))?;
            i += 1;
            if arg == "--prefix-cache" {
                prefix_cache = true;
            } else if arg == "--no-draft" {
                no_draft = true;
            } else if arg.starts_with("--") {
                if !flags.contains(&arg) && !DRAFT_FLAGS.contains(&arg) && arg != "--mmproj" {
                    return Err(invalid(format!("unknown option {arg}\n{usage}")));
                }
                let value = args
                    .get(i)
                    .and_then(|v| v.to_str())
                    .ok_or_else(|| invalid(format!("{arg} requires a value")))?;
                i += 1;
                if values.insert(arg.to_owned(), value.to_owned()).is_some() {
                    return Err(invalid(format!("{arg} given more than once")));
                }
            } else {
                positional.push(arg.to_owned());
            }
        }
        if !(1 + extra..=3 + extra).contains(&positional.len()) {
            return Err(invalid(usage));
        }
        let mut options = Self {
            model: PathBuf::new(),
            extra: positional[1..=extra].to_vec(),
            context: match positional.get(1 + extra) {
                Some(v) => v
                    .parse()
                    .map_err(|_| invalid("CONTEXT must be an integer"))?,
                None => DEFAULT_CONTEXT,
            },
            budget: 0,
            draft: None,
            prefix_cache,
            mmproj: None,
            values,
        };
        let draft_model = options.values.remove("--draft-model");
        let draft_tokens = options.get::<usize>("--draft-tokens")?;
        let draft_vocab = options.values.remove("--draft-vocab");
        if no_draft && draft_model.is_some() {
            return Err(invalid("--no-draft conflicts with --draft-model"));
        }
        let manifest = models::Manifest::builtin()?;
        let resolved = models::resolve(&manifest, &positional[0], &models::default_dir()?)?;
        options.model = resolved.model;
        let mmproj = options.values.remove("--mmproj").map(PathBuf::from);
        options.mmproj = crate::engine::projector_for(&options.model, mmproj.as_deref())?;
        let draft_model = match draft_model {
            Some(model) => Some(model.into()),
            None if no_draft => None,
            None => resolved.draft.inspect(|d| {
                eprintln!("MTP draft: {} (--no-draft disables it)", d.display());
            }),
        };
        if draft_model.is_none() && (draft_tokens.is_some() || draft_vocab.is_some()) {
            return Err(invalid(
                "--draft-tokens/--draft-vocab require an MTP draft model",
            ));
        }
        options.draft = draft_model.map(|model| DraftConfig {
            model,
            max_tokens: draft_tokens.unwrap_or(3),
            vocabulary: draft_vocab.map(Into::into),
        });
        let gib: usize = match positional.get(2 + extra) {
            Some(v) => v
                .parse()
                .map_err(|_| invalid("MEMORY_GIB must be an integer"))?,
            None => models::auto_budget_gib(
                &options.model,
                options.draft.as_ref().map(|d| d.model.as_path()),
            )?,
        };
        options.budget = gib
            .checked_mul(1024 * 1024 * 1024)
            .ok_or_else(|| invalid("memory budget overflow"))?;
        Ok(options)
    }
    pub fn get<T: FromStr>(&self, flag: &str) -> Result<Option<T>> {
        self.values
            .get(flag)
            .map(|v| {
                v.parse()
                    .map_err(|_| invalid(format!("invalid value for {flag}: {v}")))
            })
            .transpose()
    }
    pub fn string(&self, flag: &str) -> Option<&str> {
        self.values.get(flag).map(String::as_str)
    }
    /// `--temperature/--top-p/--top-k/--seed`; defaults to greedy decoding.
    pub fn sampling(&self) -> Result<(Sampling, Option<u64>)> {
        let d = Sampling::default();
        let sampling = Sampling {
            temperature: self.get("--temperature")?.unwrap_or(d.temperature),
            top_p: self.get("--top-p")?.unwrap_or(d.top_p),
            top_k: self.get("--top-k")?.unwrap_or(d.top_k),
            seed: 0,
        };
        sampling.validate()?;
        Ok((sampling, self.get("--seed")?))
    }
    pub fn load(&self) -> Result<Engine> {
        // The projector's weights and activations come out of the same budget.
        let vision = match &self.mmproj {
            Some(path) => {
                std::fs::metadata(path)
                    .map_err(|e| invalid(format!("vision projector {}: {e}", path.display())))?
                    .len() as usize
                    + crate::engine::ACTIVATION_RESERVE
            }
            None => 0,
        };
        let mut engine = Engine::load(
            &self.model,
            self.budget.saturating_sub(vision),
            self.context,
            self.draft.clone(),
            self.prefix_cache,
        )?;
        if let Some(path) = &self.mmproj {
            engine.attach_vision(path)?;
            eprintln!("vision projector: {}", path.display());
        }
        Ok(engine)
    }
}
