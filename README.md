# Ingot

> 设计与验收文档以中文为主；本 README 以英文为主，便于外部读者。

Run Qwen3.8-Flash-Next or GLM-5.3-Flash locally on an Apple Silicon Mac, plus
standard Llama, Qwen2, Qwen3 and Qwen3-MoE GGUF models. Choose a model, confirm
its download, and chat in your terminal.

[简体中文](README.zh-CN.md)

## Start here

ingot runs AI chat models privately on your own Mac: nothing you type leaves
your computer. You need an Apple Silicon Mac (M1 or later) with macOS 15+.

1. **Install:** clone this repository and run `scripts/install.sh`.
2. **Chat:** run `ingot`, pick a model, confirm the download, and type.
3. **Pick a model by memory:** 16 GB Mac → `qwen3-8b-q4_k_m` (4.7 GiB),
   32 GB → `qwen3-30b-a3b-q4_k_m` (17.3 GiB), 64 GB+ → `qwen3.8-iq2_xs`
   (63.4 GiB). `ingot models` shows what fits your Mac.
4. **Use it in your browser or apps:** run `ingot serve` and open
   <http://127.0.0.1:8080/>, or point Open WebUI, Cursor or Claude Code at it.

Guides:

- [Quick start](docs/QUICKSTART.md): install, first chat, which model, browser,
  Open WebUI, Cursor, Claude Code, phone and other computers.
- [Something went wrong?](docs/TROUBLESHOOTING.md): error messages and fixes.
- [Let your AI set it up](docs/AI_SETUP.md): instructions to hand to an AI
  coding agent.

The rest of this README is the detailed technical reference.

## Install from source

**Before you begin:** this is an early project for large models, tested on an
M2 Max with 96 GiB memory. The smallest download is **63.4 GiB (~68 GB)**;
the largest is 102.5 GiB (~110 GB). Keep enough free SSD space for your chosen
model. Memory is selected automatically, but this does not guarantee speed or
compatibility on every Mac. See [Requirements](#requirements).

### Install

There is no published binary release yet. Install from source with Rust 1.88+
and Xcode Command Line Tools. The private repository requires GitHub access:

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/install.sh
```

The script checks for an Apple Silicon Mac, macOS 15+, Xcode Command Line
Tools and Rust, builds the release binary and copies it to `~/.local/bin`. It
never uses sudo. When that directory is not on your PATH it prints the line to
add; `--yes` appends it to `~/.zshrc` instead. Other options: `--prefix DIR`
(install to `DIR/bin`), `--no-build` (reuse `target/release/ingot`),
`--binary PATH` (install a given binary) and `--dry-run` (check and print the
steps only). `cargo install --path . --locked` also works and installs to
`~/.cargo/bin`.

To keep an API server running in the background for Open WebUI, Claude Code
or other clients, see [Background service](#background-service).

Then start with **one command**:

```sh
ingot run
```

1. Choose a model by number or name, or press Enter for the default. Already
   downloaded models are preferred; otherwise the first option is the validated
   Qwen IQ2_XS baseline. The menu shows download sizes and model descriptions.
2. Read the size and destination, then type `y` to download. Pressing Enter or
   choosing `q` in the model menu cancels. Downloads resume when retried and are
   SHA-256 verified. Optional draft weights are not downloaded by this command.
3. Wait for **Ready!**, type a message and press Enter. `/help` shows commands,
   `/reset` starts a new conversation, and `/exit` quits. Ctrl-C stops a reply.

Next time, use the same command. Local models are reused. To skip the menu:

```sh
ingot run qwen3.8-iq2_xs
```

`ingot` alone also opens the guide in a terminal. In a pipe it prints help.
`ingot run MODEL --yes` explicitly accepts a download without a confirmation
prompt; without `--yes`, piped input is never treated as download permission.
Use `ingot chat MODEL` for advanced settings after downloading.

### If you get stuck

| What you see | What to do |
| --- | --- |
| `cargo: command not found` | Install Rust with [rustup](https://rustup.rs/), reopen the terminal, and retry installation. |
| Compiler/linker missing | Run `xcode-select --install`, complete the macOS installer, then retry. |
| `ingot: command not found` after installation | Use `~/.cargo/bin/ingot run` (or `$CARGO_HOME/bin/ingot` if you use a custom Cargo directory); add that bin directory to your shell's PATH. |
| Download interrupted | Run the same `ingot run MODEL` command again; completed shards and partial downloads are reused. |
| Not enough disk space | Set `INGOT_MODELS` to a directory on a larger SSD before starting, e.g. `INGOT_MODELS=/Volumes/SSD/models ingot run`. Use that same setting on later runs. |
| Model loading takes time | Wait for `Ready!`; loading large weights and compiling Metal shaders can take minutes. |
| Conversation is full | Type `/reset` to start a new conversation. |
| Need available options | Run `ingot --help` or `ingot chat --help`. |

## Engine details

A native Rust + Metal inference engine for two large Mixture-of-Experts models on
Apple Silicon:

- **Qwen3.8-Flash-Next** (GGUF architecture `qwen4exp`): 48 layers, 4-way
  hyper-connections, 36 Gated DeltaNet + 12 QSA sparse-attention layers, PLE,
  512 routed experts (top-10) plus a shared expert.
- **GLM-5.3-Flash** (GGUF architecture `glm5-next`): 45 layers, 34 KDA + 11
  compressed NoPE MLA / pooled DSA layers, 4-way mHC, 3 dense FFN layers, then
  288 routed experts (top-8) plus a shared expert.

The binary reads GGUF files directly and submits its own Metal kernels. It does
not link or spawn llama.cpp/ggml or MLX; those are used only as independent
numerical oracles during validation (see [PROVENANCE.md](PROVENANCE.md)).

Main features:

- Low-bit GGUF weights executed in their stored format (no conversion or copy).
- **MTP speculative decoding** with each model's trained NextN draft head
  (1–3 draft tokens per round, optional draft-vocabulary shortlist). Output is
  identical to plain greedy decoding; draft tokens are never emitted before the
  target model verifies them.
- **Resident or bounded-memory experts.** If the memory budget fits every
  expert, all experts stay resident and routing stays on the GPU. Otherwise
  experts are loaded into fixed per-layer slots with LRU replacement, and the
  next expert read overlaps the current expert's compute.
- **Grouped prefill**: tokens routed to the same expert are computed together.
  Qwen uses 4096/1024/128-token prefill blocks depending on the budget; GLM
  uses 64-token blocks.
- **Prefix cache** (`--prefix-cache`): keeps the state of one prompt prefix and
  reuses it in later requests that share it.
- Optional per-kernel GPU timing (`INGOT_KERNEL_PROFILE`).

## Status and limitations

- Greedy decoding (default) and temperature/top-k/top-p sampling. MTP is used
  only for greedy decoding; sampled requests use autoregressive decoding.
- One loaded model, one active session per engine, single Metal queue.
- Multi-turn text chat with system/user/assistant messages. Thinking levels
  off/low/medium/high per request; by default Qwen runs with thinking
  disabled and GLM at its default "Max" reasoning effort. Tool calling for
  Qwen models (not GLM); image input for Qwen3.8 with its vision projector.
- An OpenAI-compatible HTTP/1.1 subset, with one request per connection and
  sequential inference. Optional API-key authentication (`--api-key`); no
  TLS, cross-request batching or generic graph/plugin framework. The default
  bind address is loopback, and another address needs a key or `--insecure`.
- The two optimized architectures above plus the generic `llama`, `qwen2`,
  `qwen3`, `qwen3moe`, `gemma3`, `gemma4` and `phi3` path (comparable to llama.cpp, see
  [Generic architectures](#generic-architectures)). Other GGUF architectures
  are rejected with an error.
- Tested on exactly one machine: **Apple M2 Max (38-core GPU), 96 GiB,
  macOS 26.5.2**. Other chips, memory sizes and macOS versions are untested.
- Long-context language quality has not been independently evaluated beyond
  the recorded checks in [NATIVE_VALIDATION.md](NATIVE_VALIDATION.md).
- Model weights are not included and follow their own licenses.

## Supported weights

Tensor types accepted by the loader (`src/gguf.rs`) and executed by the Metal
kernels:

| GGML type | ID | | GGML type | ID |
| --- | ---: | --- | --- | ---: |
| F32 | 0 | | IQ2_XXS | 16 |
| F16 | 1 | | IQ2_XS | 17 |
| BF16 | 30 | | IQ3_XXS | 18 |
| Q8_0 | 8 | | IQ4_NL | 20 |
| Q2_K | 10 | | IQ3_S | 21 |
| Q4_K | 12 | | IQ2_S | 22 |
| Q5_K | 13 | | IQ4_XS | 23 |
| Q6_K | 14 | | IQ1_M | 29 |
| Q2_0 | 42 | | | |

`Q2_0` (type 42) is the convention of the pinned reference fork
`memoriaru/llama.cpp`, not an upstream ggml type; other GGUF files that use
ID 42 may not be compatible. Q4_0 (type 2) runs through the generic
projection kernels (used by the Llama 3.1 Q4_0 file below).

### Generic architectures

GGUF files with `general.architecture` `llama` (also Mistral/Mixtral GGUFs),
`qwen2`, `qwen3`, `qwen3moe`, `gemma3`, `gemma4` or `phi3` run on a generic
decoder path (`src/generic.rs`, per-architecture configuration in
`src/generic_arch.rs`): GQA attention with RoPE (adjacent-pair for Llama with
optional `rope_freqs` factors, split-half otherwise, partial rotary dimensions
and linear RoPE scaling as declared), optional QKV bias and QK-norm, then a
SwiGLU FFN, a GeGLU FFN (Gemma) or a top-k softmax MoE, with an F32 KV cache.
Gemma adds embedding scaling, post-attention/post-FFN norms, per-layer sliding
windows with their own RoPE base, and final logit soft-capping when declared;
Gemma 4 also has per-layer head sizes and KV heads, value RMS-norm, layers that
reuse keys as values, and a per-layer output scale. Phi-3 uses fused QKV and
gate/up projections; like llama.cpp, its declared sliding window is ignored. A
`llama` file that declares `attention.sliding_window` gets that window. All
weights stay resident; there is no MTP draft and no SSD expert streaming on
this path. Tokenizers: GPT-2 BPE with the `qwen2`, `llama-bpe` or `dbrx` (Phi-4)
pre-tokenizer, SentencePiece (`tokenizer.ggml.model` `llama`, scores with
`<0xXX>` byte fallback, used by Gemma 3, Phi-3, Mistral and Llama 2) and the
Gemma 4 BPE (`gemma4`), all matching `llama-tokenize`. Chat is rendered from
the file's own `tokenizer.chat_template` (minijinja, the way Hugging Face
transformers renders it, thinking disabled unless a request asks for a level,
which sets `enable_thinking` and `reasoning_effort`). Tool calls use the template's
tool prompt (a template without one rejects `tools`); Hermes-style
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` output (Qwen2.5/Qwen3)
and Llama 3.x JSON calls (a reply that is `{"name": ..., "parameters": ...}`,
or one after `<|python_tag|>`) are parsed into `tool_calls`. Rejected with an
error naming the feature: other architectures, RoPE scaling other than Llama 3
or linear, LongRoPE (`rope_factors_long`, Phi-3.5/Phi-4-mini), Gemma 4 shared
KV layers, per-layer embeddings (E2B/E4B) and MoE, and any tensor the
architecture does not consume.

Checked against upstream llama.cpp (`tools/generic-oracle.cpp`, F32 KV cache)
on the same files, on this M2 Max:

| Model (manifest name) | Logits vs llama.cpp | Greedy vs llama.cpp (short / long prompt) | Decode, ours / llama.cpp | Prefill 1730 tokens, ours / llama.cpp |
| --- | --- | --- | ---: | ---: |
| `qwen3-8b-q4_k_m` | max abs diff 5.1e-5, top-1 32/32 | 48/48, 32/32 (2892 tokens) | 53.4 / 57.4 tok/s | 460 / 547 tok/s |
| `llama3.1-8b-instruct-q4_0` | max abs diff 5.5e-5, top-1 30/30 | 48/48, 32/32 (2810 tokens) | 57.2 / 59.2 tok/s | 474 / 562 tok/s |
| `qwen3-30b-a3b-q4_k_m` | max abs diff 3.4e-5, top-1 32/32 | 43/43, 32/32 (2892 tokens) | 84.5 / 85.3 tok/s | 692 / 748 tok/s |

Later additions, same method with a different long prompt (its token count is
given; it exceeds the 1024-token sliding window of the Gemma models):

| Model (manifest name) | Logits vs llama.cpp | Greedy vs llama.cpp (short / long prompt) | Decode, ours / llama.cpp | Prefill of the long prompt, ours / llama.cpp |
| --- | --- | --- | ---: | ---: |
| `gemma3-4b-it-qat-q4_0` | max abs diff 7.5e-5, top-1 28/28 | 48/48, 32/32 (2854 tokens) | 89.2 / 85.6 tok/s | 785 / 967 tok/s |
| `gemma4-12b-it-qat-q4_0` | max abs diff 3.1e-4, top-1 32/32 | 48/48, 32/32 (2858 tokens) | 36.8 / 34.5 tok/s | 141 / 301 tok/s |
| `phi3-mini-4k-instruct-q4` | max abs diff 6.2e-5, top-1 26/26 | 48/48, 32/32 (1698 tokens) | 87.6 / 92.1 tok/s | 682 / 766 tok/s |
| `phi4-q4_0` | max abs diff 5.3e-5, top-1 27/27 | 48/48, 32/32 (2807 tokens) | 33.9 / 36.2 tok/s | 228 / 277 tok/s |

Gemma 4's global layers have head size 512, which has no flash prefill kernel
yet, so its prefill uses dense attention in 128-token blocks. Phi-3-mini-4k produces garbage past
2047 tokens in both engines, because neither applies the sliding window it was
trained with.

Greedy agreement feeds the prompt rendered by jinja2 plus our generated tokens
to llama.cpp and checks its argmax at every position. llama.cpp numbers are
`llama-bench` (tg64, and pp at the prompt length shown) on the same machine,
interleaved with our runs in one window per model with other GPU work running,
so compare within a row; ours are medians of repeated warm runs (decode: up to
128 generated tokens; prefill: 512-token blocks). Decode uses generic-path
matrix-vector kernels after llama.cpp's `kernel_mul_mv_*` (Q4_K, Q6_K, Q4_0,
plus F16/F32), fused gate/up with SwiGLU or GeGLU, fused QK-norm/RoPE/KV-cache
writes, one-kernel decode attention with sliding windows, a fused residual add
and RMS norm, and one serial encoder per layer. Other weight types keep the
shared kernels. Prefill uses a 64x64 simdgroup-matrix GEMM with vectorized
weight decode, flash-style tiled attention (`kernels/generic.metal`, the
`attention_prefill_tiles` algorithm for head dimensions 64/96/128/256) and, for
MoE, GPU expert grouping with one tiled GEMM per expert. Kept server sessions of
these attention-only models can be cut back to a shared prefix (see
[HTTP server](#http-server-and-interactive-chat)).

Weights used for all recorded results (SHA-256 in
[macos/model-sha256.txt](macos/model-sha256.txt) and
[NATIVE_VALIDATION.md](NATIVE_VALIDATION.md)):

| Model | File | Size | MTP draft head |
| --- | --- | ---: | --- |
| Qwen3.8-Flash-Next | `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`, 2 shards | ~68 GB | `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, ~4.1 GB, [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/tree/052beeaca7bec4a303e59cc7bc630c4f3a1b845d) |
| GLM-5.3-Flash | [ggml-org Q2_K_S](https://huggingface.co/ggml-org/GLM-5.3-Flash-GGUF) rev `c259a9f`, 2 shards | ~110 GB | `mtp-GLM-5.3-Flash-Q8_0.gguf`, ~9.3 GB |

An ISTA-DASLab IQ3_XXS GGUF of Qwen3.8-Flash-Next (~70.6 GiB, uses Q4_K,
Q5_K, IQ2_XS and IQ3_XXS) also runs; its quality results are still pending.

<!-- TODO: add the public source URL of the Qwen IQ2_XS GGUF and the GLM Q8_0 draft head. -->

Both shards must be in the same directory; pass the `-00001-of-00002` shard.

## Requirements

- Apple Silicon Mac with Metal (only macOS 26.5.2 tested).
- Rust 1.88+ (edition 2024) and Xcode Command Line Tools. CI checks the minimum
  supported Rust version as well as stable. Metal shaders are
  compiled by the system when the engine loads.
- Disk space for the model files above, preferably on the internal SSD.
- Memory: the engine budget (`MEMORY_GIB`, automatically recommended when
  omitted; see [Model management](#model-management)) covers dense weights,
  expert slots, session state, workspace and a host reserve; the OS file cache
  comes on top. Recorded working budgets:
  - Qwen: 8 GiB (bounded experts) up to 64 GiB (all experts resident).
  - GLM: 11/16/24/32 GiB with bounded experts. All GLM experts (~100 GB) do
    not fit in 96 GiB; a 64 GiB GLM probe caused heavy memory compression and
    was stopped. Leave headroom for the OS.

## Packaging

The source installation above is the current user path. Maintainers can build
an archive on Apple Silicon:

```sh
./scripts/release.sh   # writes dist/*.tar.gz and dist/*.tar.gz.sha256
```

Archives contain a self-contained binary with embedded Metal shader sources.
After a release is published, unpack its archive and run `./ingot run`.
`packaging/homebrew/ingot.rb` is an unfinished template, not an installable tap.
The root `setup-macos.sh` / `run-macos.sh` scripts belong to the llama.cpp
comparison engine; they do not install or launch the native ingot app.

## Build

```sh
cargo build --release --locked
```

Use a separate `--target-dir` (or `CARGO_TARGET_DIR`) for each checkout of
this crate so that different versions do not share build products.

## Run

```sh
ingot run qwen3.8-iq2_xs       # download if needed, then chat
ingot pull qwen3.8-mtp-q8_0    # optional MTP draft, ~4 GiB
ingot chat qwen3.8-iq2_xs      # advanced chat with an existing model
ingot serve qwen3.8-iq2_xs     # OpenAI-compatible API on 127.0.0.1:8080
```

`MODEL` is a name from `ingot models`, found below the model directory (see
[Model management](#model-management)), or a path to the first GGUF shard.
When a main model's MTP draft from the manifest is present it is used
automatically; `--no-draft` disables it and `--draft-model PATH` picks another
draft file. `CONTEXT` (session capacity in tokens) defaults to 8192 and
`MEMORY_GIB` to the Metal working set minus 4 GiB, within which the engine
allocates only what the weights, context state and workspace need; both may follow
`MODEL` as positional arguments.

```text
ingot generate MODEL PROMPT MAX_TOKENS [CONTEXT [MEMORY_GIB]]
        [--no-draft | --draft-model DRAFT.gguf [--draft-tokens 1|2|3] [--draft-vocab IDS]]
        [--prefix-cache]
```

`generate` runs one greedy completion of a single user message. Generated
text goes to stdout; stop reason, token IDs and statistics go to stderr.

```sh
# Qwen, bounded experts in 8 GiB, without MTP
ingot generate qwen3.8-iq2_xs 'Introduce yourself in one sentence.' 64 4096 8 --no-draft

# Qwen, all experts resident, MTP draft used automatically when pulled
ingot generate qwen3.8-iq2_xs 'Write a quicksort in Rust.' 256

GLM=/path/to/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf
ingot generate "$GLM" 'What is 1+1? Answer only.' 224 1024 32
```

`--draft-model` must be the NextN draft GGUF of the same target model; its
tokenizer and model metadata are checked. `--draft-tokens` defaults to 3.
`--draft-vocab` takes a file of little-endian 32-bit token IDs that restricts
the draft head's output vocabulary; `examples/draft_vocab.rs` builds one from a
corpus you supply (never use evaluation data).

Validation commands live under `ingot dev` (`tokenize`, `logits`, `trace`,
`dequant`, `moe-layer`); they take a first-shard path, write little-endian F32
output, refuse to overwrite existing files and default to a 32 GiB budget.

### HTTP server and interactive chat

```sh
# OpenAI-compatible server, listens on 127.0.0.1:8080 by default
ingot serve qwen3.8-iq3_xxs
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":64}'
curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Tell a joke"}],"stream":true,"temperature":0.7,"seed":1}'

# Interactive multi-turn chat: /reset clears history, /exit or Ctrl-D quits,
# Ctrl-C interrupts only the current reply
ingot chat qwen3.8-iq3_xxs --system 'Answer briefly.' --max-tokens 512
```

Both commands take `MODEL [CONTEXT [MEMORY_GIB]]` and the same draft options
and `--prefix-cache` as `generate`. `serve` also takes `--host`, `--port`, `--sessions`, `--keep-alive`, `--thinking`, `--api-key` and `--insecure` (and runs without MODEL, see [Serving every local model](#serving-every-local-model)); `chat` also
takes `--system`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`,
`--seed` and `--thinking off|low|medium|high` (reasoning is printed dimmed on
a terminal and kept as the turn's reasoning in history).

- Web chat: open `http://127.0.0.1:8080/` in a browser (the URL is printed at
  startup). The page is one file built into the binary and makes no external
  requests. It streams from `/v1/chat/completions`; has a model picker filled
  from `/v1/models`, a per-conversation system prompt, temperature, max tokens
  a JSON output switch (`response_format` `json_object`) and a thinking menu
  (default/off/low/medium/high, sent as `reasoning_effort`; reasoning shows in
  a folded "Thinking" section and is sent back in history); keeps a
  conversation list in the browser's localStorage (the page still works,
  unsaved, where storage is blocked); supports stop, regenerate, editing the
  last message and copy buttons for messages and code blocks; renders Markdown
  (code blocks, tables, lists, links limited to http/https) with all model text
  HTML-escaped first; shows tokens/s; and follows the system light/dark theme
  or a chosen one. On narrow screens the conversation list folds into a menu.
  The Monitor tab (see [Monitoring](#monitoring)) shows the loaded model,
  Metal memory, the engine state and queue, kept sessions and the speeds of
  recent requests, refreshed every 3 seconds. When the server requires an API
  key the page asks for it once and keeps it in localStorage.
- Endpoints: `GET /` (web chat, and `GET /monitor.js` for its Monitor tab),
  `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, the
  Anthropic Messages API `POST /v1/messages`
  (see [Coding agents](#coding-agents-claude-code-cursor)), `GET /metrics` and
  `GET /v1/status`. Supported: `stream` (SSE, with `usage` at the end;
  `stream_options.include_usage` adds a standard usage chunk), `max_tokens`
  (chat also accepts `max_completion_tokens`; chat defaults to the remaining
  context, completions to 16), `stop` (up to 4 strings), `temperature`,
  `top_p`, `top_k` and `seed`. `model` may be omitted; if given it must equal
  the name in `/v1/models` (the file name without the shard suffix).
- Structured outputs: `response_format` `{"type":"json_object"}` (any JSON
  object) or `{"type":"json_schema","json_schema":{"schema":{...}}}`; on
  `/v1/messages`, `output_config.format` (or the beta `output_format`)
  `{"type":"json_schema","schema":{...}}`. Decoding is constrained: at every
  step tokens whose bytes cannot continue a valid answer are masked, so a
  finished answer (`finish_reason: "stop"`) always parses and matches the
  schema; a `length` finish can be cut off mid-JSON. Supported schema subset:
  `type` (string, number, integer, boolean, null, object, array, or a list of
  these), `properties`, `required`, `additionalProperties` (only declared
  properties are generated; a schema-valued `additionalProperties` without
  `properties` describes a map), `items`, `minItems`/`maxItems`, `enum`,
  `const`, `anyOf` whose branches start with different JSON tokens (e.g.
  `{"anyOf":[{...object...},{"type":"null"}]}`), local `$ref` into
  `$defs`/`definitions` (recursion allowed), and annotations such as `title`
  and `description`. Any other keyword (`pattern`, `format`, `minLength`,
  `minimum`, `oneOf`, ...) returns 400 naming its path. Properties may come in
  any order; whitespace is limited to compact or indented layouts. It cannot be
  combined with `tools`. A constrained request reads the logits on the CPU and
  runs without MTP (the log says so); the constraint itself costs a few
  microseconds per token. GLM reasons first and is constrained after
  `</think>`. `uv run --with jsonschema tools/check_structured.py URL` checks a
  running server with 20 OpenAI and 2 Anthropic structured requests.
- Tool calling (Qwen): `tools` (function tools), `tool_choice` `auto` or
  `none`, `parallel_tool_calls`, assistant `tool_calls` and `tool` messages in
  history. Tools are rendered with the GGUF chat template's tool prompt and the
  model's `<tool_call>` output is returned as `tool_calls` with
  `finish_reason: "tool_calls"` (streamed as one delta at the end). A
  malformed or truncated call is returned as text. GLM rejects tools.
- Unsupported parameters return HTTP 400 naming the parameter, e.g. forced
  `tool_choice`, `n>1`, non-zero `presence_penalty`/`frequency_penalty`,
  `logprobs`, `echo`, and images for a model without a vision projector
  (see [Images](#images)). Prompt plus output beyond the context
  also returns 400.
- Sampling defaults to greedy (temperature 0), which can use MTP. With
  temperature > 0, MTP's greedy verification cannot preserve the sampling
  distribution, so that request runs autoregressively and the log says so.
  The same seed and parameters reproduce the same output.
- Chat templates reproduce the verified GGUF templates of the supported
  checkpoints for multi-turn system/user/assistant messages at every thinking
  level (see [Thinking levels](#thinking-levels)). Reasoning is returned in
  `reasoning_content`, and `reasoning_content` in history is replayed
  according to the template.
- Requests run one at a time on a single worker thread. Finished sessions
  are kept: a request runs in the kept session whose tokens are the longest
  prefix of its prompt (for example a conversation with a message appended),
  so only the new tokens are computed, even when unrelated requests came in
  between. For the generic attention-only architectures a kept session that
  shares at least half its tokens with the prompt is cut back to the common
  prefix and extended, which covers chat templates that re-render earlier
  turns differently (Qwen3 drops the empty think block from history); the
  recurrent Qwen3.8 and GLM states need an exact prefix. Otherwise it starts a fresh session and the least recently used
  one is dropped when the pool is full. Each kept session holds a full
  CONTEXT state; `--sessions N` (default up to 4) is checked against the
  memory budget left after the model and fails at startup if it does not fit,
  while the default shrinks to what fits. The log shows `reused=` tokens and
  which session (`session=#3(reused …)`) ran each request. A client disconnect (failed stream write or closed connection) cancels
  generation.
- The server admits at most 16 connections and 8 queued requests; excess
  requests receive HTTP 503 and may be retried later. Output events are bounded
  to 8 per request, and socket reads/writes time out after 30 seconds without
  progress. An error after SSE headers is sent as an SSE `error` object and
  closes the stream without a successful completion marker.
- `chat` keeps the tokens the model actually generated as history and
  computes only the new message each turn.

### Thinking levels

Each request may ask for a thinking level; the server maps it onto the
model's own template mechanism:

| Level | Qwen3.8 | GLM-5.3 | GGUF templates (Qwen3, ...) |
| --- | --- | --- | --- |
| off | `enable_thinking=false` (empty think block) | reply starts with an empty think block | `enable_thinking=false` |
| low | reasoning effort low instruction, open think block | `Reasoning Effort: Low` | `enable_thinking=true`, `reasoning_effort=low` |
| medium | no effort instruction, open think block | `Reasoning Effort: High` | `enable_thinking=true`, `reasoning_effort=medium` |
| high | reasoning effort xhigh instruction, open think block | `Reasoning Effort: Max` | `enable_thinking=true`, `reasoning_effort=high` |
| not given | off | Max (as before) | off |

The Qwen3.8 and GLM renderings are checked against a jinja2 rendering of the
GGUF templates (`tests/fixtures/thinking_templates.json`). Requests choose a
level with:

- OpenAI chat: `reasoning_effort` (`none`, `minimal` = low, `low`, `medium`,
  `high`, `xhigh` = high) or `chat_template_kwargs: {"enable_thinking": bool}`
  (true without an effort is high; a contradicting pair is a 400).
- Anthropic: `thinking: {"type": "disabled"}` is off; `{"type": "enabled",
  "budget_tokens": N}` is low below 4096, medium below 16384, else high (the
  budget is not enforced as a token limit; `max_tokens` is); `{"type":
  "adaptive"}` is medium.
- `serve --thinking LEVEL` sets the level for requests that give none, and
  `chat --thinking LEVEL` the level of every turn.

Reasoning is streamed separately: `reasoning_content` deltas on OpenAI chat,
`thinking` content blocks (`thinking_delta`) on Anthropic, and never in
`content`. Templates that let the model open `<think>` itself (Qwen3 with
`enable_thinking`) are handled too. Tool calls after the think block are
parsed as usual, structured outputs constrain only the answer after
`</think>`, and greedy requests keep MTP. Thinking takes tokens from
`max_tokens`; a reply cut off while reasoning finishes with `length` and
little or no `content`.

### Remote access and API keys

```sh
ingot serve qwen3.8-iq3_xxs --host 0.0.0.0 --api-key "$(openssl rand -hex 24)"
INGOT_API_KEY=... ingot serve --host 0.0.0.0     # keeps the key out of `ps`
curl -s http://HOST:8080/v1/models -H "Authorization: Bearer $KEY"
```

With `--api-key KEY` (or `INGOT_API_KEY`) every API route, `/metrics` and
`/v1/status` require `Authorization: Bearer KEY` or `x-api-key: KEY` (the
header Anthropic clients send, so `ANTHROPIC_API_KEY=KEY` works). Keys are
compared in constant time. A missing or wrong key gets HTTP 401 in the
OpenAI error shape (`code: "invalid_api_key"`) or, on `/v1/messages`, the
Anthropic one (`authentication_error`). The web page at `/` stays public and
asks for the key. `serve` refuses to listen on an address other than
loopback without a key; `--insecure` overrides that. Traffic is plain HTTP:
use a VPN, SSH tunnel or TLS proxy across untrusted networks.

### Monitoring

`GET /metrics` is Prometheus text: `ingot_requests_total{outcome=...}`
(stop, length, tool_calls, error, cancelled), prompt, reused and completion
token counters, prefill and decode tokens/s of the last request,
`ingot_queue_depth`, `ingot_requests_active`,
`ingot_model_loaded{model=...}`, Metal allocated, peak (highest sampled)
and recommended working-set bytes, kept sessions and their capacity, and
uptime. `GET /v1/status` returns the same as JSON plus the active request
(tokens generated so far), the default thinking level and the last 20
requests with their speeds; the web page's Monitor tab polls it.

```sh
curl -s localhost:8080/metrics | grep -v '^#'
curl -s localhost:8080/v1/status
```
### Images

Qwen3.8-Flash-Next reads pictures with its vision projector, the GGUF
`mmproj` file published next to the model weights (ggml-org's
`mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`, 617 MB). Give it with `--mmproj PATH`,
or put exactly one `mmproj*.gguf` in the model's directory and it is found
automatically:

```sh
ingot serve qwen3.8-iq3_xxs --mmproj ~/Models/Qwen3.8-Flash-Next-mmproj/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user",
  "content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -i photo.png)"'"}},
             {"type":"text","text":"What is in this picture?"}]}],"max_tokens":200}'

# chat: --image attaches a picture to the first message, /image PATH to the next one
ingot chat qwen3.8-iq3_xxs --mmproj ~/Models/Qwen3.8-Flash-Next-mmproj/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf --image photo.png
```

- OpenAI `image_url` parts take `data:image/png;base64,…` or
  `data:image/jpeg;base64,…` URLs; Anthropic `image` blocks take
  `{"type":"base64","media_type":"image/png"|"image/jpeg","data":…}`. Images
  are accepted in user messages only. http(s) URLs (OpenAI `url`, Anthropic
  `{"type":"url"}` sources) are fetched only when the server runs with
  `--allow-remote-images` (30 s timeout, 20 MB limit); otherwise they return
  400. A model without a projector returns 400 for any image.
- `/v1/models` reports `"vision": true` for models with a projector, and the
  web chat shows them as `· vision`. There an Image button, pasting a
  screenshot or dropping files attaches pictures to the next message; other
  image types are converted to PNG in the browser. Images are stored with the
  conversation in localStorage while it has room, otherwise only for the visit.
- Preprocessing follows the model's processor as llama.cpp implements it:
  each side is rounded to a multiple of 32 pixels and the image is scaled
  (keeping its aspect ratio) to between 8 and 4096 image tokens, one per
  32×32 pixels, with Pillow-compatible bicubic resampling. A 640×400 picture
  becomes 20×12 = 240 tokens.
- The encoder (`src/vision.rs`, `kernels/vision.metal`) is the 27-layer ViT
  with learned position embeddings resized bilinearly, 2D RoPE and the 2×2
  patch merger, in the projector's Q8_0/F16 weights on Metal, on its own
  command queue. Image tokens enter the text model at the template's
  `<|image_pad|>` with interleaved M-RoPE positions (shared temporal position,
  row and column on the other axes; the image advances `max(columns, rows)`
  positions) and the image token id for the PLE n-gram hashes, as llama.cpp
  does. Embeddings are cached by image content, so a conversation that grows
  by a message reuses the kept session, images included.
- A prompt with images runs without the MTP draft (it is trained on text
  positions). The projector's weights plus 2 GiB for its activations are taken
  out of the memory budget.
- Checked against llama.cpp mtmd with `tools/qwen-vision-oracle.cpp`: on PNG
  input the image embeddings agree to a relative L2 error of 0.1–0.9%
  (per-token cosine ≥ 0.987), and greedy answers for a synthetic diagram and a
  photographed newspaper page (PNG and JPEG) match token for token over 32
  tokens. JPEG pixels can differ from llama.cpp's stb_image decode by a few
  levels, which moves a handful of image tokens more but did not change the
  answer. A 640×400 image encodes in about 0.3 s on an M2 Max.

### Serving every local model

`ingot serve` without MODEL serves every complete local model: manifest
models found below the model directory and pulled `hf:` files. `/v1/models`
lists them (rescanned on each call, the loaded model first), and the model a
request names is loaded when it arrives. One model is resident at a time:
queued requests for the current model finish first, then it is unloaded and
the requested one is loaded; the log shows load and unload times and the
Metal memory allocated before and after. An OpenAI request naming a model
that is not local gets HTTP 404 with the local names. A request without a
model, and an Anthropic request whose model is not local (such as Claude
Code's `claude-*`), uses the loaded model, else `--default MODEL`, else the
first listed model; the log says so.

```sh
ingot serve --context 32768 --keep-alive 10m
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-q4_k_m","messages":[{"role":"user","content":"Hi"}]}'
```

Options: `--default MODEL`, `--context N` (default 8192) and `--memory-gib N`
(default automatic, per model) apply to every load; `--keep-alive DURATION`
(`30s`, `5m`, `1h`; also accepted with MODEL) unloads a model after that long
without requests, and the next request loads it again. `--host`, `--port`,
`--sessions`, `--no-draft` and `--prefix-cache` work as with MODEL.
`serve MODEL` keeps serving only that model.

### Background service

```sh
ingot service install --context 32768 --keep-alive 15m   # --dry-run prints it only
ingot service status
ingot service logs --lines 50
ingot service uninstall
```

`install` writes the per-user launchd agent
`~/Library/LaunchAgents/com.ingot.serve.plist` and loads it with `launchctl`.
It runs `ingot serve` without MODEL on 127.0.0.1 (`--port`, `--default`,
`--keep-alive` and `--context` are passed on), starts at login, restarts after
a crash and logs to `~/Library/Logs/ingot/serve.log`. `INGOT_MODELS` is
recorded when it is set. Install the binary first (`scripts/install.sh`) so the
agent does not point into `target/`. `uninstall` unloads and removes the agent
and keeps the logs.

### Coding agents (Claude Code, Cursor)

`POST /v1/messages` accepts Anthropic Messages requests: `system`, text,
`tool_use` and `tool_result` blocks, custom tools (`name`, `description`,
`input_schema`), `tool_choice` `auto`/`none`, `stop_sequences`, `stream`
(standard Anthropic SSE events) and sampling parameters. Without
`temperature` decoding is greedy (Anthropic defaults to 1.0), which keeps
tool calls deterministic and allows MTP. `max_tokens` is lowered to the
remaining context. The `model` name is echoed back; `serve MODEL` does not check it, and `serve` without MODEL loads it when it is a local model and otherwise uses the loaded one.
`thinking` selects the thinking level (see [Thinking levels](#thinking-levels));
reasoning comes back as `thinking` blocks and earlier `thinking` blocks in
history are replayed into the template. `metadata`, `context_management`,
`output_config` (except its `format`, see structured outputs above) and
`safeguards` are accepted and ignored; images, server tools and forced
`tool_choice` return 400.

Claude Code's system prompt and tools are about 25K tokens, so give the server
a long context:

```sh
ingot serve qwen3.8-iq3_xxs 65536
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local \
  ANTHROPIC_MODEL=qwen3.8 ANTHROPIC_SMALL_FAST_MODEL=qwen3.8 \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
```

OpenAI-compatible clients (Cursor, Continue, Open WebUI) use
`http://127.0.0.1:8080/v1` with the model name from `/v1/models`.

### Model management

A built-in manifest (`src/models/manifest.txt`) lists the models verified with
this engine, pinned to a Hugging Face revision with each file's size and
SHA-256:

| Name | Kind | Source |
| --- | --- | --- |
| `qwen3.8-iq2_xs` | main model | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF `IQ2_XS/` |
| `qwen3.8-iq3_xxs` | main model | same repo, `IQ3_XXS/` (shard 2 identical to IQ2_XS) |
| `qwen3.8-mtp-q8_0` | MTP draft | ggml-org/Qwen3.8-Flash-Next-GGUF |
| `glm5.3-q2_k_s` | main model | ggml-org/GLM-5.3-Flash-GGUF |
| `glm5.3-mtp-q8_0` | MTP draft | ggml-org/GLM-5.3-Flash-GGUF |

```sh
ingot models                  # present/missing, resident memory, recommended budget
ingot pull qwen3.8-iq2_xs     # download and verify; prints the first shard path
ingot chat qwen3.8-iq2_xs     # names resolve to the first shard in the model directory
```

The model directory is `--dir PATH` (models/pull only), else `INGOT_MODELS`, else
`~/.cache/ingot/models`. `models` recognises existing files by name and size
anywhere under that directory (up to 4 levels, hidden directories skipped), so
existing layouts need no migration. A model is `present` only when all its
shards share one directory; a complete copy is preferred over a partial copy
elsewhere. `pull` keeps a complete existing layout, or assembles missing shards
beside the fullest partial copy using verified hard links when possible.
For a new model, `pull` downloads into `.part` files under
`<dir>/<name>/`; rerunning resumes an interrupted download. Files are renamed
only after their SHA-256 matches; existing files are re-verified and skipped
when the hash matches, otherwise the command fails. Identical shards (such as
shard 2 of both Qwen quantizations) are hard-linked within one file system.
Network errors, HTTP errors and hash mismatches all fail with a non-zero exit.
`HF_ENDPOINT` selects a mirror.

Any GGUF on Hugging Face can be pulled by spec instead of name:

```sh
ingot pull hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
ingot pull hf:OWNER/REPO/DIR/MODEL-00001-of-00003.gguf@REVISION   # all 3 shards
ingot chat hf:OWNER/REPO/DIR/MODEL-00001-of-00003.gguf@REVISION
```

`pull` reads each file's size and SHA-256 from the repository's LFS metadata
(`/api/models/REPO/tree/REVISION`), then downloads with the same resume and
verification as named models into `<dir>/hf/OWNER/REPO/PATH`. `REVISION`
defaults to `main`. Naming any shard of a `-0000K-of-0000N` split fetches all
N parts; the first shard is printed. `chat`, `serve` and `generate` accept the
same spec once every shard is present. Missing repositories or files, gated
repositories, files without LFS metadata, network errors and hash mismatches
fail with a non-zero exit. The engine still runs only the architectures it
supports, so a pulled GGUF of another architecture is rejected when loaded.

`models` reads the Metal device's `recommendedMaxWorkingSetSize` and
recommends "resident weights + 4 GiB", capped at the working set minus 4 GiB.
Resident weights exclude Qwen's `per_layer_token_embd`, which is mmap-read by
row. When not all experts fit, the model is marked as SSD expert streaming:
dense weights stay resident and routed experts are read from SSD through fixed
slots.

### Library use

```rust,no_run
use ingot::engine::{Cancellation, Engine};
use std::path::Path;

fn main() -> ingot::Result<()> {
    let mut engine = Engine::load(
        Path::new("/path/to/model-00001-of-00002.gguf"),
        32 << 30, // memory budget in bytes
        4096,     // context capacity
        None,     // Option<DraftConfig> for MTP
        false,    // prefix cache
    )?;
    let prompt = engine.tokenizer.chat("Hello")?;
    let result = engine.generate(&prompt, 64, Cancellation::default(), |_, bytes| {
        use std::io::Write;
        std::io::stdout().write_all(bytes)
    })?;
    eprintln!("{:?} {:?}", result.stop, result.statistics);
    Ok(())
}
```

A `Cancellation` can be cloned to another thread. A cancelled or failed session
cannot be reused; drop it and create a new one. Device errors require reloading
the engine. Errors are `std::io::Error`.

### Kernel profiling

Set `INGOT_KERNEL_PROFILE=/absolute/path/kernels.json` to record GPU time per
compute encoder. The device must support stage-boundary timestamp counters.
The JSON is written on normal exit and groups `kernels` by `phase`
(`prefill`/`decode`/`draft`/`verify` for Qwen; GLM has no phase labels) and
`label`, with `count`, `total_gpu_ns` and `average_gpu_ns`; `phases` holds
command-buffer totals and dropped/failed sample counts. Labels of concurrent
encoders join the kernel names with `+`, and their time is the whole
encoder's. Profiling adds overhead; do not report profiled throughput as
performance. The Metal-only check is:

```sh
cargo test --release --lib metal::profile::tests::kernel_profile_stage_boundary_counters -- --ignored --exact --nocapture
```

## Performance

Measured 2026-10-22 on an M2 Max 38-core / 96 GiB with ingot commit
`f78aa5f`. Full requests, 256 output tokens, temperature 0, prefix cache off,
three alternating ingot/oMLX groups, wall time per request including prefill:

| Prompt (code) | ingot: Qwen IQ2_XS + Q8 MTP, 64 GiB | oMLX: oQ4e, all experts resident | Ratio |
| --- | ---: | ---: | ---: |
| 1K tokens | 7.81 s | 9.55 s | 0.82 |
| 4K tokens | 16.93 s | 16.74 s | 1.01 |

- Per-group ratios: 1K 0.765/0.843/0.850, 4K 0.997/1.018/1.018. At 4K the
  remaining gap is prefill (10.0–10.5 s vs 9.7–9.8 s); decode is on par.
  The fused prefill expert kernel merged afterwards cut 4K prefill by a
  further ~1.3% (not in this table).
- **The quantizations differ**: ingot runs a ~2-bit IQ2_XS GGUF and oMLX its
  own ~4-bit oQ4e weights. This is not a same-weights comparison. On a frozen
  2569-task quality set, IQ2_XS on ingot scored 75.9% vs 78.0% for oMLX
  oQ4e; llama.cpp on the same IQ2_XS file was within 0.23 points of ingot,
  so the gap is from the quantization, not the engine.
- oMLX 0.7.1.dev1 / mlx 0.32.3, `--memory-guard-gb 78`, expert offload
  disabled, its MTP off (faster than its MTP mode in this setup).
- Tokenization differs by one token (ingot 1023/4095 IDs vs 1024/4096).
- OS file cache and background load were not isolated (load average 1.70 at
  start, swap in use).

Earlier measurements with bounded memory and short prompts (2026-10-13) are in
[validation/native-performance-3.json](validation/native-performance-3.json).

### vs Strata on Mac

Same-cohort comparison, 2026-10-29, M2 Max 38-core / 96 GiB, ingot commit
`f8e22c0` vs strata-mac (Strata's patched llama.cpp for macOS) commit `63acca2`.
Both engines served **the same GGUF files** and the same Q8_0 MTP head through
their OpenAI-compatible servers. Requests were streamed chat completions with
the GGUF chat template and thinking off, greedy, exactly 256 output tokens, and
no prompt cache. Each request's prompt token count was identical on both
engines (1K/4K/16K = 1038–1040 / 4110–4112 / 16399–16401 tokens). Code prompts
are Rust source; prose prompts are text from *The Rust Programming Language*.
Ratio = ingot time / strata time per pair, so below 1 means ingot is faster.
Each row is 6 ABBA pairs; each block starts a fresh server and runs an untimed
~2.7K-token warm-up first.

| Model, decoding | Prompt | Code: ingot / strata s (ratio, 95% CI) | Prose: ingot / strata s (ratio, 95% CI) | Prefill tok/s ingot / strata | Decode tok/s ingot / strata |
|---|---|---:|---:|---:|---:|
| IQ2_XS, MTP | 1K | 9.3 / 10.9 (0.86, 0.75–0.97) | 10.2 / 11.8 (0.86, 0.80–0.92) | 310 / 277 | 40.4 / 34.1 |
| IQ2_XS, MTP | 4K | 19.0 / 21.0 (0.91, 0.87–0.95) | 18.6 / 21.7 (0.86, 0.83–0.89) | 352 / 320 | 35.9 / 30.3 |
| IQ2_XS, MTP | 16K | 62.2 / 68.5 (0.91, 0.89–0.93) | 64.3 / 70.3 (0.92, 0.85–0.99) | 295 / 274 | 33.8 / 26.9 |
| IQ2_XS, AR | 1K | 11.1 / 14.1 (0.79, 0.71–0.87) | 12.2 / 13.9 (0.89, 0.61–1.17) | 271 / 287 | 34.5 / 24.9 |
| IQ2_XS, AR | 4K | 20.5 / 24.0 (0.86, 0.74–0.99) | 20.3 / 23.4 (0.87, 0.81–0.93) | 325 / 327 | 33.4 / 23.2 |
| IQ2_XS, AR | 16K | 68.2 / 69.8 (0.98, 0.90–1.06) | 70.1 / 70.0 (1.00, 0.96–1.04) | 272 / 287 | 29.6 / 20.2 |
| IQ3_XXS, MTP | 1K | 10.8 / 12.6 (0.86, 0.80–0.92) | 10.8 / 12.2 (0.89, 0.85–0.93) | 290 / 258 | 35.5 / 30.8 |
| IQ3_XXS, MTP | 4K | 20.4 / 22.8 (0.90, 0.81–0.98) | 21.0 / 22.9 (0.92, 0.82–1.02) | 314 / 292 | 33.8 / 29.2 |
| IQ3_XXS, MTP | 16K | 70.5 / 73.8 (0.96, 0.90–1.01) | 71.7 / 74.1 (0.97, 0.91–1.02) | 264 / 258 | 29.0 / 24.8 |
| IQ3_XXS, AR | 1K | 11.1 / 13.8 (0.81, 0.77–0.84) | 11.0 / 13.8 (0.79, 0.74–0.85) | 312 / 309 | 33.2 / 24.5 |
| IQ3_XXS, AR | 4K | 20.6 / 23.4 (0.88, 0.82–0.94) | 20.6 / 23.2 (0.89, 0.84–0.94) | 332 / 333 | 31.1 / 23.4 |
| IQ3_XXS, AR | 16K | 68.1 / 69.0 (0.99, 0.96–1.02) | 68.3 / 69.0 (0.99, 0.96–1.02) | 278 / 292 | 28.0 / 20.1 |

- **ingot is faster on the mean in every 1K and 4K cell and in all six
  IQ2_XS MTP cells. The 95% interval lies below 1 in 16 of 24 cells**,
  including all six IQ2_XS MTP cells. IQ2_XS AR prose 1K and IQ3_XXS MTP prose
  4K are unresolved. Decode is 15–47% faster in every cell.
- **No cell is a significant loss, but 16K without MTP is a tie** (ratios
  0.98–1.00, intervals spanning 1), and IQ3_XXS MTP at 16K is not resolved
  (0.96/0.97). The cause is prefill: strata prefills 16K prompts 4–7% faster in
  AR mode (TTFT 56–57 s vs 59–61 s), which uses up ingot's decode lead. A kernel
  profile (4K → 16K prose, 4× the tokens) shows ingot's GPU prefill growing
  10.1 → 55.0 s. Its prefill attention kernel (`attention_prefill_tiles`) grows
  0.65 → 11.5 s (17.7×) and is 21% of 16K prefill. Expert and dense matmuls grow
  roughly linearly. MTP draft catch-up is only ~1.5 s of GPU time at 16K.
  Long-context prefill attention is the place to work on.
- Memory (peak, max over blocks): the system's wired memory rose by 46.6 GiB
  for ingot and 44.2 GiB for strata (IQ2_XS MTP); 54.3 vs 52.0 GiB for
  IQ3_XXS MTP. strata's weights are an mmapped file (process RSS ~47 GiB,
  footprint 2.3 GiB). ingot's are Metal buffers (RSS 1.9 GiB, footprint
  12.1 GiB at 16K, Metal allocation 40.4 GiB for IQ2_XS). Per-process RSS
  therefore isn't comparable between the two engines.
- Quality: on the first 100 Chinese, math and code tasks of the frozen
  2569-task suite (IQ2_XS, MTP, identical prompt token IDs, same caps,
  unchanged grader), ingot scored 79.0% and strata 78.7%. One task was
  correct only on ingot, none only on strata; per category zh 50/49,
  math 97/97, code 90/90. Total non-inferiority passed (one-sided lower bound
  −1.4 points ≥ −2). 250/300 outputs were byte-identical. In the timed runs the
  engines' greedy texts also diverge after tens of tokens (2-bit near-ties);
  both always produced 256 tokens.
- strata-mac ran its documented resident profile (`STRATA_SSD_EXPERTS=0`,
  `--gpu-layers 99 --batch-size 1024 --ubatch-size 1024 -ncmoe 0 -mec 0
  --load-mode mmap --lazy-mode on`; with MTP: `--spec-type draft-mtp
  --spec-draft-n-max 3 --spec-draft-p-min 0`). On top of that profile it had
  `--ctx-size 20480` (for the 16K cell), `--reasoning off`,
  `--no-cache-prompt` and `--cache-ram 0`. Its optional head-only Q4 draft was
  not tested; both engines used the Q8_0 MTP head. ingot: `serve MODEL 20480
  --sessions 1` with `--draft-model … --draft-tokens 3` or `--no-draft`.
- Background load was not isolated. Swap was nearly full (~26–30 GiB) because
  an idle Ollama runner holds a 28 GiB footprint; OrbStack, a Godot editor and
  other apps were active, and both engines read 0.1–5 GiB from SSD per request.
  The first IQ2_XS MTP cohort ran while another ingot process was active
  outside the GPU lock. It was discarded and rerun (the table shows the rerun).
  Its results pointed the same way but were noisier. Ratios within a pair are
  robust to this drift; absolute seconds are not.

## Validation

```sh
cargo fmt --all --check
cargo clippy --release --lib --bin ingot --example bench_tokens -- -D warnings
cargo test --release --lib
```

These run without model weights. Real-weight tests are `#[ignore]`d, read the
model paths from environment variables (`QWEN_MODEL`, `QWEN_DRAFT`,
`GLM_MODEL`, `GLM_DRAFT`, `DRAFT_TOKENS`, `DRAFT_VOCAB`, ...) and should be run
one at a time, for example:

```sh
QWEN_MODEL="$QWEN" cargo test --release --locked --test qwen \
  state_cancellation_memory_and_reuse -- --ignored --nocapture --test-threads=1
```

The MTP tests check that speculative output is token-for-token identical to
autoregressive output. Avoid loading two large models at the same time.

### Before publishing a release

Hosted CI's Metal step is informational because its virtual GPU is not a
substitute for real Apple Silicon. A green CI run alone is not native acceptance.
On a real Mac with the pinned main and draft weights already downloaded, run:

```sh
export QWEN_MODEL=/path/to/Qwen-first-shard.gguf
export QWEN_DRAFT=/path/to/Qwen-MTP.gguf
export GLM_MODEL=/path/to/GLM-first-shard.gguf
export GLM_DRAFT=/path/to/GLM-MTP.gguf
./scripts/validate-native.sh
```

The script checks the required files before starting, downloads nothing, runs
tests sequentially, and stops on any failure. Allow at least the default 44 GiB
Qwen / 32 GiB GLM engine budget plus OS headroom; a 96 GiB or larger Mac is the
existing validation class, not a guarantee for every model or configuration.
It records the commit, working-tree changes, Mac/OS, Rust toolchain, model paths,
budgets and test output in `evidence/native-*.log` (or a supplied log-file path).

The suite covers GPU kernels, host and CLI behavior, real-weight projections,
tokenization, cancellation, session reuse, MTP equivalence, prefix-cache reuse
and request isolation for both models. The Qwen prefix fixture is generated
from its tokenizer; no external token-file fixture is needed. Long-context
quality, performance benchmarks and ignored experimental probes are separate
checks; this script does not certify them or all Mac configurations.

Before publishing, review a **PASS** report for the release commit from a clean
checkout, retain it with the release evidence, and list the tested hardware and
weights in the release notes. Rerun after code changes. Missing weights, a failed
test, or an untested commit means the release remains unaccepted. Tag-triggered
packaging creates a **draft** release only; producing an archive is not acceptance.
The orchestration test in CI uses a mock Cargo command and cannot replace this
hardware run. See [NATIVE_VALIDATION.md](NATIVE_VALIDATION.md) for prior evidence.

The oracle tools (`tools/*oracle*.cpp`, `tools/check_native*.py`) need the
pinned reference builds described in [NATIVE_VALIDATION.md](NATIVE_VALIDATION.md).
`examples/bench_tokens.rs` is the persistent-engine benchmark used for the
numbers above.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | GGUF reader, tokenizer, Qwen/GLM model code, MTP, Metal runtime |
| `kernels/` | Metal shaders; `kernels/NOTICE` lists adapted ggml material |
| `examples/` | benchmarks, validation and draft-vocabulary tools |
| `tests/` | integration tests (most need real weights) |
| `tools/` | Python/C++ oracle, benchmark and quality scripts |
| `validation/` | small recorded validation/performance summaries |
| `macos/`, `CMakeLists.txt`, `setup-macos.sh`, `run-macos.sh` | pinned llama.cpp reference path, comparison only |

### Reference engine (comparison only)

The CMake path builds the pinned `memoriaru/llama.cpp` fork with
`macos/llama-metal.patch` as an independent reference. Its results are not
counted as ingot results.

```sh
./setup-macos.sh
cmake --build build-macos --target moe-mac-check moe-quality-logits
./run-macos.sh --model "$QWEN"
```

## Documents

Current:

- [ARCHITECTURE.md](ARCHITECTURE.md): design and trade-offs (Chinese). Written
  before MTP and the prefix cache were implemented; its "first version" scope
  and the MTP section describe the plan at that time.
- [PROVENANCE.md](PROVENANCE.md), [kernels/NOTICE](kernels/NOTICE): origin of
  the code, pinned reference revisions, adapted third-party material.

Historical records, kept for traceability and not updated for the current code:

- [NATIVE_VALIDATION.md](NATIVE_VALIDATION.md): acceptance record of
  2026-10-12/13.
- [OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md): the 2026-10-13 optimization
  spec (MTP, matrix prefill, ...), since implemented in large part.
- [ROADMAP.md](ROADMAP.md), [RUST_ENGINE.md](RUST_ENGINE.md): 2026-10-13 status
  snapshots.
- [macos/DESIGN.md](macos/DESIGN.md), [macos/VALIDATION.md](macos/VALIDATION.md):
  records of the llama.cpp experiment this repository was extracted from.

Large raw evidence lives in the git-ignored `evidence/` directory and is not
part of the repository.

## License

MIT, see [LICENSE](LICENSE). Quantization layouts/codebooks and some kernel
techniques are adapted from ggml/llama.cpp under the MIT license; see
[kernels/NOTICE](kernels/NOTICE) and [macos/NOTICE](macos/NOTICE). Model weights
are not distributed here and are subject to their own licenses.
