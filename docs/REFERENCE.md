# Ingot reference

Every command, option and API detail. New here? Start with the
[quick start](QUICKSTART.md).

## Install from source

There is no published binary release yet. Install from source with Rust 1.88+
and Xcode Command Line Tools:

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
