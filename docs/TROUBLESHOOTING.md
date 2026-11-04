# Something went wrong?

Find the message you see below (search this page for a few words of it). The
text in quotes is what Ingot or the installer prints; parts in CAPITALS
differ on your Mac. For first steps see [QUICKSTART.md](QUICKSTART.md).

When you report a problem, include the full message, the command you ran,
`ingot --version` and the output of `ingot models`.

## Installing

### macOS cannot verify Ingot

The disk image on the releases page is signed and notarized, so it opens
normally. A copy built with `scripts/package-app.sh` and no `INGOT_SIGN_ID` is
signed ad hoc, and another Mac blocks it the first time. Try to open it and
choose **Done**, then go to **System Settings > Privacy & Security** and choose
**Open Anyway**.

### The Mac app says it could not start

Choose **Show Log**. The log is at `~/Library/Logs/Ingot/ingot.log`. **Try
Again** starts the engine again; it also restarts by itself after a crash.

### A PDF cannot be attached

A scanned PDF holds pictures of pages, so there is no text to read. PDFs over
16 MiB are also refused.

### Xcode Command Line Tools missing

```text
install.sh: Xcode Command Line Tools are missing; run `xcode-select --install`, finish the installer, then rerun this script
```

Run `xcode-select --install`, click **Install** in the window that opens, wait
for it to finish, then run `scripts/install.sh` again. Compiler or linker
errors such as `xcrun: error` or `linker 'cc' not found` during the build have
the same fix.

### Rust missing

```text
install.sh: cargo not found; install Rust from https://rustup.rs (1.88 or later), open a new terminal and rerun
```

Install Rust from <https://rustup.rs>, open a **new** Terminal window and run
the installer again. If Rust is already installed but older than 1.88, the
build fails with an edition or version error: run `rustup update`.

### Wrong Mac or macOS version

```text
install.sh: ingot needs an Apple Silicon Mac (found x86_64); under Rosetta, use a native arm64 terminal
install.sh: ingot needs macOS 15 or later (Metal residency sets); this is macOS VERSION
```

Ingot needs an M-series Mac with macOS 15 or later. If you do have Apple
Silicon and still see `x86_64`, your Terminal is running under Rosetta: in
Finder, select Terminal, choose **Get Info**, untick **Open using Rosetta** and
reopen it.

### `ingot: command not found`

The install folder is not on your PATH. Run `scripts/install.sh --yes` (it adds
`~/.local/bin` to `~/.zshrc`), then open a new Terminal window. Until then you
can run `~/.local/bin/ingot` directly. If you installed with `cargo install`,
the program is in `~/.cargo/bin` instead.

## Downloading

### Download interrupted

```text
ingot: download URL: ERROR; rerun pull to resume
To retry and resume a partial download, run `ingot run NAME`.
```

Your network dropped or the server stopped answering. Run the same command
again (`ingot run NAME` or `ingot pull NAME`): it continues from where it
stopped. Nothing needs to be deleted.

If you are in a region where Hugging Face is slow or blocked, set
`HF_ENDPOINT` to a Hugging Face mirror you trust before running the command.

### Hash mismatch

```text
SHA-256 mismatch for PATH: expected HASH, got HASH; the partial download was removed
```

The downloaded file was damaged on the way. Ingot already deleted it; run the
same command again to download it fresh.

If the message does **not** end with "the partial download was removed", the
mismatching file is one that was already in your model folder (for example an
old or different copy). Move or delete the file named in the message, then run
the command again.

### Private or missing Hugging Face file

```text
ingot: https://huggingface.co/api/models/OWNER/REPO/tree/main: HTTP access denied; OWNER/REPO is private or gated
```

Check the spelling of `hf:OWNER/REPO/FILE.gguf`. Gated repositories (ones that
need you to accept a license on the website) cannot be pulled yet; pick a
public GGUF repository instead.

### Not enough disk space

The download fails with a "No space left on device" error. Free some space, or
put models on a bigger disk by setting `INGOT_MODELS` on **every** run, for
example `INGOT_MODELS=/Volumes/SSD/models ingot`. `ingot models` shows
which folder is in use on its first line.

## Starting a model

### Model not found

```text
ingot: model NAME is not complete in FOLDER; run `ingot pull NAME` or set INGOT_MODELS
```

The model is not downloaded (or only partly) in the folder Ingot looks in.
Run `ingot pull NAME`, or if you keep models elsewhere, set `INGOT_MODELS`
to that folder. All parts of a multi-part model must be in the same folder.

```text
ingot: NAME is not a file or a known model name (unknown model NAME; known models: ...)
```

The name is misspelled. Copy it from the list in the message or from
`ingot models`. You can also give the full path to a `.gguf` file.

### Not enough memory

```text
ingot: ARCH with context N needs X GiB, budget is Y GiB
ingot: memory budget N cannot hold required N bytes
```

The model, plus the room it needs to remember the conversation (the
"context"), does not fit in the memory your Mac makes available. Try, in this
order:

1. Close other large apps and try again.
2. Use a smaller context: `ingot chat MODEL 4096` (the default is 8192
   tokens), or `ingot serve --context 4096`.
3. Pick a smaller model; `ingot models` shows the memory each one wants on
   your Mac.

```text
ingot: --sessions N needs X GiB for ... but the memory budget has Y GiB left after the model (K session(s) fit); lower --sessions or CONTEXT, or raise MEMORY_GIB
```

You asked `serve` to keep more conversations in memory than fit. Leave out
`--sessions` (Ingot then picks what fits) or lower `--context`.

If the Mac becomes very slow while a model runs (memory pressure is red in
Activity Monitor), the model is too large for your Mac: choose a smaller one.

### Slow first start

The first time you load a model after downloading it or restarting the Mac,
Ingot has to read the whole model from the SSD (tens of GiB for the larger
models) and macOS compiles the GPU code. That can take minutes; wait for
**Ready!** (in `chat`) or for the first reply (in `serve`). The next start is
much faster while macOS still has the file in its cache.

With `ingot serve`, a model is loaded when the first request for it arrives,
so that first request waits for the load. If you used `--keep-alive`, an idle
model is unloaded and the next request loads it again.

### Unsupported model file

```text
ingot: unsupported model architecture: ARCH; supported: qwen4exp, glm5-next, llama, qwen2, qwen3, qwen3moe, gemma3, gemma4, phi3
ingot: unsupported tokenizer MODEL/PRE; supported: GPT-2 BPE with qwen35, glm4, qwen2, llama-bpe or dbrx pre-tokenizers, SentencePiece (llama) or gemma4
ingot: GGML type N is unsupported
```

Ingot runs only some model families and file formats (the lists in the
message are what your installed version supports). Download a model from
`ingot models` instead, or a Llama / Qwen / Gemma / Phi GGUF in a common quantization such
as Q4_K_M.

## Using the server

### Port already in use

```text
ingot: cannot listen on 127.0.0.1:8080: Address already in use (os error 48)
```

Something else already uses port 8080. Often it is Ingot itself: the
background service (`ingot service status`) or a `ingot serve` in another
Terminal window. Use that one, stop it, or pick another port:
`ingot serve --port 8081` (then use `8081` in all URLs). To see which program
holds the port: `lsof -nP -iTCP:8080 -sTCP:LISTEN`.

### An app says the model was not found

```text
model 'NAME' not found in FOLDER; local models: ...
```

The app asked for a model you have not downloaded. Use one of the names the
message lists (the same names `ingot models` shows as `present`).

```text
no complete local models in FOLDER; run `ingot pull NAME` first
```

The server found no downloaded models. Download one, or start the server with
the same `INGOT_MODELS` setting you used for the download.

### An app cannot connect

- Is the server running? `ingot service status`, or look at the Terminal
  window where you ran `ingot serve`.
- Open WebUI in Docker must use `http://host.docker.internal:8080/v1`, not
  `127.0.0.1`.
- From another device, the server must be started with `--host 0.0.0.0` and
  an `--api-key` (send it as `Authorization: Bearer KEY`), and
  both devices must be on the same network (see
  [QUICKSTART.md](QUICKSTART.md#use-it-from-your-phone-or-another-computer)).
- Cursor sends some requests through its own servers, which cannot reach your
  Mac's `127.0.0.1`.

### Server busy (HTTP 503)

```text
server busy; retry later
```

The server handles one request at a time and queues up to 8. Wait and retry,
or send fewer requests at once.

### Claude Code fails with a context error

```text
prompt plus output exceeds context, or empty request
```

Claude Code's instructions alone are about 25,000 tokens. Start the server
with `--context 65536` (for the background service, run
`ingot service install --context 65536`).

## Chatting

### The conversation is full

```text
conversation needs N tokens but the context is N; use /reset
```

Type `/reset` to start a new conversation, or start chat with a larger context,
for example `ingot chat MODEL 32768` (uses more memory).

### Background service problems

`ingot service status` shows whether it is installed and running;
`ingot service logs --lines 100` shows its recent log, where the messages on
this page appear. To change settings, run `ingot service install` again with
the new options; it replaces and restarts the service.
