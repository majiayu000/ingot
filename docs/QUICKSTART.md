# Ingot quick start

Ingot runs AI chat models on your own Mac. Nothing you type leaves your
computer. This page gets you from nothing to a working chat, then shows how to
use it from your browser, other apps and other devices.

If something goes wrong, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
If you want an AI coding agent to do the setup for you, point it at
[AI_SETUP.md](AI_SETUP.md).

## What you need

- A Mac with Apple Silicon (M1 or later) and macOS 15 or later.
- Free disk space for a model: from about 5 GiB for a small model to over
  100 GiB for the largest one (see [Which model?](#which-model-should-i-pick)).
- Xcode Command Line Tools and Rust 1.88 or later (the installer checks both
  and tells you how to get them).

Ingot is an early project. It has been tested on one machine (M2 Max,
96 GiB). Other Macs should work but are untested.

## Install

Open the **Terminal** app and run:

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/install.sh
```

The script builds Ingot (a few minutes the first time) and copies it to
`~/.local/bin`. It never asks for your password and never uses `sudo`.

If it says `~/.local/bin is not on your PATH`, either copy the `export PATH=...`
line it prints into `~/.zshrc`, or run `scripts/install.sh --yes` to let it add
that line for you. Then **open a new Terminal window**.

Check that it worked:

```sh
ingot --version
```

Useful installer options: `--dry-run` (check and show the steps, change
nothing), `--prefix DIR` (install into `DIR/bin`), `--no-build` (reuse an
existing build).

## Your first chat

```sh
ingot
```

1. A numbered list of models appears with their download sizes. Type a number
   or name and press Enter (Enter alone picks the default).
2. Ingot shows how much it will download and where. Type `y` and press Enter.
   Nothing is downloaded until you say yes. If the download stops, run the
   same command again: it continues where it left off and checks every file.
3. When you see **Ready!**, type a message and press Enter.

In the chat:

| Type | To |
| --- | --- |
| `/reset` | start a new conversation |
| `/help` | list the commands |
| `/exit` (or Ctrl-D) | quit |
| Ctrl-C | stop the current reply |

Next time, run `ingot` again; downloaded models are reused. To skip the menu,
name the model: `ingot run qwen3-8b-q4_k_m`.

## Which model should I pick?

Run `ingot models`. It lists every model, its download size, whether you
already have it, and the memory budget it recommends on **your** Mac (the
"Metal working set" line shows how much memory your Mac lets the GPU use,
usually about two thirds to three quarters of its RAM).

Sizes from `ingot models`:

| Your Mac's memory | Start with | Download | Memory it wants |
| --- | --- | ---: | ---: |
| 16 GB | `qwen3-8b-q4_k_m` (Qwen3 8B) or `llama3.1-8b-instruct-q4_0` (Llama 3.1 8B) | 4.7 / 5.6 GiB | 9 / 10 GiB |
| 32 GB | `qwen3-30b-a3b-q4_k_m` (Qwen3 30B-A3B, fast and smarter) | 17.3 GiB | 22 GiB |
| 64 GB | `qwen3.8-iq2_xs` (Qwen3.8-Flash-Next, the main tested model) | 63.4 GiB | 41 GiB |
| 96 GB or more | `qwen3.8-iq3_xxs` (higher-quality Qwen3.8) | 70.6 GiB | 48 GiB |
| 96 GB or more, lots of disk | `glm5.3-q2_k_s` (GLM-5.3-Flash, reads part of the model from disk while it runs, slower) | 102.5 GiB | 73 GiB |

Rules of thumb:

- Pick a model whose "memory it wants" is below your Mac's Metal working set,
  with room to spare for your other apps.
- The big Qwen3.8 and GLM models can still run with less memory: Ingot keeps
  part of the model on the SSD and reads it as needed. That works, but it is
  much slower. You do not need to configure anything; memory is chosen
  automatically.
- Bigger downloads are better answers but slower replies.
- The `-mtp-` entries (`qwen3.8-mtp-q8_0`, `glm5.3-mtp-q8_0`) are optional
  add-ons that make Qwen3.8 and GLM answer faster. Get one with
  `ingot pull qwen3.8-mtp-q8_0`; it is used automatically.

Models are stored in `~/.cache/ingot/models`. To keep them on another disk,
set `INGOT_MODELS` every time you run Ingot, for example
`INGOT_MODELS=/Volumes/SSD/models ingot`.

You can also download a GGUF file from Hugging Face by name:
`ingot pull hf:OWNER/REPO/FILE.gguf`. Ingot runs Llama, Qwen2, Qwen3 and
Qwen3-MoE files this way; other model families are rejected with an error.

## Use it from your browser

```sh
ingot serve
```

Leave that window open and visit <http://127.0.0.1:8080/> in your browser. You
get a chat page with a model picker, a conversation list, stop/regenerate
buttons and copy buttons for code blocks. It works offline; nothing is sent anywhere.

`ingot serve` lists every model you have downloaded and loads the one you
pick when you first use it (one at a time). Press Ctrl-C in the Terminal to
stop it.

### Keep it running in the background

So you do not need a Terminal window open:

```sh
ingot service install
```

The server now starts when you log in and restarts if it crashes. Useful
commands: `ingot service status`, `ingot service logs`,
`ingot service uninstall`. Add `--keep-alive 15m` to `install` to free the
model's memory after 15 minutes without use.

## Use it from other apps

All of these need the server running (`ingot serve` or the background
service). Use a model name from `ingot models`.

### Open WebUI

Add an **OpenAI** connection:

- URL: `http://127.0.0.1:8080/v1` (if Open WebUI runs in Docker, use
  `http://host.docker.internal:8080/v1`)
- API key: anything, for example `local`

Your downloaded models appear in the model list.

### Cursor, Continue and other OpenAI-compatible apps

Set the OpenAI base URL to `http://127.0.0.1:8080/v1`, any API key, and a
model name from `ingot models`. Note: Cursor sends some requests through its
own servers, which cannot reach `127.0.0.1` on your Mac; if Cursor says it
cannot connect, that is why. Do not expose Ingot to the internet to work
around this: it has no password protection.

### Claude Code

Claude Code sends a large set of instructions with every request, so start the
server with a long context first:

```sh
ingot serve --context 65536
```

Then, in another Terminal window, in your project folder:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local \
  ANTHROPIC_MODEL=qwen3.8-iq2_xs CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
```

Replace `qwen3.8-iq2_xs` with the model you downloaded. Tool use (editing
files, running commands) works with Qwen-family models, not with GLM.

## Use it from your phone or another computer

By default the server only answers your own Mac. To let other devices on
**your home or office network** use it:

```sh
KEY=$(openssl rand -hex 24); echo "$KEY"
ingot serve --host 0.0.0.0 --api-key "$KEY"
```

Keep the printed key; other devices need it. Find your Mac's address with
`ipconfig getifaddr en0` (Wi-Fi), then on the other device open
`http://THAT-ADDRESS:8080/` in a browser (the page asks for the key), or use
`http://THAT-ADDRESS:8080/v1` with that key in an app.

Only do this on a network you trust, and never expose the port to the
internet. macOS may ask whether to allow incoming
connections; allow it for Ingot.

## Update Ingot

```sh
cd ingot
git pull
scripts/install.sh
```

If you use the background service, it keeps running the old version until you
run `ingot service install` again (add the same options you used before); that
restarts it with the new version.
