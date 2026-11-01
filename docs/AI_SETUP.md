# Setting up Ingot with an AI coding agent

Give this file to an AI coding agent (Claude Code, Codex, Cursor and the like)
and ask it to "install Ingot for me following docs/AI_SETUP.md". The steps
below are written for the agent. A person can follow them too; the plain guide
is [QUICKSTART.md](QUICKSTART.md).

## Instructions for the agent

You are installing Ingot, a local LLM inference engine for Apple Silicon
Macs, for the user on their own Mac. Work in a terminal on that Mac.

### Ground rules

- **Never use `sudo`** and never change system settings. Everything installs
  into the user's home folder.
- **Ask before every large download.** Model files are 4.7 GiB to 102.5 GiB.
  Tell the user the model name, the download size and the destination folder,
  and wait for a clear yes. Do not pass `--yes` to `ingot run` unless the
  user approved that exact download.
- Ask before changing shell startup files (`~/.zshrc`); `scripts/install.sh`
  only does this with `--yes`.
- Ask before installing the background service (`ingot service install`),
  before binding the server to anything other than `127.0.0.1`, and before
  installing Xcode Command Line Tools or Rust.
- Do not invent flags. Check `ingot --help` and `ingot COMMAND --help`
  (`run`, `chat`, `serve`, `service`, `models`) on the installed version before
  using an option. Use only commands shown there or in this file.
- Commands that load a model use the GPU and a lot of memory, and can take
  minutes to start. Run them in the foreground only when the user wants to
  chat; run `ingot serve` as a long-running background process.
- If a step fails, look up the exact message in
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and tell the user what it means
  before trying a fix.

### Step 1: check the machine

```sh
uname -m                     # must print arm64
sw_vers -productVersion      # must be 15 or later
sysctl -n hw.memsize         # RAM in bytes, used to choose a model
xcode-select -p              # must print a path; otherwise CLT are missing
cargo --version              # must be 1.88 or later
df -h ~                      # free disk space
```

- `uname -m` is not `arm64`: stop. Ingot needs Apple Silicon (or the
  terminal runs under Rosetta; tell the user).
- Xcode Command Line Tools missing: ask the user to run
  `xcode-select --install` and finish the installer window (it needs a click).
- Rust missing or too old: ask the user, then install it from
  <https://rustup.rs> (user-level, no sudo) or run `rustup update`.

### Step 2: get the source and install

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/install.sh --dry-run    # checks prerequisites, changes nothing
scripts/install.sh              # builds and copies ingot to ~/.local/bin
```

If the script reports that `~/.local/bin` is not on PATH, ask the user whether
to add it, then run `scripts/install.sh --yes --no-build` (it appends one
`export PATH=...` line to `~/.zshrc`). In the current shell, use
`~/.local/bin/ingot` or `export PATH="$HOME/.local/bin:$PATH"`.

Verify:

```sh
ingot --version
ingot models
```

### Step 3: choose a model with the user

Read `ingot models`. It prints the "Metal working set" (GPU memory available
on this Mac) and, per model, the download size and the recommended memory
budget. Recommend the largest model whose recommended budget is comfortably
below the working set and whose download fits on disk:

| Mac RAM | Suggested model | Download |
| --- | --- | ---: |
| 16 GB | `qwen3-8b-q4_k_m` | 4.7 GiB |
| 32 GB | `qwen3-30b-a3b-q4_k_m` | 17.3 GiB |
| 64 GB | `qwen3.8-iq2_xs` | 63.4 GiB |
| 96 GB+ | `qwen3.8-iq3_xxs` | 70.6 GiB |

Confirm the choice and the download size with the user, then download:

```sh
ingot pull MODEL
```

`pull` resumes if interrupted (rerun the same command) and verifies SHA-256.
Models go to `~/.cache/ingot/models` unless `INGOT_MODELS` is set; if the
user wants another disk, set `INGOT_MODELS` and tell them it must be set the
same way on every later run (and is recorded by `ingot service install`).

### Step 4: configure what the user wants

Chat in the terminal (interactive; let the user drive it):

```sh
ingot run MODEL
```

Local server with web chat at `http://127.0.0.1:8080/`:

```sh
ingot serve                       # every downloaded model, loaded on demand
```

Background service (ask first; starts at login):

```sh
ingot service install --dry-run   # show what it would do
ingot service install
ingot service status
```

Check the server answers (no model load needed):

```sh
curl -s http://127.0.0.1:8080/v1/models
```

Client settings to give the user:

- Open WebUI, Cursor, Continue and other OpenAI-compatible clients: base URL
  `http://127.0.0.1:8080/v1`, any API key, model name from `/v1/models`. Open
  WebUI in Docker: `http://host.docker.internal:8080/v1`.
- Claude Code: the server needs a long context (`ingot serve --context 65536`
  or `ingot service install --context 65536`), then:

  ```sh
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local \
    ANTHROPIC_MODEL=MODEL CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
  ```

  Do not write these variables into the user's shell startup files unless they
  ask; otherwise their normal Claude Code would stop using the real API.

Other devices on the local network: `ingot serve --host 0.0.0.0 --api-key KEY`
(or `INGOT_API_KEY=KEY`); without a key the server refuses a non-loopback
host unless `--insecure` is given. Get the user's consent first and tell them
the key. Never expose it to the internet.

### Step 5: report back

Tell the user what was installed and where (`~/.local/bin/ingot`), which
model was downloaded and its size, whether the service is installed, which
URLs to use, and any change made to `~/.zshrc`. To undo everything:
`ingot service uninstall`, delete `~/.local/bin/ingot`, delete the model
folder (ask first; it is large and slow to re-download) and remove the
`# ingot` PATH line from `~/.zshrc` if one was added.
