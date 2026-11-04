# Ingot

[![CI](https://github.com/majiayu000/ingot/actions/workflows/ci.yml/badge.svg)](https://github.com/majiayu000/ingot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run AI models on your Apple Silicon Mac, in the terminal, in your browser, or
behind your favorite apps. Ingot is one Rust binary with its own Metal kernels.
It does not use llama.cpp or MLX at run time.

[简体中文](README.zh-CN.md) ·
[Quick start](docs/QUICKSTART.md) ·
[Reference](docs/REFERENCE.md) ·
[Performance](docs/PERFORMANCE.md)

## Get started

You need an Apple Silicon Mac and macOS 15 or later. There is no prebuilt
download yet, so build it first with Rust 1.88+ and the Xcode Command Line
Tools:

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/package-app.sh       # the Mac app and disk image, in dist/
scripts/install.sh           # or only the command line tool
```

**Mac app:** open `dist/Ingot-0.1.0.dmg` and drag Ingot to Applications. It
opens on a list of models rated for your Mac. Pick one, wait for the download,
and start typing. Chats, projects, PDF and text attachments, custom models and
settings are all in the app. Ingot stays in the menu bar so your other apps can
reach it.

**Command line:** run `ingot`, pick a model and confirm the download.

Your conversations stay on your Mac.

## What it does

- **Chat anywhere:** in the Mac app, in the terminal, or in the built-in web
  page at <http://127.0.0.1:8080/> after running `ingot serve`.
- **Works with your tools:** OpenAI, Anthropic and Ollama compatible APIs for
  Open WebUI, Cursor, Claude Code and any app with an Ollama setting.
- **Runs very large models:** Qwen3.8-Flash-Next and GLM-5.3-Flash, with the
  experts that do not fit in memory streamed from SSD.
- **Runs common GGUF models:** Llama, Qwen2, Qwen3, Qwen3-MoE, Gemma 3, Gemma 4
  and Phi-3 family files, from the built-in list or directly from Hugging Face.
- **Finds and shapes models:** search Hugging Face from the app, and save custom
  models with your own prompt and parameters, or import a Modelfile.
- **Starts your tools:** `ingot launch claude`, `codex` or `opencode`.
- **Covers the details:** tool calling, thinking levels, image input (Qwen3.8),
  API keys, Prometheus metrics, a background service, and resumable downloads
  checked with SHA-256.

## Which model?

Open the app, or run `ingot` and press Enter. Ingot checks this Mac's memory and
free disk, recommends the best model that fits, and picks a context length for
it. A model you already downloaded comes first if it fits.

Examples, with the memory each needs at a context of 8192 tokens:

| Model | Download | Memory |
| --- | ---: | ---: |
| `gemma3-4b-it-qat-q4_0` | 2.4 GiB | 5.0 GiB |
| `qwen3-8b-q4_k_m` | 4.7 GiB | 7.5 GiB |
| `qwen3-30b-a3b-q4_k_m` | 17.3 GiB | 19.3 GiB |
| `qwen3.8-iq2_xs` | 63.4 GiB | 41 GiB |

Memory here means what the GPU can use, which is less than your Mac's RAM. The
first screen of `ingot` shows the number for your Mac.

```sh
ingot models                # everything available, and what fits
ingot list                  # what you have
ingot ps                    # what is loaded
ingot stop qwen3-8b-q4_k_m  # free its memory now
ingot run qwen3-8b-q4_k_m   # skip the menu
```

To run another supported GGUF, use `ingot pull hf:OWNER/REPO/FILE.gguf`.

## Connect your apps

```sh
ingot serve qwen3-8b-q4_k_m
```

| Client | Setting |
| --- | --- |
| Browser | Open <http://127.0.0.1:8080/> |
| Open WebUI, Cursor, other OpenAI clients | Base URL `http://127.0.0.1:8080/v1`, model name from `/v1/models` |
| Apps with an Ollama setting | Server `http://127.0.0.1:8080` |
| Claude Code | [Setup guide](docs/QUICKSTART.md#claude-code) |

The Mac app starts this server for you at port 11435. Use **Copy API Address**
in its menu bar icon. The server listens on this Mac only by default. To reach it from a phone or
another computer, see [Remote access](docs/REFERENCE.md#remote-access-and-api-keys).

## Performance

Measured on one M2 Max (38-core GPU, 96 GiB). Time ratios below 1 mean Ingot is
faster.

| Compared with | Test | Result |
| --- | --- | --- |
| [Strata](docs/PERFORMANCE.md#vs-strata-on-mac) | Qwen3.8, same GGUF and MTP head, 1K and 4K prompts | Request time ratio 0.79–0.92; generation 15–47% faster |
| Strata | Same, 16K prompts without MTP | Tie (0.98–1.00); Strata's prompt processing is 4–7% faster |
| [oMLX](docs/PERFORMANCE.md) | Qwen3.8, different quantization | Ratio 0.82 at 1K, 1.01 at 4K; oMLX's 4-bit weights score higher on quality |
| [llama.cpp](docs/ENGINE.md#generic-architectures) | Seven standard GGUF models | Logits match to about 1e-4; generation within ±7%; prompt processing 8–19% slower, 53% slower on Gemma 4 |

Ingot is strongest on the Qwen3.8 and GLM-5.3 paths it was built for. Standard
GGUF models are correct and close, but llama.cpp still processes prompts faster.

## Status

Ingot is an early project.

- Tested on one machine only. Other chips, memory sizes and macOS versions are
  untested.
- The server runs one inference at a time and has no TLS.
- Tool calling is not available for GLM. Image input needs Qwen3.8 and its
  vision projector.
- Other GGUF architectures are rejected with an error.

See the [full list of limitations](docs/ENGINE.md#status-and-limitations).

## Documentation

| Guide | What it covers |
| --- | --- |
| [Quick start](docs/QUICKSTART.md) | Install, first chat, choosing a model, connecting apps and devices |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common errors and how to fix them |
| [AI setup](docs/AI_SETUP.md) | Instructions to hand to an AI coding agent |
| [Reference](docs/REFERENCE.md) | All commands, options and API details |
| [Engine](docs/ENGINE.md) | How the engine works and which weights it runs |
| [Performance](docs/PERFORMANCE.md) | Full benchmark method and results |
| [Development](docs/DEVELOPMENT.md) | Building, validation and repository layout |
| [Architecture](docs/ARCHITECTURE.md) | Design and trade-offs (Chinese) |
| [Provenance](PROVENANCE.md) | Where the code came from and adapted material |
| [History](docs/history/) | Dated records, not updated for the current code |

## License

MIT, see [LICENSE](LICENSE). Quantization layouts and some kernel techniques are
adapted from ggml/llama.cpp under the MIT license; see
[kernels/NOTICE](kernels/NOTICE) and [macos/NOTICE](macos/NOTICE). Model weights
are not included and follow their own licenses.
