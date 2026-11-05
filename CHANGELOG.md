# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Performance

- Gemma 4 prompt processing is about twice as fast: its head-size-512 global
  layers now use flash prefill attention, which also allows 512-token prefill
  blocks. A 3813-token chat prompt on an M2 Max went from 125 to 261 tok/s;
  greedy output still matches llama.cpp.

## [0.1.0] - 2026-11-05

First public release.

### Added

- Native Rust + Metal engine for Qwen3.8-Flash-Next (`qwen4exp`) and
  GLM-5.3-Flash (`glm5-next`) GGUF models on Apple Silicon. No llama.cpp, ggml
  or MLX at run time.
- Weights executed in their stored GGML types: F32, F16, BF16, Q8_0, Q2_K,
  Q4_K, Q5_K, Q6_K, Q2_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_NL and IQ4_XS.
- MTP speculative decoding with the trained NextN draft head (1–3 draft
  tokens, optional draft-vocabulary shortlist); output is token-for-token
  identical to greedy autoregressive decoding.
- Resident experts when the memory budget allows, otherwise fixed per-layer
  expert slots with LRU replacement and overlapped reads.
- Grouped prefill with GPU-side expert grouping and fused routed
  gate/up/SwiGLU expert kernels.
- Single-prefix cache across requests (`--prefix-cache`).
- Per-kernel GPU timing via `INGOT_KERNEL_PROFILE`.
- Mac app (`scripts/package-app.sh`): disk image, native window and menu bar
  panel, first-run model list, downloads with progress, crash recovery, system
  proxy support, start at login and automatic updates (Sparkle, EdDSA-signed
  appcast on the GitHub release). The app and disk image are notarized.
- App pages: Hugging Face search, model details, custom models, Connect apps and
  Settings. Chat search, projects, and PDF and text attachments. Chats are stored
  in a file on this Mac. The interface follows the system's light or dark mode.
- `ingot list`, `ps`, `show`, `stop`, `rm`, `cp`, `create` and `launch`.
- `min_p`, `repeat_penalty`, `num_ctx` and `keep_alive` per request, preloading
  and immediate unloading, `GET /v1/models/{id}` and `POST /v1/responses`.
- Templates without a system role, such as Phi-3's, no longer drop system prompts.
- Ollama-compatible API under `/api/`, checked with the real `ollama` client.
- Requests from other sites that would change state are refused with 403.
- `ingot serve` without MODEL picks a context that fits each listed model.
- `ingot run` recommends the best model for the Mac from its Metal working set and
  free disk, picks a context that fits, and marks models that will not run well.
- When CONTEXT is omitted, it stops at the limit declared by the GGUF file.
- `generate`, `tokenize`, `logits`, `trace`, `dequant` and `moe-layer`
  commands; library API (`ingot::engine::Engine`).

### Performance

Measured 2026-10-22 on an M2 Max (38-core GPU, 96 GiB), engine commit
`f78aa5f`, Qwen3.8-Flash-Next IQ2_XS GGUF with the Q8_0 MTP head (3 draft
tokens), 64 GiB budget, all experts resident. Compared with oMLX 0.7.1.dev1 /
mlx 0.32.3 serving its oQ4e weights with all experts resident and MTP off.
Full requests on code prompts, 256 output tokens, temperature 0, prefix cache
off, three alternating groups:

| Prompt | ingot | oMLX | Ratio |
| --- | ---: | ---: | ---: |
| 1K tokens | 7.81 s | 9.55 s | 0.82 |
| 4K tokens | 16.93 s | 16.74 s | 1.01 |

- The two engines run **different quantizations** (about 2-bit IQ2_XS vs
  about 4-bit oQ4e). This is a system comparison, not a same-weights engine
  comparison.
- On a frozen 2569-task quality set, IQ2_XS on ingot scored 75.9% vs 78.0%
  for oMLX oQ4e; llama.cpp on the same IQ2_XS file scored within 0.23 points of
  ingot, so the gap comes from the quantization.
- At 4K the remaining gap is in prefill (10.0–10.5 s vs 9.7–9.8 s); decode is
  on par. The fused prefill expert kernel added after this measurement cut 4K
  prefill by about 1.3% more.
- OS file cache and background load were not isolated. Tokenization differs by
  one token (1023/4095 vs 1024/4096 IDs).

### Known limitations

- Greedy decoding only; one model and one active session per engine.
- Single-user text template only; no multimodal input or tool messages.
- Tested only on one M2 Max / 96 GiB machine with macOS 26.5.2.
