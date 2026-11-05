# Ingot engine

How the engine works, what it runs, and what it does not support yet.

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
numerical oracles during validation (see [PROVENANCE.md](../PROVENANCE.md)).

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
  the recorded checks in [NATIVE_VALIDATION.md](history/NATIVE_VALIDATION.md).
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

Gemma 4's global layers have head size 512. Their flash prefill kernel has no
tail variant (its key tile would not fit in threadgroup memory), so the last
partial tile of fewer than 64 query rows uses dense attention. Re-measured
2026-11-05 on the M2 Max with another GPU job's model resident: prefill of a
3813-token chat prompt went from 125 to 261 tok/s (llama.cpp: 293 tok/s at
pp4562), and 48 greedy tokens still match llama.cpp on two long chat prompts. Phi-3-mini-4k produces garbage past
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
[HTTP server](REFERENCE.md#http-server-and-interactive-chat)).

Weights used for all recorded results (SHA-256 in
[macos/model-sha256.txt](../macos/model-sha256.txt) and
[NATIVE_VALIDATION.md](history/NATIVE_VALIDATION.md)):

| Model | File | Size | MTP draft head |
| --- | --- | ---: | --- |
| Qwen3.8-Flash-Next | `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`, 2 shards | ~68 GB | `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, ~4.1 GB, [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/tree/052beeaca7bec4a303e59cc7bc630c4f3a1b845d) |
| GLM-5.3-Flash | [ggml-org Q2_K_S](https://huggingface.co/ggml-org/GLM-5.3-Flash-GGUF) rev `c259a9f`, 2 shards | ~110 GB | `mtp-GLM-5.3-Flash-Q8_0.gguf`, ~9.3 GB |

An ISTA-DASLab IQ3_XXS GGUF of Qwen3.8-Flash-Next (~70.6 GiB, uses Q4_K,
Q5_K, IQ2_XS and IQ3_XXS) also runs; its quality results are still pending.

Both shards must be in the same directory; pass the `-00001-of-00002` shard.
