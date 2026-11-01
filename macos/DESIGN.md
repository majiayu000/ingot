# Implementation decision

2026-10-07. Required: the full Strata model on Apple Silicon, finite expert residency, SSD reads, correctness and performance evidence. A small-model chat UI does not satisfy this scope.

| System | Verified boundary | Decision |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata), `82f46a8` | Session loop/expert source depend on CUDA/HIP and x86 SIMD. Normal mode holds experts in RAM and an embedding table on SSD. | Retain source/model target; defer translating the entire pipeline. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `qwen4exp` model graph, Metal quantized expert operations, ARM CPU and lazy embeddings. | Adopt graph, tokenizer, kernels and API. |
| [Expert pool fork](https://github.com/memoriaru/llama.cpp/tree/moe-expert-pool), `555d1ec` | Scheduler maintains slots and ID remapping; misses copy from mapped host addresses. MIT. | Adapt to batched file reads; retain routing and LRU. |
| [Existing Mac port](https://github.com/jinzy0623/Strata-macOS) | README limits validation to small GGUF and excludes original expert streaming/MTP. | Do not use it as the core implementation. |
| [Flash-MoE](https://github.com/danveloper/flash-moe) | Direct Metal and parallel reads for Qwen3.5-397B; published measurements are on its author's M3 Max. | Use the read strategy as prior art; do not copy the different model's engine. Root license metadata did not establish a reusable license. |
| [Anemll Flash MLX](https://github.com/Anemll/anemll-flash-mlx) | Stable slot banks, exported sidecars and Qwen3.5 MLX models. MIT. | Defer separate format/export pipeline; inspected version did not establish a Qwen3.8 path. |
| [LM Studio](https://lmstudio.ai/docs/app) | Existing local-model desktop product. | A desktop wrapper does not implement the requested core. No private architecture claims. |

These boundaries are verified from primary source code/docs. External speed figures remain author reports. Local measurements determine this repository's claims.

## Adaptation

`STRATA_SSD_EXPERTS=1` selects lazy CPU mappings for expert tensors. The loader retains GGUF file descriptors/offsets. The context binds those ranges to existing pools. The scheduler batches misses with GCD and `pread` directly into shared Metal slots after routing readback synchronizes the queue, then remaps IDs and multiplies. A Metal backend extension confirms CPU access; private/device-only buffers are rejected at binding. Quantized bytes are unchanged. Model routing and active expert counts do not change.

Descriptors share ownership per model shard and pools hold close-on-exec duplicates. File bounds and shared-buffer access are checked at binding. Reads handle interruption/short reads and propagate pooled-read I/O failures through `GGML_STATUS_FAILED`; a damaged pool stays failed.

Accelerate BLAS must not count as a second GPU. The inherited device count disabled caching on a normal Mac build; real server startup logs exposed this and drive the fix's validation.

Slots cap resident expert weights; clean OS file pages remain reclaimable. Dense/KV/compute memory is separate. Prefill batches larger than the pool take the existing selective-copy path. No prediction or speculative I/O is added.

## Validation plan and risks

Compare ordinary/file-backed expert outputs across types, routing strides, cold loads, hits, eviction and graph rebuilds. Close original descriptors and truncate source files. Build from the pinned archive. Verify real server pool registration, ordinary/SSE requests and recovery after cancellation. Compare resident/mapped-pool/file-pool using identical model and batch settings. Verify both target-model checksums before a full-model run.

SSD locality and prefill memory are performance risks. MTP, images, long-context quality and low-memory Macs need separate evidence. Synthetic parity and another MoE model do not establish Qwen3.8 quality or speed.

## Measured optimization

The first complete-model run retained per-pool staging vectors: peak footprint was 6,506.2 MiB at 64 slots. The [pinned Metal source](https://github.com/memoriaru/llama.cpp/blob/555d1ec9daa4564cb06adb39c469acfdcd1d1e93/ggml/src/ggml-metal/ggml-metal-device.m#L2215) implements shared-buffer tensor writes with `memcpy`; its scheduler already synchronizes at routing readback. Direct positional reads into the confirmed shared slots remove staging allocation and the extra copy. Failed reads may partially overwrite slots, but return a failed graph before multiplication and leave the pool failed until recreation. Parity, truncation, repeated failure and complete-model API checks passed after this change; final performance and memory measurements are recorded in VALIDATION.md.
