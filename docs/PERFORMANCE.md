# Ingot performance

Full method and results. The [README](../README.md#performance) has the summary.

Measured 2026-10-22 on an M2 Max 38-core / 96 GiB with Ingot commit
`f78aa5f`. Full requests, 256 output tokens, temperature 0, prefix cache off,
three alternating ingot/oMLX groups, wall time per request including prefill:

| Prompt (code) | Ingot: Qwen IQ2_XS + Q8 MTP, 64 GiB | oMLX: oQ4e, all experts resident | Ratio |
| --- | ---: | ---: | ---: |
| 1K tokens | 7.81 s | 9.55 s | 0.82 |
| 4K tokens | 16.93 s | 16.74 s | 1.01 |

- Per-group ratios: 1K 0.765/0.843/0.850, 4K 0.997/1.018/1.018. At 4K the
  remaining gap is prefill (10.0–10.5 s vs 9.7–9.8 s); decode is on par.
  The fused prefill expert kernel merged afterwards cut 4K prefill by a
  further ~1.3% (not in this table).
- **The quantizations differ**: Ingot runs a ~2-bit IQ2_XS GGUF and oMLX its
  own ~4-bit oQ4e weights. This is not a same-weights comparison. On a frozen
  2569-task quality set, IQ2_XS on Ingot scored 75.9% vs 78.0% for oMLX
  oQ4e; llama.cpp on the same IQ2_XS file was within 0.23 points of Ingot,
  so the gap is from the quantization, not the engine.
- oMLX 0.7.1.dev1 / mlx 0.32.3, `--memory-guard-gb 78`, expert offload
  disabled, its MTP off (faster than its MTP mode in this setup).
- Tokenization differs by one token (Ingot 1023/4095 IDs vs 1024/4096).
- OS file cache and background load were not isolated (load average 1.70 at
  start, swap in use).

Earlier measurements with bounded memory and short prompts (2026-10-13) are in
[validation/native-performance-3.json](../validation/native-performance-3.json).

## vs Strata on Mac

Same-cohort comparison, 2026-10-29, M2 Max 38-core / 96 GiB, Ingot commit
`f8e22c0` vs strata-mac (Strata's patched llama.cpp for macOS) commit `63acca2`.
Both engines served **the same GGUF files** and the same Q8_0 MTP head through
their OpenAI-compatible servers. Requests were streamed chat completions with
the GGUF chat template and thinking off, greedy, exactly 256 output tokens, and
no prompt cache. Each request's prompt token count was identical on both
engines (1K/4K/16K = 1038–1040 / 4110–4112 / 16399–16401 tokens). Code prompts
are Rust source; prose prompts are text from *The Rust Programming Language*.
Ratio = Ingot time / strata time per pair, so below 1 means Ingot is faster.
Each row is 6 ABBA pairs; each block starts a fresh server and runs an untimed
~2.7K-token warm-up first.

| Model, decoding | Prompt | Code: Ingot / strata s (ratio, 95% CI) | Prose: Ingot / strata s (ratio, 95% CI) | Prefill tok/s Ingot / strata | Decode tok/s Ingot / strata |
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

- **Ingot is faster on the mean in every 1K and 4K cell and in all six
  IQ2_XS MTP cells. The 95% interval lies below 1 in 16 of 24 cells**,
  including all six IQ2_XS MTP cells. IQ2_XS AR prose 1K and IQ3_XXS MTP prose
  4K are unresolved. Decode is 15–47% faster in every cell.
- **No cell is a significant loss, but 16K without MTP is a tie** (ratios
  0.98–1.00, intervals spanning 1), and IQ3_XXS MTP at 16K is not resolved
  (0.96/0.97). The cause is prefill: strata prefills 16K prompts 4–7% faster in
  AR mode (TTFT 56–57 s vs 59–61 s), which uses up Ingot's decode lead. A kernel
  profile (4K → 16K prose, 4× the tokens) shows Ingot's GPU prefill growing
  10.1 → 55.0 s. Its prefill attention kernel (`attention_prefill_tiles`) grows
  0.65 → 11.5 s (17.7×) and is 21% of 16K prefill. Expert and dense matmuls grow
  roughly linearly. MTP draft catch-up is only ~1.5 s of GPU time at 16K.
  Long-context prefill attention is the place to work on.
- Memory (peak, max over blocks): the system's wired memory rose by 46.6 GiB
  for Ingot and 44.2 GiB for strata (IQ2_XS MTP); 54.3 vs 52.0 GiB for
  IQ3_XXS MTP. strata's weights are an mmapped file (process RSS ~47 GiB,
  footprint 2.3 GiB). Ingot's are Metal buffers (RSS 1.9 GiB, footprint
  12.1 GiB at 16K, Metal allocation 40.4 GiB for IQ2_XS). Per-process RSS
  therefore isn't comparable between the two engines.
- Quality: on the first 100 Chinese, math and code tasks of the frozen
  2569-task suite (IQ2_XS, MTP, identical prompt token IDs, same caps,
  unchanged grader), Ingot scored 79.0% and strata 78.7%. One task was
  correct only on Ingot, none only on strata; per category zh 50/49,
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
  not tested; both engines used the Q8_0 MTP head. Ingot: `serve MODEL 20480
  --sessions 1` with `--draft-model … --draft-tokens 3` or `--no-draft`.
- Background load was not isolated. Swap was nearly full (~26–30 GiB) because
  an idle Ollama runner holds a 28 GiB footprint; OrbStack, a Godot editor and
  other apps were active, and both engines read 0.1–5 GiB from SSD per request.
  The first IQ2_XS MTP cohort ran while another Ingot process was active
  outside the GPU lock. It was discarded and rerun (the table shows the rerun).
  Its results pointed the same way but were noisier. Ratios within a pair are
  robust to this drift; absolute seconds are not.
