# Source and ownership

This is an independent repository extracted from the local Mac experiments at
commit `63bcc9f7970c2d8a7d9e38407e0a352a112b269c` on 2026-10-11. It starts a new
Git history.

The earlier checkout began with [Niko1221/Strata](https://github.com/Niko1221/Strata)
at `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. The Mac execution path was developed
on top of a separate llama.cpp dependency. This extraction does not include the
original Strata CUDA/HIP engine, application, installers or pack pipeline.

## Native Rust execution — 2026-10-11

The project direction is now a native Rust inference engine. `src/` owns GGUF
reading, routing and Metal dispatch; `kernels/moe.metal` executes the quantized
matrix products, SwiGLU and expert combination. It now implements the complete
Qwen text-generation trunk, state and bounded expert loading. See
`NATIVE_VALIDATION.md` for the exact acceptance evidence and remaining work.

The Rust binary uses `objc2`, `objc2-foundation` and `objc2-metal` to call Apple
frameworks, and Hugging Face Tokenizers 0.22.1 for BPE. It does not link libllama or libggml and does not spawn their tools.
`kernels/quant_tables.metal` and the quantization formulas are adapted from the
pinned ggml source below, with attribution and MIT terms in `kernels/NOTICE`.
`tools/check_rust_moe.py` and `tools/check_native.py` load ggml solely as
independent test oracles. `tools/qwen-oracle.cpp` is a separate executable
linked to the fixed reference engine; the Rust binary never calls it.

## Reference runtime dependency

- Repository: [memoriaru/llama.cpp](https://github.com/memoriaru/llama.cpp)
- Revision: `555d1ec9daa4564cb06adb39c469acfdcd1d1e93`
- Source archive SHA-256:
  `774d5c5c88a7b4b6122478fe4fb5e19d950c626c64c11a670c1a997a0e4ee65d`
- This branch supplies the reference graph, tokenizer, server, Metal backend and
  expert-pool foundation. The CMake build applies `macos/llama-metal.patch` to it.
- The retained patch also includes work adapted from ggml-org/llama.cpp revision
  `b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea`. The ggml MIT notice is retained in
  `macos/NOTICE`.

The runtime patch is copied byte for byte from the frozen local experiment
commit. This extraction changes build/package identity and documentation,
not the tested kernel, scheduling or model behavior. The extra
`moe-quality-logits` build target exposes the existing evaluator source.

## Inherited tools and names

`tools/draft_vocab.py`, `tools/strata_tokenizer.py` and `tools/gguf_reader.py`
come from Strata. They are retained together because the draft-vocabulary tool
imports the tokenizer, which imports the GGUF reader. The Strata copyright and
MIT license remain in `LICENSE`. `tools/export_mtp_shortlist.py` and the Mac
evaluation/profiling tools come from the local Mac work.

Internal names such as `STRATA_SSD_EXPERTS`, other `STRATA_*` experiment switches
and the `strata.draft_vocab_ids` GGUF metadata key remain unchanged to preserve
the existing execution and artifact contract during extraction. They do not
connect this repository to Strata's remote or require its application.
The two retained Python utility module names likewise identify their origin.

## Historical evidence

`macos/DESIGN.md`, `macos/VALIDATION.md` and the small recorded validation
outputs are historical copies from the same frozen commit. Links to ignored raw
archives and old local paths may require the original experiment workspace. The
working log, the 35 MiB `optimization-results.json`, model weights, build
products and ignored raw archives were not copied. The original checkout
has been preserved; this repository does not claim to reproduce those complete
experiments merely by retaining their summaries.

The extraction source was a local checkout named `strata-mac`.

This project was developed under the working name `mac-moe`. The recorded outputs
in `validation/` keep that name in their command lines.

## Extraction verification — 2026-10-11

- A fresh build in this repository passed for `llama-server`, `llama-bench`,
  `llama-perplexity`, both required test executables, `moe-profile-model` and
  `moe-quality-logits`.
- The network download was slow and was stopped. The identical original source
  archive was copied into the new download cache after SHA-256 verification;
  CMake then extracted and patched it normally. No old binary was reused.
- `moe-mac-check` completed successfully: batch allocation, resident expert pool
  and file-backed expert pool. Both expert-pool runs used `MTL0` and reported
  `PASSED`; they were not skipped for lack of an accelerator.
- 21 copied runtime/tool/evidence files matched the frozen commit byte for byte.
  The runtime patch SHA-256 is
  `4430ca14efcf62c2c9c77f3aa67a32a7f5c7f674d999508a1547f703b6fc1d46`.
- Shell/Python syntax, JSON parsing, optional NumPy/regex imports and both
  vocabulary-tool help entries passed. Edited packaging files passed the Git
  whitespace check. The source checkout remained clean.
- Complete-model quality and speed were not rerun during this extraction.
  GLM 5.3 Flash was not implemented or verified in that extraction.

## Native GLM continuation — 2026-10-12

The selected text artifact is ggml-org/GLM-5.3-Flash-GGUF Q2_K_S at revision
`c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa` (two shards, 110,027,352,992 bytes).
The 9,430,240-byte primary shard carries vocabulary/metadata; the second carries
1,383 tensors, using F32, BF16, Q8_0 and Q2_K. Actual directory totals are
10,134,678,776 non-expert bytes and 99,883,155,456 routed-expert bytes.
The directory and bounded real weight-row reads were inspected independently;
full-shard download/hash verification and full-model acceptance are recorded
separately in NATIVE_VALIDATION.md.

Adopt/adapt/build decision: retain the existing GGUF reader and direct Metal
runtime; add an independent GLM trunk and share the existing quantized weights
and fixed expert slots. Select the maintained ggml-org Q2_K_S artifact with
standard block layouts. The smaller aj9o9 mixed-IQ2 artifact was not selected
for this baseline: its different mixed quantization would add another artifact
variable before the GLM mathematics is established. Q2_K/Q4_K are larger and
unnecessary for this first implementation. Official FP8 Safetensors would
require another weight format/conversion path and was not selected.

The separate validation-only oracle uses ggml-org/llama.cpp revision
`08ffba21e5c2f1728b6b405c05ff3913f65706d4`; it does not replace the original
Qwen reference. Apply `tools/glm-oracle-loading.patch` before building that
reference: its three changes mark the routed expert tensors for on-demand
reading. Model equations and kernels remain unchanged. The oracle uses
load mode NONE, lazy reading ON and a 4 GiB GPU expert cache. The unpatched
mmap attempt registered approximately 103 GiB with Metal and failed its first
decode with an out-of-memory error on the 96 GiB Mac; that failed run is kept
separately from the successful results. The loading patch SHA-256 is
`ba48dfeb4e3557ceb2e511fcaf2fe40c8fd5041a8b04513fdff85645f9b0c0cf`.
License/attribution is in kernels/NOTICE. Validation proceeds
through real-row dequantization, independent vocabulary/template checks,
F32 operator/state checks, teacher-forced logits/traces and complete generation.
Neither the artifact publisher's quality claims nor llama.cpp performance
are reported as native Rust results.
