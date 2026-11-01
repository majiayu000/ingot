# Local validation

Apple M2 Max, 96 GB unified memory; 2026-10-07.

Completed: ARM64 Metal build; resident pool 192 matrix cases; file-backed pool the same 192 cases. Expert-pool outputs match the ordinary Metal path byte for byte. This is parity between the two Metal paths, not an independent CPU reference for their shared kernels. Formats: Q2_K, Q3_K, Q4_K, Q6_K, Q8_0, MXFP4, Q2_0, IQ2_XS, IQ3_S, IQ2_S, IQ2_XXS, IQ1_M. Routing uses 512 experts, top-10 and 64 slots, with small synthetic weight matrices: Q2_0 uses the target down-projection input width of 640 (640×64); the other formats use 512×64. These cover the four expert formats found in the target GGUF header; both complete target files have now passed SHA-256 verification. Matrix: 1/3/8/9-token routing, cold loads, eviction, hits/reloads and graph rebuilds, comparing two weight tensors per case.

File mode places its data at an unaligned offset of `2^32 + 37` bytes, covering file offsets above 4 GiB. It also tests original descriptor closure, invalid descriptor/range rejection, truncation and repeated failure on a damaged pool. A failed read does not produce a successful graph.

Raw results: [resident](pool-resident-test.txt), [file-backed](pool-file-test.txt).

The archive-only product build and both targeted CTest cases passed (2/2). After the direct-read optimization, resident and file-backed executables again passed all 192 cases each; binding-failure cleanup was additionally exercised with private Metal buffers. The latest patch also applied cleanly to a pristine copy of the pinned archive; all nine patched files matched the tested sources. Checks were local; no hosted CI result is claimed. A synthetic two-layer qwen4exp MoE fixture also completed prefill and generation with four registered pools and observed cache misses, with Accelerate enabled. This checks the model graph and loader/context/scheduler integration; random weights do not measure language quality or large-model throughput. OpenAI chat, SSE termination, client cancellation and a subsequent request passed on a separate 20 MB synthetic fully-MoE Qwen3 fixture with a real tokenizer. Fixed-seed greedy outputs and token usage matched the ordinary Metal baseline. Six pools were confirmed bound to positional file reads; server logs confirmed cancellation. The fixture was derived from tiny-random/qwen3-moe revision `10a349dcb488b10c27aa4a3c1dbefb74c41565c3`, replacing its unsupported dense first MLP with a copy of its random expert layer. This is an integration test, not language-quality evidence. The complete target-model results are recorded below.

The initial 30B benchmark was invalid as a pool benchmark: BLAS was treated as a second GPU and disabled pools; the local Muse-Glimmer model also lacked MoE tensors. Those numbers are excluded. Real server registration must be verified before recording pool performance.

A separate synthetic PLE fixture completed 32-token prefill and 32-token generation with a 10.24 GB logical F16 table (320,001,446 rows, 16 heads, scaled row width 16) and four file-bound expert pools. Maximum RSS was 128,139,264 bytes (122.2 MiB); peak footprint was 55,730,000 bytes. The file used sparse zero pages and approximately 19 MB of allocated storage. This checks lazy mapping and PLE graph integration, not physical SSD speed or target-model quality. [Recorded measurement](ple-memory-test.json). PLE was placed on the recurrent layer, matching the target architecture; a synthetic PLE on a full-attention layer crashed during graph construction; source inspection indicates that the missing recurrent PLE history caused the failure. That unsupported layout is outside the target-model validation.

The default four-token microbatch was verified on a 70 MB synthetic 512-expert/top-10 Qwen3 MoE fixture with distinct random expert weights (seed 51210). A 128-token prefill-only probe observed progressive pool hits/misses across microbatches; six file-backed pools were registered. Greedy chat outputs and usage matched resident Metal, and SSE/cancellation/recovery passed at 8K context. [Integration result](api-512-test.json). This establishes the target routing geometry and launch path, not Qwen3.8 language quality. The default uses at most 40 experts per microbatch against 64 slots; the previous 32-token microbatch exceeded this gate and fell back to mapped selective copies during prefill.

## Complete target model

Both files from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, completed download and passed the recorded SHA-256 checks (68,026,093,024 bytes total). No weights are redistributed. The complete model loaded with 144 file-bound pools at 8K context. Three greedy math/Chinese/English requests produced complete answers and exactly matched resident Metal messages and token usage. SSE reached `[DONE]`; client cancellation appeared in server logs and a subsequent request returned `391`. [API evidence](full-model-api-test.json). These are short-prompt integration checks, not a broad language-quality or long-context evaluation.

`llama-bench` measured pp128/tg128, three repetitions, eight CPU threads, batch 256, Metal layers 99 and lazy embeddings. Separate requested contexts of 128 tokens were padded to 256 by the engine, confirmed in logs. OS file cache was uncontrolled after download/checksum; no global cache purge. These do not measure cold SSD bandwidth or 8K throughput.

| Path | Slots / microbatch | pp128 token/s | tg128 token/s | Peak RSS GiB | Peak footprint GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Resident Metal | 0 / 4 | 48.23 | 26.14 | 36.99 | 0.35 |
| Mapped expert pool | 64 / 4 | 36.11 | 17.98 | 41.22 | 4.48 |
| Direct file expert pool | 64 / 4 | 34.25 | 16.45 | 8.20 | 4.46 |
| Direct file expert pool | 128 / 8 | 48.19 | 17.89 | 12.23 | 8.59 |

[Commands, samples and memory counters](full-model-benchmark.json). RSS and footprint account differently for the mapped resident path; do not interpret footprint as total physical RAM or add the two counters. Reclaimable OS file cache also consumes system memory. The file path reduces measured process RSS; no equivalent reduction in total system RAM is claimed. Fully resident Metal is faster on this 96 GB machine.

Direct reads removed per-pool staging: the earlier 64-slot peak footprint was 6.35 GiB, versus 4.46 GiB after the change. Private buffers are rejected at binding; the negative probe returned exit 1 with `failed to bind expert file ranges`, with resources released. Default shared-buffer tests and complete-model API passed. The default remains 64 slots/4 tokens for the smaller measured memory cost; 128 slots/8 tokens is a measured prefill tradeoff. Smaller-memory hardware, cold SSD performance, images and long-context quality remain unverified. Qwen4Exp MTP validation is recorded below.


## Qwen4Exp MTP

The shipped patch now supports the verified official Q8 NextN head as an opt-in on the complete IQ2_XS target. Seven existing source files adapt loading, wide hidden capture, the single draft graph, cache filters and recurrent rollback; the original SSD/Metal pool implementation is unchanged. Eight complete fixed 1K/4K responses matched original 256-token sequences. Length three at 128 slots/microbatch eight improved measured generation while slowing prefill and 4K complete requests. Default 64 slots/microbatch four was checked with length two: the 32-token prefix, three short chats, SSE termination, cancellation and recovery matched saved results. The inherited splitter rejects microbatch <= draft length + 1; this contract is retained. Multiple concurrent sequences and broader quality are unverified.


## Current default (2026-10-10)

The four-token microbatch and MTP splitter constraints above describe historical experiments, not the current launch contract. The current SSD launcher uses batch/microbatch 1024, 64 expert slots, one sequence, and MTP off. Expert-grouped prefill and read/compute overlap are adopted. Draft length three works at microbatch 1024 and the former four-token boundary; cancellation, rollback, SSE and recovery checks passed. Local raw archives now live in ../evidence/ and are excluded from Git.
