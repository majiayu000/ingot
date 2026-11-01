# Developing Ingot

Building, packaging, validation and the repository layout.

## Build

```sh
cargo build --release --locked
```

Use a separate `--target-dir` (or `CARGO_TARGET_DIR`) for each checkout of
this crate so that different versions do not share build products.

## Packaging

The source installation above is the current user path. Maintainers can build
an archive on Apple Silicon:

```sh
./scripts/release.sh   # writes dist/*.tar.gz and dist/*.tar.gz.sha256
```

Archives contain a self-contained binary with embedded Metal shader sources.
After a release is published, unpack its archive and run `./ingot run`.
`packaging/homebrew/ingot.rb` is an unfinished template, not an installable tap.
The root `setup-macos.sh` / `run-macos.sh` scripts belong to the llama.cpp
comparison engine; they do not install or launch the native Ingot app.

## Validation

```sh
cargo fmt --all --check
cargo clippy --release --lib --bin ingot --example bench_tokens -- -D warnings
cargo test --release --lib
```

These run without model weights. Real-weight tests are `#[ignore]`d, read the
model paths from environment variables (`QWEN_MODEL`, `QWEN_DRAFT`,
`GLM_MODEL`, `GLM_DRAFT`, `DRAFT_TOKENS`, `DRAFT_VOCAB`, ...) and should be run
one at a time, for example:

```sh
QWEN_MODEL="$QWEN" cargo test --release --locked --test qwen \
  state_cancellation_memory_and_reuse -- --ignored --nocapture --test-threads=1
```

The MTP tests check that speculative output is token-for-token identical to
autoregressive output. Avoid loading two large models at the same time.

### Before publishing a release

Hosted CI's Metal step is informational because its virtual GPU is not a
substitute for real Apple Silicon. A green CI run alone is not native acceptance.
On a real Mac with the pinned main and draft weights already downloaded, run:

```sh
export QWEN_MODEL=/path/to/Qwen-first-shard.gguf
export QWEN_DRAFT=/path/to/Qwen-MTP.gguf
export GLM_MODEL=/path/to/GLM-first-shard.gguf
export GLM_DRAFT=/path/to/GLM-MTP.gguf
./scripts/validate-native.sh
```

The script checks the required files before starting, downloads nothing, runs
tests sequentially, and stops on any failure. Allow at least the default 44 GiB
Qwen / 32 GiB GLM engine budget plus OS headroom; a 96 GiB or larger Mac is the
existing validation class, not a guarantee for every model or configuration.
It records the commit, working-tree changes, Mac/OS, Rust toolchain, model paths,
budgets and test output in `evidence/native-*.log` (or a supplied log-file path).

The suite covers GPU kernels, host and CLI behavior, real-weight projections,
tokenization, cancellation, session reuse, MTP equivalence, prefix-cache reuse
and request isolation for both models. The Qwen prefix fixture is generated
from its tokenizer; no external token-file fixture is needed. Long-context
quality, performance benchmarks and ignored experimental probes are separate
checks; this script does not certify them or all Mac configurations.

Before publishing, review a **PASS** report for the release commit from a clean
checkout, retain it with the release evidence, and list the tested hardware and
weights in the release notes. Rerun after code changes. Missing weights, a failed
test, or an untested commit means the release remains unaccepted. Tag-triggered
packaging creates a **draft** release only; producing an archive is not acceptance.
The orchestration test in CI uses a mock Cargo command and cannot replace this
hardware run. See [NATIVE_VALIDATION.md](history/NATIVE_VALIDATION.md) for prior evidence.

The oracle tools (`tools/*oracle*.cpp`, `tools/check_native*.py`) need the
pinned reference builds described in [NATIVE_VALIDATION.md](history/NATIVE_VALIDATION.md).
`examples/bench_tokens.rs` is the persistent-engine benchmark used for the
numbers above.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | GGUF reader, tokenizer, Qwen/GLM model code, MTP, Metal runtime |
| `kernels/` | Metal shaders; `kernels/NOTICE` lists adapted ggml material |
| `examples/` | benchmarks, validation and draft-vocabulary tools |
| `tests/` | integration tests (most need real weights) |
| `tools/` | Python/C++ oracle, benchmark and quality scripts |
| `validation/` | small recorded validation/performance summaries |
| `macos/`, `CMakeLists.txt`, `setup-macos.sh`, `run-macos.sh` | pinned llama.cpp reference path, comparison only |

### Reference engine (comparison only)

The CMake path builds the pinned `memoriaru/llama.cpp` fork with
`macos/llama-metal.patch` as an independent reference. Its results are not
counted as Ingot results.

```sh
./setup-macos.sh
cmake --build build-macos --target moe-mac-check moe-quality-logits
./run-macos.sh --model "$QWEN"
```
