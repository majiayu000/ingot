#!/usr/bin/env bash
# Release acceptance on a real Apple Silicon Mac with existing model weights.
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: scripts/validate-native.sh [LOG_FILE]

Run the host, GPU and real-model correctness regressions, sequentially.
Requires an Apple Silicon Mac and four existing files:
  QWEN_MODEL, QWEN_DRAFT, GLM_MODEL, GLM_DRAFT
Use the first shard for each main model. No weights are downloaded.
MTP tests default to a 44 GiB Qwen / 32 GiB GLM budget; use a Mac with
enough free unified memory. See README.md for scope and release review.
Logs default to evidence/native-<UTC timestamp>-<process ID>.log.
HELP
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo 'Usage: scripts/validate-native.sh [LOG_FILE]' >&2
  exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Native acceptance requires an Apple Silicon Mac.' >&2
  exit 2
fi
for name in QWEN_MODEL QWEN_DRAFT GLM_MODEL GLM_DRAFT DRAFT_VOCAB; do
  if [[ "$name" == DRAFT_VOCAB && -z "${!name:-}" ]]; then
    continue
  fi
  if [[ -z "${!name:-}" || ! -f "${!name}" || ! -r "${!name}" ]]; then
    echo "Set $name to an existing, readable model file; see scripts/validate-native.sh --help." >&2
    exit 2
  fi
  # Cargo runs test binaries from the package root. Preserve caller-relative paths.
  absolute="$(cd "$(dirname "${!name}")" && pwd)/$(basename "${!name}")"
  printf -v "$name" '%s' "$absolute"
  export "$name"
done

root="$(cd "$(dirname "$0")/.." && pwd)"
log="${1:-$root/evidence/native-$(date -u +%Y%m%dT%H%M%SZ)-$$.log}"
mkdir -p "$(dirname "$log")"
manifest="$root/Cargo.toml"

validate() {
  date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
  git -C "$root" log -1 --format='Commit: %H'
  echo 'Working tree changes (empty means clean):'
  git -C "$root" status --short
  sw_vers
  sysctl hw.model hw.memsize machdep.cpu.brand_string
  rustc --version
  cargo --version
  for name in QWEN_MODEL QWEN_DRAFT GLM_MODEL GLM_DRAFT; do
    printf '%s=%s\n' "$name" "${!name}"
  done
  printf 'QWEN_MEMORY_GIB=%s GLM_MEMORY_GIB=%s DRAFT_TOKENS=%s DRAFT_VOCAB=%s\n' \
    "${QWEN_MEMORY_GIB:-44}" "${GLM_MEMORY_GIB:-32}" "${DRAFT_TOKENS:-3}" "${DRAFT_VOCAB:-full vocabulary}"

  cargo fmt --manifest-path "$manifest" --all --check
  cargo clippy --manifest-path "$manifest" --release --locked --lib --bin ingot \
    --example bench_tokens --test cli -- -D warnings
  # Hardware failures are fatal here, unlike the informational hosted CI job.
  cargo test --manifest-path "$manifest" --release --locked --lib -- --test-threads=1
  cargo test --manifest-path "$manifest" --release --locked --bin ingot --test cli
  cargo test --manifest-path "$manifest" --release --locked --test tokenizer --test projections \
    -- --ignored --nocapture --test-threads=1
  cargo test --manifest-path "$manifest" --release --locked --test qwen \
    state_cancellation_memory_and_reuse -- --ignored --exact --nocapture --test-threads=1
  cargo test --manifest-path "$manifest" --release --locked --test glm --test mtp --test glm_mtp \
    -- --ignored --nocapture --test-threads=1
  echo 'PASS: native correctness regression completed. Review this evidence before publishing.'
}

echo "Native acceptance log: $log"
validate 2>&1 | tee "$log"
