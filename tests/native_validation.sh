#!/usr/bin/env bash
# Verify orchestration only; this is not GPU/model acceptance evidence.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/bin"
cat > "$scratch/bin/cargo" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$NATIVE_TEST_CALLS"
if [[ "${NATIVE_TEST_FAIL:-}" == yes && "$1" == test ]]; then
  echo 'simulated GPU test failure' >&2
  exit 19
fi
echo 'mock cargo (orchestration test only)'
MOCK
chmod +x "$scratch/bin/cargo"
export PATH="$scratch/bin:$PATH"
export NATIVE_TEST_CALLS="$scratch/calls"
touch "$scratch/main.gguf" "$scratch/draft.gguf"
export QWEN_MODEL="$scratch/main.gguf" QWEN_DRAFT="$scratch/draft.gguf"
export GLM_MODEL="$scratch/main.gguf" GLM_DRAFT="$scratch/draft.gguf"
unset DRAFT_VOCAB

if QWEN_MODEL='' bash "$root/scripts/validate-native.sh" "$scratch/missing.log" > "$scratch/missing.out" 2>&1; then
  echo 'missing weights incorrectly passed' >&2
  exit 1
fi
[[ ! -e "$NATIVE_TEST_CALLS" && ! -e "$scratch/missing.log" ]]

set +e
NATIVE_TEST_FAIL=yes bash "$root/scripts/validate-native.sh" "$scratch/failed.log" > "$scratch/failed.out" 2>&1
status=$?
set -e
[[ "$status" == 19 ]]
! grep -q '^PASS:' "$scratch/failed.log"
! grep -q -- '--test mtp' "$NATIVE_TEST_CALLS"

: > "$NATIVE_TEST_CALLS"
bash "$root/scripts/validate-native.sh" "$scratch/passed.log" > "$scratch/passed.out" 2>&1
grep -q '^PASS:' "$scratch/passed.log"
grep -q -- '--test-threads=1' "$NATIVE_TEST_CALLS"
grep -q -- '--test tokenizer --test projections' "$NATIVE_TEST_CALLS"
grep -q -- 'state_cancellation_memory_and_reuse -- --ignored --exact' "$NATIVE_TEST_CALLS"
grep -q -- '--test glm --test mtp --test glm_mtp' "$NATIVE_TEST_CALLS"

# Invocation from another directory must not change the meaning of model paths.
(
  cd "$scratch"
  QWEN_MODEL=main.gguf QWEN_DRAFT=draft.gguf GLM_MODEL=main.gguf GLM_DRAFT=draft.gguf \
    bash "$root/scripts/validate-native.sh" "$scratch/relative.log" > "$scratch/relative.out" 2>&1
)
grep -Fq "QWEN_MODEL=$scratch/main.gguf" "$scratch/relative.log"
grep -Fq "GLM_DRAFT=$scratch/draft.gguf" "$scratch/relative.log"
echo 'Native validation orchestration: missing-input, failure, success and relative-path cases passed.'
