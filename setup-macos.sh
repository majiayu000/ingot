#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This project requires native Apple Silicon macOS." >&2
  exit 1
fi
cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release
cmake --build build-macos --target llama-server llama-bench llama-perplexity test-expert-pool -j 8
cmake --build build-macos --target moe-mac-check
echo "Built and verified. Start with: ./run-macos.sh --model /path/to/first-shard.gguf"
