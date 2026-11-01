#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x build-macos/bin/llama-server ]]; then
  echo "Run ./setup-macos.sh first." >&2
  exit 1
fi
export STRATA_SSD_EXPERTS=1
exec build-macos/bin/llama-server --host 127.0.0.1 --port 8080 --gpu-layers 99 \
  --load-mode mmap --lazy-mode on --moe-expert-cache 64 --ctx-size 8192 \
  --batch-size 1024 --ubatch-size 1024 --parallel 1 --jinja "$@"
