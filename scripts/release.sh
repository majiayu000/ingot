#!/usr/bin/env bash
# Build the Apple Silicon release tarball and its SHA-256 file into dist/.
# The Metal shader sources are compiled into the binary (include_str! in
# src/metal.rs), so the archive runs without the source tree.
# Optional: INGOT_SMOKE_MODEL=/path/to/primary-shard.gguf also runs
# `ingot dev tokenize` from the extracted archive (reads metadata only, no GPU).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "release.sh must run on Apple Silicon macOS" >&2
  exit 1
fi

pkgid="$(cargo pkgid)"
version="${pkgid##*[#@]}"
name=ingot
target=aarch64-apple-darwin
target_dir="${CARGO_TARGET_DIR:-$root/target}"
dist="$root/dist"
stem="$name-$version-$target"

cargo build --release --locked --bin "$name"
binary="$target_dir/release/$name"
if [[ "$(lipo -archs "$binary")" != arm64 ]]; then
  echo "unexpected architecture: $(lipo -archs "$binary")" >&2
  exit 1
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/$stem"
cp "$binary" "$stage/$stem/"
cp README.md LICENSE CHANGELOG.md "$stage/$stem/"
cp kernels/NOTICE "$stage/$stem/NOTICE"

mkdir -p "$dist"
archive="$dist/$stem.tar.gz"
tar -C "$stage" -czf "$archive" "$stem"
(cd "$dist" && shasum -a 256 "$stem.tar.gz" > "$stem.tar.gz.sha256")

# Verify the archive from an unrelated directory.
check="$stage/check"
mkdir "$check"
(cd "$dist" && shasum -a 256 -c "$stem.tar.gz.sha256")
tar -C "$check" -xzf "$archive"
(cd "$check" && "./$stem/$name" --help > /dev/null)
if [[ -n "${INGOT_SMOKE_MODEL:-}" ]]; then
  (cd "$check" && "./$stem/$name" dev tokenize "$INGOT_SMOKE_MODEL" "hello world")
fi

echo "$archive"
cat "$archive.sha256"
