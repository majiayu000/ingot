#!/usr/bin/env bash
# Install ingot for the current user: build the release binary (or reuse
# one), copy it to PREFIX/bin and print the next steps. No sudo; shell rc
# files are changed only with --yes.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/install.sh [--prefix DIR] [--binary PATH] [--no-build] [--yes] [--dry-run]

  --prefix DIR    install to DIR/bin (default: ~/.local)
  --binary PATH   install this ingot binary instead of building one
  --no-build      reuse target/release/ingot; fail if it is missing
  --yes           add DIR/bin to PATH in ~/.zshrc when it is not on PATH
  --dry-run       check prerequisites and print what would be done
EOF
}

prefix="${HOME:?HOME is not set}/.local"
binary=""
build=1
yes=0
dry_run=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) prefix="${2:?--prefix requires a directory}"; shift 2 ;;
    --binary) binary="${2:?--binary requires a path}"; build=0; shift 2 ;;
    --no-build) build=0; shift ;;
    --yes) yes=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "install.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { echo "install.sh: $*" >&2; exit 1; }
run() {
  echo "+ $*"
  if [ "$dry_run" -eq 0 ]; then "$@"; fi
}

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bindir="${prefix%/}/bin"

# Prerequisites.
[ "$(uname -s)" = Darwin ] || fail "ingot runs on macOS only (found $(uname -s))"
[ "$(uname -m)" = arm64 ] || fail "ingot needs an Apple Silicon Mac (found $(uname -m)); under Rosetta, use a native arm64 terminal"
macos="$(sw_vers -productVersion)"
[ "${macos%%.*}" -ge 15 ] || fail "ingot needs macOS 15 or later (Metal residency sets); this is macOS $macos"
if [ -z "$binary" ]; then
  xcode-select -p >/dev/null 2>&1 \
    || fail "Xcode Command Line Tools are missing; run \`xcode-select --install\`, finish the installer, then rerun this script"
  if [ "$build" -eq 1 ]; then
    command -v cargo >/dev/null 2>&1 \
      || fail "cargo not found; install Rust from https://rustup.rs (1.88 or later), open a new terminal and rerun"
  fi
fi
echo "macOS $macos on $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64)"

# Binary.
if [ -z "$binary" ]; then
  binary="$repo/target/release/ingot"
  if [ "$build" -eq 1 ]; then
    run cargo build --release --locked --manifest-path "$repo/Cargo.toml" --bin ingot
  fi
  if [ "$dry_run" -eq 0 ] || [ "$build" -eq 0 ]; then
    [ -x "$binary" ] || fail "$binary not found; build it with \`cargo build --release\` or drop --no-build"
  fi
else
  [ -x "$binary" ] || fail "$binary is not an executable file"
fi

target="$bindir/ingot"
run mkdir -p "$bindir"
# Copy to a temporary name first so a running server keeps its old binary.
run cp "$binary" "$target.new"
run mv -f "$target.new" "$target"
if [ "$dry_run" -eq 0 ]; then
  "$target" --version
fi

# PATH.
case ":$PATH:" in
  *":$bindir:"*) on_path=1 ;;
  *) on_path=0 ;;
esac
if [ "$on_path" -eq 0 ]; then
  rc="$HOME/.zshrc"
  line="export PATH=\"$bindir:\$PATH\""
  if [ "$yes" -eq 1 ]; then
    if grep -qsF "$line" "$rc"; then
      echo "$rc already adds $bindir to PATH; open a new terminal"
    else
      echo "+ append to $rc: $line"
      if [ "$dry_run" -eq 0 ]; then printf '\n# ingot\n%s\n' "$line" >>"$rc"; fi
    fi
  else
    echo
    echo "$bindir is not on your PATH. Add it (or rerun with --yes to append this to $rc):"
    echo "  $line"
  fi
fi

cat <<EOF

Installed $target$( [ "$dry_run" -eq 1 ] && echo " (dry run: nothing was changed)" )

Next steps:
  ingot                      choose a model, download it and chat
  ingot models               list models and what is already downloaded
  ingot serve                API on http://127.0.0.1:8080/v1 for every local model,
                               loaded on demand; chat page at http://127.0.0.1:8080/
  ingot service install      keep that server running in the background (launchd)

Open WebUI: add an OpenAI connection with URL http://127.0.0.1:8080/v1 and any API key
  (from Docker use http://host.docker.internal:8080/v1).
Claude Code (give the server a long context, e.g. \`ingot serve --context 65536\`):
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local \\
    ANTHROPIC_MODEL=MODEL CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
  MODEL is a name from \`ingot models\`; other names use the loaded model.
EOF
