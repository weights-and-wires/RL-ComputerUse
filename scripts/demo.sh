#!/usr/bin/env bash
# Set up and start the local demo UI: watch Qwen3-VL-4B do MiniWoB++ tasks.
#
# The first run installs everything into .demo/ (about 10 GB with the model):
#   - a Python virtual environment with the demo dependencies
#   - Playwright's Chromium
#   - MiniWoB++ at the commit training used
#   - the run-1 and run-2 LoRA adapters from Hugging Face (private repo,
#     so HF_TOKEN must be set, for example in ~/.api_keys)
# The base model (about 9 GB) downloads into the Hugging Face cache the first
# time the server loads it. Later runs skip every finished step.
#
# Usage:
#   scripts/demo.sh                  # set up if needed, then start the UI
#   scripts/demo.sh --show-browser   # also show a visible, slowed-down Chromium window
#   scripts/demo.sh --port 8001      # any other flags pass through to demo/app.py
#   scripts/demo.sh --setup-only     # install and download, don't start the server
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="$REPO/.demo"
VENV="$DEMO_DIR/venv"
MINIWOB_DIR="$DEMO_DIR/miniwob-plusplus"
MINIWOB_COMMIT="7fd85d71a4b60325c6585396ec4f48377d049838"   # same as scripts/pod_bootstrap.sh
HF_REPO="weights-and-wires/qwen3vl-4b-miniwob-grpo"
ADAPTERS="$DEMO_DIR/adapters"
RUN2_PATH="checkpoints/grpo-4b-run2/checkpoint-52"   # run 2's final adapter

SETUP_ONLY=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--setup-only" ]]; then SETUP_ONLY=1; else ARGS+=("$arg"); fi
done

step() { printf '\n==> %s\n' "$*"; }

# Load HF_TOKEN without printing it.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.api_keys" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.api_keys" >/dev/null 2>&1 || true
fi
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$DEMO_DIR"

# ---- 1. Python environment -----------------------------------------------------
if [[ ! -f "$VENV/.installed" ]]; then
  step "Create the Python environment in .demo/venv"
  if ! command -v uv >/dev/null 2>&1; then
    echo "This script needs uv. Install it from https://docs.astral.sh/uv/ and run the script again." >&2
    exit 1
  fi
  uv venv --python 3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" -r "$REPO/demo/requirements.txt"
  touch "$VENV/.installed"
fi
PY="$VENV/bin/python"

# ---- 2. Chromium for Playwright ----------------------------------------------------
if [[ ! -f "$DEMO_DIR/.chromium" ]]; then
  step "Install Playwright's Chromium"
  "$PY" -m playwright install chromium
  touch "$DEMO_DIR/.chromium"
fi

# ---- 3. MiniWoB++ pages --------------------------------------------------------------
if [[ "$(git -C "$MINIWOB_DIR" rev-parse HEAD 2>/dev/null || true)" != "$MINIWOB_COMMIT" ]]; then
  step "Get MiniWoB++ at commit ${MINIWOB_COMMIT:0:10}"
  [[ -d "$MINIWOB_DIR/.git" ]] || git clone --quiet https://github.com/Farama-Foundation/miniwob-plusplus.git "$MINIWOB_DIR"
  git -C "$MINIWOB_DIR" cat-file -e "$MINIWOB_COMMIT^{commit}" 2>/dev/null || git -C "$MINIWOB_DIR" fetch --quiet origin
  git -C "$MINIWOB_DIR" -c advice.detachedHead=false checkout --quiet "$MINIWOB_COMMIT"
fi

# ---- 4. LoRA adapters ----------------------------------------------------------------
if [[ ! -f "$ADAPTERS/run2/adapter_model.safetensors" || ! -f "$ADAPTERS/run1/adapter_model.safetensors" ]]; then
  step "Download the run-1 and run-2 adapters from Hugging Face (about 530 MB)"
  if [[ -z "$HF_TOKEN" ]]; then
    echo "The adapter repository is private. Set HF_TOKEN (for example in ~/.api_keys) and run the script again." >&2
    exit 1
  fi
  "$VENV/bin/hf" download "$HF_REPO" adapter_config.json adapter_model.safetensors \
    --local-dir "$ADAPTERS/run1" --quiet >/dev/null
  "$VENV/bin/hf" download "$HF_REPO" "$RUN2_PATH/adapter_config.json" "$RUN2_PATH/adapter_model.safetensors" \
    --local-dir "$DEMO_DIR/hf-run2" --quiet >/dev/null
  mkdir -p "$ADAPTERS/run2"
  mv "$DEMO_DIR/hf-run2/$RUN2_PATH/"adapter_* "$ADAPTERS/run2/"
  rm -rf "$DEMO_DIR/hf-run2"
fi

if [[ "$SETUP_ONLY" == 1 ]]; then
  step "Setup finished. Start the demo with scripts/demo.sh"
  exit 0
fi

# ---- 5. Start the server ---------------------------------------------------------------
step "Start the demo (the first start downloads the 9 GB base model)"
cd "$REPO"
# On Apple silicon, let PyTorch fall back to the CPU for any operation MPS lacks.
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$PY" -m demo.app \
  --miniwob-html "$MINIWOB_DIR/miniwob/html/miniwob/" \
  --adapter-run1 "$ADAPTERS/run1" \
  --adapter-run2 "$ADAPTERS/run2" \
  ${ARGS[@]+"${ARGS[@]}"}
