#!/usr/bin/env bash
# Set up the pod. Runs ON the pod, and is safe to run again (idempotent).
#
# What it does: system packages, uv, a Python 3.12 venv on the network volume,
# the pinned Python dependencies, Playwright Chromium, the MiniWoB++ clone,
# /workspace/env.sh, a model weight prefetch, and two smoke tests.
# Everything is logged to /workspace/logs/bootstrap.log.
#
# Usage: bash /workspace/RL-ComputerUse/scripts/pod_bootstrap.sh [--with-2b] [--torch-backend B]
#                                                                [--skip-prefetch] [--skip-smoke]
set -Eeuo pipefail
# Never trace: tracing could print token values from /workspace/.secrets.
{ set +x; } 2>/dev/null

WORKSPACE=/workspace
# Run 3 keeps the repository on the container disk (/root/RL-ComputerUse) and
# sets CUA_REPO; runs 1 and 2 used the default.
REPO="${CUA_REPO:-$WORKSPACE/RL-ComputerUse}"
# The venv, uv cache, and model weights live on the pod's local container disk.
# /workspace is a network filesystem (a network volume, or the pod volume disk in
# datacenters such as US-KS-2), and writing about 40 GB of small files there took
# 22 minutes in testing. The container disk is lost when the pod stops, so run
# this script again after a restart (about 5-8 minutes). Results, checkpoints,
# logs, and the MiniWoB clone stay on /workspace.
LOCAL_ROOT="${CUA_LOCAL_ROOT:-/root/cua}"
VENV="$LOCAL_ROOT/venv"
UV_HOME="$LOCAL_ROOT/uv"
HF_HOME_DIR="$LOCAL_ROOT/hf"
mkdir -p "$LOCAL_ROOT"
# Browsers load every MiniWoB page from here. On a network drive they start
# slowly, so run 3 sets CUA_MINIWOB_DIR=/root/miniwob-plusplus.
MINIWOB_DIR="${CUA_MINIWOB_DIR:-$WORKSPACE/miniwob-plusplus}"
MINIWOB_COMMIT="7fd85d71a4b60325c6585396ec4f48377d049838"
LOG="$WORKSPACE/logs/bootstrap.log"

WITH_2B=0
TORCH_BACKEND="auto"
SKIP_PREFETCH=0
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-2b)         WITH_2B=1; shift ;;
    --torch-backend)   TORCH_BACKEND="${2:?--torch-backend needs a value}"; shift 2 ;;
    --skip-prefetch)   SKIP_PREFETCH=1; shift ;;
    --skip-smoke)      SKIP_SMOKE=1; shift ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      echo "Options: --with-2b (also fetch Qwen3-VL-2B), --torch-backend B (auto, cu128, cu129),"
      echo "         --skip-prefetch, --skip-smoke"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$WORKSPACE/logs"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf 'WARNING: %s\n' "$*"; }
SMOKE_FAILURES=()

[[ -d "$REPO" ]] || { echo "ERROR: $REPO not found. From your Mac, run: scripts/sync_results.sh --push-code"; exit 1; }
[[ -f "$REPO/pyproject.toml" ]] || { echo "ERROR: $REPO/pyproject.toml missing. Push the code again."; exit 1; }

# ---- 1. GPU and driver ----------------------------------------------------------
step "GPU and driver"
nvidia-smi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
DRIVER_MAJOR="${DRIVER%%.*}"
echo "Driver version: $DRIVER"
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ && "$DRIVER_MAJOR" -lt 580 ]]; then
  warn "Driver $DRIVER is older than R580. PyPI's vllm 0.33.0 may pull CUDA 13 wheels that need R580+."
  if [[ "$TORCH_BACKEND" == "auto" ]]; then
    if [[ "$DRIVER_MAJOR" -ge 575 ]]; then TORCH_BACKEND="cu129"; else TORCH_BACKEND="cu128"; fi
    warn "Using --torch-backend=$TORCH_BACKEND. Pass --torch-backend to override."
  fi
fi
echo "Torch backend: $TORCH_BACKEND"

# ---- 2. System packages ---------------------------------------------------------
step "apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git rsync jq tmux htop curl ca-certificates build-essential \
  fonts-dejavu-core ffmpeg

# ---- 3. uv ----------------------------------------------------------------------
step "uv"
export PATH="$UV_HOME/bin:$PATH"
export UV_CACHE_DIR="$UV_HOME/cache"
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"
# The cache and the venv sit on different filesystems from the container disk,
# and hardlinks can fail on a network volume. Copy instead.
export UV_LINK_MODE=copy
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_HOME/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# ---- 4. venv on the network volume ----------------------------------------------
step "Python 3.12 venv at $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12
else
  echo "venv exists. Reusing it."
fi
PY="$VENV/bin/python"

# ---- 5. Python dependencies -----------------------------------------------------
step "Install pinned dependencies (editable) with --torch-backend=$TORCH_BACKEND"
# One call, so the resolver sees every pin together. The pins live in pyproject.toml.
# --group dev adds pytest and ruff. vLLM brings its own torch.
( cd "$REPO" && uv pip install --python "$PY" --torch-backend "$TORCH_BACKEND" -e . --group dev )
# The cache roughly doubles disk use on the container disk and is lost on pod stop
# anyway, so drop it once the venv exists.
uv cache clean >/dev/null 2>&1 || true
uv pip check --python "$PY" || warn "uv pip check reported conflicts. Read the lines above before you train."
"$PY" -c "import torch, vllm, transformers, trl, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__)"

step "vLLM kernels"
if "$VENV/bin/vllm" download-kernels --help >/dev/null 2>&1; then
  "$VENV/bin/vllm" download-kernels || warn "vllm download-kernels failed. Continuing."
else
  echo "vllm download-kernels isn't available. Skipping."
fi

# ---- 6. Playwright ----------------------------------------------------------------
step "Playwright Chromium"
# Playwright 1.44 (pinned by browsergym-core) predates Ubuntu 24.04, so its
# --with-deps step asks for packages that 24.04 renamed (libasound2 became
# libasound2t64) and fails. Install Chromium's runtime libraries with apt
# instead, then download only the browser.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  libasound2t64 libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 libcups2t64 \
  libdbus-1-3 libdrm2 libgbm1 libglib2.0-0t64 libnspr4 libnss3 libpango-1.0-0 \
  libcairo2 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 \
  libxkbcommon0 libxrandr2 libxshmfence1 fonts-liberation fonts-noto-color-emoji \
  >/dev/null
"$VENV/bin/playwright" install chromium

# ---- 7. MiniWoB++ -------------------------------------------------------------------
step "MiniWoB++ at commit ${MINIWOB_COMMIT:0:10}"
if [[ ! -d "$MINIWOB_DIR/.git" ]]; then
  git clone https://github.com/Farama-Foundation/miniwob-plusplus.git "$MINIWOB_DIR"
fi
git -C "$MINIWOB_DIR" cat-file -e "$MINIWOB_COMMIT^{commit}" 2>/dev/null || git -C "$MINIWOB_DIR" fetch origin
git -C "$MINIWOB_DIR" reset --hard "$MINIWOB_COMMIT"
MINIWOB_HTML="$MINIWOB_DIR/miniwob/html/miniwob/"
[[ -d "$MINIWOB_HTML" ]] || { echo "ERROR: $MINIWOB_HTML missing after checkout."; exit 1; }

# ---- 8. env.sh ----------------------------------------------------------------------
step "Write $WORKSPACE/env.sh"
cat > "$WORKSPACE/env.sh" <<EOF
# Generated by scripts/pod_bootstrap.sh. Source it in every shell: source /workspace/env.sh
# MINIWOB_URL needs the trailing slash: BrowserGym concatenates it with the task name.
export MINIWOB_URL="file://${MINIWOB_HTML}"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="$UV_HOME/cache"
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"
export UV_LINK_MODE=copy
export PATH="$UV_HOME/bin:\$PATH"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# HF_TOKEN and WANDB_API_KEY. runpod_up.sh writes this file (mode 600).
if [ -f "$WORKSPACE/.secrets" ]; then
  . "$WORKSPACE/.secrets"
fi
EOF
touch "$WORKSPACE/.secrets" 2>/dev/null || true
chmod 600 "$WORKSPACE/.secrets" 2>/dev/null || true

# Load env.sh in new login and tmux shells. The guard keeps this idempotent.
grep -qF "source $WORKSPACE/env.sh" "$HOME/.bashrc" 2>/dev/null \
  || echo "[ -f $WORKSPACE/env.sh ] && source $WORKSPACE/env.sh" >> "$HOME/.bashrc"

set +u
# shellcheck disable=SC1091
source "$WORKSPACE/env.sh"
set -u
echo "MINIWOB_URL=$MINIWOB_URL"
# Report presence only. Never print the values.
[[ -n "${HF_TOKEN:-}" ]]       && echo "HF_TOKEN: set"       || warn "HF_TOKEN isn't set. Downloads can be rate-limited."
[[ -n "${WANDB_API_KEY:-}" ]]  && echo "WANDB_API_KEY: set"  || warn "WANDB_API_KEY isn't set. Run scripts/runpod_up.sh again, or write /workspace/.secrets by hand."

# ---- 9. Model weights ---------------------------------------------------------------
if [[ "$SKIP_PREFETCH" -eq 0 ]]; then
  step "Prefetch model weights into $HF_HOME"
  if command -v hf >/dev/null 2>&1; then HF_CLI=(hf download)
  else HF_CLI=(huggingface-cli download); fi
  "${HF_CLI[@]}" Qwen/Qwen3-VL-4B-Instruct
  if [[ "$WITH_2B" -eq 1 ]]; then
    "${HF_CLI[@]}" Qwen/Qwen3-VL-2B-Instruct
  fi
else
  echo "Skipping the weight prefetch."
fi

# ---- 10. Smoke tests ----------------------------------------------------------------
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  step "Smoke test: pytest -q tests"
  ( cd "$REPO" && "$PY" -m pytest -q tests ) || SMOKE_FAILURES+=("pytest")

  step "Smoke test: python -m cua.env --task click-button --seed 0"
  ( cd "$REPO" && "$PY" -m cua.env --task click-button --seed 0 ) || SMOKE_FAILURES+=("cua.env")
fi

step "Summary"
if [[ ${#SMOKE_FAILURES[@]} -gt 0 ]]; then
  echo "FAILED: ${SMOKE_FAILURES[*]}"
  echo "See $LOG. Common fixes are in docs/RUNBOOK.md under 'Smoke test fails'."
  exit 1
fi
echo "Bootstrap finished. Next: tmux new -s run, then: source /workspace/env.sh && cd $REPO && bash scripts/saturday.sh"
