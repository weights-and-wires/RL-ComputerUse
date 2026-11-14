#!/usr/bin/env bash
# Copy files between this Mac and the pod with rsync over SSH.
# Reads the SSH host and port from .runpod/state.env (written by runpod_up.sh).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_runpod.sh"

usage() {
  cat <<'EOF'
Usage: scripts/sync_results.sh [--adapters] [--push-code] [--dry-run]

With no flags, copy the pod's /workspace/RL-ComputerUse/results/ to ./results/.

Options:
  --adapters    Also copy every final adapter, /workspace/ckpt/*/final, to
                ./results/adapters/NAME/.
  --push-code   Instead of pulling, copy this repository to the pod at
                /workspace/RL-ComputerUse. Excludes results/, .runpod/, .venv,
                __pycache__, wandb/, and .env. Installs rsync on the pod first
                if it's missing.
  --dry-run     Show what rsync would transfer without changing any files.
  -h, --help    Show this help.

The script copies nothing that contains secrets. The pod's /workspace/.secrets
sits outside the synced folders.
EOF
}

PULL_ADAPTERS=0
PUSH_CODE=0
DRY_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapters)  PULL_ADAPTERS=1; shift ;;
    --push-code) PUSH_CODE=1; shift ;;
    --dry-run)   DRY_FLAG="1"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) usage >&2; die "Unknown option: $1" ;;
  esac
done

require_cmd rsync "It ships with macOS."
load_state
need_ssh_state
RSH="$(rsync_ssh_cmd)"
REMOTE="root@$SSH_HOST"

if [[ "$PUSH_CODE" -eq 1 ]]; then
  info "Pushing code to $REMOTE:$REMOTE_REPO ..."
  rp_ssh "mkdir -p $REMOTE_REPO && (command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync))"
  rsync -az --no-owner --no-group --partial --delete ${DRY_FLAG:+-n} -e "$RSH" \
    --exclude 'results/' --exclude '.runpod/' --exclude '.venv/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.egg-info' \
    --exclude 'wandb/' --exclude '.env' --exclude '.git/' \
    --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
    "$REPO_ROOT/" "$REMOTE:$REMOTE_REPO/"
  info "Code pushed."
  exit 0
fi

mkdir -p "$REPO_ROOT/results"
info "Pulling results from the pod..."
rsync -az --partial ${DRY_FLAG:+-n} -e "$RSH" "$REMOTE:$REMOTE_REPO/results/" "$REPO_ROOT/results/"

if [[ "$PULL_ADAPTERS" -eq 1 ]]; then
  info "Pulling final adapters..."
  # List adapter folders over SSH instead of relying on remote glob expansion.
  names="$(rp_ssh 'for d in /workspace/ckpt/*/final; do [ -d "$d" ] && basename "$(dirname "$d")"; done' || true)"
  if [[ -z "$names" ]]; then
    info "No /workspace/ckpt/*/final folders found."
  else
    while IFS= read -r name; do
      [[ -n "$name" ]] || continue
      mkdir -p "$REPO_ROOT/results/adapters/$name"
      rsync -az --partial ${DRY_FLAG:+-n} -e "$RSH" \
        "$REMOTE:/workspace/ckpt/$name/final/" "$REPO_ROOT/results/adapters/$name/"
      info "  adapter: $name"
    done <<<"$names"
  fi
fi

info "Done. Results are in $REPO_ROOT/results/."
