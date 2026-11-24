#!/usr/bin/env bash
# Start run 3 on the pod, or report that it already runs. Safe to call again.
#
# The monitor calls it over SSH:
#   ssh root@<pod> 'cd /root/RL-ComputerUse && bash scripts/run3_pod_start.sh'
#
# What it does, in order:
#   1. If tmux session run3 is alive, prints ALREADY_RUNNING and exits 0
#      (it also restarts the sync loop if that one died).
#   2. Bootstraps the pod (scripts/pod_bootstrap.sh) unless it already is:
#      /workspace/.bootstrapped exists and the venv on /root is still there.
#   3. Restores run-2 inputs, results/run3/, and the newest synced checkpoint
#      from Hugging Face where they're missing (python -m cua.hfsync restore).
#   4. Starts the sync loop in tmux session run3-sync.
#   5. Starts scripts/run3.sh in tmux session run3, logging to results/run3/run.log.
#   6. Prints STARTED and exits 0. Any failure exits non-zero with a message.
#
# These variables pass through to run3.sh when set: GRPO_MAX_HOURS, RFT,
# PRICE_PER_HR, BEST_CKPT, CUA_SETS, RUN2_ADAPTER, RUN2_HF_PATH, FORCE_EVAL,
# SYNC_INTERVAL_S.
set -uo pipefail
{ set +x; } 2>/dev/null   # never trace: the environment holds tokens

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || exit 1

RES="results/run3"
LOG_DIR=/workspace/logs
LOCAL_ROOT="${CUA_LOCAL_ROOT:-/root/cua}"
MINIWOB_DIR=/root/miniwob-plusplus   # container disk: browsers must not load pages from a network drive
mkdir -p "$LOG_DIR"

fail() { echo "FAILED: $*"; exit 1; }
have_tmux() { command -v tmux >/dev/null 2>&1; }

PASS_VARS=(GRPO_MAX_HOURS RFT PRICE_PER_HR BEST_CKPT CUA_SETS RUN2_ADAPTER RUN2_HF_PATH FORCE_EVAL SYNC_INTERVAL_S)
env_prefix() {
  local v out="env"
  for v in "${PASS_VARS[@]}"; do
    [[ -n "${!v:-}" ]] && out+=" $v=$(printf '%q' "${!v}")"
  done
  echo "$out"
}

start_sync() {
  if ! tmux has-session -t run3-sync 2>/dev/null; then
    tmux new-session -d -s run3-sync \
      "cd $(printf '%q' "$REPO") && source /workspace/env.sh && mkdir -p $RES/logs && $(env_prefix) python -m cua.hfsync loop >> $RES/logs/sync.log 2>&1" \
      || return 1
  fi
}

# ---- 1. Already running? -------------------------------------------------------------
if have_tmux && tmux has-session -t run3 2>/dev/null; then
  if [[ -f /workspace/env.sh ]]; then start_sync || echo "WARNING: couldn't restart the sync loop."; fi
  echo "ALREADY_RUNNING"
  exit 0
fi

# ---- 2. Secrets and bootstrap ----------------------------------------------------------
if [[ -f /workspace/.secrets ]]; then
  set +u
  # shellcheck disable=SC1091
  source /workspace/.secrets
  set -u
fi
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN isn't set in /workspace/.secrets."
[[ -n "${HF_ADAPTER_REPO:-}" ]] || fail "HF_ADAPTER_REPO isn't set in /workspace/.secrets."

# A stopped and restarted pod keeps /workspace but loses /root, so check the venv too.
if [[ ! -f /workspace/.bootstrapped || ! -x "$LOCAL_ROOT/venv/bin/python" || ! -f /workspace/env.sh ]]; then
  echo "Bootstrapping the pod (about 10 minutes). Log: $LOG_DIR/bootstrap.log"
  rm -f /workspace/.bootstrapped
  if ! CUA_REPO="$REPO" CUA_MINIWOB_DIR="$MINIWOB_DIR" bash scripts/pod_bootstrap.sh >"$LOG_DIR/run3_bootstrap.out" 2>&1; then
    tail -n 30 "$LOG_DIR/run3_bootstrap.out"
    fail "pod_bootstrap.sh failed. See $LOG_DIR/bootstrap.log."
  fi
  date -u +%FT%TZ > /workspace/.bootstrapped
fi

set +u
# shellcheck disable=SC1091
source /workspace/env.sh
set -u
have_tmux || fail "tmux isn't installed, even after the bootstrap."

# ---- 3. Restore from Hugging Face --------------------------------------------------------
echo "Restoring from Hugging Face where needed. Log: $LOG_DIR/run3_restore.log"
if ! python -m cua.hfsync restore >>"$LOG_DIR/run3_restore.log" 2>&1; then
  tail -n 30 "$LOG_DIR/run3_restore.log"
  fail "restore failed."
fi
mkdir -p "$RES/logs"

# ---- 4. Sync loop ---------------------------------------------------------------------
start_sync || fail "couldn't start tmux session run3-sync."

# ---- 5. The pipeline ------------------------------------------------------------------
tmux new-session -d -s run3 \
  "cd $(printf '%q' "$REPO") && source /workspace/env.sh && $(env_prefix) bash scripts/run3.sh 2>&1 | tee -a $RES/run.log" \
  || fail "couldn't start tmux session run3."
sleep 3
if ! tmux has-session -t run3 2>/dev/null; then
  tail -n 20 "$RES/run.log" 2>/dev/null
  fail "tmux session run3 ended at once. See $RES/run.log."
fi
echo "STARTED"
exit 0
