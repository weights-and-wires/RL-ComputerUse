#!/usr/bin/env bash
# Run the training-day pipeline in resumable phases. Runs ON the pod, inside tmux.
#
# Each phase writes a marker file results/.phases/<phase>.done when it succeeds
# and logs to results/logs/<phase>.log. A normal run skips phases that have a
# marker, so after a crash or a pod swap you run the same command again.
set -uo pipefail
{ set +x; } 2>/dev/null   # never trace: the environment can hold tokens

PHASES=(smoke baseline select base_eval rft grpo final_eval report push)

usage() {
  cat <<'EOF'
Usage: bash scripts/saturday.sh [--from PHASE | --only PHASE | --redo PHASE | --status] [-- EXTRA_ARGS]

Phases, in order:
  smoke       Environment smoke test, plus a 2-seed greedy eval on click-button.
  baseline    Sampled baseline on every candidate task (run name: baseline).
  select      Pick the training tasks from the baseline (results/selected_tasks.json).
  base_eval   Greedy eval of the base model on the selected and held-out tasks
              (base-train, base-heldout).
  rft         Rejection-sampling warm start, unless the skip rule applies.
              Evals: rft-train, rft-heldout. Records results/.phases/init_adapter.
  grpo        GRPO training. Resumes from checkpoints automatically.
  final_eval  Greedy eval of the final adapter (grpo-train, grpo-heldout).
  report      Build results/report/index.html.
  push        Upload the final adapter to Hugging Face if HF_ADAPTER_REPO is set.

Options:
  --from PHASE    Run PHASE and every later phase, even when markers exist.
  --only PHASE    Run only PHASE, even when its marker exists.
  --redo PHASE    Delete the marker for PHASE, then run as usual (PHASE and any
                  phase without a marker).
  --status        Show which phases are done and exit.
  -- ARGS         Extra arguments appended to the trainer command in the grpo
                  phase, for example: -- --set grpo.learning_rate=5e-6

Environment variables:
  MODEL=4b|2b         Model size. Default: 4b. 2b adds the 2B model name and
                      separate adapter folders (rft-2b, grpo-2b-trackA).
  TRAINER=nano|trl    GRPO implementation. Default: the `trainer:` key in the config.
  GRPO_CONFIG=PATH    Override the GRPO config (for example configs/grpo_trackB.yaml).
  CUA_SETS="--set K=V ..."  Extra --set arguments added to every eval, rft, and grpo command.
                      Use the same value for base evals and trained evals so they stay comparable.
  GRPO_MAX_HOURS=5    Stop the grpo phase after this many hours. Checkpoints stay.
  BEST_CKPT=PATH      Adapter that final_eval evaluates. Default: the final adapter,
                      or the newest checkpoint when no final adapter exists.
  PRICE_PER_HR=2.99   Hourly price used for the spend estimate.
  SELECT_MIN, SELECT_MAX   Override the success-rate band in the select phase.
  HF_ADAPTER_REPO     Hugging Face repo for the push phase. Default:
                      weights-and-wires/qwen3vl-4b-miniwob-grpo (private). Set it to "" to skip.
  HF_TOKEN            Needed by the push phase. Read from the environment, never printed.
EOF
}

# ---- paths and settings -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || exit 1

if [[ -f /workspace/env.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /workspace/env.sh
  set -u
fi

RES="results"
PH="$RES/.phases"
LOGS="$RES/logs"
mkdir -p "$PH" "$LOGS"

MODEL="${MODEL:-4b}"
PRICE_PER_HR="${PRICE_PER_HR:-2.99}"
GRPO_MAX_HOURS="${GRPO_MAX_HOURS:-5}"
# The push phase uploads the final adapter here (created private). Set
# HF_ADAPTER_REPO="" to skip the upload.
HF_ADAPTER_REPO="${HF_ADAPTER_REPO-weights-and-wires/qwen3vl-4b-miniwob-grpo}"

case "$MODEL" in
  4b) GRPO_CONFIG_DEFAULT="configs/grpo_qwen3vl4b.yaml"; MODEL_SET=(); RFT_SET=()
      RFT_DIR="/workspace/ckpt/rft-4b" ;;
  2b) GRPO_CONFIG_DEFAULT="configs/grpo_qwen3vl2b.yaml"
      MODEL_SET=(--set model.name=Qwen/Qwen3-VL-2B-Instruct)
      RFT_DIR="/workspace/ckpt/rft-2b"
      RFT_SET=(--set run_name=rft-2b --set "sft.output_dir=$RFT_DIR") ;;
  *)  echo "MODEL must be 4b or 2b." >&2; exit 1 ;;
esac
GRPO_CONFIG="${GRPO_CONFIG:-$GRPO_CONFIG_DEFAULT}"

# CUA_SETS holds extra --set arguments for every cua command (eval, rft, grpo), for example
# CUA_SETS="--set env.first_obs_image=false --set env.screenshot_long_side=384".
GLOBAL_SET=()
if [[ -n "${CUA_SETS:-}" ]]; then read -r -a GLOBAL_SET <<<"$CUA_SETS"; fi

# The trainer comes from the config unless TRAINER overrides it.
if [[ -n "${TRAINER:-}" ]]; then
  TRAINER_SET=(--set "trainer=$TRAINER")
  TRAINER_EFFECTIVE="$TRAINER"
else
  TRAINER_SET=()
  TRAINER_EFFECTIVE="$(sed -n 's/^trainer:[[:space:]]*\([a-z]*\).*/\1/p' "$GRPO_CONFIG" | head -n1)"
  TRAINER_EFFECTIVE="${TRAINER_EFFECTIVE:-nano}"
fi
[[ "$TRAINER_EFFECTIVE" == "nano" || "$TRAINER_EFFECTIVE" == "trl" ]] \
  || { echo "Unknown trainer '$TRAINER_EFFECTIVE'. Use nano or trl." >&2; exit 1; }

# The GRPO output folder, read from the `grpo:` block of the config.
GRPO_OUT="$(awk '/^grpo:/{g=1; next} /^[^ #]/{g=0} g && /^  output_dir:/{print $2; exit}' "$GRPO_CONFIG")"

# ---- argument parsing -----------------------------------------------------------------
MODE="all"; TARGET=""; EXTRA_ARGS=()
valid_phase() { local p; for p in "${PHASES[@]}"; do [[ "$p" == "$1" ]] && return 0; done; return 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from|--only|--redo)
      MODE="${1#--}"; TARGET="${2:-}"
      valid_phase "$TARGET" || { echo "Unknown phase '$TARGET'. Phases: ${PHASES[*]}" >&2; exit 1; }
      shift 2 ;;
    --status) MODE="status"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) usage >&2; echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---- helpers --------------------------------------------------------------------------
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# run CMD...: echo the command, then run it. Phase functions run with `set -e`.
run() { say "\$ $*"; "$@"; }

estimate_spend() {
  # Total phase seconds recorded so far, converted to dollars.
  local secs=0
  [[ -f "$PH/.elapsed_seconds" ]] && secs="$(cat "$PH/.elapsed_seconds")"
  awk -v s="$secs" -v p="$PRICE_PER_HR" 'BEGIN { printf "%.1f h, about $%.2f", s/3600, s/3600*p }'
}

add_elapsed() {
  local secs=0
  [[ -f "$PH/.elapsed_seconds" ]] && secs="$(cat "$PH/.elapsed_seconds")"
  echo $((secs + $1)) > "$PH/.elapsed_seconds"
}

# eval_cmd RUN_NAME MODE TASKS [extra --set args...]
eval_cmd() {
  local name="$1" mode="$2" tasks="$3"; shift 3
  run python -m cua.eval --config configs/eval.yaml \
    --set "eval.mode=$mode" --set "eval.tasks=$tasks" --set "run_name=$name" \
    "${MODEL_SET[@]}" "${GLOBAL_SET[@]}" "$@"
}

require_file() { [[ -s "$1" ]] || { echo "Expected file missing or empty: $1"; return 1; }; }

# ---- phases ----------------------------------------------------------------------------
# A phase that skips its work and should stay "not done" touches $PH/.no_marker.

phase_smoke() {
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  [[ -n "${HF_TOKEN:-}" ]]      && echo "HF_TOKEN: set"      || echo "WARNING: HF_TOKEN isn't set."
  [[ -n "${WANDB_API_KEY:-}" ]] && echo "WANDB_API_KEY: set" || echo "WARNING: WANDB_API_KEY isn't set."
  run python -m cua.env --task click-button --seed 0
  # A 2-seed greedy eval exercises vLLM, the prompt format, and the env pool.
  # It relies on --set values overriding the mode preset (num_seeds=50).
  eval_cmd smoke greedy click-button --set eval.num_seeds=2 --set eval.gif_seeds=0
}

phase_baseline() {
  eval_cmd baseline baseline candidates
  require_file "$RES/eval/baseline/per_task.csv"
}

phase_select() {
  local extra=()
  [[ -n "${SELECT_MIN:-}" ]] && extra+=(--set "select.min_success=$SELECT_MIN")
  [[ -n "${SELECT_MAX:-}" ]] && extra+=(--set "select.max_success=$SELECT_MAX")
  run python -m cua.select_tasks --baseline "$RES/eval/baseline/per_task.csv" "${extra[@]}"
  require_file "$RES/selected_tasks.json"
  local n; n="$(jq '.selected | length' "$RES/selected_tasks.json")"
  echo "Selected $n tasks:"
  jq -r '.selected | join(", ")' "$RES/selected_tasks.json"
  if [[ "$n" -lt 8 ]]; then
    echo "Only $n tasks qualify (fewer than 8). Widen the band, then rerun this phase:"
    echo "  SELECT_MIN=0.05 SELECT_MAX=0.90 bash scripts/saturday.sh --only select"
    return 1
  fi
}

phase_base_eval() {
  eval_cmd base-train greedy selected
  eval_cmd base-heldout greedy heldout
}

phase_rft() {
  local rc=0
  python -m cua.rft should_skip --config configs/rft.yaml "${MODEL_SET[@]}" "${RFT_SET[@]}" "${GLOBAL_SET[@]}" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    echo "RFT skip rule met (format errors low and success high). No warm start."
    rm -f "$PH/init_adapter"
    return 0
  elif [[ "$rc" -ne 1 ]]; then
    echo "cua.rft should_skip failed with exit code $rc."
    return 1
  fi
  run python -m cua.rft collect --config configs/rft.yaml "${MODEL_SET[@]}" "${RFT_SET[@]}" "${GLOBAL_SET[@]}"
  run python -m cua.rft train --config configs/rft.yaml "${MODEL_SET[@]}" "${RFT_SET[@]}" "${GLOBAL_SET[@]}"
  require_file "$RFT_DIR/final/adapter_config.json"
  echo "$RFT_DIR/final" > "$PH/init_adapter"
  eval_cmd rft-train greedy selected --set "model.adapter=$RFT_DIR/final"
  eval_cmd rft-heldout greedy heldout --set "model.adapter=$RFT_DIR/final"
}

phase_grpo() {
  local adapter_args=() resume_args=()
  if [[ -s "$PH/init_adapter" ]]; then
    adapter_args=(--set "model.adapter=$(cat "$PH/init_adapter")")
    echo "Starting from the RFT adapter: $(cat "$PH/init_adapter")"
  fi
  if [[ -n "$GRPO_OUT" ]] && compgen -G "$GRPO_OUT/checkpoint-*" >/dev/null; then
    resume_args=(--resume)
    echo "Found checkpoints in $GRPO_OUT. Resuming."
  fi

  local trainer_cmd
  if [[ "$TRAINER_EFFECTIVE" == "nano" ]]; then
    trainer_cmd=(python -m cua.nano_grpo.loop)
  else
    trainer_cmd=(python -m cua.train_grpo)
  fi

  # timeout stops the run at the time budget. Checkpoints (every save_steps) survive.
  # SIGINT (not TERM) arrives in Python as KeyboardInterrupt, which nano_grpo
  # catches to save one last checkpoint before it exits.
  local cmd=(timeout --signal=INT --kill-after=300 "${GRPO_MAX_HOURS}h"
    "${trainer_cmd[@]}" --config "$GRPO_CONFIG"
    "${TRAINER_SET[@]}" "${GLOBAL_SET[@]}" "${adapter_args[@]}" "${EXTRA_ARGS[@]}" "${resume_args[@]}")
  say "\$ ${cmd[*]}"
  local rc=0
  "${cmd[@]}" || rc=$?
  if [[ "$rc" -eq 124 ]]; then
    echo "Stopped at the ${GRPO_MAX_HOURS} h time budget (GRPO_MAX_HOURS). Checkpoints are in $GRPO_OUT."
    return 0
  fi
  return "$rc"
}

latest_checkpoint() {
  # Highest checkpoint-N in GRPO_OUT.
  /bin/ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -n1
}

phase_final_eval() {
  local adapter="${BEST_CKPT:-}"
  if [[ -z "$adapter" ]]; then
    if [[ -f "$GRPO_OUT/final/adapter_config.json" ]]; then
      adapter="$GRPO_OUT/final"
    else
      local n; n="$(latest_checkpoint)"
      [[ -n "$n" ]] || { echo "No adapter found in $GRPO_OUT. Set BEST_CKPT=PATH."; return 1; }
      adapter="$GRPO_OUT/checkpoint-$n"
    fi
  fi
  [[ -f "$adapter/adapter_config.json" ]] || { echo "Not a PEFT adapter folder: $adapter"; return 1; }
  echo "Evaluating adapter: $adapter"
  echo "$adapter" > "$PH/final_adapter"
  eval_cmd grpo-train greedy selected --set "model.adapter=$adapter"
  eval_cmd grpo-heldout greedy heldout --set "model.adapter=$adapter"
}

phase_report() {
  run python -m cua.report
  [[ -f "$RES/report/index.html" ]] && echo "Report: $REPO/$RES/report/index.html"
  return 0
}

phase_push() {
  if [[ -z "${HF_ADAPTER_REPO:-}" ]]; then
    echo "HF_ADAPTER_REPO isn't set. Skipping the push. Keep the adapter with scripts/sync_results.sh --adapters."
    touch "$PH/.no_marker"; return 0
  fi
  [[ -s "$PH/final_adapter" ]] || { echo "No final adapter recorded. Run final_eval first."; return 1; }
  local adapter; adapter="$(cat "$PH/final_adapter")"
  # HF_TOKEN comes from the environment; huggingface_hub reads it itself.
  ADAPTER="$adapter" REPO_ID="$HF_ADAPTER_REPO" python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = os.environ["REPO_ID"]
api.create_repo(repo_id, private=True, exist_ok=True)
api.upload_folder(folder_path=os.environ["ADAPTER"], repo_id=repo_id,
                  commit_message="Upload GRPO LoRA adapter")
print(f"Uploaded {os.environ['ADAPTER']} to {repo_id} (private).")
PY
}

# ---- runner ----------------------------------------------------------------------------
resume_hint() {
  local prefix=""
  [[ "$MODEL" != "4b" ]] && prefix+="MODEL=$MODEL "
  [[ -n "${TRAINER:-}" ]] && prefix+="TRAINER=$TRAINER "
  echo "${prefix}bash scripts/saturday.sh --from $1"
}

run_phase() {
  local name="$1" log="$LOGS/$1.log" start end rc
  say "==== phase: $name ===="
  say "Log: $log"
  rm -f "$PH/.no_marker"
  start="$(date +%s)"
  # Subshell with -e, so the first failing command ends the phase. The pipe to
  # tee gives you live output and a log; PIPESTATUS[0] is the phase's exit code.
  ( set -Eeuo pipefail; "phase_$name" ) 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"
  end="$(date +%s)"
  add_elapsed $((end - start))
  say "Phase $name took $(( (end - start) / 60 )) min $(( (end - start) % 60 )) s. Spend so far: $(estimate_spend) at \$$PRICE_PER_HR/h."
  if [[ "$rc" -ne 0 ]]; then
    echo
    echo "PHASE FAILED: $name (exit code $rc). Last 40 lines of $log:"
    echo "-----"
    tail -n 40 "$log"
    echo "-----"
    echo "Fix the problem, then resume with:"
    echo "  $(resume_hint "$name")"
    exit "$rc"
  fi
  if [[ -f "$PH/.no_marker" ]]; then
    rm -f "$PH/.no_marker"
  else
    date -u +%FT%TZ > "$PH/$name.done"
  fi
}

show_status() {
  local p
  echo "Model: $MODEL | trainer: $TRAINER_EFFECTIVE | GRPO config: $GRPO_CONFIG"
  for p in "${PHASES[@]}"; do
    if [[ -f "$PH/$p.done" ]]; then printf '  %-11s done (%s)\n' "$p" "$(cat "$PH/$p.done")"
    else printf '  %-11s pending\n' "$p"; fi
  done
  echo "Spend so far: $(estimate_spend)"
}

case "$MODE" in
  status) show_status; exit 0 ;;
  redo)   rm -f "$PH/$TARGET.done"; MODE="all" ;;
esac

say "Model: $MODEL | trainer: $TRAINER_EFFECTIVE | config: $GRPO_CONFIG | price: \$$PRICE_PER_HR/h"

started=0
for p in "${PHASES[@]}"; do
  case "$MODE" in
    only) [[ "$p" == "$TARGET" ]] || continue ;;
    from) [[ "$p" == "$TARGET" ]] && started=1
          [[ "$started" -eq 1 ]] || continue ;;
  esac
  if [[ "$MODE" == "all" && -f "$PH/$p.done" ]]; then
    say "Skipping $p (done). Use --redo $p to run it again."
    continue
  fi
  run_phase "$p"
done

echo
say "All requested phases finished. Spend estimate: $(estimate_spend)."
echo "Next, from your Mac: scripts/sync_results.sh --adapters, then scripts/runpod_down.sh"
