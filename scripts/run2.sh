#!/usr/bin/env bash
# Run 2 pipeline (docs/PLAN_RUN2.md): continue GRPO from the run-1 adapter.
# Runs ON the pod, inside tmux, in resumable phases like saturday.sh.
#
# Everything goes to results/run2/ (CUA_RESULTS_DIR), so run 1's results stay
# untouched. Each phase writes results/run2/.phases/<phase>.done on success and
# logs to results/run2/logs/<phase>.log.
set -uo pipefail
{ set +x; } 2>/dev/null   # never trace: the environment can hold tokens

PHASES=(baseline select heldout_screen pre_eval grpo post_eval compare report)

usage() {
  cat <<'EOF'
Usage: RUN1_ADAPTER=DIR bash scripts/run2.sh [--from PHASE | --only PHASE | --redo PHASE | --status] [-- EXTRA_ARGS]

Phases, in order:
  baseline        Sampled baseline of the run-1 adapter on every run-2 candidate
                  (run name: baseline).
  select          Pick the run-2 pool, band 5%-85%, up to 16 tasks
                  (results/run2/selected_tasks.json).
  heldout_screen  Sampled screen of the base model on the held-out-v2 candidates,
                  then keep the 5 nearest 50% (results/run2/heldout_v2.json).
  pre_eval        Greedy evals before training:
                    rft1-pool1     run-1 RFT adapter on run 1's pool (skipped without RFT1_ADAPTER)
                    base-pool2     base model on the run-2 pool
                    run1-pool2     run-1 adapter on the run-2 pool
                    base-heldout2  base model on held-out v2 (100 seeds)
                    run1-heldout2  run-1 adapter on held-out v2 (100 seeds)
  grpo            GRPO from the run-1 adapter (configs/grpo_run2.yaml), time-capped.
  post_eval       Greedy evals of the run-2 adapter: run2-pool2, run2-heldout2, run2-pool1.
  compare         Paired comparisons into results/run2/compare/*.md and *.json.
  report          Build results/run2/report/index.html (report preset run2).

Options:
  --from PHASE    Run PHASE and every later phase, even when markers exist.
  --only PHASE    Run only PHASE, even when its marker exists.
  --redo PHASE    Delete the marker for PHASE, then run as usual.
  --status        Show which phases are done and exit.
  -- ARGS         Extra arguments for the trainer in the grpo phase, for example:
                  -- --set grpo.learning_rate=5e-6

Environment variables:
  RUN1_ADAPTER      Required. Folder with the run-1 final adapter (adapter_config.json).
  RFT1_ADAPTER      Optional. Folder with the run-1 RFT adapter, for the rft1-pool1 eval.
  RUN1_RESULTS      Optional. Run 1's results folder (with eval/grpo-train), for the
                    retention comparison. Default: results.
  GRPO_MAX_HOURS=5  Stop the grpo phase after this many hours. Checkpoints stay.
  BEST_CKPT=PATH    Adapter that post_eval evaluates. Default: the final adapter,
                    or the newest checkpoint.
  PRICE_PER_HR=2.19 Hourly price used for the spend estimate.
  CUA_SETS="--set K=V ..."  Extra --set arguments for every eval and grpo command.
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

RES="results/run2"
export CUA_RESULTS_DIR="$RES"
PH="$RES/.phases"
LOGS="$RES/logs"
mkdir -p "$PH" "$LOGS"

PRICE_PER_HR="${PRICE_PER_HR:-2.19}"
GRPO_MAX_HOURS="${GRPO_MAX_HOURS:-5}"
GRPO_CONFIG="configs/grpo_run2.yaml"
GRPO_OUT="$(awk '/^grpo:/{g=1; next} /^[^ #]/{g=0} g && /^  output_dir:/{print $2; exit}' "$GRPO_CONFIG")"
RUN1_RESULTS="${RUN1_RESULTS:-results}"

GLOBAL_SET=()
if [[ -n "${CUA_SETS:-}" ]]; then read -r -a GLOBAL_SET <<<"$CUA_SETS"; fi

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

if [[ "$MODE" != "status" ]]; then
  [[ -n "${RUN1_ADAPTER:-}" && -f "${RUN1_ADAPTER}/adapter_config.json" ]] \
    || { echo "Set RUN1_ADAPTER to the run-1 adapter folder (it must hold adapter_config.json)." >&2; exit 1; }
fi

# ---- helpers --------------------------------------------------------------------------
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { say "\$ $*"; "$@"; }

estimate_spend() {
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
    "${GLOBAL_SET[@]}" "$@"
}

final_adapter() {
  if [[ -n "${BEST_CKPT:-}" ]]; then echo "$BEST_CKPT"; return; fi
  if [[ -f "$GRPO_OUT/final/adapter_config.json" ]]; then echo "$GRPO_OUT/final"; return; fi
  local n
  n="$(/bin/ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -n1)"
  [[ -n "$n" ]] && echo "$GRPO_OUT/checkpoint-$n"
}

# ---- phases ----------------------------------------------------------------------------
phase_baseline() {
  eval_cmd baseline baseline candidates_run2 --set "model.adapter=$RUN1_ADAPTER"
}

phase_select() {
  run python -m cua.select_tasks --candidates candidates_run2 \
    --baseline "$RES/eval/baseline/per_task.csv" \
    --set select.min_success=0.05 --set select.max_success=0.85 --set select.max_tasks=16
}

phase_heldout_screen() {
  # Base model on purpose: picking held-out tasks with either adapter would
  # favor that adapter in the comparison.
  eval_cmd heldout-screen baseline heldout_v2_candidates
  run python -m cua.select_tasks --heldout-extra 5 --baseline "$RES/eval/heldout-screen/per_task.csv"
}

phase_pre_eval() {
  if [[ -n "${RFT1_ADAPTER:-}" && -f "${RFT1_ADAPTER}/adapter_config.json" ]]; then
    eval_cmd rft1-pool1 greedy run1_pool --set "model.adapter=$RFT1_ADAPTER"
  else
    echo "RFT1_ADAPTER isn't set; skipping the rft1-pool1 eval."
  fi
  eval_cmd base-pool2 greedy selected
  eval_cmd run1-pool2 greedy selected --set "model.adapter=$RUN1_ADAPTER"
  eval_cmd base-heldout2 greedy heldout_v2 --set eval.num_seeds=100
  eval_cmd run1-heldout2 greedy heldout_v2 --set eval.num_seeds=100 --set "model.adapter=$RUN1_ADAPTER"
}

phase_grpo() {
  local adapter_args=(--set "model.adapter=$RUN1_ADAPTER") resume_args=()
  if compgen -G "$GRPO_OUT/checkpoint-*" >/dev/null; then
    resume_args=(--resume)   # --resume loads the newest checkpoint instead of RUN1_ADAPTER
    echo "Found checkpoints in $GRPO_OUT. Resuming."
  fi
  # SIGINT reaches Python as KeyboardInterrupt; nano_grpo saves a checkpoint and exits.
  local cmd=(timeout --signal=INT --kill-after=300 "${GRPO_MAX_HOURS}h"
    python -m cua.nano_grpo.loop --config "$GRPO_CONFIG"
    "${GLOBAL_SET[@]}" "${adapter_args[@]}" "${EXTRA_ARGS[@]}" "${resume_args[@]}")
  say "\$ ${cmd[*]}"
  local rc=0
  "${cmd[@]}" || rc=$?
  if [[ "$rc" -eq 124 ]]; then
    echo "Stopped at the ${GRPO_MAX_HOURS} h time budget. Checkpoints are in $GRPO_OUT."
    return 0
  fi
  return "$rc"
}

phase_post_eval() {
  local adapter; adapter="$(final_adapter)"
  [[ -n "$adapter" && -f "$adapter/adapter_config.json" ]] || { echo "No run-2 adapter found in $GRPO_OUT. Set BEST_CKPT=PATH."; return 1; }
  echo "Evaluating adapter: $adapter"
  echo "$adapter" > "$PH/final_adapter"
  eval_cmd run2-pool2 greedy selected --set "model.adapter=$adapter"
  eval_cmd run2-heldout2 greedy heldout_v2 --set eval.num_seeds=100 --set "model.adapter=$adapter"
  eval_cmd run2-pool1 greedy run1_pool --set "model.adapter=$adapter"
}

phase_compare() {
  local out="$RES/compare" e="$RES/eval"
  mkdir -p "$out"
  compare() {  # compare NAME A_DIR B_DIR
    if [[ -f "$2/episodes.jsonl" && -f "$3/episodes.jsonl" ]]; then
      run python -m cua.compare "$2" "$3" --json "$out/$1.json" | tee "$out/$1.md"
    else
      echo "Skipping $1: missing $2 or $3."
    fi
  }
  compare pool2-run2-vs-run1     "$e/run1-pool2"    "$e/run2-pool2"
  compare pool2-run1-vs-base     "$e/base-pool2"    "$e/run1-pool2"
  compare heldout2-run2-vs-run1  "$e/run1-heldout2" "$e/run2-heldout2"
  compare heldout2-run1-vs-base  "$e/base-heldout2" "$e/run1-heldout2"
  compare pool1-run2-vs-run1     "$RUN1_RESULTS/eval/grpo-train" "$e/run2-pool1"
  compare pool1-rft1-vs-base     "$RUN1_RESULTS/eval/base-train" "$e/rft1-pool1"
  compare pool1-run1-vs-rft1     "$e/rft1-pool1"    "$RUN1_RESULTS/eval/grpo-train"
  return 0
}

phase_report() {
  run python -m cua.report --preset run2
  [[ -f "$RES/report/index.html" ]] && echo "Report: $REPO/$RES/report/index.html"
  return 0
}

# ---- runner ----------------------------------------------------------------------------
run_phase() {
  local name="$1" log="$LOGS/$1.log" start end rc
  say "==== phase: $name ===="
  say "Log: $log"
  start="$(date +%s)"
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
    echo "  RUN1_ADAPTER=$RUN1_ADAPTER bash scripts/run2.sh --from $name"
    exit "$rc"
  fi
  date -u +%FT%TZ > "$PH/$name.done"
}

show_status() {
  local p
  echo "Run 2 | GRPO config: $GRPO_CONFIG | results: $RES"
  for p in "${PHASES[@]}"; do
    if [[ -f "$PH/$p.done" ]]; then printf '  %-15s done (%s)\n' "$p" "$(cat "$PH/$p.done")"
    else printf '  %-15s pending\n' "$p"; fi
  done
  echo "Spend so far: $(estimate_spend)"
}

case "$MODE" in
  status) show_status; exit 0 ;;
  redo)   rm -f "$PH/$TARGET.done"; MODE="all" ;;
esac

say "Run 2 | adapter: $RUN1_ADAPTER | config: $GRPO_CONFIG | price: \$$PRICE_PER_HR/h"

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
