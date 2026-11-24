#!/usr/bin/env bash
# Run 3 pipeline (docs/PLAN_RUN3.md): GRPO on Track B, continued from the run-2 adapter.
# Runs ON the pod, inside tmux session run3 (scripts/run3_pod_start.sh starts it).
#
# Everything goes to results/run3/ (CUA_RESULTS_DIR). Each phase writes
# results/run3/.phases/<phase>.started when it starts, <phase>.done on success,
# and <phase>.failed on failure, and logs to results/run3/logs/<phase>.log.
# Phases that are done are skipped, so after a pod loss the same command
# continues from the first unfinished phase. The contract with the monitor is
# docs/RUN3_INTERFACES.md.
set -uo pipefail
{ set +x; } 2>/dev/null   # never trace: the environment can hold tokens

PHASES=(smoke screen select rft pre_eval grpo post_eval compare report)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || exit 1

# Fast path for the monitor: no env.sh, no torch, one JSON object, exit 0.
if [[ "${1:-}" == "--status-json" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m cua.run3_status || echo '{"run": "run3", "error": "run3_status failed"}'
  else
    echo '{"run": "run3", "error": "python3 not found"}'
  fi
  exit 0
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/run3.sh [--from PHASE | --only PHASE | --redo PHASE | --status | --status-json] [-- EXTRA_ARGS]

Phases, in order:
  smoke      Fail-fast checks: inputs, task names, 2 GRPO steps on 2 tasks, 1 more
             step with --resume, and a 3-seed Track B eval of the smoke adapter.
  screen     Sampled screen of the run-2 adapter on Track B, 16 seeds, on every
             registered run-3 candidate (run name: baseline).
  select     Mix the run-3 pool (results/run3/selected_tasks.json): up to 16 learn
             tasks in the 5%-85% band, at least 3 from each source, plus up to 4
             solved replay tasks from the run-1 and run-2 pools.
  rft        Optional RFT warm start on Track B successes (RFT=auto: only when the
             screen's format-error rate is 5% or more).
  pre_eval   Greedy Track B evals before training: base-pool3, run2-pool3,
             base-heldout3, run2-heldout3 (100 seeds), rft3-pool3 (if RFT ran), and
             run2A-pool2 (Track A, only if run 2's own eval isn't available).
  grpo       GRPO from the start adapter (configs/grpo_run3.yaml), time-capped.
  post_eval  Greedy evals of the run-3 adapter: run3-pool3, run3-heldout3 (Track B),
             and run3A-pool2 (Track A, for forgetting against run 2's 72.6%).
  compare    Paired comparisons into results/run3/compare/*.md and *.json.
  report     Build results/run3/report/index.html (report preset run3).

Options:
  --from PHASE    Run PHASE and every later phase, even when markers exist.
  --only PHASE    Run only PHASE, even when its marker exists.
  --redo PHASE    Delete the marker for PHASE, rerun its evals, then run as usual.
  --status        Show which phases are done and exit.
  --status-json   Print the status as one JSON object for the monitor and exit.
  -- ARGS         Extra arguments for the trainer in the grpo phase.

Environment variables:
  RUN2_ADAPTER=/workspace/inputs/run2-adapter  Run-2 adapter (restored from the Hub).
  GRPO_MAX_HOURS=9   Total GRPO time, summed over pod restarts. Checkpoints stay.
  RFT=auto           auto, on, or off.
  BEST_CKPT=PATH     Adapter that post_eval evaluates. Default: the final adapter.
  PRICE_PER_HR=2.90  Hourly price for the spend estimate in the log.
  FORCE_EVAL=1       Rerun evals whose summary.json already exists.
  CUA_SETS="--set K=V ..."  Extra --set arguments for every eval and grpo command.

To stop GRPO early and go on to the post-training evals: touch results/run3/.stop_grpo
EOF
}

if [[ -f /workspace/env.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /workspace/env.sh
  set -u
fi

RES="results/run3"
export CUA_RESULTS_DIR="$RES"
PH="$RES/.phases"
LOGS="$RES/logs"
mkdir -p "$PH" "$LOGS"

RUN2_ADAPTER="${RUN2_ADAPTER:-/workspace/inputs/run2-adapter}"
RUN2_RESULTS="${RUN2_RESULTS:-results/run2}"
PRICE_PER_HR="${PRICE_PER_HR:-2.90}"
GRPO_MAX_HOURS="${GRPO_MAX_HOURS:-9}"
RFT="${RFT:-auto}"
FORCE_EVAL="${FORCE_EVAL:-0}"
GRPO_CONFIG="configs/grpo_run3.yaml"
RFT_CONFIG="configs/rft_run3.yaml"
RFT_OUT=/workspace/ckpt/rft-4b-run3/final
GRPO_OUT="$(awk '/^grpo:/{g=1; next} /^[^ #]/{g=0} g && /^  output_dir:/{print $2; exit}' "$GRPO_CONFIG")"
TRACK_B=(--set env.track=B --set env.max_turns=10)

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

# ---- helpers --------------------------------------------------------------------------
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { say "\$ $*"; "$@"; }
now_iso() { date -u +%FT%TZ; }

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
# Skips an eval whose summary.json exists (cua.eval writes it last), so a phase
# that a pod loss cut short reruns only its unfinished evals.
eval_cmd() {
  local name="$1" mode="$2" tasks="$3"; shift 3
  if [[ "$FORCE_EVAL" != "1" && -f "$RES/eval/$name/summary.json" ]]; then
    say "Skipping eval $name (summary.json exists; FORCE_EVAL=1 reruns it)."
    return 0
  fi
  run python -m cua.eval --config configs/eval.yaml --no-wandb \
    --set "eval.mode=$mode" --set "eval.tasks=$tasks" --set "run_name=$name" \
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "$@"
}

start_adapter() {
  if [[ -f "$PH/start_adapter" ]]; then cat "$PH/start_adapter"; else echo "$RUN2_ADAPTER"; fi
}

final_adapter() {
  if [[ -n "${BEST_CKPT:-}" ]]; then echo "$BEST_CKPT"; return; fi
  if [[ -f "$GRPO_OUT/final/adapter_config.json" ]]; then echo "$GRPO_OUT/final"; return; fi
  local n
  n="$(/bin/ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sed 's/.*checkpoint-//' | sort -n | tail -n1)"
  [[ -n "$n" ]] && echo "$GRPO_OUT/checkpoint-$n"
}

need_run2_adapter() {
  [[ -f "$RUN2_ADAPTER/adapter_config.json" ]] \
    || { echo "The run-2 adapter isn't at $RUN2_ADAPTER. Run: python -m cua.hfsync restore"; return 1; }
}

# ---- phases ----------------------------------------------------------------------------
phase_smoke() {
  need_run2_adapter
  case "${MINIWOB_URL:-}" in
    "") echo "MINIWOB_URL isn't set. Source /workspace/env.sh."; return 1 ;;
    file:///workspace/*) echo "WARNING: MiniWoB pages load from /workspace ($MINIWOB_URL). If that is a network" \
                              "drive, browsers start slowly. run3_pod_start.sh puts MiniWoB on /root." ;;
  esac
  python - <<'EOF'
from cua.config import load_tasks, resolve_task_list
spec = load_tasks()  # also refuses held-out tasks among the training candidates
held = resolve_task_list("heldout_v2")
assert len(held) == 10, held
assert not set(held) & set(spec.train_candidates_run3)
print(f"{len(spec.train_candidates_run3)} run-3 candidates, {len(held)} held-out-v2 tasks")
EOF
  run python -m cua.run3_pool registered --out "$RES/candidates.json"

  local sres="$RES/smoke" sck=/workspace/ckpt/smoke-run3
  rm -rf "$sres" "$sck"
  mkdir -p "$sres"
  local smoke_set=(--set "model.adapter=$RUN2_ADAPTER" --set "grpo.output_dir=$sck" --set run_name=smoke-run3
    --set "tasks=click-button,multi-layouts" --set grpo.prompts_per_step=4 --set grpo.num_generations=4
    --set grpo.dynamic_sampling.max_rounds=1 --set grpo.save_steps=2 --set grpo.eval_every=0
    --set grpo.samples_every=1 --set grpo.warmup_steps=1 --set wandb.mode=disabled)
  say "Smoke GRPO: 2 steps from the run-2 adapter."
  run env CUA_RESULTS_DIR="$sres" timeout 1800 python -m cua.nano_grpo.loop --config "$GRPO_CONFIG" \
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "${smoke_set[@]}" --set grpo.max_steps=2
  say "Smoke GRPO: 1 more step with --resume."
  run env CUA_RESULTS_DIR="$sres" timeout 1200 python -m cua.nano_grpo.loop --config "$GRPO_CONFIG" \
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "${smoke_set[@]}" --set grpo.max_steps=3 --resume 2>&1 | tee "$sres/resume.log"
  say "Smoke eval: 3 seeds, Track B, smoke adapter."
  run env CUA_RESULTS_DIR="$sres" timeout 900 python -m cua.eval --config configs/eval.yaml --no-wandb \
    --set eval.mode=greedy --set "eval.tasks=click-button,multi-layouts" --set eval.num_seeds=3 \
    --set eval.gif_seeds=1 --set run_name=smoke-eval --set "model.adapter=$sck/final" \
    "${TRACK_B[@]}" ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"}
  SRES="$sres" SCK="$sck" python - <<'EOF'
import json, os
from pathlib import Path
res, ck = Path(os.environ["SRES"]), Path(os.environ["SCK"])
rows = [json.loads(l) for l in (res / "train/smoke-run3/metrics.jsonl").read_text().splitlines() if l.strip()]
steps = [r["step"] for r in rows if "time/rollout_s" in r]
assert steps == [1, 2, 3], f"metric steps {steps}, want [1, 2, 3]"
state = json.loads((ck / "checkpoint-3/trainer_state.json").read_text())
assert state["step"] == 3 and "sampler" in state, state.keys()
assert (ck / "final/adapter_model.safetensors").exists()
assert "resumed from" in (res / "resume.log").read_text(), "the second run didn't resume"
for n in (1, 2, 3):
    ep = json.loads((res / f"samples/step-{n:06d}/episode.json").read_text())
    assert ep["turns"] and ep["turns"][0]["image"], ep
s = json.loads((res / "eval/smoke-eval/summary.json").read_text())
assert s["track"] == "B" and s["n_crashed"] < 6 and s["overall_success"] is not None, s
if not any(r.get("episodes/trained", 0) > 0 for r in rows):
    print("WARNING: no smoke step had a mixed group, so the train phase didn't run. "
          "The grpo phase is its first real test.")
for r in rows:
    if "time/rollout_s" in r:
        print(f"step {r['step']}: rollout {r['time/rollout_s']:.0f} s, train {r['time/train_s']:.0f} s, "
              f"trained episodes {r.get('episodes/trained', 0):.0f}, format errors per turn "
              f"{r.get('format_error/rate_per_turn', float('nan')):.2f}")
print(f"smoke eval: success {s['overall_success']}, format errors {s['format_error_rate']}")
print("SMOKE OK")
EOF
  rm -rf "$sck"
}

phase_screen() {
  need_run2_adapter
  [[ -f "$RES/candidates.json" ]] || run python -m cua.run3_pool registered --out "$RES/candidates.json"
  local tasks
  tasks="$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["tasks"]))' "$RES/candidates.json")"
  eval_cmd baseline baseline "$tasks" "${TRACK_B[@]}" --set "model.adapter=$RUN2_ADAPTER"
}

phase_select() {
  run python -m cua.run3_pool select --baseline "$RES/eval/baseline/per_task.csv" \
    --candidates "$RES/candidates.json"
  # Held-out v2 is run 2's list, unchanged, so held-out numbers compare across runs.
  RUN2_RESULTS="$RUN2_RESULTS" python - <<'EOF'
import json, os
from pathlib import Path
from cua.config import load_tasks
frozen = load_tasks().heldout_v2
src = Path(os.environ["RUN2_RESULTS"]) / "heldout_v2.json"
if src.exists():
    theirs = json.loads(src.read_text())["heldout_v2"]
    assert theirs == frozen, f"run 2's heldout_v2.json {theirs} differs from configs/tasks.yaml {frozen}"
Path("results/run3/heldout_v2.json").write_text(json.dumps({"heldout_v2": frozen, "source": str(src)}, indent=2))
print(f"Held-out v2: {frozen}")
EOF
}

phase_rft() {
  need_run2_adapter
  echo "$RUN2_ADAPTER" > "$PH/start_adapter"
  case "$RFT" in
    off) echo "RFT=off: GRPO starts from the run-2 adapter."; return 0 ;;
    auto)
      if python -m cua.rft should_skip --config "$RFT_CONFIG"; then
        echo "RFT=auto: skipping RFT (the format-error rate is low). GRPO starts from the run-2 adapter."
        return 0
      fi ;;
    on) ;;
    *) echo "RFT must be auto, on, or off, not '$RFT'."; return 1 ;;
  esac
  run python -m cua.rft collect --config "$RFT_CONFIG" --set "model.adapter=$RUN2_ADAPTER" ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"}
  run python -m cua.rft train --config "$RFT_CONFIG" --set "model.adapter=$RUN2_ADAPTER" ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"}
  [[ -f "$RFT_OUT/adapter_config.json" ]] || { echo "RFT finished without $RFT_OUT"; return 1; }
  echo "$RFT_OUT" > "$PH/start_adapter"
}

phase_pre_eval() {
  need_run2_adapter
  local start; start="$(start_adapter)"
  eval_cmd base-pool3    greedy selected   "${TRACK_B[@]}"
  eval_cmd run2-pool3    greedy selected   "${TRACK_B[@]}" --set "model.adapter=$RUN2_ADAPTER"
  eval_cmd base-heldout3 greedy heldout_v2 "${TRACK_B[@]}" --set eval.num_seeds=100
  eval_cmd run2-heldout3 greedy heldout_v2 "${TRACK_B[@]}" --set eval.num_seeds=100 --set "model.adapter=$RUN2_ADAPTER"
  if [[ "$start" != "$RUN2_ADAPTER" ]]; then
    eval_cmd rft3-pool3 greedy selected "${TRACK_B[@]}" --set "model.adapter=$start"
  fi
  if [[ ! -f "$RUN2_RESULTS/eval/run2-pool2/episodes.jsonl" ]]; then
    echo "Run 2's own Track A eval isn't here; evaluating the run-2 adapter on Track A instead."
    eval_cmd run2A-pool2 greedy run2_pool --set "model.adapter=$RUN2_ADAPTER"
  fi
}

phase_grpo() {
  local start; start="$(start_adapter)"
  [[ -f "$start/adapter_config.json" ]] || { echo "Start adapter $start is missing."; return 1; }
  local resume_args=()
  if compgen -G "$GRPO_OUT/checkpoint-[0-9]*" >/dev/null; then
    resume_args=(--resume)   # loads the newest checkpoint instead of the start adapter
    echo "Found checkpoints in $GRPO_OUT. Resuming."
  fi
  # The time cap counts GRPO time over every pod, so a restart doesn't reset it.
  local used cap left
  used="$(cat "$PH/grpo.elapsed_s" 2>/dev/null || echo 0)"
  [[ "$used" =~ ^[0-9]+$ ]] || used=0
  cap="$(awk -v h="$GRPO_MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')"
  left=$((cap - used))
  if (( left < 300 )); then
    echo "The GRPO time budget is used up ($used s of $cap s). Going on with the newest checkpoint."
    [[ -n "$(final_adapter)" ]] || { echo "No checkpoint in $GRPO_OUT."; return 1; }
    return 0
  fi
  echo "GRPO time: $used s used, $left s left of $cap s (GRPO_MAX_HOURS=$GRPO_MAX_HOURS)."
  # Record GRPO time every minute. Its output goes to /dev/null, so tee doesn't wait for it.
  ( t0="$(date +%s)"
    while sleep 60; do echo $((used + $(date +%s) - t0)) > "$PH/grpo.elapsed_s.tmp" && mv "$PH/grpo.elapsed_s.tmp" "$PH/grpo.elapsed_s"; done
  ) >/dev/null 2>&1 &
  local ticker=$! t_start; t_start="$(date +%s)"
  # SIGINT reaches Python as KeyboardInterrupt; nano_grpo saves a checkpoint and exits.
  local cmd=(timeout --signal=INT --kill-after=300 "${left}s"
    python -m cua.nano_grpo.loop --config "$GRPO_CONFIG"
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} --set "model.adapter=$start" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} ${resume_args[@]+"${resume_args[@]}"})
  say "\$ ${cmd[*]}"
  local rc=0
  "${cmd[@]}" || rc=$?
  kill "$ticker" 2>/dev/null || true
  echo $((used + $(date +%s) - t_start)) > "$PH/grpo.elapsed_s"
  if [[ "$rc" -eq 124 ]]; then
    echo "Stopped at the ${GRPO_MAX_HOURS} h time budget. Checkpoints are in $GRPO_OUT."
    return 0
  fi
  [[ "$rc" -eq 0 && -f "$RES/.stop_grpo" ]] && echo "Stopped early by $RES/.stop_grpo."
  return "$rc"
}

phase_post_eval() {
  local adapter; adapter="$(final_adapter)"
  [[ -n "$adapter" && -f "$adapter/adapter_config.json" ]] || { echo "No run-3 adapter in $GRPO_OUT. Set BEST_CKPT=PATH."; return 1; }
  echo "Evaluating adapter: $adapter"
  echo "$adapter" > "$PH/final_adapter"
  eval_cmd run3-pool3    greedy selected   "${TRACK_B[@]}" --set "model.adapter=$adapter"
  eval_cmd run3-heldout3 greedy heldout_v2 "${TRACK_B[@]}" --set eval.num_seeds=100 --set "model.adapter=$adapter"
  eval_cmd run3A-pool2   greedy run2_pool  --set "model.adapter=$adapter"
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
  local track_a_ref="$RUN2_RESULTS/eval/run2-pool2"
  [[ -f "$track_a_ref/episodes.jsonl" ]] || track_a_ref="$e/run2A-pool2"
  compare pool3-run3-vs-run2         "$e/run2-pool3"    "$e/run3-pool3"
  compare pool3-run2-vs-base         "$e/base-pool3"    "$e/run2-pool3"
  compare pool3-run3-vs-base         "$e/base-pool3"    "$e/run3-pool3"
  compare heldout3-run3-vs-run2      "$e/run2-heldout3" "$e/run3-heldout3"
  compare heldout3-run2-vs-base      "$e/base-heldout3" "$e/run2-heldout3"
  compare heldout3-run3-vs-base      "$e/base-heldout3" "$e/run3-heldout3"
  compare trackA-pool2-run3-vs-run2  "$track_a_ref"     "$e/run3A-pool2"
  compare pool3-rft3-vs-run2         "$e/run2-pool3"    "$e/rft3-pool3"
  compare pool3-run3-vs-rft3         "$e/rft3-pool3"    "$e/run3-pool3"
  return 0
}

phase_report() {
  run python -m cua.report --preset run3
  [[ -f "$RES/report/index.html" ]] && echo "Report: $REPO/$RES/report/index.html"
  return 0
}

# ---- runner ----------------------------------------------------------------------------
run_phase() {
  local name="$1" log="$LOGS/$1.log" start end rc
  say "==== phase: $name ===="
  say "Log: $log"
  now_iso > "$PH/$name.started"
  rm -f "$PH/$name.failed"
  echo "$name" > "$PH/.current"
  start="$(date +%s)"
  ( set -Eeuo pipefail; "phase_$name" ) 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"
  end="$(date +%s)"
  rm -f "$PH/.current"
  add_elapsed $((end - start))
  say "Phase $name took $(( (end - start) / 60 )) min $(( (end - start) % 60 )) s. Spend so far: $(estimate_spend) at \$$PRICE_PER_HR/h."
  if [[ "$rc" -ne 0 ]]; then
    echo "$(now_iso) exit $rc" > "$PH/$name.failed"
    echo
    echo "PHASE FAILED: $name (exit code $rc). Last 40 lines of $log:"
    echo "-----"
    tail -n 40 "$log"
    echo "-----"
    echo "Fix the problem, then resume with: bash scripts/run3.sh"
    exit "$rc"
  fi
  now_iso > "$PH/$name.done"
}

show_status() {
  local p
  echo "Run 3 | GRPO config: $GRPO_CONFIG | results: $RES"
  for p in "${PHASES[@]}"; do
    if [[ -f "$PH/$p.done" ]]; then printf '  %-10s done (%s)\n' "$p" "$(cat "$PH/$p.done")"
    elif [[ -f "$PH/$p.failed" ]]; then printf '  %-10s failed (%s)\n' "$p" "$(cat "$PH/$p.failed")"
    else printf '  %-10s pending\n' "$p"; fi
  done
  echo "Spend so far: $(estimate_spend)"
}

case "$MODE" in
  status) show_status; exit 0 ;;
  redo)   rm -f "$PH/$TARGET.done"; FORCE_EVAL=1; MODE="all" ;;
esac

# One pipeline at a time.
if [[ -f "$PH/.pid" ]]; then
  other="$(cat "$PH/.pid")"
  if [[ "$other" =~ ^[0-9]+$ && "$other" != "$$" ]] && kill -0 "$other" 2>/dev/null; then
    echo "run3.sh is already running (PID $other)." >&2
    exit 1
  fi
fi
echo "$$" > "$PH/.pid"
trap 'rm -f "$PH/.pid" "$PH/.current"' EXIT

say "Run 3 | run-2 adapter: $RUN2_ADAPTER | config: $GRPO_CONFIG | GRPO cap: ${GRPO_MAX_HOURS} h | RFT=$RFT | price: \$$PRICE_PER_HR/h"

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
all_done=1
for p in "${PHASES[@]}"; do [[ -f "$PH/$p.done" ]] || all_done=0; done
if [[ "$all_done" -eq 1 ]]; then echo "ALL DONE"; fi
