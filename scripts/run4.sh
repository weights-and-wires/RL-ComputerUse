#!/usr/bin/env bash
# Run 4 pipeline (docs/PLAN_RUN4.md): GRPO on Track B from the run-3 adapter, with
# 20 turns, a history window, double_click and left_click_drag, and 64 browsers.
# Runs ON the pod, inside tmux session run4 (scripts/run4_pod_start.sh starts it).
#
# Everything goes to results/run4/ (CUA_RESULTS_DIR). Each phase writes
# results/run4/.phases/<phase>.started when it starts, <phase>.done on success,
# and <phase>.failed on failure, and logs to results/run4/logs/<phase>.log.
# Phases that are done are skipped, so after a pod loss the same command
# continues from the first unfinished phase. The contract with the monitor is
# docs/RUN3_INTERFACES.md with run3 replaced by run4; docs/RUN4_INTERFACES.md
# lists the differences.
set -uo pipefail
{ set +x; } 2>/dev/null   # never trace: the environment can hold tokens

PHASES=(smoke screen select rft pre_eval grpo post_eval compare report)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || exit 1

# Fast path for the monitor: no env.sh, no torch, one JSON object, exit 0.
if [[ "${1:-}" == "--status-json" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m cua.run_status --run run4 || echo '{"run": "run4", "error": "run_status failed"}'
  else
    echo '{"run": "run4", "error": "python3 not found"}'
  fi
  exit 0
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/run4.sh [--from PHASE | --only PHASE | --redo PHASE | --status | --status-json] [-- EXTRA_ARGS]

Phases, in order:
  smoke      Fail-fast checks of everything run 4 adds: task lists, the browser count
             (64, or half if that fails), drag and double-click in real browsers, the
             history window (prompt = prefix of the trained sequence), 2 GRPO steps on
             2 tasks with the ratio cap and sampler checks, a mid-training eval and best
             adapter, 1 more step with --resume, and a 3-seed eval of the smoke adapter.
  screen     Sampled screen of the run-3 adapter at run 4's settings, 16 seeds, on every
             registered run-4 candidate (run name: baseline).
  select     The run-4 pool (results/run4/selected_tasks.json): up to 18 learn tasks,
             4 slots for drag tasks and 5 for long-horizon tasks, plus up to 3 solved
             replay tasks from the run-3 pool. Also the per-task turn budgets.
  rft        Kept for the phase contract. Run 4 has no RFT: it records the start adapter.
  pre_eval   Greedy evals of the run-3 adapter at run 4's settings: run3-heldout4
             (100 seeds) and run3-full4 (full suite, 20 seeds). The base model too, as
             BASE_EVAL says.
  grpo       GRPO from the run-3 adapter (configs/grpo_run4.yaml), time-capped.
  post_eval  Choose between the final and the best mid-training adapter (select-final,
             baseline seeds), then run4-heldout4 and run4-full4 with the chosen one.
  compare    Paired comparisons into results/run4/compare/*.md and *.json.
  report     Build results/run4/report/index.html (report preset run4).

Options:
  --from PHASE    Run PHASE and every later phase, even when markers exist.
  --only PHASE    Run only PHASE, even when its marker exists.
  --redo PHASE    Delete the marker for PHASE, rerun its evals, then run as usual.
  --status        Show which phases are done and exit.
  --status-json   Print the status as one JSON object for the monitor and exit.
  -- ARGS         Extra arguments for the trainer in the grpo phase.

Environment variables:
  RUN3_ADAPTER=/workspace/inputs/run3-adapter  Run-3 adapter (restored from the Hub).
  GRPO_MAX_HOURS=10  Total GRPO time, summed over pod restarts. 7.5 fits Secure pricing.
  BASE_EVAL=auto     Base-model evals: off, heldout, full, or auto (full at PRICE_PER_HR
                     of 3.00 or less, else heldout).
  TURN_BUDGETS=1     Per-task turn limits from the screen; 0 gives every task 20 turns.
  CUA_POOL_SIZE=N    Browsers to try in smoke. Default: 64, lowered for small pods.
  BEST_CKPT=PATH     Adapter that post_eval evaluates. Default: final or best (see above).
  PRICE_PER_HR=2.69  Hourly price for the spend estimate in the log.
  FORCE_EVAL=1       Rerun evals whose summary.json already exists.
  CUA_SETS="--set K=V ..."  Extra --set arguments for every eval and grpo command.

To stop GRPO early and go on to the post-training evals: touch results/run4/.stop_grpo
EOF
}

if [[ -f /workspace/env.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /workspace/env.sh
  set -u
fi

RES="results/run4"
export CUA_RESULTS_DIR="$RES"
PH="$RES/.phases"
LOGS="$RES/logs"
mkdir -p "$PH" "$LOGS"

RUN3_ADAPTER="${RUN3_ADAPTER:-/workspace/inputs/run3-adapter}"
PRICE_PER_HR="${PRICE_PER_HR:-2.69}"
GRPO_MAX_HOURS="${GRPO_MAX_HOURS:-10}"
BASE_EVAL="${BASE_EVAL:-auto}"
TURN_BUDGETS="${TURN_BUDGETS:-1}"
RFT="${RFT:-off}"
FORCE_EVAL="${FORCE_EVAL:-0}"
GRPO_CONFIG="configs/grpo_run4.yaml"
EVAL_CONFIG="configs/eval_run4.yaml"
GRPO_OUT="$(awk '/^grpo:/{g=1; next} /^[^ #]/{g=0} g && /^  output_dir:/{print $2; exit}' "$GRPO_CONFIG")"
BUDGETS_FILE="$RES/turn_budgets.json"
POOL_FILE="$RES/pool_size.txt"

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

# The browser count the smoke check passed (or 16 before it ran).
pool_size() {
  if [[ -f "$POOL_FILE" ]]; then cat "$POOL_FILE"; else echo 16; fi
}

# Sets RUN4_SETS to the --set arguments every run-4 command shares: the browser
# count, and the turn budgets once the select phase wrote them.
RUN4_SETS=()
set_run4_sets() {
  RUN4_SETS=(--set "env.pool_size=$(pool_size)")
  if [[ "$TURN_BUDGETS" == "1" && -f "$BUDGETS_FILE" ]]; then RUN4_SETS+=(--set "env.turn_budgets=$BUDGETS_FILE"); fi
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
  set_run4_sets
  run python -m cua.eval --config "$EVAL_CONFIG" --no-wandb \
    --set "eval.mode=$mode" --set "eval.tasks=$tasks" --set "run_name=$name" \
    "${RUN4_SETS[@]}" ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "$@"
}

start_adapter() {
  if [[ -f "$PH/start_adapter" ]]; then cat "$PH/start_adapter"; else echo "$RUN3_ADAPTER"; fi
}

# The newest adapter GRPO left: final/ if it exists, else the newest checkpoint.
grpo_final_adapter() {
  if [[ -f "$GRPO_OUT/final/adapter_config.json" ]]; then echo "$GRPO_OUT/final"; return; fi
  local n
  n="$(/bin/ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sed 's/.*checkpoint-//' | sort -n | tail -n1)"
  [[ -n "$n" ]] && echo "$GRPO_OUT/checkpoint-$n"
}

need_run3_adapter() {
  [[ -f "$RUN3_ADAPTER/adapter_config.json" ]] \
    || { echo "The run-3 adapter isn't at $RUN3_ADAPTER. Run: python -m cua.hfsync --run run4 restore"; return 1; }
}

base_eval_mode() {
  case "$BASE_EVAL" in
    off|heldout|full) echo "$BASE_EVAL" ;;
    auto) awk -v p="$PRICE_PER_HR" 'BEGIN { print (p <= 3.0) ? "full" : "heldout" }' ;;
    *) echo "heldout" ;;
  esac
}

# ---- phases ----------------------------------------------------------------------------
phase_smoke() {
  need_run3_adapter
  case "${MINIWOB_URL:-}" in
    "") echo "MINIWOB_URL isn't set. Source /workspace/env.sh."; return 1 ;;
    file:///workspace/*) echo "WARNING: MiniWoB pages load from /workspace ($MINIWOB_URL). If that is a network" \
                              "drive, browsers start slowly. run4_pod_start.sh puts MiniWoB on /root." ;;
  esac
  python - <<'EOF'
from cua.config import load_tasks, resolve_task_list
spec = load_tasks()  # also refuses held-out tasks among the training candidates
held = resolve_task_list("heldout_v2")
assert len(held) == 10, held
assert not set(held) & set(spec.train_candidates_run4)
assert "stock-market" not in spec.train_candidates_run4
print(f"{len(spec.train_candidates_run4)} run-4 candidates, {len(held)} held-out-v2 tasks")
EOF
  run python -m cua.run_pool --run run4 registered --out "$RES/candidates.json"

  # 1. Browsers: try 64 (or CUA_POOL_SIZE), halve once if that fails.
  local want="${CUA_POOL_SIZE:-}"
  [[ -n "$want" ]] || want="$(python -m cua.pipeline pool-size --want 64)"
  say "Smoke: $want browsers at once."
  run timeout 900 python -m cua.smoke4 browsers --config "$GRPO_CONFIG" --size "$want" --out "$POOL_FILE"
  say "Browsers: $(pool_size)."

  # 2. Drag, double-click, and click in real browsers. 3. The history window.
  run timeout 600 python -m cua.smoke4 actions --config "$GRPO_CONFIG"
  run timeout 900 python -m cua.smoke4 context --config "$GRPO_CONFIG"

  # 4. GRPO: 2 steps, a mid-training eval at step 2 (best adapter), then 1 more step with --resume.
  local sres="$RES/smoke" sck=/workspace/ckpt/smoke-run4
  rm -rf "$sres" "$sck"
  mkdir -p "$sres"
  local smoke_set=(--set "model.adapter=$RUN3_ADAPTER" --set "grpo.output_dir=$sck" --set run_name=smoke-run4
    --set "tasks=choose-date,right-angle" --set grpo.prompts_per_step=4 --set grpo.num_generations=4
    --set grpo.dynamic_sampling.max_rounds=3 --set grpo.dynamic_sampling.min_mixed_groups=1
    --set grpo.save_steps=2 --set grpo.eval_every=2 --set grpo.eval_num_seeds=2
    --set grpo.samples_every=1 --set grpo.warmup_steps=1 --set wandb.mode=disabled
    --set "env.pool_size=$(pool_size)")
  say "Smoke GRPO: 2 steps from the run-3 adapter."
  run env CUA_RESULTS_DIR="$sres" timeout 2400 python -m cua.nano_grpo.loop --config "$GRPO_CONFIG" \
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "${smoke_set[@]}" --set grpo.max_steps=2
  say "Smoke GRPO: 1 more step with --resume."
  run env CUA_RESULTS_DIR="$sres" timeout 1500 python -m cua.nano_grpo.loop --config "$GRPO_CONFIG" \
    ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"} "${smoke_set[@]}" --set grpo.max_steps=3 --resume 2>&1 | tee "$sres/resume.log"
  say "Smoke eval: 3 seeds, run 4's settings, smoke adapter."
  run env CUA_RESULTS_DIR="$sres" timeout 900 python -m cua.eval --config "$EVAL_CONFIG" --no-wandb \
    --set eval.mode=greedy --set "eval.tasks=choose-date,use-slider" --set eval.num_seeds=3 \
    --set eval.gif_seeds=1 --set run_name=smoke-eval --set "model.adapter=$sck/final" \
    --set "env.pool_size=$(pool_size)" ${GLOBAL_SET[@]+"${GLOBAL_SET[@]}"}
  run python -m cua.pipeline metrics-check "$sres/train/smoke-run4/metrics.jsonl"
  SRES="$sres" SCK="$sck" python - <<'EOF'
import json, math, os
from pathlib import Path
res, ck = Path(os.environ["SRES"]), Path(os.environ["SCK"])
rows = [json.loads(l) for l in (res / "train/smoke-run4/metrics.jsonl").read_text().splitlines() if l.strip()]
train = [r for r in rows if r.get("kind") == "train"]
evals = [r for r in rows if r.get("kind") == "eval"]
assert [r["step"] for r in train] == [1, 2, 3], f"train steps {[r['step'] for r in train]}, want [1, 2, 3]"
assert evals and evals[0]["step"] == 2 and "eval/success" in evals[0], f"eval rows {evals}"
state = json.loads((ck / "checkpoint-3/trainer_state.json").read_text())
assert state["step"] == 3 and "sampler" in state, state.keys()
assert (ck / "final/adapter_model.safetensors").exists()
best = json.loads((ck / "best/best.json").read_text())
assert best["step"] == 2 and (ck / "best/adapter_config.json").exists(), best
assert "resumed from" in (res / "resume.log").read_text(), "the second run didn't resume"
for n in (1, 2, 3):
    ep = json.loads((res / f"samples/step-{n:06d}/episode.json").read_text())
    assert ep["turns"] and ep["turns"][0]["image"], ep
trained = [r for r in train if r.get("episodes/trained", 0) > 0]
# The ratio cap and the sampler checks only run in the train phase, so it must run here.
assert trained, "no smoke step had a mixed group, so the ratio cap and sampler checks never ran"
for key in ("train/ratio_capped_frac", "train/ratio_over_cap_frac", "train/tis_weight_mean",
            "train/tis_capped_frac", "sampler/dropped_episodes", "sampler/mean_abs_logp_diff"):
    vals = [r.get(key) for r in trained]
    assert all(isinstance(v, (int, float)) for v in vals), f"{key}: {vals}"
assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in trained)
if not any(r.get("train/sequences", 0) > r.get("episodes/trained", 0) for r in trained):
    print("WARNING: no trained episode passed 4 turns, so no trained episode had a second segment. "
          "cua.smoke4 context checked the window; the grpo phase is its first training test.")
s = json.loads((res / "eval/smoke-eval/summary.json").read_text())
assert s["track"] == "B" and s["max_turns"] == 20 and s["context_window_turns"] == 4, s
assert s["track_b_extra_tools"] is True and s["n_crashed"] < 6 and s["overall_success"] is not None, s
for r in train:
    print(f"step {r['step']}: rollout {r['time/rollout_s']:.0f} s, train {r['time/train_s']:.0f} s, "
          f"trained episodes {r.get('episodes/trained', 0):.0f}, sequences {r.get('train/sequences', 0):.0f}, "
          f"turns {r.get('turns/mean', float('nan')):.1f}, ratio capped {r.get('train/ratio_capped_frac', float('nan')):.4f}, "
          f"dropped {r.get('sampler/dropped_episodes', 0):.0f}, mismatch {r.get('sampler/mean_abs_logp_diff', float('nan')):.3f}")
print(f"smoke eval: success {s['overall_success']}, format errors {s['format_error_rate']}")
print("SMOKE OK")
EOF
  rm -rf "$sck"
}

phase_screen() {
  need_run3_adapter
  [[ -f "$RES/candidates.json" ]] || run python -m cua.run_pool --run run4 registered --out "$RES/candidates.json"
  local tasks
  tasks="$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["tasks"]))' "$RES/candidates.json")"
  # No turn budgets yet: they come from this screen, so every task gets 20 turns here.
  local saved="$TURN_BUDGETS"; TURN_BUDGETS=0
  eval_cmd baseline baseline "$tasks" --set "model.adapter=$RUN3_ADAPTER"
  TURN_BUDGETS="$saved"
}

phase_select() {
  run python -m cua.run_pool --run run4 select --baseline "$RES/eval/baseline/per_task.csv" \
    --candidates "$RES/candidates.json"
  if [[ "$TURN_BUDGETS" == "1" ]]; then
    run python -m cua.run_pool --run run4 budgets --episodes "$RES/eval/baseline/episodes.jsonl" \
      --out "$BUDGETS_FILE" --max-turns 20 --floor 10
  else
    echo "TURN_BUDGETS=0: every task gets 20 turns."
  fi
  # Held-out v2 is run 2's list, unchanged, so held-out numbers compare across runs.
  python - <<'EOF'
import json
from pathlib import Path
from cua.config import load_tasks
frozen = load_tasks().heldout_v2
src = Path("results/run3/heldout_v2.json")
if src.exists():
    theirs = json.loads(src.read_text())["heldout_v2"]
    assert theirs == frozen, f"run 3's heldout_v2.json {theirs} differs from configs/tasks.yaml {frozen}"
Path("results/run4/heldout_v2.json").write_text(json.dumps({"heldout_v2": frozen, "source": str(src)}, indent=2))
print(f"Held-out v2: {frozen}")
EOF
}

phase_rft() {
  # Run 4 has no RFT (the Track B format-error rate is near 0). The phase stays
  # so the phase list matches run 3's contract.
  need_run3_adapter
  [[ "$RFT" == "off" || "$RFT" == "auto" ]] || { echo "Run 4 has no RFT config; RFT must be off."; return 1; }
  echo "$RUN3_ADAPTER" > "$PH/start_adapter"
  echo "No RFT in run 4: GRPO starts from the run-3 adapter."
}

phase_pre_eval() {
  need_run3_adapter
  eval_cmd run3-heldout4 greedy heldout_v2 --set eval.num_seeds=100 --set "model.adapter=$RUN3_ADAPTER"
  eval_cmd run3-full4    greedy full_suite --set eval.num_seeds=20  --set "model.adapter=$RUN3_ADAPTER"
  local base; base="$(base_eval_mode)"
  echo "Base-model evals: $base (BASE_EVAL=$BASE_EVAL, PRICE_PER_HR=$PRICE_PER_HR)."
  if [[ "$base" == "heldout" || "$base" == "full" ]]; then
    eval_cmd base-heldout4 greedy heldout_v2 --set eval.num_seeds=100
  fi
  if [[ "$base" == "full" ]]; then
    eval_cmd base-full4 greedy full_suite --set eval.num_seeds=20
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
    [[ -n "$(grpo_final_adapter)" ]] || { echo "No checkpoint in $GRPO_OUT."; return 1; }
    return 0
  fi
  echo "GRPO time: $used s used, $left s left of $cap s (GRPO_MAX_HOURS=$GRPO_MAX_HOURS)."
  # Record GRPO time every minute. Its output goes to /dev/null, so tee doesn't wait for it.
  ( t0="$(date +%s)"
    while sleep 60; do echo $((used + $(date +%s) - t0)) > "$PH/grpo.elapsed_s.tmp" && mv "$PH/grpo.elapsed_s.tmp" "$PH/grpo.elapsed_s"; done
  ) >/dev/null 2>&1 &
  local ticker=$! t_start; t_start="$(date +%s)"
  set_run4_sets
  # SIGINT reaches Python as KeyboardInterrupt; nano_grpo saves a checkpoint and exits.
  local cmd=(timeout --signal=INT --kill-after=300 "${left}s"
    python -m cua.nano_grpo.loop --config "$GRPO_CONFIG" "${RUN4_SETS[@]}"
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

# Chooses the adapter post_eval reports: BEST_CKPT, else final vs best (cua.pipeline
# choose-adapter), and records the choice in results/run4/adapter_choice.json.
choose_adapter() {
  rm -f "$RES/adapter_choice.json"   # never report a stale choice
  if [[ -n "${BEST_CKPT:-}" ]]; then
    printf '{"choice": "BEST_CKPT", "adapter": "%s"}\n' "$BEST_CKPT" > "$RES/adapter_choice.json"
    echo "$BEST_CKPT"; return
  fi
  local final; final="$(grpo_final_adapter)"
  [[ -n "$final" ]] || return 1
  local best="$GRPO_OUT/best"
  if [[ -f "$best/best.json" && -f "$best/adapter_config.json" ]]; then
    # Score the final adapter the way the mid-training evals scored best/: greedy,
    # pool tasks, 20 baseline-range seeds (not the headline seeds).
    eval_cmd select-final greedy selected --set eval.seed_range=baseline --set eval.num_seeds=20 \
      --set eval.gif_seeds=0 --set "model.adapter=$final" >&2
  fi
  # Without best/, this records "final".
  python -m cua.pipeline choose-adapter --final "$final" --best "$best" \
    --final-eval "$RES/eval/select-final/summary.json" --out "$RES/adapter_choice.json"
}

phase_post_eval() {
  local adapter; adapter="$(choose_adapter | tail -n1)"
  [[ -n "$adapter" && -f "$adapter/adapter_config.json" ]] || { echo "No run-4 adapter in $GRPO_OUT. Set BEST_CKPT=PATH."; return 1; }
  if [[ -f "$RES/adapter_choice.json" ]]; then cat "$RES/adapter_choice.json"; echo; fi
  echo "Evaluating adapter: $adapter"
  echo "$adapter" > "$PH/final_adapter"
  eval_cmd run4-heldout4 greedy heldout_v2 --set eval.num_seeds=100 --set "model.adapter=$adapter"
  eval_cmd run4-full4    greedy full_suite --set eval.num_seeds=20  --set "model.adapter=$adapter"
}

phase_compare() {
  local out="$RES/compare" e="$RES/eval"
  mkdir -p "$out"
  compare() {  # compare NAME A_DIR B_DIR [cua.compare flags...]
    local name="$1" a="$2" b="$3"; shift 3
    if [[ -f "$a/episodes.jsonl" && -f "$b/episodes.jsonl" ]]; then
      run python -m cua.compare "$a" "$b" --json "$out/$name.json" "$@" | tee "$out/$name.md"
    else
      echo "Skipping $name: missing $a or $b."
    fi
  }
  compare heldout4-run4-vs-run3        "$e/run3-heldout4" "$e/run4-heldout4"
  compare heldout4-no-sign-run4-vs-run3 "$e/run3-heldout4" "$e/run4-heldout4" --exclude-tasks sign-agreement
  compare full4-run4-vs-run3           "$e/run3-full4"    "$e/run4-full4"
  compare full4-pool-run4-vs-run3      "$e/run3-full4"    "$e/run4-full4" --tasks selected
  compare full4-turn-limited-run4-vs-run3 "$e/run3-full4" "$e/run4-full4" --tasks run4_turn_limited
  compare full4-new-action-run4-vs-run3 "$e/run3-full4"   "$e/run4-full4" --tasks run4_new_action
  compare full4-no-pool-run4-vs-run3   "$e/run3-full4"    "$e/run4-full4" --exclude-tasks selected
  compare heldout4-run3-vs-base        "$e/base-heldout4" "$e/run3-heldout4"
  compare heldout4-run4-vs-base        "$e/base-heldout4" "$e/run4-heldout4"
  compare full4-run3-vs-base           "$e/base-full4"    "$e/run3-full4"
  compare full4-run4-vs-base           "$e/base-full4"    "$e/run4-full4"
  return 0
}

phase_report() {
  run python -m cua.report --preset run4
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
    echo "Fix the problem, then resume with: bash scripts/run4.sh"
    exit "$rc"
  fi
  now_iso > "$PH/$name.done"
}

show_status() {
  local p
  echo "Run 4 | GRPO config: $GRPO_CONFIG | results: $RES | browsers: $(pool_size)"
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
    echo "run4.sh is already running (PID $other)." >&2
    exit 1
  fi
fi
echo "$$" > "$PH/.pid"
trap 'rm -f "$PH/.pid" "$PH/.current"' EXIT

say "Run 4 | run-3 adapter: $RUN3_ADAPTER | config: $GRPO_CONFIG | GRPO cap: ${GRPO_MAX_HOURS} h | BASE_EVAL=$BASE_EVAL | TURN_BUDGETS=$TURN_BUDGETS | price: \$$PRICE_PER_HR/h"

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
