"""Pick the training task pool from the sampled-baseline results.

Command line:

    python -m cua.select_tasks --baseline results/eval/baseline/per_task.csv

Why this step matters: GRPO learns from groups of episodes that differ in
outcome. A task the base model always solves (rate 1.0) or never solves (rate
0.0) gives every episode in a group the same reward, so the group contributes
no gradient. Tasks in the middle, where the model sometimes succeeds, teach the
most. The thresholds live in `configs/eval.yaml` under `select:`.

Output: `results/selected_tasks.json`, which `cua.config.load_selected_tasks` reads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from cua.config import add_config_args, config_from_args, load_tasks, results_dir, selected_tasks_path

# Among tasks that qualify, prefer rates near this value when we must cap the
# list. Near 0.5 the success and failure outcomes are balanced, which gives the
# largest reward variance inside a group.
TARGET_RATE = 0.45


def read_rates(path: Path) -> dict[str, tuple[int, float]]:
    """Read `per_task.csv` into {task: (n, success_rate)}. Skips tasks with no valid episodes."""
    rates: dict[str, tuple[int, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("success_rate"):  # blank: every episode crashed
                continue
            n = int(float(row.get("n") or 0))
            if n > 0:
                rates[row["task"]] = (n, float(row["success_rate"]))
    return rates


def select(
    rates: dict[str, tuple[int, float]],
    candidates: list[str],
    min_success: float,
    max_success: float,
    min_tasks: int,
    max_tasks: int,
) -> dict:
    """Apply the selection rules and return the JSON-ready result dict."""
    dropped: dict[str, str] = {}
    for task in candidates:
        if task not in rates:
            dropped[task] = "missing from baseline (no valid episodes)"

    def band_distance(rate: float) -> float:
        """How far a rate is from the [min, max] band. Zero inside the band."""
        if rate < min_success:
            return min_success - rate
        if rate > max_success:
            return rate - max_success
        return 0.0

    present = [t for t in candidates if t in rates]
    selected = [t for t in present if band_distance(rates[t][1]) == 0.0]
    widened: list[str] = []

    if len(selected) < min_tasks:
        # Not enough tasks in the band: add the nearest ones. Tasks the model
        # never solved are the last resort, because with zero successes a
        # group has no positive example to learn from.
        rest = [t for t in present if t not in selected]
        nonzero = sorted((t for t in rest if rates[t][1] > 0), key=lambda t: (band_distance(rates[t][1]), abs(rates[t][1] - TARGET_RATE)))
        zero = sorted((t for t in rest if rates[t][1] == 0), key=lambda t: t)
        for task in nonzero + zero:
            if len(selected) >= min_tasks:
                break
            selected.append(task)
            widened.append(task)

    if len(selected) > max_tasks:
        # Keep the tasks whose rates are closest to the target.
        ranked = sorted(selected, key=lambda t: abs(rates[t][1] - TARGET_RATE))
        keep = set(ranked[:max_tasks])
        for task in selected:
            if task not in keep:
                dropped[task] = f"over cap of {max_tasks} (rate {rates[task][1]:.2f} is far from {TARGET_RATE})"
        selected = [t for t in selected if t in keep]

    for task in present:
        if task in selected or task in dropped:
            continue
        rate = rates[task][1]
        if rate < min_success:
            dropped[task] = f"too hard (rate {rate:.2f} < {min_success})" if rate > 0 else "zero successes"
        else:
            dropped[task] = f"too easy (rate {rate:.2f} > {max_success})"

    return {
        "selected": selected,
        "rates": {t: rates[t][1] for t in selected},
        "dropped": dropped,
        "criteria": {
            "min_success": min_success,
            "max_success": max_success,
            "min_tasks": min_tasks,
            "max_tasks": max_tasks,
            "target_rate": TARGET_RATE,
            "widened_beyond_band": widened,
            "baseline_n": {t: rates[t][0] for t in present},
        },
    }


def print_table(result: dict, rates: dict[str, tuple[int, float]]) -> None:
    print(f"{'task':28s} {'n':>4s} {'rate':>6s}  status")
    selected = set(result["selected"])
    widened = set(result["criteria"]["widened_beyond_band"])
    for task, (n, rate) in sorted(rates.items(), key=lambda kv: kv[1][1]):
        if task in selected:
            status = "selected (widened)" if task in widened else "selected"
        else:
            status = "dropped: " + result["dropped"].get(task, "")
        print(f"{task:28s} {n:4d} {rate:6.2f}  {status}")
    for task, reason in result["dropped"].items():
        if task not in rates:
            print(f"{task:28s} {'-':>4s} {'-':>6s}  dropped: {reason}")
    print(f"\nSelected {len(selected)} tasks.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Choose training tasks from the sampled baseline and write results/selected_tasks.json."
    )
    add_config_args(parser, "configs/eval.yaml")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to the baseline per_task.csv. Default: results/eval/baseline/per_task.csv.",
    )
    parser.add_argument("--out", default=None, help="Output JSON path. Default: results/selected_tasks.json.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    baseline = Path(args.baseline) if args.baseline else results_dir(cfg) / "eval" / "baseline" / "per_task.csv"
    if not baseline.exists():
        raise SystemExit(f"{baseline} not found. Run the sampled baseline first (eval.mode=baseline).")

    sel = cfg.select
    rates = read_rates(baseline)
    result = select(
        rates,
        load_tasks().train_candidates,
        float(sel.min_success),
        float(sel.max_success),
        int(sel.min_tasks),
        int(sel.max_tasks),
    )
    print_table(result, rates)

    out = Path(args.out) if args.out else selected_tasks_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    if len(result["selected"]) < int(sel.min_tasks):
        print(f"WARNING: only {len(result['selected'])} tasks available; fewer than min_tasks={sel.min_tasks}.")


if __name__ == "__main__":
    main()
