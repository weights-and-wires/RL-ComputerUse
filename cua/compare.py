"""Compare two eval runs on the same (task, seed) pairs, with a paired bootstrap.

Command line:

    python -m cua.compare results/run2/eval/run1-pool2 results/run2/eval/run2-pool2
    python -m cua.compare A_DIR B_DIR --json results/run2/compare/pool2.json
    python -m cua.compare A_DIR B_DIR --tasks run4_turn_limited       # a subset of tasks
    python -m cua.compare A_DIR B_DIR --exclude-tasks sign-agreement  # all tasks but these

Each argument is an eval output folder that holds `episodes.jsonl`. The script
pairs episodes by (task, seed), using sample 0, and drops a pair when either
episode crashed. It reports, per task and overall, the success rate of A and of
B, the difference B - A in percentage points, and a 95% interval for that
difference.

Why paired: both models face the identical page for a given seed, so most of
the seed-to-seed difficulty cancels out. A paired interval is narrower than
comparing two independent success rates, so smaller real gains become visible.

How the interval works: resample seeds with replacement within each task (a
stratified bootstrap), recompute the difference, repeat 10,000 times, and take
the 2.5th and 97.5th percentiles. "Overall" weights each task equally, which
matches the overall success rate when every task has the same number of seeds.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional


def load_outcomes(run_dir: Path) -> dict[tuple[str, int], bool]:
    """Reads {(task, seed): success} from a run's episodes.jsonl. Crashed episodes are left out."""
    path = Path(run_dir) / "episodes.jsonl"
    out: dict[tuple[str, int], bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        if int(ep.get("sample_idx", 0)) != 0 or ep.get("crashed"):
            continue
        out[(ep["task"], int(ep["seed"]))] = bool(ep["success"])
    return out


def filter_tasks(
    outcomes: dict[tuple[str, int], bool], keep: Optional[set] = None, drop: Optional[set] = None
) -> dict[tuple[str, int], bool]:
    """Keeps only the tasks in `keep` (when given) and removes those in `drop`."""
    return {k: v for k, v in outcomes.items()
            if (keep is None or k[0] in keep) and (drop is None or k[0] not in drop)}


def pair(a: dict[tuple[str, int], bool], b: dict[tuple[str, int], bool]) -> dict[str, list[tuple[int, int]]]:
    """Returns {task: [(a_success, b_success), ...]} over the (task, seed) keys both runs have."""
    by_task: dict[str, list[tuple[int, int]]] = {}
    for key in sorted(a.keys() & b.keys()):
        by_task.setdefault(key[0], []).append((int(a[key]), int(b[key])))
    return by_task


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def paired_bootstrap(
    by_task: dict[str, list[tuple[int, int]]], n_boot: int = 10_000, seed: int = 0
) -> dict:
    """Per-task and overall difference (B - A) with 95% stratified bootstrap intervals."""
    rng = random.Random(seed)
    tasks = sorted(by_task)
    result: dict = {"tasks": {}, "overall": {}}
    if not tasks:
        return result

    def diff(pairs: list[tuple[int, int]]) -> float:
        return sum(b - a for a, b in pairs) / len(pairs)

    boots: dict[str, list[float]] = {t: [] for t in tasks}
    overall_boot: list[float] = []
    for _ in range(n_boot):
        per_task = []
        for t in tasks:
            pairs = by_task[t]
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            d = diff(sample)
            boots[t].append(d)
            per_task.append(d)
        overall_boot.append(sum(per_task) / len(per_task))

    for t in tasks:
        pairs = by_task[t]
        vals = sorted(boots[t])
        result["tasks"][t] = {
            "n": len(pairs),
            "a": sum(a for a, _ in pairs) / len(pairs),
            "b": sum(b for _, b in pairs) / len(pairs),
            "diff": diff(pairs),
            "ci95": [_percentile(vals, 0.025), _percentile(vals, 0.975)],
            # Discordant pairs: how many seeds flipped each way.
            "b_only": sum(1 for a, b in pairs if b and not a),
            "a_only": sum(1 for a, b in pairs if a and not b),
        }
    rows = result["tasks"].values()
    vals = sorted(overall_boot)
    result["overall"] = {
        "n": sum(r["n"] for r in rows),
        "a": sum(r["a"] for r in rows) / len(tasks),
        "b": sum(r["b"] for r in rows) / len(tasks),
        "diff": sum(r["diff"] for r in rows) / len(tasks),
        "ci95": [_percentile(vals, 0.025), _percentile(vals, 0.975)],
    }
    return result


def format_table(result: dict, name_a: str, name_b: str) -> str:
    """Markdown table of the comparison, in percentage points."""
    lines = [
        f"| Task | n | {name_a} | {name_b} | Difference | 95% interval | Flips (+/-) |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(label: str, r: dict, flips: str) -> str:
        lo, hi = r["ci95"]
        return (
            f"| {label} | {r['n']} | {100 * r['a']:.1f}% | {100 * r['b']:.1f}% | "
            f"{100 * r['diff']:+.1f} | {100 * lo:+.1f} to {100 * hi:+.1f} | {flips} |"
        )

    for task, r in result["tasks"].items():
        lines.append(row(task, r, f"{r['b_only']}/{r['a_only']}"))
    if result["overall"]:
        lines.append(row("**Overall**", result["overall"], ""))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two eval runs (B minus A) on shared task seeds.")
    parser.add_argument("run_a", help="Eval folder of the reference model (holds episodes.jsonl).")
    parser.add_argument("run_b", help="Eval folder of the model being compared.")
    parser.add_argument("--json", default=None, help="Also write the result to this JSON file.")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap resamples. Default: 10000.")
    parser.add_argument("--tasks", default=None,
                        help="Compare only these tasks: comma-separated names or a task-list name (cua.config).")
    parser.add_argument("--exclude-tasks", default=None, help="Leave out these tasks (same forms as --tasks).")
    args = parser.parse_args(argv)

    a, b = load_outcomes(Path(args.run_a)), load_outcomes(Path(args.run_b))
    if args.tasks or args.exclude_tasks:
        from cua.config import resolve_task_list

        keep = set(resolve_task_list(args.tasks)) if args.tasks else None
        drop = set(resolve_task_list(args.exclude_tasks)) if args.exclude_tasks else None
        a, b = filter_tasks(a, keep, drop), filter_tasks(b, keep, drop)
    by_task = pair(a, b)
    if not by_task:
        raise SystemExit("The two runs share no (task, seed) pairs.")
    result = paired_bootstrap(by_task, n_boot=args.n_boot)
    name_a, name_b = Path(args.run_a).name, Path(args.run_b).name
    unpaired = len(a.keys() ^ b.keys())
    print(format_table(result, name_a, name_b))
    if unpaired:
        print(f"\n{unpaired} (task, seed) pairs appear in only one run or crashed, and are left out.")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"a": args.run_a, "b": args.run_b, "tasks": args.tasks,
                                   "exclude_tasks": args.exclude_tasks, **result}, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
