"""Build the run-3 training pool: a mix of both earlier pools plus Track-B tasks.

Command line:

    python -m cua.run3_pool registered --out results/run3/candidates.json
    python -m cua.run3_pool select --baseline results/run3/eval/baseline/per_task.csv

`registered` keeps the run-3 candidates that browsergym-miniwob registers, so
a misspelled or missing task can't crash the screening eval. `select` reads the
sampled screen of the run-2 adapter (Track B, 16 seeds) and writes
`selected_tasks.json` in the same format as `cua.select_tasks`, plus the fields
`learn`, `replay`, and `source`.

Why a mix: run 2 trained only on its own pool, and run 3 changes the action
space. Training only on new tasks risks forgetting the old ones. The pool has
two parts:

- Learn tasks: in the 5%-85% band, where groups are often mixed. Each source
  (run-1 pool, run-2 pool, Track-B extras) gets a minimum share, so the pool
  doesn't collapse onto one family. The rest of the slots go to the tasks
  nearest 45% success.
- Replay tasks: tasks from the run-1 and run-2 pools that the model already
  solves (above 85%). Their groups are mostly all-success, so they cost
  rollouts but no training. The adaptive sampler gives them a small floor
  weight. If one starts to fail, its groups turn mixed, its weight rises, and
  GRPO trains on it again: replay doubles as a forgetting alarm.

The module is pure Python, so the tests run without a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from cua.config import load_tasks, results_dir, selected_tasks_path
from cua.select_tasks import TARGET_RATE, read_rates

# Source groups, in priority order. A task in two groups counts for the first.
SOURCES = ("run2_pool", "run1_pool", "trackb_extra", "other")


def task_sources(spec=None) -> dict[str, str]:
    """Maps every run-3 candidate to its source group."""
    spec = spec or load_tasks()
    groups = {
        "run2_pool": spec.run2_pool,
        "run1_pool": spec.run1_pool,
        "trackb_extra": spec.train_candidates_run3_extra,
        "other": spec.train_candidates_run3,
    }
    out: dict[str, str] = {}
    for name in SOURCES:
        for task in groups[name]:
            out.setdefault(task, name)
    return out


def mix_pool(
    rates: dict[str, tuple[int, float]],
    sources: dict[str, str],
    min_success: float = 0.05,
    max_success: float = 0.85,
    max_learn: int = 16,
    min_learn: int = 10,
    min_per_source: int = 3,
    max_replay: int = 4,
    target: float = TARGET_RATE,
) -> dict:
    """Chooses learn and replay tasks. Returns the JSON-ready selection.

    Args:
      rates: {task: (n, success_rate)} from the screening eval.
      sources: {task: source group}, from `task_sources`. Only these tasks are
        candidates.
      min_success, max_success: The learn band, inclusive.
      max_learn: Most learn tasks.
      min_learn: If fewer tasks are in the band, add the nearest tasks with at
        least one success until there are this many.
      min_per_source: Learn tasks reserved for each of run2_pool, run1_pool,
        and trackb_extra, when that many are in the band.
      max_replay: Most replay tasks.
      target: Preferred success rate when ranking.
    """
    present = [t for t in sources if t in rates]
    dropped: dict[str, str] = {t: "missing from the screen (no valid episodes)" for t in sources if t not in rates}

    def rate(t: str) -> float:
        return rates[t][1]

    def closeness(t: str) -> tuple[float, str]:
        return (abs(rate(t) - target), t)

    in_band = [t for t in present if min_success <= rate(t) <= max_success]
    learn: list[str] = []
    for src in ("run2_pool", "run1_pool", "trackb_extra"):
        ranked = sorted((t for t in in_band if sources[t] == src), key=closeness)
        learn += ranked[: min(min_per_source, max_learn - len(learn))]
    rest = sorted((t for t in in_band if t not in learn), key=closeness)
    for t in rest:
        if len(learn) >= max_learn:
            dropped[t] = f"over cap of {max_learn} learn tasks (rate {rate(t):.2f} is far from {target})"
            continue
        learn.append(t)

    widened: list[str] = []
    if len(learn) < min_learn:
        def band_distance(t: str) -> float:
            return min_success - rate(t) if rate(t) < min_success else rate(t) - max_success

        nearest = sorted(
            (t for t in present if t not in learn and 0.0 < rate(t) < 1.0),
            key=lambda t: (band_distance(t), t),
        )
        for t in nearest[: min_learn - len(learn)]:
            learn.append(t)
            widened.append(t)

    # Replay: solved tasks from the earlier pools, most fragile first, taken
    # alternately from the two pools so both stay represented.
    queues = {
        src: sorted(
            (t for t in present if sources[t] == src and rate(t) > max_success and t not in learn),
            key=lambda t: (rate(t), t),
        )
        for src in ("run2_pool", "run1_pool")
    }
    replay: list[str] = []
    while len(replay) < max_replay and any(queues.values()):
        for src in ("run2_pool", "run1_pool"):
            if queues[src] and len(replay) < max_replay:
                replay.append(queues[src].pop(0))
    for src_queue in queues.values():
        for t in src_queue:
            dropped.setdefault(t, f"solved (rate {rate(t):.2f}); over the cap of {max_replay} replay tasks")

    selected = learn + replay
    for t in present:
        if t in selected or t in dropped:
            continue
        r = rate(t)
        if r == 0.0:
            dropped[t] = "zero successes"
        elif r < min_success:
            dropped[t] = f"too hard (rate {r:.2f} < {min_success})"
        else:
            dropped[t] = f"too easy (rate {r:.2f} > {max_success}) and not in an earlier pool"

    return {
        "selected": selected,
        "learn": learn,
        "replay": replay,
        "rates": {t: rate(t) for t in selected},
        "source": {t: sources[t] for t in selected},
        "dropped": dropped,
        "criteria": {
            "min_success": min_success,
            "max_success": max_success,
            "max_learn": max_learn,
            "min_learn": min_learn,
            "min_per_source": min_per_source,
            "max_replay": max_replay,
            "target_rate": target,
            "widened_beyond_band": widened,
            "screen_n": {t: rates[t][0] for t in present},
        },
    }


def registered_tasks(names: list[str]) -> tuple[list[str], list[str]]:
    """Splits names into (registered, missing) for browsergym-miniwob. Needs BrowserGym."""
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401  (registers the environments)

    ok, missing = [], []
    for name in names:
        try:
            gym.spec(f"browsergym/miniwob.{name}")
            ok.append(name)
        except Exception:  # noqa: BLE001 - gymnasium raises several error types
            missing.append(name)
    return ok, missing


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the run-3 task pool.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("registered", help="Write the run-3 candidates that BrowserGym registers.")
    reg.add_argument("--out", required=True)
    sel = sub.add_parser("select", help="Mix the pool from the screening eval.")
    sel.add_argument("--baseline", default=None, help="per_task.csv of the screen. Default: <results>/eval/baseline.")
    sel.add_argument("--candidates", default=None, help="JSON from `registered`. Default: every run-3 candidate.")
    sel.add_argument("--out", default=None, help="Default: <results>/selected_tasks.json.")
    sel.add_argument("--max-learn", type=int, default=16)
    sel.add_argument("--max-replay", type=int, default=4)
    sel.add_argument("--min-per-source", type=int, default=3)
    sel.add_argument("--min-success", type=float, default=0.05)
    sel.add_argument("--max-success", type=float, default=0.85)
    args = parser.parse_args(argv)

    if args.cmd == "registered":
        names = load_tasks().train_candidates_run3
        ok, missing = registered_tasks(names)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"tasks": ok, "missing": missing}, indent=2))
        print(f"{len(ok)} of {len(names)} run-3 candidates are registered.")
        if missing:
            print(f"Not registered, left out: {', '.join(missing)}")
        if not ok:
            raise SystemExit("No run-3 candidate is registered. Check the browsergym-miniwob install.")
        return

    baseline = Path(args.baseline) if args.baseline else results_dir() / "eval" / "baseline" / "per_task.csv"
    if not baseline.exists():
        raise SystemExit(f"{baseline} not found. Run the screen phase first.")
    sources = task_sources()
    if args.candidates:
        allowed = set(json.loads(Path(args.candidates).read_text())["tasks"])
        sources = {t: s for t, s in sources.items() if t in allowed}
    result = mix_pool(
        read_rates(baseline), sources,
        min_success=args.min_success, max_success=args.max_success,
        max_learn=args.max_learn, min_per_source=args.min_per_source, max_replay=args.max_replay,
    )
    for t in result["selected"]:
        kind = "replay" if t in result["replay"] else "learn"
        print(f"{t:28s} {result['rates'][t]:6.2f}  {kind:6s}  {result['source'][t]}")
    out = Path(args.out) if args.out else selected_tasks_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Selected {len(result['learn'])} learn and {len(result['replay'])} replay tasks. Wrote {out}")
    if len(result["learn"]) < 6:
        raise SystemExit(f"Only {len(result['learn'])} learn tasks; GRPO would have too little to learn from.")


if __name__ == "__main__":
    main()
