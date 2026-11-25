"""Build a run's training pool from a screen, and run 4's eval task lists.

Command line:

    python -m cua.run_pool registered --run run4 --out results/run4/candidates.json
    python -m cua.run_pool select --run run4 --baseline results/run4/eval/baseline/per_task.csv
    python -m cua.run_pool budgets --episodes results/run4/eval/baseline/episodes.jsonl \\
        --out results/run4/turn_budgets.json

`registered` keeps the run's candidates that browsergym-miniwob registers, so
a misspelled or missing task can't crash the screening eval. For run 4 it also
writes `full_suite.json` next to `--out`: every registered task except
`full_suite_exclude` in configs/tasks.yaml.

`select` reads the sampled screen of the start adapter and writes
`selected_tasks.json` in the format of `cua.select_tasks`, plus the fields
`learn`, `replay`, and `source`. `budgets` writes per-task turn limits.

The pool has two parts:

- Learn tasks: in the 5%-85% band, where groups are often mixed. Some
  sources get reserved slots, so the pool doesn't collapse onto one family.
  The rest of the slots go to the tasks nearest 45% success.
- Replay tasks: tasks from earlier pools that the model already solves (above
  85%). Their groups are mostly all-success, so they cost rollouts but no
  training. The adaptive sampler gives them a small floor weight. If one
  starts to fail, its groups turn mixed and GRPO trains on it again: replay
  doubles as a forgetting alarm.

Run 3 reserves 3 slots each for the run-2 pool, the run-1 pool, and the
Track-B extras, and replays the run-1 and run-2 pools (`cua.run3_pool`). Run 4
reserves slots for the drag tasks (`new_action`, where one success in the
screen is enough) and the long-horizon tasks, fills the rest by closeness to
45% from any source, and replays the run-3 pool.

The selection code is pure Python, so the tests run without a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

from cua.config import TaskSpec, load_tasks, results_dir
from cua.select_tasks import TARGET_RATE, read_rates

# Source groups per run, in priority order. A task in two groups counts for the first.
RUN_SOURCES: dict[str, tuple[str, ...]] = {
    "run3": ("run2_pool", "run1_pool", "trackb_extra", "other"),
    "run4": ("new_action", "long_horizon", "run3_pool", "other"),
}

# Run 4's pool settings (docs/PLAN_RUN4.md).
RUN4_RESERVED = (("new_action", 4), ("long_horizon", 5))
RUN4_REPLAY_SOURCES = ("run3_pool",)
RUN4_MIN_SUCCESS_BY_SOURCE = {"new_action": 0.01}  # any success in the screen gives GRPO a signal
RUN4_MAX_LEARN = 18
RUN4_MAX_REPLAY = 3


def candidates(run: str, spec: Optional[TaskSpec] = None) -> list[str]:
    """The screening candidates of a run."""
    spec = spec or load_tasks()
    if run == "run3":
        return spec.train_candidates_run3
    if run == "run4":
        return spec.train_candidates_run4
    raise ValueError(f"Unknown run {run!r}. Use run3 or run4.")


def task_sources(run: str = "run3", spec: Optional[TaskSpec] = None) -> dict[str, str]:
    """Maps every candidate of a run to its source group."""
    spec = spec or load_tasks()
    if run == "run3":
        groups = {
            "run2_pool": spec.run2_pool,
            "run1_pool": spec.run1_pool,
            "trackb_extra": spec.train_candidates_run3_extra,
            "other": spec.train_candidates_run3,
        }
    elif run == "run4":
        groups = {
            "new_action": spec.train_candidates_run4_new_action,
            "long_horizon": spec.run4_long_horizon,
            "run3_pool": spec.run3_pool,
            "other": spec.train_candidates_run4,
        }
    else:
        raise ValueError(f"Unknown run {run!r}. Use run3 or run4.")
    allowed = set(candidates(run, spec))
    out: dict[str, str] = {}
    for name in RUN_SOURCES[run]:
        for task in groups[name]:
            if task in allowed:
                out.setdefault(task, name)
    return out


def mix_pool(
    rates: dict[str, tuple[int, float]],
    sources: dict[str, str],
    min_success: float = 0.05,
    max_success: float = 0.85,
    max_learn: int = 16,
    min_learn: int = 10,
    reserved: Iterable[tuple[str, int]] = (("run2_pool", 3), ("run1_pool", 3), ("trackb_extra", 3)),
    replay_sources: tuple[str, ...] = ("run2_pool", "run1_pool"),
    max_replay: int = 4,
    target: float = TARGET_RATE,
    min_success_by_source: Optional[dict[str, float]] = None,
    exclude: Optional[dict[str, str]] = None,
) -> dict:
    """Chooses learn and replay tasks. Returns the JSON-ready selection.

    Args:
      rates: {task: (n, success_rate)} from the screening eval.
      sources: {task: source group}. Only these tasks are candidates.
      min_success, max_success: The learn band, inclusive.
      max_learn: Most learn tasks.
      min_learn: If fewer tasks are in the band, add the nearest tasks with at
        least one success until there are this many.
      reserved: (source, slots) pairs, in order: learn slots reserved for each
        source when that many of its tasks are in the band.
      replay_sources: Sources whose solved tasks can be replayed, taken in turn.
      max_replay: Most replay tasks.
      target: Preferred success rate when ranking.
      min_success_by_source: A lower band edge per source (run 4: any success
        counts for the drag tasks).
      exclude: {task: reason} never to select.
    """
    floors = dict(min_success_by_source or {})
    exclude = dict(exclude or {})
    dropped: dict[str, str] = {t: f"excluded: {why}" for t, why in exclude.items() if t in sources}
    present = [t for t in sources if t in rates and t not in exclude]
    for t in sources:
        if t not in rates and t not in exclude:
            dropped[t] = "missing from the screen (no valid episodes)"

    def rate(t: str) -> float:
        return rates[t][1]

    def low(t: str) -> float:
        return floors.get(sources[t], min_success)

    def closeness(t: str) -> tuple[float, str]:
        return (abs(rate(t) - target), t)

    in_band = [t for t in present if low(t) <= rate(t) <= max_success]
    learn: list[str] = []
    for src, slots in reserved:
        ranked = sorted((t for t in in_band if sources[t] == src and t not in learn), key=closeness)
        learn += ranked[: max(0, min(slots, max_learn - len(learn)))]
    rest = sorted((t for t in in_band if t not in learn), key=closeness)
    for t in rest:
        if len(learn) >= max_learn:
            dropped[t] = f"over cap of {max_learn} learn tasks (rate {rate(t):.2f} is far from {target})"
            continue
        learn.append(t)

    widened: list[str] = []
    if len(learn) < min_learn:
        def band_distance(t: str) -> float:
            return low(t) - rate(t) if rate(t) < low(t) else rate(t) - max_success

        nearest = sorted(
            (t for t in present if t not in learn and 0.0 < rate(t) < 1.0),
            key=lambda t: (band_distance(t), t),
        )
        for t in nearest[: min_learn - len(learn)]:
            learn.append(t)
            widened.append(t)

    # Replay: solved tasks from the earlier pools, most fragile first, taken
    # alternately from the replay sources so each stays represented.
    queues = {
        src: sorted(
            (t for t in present if sources[t] == src and rate(t) > max_success and t not in learn),
            key=lambda t: (rate(t), t),
        )
        for src in replay_sources
    }
    replay: list[str] = []
    while len(replay) < max_replay and any(queues.values()):
        for src in replay_sources:
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
        elif r < low(t):
            dropped[t] = f"too hard (rate {r:.2f} < {low(t)})"
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
            "reserved": [list(r) for r in reserved],
            "min_success_by_source": floors,
            "replay_sources": list(replay_sources),
            "max_replay": max_replay,
            "target_rate": target,
            "widened_beyond_band": widened,
            "screen_n": {t: rates[t][0] for t in present},
        },
    }


def select_run4(rates: dict[str, tuple[int, float]], sources: dict[str, str],
                spec: Optional[TaskSpec] = None) -> dict:
    """Run 4's pool: reserved slots for drag and long-horizon tasks, replay from the run-3 pool."""
    spec = spec or load_tasks()
    return mix_pool(
        rates, sources, max_learn=RUN4_MAX_LEARN, min_learn=10, reserved=RUN4_RESERVED,
        replay_sources=RUN4_REPLAY_SOURCES, max_replay=RUN4_MAX_REPLAY,
        min_success_by_source=RUN4_MIN_SUCCESS_BY_SOURCE, exclude=spec.run4_exclude,
    )


# ---------------------------------------------------------------------------
# Turn budgets (run 4)
# ---------------------------------------------------------------------------


def turn_budgets_from_episodes(
    episodes: Iterable[dict], max_turns: int = 20, floor: int = 10, min_successes: int = 3
) -> dict:
    """Per-task turn limits from a screen's episodes (`episodes.jsonl` rows).

    A task the screen solved at least `min_successes` times gets
    `2 * p90 + 2` turns, where p90 is the 90th percentile of the turns its
    successes took, clamped to [floor, max_turns]. Other tasks get
    `max_turns`. The floor is run 3's limit, so no task gets fewer turns than
    in run 3. Short tasks stop failing episodes early, which keeps rollout
    rounds from waiting on 20-turn failures; long tasks keep every turn.

    Returns {"budgets": {task: turns}, "max_turns", "floor", "rule"}.
    """
    turns: dict[str, list[int]] = {}
    seen: set[str] = set()
    for ep in episodes:
        if ep.get("crashed"):
            continue
        seen.add(ep["task"])
        if ep.get("success"):
            turns.setdefault(ep["task"], []).append(int(ep["num_turns"]))
    budgets: dict[str, int] = {}
    for task in sorted(seen):
        ok = sorted(turns.get(task, []))
        if len(ok) < min_successes:
            budgets[task] = int(max_turns)
            continue
        p90 = ok[min(len(ok) - 1, math.ceil(0.9 * len(ok)) - 1)]
        budgets[task] = int(min(max_turns, max(floor, 2 * p90 + 2)))
    return {"budgets": budgets, "max_turns": int(max_turns), "floor": int(floor),
            "rule": f"2 * p90(turns of successes) + 2, clamped to [{floor}, {max_turns}]; "
                    f"{max_turns} with fewer than {min_successes} successes"}


# ---------------------------------------------------------------------------
# BrowserGym registry
# ---------------------------------------------------------------------------


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


def all_registered() -> list[str]:
    """Every MiniWoB task that browsergym-miniwob registers. Needs BrowserGym."""
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401

    prefix = "browsergym/miniwob."
    return sorted(k[len(prefix):] for k in gym.envs.registry if k.startswith(prefix))


def full_suite(registered: Iterable[str], exclude: dict[str, str]) -> dict:
    """Run 4's full-suite task list: every registered task except `exclude`."""
    names = sorted(set(registered))
    return {"tasks": [t for t in names if t not in exclude],
            "excluded": {t: why for t, why in sorted(exclude.items()) if t in names}}


def main(argv: Optional[list[str]] = None, run: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a run's task pool.")
    parser.add_argument("--run", default=run or "run4", choices=sorted(RUN_SOURCES))
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("registered", help="Write the run's candidates that BrowserGym registers.")
    reg.add_argument("--out", required=True)
    sel = sub.add_parser("select", help="Mix the pool from the screening eval.")
    sel.add_argument("--baseline", default=None, help="per_task.csv of the screen. Default: <results>/eval/baseline.")
    sel.add_argument("--candidates", default=None, help="JSON from `registered`. Default: every candidate.")
    sel.add_argument("--out", default=None, help="Default: <results>/selected_tasks.json.")
    # Run 3's tuning flags (run 4 uses the RUN4_* settings above).
    sel.add_argument("--max-learn", type=int, default=16)
    sel.add_argument("--max-replay", type=int, default=4)
    sel.add_argument("--min-per-source", type=int, default=3)
    sel.add_argument("--min-success", type=float, default=0.05)
    sel.add_argument("--max-success", type=float, default=0.85)
    bud = sub.add_parser("budgets", help="Write per-task turn limits from the screen's episodes.")
    bud.add_argument("--episodes", required=True)
    bud.add_argument("--out", required=True)
    bud.add_argument("--max-turns", type=int, default=20)
    bud.add_argument("--floor", type=int, default=10)
    args = parser.parse_args(argv)
    spec = load_tasks()

    if args.cmd == "registered":
        names = candidates(args.run, spec)
        ok, missing = registered_tasks(names)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"tasks": ok, "missing": missing}, indent=2))
        print(f"{len(ok)} of {len(names)} {args.run} candidates are registered.")
        if missing:
            print(f"Not registered, left out: {', '.join(missing)}")
        if not ok:
            raise SystemExit(f"No {args.run} candidate is registered. Check the browsergym-miniwob install.")
        if args.run == "run4":
            suite = full_suite(all_registered(), spec.full_suite_exclude)
            (out.parent / "full_suite.json").write_text(json.dumps(suite, indent=2))
            print(f"Full suite: {len(suite['tasks'])} tasks ({len(suite['excluded'])} excluded).")
        return

    if args.cmd == "budgets":
        rows = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
        result = turn_budgets_from_episodes(rows, max_turns=args.max_turns, floor=args.floor)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        short = {t: b for t, b in result["budgets"].items() if b < args.max_turns}
        print(f"Turn budgets for {len(result['budgets'])} tasks; {len(short)} below {args.max_turns}. Wrote {args.out}")
        return

    baseline = Path(args.baseline) if args.baseline else results_dir() / "eval" / "baseline" / "per_task.csv"
    if not baseline.exists():
        raise SystemExit(f"{baseline} not found. Run the screen phase first.")
    sources = task_sources(args.run, spec)
    if args.candidates:
        allowed = set(json.loads(Path(args.candidates).read_text())["tasks"])
        sources = {t: s for t, s in sources.items() if t in allowed}
    rates = read_rates(baseline)
    if args.run == "run4":
        result = select_run4(rates, sources, spec)
    else:
        result = mix_pool(
            rates, sources, min_success=args.min_success, max_success=args.max_success,
            max_learn=args.max_learn, max_replay=args.max_replay,
            reserved=[(src, args.min_per_source) for src in ("run2_pool", "run1_pool", "trackb_extra")],
        )
    for t in result["selected"]:
        kind = "replay" if t in result["replay"] else "learn"
        print(f"{t:28s} {result['rates'][t]:6.2f}  {kind:6s}  {result['source'][t]}")
    out = Path(args.out) if args.out else results_dir() / "selected_tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Selected {len(result['learn'])} learn and {len(result['replay'])} replay tasks. Wrote {out}")
    if len(result["learn"]) < 6:
        raise SystemExit(f"Only {len(result['learn'])} learn tasks; GRPO would have too little to learn from.")


if __name__ == "__main__":
    main()
