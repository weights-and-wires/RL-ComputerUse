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

The module is pure Python, so the tests run without a GPU. The shared code
lives in `cua.run_pool`, which run 4 uses too.
"""

from __future__ import annotations

from typing import Optional

from cua import run_pool
from cua.config import load_tasks
from cua.run_pool import registered_tasks  # noqa: F401 - part of this module's API
from cua.select_tasks import TARGET_RATE

# Source groups, in priority order. A task in two groups counts for the first.
SOURCES = run_pool.RUN_SOURCES["run3"]


def task_sources(spec=None) -> dict[str, str]:
    """Maps every run-3 candidate to its source group."""
    return run_pool.task_sources("run3", spec or load_tasks())


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
    """Run 3's pool: `min_per_source` learn slots each for run2_pool, run1_pool, and trackb_extra.

    See `cua.run_pool.mix_pool` for the arguments.
    """
    result = run_pool.mix_pool(
        rates, sources, min_success=min_success, max_success=max_success, max_learn=max_learn,
        min_learn=min_learn,
        reserved=[(src, min_per_source) for src in ("run2_pool", "run1_pool", "trackb_extra")],
        replay_sources=("run2_pool", "run1_pool"), max_replay=max_replay, target=target,
    )
    result["criteria"]["min_per_source"] = min_per_source
    return result


def main(argv: Optional[list[str]] = None) -> None:
    """`python -m cua.run3_pool registered|select ...`, as in run 3."""
    run_pool.main(argv, run="run3")


if __name__ == "__main__":
    main()
