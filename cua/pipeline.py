"""Small decisions that scripts/run4.sh makes, as pure functions with a CLI.

    python -m cua.pipeline pool-size [--want 64]
    python -m cua.pipeline choose-adapter --final DIR --best DIR --final-eval SUMMARY [--margin 0.02] --out JSON
    python -m cua.pipeline metrics-check METRICS.jsonl

Only the standard library, so the pipeline can call it before the venv is
warm and the tests run anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# Measured on run 3's pod: about 0.35 GB per headless Chromium plus 0.1 GB for
# its worker process. 0.6 GB leaves headroom for heavy pages.
GB_PER_BROWSER = 0.6
# RAM that vLLM, the trainer, and the page cache need outside the browsers.
RESERVE_GB = 40.0
# Each browser step mostly waits (100 ms slow-mo per Playwright call and a
# 0.5 s pre-observation delay), so 4 browsers per vCPU still keep up.
BROWSERS_PER_CPU = 4
MIN_BROWSERS = 16


def pool_size(want: int, nproc: int, mem_available_gb: Optional[float]) -> int:
    """How many browsers to run: `want`, unless CPU or RAM can't carry them.

    Never below MIN_BROWSERS (run 3's evals used 16) unless `want` is smaller.
    """
    size = min(int(want), BROWSERS_PER_CPU * max(1, int(nproc)))
    if mem_available_gb is not None:
        size = min(size, int(max(0.0, mem_available_gb - RESERVE_GB) / GB_PER_BROWSER))
    return max(min(int(want), MIN_BROWSERS), size)


def mem_available_gb(meminfo: str = "/proc/meminfo") -> Optional[float]:
    """MemAvailable in GB, or None off Linux."""
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return None


def cgroup_cpus(path: str = "/sys/fs/cgroup/cpu.max") -> Optional[int]:
    """The container's CPU quota in whole CPUs (cgroup v2), or None when unlimited or unknown.

    `os.cpu_count()` reports the host's CPUs, which on Runpod is often far
    more than the pod may use.
    """
    try:
        quota, period = Path(path).read_text().split()[:2]
        if quota == "max":
            return None
        return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        return None


def choose_adapter(final_score: Optional[float], best: Optional[dict], margin: float = 0.02) -> str:
    """Returns "final" or "best": which adapter post_eval should report.

    Both scores come from the same greedy eval on the same seeds (run 4: the
    pool tasks on the baseline seed range, which the headline evals never
    use). The best checkpoint wins only when it beats the final adapter by
    more than `margin`, because the mid-training score of 20 seeds per task
    is noisy and a later adapter has trained on more groups.
    """
    if not best or best.get("eval_success") is None:
        return "final"
    if final_score is None:
        return "best"
    return "best" if float(best["eval_success"]) > float(final_score) + margin else "final"


def check_metrics(rows: list[dict]) -> list[str]:
    """Problems with a metrics.jsonl from run 4's loop: every row has a kind, and steps are sane."""
    problems = []
    train_steps = [r.get("step") for r in rows if r.get("kind") == "train"]
    for i, r in enumerate(rows):
        if r.get("kind") not in ("train", "eval"):
            problems.append(f"row {i} has kind {r.get('kind')!r}")
        if r.get("kind") == "eval" and r.get("step") not in train_steps:
            problems.append(f"eval row {i} (step {r.get('step')}) has no training row with that step")
    if train_steps != sorted(set(train_steps)):
        problems.append(f"training steps aren't unique and increasing: {train_steps}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Small decisions for scripts/run4.sh.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("pool-size", help="Print how many browsers to run.")
    ps.add_argument("--want", type=int, default=64)
    ca = sub.add_parser("choose-adapter", help="Choose between the final and the best adapter.")
    ca.add_argument("--final", required=True)
    ca.add_argument("--best", required=True, help="The best/ folder (holds best.json).")
    ca.add_argument("--final-eval", required=True, help="summary.json of the final adapter's selection eval.")
    ca.add_argument("--margin", type=float, default=0.02)
    ca.add_argument("--out", required=True)
    mc = sub.add_parser("metrics-check", help="Check the kind field and steps of a metrics.jsonl.")
    mc.add_argument("path")
    args = parser.parse_args(argv)

    if args.cmd == "pool-size":
        nproc = cgroup_cpus() or os.cpu_count() or 1
        print(pool_size(args.want, nproc, mem_available_gb()))
        return 0

    if args.cmd == "metrics-check":
        rows = [json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()]
        problems = check_metrics(rows)
        for p in problems:
            print(f"metrics problem: {p}")
        return 1 if problems else 0

    best = None
    try:
        best = json.loads((Path(args.best) / "best.json").read_text())
        if not (Path(args.best) / "adapter_config.json").exists():
            best = None
    except (OSError, ValueError):
        pass
    final_score = None
    try:
        final_score = json.loads(Path(args.final_eval).read_text()).get("overall_success")
    except (OSError, ValueError):
        pass
    choice = choose_adapter(final_score, best, args.margin)
    path = args.best if choice == "best" else args.final
    record = {"choice": choice, "adapter": path, "final_adapter": args.final, "final_eval_success": final_score,
              "best": best, "margin": args.margin}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
