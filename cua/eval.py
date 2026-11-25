"""Evaluate a model (base or LoRA adapter) on MiniWoB++ tasks.

Command line:

    python -m cua.eval --config configs/eval.yaml \\
        --set eval.mode=greedy --set eval.tasks=heldout --set run_name=base-heldout \\
        [--set model.adapter=/workspace/ckpt/grpo-4b-trackA/final]

Outputs go to `results/eval/<run_name>/`:

    episodes.jsonl   one line per episode (Episode.summary())
    per_task.csv     one row per task
    summary.json     headline numbers and metadata
    gifs/            <task>_seed<seed>.gif replays with a caption strip

Metrics:

- success rate: fraction of episodes the environment scored as solved.
- pass@k: the chance that at least one of k samples solves a (task, seed).
  We use the unbiased estimator 1 - C(n-c, k) / C(n, k), where n is the number
  of samples drawn and c the number that succeeded. It equals the plain
  success rate when k = 1.
- format error rate: format-error turns divided by all turns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from cua.config import (
    Config,
    add_config_args,
    load_config,
    load_tasks,
    resolve_task_list,
    results_dir,
)
from cua.types import Episode, EpisodeSpec

# Order used for columns and the report's failure-mode table.
FAILURE_MODES = [
    "success",
    "wrong_action",
    "invalid_action",
    "format_error",
    "ran_out_of_turns",
    "crashed",
]


# ---------------------------------------------------------------------------
# Config with presets
# ---------------------------------------------------------------------------


def load_eval_config(path: str, overrides: Optional[list[str]] = None) -> Config:
    """Load the eval config and apply `presets[eval.mode]` under explicit overrides.

    The first load only reads `eval.mode` (which may itself come from `--set`).
    The second load applies the preset values *before* your `--set` values, so
    anything you pass explicitly wins over the preset.
    """
    overrides = list(overrides or [])
    first = load_config(path, overrides)
    preset = (first.get("presets") or {}).get(first.eval.mode)
    if not preset:
        return first
    # json.dumps output is valid YAML, and load_config parses override values as YAML.
    preset_overrides = [f"eval.{key}={json.dumps(value)}" for key, value in preset.items()]
    return load_config(path, preset_overrides + overrides)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c correct, budget k.

    If fewer than k samples failed, every size-k subset contains a success.
    """
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_metrics(episodes: list[Episode], k: int) -> dict[str, Any]:
    """Aggregate episodes into per-task and overall numbers.

    Crashed episodes (browser failures, not model failures) are excluded from
    the rates so a flaky browser does not look like a bad model. They are
    counted separately in `n_crashed`.
    """
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.task].append(ep)

    per_task: dict[str, dict[str, Any]] = {}
    for task, eps in by_task.items():
        valid = [e for e in eps if not e.crashed]
        groups: dict[int, list[Episode]] = defaultdict(list)  # seed -> samples
        for e in valid:
            groups[e.seed].append(e)
        pk = [pass_at_k(len(g), sum(e.success for e in g), k) for g in groups.values()]
        turns = sum(e.num_turns for e in valid)
        modes = Counter(e.failure_mode() for e in eps)
        per_task[task] = {
            "n": len(valid),
            "n_crashed": len(eps) - len(valid),
            "success_rate": _mean([float(e.success) for e in valid]),
            "pass_at_k": _mean(pk),
            "mean_turns": _mean([float(e.num_turns) for e in valid]),
            "format_error_rate": (sum(e.num_format_errors for e in valid) / turns) if turns else None,
            "failure_modes": {m: modes.get(m, 0) for m in FAILURE_MODES},
        }

    # Macro average: every task counts equally, even if one lost episodes to crashes.
    rated = [t for t in per_task.values() if t["success_rate"] is not None]
    valid_all = [e for e in episodes if not e.crashed]
    total_turns = sum(e.num_turns for e in valid_all)
    overall_modes = Counter(e.failure_mode() for e in episodes)
    return {
        "overall_success": _mean([t["success_rate"] for t in rated]),
        "overall_success_pooled": _mean([float(e.success) for e in valid_all]),
        "pass_at_k": _mean([t["pass_at_k"] for t in rated]),
        "format_error_rate": (
            sum(e.num_format_errors for e in valid_all) / total_turns if total_turns else None
        ),
        "mean_turns": _mean([float(e.num_turns) for e in valid_all]),
        "n_episodes": len(valid_all),
        "n_crashed": len(episodes) - len(valid_all),
        "per_task": dict(sorted(per_task.items())),
        "failure_modes": {m: overall_modes.get(m, 0) for m in FAILURE_MODES},
    }


# ---------------------------------------------------------------------------
# GIFs
# ---------------------------------------------------------------------------


def _caption_frame(frame: Image.Image, lines: list[str], width: int) -> Image.Image:
    """Return `frame` with a dark caption strip underneath."""
    font = ImageFont.load_default()
    line_h = 14
    strip_h = line_h * len(lines) + 8
    canvas = Image.new("RGB", (width, frame.height + strip_h), (24, 24, 28))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        # Cut long lines so text never runs off the image (the default font is ~6 px/char).
        max_chars = max(10, width // 6)
        draw.text((6, frame.height + 4 + i * line_h), line[:max_chars], fill=(235, 235, 235), font=font)
    return canvas


def write_gif(ep: Episode, path: Path, fps: float) -> bool:
    """Write a replay GIF with a caption (task, step, action, outcome). Returns True on success.

    `ep.frames[0]` is the page after reset and `ep.frames[i]` is the page after
    turn i (rollout keeps these aligned), so frame i is captioned with turn i-1's action.
    """
    if not ep.frames:
        return False
    base_size = ep.frames[0].size
    outcome_final = "SUCCESS" if ep.success else ("CRASHED" if ep.crashed else "FAIL")
    rendered: list[Image.Image] = []
    for i, frame in enumerate(ep.frames):
        frame = frame.convert("RGB")
        if frame.size != base_size:  # GIF frames must share one size
            frame = frame.resize(base_size)
        if i == 0:
            action = "(start)"
        elif i - 1 < len(ep.turns):
            turn = ep.turns[i - 1]
            action = turn.parsed.action or f"(no valid action: {turn.parsed.format_error})"
        else:
            action = ""
        is_last = i == len(ep.frames) - 1
        lines = [
            f"{ep.task}  seed {ep.seed}  step {i}/{ep.num_turns}",
            f"action: {action}",
            f"result: {outcome_final if is_last else 'running'}",
        ]
        rendered.append(_caption_frame(frame, lines, max(base_size[0], 320)))
    # Hold the last frame so you can read the outcome before the loop restarts.
    rendered.extend([rendered[-1]] * 3)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        import numpy as np

        # `fps` is unambiguous across imageio versions (`duration` changed units).
        imageio.mimsave(path, [np.asarray(f) for f in rendered], format="GIF", fps=fps, loop=0)
    except Exception:
        # Fall back to Pillow so a missing or odd imageio never loses the replay.
        rendered[0].save(
            path, save_all=True, append_images=rendered[1:], duration=int(1000 / fps), loop=0
        )
    return True


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------


def _log_wandb(cfg: Config, summary: dict[str, Any]) -> None:
    """Log the summary and a per-task table. Skips quietly without a key."""
    if not os.environ.get("WANDB_API_KEY"):  # check presence only; never print it
        print("WANDB_API_KEY not set; skipping W&B logging.")
        return
    try:
        import wandb

        started_run = wandb.run is None
        if started_run:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                job_type="eval",
                name=cfg.run_name,
                config=cfg.to_dict(),
                mode=cfg.wandb.get("mode", "online"),
            )
        columns = ["task", "n", "success_rate", "pass_at_k", "mean_turns", "format_error_rate"]
        table = wandb.Table(columns=columns)
        for task, row in summary["per_task"].items():
            table.add_data(task, *[row[c] for c in columns[1:]])
        # When a training run is already active (mid-training eval), prefix the
        # keys so they do not collide with the trainer's own metrics.
        prefix = "" if started_run else f"eval/{cfg.run_name}/"
        scalars = {
            f"{prefix}overall_success": summary["overall_success"],
            f"{prefix}pass_at_k": summary["pass_at_k"],
            f"{prefix}format_error_rate": summary["format_error_rate"],
            f"{prefix}mean_turns": summary["mean_turns"],
            f"{prefix}per_task": table,
        }
        wandb.log(scalars)
        if started_run:
            wandb.finish()
    except Exception as exc:  # logging must never fail an evaluation
        print(f"W&B logging failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def evaluate(
    cfg: Config,
    policy: Any = None,
    pool: Any = None,
    out_dir: Optional[Path] = None,
    log_wandb: bool = True,
) -> dict[str, Any]:
    """Run an evaluation and return the summary dict.

    Pass `policy` and `pool` to reuse existing ones (mid-training eval). Objects
    you pass in are never closed; objects created here are closed on exit.
    `cfg` should come from `load_eval_config` so presets are already applied.
    """
    # Imported here so `python -m cua.eval --help` works without a GPU stack.
    from cua.env import EnvPool
    from cua.policy import VLLMPolicy, sampling_from_cfg
    from cua.rollout import run_episodes

    ev = cfg.eval
    tasks = resolve_task_list(ev.tasks, cfg)
    seeds = load_tasks().seed_range(ev.seed_range).seeds(int(ev.num_seeds))
    temperature = float(ev.temperature)
    greedy = ev.mode == "greedy" or temperature == 0.0
    # Greedy decoding is deterministic, so extra samples would be identical copies.
    k = 1 if greedy else max(1, int(ev.k))
    gif_seeds = int(ev.get("gif_seeds", 0))
    out_dir = Path(out_dir) if out_dir else results_dir(cfg) / "eval" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task-major order keeps each browser on one task for long stretches.
    # Switching tasks forces a slow env rebuild (see MiniWoBEnv).
    specs = [
        EpisodeSpec(
            task=task,
            seed=seed,
            sample_idx=j,
            record=(seed_idx < gif_seeds and j == 0),
        )
        for task in tasks
        for seed_idx, seed in enumerate(seeds)
        for j in range(k)
    ]

    own_policy, own_pool = policy is None, pool is None
    t_start = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        if own_policy:
            policy = VLLMPolicy(cfg)
        if own_pool:
            pool = EnvPool(cfg)
        sampling = sampling_from_cfg(cfg, greedy=greedy)
        if not greedy:
            sampling.temperature = temperature
        t_load = time.perf_counter() - t_start
        episodes = run_episodes(
            specs, policy.generate, pool, cfg, sampling, track=cfg.env.track, stats=stats
        )
    finally:
        # Close only what we created, and close the pool before the policy.
        if own_pool and pool is not None:
            pool.close()
        if own_policy and policy is not None:
            policy.close()

    metrics = compute_metrics(episodes, k)
    adapter = getattr(policy, "adapter_path", None) or cfg.model.get("adapter")
    summary: dict[str, Any] = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "adapter": adapter,
        "mode": ev.mode,
        "task_set": ev.tasks if isinstance(ev.tasks, str) else ",".join(ev.tasks),
        "tasks": tasks,
        "seed_range": ev.seed_range,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "k": k,
        "temperature": 0.0 if greedy else temperature,
        "track": cfg.env.track,
        "max_turns": int(cfg.env.max_turns),
        # Run 4 settings that change what the model sees (absent or 0 before run 4).
        "context_window_turns": int(cfg.env.get("context_window_turns", 0) or 0),
        "track_b_extra_tools": bool(cfg.env.get("track_b_extra_tools", False)),
        "turn_budgets": cfg.env.get("turn_budgets"),
        "format_error_rate_definition": "format-error turns / all turns",
        **metrics,
        "timings": {
            "total_s": round(time.perf_counter() - t_start, 1),
            "load_s": round(t_load, 1),
            **{key: round(val, 1) if key.endswith("_s") else int(val) for key, val in stats.items()},
        },
    }

    # ---- write files --------------------------------------------------------
    with open(out_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep.summary()) + "\n")

    with open(out_dir / "per_task.csv", "w", newline="") as f:
        cols = ["task", "n", "n_crashed", "success_rate", "pass_at_k", "mean_turns", "format_error_rate"]
        writer = csv.writer(f)
        writer.writerow(cols + [f"fm_{m}" for m in FAILURE_MODES])
        for task, row in summary["per_task"].items():
            writer.writerow(
                [task]
                + ["" if row[c] is None else row[c] for c in cols[1:]]
                + [row["failure_modes"][m] for m in FAILURE_MODES]
            )

    n_gifs = 0
    for ep in episodes:
        if ep.frames and ep.sample_idx == 0:
            n_gifs += write_gif(ep, out_dir / "gifs" / f"{ep.task}_seed{ep.seed}.gif", float(ev.get("gif_fps", 2)))
            ep.frames.clear()  # release the memory; frames are only needed for GIFs
    summary["n_gifs"] = n_gifs

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, out_dir)
    if log_wandb:
        _log_wandb(cfg, summary)
    return summary


def _fmt(x: Optional[float], pct: bool = True) -> str:
    if x is None:
        return "  n/a"
    return f"{100 * x:5.1f}%" if pct else f"{x:5.2f}"


def _print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print(f"\n=== {summary['run_name']}  ({summary['mode']}, k={summary['k']}, T={summary['temperature']}) ===")
    print(f"{'task':28s} {'n':>4s} {'success':>8s} {'pass@k':>8s} {'turns':>6s} {'fmt_err':>8s}")
    for task, r in summary["per_task"].items():
        print(
            f"{task:28s} {r['n']:4d} {_fmt(r['success_rate']):>8s} {_fmt(r['pass_at_k']):>8s} "
            f"{_fmt(r['mean_turns'], False):>6s} {_fmt(r['format_error_rate']):>8s}"
        )
    print(
        f"{'OVERALL':28s} {summary['n_episodes']:4d} {_fmt(summary['overall_success']):>8s} "
        f"{_fmt(summary['pass_at_k']):>8s} {_fmt(summary['mean_turns'], False):>6s} "
        f"{_fmt(summary['format_error_rate']):>8s}"
    )
    if summary["n_crashed"]:
        print(f"WARNING: {summary['n_crashed']} episodes crashed and are excluded from the rates.")
    t = summary["timings"]
    print(
        f"time: total {t['total_s']}s, generation {t.get('generation_s', 0)}s, "
        f"browsers {t.get('env_s', 0)}s. Results in {out_dir}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MiniWoB++ tasks and write results/eval/<run_name>/."
    )
    add_config_args(parser, "configs/eval.yaml")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
    args = parser.parse_args(argv)
    cfg = load_eval_config(args.config, args.overrides)
    evaluate(cfg, log_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
