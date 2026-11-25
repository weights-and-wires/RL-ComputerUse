#!/usr/bin/env python3
"""Build read-only dashboard data snapshots for finished runs 1 and 2.

Reads the repo's local results and writes the same data schema that the live
monitor writes (see docs/RUN3_INTERFACES.md): state.json, metrics.jsonl,
evals.json, log.txt, events.jsonl, notes.json, and samples/. A snapshot has no
system.jsonl, because those runs had no GPU telemetry; the dashboard skips
panels without data.

Usage:
  python3 dashboard/snapshot_runs.py --out ~/cua-runs-build            # runs 1 and 2
  python3 dashboard/snapshot_runs.py --out OUT --runs run2 --samples 12

Each run lands in OUT/<run>/data/. The state has `"snapshot": true` and
`"run_state": "done"`. There is no demo flag.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "monitor"))
from common import (  # noqa: E402
    Scrubber, atomic_write_json, atomic_write_text, build_evals, clean_log_text, load_env, now_iso,
    read_jsonl, sample_index,
)
from demo_data import gif_to_jpgs  # noqa: E402

RUNS = {
    # run name: (label, results folder, metrics path under it, eval folder, log file, episodes folder for samples)
    "run1": {
        "label": "Run 1", "root": ROOT / "results", "metrics": "train/grpo-4b-trackA/metrics.jsonl",
        "eval": "eval", "log": "run.log", "samples_from": "eval/grpo-train",
        "note": "Run 1 trained Qwen3-VL-4B with GRPO on MiniWoB++ tasks, with element IDs (Track A).",
    },
    "run2": {
        "label": "Run 2", "root": ROOT / "results" / "run2", "metrics": "train/grpo-4b-run2/metrics.jsonl",
        "eval": "eval", "log": "run.log", "samples_from": "eval/run2-pool2",
        "note": "Run 2 continued from run 1 on a wider task pool.",
    },
}


def snapshot(run: str, out: Path, n_samples: int, scrub: Scrubber) -> dict:
    cfg = RUNS[run]
    root: Path = cfg["root"]
    data = out / run / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    counts = {}

    metrics = read_jsonl(root / cfg["metrics"])
    if metrics:
        atomic_write_text(data / "metrics.jsonl", "".join(json.dumps(scrub.obj(m)) + "\n" for m in metrics))
    counts["metrics"] = len(metrics)

    evals = build_evals(root / cfg["eval"])
    atomic_write_json(data / "evals.json", scrub.obj(evals))
    counts["evals"] = len(evals)

    try:
        log = (root / cfg["log"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    if log:
        atomic_write_text(data / "log.txt", clean_log_text(log, scrub, 2000))

    # Samples: pick evenly spaced finished episodes that have a GIF.
    sd = data / "samples"
    sd.mkdir()
    src = root / cfg["samples_from"]
    gifs = sorted((src / "gifs").glob("*.gif")) if (src / "gifs").is_dir() else []
    eps = {(e.get("task"), e.get("seed")): e for e in read_jsonl(src / "episodes.jsonl")}
    step_max = max([int(m.get("step", 0)) for m in metrics] or [0])
    picks = gifs[:: max(1, len(gifs) // max(1, n_samples))][:n_samples] if gifs else []
    for k, gif in enumerate(picks):
        task, _, seed = gif.stem.rpartition("_seed")
        ep = eps.get((task, int(seed) if seed.isdigit() else seed), {})
        step = max(1, round((k + 1) * step_max / (len(picks) + 1))) if step_max else k + 1
        folder = sd / f"step-{step:06d}"
        folder.mkdir(exist_ok=True)
        imgs = gif_to_jpgs(gif, folder, max(1, int(ep.get("num_turns") or 1)) + 1)
        acts, resps, errs = ep.get("actions") or [], ep.get("responses") or [], ep.get("env_errors") or []
        turns = [{"image": imgs[i] if i < len(imgs) else None,
                  "response": scrub(resps[i]) if i < len(resps) else "",
                  "action": acts[i] if i < len(acts) else "",
                  "error": errs[i] if i < len(errs) else ""} for i in range(max(len(acts), 1))]
        atomic_write_json(folder / "episode.json", {
            "step": step, "task": task, "seed": ep.get("seed", seed),
            "reward": ep.get("reward_total", ep.get("success")), "success": bool(ep.get("success")), "turns": turns})
    atomic_write_json(sd / "index.json", sample_index(sd))
    counts["samples"] = len(picks)

    # No telemetry for these runs: empty files, so the page shows "No GPU readings".
    now = now_iso()
    atomic_write_json(data / "notes.json", [{"t": now, "text": cfg["note"] + " This page is a finished snapshot rebuilt from the saved results."}])
    atomic_write_text(data / "events.jsonl", json.dumps(
        {"t": now, "kind": "note", "msg": f"{cfg['label']} finished earlier. This is a read-only snapshot of its saved results."}) + "\n")
    state = {
        "run": run, "run_label": cfg["label"], "snapshot": True, "demo": False, "run_state": "done",
        "updated": now, "started": None,
        "phases": [{"name": "grpo", "status": "done", "started": None, "finished": None},
                   {"name": "post_eval", "status": "done" if evals else "pending", "started": None, "finished": None}],
        "current_phase": None,
        "grpo": {"step": step_max or None, "max_steps": step_max or None},
        "pod": {}, "spend_usd": None, "budget_usd": None, "eta": None,
        "restarts": None, "pod_generations": None, "wandb_url": None, "hf_url": None, "last_error": None,
    }
    atomic_write_json(data / "state.json", scrub.obj(state))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder; each run goes to OUT/<run>/data")
    ap.add_argument("--runs", nargs="+", default=sorted(RUNS), choices=sorted(RUNS))
    ap.add_argument("--samples", type=int, default=12, help="sample episodes per run (needs ffmpeg)")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    scrub = Scrubber(load_env(ROOT / ".env"))
    for run in args.runs:
        c = snapshot(run, out, args.samples, scrub)
        print(f"{run}: " + ", ".join(f"{k} {v}" for k, v in c.items()) + f" -> {out / run / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
