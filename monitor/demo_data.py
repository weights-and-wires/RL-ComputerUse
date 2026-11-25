#!/usr/bin/env python3
"""Write a demo data/ set from run-2 results, for dashboard development.

Every file this script writes is demo data: data/state.json carries
`"demo": true`, and data/.demo lists the files it wrote. The real collector
deletes them all when a real run starts.

Usage:
  python3 monitor/demo_data.py --src results/run2 [--data ~/cua-run4/data] [--progress 0.75]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    Paths, Scrubber, atomic_write_json, atomic_write_text, build_evals, clean_log_text,
    load_env, now_iso, read_jsonl, sample_index,
)

PHASES = ["smoke", "screen", "select", "rft", "pre_eval", "grpo", "post_eval", "compare", "report"]  # run3.sh
COST_PER_HR = 2.69
DEMO_MARK = ".demo"


def gif_to_jpgs(gif: Path, outdir: Path, max_turns: int) -> list[str]:
    """Split a GIF into turn-K.jpg frames (640 px wide at most) with ffmpeg."""
    if not shutil.which("ffmpeg"):
        return []
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(gif), "-fps_mode", "passthrough",
             "-vf", "scale='min(640,iw)':-2", "-q:v", "7", f"{td}/f-%03d.jpg"],
            capture_output=True,
        )
        if r.returncode != 0:
            return []
        frames = sorted(Path(td).glob("f-*.jpg"))[:max_turns]
        names = []
        for k, f in enumerate(frames):
            name = f"turn-{k}.jpg"
            shutil.copy(f, outdir / name)
            names.append(name)
        return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="run-2 results folder (results/run2)")
    ap.add_argument("--data", default=None, help="output folder (default: the data folder of the selected run, ~/cua-run4/data)")
    ap.add_argument("--progress", type=float, default=0.75, help="fraction of GRPO done in the demo")
    ap.add_argument("--samples", type=int, default=14)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    paths = Paths()
    data = Path(args.data).expanduser() if args.data else paths.data
    data.mkdir(parents=True, exist_ok=True)
    scrub = Scrubber(load_env(paths.env))
    rng = random.Random(3)
    written: list[str] = []

    # --- metrics -------------------------------------------------------------
    metrics = read_jsonl(src / "train/grpo-4b-run2/metrics.jsonl")
    max_steps = len(metrics) or 54
    cut = max(1, int(max_steps * args.progress))
    shown = metrics[:cut]
    atomic_write_text(data / "metrics.jsonl", "".join(json.dumps(m) + "\n" for m in shown))
    written.append("metrics.jsonl")

    # --- timeline ------------------------------------------------------------
    now = time.time()
    # Phase durations in minutes, roughly as in run 2.
    durations = {"smoke": 12, "screen": 40, "select": 4, "rft": 45, "pre_eval": 60,
                 "grpo": 5.5 * 60, "post_eval": 60, "compare": 2, "report": 4}
    grpo_elapsed = durations["grpo"] * 60 * args.progress
    t = now - grpo_elapsed - sum(durations[p] for p in PHASES[:PHASES.index("grpo")]) * 60 - 600
    started = t
    events = [{"t": now_iso(t), "kind": "pod_created",
               "msg": "[demo] Created pod demo0000pod (NVIDIA H100 80GB HBM3, SECURE, $2.69/hr), generation 1."}]
    t += 300
    events.append({"t": now_iso(t), "kind": "note", "msg": "[demo] SSH is up; secrets and code shipped; run3_pod_start.sh printed STARTED."})
    t += 300
    phases = []
    for p in PHASES:
        ph = {"name": p, "status": "pending", "started": None, "finished": None}
        if p == "grpo":
            ph.update(status="running", started=now_iso(t))
            events.append({"t": now_iso(t), "kind": "phase_started", "msg": "[demo] Phase grpo started."})
            # A crash and a restart in the middle of GRPO, to show those events.
            tc = t + grpo_elapsed * 0.4
            events.append({"t": now_iso(tc), "kind": "restart",
                           "msg": "[demo] tmux session run3 died during grpo; restarted with run3_pod_start.sh (restart 1)."})
            phases.append(ph)
            break_after = True
        else:
            ph.update(status="done", started=now_iso(t), finished=now_iso(t + durations[p] * 60))
            events.append({"t": now_iso(t), "kind": "phase_started", "msg": f"[demo] Phase {p} started."})
            t += durations[p] * 60
            events.append({"t": now_iso(t), "kind": "phase_done", "msg": f"[demo] Phase {p} done."})
            phases.append(ph)
            break_after = False
        if break_after:
            break
    for p in PHASES[len(phases):]:
        phases.append({"name": p, "status": "pending", "started": None, "finished": None})
    events.sort(key=lambda e: e["t"])
    atomic_write_text(data / "events.jsonl", "".join(json.dumps(e) + "\n" for e in events))
    written.append("events.jsonl")

    # --- system.jsonl --------------------------------------------------------
    rows = []
    minutes = int((now - started) / 60)
    for i in range(minutes + 1):
        ts = started + i * 60
        frac_grpo = ts > now - grpo_elapsed
        phase_cycle = (i % 7) < 3  # rollout vs train
        util = (rng.randint(30, 60) if phase_cycle else rng.randint(92, 100)) if frac_grpo else rng.randint(0, 45)
        rows.append({
            "t": now_iso(ts),
            "gpu_util": util,
            "gpu_mem_mb": rng.randint(48000, 62000) if frac_grpo else rng.randint(35000, 71000),
            "gpu_power_w": round(120 + util * 5.6 + rng.uniform(-20, 20)),
            "spend_usd": round((ts - started) / 3600 * COST_PER_HR, 3),
        })
    atomic_write_text(data / "system.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    written.append("system.jsonl")

    # --- evals ---------------------------------------------------------------
    evals = build_evals(src / "eval")
    atomic_write_json(data / "evals.json", evals)
    written.append("evals.json")

    # --- log -----------------------------------------------------------------
    try:
        log = (src / "run.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    lines = log.replace("\r", "\n").split("\n")
    # Stop the log in the middle of GRPO so it matches the demo state.
    for i, ln in enumerate(lines):
        if f"step {cut + 1}" in ln.lower() or f"step={cut + 1}" in ln.lower():
            lines = lines[:i]
            break
    atomic_write_text(data / "log.txt", "[demo] This log is run 2's, shown as demo data.\n" + clean_log_text("\n".join(lines), scrub))
    written.append("log.txt")

    # --- samples -------------------------------------------------------------
    sd = data / "samples"
    if sd.exists():
        shutil.rmtree(sd)
    sd.mkdir(parents=True)
    eps = read_jsonl(src / "eval/run2-pool2/episodes.jsonl")
    by_key = {(e.get("task"), e.get("seed")): e for e in eps}
    gifs = sorted((src / "eval/run2-pool2/gifs").glob("*.gif"))
    rng.shuffle(gifs)
    step_list = sorted(rng.sample(range(1, cut + 1), min(args.samples, cut, len(gifs))))
    for step, gif in zip(step_list, gifs):
        stem = gif.stem  # task_seedN
        task, _, seed = stem.rpartition("_seed")
        ep = by_key.get((task, int(seed) if seed.isdigit() else seed), {})
        folder = sd / f"step-{step:06d}"
        folder.mkdir()
        n_turns = max(1, int(ep.get("num_turns") or 1))
        imgs = gif_to_jpgs(gif, folder, n_turns + 1)
        acts = ep.get("actions") or []
        resps = ep.get("responses") or []
        errs = ep.get("env_errors") or []
        turns = []
        for k in range(max(len(acts), 1)):
            turns.append({
                "image": imgs[k] if k < len(imgs) else None,
                "response": scrub(resps[k]) if k < len(resps) else "",
                "action": acts[k] if k < len(acts) else "",
                "error": errs[k] if k < len(errs) else "",
            })
        atomic_write_json(folder / "episode.json", {
            "step": step, "task": task, "seed": ep.get("seed", seed),
            "reward": ep.get("reward_total", ep.get("success")),
            "success": bool(ep.get("success")), "turns": turns, "demo": True,
        })
    atomic_write_json(sd / "index.json", sample_index(sd))
    written.append("samples")

    # --- notes ---------------------------------------------------------------
    notes = [
        {"t": now_iso(now - 3600), "text": "[demo] GRPO is moving: reward is flat around 0.4 and mixed groups hold at 7 to 11 per step. Step times sit under 400 s, and spend is on track for the $50 cap."},
        {"t": now_iso(now - 1800), "text": f"[demo] GRPO is at step {cut} of {max_steps}. The run restarted once after its tmux session died and resumed from the last checkpoint. Nothing needs your attention."},
    ]
    atomic_write_json(data / "notes.json", notes)
    written.append("notes.json")

    # --- state ---------------------------------------------------------------
    last = shown[-1] if shown else {}
    spend = round((now - started) / 3600 * COST_PER_HR, 2)
    sec_per_step = grpo_elapsed / max(1, cut)
    remaining = (max_steps - cut) * sec_per_step + (60 + 2 + 4) * 60
    state = {
        "demo": True,
        "run_state": "running",
        "updated": now_iso(now),
        "started": now_iso(started),
        "phases": phases,
        "current_phase": "grpo",
        "grpo": {"step": int(last.get("step", cut)), "max_steps": max_steps,
                 "last_checkpoint": (cut // 25) * 25 or None, "synced_checkpoint": (cut // 25) * 25 or None},
        "pod": {"id": "demo0000pod", "gpu": "NVIDIA H100 80GB HBM3", "cloud": "SECURE",
                "cost_per_hr": COST_PER_HR, "uptime_s": int(now - started), "generation": 1},
        "spend_usd": spend,
        "budget_usd": 50,
        "eta": now_iso(now + remaining),
        "restarts": 1,
        "pod_generations": 1,
        "wandb_url": None,
        "hf_url": None,
        "last_error": None,
        "gpu": {"util": rows[-1]["gpu_util"], "mem_used_mb": rows[-1]["gpu_mem_mb"],
                "mem_total_mb": 81559, "power_w": rows[-1]["gpu_power_w"]},
    }
    atomic_write_json(data / "state.json", state)
    written.append("state.json")
    atomic_write_json(data / DEMO_MARK, {"demo": True, "written": now_iso(now), "files": written})
    print(f"Wrote demo data to {data}: {len(shown)} metric rows, {len(rows)} system rows, "
          f"{len(evals)} evals, {len(step_list)} samples, {len(events)} events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
