#!/usr/bin/env python3
"""Build a demo data set for the dashboard from results/run2/.

Writes dashboard/fixtures/data/ following docs/RUN3_INTERFACES.md. System
telemetry, events, and the sample click coordinates are synthetic; the
metrics, evals, and log come from run 2.
"""
import csv, json, math, random, re, shutil, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
R2 = ROOT / "results" / "run2"
random.seed(3)
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "samples").mkdir(parents=True)

def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]

metrics = [json.loads(l) for l in open(R2 / "train/grpo-4b-run2/metrics.jsonl")]
shutil.copy(R2 / "train/grpo-4b-run2/metrics.jsonl", OUT / "metrics.jsonl")
steps = len(metrics)

now = datetime.now(timezone.utc).replace(microsecond=0)
start = now - timedelta(hours=3, minutes=40)
iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")

# system.jsonl: one line per minute
spend, rate = 0.0, 2.99
lines = []
t = start
while t <= now:
    phase_boost = 1 if (t - start).total_seconds() > 1500 else 0.25
    lines.append({"t": iso(t), "gpu_util": int(min(100, max(0, random.gauss(90 if phase_boost == 1 else 30, 6)))),
                  "gpu_mem_mb": int(random.gauss(58000, 1500) if phase_boost == 1 else 21000),
                  "gpu_power_w": int(random.gauss(560, 40) if phase_boost == 1 else 190),
                  "spend_usd": round(spend, 3)})
    spend += rate / 60
    t += timedelta(minutes=1)
(OUT / "system.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")

phases = [("setup", 0, 12), ("baseline", 12, 40), ("grpo", 40, None), ("post_eval", None, None), ("upload", None, None)]
ph = []
for name, a, b in phases:
    if b is not None:
        ph.append({"name": name, "status": "done", "started": iso(start + timedelta(minutes=a)), "finished": iso(start + timedelta(minutes=b))})
    elif a is not None:
        ph.append({"name": name, "status": "running", "started": iso(start + timedelta(minutes=a)), "finished": None})
    else:
        ph.append({"name": name, "status": "pending", "started": None, "finished": None})
state = {
    "demo": True, "run_state": "running", "updated": iso(now), "started": iso(start), "phases": ph,
    "current_phase": "grpo",
    "grpo": {"step": metrics[-1]["step"], "max_steps": 80, "last_checkpoint": 50, "synced_checkpoint": 50},
    "pod": {"id": "demo", "gpu": "1x H100 80GB", "cost_per_hr": rate, "uptime_s": int((now - start).total_seconds()), "generation": 1},
    "spend_usd": round(spend, 2), "budget_usd": 50, "eta": iso(now + timedelta(minutes=70)),
    "restarts": 1, "pod_generations": 1, "wandb_url": "https://wandb.ai/", "hf_url": "https://huggingface.co/", "last_error": None,
}
(OUT / "state.json").write_text(json.dumps(state, indent=1))

# evals.json from run 2 eval summaries
evals = {}
for name in ["base-heldout2", "run2-heldout2", "base-pool2", "run2-pool2"]:
    s = json.load(open(R2 / "eval" / name / "summary.json"))
    tasks = {k: {"success": v["success_rate"], "n": v["n"]} for k, v in s["per_task"].items()}
    k = sum(round(v["success"] * v["n"]) for v in tasks.values())
    n = sum(v["n"] for v in tasks.values())
    evals[name] = {"mode": s["mode"], "track": s["track"], "tasks": tasks,
                   "mean": round(k / n, 4), "ci95": wilson(k, n)}
(OUT / "evals.json").write_text(json.dumps(evals, indent=1))

# log: tail of run 2 log, scrubbed
raw = (R2 / "run.log").read_text(errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")[-2000:]
scrub = lambda s: re.sub(r"(hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,}|\b[0-9a-f]{40}\b)", "[redacted]", s)
(OUT / "log.txt").write_text("\n".join(scrub(l) for l in raw) + "\n")

ev = [
    (0, "pod_created", "Pod created: 1x H100 80GB (demo)"),
    (12, "phase_done", "setup finished"), (12, "phase_started", "baseline started"),
    (40, "phase_done", "baseline finished"), (40, "phase_started", "grpo started"),
    (95, "restart", "GRPO process exited (CUDA OOM); restarted from checkpoint 25"),
    (150, "note", "Demo events: invented for layout, not from run 2."),
]
(OUT / "events.jsonl").write_text("\n".join(json.dumps({"t": iso(start + timedelta(minutes=m)), "kind": k, "msg": s}) for m, k, s in ev) + "\n")
(OUT / "notes.json").write_text(json.dumps([
    {"t": iso(now - timedelta(minutes=50)), "text": "Run 2 **demo data** is on screen. Notes support *italics*, `code`, lists and [links](https://example.com)."},
    {"t": iso(now - timedelta(minutes=5)), "text": "- Mixed groups are rising.\n- Next: held-out eval after step 80.\n\n<script>alert(1)</script> is escaped, not run."},
], indent=1))

# samples: first frame of an eval gif, with a made-up click
try:
    from PIL import Image
except ImportError:
    Image = None
index = []
if Image:
    picks = [("click-checkboxes-large", 0, True), ("click-checkboxes-large", 1, False), ("click-test-2", 0, True)]
    for i, (task, seed, ok) in enumerate(picks):
        g = R2 / "eval/run2-heldout2/gifs" / f"{task}_seed{seed}.gif"
        if not g.exists():
            continue
        step = 50 - 10 * i
        d = OUT / "samples" / f"step-{step:06d}"
        d.mkdir(exist_ok=True)
        im = Image.open(g)
        turns = []
        for k in range(min(3, im.n_frames)):
            im.seek(k)
            f = im.convert("RGB")
            f.thumbnail((640, 640))
            fn = f"turn-{k}.jpg"
            f.save(d / fn, quality=70)
            x, y = random.randint(150, 850), random.randint(150, 850)
            turns.append({"image": fn, "response": f"I will click the box.\n<tool_call>\n{{\"name\": \"mouse_click\", \"arguments\": {{\"coordinate\": [{x}, {y}]}}}}\n</tool_call>",
                          "action": f"click(coordinate=[{x}, {y}])", "error": ""})
        rw = 1.04 if ok else 0.0
        (d / "episode.json").write_text(json.dumps({"step": step, "task": task, "seed": seed, "reward": rw, "success": ok, "turns": turns}))
        index.append({"step": step, "task": task, "success": ok, "reward": rw, "dir": d.name})
index.sort(key=lambda r: -r["step"])
(OUT / "samples" / "index.json").write_text(json.dumps(index))
print("fixture written to", OUT, "steps", steps, "samples", len(index))
