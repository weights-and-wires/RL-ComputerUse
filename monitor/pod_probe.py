"""Runs ON THE POD (sent over SSH stdin as `python3 - ARGS`). Prints one JSON object.

Collects, in one SSH round trip, what the collector needs: the status JSON, new
metrics bytes, the run.log tail, eval summaries, sample folder names, GPU
numbers, and wandb_run.json. Standard library only; never imports torch.
"""

import json
import os
import subprocess
import sys

R = "results/run3"
argv = sys.argv[1:]


def arg(name, default):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


out = {}

try:
    p = subprocess.run(["bash", "scripts/run3.sh", "--status-json"], capture_output=True, timeout=30)
    t = p.stdout.decode(errors="replace").strip()
    out["status"] = json.loads(t[t.find("{"):]) if "{" in t else None
except Exception as e:  # noqa: BLE001
    out["status"] = None
    out["status_error"] = repr(e)[:300]

# metrics.jsonl from a byte offset
mpath = f"{R}/train/grpo-4b-run3/metrics.jsonl"
off = int(arg("--metrics-offset", "0"))
try:
    size = os.path.getsize(mpath)
    start = off if 0 <= off <= size else 0
    with open(mpath, "rb") as f:
        if start > 0:
            # A GRPO resume trims and rewrites the file. If the saved offset no
            # longer sits right after a newline, read from the start again; the
            # collector sorts out replayed steps.
            f.seek(start - 1)
            if f.read(1) != b"\n":
                start = 0
        f.seek(start)
        data = f.read(8_000_000)
    # Only whole lines.
    cut = data.rfind(b"\n") + 1
    out["metrics"] = {"size": size, "start": start, "end": start + cut, "text": data[:cut].decode(errors="replace")}
except OSError:
    out["metrics"] = None

# run.log tail
lines = int(arg("--log-lines", "2000"))
try:
    with open(f"{R}/run.log", "rb") as f:
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - 3_000_000))
        txt = f.read().decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    keep = [ln for ln in txt.split("\n") if ln.strip()][-lines:]
    out["log_tail"] = "\n".join(keep)
except OSError:
    out["log_tail"] = None

# evals
evals = {}
ed = f"{R}/eval"
if os.path.isdir(ed):
    for name in sorted(os.listdir(ed)):
        d = os.path.join(ed, name)
        if not os.path.isdir(d):
            continue
        e = {}
        try:
            with open(os.path.join(d, "per_task.csv"), encoding="utf-8") as f:
                e["per_task"] = f.read(200_000)
        except OSError:
            pass
        try:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
            keep = ("run_name", "mode", "track", "task_set", "overall_success", "overall_success_pooled",
                    "n_episodes", "n_crashed", "num_seeds", "k", "temperature", "adapter")
            e["summary"] = {k: s.get(k) for k in keep if k in s}
            pt = s.get("per_task")
            if isinstance(pt, dict):
                e["summary"]["per_task"] = {k: {"n": v.get("n"), "success_rate": v.get("success_rate")}
                                            for k, v in pt.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            pass
        if e:
            evals[name] = e
out["evals"] = evals

# samples
sd = f"{R}/samples"
try:
    out["samples"] = sorted(x for x in os.listdir(sd) if x.startswith("step-") and os.path.isfile(os.path.join(sd, x, "episode.json")))
except OSError:
    out["samples"] = []

# GPU
try:
    p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, timeout=15)
    f = [x.strip() for x in p.stdout.decode().splitlines()[0].split(",")]

    def num(x):
        try:
            return round(float(x))
        except ValueError:
            return None
    out["gpu"] = {"util": num(f[0]), "mem_used_mb": num(f[1]), "mem_total_mb": num(f[2]), "power_w": num(f[3])}
except Exception:  # noqa: BLE001
    out["gpu"] = None

try:
    with open(f"{R}/wandb_run.json", encoding="utf-8") as f:
        out["wandb_run"] = json.load(f)
except (OSError, ValueError):
    out["wandb_run"] = None

print(json.dumps(out))
