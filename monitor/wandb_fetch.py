#!/usr/bin/env python3
"""Fetch a W&B run's system metrics and step history with the public API.

Runs in ~/cua-run3/venv (which has `wandb`). Reads WANDB_API_KEY from the
environment. Writes {"system": [...], "history": [...]} to --out.
"""

import argparse
import json
import math
import os
import sys
import tempfile

SYS_KEYS = {
    "system.gpu.0.gpu": "gpu_util",
    "system.gpu.0.memoryAllocated": "gpu_mem_pct",
    "system.gpu.0.memoryAllocatedBytes": "gpu_mem_bytes",
    "system.gpu.0.powerWatts": "gpu_power_w",
    "system.gpu.0.temp": "gpu_temp_c",
    "system.cpu": "cpu_pct",
    "system.memory": "ram_pct",
}


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="entity/project/run_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import wandb  # noqa: PLC0415

    api = wandb.Api(timeout=60)
    run = api.run(args.path)
    system = []
    rows = None
    for stream in ("system", "events"):
        try:
            rows = run.history(samples=2000, stream=stream, pandas=False)
            if rows:
                break
        except Exception:  # noqa: BLE001
            rows = None
    import datetime as dt
    for r in rows or []:
        ts = r.get("_timestamp")
        if ts is None:
            continue
        out = {"t": dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")}
        for k, name in SYS_KEYS.items():
            if k in r:
                out[name] = clean(r[k])
        if len(out) > 1:
            system.append(out)
    history = []
    for r in run.scan_history(page_size=1000):
        step = r.get("step", r.get("_step"))
        if step is None:
            continue
        row = {k: clean(v) for k, v in r.items() if not k.startswith("_") and clean(v) is not None}
        row["step"] = step
        history.append(row)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(args.out)))
    with os.fdopen(fd, "w") as f:
        json.dump({"system": system, "history": history, "url": run.url}, f)
    os.replace(tmp, args.out)
    print(f"system rows {len(system)}, history rows {len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
