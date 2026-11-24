"""The `--status-json` view of run 3, for the monitor.

    python3 -m cua.run3_status [--results results/run3] [--ckpt /workspace/ckpt/grpo-4b-run3]

`scripts/run3.sh --status-json` calls this. It prints one JSON object (schema
in docs/RUN3_INTERFACES.md) and always exits 0. It uses only the standard
library and reads only small files, so it finishes in well under a second
(plus at most 3 s for nvidia-smi). It must never import torch.

Files it reads, all written by scripts/run3.sh unless noted:

- `.phases/<phase>.done`: finished (UTC timestamp inside).
- `.phases/<phase>.started`: start time of the latest attempt.
- `.phases/<phase>.failed`: the latest attempt failed (timestamp and exit code).
- `.phases/.current`: the phase running now.
- `.phases/.pid`: PID of the running run3.sh.
- `.phases/.elapsed_seconds`: summed wall time of finished phase attempts.
- `train/grpo-4b-run3/metrics.jsonl`: the GRPO loop's per-step metrics.
- `.synced_checkpoint`: written by the sync loop (`cua/hfsync.py`).
- `run.log`: the whole-run log; its last line is `ALL DONE` on success.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Keep in sync with PHASES in scripts/run3.sh.
PHASES = ["smoke", "screen", "select", "rft", "pre_eval", "grpo", "post_eval", "compare", "report"]
RUN_NAME = "grpo-4b-run3"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "run3"
DEFAULT_CKPT = Path("/workspace/ckpt") / RUN_NAME
DEFAULT_CONFIG = REPO_ROOT / "configs" / "grpo_run3.yaml"


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _parse_time(text: Optional[str]) -> Optional[str]:
    """Returns the first ISO-8601 timestamp in a marker file, or None."""
    if not text:
        return None
    m = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?", text)
    return m.group(0) if m else None


def _epoch(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.strptime(iso.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def last_line(path: Path, max_bytes: int = 4096) -> str:
    """Last non-empty line of a file, reading only its tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in tail.replace("\r", "\n").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def last_step(metrics: Path, max_bytes: int = 256 * 1024) -> Optional[int]:
    """Step of the newest training row in metrics.jsonl (eval-only rows count too)."""
    try:
        with open(metrics, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    best = None
    for line in tail.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # the first line of the tail is usually cut, and the last may be half-written
        if isinstance(row, dict) and isinstance(row.get("step"), int):
            best = row["step"] if best is None else max(best, row["step"])
    return best


def newest_checkpoint(ckpt_dir: Path) -> Optional[int]:
    """Largest N with a finished `checkpoint-N` folder (`.tmp` folders don't count)."""
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return None
    steps = [int(n.split("-", 1)[1]) for n in names
             if re.fullmatch(r"checkpoint-\d+", n) and (ckpt_dir / n / "trainer_state.json").exists()]
    return max(steps) if steps else None


def config_max_steps(config: Path) -> Optional[int]:
    """`grpo.max_steps` from the run-3 config, read with a regex (no YAML import)."""
    text = _read(config) or ""
    m = re.search(r"^grpo:\s*$(.*?)(?=^\S)", text + "\nend:\n", re.S | re.M)
    if m:
        mm = re.search(r"^\s+max_steps:\s*(\d+)", m.group(1), re.M)
        if mm:
            return int(mm.group(1))
    return None


def gpu_stats(runner: Callable[..., Any] = subprocess.run) -> dict[str, Optional[float]]:
    """GPU 0 utilization, memory, and power from nvidia-smi. Nones when it's missing."""
    empty = {"util": None, "mem_used_mb": None, "mem_total_mb": None, "power_w": None}
    try:
        out = runner(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return empty
    if getattr(out, "returncode", 1) != 0 or not out.stdout.strip():
        return empty
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

    def num(x: str) -> Optional[float]:
        try:
            v = float(x)
        except ValueError:
            return None
        return int(v) if v.is_integer() else round(v, 1)

    vals = [num(p) for p in parts] + [None] * 4
    return {"util": vals[0], "mem_used_mb": vals[1], "mem_total_mb": vals[2], "power_w": vals[3]}


def build_status(
    results: Path = DEFAULT_RESULTS,
    ckpt_dir: Path = DEFAULT_CKPT,
    config: Path = DEFAULT_CONFIG,
    now: Optional[float] = None,
    alive: Callable[[int], bool] = pid_alive,
    gpu: Optional[Callable[[], dict]] = None,
) -> dict[str, Any]:
    """Builds the status dict. Every argument has a default so tests can inject fakes."""
    now = time.time() if now is None else now
    ph = Path(results) / ".phases"
    pid_text = _read(ph / ".pid")
    running = bool(pid_text and pid_text.isdigit() and alive(int(pid_text)))
    current = _read(ph / ".current") if running else None

    phases, failed_phase = [], None
    for name in PHASES:
        done = (ph / f"{name}.done").exists()
        started = _parse_time(_read(ph / f"{name}.started"))
        finished = _parse_time(_read(ph / f"{name}.done")) if done else None
        if done:
            status = "done"
        elif running and current == name:
            status = "running"
        elif (ph / f"{name}.failed").exists():
            status = "failed"
            failed_phase = name
        else:
            status = "pending"
        phases.append({"name": name, "status": status, "started": started, "finished": finished})

    all_done = all(p["status"] == "done" for p in phases)
    if current is None and not all_done:
        current = next((p["name"] for p in phases if p["status"] != "done"), None)
    if all_done:
        current = None

    elapsed = 0.0
    try:
        elapsed = float(_read(ph / ".elapsed_seconds") or 0)
    except ValueError:
        pass
    if running and current:
        start = _epoch(_parse_time(_read(ph / f"{current}.started")))
        if start:
            elapsed += max(0.0, now - start)

    synced = _read(Path(results) / ".synced_checkpoint")
    return {
        "run": "run3",
        "done": all_done,
        "all_done_line": last_line(Path(results) / "run.log") == "ALL DONE",
        "running": running,
        "current_phase": current,
        "failed_phase": failed_phase,
        "phases": phases,
        "grpo": {
            "step": last_step(Path(results) / "train" / RUN_NAME / "metrics.jsonl"),
            "max_steps": config_max_steps(Path(config)),
            "last_checkpoint": newest_checkpoint(Path(ckpt_dir)),
            "synced_checkpoint": int(synced) if synced and synced.isdigit() else None,
        },
        "elapsed_s": int(elapsed),
        "gpu": (gpu or gpu_stats)(),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Print run 3's status as one JSON object.")
    parser.add_argument("--results", default=os.environ.get("RUN3_RESULTS", str(DEFAULT_RESULTS)))
    parser.add_argument("--ckpt", default=os.environ.get("RUN3_CKPT", str(DEFAULT_CKPT)))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    try:
        status = build_status(Path(args.results), Path(args.ckpt), Path(args.config))
    except Exception as exc:  # noqa: BLE001 - the monitor needs JSON even when something is off
        status = {"run": "run3", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(status))


if __name__ == "__main__":
    main()
