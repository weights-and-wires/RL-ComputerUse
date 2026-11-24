"""Tests for `scripts/run3.sh --status-json` (cua/run3_status.py), with fake marker folders.

No GPU, tmux, or nvidia-smi needed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from cua import run3_status as st

REPO = Path(__file__).resolve().parent.parent
NO_GPU = {"util": None, "mem_used_mb": None, "mem_total_mb": None, "power_w": None}


def make_results(tmp_path: Path) -> Path:
    res = tmp_path / "results" / "run3"
    (res / ".phases").mkdir(parents=True)
    return res


def status(res: Path, ckpt: Path, **kw):
    kw.setdefault("gpu", lambda: dict(NO_GPU))
    kw.setdefault("alive", lambda pid: False)
    return st.build_status(res, ckpt, REPO / "configs" / "grpo_run3.yaml", **kw)


def test_phases_match_run3_sh():
    text = (REPO / "scripts" / "run3.sh").read_text()
    m = re.search(r"^PHASES=\(([^)]*)\)", text, re.M)
    assert m and m.group(1).split() == st.PHASES


def test_fresh_run_has_every_phase_pending(tmp_path):
    res = make_results(tmp_path)
    s = status(res, tmp_path / "ckpt")
    assert s["run"] == "run3"
    assert s["done"] is False and s["running"] is False and s["all_done_line"] is False
    assert [p["status"] for p in s["phases"]] == ["pending"] * len(st.PHASES)
    assert s["current_phase"] == "smoke"
    assert s["failed_phase"] is None
    assert s["grpo"] == {"step": None, "max_steps": 400, "last_checkpoint": None, "synced_checkpoint": None}
    assert s["elapsed_s"] == 0
    assert set(s["gpu"]) == {"util", "mem_used_mb", "mem_total_mb", "power_w"}


def test_running_grpo_with_checkpoints_and_metrics(tmp_path):
    res = make_results(tmp_path)
    ph = res / ".phases"
    for name in ("smoke", "screen", "select", "rft", "pre_eval"):
        (ph / f"{name}.started").write_text("2026-11-25T01:00:00Z\n")
        (ph / f"{name}.done").write_text("2026-11-25T02:00:00Z\n")
    (ph / "grpo.started").write_text("2026-11-25T03:00:00Z\n")
    (ph / ".current").write_text("grpo\n")
    (ph / ".pid").write_text("4242\n")
    (ph / ".elapsed_seconds").write_text("7200\n")
    metrics = res / "train" / "grpo-4b-run3" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    rows = [{"step": i, "time/rollout_s": 1.0} for i in range(1, 121)] + [{"step": 100, "eval/success": 0.6}]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"step": 121, "half-writ')
    ckpt = tmp_path / "ckpt"
    for n in (90, 100):
        (ckpt / f"checkpoint-{n}").mkdir(parents=True)
        (ckpt / f"checkpoint-{n}" / "trainer_state.json").write_text("{}")
    (ckpt / "checkpoint-110.tmp").mkdir()
    (res / ".synced_checkpoint").write_text("90\n")

    now = st._epoch("2026-11-25T03:30:00Z")
    s = status(res, ckpt, now=now, alive=lambda pid: pid == 4242)
    by = {p["name"]: p for p in s["phases"]}
    assert s["running"] is True
    assert s["current_phase"] == "grpo"
    assert by["grpo"]["status"] == "running" and by["grpo"]["started"] == "2026-11-25T03:00:00Z"
    assert by["pre_eval"] == {"name": "pre_eval", "status": "done",
                              "started": "2026-11-25T01:00:00Z", "finished": "2026-11-25T02:00:00Z"}
    assert by["post_eval"]["status"] == "pending"
    assert s["grpo"] == {"step": 120, "max_steps": 400, "last_checkpoint": 100, "synced_checkpoint": 90}
    assert s["elapsed_s"] == 7200 + 1800


def test_dead_pipeline_shows_failed_phase(tmp_path):
    res = make_results(tmp_path)
    ph = res / ".phases"
    (ph / "smoke.done").write_text("2026-11-25T01:00:00Z\n")
    (ph / "screen.started").write_text("2026-11-25T01:00:00Z\n")
    (ph / "screen.failed").write_text("2026-11-25T01:10:00Z exit 1\n")
    (ph / ".pid").write_text("999999\n")  # stale: run3.sh died
    (ph / ".current").write_text("screen\n")
    s = status(res, tmp_path / "ckpt", alive=lambda pid: False)
    assert s["running"] is False
    assert s["failed_phase"] == "screen"
    assert s["current_phase"] == "screen"
    assert [p["status"] for p in s["phases"]][:2] == ["done", "failed"]


def test_retry_of_a_failed_phase_shows_running(tmp_path):
    res = make_results(tmp_path)
    ph = res / ".phases"
    (ph / "smoke.failed").write_text("x")
    (ph / "smoke.started").write_text("2026-11-25T01:00:00Z")
    (ph / ".current").write_text("smoke")
    (ph / ".pid").write_text("7")
    s = status(res, tmp_path / "ckpt", alive=lambda pid: True)
    assert s["phases"][0]["status"] == "running"
    assert s["failed_phase"] is None


def test_all_done(tmp_path):
    res = make_results(tmp_path)
    for name in st.PHASES:
        (res / ".phases" / f"{name}.done").write_text("2026-11-25T09:00:00Z\n")
    (res / "run.log").write_text("[12:00:00] All requested phases finished.\r\nALL DONE\n\n")
    s = status(res, tmp_path / "ckpt")
    assert s["done"] is True and s["all_done_line"] is True
    assert s["current_phase"] is None


def test_gpu_stats_parses_nvidia_smi_and_survives_its_absence():
    class Out:
        returncode = 0
        stdout = "97, 61000, 81559, 610.52\n"

    assert st.gpu_stats(lambda *a, **k: Out()) == {"util": 97, "mem_used_mb": 61000, "mem_total_mb": 81559,
                                                    "power_w": 610.5}

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert st.gpu_stats(missing) == NO_GPU


def test_status_json_cli_is_fast_and_never_imports_torch(tmp_path):
    res = make_results(tmp_path)
    (res / ".phases" / "smoke.done").write_text("2026-11-25T01:00:00Z\n")
    env = dict(os.environ, RUN3_RESULTS=str(res), RUN3_CKPT=str(tmp_path / "ckpt"))
    t0 = time.perf_counter()
    out = subprocess.run(["bash", str(REPO / "scripts" / "run3.sh"), "--status-json"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert time.perf_counter() - t0 < 5
    assert out.returncode == 0
    data = json.loads(out.stdout)
    assert data["phases"][0]["status"] == "done" and data["current_phase"] == "screen"

    probe = ("import runpy, sys; sys.argv = ['x', '--results', sys.argv[1]]; "
             "runpy.run_module('cua.run3_status', run_name='__main__'); "
             "assert 'torch' not in sys.modules, 'imported torch'")
    out = subprocess.run([sys.executable, "-c", probe, str(res)], capture_output=True, text=True,
                         cwd=REPO, timeout=30)
    assert out.returncode == 0, out.stderr
