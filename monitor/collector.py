"""Collector and note writer for the run-3 monitor. Both run as threads in the monitor.

The collector fills ~/cua-run3/data/ every 60 s while a pod is up. Every
failure here is logged and swallowed: the collector must never break the
monitor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from common import (
    append_jsonl_atomic, atomic_write_json, atomic_write_text, build_evals, clean_log_text,
    now_iso, read_json, read_jsonl, sample_index,
)

HERE = Path(__file__).resolve().parent
PROBE = (HERE / "pod_probe.py").read_bytes()
KEEP_SAMPLES = 30
COLLECT_STATES = {"bootstrapping", "running", "restarting", "stopping", "uploading"}


class Collector:
    def __init__(self, mon, clock=time.time, wandb_every_s: float = 300):
        self.mon = mon
        self.paths = mon.paths
        self.scrub = mon.scrub
        self.clock = clock
        self.wandb_every_s = wandb_every_s
        self.cache_file = self.paths.cache / "collector.json"

    # --- cache --------------------------------------------------------------------

    def cache(self) -> dict:
        c = read_json(self.cache_file, default=None) or {}
        c.setdefault("metrics_offset", 0)
        c.setdefault("metrics_pod", None)
        c.setdefault("last_step", None)
        c.setdefault("wandb_last_ts", 0)
        return c

    def save_cache(self, c: dict) -> None:
        atomic_write_json(self.cache_file, c, mode=0o600)

    # --- one pass -------------------------------------------------------------------

    def active(self) -> bool:
        return self.mon.st.get("run_state") in COLLECT_STATES and self.mon.live_pod_exists()

    def collect_once(self) -> bool:
        rec = dict(self.mon.pod_rec() or {})
        conn = self.mon.pod_conn(rec) if rec else None
        if conn is None:
            return False
        c = self.cache()
        if c["metrics_pod"] != rec.get("id"):
            # New pod: its restored metrics.jsonl may differ in length, so read it
            # from the start and rely on step de-duplication.
            c["metrics_offset"], c["metrics_pod"] = 0, rec.get("id")
        r = conn.run(f"python3 - --metrics-offset {int(c['metrics_offset'])} --log-lines 2000", stdin=PROBE, timeout=120)
        if not r.ok:
            self.mon.log(f"collector: probe failed (rc {r.rc}): {r.err.strip()[-200:]}")
            return False
        try:
            line = [ln for ln in r.out.strip().splitlines() if ln.startswith("{")][-1]
            probe = json.loads(line)
        except (IndexError, ValueError):
            self.mon.log("collector: probe printed no JSON")
            return False
        self.apply_probe(probe, c, conn)
        self.save_cache(c)
        return True

    def apply_probe(self, probe: dict, c: dict, conn) -> None:
        mon = self.mon
        status = probe.get("status")
        if isinstance(status, dict) and mon.lock.acquire(timeout=5):
            try:
                if mon.st.get("run_state") in COLLECT_STATES:
                    mon.absorb_status(status)
                    mon.save()
            finally:
                mon.lock.release()
        self._guard("metrics", self.metrics, probe.get("metrics"), c)
        self._guard("log", self.log_tail, probe.get("log_tail"))
        self._guard("evals", self.evals, probe.get("evals") or {})
        self._guard("samples", self.samples, probe.get("samples") or [], conn)
        gpu = probe.get("gpu") or ((status or {}).get("gpu") if isinstance(status, dict) else None)
        self._guard("system", self.system, gpu)
        wr = probe.get("wandb_run")
        if isinstance(wr, dict) and wr.get("url"):
            mon.st["wandb_url"] = wr["url"]
        if isinstance(wr, dict) and self.clock() - c.get("wandb_last_ts", 0) >= self.wandb_every_s:
            c["wandb_last_ts"] = self.clock()
            self._guard("wandb", self.wandb, wr, c)

    def _guard(self, what, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.mon.log(f"collector: {what} failed: {e!r}")

    # --- pieces ---------------------------------------------------------------------

    def metrics(self, m, c: dict) -> None:
        if not m:
            return
        # If the pod's file shrank (a restored pod), the probe reads it from the
        # start; rows at or below the last step seen are skipped.
        rows = []
        last = c.get("last_step")
        if last is None:
            local = read_jsonl(self.paths.d("metrics.jsonl"))
            steps = [r.get("step") for r in local if isinstance(r.get("step"), (int, float))]
            last = max(steps) if steps else None
        # A GRPO resume replays the steps after its checkpoint, and the pod trims
        # those rows from its own file. Rows at or below the last step seen are
        # either already copied (a re-read from offset 0) or replays. Replays
        # arrive as new data after the checkpoint step, so the first new row
        # whose step is at or below `last` and that the local copy holds with
        # different values starts a rewind: local rows from that step on are
        # dropped and the pod's rows win (newest values per step).
        local_rows = None
        rewind_from = None
        for ln in m.get("text", "").splitlines():
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            step = row.get("step")
            if isinstance(step, (int, float)) and last is not None and step <= last and rewind_from is None:
                if local_rows is None:
                    local_rows = {r.get("step"): r for r in read_jsonl(self.paths.d("metrics.jsonl"))}
                old = local_rows.get(step)
                if old is None or old == self.scrub.obj(row):
                    continue  # already copied
                rewind_from = step
            rows.append(self.scrub.obj(row))
            if isinstance(step, (int, float)):
                last = step
        if rewind_from is not None:
            kept = [r for r in read_jsonl(self.paths.d("metrics.jsonl"))
                    if not (isinstance(r.get("step"), (int, float)) and r["step"] >= rewind_from)]
            atomic_write_text(self.paths.d("metrics.jsonl"),
                              "".join(json.dumps(r) + "\n" for r in kept + rows))
            self.mon.event("note", f"GRPO resumed: replaced metric rows from step {rewind_from} on.")
        else:
            append_jsonl_atomic(self.paths.d("metrics.jsonl"), rows)
        c["metrics_offset"] = m["end"]
        c["last_step"] = last

    def log_tail(self, text) -> None:
        if text is None:
            return
        atomic_write_text(self.paths.d("log.txt"), clean_log_text(text, self.scrub, 2000))

    def evals(self, evals: dict) -> None:
        if not evals:
            return
        root = self.paths.cache / "evals"
        for name, e in evals.items():
            if "/" in name or name.startswith("."):
                continue
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            if e.get("per_task"):
                atomic_write_text(d / "per_task.csv", e["per_task"])
            if e.get("summary"):
                atomic_write_json(d / "summary.json", e["summary"])
        atomic_write_json(self.paths.d("evals.json"), self.scrub.obj(build_evals(root)))

    def samples(self, names: list, conn) -> None:
        sd = self.paths.d("samples")
        sd.mkdir(parents=True, exist_ok=True)
        names = [n for n in names if n.startswith("step-") and "/" not in n and " " not in n]
        want = sorted(names)[-KEEP_SAMPLES:]
        have = {p.name for p in sd.iterdir() if p.is_dir()}
        missing = [n for n in want if n not in have]
        if missing:
            with tempfile.TemporaryDirectory(dir=self.paths.cache) as td:
                r = conn.pipe_to("tar -C results/run3/samples -cf - " + " ".join(missing), ["tar", "-xf", "-", "-C", td], timeout=300)
                if r.ok:
                    for n in missing:
                        src = Path(td) / n
                        if not (src / "episode.json").exists():
                            continue
                        ep = read_json(src / "episode.json", default={}) or {}
                        atomic_write_json(src / "episode.json", self.scrub.obj(ep))
                        dst = sd / n
                        if dst.exists():
                            shutil.rmtree(dst)
                        os.replace(src, dst)
                else:
                    self.mon.log(f"collector: sample tar failed (rc {r.rc}): {r.err.strip()[-200:]}")
        keep = set(sorted({p.name for p in sd.iterdir() if p.is_dir() and p.name.startswith("step-")})[-KEEP_SAMPLES:])
        for p in sd.iterdir():
            if p.is_dir() and p.name not in keep:
                shutil.rmtree(p, ignore_errors=True)
        atomic_write_json(sd / "index.json", sample_index(sd))

    def system(self, gpu) -> None:
        gpu = gpu or {}
        append_jsonl_atomic(self.paths.d("system.jsonl"), [{
            "t": now_iso(self.clock()),
            "gpu_util": gpu.get("util"),
            "gpu_mem_mb": gpu.get("mem_used_mb"),
            "gpu_power_w": gpu.get("power_w"),
            "spend_usd": round(self.mon.spend(), 3),
        }])

    def wandb(self, wr: dict, c: dict) -> None:
        """Fetch W&B history in a subprocess with a timeout. Any failure is only logged."""
        py = self.paths.home / "venv" / "bin" / "python"
        if not py.exists() or not all(wr.get(k) for k in ("entity", "project", "id")):
            return
        key = self.mon.env.get("WANDB_API_KEY")
        if not key:
            return
        out = self.paths.cache / "wandb.json"
        env = dict(os.environ, WANDB_API_KEY=key, WANDB_SILENT="true")
        try:
            p = subprocess.run([str(py), str(HERE / "wandb_fetch.py"), "--path", f"{wr['entity']}/{wr['project']}/{wr['id']}",
                                "--out", str(out)], env=env, capture_output=True, timeout=240)
        except subprocess.TimeoutExpired:
            self.mon.log("collector: wandb fetch timed out")
            return
        if p.returncode != 0:
            self.mon.log(f"collector: wandb fetch failed: {self.scrub(p.stderr.decode(errors='replace'))[-300:]}")
            return
        data = read_json(out, default={}) or {}
        sysrows = data.get("system") or []
        if sysrows:
            atomic_write_text(self.paths.d("wandb_system.jsonl"),
                              "".join(json.dumps(self.scrub.obj(r)) + "\n" for r in sysrows))
        hist = data.get("history") or []
        last = c.get("last_step")
        add = []
        for row in sorted((h for h in hist if isinstance(h.get("step"), (int, float))), key=lambda h: h["step"]):
            if last is None or row["step"] > last:
                row = dict(row, _source="wandb")
                add.append(self.scrub.obj(row))
                last = row["step"]
        if add:
            append_jsonl_atomic(self.paths.d("metrics.jsonl"), add)
            c["last_step"] = last
            self.mon.log(f"collector: added {len(add)} metric rows from W&B that the pod's file lacked")

    # --- loop -----------------------------------------------------------------------

    def loop(self, stop) -> None:
        while not stop.is_set():
            t0 = time.time()
            try:
                if self.active():
                    self.collect_once()
            except Exception as e:  # noqa: BLE001
                self.mon.log(f"collector: pass crashed: {e!r}")
            stop.wait(max(5, self.mon.cfg["collect_s"] - (time.time() - t0)) if self.active() else 15)


class NoteWriter:
    """Every 30 minutes, ask `claude -p` for a short plain-language status note."""

    def __init__(self, mon, clock=time.time, runner=None):
        self.mon = mon
        self.clock = clock
        self.runner = runner or self._run_claude
        self.prompt = (HERE / "prompt.md").read_text(encoding="utf-8")

    def context(self) -> str:
        d = self.mon.paths.data
        state = read_json(d / "state.json", default={}) or {}
        events = read_jsonl(d / "events.jsonl")[-15:]
        metrics = read_jsonl(d / "metrics.jsonl")[-3:]
        keys = ("step", "reward/total", "success/overall", "groups/zero_variance_frac", "dyn/mixed_groups",
                "episodes/trained", "time/rollout_s", "time/train_s", "train/clip_frac", "train/loss", "eval/success")
        metrics = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in keys} for m in metrics]
        try:
            log = (d / "log.txt").read_text(encoding="utf-8").splitlines()[-40:]
        except OSError:
            log = []
        notes = read_json(d / "notes.json", default=[]) or []
        prev = notes[-1]["text"] if notes else "none"
        return (f"## data/state.json\n{json.dumps(state, indent=1)}\n\n## Last events\n"
                + "\n".join(json.dumps(e) for e in events)
                + f"\n\n## Last metric rows\n" + "\n".join(json.dumps(m) for m in metrics)
                + "\n\n## Last 40 log lines\n" + "\n".join(log)
                + f"\n\n## Previous note\n{prev}\n")

    def _run_claude(self, text: str) -> str | None:
        argv = ["claude", "-p", "--allowedTools", "Read"]
        model = self.mon.cfg.get("notes_model")
        if model:
            argv += ["--model", model]
        try:
            p = subprocess.run(argv, input=text.encode(), capture_output=True, timeout=300,
                               cwd=str(self.mon.paths.data))
        except (subprocess.TimeoutExpired, OSError) as e:
            self.mon.log(f"notes: claude failed: {e!r}")
            return None
        if p.returncode != 0:
            self.mon.log(f"notes: claude exited {p.returncode}")
            return None
        return p.stdout.decode(errors="replace")

    def write_once(self) -> bool:
        out = self.runner(self.prompt + "\n\n" + self.context())
        if not out:
            return False
        lines = [ln for ln in out.strip().splitlines() if not ln.startswith("[claude-code:")]
        text = self.mon.scrub("\n".join(lines).strip())[:1500]
        if len(text) < 20:
            return False
        path = self.mon.paths.d("notes.json")
        notes = read_json(path, default=[]) or []
        notes.append({"t": now_iso(self.clock()), "text": text})
        atomic_write_json(path, notes[-200:])
        return True

    def loop(self, stop) -> None:
        last = 0.0
        written_terminal = False
        while not stop.is_set():
            st = self.mon.st.get("run_state")
            active = st not in ("idle", "done", "failed")
            terminal_once = st in ("done", "failed") and not written_terminal and last > 0
            if (active and time.time() - last >= self.mon.cfg["notes_s"]) or terminal_once:
                try:
                    self.write_once()
                except Exception as e:  # noqa: BLE001
                    self.mon.log(f"notes: {e!r}")
                last = time.time()
                written_terminal = terminal_once or written_terminal
            if active:
                written_terminal = False
            stop.wait(30)
