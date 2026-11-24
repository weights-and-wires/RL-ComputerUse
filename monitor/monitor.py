#!/usr/bin/env python3
"""Run-3 monitor: creates and watches the Runpod pod, enforces the budget, tears down.

Runs on Bazzite as the user systemd unit `cua-run3-monitor`. The state lives in
~/cua-run3/state.json and survives restarts of this process. See README.md.

Usage:
  monitor.py run            Main loop (what the systemd unit runs).
  monitor.py once           One state-machine tick, then exit.
  monitor.py status         Print a short status summary.
  monitor.py plan           Print the pod plan and live prices (read-only API calls).
  monitor.py stop|kill      Queue a graceful stop or an immediate teardown.
  monitor.py --dry-run      Run the built-in scenario against a fake pod and API.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    Paths, Scrubber, append_jsonl_atomic, atomic_write_json, load_env, now_iso, parse_iso, read_json,
)
from podssh import REMOTE_REPO, SSHPod  # noqa: E402
from runpod import GONE_STATUSES, ApiError, RunpodAPI, normalize_pod  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "budget_usd": 50.0,
    "soft_stop_usd": 42.0,
    "hard_stop_usd": 48.0,
    "wall_limit_h": 20.0,
    "poll_s": 120,
    "fast_poll_s": 20,
    "collect_s": 60,
    "notes_s": 1800,
    "ssh_lost_after_s": 900,
    "bootstrap_timeout_s": 1500,
    "max_generations": 4,
    "max_phase_failures": 3,
    "max_create_attempts": 30,
    "upload_attempts": 2,
    "upload_timeout_s": 5400,
    "start_timeout_s": 2400,
    "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
    "container_disk_gb": 150,
    "min_vcpu_per_gpu": 16,
    "min_ram_per_gpu": 64,
    "start_ssh": True,
    # Tried in order until one has stock.
    "gpu_candidates": [
        # Community first: about $2.69/hr against $4.39/hr on Secure (2026-11-24),
        # which is what lets the plan's 12 h fit the $50 cap. A lost community
        # pod is recreated and resumes from the last synced checkpoint.
        ["NVIDIA H100 80GB HBM3", "COMMUNITY"],
        ["NVIDIA H100 80GB HBM3", "SECURE"],
        ["NVIDIA H100 PCIe", "COMMUNITY"],
        ["NVIDIA H100 NVL", "COMMUNITY"],
        ["NVIDIA H100 PCIe", "SECURE"],
        ["NVIDIA H100 NVL", "SECURE"],
    ],
    # Used for spend until the API reports the pod's own hourly rate.
    "fallback_cost_per_hr": 4.50,
    "ssh_key": "~/.ssh/id_ed25519",
}

ACTIVE_POD_STATES = {"bootstrapping", "running", "restarting", "stopping", "uploading"}
TERMINAL = {"done", "failed"}


def load_config(paths: Paths) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(read_json(paths.home / "config.json", default={}) or {})
    return cfg


def new_state() -> dict:
    return {
        "version": 1,
        "demo": False,
        "run_state": "idle",
        "started_ts": None,
        "updated": None,
        "pods": [],             # every pod this monitor created or adopted
        "current_pod": None,
        "generation": 0,
        "restarts": 0,
        "fail_phase": None,
        "fail_count": 0,
        "fail_progress": None,
        "ssh_fail_since": None,
        "soft_stop_sent": False,
        "stop_requested": False,
        "teardown_reason": None,
        "final_outcome": None,  # done | failed, set when teardown starts
        "upload_verified": False,
        "upload_attempts": 0,
        "create_attempts": 0,
        "seen_phases": {},
        "last_status": None,
        "last_status_ts": None,
        "last_error": None,
        "adopt_id": None,
        "wandb_url": None,
        "eta": None,
    }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class Monitor:
    def __init__(self, paths: Paths, api, pod_factory, clock=time.time, config: dict | None = None,
                 env: dict | None = None, out=None, local_tar_argv=None):
        self.paths = paths
        self.api = api
        self.pod_factory = pod_factory
        self.clock = clock
        self.cfg = config or load_config(paths)
        self.env = env if env is not None else load_env(paths.env)
        self.scrub = Scrubber(self.env)
        self.out = out or sys.stdout
        self.lock = threading.RLock()
        self.local_tar_argv = local_tar_argv
        self.st = self._load()

    # --- persistence ------------------------------------------------------------

    def _load(self) -> dict:
        st = read_json(self.paths.state, default=None)
        base = new_state()
        if isinstance(st, dict):
            base.update(st)
        return base

    def save(self) -> None:
        with self.lock:
            self.st["updated"] = now_iso(self.clock())
            atomic_write_json(self.paths.state, self.st, mode=0o600)
            self.write_public_state()

    def log(self, msg: str) -> None:
        line = f"[{now_iso(self.clock())}] {self.scrub(msg)}"
        print(line, file=self.out, flush=True)
        try:
            with open(self.paths.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def event(self, kind: str, msg: str) -> None:
        self.log(f"EVENT {kind}: {msg}")
        append_jsonl_atomic(self.paths.d("events.jsonl"),
                            [{"t": now_iso(self.clock()), "kind": kind, "msg": self.scrub(msg)}])

    def set_state(self, new: str, why: str = "") -> None:
        old = self.st["run_state"]
        if old == new:
            return
        self.st["run_state"] = new
        self.log(f"state {old} -> {new}{': ' + why if why else ''}")
        self.save()

    def error(self, msg: str, loud: bool = True) -> None:
        self.st["last_error"] = self.scrub(msg)[:1000]
        if loud:
            self.event("error", msg)
        else:
            self.log("error: " + msg)

    # --- pods we own --------------------------------------------------------------

    def pod_rec(self, pod_id: str | None = None) -> dict | None:
        pid = pod_id or self.st["current_pod"]
        for p in self.st["pods"]:
            if p["id"] == pid:
                return p
        return None

    def owned_ids(self) -> set[str]:
        return {p["id"] for p in self.st["pods"]}

    def pod_conn(self, rec: dict | None = None):
        rec = rec or self.pod_rec()
        if not rec or not rec.get("ssh_host") or not rec.get("ssh_port"):
            return None
        return self.pod_factory(rec["ssh_host"], rec["ssh_port"])

    # --- spend --------------------------------------------------------------------

    def pod_spend(self, rec: dict, now: float | None = None) -> float:
        now = now or self.clock()
        start = rec.get("billing_start_ts") or rec.get("created_ts") or now
        end = rec.get("ended_ts") or now
        rate = rec.get("cost_per_hr") or self.cfg["fallback_cost_per_hr"]
        return max(0.0, (end - start) / 3600.0 * float(rate))

    def spend(self) -> float:
        """Total spend over every pod generation. Doesn't modify state (the collector calls it too)."""
        now = self.clock()
        return round(sum(self.pod_spend(rec, now) for rec in list(self.st["pods"])), 4)

    def refresh_spend(self) -> float:
        now = self.clock()
        for rec in self.st["pods"]:
            rec["spend_usd"] = round(self.pod_spend(rec, now), 4)
            rec["uptime_s"] = int((rec.get("ended_ts") or now) - (rec.get("billing_start_ts") or rec.get("created_ts") or now))
        return self.spend()

    # --- public data/state.json ---------------------------------------------------

    def write_public_state(self) -> None:
        st = self.st
        if st["run_state"] == "idle" and self.paths.d(".demo").exists():
            return  # keep the demo data until a real run starts
        status = st.get("last_status") or {}
        rec = self.pod_rec()
        hf_repo = self.env.get("HF_ADAPTER_REPO")
        spend = self.refresh_spend()
        pub = {
            "demo": False,
            "run_state": st["run_state"],
            "updated": now_iso(self.clock()),
            "started": now_iso(st["started_ts"]) if st.get("started_ts") else None,
            "phases": status.get("phases") or [],
            "current_phase": status.get("current_phase"),
            "grpo": status.get("grpo") or {},
            "pod": {
                "id": rec["id"] if rec else None,
                "gpu": rec.get("gpu") if rec else None,
                "cloud": rec.get("cloud") if rec else None,
                "cost_per_hr": rec.get("cost_per_hr") if rec else None,
                "uptime_s": rec.get("uptime_s") if rec else None,
                "generation": rec.get("generation") if rec else st["generation"],
            },
            "spend_usd": round(spend, 2),
            "budget_usd": self.cfg["budget_usd"],
            "eta": st.get("eta"),
            "restarts": st["restarts"],
            "pod_generations": st["generation"],
            "wandb_url": st.get("wandb_url"),
            "hf_url": f"https://huggingface.co/{hf_repo}" if hf_repo else None,
            "last_error": st.get("last_error"),
            "gpu": status.get("gpu"),
            "teardown_reason": st.get("teardown_reason"),
            "soft_stop_sent": st.get("soft_stop_sent"),
        }
        atomic_write_json(self.paths.d("state.json"), self.scrub.obj(pub))

    # --- commands from run3_go.sh ---------------------------------------------------

    def armed(self) -> dict | None:
        if not self.paths.armed.exists():
            return None
        return read_json(self.paths.armed, default={}) or {}

    def take_command(self) -> str | None:
        cmd = read_json(self.paths.command, default=None)
        if not cmd:
            return None
        try:
            self.paths.command.unlink()
        except OSError:
            pass
        return (cmd or {}).get("cmd")

    def handle_command(self, cmd: str) -> None:
        state = self.st["run_state"]
        self.log(f"command received: {cmd} (state {state})")
        if cmd == "stop":
            if state in TERMINAL:
                return
            self.st["stop_requested"] = True
            if state in ("idle", "creating_pod") and not self.st["current_pod"]:
                self.event("note", "Stop requested before a pod existed. Nothing to tear down.")
                self.finish("done" if state == "idle" else "failed", "stopped before a pod existed")
                return
            rec = self.pod_rec()
            if state == "bootstrapping" and rec and not (rec.get("bootstrap") or {}).get("started"):
                # Nothing has run yet, so there is nothing to upload.
                self.event("note", "Stop requested during bootstrap, before the pipeline started. Deleting the pod.")
                self.st["teardown_reason"], self.st["final_outcome"] = "owner_stop", "failed"
                self.set_state("deleting", "stop during bootstrap")
            elif state in ("running", "restarting", "bootstrapping"):
                self.soft_stop("Stop requested by the owner.")
                self.set_state("stopping", "owner requested a graceful stop")
            elif state == "recreating_pod":
                self.begin_teardown("owner_stop")
        elif cmd == "kill":
            if state in TERMINAL and not self.live_pod_exists():
                return
            self.begin_teardown("owner_kill")
        self.save()

    def live_pod_exists(self) -> bool:
        rec = self.pod_rec()
        return bool(rec and not rec.get("deleted_ts"))

    # --- the tick -------------------------------------------------------------------

    def tick(self) -> None:
        with self.lock:
            cmd = self.take_command()
            if cmd:
                self.handle_command(cmd)
            state = self.st["run_state"]
            if state not in TERMINAL and state != "idle":
                if self.check_limits():
                    self.save()
                    return
            handler = getattr(self, f"on_{state}", None)
            if handler is None:
                self.error(f"unknown state {state}; resetting to failed")
                self.set_state("failed")
                return
            try:
                handler()
            except Exception as e:  # never die on one bad tick
                self.error(f"tick crashed in {state}: {e!r}\n{traceback.format_exc()[-1500:]}")
            self.save()

    def check_limits(self) -> bool:
        """Budget and wall-clock enforcement. Returns True when it changed the state."""
        spend = self.spend()
        state = self.st["run_state"]
        cfg = self.cfg
        if state in ("uploading", "deleting"):
            return False
        hard = spend >= cfg["hard_stop_usd"]
        started = self.st.get("started_ts")
        timeout = started is not None and self.clock() - started >= cfg["wall_limit_h"] * 3600
        if hard or timeout:
            reason = "hard_budget" if hard else "wall_timeout"
            self.event("error", f"Hard stop ({reason}): spend ${spend:.2f}, "
                                f"{(self.clock() - (started or self.clock())) / 3600:.1f} h since start. Tearing down now.")
            self.begin_teardown(reason)
            return True
        if spend >= cfg["soft_stop_usd"] and not self.st["soft_stop_sent"] and state in ("running", "stopping", "restarting"):
            self.soft_stop(f"Spend ${spend:.2f} reached the ${cfg['soft_stop_usd']:.0f} soft limit.", kind="budget_soft_stop")
        return False

    def soft_stop(self, why: str, kind: str = "note") -> bool:
        if self.st["soft_stop_sent"]:
            return True
        conn = self.pod_conn()
        if not conn:
            return False
        r = conn.run("mkdir -p results/run3 && touch results/run3/.stop_grpo", timeout=60)
        if r.ok:
            self.st["soft_stop_sent"] = True
            self.event(kind, f"{why} Touched results/run3/.stop_grpo: GRPO saves and ends at the next step, then post-training evals run.")
            return True
        self.log(f"soft stop failed (rc {r.rc}); will retry: {r.err[-200:]}")
        return False

    # --- idle -------------------------------------------------------------------------

    def on_idle(self) -> None:
        arm = self.armed()
        if arm is None:
            return
        self.reset_for_new_run()
        self.st["started_ts"] = self.clock()
        self.st["adopt_id"] = arm.get("adopt")
        self.clear_demo_data()
        self.event("note", "Monitor armed. Run 3 starting." + (f" Adopting pod {arm['adopt']}." if arm.get("adopt") else ""))
        self.set_state("creating_pod", "armed")

    def reset_for_new_run(self) -> None:
        keep = {"pods"}
        fresh = new_state()
        for k, v in fresh.items():
            if k not in keep:
                self.st[k] = v
        self.st["pods"] = []

    def clear_demo_data(self) -> None:
        mark = self.paths.d(".demo")
        pub = read_json(self.paths.d("state.json"), default={}) or {}
        if not mark.exists() and not pub.get("demo"):
            return
        import shutil
        info = read_json(mark, default={}) or {}
        for name in info.get("files") or ["metrics.jsonl", "system.jsonl", "evals.json", "log.txt",
                                          "events.jsonl", "notes.json", "samples", "state.json"]:
            p = self.paths.d(name)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass
        for extra in ("wandb_system.jsonl",):
            try:
                self.paths.d(extra).unlink()
            except OSError:
                pass
        try:
            mark.unlink()
        except OSError:
            pass
        # The collector keeps offsets for the files it mirrors; reset them too.
        try:
            (self.paths.cache / "collector.json").unlink()
        except OSError:
            pass
        self.log("cleared demo data from data/")

    # --- creating_pod ---------------------------------------------------------------

    def pod_name(self, gen: int) -> str:
        return f"cua-run3-{int(self.st['started_ts'] or 0)}-g{gen}"

    def public_keys(self) -> str:
        keys = []
        for p in (Path(os.path.expanduser(self.cfg["ssh_key"]) + ".pub"), self.paths.extra_keys):
            try:
                for line in Path(p).read_text().splitlines():
                    line = line.strip()
                    if line.startswith("ssh-") and line not in keys:
                        keys.append(line)
            except OSError:
                pass
        return "\n".join(keys)

    def pod_body(self, gpu: str, cloud: str, name: str, start_ssh: bool) -> dict:
        keys = self.public_keys()
        body = {
            "name": name,
            "image": self.cfg["image"],
            "gpu": {"id": gpu, "count": 1, "minRamPerGpu": self.cfg["min_ram_per_gpu"],
                    "minVcpuCountPerGpu": self.cfg["min_vcpu_per_gpu"]},
            "cloud": cloud,
            "disk": self.cfg["container_disk_gb"],
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": keys, "SSH_PUBLIC_KEY": keys},
        }
        if start_ssh:
            body["startSsh"] = True
        return body

    def record_pod(self, raw: dict, gen: int, adopted: bool, gpu: str | None = None, cloud: str | None = None) -> dict:
        n = normalize_pod(raw)
        now = self.clock()
        billing_start = now
        if adopted:
            billing_start = parse_iso(n.get("started_at")) or parse_iso(n.get("created_at")) or now
        rec = {
            "id": n["id"], "name": n.get("name"), "generation": gen, "adopted": adopted,
            "gpu": n.get("gpu") or gpu, "cloud": n.get("cloud") or cloud,
            "cost_per_hr": n.get("cost_per_hr"), "created_ts": now, "billing_start_ts": billing_start,
            "ended_ts": None, "deleted_ts": None, "ssh_host": n.get("ssh_host"), "ssh_port": n.get("ssh_port"),
            "api_status": n.get("status"), "bootstrap": {}, "spend_usd": 0.0, "uptime_s": 0,
        }
        self.st["pods"].append(rec)
        self.st["current_pod"] = rec["id"]
        self.st["generation"] = gen
        self.st["ssh_fail_since"] = None
        return rec

    def on_creating_pod(self) -> None:
        if self.st["current_pod"] and self.live_pod_exists():
            self.set_state("bootstrapping")
            return
        if not self.armed():
            self.log("ARMED is missing, so the monitor won't create a pod. Waiting.")
            return
        gen = self.st["generation"] + 1
        if gen > self.cfg["max_generations"]:
            self.error(f"Pod generation cap ({self.cfg['max_generations']}) reached. Not creating another pod.")
            self.finish("failed", "generation cap reached")
            return

        adopt = self.st.get("adopt_id")
        if adopt:
            raw = self.api.get_pod(adopt)
            if not raw:
                self.error(f"Pod {adopt} to adopt doesn't exist (404).")
                self.finish("failed", "adopt target missing")
                return
            rec = self.record_pod(raw, gen, adopted=True)
            self.st["adopt_id"] = None
            self.event("pod_created", f"Adopted pod {rec['id']} ({rec['gpu']}, {rec['cloud']}, "
                                      f"${rec['cost_per_hr'] or '?'}/hr), generation {gen}.")
            self.set_state("bootstrapping", "adopted")
            return

        # Idempotency: if an earlier tick created the pod but crashed before recording
        # it, find it by its unique name instead of creating a second one.
        name = self.pod_name(gen)
        try:
            for raw in self.api.list_pods():
                if raw.get("name") == name and raw.get("id") not in self.owned_ids():
                    rec = self.record_pod(raw, gen, adopted=False)
                    self.event("pod_created", f"Found pod {rec['id']} ({name}) from an interrupted create; using it.")
                    self.set_state("bootstrapping")
                    return
        except ApiError as e:
            self.log(f"list_pods failed: {e}")

        self.st["create_attempts"] += 1
        tried = []
        for gpu, cloud in self.cfg["gpu_candidates"]:
            start_ssh = bool(self.cfg.get("start_ssh", True))
            for _ in range(2):
                try:
                    raw = self.api.create_pod(self.pod_body(gpu, cloud, name, start_ssh))
                except ApiError as e:
                    if e.status == 422 and start_ssh and "ssh" in e.detail.lower():
                        start_ssh = False  # retry without startSsh; the image reads PUBLIC_KEY itself
                        continue
                    tried.append(f"{gpu}/{cloud}: {e}")
                    if e.status in (401, 403):
                        self.error(f"Runpod refused the API key ({e}). Fix RUNPOD_API_KEY in ~/cua-run3/.env.")
                        return
                    if e.status == 0 or e.status == 429 or e.status >= 500:
                        self.log(f"create_pod transient error: {e}; retrying next tick")
                        return
                    break
                if not raw or not raw.get("id"):
                    tried.append(f"{gpu}/{cloud}: no pod id in response")
                    break
                rec = self.record_pod(raw, gen, adopted=False, gpu=gpu, cloud=cloud)
                self.st["create_attempts"] = 0
                kind = "pod_created" if gen == 1 else "pod_recreated"
                fallback = "" if (gpu, cloud) == tuple(self.cfg["gpu_candidates"][0]) else " (fallback choice)"
                self.event(kind, f"Created pod {rec['id']} on {gpu} / {cloud}{fallback}, "
                                 f"${rec['cost_per_hr'] or '?'}/hr, generation {gen}.")
                self.set_state("bootstrapping", "pod created")
                return
        self.error(f"No H100 stock for any candidate (attempt {self.st['create_attempts']}): " + " | ".join(tried), loud=self.st["create_attempts"] % 5 == 1)
        if self.st["create_attempts"] >= self.cfg["max_create_attempts"]:
            self.finish("failed", "could not create a pod")

    # --- bootstrapping ---------------------------------------------------------------

    def refresh_pod(self, rec: dict) -> str:
        """GET the pod. Returns 'ok', 'gone', or 'unknown' (API trouble)."""
        try:
            raw = self.api.get_pod(rec["id"])
        except ApiError as e:
            self.log(f"get_pod failed: {e}")
            return "unknown"
        if raw is None:
            rec["api_status"] = "NOT_FOUND"
            return "gone"
        n = normalize_pod(raw)
        rec["api_status"] = n.get("status")
        if n.get("cost_per_hr"):
            rec["cost_per_hr"] = n["cost_per_hr"]
        if n.get("ssh_host") and n.get("ssh_port"):
            rec["ssh_host"], rec["ssh_port"] = n["ssh_host"], n["ssh_port"]
        if n.get("gpu"):
            rec["gpu"] = n["gpu"]
        if (n.get("status") or "").upper() in GONE_STATUSES:
            return "gone"
        return "ok"

    def on_bootstrapping(self) -> None:
        rec = self.pod_rec()
        if not rec:
            self.set_state("creating_pod")
            return
        health = self.refresh_pod(rec)
        if health == "gone":
            self.pod_lost(rec, f"API reports pod {rec['id']} as {rec['api_status']} during bootstrap")
            return
        age = self.clock() - rec["created_ts"]
        conn = self.pod_conn(rec)
        if conn is None or not conn.run("true", timeout=30, in_repo=False).ok:
            if age > self.cfg["bootstrap_timeout_s"]:
                self.pod_lost(rec, f"pod {rec['id']} never accepted SSH within {self.cfg['bootstrap_timeout_s'] // 60} min")
            else:
                self.log(f"waiting for SSH on {rec['id']} ({int(age)} s, status {rec['api_status']})")
            return
        b = rec.setdefault("bootstrap", {})
        if not b.get("ssh_ok"):
            b["ssh_ok"] = True
            self.event("note", f"SSH is up on pod {rec['id']} ({rec['ssh_host']}:{rec['ssh_port']}).")
        if not self.write_secrets(conn):
            return
        b["secrets"] = True
        if not b.get("code"):
            if not self.ship_code(conn):
                return
            b["code"] = True
        self.save()
        if self.start_run(conn, why="bootstrap"):
            b["started"] = True

    def write_secrets(self, conn) -> bool:
        lines = []
        for k in ("HF_TOKEN", "WANDB_API_KEY", "HF_ADAPTER_REPO"):
            v = self.env.get(k)
            if v:
                lines.append(f"export {k}={shlex.quote(v)}\n")
        missing = [k for k in ("HF_TOKEN", "WANDB_API_KEY", "HF_ADAPTER_REPO") if not self.env.get(k)]
        if missing:
            self.log(f"warning: {', '.join(missing)} missing from ~/cua-run3/.env")
        r = conn.run("umask 077; mkdir -p /workspace && cat > /workspace/.secrets.tmp && "
                     "chmod 600 /workspace/.secrets.tmp && mv /workspace/.secrets.tmp /workspace/.secrets",
                     stdin="".join(lines).encode(), timeout=60, in_repo=False)
        if not r.ok:
            self.error(f"writing /workspace/.secrets failed (rc {r.rc}): {r.err[-300:]}", loud=False)
        return r.ok

    def tar_argv(self) -> list[str]:
        if self.local_tar_argv:
            return self.local_tar_argv
        ex = []
        for pat in (".env", ".env.*", "results", ".runpod", ".venv", "venv", "__pycache__", "wandb", "*.pyc",
                    ".pytest_cache", ".ruff_cache", ".demo", "results_live", "results_test_run"):
            ex += ["--exclude", pat]
        return ["tar", "-C", str(self.paths.repo), *ex, "-czf", "-", "."]

    def ship_code(self, conn) -> bool:
        if not (self.paths.repo / "scripts" / "run3_pod_start.sh").exists() and not self.local_tar_argv:
            self.error(f"{self.paths.repo}/scripts/run3_pod_start.sh is missing. Sync the repo with run3_go.sh.")
            return False
        r = conn.pipe_from(self.tar_argv(), f"mkdir -p {REMOTE_REPO} && tar -xzf - -C {REMOTE_REPO}", timeout=900)
        if not r.ok:
            self.error(f"shipping code failed (rc {r.rc}): {r.err[-300:]}", loud=False)
            return False
        self.event("note", f"Shipped the code from ~/cua-run3/repo to {REMOTE_REPO}.")
        return True

    def start_run(self, conn, why: str) -> bool:
        r = conn.run("bash scripts/run3_pod_start.sh", timeout=self.cfg["start_timeout_s"])
        text = (r.out or "")[-2000:]
        if r.ok and ("STARTED" in text or "ALREADY_RUNNING" in text):
            word = "ALREADY_RUNNING" if "ALREADY_RUNNING" in text else "STARTED"
            if why == "bootstrap":
                self.event("note", f"run3_pod_start.sh printed {word}. The pipeline is running.")
            self.set_state("stopping" if self.st.get("stop_requested") else "running", why)
            return True
        msg = f"run3_pod_start.sh failed (rc {r.rc}): {(r.err or text)[-400:]}"
        if why == "bootstrap":
            self.note_failure("pod_start", None)
            if self.st["fail_count"] >= self.cfg["max_phase_failures"]:
                self.error(msg + " Giving up after repeated failures.")
                self.begin_teardown("phase_failed")
                return False
        self.error(msg)
        return False

    # --- running ---------------------------------------------------------------------

    def status_json(self, conn):
        r = conn.run("bash scripts/run3.sh --status-json", timeout=60)
        if r.ssh_failed:
            return None, f"ssh failed (rc {r.rc}): {r.err.strip()[-200:]}"
        try:
            text = r.out.strip()
            start = text.find("{")
            status = json.loads(text[start:])
        except ValueError:
            return False, f"--status-json printed no JSON (rc {r.rc}): {(r.out + r.err)[-300:]}"
        if not isinstance(status, dict) or "running" not in status or status.get("error"):
            # run3_status.py prints {"run": "run3", "error": ...} when it breaks. That
            # says nothing about the run, so it must not count as a dead run.
            return False, f"--status-json reported an error: {str(status)[:300]}"
        return status, None

    def on_running(self) -> None:
        self._health(stopping=False)

    def on_stopping(self) -> None:
        if not self.st["soft_stop_sent"]:
            self.soft_stop("Stop requested by the owner.")
        self._health(stopping=True)

    def on_restarting(self) -> None:
        self._health(stopping=False)

    def _health(self, stopping: bool) -> None:
        rec = self.pod_rec()
        if not rec:
            self.set_state("creating_pod")
            return
        health = self.refresh_pod(rec)
        if health == "gone":
            self.pod_lost(rec, f"API reports pod {rec['id']} as {rec['api_status']}")
            return
        conn = self.pod_conn(rec)
        status, err = (None, "no SSH endpoint yet") if conn is None else self.status_json(conn)
        now = self.clock()
        if status is None:
            if self.st["ssh_fail_since"] is None:
                self.st["ssh_fail_since"] = now
            down = now - self.st["ssh_fail_since"]
            self.log(f"SSH check failed for {int(down)} s: {err}")
            if down >= self.cfg["ssh_lost_after_s"] and health == "ok":
                self.pod_lost(rec, f"SSH failed for {int(down // 60)} min while the API says {rec['api_status']}")
            return
        self.st["ssh_fail_since"] = None
        if status is False:
            self.error(err, loud=False)
            return
        self.absorb_status(status)
        if status.get("all_done_line") or (status.get("done") and not status.get("running")):
            self.event("note", "The pipeline finished (ALL DONE).")
            self.begin_teardown("done")
            return
        if status.get("done"):
            return  # every phase is done; run3.sh is finishing up (wait for ALL DONE)
        if not status.get("running"):
            phase = status.get("failed_phase") or status.get("current_phase") or "unknown"
            self.note_failure(phase, status)
            if self.st["fail_count"] >= self.cfg["max_phase_failures"]:
                self.error(f"Phase {phase} failed {self.st['fail_count']} times in a row. Not retrying; tearing down.")
                self.begin_teardown("phase_failed")
                return
            self.st["restarts"] += 1
            self.set_state("restarting", f"run3 tmux session dead in phase {phase}")
            ok = self.start_run(conn, why="restart")
            self.event("restart", f"The run3 tmux session was dead in phase {phase} "
                                  f"(failure {self.st['fail_count']} in a row). "
                                  f"{'Restarted' if ok else 'Restart FAILED'} with run3_pod_start.sh (restart {self.st['restarts']}).")
            return
        if self.st["run_state"] == "restarting":
            self.set_state("stopping" if stopping else "running")

    def progress_key(self, status: dict | None):
        if not status:
            return None
        done = sum(1 for p in status.get("phases") or [] if p.get("status") == "done")
        return [done, (status.get("grpo") or {}).get("step")]

    def note_failure(self, phase: str, status: dict | None) -> None:
        key = self.progress_key(status)
        if phase == self.st.get("fail_phase") and key == self.st.get("fail_progress"):
            self.st["fail_count"] += 1
        else:
            self.st["fail_phase"], self.st["fail_count"] = phase, 1
        self.st["fail_progress"] = key

    def absorb_status(self, status: dict) -> None:
        now = self.clock()
        self.st["last_status"] = self.scrub.obj(status)
        self.st["last_status_ts"] = now
        seen = self.st.setdefault("seen_phases", {})
        for ph in status.get("phases") or []:
            name, s = ph.get("name"), ph.get("status")
            if not name or seen.get(name) == s:
                continue
            seen[name] = s
            if s == "running":
                self.event("phase_started", f"Phase {name} started.")
            elif s == "done":
                self.event("phase_done", f"Phase {name} done.")
            elif s == "failed":
                self.event("error", f"Phase {name} failed.")
        self.st["eta"] = self.estimate_eta(status)

    def estimate_eta(self, status: dict):
        g = status.get("grpo") or {}
        step, mx = g.get("step"), g.get("max_steps")
        phases = status.get("phases") or []
        grpo = next((p for p in phases if p.get("name") == "grpo"), None)
        if not (step and mx and grpo and grpo.get("started")):
            return None
        t0 = parse_iso(grpo["started"])
        if not t0:
            return None
        now = self.clock()
        if grpo.get("status") == "done":
            rate_left = 0
        else:
            rate_left = (now - t0) / max(1, step) * max(0, mx - step)
        names = [p.get("name") for p in phases]
        after = names[names.index("grpo") + 1:] if "grpo" in names else []
        tail = 3600 if any(n and "eval" in n for n in after) else 900
        return now_iso(now + rate_left + tail)

    # --- loss and recreation -------------------------------------------------------------

    def pod_lost(self, rec: dict, why: str) -> None:
        self.event("pod_lost", f"Lost pod {rec['id']} (generation {rec['generation']}): {why}.")
        rec["ended_ts"] = rec.get("ended_ts") or self.clock()
        self.st["ssh_fail_since"] = None
        self.set_state("recreating_pod", why)

    def delete_owned(self, rec: dict) -> bool:
        """Delete one pod this monitor owns, then confirm it's gone. True when gone."""
        if rec["id"] not in self.owned_ids():
            raise RuntimeError(f"refusing to delete pod {rec['id']}: not created or adopted by this monitor")
        try:
            code = self.api.delete_pod(rec["id"])
        except ApiError as e:
            self.error(f"DELETE pod {rec['id']} failed: {e}")
            return False
        try:
            raw = self.api.get_pod(rec["id"])
        except ApiError as e:
            self.error(f"Pod {rec['id']}: DELETE returned {code}, but the follow-up GET failed ({e}). Will check again.")
            return False
        gone = raw is None or (normalize_pod(raw).get("status") or "").upper() == "TERMINATED"
        if not gone:
            self.error(f"Pod {rec['id']}: DELETE returned {code}, but the pod still shows "
                       f"{normalize_pod(raw).get('status')}. Will retry.")
            return False
        rec["deleted_ts"] = self.clock()
        rec["ended_ts"] = rec.get("ended_ts") or rec["deleted_ts"]
        self.event("pod_deleted", f"Deleted pod {rec['id']} (generation {rec['generation']}); "
                                  f"the API confirms it's gone. Pod spend ${self.pod_spend(rec):.2f}.")
        return True

    def on_recreating_pod(self) -> None:
        rec = self.pod_rec()
        if rec and not rec.get("deleted_ts"):
            try:
                raw = self.api.get_pod(rec["id"])
            except ApiError as e:
                self.log(f"get_pod failed while recreating: {e}")
                return
            if raw is None:
                rec["deleted_ts"] = self.clock()
                self.log(f"pod {rec['id']} is already gone (404)")
            elif not self.delete_owned(rec):
                return
        self.st["current_pod"] = None
        if self.st["generation"] >= self.cfg["max_generations"]:
            self.error(f"Pod generation cap ({self.cfg['max_generations']}) reached after losing a pod. "
                       "Stopping; Hugging Face has the last synced checkpoint.")
            self.finish("failed", "generation cap reached")
            return
        if self.st.get("stop_requested"):
            self.finish("failed", "pod lost after a stop request; not recreating")
            return
        if not self.armed():
            self.error("ARMED is missing, so the monitor won't create a replacement pod.")
            self.finish("failed", "pod lost and monitor disarmed")
            return
        # A new pod starts with a clean failure streak.
        self.st["fail_phase"], self.st["fail_count"], self.st["fail_progress"] = None, 0, None
        self.set_state("creating_pod", "replacing the lost pod")
        self.on_creating_pod()

    # --- teardown ----------------------------------------------------------------------------

    def begin_teardown(self, reason: str) -> None:
        self.st["teardown_reason"] = reason
        self.st["final_outcome"] = "done" if reason == "done" else "failed"
        self.st["upload_attempts"] = 0
        self.st["upload_verified"] = False
        if not self.live_pod_exists():
            self.finish(self.st["final_outcome"], f"teardown ({reason}) with no pod alive")
            return
        self.set_state("uploading", reason)

    def on_uploading(self) -> None:
        rec = self.pod_rec()
        reason = self.st["teardown_reason"]
        conn = self.pod_conn(rec) if rec else None
        attempts = 1 if reason == "owner_kill" else self.cfg["upload_attempts"]
        while not self.st["upload_verified"] and self.st["upload_attempts"] < attempts and conn is not None:
            self.st["upload_attempts"] += 1
            n = self.st["upload_attempts"]
            self.log(f"upload attempt {n}/{attempts}")
            self.save()
            r = conn.run("{ [ -f /workspace/env.sh ] && . /workspace/env.sh; } ; "
                         "python scripts/upload_run3.py --final --verify", timeout=self.cfg["upload_timeout_s"])
            if r.ok:
                self.st["upload_verified"] = True
                self.event("upload_verified", f"upload_run3.py --final --verify succeeded (attempt {n}).")
            else:
                self.error(f"Upload attempt {n} failed (rc {r.rc}): {(r.err or r.out)[-400:]}")
        spend = self.spend()
        if self.st["upload_verified"]:
            self.set_state("deleting", "upload verified")
            self.on_deleting()
            return
        hard = reason in ("hard_budget", "wall_timeout")
        if reason == "owner_kill":
            self.event("error", "Owner ran --kill: deleting the pod although the upload is NOT verified.")
            self.set_state("deleting", "kill")
            self.on_deleting()
            return
        if spend >= self.cfg["hard_stop_usd"]:
            self.event("error", f"Upload NOT verified, but spend ${spend:.2f} >= ${self.cfg['hard_stop_usd']:.0f}. "
                                "Deleting the pod anyway to stop billing. Data not on Hugging Face is LOST.")
            self.st["final_outcome"] = "failed"
            self.set_state("deleting", "spend at hard limit without verified upload")
            self.on_deleting()
            return
        self.event("error", f"UPLOAD NOT VERIFIED ({'hard stop' if hard else reason}). The pod {rec['id'] if rec else '?'} "
                            f"is LEFT RUNNING at ${(rec or {}).get('cost_per_hr') or '?'}/hr so nothing is lost. "
                            "Owner action needed: fix and run run3_go.sh --kill, or delete it in the Runpod console. "
                            f"The monitor deletes it automatically if spend reaches ${self.cfg['hard_stop_usd']:.0f}.")
        self.finish("failed", "upload not verified; pod left running")

    def on_deleting(self) -> None:
        rec = self.pod_rec()
        if rec and not rec.get("deleted_ts"):
            if not self.delete_owned(rec):
                return
        self.finish(self.st.get("final_outcome") or "done", f"teardown complete ({self.st['teardown_reason']})")

    def finish(self, outcome: str, why: str) -> None:
        self.st["final_outcome"] = outcome
        self.set_state(outcome, why)
        self.event("note", f"Run 3 monitor finished: {outcome.upper()} ({why}). Total spend ${self.spend():.2f}.")
        try:
            self.paths.armed.unlink()
        except OSError:
            pass

    # --- terminal states ----------------------------------------------------------------------

    def on_done(self) -> None:
        pass

    def on_failed(self) -> None:
        # A pod can be left running on purpose (unverified upload). Keep its spend
        # current and enforce the hard limit on it.
        rec = self.pod_rec()
        if not rec or rec.get("deleted_ts"):
            return
        health = self.refresh_pod(rec)
        if health == "gone":
            rec["ended_ts"] = rec.get("ended_ts") or self.clock()
            rec["deleted_ts"] = rec["ended_ts"]
            self.event("pod_deleted", f"Pod {rec['id']} is gone (deleted outside the monitor).")
            return
        if self.spend() >= self.cfg["hard_stop_usd"]:
            self.event("error", f"Spend reached ${self.cfg['hard_stop_usd']:.0f} on the pod left running after a failed upload. Deleting it now.")
            self.delete_owned(rec)

    # --- loop -------------------------------------------------------------------------------

    def poll_interval(self) -> float:
        s = self.st["run_state"]
        if s in ("creating_pod", "bootstrapping", "recreating_pod", "restarting", "deleting"):
            return self.cfg["fast_poll_s"]
        if s == "idle":
            return 15
        if s in TERMINAL:
            return 300 if self.live_pod_exists() else 60
        return self.cfg["poll_s"]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def real_monitor(paths: Paths) -> Monitor:
    env = load_env(paths.env)
    key = env.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit(f"RUNPOD_API_KEY is missing from {paths.env}")
    cfg = load_config(paths)
    ssh_key = os.path.expanduser(cfg["ssh_key"])
    return Monitor(paths, RunpodAPI(key), lambda h, p: SSHPod(h, p, ssh_key), config=cfg, env=env)


def run_loop(paths: Paths) -> int:
    paths.home.mkdir(parents=True, exist_ok=True)
    lock = open(paths.lock, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another monitor process holds the lock. Exiting.", file=sys.stderr)
        return 1
    mon = real_monitor(paths)
    from collector import Collector, NoteWriter
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    col = Collector(mon)
    notes = NoteWriter(mon)
    threading.Thread(target=col.loop, args=(stop,), daemon=True, name="collector").start()
    threading.Thread(target=notes.loop, args=(stop,), daemon=True, name="notes").start()
    mon.log(f"monitor started in state {mon.st['run_state']} (pid {os.getpid()})")
    mon.save()
    while not stop.is_set():
        mon.tick()
        deadline = time.time() + mon.poll_interval()
        while time.time() < deadline and not stop.is_set():
            if mon.paths.command.exists():
                break
            stop.wait(3)
    mon.log("monitor stopping (signal); state is saved")
    return 0


def print_status(paths: Paths) -> int:
    st = read_json(paths.state, default=None)
    pub = read_json(paths.d("state.json"), default={}) or {}
    if not st:
        print(f"No monitor state yet ({paths.state}). Data state: {pub.get('run_state')} demo={pub.get('demo')}")
        return 0
    print(f"run_state:   {st['run_state']}   (armed: {paths.armed.exists()})")
    print(f"spend:       ${pub.get('spend_usd')} of ${pub.get('budget_usd')}; soft stop sent: {st.get('soft_stop_sent')}")
    print(f"generation:  {st.get('generation')}   restarts: {st.get('restarts')}   teardown: {st.get('teardown_reason')}")
    s = st.get("last_status") or {}
    print(f"phase:       {s.get('current_phase')}   grpo: {s.get('grpo')}")
    for p in st.get("pods", []):
        print(f"pod {p['id']}: gen {p['generation']} {p.get('gpu')} {p.get('cloud')} ${p.get('cost_per_hr')}/hr "
              f"spend ${p.get('spend_usd')} api={p.get('api_status')} ssh={p.get('ssh_host')}:{p.get('ssh_port')} "
              f"deleted={'yes' if p.get('deleted_ts') else 'no'}")
    if st.get("last_error"):
        print(f"last_error:  {st['last_error'][:300]}")
    return 0


def print_plan(paths: Paths) -> int:
    cfg = load_config(paths)
    env = load_env(paths.env)
    prices = RunpodAPI(env.get("RUNPOD_API_KEY", "")).gpu_prices() if env.get("RUNPOD_API_KEY") else {}
    print("Pod plan (tried in order until one has stock):")
    for gpu, cloud in cfg["gpu_candidates"]:
        g = prices.get(gpu) or {}
        price = g.get("securePrice" if cloud == "SECURE" else "communityPrice")
        hrs = cfg["hard_stop_usd"] / price if price else None
        print(f"  {gpu:24s} {cloud:9s} ${price if price is not None else '?':>5}/hr"
              + (f"   (${cfg['hard_stop_usd']:.0f} hard stop after ~{hrs:.1f} h)" if hrs else ""))
    print(f"Image {cfg['image']}, container disk {cfg['container_disk_gb']} GB, no network volume.")
    print(f"Budget ${cfg['budget_usd']:.0f}: soft stop (end GRPO early) at ${cfg['soft_stop_usd']:.0f}, "
          f"forced teardown at ${cfg['hard_stop_usd']:.0f} or after {cfg['wall_limit_h']:.0f} h.")
    print(f"Pod generations capped at {cfg['max_generations']}; a phase that fails {cfg['max_phase_failures']} times in a row ends the run.")
    return 0


def queue_command(paths: Paths, cmd: str) -> int:
    atomic_write_json(paths.command, {"cmd": cmd, "t": now_iso()})
    print(f"Queued '{cmd}'. The monitor picks it up within a few seconds if it's running.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run-3 monitor")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "once", "status", "plan", "stop", "kill"])
    ap.add_argument("--dry-run", action="store_true", help="run the built-in fake-pod scenario and exit")
    ap.add_argument("--home", default=None, help="override ~/cua-run3")
    args = ap.parse_args(argv)
    if args.dry_run:
        from fakes import run_dry_scenario
        return run_dry_scenario(verbose=True)
    paths = Paths(args.home)
    if args.command == "run":
        return run_loop(paths)
    if args.command == "once":
        m = real_monitor(paths)
        m.tick()
        return 0
    if args.command == "status":
        return print_status(paths)
    if args.command == "plan":
        return print_plan(paths)
    return queue_command(paths, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
