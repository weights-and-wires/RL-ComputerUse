"""Fake Runpod API, fake pod, and fake clock for `monitor.py --dry-run` and the tests.

Nothing here touches the network or a real pod.
"""

from __future__ import annotations

import io
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

from common import Paths, atomic_write_json, read_json, read_jsonl
from podssh import Result
from runpod import ApiError

PHASES = ["baseline", "select", "pre_eval", "grpo", "post_eval", "report"]


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakePod:
    """Simulates the pod-side contract: run3_pod_start.sh, run3.sh --status-json, upload."""

    def __init__(self, pod_id: str):
        self.id = pod_id
        self.reachable = True
        self.tmux = False
        self.started = 0
        self.start_fail = False
        self.secrets = b""
        self.code_shipped = 0
        self.stop_grpo = False
        self.upload_ok = True
        self.uploads = 0
        self.phase_idx = 0
        self.grpo_step = 0
        self.done = False
        self.failed_phase = None
        self.metrics: list[dict] = []
        self.samples: list[str] = []
        self.evals: dict = {}
        self.log = "starting\n"

    # Scenario controls
    def advance(self, phase: str, step: int = 0) -> None:
        self.phase_idx = PHASES.index(phase)
        self.grpo_step = step
        for s in range(len(self.metrics) + 1, step + 1):
            self.metrics.append({"step": s, "reward/total": 0.3 + s / 1000, "success/overall": 0.4})
        self.failed_phase = None

    def crash(self) -> None:
        self.tmux = False
        self.failed_phase = PHASES[self.phase_idx]

    def finish(self) -> None:
        self.phase_idx = len(PHASES)
        self.done = True
        self.tmux = False
        self.log += "ALL DONE\n"

    def status(self) -> dict:
        phases = []
        for i, p in enumerate(PHASES):
            if i < self.phase_idx or self.done:
                s = "done"
            elif i == self.phase_idx:
                s = "failed" if self.failed_phase == p else ("running" if self.tmux else "pending")
            else:
                s = "pending"
            phases.append({"name": p, "status": s, "started": "2027-01-15T08:00:00Z" if s != "pending" else None,
                           "finished": None})
        return {
            "run": "run3", "done": self.done, "all_done_line": self.done, "running": self.tmux,
            "current_phase": None if self.done else PHASES[self.phase_idx], "failed_phase": self.failed_phase,
            "phases": phases,
            "grpo": {"step": self.grpo_step, "max_steps": 400, "last_checkpoint": self.grpo_step // 25 * 25,
                     "synced_checkpoint": self.grpo_step // 25 * 25},
            "elapsed_s": 100, "gpu": {"util": 97, "mem_used_mb": 61000, "mem_total_mb": 81559, "power_w": 610},
        }

    def probe(self, offset: int) -> dict:
        text = "".join(json.dumps(m) + "\n" for m in self.metrics)
        b = text.encode()
        start = offset if offset <= len(b) else 0
        return {
            "status": self.status(),
            "metrics": {"size": len(b), "start": start, "end": len(b), "text": b[start:].decode()},
            "log_tail": self.log,
            "evals": self.evals,
            "samples": self.samples,
            "gpu": {"util": 97, "mem_used_mb": 61000, "mem_total_mb": 81559, "power_w": 610},
            "wandb_run": {"entity": "e", "project": "p", "id": "abc", "url": "https://wandb.ai/e/p/runs/abc"},
        }


class FakeConn:
    def __init__(self, world: "FakeWorld", pod_id: str):
        self.world = world
        self.pod_id = pod_id

    @property
    def pod(self) -> FakePod | None:
        return self.world.pods.get(self.pod_id)

    def _down(self):
        return Result(255, "", "ssh: connect to host: Connection refused")

    def run(self, cmd, stdin=None, timeout=120, in_repo=True, binary_out=False) -> Result:
        pod = self.pod
        self.world.ssh_log.append((self.pod_id, cmd.split("\n")[0][:80]))
        if pod is None or not pod.reachable or self.world.api_pods.get(self.pod_id, {}).get("status") != "RUNNING":
            return self._down()
        if cmd == "true":
            return Result(0, "", "")
        if ".secrets" in cmd:
            pod.secrets = stdin or b""
            return Result(0, "", "")
        if "run3_pod_start.sh" in cmd:
            if pod.start_fail:
                return Result(1, "bootstrap failed\n", "pip error")
            if pod.tmux:
                return Result(0, "ALREADY_RUNNING\n", "")
            pod.tmux, pod.failed_phase = True, None
            pod.started += 1
            return Result(0, "restoring...\nSTARTED\n", "")
        if "run3.sh --status-json" in cmd:
            return Result(0, json.dumps(pod.status()) + "\n", "")
        if ".stop_grpo" in cmd:
            pod.stop_grpo = True
            return Result(0, "", "")
        if "upload_run3.py" in cmd:
            pod.uploads += 1
            return Result(0, "UPLOAD VERIFIED\n", "") if pod.upload_ok else Result(1, "", "missing_on_hub=3")
        if "python3 -" in cmd:
            off = int(cmd.split("--metrics-offset")[1].split()[0])
            return Result(0, json.dumps(pod.probe(off)) + "\n", "")
        return Result(127, "", f"fake pod: unknown command {cmd[:60]}")

    def pipe_from(self, local_argv, remote_cmd, timeout=900, in_repo=False) -> Result:
        pod = self.pod
        if pod is None or not pod.reachable:
            return self._down()
        pod.code_shipped += 1
        return Result(0, "", "")

    def pipe_to(self, remote_cmd, local_argv, timeout=600, in_repo=True) -> Result:
        pod = self.pod
        if pod is None or not pod.reachable:
            return self._down()
        target = Path(local_argv[local_argv.index("-C") + 1])
        for name in remote_cmd.split("-cf -", 1)[1].split():
            d = target / name
            d.mkdir(parents=True, exist_ok=True)
            step = int(name.split("-")[1])
            atomic_write_json(d / "episode.json", {"step": step, "task": "click-button", "seed": step, "reward": 1.0,
                                                   "success": True, "turns": [{"image": "turn-0.jpg", "response":
                                                   "token hf_ABCDEFGHIJKLMNOPQRSTUVWX leaked", "action": "click(coordinate=[1, 2])", "error": ""}]})
            (d / "turn-0.jpg").write_bytes(b"\xff\xd8fakejpeg")
        return Result(0, "", "")


class FakeWorld:
    """The fake Runpod account: pods, stock, and the SSH side of each pod."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.api_pods: dict[str, dict] = {}
        self.pods: dict[str, FakePod] = {}
        self.no_stock: set[tuple[str, str]] = set()
        self.creates = 0
        self.deletes: list[str] = []
        self.ssh_log: list = []
        self.ids = (f"fakepod{n:03d}" for n in itertools.count(1))
        self.api_down = False
        # Somebody else's pod in the same account. The monitor must never touch it.
        self.api_pods["foreignpod1"] = {"id": "foreignpod1", "name": "someone-else", "status": "RUNNING", "cost": 1.0}

    # --- API surface (same methods as runpod.RunpodAPI) ---------------------------
    def _check(self):
        if self.api_down:
            raise ApiError(0, "network error: fake outage")

    def get_pod(self, pod_id):
        self._check()
        p = self.api_pods.get(pod_id)
        if p is None or p.get("status") == "GONE":
            return None
        return json.loads(json.dumps(p))

    def list_pods(self):
        self._check()
        return [dict(p) for p in self.api_pods.values() if p.get("status") != "GONE"]

    def create_pod(self, body):
        self._check()
        gpu, cloud = body["gpu"]["id"], body["cloud"]
        if (gpu, cloud) in self.no_stock:
            raise ApiError(400, "this GPU and data center combination could not be placed")
        assert body["disk"] >= 100 and "PUBLIC_KEY" in body["env"] and "mounts" not in body
        pid = next(self.ids)
        self.creates += 1
        cost = {"SECURE": 4.39, "COMMUNITY": 2.69}[cloud] if "HBM3" in gpu else 2.89
        self.api_pods[pid] = {
            "id": pid, "name": body["name"], "status": "RUNNING", "cost": cost, "cloud": cloud,
            "gpu": {"id": gpu, "count": 1}, "env": dict(body["env"]),
            "ssh": {"direct": {"host": f"10.0.0.{len(self.api_pods)}", "port": 22000 + len(self.api_pods)}},
            "createdAt": "2027-01-15T08:00:00Z", "startedAt": "2027-01-15T08:00:30Z",
        }
        self.pods[pid] = FakePod(pid)
        return json.loads(json.dumps(self.api_pods[pid]))

    def delete_pod(self, pod_id):
        self._check()
        assert pod_id != "foreignpod1", "monitor tried to delete a pod it doesn't own"
        self.deletes.append(pod_id)
        if pod_id not in self.api_pods or self.api_pods[pod_id]["status"] == "GONE":
            return 404
        self.api_pods[pod_id]["status"] = "GONE"
        return 204

    def gpu_prices(self):
        return {}

    # --- scenario helpers -------------------------------------------------------------
    def conn(self, host, port):
        for pid, p in self.api_pods.items():
            d = (p.get("ssh") or {}).get("direct") or {}
            if d.get("host") == host and d.get("port") == port:
                return FakeConn(self, pid)
        return FakeConn(self, "none")

    def lose(self, pod_id, how="TERMINATED"):
        if how == "404":
            self.api_pods[pod_id]["status"] = "GONE"
        else:
            self.api_pods[pod_id]["status"] = how


def make_home(tmp: Path, armed: bool = True) -> Paths:
    paths = Paths(tmp)
    paths.data.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)
    (paths.repo / "scripts").mkdir(parents=True, exist_ok=True)
    (paths.repo / "scripts" / "run3_pod_start.sh").write_text("#!/bin/bash\n")
    paths.env.write_text("RUNPOD_API_KEY=rpa_FAKEFAKEFAKEFAKEFAKEFAKE\nHF_TOKEN=hf_FAKEFAKEFAKEFAKEFAKEFAKE\n"
                         "WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567\nHF_ADAPTER_REPO=someone/qwen-run3\n")
    paths.extra_keys.write_text("ssh-ed25519 AAAAfakeMacKey mac\n")
    if armed:
        atomic_write_json(paths.armed, {"t": "now"})
    return paths


def build(tmp: Path, armed: bool = True, config: dict | None = None):
    from common import load_env
    from monitor import DEFAULT_CONFIG, Monitor
    clock = FakeClock()
    world = FakeWorld(clock)
    paths = make_home(tmp, armed)
    cfg = dict(DEFAULT_CONFIG, ssh_key=str(tmp / "nokey"), **(config or {}))
    out = io.StringIO()
    mon = Monitor(paths, world, world.conn, clock=clock, config=cfg, env=load_env(paths.env), out=out,
                  local_tar_argv=["true"])
    return mon, world, clock, paths, out


def events(paths: Paths) -> list[str]:
    return [e["kind"] for e in read_jsonl(paths.d("events.jsonl"))]


# ---------------------------------------------------------------------------
# The dry-run scenarios (also used by the tests)
# ---------------------------------------------------------------------------


def scenario_full(tmp: Path, say=print):
    """create -> running -> crash -> restart -> pod lost -> recreate -> soft stop -> done -> upload -> delete."""
    mon, world, clock, paths, out = build(tmp)
    world.no_stock.add(("NVIDIA H100 80GB HBM3", "COMMUNITY"))  # forces the fallbacks
    world.no_stock.add(("NVIDIA H100 80GB HBM3", "SECURE"))

    def tick(label, dt=120):
        clock.advance(dt)
        mon.tick()
        say(f"  {label:<44s} -> state={mon.st['run_state']:<15s} spend=${mon.spend():6.2f} "
            f"gen={mon.st['generation']} restarts={mon.st['restarts']}")

    tick("tick 1: armed, idle")
    tick("tick 2: create pod (SXM out of stock on both clouds)")
    p1 = mon.st["current_pod"]
    tick("tick 3: bootstrap (ssh, secrets, code, start)")
    pod = world.pods[p1]
    pod.advance("grpo", 12)
    tick("tick 4: healthy in grpo")
    pod.crash()
    tick("tick 5: tmux died -> restart")
    pod.advance("grpo", 30)
    tick("tick 6: healthy again")
    world.lose(p1, "404")
    tick("tick 7: pod vanished (404) -> pod_lost")
    tick("tick 8: delete old, create generation 2")
    p2 = mon.st["current_pod"]
    pod2 = world.pods[p2]
    tick("tick 9: bootstrap generation 2 (resume)")
    pod2.advance("grpo", 120)
    tick("tick 10: healthy in grpo on generation 2")
    clock.advance(14.4 * 3600)  # long GRPO stretch; spend passes the $42 soft limit
    tick("tick 11: spend over soft limit -> .stop_grpo")
    tick("tick 12: still running, no second soft stop")
    pod2.advance("post_eval", 140)
    tick("tick 13: post-training evals")
    pod2.finish()
    tick("tick 14: ALL DONE -> begin teardown")
    tick("tick 15: upload verified -> delete -> done")
    return mon, world, paths, p1, p2


def scenario_hard_budget(tmp: Path, say=print):
    mon, world, clock, paths, out = build(tmp)
    clock.advance(1); mon.tick(); mon.tick(); mon.tick()
    pod = world.pods[mon.st["current_pod"]]
    pod.advance("grpo", 50)
    mon.tick()
    say(f"  running: state={mon.st['run_state']} spend=${mon.spend():.2f}")
    clock.advance(18 * 3600)  # 2.69/hr * 18 h = $48.4 (community SXM, the first choice)
    mon.tick()
    say(f"  hard stop tick: state={mon.st['run_state']} reason={mon.st['teardown_reason']}")
    mon.tick()
    say(f"  after 18 h: state={mon.st['run_state']} spend=${mon.spend():.2f} reason={mon.st['teardown_reason']} "
        f"uploads={pod.uploads} deleted={world.deletes}")
    return mon, world, paths, pod


def run_dry_scenario(verbose=True) -> int:
    say = print if verbose else (lambda *a: None)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        say("Scenario 1: full lifecycle against a fake pod")
        mon, world, paths, p1, p2 = scenario_full(Path(td) / "a", say)
        ev = events(paths)
        say("  events: " + ", ".join(ev))
        checks = {
            "two pods created": world.creates == 2,
            "fallback GPU recorded": mon.pod_rec(p1)["gpu"] == "NVIDIA H100 PCIe",
            "one restart": mon.st["restarts"] == 1,
            "pod_lost + pod_recreated": "pod_lost" in ev and "pod_recreated" in ev,
            "budget_soft_stop once": ev.count("budget_soft_stop") == 1 and world.pods[p2].stop_grpo,
            "upload verified": "upload_verified" in ev and world.pods[p2].uploads == 1,
            "pod 2 deleted and confirmed": p2 in world.deletes and world.api_pods[p2]["status"] == "GONE",
            "foreign pod untouched": "foreignpod1" not in world.deletes,
            "final state done": mon.st["run_state"] == "done",
            "ARMED removed": not paths.armed.exists(),
            "no secret in data/": not any(s in (paths.data / f).read_text() for f in ("events.jsonl", "state.json")
                                          for s in ("hf_FAKE", "rpa_FAKE", "0123456789abcdef0123")),
        }
        for k, v in checks.items():
            say(f"  [{'ok' if v else 'FAIL'}] {k}")
            ok &= v
        say("Scenario 2: hard budget stop")
        mon, world, paths, pod = scenario_hard_budget(Path(td) / "b", say)
        checks = {
            "hard stop teardown": mon.st["teardown_reason"] == "hard_budget",
            "uploaded then deleted": pod.uploads == 1 and world.deletes == [pod.id],
            "state failed (run cut short)": mon.st["run_state"] == "failed",
        }
        for k, v in checks.items():
            say(f"  [{'ok' if v else 'FAIL'}] {k}")
            ok &= v
    say("DRY RUN " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_dry_scenario())
