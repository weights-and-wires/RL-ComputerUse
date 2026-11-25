"""Tests for the run monitor. Plain standard library: `python3 -m unittest discover monitor/tests`."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from collector import Collector, NoteWriter  # noqa: E402
from common import (  # noqa: E402
    Scrubber, atomic_write_json, build_evals, parse_eval_dir, read_json, read_jsonl, wilson_ci,
)
from fakes import PHASES, build, events, scenario_full, scenario_hard_budget  # noqa: E402
from runpod import normalize_pod  # noqa: E402

H = 3600


def quiet(*_a):
    pass


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmp = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def running(self, **cfg):
        """A monitor with one pod in grpo, state running."""
        mon, world, clock, paths, out = build(self.tmp, config=cfg or None)
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        self.assertEqual(mon.st["run_state"], "running")
        pod = world.pods[mon.st["current_pod"]]
        pod.advance("grpo", 10)
        clock.advance(60)
        mon.tick()
        return mon, world, clock, paths, pod


class TestLifecycle(Base):
    def test_explicit_gpu_order_and_pod_env(self):
        order = [["NVIDIA A100-SXM4-80GB", "SECURE"], ["NVIDIA A100 80GB PCIe", "SECURE"]]
        mon, world, clock, paths, out = build(self.tmp, config={
            "gpu_order": order, "pod_env": {"GRPO_MAX_HOURS": "12", "BAD NAME": "x", "X": "a;rm -rf /"}})
        world.no_stock.add(("NVIDIA A100-SXM4-80GB", "SECURE"))
        for _ in range(2):  # idle -> creating_pod, then the create call
            clock.advance(60)
            mon.tick()
        rec = mon.pod_rec(mon.st["current_pod"])
        # SXM had no stock, so the second entry wins at once; no Community wait.
        self.assertEqual((rec["gpu"], rec["cloud"]), ("NVIDIA A100 80GB PCIe", "SECURE"))
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        starts = [c for _, c in world.ssh_log if "run4_pod_start.sh" in c]
        self.assertTrue(starts)
        self.assertIn("GRPO_MAX_HOURS=12 ", starts[0])
        self.assertIn("PRICE_PER_HR=", starts[0])
        self.assertNotIn("rm -rf", starts[0])
        self.assertNotIn("BAD NAME", starts[0])

    def test_full_scenario(self):
        mon, world, paths, p1, p2 = scenario_full(self.tmp, say=quiet)
        ev = events(paths)
        self.assertEqual(world.creates, 2)
        self.assertEqual(mon.pod_rec(p1)["gpu"], "NVIDIA H100 PCIe")
        self.assertEqual(mon.st["restarts"], 1)
        for k in ("pod_created", "restart", "pod_lost", "pod_recreated", "budget_soft_stop", "upload_verified", "pod_deleted"):
            self.assertIn(k, ev)
        self.assertEqual(ev.count("budget_soft_stop"), 1)
        self.assertTrue(world.pods[p2].stop_grpo)
        self.assertEqual(world.api_pods[p2]["status"], "GONE")
        self.assertNotIn("foreignpod1", world.deletes)
        self.assertEqual(mon.st["run_state"], "done")
        self.assertFalse(paths.armed.exists())
        # Secrets went over stdin to the pod and contain the right names.
        sec = world.pods[p2].secrets.decode()
        self.assertIn("export HF_TOKEN=", sec)
        self.assertIn("export HF_ADAPTER_REPO=", sec)
        # Spend persisted per generation.
        self.assertGreater(mon.pod_rec(p1)["spend_usd"], 0)
        pub = read_json(paths.d("state.json"))
        self.assertEqual(pub["pod_generations"], 2)
        self.assertGreaterEqual(pub["spend_usd"], 42)

    def test_hard_budget(self):
        mon, world, paths, pod = scenario_hard_budget(self.tmp, say=quiet)
        self.assertEqual(mon.st["teardown_reason"], "hard_budget")
        self.assertEqual(pod.uploads, 1)
        self.assertEqual(world.deletes, [pod.id])
        self.assertEqual(mon.st["run_state"], "failed")

    def test_wall_timeout(self):
        mon, world, clock, paths, pod = self.running(fallback_cost_per_hr=0.01)
        world.api_pods[pod.id]["cost"] = 0.5  # cheap, so only the clock limit trips
        mon.pod_rec()["cost_per_hr"] = 0.5
        clock.advance(20 * H)
        mon.tick()
        self.assertEqual(mon.st["teardown_reason"], "wall_timeout")
        mon.tick()
        self.assertEqual(world.deletes, [pod.id])

    def test_status_error_is_not_a_crash(self):
        mon, world, clock, paths, pod = self.running()
        orig = pod.status
        pod.status = lambda: {"run": "run4", "error": "ValueError: boom"}
        for _ in range(5):
            clock.advance(120)
            mon.tick()
        self.assertEqual(mon.st["restarts"], 0)
        self.assertEqual(mon.st["run_state"], "running")
        pod.status = orig

    def test_done_but_still_running_waits(self):
        mon, world, clock, paths, pod = self.running()
        pod.phase_idx, pod.done, pod.tmux = len(PHASES), True, True
        pod.log = "report\n"
        orig = pod.status
        pod.status = lambda: dict(orig(), all_done_line=False, running=True)
        clock.advance(120)
        mon.tick()
        self.assertEqual(mon.st["run_state"], "running")
        pod.status = orig
        pod.tmux = False
        clock.advance(120)
        mon.tick()
        self.assertEqual(mon.st["run_state"], "uploading")

    def test_not_armed_never_creates(self):
        mon, world, clock, paths, _ = build(self.tmp, armed=False)
        for _ in range(5):
            clock.advance(60)
            mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["run_state"], "idle")

    def test_same_phase_fails_three_times(self):
        mon, world, clock, paths, pod = self.running()
        for i in range(3):
            pod.crash()
            clock.advance(120)
            mon.tick()
        self.assertEqual(mon.st["teardown_reason"], "phase_failed")
        self.assertEqual(mon.st["restarts"], 2)
        clock.advance(120)
        mon.tick()
        self.assertEqual(world.deletes, [pod.id])
        self.assertEqual(mon.st["run_state"], "failed")

    def test_progress_resets_failure_streak(self):
        mon, world, clock, paths, pod = self.running()
        for step in (20, 30, 40, 50):
            pod.crash()
            clock.advance(120)
            mon.tick()
            pod.advance("grpo", step)
        self.assertEqual(mon.st["restarts"], 4)
        self.assertEqual(mon.st["run_state"], "running")

    def test_ssh_lost_15_min_recreates(self):
        mon, world, clock, paths, pod = self.running()
        pod.reachable = False
        for _ in range(8):  # first failure at t=0, then 14 minutes more
            clock.advance(120)
            mon.tick()
            self.assertEqual(mon.st["run_state"], "running")
        clock.advance(120)
        mon.tick()  # 16 minutes down
        self.assertEqual(mon.st["run_state"], "recreating_pod")
        clock.advance(20)
        mon.tick()
        self.assertIn(pod.id, world.deletes)
        self.assertEqual(mon.st["generation"], 2)
        self.assertEqual(world.creates, 2)

    def test_exited_pod_is_deleted_and_replaced(self):
        mon, world, clock, paths, pod = self.running()
        world.lose(pod.id, "EXITED")
        clock.advance(120)
        mon.tick()
        mon.tick()
        self.assertIn(pod.id, world.deletes)
        self.assertEqual(world.creates, 2)

    def test_generation_cap(self):
        mon, world, clock, paths, pod = self.running(max_generations=2)
        world.lose(pod.id, "404")
        for _ in range(4):
            clock.advance(60)
            mon.tick()
        second = world.pods[mon.st["current_pod"]]
        world.lose(second.id, "404")
        for _ in range(4):
            clock.advance(60)
            mon.tick()
        self.assertEqual(world.creates, 2)
        self.assertEqual(mon.st["run_state"], "failed")

    def test_upload_fails_twice_below_limit_leaves_pod(self):
        mon, world, clock, paths, pod = self.running()
        pod.upload_ok = False
        pod.finish()
        clock.advance(120)
        mon.tick()
        mon.tick()
        self.assertEqual(pod.uploads, 2)
        self.assertEqual(mon.st["run_state"], "failed")
        self.assertEqual(world.deletes, [])
        self.assertTrue(any("UPLOAD NOT VERIFIED" in e["msg"] for e in read_jsonl(paths.d("events.jsonl"))))
        # Later the spend reaches the hard limit: the monitor deletes it.
        clock.advance(18 * H)
        mon.tick()
        self.assertEqual(world.deletes, [pod.id])

    def test_upload_fails_at_hard_limit_deletes(self):
        mon, world, clock, paths, pod = self.running()
        pod.upload_ok = False
        clock.advance(18 * H)
        mon.tick()
        mon.tick()
        self.assertEqual(pod.uploads, 2)
        self.assertEqual(world.deletes, [pod.id])
        msgs = " ".join(e["msg"] for e in read_jsonl(paths.d("events.jsonl")))
        self.assertIn("Deleting the pod anyway", msgs)

    def test_kill(self):
        mon, world, clock, paths, pod = self.running()
        pod.upload_ok = False
        atomic_write_json(paths.command, {"cmd": "kill"})
        mon.tick()
        mon.tick()
        self.assertEqual(pod.uploads, 1)
        self.assertEqual(world.deletes, [pod.id])

    def test_graceful_stop(self):
        mon, world, clock, paths, pod = self.running()
        atomic_write_json(paths.command, {"cmd": "stop"})
        mon.tick()
        self.assertEqual(mon.st["run_state"], "stopping")
        self.assertTrue(pod.stop_grpo)
        pod.crash()  # a crash while stopping still restarts
        clock.advance(120)
        mon.tick()
        self.assertEqual(mon.st["run_state"], "stopping")
        pod.finish()
        clock.advance(120)
        mon.tick()
        mon.tick()
        self.assertEqual(mon.st["run_state"], "done")
        self.assertEqual(world.deletes, [pod.id])

    def test_monitor_restart_resumes(self):
        from monitor import Monitor
        mon, world, clock, paths, pod = self.running()
        mon2 = Monitor(paths, world, world.conn, clock=clock, config=mon.cfg, env=mon.env, out=mon.out,
                       local_tar_argv=["true"])
        self.assertEqual(mon2.st["run_state"], "running")
        self.assertEqual(mon2.st["current_pod"], pod.id)
        pod.finish()
        clock.advance(120)
        mon2.tick()
        mon2.tick()
        self.assertEqual(mon2.st["run_state"], "done")
        self.assertEqual(world.creates, 1)

    def test_interrupted_create_is_found_by_name(self):
        mon, world, clock, paths, _ = build(self.tmp)
        clock.advance(60)
        mon.tick()  # idle -> creating_pod
        # Simulate a crash after POST but before the monitor recorded the pod.
        body = mon.pod_body("NVIDIA H100 80GB HBM3", "SECURE", mon.pod_name(1), True)
        world.create_pod(body)
        mon.tick()
        self.assertEqual(world.creates, 1)
        self.assertEqual(mon.st["run_state"], "bootstrapping")

    def test_refuses_foreign_delete(self):
        mon, world, clock, paths, pod = self.running()
        with self.assertRaises(RuntimeError):
            mon.delete_owned({"id": "foreignpod1"})

    def test_adopt(self):
        mon, world, clock, paths, _ = build(self.tmp, armed=False)
        raw = world.create_pod(mon.pod_body("NVIDIA H100 PCIe", "COMMUNITY", "by-hand", True))
        world.creates = 0
        atomic_write_json(paths.armed, {"adopt": raw["id"]})
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["current_pod"], raw["id"])
        self.assertTrue(mon.pod_rec()["adopted"])
        self.assertEqual(mon.st["run_state"], "running")

    def test_start_failure_three_times(self):
        mon, world, clock, paths, _ = build(self.tmp)
        clock.advance(60); mon.tick(); mon.tick()
        world.pods[mon.st["current_pod"]].start_fail = True
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        self.assertEqual(mon.st["teardown_reason"], "phase_failed")

    def test_no_stock_anywhere_keeps_trying(self):
        mon, world, clock, paths, _ = build(self.tmp, config={"max_create_attempts": 3})
        for g in mon.cfg["gpu_types"]:
            for c in ("COMMUNITY", "SECURE"):
                world.no_stock.add((g, c))
        for _ in range(5):
            clock.advance(60)
            mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["run_state"], "failed")

    def test_community_fallback(self):
        mon, world, clock, paths, _ = build(self.tmp)
        for g in mon.cfg["gpu_types"]:
            world.no_stock.add((g, "SECURE"))
        clock.advance(60); mon.tick(); mon.tick()
        rec = mon.pod_rec()
        self.assertEqual((rec["gpu"], rec["cloud"]), ("NVIDIA H100 80GB HBM3", "COMMUNITY"))

    def test_demo_data_cleared_on_start(self):
        mon, world, clock, paths, _ = build(self.tmp)
        atomic_write_json(paths.d("state.json"), {"demo": True})
        (paths.d("metrics.jsonl")).write_text('{"step": 1}\n')
        atomic_write_json(paths.d(".demo"), {"files": ["metrics.jsonl", "state.json"]})
        clock.advance(60)
        mon.tick()
        self.assertFalse(paths.d(".demo").exists())
        self.assertFalse(paths.d("metrics.jsonl").exists())
        self.assertFalse(read_json(paths.d("state.json"))["demo"])


class TestGpuPolicy(Base):
    def test_community_retried_for_20_min_before_secure(self):
        mon, world, clock, paths, _ = build(self.tmp)
        types = mon.cfg["gpu_types"]
        self.assertEqual(types, ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"])
        self.assertEqual(mon.cfg["community_wait_s"], 1200)
        for g in types:
            world.no_stock.add((g, "COMMUNITY"))
        clock.advance(1)
        mon.tick()
        for _ in range(58):  # 58 ticks x 20 s: just inside 20 minutes
            mon.tick()
            clock.advance(20)
        self.assertEqual(mon.st["run_state"], "creating_pod")
        self.assertTrue(world.attempts)
        self.assertTrue(all(c == "COMMUNITY" for _, c in world.attempts), "no Secure try inside the window")
        self.assertEqual(world.attempts[:3], [(g, "COMMUNITY") for g in types])  # SXM, PCIe, NVL, then again
        clock.advance(120)
        mon.tick()
        rec = mon.pod_rec()
        self.assertEqual((rec["gpu"], rec["cloud"]), (types[0], "SECURE"))
        msgs = [e["msg"] for e in read_jsonl(paths.d("events.jsonl")) if e["kind"] == "gpu_fallback"]
        self.assertTrue(any("Secure Cloud" in m for m in msgs))
        self.assertTrue(any("PCIe" in m and "Community" in m for m in msgs))

    def test_secure_order_is_sxm_pcie_nvl(self):
        mon, world, clock, paths, _ = build(self.tmp)
        types = mon.cfg["gpu_types"]
        for g in types:
            world.no_stock.add((g, "COMMUNITY"))
        world.no_stock.update({(types[0], "SECURE"), (types[1], "SECURE")})
        clock.advance(1)
        mon.tick()
        mon.tick()
        clock.advance(1300)
        mon.tick()
        secure = [a for a in world.attempts if a[1] == "SECURE"]
        self.assertEqual(secure, [(g, "SECURE") for g in types])
        self.assertEqual((mon.pod_rec()["gpu"], mon.pod_rec()["cloud"]), (types[2], "SECURE"))

    def test_community_pcie_is_second_choice(self):
        mon, world, clock, paths, _ = build(self.tmp)
        world.no_stock.add((mon.cfg["gpu_types"][0], "COMMUNITY"))
        clock.advance(1)
        mon.tick()
        mon.tick()
        rec = mon.pod_rec()
        self.assertEqual((rec["gpu"], rec["cloud"]), ("NVIDIA H100 PCIe", "COMMUNITY"))
        self.assertEqual(world.attempts, [("NVIDIA H100 80GB HBM3", "COMMUNITY"), ("NVIDIA H100 PCIe", "COMMUNITY")])
        self.assertIn("gpu_fallback", events(paths))

    def test_community_wait_is_configurable(self):
        mon, world, clock, paths, _ = build(self.tmp, config={"community_wait_s": 60})
        for g in mon.cfg["gpu_types"]:
            world.no_stock.add((g, "COMMUNITY"))
        clock.advance(1)
        mon.tick()
        mon.tick()
        clock.advance(61)
        mon.tick()
        self.assertEqual(mon.pod_rec()["cloud"], "SECURE")


class TestBalance(Base):
    def test_refuses_below_remaining_budget_plus_margin(self):
        mon, world, clock, paths, _ = build(self.tmp)
        world.balance = 49.99  # needs 48 + 2
        clock.advance(1)
        mon.tick()
        mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(world.attempts, [])
        self.assertEqual(mon.st["run_state"], "failed")
        errs = [e["msg"] for e in read_jsonl(paths.d("events.jsonl")) if e["kind"] == "error"]
        self.assertTrue(any("49.99" in m and "50.00" in m for m in errs))
        self.assertFalse(paths.armed.exists())

    def test_creates_at_exactly_the_threshold(self):
        mon, world, clock, paths, _ = build(self.tmp)
        world.balance = 50.0
        clock.advance(1)
        mon.tick()
        mon.tick()
        self.assertEqual(world.creates, 1)

    def test_replacement_pod_needs_less_after_spend(self):
        mon, world, clock, paths, pod = self.running()
        clock.advance(10 * H)  # 10 h x $2.69 = $26.9 spent
        world.balance = 30.0   # needs 48 - 26.9 + 2 = 23.1: enough
        world.lose(pod.id, "404")
        for _ in range(3):
            clock.advance(20)
            mon.tick()
        self.assertEqual(world.creates, 2)
        self.assertNotEqual(mon.st["run_state"], "failed")

    def test_replacement_pod_refused_when_balance_is_short(self):
        mon, world, clock, paths, pod = self.running()
        clock.advance(10 * H)
        world.balance = 10.0
        world.lose(pod.id, "404")
        for _ in range(3):
            clock.advance(20)
            mon.tick()
        self.assertEqual(world.creates, 1)
        self.assertEqual(mon.st["run_state"], "failed")

    def test_unreadable_balance_retries_then_fails_without_a_pod(self):
        mon, world, clock, paths, _ = build(self.tmp)
        world.balance = None
        clock.advance(1)
        for _ in range(3):
            mon.tick()
        self.assertEqual(mon.st["run_state"], "creating_pod")
        for _ in range(3):
            mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["run_state"], "failed")

    def test_balance_recovers_after_a_blip(self):
        mon, world, clock, paths, _ = build(self.tmp)
        world.balance = None
        clock.advance(1)
        mon.tick()
        mon.tick()
        world.balance = 100.0
        mon.tick()
        self.assertEqual(world.creates, 1)

    def test_plan_prints_balance_without_the_key(self):
        import io, contextlib
        from monitor import print_plan
        mon, world, clock, paths, _ = build(self.tmp)
        buf = io.StringIO()
        import monitor as M
        orig = M.RunpodAPI
        class FakeApi:
            def __init__(self, key): pass
            def gpu_prices(self): return {"NVIDIA H100 80GB HBM3": {"communityPrice": 2.69, "securePrice": 4.39}}
            def client_balance(self): return 30.78
        M.RunpodAPI = FakeApi
        try:
            with contextlib.redirect_stdout(buf):
                print_plan(paths)
        finally:
            M.RunpodAPI = orig
        out = buf.getvalue()
        self.assertIn("$30.78", out)
        self.assertIn("TOO LOW", out)
        self.assertNotIn("rpa_", out)


class TestRunNames(Base):
    def test_files_are_checked_before_arming(self):
        mon, world, clock, paths, _ = build(self.tmp)
        (paths.repo / "scripts" / "upload_run4.py").unlink()
        clock.advance(1)
        mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["run_state"], "failed")
        self.assertFalse(paths.armed.exists())
        msgs = " ".join(e["msg"] for e in read_jsonl(paths.d("events.jsonl")))
        self.assertIn("upload_run4.py", msgs)

    def test_all_names_derive_from_the_run(self):
        from common import RunNames
        n = RunNames("run4")
        self.assertEqual((n.pod_start, n.run_sh, n.upload), ("scripts/run4_pod_start.sh", "scripts/run4.sh", "scripts/upload_run4.py"))
        self.assertEqual((n.tmux, n.tmux_sync, n.unit), ("run4", "run4-sync", "cua-run4-monitor"))
        self.assertEqual(n.metrics, "results/run4/train/grpo-4b-run4/metrics.jsonl")
        self.assertEqual(n.stop_flag, "results/run4/.stop_grpo")
        self.assertEqual(n.label, "Run 4")
        self.assertEqual(n.pod_name(1700000000, 2), "cua-run4-1700000000-g2")
        self.assertEqual(n.hf_ckpt, "checkpoints/grpo-4b-run4")

    def test_default_run_is_run4_and_home_follows(self):
        import os
        from common import Paths
        saved = {k: os.environ.pop(k, None) for k in ("CUA_RUN", "CUA_RUN_HOME")}
        try:
            p = Paths()
            self.assertEqual(p.run, "run4")
            self.assertEqual(p.home.resolve(), (Path.home() / "cua-run4").resolve())
            self.assertEqual(Paths(run="run3").home.resolve(), (Path.home() / "cua-run3").resolve())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_install_folder_sets_the_run(self):
        from common import infer_run
        self.assertEqual(infer_run(Path("/home/u/cua-run3/monitor")), ("run3", Path("/home/u/cua-run3")))
        self.assertEqual(infer_run(Path("/Users/u/repo/monitor")), ("run4", None))

    def test_pod_commands_use_the_run(self):
        mon, world, clock, paths, _ = build(self.tmp)
        clock.advance(1)
        for _ in range(4):
            mon.tick()
        cmds = " ".join(c for _, c in world.ssh_log)
        self.assertIn("scripts/run4_pod_start.sh", cmds)
        self.assertIn("scripts/run4.sh --status-json", cmds)
        self.assertNotIn("run3", cmds)

    def test_run3_still_works_with_its_names(self):
        mon, world, clock, paths, _ = build(self.tmp, run="run3")
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        self.assertEqual(mon.st["run_state"], "running")
        self.assertEqual(read_json(paths.d("state.json"))["run"], "run3")
        self.assertIn(mon.n.pod_name(mon.st["started_ts"], 1), [p["name"] for p in world.api_pods.values()])


class TestCollector(Base):
    def test_collect(self):
        mon, world, clock, paths, pod = self.running()
        pod.advance("grpo", 5)
        pod.samples = [f"step-{i:06d}" for i in range(1, 41)]
        ev = self.tmp / "evsrc"
        pod.evals = {"pre-pool": {"per_task": "task,n,success_rate\nclick-button,100,0.5\nenter-text,100,0.7\n",
                                  "summary": {"mode": "greedy", "track": "B"}}}
        pod.log = "a\rb\r\nkey hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ end\n"
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        self.assertTrue(col.collect_once())
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual([r["step"] for r in rows], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pod.advance("grpo", 12)
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual(rows[-1]["step"], 12)
        self.assertEqual(len(rows), 12)
        # Samples: newest 30 kept, scrubbed, index newest first.
        idx = read_json(paths.d("samples/index.json"))
        self.assertEqual(len(idx), 30)
        self.assertEqual(idx[0]["dir"], "step-000040")
        ep = (paths.d("samples/step-000040/episode.json")).read_text()
        self.assertNotIn("hf_ABCDEF", ep)
        self.assertIn("[redacted]", ep)
        # Evals.
        evals = read_json(paths.d("evals.json"))
        self.assertAlmostEqual(evals["pre-pool"]["mean"], 0.6)
        self.assertEqual(evals["pre-pool"]["track"], "B")
        # Log scrubbed, carriage returns split.
        log = paths.d("log.txt").read_text()
        self.assertIn("a\nb\n", log)
        self.assertNotIn("hf_ABC", log)
        # System line with spend.
        sysrows = read_jsonl(paths.d("system.jsonl"))
        self.assertEqual(sysrows[-1]["gpu_util"], 97)
        self.assertIn("spend_usd", sysrows[-1])
        self.assertEqual(mon.st["wandb_url"], "https://wandb.ai/e/p/runs/abc")

    def test_restored_pod_dedupes(self):
        mon, world, clock, paths, pod = self.running()
        pod.advance("grpo", 8)
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        col.collect_once()
        # Replace the pod with one whose restored metrics stop at step 5, then grow to 9.
        world.lose(pod.id, "404")
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        pod2 = world.pods[mon.st["current_pod"]]
        pod2.advance("grpo", 13)
        col.collect_once()
        steps = [r["step"] for r in read_jsonl(paths.d("metrics.jsonl"))]
        self.assertEqual(steps, list(range(1, 14)))

    def test_resume_replay_replaces_rows(self):
        # GRPO resumed from checkpoint 5 on the same pod: the pod trimmed steps 6-8
        # and replayed them with new values. The local copy must take the new ones.
        mon, world, clock, paths, pod = self.running()
        pod.advance("grpo", 8)
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        col.collect_once()
        pod.metrics = pod.metrics[:5]
        col.collect_once()  # the trimmed file is shorter, so the probe rereads it
        pod.advance("grpo", 9)
        for m in pod.metrics[5:]:
            m["reward/total"] = 0.9
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual([r["step"] for r in rows], list(range(1, 10)))
        self.assertEqual([r["reward/total"] for r in rows[5:]], [0.9] * 4)

    # --- the run-3 false rewind ----------------------------------------------------

    @staticmethod
    def train_row(step, **extra):
        return dict({"step": step, "reward/total": 0.3 + step / 1000, "success/overall": 0.4}, **extra)

    def test_eval_row_sharing_a_step_is_not_a_resume(self):
        # Run 3's pattern: train rows 1..30 and a mid-training eval row for step 25
        # appended right after train row 25, copied while the run is live.
        mon, world, clock, paths, pod = self.running()
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        pod.metrics = [self.train_row(s) for s in range(1, 25)]
        col.collect_once()
        pod.metrics.append(self.train_row(25))
        col.collect_once()  # train row 25 is copied; the eval row doesn't exist yet
        pod.metrics.append({"step": 25, "eval/success": 0.7})
        col.collect_once()  # the eval row arrives with the same step as a copied row
        pod.metrics += [self.train_row(s) for s in range(26, 31)]
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual(len(rows), 31)
        self.assertEqual(rows, pod.metrics)
        self.assertEqual(sum(1 for r in rows if r["step"] == 25), 2)
        self.assertEqual([r["step"] for r in rows if "reward/total" in r], list(range(1, 31)))
        self.assertNotIn("GRPO resumed", " ".join(e["msg"] for e in read_jsonl(paths.d("events.jsonl"))))

    def test_eval_row_with_kind_field_in_one_read(self):
        mon, world, clock, paths, pod = self.running()
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        pod.metrics = [self.train_row(s, kind="train") for s in range(1, 26)]
        pod.metrics.append({"step": 25, "kind": "eval", "eval/success": 0.7, "eval/n": 40})
        pod.metrics += [self.train_row(s, kind="train") for s in range(26, 31)]
        col.collect_once()
        pod.metrics.append({"step": 30, "kind": "eval", "eval/success": 0.72})
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual(rows, pod.metrics)
        self.assertEqual([r["step"] for r in rows if r.get("kind") == "eval"], [25, 30])
        self.assertNotIn("GRPO resumed", " ".join(e["msg"] for e in read_jsonl(paths.d("events.jsonl"))))

    def test_resume_with_eval_rows_replaces_from_the_checkpoint(self):
        # GRPO resumes from checkpoint 25 on the same pod: the pod trims every row
        # after the checkpoint (train and eval) and replays them.
        mon, world, clock, paths, pod = self.running()
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        pod.metrics = [self.train_row(s) for s in range(1, 26)] + [{"step": 25, "eval/success": 0.7}]
        pod.metrics += [self.train_row(s) for s in range(26, 41)]
        col.collect_once()
        pod.metrics = pod.metrics[:26]  # rows through the step-25 eval row survive
        col.collect_once()
        pod.metrics += [self.train_row(s, **{"reward/total": 0.9}) for s in range(26, 36)]
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows, pod.metrics)
        evs = [e["msg"] for e in read_jsonl(paths.d("events.jsonl")) if "GRPO resumed" in e["msg"]]
        self.assertEqual(len(evs), 1)
        self.assertIn("from step 26", evs[0])

    def test_unchanged_new_pod_adds_no_rewind(self):
        mon, world, clock, paths, pod = self.running()
        col = Collector(mon, clock=clock, wandb_every_s=10**9)
        pod.metrics = [self.train_row(s) for s in range(1, 11)] + [{"step": 10, "eval/success": 0.5}]
        col.collect_once()
        world.lose(pod.id, "404")
        for _ in range(3):
            clock.advance(60)
            mon.tick()
        pod2 = world.pods[mon.st["current_pod"]]
        pod2.metrics = list(pod.metrics) + [self.train_row(11)]
        col.collect_once()
        rows = read_jsonl(paths.d("metrics.jsonl"))
        self.assertEqual(rows, pod2.metrics)
        self.assertNotIn("GRPO resumed", " ".join(e["msg"] for e in read_jsonl(paths.d("events.jsonl"))))

    def test_collector_failure_is_harmless(self):
        mon, world, clock, paths, pod = self.running()
        pod.reachable = False
        col = Collector(mon, clock=clock)
        self.assertFalse(col.collect_once())

    def test_notes(self):
        mon, world, clock, paths, pod = self.running()
        nw = NoteWriter(mon, clock=clock, runner=lambda text: "[claude-code:unrecognized_model] x\nGRPO is at step 10. Healthy. Spend is low.")
        self.assertTrue(nw.write_once())
        notes = read_json(paths.d("notes.json"))
        self.assertEqual(notes[-1]["text"], "GRPO is at step 10. Healthy. Spend is low.")
        bad = NoteWriter(mon, clock=clock, runner=lambda text: None)
        self.assertFalse(bad.write_once())


class TestCommon(unittest.TestCase):
    def test_scrub(self):
        s = Scrubber({"HF_TOKEN": "hf_secretvalue123", "WANDB_API_KEY": "wandb_v1_abcdefghijklmnop", "OTHER": "x"})
        text = ("hf_secretvalue123 wandb_v1_abcdefghijklmnop hf_ABCDEFGHIJKLMNOPQRSTUV rpa_ABCDEFGHIJKLMNOPQRSTUV "
                + "a" * 40 + " ok")
        out = s(text)
        self.assertEqual(out, " ".join(["[redacted]"] * 5) + " ok")
        self.assertEqual(s.obj({"k": ["hf_secretvalue123"]}), {"k": ["[redacted]"]})

    def test_wilson(self):
        lo, hi = wilson_ci(62, 100)
        self.assertAlmostEqual(lo, 0.5221, places=3)
        self.assertAlmostEqual(hi, 0.7093, places=3)
        self.assertIsNone(wilson_ci(0, 0))

    def test_parse_eval_summary_only(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "e1"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({"mode": "greedy", "track": "B", "overall_success": 0.5,
                                                        "per_task": {"a": {"n": 10, "success_rate": 0.5}}}))
            e = parse_eval_dir(d)
            self.assertEqual(e["tasks"]["a"], {"success": 0.5, "n": 10})
            self.assertEqual(e["mean"], 0.5)
            self.assertEqual(list(build_evals(Path(td))), ["e1"])

    def test_normalize_v2_and_v1(self):
        v2 = {"id": "p", "status": "RUNNING", "cost": 2.69, "gpu": {"id": "NVIDIA H100 80GB HBM3"}, "cloud": "COMMUNITY",
              "ssh": {"direct": {"host": "1.2.3.4", "port": 4567}}, "runtime": {"uptime": 12}}
        n = normalize_pod(v2)
        self.assertEqual((n["ssh_host"], n["ssh_port"], n["cost_per_hr"], n["uptime_s"]), ("1.2.3.4", 4567, 2.69, 12))
        v2b = {"id": "p", "status": "RUNNING", "ssh": {"direct": None},
               "runtime": {"ports": [{"private": 22, "public": 3333, "type": "tcp", "ip": "5.6.7.8"}]}}
        self.assertEqual(normalize_pod(v2b)["ssh_port"], 3333)
        v1 = {"id": "p", "desiredStatus": "RUNNING", "publicIp": "9.9.9.9", "portMappings": {"22": 1234}, "costPerHr": 2.19}
        n = normalize_pod(v1)
        self.assertEqual((n["status"], n["ssh_host"], n["ssh_port"], n["cost_per_hr"]), ("RUNNING", "9.9.9.9", 1234, 2.19))


class TestRealPieces(Base):
    def test_pod_probe_script(self):
        """Run the real pod_probe.py against a fake repo laid out like the pod."""
        import subprocess
        repo = self.tmp / "repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "scripts" / "run4.sh").write_text('echo \'{"run": "run4", "done": false, "running": true}\'\n')
        r3 = repo / "results/run4"
        (r3 / "train/grpo-4b-run4").mkdir(parents=True)
        (r3 / "train/grpo-4b-run4/metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step": 3, "partial')
        (r3 / "run.log").write_text("x\r\ny\rz\n")
        (r3 / "eval/pre").mkdir(parents=True)
        (r3 / "eval/pre/per_task.csv").write_text("task,n,success_rate\na,10,0.5\n")
        (r3 / "samples/step-000010").mkdir(parents=True)
        (r3 / "samples/step-000010/episode.json").write_text("{}")
        (r3 / "wandb_run.json").write_text('{"entity": "e", "project": "p", "id": "i", "url": "u"}')
        src = (HERE.parent / "pod_probe.py").read_bytes()
        p = subprocess.run([sys.executable, "-", "--metrics-offset", "0", "--run", "run4"], input=src, cwd=repo,
                           capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = json.loads(p.stdout)
        self.assertEqual(out["status"]["run"], "run4")
        self.assertFalse(out["metrics"]["reset"])
        self.assertEqual(out["metrics"]["text"], '{"step": 1}\n{"step": 2}\n')  # whole lines only
        self.assertEqual(out["log_tail"], "x\ny\nz")
        self.assertIn("per_task", out["evals"]["pre"])
        self.assertEqual(out["samples"], ["step-000010"])
        self.assertEqual(out["wandb_run"]["id"], "i")

    def test_code_tar_excludes_secrets(self):
        import subprocess
        mon, world, clock, paths, _ = build(self.tmp)
        mon.local_tar_argv = None
        (paths.repo / ".env").write_text("HF_TOKEN=hf_x\n")
        (paths.repo / "results/run2").mkdir(parents=True)
        (paths.repo / "results/run2/big.bin").write_text("x")
        (paths.repo / "cua").mkdir()
        (paths.repo / "cua/a.py").write_text("print(1)\n")
        p = subprocess.run(mon.tar_argv(), capture_output=True)
        self.assertEqual(p.returncode, 0)
        names = subprocess.run(["tar", "-tzf", "-"], input=p.stdout, capture_output=True).stdout.decode()
        self.assertIn("cua/a.py", names)
        self.assertIn("scripts/run4_pod_start.sh", names)
        self.assertNotIn(".env", names)
        self.assertNotIn("results", names)


if __name__ == "__main__":
    unittest.main()
