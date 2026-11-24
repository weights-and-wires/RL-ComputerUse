"""Tests for the run-3 monitor. Plain standard library: `python3 -m unittest discover monitor/tests`."""

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
        pod.status = lambda: {"run": "run3", "error": "ValueError: boom"}
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
        for g, c in mon.cfg["gpu_candidates"]:
            world.no_stock.add((g, c))
        for _ in range(5):
            clock.advance(60)
            mon.tick()
        self.assertEqual(world.creates, 0)
        self.assertEqual(mon.st["run_state"], "failed")

    def test_community_fallback(self):
        mon, world, clock, paths, _ = build(self.tmp)
        for g, c in mon.cfg["gpu_candidates"]:
            if c == "SECURE":
                world.no_stock.add((g, c))
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
        (repo / "scripts" / "run3.sh").write_text('echo \'{"run": "run3", "done": false, "running": true}\'\n')
        r3 = repo / "results/run3"
        (r3 / "train/grpo-4b-run3").mkdir(parents=True)
        (r3 / "train/grpo-4b-run3/metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step": 3, "partial')
        (r3 / "run.log").write_text("x\r\ny\rz\n")
        (r3 / "eval/pre").mkdir(parents=True)
        (r3 / "eval/pre/per_task.csv").write_text("task,n,success_rate\na,10,0.5\n")
        (r3 / "samples/step-000010").mkdir(parents=True)
        (r3 / "samples/step-000010/episode.json").write_text("{}")
        (r3 / "wandb_run.json").write_text('{"entity": "e", "project": "p", "id": "i", "url": "u"}')
        src = (HERE.parent / "pod_probe.py").read_bytes()
        p = subprocess.run([sys.executable, "-", "--metrics-offset", "0"], input=src, cwd=repo, capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = json.loads(p.stdout)
        self.assertEqual(out["status"]["run"], "run3")
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
        self.assertIn("scripts/run3_pod_start.sh", names)
        self.assertNotIn(".env", names)
        self.assertNotIn("results", names)


if __name__ == "__main__":
    unittest.main()
