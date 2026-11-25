"""Tests for run 4 (docs/PLAN_RUN4.md): new actions, the history window, turn budgets,
the ratio cap and sampler checks, the run-name-aware status, pool, and sync code,
the metrics `kind` field, and the pipeline glue. No GPU, BrowserGym, or vLLM needed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from cua import actions, pipeline, prompts, report, rollout, run_pool, run_status, smoke4
from cua.actions import ActionError, parse_response, render_action
from cua.config import load_config, load_tasks, resolve_task_list, turn_limit
from cua.types import EpisodeSpec, Observation, SamplingSpec, StepResult

REPO = Path(__file__).resolve().parent.parent
SIZE = (500, 300)


# ---------------------------------------------------------------------------
# Actions: double_click and left_click_drag
# ---------------------------------------------------------------------------


def call(name: str, **args) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def test_double_click_and_drag_render_browsergym_actions():
    assert render_action("double_click", {"coordinate": [500, 500]}, "B", SIZE, extended=True) == "mouse_dblclick(250, 150)"
    drag = {"start_coordinate": [100, 500], "end_coordinate": [900, 1000]}
    assert render_action("left_click_drag", drag, "B", SIZE, extended=True) == "mouse_drag_and_drop(50, 150, 450, 299)"


@pytest.mark.parametrize("args", [
    {"coordinate": [100, 500], "coordinate2": [900, 1000]},       # Qwen mobile swipe shape
    {"start_coordinate": [100, 500], "coordinate": [900, 1000]},  # Qwen computer-use end point
])
def test_drag_accepts_qwen_argument_shapes(args):
    assert render_action("left_click_drag", args, "B", SIZE, extended=True) == "mouse_drag_and_drop(50, 150, 450, 299)"


def test_drag_without_a_start_point_is_a_recoverable_error():
    parsed = parse_response(call("left_click_drag", coordinate=[10, 10]), "B", SIZE, extended=True)
    assert parsed.action is None and not parsed.ends_episode
    assert "coordinate" in parsed.format_error and "start_coordinate" in parsed.format_error


@pytest.mark.parametrize("bad", [[1], "1,2", None, [1, 2, 3]])
def test_drag_rejects_bad_points(bad):
    with pytest.raises(ActionError):
        render_action("left_click_drag", {"start_coordinate": bad, "end_coordinate": [1, 2]}, "B", SIZE, extended=True)


def test_new_tools_exist_only_when_extended_and_only_on_track_b():
    assert parse_response(call("double_click", coordinate=[5, 5]), "B", SIZE).format_error.startswith("Unknown tool")
    assert parse_response(call("double_click", coordinate=[5, 5]), "B", SIZE, extended=True).action == "mouse_dblclick(2, 2)"
    assert [f.__name__ for f in actions.tools_for("B")] == ["mouse_click", "keyboard_type", "keyboard_press", "scroll", "noop"]
    assert [f.__name__ for f in actions.tools_for("B", True)][-2:] == ["double_click", "left_click_drag"]
    assert actions.tools_for("A", True) == actions.tools_for("A")
    # Run 3's click is unchanged.
    assert parse_response(call("mouse_click", coordinate=[500, 500]), "B", SIZE, extended=True).action == "mouse_click(250, 150)"


def test_extended_tool_schemas_require_every_argument():
    pytest.importorskip("transformers")
    schemas = {s["function"]["name"]: s["function"] for s in actions.tool_schemas("B", extended=True)}
    drag = schemas["left_click_drag"]["parameters"]
    assert set(drag["required"]) == {"start_coordinate", "end_coordinate"}
    assert drag["properties"]["start_coordinate"]["type"] == "array"
    assert schemas["double_click"]["parameters"]["required"] == ["coordinate"]


def test_system_prompt_mentions_new_tools_only_when_extended():
    assert "left_click_drag" in prompts.system_prompt("B", extended=True)
    assert "double_click" not in prompts.system_prompt("B")
    assert prompts.system_prompt("A", extended=True) == prompts.system_prompt("A")


# ---------------------------------------------------------------------------
# Rollout: history window and turn budgets
# ---------------------------------------------------------------------------


class FakeEnv:
    """A browser that never ends an episode; it tags each screenshot with the step."""

    reset_frame = None

    def __init__(self, succeed_at: int | None = None):
        self.succeed_at = succeed_at

    def _obs(self, n: int, error: str = "") -> Observation:
        return Observation(goal="Click OK.", screenshot=Image.new("RGB", (64, 40), (n, 0, 0)), axtree="",
                           last_action_error=error, step=n, screen_size=(640, 400))

    def reset(self, task, seed, record=False):
        self.n = 0
        return self._obs(0)

    def step(self, action):
        self.n += 1
        ok = self.succeed_at is not None and self.n >= self.succeed_at
        return StepResult(self._obs(self.n), ok, 1.0 if ok else 0.0, ok)


class FakePool:
    def __init__(self, n=2, **kw):
        self.envs = [FakeEnv(**kw) for _ in range(n)]


def windowed_cfg(window=4, max_turns=10, budgets=None):
    return load_config(REPO / "configs/base.yaml", [
        "env.track=B", "env.track_b_extra_tools=true", f"env.max_turns={max_turns}",
        f"env.context_window_turns={window}",
    ] + ([f"env.turn_budgets={json.dumps(budgets)}"] if budgets else []))


def scripted(texts):
    policy = smoke4.ScriptedPolicy(texts)
    return policy


def test_window_splits_an_episode_into_segments_with_a_history_log():
    cfg = windowed_cfg(window=4, max_turns=10)
    policy = scripted([call("noop")] * 3 + [call("double_click", coordinate=[1, 2])] * 7)
    [ep] = rollout.run_episodes([EpisodeSpec("t", 0)], policy.generate, FakePool(1), cfg, SamplingSpec(), progress=False)
    assert len(ep.turns) == 10 and ep.truncated
    assert [t.segment for t in ep.turns] == [0] * 4 + [1] * 4 + [2] * 2
    convs = ep.conversations()
    assert len(convs) == 3 and convs[-1] is ep.messages and convs[0] is ep.segments[0]
    # Closed segments end with the assistant reply (the next segment shows that screenshot).
    assert all(c[-1]["role"] == "assistant" for c in convs[:2])
    log = convs[2][1]["content"][1]["text"].strip().splitlines()
    assert log[0] == prompts.HISTORY_HEADER and len(log) == 1 + 8
    assert log[1] == "1. noop()" and log[4] == "4. double_click(coordinate=[1, 2])"
    # Every turn's prompt_len points at its reply in its own segment.
    for t in ep.turns:
        conv = convs[t.segment]
        assert conv[t.prompt_len]["role"] == "assistant" and conv[t.prompt_len]["content"] == t.generation.text
    # At most `window` screenshots in any prompt.
    from cua.policy import collect_images
    for t in ep.turns:
        assert len(collect_images(convs[t.segment][: t.prompt_len])) <= 4


def test_without_a_window_the_episode_is_one_conversation():
    cfg = windowed_cfg(window=0, max_turns=6)
    [ep] = rollout.run_episodes([EpisodeSpec("t", 0)], scripted([call("noop")]).generate, FakePool(1), cfg,
                                SamplingSpec(), progress=False)
    assert ep.segments == [] and ep.conversations() == [ep.messages] and len(ep.turns) == 6


def test_turn_budgets_cap_each_task_and_set_the_efficiency_bonus():
    cfg = windowed_cfg(window=4, max_turns=20, budgets={"short": 10})
    assert turn_limit(cfg, "short") == 10 and turn_limit(cfg, "other") == 20
    specs = [EpisodeSpec("short", 0), EpisodeSpec("other", 0)]
    eps = rollout.run_episodes(specs, scripted([call("noop")]).generate, FakePool(2), cfg, SamplingSpec(), progress=False)
    assert [len(e.turns) for e in eps] == [10, 20]
    # A success on turn 5 of a 10-turn budget earns 0.1 * (1 - 5/10).
    pool = FakePool(1, succeed_at=5)
    [ep] = rollout.run_episodes([EpisodeSpec("short", 0)], scripted([call("noop")]).generate, pool, cfg,
                                SamplingSpec(), progress=False)
    assert ep.success and ep.reward.efficiency_bonus == pytest.approx(0.05)


def test_turn_budgets_file_must_exist(tmp_path):
    cfg = windowed_cfg()
    cfg.env["turn_budgets"] = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        turn_limit(cfg, "x")
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"budgets": {"x": 12}}))
    cfg.env["turn_budgets"] = str(path)
    assert turn_limit(cfg, "x") == 10  # capped at env.max_turns


def _chat_template_processor():
    """A tokenizer-only stand-in for the Qwen3-VL processor (renders text, no images)."""
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        tmpl = json.loads(Path(hf_hub_download("Qwen/Qwen3-VL-4B-Instruct", "chat_template.json")).read_text())
    except Exception as exc:  # noqa: BLE001 - not cached, no network
        pytest.skip(f"Qwen3-VL tokenizer not available offline: {type(exc).__name__}")

    class Proc:
        tokenizer = tok

        def apply_chat_template(self, messages, **kw):
            return tok.apply_chat_template(messages, chat_template=tmpl["chat_template"], **kw)

    return Proc()


def test_windowed_prompts_are_exact_prefixes_of_the_trained_sequences():
    """What vLLM sees at each turn must be the start of what the trainer scores, token for token."""
    proc = _chat_template_processor()
    cfg = windowed_cfg(window=4, max_turns=12)
    problems = smoke4.run_context_check(cfg, FakePool(1), proc, turns=12, window=4, tokenize=False)
    assert problems == []


def test_prefix_check_catches_a_changed_prompt():
    proc = _chat_template_processor()
    cfg = windowed_cfg(window=4, max_turns=6)
    policy = smoke4.ScriptedPolicy([call("noop")], proc)
    eps = rollout.run_episodes([EpisodeSpec("t", 0)], policy.generate, FakePool(1), cfg, SamplingSpec(), progress=False)
    tools = actions.tool_schemas("B", True)
    assert smoke4.check_prefixes(policy, eps, proc, tools, max_images=4) == []
    messages, n, text = policy.prompts[2]
    policy.prompts[2] = (messages, n, text.replace("Goal", "Gaol", 1))
    assert any("prefix" in p for p in smoke4.check_prefixes(policy, eps, proc, tools))


# ---------------------------------------------------------------------------
# Status, pool, and sync with a run name
# ---------------------------------------------------------------------------

NO_GPU = {"util": None, "mem_used_mb": None, "mem_total_mb": None, "power_w": None}


def test_status_for_run4_reads_run4_metrics_and_skips_eval_rows(tmp_path):
    res = tmp_path / "results" / "run4"
    (res / ".phases").mkdir(parents=True)
    m = res / "train" / "grpo-4b-run4" / "metrics.jsonl"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"step": 24, "kind": "train"}) + "\n" + json.dumps({"step": 25, "kind": "train"}) + "\n"
                 + json.dumps({"step": 25, "kind": "eval"}) + "\n" + json.dumps({"step": 99, "kind": "eval"}) + "\n")
    s = run_status.build_status(res, tmp_path / "ckpt", REPO / "configs/grpo_run4.yaml", gpu=lambda: dict(NO_GPU),
                                alive=lambda pid: False, run="run4")
    assert s["run"] == "run4" and s["grpo"]["step"] == 25 and s["grpo"]["max_steps"] == 400
    assert [p["name"] for p in s["phases"]] == run_status.PHASES


def test_phases_match_run4_sh():
    text = (REPO / "scripts" / "run4.sh").read_text()
    m = re.search(r"^PHASES=\(([^)]*)\)", text, re.M)
    assert m and m.group(1).split() == run_status.PHASES


def test_run4_status_json_cli_is_fast_and_never_imports_torch(tmp_path):
    res = tmp_path / "results" / "run4"
    (res / ".phases").mkdir(parents=True)
    (res / ".phases" / "smoke.done").write_text("2026-11-26T01:00:00Z\n")
    env = dict(os.environ, RUN4_RESULTS=str(res), RUN4_CKPT=str(tmp_path / "ckpt"))
    t0 = time.perf_counter()
    out = subprocess.run(["bash", str(REPO / "scripts" / "run4.sh"), "--status-json"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert time.perf_counter() - t0 < 5 and out.returncode == 0
    data = json.loads(out.stdout)
    assert data["run"] == "run4" and data["phases"][0]["status"] == "done" and data["current_phase"] == "screen"
    probe = ("import runpy, sys; sys.argv = ['x', '--run', 'run4', '--results', sys.argv[1]]; "
             "runpy.run_module('cua.run_status', run_name='__main__'); "
             "assert 'torch' not in sys.modules, 'imported torch'")
    out = subprocess.run([sys.executable, "-c", probe, str(res)], capture_output=True, text=True, cwd=REPO, timeout=30)
    assert out.returncode == 0, out.stderr
    # Run 3's command still reports run 3.
    out = subprocess.run([sys.executable, "-m", "cua.run3_status", "--results", str(res)], capture_output=True,
                         text=True, cwd=REPO, timeout=30)
    assert json.loads(out.stdout)["run"] == "run3"


def test_run4_task_lists_are_safe_and_complete():
    spec = load_tasks()
    cands = set(spec.train_candidates_run4)
    held = set(spec.heldout) | set(spec.heldout_v2) | set(spec.heldout_v2_candidates)
    assert not cands & held
    assert "stock-market" not in cands and "draw-circle" not in cands
    assert set(spec.train_candidates_run4_new_action) <= cands and {"drag-box", "use-slider"} <= cands
    assert set(spec.run4_turn_limited) <= set(spec.run4_long_horizon) <= cands
    assert set(spec.run3_pool) <= set(spec.train_candidates_run3)
    assert not any(t.endswith("-nodelay") for t in cands)
    assert resolve_task_list("candidates_run4") == spec.train_candidates_run4
    assert resolve_task_list("run4_turn_limited") == spec.run4_turn_limited
    sel = REPO / "results/run3/selected_tasks.json"
    if sel.exists():
        assert json.loads(sel.read_text())["selected"] == spec.run3_pool


def test_run4_sources_and_pool_selection():
    spec = load_tasks()
    src = run_pool.task_sources("run4", spec)
    assert set(src) == set(spec.train_candidates_run4)
    assert src["use-slider"] == "new_action" and src["use-colorwheel"] == "new_action"
    assert src["choose-date"] == "long_horizon" and src["bisect-angle"] == "run3_pool" and src["click-button"] == "other"
    rates = {t: (16, 0.0) for t in src}
    rates.update({"use-slider": (16, 1 / 16), "drag-box": (16, 0.30), "drag-items": (16, 0.0),
                  "book-flight": (16, 0.10), "search-engine": (16, 0.50), "choose-list": (16, 0.4),
                  "bisect-angle": (16, 0.45), "click-button": (16, 0.44), "click-link": (16, 0.9),
                  "multi-layouts": (16, 1.0), "form-sequence-2": (16, 0.95), "find-greatest": (16, 0.99),
                  "simple-arithmetic": (16, 0.97), "stock-market": (16, 0.5)})
    # Enough in-band tasks that the pool doesn't widen toward the solved ones.
    rates.update({t: (16, 0.5) for t in ("enter-date", "login-user", "enter-time", "copy-paste", "click-pie")})
    out = run_pool.select_run4(rates, src, spec)
    # One success in 16 is enough for a drag task; drag tasks get the first slots.
    assert out["learn"][:2] == ["drag-box", "use-slider"]
    assert {"book-flight", "search-engine", "choose-list"} <= set(out["learn"])
    assert out["dropped"]["drag-items"] == "zero successes"
    assert "stock-market" not in out["selected"]
    assert out["replay"] == ["form-sequence-2", "simple-arithmetic", "find-greatest"]
    assert out["dropped"]["click-link"].startswith("too easy")


def test_turn_budgets_from_screen_episodes():
    eps = ([{"task": "quick", "success": True, "num_turns": 1}] * 9 + [{"task": "quick", "success": True, "num_turns": 6}]
           + [{"task": "long", "success": True, "num_turns": 9}] * 5
           + [{"task": "hard", "success": False, "num_turns": 20}] * 16
           + [{"task": "crashy", "success": True, "num_turns": 1, "crashed": "boom"}])
    out = run_pool.turn_budgets_from_episodes(eps, max_turns=20, floor=10)
    assert out["budgets"] == {"hard": 20, "long": 20, "quick": 10}


def test_full_suite_list():
    out = run_pool.full_suite(["a", "stock-market", "b", "a"], {"stock-market": "coin flip", "zzz": "absent"})
    assert out == {"tasks": ["a", "b"], "excluded": {"stock-market": "coin flip"}}
    spec = load_tasks()
    assert "stock-market" in spec.full_suite_exclude and "use-autocomplete-nodelay" in spec.full_suite_exclude


from cua import hfsync as hs  # noqa: E402


class FakeHub:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.ops = []

    def list_files(self, prefix):
        return {p: len(b) for p, b in self.files.items() if p.startswith(prefix.rstrip("/") + "/")}

    def upload_folder(self, local, remote, message):
        self.ops.append(("folder", remote, message))
        for path in Path(local).rglob("*"):
            if path.is_file():
                self.files[f"{remote}/{path.relative_to(local).as_posix()}"] = path.read_bytes()

    def upload_files(self, files, message):
        for rpath, data in files:
            self.ops.append(("file", rpath, message))
            self.files[rpath] = Path(data).read_bytes() if isinstance(data, Path) else bytes(data)

    def download(self, remote_paths, dest):
        for rpath in remote_paths:
            out = Path(dest) / rpath
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.files[rpath])


def run4_layout(tmp_path):
    lay = hs.Layout.for_run("run4", repo_root=tmp_path / "repo")
    lay.ckpt_dir = tmp_path / "ws" / "ckpt" / "grpo-4b-run4"
    lay.rft_dir = tmp_path / "ws" / "ckpt" / "rft-4b-run4" / "final"
    lay.run2_adapter_dir = tmp_path / "ws" / "inputs" / "run3-adapter"
    return lay


def adapter(d: Path, tag="x", best_step=None):
    d.mkdir(parents=True, exist_ok=True)
    for f in hs.ADAPTER_FILES:
        (d / f).write_text(f"{f}-{tag}")
    if best_step is not None:
        (d / "best.json").write_text(json.dumps({"step": best_step, "eval_success": 0.7}))


def test_run4_layout_names():
    lay = hs.Layout.for_run("run4")
    assert lay.remote_ckpt == "checkpoints/grpo-4b-run4" and lay.remote_results == "results/run4"
    assert lay.prev_remote_ckpt == "checkpoints/grpo-4b-run3" and lay.remote_best == "checkpoints/grpo-4b-run4/best"
    assert str(lay.run2_adapter_dir).endswith("run3-adapter")
    env = {"RUN4_RESULTS": "/x/r4", "RUN3_ADAPTER": "/x/a3"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        lay = hs.Layout.from_env("run4")
        assert str(lay.results) == "/x/r4" and str(lay.run2_adapter_dir) == "/x/a3"
        assert hs.Layout.from_env().run == "run3"
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_run4_restore_takes_run3_final_adapter_and_result_files(tmp_path):
    lay = run4_layout(tmp_path)
    files = {f"checkpoints/grpo-4b-run3/final/{f}": b"r3" for f in hs.ADAPTER_FILES}
    files.update({f"checkpoints/grpo-4b-run3/checkpoint-240/{f}": b"old" for f in hs.ADAPTER_FILES})
    files["results/run3/heldout_v2.json"] = b"{}"
    files["results/run2/eval/run2-heldout2/summary.json"] = b"{}"
    hub = FakeHub(files)
    did = hs.restore(hub, lay)
    assert did["run3_adapter"] == "checkpoints/grpo-4b-run3/final"
    assert (lay.run2_adapter_dir / "adapter_model.safetensors").read_bytes() == b"r3"
    assert (lay.repo_root / "results/run3/heldout_v2.json").exists()
    assert (lay.repo_root / "results/run2/eval/run2-heldout2/summary.json").exists()
    assert hub.ops == []


def test_best_adapter_syncs_once_per_version_and_restores(tmp_path):
    lay = run4_layout(tmp_path / "pod1")
    hub = FakeHub({f"checkpoints/grpo-4b-run3/final/{f}": b"r3" for f in hs.ADAPTER_FILES})
    (lay.results / ".phases").mkdir(parents=True)
    adapter(lay.best_dir, "b25", best_step=25)
    state = hs.SyncState()
    did = hs.sync_once(hub, lay, state)
    assert "best-step-25" in did["adapters"]
    assert hs.sync_once(hub, lay, state)["adapters"] == []
    adapter(lay.best_dir, "b50", best_step=50)
    assert hs.sync_once(hub, lay, state)["adapters"] == ["best-step-50"]
    assert all(msg.startswith("run4:") for _, _, msg in hub.ops)
    assert "checkpoints/grpo-4b-run4/best/best.json" in hs.final_upload_map(lay)

    lay2 = run4_layout(tmp_path / "pod2")
    did = hs.restore(hub, lay2)
    assert "checkpoints/grpo-4b-run4/best" in did["adapters"]
    assert json.loads((lay2.best_dir / "best.json").read_text())["step"] == 50


# ---------------------------------------------------------------------------
# Pipeline glue, configs, compare, and report
# ---------------------------------------------------------------------------


def test_pool_size_respects_cpu_and_memory():
    assert pipeline.pool_size(64, 32, 250.0) == 64
    assert pipeline.pool_size(64, 12, 250.0) == 48          # 4 per vCPU
    assert pipeline.pool_size(64, 32, 60.0) == 33           # (60 - 40) / 0.6
    assert pipeline.pool_size(64, 2, 41.0) == 16            # never below run 3's 16
    assert pipeline.pool_size(8, 32, None) == 8


def test_choose_adapter_prefers_final_unless_best_clearly_wins():
    assert pipeline.choose_adapter(0.70, None) == "final"
    assert pipeline.choose_adapter(0.70, {"eval_success": 0.71}) == "final"
    assert pipeline.choose_adapter(0.70, {"eval_success": 0.73}) == "best"
    assert pipeline.choose_adapter(None, {"eval_success": 0.5}) == "best"


def test_choose_adapter_cli(tmp_path):
    best = tmp_path / "best"
    adapter(best, best_step=75)
    (tmp_path / "s.json").write_text(json.dumps({"overall_success": 0.6}))
    out = tmp_path / "choice.json"
    res = subprocess.run([sys.executable, "-m", "cua.pipeline", "choose-adapter", "--final", "/f", "--best", str(best),
                          "--final-eval", str(tmp_path / "s.json"), "--out", str(out)], capture_output=True, text=True,
                         cwd=REPO, timeout=30)
    assert res.returncode == 0 and res.stdout.strip() == str(best)
    assert json.loads(out.read_text())["choice"] == "best"


def test_grpo_run4_config():
    cfg = load_config(REPO / "configs/grpo_run4.yaml")
    g = cfg.grpo
    assert cfg.run_name == "grpo-4b-run4" and g.output_dir == "/workspace/ckpt/grpo-4b-run4"
    assert cfg.env.max_turns == 20 and cfg.env.context_window_turns == 4 and cfg.env.track_b_extra_tools
    assert g.ratio_cap > 1 + g.epsilon_high and g.sampler_check.enabled and g.sampler_check.tis_cap == 2.0
    assert g.max_tokens_norm == 10 * cfg.generation.max_new_tokens
    assert g.keep_best and g.eval_seed_range == "baseline" and g.stop_file == ".stop_grpo"
    assert g.train_seed_offset > load_config(REPO / "configs/grpo_run3.yaml").grpo.train_seed_offset + 1000
    # A windowed prompt can never need more images than vLLM allows.
    assert cfg.env.context_window_turns + 1 <= cfg.env.max_turns + 1
    e = load_config(REPO / "configs/eval_run4.yaml")
    for key in ("track", "max_turns", "context_window_turns", "track_b_extra_tools"):
        assert e.env[key] == cfg.env[key]


def test_report_preset_run4_names_match_run4_sh():
    preset = report.PRESETS["run4"]
    names = {n for pool in preset["POOLS"].values() for n in pool.values()}
    names |= {n for _, run, cols in preset["JOURNEY"] if run == "run4" for n in cols.values()}
    evals = set(re.findall(r"eval_cmd\s+([a-zA-Z0-9-]+)\s", (REPO / "scripts/run4.sh").read_text()))
    assert names <= evals, names - evals


def test_report_run4_goals_and_journey(tmp_path):
    root = tmp_path / "results" / "run4"

    def summary(path: Path, overall: float, per_task: dict):
        path.mkdir(parents=True, exist_ok=True)
        (path / "summary.json").write_text(json.dumps({"overall_success": overall, "n_episodes": 10,
                                                       "per_task": {t: {"success_rate": r} for t, r in per_task.items()}}))

    summary(root / "eval/run4-heldout4", 0.75, {})
    summary(root / "eval/run4-full4", 0.6, {"book-flight": 0.5, "search-engine": 0.2, "drag-box": 0.1, "use-slider": 0.0})
    summary(root / "eval/run3-full4", 0.5, {"book-flight": 0.0, "search-engine": 0.0})
    summary(tmp_path / "results/run3/eval/run3-heldout3", 0.715, {})
    (root / "compare").mkdir(parents=True)
    (root / "compare/full4-run4-vs-run3.json").write_text(json.dumps({"overall": {"a": 0.5, "b": 0.6, "diff": 0.1,
                                                                                  "ci95": [0.05, 0.15]}}))
    m = root / "train/grpo-4b-run4/metrics.jsonl"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"step": 1, "kind": "train", "time/rollout_s": 1, "grad_norm_max": 0.2, "train/approx_kl": 0.01})
                 + "\n" + json.dumps({"step": 1, "kind": "eval", "eval/success": 0.5}) + "\n")
    lists = {"turn_limited": ["book-flight", "search-engine"], "new_action": ["drag-box", "use-slider"]}
    goals = report.run4_goals(root, lists)
    assert [met for _, met, _ in goals] == [True, True, True, True, True]
    keys = list(report.PRESETS["run4"]) + ["JOURNEY", "RUN4_GOALS"]
    saved = {k: getattr(report, k) for k in keys if hasattr(report, k)}
    report.apply_preset("run4")
    try:
        html = report.journey_section(tmp_path / "results")
        assert "71.5%" in html and "75.0%" in html and "n/a" in html
        assert "Run 4 success criteria" in report.goals_section(root)
    finally:
        for k, v in saved.items():
            setattr(report, k, v)


def test_compare_filters_tasks():
    from cua.compare import filter_tasks

    out = {("a", 0): True, ("b", 0): False, ("c", 1): True}
    assert filter_tasks(out, keep={"a", "b"}) == {("a", 0): True, ("b", 0): False}
    assert filter_tasks(out, drop={"a"}) == {("b", 0): False, ("c", 1): True}
