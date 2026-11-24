"""Tests for the dashboard sample writer (cua/samples.py) and the loop's run-3 helpers."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from PIL import Image

from cua import samples


def fake_turn(text: str, tool: str, args: dict, error: str = "", fmt: str | None = None, size=(1000, 600)):
    return SimpleNamespace(
        observation=SimpleNamespace(screenshot=Image.new("RGB", size, "white")),
        generation=SimpleNamespace(text=text),
        parsed=SimpleNamespace(tool_name=tool, tool_args=args, action=None, format_error=fmt),
        env_error=error,
    )


def fake_episode(task="click-shape", seed=300123, success=True, reward=1.05, turns=None, crashed=""):
    return SimpleNamespace(
        task=task, seed=seed, success=success, crashed=crashed,
        reward=SimpleNamespace(total=reward),
        turns=turns if turns is not None else [
            fake_turn("The circle is at the top.", "mouse_click", {"coordinate": [412, 230]}),
            fake_turn("Type the answer.", "keyboard_type", {"text": "hi"}, error="element not found"),
        ],
    )


def test_write_sample_matches_the_contract(tmp_path):
    out = samples.write_sample(fake_episode(), 120, tmp_path)
    assert out == tmp_path / "samples" / "step-000120"
    ep = json.loads((out / "episode.json").read_text())
    assert ep == {
        "step": 120, "task": "click-shape", "seed": 300123, "reward": 1.05, "success": True,
        "turns": [
            {"image": "turn-0.jpg", "response": "The circle is at the top.",
             "action": "mouse_click(coordinate=[412, 230])", "error": ""},
            {"image": "turn-1.jpg", "response": "Type the answer.",
             "action": 'keyboard_type(text="hi")', "error": "element not found"},
        ],
    }
    with Image.open(out / "turn-0.jpg") as img:
        assert img.format == "JPEG" and img.size == (640, 384)
    assert not list((tmp_path / "samples").glob("*.tmp"))


def test_write_sample_keeps_small_images_and_replaces_old_folder(tmp_path):
    ep = fake_episode(turns=[fake_turn("x", "noop", {}, size=(498, 321))])
    samples.write_sample(ep, 5, tmp_path)
    (tmp_path / "samples/step-000005/stale.txt").write_text("old")
    out = samples.write_sample(ep, 5, tmp_path)
    assert not (out / "stale.txt").exists()
    with Image.open(out / "turn-0.jpg") as img:
        assert img.size == (498, 321)
    assert json.loads((out / "episode.json").read_text())["turns"][0]["action"] == "noop()"


def test_format_error_and_missing_tool_call():
    turn = fake_turn("no tool call here", "", {}, fmt="no <tool_call> found")
    turn.parsed.action = None
    assert samples.action_text(turn.parsed) == ""
    ep = fake_episode(turns=[turn])
    assert ep.turns[0].parsed.format_error


def test_pick_episode_prefers_mixed_groups_and_the_wanted_outcome():
    a_ok = fake_episode("a", 1, True)
    a_bad = fake_episode("a", 1, False)
    b_ok = fake_episode("b", 2, True)  # all-success group
    crashed = fake_episode("c", 3, False, crashed="boom")
    eps = [a_ok, a_bad, b_ok, crashed]
    rng = random.Random(0)
    assert samples.pick_episode(eps, {("a", 1)}, want_success=False, rng=rng) is a_bad
    assert samples.pick_episode(eps, {("a", 1)}, want_success=True, rng=rng) is a_ok
    assert samples.pick_episode([b_ok, crashed], set(), want_success=False, rng=rng) is b_ok
    assert samples.pick_episode([crashed], set(), want_success=True) is None


# The loop's helpers live in a module that imports torch and the model stack.
loop = pytest.importorskip("cua.nano_grpo.loop")


def test_stop_flag(tmp_path):
    flag = tmp_path / ".stop_grpo"
    assert not loop.stop_requested(flag)
    assert not loop.stop_requested(None)
    flag.touch()
    assert loop.stop_requested(flag)


def test_trim_metrics_after_resume(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": s} for s in (1, 2, 3)] + [{"step": 2, "eval/success": 0.5}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"step": 4, "cut')
    assert loop.trim_metrics(path, 2) == 2
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 1}, {"step": 2}, {"step": 2, "eval/success": 0.5}]
    assert loop.trim_metrics(path, 2) == 0
    assert loop.trim_metrics(tmp_path / "missing.jsonl", 0) == 0


def test_wandb_run_json(tmp_path):
    run = SimpleNamespace(entity="me", project="rl-computer-use", id="abc123", url="https://wandb.ai/x")
    loop.write_wandb_run_json(run, tmp_path / "wandb_run.json")
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == {
        "entity": "me", "project": "rl-computer-use", "id": "abc123", "url": "https://wandb.ai/x"}
