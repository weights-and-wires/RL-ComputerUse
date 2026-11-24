"""Tests for the run-3 task pool (cua/run3_pool.py), task lists, and configs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cua import report, run3_pool
from cua.config import load_config, load_tasks, resolve_task_list

REPO = Path(__file__).resolve().parent.parent


def rates(**kw: float) -> dict[str, tuple[int, float]]:
    return {t.replace("_", "-"): (16, r) for t, r in kw.items()}


SOURCES = {
    "r2a": "run2_pool", "r2b": "run2_pool", "r2c": "run2_pool", "r2d": "run2_pool", "r2-solved": "run2_pool",
    "r2-solved2": "run2_pool",
    "r1a": "run1_pool", "r1b": "run1_pool", "r1-solved": "run1_pool",
    "ba": "trackb_extra", "bb": "trackb_extra", "bc": "trackb_extra", "bd": "trackb_extra",
    "o1": "other", "o2": "other", "o-easy": "other", "o-zero": "other",
}
RATES = {
    "r2a": (16, 0.45), "r2b": (16, 0.30), "r2c": (16, 0.70), "r2d": (16, 0.50), "r2-solved": (16, 0.94),
    "r2-solved2": (16, 1.0),
    "r1a": (16, 0.80), "r1b": (16, 0.10), "r1-solved": (16, 1.0),
    "ba": (16, 0.06), "bb": (16, 0.85), "bc": (16, 0.40), "bd": (16, 0.44),
    "o1": (16, 0.45), "o2": (16, 0.46), "o-easy": (16, 0.95), "o-zero": (16, 0.0),
}


def test_mix_reserves_a_share_per_source_then_fills_by_closeness():
    out = run3_pool.mix_pool(RATES, SOURCES, max_learn=8, min_per_source=2, max_replay=3, min_learn=1)
    learn = out["learn"]
    assert len(learn) == 8
    # Two per source first (closest to 0.45), then the closest of the rest.
    assert learn[:6] == ["r2a", "r2d", "r1b", "r1a", "bd", "bc"]
    assert set(learn[6:]) == {"o1", "o2"}
    assert out["dropped"]["r2b"].startswith("over cap")
    # Replay alternates between the two earlier pools, most fragile first.
    assert out["replay"] == ["r2-solved", "r1-solved", "r2-solved2"]
    assert out["selected"] == learn + out["replay"]
    assert set(out["rates"]) == set(out["selected"])
    assert out["source"]["bd"] == "trackb_extra"
    assert out["dropped"]["o-easy"].startswith("too easy")
    assert out["dropped"]["o-zero"] == "zero successes"


def test_band_edges_are_inclusive_and_missing_tasks_are_reported():
    srcs = dict(SOURCES, absent="other")
    out = run3_pool.mix_pool(RATES, srcs, max_learn=16, min_per_source=3, max_replay=0, min_learn=1)
    assert "ba" in out["learn"] and "bb" in out["learn"]  # 0.06 and 0.85
    assert out["replay"] == []
    assert out["dropped"]["absent"].startswith("missing")


def test_widening_adds_nearest_tasks_with_some_success():
    r = {"a": (16, 0.9), "b": (16, 0.02), "c": (16, 0.0), "d": (16, 0.5)}
    s = {"a": "other", "b": "other", "c": "other", "d": "other"}
    out = run3_pool.mix_pool(r, s, min_learn=3)
    assert out["learn"][0] == "d"
    assert set(out["learn"][1:]) == {"a", "b"}
    assert set(out["criteria"]["widened_beyond_band"]) == {"a", "b"}


def test_task_sources_cover_every_candidate_with_pool_priority():
    spec = load_tasks()
    src = run3_pool.task_sources(spec)
    assert set(src) == set(spec.train_candidates_run3)
    assert src["social-media"] == "run2_pool"  # in both earlier pools; run 2 wins
    assert src["enter-date"] == "run1_pool"
    assert src["circle-center"] == "trackb_extra"
    assert src["click-button"] == "other"


def test_run3_task_lists_are_safe():
    spec = load_tasks()
    held = set(spec.heldout) | set(spec.heldout_v2) | set(spec.heldout_v2_candidates)
    assert not held & set(spec.train_candidates_run3)
    assert set(spec.train_candidates_run3_extra).isdisjoint(spec.train_candidates_run2)
    assert len(spec.heldout_v2) == 10 and set(spec.heldout) <= set(spec.heldout_v2)
    assert resolve_task_list("run2_pool") == spec.run2_pool
    assert resolve_task_list("candidates_run3") == spec.train_candidates_run3
    for bad in ("hover-shape", "drag-box", "use-slider", "highlight-text", "click-test-transfer"):
        assert bad not in spec.train_candidates_run3


def test_frozen_lists_match_run2_results_when_present():
    spec = load_tasks()
    held = REPO / "results/run2/heldout_v2.json"
    sel = REPO / "results/run2/selected_tasks.json"
    if not held.exists() or not sel.exists():
        pytest.skip("run 2's results aren't in this checkout")
    assert json.loads(held.read_text())["heldout_v2"] == spec.heldout_v2
    assert json.loads(sel.read_text())["selected"] == spec.run2_pool


def test_heldout_v2_falls_back_to_the_frozen_list(tmp_path, monkeypatch):
    monkeypatch.setenv("CUA_RESULTS_DIR", str(tmp_path))
    assert resolve_task_list("heldout_v2") == load_tasks().heldout_v2


def test_grpo_run3_config():
    cfg = load_config(REPO / "configs/grpo_run3.yaml")
    g = cfg.grpo
    assert cfg.env.track == "B" and cfg.env.max_turns == 10
    assert cfg.run_name == "grpo-4b-run3" and g.output_dir == "/workspace/ckpt/grpo-4b-run3"
    assert g.sampler == "adaptive" and g.adaptive.floor == pytest.approx(0.10)
    assert g.dynamic_sampling.enabled and g.updates_per_step == 2 and g.assistant_only_logits
    assert g.eval_num_seeds == 20
    assert g.stop_file == ".stop_grpo" and g.samples_every > 0
    # Fresh seeds past run 2's (offset 100000, about 100 seeds per task).
    run2 = load_config(REPO / "configs/grpo_run2.yaml").grpo.train_seed_offset
    assert g.train_seed_offset >= run2 + 100000
    assert load_config(REPO / "configs/rft_run3.yaml").env.track == "B"


def test_report_preset_run3_names_match_run3_sh():
    preset = report.PRESETS["run3"]
    names = {n for pool in preset["POOLS"].values() for n in pool.values()}
    text = (REPO / "scripts/run3.sh").read_text()
    evals = set(re.findall(r"eval_cmd\s+([a-zA-Z0-9-]+)\s", text))
    assert names <= evals, names - evals
