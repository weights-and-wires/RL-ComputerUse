"""Tests for the run-2 additions: samplers, dynamic sampling, sparse log-probs,
paired comparison, held-out screening, and the task-list guards.

None of these need a GPU, BrowserGym, or vLLM. Tests that need torch skip
without it.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from cua import compare
from cua.config import load_tasks, resolve_task_list
from cua.nano_grpo.sampler import (
    AdaptiveTaskSampler,
    RoundRobinSampler,
    make_sampler,
    mixed_group_prob,
)
from cua.select_tasks import pick_heldout_extra

# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def test_mixed_group_prob_matches_plan_table():
    assert mixed_group_prob(0.5, 8) == pytest.approx(0.9922, abs=1e-4)
    assert mixed_group_prob(0.1, 8) == pytest.approx(0.5695, abs=1e-4)
    assert mixed_group_prob(0.9, 8) == pytest.approx(mixed_group_prob(0.1, 8))
    assert mixed_group_prob(0.0, 8) == 0.0
    assert mixed_group_prob(1.0, 8) == 0.0


def test_round_robin_reproduces_run1_order():
    # Run 1's next_prompts: the k-th prompt is task k % n with seed index k // n.
    tasks = ["a", "b", "c"]
    s = RoundRobinSampler(tasks, seed_start=1000)
    got = s.sample(4) + s.sample(4)
    want = [(tasks[k % 3], 1000 + k // 3) for k in range(8)]
    assert got == want


def test_round_robin_state_round_trip():
    s = RoundRobinSampler(["a", "b"], 0)
    s.sample(5)
    t = RoundRobinSampler(["a", "b"], 0)
    t.load_state_dict(s.state_dict())
    assert t.sample(3) == s.sample(3)


def test_adaptive_weights_favor_the_middle_band():
    s = AdaptiveTaskSampler(["mid", "done", "never"], 0, 8, init_rates={"mid": 0.5, "done": 1.0, "never": 0.0})
    w = s.weights()
    assert w["mid"] > 0.99
    assert w["done"] == pytest.approx(0.05)  # the floor
    assert w["never"] == pytest.approx(0.05)


def test_adaptive_sampling_frequency_follows_weights():
    s = AdaptiveTaskSampler(
        ["mid", "done"], 0, 8, init_rates={"mid": 0.5, "done": 1.0}, ema_alpha=0.0, rng=random.Random(1)
    )
    counts = {"mid": 0, "done": 0}
    for _ in range(2000):
        task, _ = s.sample(1)[0]
        counts[task] += 1
    # Expected share of "done": 0.05 / (0.05 + 0.992), about 5%.
    assert 0.02 < counts["done"] / 2000 < 0.09


def test_adaptive_never_reuses_a_seed_and_spreads_over_tasks():
    tasks = ["a", "b", "c"]
    s = AdaptiveTaskSampler(tasks, 500, 8, rng=random.Random(0))
    first = s.sample(3)
    assert sorted(t for t, _ in first) == tasks  # without replacement while possible
    prompts = first + s.sample(7) + s.sample(7)
    assert len(set(prompts)) == len(prompts)
    for t in tasks:
        seeds = [seed for task, seed in prompts if task == t]
        assert seeds == list(range(500, 500 + len(seeds)))


def test_adaptive_update_is_an_ema():
    s = AdaptiveTaskSampler(["a"], 0, 8, init_rates={"a": 0.5}, ema_alpha=0.25)
    s.update("a", [True] * 8)
    assert s.rates()["a"] == pytest.approx(0.625)
    s.update("unknown-task", [True])  # ignored
    s.update("a", [])  # ignored
    assert s.rates()["a"] == pytest.approx(0.625)


def test_adaptive_state_round_trip_is_exact():
    s = AdaptiveTaskSampler(["a", "b", "c"], 0, 8, rng=random.Random(3))
    s.sample(5)
    s.update("a", [True, False])
    state = json.loads(json.dumps(s.state_dict()))  # what the checkpoint stores
    t = AdaptiveTaskSampler(["a", "b", "c"], 0, 8, rng=random.Random(99))
    t.load_state_dict(state)
    assert t.rates() == s.rates()
    assert t.sample(6) == s.sample(6)


def test_make_sampler_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_sampler("greedy", ["a"], 0, 8)


# ---------------------------------------------------------------------------
# Grouping and dynamic sampling (cua/nano_grpo/loop.py)
# ---------------------------------------------------------------------------

loop = pytest.importorskip("cua.nano_grpo.loop")


def fake_episode(task: str, seed: int, success: bool, crashed: bool = False):
    return SimpleNamespace(
        task=task, seed=seed, success=success, crashed=crashed,
        reward=SimpleNamespace(total=1.0 if success else 0.0),
    )


def test_build_training_groups_keeps_only_mixed_groups():
    eps = (
        [fake_episode("a", 1, s) for s in (True, False, True, False)]  # mixed
        + [fake_episode("b", 1, True) for _ in range(4)]  # all pass
        + [fake_episode("c", 1, False) for _ in range(3)] + [fake_episode("c", 1, True, crashed=True)]
    )
    kept, rewards, counters = loop.build_training_groups(eps, scale_rewards=True)
    assert len(kept) == 1 and len(kept[0]) == 4
    assert sum(adv for _, adv in kept[0]) == pytest.approx(0.0, abs=1e-6)
    assert len(rewards) == 3
    assert counters["episodes/crashed"] == 1.0
    assert counters["groups/used"] == 1.0


class ScriptedRoll:
    """Plays back a fixed success pattern per round: one bool list per group."""

    def __init__(self, rounds: list[list[list[bool]]]):
        self.rounds = rounds
        self.calls = 0

    def __call__(self, specs):
        pattern = self.rounds[self.calls]
        self.calls += 1
        groups: dict[tuple[str, int], list] = {}
        for spec in specs:
            groups.setdefault((spec.task, spec.seed), []).append(spec)
        out = []
        for (key, members), outcomes in zip(groups.items(), pattern):
            out += [fake_episode(key[0], key[1], ok) for ok, _ in zip(outcomes, members)]
        return out


MIXED = [True, False, False, False]
PASS = [True] * 4


def test_collect_groups_stops_once_enough_groups_are_mixed():
    roll = ScriptedRoll([[MIXED, PASS], [MIXED, MIXED], [MIXED, MIXED]])
    sampler = RoundRobinSampler(["a", "b"], 0)
    eps, mixed, rewards, crashed, rounds = loop.collect_groups(
        sampler, roll, prompts_per_step=2, num_generations=4, min_mixed=3, max_rounds=3, scale_rewards=True
    )
    assert rounds == 2 and roll.calls == 2
    assert len(mixed) == 3
    assert len(eps) == 16 and len(rewards) == 4 and crashed == 0


def test_collect_groups_respects_max_rounds():
    roll = ScriptedRoll([[PASS, PASS]] * 5)
    _, mixed, _, _, rounds = loop.collect_groups(
        RoundRobinSampler(["a", "b"], 0), roll, 2, 4, min_mixed=6, max_rounds=3, scale_rewards=True
    )
    assert rounds == 3 and mixed == []


def test_collect_groups_with_one_round_is_run1_behavior():
    roll = ScriptedRoll([[PASS, PASS], [MIXED, MIXED]])
    _, mixed, _, _, rounds = loop.collect_groups(
        RoundRobinSampler(["a", "b"], 0), roll, 2, 4, min_mixed=6, max_rounds=1, scale_rewards=True
    )
    assert rounds == 1 and mixed == []


def test_collect_groups_feeds_every_group_to_the_sampler():
    roll = ScriptedRoll([[PASS, [False] * 4]])
    sampler = AdaptiveTaskSampler(["a", "b"], 0, 4, init_rates={"a": 0.5, "b": 0.5}, ema_alpha=0.5, rng=random.Random(0))
    loop.collect_groups(sampler, roll, 2, 4, min_mixed=1, max_rounds=1, scale_rewards=True)
    rates = sorted(sampler.rates().values())
    assert rates == [pytest.approx(0.25), pytest.approx(0.75)]


# ---------------------------------------------------------------------------
# Assistant-only log-probs (cua/nano_grpo/loss.py)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
L = pytest.importorskip("cua.nano_grpo.loss")


def test_logit_positions_shift_and_union():
    mask = torch.tensor([[0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 0, 0]])
    assert L.logit_positions(mask).tolist() == [0, 1, 2, 4]


def test_sparse_logprobs_match_full_logprobs_and_gradients():
    torch.manual_seed(0)
    B, T, V = 2, 12, 50
    full_logits = torch.randn(B, T, V, requires_grad=True)
    ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T)
    mask[0, 3:6] = 1
    mask[1, 8:11] = 1
    pos = L.logit_positions(mask)

    full = L.token_logprobs(full_logits, ids, chunk_size=4)
    sparse = L.token_logprobs_at(full_logits[:, pos], ids, pos, chunk_size=2)
    assert torch.allclose(full * mask, sparse * mask, atol=1e-6)

    adv = torch.tensor([1.0, -0.5])
    g_full = torch.autograd.grad(L.clipped_pg_loss(full, full.detach(), adv, mask, 0.2, 0.28)[0], full_logits)[0]
    g_sparse = torch.autograd.grad(L.clipped_pg_loss(sparse, sparse.detach(), adv, mask, 0.2, 0.28)[0], full_logits)[0]
    assert torch.allclose(g_full, g_sparse, atol=1e-6)


def test_sparse_logprobs_handle_no_assistant_tokens():
    logits = torch.randn(1, 0, 10)
    out = L.token_logprobs_at(logits, torch.zeros(1, 5, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert out.shape == (1, 5) and out.abs().sum() == 0


# ---------------------------------------------------------------------------
# Paired comparison, held-out screening, task lists
# ---------------------------------------------------------------------------


def test_paired_bootstrap_on_known_pairs():
    by_task = {"t": [(0, 1)] * 5 + [(1, 1)] * 5}
    r = compare.paired_bootstrap(by_task, n_boot=500)
    assert r["tasks"]["t"]["diff"] == pytest.approx(0.5)
    assert r["tasks"]["t"]["b_only"] == 5 and r["tasks"]["t"]["a_only"] == 0
    lo, hi = r["overall"]["ci95"]
    assert 0.0 < lo <= 0.5 <= hi <= 1.0


def test_compare_pairs_only_shared_seeds(tmp_path):
    def write(name, rows):
        d = tmp_path / name
        d.mkdir()
        (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return d

    a = write("a", [{"task": "t", "seed": s, "sample_idx": 0, "success": False, "crashed": False} for s in range(3)])
    b = write("b", [{"task": "t", "seed": s, "sample_idx": 0, "success": True, "crashed": s == 1} for s in range(4)])
    by_task = compare.pair(compare.load_outcomes(a), compare.load_outcomes(b))
    assert by_task == {"t": [(0, 1), (0, 1)]}  # seed 1 crashed in b, seed 3 missing in a


def test_pick_heldout_extra_prefers_rates_near_half():
    rates = {"x": (16, 0.5), "y": (16, 0.9), "z": (16, 0.4), "w": (16, 0.0)}
    r = pick_heldout_extra(rates, ["w", "x", "y", "z", "missing"], 2)
    assert r["extra"] == ["x", "z"]
    assert r["heldout_v2"] == load_tasks().heldout + ["x", "z"]
    assert r["missing"] == ["missing"]


def test_run2_task_lists():
    spec = load_tasks()
    run2 = resolve_task_list("candidates_run2")
    assert run2[: len(spec.train_candidates)] == spec.train_candidates
    assert len(run2) == len(set(run2))
    assert set(spec.heldout_v2_candidates).isdisjoint(run2)
    assert set(spec.heldout).isdisjoint(run2)
    assert len(resolve_task_list("run1_pool")) == 10
