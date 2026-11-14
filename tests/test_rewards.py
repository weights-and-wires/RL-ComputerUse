"""Tests for cua.rewards."""

from __future__ import annotations

import pytest

from cua.rewards import compute_reward, episode_stats, group_stats, reward_from_counts
from cua.types import Episode, Generation, ParsedAction, Turn

# base.yaml: format_penalty 0.1, cap 0.3, efficiency_coef 0.1, max_turns 8.

# -- reward arithmetic -----------------------------------------------------------


def test_failure_without_errors_is_zero(cfg):
    r = reward_from_counts(False, 0, 8, cfg)
    assert (r.success, r.format_penalty, r.efficiency_bonus, r.total) == (0.0, 0.0, 0.0, 0.0)


def test_success_with_efficiency_bonus(cfg):
    r = reward_from_counts(True, 0, 2, cfg)
    assert r.success == 1.0
    assert r.efficiency_bonus == pytest.approx(0.1 * (1 - 2 / 8))
    assert r.total == pytest.approx(1.075)


def test_success_on_the_last_turn_gets_no_bonus(cfg):
    assert reward_from_counts(True, 0, 8, cfg).efficiency_bonus == pytest.approx(0.0)


def test_bonus_is_success_only(cfg):
    """A fast failure must not earn a bonus."""
    assert reward_from_counts(False, 0, 1, cfg).efficiency_bonus == 0.0


def test_format_penalty_scales_then_caps(cfg):
    assert reward_from_counts(False, 1, 3, cfg).format_penalty == pytest.approx(-0.1)
    assert reward_from_counts(False, 2, 3, cfg).format_penalty == pytest.approx(-0.2)
    assert reward_from_counts(False, 3, 3, cfg).format_penalty == pytest.approx(-0.3)
    assert reward_from_counts(False, 7, 8, cfg).format_penalty == pytest.approx(-0.3)  # capped


def test_success_with_format_error(cfg):
    r = reward_from_counts(True, 1, 8, cfg)
    assert r.total == pytest.approx(1.0 - 0.1 + 0.0)


def test_total_is_the_sum_of_the_parts(cfg):
    r = reward_from_counts(True, 2, 4, cfg)
    assert r.total == pytest.approx(r.success + r.format_penalty + r.efficiency_bonus)
    assert set(r.to_dict()) == {"reward/total", "reward/success", "reward/format_penalty", "reward/efficiency_bonus"}


def test_bonus_never_goes_negative(cfg):
    assert reward_from_counts(True, 0, 20, cfg).efficiency_bonus == 0.0


def test_reward_reads_config_values(cfg):
    cfg.rewards.format_penalty = 0.05
    cfg.rewards.format_penalty_cap = 0.5
    cfg.rewards.efficiency_coef = 0.2
    cfg.env.max_turns = 4
    r = reward_from_counts(True, 2, 1, cfg)
    assert r.format_penalty == pytest.approx(-0.1)
    assert r.efficiency_bonus == pytest.approx(0.2 * 0.75)


# -- episodes --------------------------------------------------------------------


def make_episode(task="click-button", success=False, turns=2, format_errors=0, env_errors=0, **kwargs) -> Episode:
    """Build an Episode with `turns` turns, the first `format_errors` of which are format errors."""
    ep_turns = []
    for i in range(turns):
        parsed = ParsedAction(action=None if i < format_errors else "noop()", format_error="bad" if i < format_errors else None)
        ep_turns.append(
            Turn(
                step=i,
                observation=None,  # type: ignore[arg-type]  (the reward code never reads it)
                prompt_len=2,
                generation=Generation(text=""),
                parsed=parsed,
                env_error="boom" if i < env_errors else "",
            )
        )
    return Episode(task=task, seed=0, turns=ep_turns, success=success, **kwargs)


def test_compute_reward_uses_episode_counts(cfg):
    ep = make_episode(success=True, turns=4, format_errors=1)
    r = compute_reward(ep, cfg)
    assert r.total == pytest.approx(1.0 - 0.1 + 0.1 * (1 - 4 / 8))


def test_crashed_episode_gets_zero_reward(cfg):
    ep = make_episode(success=True, turns=2, crashed="Chromium died")
    assert compute_reward(ep, cfg).total == 0.0


# -- group statistics ------------------------------------------------------------


def test_group_stats_zero_variance_fraction():
    stats = group_stats([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    assert stats["groups/zero_variance_frac"] == pytest.approx(2 / 3)
    assert stats["groups/mean_std"] == pytest.approx(0.5 / 3)  # population std of [1,0,1,0] is 0.5
    assert stats["groups/mean_reward"] == pytest.approx(0.5)
    assert stats["groups/count"] == 3


def test_group_stats_tiny_differences_count_as_zero_variance():
    assert group_stats([[0.5, 0.5 + 1e-9]])["groups/zero_variance_frac"] == 1.0


def test_group_stats_empty():
    stats = group_stats([])
    assert stats["groups/zero_variance_frac"] == 0.0 and stats["groups/count"] == 0.0
    assert group_stats([[], []])["groups/count"] == 0.0


def test_group_stats_single_sample_group_is_zero_variance():
    assert group_stats([[0.7]])["groups/zero_variance_frac"] == 1.0


# -- episode statistics ----------------------------------------------------------


def _scored(ep: Episode, cfg) -> Episode:
    ep.reward = compute_reward(ep, cfg)
    return ep


def test_episode_stats(cfg):
    episodes = [
        _scored(make_episode("click-button", success=True, turns=2), cfg),
        _scored(make_episode("click-button", success=False, turns=4, truncated=True), cfg),
        _scored(make_episode("enter-text", success=False, turns=2, format_errors=1), cfg),
        _scored(make_episode("enter-text", success=False, turns=2, env_errors=1), cfg),
        _scored(make_episode("enter-text", success=True, turns=1, crashed="died"), cfg),
    ]
    stats = episode_stats(episodes)

    assert stats["episodes/count"] == 4  # the crashed episode is excluded
    assert stats["episodes/crashed"] == 1
    assert stats["success/overall"] == pytest.approx(1 / 4)
    assert stats["success/click-button"] == pytest.approx(1 / 2)
    assert stats["success/enter-text"] == 0.0
    assert stats["turns/mean"] == pytest.approx((2 + 4 + 2 + 2) / 4)
    assert stats["format_error/rate_per_turn"] == pytest.approx(1 / 10)
    assert stats["format_error/rate_per_episode"] == pytest.approx(1 / 4)

    # Reward components are logged separately.
    assert stats["reward/success"] == pytest.approx(0.25)
    assert stats["reward/format_penalty"] == pytest.approx(-0.1 / 4)
    assert stats["reward/efficiency_bonus"] == pytest.approx(0.1 * (1 - 2 / 8) / 4)
    assert stats["reward/total"] == pytest.approx(
        stats["reward/success"] + stats["reward/format_penalty"] + stats["reward/efficiency_bonus"]
    )

    assert stats["failure/success"] == pytest.approx(1 / 4)
    assert stats["failure/ran_out_of_turns"] == pytest.approx(1 / 4)
    assert stats["failure/format_error"] == pytest.approx(1 / 4)
    assert stats["failure/invalid_action"] == pytest.approx(1 / 4)
    assert stats["failure/wrong_action"] == 0.0
    assert sum(v for k, v in stats.items() if k.startswith("failure/")) == pytest.approx(1.0)


def test_episode_stats_empty_and_all_crashed():
    assert episode_stats([]) == {"episodes/count": 0.0, "episodes/crashed": 0.0}
    stats = episode_stats([make_episode(crashed="x")])
    assert stats == {"episodes/count": 0.0, "episodes/crashed": 1.0}


def test_episode_stats_values_are_plain_floats(cfg):
    stats = episode_stats([_scored(make_episode(success=True), cfg)])
    assert all(isinstance(v, float) for v in stats.values())
