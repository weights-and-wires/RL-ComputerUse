"""Episode reward and logging statistics.

The reward has three separately logged parts, so you can spot reward hacking:

- success: +1.0 when MiniWoB reports success.
- format penalty: -0.1 per format error, capped at -0.3 per episode.
- efficiency bonus: 0.1 * (1 - turns_used / max_turns), only on success.

Why the bonus applies only on success: a bonus for finishing fast on a failed
episode would reward giving up early.
"""

from __future__ import annotations

import math
from collections import defaultdict

from cua.config import Config
from cua.types import Episode, RewardBreakdown

# Same labels as Episode.failure_mode(), minus "crashed" (crashed episodes are
# excluded from the statistics). A fixed list keeps W&B keys stable across steps.
FAILURE_MODES = ("success", "format_error", "ran_out_of_turns", "invalid_action", "wrong_action")


def reward_from_counts(success: bool, num_format_errors: int, turns_used: int, cfg: Config) -> RewardBreakdown:
    """Compute the reward from three numbers. Both trainers call this.

    Args:
        success: Whether MiniWoB reported success.
        num_format_errors: Turns whose response had a format error.
        turns_used: Turns the agent took.
        cfg: Config with `rewards.*` and `env.max_turns`.

    Returns:
        A `RewardBreakdown`. `format_penalty` is zero or negative.
    """
    rewards = cfg.rewards
    max_turns = max(int(cfg.env.max_turns), 1)

    penalty = -min(max(num_format_errors, 0) * float(rewards.format_penalty), float(rewards.format_penalty_cap))

    bonus = 0.0
    if success:
        # Clamp so an out-of-range turns_used can never produce a negative bonus.
        used_fraction = min(max(turns_used / max_turns, 0.0), 1.0)
        bonus = float(rewards.efficiency_coef) * (1.0 - used_fraction)

    return RewardBreakdown(
        success=1.0 if success else 0.0,
        format_penalty=penalty,
        efficiency_bonus=bonus,
    )


def compute_reward(ep: Episode, cfg: Config) -> RewardBreakdown:
    """Compute the reward for a finished episode.

    A crashed episode gets an all-zero breakdown. The trainers drop it from the
    batch, and zeros make sure a stray read of its reward cannot teach anything.
    """
    if ep.crashed:
        return RewardBreakdown()
    return reward_from_counts(ep.success, ep.num_format_errors, ep.num_turns, cfg)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pstdev(values: list[float]) -> float:
    """Population standard deviation (divides by n)."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def group_stats(rewards_by_group: list[list[float]]) -> dict[str, float]:
    """Summarize GRPO groups (all samples for one prompt).

    A group with zero reward variance gives zero advantage to every sample, so
    it teaches nothing. A high `groups/zero_variance_frac` means the task pool
    is too easy or too hard.

    Returns:
        `groups/zero_variance_frac` (groups with std < 1e-6),
        `groups/mean_std`, `groups/mean_reward`, and `groups/count`.
    """
    groups = [g for g in rewards_by_group if g]
    if not groups:
        return {
            "groups/zero_variance_frac": 0.0,
            "groups/mean_std": 0.0,
            "groups/mean_reward": 0.0,
            "groups/count": 0.0,
        }
    stds = [_pstdev(g) for g in groups]
    return {
        "groups/zero_variance_frac": sum(1 for s in stds if s < 1e-6) / len(groups),
        "groups/mean_std": _mean(stds),
        "groups/mean_reward": _mean([_mean(g) for g in groups]),
        "groups/count": float(len(groups)),
    }


def episode_stats(episodes: list[Episode]) -> dict[str, float]:
    """Build a flat, W&B-ready dict of metrics for a batch of episodes.

    Crashed episodes are excluded from every rate and mean and reported only in
    `episodes/crashed`. The failure-mode fractions (`failure/<mode>`) sum to 1
    over the non-crashed episodes.
    """
    valid = [ep for ep in episodes if not ep.crashed]
    stats: dict[str, float] = {
        "episodes/count": float(len(valid)),
        "episodes/crashed": float(len(episodes) - len(valid)),
    }
    if not valid:
        return stats
    n = len(valid)

    # Reward components: reward/total, reward/success, reward/format_penalty, ...
    components: dict[str, list[float]] = defaultdict(list)
    for ep in valid:
        for key, value in ep.reward.to_dict().items():
            components[key].append(value)
    for key, values in components.items():
        stats[key] = _mean(values)

    stats["success/overall"] = sum(ep.success for ep in valid) / n
    per_task: dict[str, list[float]] = defaultdict(list)
    for ep in valid:
        per_task[ep.task].append(float(ep.success))
    for task, values in sorted(per_task.items()):
        stats[f"success/{task}"] = _mean(values)

    total_turns = sum(ep.num_turns for ep in valid)
    total_errors = sum(ep.num_format_errors for ep in valid)
    stats["format_error/rate_per_turn"] = total_errors / total_turns if total_turns else 0.0
    stats["format_error/rate_per_episode"] = sum(1 for ep in valid if ep.num_format_errors) / n
    stats["turns/mean"] = total_turns / n

    modes = [ep.failure_mode() for ep in valid]
    for mode in FAILURE_MODES:
        stats[f"failure/{mode}"] = modes.count(mode) / n
    return stats
