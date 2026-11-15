"""Choosing which (task, seed) prompts to roll out: round-robin or adaptive.

GRPO learns only from groups whose episodes have different outcomes. For a task
with success rate p and G episodes per group, the chance that a group is mixed
(at least one success and at least one failure) is:

    P(mixed) = 1 - p^G - (1 - p)^G

With G = 8 that is 99% at p = 0.5, 57% at p = 0.1 or 0.9, and only 15% at
p = 0.98. Run 1 rotated through tasks round-robin, so a task that was already
mastered took the same share of rollouts as one in the useful middle band, and
about 77% of all groups taught nothing.

`AdaptiveTaskSampler` keeps a running estimate of each task's success rate and
samples tasks with weight P(mixed), with a small floor so mastered and unsolved
tasks still get an occasional rollout (enough to notice when one changes).

Both samplers are pure Python with no I/O. Their whole state fits in a small
dict, so a checkpoint resumes them exactly.
"""

from __future__ import annotations

import random
from typing import Optional


def mixed_group_prob(p: float, group_size: int) -> float:
    """Chance that a group of `group_size` episodes has at least one success and one failure."""
    p = min(max(p, 0.0), 1.0)
    return 1.0 - p**group_size - (1.0 - p) ** group_size


class RoundRobinSampler:
    """Run 1's sampler: the k-th prompt is task k % n with seed index k // n.

    Every task gets an equal share and a (task, seed) pair is never reused.
    """

    def __init__(self, tasks: list[str], seed_start: int):
        self.tasks = list(tasks)
        self.seed_start = seed_start
        self.pos = 0

    def sample(self, n: int) -> list[tuple[str, int]]:
        k0, self.pos = self.pos, self.pos + n
        return [(self.tasks[k % len(self.tasks)], self.seed_start + k // len(self.tasks)) for k in range(k0, k0 + n)]

    def update(self, task: str, outcomes: list[bool]) -> None:
        """Round-robin ignores outcomes."""

    def weights(self) -> dict[str, float]:
        return {t: 1.0 / len(self.tasks) for t in self.tasks}

    def rates(self) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict:
        return {"kind": "round_robin", "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"])


class AdaptiveTaskSampler:
    """Samples tasks in proportion to how likely their groups are to be informative.

    Args:
      tasks: The training pool.
      seed_start: First training seed. Each task counts its own seeds up from
        here, so a (task, seed) pair is never reused.
      group_size: G, the episodes per group. Used in the weight formula.
      init_rates: Starting success-rate estimate per task, for example the
        sampled-baseline rates. Missing tasks start at 0.5.
      ema_alpha: Weight of the newest group in the running estimate. With 0.2,
        the estimate mostly reflects the last 5 groups (about 40 episodes).
      floor: Smallest weight a task can have, so it still gets sampled now and then.
      rng: Random source. Pass one seeded `random.Random` for reproducibility.
    """

    def __init__(
        self,
        tasks: list[str],
        seed_start: int,
        group_size: int,
        init_rates: Optional[dict[str, float]] = None,
        ema_alpha: float = 0.2,
        floor: float = 0.05,
        rng: Optional[random.Random] = None,
    ):
        if not tasks:
            raise ValueError("the task pool is empty")
        self.tasks = list(tasks)
        self.seed_start = seed_start
        self.group_size = group_size
        self.ema_alpha = ema_alpha
        self.floor = floor
        self.rng = rng or random.Random(0)
        init_rates = init_rates or {}
        self.rate = {t: float(init_rates.get(t, 0.5)) for t in self.tasks}
        self.next_seed = {t: 0 for t in self.tasks}
        self.groups_seen = {t: 0 for t in self.tasks}

    def weights(self) -> dict[str, float]:
        """Unnormalized sampling weight per task: max(P(mixed), floor)."""
        return {t: max(mixed_group_prob(self.rate[t], self.group_size), self.floor) for t in self.tasks}

    def rates(self) -> dict[str, float]:
        return dict(self.rate)

    def sample(self, n: int) -> list[tuple[str, int]]:
        """Returns n (task, seed) prompts.

        Tasks are drawn by weight without replacement, so one call spreads over
        as many different tasks as it can. Only when n exceeds the pool size
        does a task appear twice (with different seeds).
        """
        out: list[tuple[str, int]] = []
        w = self.weights()
        while len(out) < n:
            # Weighted sampling without replacement (Efraimidis-Spirakis): sort
            # by u^(1/w) with u uniform in (0, 1), and take the largest keys.
            keys = sorted(self.tasks, key=lambda t: self.rng.random() ** (1.0 / w[t]), reverse=True)
            for task in keys[: n - len(out)]:
                out.append((task, self.seed_start + self.next_seed[task]))
                self.next_seed[task] += 1
        return out

    def update(self, task: str, outcomes: list[bool]) -> None:
        """Folds one group's outcomes (success per episode) into the task's estimate."""
        if task not in self.rate or not outcomes:
            return
        group_rate = sum(outcomes) / len(outcomes)
        self.rate[task] = (1.0 - self.ema_alpha) * self.rate[task] + self.ema_alpha * group_rate
        self.groups_seen[task] += 1

    def state_dict(self) -> dict:
        return {
            "kind": "adaptive",
            "rate": dict(self.rate),
            "next_seed": dict(self.next_seed),
            "groups_seen": dict(self.groups_seen),
            "rng": _rng_state_to_json(self.rng.getstate()),
        }

    def load_state_dict(self, state: dict) -> None:
        # Only tasks in the current pool are restored; a changed pool keeps defaults for new tasks.
        for t in self.tasks:
            if t in state["rate"]:
                self.rate[t] = float(state["rate"][t])
                self.next_seed[t] = int(state["next_seed"][t])
                self.groups_seen[t] = int(state.get("groups_seen", {}).get(t, 0))
        if state.get("rng") is not None:
            self.rng.setstate(_rng_state_from_json(state["rng"]))


def _rng_state_to_json(state: tuple) -> list:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_state_from_json(state: list) -> tuple:
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def make_sampler(kind: str, tasks: list[str], seed_start: int, group_size: int, **adaptive_kwargs):
    """Builds the sampler named by `grpo.sampler` in the config."""
    if kind == "round_robin":
        return RoundRobinSampler(tasks, seed_start)
    if kind == "adaptive":
        return AdaptiveTaskSampler(tasks, seed_start, group_size, **adaptive_kwargs)
    raise ValueError(f"grpo.sampler must be 'round_robin' or 'adaptive', got {kind!r}")
