"""Run many episodes at once with turn-level batching.

The problem: a GPU is efficient when it generates many responses in one call,
but each browser step takes about a second, and every episode needs several
steps. If you ran episodes one at a time, the GPU would idle almost all the time.

The solution is to advance all active episodes one turn at a time ("ticks"):

    tick:  1. start queued episodes on free browsers (reset, in parallel)
           2. build every active conversation, call generate() ONCE
           3. parse each response, step all browsers in parallel
           4. finished episodes free their browser for the next queued episode

Each browser is a separate worker process (RemoteEnv), so stepping them from
threads is safe: the threads only wait on pipes, and the Playwright objects
stay in their own processes.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from cua import actions, prompts, rewards
from cua.config import Config, turn_limit
from cua.samples import action_text
from cua.types import (
    CONTEXT_OVERFLOW,
    Episode,
    EpisodeSpec,
    GenerateFn,
    Generation,
    Observation,
    ParsedAction,
    SamplingSpec,
    StepResult,
    Track,
    Turn,
)

__all__ = ["run_episodes", "make_group_specs"]


def make_group_specs(
    tasks: list[str], seeds: list[int], group_size: int, record: bool = False
) -> list[EpisodeSpec]:
    """Expand (task, seed) pairs into GRPO groups.

    `tasks` and `seeds` are parallel lists: group i is `(tasks[i], seeds[i])`,
    repeated `group_size` times with `sample_idx` 0..group_size-1. All members of
    a group share the same task and seed, so they face the identical page and
    only differ in what the model samples. GRPO compares rewards inside a group.
    """
    if len(tasks) != len(seeds):
        raise ValueError(f"tasks ({len(tasks)}) and seeds ({len(seeds)}) must have equal length")
    return [
        EpisodeSpec(task=task, seed=seed, sample_idx=j, record=record)
        for task, seed in zip(tasks, seeds)
        for j in range(group_size)
    ]


@dataclass
class _Slot:
    """Bookkeeping for one running episode."""

    index: int  # position in the input `specs` list, so we can return results in order
    spec: EpisodeSpec
    env: Any  # a RemoteEnv
    episode: Episode
    obs: Observation
    limit: int = 0  # this episode's turn limit (env.max_turns, or the task's budget)
    seg_turns: int = 0  # turns in the current segment (history window only)


def start_new_segment(ep: Episode, obs: Observation, track: Track, extended: bool) -> None:
    """Closes the current conversation and opens the next segment of a windowed episode.

    The finished segment drops its trailing tool message: the model never
    answers it there, and the new segment shows the same screenshot again.
    The new segment opens with a text log of every turn so far. See the
    history-window notes in `cua.prompts`.
    """
    seg = ep.messages  # the same list object the prompts used, so callers can match them
    if seg and seg[-1].get("role") == "tool":
        seg.pop()
    ep.segments.append(seg)
    history = [
        prompts.history_line(i + 1, action_text(t.parsed), t.env_error) for i, t in enumerate(ep.turns)
    ]
    ep.messages = prompts.segment_messages(obs, track, True, history, extended)


def _timed(fn: Callable[..., Any], *args: Any) -> tuple[Any, Optional[Exception]]:
    """Call `fn`, returning (result, None) or (None, exception).

    Env calls can fail (browser crash, timeout). We catch the exception per
    episode so one bad browser never aborts a whole batch of 64 episodes.
    """
    try:
        return fn(*args), None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return None, exc


def _bump(stats: Optional[dict], key: str, value: float) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0.0) + value


def run_episodes(
    specs: list[EpisodeSpec],
    generate: GenerateFn,
    pool: Any,
    cfg: Config,
    sampling: SamplingSpec,
    track: Optional[Track] = None,
    progress: bool = True,
    stats: Optional[dict] = None,
) -> list[Episode]:
    """Run every spec to completion and return the episodes in spec order.

    Args:
        specs: Episodes to run. At most `len(pool.envs)` run at the same time.
        generate: Maps (conversations, sampling, tool schemas) to Generations.
            Pass `VLLMPolicy.generate`, or a fake in tests.
        pool: An EnvPool of RemoteEnvs.
        cfg: Config. Uses `cfg.env.max_turns`, `cfg.env.first_obs_image`,
            `cfg.env.track`, and, when set, `env.turn_budgets` (per-task turn
            limits), `env.context_window_turns` (history window), and
            `env.track_b_extra_tools` (double_click and left_click_drag).
        sampling: Decoding settings for every generate() call.
        track: "A" or "B". Defaults to `cfg.env.track`.
        progress: Show a tqdm progress bar.
        stats: Optional dict that this function fills with timing totals:
            `generation_s`, `reset_s`, `step_s`, `env_s` (reset + step),
            `ticks`, `generate_calls`, and `episodes`. Use it to see whether
            generation or the browsers limit your throughput.

    Episode rules:
        - Success or an env `terminated` flag ends the episode.
        - A response with no tool call ends it (format error, nothing to execute).
        - A tool call with bad arguments sends the error back as the tool result
          and the episode continues.
        - Reaching the turn limit ends it with `truncated=True`.
        - An env exception sets `episode.crashed` and ends it.
        - With `env.context_window_turns = W > 0`, every W turns the episode
          starts a new conversation (segment) that holds a text log of the
          earlier turns and only the newest screenshots. Each prompt is still
          a prefix of its finished segment, so training sees the same tokens.
    """
    track = track or cfg.env.track
    with_image = bool(cfg.env.get("first_obs_image", True))
    extended = track == "B" and actions.extended_tools(cfg)
    window = int(cfg.env.get("context_window_turns", 0) or 0)
    tools = actions.tool_schemas(track, extended)

    results: list[Optional[Episode]] = [None] * len(specs)
    queue: deque[int] = deque(range(len(specs)))
    free_envs = list(pool.envs)
    active: list[_Slot] = []

    bar = _make_bar(len(specs), progress)
    n_success = n_done = 0

    def finish(slot_or_none: Optional[_Slot], ep: Episode, env: Any, index: int) -> None:
        """Compute the reward, store the episode, and free its browser."""
        nonlocal n_success, n_done
        try:
            ep.reward = rewards.compute_reward(ep, cfg)
        except Exception as exc:  # a reward bug must not lose the whole batch
            ep.crashed = ep.crashed or f"reward error: {exc}"
        results[index] = ep
        free_envs.append(env)
        n_done += 1
        n_success += int(ep.success)
        bar.update(1)
        bar.set_postfix(active=len(active), success=f"{n_success}/{n_done}")

    # One thread per browser. Threads suffice because each RemoteEnv call just
    # blocks on a pipe to its own worker process.
    with ThreadPoolExecutor(max_workers=max(1, len(pool.envs))) as executor:
        while queue or active:
            _bump(stats, "ticks", 1)

            # ---- 1. Start queued episodes on free browsers ------------------
            starting: list[tuple[int, EpisodeSpec, Any]] = []
            while queue and free_envs:
                idx = queue.popleft()
                starting.append((idx, specs[idx], free_envs.pop()))
            if starting:
                t0 = time.perf_counter()
                reset_out = list(
                    executor.map(
                        lambda item: _timed(item[2].reset, item[1].task, item[1].seed, item[1].record),
                        starting,
                    )
                )
                _bump(stats, "reset_s", time.perf_counter() - t0)
                _bump(stats, "env_s", time.perf_counter() - t0)
                for (idx, spec, env), (obs, exc) in zip(starting, reset_out):
                    ep = Episode(
                        task=spec.task, seed=spec.seed, track=track, sample_idx=spec.sample_idx
                    )
                    if exc is not None:
                        ep.crashed = f"reset failed: {exc}"
                        finish(None, ep, env, idx)
                        continue
                    ep.messages = prompts.initial_messages(obs, track, with_image=with_image, extended=extended)
                    if spec.record:
                        # Use the full-size reset frame the env kept, so every GIF
                        # frame has the same size as the full-size step frames.
                        first = getattr(env, "reset_frame", None) or obs.screenshot
                        if first is not None:
                            ep.frames.append(first.copy())
                    active.append(_Slot(idx, spec, env, ep, obs, limit=turn_limit(cfg, spec.task)))

            if not active:
                continue  # every started episode crashed on reset; try the queue again

            # ---- 1b. History window: open a new segment where one is due ----
            if window > 0:
                for slot in active:
                    if slot.seg_turns >= window:
                        start_new_segment(slot.episode, slot.obs, track, extended)
                        slot.seg_turns = 0

            # ---- 2. One generate() call for all active episodes -------------
            t0 = time.perf_counter()
            generations: list[Generation] = generate(
                [slot.episode.messages for slot in active], sampling, tools
            )
            _bump(stats, "generation_s", time.perf_counter() - t0)
            _bump(stats, "generate_calls", 1)
            if len(generations) != len(active):
                raise RuntimeError(
                    f"generate() returned {len(generations)} results for {len(active)} prompts"
                )

            # ---- 3. Parse responses and decide what each episode does -------
            to_step: list[tuple[_Slot, ParsedAction, Generation, int]] = []
            still_active: list[_Slot] = []
            for slot, gen in zip(active, generations):
                ep = slot.episode
                if gen.finish_reason == CONTEXT_OVERFLOW:
                    # The conversation no longer fits in vLLM's context. End it
                    # like running out of turns: no reward, no format penalty.
                    ep.truncated = True
                    finish(slot, ep, slot.env, slot.index)
                    continue
                parsed = actions.parse_response(gen.text, track, slot.obs.screen_size, extended)
                prompt_len = len(ep.messages)  # messages[:prompt_len] is this turn's input
                segment = len(ep.segments)
                slot.seg_turns += 1

                if parsed.ends_episode:
                    # No tool call at all: nothing to run, and the episode ends.
                    # We still store the reply so RFT/GRPO can see what the model said.
                    ep.turns.append(
                        Turn(len(ep.turns) + 1, slot.obs, prompt_len, gen, parsed,
                             env_error=parsed.format_error or "", segment=segment)
                    )
                    prompts.append_turn(ep.messages, gen.text, None, "")
                    _hold_frame(ep, slot.spec)
                    finish(slot, ep, slot.env, slot.index)
                elif parsed.action is None:
                    # A tool call exists but is invalid (unknown tool, bad args).
                    # Tell the model, using the same text style TRL uses for tool
                    # errors, and let it try again. No browser step happens.
                    error_text = str({"error": parsed.format_error or "invalid tool call"})
                    ep.turns.append(
                        Turn(len(ep.turns) + 1, slot.obs, prompt_len, gen, parsed,
                             env_error=parsed.format_error or "", segment=segment)
                    )
                    content = prompts.tool_result_content(slot.obs, track, error=error_text)
                    prompts.append_turn(ep.messages, gen.text, content, parsed.tool_name or "unknown")
                    _hold_frame(ep, slot.spec)
                    if len(ep.turns) >= slot.limit:
                        ep.truncated = True
                        finish(slot, ep, slot.env, slot.index)
                    else:
                        still_active.append(slot)
                else:
                    to_step.append((slot, parsed, gen, prompt_len, segment))
                    still_active.append(slot)

            # ---- 4. Step all browsers in parallel ---------------------------
            if to_step:
                t0 = time.perf_counter()
                step_out = list(
                    executor.map(lambda item: _timed(item[0].env.step, item[1].action), to_step)
                )
                _bump(stats, "step_s", time.perf_counter() - t0)
                _bump(stats, "env_s", time.perf_counter() - t0)

                for (slot, parsed, gen, prompt_len, segment), (res, exc) in zip(to_step, step_out):
                    ep = slot.episode
                    if exc is not None:
                        ep.crashed = f"step failed: {exc}"
                        ep.turns.append(
                            Turn(len(ep.turns) + 1, slot.obs, prompt_len, gen, parsed,
                                 env_error=ep.crashed, segment=segment)
                        )
                        still_active.remove(slot)
                        finish(slot, ep, slot.env, slot.index)
                        continue

                    res: StepResult
                    new_obs = res.observation
                    ep.turns.append(
                        Turn(len(ep.turns) + 1, slot.obs, prompt_len, gen, parsed,
                             env_error=new_obs.last_action_error, segment=segment)
                    )
                    content = prompts.tool_result_content(
                        new_obs, track, error=new_obs.last_action_error
                    )
                    prompts.append_turn(ep.messages, gen.text, content, parsed.tool_name)
                    ep.success = bool(res.success)
                    ep.raw_env_reward = float(res.raw_env_reward)
                    ep.terminated = bool(res.terminated)
                    if slot.spec.record:
                        frame = res.frame if res.frame is not None else new_obs.screenshot
                        if frame is not None:
                            ep.frames.append(frame.copy())
                    slot.obs = new_obs

                    if ep.success or ep.terminated:
                        still_active.remove(slot)
                        finish(slot, ep, slot.env, slot.index)
                    elif len(ep.turns) >= slot.limit:
                        ep.truncated = True
                        still_active.remove(slot)
                        finish(slot, ep, slot.env, slot.index)

            active = still_active
            bar.set_postfix(active=len(active), success=f"{n_success}/{n_done}")

    bar.close()
    _bump(stats, "episodes", len(specs))
    return [ep for ep in results if ep is not None]


def _hold_frame(ep: Episode, spec: EpisodeSpec) -> None:
    """Repeat the last frame for a turn that did not change the page.

    Keeping `len(ep.frames) == len(ep.turns) + 1` lets the GIF writer pair
    frame i+1 with turn i without any bookkeeping.
    """
    if spec.record and ep.frames:
        ep.frames.append(ep.frames[-1].copy())


class _NullBar:
    """Stand-in for tqdm when progress is off or tqdm is missing."""

    def update(self, n: int = 1) -> None: ...
    def set_postfix(self, **kwargs: Any) -> None: ...
    def close(self) -> None: ...


def _make_bar(total: int, progress: bool) -> Any:
    if not progress:
        return _NullBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullBar()
    return tqdm(total=total, desc="episodes", unit="ep", dynamic_ncols=True)
