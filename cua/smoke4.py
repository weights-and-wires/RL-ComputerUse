"""Fail-fast checks for run 4's new mechanisms, before any GPU time goes to training.

    python -m cua.smoke4 browsers --config configs/grpo_run4.yaml --size 64 --out results/run4/pool_size
    python -m cua.smoke4 actions  --config configs/grpo_run4.yaml
    python -m cua.smoke4 context  --config configs/grpo_run4.yaml

`scripts/run4.sh` runs all three in its `smoke` phase. None of them needs a
GPU or a model: a scripted policy stands in for vLLM.

- `browsers` starts `--size` browsers at once, resets and steps each, and
  halves the size once if more than 5% crash or a step takes over 20 s. It
  writes the size that passed to `--out`, and every later phase uses it.
- `actions` runs left_click_drag, double_click, and mouse_click in real
  browsers on a drag task and checks that BrowserGym runs each without an
  error.
- `context` runs a 12-turn scripted episode with the history window and
  checks that every prompt the rollout sent is an exact prefix of the
  conversation the trainer scores, in rendered text and in token IDs, and
  that no prompt holds more screenshots than the window allows.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from cua import actions, prompts
from cua.config import Config, add_config_args, config_from_args
from cua.types import Generation, SamplingSpec


def tool_call(name: str, **arguments: Any) -> str:
    """A model response with one tool call, in the format the parser reads."""
    return f"Acting.\n<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


class ScriptedPolicy:
    """Stands in for VLLMPolicy.generate: turn k of every episode answers `script[k]`.

    With a processor, it also renders every prompt the way VLLMPolicy does and
    keeps the text, so `check_prefixes` can compare it with training's view.
    """

    def __init__(self, script: list[str], processor: Any = None):
        self.script = script
        self.processor = processor
        self.prompts: list[tuple[list[dict], int, str]] = []  # (conversation, prompt_len, rendered text)

    def generate(self, batch: list[list[dict]], sampling: SamplingSpec, tools: list[dict]) -> list[Generation]:
        out = []
        for messages in batch:
            k = sum(1 for m in messages if m.get("role") == "assistant")
            # The turn index across segments: earlier segments' turns are in the history log.
            log = next((b.get("text", "").strip() for b in messages[1].get("content", [])
                        if isinstance(b, dict) and b.get("text", "").strip().startswith(prompts.HISTORY_HEADER)), "")
            k += max(0, len(log.splitlines()) - 1) if log else 0
            text = self.script[min(k, len(self.script) - 1)]
            if self.processor is not None:
                from cua.policy import render_prompt

                self.prompts.append((messages, len(messages), render_prompt(self.processor, messages, tools, True)))
            out.append(Generation(text=text, token_ids=[], logprobs=[], finish_reason="stop"))
        return out


def check_prefixes(policy: ScriptedPolicy, episodes: list, processor: Any, tools: list[dict],
                   max_images: Optional[int] = None) -> list[str]:
    """Problems where a rollout prompt isn't an exact prefix of what training scores.

    Training (nano_grpo.loop.train_phase) tokenizes each finished conversation
    with `tokenize_episode`. For every recorded prompt, the rendered prompt
    plus the response plus `<|im_end|>` must start the rendered conversation,
    and the token IDs must match position by position. Every turn of every
    episode must have been checked this way.
    """
    from cua.policy import collect_images, render_prompt

    problems: list[str] = []
    tok = processor.tokenizer
    finished = {}
    for ep in episodes:
        for conv in ep.conversations():
            finished[id(conv)] = (conv, render_prompt(processor, conv, tools, add_generation_prompt=False))
    checked = 0
    for messages, n, text in policy.prompts:
        if id(messages) not in finished:
            problems.append(f"a prompt of {n} messages belongs to no finished conversation")
            continue
        conv, full = finished[id(messages)]
        if n >= len(conv) or conv[n].get("role") != "assistant":
            problems.append(f"prompt of {n} messages has no assistant reply after it")
            continue
        expect = text + conv[n]["content"] + "<|im_end|>"
        if not full.startswith(expect):
            problems.append(f"prompt of {n} messages isn't a text prefix of its conversation")
            continue
        a = tok(expect, add_special_tokens=False)["input_ids"]
        b = tok(full, add_special_tokens=False)["input_ids"]
        if b[: len(a)] != a:
            problems.append(f"prompt of {n} messages isn't a token prefix of its conversation")
        n_images = len(collect_images(conv[:n]))
        if max_images is not None and n_images > max_images:
            problems.append(f"prompt of {n} messages holds {n_images} screenshots, more than {max_images}")
        checked += 1
    want = sum(len(ep.turns) for ep in episodes)
    if checked != want:
        problems.append(f"checked {checked} prompts for {want} turns")
    return problems


def run_context_check(cfg: Config, pool: Any, processor: Any, task: str = "click-button",
                      turns: int = 12, window: int = 4, tokenize: bool = True) -> list[str]:
    """Runs one scripted `turns`-turn episode with a window of `window` turns and checks it.

    Set `tokenize=False` when `processor` can render chat text but can't
    process images (the laptop tests use a tokenizer-only stand-in).
    Returns a list of problems (empty when everything holds).
    """
    from cua import rollout
    from cua.types import EpisodeSpec

    cfg = Config(json.loads(json.dumps(cfg, default=str)))
    cfg.env["max_turns"] = turns
    cfg.env["context_window_turns"] = window
    cfg.env["turn_budgets"] = None
    extended = actions.extended_tools(cfg)
    tools = actions.tool_schemas(cfg.env.track, extended)
    # noop and keyboard_press keep the page unchanged, so the episode runs all turns.
    script = [tool_call("noop") if k % 2 else tool_call("keyboard_press", key="Shift") for k in range(turns)]
    policy = ScriptedPolicy(script, processor)
    eps = rollout.run_episodes([EpisodeSpec(task, 0)], policy.generate, pool, cfg, SamplingSpec(), progress=False)
    problems = []
    ep = eps[0]
    if ep.crashed:
        return [f"the scripted episode crashed: {ep.crashed}"]
    want_segments = (len(ep.turns) + window - 1) // window
    if len(ep.conversations()) != want_segments:
        problems.append(f"{len(ep.turns)} turns gave {len(ep.conversations())} conversations, want {want_segments}")
    if len(ep.turns) != turns and not ep.success:
        problems.append(f"the episode took {len(ep.turns)} turns, want {turns}")
    for ci, conv in enumerate(ep.conversations()):
        n_assist = sum(1 for m in conv if m.get("role") == "assistant")
        if n_assist != len(ep.turns_in(ci)):
            problems.append(f"segment {ci} has {n_assist} assistant messages for {len(ep.turns_in(ci))} turns")
        if ci > 0:
            log = conv[1]["content"][1]["text"].strip()
            if not log.startswith(prompts.HISTORY_HEADER) or len(log.splitlines()) != 1 + ci * window:
                problems.append(f"segment {ci} has a wrong history log: {log[:120]!r}")
    # A segment's prompt holds its opening screenshot plus one per earlier turn in it.
    problems += check_prefixes(policy, eps, processor, tools, max_images=window)
    if tokenize:
        from cua.policy import tokenize_episode

        for ci, conv in enumerate(ep.conversations()):
            enc = tokenize_episode(processor, conv, tools)
            n_spans = _count_spans(enc["assistant_mask"][0].tolist())
            if n_spans != len(ep.turns_in(ci)):
                problems.append(f"segment {ci}: {n_spans} assistant spans for {len(ep.turns_in(ci))} turns")
    return problems


def _count_spans(mask_row: list[int]) -> int:
    return sum(1 for i, m in enumerate(mask_row) if m and (i == 0 or not mask_row[i - 1]))


def run_actions_check(cfg: Config, pool: Any) -> list[str]:
    """Runs drag, double-click, and click in real browsers; returns problems."""
    from cua import rollout
    from cua.types import EpisodeSpec

    cfg = Config(json.loads(json.dumps(cfg, default=str)))
    cfg.env["track_b_extra_tools"] = True
    cfg.env["turn_budgets"] = None
    cfg.env["max_turns"] = 3
    script = [
        tool_call("left_click_drag", start_coordinate=[150, 500], end_coordinate=[700, 520]),
        tool_call("double_click", coordinate=[500, 500]),
        tool_call("mouse_click", coordinate=[500, 500]),
    ]
    policy = ScriptedPolicy(script)
    specs = [EpisodeSpec(task, 0) for task in ("use-slider", "drag-box")]
    eps = rollout.run_episodes(specs, policy.generate, pool, cfg, SamplingSpec(), progress=False)
    problems = []
    want = ("mouse_drag_and_drop(", "mouse_dblclick(", "mouse_click(")
    for ep in eps:
        if ep.crashed:
            problems.append(f"{ep.task}: crashed ({ep.crashed[:200]})")
            continue
        for turn, prefix in zip(ep.turns, want):
            action = turn.parsed.action or ""
            if not action.startswith(prefix):
                problems.append(f"{ep.task} turn {turn.step}: parsed {action!r}, want {prefix}...")
            if turn.env_error:
                problems.append(f"{ep.task} turn {turn.step}: BrowserGym error {turn.env_error[:200]!r}")
        print(f"{ep.task}: {[t.parsed.action for t in ep.turns]} success={ep.success}")
    return problems


def stress_browsers(cfg: Config, size: int, make_pool: Callable[[Config, int], Any]) -> dict:
    """Resets and steps `size` browsers at once. Returns counts and the slowest call."""
    t0 = time.perf_counter()
    crashes, slowest = 0, 0.0
    with make_pool(cfg, size) as pool:
        def one(env) -> tuple[bool, float]:
            start = time.perf_counter()
            try:
                env.reset("click-button", 0)
                env.step("noop()")
                return True, time.perf_counter() - start
            except Exception:  # noqa: BLE001 - counted, not raised
                return False, time.perf_counter() - start

        with ThreadPoolExecutor(max_workers=size) as ex:
            for _round in range(2):  # the second round measures warm browsers
                results = list(ex.map(one, pool.envs))
                crashes += sum(1 for ok, _ in results if not ok)
                if _round == 1:
                    slowest = max(dt for _, dt in results)
    return {"size": size, "crashes": crashes, "slowest_warm_s": round(slowest, 1),
            "total_s": round(time.perf_counter() - t0, 1)}


def choose_pool_size(cfg: Config, want: int, make_pool: Callable[[Config, int], Any],
                     max_crash_frac: float = 0.05, max_step_s: float = 20.0) -> tuple[int, list[dict]]:
    """Tries `want` browsers, then half once. Returns (size that passed or 0, reports)."""
    reports = []
    for size in (want, max(1, want // 2)):
        rep = stress_browsers(cfg, size, make_pool)
        rep["ok"] = rep["crashes"] <= max_crash_frac * 2 * size and rep["slowest_warm_s"] <= max_step_s
        reports.append(rep)
        print(f"browsers: {rep}")
        if rep["ok"]:
            return size, reports
        if size == 1:
            break
    return 0, reports


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run 4's fail-fast checks (no GPU needed).")
    add_config_args(parser, "configs/grpo_run4.yaml")
    parser.add_argument("check", choices=("browsers", "actions", "context"))
    parser.add_argument("--size", type=int, default=64, help="browsers: how many to try first.")
    parser.add_argument("--out", default=None, help="browsers: file that receives the size that passed.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    from cua.env import EnvPool

    if args.check == "browsers":
        size, reports = choose_pool_size(cfg, args.size, lambda c, n: EnvPool(c, size=n))
        if args.out:
            Path(args.out).write_text(f"{size}\n")
            Path(args.out).with_suffix(".json").write_text(json.dumps(reports, indent=2))
        if not size:
            print("FAILED: even half the browsers crash or stall. Set CUA_POOL_SIZE lower.")
            return 1
        print(f"browsers OK: {size}")
        return 0

    with EnvPool(cfg, size=2) as pool:
        if args.check == "actions":
            problems = run_actions_check(cfg, pool)
        else:
            from transformers import AutoProcessor

            from cua.policy import _configure_processor

            processor = AutoProcessor.from_pretrained(cfg.model.name)
            _configure_processor(processor, cfg)
            problems = run_context_check(cfg, pool, processor,
                                         window=int(cfg.env.get("context_window_turns", 0) or 4))
    for p in problems:
        print(f"PROBLEM: {p}")
    print(f"{args.check}: {'OK' if not problems else 'FAILED'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
