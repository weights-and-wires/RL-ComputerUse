"""Prompt and chat-message construction.

Every code path (eval, RFT data collection, nano GRPO, and the TRL environment)
builds the same conversation through this module, so the model sees one format:

    system   our instructions (the chat template adds the tool schemas)
    user     [image (first screenshot, optional), text (goal and page text)]
    assistant  raw model text, verbatim, including its <tool_call> block
    tool     [image (new screenshot), text (error, goal, page text)]
    assistant
    tool
    ...

Why the assistant message holds the raw text and not structured `tool_calls`:
re-rendering the conversation then reproduces the exact tokens the model
sampled, which RL training needs for correct log-probs.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

from cua.types import Observation, Track

_COMMON = (
    "You operate a web page to complete the task the user gives you. "
    "Each turn you see the current page and the task goal.\n\n"
    "On every turn, write at most one or two short sentences of reasoning, "
    "then call exactly one tool. Write nothing after the tool call.\n\n"
)

_TRACK_A = (
    "Refer to page elements by their bid: the number in square brackets before "
    "an element in the page text. For the line [12] button 'Submit', call click "
    'with the bid "12". Pass the bid as a string, without brackets.\n\n'
)

_TRACK_B = (
    "You see only a screenshot. Give positions as coordinates normalized to the "
    "range 0-1000 on the screenshot: x grows to the right and y grows downward, "
    "so (0, 0) is the top-left corner and (1000, 1000) is the bottom-right "
    "corner. To click, call mouse_click with coordinate [x, y]. To type into a field, click it first, then call keyboard_type.\n\n"
)

# Run 4 (env.track_b_extra_tools): the two extra Track B tools.
_TRACK_B_EXTRA = (
    "To double-click, call double_click with coordinate [x, y]. To drag an item, "
    "move a slider, or select text, call left_click_drag with start_coordinate and "
    "end_coordinate.\n\n"
)

_CLOSING = (
    "If an action fails, the next message starts with the error. Read it and "
    "try a different action. The episode ends when the task is complete. You "
    "have a limited number of turns, so act directly."
)


def system_prompt(track: Track, extended: bool = False) -> str:
    """Return the system prompt for a track.

    The prompt stays short because the chat template already adds the full tool
    schemas, and the whole system prompt is repeated in every training sample.

    Args:
        track: "A" or "B".
        extended: Describe the extra Track B tools (run 4). Ignored on Track A.
    """
    if track not in ("A", "B"):
        raise ValueError(f"Unknown track {track!r}. Use 'A' or 'B'.")
    if track == "A":
        return _COMMON + _TRACK_A + _CLOSING
    return _COMMON + _TRACK_B + (_TRACK_B_EXTRA if extended else "") + _CLOSING


def observation_text(obs: Observation, track: Track) -> str:
    """Format the text part of an observation.

    Track A returns the goal, the last error (only if there is one), and the
    AXTree. Track B omits the AXTree, so the model must rely on the screenshot.
    """
    parts = [f"Goal: {obs.goal}"]
    if obs.last_action_error:
        parts.append(f"Last action error: {obs.last_action_error}")
    if track == "A":
        parts.append("Page (accessibility tree):\n" + (obs.axtree or "(empty)"))
    return "\n\n".join(parts)


def _content_blocks(obs: Observation, track: Track, include_image: bool) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if include_image and obs.screenshot is not None:
        blocks.append({"type": "image", "image": obs.screenshot})
    blocks.append({"type": "text", "text": observation_text(obs, track)})
    return blocks


# The opening user message starts with this sentence, followed by the first
# observation blocks. The TRL path puts this sentence in its dataset prompt and
# TRL appends reset()'s blocks after it, so both paths render identical text.
USER_PREAMBLE = "Complete the task in the browser. The page follows."


def first_observation_blocks(obs: Observation, track: Track, with_image: bool) -> list[dict[str, Any]]:
    """Content blocks for the first observation (what TRL's reset() returns).

    Always a list, even without an image, so TRL takes the same code path
    (append blocks to the user message) in both settings.
    """
    return _content_blocks(obs, track, with_image)


def initial_messages(
    obs: Observation, track: Track, with_image: bool, extended: bool = False
) -> list[dict[str, Any]]:
    """Build `[system, user]` for the first turn.

    Args:
        obs: The observation from `env.reset()`.
        track: "A" or "B".
        with_image: Include the first screenshot in the user message. Pass
            `cfg.env.first_obs_image` so training and eval agree.
        extended: Describe the extra Track B tools (run 4).
    """
    return [
        {"role": "system", "content": system_prompt(track, extended)},
        {
            "role": "user",
            "content": [{"type": "text", "text": USER_PREAMBLE}]
            + first_observation_blocks(obs, track, with_image),
        },
    ]


# ---------------------------------------------------------------------------
# History window (run 4: env.context_window_turns)
# ---------------------------------------------------------------------------
#
# With a window of W turns, an episode is a chain of *segments*. Each segment
# is a complete conversation of its own:
#
#     system   the same system prompt
#     user     preamble, a text log of every earlier turn, then the current
#              observation (screenshot and goal)
#     assistant / tool / assistant / tool ...   at most W turns
#
# At turn t of a segment, the prompt is the segment's messages up to that
# turn, so it is always a prefix of the finished segment. Training scores each
# segment as one sequence, so the trainer sees exactly the tokens the sampler
# saw, and vLLM's prefix cache still hits inside a segment. The model sees
# between 1 and W screenshots, never more.

HISTORY_HEADER = "Earlier actions in this episode, oldest first:"
_MAX_ARG_CHARS = 80


def history_line(index: int, action: str, error: str = "") -> str:
    """One line of the earlier-actions log, for example `3. mouse_click(coordinate=[412, 230])`.

    Args:
        index: The 1-based turn number.
        action: The tool call as text (`cua.samples.action_text`), or empty
            when the response had no valid tool call.
        error: The error that turn produced, if any.
    """
    text = action or "(no valid tool call)"
    if len(text) > _MAX_ARG_CHARS:
        text = text[: _MAX_ARG_CHARS - 3] + "..."
    if error:
        err = " ".join(str(error).split())
        text += f" -> error: {err[:_MAX_ARG_CHARS]}"
    return f"{index}. {text}"


def segment_messages(
    obs: Observation, track: Track, with_image: bool, history: list[str], extended: bool = False
) -> list[dict[str, Any]]:
    """Build `[system, user]` that opens a new segment of a windowed episode.

    Args:
        obs: The current observation (the result of the last action).
        track: "A" or "B".
        with_image: Include the screenshot. A new segment always needs it,
            because the earlier screenshots are gone, so callers pass True.
        history: Lines from `history_line`, one per earlier turn.
        extended: Describe the extra Track B tools (run 4).
    """
    # The chat template joins text blocks without a separator, so the log
    # carries its own blank lines.
    log = "\n\n" + HISTORY_HEADER + "\n" + "\n".join(history) + "\n\n" if history else ""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": USER_PREAMBLE}]
    if log:
        blocks.append({"type": "text", "text": log})
    return [
        {"role": "system", "content": system_prompt(track, extended)},
        {"role": "user", "content": blocks + first_observation_blocks(obs, track, with_image)},
    ]


def tool_result_content(obs: Observation, track: Track, error: str = "") -> list[dict[str, Any]]:
    """Build the content of a `tool` message: new screenshot plus text.

    The TRL environment tools and `cua.rollout` both call this, so the model
    sees identical tool results in training and evaluation.

    Args:
        obs: The observation after the action (or the unchanged one if the
            action never ran).
        track: "A" or "B".
        error: An extra error message, such as a tool-argument error from the
            parser. It is combined with `obs.last_action_error` when both exist.
    """
    messages = [m for m in (error, obs.last_action_error) if m]
    merged = "; ".join(dict.fromkeys(messages))  # Drop duplicates, keep order.
    obs = dataclasses.replace(obs, last_action_error=merged)
    return _content_blocks(obs, track, include_image=True)


def append_turn(
    messages: list[dict[str, Any]],
    response_text: str,
    tool_content: Optional[list[dict[str, Any]]],
    tool_name: str,
) -> None:
    """Append one assistant message and, if given, its tool result, in place.

    Args:
        messages: The conversation so far.
        response_text: The raw model text, stored verbatim.
        tool_content: The output of `tool_result_content`, or None when the
            episode ended and no tool result follows.
        tool_name: The tool the model called. The chat template prints it in
            the tool message.
    """
    messages.append({"role": "assistant", "content": response_text})
    if tool_content is not None:
        messages.append({"role": "tool", "name": tool_name, "content": tool_content})


def estimate_tokens(
    messages: list[dict[str, Any]],
    chars_per_token: float = 3.5,
    pixels_per_token: int = 1024,
    per_message_overhead: int = 4,
) -> int:
    """Estimate the prompt length in tokens without loading a tokenizer.

    This is a rough budget check for tests, not an exact count. Text costs
    `len / chars_per_token`. An image costs `width * height / pixels_per_token + 2`,
    where Qwen3-VL uses one token per 32x32 pixels and 2 tokens mark the image
    boundaries. The estimate excludes the tool schemas that the chat template
    adds, so keep some headroom.
    """
    total = 0.0
    for message in messages:
        total += per_message_overhead
        content = message.get("content")
        if isinstance(content, str):
            total += len(content) / chars_per_token
            continue
        for block in content or []:
            if block.get("type") == "image":
                image = block.get("image")
                width, height = getattr(image, "size", (0, 0))
                total += width * height / pixels_per_token + 2
            else:
                total += len(block.get("text", "")) / chars_per_token
    return math.ceil(total)
