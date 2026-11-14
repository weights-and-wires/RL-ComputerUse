"""Tests for cua.prompts and the pure helpers in cua.env."""

from __future__ import annotations

import pytest
from PIL import Image

from cua.env import (
    TRUNCATION_MARKER,
    format_observation,
    normalize_miniwob_url,
    resize_long_side,
    truncate_axtree,
)
from cua.prompts import (
    append_turn,
    estimate_tokens,
    USER_PREAMBLE,
    initial_messages,
    observation_text,
    system_prompt,
    tool_result_content,
)

# -- system prompt -------------------------------------------------------------


@pytest.mark.parametrize("track", ["A", "B"])
def test_system_prompt_is_concise(track):
    assert len(system_prompt(track).split()) <= 180


def test_system_prompt_per_track():
    a, b = system_prompt("A"), system_prompt("B")
    assert "bid" in a and "square brackets" in a
    assert "0-1000" in b and "bid" not in b
    assert "exactly one tool" in a and "exactly one tool" in b
    with pytest.raises(ValueError):
        system_prompt("C")  # type: ignore[arg-type]


# -- observation text ----------------------------------------------------------


def test_observation_text_track_a(make_obs):
    text = observation_text(make_obs(), "A")
    assert text == "Goal: Click on the \"OK\" button.\n\nPage (accessibility tree):\n[1] button 'OK'"


def test_observation_text_includes_error_only_when_present(make_obs):
    assert "Last action error" not in observation_text(make_obs(), "A")
    text = observation_text(make_obs(error="TimeoutError: element not found"), "A")
    assert "Last action error: TimeoutError: element not found" in text
    # The error sits between the goal and the page text.
    assert text.index("Goal") < text.index("Last action error") < text.index("Page (accessibility tree)")


def test_observation_text_track_b_has_no_axtree(make_obs):
    text = observation_text(make_obs(), "B")
    assert text.startswith("Goal:")
    assert "accessibility" not in text and "[1] button" not in text


def test_observation_text_empty_axtree(make_obs):
    assert "(empty)" in observation_text(make_obs(axtree=""), "A")


# -- message shapes ------------------------------------------------------------


def test_tool_result_content_blocks(make_obs):
    blocks = tool_result_content(make_obs(), "A")
    assert [b["type"] for b in blocks] == ["image", "text"]
    assert isinstance(blocks[0]["image"], Image.Image)
    assert blocks[1]["text"].startswith("Goal:")


def test_tool_result_content_without_screenshot(make_obs):
    blocks = tool_result_content(make_obs(with_image=False), "A")
    assert [b["type"] for b in blocks] == ["text"]


def test_tool_result_content_merges_errors_without_mutating_obs(make_obs):
    obs = make_obs(error="env says no")
    text = tool_result_content(obs, "A", error="bad bid")[1]["text"]
    assert "bad bid" in text and "env says no" in text
    assert obs.last_action_error == "env says no"
    # The same message twice appears once.
    text = tool_result_content(make_obs(error="same"), "A", error="same")[1]["text"]
    assert text.count("same") == 1


def test_initial_messages_with_image(obs):
    messages = initial_messages(obs, "A", with_image=True)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert isinstance(messages[0]["content"], str)
    # The preamble text first (it matches the TRL dataset prompt), then the
    # first observation: screenshot and page text.
    assert [b["type"] for b in messages[1]["content"]] == ["text", "image", "text"]
    assert messages[1]["content"][0]["text"] == USER_PREAMBLE


def test_initial_messages_without_image(obs):
    messages = initial_messages(obs, "A", with_image=False)
    assert [b["type"] for b in messages[1]["content"]] == ["text", "text"]


def test_append_turn_shapes(obs):
    messages = initial_messages(obs, "A", with_image=True)
    content = tool_result_content(obs, "A")
    append_turn(messages, "Click it.\n<tool_call>...</tool_call>", content, "click")
    append_turn(messages, "Done.", None, "")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == "Click it.\n<tool_call>...</tool_call>"  # raw text, verbatim
    assert messages[3]["name"] == "click"
    assert messages[3]["content"] is content
    assert "name" not in messages[4]


# -- AXTree truncation ---------------------------------------------------------


def test_truncate_short_text_is_unchanged():
    assert truncate_axtree("[1] button 'OK'", 100) == "[1] button 'OK'"


def test_truncate_long_text_fits_and_is_marked():
    text = "\n".join(f"[{i}] link 'Item {i}'" for i in range(500))
    out = truncate_axtree(text, 2000)
    assert len(out) <= 2000
    assert out.endswith(TRUNCATION_MARKER)
    # The cut falls on a line boundary: every kept line is a complete original line.
    kept = out.splitlines()[:-1]
    assert all(line in text.splitlines() for line in kept)


def test_truncate_without_newlines():
    out = truncate_axtree("x" * 5000, 100)
    assert len(out) == 100 and out.endswith(TRUNCATION_MARKER)


def test_truncate_edge_cases():
    assert truncate_axtree("abc", 0) == ""
    assert len(truncate_axtree("x" * 50, 5)) <= 5


# -- token budget --------------------------------------------------------------


def test_estimate_tokens_counts_text_and_images(make_obs):
    messages = [{"role": "user", "content": "x" * 350}]
    assert estimate_tokens(messages) == 4 + 100
    image = Image.new("RGB", (512, 320))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    assert estimate_tokens(messages) == 4 + 512 * 320 // 1024 + 2


def test_eight_turns_fit_in_the_context_window(make_obs):
    """8 turns of a 512x320 screenshot and a 2000-char AXTree stay under 8192 tokens."""
    axtree = truncate_axtree("\n".join(f"[{i}] link 'Item number {i}'" for i in range(400)), 2000)
    assert len(axtree) == pytest.approx(2000, abs=60)

    def obs_at(step):
        return make_obs(axtree=axtree, size=(512, 320), step=step, goal="Click on the link with the text 'Item number 3'.")

    messages = initial_messages(obs_at(0), "A", with_image=True)
    response = 'I will click the link.\n<tool_call>\n{"name": "click", "arguments": {"bid": "12"}}\n</tool_call>'
    for turn in range(8):
        # The last turn ends the episode, so no tool result follows it.
        content = tool_result_content(obs_at(turn + 1), "A") if turn < 7 else None
        append_turn(messages, response, content, "click")

    assert sum(1 for m in messages if m["role"] == "assistant") == 8
    # The estimate skips the tool schemas that the chat template adds, so leave 600 tokens for them.
    assert estimate_tokens(messages) + 600 < 8192


# -- format_observation (Track B needs no BrowserGym) ---------------------------


def _raw_obs(np, height=200, width=400):
    return {
        "goal": "Click the button.",
        "screenshot": np.zeros((height, width, 3), dtype="uint8"),
        "last_action_error": "",
        "focused_element_bid": "7",
    }


def test_format_observation_track_b(cfg):
    np = pytest.importorskip("numpy")
    cfg.env.track = "B"
    cfg.env.screenshot_long_side = 200
    obs = format_observation(_raw_obs(np), cfg, step=3)
    assert obs.axtree == ""  # Track B never shows the AXTree.
    assert obs.screenshot.size == (200, 100)  # Resized to the long side.
    assert obs.screen_size == (400, 200)  # The raw size is preserved for coordinate mapping.
    assert (obs.goal, obs.step, obs.focused_bid) == ("Click the button.", 3, "7")


def test_format_observation_never_upscales(cfg):
    np = pytest.importorskip("numpy")
    cfg.env.track = "B"
    cfg.env.screenshot_long_side = 512
    obs = format_observation(_raw_obs(np, height=100, width=200), cfg)
    assert obs.screenshot.size == (200, 100)


def test_format_observation_goal_falls_back_to_goal_object(cfg):
    np = pytest.importorskip("numpy")
    cfg.env.track = "B"
    raw = _raw_obs(np)
    raw["goal"] = ""
    raw["goal_object"] = ({"type": "text", "text": "From the goal object."},)
    assert format_observation(raw, cfg).goal == "From the goal object."


def test_resize_long_side_keeps_aspect_ratio():
    out = resize_long_side(Image.new("RGB", (1000, 500)), 250)
    assert out.size == (250, 125)


# -- MINIWOB_URL ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("/workspace/miniwob/html/miniwob", "file:///workspace/miniwob/html/miniwob/"),
        ("/workspace/miniwob/html/miniwob/", "file:///workspace/miniwob/html/miniwob/"),
        ("file:///a/b", "file:///a/b/"),
        ("http://localhost:8000/miniwob//", "http://localhost:8000/miniwob/"),
    ],
)
def test_normalize_miniwob_url(value, expected):
    assert normalize_miniwob_url(value) == expected
