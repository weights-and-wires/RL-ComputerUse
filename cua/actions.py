"""Tool definitions and tool-call parsing for the browser agent.

The model acts through Qwen's native tool-calling format:

    The OK button has bid 12.
    <tool_call>
    {"name": "click", "arguments": {"bid": "12"}}
    </tool_call>

Why this format: TRL's `environment_factory` exposes env methods as tools and
parses this same `<tool_call>` JSON. If eval and RFT used another format, the
model would see a different distribution than it trains on.

This module has three jobs:

1. Define each tool ONCE as a stub function. Its signature and docstring become
   the JSON schema that the chat template shows the model
   (`transformers.utils.get_json_schema` reads them).
2. `render_action` turns a validated tool call into a BrowserGym action string.
3. `parse_response` pulls the first tool call out of raw model text. It never
   raises, because a model sampled at temperature 1.0 produces odd output.

The stubs never run. The environment executes the BrowserGym string.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Callable

from cua.types import ParsedAction, Track

# Pixels per scroll call, measured in screenshot pixels. BrowserGym divides by
# its 1.5 zoom factor itself, so 150 here is 100 CSS pixels on the page.
SCROLL_PIXELS = 150

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


class ActionError(ValueError):
    """Raised by `render_action` when a tool call is invalid.

    The message goes back to the model as the tool result, so write it for the
    model: say what is wrong and what a valid call looks like.
    """


# ---------------------------------------------------------------------------
# Tool stubs. Docstrings follow Google style because get_json_schema parses the
# "Args:" section into the per-parameter descriptions. Keep them short: the
# whole schema is repeated in every prompt.
# ---------------------------------------------------------------------------


def click(bid: str) -> str:
    """Click an element on the page.

    Args:
        bid: The element ID, the number in square brackets in the page text, for example "12".

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def fill(bid: str, text: str) -> str:
    """Replace the content of a text field with the given text.

    Args:
        bid: The element ID of the text field, for example "7".
        text: The text to enter.

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def select_option(bid: str, option: str) -> str:
    """Choose an option in a dropdown list.

    Args:
        bid: The element ID of the dropdown, for example "3".
        option: The visible label of the option to select.

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def press(bid: str, key: str) -> str:
    """Focus an element and press a key or key combination.

    Args:
        bid: The element ID to focus, for example "5".
        key: The key to press, for example "Enter", "Tab", or "Control+a".

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def scroll(direction: str) -> str:
    """Scroll the page vertically by a fixed amount.

    Args:
        direction: Either "up" or "down".

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def noop() -> str:
    """Do nothing for one step. Use it to wait for the page to update.

    Returns:
        The new page state.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def mouse_click(coordinate: list[int]) -> str:
    """Click at a position on the screenshot.

    Args:
        coordinate: The position as [x, y], each from 0 to 1000. x runs from the
            left edge (0) to the right edge (1000); y runs from the top edge (0)
            to the bottom edge (1000).

    Returns:
        The new screenshot.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def keyboard_type(text: str) -> str:
    """Type text into the focused element. Click a text field first.

    Args:
        text: The text to type.

    Returns:
        The new screenshot.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


def keyboard_press(key: str) -> str:
    """Press a key or key combination on the focused element.

    Args:
        key: The key to press, for example "Enter", "Tab", or "Control+a".

    Returns:
        The new screenshot.
    """
    raise NotImplementedError("Tool stubs only describe the interface.")


TRACK_A_TOOLS: list[Callable] = [click, fill, select_option, press, scroll, noop]
TRACK_B_TOOLS: list[Callable] = [mouse_click, keyboard_type, keyboard_press, scroll, noop]


def tools_for(track: Track) -> list[Callable]:
    """Return the tool stubs for a track ("A" or "B")."""
    if track == "A":
        return list(TRACK_A_TOOLS)
    if track == "B":
        return list(TRACK_B_TOOLS)
    raise ValueError(f"Unknown track {track!r}. Use 'A' or 'B'.")


def tool_schemas(track: Track) -> list[dict]:
    """Return JSON schemas for a track's tools, ready for `tools=` in a chat template."""
    # Lazy import keeps `import cua.actions` fast and usable without transformers.
    from transformers.utils import get_json_schema

    return [get_json_schema(fn) for fn in tools_for(track)]


def _tool_names(track: Track) -> list[str]:
    return [fn.__name__ for fn in tools_for(track)]


# ---------------------------------------------------------------------------
# Argument validation. Each validator takes the raw JSON value and returns the
# normalized value, or raises ValueError with a short reason.
# ---------------------------------------------------------------------------

_BID_RE = re.compile(r"^[A-Za-z0-9]+$")


def _as_bid(value: Any) -> str:
    # Models often emit the bid as a JSON number, so we coerce ints to strings.
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not _BID_RE.match(value.strip()):
        raise ValueError(
            "the bid must be a non-empty string of letters and digits without brackets, for example \"12\""
        )
    return value.strip()


def _as_text(value: Any) -> str:
    """Accept any string (including empty) and plain numbers."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("expected a string")
    return value if isinstance(value, str) else str(value)


def _as_nonempty_text(value: Any) -> str:
    text = _as_text(value)
    if not text.strip():
        raise ValueError("expected a non-empty string")
    return text


def _as_direction(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in ("up", "down"):
        return value.strip().lower()
    raise ValueError('expected "up" or "down"')


def _as_coord(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("expected a number from 0 to 1000")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("expected a number from 0 to 1000") from None
    if not math.isfinite(number):
        raise ValueError("expected a finite number from 0 to 1000")
    return number


def _as_coord_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected [x, y]")
    return _as_coord(value[0]), _as_coord(value[1])


def _normalize_click_args(args: dict) -> dict:
    """Accept the click argument shapes the model produces.

    Qwen3-VL was trained to click with `{"coordinate": [x, y]}`, which is the
    schema. Older prompts used `{"x": X, "y": Y}`, and the model sometimes
    writes `{"x": [X, Y]}`. Map both onto `coordinate`.
    """
    if "coordinate" in args:
        return args
    if set(args) == {"x", "y"}:
        return {"coordinate": [args["x"], args["y"]]}
    if set(args) == {"x"} and isinstance(args["x"], (list, tuple)):
        return {"coordinate": args["x"]}
    return args


# Argument name -> validator, in the order the BrowserGym call uses them.
_ARG_SPECS: dict[str, list[tuple[str, Callable[[Any], Any]]]] = {
    "click": [("bid", _as_bid)],
    "fill": [("bid", _as_bid), ("text", _as_text)],
    "select_option": [("bid", _as_bid), ("option", _as_nonempty_text)],
    "press": [("bid", _as_bid), ("key", _as_nonempty_text)],
    "scroll": [("direction", _as_direction)],
    "noop": [],
    "mouse_click": [("coordinate", _as_coord_pair)],
    "keyboard_type": [("text", _as_text)],
    "keyboard_press": [("key", _as_nonempty_text)],
}


def _q(text: str) -> str:
    """Quote `text` as a Python string literal.

    BrowserGym parses the action string as Python. `repr()` escapes quotes,
    backslashes, and newlines, so text such as `it's "x"\\n` stays one literal
    instead of breaking out of the call.
    """
    return repr(text)


def _normalized_to_pixel(value: float, size: int) -> int:
    """Map a 0-1000 coordinate to a pixel index inside [0, size - 1]."""
    return min(max(round(value / 1000 * size), 0), size - 1)


def render_action(name: str, args: dict, track: Track, screen_size: tuple[int, int]) -> str:
    """Validate a tool call and build the BrowserGym action string.

    Args:
        name: The tool name from the model's tool call.
        args: The tool arguments from the model's tool call.
        track: "A" (element IDs) or "B" (coordinates).
        screen_size: The raw screenshot size as (width, height). Track B uses
            it to turn 0-1000 coordinates into pixels.

    Returns:
        An action string such as `click('12')` or `mouse_click(249, 160)`.

    Raises:
        ActionError: If the tool is unknown, belongs to the other track, or has
            missing, extra, or badly typed arguments.
    """
    available = _tool_names(track)
    if not isinstance(name, str) or name not in available:
        raise ActionError(f"Unknown tool {name!r}. Available tools: {', '.join(available)}.")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ActionError(f"The arguments for {name} must be a JSON object.")

    if name == "mouse_click":
        args = _normalize_click_args(args)
    spec = _ARG_SPECS[name]
    expected = [arg for arg, _ in spec]
    extra = [key for key in args if key not in expected]
    if extra:
        raise ActionError(
            f"Unexpected argument(s) {', '.join(map(repr, extra))} for {name}. "
            f"Expected: {', '.join(expected) or 'no arguments'}."
        )

    values: dict[str, Any] = {}
    for arg, check in spec:
        if arg not in args:
            raise ActionError(f"Missing required argument {arg!r} for {name}.")
        try:
            values[arg] = check(args[arg])
        except ValueError as exc:
            raise ActionError(f"Invalid argument {arg!r} for {name}: {exc}.") from None

    if name == "click":
        return f"click({_q(values['bid'])})"
    if name == "fill":
        return f"fill({_q(values['bid'])}, {_q(values['text'])})"
    if name == "select_option":
        return f"select_option({_q(values['bid'])}, {_q(values['option'])})"
    if name == "press":
        return f"press({_q(values['bid'])}, {_q(values['key'])})"
    if name == "scroll":
        return f"scroll(0, {SCROLL_PIXELS if values['direction'] == 'down' else -SCROLL_PIXELS})"
    if name == "noop":
        return "noop()"
    if name == "keyboard_type":
        return f"keyboard_type({_q(values['text'])})"
    if name == "keyboard_press":
        return f"keyboard_press({_q(values['key'])})"
    if name == "mouse_click":
        width, height = screen_size
        if width <= 0 or height <= 0:
            raise ActionError("The screenshot size is unknown, so the click position cannot be mapped.")
        x, y = values["coordinate"]
        px = _normalized_to_pixel(x, width)
        py = _normalized_to_pixel(y, height)
        return f"mouse_click({px}, {py})"
    raise ActionError(f"Unknown tool {name!r}.")  # Unreachable: _ARG_SPECS covers all tools.


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _loads_lenient(raw: str) -> Any:
    """Parse a JSON-ish object, tolerating common model mistakes.

    Handles, in order: strict JSON followed by junk, trailing commas, and
    Python-style literals (single quotes, True/None).

    Raises:
        ValueError: If nothing parses.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    raw = raw[start:]

    # 1. Strict JSON. raw_decode stops after the first object, so trailing
    #    text such as "<|im_end|>" or more prose does not matter.
    try:
        return json.JSONDecoder().raw_decode(raw)[0]
    except ValueError:
        pass

    # 2. Trim to the last brace and remove trailing commas before } or ].
    end = raw.rfind("}")
    candidate = _TRAILING_COMMA_RE.sub(r"\1", raw[: end + 1] if end != -1 else raw)
    try:
        return json.loads(candidate)
    except ValueError:
        pass

    # 3. Python literal syntax covers single quotes. literal_eval never runs code.
    try:
        return ast.literal_eval(candidate)
    except Exception as exc:  # noqa: BLE001  (SyntaxError, ValueError, RecursionError, ...)
        raise ValueError("invalid JSON") from exc


def _split_tool_call(text: str) -> tuple[str, str | None]:
    """Return (reasoning, tool-call body). The body is None if there is no tag."""
    start = text.find(_TOOL_OPEN)
    if start == -1:
        return text.strip(), None
    reasoning = text[:start].strip()
    rest = text[start + len(_TOOL_OPEN):]
    # The body ends at the closing tag, or at a second opening tag when the
    # model forgot the closing tag, or at the end of the text. First call wins.
    cut = len(rest)
    for marker in (_TOOL_CLOSE, _TOOL_OPEN):
        pos = rest.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return reasoning, rest[:cut]


def _parse(text: str, track: Track, screen_size: tuple[int, int]) -> ParsedAction:
    reasoning, body = _split_tool_call(text)
    if body is None:
        return ParsedAction(
            action=None,
            format_error="No <tool_call> block found. Call exactly one tool.",
            ends_episode=True,
            reasoning=reasoning,
        )

    def unparseable(reason: str) -> ParsedAction:
        # An unreadable tool call ends the episode, the same as TRL does.
        return ParsedAction(action=None, format_error=reason, ends_episode=True, reasoning=reasoning)

    try:
        call = _loads_lenient(body)
    except ValueError as exc:
        return unparseable(f"The <tool_call> content is not valid JSON ({exc}).")
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"].strip():
        return unparseable('The <tool_call> JSON must be an object with a "name" string.')

    name = call["name"].strip()
    args = call.get("arguments", {})
    if args is None:
        args = {}
    if isinstance(args, str):
        # Some models double-encode: "arguments": "{\"bid\": \"3\"}".
        if args.strip():
            try:
                args = _loads_lenient(args)
            except ValueError:
                pass  # Keep the string; render_action reports a clear error.
        else:
            args = {}

    try:
        action = render_action(name, args, track, screen_size)
    except ActionError as exc:
        # Bad arguments or unknown tool: the episode continues, and the error
        # text goes back to the model as the tool result.
        return ParsedAction(
            action=None,
            tool_name=name,
            tool_args=args if isinstance(args, dict) else {},
            format_error=str(exc),
            ends_episode=False,
            reasoning=reasoning,
        )
    return ParsedAction(action=action, tool_name=name, tool_args=dict(args), reasoning=reasoning)


def parse_response(text: str, track: Track, screen_size: tuple[int, int]) -> ParsedAction:
    """Parse raw model text into a `ParsedAction`. This function never raises.

    Rules:

    - The first `<tool_call>` block wins. Later blocks are ignored.
    - Everything before that block is the reasoning.
    - The closing tag is optional. Single quotes, trailing commas, and
      `arguments` given as a JSON string are accepted.
    - No tool call, or JSON that does not parse: `ends_episode=True`.
    - Unknown tool or bad arguments: `ends_episode=False`, `action=None`.

    Args:
        text: The raw generated text for one turn.
        track: "A" or "B".
        screen_size: The raw screenshot size as (width, height).

    Returns:
        A `ParsedAction`. Check `format_error` first, then `action`.
    """
    try:
        return _parse(text if isinstance(text, str) else "", track, screen_size)
    except Exception as exc:  # noqa: BLE001  (the contract says: never raise)
        return ParsedAction(
            action=None,
            format_error=f"Could not parse the response ({type(exc).__name__}).",
            ends_episode=True,
        )
