"""Tests for cua.actions: rendering, parsing, and tool schemas."""

from __future__ import annotations

import ast
import json

import pytest

from cua.actions import (
    TRACK_A_TOOLS,
    TRACK_B_TOOLS,
    ActionError,
    parse_response,
    render_action,
    tool_schemas,
    tools_for,
)

SIZE = (500, 300)  # (width, height): round numbers make the coordinate math obvious


def call(name: str, args: dict | None = None) -> str:
    """Build a well-formed tool-call block."""
    payload = {"name": name} if args is None else {"name": name, "arguments": args}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


# -- valid calls -------------------------------------------------------------

TRACK_A_CASES = [
    ("click", {"bid": "12"}, "click('12')"),
    ("click", {"bid": 12}, "click('12')"),  # ints are coerced to strings
    ("click", {"bid": "a7"}, "click('a7')"),  # frame bids contain letters
    ("fill", {"bid": "7", "text": "hello"}, "fill('7', 'hello')"),
    ("fill", {"bid": "7", "text": ""}, "fill('7', '')"),
    ("select_option", {"bid": "3", "option": "Blue"}, "select_option('3', 'Blue')"),
    ("press", {"bid": "5", "key": "Enter"}, "press('5', 'Enter')"),
    ("scroll", {"direction": "down"}, "scroll(0, 150)"),
    ("scroll", {"direction": "up"}, "scroll(0, -150)"),
    ("noop", {}, "noop()"),
]


@pytest.mark.parametrize("name,args,expected", TRACK_A_CASES)
def test_track_a_valid_calls(name, args, expected):
    assert render_action(name, args, "A", SIZE) == expected
    parsed = parse_response(call(name, args), "A", SIZE)
    assert parsed.format_error is None
    assert parsed.action == expected
    assert parsed.tool_name == name
    assert not parsed.ends_episode


TRACK_B_CASES = [
    ("mouse_click", {"coordinate": [500, 500]}, "mouse_click(250, 150)"),
    ("mouse_click", {"coordinate": [0, 0]}, "mouse_click(0, 0)"),
    ("keyboard_type", {"text": "hello"}, "keyboard_type('hello')"),
    ("keyboard_press", {"key": "Enter"}, "keyboard_press('Enter')"),
    ("scroll", {"direction": "down"}, "scroll(0, 150)"),
    ("noop", {}, "noop()"),
]


@pytest.mark.parametrize("name,args,expected", TRACK_B_CASES)
def test_track_b_valid_calls(name, args, expected):
    assert render_action(name, args, "B", SIZE) == expected
    assert parse_response(call(name, args), "B", SIZE).action == expected


def test_noop_without_arguments_key():
    parsed = parse_response(call("noop"), "A", SIZE)
    assert parsed.action == "noop()"


# -- coordinate mapping ------------------------------------------------------


def test_coordinates_clamp_to_the_screenshot():
    assert render_action("mouse_click", {"x": 1000, "y": 1000}, "B", SIZE) == "mouse_click(499, 299)"
    assert render_action("mouse_click", {"x": -50, "y": 5000}, "B", SIZE) == "mouse_click(0, 299)"


def test_coordinates_use_the_raw_screen_size():
    assert render_action("mouse_click", {"x": 250, "y": 750}, "B", (400, 200)) == "mouse_click(100, 150)"


def test_coordinates_accept_floats_and_numeric_strings():
    assert render_action("mouse_click", {"x": 500.0, "y": "500"}, "B", SIZE) == "mouse_click(250, 150)"


@pytest.mark.parametrize("bad", [None, True, "left", float("nan"), float("inf"), [1]])
def test_bad_coordinates_are_rejected(bad):
    with pytest.raises(ActionError):
        render_action("mouse_click", {"x": bad, "y": 10}, "B", SIZE)


def test_unknown_screen_size_is_rejected():
    with pytest.raises(ActionError):
        render_action("mouse_click", {"x": 1, "y": 1}, "B", (0, 0))


# -- quoting -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["it's", 'say "hi"', "line1\nline2", "back\\slash", "mixed ' and \" and \n", "unicode: café"],
)
def test_text_survives_quoting(text):
    """The action string must parse as Python and round-trip the original text."""
    action = render_action("fill", {"bid": "1", "text": text}, "A", SIZE)
    node = ast.parse(action, mode="eval").body
    assert isinstance(node, ast.Call) and node.func.id == "fill"
    assert [ast.literal_eval(a) for a in node.args] == ["1", text]
    # An injected second statement must not escape the string literal.
    assert "\n" not in action


def test_quote_injection_stays_inside_the_literal():
    text = "x'); noop(); fill('1', 'y"
    action = render_action("keyboard_type", {"text": text}, "B", SIZE)
    node = ast.parse(action, mode="eval").body
    assert ast.literal_eval(node.args[0]) == text


# -- validation --------------------------------------------------------------


@pytest.mark.parametrize("bid", ["", "  ", "a b", "[12]", "12;", None, True, 1.5, ["1"], {"a": 1}])
def test_bad_bids_are_rejected(bid):
    with pytest.raises(ActionError):
        render_action("click", {"bid": bid}, "A", SIZE)


def test_missing_arguments():
    with pytest.raises(ActionError, match="Missing"):
        render_action("click", {}, "A", SIZE)
    with pytest.raises(ActionError, match="text"):
        render_action("fill", {"bid": "1"}, "A", SIZE)


def test_extra_arguments_are_rejected():
    with pytest.raises(ActionError, match="Unexpected"):
        render_action("click", {"bid": "1", "force": True}, "A", SIZE)


def test_bad_scroll_direction():
    with pytest.raises(ActionError):
        render_action("scroll", {"direction": "left"}, "A", SIZE)


def test_unknown_tool_and_wrong_track():
    with pytest.raises(ActionError, match="Available tools"):
        render_action("teleport", {}, "A", SIZE)
    with pytest.raises(ActionError, match="Available tools"):
        render_action("mouse_click", {"x": 1, "y": 1}, "A", SIZE)
    with pytest.raises(ActionError, match="Available tools"):
        render_action("click", {"bid": "1"}, "B", SIZE)


def test_non_dict_arguments():
    with pytest.raises(ActionError):
        render_action("click", ["1"], "A", SIZE)  # type: ignore[arg-type]


# -- malformed model output --------------------------------------------------


def test_no_tag_ends_the_episode():
    parsed = parse_response("I will click the button.", "A", SIZE)
    assert parsed.action is None
    assert parsed.format_error
    assert parsed.ends_episode
    assert parsed.reasoning == "I will click the button."


def test_bad_json_ends_the_episode():
    parsed = parse_response("<tool_call>{not json at all</tool_call>", "A", SIZE)
    assert parsed.action is None and parsed.format_error and parsed.ends_episode


def test_empty_tag_ends_the_episode():
    parsed = parse_response("<tool_call></tool_call>", "A", SIZE)
    assert parsed.ends_episode and parsed.format_error


def test_unknown_tool_continues_the_episode():
    parsed = parse_response(call("teleport", {}), "A", SIZE)
    assert parsed.action is None
    assert "Unknown tool" in parsed.format_error
    assert not parsed.ends_episode
    assert parsed.tool_name == "teleport"


def test_wrong_track_tool_continues_the_episode():
    parsed = parse_response(call("mouse_click", {"x": 1, "y": 2}), "A", SIZE)
    assert parsed.action is None and parsed.format_error and not parsed.ends_episode
    parsed = parse_response(call("click", {"bid": "1"}), "B", SIZE)
    assert parsed.action is None and parsed.format_error and not parsed.ends_episode


def test_missing_arguments_continue_the_episode():
    parsed = parse_response(call("fill", {"bid": "1"}), "A", SIZE)
    assert parsed.action is None and "text" in parsed.format_error and not parsed.ends_episode


def test_extra_prose_is_split_from_the_call():
    text = "The button has bid 4.\n" + call("click", {"bid": "4"}) + "\nHope that works!"
    parsed = parse_response(text, "A", SIZE)
    assert parsed.action == "click('4')"
    assert parsed.reasoning == "The button has bid 4."


def test_two_tool_calls_first_wins():
    text = call("click", {"bid": "1"}) + "\n" + call("click", {"bid": "2"})
    assert parse_response(text, "A", SIZE).action == "click('1')"


def test_two_calls_without_closing_tags_first_wins():
    text = '<tool_call>{"name": "click", "arguments": {"bid": "1"}}\n<tool_call>{"name": "click", "arguments": {"bid": "2"}}'
    assert parse_response(text, "A", SIZE).action == "click('1')"


def test_arguments_given_as_a_json_string():
    text = '<tool_call>{"name": "click", "arguments": "{\\"bid\\": \\"3\\"}"}</tool_call>'
    parsed = parse_response(text, "A", SIZE)
    assert parsed.action == "click('3')"
    assert parsed.tool_args == {"bid": "3"}


def test_arguments_string_that_is_not_json():
    text = '<tool_call>{"name": "click", "arguments": "bid 3"}</tool_call>'
    parsed = parse_response(text, "A", SIZE)
    assert parsed.action is None and parsed.format_error and not parsed.ends_episode


def test_arguments_that_are_not_an_object():
    parsed = parse_response('<tool_call>{"name": "click", "arguments": [1]}</tool_call>', "A", SIZE)
    assert parsed.action is None and parsed.format_error and not parsed.ends_episode


def test_missing_closing_tag():
    parsed = parse_response('<tool_call>\n{"name": "click", "arguments": {"bid": "9"}}\n', "A", SIZE)
    assert parsed.action == "click('9')"


def test_single_quotes():
    parsed = parse_response("<tool_call>{'name': 'click', 'arguments': {'bid': '9'}}</tool_call>", "A", SIZE)
    assert parsed.action == "click('9')"


def test_trailing_commas():
    text = '<tool_call>{"name": "fill", "arguments": {"bid": "2", "text": "x",},}</tool_call>'
    assert parse_response(text, "A", SIZE).action == "fill('2', 'x')"


def test_end_of_turn_token_after_the_json_is_ignored():
    text = '<tool_call>{"name": "noop", "arguments": {}}<|im_end|>'
    assert parse_response(text, "A", SIZE).action == "noop()"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<tool_call>",
        "</tool_call>",
        "<tool_call>[1, 2]</tool_call>",
        '<tool_call>{"name": 5}</tool_call>',
        '<tool_call>{"name": ""}</tool_call>',
        '<tool_call>{"arguments": {}}</tool_call>',
        "<tool_call>{{{{</tool_call>",
        "<tool_call>" + "{" * 5000,
        "\x00\x01 <tool_call> \ud800",
        "<tool_call>null</tool_call>",
    ],
)
def test_parser_never_raises(text):
    parsed = parse_response(text, "A", SIZE)
    assert parsed.action is None
    assert parsed.format_error


def test_parser_tolerates_non_string_input():
    assert parse_response(None, "A", SIZE).format_error  # type: ignore[arg-type]


# -- tool lists and schemas --------------------------------------------------


def test_tool_lists():
    assert [f.__name__ for f in tools_for("A")] == ["click", "fill", "select_option", "press", "scroll", "noop"]
    assert [f.__name__ for f in tools_for("B")] == ["mouse_click", "keyboard_type", "keyboard_press", "scroll", "noop"]
    assert tools_for("A") == TRACK_A_TOOLS
    assert tools_for("B") == TRACK_B_TOOLS
    with pytest.raises(ValueError):
        tools_for("C")  # type: ignore[arg-type]


def test_every_tool_has_a_docstring_with_args_for_each_parameter():
    import inspect

    for fn in TRACK_A_TOOLS + TRACK_B_TOOLS:
        doc = inspect.getdoc(fn)
        assert doc
        for param in inspect.signature(fn).parameters:
            assert f"{param}:" in doc, f"{fn.__name__} does not document {param}"


@pytest.mark.parametrize("track", ["A", "B"])
def test_tool_schemas(track):
    pytest.importorskip("transformers")
    schemas = tool_schemas(track)
    assert [s["function"]["name"] for s in schemas] == [f.__name__ for f in tools_for(track)]
    for schema in schemas:
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        # Every parameter must be required: the parser rejects missing ones.
        assert set(function["parameters"].get("required", [])) == set(function["parameters"]["properties"])


def test_click_schema_details():
    pytest.importorskip("transformers")
    schema = {s["function"]["name"]: s["function"] for s in tool_schemas("A")}["click"]
    assert schema["parameters"]["properties"]["bid"]["type"] == "string"
    mouse = {s["function"]["name"]: s["function"] for s in tool_schemas("B")}["mouse_click"]
    assert mouse["parameters"]["properties"]["coordinate"]["type"] == "array"


def test_mouse_click_accepts_legacy_argument_shapes():
    assert render_action("mouse_click", {"x": 500, "y": 500}, "B", SIZE) == "mouse_click(250, 150)"
    assert render_action("mouse_click", {"x": [500, 500]}, "B", SIZE) == "mouse_click(250, 150)"


@pytest.mark.parametrize("bad", [[1], [1, 2, 3], "1,2", None])
def test_mouse_click_rejects_bad_coordinate_pairs(bad):
    with pytest.raises(ActionError):
        render_action("mouse_click", {"coordinate": bad}, "B", SIZE)
