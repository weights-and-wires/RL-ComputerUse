"""Shared data types for the whole `cua` package.

Every module passes these objects around instead of loose dicts, so there is
exactly one definition of "an observation", "a model generation", and "an
episode". If you change a field here, search the package for its uses.

Data flow for one episode (see cua/rollout.py):

    env.reset(task, seed) -> Observation
    prompts.initial_messages(obs) -> [system, user] chat messages (HF format)
    policy.generate(conversations, sampling, tools) -> Generation per episode
    actions.parse_response(text) -> ParsedAction (a BrowserGym action string)
    env.step(action) -> StepResult (new Observation, success, terminated)
    prompts.append_turn(...) adds the assistant reply and the tool result
    ... repeat until the task ends, the model stops calling tools, or max_turns ...
    rewards.compute_reward(episode) -> RewardBreakdown
    Episode collects all of the above.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional

from PIL import Image

# "A" = element-ID actions (click("12")) with screenshot + AXTree.
# "B" = coordinate actions (mouse_click(500, 300)) with screenshot only.
Track = Literal["A", "B"]


@dataclass
class Observation:
    """What the agent sees at one turn, already formatted for the prompt."""

    goal: str
    """Natural-language task instruction, for example 'Click on the "OK" button.'"""

    screenshot: Optional[Image.Image]
    """Screenshot resized so the long side equals `env.screenshot_long_side`."""

    axtree: str
    """Flattened accessibility tree with BrowserGym element IDs (bids),
    truncated to `env.max_axtree_chars`. Empty string on Track B."""

    last_action_error: str = ""
    """Error message from the previous action, or empty if it succeeded."""

    step: int = 0
    """0 after reset, then 1, 2, ... after each env.step()."""

    screen_size: tuple[int, int] = (0, 0)
    """(width, height) of the raw BrowserGym screenshot in pixels, before any
    resizing. BrowserGym's coordinate actions (mouse_click and friends) take
    positions in this space; BrowserGym itself divides by its 1.5 zoom factor
    to get CSS pixels. Track B maps the model's 0-1000 normalized coordinates
    into this space."""

    focused_bid: str = ""
    """BrowserGym ID of the focused element, if any. Helps with keyboard tasks."""


@dataclass
class StepResult:
    """What MiniWoBEnv.step() returns."""

    observation: Observation
    success: bool
    """True when BrowserGym's reward is 1.0 (MiniWoB raw reward > 0)."""
    raw_env_reward: float
    """MiniWoB's own raw score, from info["task_info"]["RAW_REWARD_GLOBAL"]."""
    terminated: bool
    """True when the page reports the episode is over (success or failure)."""
    frame: Optional[Image.Image] = None
    """Full-size screenshot after the action, only when recording."""


@dataclass
class SamplingSpec:
    """Decoding settings for one generate() call. Shared by vLLM and HF paths."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 160
    return_logprobs: bool = False
    """Ask vLLM for the log-prob of each sampled token (nano GRPO needs it)."""
    seed: Optional[int] = None


@dataclass
class EpisodeSpec:
    """One episode to run: which task, which seed, which sample in its group."""

    task: str
    seed: int
    sample_idx: int = 0
    record: bool = False
    """Keep full-size frames for a GIF replay."""


# Generation.finish_reason when the prompt no longer fits in the model's context.
CONTEXT_OVERFLOW = "context_overflow"


@dataclass
class Generation:
    """One sampled model response for one turn.

    `token_ids` and `logprobs` are only filled by the vLLM policy when
    log-probs are requested. The nano GRPO loop needs them; eval does not.
    """

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""


@dataclass
class ParsedAction:
    """Result of parsing one model response. The parser never raises.

    The model acts through Qwen's native tool-calling format (the same format
    TRL's environment_factory uses during GRPO), for example:

        The OK button has bid 12.
        <tool_call>
        {"name": "click", "arguments": {"bid": "12"}}
        </tool_call>
    """

    action: Optional[str]
    """BrowserGym action string to execute, for example "click('12')".
    None when there is nothing valid to execute."""

    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)

    format_error: Optional[str] = None
    """None if the response is well formed, else a short reason. Two kinds:
    - No parseable <tool_call>: the episode ends here (TRL behaves the same
      way: a completion without a tool call ends the rollout).
    - A tool call with an unknown name or bad arguments: the error goes back
      to the model as the tool result and the episode continues."""

    ends_episode: bool = False
    """True when the response had no tool call at all."""

    reasoning: str = ""
    """Free text before the tool call, kept for logs and GIF captions."""


@dataclass
class Turn:
    """One step of an episode: what the model saw, said, and did."""

    step: int
    observation: Observation
    """The observation the model acted on."""
    prompt_len: int
    """Number of messages in Episode.messages that formed this turn's prompt.
    Episode.messages[:prompt_len] is the exact input; the next message is this
    turn's assistant reply. nano_grpo and RFT rebuild per-turn inputs from it."""
    generation: Generation
    parsed: ParsedAction
    env_error: str = ""
    """`last_action_error` reported by BrowserGym after executing the action,
    or the tool-argument error from the parser."""


@dataclass
class RewardBreakdown:
    """Episode reward split into its parts so each can be logged separately."""

    success: float = 0.0
    format_penalty: float = 0.0
    efficiency_bonus: float = 0.0

    @property
    def total(self) -> float:
        return self.success + self.format_penalty + self.efficiency_bonus

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/success": self.success,
            "reward/format_penalty": self.format_penalty,
            "reward/efficiency_bonus": self.efficiency_bonus,
        }


@dataclass
class Episode:
    """A complete episode from reset to termination or the turn limit."""

    task: str
    """MiniWoB task name without prefix, for example 'click-button'."""
    seed: int
    track: Track = "A"
    sample_idx: int = 0
    """Index within a GRPO group or pass@k batch (0..G-1)."""

    turns: list[Turn] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    """The full conversation in HF chat format: system, user (goal and first
    observation), then alternating assistant (raw model text) and tool
    (screenshot and page text) messages. Images are PIL objects inside
    content blocks. This is exactly what RFT trains on."""
    success: bool = False
    raw_env_reward: float = 0.0
    """Last reward value BrowserGym returned (MiniWoB's own score)."""
    terminated: bool = False
    truncated: bool = False
    """True when the episode hit max_turns without the task ending."""
    crashed: str = ""
    """Non-empty if the browser or env raised; the episode then counts as a
    failure and is excluded from training batches."""
    reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    frames: list[Image.Image] = field(default_factory=list)
    """Screenshots at full resolution for GIF replays (only when recording)."""

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_format_errors(self) -> int:
        return sum(1 for t in self.turns if t.parsed.format_error)

    def failure_mode(self) -> str:
        """Coarse label used by the report's failure-mode table."""
        if self.success:
            return "success"
        if self.crashed:
            return "crashed"
        if self.num_format_errors:
            return "format_error"
        if self.truncated:
            return "ran_out_of_turns"
        if any(t.env_error for t in self.turns):
            return "invalid_action"
        return "wrong_action"

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-serializable record (no images) for JSONL logs."""
        return {
            "task": self.task,
            "seed": self.seed,
            "track": self.track,
            "sample_idx": self.sample_idx,
            "success": self.success,
            "raw_env_reward": self.raw_env_reward,
            "num_turns": self.num_turns,
            "num_format_errors": self.num_format_errors,
            "truncated": self.truncated,
            "crashed": self.crashed,
            "failure_mode": self.failure_mode(),
            "actions": [t.parsed.action for t in self.turns],
            "responses": [t.generation.text for t in self.turns],
            "env_errors": [t.env_error for t in self.turns],
            **asdict(self.reward),
            "reward_total": self.reward.total,
        }


# A policy, seen from the rollout runner, is any function that takes a batch of
# chat conversations (HF message format, one list of messages per episode),
# the sampling settings, and the tool schemas (rendered into the system prompt
# by the chat template), and returns one Generation per conversation.
# VLLMPolicy.generate in policy.py is the main implementation; tests can pass
# a fake that returns canned text.
GenerateFn = Callable[
    [list[list[dict[str, Any]]], SamplingSpec, list[dict[str, Any]]],
    list[Generation],
]
