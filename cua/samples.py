"""Sample rollouts for the run-3 dashboard.

The GRPO loop calls `write_sample` every `grpo.samples_every` steps. Each call
writes one folder, `<results>/samples/step-NNNNNN/`, that holds:

- `episode.json`: step, task, seed, reward, success, and one entry per turn
  with the screenshot file name, the model's text, the action, and any error.
- `turn-K.jpg`: the screenshot the model saw at turn K, at most 640 pixels
  wide, JPEG quality 70.

The folder appears atomically (written aside, then renamed), so a reader never
sees half a sample. See docs/RUN3_INTERFACES.md for the schema.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

MAX_WIDTH = 640
JPEG_QUALITY = 70


def sample_dir(root: Path, step: int) -> Path:
    return Path(root) / "samples" / f"step-{step:06d}"


def action_text(parsed: Any) -> str:
    """Shows a tool call as `name(arg=value, ...)`, for example `mouse_click(coordinate=[412, 230])`."""
    name = getattr(parsed, "tool_name", "") or ""
    args = getattr(parsed, "tool_args", None) or {}
    if name:
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        return f"{name}({inner})"
    return getattr(parsed, "action", None) or ""


def _save_jpeg(image: Any, path: Path) -> None:
    img = image.convert("RGB")
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, max(1, round(img.height * MAX_WIDTH / img.width))))
    img.save(path, format="JPEG", quality=JPEG_QUALITY)


def write_sample(ep: Any, step: int, root: Path) -> Path:
    """Writes one episode as a dashboard sample and returns its folder.

    `ep` is a `cua.types.Episode` (anything with the same attributes works).
    An existing folder for the same step is replaced.
    """
    final = sample_dir(root, step)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    turns = []
    for k, turn in enumerate(ep.turns):
        image_name = ""
        shot = getattr(turn.observation, "screenshot", None)
        if shot is not None:
            image_name = f"turn-{k}.jpg"
            _save_jpeg(shot, tmp / image_name)
        parsed = turn.parsed
        turns.append({
            "image": image_name,
            "response": turn.generation.text,
            "action": action_text(parsed),
            "error": turn.env_error or (parsed.format_error or ""),
        })
    record = {
        "step": int(step),
        "task": ep.task,
        "seed": int(ep.seed),
        "reward": float(ep.reward.total),
        "success": bool(ep.success),
        "turns": turns,
    }
    (tmp / "episode.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return final


def pick_episode(episodes: list[Any], mixed_keys: set, want_success: bool,
                 rng: Optional[random.Random] = None) -> Optional[Any]:
    """Chooses the episode to show.

    Prefers episodes from mixed groups (the ones GRPO learns from) whose outcome
    matches `want_success`, then any mixed-group episode, then any live episode.
    Alternating `want_success` between calls shows both successes and failures.
    `mixed_keys` holds the (task, seed) of every mixed group.
    """
    rng = rng or random.Random(0)
    live = [ep for ep in episodes if not ep.crashed and ep.turns]
    mixed = [ep for ep in live if (ep.task, ep.seed) in mixed_keys]
    for pool in ([ep for ep in mixed if bool(ep.success) == want_success], mixed, live):
        if pool:
            return rng.choice(pool)
    return None
