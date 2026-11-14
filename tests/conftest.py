"""Shared pytest fixtures.

These tests never need BrowserGym, vLLM, or a GPU. Tests that need
`transformers` or `numpy` call `pytest.importorskip` and skip when missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# Make `import cua` work no matter where pytest starts.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cua.config import load_config  # noqa: E402
from cua.types import Observation  # noqa: E402


@pytest.fixture
def cfg():
    """A fresh Config built from configs/base.yaml. Tests may mutate it."""
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture
def make_obs():
    """Factory for a fake Observation with a small PIL screenshot."""

    def _make(
        goal: str = 'Click on the "OK" button.',
        axtree: str = "[1] button 'OK'",
        error: str = "",
        step: int = 0,
        size: tuple[int, int] = (64, 40),
        with_image: bool = True,
    ) -> Observation:
        return Observation(
            goal=goal,
            screenshot=Image.new("RGB", size, "white") if with_image else None,
            axtree=axtree,
            last_action_error=error,
            step=step,
            screen_size=size,
        )

    return _make


@pytest.fixture
def obs(make_obs):
    """A default fake Observation."""
    return make_obs()
