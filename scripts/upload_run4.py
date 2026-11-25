#!/usr/bin/env python3
"""Upload everything run 4 produced to Hugging Face, then verify it.

Usage, on the pod, from the repository root:

    python scripts/upload_run4.py --final --verify

`--final` uploads the GRPO checkpoints, the final and best adapters, the RFT
adapter (if the rft phase ran), all of results/run4/, and the W&B files of the
GRPO run to HF_ADAPTER_REPO, under checkpoints/grpo-4b-run4/,
checkpoints/rft-4b-run4/, and results/run4/. It uploads only files that
aren't already on the Hub with the same size, so running it again is cheap.

`--verify` then lists the Hub and checks that every local file is there with
the same size. The exit code is 0 only when that check passes, and the last
output line is `VERIFIED`. `--verify` without `--final` only checks.

The sync loop (tmux session run4-sync) is stopped first, so two processes
don't commit at the same time. The code is shared with run 3
(`cua.hfsync.upload_cli`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cua import hfsync  # noqa: E402

if __name__ == "__main__":
    sys.exit(hfsync.upload_cli("run4"))
