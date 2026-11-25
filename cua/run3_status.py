"""The `--status-json` view of run 3, for the monitor: `python3 -m cua.run3_status`.

A thin wrapper around `cua.run_status`, which serves run 3 and run 4 alike.
`scripts/run3.sh --status-json` calls it. It must never import torch.
"""

from __future__ import annotations

from typing import Optional

from cua import run_status as _impl
from cua.run_status import (  # noqa: F401 - re-exported for callers and tests
    DEFAULT_CKPT,
    DEFAULT_CONFIG,
    DEFAULT_RESULTS,
    PHASES,
    RUN_NAME,
    _epoch,
    _parse_time,
    _read,
    build_status,
    config_max_steps,
    gpu_stats,
    last_line,
    last_step,
    newest_checkpoint,
    pid_alive,
)


def main(argv: Optional[list[str]] = None) -> None:
    _impl.main(argv, run="run3")


if __name__ == "__main__":
    main()
