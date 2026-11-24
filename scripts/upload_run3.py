#!/usr/bin/env python3
"""Upload everything run 3 produced to Hugging Face, then verify it.

Usage, on the pod, from the repository root:

    python scripts/upload_run3.py --final --verify

`--final` uploads the GRPO checkpoints, the final adapter, the RFT adapter (if
the rft phase ran), all of results/run3/, and the W&B files of the GRPO run to
HF_ADAPTER_REPO, under checkpoints/grpo-4b-run3/, checkpoints/rft-4b-run3/,
and results/run3/. It uploads only files that aren't already on the Hub with
the same size, so running it again is cheap.

`--verify` then lists the Hub and checks that every local file is there with
the same size. The exit code is 0 only when that check passes. `--verify`
without `--final` only checks.

The sync loop (tmux session run3-sync) is stopped first, so two processes
don't commit at the same time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cua import hfsync  # noqa: E402


def stop_sync_session() -> None:
    try:
        subprocess.run(["tmux", "kill-session", "-t", "run3-sync"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass  # no tmux, or no session: nothing to stop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload run 3 to Hugging Face and verify it.")
    parser.add_argument("--final", action="store_true", help="Upload everything that isn't on the Hub yet.")
    parser.add_argument("--verify", action="store_true", help="Check every local file is on the Hub; exit 1 if not.")
    args = parser.parse_args(argv)
    if not args.final and not args.verify:
        parser.error("pass --final, --verify, or both")

    lay = hfsync.Layout.from_env()
    hub = hfsync.hub_from_env()
    if args.final:
        stop_sync_session()
        ok = hfsync.upload_final(hub, lay, do_verify=args.verify)
    else:
        missing, mismatched = hfsync.verify(hfsync.final_upload_map(lay), hfsync.remote_listing(hub, lay))
        ok = not missing and not mismatched
        hfsync.log(f"{len(missing)} missing, {len(mismatched)} with another size")
    print("VERIFIED" if ok and args.verify else ("UPLOADED" if ok else "NOT VERIFIED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
