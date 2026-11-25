#!/usr/bin/env bash
# Run 3 trigger. A thin wrapper: scripts/run4_go.sh with RUN=run3.
# Run 3 finished; this stays only so the old command still works.
set -Eeuo pipefail
RUN=run3 exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run4_go.sh" "$@"
