#!/usr/bin/env bash
# Install or update the multi-run dashboard on Bazzite. Runs on your Mac.
#
# Usage: dashboard/deploy.sh [--snapshots]
#   --snapshots   Rebuild the run 1 and run 2 snapshots from results/ and copy them to
#                 ~/cua-runs/run1/data and ~/cua-runs/run2/data on Bazzite.
#
# The script puts the server in ~/cua-dashboard, installs the cua-dashboard user unit
# (port 8790), and replaces the old cua-run3-dashboard unit. It never touches ~/cua-run3.
set -Eeuo pipefail
{ set +x; } 2>/dev/null

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${BAZZITE_HOST:-ifkash@192.168.1.4}"
KEY="${BAZZITE_KEY:-$HOME/.ssh/id_ed25519_kh}"
SSH=(ssh -i "$KEY" -o ConnectTimeout=10 "$HOST")
RSH="ssh -i $KEY -o ConnectTimeout=10"
SNAP=0
for a in "$@"; do
  case "$a" in
    --snapshots) SNAP=1 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

if [[ "$SNAP" -eq 1 ]]; then
  tmp="$(mktemp -d)"
  python3 "$REPO/dashboard/snapshot_runs.py" --out "$tmp"
  for r in run1 run2; do
    "${SSH[@]}" "mkdir -p ~/cua-runs/$r"
    rsync -a --delete -e "$RSH" "$tmp/$r/data/" "$HOST:cua-runs/$r/data/"
  done
  rm -rf "$tmp"
fi

"${SSH[@]}" "mkdir -p ~/cua-dashboard ~/cua-run4/data ~/cua-runs"
rsync -a --delete -e "$RSH" --exclude __pycache__ --exclude fixtures --exclude '._*' "$REPO/dashboard/" "$HOST:cua-dashboard/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p ~/.config/systemd/user
cp ~/cua-dashboard/cua-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
# Free port 8790: stop the old unit and disable it (its files stay where they are).
if systemctl --user list-unit-files cua-run3-dashboard.service >/dev/null 2>&1; then
  systemctl --user disable --now cua-run3-dashboard.service 2>/dev/null || true
fi
systemctl --user enable cua-dashboard.service >/dev/null 2>&1
systemctl --user restart cua-dashboard.service
sleep 2
echo "cua-dashboard: enabled=$(systemctl --user is-enabled cua-dashboard) active=$(systemctl --user is-active cua-dashboard)"
curl -s -o /dev/null -w "local / -> %{http_code}\n" http://127.0.0.1:8790/
REMOTE
