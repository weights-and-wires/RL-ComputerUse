#!/usr/bin/env bash
# Install or update the run monitor on Bazzite. Runs on your Mac. Doesn't start
# the monitor, doesn't arm it, and doesn't create a pod.
#
# Usage: monitor/deploy.sh [--run run4] [--env] [--demo]
#   --run NAME  Run to install (default run4, or $RUN). Installs into ~/cua-NAME.
#   --env       Copy the repo's .env to ~/cua-NAME/.env on Bazzite (mode 600), replacing it.
#   --demo      Regenerate the demo data in ~/cua-NAME/data/ from results/run2.
set -Eeuo pipefail
{ set +x; } 2>/dev/null

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${BAZZITE_HOST:-ifkash@192.168.1.4}"
KEY="${BAZZITE_KEY:-$HOME/.ssh/id_ed25519_kh}"
SSH=(ssh -i "$KEY" -o ConnectTimeout=10 "$HOST")
RSH="ssh -i $KEY -o ConnectTimeout=10"
RUN="${RUN:-run4}"
COPY_ENV=0; DEMO=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="${2:?--run needs a name}"; shift 2 ;;
    --env) COPY_ENV=1; shift ;;
    --demo) DEMO=1; shift ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$RUN" =~ ^run[0-9]+[a-z]?$ ]] || { echo "Run name must look like run4." >&2; exit 2; }
HOME_DIR="cua-$RUN"
UNIT="cua-$RUN-monitor"
[[ -f "$REPO/monitor/$UNIT.service" ]] || { echo "No $REPO/monitor/$UNIT.service" >&2; exit 1; }

echo "Syncing monitor/ to $HOST:~/$HOME_DIR/monitor/"
"${SSH[@]}" "mkdir -p ~/$HOME_DIR/data ~/$HOME_DIR/cache && chmod 755 ~/$HOME_DIR ~/$HOME_DIR/data && chmod 700 ~/$HOME_DIR/cache"
rsync -a --delete -e "$RSH" --exclude __pycache__ "$REPO/monitor/" "$HOST:$HOME_DIR/monitor/"

# The Mac's Runpod key (runpod_up.sh default), so you can SSH into the pods too.
if [[ -f "$HOME/.ssh/id_ed25519.pub" ]]; then
  "${SSH[@]}" "cat > ~/$HOME_DIR/authorized_keys.extra" < "$HOME/.ssh/id_ed25519.pub"
fi

if [[ "$COPY_ENV" -eq 1 ]]; then
  [[ -f "$REPO/.env" ]] || { echo "No $REPO/.env" >&2; exit 1; }
  echo "Copying .env (mode 600). Values aren't printed."
  "${SSH[@]}" "umask 077; cat > ~/$HOME_DIR/.env.tmp && chmod 600 ~/$HOME_DIR/.env.tmp && mv ~/$HOME_DIR/.env.tmp ~/$HOME_DIR/.env" < "$REPO/.env"
fi

echo "Setting up the W&B venv (~/$HOME_DIR/venv) and the systemd user unit"
"${SSH[@]}" "HOME_DIR=$HOME_DIR UNIT=$UNIT bash -s" <<'REMOTE'
set -euo pipefail
cd ~/"$HOME_DIR"
if [ ! -x venv/bin/python ] || ! venv/bin/python -c "import wandb" 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then
    uv venv -q --python 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install -q wandb
  else
    python3 -m venv venv && venv/bin/pip -q install wandb
  fi
fi
venv/bin/python -c "import wandb; print('wandb', wandb.__version__)"
mkdir -p ~/.config/systemd/user
cp monitor/"$UNIT".service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable "$UNIT".service >/dev/null 2>&1
echo "unit: enabled=$(systemctl --user is-enabled "$UNIT") active=$(systemctl --user is-active "$UNIT" || true)"
[ -f ARMED ] && echo "WARNING: ~/$HOME_DIR/ARMED exists" || echo "ARMED: absent (the monitor won't create a pod)"
REMOTE

if [[ "$DEMO" -eq 1 ]]; then
  rsync -a -e "$RSH" --exclude report "$REPO/results/run2/" "$HOST:$HOME_DIR/demo_src/run2/"
  "${SSH[@]}" "cd ~/$HOME_DIR && python3 monitor/demo_data.py --src demo_src/run2"
fi
echo "Deployed. Start a run with scripts/${RUN}_go.sh."
