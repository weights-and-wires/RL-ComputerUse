#!/usr/bin/env bash
# Install or update the run-3 monitor on Bazzite. Runs on your Mac. Doesn't start
# the monitor and doesn't create a pod.
#
# Usage: monitor/deploy.sh [--env] [--demo]
#   --env    Copy the repo's .env to ~/cua-run3/.env on Bazzite (mode 600), replacing it.
#   --demo   Regenerate the demo data in ~/cua-run3/data/ from results/run2.
set -Eeuo pipefail
{ set +x; } 2>/dev/null

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${BAZZITE_HOST:-ifkash@192.168.1.4}"
KEY="${BAZZITE_KEY:-$HOME/.ssh/id_ed25519_kh}"
SSH=(ssh -i "$KEY" -o ConnectTimeout=10 "$HOST")
RSH="ssh -i $KEY -o ConnectTimeout=10"
COPY_ENV=0; DEMO=0
for a in "$@"; do
  case "$a" in
    --env) COPY_ENV=1 ;;
    --demo) DEMO=1 ;;
    -h|--help) sed -n 2,8p "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

echo "Syncing monitor/ to $HOST:~/cua-run3/monitor/"
"${SSH[@]}" 'mkdir -p ~/cua-run3/data ~/cua-run3/cache && chmod 755 ~/cua-run3 ~/cua-run3/data && chmod 700 ~/cua-run3/cache'
rsync -a --delete -e "$RSH" --exclude __pycache__ "$REPO/monitor/" "$HOST:cua-run3/monitor/"

# The Mac's Runpod key (runpod_up.sh default), so you can SSH into the pods too.
if [[ -f "$HOME/.ssh/id_ed25519.pub" ]]; then
  "${SSH[@]}" 'cat > ~/cua-run3/authorized_keys.extra' < "$HOME/.ssh/id_ed25519.pub"
fi

if [[ "$COPY_ENV" -eq 1 ]]; then
  [[ -f "$REPO/.env" ]] || { echo "No $REPO/.env" >&2; exit 1; }
  echo "Copying .env (mode 600). Values aren't printed."
  "${SSH[@]}" 'umask 077; cat > ~/cua-run3/.env.tmp && chmod 600 ~/cua-run3/.env.tmp && mv ~/cua-run3/.env.tmp ~/cua-run3/.env' < "$REPO/.env"
fi

echo "Setting up the W&B venv (~/cua-run3/venv) and the systemd user unit"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/cua-run3
if [ ! -x venv/bin/python ] || ! venv/bin/python -c "import wandb" 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then
    uv venv -q --python 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install -q wandb
  else
    python3 -m venv venv && venv/bin/pip -q install wandb
  fi
fi
venv/bin/python -c "import wandb; print('wandb', wandb.__version__)"
mkdir -p ~/.config/systemd/user
cp monitor/cua-run3-monitor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable cua-run3-monitor.service >/dev/null 2>&1
echo "unit: enabled=$(systemctl --user is-enabled cua-run3-monitor) active=$(systemctl --user is-active cua-run3-monitor || true)"
[ -f ARMED ] && echo "WARNING: ~/cua-run3/ARMED exists" || echo "ARMED: absent (the monitor won't create a pod)"
REMOTE

if [[ "$DEMO" -eq 1 ]]; then
  rsync -a -e "$RSH" --exclude report "$REPO/results/run2/" "$HOST:cua-run3/demo_src/run2/"
  "${SSH[@]}" 'cd ~/cua-run3 && python3 monitor/demo_data.py --src demo_src/run2'
fi
echo "Deployed. Start a run with scripts/run3_go.sh."
