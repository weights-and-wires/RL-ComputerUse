#!/usr/bin/env bash
# Start, inspect, or stop run 3. Runs on your Mac; the work happens on Bazzite.
set -Eeuo pipefail
{ set +x; } 2>/dev/null

usage() {
  cat <<'EOF'
Usage: scripts/run3_go.sh [--yes] [--adopt POD_ID] [--from-mac]
       scripts/run3_go.sh --status | --stop | --kill [--yes]

Without a mode flag, the script starts run 3:
  1. Syncs monitor/ to Bazzite (~/cua-run3/monitor).
  2. Shows the pod plan with live prices and the budget rules.
  3. Updates ~/cua-run3/repo on Bazzite. If your checkout is clean and matches
     origin/main, Bazzite pulls main from GitHub. Otherwise, or with --from-mac,
     the script copies your working tree (tracked and untracked, not ignored).
  4. Asks for confirmation, writes ~/cua-run3/ARMED, and starts the
     cua-run3-monitor user service, which creates the pod.

Options:
  --yes            Don't ask for confirmation.
  --adopt POD_ID   Use a pod you created by hand instead of creating one.
  --from-mac       Copy your working tree even if it matches origin/main.
  --status         Show the monitor state, service state, and recent events.
  --stop           Graceful stop: end GRPO early (.stop_grpo), let the post-training
                   evals finish, upload, then delete the pod.
  --kill           Upload once right now, then delete the pod even if the upload fails.

Environment:
  BAZZITE_HOST     SSH target. Default: ifkash@192.168.1.4, with ifkash@100.115.46.42
                   (Tailscale) as the fallback.
  BAZZITE_KEY      SSH key for Bazzite. Default: ~/.ssh/id_ed25519_kh.
EOF
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${BAZZITE_KEY:-$HOME/.ssh/id_ed25519_kh}"
MODE=go; YES=0; ADOPT=""; FROM_MAC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1; shift ;;
    --adopt) ADOPT="${2:?--adopt needs a pod id}"; shift 2 ;;
    --from-mac) FROM_MAC=1; shift ;;
    --status) MODE=status; shift ;;
    --stop) MODE=stop; shift ;;
    --kill) MODE=kill; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$ADOPT" =~ ^[A-Za-z0-9_-]*$ ]] || { echo "Pod id has unexpected characters." >&2; exit 2; }

pick_host() {
  local h
  for h in ${BAZZITE_HOST:-ifkash@192.168.1.4 ifkash@100.115.46.42}; do
    if ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=6 "$h" true 2>/dev/null; then echo "$h"; return 0; fi
  done
  return 1
}
HOST="$(pick_host)" || { echo "ERROR: can't reach Bazzite over SSH." >&2; exit 1; }
B() { ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$@"; }
RSH="ssh -i $KEY -o ConnectTimeout=10"
confirm() { [[ "$YES" -eq 1 ]] && return 0; local r; read -r -p "$1 [y/N] " r; [[ "$r" == y || "$r" == Y || "$r" == yes ]]; }
remote_state() { B 'python3 -c "import json,os;print(json.load(open(os.path.expanduser(\"~/cua-run3/state.json\"))).get(\"run_state\",\"idle\"))" 2>/dev/null || echo idle'; }
ensure_running() { B 'systemctl --user is-active -q cua-run3-monitor || systemctl --user start cua-run3-monitor'; }

case "$MODE" in
  status)
    B 'cd ~/cua-run3 && python3 monitor/monitor.py status; echo; echo "service: $(systemctl --user is-active cua-run3-monitor)"; echo "recent events:"; tail -n 8 data/events.jsonl 2>/dev/null | python3 -c "import json,sys
for l in sys.stdin:
    e=json.loads(l); print(\"  \", e[\"t\"], e[\"kind\"], \"-\", e[\"msg\"][:160])"'
    exit 0 ;;
  stop)
    B 'python3 ~/cua-run3/monitor/monitor.py stop'
    ensure_running
    echo "The monitor touches .stop_grpo, waits for ALL DONE, uploads, and deletes the pod."
    exit 0 ;;
  kill)
    confirm "Upload once now and then DELETE the pod even if the upload fails?" || exit 1
    B 'python3 ~/cua-run3/monitor/monitor.py kill'
    ensure_running
    echo "Queued. Follow it with: scripts/run3_go.sh --status"
    exit 0 ;;
esac

# ---- go ------------------------------------------------------------------------
echo "Bazzite: $HOST"
rsync -a --delete -e "$RSH" --exclude __pycache__ "$REPO/monitor/" "$HOST:cua-run3/monitor/"
[[ -f "$HOME/.ssh/id_ed25519.pub" ]] && B 'cat > ~/cua-run3/authorized_keys.extra' < "$HOME/.ssh/id_ed25519.pub"
B 'test -s ~/cua-run3/.env && test -f ~/.config/systemd/user/cua-run3-monitor.service' \
  || { echo "ERROR: Bazzite isn't set up. Run: monitor/deploy.sh --env" >&2; exit 1; }

state="$(remote_state)"
case "$state" in
  idle|done|failed) ;;
  *) echo "A run is already in progress (state: $state). Use --status, --stop, or --kill." >&2; exit 1 ;;
esac
if [[ "$state" == failed ]] && B 'python3 - <<"PY"
import json, os, sys
st = json.load(open(os.path.expanduser("~/cua-run3/state.json")))
sys.exit(0 if any(not p.get("deleted_ts") for p in st.get("pods", [])) else 1)
PY'; then
  echo "The last run failed and left a pod running. Resolve it first (--kill, or the Runpod console)." >&2
  exit 1
fi

echo
B 'python3 ~/cua-run3/monitor/monitor.py plan'
echo

# Repo source.
cd "$REPO"
SRC="github"
git fetch -q origin main 2>/dev/null || true
if [[ "$FROM_MAC" -eq 1 || -n "$(git status --porcelain)" || "$(git rev-parse HEAD)" != "$(git rev-parse origin/main 2>/dev/null)" ]]; then
  SRC="mac"
fi
if [[ "$SRC" == github ]]; then
  echo "Code: GitHub main ($(git rev-parse --short origin/main)), cloned or pulled on Bazzite."
else
  echo "Code: your working tree at $(git rev-parse --short HEAD)$( [[ -n "$(git status --porcelain)" ]] && echo ' plus uncommitted changes'), copied to Bazzite."
fi
[[ -n "$ADOPT" ]] && echo "Pod: adopt $ADOPT (no new pod is created for generation 1)."
echo
confirm "Start run 3 now? The monitor creates the pod and billing starts." || { echo "Cancelled."; exit 1; }

if [[ "$SRC" == github ]]; then
  B 'set -e; cd ~/cua-run3
     if [ -d repo/.git ]; then git -C repo fetch -q origin main && git -C repo checkout -q -B main origin/main && git -C repo reset -q --hard origin/main
     else rm -rf repo && git clone -q -b main git@github.com:weights-and-wires/RL-ComputerUse.git repo; fi
     git -C repo log -1 --format="repo at %h %s"'
else
  # Ship tracked and untracked (not ignored) files as one tar, then swap the folder.
  # Dot-folders (.cloudflare, .claude, ...) and .env files never leave the Mac.
  list="$(mktemp)"
  git ls-files -co --exclude-standard \
    | grep -vE '^(\.env|\.[^/]+/|results/|results_live/|demo/)' \
    | while IFS= read -r f; do [[ -e "$f" ]] && printf '%s\n' "$f"; done > "$list"
  rev="$(git rev-parse HEAD)$( [[ -n "$(git status --porcelain)" ]] && echo '+dirty')"
  COPYFILE_DISABLE=1 tar -czf - -T "$list" \
    | B "set -e; cd ~/cua-run3; rm -rf repo.new; mkdir repo.new; tar -xzf - -C repo.new 2>/dev/null
         echo '$rev' > repo.new/REVISION; rm -rf repo; mv repo.new repo
         echo \"repo: \$(find repo -type f | wc -l) files, revision \$(cat repo/REVISION)\""
  rm -f "$list"
fi
for f in scripts/run3_pod_start.sh scripts/run3.sh scripts/upload_run3.py; do
  B "test -f ~/cua-run3/repo/$f" || { echo "ERROR: ~/cua-run3/repo/$f is missing. Not starting." >&2; exit 1; }
done

ARM="{\"t\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"by\": \"run3_go.sh\"$( [[ -n "$ADOPT" ]] && printf ', "adopt": "%s"' "$ADOPT")}"
B "set -e; systemctl --user stop cua-run3-monitor 2>/dev/null || true
   cd ~/cua-run3
   if [ -f state.json ]; then mv state.json state.\$(date +%Y%m%d-%H%M%S).json; fi
   rm -f command.json
   printf '%s\n' '$ARM' > ARMED
   systemctl --user enable -q cua-run3-monitor
   systemctl --user start cua-run3-monitor
   sleep 5; systemctl --user is-active cua-run3-monitor"
echo "Run 3 armed. The monitor is creating the pod."
echo "Follow it: scripts/run3_go.sh --status   (dashboard: https://run.ifkash.dev)"
