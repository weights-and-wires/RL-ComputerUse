#!/usr/bin/env bash
# Stop or delete the Runpod pod. Runs on your Mac.
#
# The script never deletes the network volume unless you pass --delete-volume
# AND type the volume name when asked.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_runpod.sh"

usage() {
  cat <<'EOF'
Usage: scripts/runpod_down.sh [--stop | --delete-pod] [--delete-volume] [options]

Before it acts, the script runs scripts/sync_results.sh --adapters to copy your
results and adapters to this Mac.

Modes:
  --stop            Stop the pod (default). GPU billing ends. Your network volume
                    and its files stay.
  --delete-pod      Delete the pod. The network volume and its files stay.
  --delete-volume   Also delete the network volume 'rl-cua-vol' and everything on
                    it. Implies --delete-pod. You must type the volume name to
                    confirm, and --yes doesn't skip that prompt.

Options:
  --no-sync         Skip the results sync.
  --yes             Skip the y/N prompts (not the volume-name prompt).
  -h, --help        Show this help.

Billing: stopping a pod ends its GPU charge. Runpod can still bill storage for
a stopped pod's container disk, and your network volume bills at its own
per-GB rate whether or not a pod uses it. Check the Runpod console for current
rates.

If --stop fails, Runpod may not allow stopping a pod that has a network volume
attached in your setup. Use --delete-pod instead. Your files live on the
volume, so scripts/runpod_up.sh can create a fresh pod on the same volume.
EOF
}

MODE="stop"
DELETE_VOLUME=0
DO_SYNC=1
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)          MODE="stop"; shift ;;
    --delete-pod)    MODE="delete"; shift ;;
    --delete-volume) DELETE_VOLUME=1; MODE="delete"; shift ;;
    --no-sync)       DO_SYNC=0; shift ;;
    --yes|-y)        ASSUME_YES=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) usage >&2; die "Unknown option: $1" ;;
  esac
done

load_api_keys
require_cmd curl "It ships with macOS."
require_cmd jq "Run: brew install jq"
init_tmp
load_state

[[ -n "${POD_ID:-}" || "$DELETE_VOLUME" -eq 1 ]] || die "No pod recorded in $STATE_FILE. Nothing to stop."

# ---- sync first -----------------------------------------------------------------
if [[ "$DO_SYNC" -eq 1 && -n "${POD_ID:-}" && -n "${SSH_HOST:-}" ]]; then
  info "Syncing results and adapters to this Mac before shutting down..."
  if ! "$SCRIPT_DIR/sync_results.sh" --adapters; then
    warn "The sync failed. Anything not on your Mac exists only on the pod or the volume."
    if [[ "$ASSUME_YES" -eq 1 ]] || confirm "Continue anyway?"; then :; else
      info "Cancelled."; exit 1
    fi
  fi
fi

# ---- pod ------------------------------------------------------------------------
if [[ -n "${POD_ID:-}" ]]; then
  if [[ "$MODE" == "stop" ]]; then
    info "Stopping pod $POD_ID..."
    code="$(rp_call POST "/pods/$POD_ID/stop" "$RP_TMP/resp.json")" || true
    if [[ "$code" =~ ^2 ]]; then
      POD_STATUS="EXITED"; save_state
      info "Stop requested. GPU billing ends."
      info "Verify in the Runpod console that the pod shows as Exited."
      info "To resume, run scripts/runpod_up.sh. It restarts the stopped pod."
    else
      warn "Stop failed (HTTP $code): $(rp_error_text "$RP_TMP/resp.json")"
      warn "Pods with a network volume can't always be stopped. The pod is still running and billing."
      warn "Run scripts/runpod_down.sh --delete-pod --no-sync. Your files stay on the volume."
      exit 1
    fi
  else
    if [[ "$ASSUME_YES" -ne 1 ]]; then
      confirm "Delete pod $POD_ID? The network volume stays." || { info "Cancelled."; exit 1; }
    fi
    info "Deleting pod $POD_ID..."
    code="$(rp_call DELETE "/pods/$POD_ID" "$RP_TMP/resp.json")" || true
    if [[ "$code" =~ ^2 || "$code" == "404" ]]; then
      POD_ID=""; POD_STATUS=""; SSH_HOST=""; SSH_PORT=""; COST_PER_HR=""
      save_state
      info "Pod deleted. GPU billing ends. The network volume still bills storage."
    else
      die "Delete failed (HTTP $code): $(rp_error_text "$RP_TMP/resp.json")"
    fi
  fi
fi

# ---- volume (explicit flag and typed confirmation only) ----------------------------
if [[ "$DELETE_VOLUME" -eq 1 ]]; then
  [[ -n "${VOLUME_ID:-}" ]] || die "No volume recorded in $STATE_FILE."
  warn "This permanently deletes volume '$VOLUME_NAME' ($VOLUME_ID) and everything on it:"
  warn "checkpoints, the venv, model weights, and results not yet synced. It can't be undone."
  read -r -p "Type the volume name ($VOLUME_NAME) to confirm: " typed
  [[ "$typed" == "$VOLUME_NAME" ]] || { info "Name didn't match. Volume kept."; exit 1; }
  code="$(rp_call DELETE "/networkvolumes/$VOLUME_ID" "$RP_TMP/resp.json")" || true
  if [[ "$code" =~ ^2 ]]; then
    VOLUME_ID=""; DATACENTER=""
    save_state
    info "Volume deleted."
  else
    die "Volume delete failed (HTTP $code): $(rp_error_text "$RP_TMP/resp.json"). A volume can't be deleted while a pod uses it."
  fi
else
  info "The network volume '$VOLUME_NAME' was not touched."
fi
