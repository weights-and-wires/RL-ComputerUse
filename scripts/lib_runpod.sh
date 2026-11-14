# shellcheck shell=bash
# Shared helpers for the Mac-side Runpod scripts (runpod_up.sh, runpod_down.sh,
# sync_results.sh). Source this file; don't run it.
#
# Secret handling rules this file enforces:
#   - Keys come from ~/.api_keys and live only in shell variables.
#   - The Runpod API key reaches curl through stdin (`curl -K -`), never argv,
#     so `ps` can't see it.
#   - Request bodies go through mode-600 temp files, never argv.
#   - Nothing here prints, logs, or writes a secret value.

# Never trace: `bash -x` would print secret values as they're expanded.
{ set +x; } 2>/dev/null

RP_API="https://rest.runpod.io/v1"
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/.." && pwd)"
STATE_DIR="$REPO_ROOT/.runpod"
STATE_FILE="$STATE_DIR/state.env"

VOLUME_NAME="rl-cua-vol"
REMOTE_REPO="/workspace/RL-ComputerUse"

# Pods get new IPs and ports on every start, so host-key checking only produces
# false alarms. The connection still uses your SSH key.
SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ServerAliveInterval=30
  -o ConnectTimeout=10
)

info() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command '$1'. $2"
}

# Source ~/.api_keys without echoing it. Sets RUNPOD_API_KEY, HF_TOKEN, and
# WANDB_API_KEY in this shell only.
load_api_keys() {
  { set +x; } 2>/dev/null
  [[ -f "$HOME/.api_keys" ]] || die "$HOME/.api_keys not found. Add 'export RUNPOD_API_KEY=...' lines to it."
  # shellcheck disable=SC1091
  source "$HOME/.api_keys"
  [[ -n "${RUNPOD_API_KEY:-}" ]] || die "RUNPOD_API_KEY is empty in ~/.api_keys."
  # Export so `jq` can read the values with $ENV and no secret lands on argv.
  export RUNPOD_API_KEY HF_TOKEN WANDB_API_KEY
}

# Private temp directory, removed on exit.
RP_TMP=""
init_tmp() {
  umask 077
  RP_TMP="$(mktemp -d)"
  trap 'rm -rf "$RP_TMP"' EXIT
}

# rp_call METHOD PATH OUTFILE [BODYFILE]
# Prints the HTTP status code (000 on a network failure). The response body
# goes to OUTFILE. Never print OUTFILE for pod calls without `del(.env)`,
# because pod objects echo the env block, which can hold secrets.
rp_call() {
  local method="$1" path="$2" out="$3" body="${4:-}"
  local args=(-sS --max-time 90 -X "$method" -o "$out" -w '%{http_code}'
    -H 'Content-Type: application/json' -K -)
  [[ -n "$body" ]] && args+=(--data-binary "@$body")
  # printf is a shell builtin, so the key never appears in a process listing.
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" \
    | curl "${args[@]}" "$RP_API$path" 2>/dev/null || true
  # On a network failure, curl's -w output is already "000".
}

# Print the error message from a response file, without the env block.
rp_error_text() {
  jq -r 'del(.env) | (.message // .error // .) | tostring' "$1" 2>/dev/null | head -c 600
}

# State file helpers. The state file holds IDs and SSH coordinates only.
STATE_VARS=(VOLUME_ID DATACENTER POD_ID POD_STATUS GPU_ID SSH_HOST SSH_PORT SSH_KEY COST_PER_HR)

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  umask 077
  mkdir -p "$STATE_DIR"
  local tmp="$STATE_FILE.tmp" v
  : > "$tmp"
  for v in "${STATE_VARS[@]}"; do
    printf '%s=%q\n' "$v" "${!v:-}" >> "$tmp"
  done
  mv "$tmp" "$STATE_FILE"
}

# Run a command on the pod over SSH. Requires SSH_HOST, SSH_PORT, SSH_KEY.
need_ssh_state() {
  [[ -n "${SSH_HOST:-}" && -n "${SSH_PORT:-}" && -n "${SSH_KEY:-}" ]] \
    || die "No SSH details in $STATE_FILE. Run scripts/runpod_up.sh first."
}

rp_ssh() {
  ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

# The `-e` argument for rsync, as one string.
rsync_ssh_cmd() {
  local s="ssh" o
  for o in "${SSH_OPTS[@]}"; do s+=" $o"; done
  printf '%s -i %q -p %s' "$s" "$SSH_KEY" "$SSH_PORT"
}

# confirm "Question" -> returns 0 on yes.
confirm() {
  local reply
  read -r -p "$1 [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" || "$reply" == "yes" ]]
}
