#!/usr/bin/env bash
# Create (or reuse) the network volume and start a Runpod H100 pod. Runs on your Mac.
#
# Uses the Runpod REST API v1 with curl and jq. The optional `runpodctl` CLI
# (brew install runpod/runpodctl/runpodctl) adds two conveniences: it lists
# datacenters and GPU stock, and it supplies the hourly price shown before you
# confirm. The script works without it.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_runpod.sh"

usage() {
  cat <<'EOF'
Usage: scripts/runpod_up.sh [--datacenter ID] [--gpu sxm|pcie] [options]

Find or create the 100 GB network volume "rl-cua-vol", then create a Secure
Cloud H100 pod that mounts it at /workspace. The script shows the hourly price
and asks for confirmation before it creates anything.

Options:
  --datacenter ID        Datacenter for a new volume, for example EU-RO-1. Required
                         the first time. Ignored when the volume already exists,
                         because a volume can't move between datacenters.
  --gpu sxm|pcie         H100 variant. Default: sxm ("NVIDIA H100 80GB HBM3").
                         pcie selects "NVIDIA H100 PCIe".
  --image IMAGE          Container image. Default:
                         runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
  --ssh-key PATH         Public key to install on the pod. Default:
                         ~/.ssh/id_ed25519.pub. The private key must sit next to it.
  --volume-gb N          Size for a new volume. Default: 100.
  --name NAME            Pod name. Default: rl-cua-pod.
  --allowed-cuda LIST    Optional comma-separated CUDA versions, for example
                         "12.9,13.0", passed as allowedCudaVersions.
  --no-secrets-in-env    Keep HF_TOKEN and WANDB_API_KEY out of the Runpod pod
                         config. The script copies them over SSH instead.
  --no-network-volume    Use a pod volume disk of --volume-gb at /workspace instead of
                         a network volume. Use this for datacenters that don't support
                         network volumes (for example US-KS-2, the only H100 PCIe
                         stock). Requires --datacenter. The data survives a pod stop,
                         but not a pod delete.
  --yes                  Skip the confirmation prompt.
  --dry-run              Print the pod request body with secrets redacted, then
                         exit. Makes no API calls.
  -h, --help             Show this help.

Secrets: values come from ~/.api_keys and are never printed or logged. By
default HF_TOKEN and WANDB_API_KEY go into the pod "env" field, which Runpod
stores in the pod configuration and shows in its console. Use
--no-secrets-in-env to avoid that. In both modes the script also writes
/workspace/.secrets (mode 600) over SSH, because SSH login shells don't always
inherit pod environment variables.

The pod bills every hour until you stop or delete it. Run
scripts/runpod_down.sh when you finish.
EOF
}

# ---- defaults and flags -----------------------------------------------------
GPU_KIND="sxm"
IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
PUBKEY_FILE="$HOME/.ssh/id_ed25519.pub"
VOLUME_GB=100
POD_NAME="rl-cua-pod"
ALLOWED_CUDA=""
SECRETS_IN_ENV=1
ASSUME_YES=0
DRY_RUN=0
FLAG_DC=""
NO_NET_VOLUME=0
HOURS_ESTIMATE=9
BUDGET_USD=50
FALLBACK_PRICE="2.99"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datacenter)         FLAG_DC="${2:?--datacenter needs a value}"; shift 2 ;;
    --gpu)                GPU_KIND="${2:?--gpu needs sxm or pcie}"; shift 2 ;;
    --image)              IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --ssh-key)            PUBKEY_FILE="${2:?--ssh-key needs a path}"; shift 2 ;;
    --volume-gb)          VOLUME_GB="${2:?--volume-gb needs a number}"; shift 2 ;;
    --name)               POD_NAME="${2:?--name needs a value}"; shift 2 ;;
    --allowed-cuda)       ALLOWED_CUDA="${2:?--allowed-cuda needs a value}"; shift 2 ;;
    --no-secrets-in-env)  SECRETS_IN_ENV=0; shift ;;
    --no-network-volume)  NO_NET_VOLUME=1; shift ;;
    --yes|-y)             ASSUME_YES=1; shift ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) usage >&2; die "Unknown option: $1" ;;
  esac
done

case "$GPU_KIND" in
  sxm)  GPU_ID="NVIDIA H100 80GB HBM3"; GPU_PATTERN="H100.*(SXM|HBM3)" ;;
  pcie) GPU_ID="NVIDIA H100 PCIe";      GPU_PATTERN="H100.*PCIe" ;;
  *)    die "--gpu must be sxm or pcie." ;;
esac

load_api_keys
require_cmd curl "It ships with macOS."
require_cmd jq "Run: brew install jq"
[[ -f "$PUBKEY_FILE" ]] || die "SSH public key not found: $PUBKEY_FILE (use --ssh-key)."
SSH_KEY="${PUBKEY_FILE%.pub}"
[[ -f "$SSH_KEY" ]] || die "Private key not found next to the public key: $SSH_KEY"
PUBKEY="$(cat "$PUBKEY_FILE")"

[[ -n "${HF_TOKEN:-}" ]]       || warn "HF_TOKEN is empty in ~/.api_keys. Model downloads are rate-limited without it."
[[ -n "${WANDB_API_KEY:-}" ]]  || warn "WANDB_API_KEY is empty in ~/.api_keys. Training logs to W&B will fail."

init_tmp
load_state
[[ -n "$FLAG_DC" ]] && DATACENTER="$FLAG_DC"
SSH_KEY="${PUBKEY_FILE%.pub}"   # after load_state, so the flag wins over a stale state value
GPU_ID_FOR_STATE="$GPU_ID"

# ---- request body -------------------------------------------------------------
# build_body real|redacted
# Public values go in with --arg. Secrets are read from the environment with
# $ENV inside jq, so they never appear on a command line.
build_body() {
  local mode="$1"
  jq -n \
    --arg name "$POD_NAME" --arg image "$IMAGE" --arg gpu "$GPU_ID" \
    --arg vol "${VOLUME_ID:-}" --arg dc "${DATACENTER:-}" --arg pk "$PUBKEY" \
    --arg mode "$mode" --argjson secrets "$SECRETS_IN_ENV" --arg cuda "$ALLOWED_CUDA" \
    --argjson nonet "$NO_NET_VOLUME" --argjson volgb "$VOLUME_GB" \
    '{
       name: $name,
       imageName: $image,
       computeType: "GPU",
       cloudType: "SECURE",
       gpuTypeIds: [$gpu],
       gpuCount: 1,
       minVCPUPerGPU: 16,
       minRAMPerGPU: 64,
       containerDiskInGb: 150,
       volumeInGb: 0,
       networkVolumeId: $vol,
       volumeMountPath: "/workspace",
       ports: ["22/tcp"],
       dataCenterIds: [$dc],
       env: { PUBLIC_KEY: $pk, SSH_PUBLIC_KEY: $pk }
     }
     | if $nonet == 1 then (del(.networkVolumeId) | .volumeInGb = $volgb) else . end
     | if $cuda != "" then .allowedCudaVersions = ($cuda | split(",")) else . end
     | if $secrets == 1 then
         .env += (
           if $mode == "redacted" then
             { HF_TOKEN: "<redacted>", WANDB_API_KEY: "<redacted>" }
           else
             { HF_TOKEN: $ENV.HF_TOKEN, WANDB_API_KEY: $ENV.WANDB_API_KEY }
           end
           | with_entries(select(.value != null and .value != ""))
         )
       else . end'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  VOLUME_ID="${VOLUME_ID:-<volume id, found or created at run time>}"
  DATACENTER="${DATACENTER:-<datacenter, taken from the volume at run time>}"
  info "Dry run: no API calls made. Request body for POST $RP_API/pods (secrets redacted):"
  build_body redacted
  info ""
  info "If the volume doesn't exist yet, the script first sends POST $RP_API/networkvolumes with:"
  jq -n --arg n "$VOLUME_NAME" --argjson s "$VOLUME_GB" --arg d "${FLAG_DC:-<datacenter>}" \
    '{name: $n, size: $s, dataCenterId: $d}'
  exit 0
fi

# ---- helpers ------------------------------------------------------------------
refresh_pod() {
  # Fills POD_STATUS, PUBLIC_IP, SSH_PORT_NEW, COST from GET /pods/{id}.
  local out="$RP_TMP/pod_get.json" code
  code="$(rp_call GET "/pods/$POD_ID" "$out")" || true
  [[ "$code" == "200" ]] || return 1
  POD_STATUS="$(jq -r '.desiredStatus // empty' "$out")"
  PUBLIC_IP="$(jq -r '.publicIp // empty' "$out")"
  SSH_PORT_NEW="$(jq -r '.portMappings["22"] // empty' "$out")"
  COST_PER_HR="$(jq -r '.costPerHr // empty' "$out")"
}

wait_for_pod() {
  info "Waiting for the pod to get a public IP and an SSH port (timeout 10 minutes)..."
  local waited=0
  while (( waited < 600 )); do
    if refresh_pod && [[ -n "${PUBLIC_IP:-}" && -n "${SSH_PORT_NEW:-}" ]]; then
      SSH_HOST="$PUBLIC_IP"; SSH_PORT="$SSH_PORT_NEW"
      save_state
      info "Pod is reachable at $SSH_HOST:$SSH_PORT."
      return 0
    fi
    sleep 10; waited=$((waited + 10))
    printf '.'
  done
  printf '\n'
  warn "Timed out after 10 minutes. The pod (ID $POD_ID) still exists and still bills."
  warn "Check the Runpod console, then run this script again, or run scripts/runpod_down.sh --delete-pod."
  exit 1
}

wait_for_ssh() {
  info "Waiting for sshd to accept your key (up to 3 minutes)..."
  local waited=0
  while (( waited < 180 )); do
    if rp_ssh -o BatchMode=yes true 2>/dev/null; then
      info "SSH works."
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  warn "SSH didn't accept the connection. Check that the public key is in the Runpod console under Settings > SSH Public Keys."
  return 1
}

push_secrets() {
  # The values travel on SSH stdin, not argv, and land in a mode-600 file.
  { set +x; } 2>/dev/null
  info "Writing /workspace/.secrets on the pod (mode 600)."
  {
    [[ -n "${HF_TOKEN:-}" ]]      && printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
    [[ -n "${WANDB_API_KEY:-}" ]] && printf 'export WANDB_API_KEY=%q\n' "$WANDB_API_KEY"
    true
  } | rp_ssh 'umask 077; cat > /workspace/.secrets.tmp && chmod 600 /workspace/.secrets.tmp && mv /workspace/.secrets.tmp /workspace/.secrets'
}

finish() {
  # Prints connection details and the billing reminder.
  local price="${COST_PER_HR:-}"
  info ""
  info "Pod $POD_ID is up."
  info "Connect with:"
  info "  ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST"
  [[ -n "$price" ]] && info "Cost: $price per hour (costPerHr from the Runpod API)."
  info "State saved to $STATE_FILE (IDs and SSH details only, no secrets)."
  info ""
  info "Next: scripts/sync_results.sh --push-code, then run scripts/pod_bootstrap.sh on the pod."
  info "REMINDER: the pod bills every hour until you stop or delete it."
  info "          Run scripts/runpod_down.sh when you finish."
}

# ---- reuse an existing pod from the state file --------------------------------
if [[ -n "${POD_ID:-}" ]]; then
  if refresh_pod; then
    case "$POD_STATUS" in
      RUNNING)
        info "Pod $POD_ID is already running. Refreshing connection details."
        wait_for_pod; wait_for_ssh && push_secrets; finish; exit 0 ;;
      EXITED)
        info "Pod $POD_ID exists but is stopped."
        if [[ "$ASSUME_YES" -eq 1 ]] || confirm "Start it again? Billing resumes."; then
          code="$(rp_call POST "/pods/$POD_ID/start" "$RP_TMP/start.json")" || true
          if [[ "$code" =~ ^2 ]]; then
            wait_for_pod; wait_for_ssh && push_secrets; finish; exit 0
          fi
          warn "Start failed (HTTP $code): $(rp_error_text "$RP_TMP/start.json")"
          warn "The GPU may be taken. Run scripts/runpod_down.sh --delete-pod, then run this script for a fresh pod on the same volume."
          exit 1
        fi
        exit 0 ;;
      *) info "Pod $POD_ID has status $POD_STATUS. Creating a new pod."; POD_ID="" ;;
    esac
  else
    info "Pod $POD_ID from the state file no longer exists. Creating a new pod."
    POD_ID=""; SSH_HOST=""; SSH_PORT=""
  fi
fi

# ---- step 1: find the network volume --------------------------------------------
CREATE_VOLUME=0
if [[ "$NO_NET_VOLUME" -eq 1 ]]; then
  [[ -n "${DATACENTER:-}" ]] || die "--no-network-volume needs --datacenter."
  VOLUME_ID=""
  info "Skipping the network volume: the pod gets a ${VOLUME_GB} GB volume disk at /workspace."
else
info "Looking for network volume '$VOLUME_NAME'..."
code="$(rp_call GET /networkvolumes "$RP_TMP/vols.json")" || true
[[ "$code" == "200" ]] || die "Listing network volumes failed (HTTP $code): $(rp_error_text "$RP_TMP/vols.json")"

found="$(jq -c --arg n "$VOLUME_NAME" \
  '(if type == "array" then . else (.networkVolumes // []) end) | map(select(.name == $n)) | first // empty' \
  "$RP_TMP/vols.json")"

CREATE_VOLUME=0
if [[ -n "$found" ]]; then
  VOLUME_ID="$(jq -r '.id' <<<"$found")"
  vol_dc="$(jq -r '.dataCenterId' <<<"$found")"
  if [[ -n "${DATACENTER:-}" && "$DATACENTER" != "$vol_dc" ]]; then
    warn "Volume lives in $vol_dc, so the pod uses $vol_dc instead of $DATACENTER."
  fi
  DATACENTER="$vol_dc"
  info "Found volume $VOLUME_ID in $DATACENTER."
else
  if [[ -z "${DATACENTER:-}" ]]; then
    info "No volume named '$VOLUME_NAME' exists, and you didn't pass --datacenter."
    info "Pick a datacenter that has H100 stock, then run again with --datacenter ID."
    if command -v runpodctl >/dev/null 2>&1; then
      info ""; info "Datacenters:"; runpodctl datacenter list || true
      info ""; info "GPU availability:"; runpodctl gpu list || true
    else
      info "Install runpodctl (brew install runpod/runpodctl/runpodctl) to list datacenters and"
      info "stock from the terminal, or check the Storage page in the Runpod console."
    fi
    exit 2
  fi
  CREATE_VOLUME=1
fi
fi

# ---- step 2: price, estimate, confirmation ----------------------------------------
PRICE=""
if command -v runpodctl >/dev/null 2>&1; then
  PRICE="$(runpodctl gpu list --include-unavailable 2>/dev/null \
    | jq -r --arg id "$GPU_ID" --arg pat "$GPU_PATTERN" \
        '[.. | objects
          | select(((.gpuId? // "") | tostring) == $id or ((.displayName? // "") | tostring | test($pat; "i")))
          | .securePricePerHr? | select(. != null)] | first // empty' 2>/dev/null)" || PRICE=""
fi

if [[ -n "$PRICE" ]]; then
  price_note="$PRICE per hour (Secure Cloud list price from runpodctl)"
  price_calc="$PRICE"
else
  price_note="unknown until the pod exists (the script prints costPerHr after creation)"
  price_calc="$FALLBACK_PRICE"
fi
est_cost="$(awk -v p="$price_calc" -v h="$HOURS_ESTIMATE" 'BEGIN { printf "%.2f", p * h }')"
max_hours="$(awk -v p="$price_calc" -v b="$BUDGET_USD" 'BEGIN { printf "%.1f", b / p }')"

info ""
info "Plan"
info "  GPU:         1x $GPU_ID, Secure Cloud, at least 16 vCPUs and 64 GB RAM"
info "  Datacenter:  $DATACENTER"
if [[ "$NO_NET_VOLUME" -eq 1 ]]; then
  info "  Volume:      ${VOLUME_GB} GB pod volume disk at /workspace (lost when the pod is deleted)"
elif [[ "$CREATE_VOLUME" -eq 1 ]]; then
  info "  Volume:      create '$VOLUME_NAME' (${VOLUME_GB} GB), mounted at /workspace"
else
  info "  Volume:      reuse '$VOLUME_NAME' ($VOLUME_ID), mounted at /workspace"
fi
info "  Image:       $IMAGE"
info "  Price:       $price_note"
info "  Estimate:    ${HOURS_ESTIMATE} hours x \$$price_calc = about \$$est_cost"
info "  Budget:      \$$BUDGET_USD lasts about $max_hours hours at this price"
[[ -z "$PRICE" ]] && info "               (estimate uses an assumed \$$FALLBACK_PRICE per hour)"
if [[ "$SECRETS_IN_ENV" -eq 1 ]]; then
  info "  Secrets:     HF_TOKEN and WANDB_API_KEY go into the pod env field. Runpod stores them"
  info "               in the pod configuration, visible in its console. Use --no-secrets-in-env to avoid that."
else
  info "  Secrets:     copied over SSH to /workspace/.secrets (mode 600); not in the pod config."
fi
info "  The pod bills until you stop or delete it."
info ""

if [[ "$ASSUME_YES" -ne 1 ]]; then
  confirm "Create these resources now?" || { info "Cancelled. Nothing was created."; exit 0; }
fi

# ---- step 3: create the volume (if needed) and the pod ----------------------------
if [[ "$CREATE_VOLUME" -eq 1 ]]; then
  info "Creating volume '$VOLUME_NAME' (${VOLUME_GB} GB) in $DATACENTER..."
  jq -n --arg n "$VOLUME_NAME" --argjson s "$VOLUME_GB" --arg d "$DATACENTER" \
    '{name: $n, size: $s, dataCenterId: $d}' > "$RP_TMP/vol_body.json"
  code="$(rp_call POST /networkvolumes "$RP_TMP/vol_resp.json" "$RP_TMP/vol_body.json")" || true
  [[ "$code" =~ ^2 ]] || die "Volume creation failed (HTTP $code): $(rp_error_text "$RP_TMP/vol_resp.json")"
  VOLUME_ID="$(jq -r '.id' "$RP_TMP/vol_resp.json")"
  [[ -n "$VOLUME_ID" && "$VOLUME_ID" != "null" ]] || die "Volume created, but the response had no id."
  info "Created volume $VOLUME_ID."
  GPU_ID="$GPU_ID_FOR_STATE"
  save_state
fi

info "Creating the pod..."
build_body real > "$RP_TMP/pod_body.json"
code="$(rp_call POST /pods "$RP_TMP/pod_resp.json" "$RP_TMP/pod_body.json")" || true
rm -f "$RP_TMP/pod_body.json"   # it holds secrets; delete it as soon as it has been sent
if [[ ! "$code" =~ ^2 ]]; then
  warn "Pod creation failed (HTTP $code): $(rp_error_text "$RP_TMP/pod_resp.json")"
  warn "A common cause is no H100 capacity in $DATACENTER. Try --gpu pcie, or wait and retry."
  warn "A volume stays in its datacenter. To use another datacenter, create a new volume there."
  exit 1
fi
POD_ID="$(jq -r '.id' "$RP_TMP/pod_resp.json")"
[[ -n "$POD_ID" && "$POD_ID" != "null" ]] || die "Pod created, but the response had no id."
COST_PER_HR="$(jq -r '.costPerHr // empty' "$RP_TMP/pod_resp.json")"
SSH_KEY="${PUBKEY_FILE%.pub}"
GPU_ID="$GPU_ID_FOR_STATE"
POD_STATUS="RUNNING"
SSH_HOST=""; SSH_PORT=""
save_state   # record the pod ID now, so a later failure can't orphan it
info "Created pod $POD_ID."

# ---- step 4: wait until SSH is reachable -------------------------------------------
wait_for_pod
if wait_for_ssh; then
  push_secrets
else
  warn "Skipping the secrets copy. Run this script again after SSH works."
fi
finish
