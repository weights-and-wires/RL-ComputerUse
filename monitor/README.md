# Run-3 monitor

The run-3 monitor runs unattended on Bazzite. It creates the Runpod H100 pod, starts
the pipeline, restarts what fails, replaces a lost pod, enforces the $50 budget, uploads
the results to Hugging Face, and deletes the pod. It also fills `~/cua-run3/data/`, which
the dashboard reads.

The contract with the pod side and the dashboard is in
[the run-3 interfaces page](../docs/RUN3_INTERFACES.md).

## Start a run

To start run 3, run the following command on your Mac:

```sh
scripts/run3_go.sh
```

The script syncs `monitor/` to Bazzite, prints the pod plan with live prices, updates
`~/cua-run3/repo`, and asks for confirmation. After you confirm, it writes
`~/cua-run3/ARMED` and starts the `cua-run3-monitor` user service, which creates the pod.

The script picks the code source as follows:

- If your checkout is clean and matches `origin/main`, Bazzite pulls `main` from GitHub.
- Otherwise, or if you pass `--from-mac`, the script copies your working tree: tracked
  and untracked files that Git doesn't ignore. It never copies `.env` files, dot-folders,
  or `results/`.

The script accepts the following options:

| Option | Effect |
|---|---|
| `--yes` | Skips the confirmation prompt. |
| `--adopt POD_ID` | Uses a pod that you created by hand as generation 1. |
| `--status` | Prints the monitor state, the service state, and recent events. |
| `--stop` | Ends GRPO early, waits for the post-training evals and `ALL DONE`, uploads, then deletes the pod. |
| `--kill` | Uploads once right away, then deletes the pod even if the upload fails. |

To connect to a pod by using SSH from your Mac, use your Runpod key
(`~/.ssh/id_ed25519`). The monitor installs it on every pod next to Bazzite's key. The
host and port are in the `--status` output.

## How the monitor works

The monitor is one Python process (standard library only) with three threads:

- The state machine, which polls about every 2 minutes.
- The collector, which fills `data/` every 60 seconds while a pod is up.
- The note writer, which asks `claude -p` for a short status note every 30 minutes.

### States

The state machine moves through the following states:

| State | What the monitor does |
|---|---|
| `idle` | Waits for `~/cua-run3/ARMED`. Without that file, the monitor never creates a pod, so the service is safe to start at boot. |
| `creating_pod` | Creates a pod. It tries H100 SXM, PCIe, and NVL on Secure Cloud, then the same on Community Cloud, and records which one it got. |
| `bootstrapping` | Waits for SSH, writes `/workspace/.secrets` through stdin, ships `~/cua-run3/repo` with `tar` over SSH, and runs `scripts/run3_pod_start.sh`. |
| `running` | Runs `scripts/run3.sh --status-json` over SSH and records phase changes. |
| `restarting` | Reruns `run3_pod_start.sh` after the `run3.sh` process dies. The pipeline resumes and skips finished phases. |
| `recreating_pod` | Deletes the lost pod if it still exists, then creates the next generation, which restores from Hugging Face. |
| `stopping` | Like `running`, after a graceful stop request. |
| `uploading` | Runs `python scripts/upload_run3.py --final --verify` on the pod, up to 2 attempts. |
| `deleting` | Deletes the pod and confirms through the API that it's gone. |
| `done`, `failed` | Final states. The monitor removes `ARMED`. |

### Failure handling

The monitor handles failures as follows:

- **Crash:** if `run3.sh` isn't running and the pipeline isn't done, the monitor restarts
  it. If the same phase dies 3 times in a row without progress, the monitor stops
  retrying and tears down. A finished phase or a higher GRPO step resets the count.
- **Lost pod:** if the API returns 404, `TERMINATED`, `EXITED`, or `ERROR`, or if SSH
  fails for 15 minutes while the API reports `RUNNING`, the monitor replaces the pod. It
  caps pods at 4 generations.
- **Broken status:** if `--status-json` prints an error object or no JSON, the monitor
  logs it and waits. It doesn't treat that as a crash.
- **No stock:** the monitor retries every 20 seconds and fails after 30 attempts. No pod
  exists in that case, so nothing bills.

### Budget

The monitor computes spend as each pod's hourly rate (the API's `cost` field) times its
wall-clock life, summed over all generations. It enforces the following limits:

| Limit | Action |
|---|---|
| $42 | Touches `results/run3/.stop_grpo` once (event `budget_soft_stop`). GRPO saves and ends, and the post-training evals run. |
| $48, or 20 hours after the start | Tears down at once, whatever the state. |

The monitor deletes a pod only after a verified upload, with two exceptions. If the
upload fails and spend is $48 or more, it deletes the pod anyway and records an `error`
event that says so. If the upload fails below $48, it leaves the pod running, marks the
run `failed`, and records a loud `error` event; you decide what to do. The monitor still
deletes that pod if spend later reaches $48.

The monitor deletes only pods that it created or adopted. It stores their IDs in
`state.json` and refuses any other ID.

## Files

The monitor uses the following files on Bazzite:

| Path | Contents |
|---|---|
| `~/cua-run3/.env` | `RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, and `HF_ADAPTER_REPO`. Mode 600. |
| `~/cua-run3/ARMED` | Present while a run is allowed to create pods. `run3_go.sh` writes it. |
| `~/cua-run3/state.json` | Durable monitor state: state, pods, spend per generation, and restart counts. Mode 600. |
| `~/cua-run3/command.json` | A queued `stop` or `kill` command. |
| `~/cua-run3/config.json` | Optional overrides for the defaults in `DEFAULT_CONFIG` in `monitor.py`. |
| `~/cua-run3/monitor.log` | The monitor log. `journalctl --user -u cua-run3-monitor` has the same lines. |
| `~/cua-run3/authorized_keys.extra` | Your Mac's public key, added to every pod. |
| `~/cua-run3/repo/` | The code that the monitor ships to pods. |
| `~/cua-run3/venv/` | A Python venv with `wandb`, used only for W&B fetches. |
| `~/cua-run3/data/` | The dashboard data, as the interfaces page describes. |

The collector also writes `data/wandb_system.jsonl`, the W&B system metrics history
(GPU use, memory, and power). It adds W&B metric rows that are missing from
`metrics.jsonl` and marks them with `"_source": "wandb"`. A W&B failure is only logged.

All files in `data/` pass through the secret scrubber and are written atomically.

The monitor code has the following parts:

| File | Purpose |
|---|---|
| `monitor.py` | State machine and command-line interface. |
| `collector.py` | Collector and note writer threads. |
| `pod_probe.py` | Runs on the pod through SSH stdin and gathers collector data in one round trip. |
| `runpod.py` | Runpod REST v2 client. |
| `podssh.py` | SSH helpers. |
| `common.py` | Atomic writes, scrubbing, eval parsing, and Wilson intervals. |
| `wandb_fetch.py` | W&B public API fetch, run in the venv. |
| `demo_data.py` | Writes demo data from run 2 for dashboard work. |
| `fakes.py` | Fake Runpod API and fake pod for the dry run and the tests. |
| `prompt.md` | The prompt for status notes. |
| `deploy.sh` | Installs the monitor on Bazzite from your Mac. |

## Install or update

To install or update the monitor on Bazzite, run the following command on your Mac:

```sh
monitor/deploy.sh --env
```

The script syncs the code, copies `.env` with mode 600, builds the venv, and installs
and enables the user unit. It doesn't start the service. Omit `--env` to keep the `.env`
that's on Bazzite. Add `--demo` to regenerate the demo data.

## Test

To run the tests on any machine with Python 3.10 or later, run the following commands:

```sh
python3 -m unittest discover -s monitor/tests
python3 monitor/monitor.py --dry-run
```

The dry run drives the state machine against a fake API and a fake pod through
create, run, crash, restart, pod loss, recreation, soft stop, done, upload, and delete,
and then through a hard budget stop.

## Demo data

Until a real run starts, `data/` holds demo data built from run 2. `data/state.json`
has `"demo": true`, and `data/.demo` lists the demo files. When the monitor leaves
`idle` for a real run, it deletes them.

## Recover from problems

- **Check the state:** run `scripts/run3_go.sh --status`, or on Bazzite run
  `python3 ~/cua-run3/monitor/monitor.py status`.
- **Monitor process crashed:** systemd restarts it after 30 seconds, and it resumes
  from `state.json`. A pod creation that the crash interrupted is found by its unique
  name, so the monitor doesn't create a second pod.
- **Run failed with a pod left running:** read the last `error` events in
  `data/events.jsonl`. Connect to the pod and fix the upload by hand, or run
  `scripts/run3_go.sh --kill` to upload once more and delete the pod.
- **Run a new attempt after `done` or `failed`:** run `scripts/run3_go.sh` again. It
  archives the old `state.json` as `state.TIMESTAMP.json`.
- **Stop the monitor without touching the pod:** run
  `systemctl --user stop cua-run3-monitor`. The pod keeps billing until you start the
  monitor again or delete the pod.
- **Change limits:** put overrides in `~/cua-run3/config.json`, for example
  `{"hard_stop_usd": 45}`, then restart the service.
