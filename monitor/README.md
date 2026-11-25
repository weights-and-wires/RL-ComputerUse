# Run monitor

The run monitor runs unattended on Bazzite. It creates the Runpod H100 pod, starts
the pipeline, restarts what fails, replaces a lost pod, enforces the $50 budget, uploads
the results to Hugging Face, and deletes the pod. It also fills `~/cua-run4/data/`, which
the dashboard reads.

The monitor works for any run. One setting, `run` (default `run4`), decides every pod
path, script name, tmux session, pod name, and label. For `run4`, the monitor uses
`scripts/run4.sh`, `scripts/run4_pod_start.sh`, `scripts/upload_run4.py`, the tmux
sessions `run4` and `run4-sync`, `results/run4/`, and the home folder `~/cua-run4/`.

The monitor picks the run in this order: the `--run` option, the `CUA_RUN` environment
variable, the folder that holds the code (code in `~/cua-run3/monitor/` means `run3`),
then `run4`. The contract with the pod side and the dashboard is in
[the run-3 interfaces page](../docs/RUN3_INTERFACES.md), with the names above
substituted for run 4.

## Start a run

To start run 4, run the following command on your Mac:

```sh
scripts/run4_go.sh
```

The script syncs `monitor/` to Bazzite, prints the pod plan with live prices and your
Runpod balance, updates `~/cua-run4/repo`, and asks for confirmation. After you confirm,
it checks that `scripts/run4_pod_start.sh`, `scripts/run4.sh`, and `scripts/upload_run4.py`
exist in the repo, writes `~/cua-run4/ARMED`, and starts the `cua-run4-monitor` user
service, which creates the pod. The monitor checks the same three files again before it
creates a pod, and it fails the run without a pod if one is missing.

`scripts/run3_go.sh` still works. It runs the same script with `RUN=run3`.

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
| `idle` | Waits for `~/cua-run4/ARMED`. Without that file, the monitor never creates a pod, so the service is safe to start at boot. |
| `creating_pod` | Checks your Runpod balance, then creates a pod by the [GPU policy](#gpu-policy), and records which GPU it got. |
| `bootstrapping` | Waits for SSH, writes `/workspace/.secrets` through stdin, ships `~/cua-run4/repo` with `tar` over SSH, and runs `scripts/run4_pod_start.sh`. |
| `running` | Runs `scripts/run4.sh --status-json` over SSH and records phase changes. |
| `restarting` | Reruns `run4_pod_start.sh` after the `run4.sh` process dies. The pipeline resumes and skips finished phases. |
| `recreating_pod` | Deletes the lost pod if it still exists, then creates the next generation, which restores from Hugging Face. |
| `stopping` | Like `running`, after a graceful stop request. |
| `uploading` | Runs `python scripts/upload_run4.py --final --verify` on the pod, up to 2 attempts. |
| `deleting` | Deletes the pod and confirms through the API that it's gone. |
| `done`, `failed` | Final states. The monitor removes `ARMED`. |

### Failure handling

The monitor handles failures as follows:

- **Crash:** if `run4.sh` isn't running and the pipeline isn't done, the monitor restarts
  it. If the same phase dies 3 times in a row without progress, the monitor stops
  retrying and tears down. A finished phase or a higher GRPO step resets the count.
- **Lost pod:** if the API returns 404, `TERMINATED`, `EXITED`, or `ERROR`, or if SSH
  fails for 15 minutes while the API reports `RUNNING`, the monitor replaces the pod. It
  caps pods at 4 generations.
- **Broken status:** if `--status-json` prints an error object or no JSON, the monitor
  logs it and waits. It doesn't treat that as a crash.
- **No stock:** the monitor retries every 20 seconds and fails after 120 attempts. No pod
  exists in that case, so nothing bills.

### GPU policy

The monitor creates a pod in this order:

1. Community Cloud H100 SXM. If it has no stock, the monitor tries Community PCIe, then
   Community NVL, and starts over at SXM. It keeps retrying for 20 minutes
   (`community_wait_s`).
2. After the wait, Secure Cloud in the same order: SXM, PCIe, NVL. Community stays on
   the list behind Secure as a last resort.

The monitor records a `gpu_fallback` event the first time it skips a GPU and when it
moves from Community to Secure. The `pod_created` event says when the pod is a fallback.

### Balance check

Before it creates any pod, including a replacement, the monitor asks Runpod for your
account balance (the GraphQL query `myself { clientBalance }`). It refuses to create the
pod when the balance is lower than the hard stop minus the spend so far, plus $2. In that
case it records an `error` event with the numbers, sets the state to `failed`, and
removes `ARMED`. If the query fails 5 times in a row, the monitor also fails the run
instead of creating a pod it can't check. The monitor never prints the API key.
`scripts/run4_go.sh` shows the balance on its confirmation screen.

### Budget

The monitor computes spend as each pod's hourly rate (the API's `cost` field) times its
wall-clock life, summed over all generations. It enforces the following limits:

| Limit | Action |
|---|---|
| $42 | Touches `results/run4/.stop_grpo` once (event `budget_soft_stop`). GRPO saves and ends, and the post-training evals run. |
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
| `~/cua-run4/.env` | `RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, and `HF_ADAPTER_REPO`. Mode 600. |
| `~/cua-run4/ARMED` | Present while a run is allowed to create pods. `run4_go.sh` writes it. |
| `~/cua-run4/state.json` | Durable monitor state: state, pods, spend per generation, and restart counts. Mode 600. |
| `~/cua-run4/command.json` | A queued `stop` or `kill` command. |
| `~/cua-run4/config.json` | Optional overrides for the defaults in `DEFAULT_CONFIG` in `monitor.py`. |
| `~/cua-run4/monitor.log` | The monitor log. `journalctl --user -u cua-run4-monitor` has the same lines. |
| `~/cua-run4/authorized_keys.extra` | Your Mac's public key, added to every pod. |
| `~/cua-run4/repo/` | The code that the monitor ships to pods. |
| `~/cua-run4/venv/` | A Python venv with `wandb`, used only for W&B fetches. |
| `~/cua-run4/data/` | The dashboard data, as the interfaces page describes. |

The collector also writes `data/wandb_system.jsonl`, the W&B system metrics history
(GPU use, memory, and power). It adds W&B metric rows that are missing from
`metrics.jsonl` and marks them with `"_source": "wandb"`. A W&B failure is only logged.

The collector copies `metrics.jsonl` by byte offset and appends new rows as they are. A
step number can appear twice, because a mid-training eval row shares the step of a
training row; that's normal and the collector keeps both. The collector replaces rows
only when the pod file shrank or was rewritten, which happens when GRPO resumes from a
checkpoint or when a new pod restores the file. The pod probe reports that as `reset`.
In that case the copy keeps the rows that match the pod's file and takes the pod's rows
from the first difference on, and the collector records a `GRPO resumed` note.

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

The script syncs the code to `~/cua-run4/monitor`, copies `.env` with mode 600, builds the
venv, and installs and enables the `cua-run4-monitor` user unit. It doesn't start the
service and doesn't arm it. Omit `--env` to keep the `.env` that's on Bazzite. Add
`--demo` to write demo data, and `--run NAME` to install a different run.

## Dashboard

The dashboard is a read-only server on Bazzite (user unit `cua-dashboard`, port 8790,
public at `https://run.ifkash.dev`). It shows the current run at `/` and every earlier
run in its own tab: `/run3/` (from `~/cua-run3/data`), `/run2/`, and `/run1/` (static
snapshots in `~/cua-runs/<run>/data`). Until run 4 starts, the Run 4 tab shows a
"hasn't started yet" page with the plan, instead of another run's data.

To install or update the dashboard, run the following command on your Mac:

```sh
dashboard/deploy.sh --snapshots
```

The `--snapshots` option rebuilds the run 1 and run 2 snapshots from `results/` with
`dashboard/snapshot_runs.py` and copies them to Bazzite. The script also replaces the old
`cua-run3-dashboard` unit, which it disables but doesn't delete.

The page reads the run name from `data/state.json` (`run` and `run_label`). It separates
mid-training exam rows from training rows by the `kind` field, or by their keys when
there's no `kind` (run 3 data). It pairs exams by a shared name suffix, for example
`pre-pool` with `post-pool`.

## Test

To run the tests on any machine with Python 3.10 or later, run the following commands:

```sh
python3 -m pytest -q monitor/tests        # or: cd monitor && python3 -m unittest discover -s tests
python3 monitor/monitor.py --dry-run
```

The dry run drives the state machine against a fake API and a fake pod through
create, run, crash, restart, pod loss, recreation, soft stop, done, upload, and delete;
then a hard budget stop; then the Community-to-Secure fallback; then a refusal because
of a low balance.

## Demo data

Until a real run starts, `data/` holds demo data built from run 2. `data/state.json`
has `"demo": true`, and `data/.demo` lists the demo files. When the monitor leaves
`idle` for a real run, it deletes them. The dashboard treats demo data as "not started".

## Recover from problems

- **Check the state:** run `scripts/run4_go.sh --status`, or on Bazzite run
  `python3 ~/cua-run4/monitor/monitor.py status`.
- **Monitor process crashed:** systemd restarts it after 30 seconds, and it resumes
  from `state.json`. A pod creation that the crash interrupted is found by its unique
  name, so the monitor doesn't create a second pod.
- **Run failed with a pod left running:** read the last `error` events in
  `data/events.jsonl`. Connect to the pod and fix the upload by hand, or run
  `scripts/run4_go.sh --kill` to upload once more and delete the pod.
- **Run a new attempt after `done` or `failed`:** run `scripts/run4_go.sh` again. It
  archives the old `state.json` as `state.TIMESTAMP.json`.
- **Stop the monitor without touching the pod:** run
  `systemctl --user stop cua-run4-monitor`. The pod keeps billing until you start the
  monitor again or delete the pod.
- **Change limits:** put overrides in `~/cua-run4/config.json`, for example
  `{"hard_stop_usd": 45}`, then restart the service.
