# Run 3 interfaces

This page fixes the contracts between the three parts of run 3, so each part can be
built and tested on its own:

- **Pod side** (`scripts/run3*.sh`, `scripts/*run3*.py`, `cua/`, `configs/`): trains and
  evaluates on a Runpod H100.
- **Monitor** (`monitor/`, runs on the Bazzite home machine): creates and watches the
  pod, restarts what fails, enforces the budget, uploads, and deletes the pod.
- **Dashboard** (`dashboard/`, served from Bazzite, public at `https://run.ifkash.dev`
  through Caddy on `vm.ifkash.dev`): shows the run live.

Decisions agreed with the owner on 2026-11-24:

- Run 3 trains **Track B** (screenshot only, `coordinate: [x, y]` actions) starting from
  the run-2 adapter, on a re-screened, mixed task pool (both earlier pools) to guard
  against forgetting. The model stays Qwen3-VL-4B.
- GPU: 1x H100 80GB on Runpod, container disk only (no network volume). **Budget cap
  $50** for the whole run, enforced by the monitor.
- If the pod disappears, the monitor creates a new pod and resumes from the last
  checkpoint synced to Hugging Face.
- The dashboard is public. It never shows a secret.

## Pod layout

| Path on the pod | Meaning |
|---|---|
| `/root/RL-ComputerUse` | The repository. The monitor copies it from its own checkout of `main` with `tar` over SSH. |
| `/workspace/.secrets` | `export NAME=value` lines (`HF_TOKEN`, `WANDB_API_KEY`, `HF_ADAPTER_REPO`), mode 600, written by the monitor. |
| `/workspace/env.sh` | Written by `pod_bootstrap.sh`; sources `.secrets`. |
| `/workspace/ckpt/grpo-4b-run3/checkpoint-N` | GRPO checkpoints. |
| `results/run3/` (in the repo) | Everything run 3 writes (`CUA_RESULTS_DIR`). |
| `results/run3/run.log` | The whole-run log. Its last line is `ALL DONE` on success. |
| `results/run3/.phases/<phase>.done` | Phase markers (UTC timestamp inside). |
| `results/run3/logs/<phase>.log` | Per-phase logs. |
| `results/run3/train/grpo-4b-run3/metrics.jsonl` | One JSON object per training step (same keys as run 2, plus any new ones). |
| `results/run3/eval/<name>/per_task.csv`, `summary.json`, `episodes.jsonl` | Eval outputs, as in run 2. |
| `results/run3/samples/step-NNNNNN/` | Sample rollouts for the dashboard: `episode.json` plus `turn-K.jpg` (at most 640 px wide, JPEG quality 70). |
| `results/run3/wandb_run.json` | `{"entity", "project", "id", "url"}` of the GRPO W&B run, written when GRPO starts. |
| `results/run3/.stop_grpo` | If this file exists, GRPO saves a checkpoint and ends at the next step boundary, and the pipeline goes on to post-training evals. |

`episode.json` in a sample folder:

```json
{"step": 120, "task": "click-button", "seed": 100123, "reward": 1.0, "success": true,
 "turns": [{"image": "turn-0.jpg", "response": "<model text>", "action": "click(coordinate=[x, y])", "error": ""}]}
```

## Pod-side commands the monitor calls

Every command runs as `ssh root@<pod> 'cd /root/RL-ComputerUse && <command>'`.

| Command | Contract |
|---|---|
| `bash scripts/run3_pod_start.sh` | Idempotent. If tmux session `run3` is alive, prints `ALREADY_RUNNING` and exits 0. Otherwise: bootstraps the pod if `/workspace/.bootstrapped` is missing, restores `results/run3/` and the newest checkpoint from Hugging Face if they are missing locally, starts the sync loop (tmux session `run3-sync`), and starts `scripts/run3.sh` (resumable, skips done phases) in tmux session `run3`. Prints `STARTED` and exits 0. Exits non-zero with a message on failure. |
| `bash scripts/run3.sh --status-json` | Prints one JSON object (schema below) and exits 0. Must be fast (under 5 s) and must not import torch. |
| `touch results/run3/.stop_grpo` | Graceful early end of GRPO (budget). |
| `python scripts/upload_run3.py --final --verify` | Uploads everything (checkpoints, final adapter, `results/run3/`, W&B files) to `HF_ADAPTER_REPO` under `checkpoints/grpo-4b-run3/` and `results/run3/`, then verifies every local file exists on the Hub with the same size. Exit 0 only if verified. |

`--status-json` schema:

```json
{
  "run": "run3",
  "done": false,
  "all_done_line": false,
  "running": true,
  "current_phase": "grpo",
  "failed_phase": null,
  "phases": [{"name": "baseline", "status": "done|running|pending|failed", "started": "ISO-8601 or null", "finished": "ISO-8601 or null"}],
  "grpo": {"step": 120, "max_steps": 400, "last_checkpoint": 100, "synced_checkpoint": 100},
  "elapsed_s": 12345,
  "gpu": {"util": 97, "mem_used_mb": 61000, "mem_total_mb": 81559, "power_w": 610}
}
```

The sync loop uploads each new checkpoint and the small files under `results/run3/`
(markers, logs, metrics, evals, samples) to Hugging Face at least every 10 minutes and
right after a phase finishes, and records the last synced checkpoint in
`results/run3/.synced_checkpoint`. A recreated pod restores from there.

## Monitor state on Bazzite

Home: `~/cua-run3/` on Bazzite (`ifkash@192.168.1.4`, Tailscale `100.115.46.42`).

| Path | Meaning |
|---|---|
| `~/cua-run3/.env` | Copy of the Mac repo's `.env` (`RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `HF_ADAPTER_REPO`). Mode 600. |
| `~/cua-run3/repo/` | Checkout of `main` that the monitor ships to pods. |
| `~/cua-run3/state.json` | Monitor's durable state (survives restarts of the monitor itself). |
| `~/cua-run3/data/` | Everything the dashboard reads (schemas below). Written atomically (write to a temp file, then rename). |

Files in `~/cua-run3/data/`:

| File | Contents |
|---|---|
| `state.json` | `{"run_state": "idle|creating_pod|bootstrapping|running|restarting|recreating_pod|stopping|uploading|deleting|done|failed", "updated": ISO, "started": ISO, "phases": [...from --status-json], "current_phase", "grpo": {...}, "pod": {"id", "gpu", "cost_per_hr", "uptime_s", "generation"}, "spend_usd", "budget_usd": 50, "eta": ISO or null, "restarts": N, "pod_generations": N, "wandb_url", "hf_url", "last_error"}` |
| `metrics.jsonl` | Copy of the pod's GRPO `metrics.jsonl` (append-only). |
| `system.jsonl` | One line per minute: `{"t": ISO, "gpu_util", "gpu_mem_mb", "gpu_power_w", "spend_usd"}`. |
| `evals.json` | `{"<eval name>": {"mode", "track", "tasks": {"<task>": {"success": 0.62, "n": 100}}, "mean": 0.62, "ci95": [0.58, 0.66]}}` |
| `log.txt` | Last 2,000 lines of `run.log`, carriage returns turned into newlines, secrets scrubbed. |
| `events.jsonl` | Timeline: `{"t": ISO, "kind": "pod_created|phase_started|phase_done|restart|pod_lost|pod_recreated|budget_soft_stop|upload_verified|pod_deleted|error|note", "msg": "..."}` |
| `notes.json` | `[{"t": ISO, "text": "markdown"}]` short status notes, newest last. |
| `samples/step-NNNNNN/` | Mirror of the pod's sample folders (newest 30 kept). |
| `samples/index.json` | `[{"step", "task", "success", "reward", "dir"}]` newest first. |

Secret scrubbing: before writing anything into `data/`, replace every value of
`RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, and anything matching `hf_[A-Za-z0-9]{20,}`,
`rpa_[A-Za-z0-9]{20,}`, or a 40-hex W&B key, with `[redacted]`.

## Dashboard

- Server: a small Python process on Bazzite, user systemd unit `cua-run3-dashboard`,
  listening on `0.0.0.0:8790`. It serves `dashboard/` static files and `~/cua-run3/data/`
  under `/data/`, plus `GET /api/log/stream` (server-sent events of new `log.txt` lines).
- Public URL: `https://run.ifkash.dev`, a Caddy site block on `vm.ifkash.dev` with
  `reverse_proxy 100.115.46.42:8790`, and a Cloudflare DNS record for `run.ifkash.dev`
  that matches how `vm.ifkash.dev` resolves (DNS only, so Caddy gets the certificate).

## Changes by the pod-side agent

The pod side implements the contract as written, with the following additions
and clarifications. None of them changes a command, a path, or a schema field
that the monitor or the dashboard reads.

- **Phase names.** The `phases` list in `--status-json` has these names, in
  order: `smoke`, `screen`, `select`, `rft`, `pre_eval`, `grpo`, `post_eval`,
  `compare`, and `report`. The `baseline` in the schema example is only an
  example. The `rft` phase is usually a quick skip.
- **Extra files under `results/run3/.phases/`.** `<phase>.started` holds the
  start of the latest attempt, `<phase>.failed` the time and exit code of a
  failed attempt, `.current` the running phase, and `.pid` the PID of
  `run3.sh`. `grpo.elapsed_s` holds the GRPO time summed over every pod, and
  `start_adapter` and `final_adapter` hold adapter paths. The sync loop never
  uploads `.pid`, `.current`, or `results/run3/.sync_state.json`.
- **Status fields.** `running` is true while the PID in `.pid` is alive. When
  nothing runs, `current_phase` is the first unfinished phase, or `null` when
  every phase is done. `grpo.step` and `grpo.last_checkpoint` are `null`
  before GRPO starts. `gpu` fields are `null` when `nvidia-smi` isn't
  available. The status reads `RUN3_RESULTS` and `RUN3_CKPT` when they're set.
- **Bootstrap check.** `run3_pod_start.sh` bootstraps when
  `/workspace/.bootstrapped` is missing, and also when the venv on `/root` or
  `/workspace/env.sh` is missing, because a stopped pod keeps `/workspace` but
  loses `/root`.
- **Inputs on the pod.** The run-2 adapter goes to
  `/workspace/inputs/run2-adapter`, and run 2's result files go to
  `results/run2/`. MiniWoB lives on `/root/miniwob-plusplus`. Bootstrap and
  restore logs are in `/workspace/logs/`. When set in the SSH command, these
  variables pass through to `run3.sh`: `GRPO_MAX_HOURS`, `RFT`,
  `PRICE_PER_HR`, `BEST_CKPT`, `CUA_SETS`, `RUN2_ADAPTER`, `RUN2_HF_PATH`,
  `FORCE_EVAL`, and `SYNC_INTERVAL_S`.
- **Sync timing.** The sync loop uploads every 5 minutes, and within about 20
  seconds of a new checkpoint or a new `.done` marker. Its log is
  `results/run3/logs/sync.log`. It exits after one last pass once `run.log`
  ends with `ALL DONE`.
- **Remote paths.** Besides `checkpoints/grpo-4b-run3/checkpoint-N/` and
  `checkpoints/grpo-4b-run3/final/`, the RFT adapter goes to
  `checkpoints/rft-4b-run3/final/` (only if the `rft` phase ran). The W&B
  files of the GRPO run go to `results/run3/wandb/<run folder>/`.
- **`metrics.jsonl` after a resume.** When GRPO resumes from checkpoint N, it
  drops rows with a step above N, because those steps run again. A copy that
  only appends sees those step numbers twice. Keep the last row for each step.
  Rows that hold only `{"step", "eval/success"}` are mid-training evals.
- **Samples.** GRPO writes one sample every 5 steps, from a mixed group when it
  can, alternating between a success and a failure. Steps that run again after
  a resume replace their sample folder.
- **`upload_run3.py`.** `--final` first stops tmux session `run3-sync`, so two
  processes don't commit at the same time. The last line of output is
  `VERIFIED` when the check passes and `NOT VERIFIED` when it fails.
- **Early stop flag.** If `.stop_grpo` exists before GRPO's first step, GRPO
  saves the start adapter as its final adapter, and the post-training evals
  run on it.
