# Run it yourself

This page is the short version of [docs/RUNBOOK.md](docs/RUNBOOK.md). It covers
the path that was tested end to end: one H100 PCIe pod in Runpod datacenter
US-KS-2, about $2.89 per hour. The full run takes about 9 GPU hours, roughly
$26.

For run 2 (GRPO continued from the run-1 adapter), see the launch commands in
[docs/PLAN_RUN2.md](docs/PLAN_RUN2.md) instead.

Commands marked **Mac** run from the repository root on your Mac. Commands
marked **Pod** run over SSH on the pod.

## 1. Create the pod

**Mac:**

```bash
scripts/runpod_up.sh --datacenter US-KS-2 --gpu pcie --no-network-volume --no-secrets-in-env
```

The script shows the price and asks you to type `y`. It waits until SSH works,
then copies your Hugging Face and W&B tokens to the pod over SSH.

US-KS-2 is the only datacenter with H100 PCIe stock, and it doesn't support
network volumes, so `--no-network-volume` gives the pod a 100 GB volume disk
at `/workspace` instead. That disk survives a pod stop but not a delete. To
use an H100 SXM ($4.39 per hour) with a network volume, replace the flags with
`--datacenter US-NE-1` and drop `--gpu pcie --no-network-volume`.

## 2. Copy the code and set up the pod

**Mac:**

```bash
scripts/sync_results.sh --push-code
```

```bash
source .runpod/state.env && ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST"
```

**Pod:**

```bash
tmux new -s run
```

```bash
bash /workspace/RL-ComputerUse/scripts/pod_bootstrap.sh
```

Setup takes about 2-3 minutes. It ends with `132 passed`, an accessibility tree
for `click-button`, and `Bootstrap finished`. The Python environment and model
weights live on the pod's local disk (`/root/cua`), so run the bootstrap again
after any pod restart.

## 3. Run the whole pipeline

**Pod**, inside `tmux`:

```bash
source /workspace/env.sh && cd /workspace/RL-ComputerUse
```

```bash
PRICE_PER_HR=2.89 bash scripts/saturday.sh
```

This runs every phase in order: smoke test, sampled baseline, task selection,
base evals, warm start, GRPO, final evals, and the report. It skips phases that
already finished, so if anything stops, run the same command again.

To detach from `tmux` and leave it running, press `Ctrl-b`, then `d`. To come
back, run `tmux attach -t run`. To see which phases are done and the spend so
far, run `bash scripts/saturday.sh --status`.

## 4. Watch the run

- **W&B:** open the `rl-computer-use` project. The charts that matter are listed
  in section 9 of [docs/LEARNING.md](docs/LEARNING.md).
- **Logs:** `results/logs/PHASE.log` on the pod, for example
  `tail -f results/logs/grpo.log`.
- In the first 30 minutes of GRPO, check that a step takes under 3 minutes and
  that `reward/total` isn't NaN. If a step is slower, see "a step takes more
  than 3 minutes" in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## 5. Get the results and stop paying

**Mac:**

```bash
scripts/sync_results.sh --adapters
```

```bash
open results/report/index.html
```

```bash
scripts/runpod_down.sh --delete-pod
```

The `push` phase uploads the final LoRA adapter to the private Hugging Face
repo `weights-and-wires/qwen3vl-4b-miniwob-grpo`. To skip the upload, run the
pipeline with `HF_ADAPTER_REPO=""`.

`runpod_down.sh` copies results to your Mac first. Use `--stop` instead of
`--delete-pod` if you want to come back to the same disk later; a stopped pod
still bills a small amount for its disk. Check the Runpod console afterward to
confirm no pod is running.

## What to expect

These numbers come from the test run on the same pod type:

- **Sampled baseline:** about 1 second per episode with 16 browsers, so the
  40-task baseline (640 episodes) takes 10-15 minutes.
- **GRPO step:** 2-2.5 minutes for 64 episodes (60-90 s of rollouts, 10-60 s
  of training), not the 1 minute the plan estimated. In the default 5-hour
  budget you get about 120-140 steps, not 200.
- **GPU memory:** about 39 GB of 80 GB during GRPO.
- **The base model is strong.** Qwen3-VL-4B solved most easy tasks 80-100% of
  the time, and those teach GRPO nothing. The candidate pool includes 15
  harder tasks for this reason; 6 of them landed in the useful 10-80% band in
  testing. Watch `groups/zero_variance_frac` in W&B: if it stays above 0.7,
  most steps aren't learning.
- **Results from the test run** are in `results_test_run/`, kept apart from
  your real run's `results/`.

## If something goes wrong

| Symptom | What to do |
|---|---|
| `runpod_up.sh` says there's no stock | Try `--gpu sxm` with `--datacenter US-NE-1` (drop `--no-network-volume`), or wait and retry. |
| The bootstrap fails | Read `/workspace/logs/bootstrap.log`, fix the cause, and run the bootstrap again. It's safe to rerun. |
| A phase crashes | Read `results/logs/PHASE.log`, then rerun `bash scripts/saturday.sh`. |
| Fewer than 8 tasks pass task selection | `SELECT_MIN=0.05 SELECT_MAX=0.90 bash scripts/saturday.sh` |
| GRPO is too slow or runs out of memory | `MODEL=2b bash scripts/saturday.sh --from base_eval` switches to the 2B model. |
