# Runbook: Saturday 2026-11-21

This runbook is the hour-by-hour checklist for training day. It follows the
schedule in [PLAN.md](PLAN.md). Commands marked **Mac** run in a terminal on
your Mac from the repository root. Commands marked **Pod** run over SSH on the
Runpod pod, inside `tmux`.

The core run takes about 9 GPU hours, roughly $27 at $2.99 per hour. The budget
is $50.

## Secrets handling

- Your keys live in `~/.api_keys` on the Mac as `export NAME=value` lines:
  `RUNPOD_API_KEY`, `HF_TOKEN`, and `WANDB_API_KEY`.
- The scripts `source` that file at runtime. They never print, log, or write
  the values into the repository.
- `scripts/runpod_up.sh` sends `HF_TOKEN` and `WANDB_API_KEY` to the pod in the
  `env` field of the create request. Runpod stores that field in the pod
  configuration, and it's visible in the Runpod console. To avoid that, pass
  `--no-secrets-in-env`. In both modes the script also writes
  `/workspace/.secrets` (mode 600) over SSH, and `/workspace/env.sh` sources
  it. The values go through SSH stdin, not the command line, so `ps` can't
  show them.
- Delete or rotate the HF and W&B tokens if you share the pod's console view.
- `HF_TOKEN` needs write access only if you want the push phase to upload the
  adapter. A read-only token still downloads the model. If the token is
  read-only, keep the adapter with `scripts/sync_results.sh --adapters`.

## Friday evening prep

Complete the following steps on Friday 2026-11-20.

1. Open `~/.api_keys` and confirm it has all three lines. Don't paste values
   into chat or into the repository.

   ```bash
   export RUNPOD_API_KEY=...
   export HF_TOKEN=...
   export WANDB_API_KEY=...
   ```

   Check that the names exist without printing the values:

   ```bash
   grep -c '^export \(RUNPOD_API_KEY\|HF_TOKEN\|WANDB_API_KEY\)=' ~/.api_keys
   ```

   The output is `3`.
2. In the Runpod console, go to **Settings > SSH Public Keys** and add the
   contents of `~/.ssh/id_ed25519.pub`. If you don't have that key, run
   `ssh-keygen -t ed25519`.
3. In the Runpod console, check that your credit is at least $50.
4. Install the tools the scripts need:

   ```bash
   brew install jq
   brew install runpod/runpodctl/runpodctl   # optional: datacenter list, GPU stock, and price
   ```

5. Optional: if you installed `runpodctl`, find a datacenter with H100 stock
   and write down its ID:

   ```bash
   runpodctl datacenter list
   runpodctl gpu list
   ```

6. Read [LEARNING.md](LEARNING.md), especially section 9, "How to read the
   training dashboard".
7. Run a dry run of the pod request. It makes no API calls and prints the
   request body with secrets redacted:

   ```bash
   scripts/runpod_up.sh --datacenter DATACENTER_ID --dry-run
   ```

   Replace `DATACENTER_ID` with the ID from step 5, for example `EU-RO-1`.
   Check that the output shows `"cloudType": "SECURE"`,
   `"minVCPUPerGPU": 16`, and `<redacted>` for both tokens.
8. Make the scripts executable:

   ```bash
   chmod +x scripts/*.sh
   ```

## Saturday timeline

| Time | Phase | Command | GPU hours | Cumulative spend |
|---|---|---|---|---|
| 09:00 | Create pod, bootstrap, smoke | `runpod_up.sh`, `pod_bootstrap.sh`, `saturday.sh --only smoke` | 0.5 | $1.50 |
| 09:30 | Sampled baseline, select tasks | `saturday.sh --only baseline`, `--only select` | 0.75 | $3.70 |
| 10:15 | Greedy base eval | `saturday.sh --only base_eval` | 0.5 | $5.20 |
| 10:45 | RFT warm start (optional) | `saturday.sh --only rft` | 1.0 | $8.20 |
| 11:45 | GRPO, Track A | `saturday.sh --only grpo` | 5.0 | $23.10 |
| 16:45 | Final eval | `saturday.sh --only final_eval` | 0.75 | $25.30 |
| 17:30 | Report, push, stop pod | `report`, `push`, `runpod_down.sh` | 0.25 | $26 |
| Optional | Track B short run | `GRPO_CONFIG=configs/grpo_trackB.yaml` | 2.0 | $32 |

Spend assumes $2.99 per hour. The Runpod console is the source of truth for
billing. The pod bills while you read, wait, or debug, so real spend runs ahead
of the table.

### 09:00 Create the pod and bootstrap

**Mac**: create the volume and pod.

```bash
scripts/runpod_up.sh --datacenter DATACENTER_ID
```

If the volume `rl-cua-vol` already exists, omit `--datacenter`. The script
shows the hourly price, an estimate for 9 hours, and how many hours $50 buys.
Type `y` to confirm.

The output ends with an `ssh` command and a cost per hour. The pod is ready
when you see "SSH works." and "Writing /workspace/.secrets".

**Mac**: copy the code, then connect.

```bash
scripts/sync_results.sh --push-code
source .runpod/state.env && ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST"
```

**Pod**: start `tmux`, so the run survives a dropped connection, and bootstrap.

```bash
tmux new -s run
bash /workspace/RL-ComputerUse/scripts/pod_bootstrap.sh
```

The bootstrap takes about 15 minutes on a fresh volume and about 3 minutes
when the volume already has the venv. Add `--with-2b` to prefetch the 2B model
as a fallback (about 5 GB more).

Expect the following output, in order:

1. `nvidia-smi` shows one H100 80GB, then `Driver version: 5xx.xx`.
2. `uv pip check` reports no conflicts.
3. A line like `torch 2.14.0 cuda 12.9 available True`, then the vLLM,
   transformers, TRL, and PEFT versions.
4. Chromium downloads, then the MiniWoB++ checkout.
5. `MINIWOB_URL=file:///workspace/miniwob-plusplus/miniwob/html/miniwob/`.
6. `HF_TOKEN: set` and `WANDB_API_KEY: set`.
7. The model download progress.
8. `pytest` ends with `passed`, then the `cua.env` smoke test prints an
   observation summary, then `Bootstrap finished`.

The log is `/workspace/logs/bootstrap.log`.

**Decision point: the bootstrap smoke test fails.**

| Symptom | Fix |
|---|---|
| `Executable doesn't exist` or missing shared libraries (`libnss3`, `libatk`) | `playwright install --with-deps chromium` in `/root/cua/venv`, then run the bootstrap again. |
| `ValueError` about `MINIWOB_URL`, or `core is not defined` | Run `source /workspace/env.sh && echo $MINIWOB_URL`. The value must end with `miniwob/html/miniwob/`, with a trailing slash. Check that `/workspace/miniwob-plusplus` is at commit `7fd85d7` with `git -C /workspace/miniwob-plusplus rev-parse HEAD`. |
| `Target page, context or browser has been closed` | Check `free -g` and `df -h /dev/shm`. Chromium needs `--disable-dev-shm-usage` and `--no-sandbox`, which `cua.env` sets. If the error persists, copy the last 40 lines of the log and fix `cua/env.py`. |
| `pytest` fails in `tests/` | Read the failure. It's a code bug, not a pod problem. Fix it on the Mac, run `scripts/sync_results.sh --push-code`, and run `cd /workspace/RL-ComputerUse && pytest -q tests`. |

### 09:15 Smoke test the full stack

**Pod**:

```bash
source /workspace/env.sh && cd /workspace/RL-ComputerUse
bash scripts/saturday.sh --only smoke
```

This runs the environment smoke test and a 2-seed greedy eval on
`click-button` through vLLM. Expect one episode summary, then
`Phase smoke took N min`. The first vLLM start takes 2-4 minutes (model load
and CUDA graph capture).

**Decision point: vLLM reports a CUDA or driver mismatch.** Symptoms:
`CUDA driver version is insufficient`, `libcudart.so.13: cannot open shared
object file`, or `no kernel image is available`. PyPI's `vllm==0.33.0` can pull
CUDA 13 wheels that need a driver of R580 or newer. Rebuild the venv with CUDA
12.9 wheels:

```bash
rm -rf /root/cua/venv
bash scripts/pod_bootstrap.sh --torch-backend=cu129 --skip-prefetch
```

If the driver is older than R575, use `--torch-backend=cu128`. The bootstrap
also picks a backend automatically when the driver is older than R580.

**Decision point: out-of-memory in the smoke eval.** Run the eval by hand with
a lower memory share:

```bash
python -m cua.eval --config configs/eval.yaml --set eval.mode=greedy \
  --set eval.tasks=click-button --set eval.num_seeds=2 --set run_name=smoke \
  --set vllm.gpu_memory_utilization=0.6
```

If W&B can't connect, run `export WANDB_MODE=offline` and sync the logs later
with `wandb sync`.

### 09:30 Sampled baseline and task selection

**Pod**:

```bash
bash scripts/saturday.sh --only baseline
bash scripts/saturday.sh --only select
```

The baseline samples 16 seeds at temperature 1.0 for each of the 40 candidate
tasks (640 episodes) and takes about 10-15 minutes. Expect
`results/eval/baseline/per_task.csv`. Open it:

```bash
column -s, -t results/eval/baseline/per_task.csv | head -30
```

The `select` phase prints the selected tasks. A healthy result is 10-15 tasks
with a base success rate between 10% and 80%.

**Decision point: fewer than 8 tasks qualify.** The `select` phase fails on
purpose. Widen the band to 5-90% and run it again:

```bash
SELECT_MIN=0.05 SELECT_MAX=0.90 bash scripts/saturday.sh --only select
```

If the output still has fewer than 8 tasks, the prompt or parser probably has
a problem, because nearly every task fails. Read two GIFs or episode records
in `results/eval/baseline/episodes.jsonl` before you spend more GPU time.

### 10:15 Greedy base eval

**Pod**:

```bash
bash scripts/saturday.sh --only base_eval
```

This runs 50 seeds per task at temperature 0 on the selected tasks
(`base-train`) and on the 5 held-out tasks (`base-heldout`). The results go to
`results/eval/base-train/` and `results/eval/base-heldout/`, with GIFs for the
first 3 seeds of each task. Write down the overall success rate for both runs.
These are your "before" numbers.

### 10:45 Optional: RFT warm start

**Pod**:

```bash
bash scripts/saturday.sh --only rft
```

The phase first runs `python -m cua.rft should_skip`. RFT is skipped when the
baseline format-error rate is below 5% and the mean success rate on the
selected tasks is above 30%. In that case the log says "RFT skip rule met" and
the GRPO phase starts from the base model.

Otherwise the phase collects successful episodes (32 per task), trains one
epoch of LoRA SFT, and evaluates the adapter as `rft-train` and `rft-heldout`.
It records the adapter path in `results/.phases/init_adapter`, and the GRPO
phase starts from it automatically.

**Decision point: RFT made things worse.** If `rft-train` success is more than
5 points below `base-train`, or `rft-heldout` drops by more than 5 points,
discard the adapter and train GRPO from the base model:

```bash
rm results/.phases/init_adapter
```

### 11:45 GRPO training

**Pod**:

```bash
bash scripts/saturday.sh --only grpo
```

The default trainer is `nano` (`cua/nano_grpo`). It steps 32 browsers in
parallel, so one step of 64 episodes takes about 2-2.5 minutes on an H100 PCIe (measured in testing). The first step
takes longer because of the vLLM start and the first browser launches.
Checkpoints go to `/workspace/ckpt/grpo-4b-trackA/checkpoint-N` every 25
steps, and the phase stops itself after `GRPO_MAX_HOURS` hours (default 5).

Open W&B in your browser (project `rl-computer-use`) and watch the charts in
the next section. Check the following within the first 30 minutes:

- A step takes under 3 minutes.
- No crash, no OOM.
- `reward/total` is not NaN.

If the connection drops, reconnect and run `tmux attach -t run`. If the
process died, run `bash scripts/saturday.sh --only grpo` again. It resumes from
the newest checkpoint.

**Decision point: a step takes more than 3 minutes.** Apply the following
fixes in order, and stop when a step is back under 3 minutes:

1. Cut the workload. Stop the run with `Ctrl-C`, then resume with a smaller
   setting:

   ```bash
   bash scripts/saturday.sh --only grpo -- \
     --set env.max_turns=6 --set env.screenshot_long_side=384 --set grpo.prompts_per_step=4
   ```

   Evals must use the same observation settings as training. Set
   `CUA_SETS="--set env.max_turns=6 --set env.screenshot_long_side=384"` in the
   shell before every later phase, and rerun `bash scripts/saturday.sh --redo
   base_eval` for a fair comparison (about 0.5 hours).
2. Switch to the 2B model, which is about twice as fast per rollout:

   ```bash
   MODEL=2b bash scripts/saturday.sh --from base_eval
   ```

   This reruns the base evals with 2B, then RFT and GRPO. It fetches the 2B
   weights on first use. Run `bash scripts/pod_bootstrap.sh --with-2b
   --skip-smoke` beforehand to prefetch them.

**Decision point: nano crashes with an error you can't fix in the first 30
minutes.** Switch to the TRL path:

```bash
TRAINER=trl bash scripts/saturday.sh --only grpo -- \
  --set grpo.per_device_train_batch_size=4 --set grpo.gradient_accumulation_steps=16
```

TRL calls the environment serially, so a step takes about 4 minutes. Reduce
the cost with `--set grpo.prompts_per_step=4` and `--set env.max_turns=6`.

If TRL fails on images returned from `reset`, disable the first screenshot:

```bash
CUA_SETS="--set env.first_obs_image=false" TRAINER=trl \
  bash scripts/saturday.sh --only grpo -- \
  --set grpo.per_device_train_batch_size=4 --set grpo.gradient_accumulation_steps=16
```

Keep `CUA_SETS` set for every later eval, and rerun `--redo base_eval`, so the
base and trained numbers use the same observations.

**Decision point: the zero-variance fraction is above 60% after 50 steps.**
Most groups give no learning signal, so the pool is wrong. Stop the run
(`Ctrl-C`). In W&B, find the `success/<task>` charts and drop the tasks with a
training success rate above 90% (mastered) or below 5% (impossible):

```bash
jq '.selected -= ["TASK_ONE", "TASK_TWO"]' results/selected_tasks.json > /tmp/sel.json \
  && mv /tmp/sel.json results/selected_tasks.json
bash scripts/saturday.sh --only grpo
bash scripts/saturday.sh --redo base_eval   # later, so the base numbers use the new pool
```

Replace `TASK_ONE` and `TASK_TWO` with the task names to drop.

**Decision point: training collapses.** Reward drops for 10 or more steps,
the format-error rate rises, or entropy spikes. Delete the checkpoints from the
bad period so the newest checkpoint is a good one, then resume with a lower
learning rate:

```bash
/bin/ls /workspace/ckpt/grpo-4b-trackA
rm -rf /workspace/ckpt/grpo-4b-trackA/checkpoint-BAD_STEP   # repeat for each later checkpoint
bash scripts/saturday.sh --only grpo -- --set grpo.learning_rate=5e-6
```

Replace `BAD_STEP` with a step number from the listing, for example `125`.

**Decision point: out-of-memory.** Lower the memory knobs, then resume:

```bash
bash scripts/saturday.sh --only grpo -- \
  --set grpo.vllm_gpu_memory_utilization=0.30 --set grpo.micro_batch_episodes=1
```

For the TRL path, lower `grpo.per_device_train_batch_size` and raise
`grpo.gradient_accumulation_steps` so that the product stays 64.

### Budget checkpoints

The Runpod console shows the real spend. Compare it with the following
checkpoints. `bash scripts/saturday.sh --status` prints an estimate from phase
durations only, and it misses idle time.

| Spend | Where you should be | If you're behind |
|---|---|---|
| $10 | GRPO started | Skip RFT (`touch results/.phases/rft.done`) |
| $20 | GRPO at step 80 or later | Cut `max_steps` to 150 with `-- --set grpo.max_steps=150` |
| $30 | GRPO finished, or at its last checkpoint | Stop GRPO (`Ctrl-C`) and run `final_eval` on the newest checkpoint |
| $40 | Core work done | Stop starting new work. Go to the end-of-day steps. Track B only if you have $10 or more left |

Spend above $40 for the core run leaves too little for Track B and mistakes.

### 16:45 Final eval

**Pod**:

```bash
bash scripts/saturday.sh --only final_eval
```

By default the phase evaluates `/workspace/ckpt/grpo-4b-trackA/final`, or the
newest checkpoint if the run was cut short. To evaluate a specific checkpoint,
for example the best one from the mid-training evals:

```bash
BEST_CKPT=/workspace/ckpt/grpo-4b-trackA/checkpoint-150 bash scripts/saturday.sh --only final_eval
```

The phase writes `results/eval/grpo-train/` and `results/eval/grpo-heldout/`.
The success criteria from [PLAN.md](PLAN.md) are:

- `grpo-train` success is at least 15 percentage points above `base-train`.
- `grpo-heldout` success is no lower than `base-heldout`.

### 17:30 Report and push

**Pod**:

```bash
bash scripts/saturday.sh --only report
HF_ADAPTER_REPO=YOUR_HF_USER/qwen3vl-4b-miniwob-grpo bash scripts/saturday.sh --only push
```

Replace `YOUR_HF_USER` with your Hugging Face user name. The push phase creates
a private repository and uploads the final adapter. It needs a write-enabled
`HF_TOKEN` on the pod.

### Optional: Track B

If you have at least $10 and 2 hours left, run the coordinate-only stretch run:

```bash
GRPO_CONFIG=configs/grpo_trackB.yaml bash scripts/saturday.sh --only grpo
```

Track B writes to `/workspace/ckpt/grpo-4b-trackB`. Don't evaluate it with
`--only final_eval`: that phase uses Track A settings and overwrites the
`grpo-train` results. Instead, evaluate the base model and the Track B adapter
with Track B settings and their own run names. Use a smaller seed count to
save time:

```bash
TB="--set env.track=B --set env.max_turns=10 --set eval.num_seeds=20"
python -m cua.eval --config configs/eval.yaml --set eval.mode=greedy --set eval.tasks=selected $TB --set run_name=trackB-base-train
python -m cua.eval --config configs/eval.yaml --set eval.mode=greedy --set eval.tasks=selected $TB --set run_name=trackB-grpo-train --set model.adapter=/workspace/ckpt/grpo-4b-trackB/final
```

If the run stopped at its time budget, there's no `final` folder. Use the
newest `checkpoint-N` folder instead. The report doesn't chart Track B runs;
compare the two `summary.json` files in `results/eval/`.

## What to watch in W&B

The W&B project is `rl-computer-use`. The local copy of every metric is
`results/train/RUN_NAME/metrics.jsonl`. The following table lists the charts,
using the names that `nano_grpo` logs. The TRL path logs the reward and success
charts under the same names, prefixed by TRL's own `train/` grouping.

| Chart | Healthy | Unhealthy | Action |
|---|---|---|---|
| `reward/total` | Rises slowly, noisy | Flat for 100+ steps, or a sudden drop | Check the zero-variance fraction. If it's normal, lower the learning rate. |
| `success/overall` and `success/<task>` | Rises on most tasks | One task rises while the others fall | Look for a reward exploit in that task's GIFs. |
| `groups/zero_variance_frac` | Below 50% and falling | Above 60% after 50 steps | Re-select the pool (see the decision point). |
| `format_error/rate_per_turn` | Near zero after RFT | Rising | The policy is degenerating. Lower the learning rate and resume. |
| `train/entropy_proxy` (mean negative log-prob of sampled tokens) | Slowly decreasing | Collapsing to near zero, or spiking | Collapse: lower the learning rate and resume from a good checkpoint. |
| `turns/mean` | Stable or slowly dropping | Collapsing to 1 | The model stopped acting. Check the format-error rate. |
| `grad_norm` | Stable, below the clip value of 1.0 most steps | Repeated spikes | Lower the learning rate. |
| `eval/success` (every 50 steps) | Rising | Falling while `reward/total` rises | Overfitting or a reward exploit. Prefer an earlier checkpoint with `BEST_CKPT`. |
| `time/rollout_s` and `time/train_s` | About 2-2.5 minutes per step with `nano` | More than 3 minutes per step | See the throughput decision point. |

## End of day

1. **Mac**: sync everything to your Mac and open the report.

   ```bash
   scripts/sync_results.sh --adapters
   open results/report/index.html
   ```

   Adapters land in `results/adapters/NAME/`. Check that `results/eval/`
   has all six eval folders: `base-train`, `base-heldout`, `rft-train`
   (if RFT ran), `rft-heldout`, `grpo-train`, `grpo-heldout`.
2. **Mac**: stop the pod. The command syncs again and then stops the pod.
   Your network volume stays.

   ```bash
   scripts/runpod_down.sh --stop
   ```

   If the stop fails because the pod has a network volume attached, run
   `scripts/runpod_down.sh --delete-pod`. The volume and its files stay.
3. In the Runpod console, confirm that the pod no longer shows as Running.
4. Decide what to do with the volume:
   - Keep it for a second session. A 100 GB network volume bills about $0.07
     per GB per month, so about $7 per month. Check the Runpod console for
     the current rate.
   - Delete it when you're finished and have copied the adapter and results:
     `scripts/runpod_down.sh --delete-volume`. The script asks you to type
     the volume name. Deleting the volume erases the venv, the model weights,
     and the checkpoints.

## If the pod dies

The pod can fail, get evicted, or lose its GPU. Everything that matters is on
the network volume (`/workspace`): the code, the venv, the checkpoints, and
`results/`.

1. **Mac**: check the pod status in the Runpod console. If it's gone or
   unhealthy, delete it, then create a fresh pod on the same volume.

   ```bash
   scripts/runpod_down.sh --delete-pod --no-sync
   scripts/runpod_up.sh
   ```

   The script reuses `rl-cua-vol` and its datacenter. If that datacenter has
   no H100 capacity, try `--gpu pcie`. A volume can't move datacenters, so as
   a last resort create a new volume in another datacenter and copy the latest
   checkpoint from your Mac with `rsync`.
2. **Mac**: push the code again, then connect.

   ```bash
   scripts/sync_results.sh --push-code
   ```

3. **Pod**: bootstrap again. The venv, weights, and MiniWoB++ clone are on the
   volume, so this takes about 3 minutes. The container disk is new, so the
   apt packages and Chromium reinstall.

   ```bash
   tmux new -s run
   bash /workspace/RL-ComputerUse/scripts/pod_bootstrap.sh --skip-prefetch
   ```

4. **Pod**: resume.

   ```bash
   source /workspace/env.sh && cd /workspace/RL-ComputerUse
   bash scripts/saturday.sh --status
   bash scripts/saturday.sh
   ```

   Completed phases have marker files in `results/.phases/` and are skipped.
   The `grpo` phase resumes from the newest `checkpoint-N`. You lose at most
   25 steps (about 25 minutes).

If `--only grpo` doesn't resume as expected, list the checkpoints
(`/bin/ls /workspace/ckpt/grpo-4b-trackA`). A checkpoint folder must contain
`adapter_config.json` and `adapter_model.safetensors`. Delete a half-written
newest folder and run again.
