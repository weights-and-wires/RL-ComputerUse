# Run 2 plan: continue GRPO from the run-1 adapter

This plan describes a second GRPO run for Qwen3-VL-4B on MiniWoB++ Track A. It
starts from run 1's final adapter and changes how the trainer spends its compute,
because run 1 wasted most of it. Status: **agreed on 2026-11-15 and
implemented**. Nothing has run on a GPU yet; the smoke test is the first check.

- Planned on: 2026-11-15.
- Starting weights: `weights-and-wires/qwen3vl-4b-miniwob-grpo`, repository root
  (identical to `checkpoints/grpo-4b-trackA/checkpoint-158`).
- GPU: 1x A100 SXM 80GB on Runpod, about $2.19 per hour.
- Budget: about $20 expected, $30 cap including smoke tests.

## What run 1 tells us

Run 1 (RFT warm start plus 158 GRPO steps) raised greedy success on the 10
training tasks from 27.3% to 57.8%, and on the 5 held-out tasks from 55.6% to
62.8%. The training metrics in `results/train/grpo-4b-trackA/metrics.jsonl` show
where the compute went.

### Most rollouts taught nothing

`groups/zero_variance_frac` averaged 0.77. Of the 8 groups per step, about 1.7
had mixed outcomes, so each step trained on about 13 of its 64 episodes. Twenty
of the 158 steps trained on nothing at all.

The probability that a group of *G* episodes has mixed outcomes, for a task
with success rate *p*, is 1 - *p*^*G* - (1 - *p*)^*G*. The following table shows
that probability for *G* = 8:

| Task success rate *p* | Chance a group of 8 is mixed |
|---|---|
| 0.05 or 0.95 | 34% |
| 0.10 or 0.90 | 57% |
| 0.20 or 0.80 | 83% |
| 0.50 | 99% |
| 0.98 | 15% |

Tasks drift out of the useful band as training works. Rollout success per
40-step window shows this:

| Task | Steps 1-40 | Steps 121-158 | Comment |
|---|---|---|---|
| enter-time | 0.98 | 1.00 | Mastered from step 1, so the RFT warm start solved it, not GRPO. |
| enter-date | 0.00 | 0.99 | Learned around step 90, pure GRPO (RFT kept no episodes). |
| find-word | 0.50 | 0.81 | Learned late. |
| click-widget | 0.88 | 0.80 | Near ceiling the whole run. |
| click-collapsible-2 | 0.05 | 0.05 | Almost never solved, so almost never mixed. |
| click-menu | 0.40 | 0.39 | Mixed groups, but greedy stayed at 28%. |

From step 1, enter-time consumed one-tenth of every rollout batch for nothing.

### Training, not rollouts, is the slow part

A step took 45 s of rollouts for 64 episodes (0.7 s per episode with 32 browsers
in parallel) and 50-80 s of training for about 13 episodes (median 4.5 s per
trained episode). A trained episode costs about six times as much as a
rolled-out one. This shapes the whole plan: **spend cheap rollouts to find
informative groups, and spend expensive training only on them.**

The 30 "OOM" lines in `results/run.log` are PyTorch allocator warnings, not
crashes: `micro_batch_episodes` stayed at 2 all run. Each warning is a failed
6.4 GB allocation, which matches the full logits tensor for two episodes
(2 x ~10,000 tokens x 151,936 vocabulary x 2 bytes). The trainer computes logits
for every position, but only about 300 assistant tokens per episode need them.
The allocator then frees its cache and retries, which explains the steps that
took 120-200 s.

### What we can't attribute

`cua/compare.py` (change 5) also settles one run-1 question: the held-out gain
of +7.2 points is real. Its paired 95% interval is +4.0 to +10.4 points, from
login-user-popup (+18) and email-inbox-forward-nl (+10).

The RFT adapter was never evaluated alone. The enter-time numbers suggest RFT
did more than we credited. The adapter exists at `checkpoints/rft-4b/final`, so
run 2 evaluates it once (about 10 minutes) to settle this.

## Goals and success criteria

Run 2 succeeds when the following are true, with all numbers from greedy
decoding on unseen eval seeds:

1. **New pool gain:** on the run-2 training pool, run 2 beats the run-1 adapter
   by at least 10 percentage points, and the paired 95% bootstrap interval
   excludes 0.
2. **Retention:** on run 1's pool, run 2 scores within 5 points of run 1's
   57.8%. Tasks dropped from training must not quietly regress.
3. **Generalization:** on the enlarged held-out set (10 tasks x 100 seeds),
   run 2 is not worse than run 1, and we report the paired interval for both
   run 1 versus base and run 2 versus run 1.

Expected outcome, stated plainly: the easy gains are gone. Run 1's +30.5 points
came mostly from three tasks that went to about 100% (enter-date, find-word,
enter-time). The remaining tasks gained 11-22 points each. For run 2 we expect
+10 to +15 points on the new pool and 0 to +5 on held-out. A result below +10
on the pool still teaches something, if the zero-variance fraction dropped as
planned.

## Changes

The plan has five changes. Changes 1-3 carry the expected gain; 4 and 5 make the
result measurable.

### 1. Re-select the task pool with the run-1 adapter

Run the sampled baseline (16 seeds, temperature 1.0) again, but with the run-1
adapter loaded, on an extended candidate list. Then select tasks in a wider
band, 5% to 85%, up to 16 tasks.

- The run-1 adapter, not the base model, is the policy we train, so its rates
  decide what's informative. Expect enter-date, enter-time, find-word, and
  click-widget to drop out as too easy.
- Add about 20 candidates from the 125 registered MiniWoB tasks that Track A
  can do (no drag, draw, or slider tasks, no pure-coordinate tasks):
  `click-menu-2`, `click-pie`, `click-shades`, `click-color`,
  `click-tab-2-hard`, `copy-paste`, `copy-paste-2`, `email-inbox-important`,
  `email-inbox-star-reply`, `social-media-all`, `number-checkboxes`,
  `buy-ticket`, `phone-book`, `stock-market`, `find-greatest`,
  `identify-shape`, `text-transform`, `guess-number`, `multi-layouts`,
  `use-spinner`, `scroll-text-2`, and `odd-or-even`.
- Exclude anything close to a held-out task: the `email-inbox-forward*` and
  `email-inbox-nl-turk` family, `login-user-popup`, `click-test*`, and
  `enter-text-2`.

Cost: about 62 tasks x 16 episodes, roughly 20 minutes.

### 2. Adaptive task weights

Run 1 rotated through tasks round-robin, so a mastered task got the same share
as an unsolved one. Run 2 keeps a running estimate of each task's rollout
success rate (an exponential moving average, initialized from the baseline) and
samples tasks with weight equal to the chance that a group is mixed:

```text
weight(task) = max(1 - p^G - (1 - p)^G, 0.05)
```

- A task at 50% gets weight 0.99, a task at 98% gets 0.15, and a task the model
  never solves gets the floor of 0.05.
- The floor keeps a little replay of mastered and unsolved tasks, so the
  trainer notices when one starts failing again or starts succeeding.
- Each task keeps its own seed counter, so a (task, seed) pair is still never
  reused. The estimates and counters go into the checkpoint, so `--resume` is
  exact.

### 3. Dynamic sampling

Adaptive weights raise the share of mixed groups, but each group is still a
coin flip. Dynamic sampling, from the DAPO paper, rolls out more groups than it
needs and trains only on the mixed ones:

1. Roll out one round: 8 groups x 8 episodes.
2. Keep the groups with mixed rewards.
3. If fewer than 6 mixed groups are kept and fewer than 3 rounds ran, go back
   to step 1.
4. Train on at most 8 mixed groups (64 episodes), which bounds training time.

With adaptive weights we expect about half the groups to be mixed, so most steps
need 2 rounds. Each step then trains on about 48 episodes instead of 13.

### 4. Faster, steadier training phase

Three changes target the 4.5 s per trained episode:

- **Logits only where needed.** Ask the model for logits only at assistant-token
  positions (the `logits_to_keep` argument, with an index tensor), which removes
  the 6 GB logits tensor. With that gone, we expect fewer allocator retries and
  less time per episode. The smoke test measures the actual speedup; the plan
  doesn't assume one.
- **Two updates per step.** With about 48 episodes, `updates_per_step: 2` splits
  them into two minibatches of about 24. The second update runs on weights that
  already moved, so the PPO ratio leaves 1 and clipping becomes active. You'll
  see nonzero `train/clip_frac` for the first time. This costs one extra
  no-gradient forward pass (about 30% more training time).
- **Fresh optimizer.** Start AdamW from scratch on the run-1 weights, keep the
  learning rate at 1e-5 with 10 warmup steps. Run 1 showed no instability at
  that rate.

Group size stays at 8. Raising it to 16 for hard tasks would lift the mixed
chance for a 10% task from 57% to 82%, but it doubles that group's training
cost. Dynamic sampling gets the same effect by spending rollouts instead.

### 5. Measurements we can trust

- **Bigger held-out set.** Keep the 5 held-out tasks and add 5 more, from tasks
  in neither the candidate list nor the held-out families: candidates are
  `click-checkboxes-transfer`, `email-inbox-delete`, `form-sequence-3`,
  `read-table-2`, `focus-text`, `order-food`, `sign-agreement`, and
  `count-sides`. Some are near relatives of training tasks (for example
  `email-inbox-delete`), so they measure near transfer rather than far
  transfer. Screen them with the **base** model (16 seeds, a few minutes)
  and keep the 5 with rates nearest 50%, so a change can show. Choosing with
  the base model avoids picking tasks that favor either adapter. Evaluate 100
  seeds per held-out task instead of 50.
- **Paired comparisons.** Every model runs on the same (task, seed) pairs. A
  small `cua/compare.py` reads two `episodes.jsonl` files and reports the
  difference with a paired bootstrap interval, per task and overall. With 500
  paired episodes, an interval of about plus or minus 4 points is typical.
- **Mid-training evals every 25 steps with 20 seeds** per pool task, instead of
  every 50 steps with 10. That's about 3 minutes each, and the curve stops
  being noise.
- **RFT attribution.** Evaluate the run-1 RFT adapter on run 1's pool.

## What's not in run 2

- **Track B (coordinate clicks).** It has more headroom, but it changes the
  action space and would start from the base model. Mixing it in would muddy
  the run-2 comparison. It's a good run 3.
- **A new RFT stage.** The run-1 adapter already has a near-zero format error
  rate, which was RFT's main job.
- **Larger groups (*G* = 16).** See change 4.

## Time and cost

The following table estimates GPU time at $2.19 per hour. Eval times assume
about 1 second per episode, as in run 1.

| Phase | Work | Hours |
|---|---|---|
| Setup | Pod, environment, download adapters, tests | 0.5 |
| Smoke test | 10 GRPO steps with the new code; check step time, memory, and logs | 0.3 |
| Pre-training evals | Sampled baseline (run 1), held-out screening (base), RFT adapter on pool 1, base and run 1 on pool 2 and held-out v2 | 1.4 |
| GRPO | Time-capped | 5.0 |
| Post-training evals | Run 2 on pool 2, held-out v2, and pool 1 | 0.6 |
| Report and upload | Report, Hugging Face upload, teardown | 0.3 |
| **Total** | | **8.1 (about $18)** |

Expected GRPO throughput, which the smoke test confirms or corrects:

| | Run 1 | Run 2, if the logits fix halves training time | Run 2, no speedup |
|---|---|---|---|
| Rollout time per step | 45 s | 90 s (2 rounds) | 90 s |
| Trained episodes per step | 13 | 48 | 48 |
| Training time per step | 60-80 s | about 140 s | about 280 s |
| Steps in 5 hours | 158 | about 75 | about 50 |
| Trained episodes in total | about 2,050 | about 3,600 | about 2,400 |

Fewer steps is expected: each step carries more signal. If the smoke test
shows training slower than 5 s per episode, lower `max_train_groups` to 6.

## Implementation

The following changes implement the plan:

1. `cua/nano_grpo/sampler.py`: `RoundRobinSampler` (run 1's order, unchanged)
   and `AdaptiveTaskSampler`. Both save their state in the checkpoint.
2. `cua/nano_grpo/loop.py`: `collect_groups` runs the dynamic-sampling rounds;
   the training phase asks for assistant-only logits; new log keys
   `dyn/rounds`, `dyn/mixed_groups`, `dyn/groups_rolled`, `task_rate/<task>`,
   and `task_weight/<task>`. Run 1's config still reproduces run 1:
   `sampler` defaults to `round_robin` and dynamic sampling is off unless a
   config turns it on.
3. `cua/nano_grpo/loss.py`: `logit_positions` and `token_logprobs_at`.
4. `configs/grpo_run2.yaml`: the run-2 settings, including
   `eval_num_seeds: 20` for mid-training evals and `train_seed_offset: 100000`
   so run 2 never replays a run-1 training seed.
5. `configs/tasks.yaml` and `cua/config.py`: the task lists
   `candidates_run2`, `run1_pool`, `heldout_v2_candidates`, and `heldout_v2`.
   The loader refuses any held-out task that is also a training candidate.
6. `cua/select_tasks.py`: `--candidates` picks the list to select from, and
   `--heldout-extra N` screens the held-out-v2 candidates.
7. `cua/compare.py`: the paired bootstrap comparison.
8. `cua/report.py`: `--preset run2` labels the three models Base, Run 1, and
   Run 2, and measures deltas against run 1. This also fixes a bug in the GIF
   grid that copied files from the wrong eval folder when both pools had GIFs.
9. `scripts/run2.sh`: the resumable run-2 pipeline. The selection band
   (5%-85%, up to 16 tasks) is passed there with `--set`.
10. `tests/test_run2.py`: 21 tests. The assistant-only log-probs match the
    full-logits path, gradients included. On a tiny randomly initialized
    Qwen3-VL (transformers 5.22.1 on the Mac) `logits_to_keep` with a position
    tensor gave identical log-probs. The pod pins transformers 5.20.2 and
    wraps the model in PEFT, so the smoke test is the first check on that
    exact stack.

## Launch commands

These commands assume the run-1 pod setup: repository and Python environment
on `/root`, checkpoints on `/workspace/ckpt`.

On the pod, download both run-1 adapters and run 1's eval results (for the
retention and RFT comparisons):

```bash
hf download weights-and-wires/qwen3vl-4b-miniwob-grpo --local-dir /root/run1 --include "adapter_*" "*.json" "chat_template.jinja" "checkpoints/rft-4b/final/*" "results/eval/base-train/*" "results/eval/grpo-train/*"
```

Run the unit tests and a 10-step smoke run of the new trainer:

```bash
cd /root/RL-ComputerUse && source /workspace/env.sh && python -m pytest -q
```

```bash
python -m cua.nano_grpo.loop --config configs/grpo_run2.yaml --set model.adapter=/root/run1 --set grpo.max_steps=10 --set grpo.output_dir=/workspace/ckpt/smoke-run2 --set run_name=smoke-run2
```

Then start the full run in tmux:

```bash
tmux new -s run2
```

```bash
RUN1_ADAPTER=/root/run1 RFT1_ADAPTER=/root/run1/checkpoints/rft-4b/final RUN1_RESULTS=/root/run1/results GRPO_MAX_HOURS=5 PRICE_PER_HR=2.19 bash scripts/run2.sh
```

`scripts/run2.sh` runs these phases in order, skipping any that already
finished: `baseline` (run-1 adapter, extended candidates), `select`,
`heldout_screen` (base model), `pre_eval` (RFT adapter on pool 1; base and run
1 on pool 2 and held-out v2), `grpo`, `post_eval`, `compare`, and `report`.
Everything goes to `results/run2/`. To see progress, run
`bash scripts/run2.sh --status`.

In the first 30 minutes of the `grpo` phase, check these values in
`results/run2/logs/grpo.log` or W&B:

- `time/train_s` divided by `episodes/trained`: under 5 s per episode. If
  it's higher, stop with `Ctrl-C` and rerun the phase with
  `-- --set grpo.dynamic_sampling.max_train_groups=6`.
- `dyn/rounds`: mostly 1 or 2. Always 3 means the pool is too easy or too hard.
- `train/clip_frac`: above 0 (the second update is working).

The phase results land in `results/run2/compare/*.md`. The headline numbers are
`pool2-run2-vs-run1.md` (goal 1), `pool1-run2-vs-run1.md` (goal 2), and
`heldout2-run2-vs-run1.md` (goal 3).

The home monitor (`~/cua-monitor/loop.sh` on the home machine) needs only the
new pod's SSH details and the `run2` tmux session name. Its teardown pushes code
to GitHub, uploads checkpoints and results to Hugging Face under
`checkpoints/grpo-4b-run2`, verifies the upload, and deletes the pod.

## Decisions

You agreed to the plan as drafted on 2026-11-15:

1. Changes 1-5 as scoped; Track B waits for run 3.
2. Selection band 5%-85%, up to 16 tasks.
3. `updates_per_step: 2`.
4. About $18 expected, $30 cap, on the same A100 SXM 80GB pod type as run 1.
