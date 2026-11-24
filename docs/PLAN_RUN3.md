# Run 3 plan: Track B GRPO from the run-2 adapter

This plan describes the third GRPO run for Qwen3-VL-4B on MiniWoB++. Run 3
trains Track B (screenshot only, native `coordinate: [x, y]` clicks) for the
first time. It starts from the run-2 adapter and trains on a re-screened pool
that mixes both earlier pools with new coordinate tasks. Status: implemented on
2026-11-24. Nothing has run on a GPU yet; the `smoke` phase is the first check.

- Starting weights: the run-2 adapter, `checkpoints/grpo-4b-run2/checkpoint-52`
  in `HF_ADAPTER_REPO`. Run 2 stopped at its time cap, so it has no `final`
  folder; `checkpoint-52` is the adapter that its post-training evals used.
  `cua/hfsync.py` picks `final` if it exists, and otherwise the newest
  checkpoint with an adapter. To force a path, set `RUN2_HF_PATH`.
- GPU: 1x H100 80GB on Runpod, container disk only, about $2.70-3.00 per hour.
- Budget: about $35 expected, a soft stop at about $42 (`.stop_grpo`), and a
  hard cap of $50. The monitor enforces both.
- Contract with the monitor and the dashboard:
  [RUN3_INTERFACES.md](RUN3_INTERFACES.md).

## What runs 1 and 2 tell us

The following lessons shape this plan:

- **Count mixed groups, not steps.** Run 2 took 52 steps in 5 hours, and each
  step trained on 6 mixed groups (48 episodes), so about 312 groups in total.
  Run 1 took 158 steps but trained on only about 1.7 groups per step. Run 3
  logs the running total as `dyn/trained_groups_total`.
- **Track B formatting matters.** An early Track B build couldn't parse about
  30% of responses. The native Qwen3-VL format, `mouse_click` with
  `{"coordinate": [x, y]}` in 0-1000 units, fixed that. A 64-episode Track B
  screen of the base model had a format-error rate of 0% (results_test_run).
- **Browsers must not load pages from a slow network drive.** Run 3 keeps the
  repository, the venv, and the MiniWoB pages on the container disk (`/root`).
- **Long pages overflow the context.** Track B sends no AXTree, so prompts are
  much shorter than on Track A. An episode that still overflows ends as
  truncated, without a format penalty.
- **Mid-run checks on 10 seeds swing a lot.** Mid-training evals use 20 seeds
  per task, and the headline evals use 50 seeds (pool) and 100 seeds
  (held-out).
- **RFT helps where the protocol is the problem.** Run 1's RFT adapter gained
  4.8 points overall, almost all on enter-time (36% to 94%). GRPO then added
  25.8 points. See the RFT decision in the next section.

## Decisions

1. **Track B from the run-2 adapter.** Track B has more headroom than Track A,
   and the run-2 adapter already knows the tasks. Starting from the base model
   would spend the budget relearning them.
2. **Re-screen with the policy you train.** A sampled screen (16 seeds,
   temperature 1.0) of the run-2 adapter on Track B decides which tasks are
   informative. The base model's rates would be the wrong guide.
3. **A mixed pool with replay.** See the next section.
4. **Keep what worked in run 2:** the adaptive sampler, DAPO dynamic sampling,
   assistant-only logits, PPO clipping with 2 updates per step, and fresh seeds
   past every earlier run.
5. **RFT is optional and gated.** The `rft` phase runs only when the Track B
   screen has a format-error rate of 5% or more (`RFT=auto`). The reasons:
   - Run 1's RFT gain came from fixing the action protocol, and Track B's
     native format already has a near-zero error rate.
   - RFT pushes tasks to saturation, which takes them out of the GRPO band.
     enter-time went to 94% after RFT and taught GRPO nothing for all of run 1.
   - It's cheap if it does run: about 0.45 hours (about $1.30) to sample 32
     episodes per pool task, keep the clean successes, and fine-tune the run-2
     adapter for one epoch at a learning rate of 2e-5.

   Set `RFT=on` or `RFT=off` to override the gate. When RFT runs, GRPO starts
   from the RFT adapter and the `rft3-pool3` eval measures RFT alone.
6. **Held-out v2 stays exactly as run 2 chose it.** Its 10 tasks are frozen in
   `configs/tasks.yaml`, so held-out numbers compare across runs.

## Task pool

The pool has up to 16 *learn* tasks and up to 4 *replay* tasks.
`cua/run3_pool.py` builds it from the screen. The candidates are as follows:

- Run 1's 40 candidates and run 2's 22 extras.
- Nine Track-B-only tasks that need pixel positions or a visual reading:
  `bisect-angle`, `circle-center`, `find-midpoint`, `right-angle`,
  `ascending-numbers`, `use-colorwheel`, `use-colorwheel-2`,
  `visual-addition`, and `tic-tac-toe`. The list leaves out drag, draw,
  slider, and hover tasks, because Track B has no tool for them, and
  `count-sides`, which is held out.

The `smoke` phase drops any candidate that browsergym-miniwob doesn't register.
Each candidate gets one source group, in this order of priority: run-2 pool,
run-1 pool, Track-B extras, and other.

To pick learn tasks, `select` takes the tasks in the 5%-85% band. It reserves
up to 3 slots for each of the three named sources, choosing the tasks nearest
45% success. It fills the remaining slots with the nearest-45% tasks from any
source. This keeps the pool from collapsing onto one family, for example onto
email tasks only.

To pick replay tasks, `select` takes tasks from the run-1 and run-2 pools that
score above 85%. It alternates between the two pools and prefers the lowest
rates, which are the most fragile. Replay groups are mostly all-success, so
they cost rollouts but no training. The adaptive sampler gives them a floor
weight of 0.10. If a replay task starts to fail, its groups turn mixed, its
weight rises, and GRPO trains on it again. Replay doubles as a forgetting
alarm.

## Hyperparameters

The following table lists the settings in `configs/grpo_run3.yaml` that differ
from run 2 or that matter most:

| Setting | Value | Reason |
|---|---|---|
| `env.track`, `env.max_turns` | B, 10 | Track B needs a click before each typing action. |
| `learning_rate`, `warmup_steps` | 1e-5, 10 | Run 2 was stable at this rate with a fresh AdamW. |
| `num_generations`, `prompts_per_step` | 8, 8 | Unchanged. Dynamic sampling raises the mixed share instead of larger groups. |
| `dynamic_sampling` | 6 mixed groups, up to 3 rounds, train on at most 6 | Run 2 found 8 mixed groups per step in 2.1 rounds on average. |
| `updates_per_step` | 2 | Keeps the PPO clip active. |
| `adaptive.floor` | 0.10 | Run 2 used 0.05. Run 3 has an explicit retention goal. |
| `save_steps` | 10 | Run 2 used 25. A lost pod costs at most 10 steps, about 40 minutes. |
| `eval_every`, `eval_num_seeds` | 25, 20 | Mid-training evals that don't swing as much as 10 seeds. |
| `train_seed_offset` | 300000 | Run 1 used seeds from 1000, run 2 from 101000 (about 100 per task). |
| `max_steps` | 400 | A ceiling. The time cap ends the run first. |
| `samples_every` | 5 | One dashboard sample every 5 steps. |

## Phases, time, and cost

The estimates come from run 2's logs (`results/run2/run.log`, eval
`summary.json` timings, and `metrics.jsonl`) on an A100 at $2.19 per hour:

- The sampled baseline ran 992 episodes in 15 minutes (0.93 s per episode).
- `pre_eval` ran 3,100 greedy episodes in 64 minutes, including 5 model loads
  (1.24 s per episode overall).
- `post_eval` ran 2,300 episodes in 34 minutes (0.89 s per episode).
- A GRPO step took 333 s: 108 s of rollouts and 225 s of training for 48
  episodes. A mid-training eval of 320 episodes took 3.7 minutes.

The table assumes $2.90 per hour and the same per-episode times as run 2. An
H100 and Track B's shorter prompts probably make training faster, but the plan
doesn't count on it.

| Phase | Work | Hours | Cost |
|---|---|---|---|
| Pod start | Pod creation, image pull | 0.10 | $0.29 |
| Bootstrap and restore | `pod_bootstrap.sh`, 9 GB base model, run-2 adapter | 0.25 | $0.73 |
| `smoke` | 3 GRPO steps on 2 tasks (2 fresh, 1 resumed), 3-seed eval | 0.30 | $0.87 |
| `screen` | 71 tasks x 16 seeds, sampled, Track B | 0.30 | $0.87 |
| `select` | Pool mixing | 0.00 | $0.00 |
| `rft` | Skipped when the format-error rate is under 5% (expected) | 0.00 (0.45 if it runs) | $0.00 ($1.31) |
| `pre_eval` | Base and run 2 on the pool (20 x 50) and held-out (10 x 100), Track B | 1.40 | $4.06 |
| `grpo` | `GRPO_MAX_HOURS=9`, about 90-150 steps, 540-900 trained groups | 9.00 | $26.10 |
| `post_eval` | Run 3 on pool and held-out (Track B) and run-2 pool (Track A) | 0.70 | $2.03 |
| `compare`, `report` | Paired bootstraps and the HTML report | 0.02 | $0.06 |
| Final upload | `upload_run3.py --final --verify`, pod deletion | 0.15 | $0.44 |
| **Total** | | **12.2** | **about $35** |

At $2.70 per hour the total is about $33, and at $3.00 about $37. A pod loss
costs about 0.5 hours for the new pod and its bootstrap plus up to 10 lost
steps, roughly $2-4. One loss still ends below the $42 soft stop. If the soft
stop fires during GRPO, `post_eval` and the upload add about $2.50, which ends
near $45, under the $50 cap. If run 2's own Track A eval isn't on the Hub,
`pre_eval` evaluates the run-2 adapter on Track A as well (about 0.25 hours).

## Success criteria

All numbers come from greedy decoding on eval seeds 0-49 (pool) or 0-99
(held-out), identical across models, with paired 95% intervals from
`cua/compare.py`. Run 3 succeeds when the following are true:

1. **Track B pool gain:** run 3 beats the run-2 adapter on the run-3 pool by
   at least 10 points, and the interval excludes 0
   (`compare/pool3-run3-vs-run2.md`).
2. **Track B generalization:** on held-out v2, run 3 isn't worse than the
   run-2 adapter: the lower end of the interval is above -3 points
   (`compare/heldout3-run3-vs-run2.md`).
3. **Track A retention:** on run 2's pool, run 3 on Track A scores within 5
   points of run 2's 72.6% (`compare/trackA-pool2-run3-vs-run2.md`).
4. **Signal per dollar:** GRPO trains on at least 500 mixed groups, against
   run 2's 312.
5. **Budget:** the run ends without the soft stop and never passes $50.

Expected outcome, stated plainly: Track B starts lower than Track A on the same
tasks, so the pool gain can be larger than run 2's. Track A retention is the
open question, because run 3 trains no Track A episodes. A drop of more than 5
points would argue for mixed-track training in a later run.

## How the pipeline survives a lost pod

The pod has no network volume, so a lost pod loses its disk. The following
mechanisms bring a new pod back to the last synced state:

- The sync loop (tmux session `run3-sync`) uploads each new checkpoint once,
  in the background, and records it in `results/run3/.synced_checkpoint`. It
  uploads small files under `results/run3/` every 5 minutes and right after a
  phase finishes. Adapters go up before markers, so a `grpo.done` on the Hub
  implies that the final adapter is there too.
- `run3_pod_start.sh` restores the run-2 adapter, `results/run3/`, the newest
  synced checkpoint, and any finished adapter, each only when it's missing.
  With nothing synced yet, GRPO starts fresh from the run-2 adapter.
- `run3.sh` skips finished phases. Inside an eval phase, it skips each eval
  whose `summary.json` exists. GRPO resumes with `--resume`, which restores the
  optimizer, RNG, sampler state, and W&B run ID.
- The GRPO time cap counts time over every pod (`.phases/grpo.elapsed_s`), so a
  restart doesn't reset it.

## Risks

- **The HF token can't see the adapter repository.** On 2026-11-24, the
  fine-grained `HF_TOKEN` in the Mac's `.env` got a 404 when listing
  `HF_ADAPTER_REPO`. If the monitor writes that token to the pod, restore and
  sync fail at once. Give the token read and write access to the repository
  before launch.
- **Track B episodes are longer.** More turns per episode can make rollouts
  slower. The `smoke` phase prints rollout and training seconds per step.
- **Track-A forgetting.** Run 3 measures it but doesn't train against it.
- **Unregistered tasks.** The new Track-B tasks weren't checked against
  browsergym-miniwob 0.14.3 on a pod. The `smoke` phase drops any it doesn't
  register.
- **Coordinate tasks that never succeed.** If the run-2 adapter scores 0% on
  them, they drop out of the band and the pool leans on the earlier pools.
- **An early stop flag.** If `.stop_grpo` exists before GRPO's first step,
  GRPO saves the start adapter as its final adapter, and the post-training
  evals measure the run-2 adapter again.
- **Hub storage.** Every synced checkpoint is about 0.8 GB, so run 3 adds
  about 10-15 GB to the repository.

## Launch

The monitor launches and watches run 3. The commands it uses are in
[RUN3_INTERFACES.md](RUN3_INTERFACES.md). To start it by hand on a pod that has
`/workspace/.secrets`, run the following command:

```bash
cd /root/RL-ComputerUse && bash scripts/run3_pod_start.sh
```

To check progress, run `bash scripts/run3.sh --status`. To end GRPO early and
go on to the post-training evals, create `results/run3/.stop_grpo`.

In the first hour of the `grpo` phase, check these values in
`results/run3/logs/grpo.log` or W&B:

- `dyn/rounds`: mostly 1 or 2. Always 3 means the pool is too easy or too hard.
- `train/clip_frac`: above 0.
- `format_error/rate_per_turn`: under 5%.
- `time/train_s`: under about 250 s per step.
