# Run 4 plan: longer episodes, drag tools, and a guarded loss

This plan describes the fourth and last GRPO run for Qwen3-VL-4B on MiniWoB++.
Run 4 trains Track B (screenshot only) from the run-3 adapter. It doubles the
turn limit to 20, adds `double_click` and `left_click_drag`, keeps prompts
small with a history window, and guards the loss against the gap between
vLLM's and the trainer's log-probabilities. Status: implemented on 2026-11-25.
Nothing has run on a GPU yet; the `smoke` phase is the first check.

- Starting weights: the run-3 adapter, `checkpoints/grpo-4b-run3/final` in
  `HF_ADAPTER_REPO` (step 240, confirmed in the Hub listing). To force another
  path, set `RUN3_HF_PATH`.
- GPU: 1x H100 80GB on Runpod, container disk only. Community H100 SXM costs
  about $2.69 per hour; Secure costs $4.39 per hour.
- Budget: a hard cap of $50 and a soft stop at $42 (`.stop_grpo`). The monitor
  enforces both.
- Contract with the monitor and the dashboard: the run-3 contract with `run3`
  replaced by `run4`. For the differences, see
  [RUN4_INTERFACES.md](RUN4_INTERFACES.md).

Run 4 is the last run, so the plan favors reliability over ambition. Every
mechanism that run 4 adds runs in the `smoke` phase, which fails before any
GPU time goes to training. Each mechanism also has a switch that turns it off
(see [Switches](#switches)).

## Goals

Run 4 has the following goals:

- Solve tasks that run out of turns at 10 turns.
- Make drag and slider tasks possible at all.
- Raise the held-out v2 score above run 3's 71.5% and the full-suite score
  above the run-3 adapter's.
- Train without a ratio blow-up like run 3's step 207.
- End with one number for every model in the project's history, base to run 4.

## What run 3 showed

The following results come from `results/run3/` (compare files, eval
`per_task.csv` files, `metrics.jsonl`, and `run.log`):

- **The pool gained a lot.** Track B pool success rose from 39.2% (base) to
  79.4% (run 3). Held-out v2 (10 tasks, 100 seeds) rose from 64.4% to 71.5%,
  8.4 points above the run-2 adapter (interval +6.4 to +10.4).
- **The pool saturated.** The share of zero-variance groups rose from 0.47 to
  0.70, and dynamic sampling needed 2.8 of 3 rounds per step by the end.
- **Rollouts are the bottleneck.** A step took 82 s of rollouts and 32 s of
  training on average (240 steps). Browsers, not generation, set the pace:
  the screen spent 741 s in browsers and 306 s generating.
- **One step blew up.** Step 207 had a ratio of 3.2e6, an approximate KL of
  169, a gradient norm of 3,709, and a loss of 14.8. The next step was
  normal. That step also had a token-count mismatch between vLLM and the
  trainer (`sampler/length_mismatch_skips` = 1), which happened in only 2% of
  steps. Ratios of 5-33, and up to 533, appear regularly.
- **Ten turns is too few.** In the run-3 screen, 8 tasks failed by running out
  of turns in 62%-100% of episodes: `book-flight`, `social-media-some`,
  `search-engine`, `email-inbox-reply`, `number-checkboxes`, `choose-list`,
  `grid-coordinate`, and `guess-number`.
- **Track B can't drag.** `use-colorwheel` and `use-colorwheel-2` scored 0%
  and ran out of turns every time; the slider and drag tasks weren't even
  candidates.
- **Two tasks never move.** `stock-market` scored 46%-54% for every model,
  and `sign-agreement` scored exactly 42% for every model. See
  [Two tasks that never move](#two-tasks-that-never-move).

## Decisions

### Drop Track A

Run 4 trains and evaluates Track B only. Run 3 measured Track A forgetting
once; another Track A eval costs time and buys nothing for the goals.

### Allow 20 turns, with per-task budgets

`env.max_turns` is 20. Long tasks get every turn. Short tasks get a budget, so
a failing episode on a 2-turn task doesn't hold its rollout round for 20
turns. The `select` phase computes the budgets from the screen
(`results/run4/turn_budgets.json`, `cua.run_pool.turn_budgets_from_episodes`):

- A task with at least 3 successes in the screen gets `2 * p90 + 2` turns,
  where p90 is the 90th percentile of the turns its successes took, clamped
  to 10-20.
- Any other task gets 20 turns.

The floor of 10 is run 3's limit, so no task gets fewer turns than in run 3.
The screen itself runs every task at 20 turns, because the budgets come from
it. The before and after evals use the same file, so they stay paired. The
efficiency bonus stays, measured against the task's own budget.

### Keep prompts small with a history window

At 20 turns, the full history doesn't fit what 64 parallel episodes can keep
in vLLM's KV cache. Qwen3-VL-4B stores 147 KB per token (36 layers, 8 KV
heads, 128 dimensions, key and value, bf16). A turn adds about 330 tokens (a
160-token screenshot, the goal text, and the reply), so a 20-turn
conversation reaches 7,500-9,000 tokens. Sixty-four of them at an average of
3,500 tokens need about 33 GB of cache, against about 18 GB that vLLM has at
run 3's memory share.

Run 4 therefore splits each episode into *segments* of 4 turns
(`env.context_window_turns: 4`). Each segment is its own conversation: the
system prompt, a user message with a text log of every earlier action (one
line per turn, for example `3. mouse_click(coordinate=[412, 230])`) and the
current screenshot, then up to 4 turns. A prompt holds 1-4 screenshots and at
most about 2,800 tokens. Episodes of 4 turns or fewer look exactly as they did
in run 3.

Segments, not a sliding window, because of how training scores tokens. The
trainer must see exactly the tokens the sampler saw. With a sliding window,
every turn's prompt differs from the next one (the oldest screenshot drops
out), so the trainer would need one forward pass per turn, about 4 times run
3's training tokens, and vLLM's prefix cache would never hit. With segments,
every prompt is a prefix of its finished segment, so the trainer scores one
sequence per segment (about 1.4 times run 3's tokens per turn) and the prefix
cache hits inside each segment.

The same code builds the prompts in rollouts, mid-training evals, and
headline evals (`cua/rollout.py`), and training reads the same message lists
(`Episode.conversations()`). Two checks hold this in place:

- `tests/test_run4.py` renders every prompt of a 12-turn episode with
  Qwen3-VL's real chat template and tokenizer and checks that each one, plus
  its reply, is an exact text and token prefix of the trained sequence.
- `python -m cua.smoke4 context` runs the same check in a real browser with
  the full processor on the pod.

### Add double-click and drag

Track B gains two tools when `env.track_b_extra_tools` is true:

| Tool | Arguments | BrowserGym action |
|---|---|---|
| `double_click` | `coordinate: [x, y]` | `mouse_dblclick(px, py)` |
| `left_click_drag` | `start_coordinate: [x, y]`, `end_coordinate: [x, y]` | `mouse_drag_and_drop(fx, fy, tx, ty)` |

Coordinates use the same 0-1000 units as `mouse_click`. The names match
Qwen3-VL's own computer-use actions. Qwen's `left_click_drag` takes only an
end point and drags from wherever the cursor is, which takes two turns and
needs cursor state; run 4 asks for both points in one call instead. The
parser also accepts Qwen's mobile `swipe` shape (`coordinate` and
`coordinate2`) and `start_coordinate` with `coordinate`. A lone `coordinate`
returns an error that names both arguments, and the episode goes on.

Both BrowserGym functions exist in browsergym-core 0.14.3, in the `coord`
action subset that `cua/env.py` already enables (checked in the 0.14.3 wheel).
`mouse_drag_and_drop` moves the mouse in one step from start to end. jQuery UI
draggables, sortables, and sliders accept one large move, but no browser has
confirmed this for every task. The `smoke` phase runs a drag, a double-click,
and a click in real browsers and fails on any BrowserGym error.

### Guard the loss against sampler mismatch

The trainer computes the PPO ratio from its own log-probabilities before and
after each update, but vLLM drew the tokens. Run 4 adds three layers, all in
`cua/nano_grpo/loss.py` and turned on in `configs/grpo_run4.yaml`:

1. **Drop episodes that don't line up.** Before the first update, a no-grad
   pass scores every sequence and lines its assistant tokens up with vLLM's
   tokens, turn by turn. An episode is left out when its tokens don't line up
   (re-tokenizing the text gave other tokens, as at step 207) or when its mean
   absolute log-probability difference is above 0.15 (run 3's per-step mean
   was 0.01-0.09). At most a quarter of a step's episodes are dropped, so a
   systematic shift shows up in the metrics instead of silently emptying the
   batch.
2. **Truncated importance sampling.** Each token's loss is weighted by
   `min(pi_trainer / pi_vllm, 2)`. A token that the trainer finds far less
   likely than vLLM did gets a weight near 0, so it can't produce a huge
   gradient.
3. **Cap the ratio.** PPO's clip bounds the objective for positive advantages
   only. For a negative advantage, `min(r * A, clip(r) * A)` keeps falling as
   `r` grows, which is how one token produced a loss of 14.8 at step 207. The
   dual clip (`ratio_cap: 3.0`) stops the objective at `3 * A`, so tokens with
   a larger ratio add a constant and no gradient.

The loop logs how often each layer acts: `sampler/dropped_episodes`,
`sampler/dropped_unaligned`, `sampler/dropped_over_threshold`,
`sampler/drop_capped`, `train/tis_weight_mean`, `train/tis_capped_frac`,
`train/ratio_capped_frac`, and `train/ratio_over_cap_frac`, plus
`grad_norm_max`. Unit tests cover the math, and a CPU test runs the whole
training phase with a tiny model.

The dr_grpo divisor is pinned to run 3's value (`max_tokens_norm: 1600`, 10
turns of 160 tokens), so doubling the turn limit doesn't halve the gradient
per token. With segments, the loss is still normalized per episode, so a
split episode weighs the same as an unsplit one.

### Run 64 browsers

Run 3 used 32 browsers in training and 16 in evals. Browser time dominated
both. Run 4 sizes the pool as follows:

- `python -m cua.pipeline pool-size` takes 64, lowered to 4 browsers per vCPU
  (from the container's CPU quota) and to what fits in RAM at 0.6 GB per
  browser after 40 GB for vLLM, the trainer, and the page cache. It never
  goes below 16.
- `python -m cua.smoke4 browsers` starts that many browsers at once, resets
  and steps each twice, and halves the count once if more than 5% fail or a
  warm step takes over 20 s. Every later phase uses the count that passed
  (`results/run4/pool_size.txt`).

A pod with 16 vCPUs and 128 GB of RAM gets 64 browsers; one with 12 vCPUs gets
48. To force a count, set `CUA_POOL_SIZE`.

Each rollout round has 12 groups (96 episodes) instead of 8, so browsers that
finish early start the next episode instead of waiting for the round's
longest one. vLLM's memory share rises from 0.35 to 0.42 for the larger KV
cache; run 3's training peak was 17 GB, so about 30 GB remain for training.

### Exclude stock-market from training

`stock-market` is a timing coin flip. The page redraws the price every 100 ms
along a random walk, and the goal asks to buy below `prices[75]`, the price
at 7.5 s. Between the screenshot and the click, 1-3 s pass (generation for
the whole batch, Playwright's 100 ms slow-motion, and the 0.5 s
pre-observation delay), which is 10-30 price steps. So the price the model
sees says almost nothing about the price when it clicks, and a click before
7.5 s succeeds about half the time whatever the model does. After 10 s the
price list runs out and the button always fails. This explains 46%-54% for
every model. Its mixed groups gave run 3 pure noise as advantages, because
run 3 trained on it as a learn task.

A fix would change the task's timing, which makes it another task. Run 4
leaves it out of the pool and the full suite (`run4_exclude` and
`full_suite_exclude` in `configs/tasks.yaml`).

### Keep sign-agreement in held-out, and read it in two parts

`sign-agreement` draws one of three problems. Problem 0 asks to click
**Cancel**, which is always clickable. Problems 1 and 2 ask to scroll a
disabled text area to the bottom, type a name, and click a button; the name
field and the second button stay disabled until the text area is at its
bottom. An emulation of the task's seeded random draws (BrowserGym's seed,
then `Math.seedrandom`, then the word generator and `core.randi(0, 3)`, run
with Node.js on the pinned MiniWoB++ source) shows that exactly 42 of eval
seeds 0-99 draw problem 0, and they are exactly the 42 seeds that the run-3
adapter solved. So every model solved every problem-0 seed and no problem-1 or
problem-2 seed. The list is `sign_agreement_cancel_only_seeds` in
`configs/tasks.yaml`.

The other problems need 1,024-2,056 characters of text scrolled in a
140-pixel text area: about 10-20 of Track B's 100-pixel scrolls, plus 3 more
actions. That is out of reach at 10 turns and at the edge at 20. BrowserGym
overrides the task's own 10-second limit, so time isn't the problem.

Run 4 keeps the task, because held-out v2 must stay exactly as runs 2 and 3
used it. The `compare` phase also writes the held-out comparison without it
(`heldout4-no-sign-run4-vs-run3`). A success on a seed outside the 42 (in
`eval/run4-heldout4/episodes.jsonl`) is evidence of real long-horizon
progress. Whether a mouse wheel scrolls a
disabled text area in headless Chromium isn't verified.

### Report the better of the final and the best adapter

Mid-training evals run every 25 steps on the pool with 20 seeds from the
*baseline* range (seeds 100-119), not the eval range. When one sets a record,
the loop saves that adapter as `best/` with `best.json`, and the sync loop
uploads it. In `post_eval`, the final adapter gets the same eval
(`select-final`), and `cua.pipeline choose-adapter` picks `best/` only when it
beats the final adapter by more than 2 points; otherwise the final adapter
wins, because it trained on more groups and one 20-seed eval is noisy. The
choice goes to `results/run4/adapter_choice.json`. Because selection uses
seeds the headline evals never use, the choice doesn't inflate the reported
numbers.

### Mark every metrics row with a kind

Mid-training eval rows in `metrics.jsonl` repeat the step of the training row
before them. Every row now has `"kind": "train"` or `"kind": "eval"`, so
readers can tell them apart. `--status-json` reports the newest training step
and ignores eval rows. W&B doesn't receive the field.

### Skip RFT

Track B's format-error rate is near 0, so RFT has nothing to fix. The `rft`
phase stays in the phase list for the contract and only records the start
adapter.

### Measure the base model where it's cheap

`BASE_EVAL=auto` evaluates the base model on held-out v2 always and on the
full suite at $3.00 per hour or less (about $1.20 at Community pricing). At
Secure pricing it skips the full suite.

## Task pool

The run-4 candidates are run 3's 71 candidates without `stock-market`, the 18
drag tasks, and 13 registered tasks that no run screened: 99 tasks in all
(`candidates_run4`). The `smoke` phase drops any that browsergym-miniwob
doesn't register. Each candidate has one source, in this order of priority:

| Source | Tasks |
|---|---|
| `new_action` | `use-slider`, `use-slider-2`, `use-colorwheel`, `use-colorwheel-2`, `drag-box`, `drag-circle`, `drag-cube`, `drag-items`, `drag-items-grid`, `drag-shapes`, `drag-shapes-2`, `drag-single-shape`, `drag-sort-numbers`, `highlight-text`, `highlight-text-2`, `resize-textarea`, `draw-line`, and `text-editor` |
| `long_horizon` | The 17 tasks that ran out of turns in half or more of the run-3 screen (`run4_long_horizon`) |
| `run3_pool` | Run 3's 18 pool tasks |
| `other` | Every other candidate |

The `screen` phase runs the run-3 adapter at run 4's settings on every
candidate, 16 seeds each, sampled at temperature 1.0. `select`
(`cua.run_pool.select_run4`) then builds the pool:

- **Learn tasks, up to 18.** It takes tasks in the 5%-85% band and reserves 4
  slots for drag tasks and 5 for long-horizon tasks, nearest 45% success
  first. For drag tasks, one success in 16 is enough, because any success
  gives GRPO a signal. It fills the rest from any source, nearest 45% first,
  so the pool has learnable tasks from step 1 even if the drag tasks stay
  near 0%. Tasks with no success can't teach anything and stay out.
- **Replay tasks, up to 3.** It takes run-3 pool tasks above 85%, most
  fragile first. The adaptive sampler gives them a floor weight of 0.10; a
  replay task that starts to fail turns mixed and trains again.
- **Mastered tasks** above 85% that aren't replay tasks stay out.

Held-out v2 is unchanged: the 10 tasks frozen in `configs/tasks.yaml`, which
`select` checks against `results/run3/heldout_v2.json`.

## Evals

Every headline eval is greedy, on eval seeds that are identical across
models, at run 4's settings (20 turns, the history window, the drag tools, and
the turn budgets):

| Eval | Tasks | Seeds | Models |
|---|---|---|---|
| `*-heldout4` | Held-out v2, 10 tasks | 100 (0-99) | Run 3 and run 4; base |
| `*-full4` | The full suite: every registered task except `full_suite_exclude` (`stock-market`, `draw-circle`, and the six `-nodelay` duplicates), 117 tasks with BrowserGym 0.14.3 | 20 (0-19) | Run 3 and run 4; base at Community pricing |
| `select-final` | The run-4 pool | 20 (100-119) | The final adapter, to choose between it and `best/` |

The `compare` phase writes paired 95% intervals (`cua/compare.py`, stratified
bootstrap) for held-out v2, held-out v2 without `sign-agreement`, the full
suite, and the full suite's pool tasks, turn-limited tasks, drag tasks, and
non-pool tasks. `cua.compare` gained `--tasks` and `--exclude-tasks` for these
subsets.

The report (`python -m cua.report --preset run4`) shows the success criteria,
the full journey from base to run 4 (held-out v2 on Track A from run 2's
evals, on Track B at 10 turns from run 3's evals, and on Track B at 20 turns
from run 4's evals, plus the full suite), the headline cards, per-task charts,
training curves, and replays. Each journey column is one setting, so compare
numbers within a column, not across columns.

## Hyperparameters

The following table lists the settings in `configs/grpo_run4.yaml` that differ
from run 3 or that matter most:

| Setting | Value | Reason |
|---|---|---|
| `env.max_turns` | 20, with per-task budgets of 10-20 | Eight tasks ran out of turns at 10. |
| `env.context_window_turns` | 4 | At most 4 screenshots per prompt; see [Keep prompts small with a history window](#keep-prompts-small-with-a-history-window). |
| `env.track_b_extra_tools` | true | `double_click` and `left_click_drag`. |
| `env.pool_size` | 64, or what the smoke check passed | Browsers set the pace. |
| `prompts_per_step` | 12 | Keeps 64 browsers busy; dynamic sampling stops at 6 mixed groups. |
| `vllm_gpu_memory_utilization` | 0.42 | KV cache for 64 prompts of about 2,000 tokens. |
| `learning_rate`, `warmup_steps` | 1e-5, 10 | Unchanged; fresh AdamW. |
| `updates_per_step`, `max_train_groups` | 2, 6 | Unchanged. |
| `max_tokens_norm` | 1600 | Run 3's gradient scale per token. |
| `ratio_cap` | 3.0 | Dual clip for negative advantages. |
| `sampler_check` | TIS cap 2.0, drop above 0.15 or unaligned, at most 25% | Run 3's step 207. |
| `eval_seed_range`, `keep_best` | `baseline`, true | Choose the best checkpoint without the headline seeds. |
| `train_seed_offset` | 500000 | Run 3 used seeds from 301000. |
| `save_steps`, `eval_every`, `eval_num_seeds` | 10, 25, 20 | Unchanged. |

## Phases, time, and cost

The estimates come from run 3's logs and adjust them for 20 turns and 64
browsers:

- **Browser throughput.** Run 3's standalone evals used 16 browsers and took
  0.195-0.207 s per episode-turn. Its mid-training evals used 32 browsers and
  took 0.123 s. The plan assumes 0.09 s with 64 browsers. If 64 browsers are
  no faster than 32 (a CPU-bound pod), every eval takes 37% longer.
- **Turns per episode.** At 10 turns, run 3's screen averaged 4.6 turns, and
  23% of episodes ran out of turns. At 20 turns, failing episodes run longer,
  and drag tasks mostly fail, so the plan assumes 7 turns per episode in the
  screen, the full suite, and GRPO, and 5.9 in held-out v2.
- **GRPO steps.** Run 3 averaged 82 s of rollouts and 32 s of training. Run 4
  assumes about 95 s per 96-episode round (bounded by the longest episode, up
  to 20 turns) and 2 rounds per step, so 190 s of rollouts, and 65 s of
  training for twice run 3's tokens per episode. With a mid-training eval
  every 25 steps (about 4 minutes), a step takes about 4.4 minutes: about 135
  steps in 10 hours (about 105 if browsers are slower), and about 800 trained
  groups.

The following table shows the phases at both price points. Community runs
`GRPO_MAX_HOURS=10` and `BASE_EVAL=auto` (full); Secure runs
`GRPO_MAX_HOURS=6.5`, and `BASE_EVAL=auto` skips the base model's full suite.

| Phase | Work | Hours (Community) | Cost at $2.69 | Hours (Secure) | Cost at $4.39 |
|---|---|---|---|---|---|
| Pod start | Pod creation, image pull | 0.10 | $0.27 | 0.10 | $0.44 |
| Bootstrap and restore | `pod_bootstrap.sh`, base model, run-3 adapter and results | 0.25 | $0.67 | 0.25 | $1.10 |
| `smoke` | Browser stress, drag and window checks, 3 GRPO steps, mid-training eval, 3-seed eval | 0.40 | $1.08 | 0.40 | $1.76 |
| `screen` | 99 tasks x 16 seeds, sampled, 20 turns | 0.40 | $1.08 | 0.40 | $1.76 |
| `select`, `rft` | Pool, budgets, held-out check | 0.00 | $0.00 | 0.00 | $0.00 |
| `pre_eval` | Run 3: held-out (1,000 episodes) and full suite (2,340); base: held-out, and full suite at Community | 1.20 | $3.23 | 0.77 | $3.38 |
| `grpo` | Time-capped | 10.00 | $26.90 | 6.50 | $28.54 |
| `post_eval` | `select-final`, run 4 on held-out and the full suite | 0.70 | $1.88 | 0.70 | $3.07 |
| `compare`, `report` | Paired bootstraps and the report | 0.05 | $0.13 | 0.05 | $0.22 |
| Final upload | `upload_run4.py --final --verify`, pod deletion | 0.15 | $0.40 | 0.15 | $0.66 |
| **Total** | | **13.25** | **about $35.60** | **9.32** | **about $40.90** |

A lost pod costs about 0.6 hours for a new pod and its bootstrap, plus up to
10 steps (about 45 minutes): about 1.35 hours, $3.60 at Community or $5.90 at
Secure. At Community, one loss ends near $39, under the soft stop. At Secure,
the soft stop at $42 ends GRPO early instead; `post_eval` and the upload then
add about $4, which ends near $46, under the $50 cap. If the monitor leaves
`GRPO_MAX_HOURS` at 10 on a Secure pod, the soft stop fires after about 7.6
hours of GRPO and the run ends near $46 as well.

## Success criteria

All numbers come from greedy decoding at run 4's settings with paired 95%
intervals from `cua/compare.py`. The report's criteria table
(`cua/report.py`, `run4_goals`) checks each one. Run 4 succeeds when the
following are true:

1. **Held-out v2 above 71.5%.** Run 4 beats run 3's held-out v2 result
   (`run4-heldout4`; the paired comparison with the run-3 adapter at run 4's
   settings is `compare/heldout4-run4-vs-run3.md`).
2. **Turn-limited tasks reach 30%.** The 8 tasks that ran out of turns at 10
   turns (`run4_turn_limited`, about 0%-19% in run 3's screen) average at
   least 30% in `run4-full4` (`compare/full4-turn-limited-run4-vs-run3.md`).
3. **Some drag-task success.** At least one drag task scores above 0% in
   `run4-full4` (`compare/full4-new-action-run4-vs-run3.md`).
4. **Full suite up.** Run 4 beats the run-3 adapter on the full suite, and the
   interval excludes 0 (`compare/full4-run4-vs-run3.md`).
5. **No ratio blow-ups.** Every step has a gradient norm of at most 10 and an
   approximate KL of at most 1 (run 3's step 207: 3,709 and 169).

Expected outcome, stated plainly: criteria 2 and 5 are the most likely to
hold, because 20 turns and the guard address their causes directly. Criterion
3 depends on whether the run-3 adapter ever succeeds at a drag in the screen;
if it doesn't, GRPO gets no drag signal, and the report says so. Criterion 1
needs about +3 points on tasks the model already solves 71% of the time, and
the 20-turn settings can move the run-3 adapter's own held-out score either
way.

## Switches

Each mechanism that run 4 adds can be turned off without code changes. Each
is on by default because the `smoke` phase exercises it and fails the run
before training if it breaks. If a mechanism fails on the pod, turn it off,
then rerun with `bash scripts/run4.sh --redo smoke`:

| Mechanism | Turn off with |
|---|---|
| History window | `CUA_SETS="--set env.context_window_turns=0"` (full history; lower the pool size, because the KV cache then holds fewer prompts) |
| Drag and double-click | `CUA_SETS="--set env.track_b_extra_tools=false"` |
| Turn budgets | `TURN_BUDGETS=0` |
| Sampler checks and TIS | `CUA_SETS="--set grpo.sampler_check.enabled=false"` |
| Ratio cap | `CUA_SETS="--set grpo.ratio_cap=null"` |
| 64 browsers | `CUA_POOL_SIZE=32` |
| Best-adapter choice | `BEST_CKPT=PATH` |

`CUA_SETS` reaches every eval and GRPO command, so the before and after
evals stay paired.

## Risks

- **Drags that BrowserGym makes in one move.** Some drag widgets might need
  intermediate mouse moves. The smoke check catches an error, but not a drag
  that runs without an error and does nothing; the screen then shows 0% on
  that task, and the pool leaves it out.
- **No drag success in the screen.** If the run-3 adapter never succeeds at a
  drag task, the pool has no drag tasks, and criterion 3 fails. The other
  criteria don't depend on it.
- **The history log is a new prompt format.** Turns after the fourth see a
  text log that the run-3 adapter never saw. The screen measures the effect
  before training, and GRPO trains on it.
- **Throughput.** The time table assumes 64 browsers are 1.4 times faster
  than 32. On a CPU-bound pod, GRPO gets about 105 steps in 10 hours instead
  of 135. The `smoke` log prints rollout and training seconds per step.
- **Memory.** vLLM at 0.42 plus training at about 20-30 GB fits in 80 GB
  with margin, but long segments with micro-batches of 2 aren't measured.
  The loop halves the micro-batch once on an out-of-memory error.
- **The drop threshold.** 0.15 comes from run 3's per-step means, not
  per-episode values. `sampler/drop_capped` shows when the 25% cap binds; if
  it binds often, the threshold is too tight.
- **Hub storage.** About 0.8 GB per synced checkpoint plus each new `best/`
  adapter (about 0.26 GB).

## Launch

The monitor launches and watches run 4 with the commands in
[RUN4_INTERFACES.md](RUN4_INTERFACES.md). To start it by hand on a pod that
has `/workspace/.secrets`, run the following command:

```bash
cd /root/RL-ComputerUse && bash scripts/run4_pod_start.sh
```

To check progress, run `bash scripts/run4.sh --status`. To end GRPO early and
go on to the post-training evals, create `results/run4/.stop_grpo`.

In the first hour of the `grpo` phase, check these values in
`results/run4/logs/grpo.log` or W&B:

- `dyn/rounds`: mostly 1 or 2.
- `train/sequences` above `episodes/trained`: the window splits long episodes.
- `sampler/dropped_episodes`: mostly 0; `sampler/drop_capped` 0.
- `train/ratio_capped_frac`: small, under 0.01.
- `time/rollout_s`: about 190 s or less.
