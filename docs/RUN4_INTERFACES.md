# Run 4 interfaces

Run 4 keeps the contract in [RUN3_INTERFACES.md](RUN3_INTERFACES.md), with
`run3` replaced by `run4` everywhere: commands, tmux sessions, paths, Hub
folders, and the `"run"` field. This page lists only what differs or what run
4 adds. The plan is [PLAN_RUN4.md](PLAN_RUN4.md).

## Commands the monitor calls

The following commands match run 3's one for one:

| Run 3 | Run 4 |
|---|---|
| `bash scripts/run3_pod_start.sh` | `bash scripts/run4_pod_start.sh` (tmux sessions `run4` and `run4-sync`) |
| `bash scripts/run3.sh --status-json` | `bash scripts/run4.sh --status-json` |
| `touch results/run3/.stop_grpo` | `touch results/run4/.stop_grpo` |
| `python scripts/upload_run3.py --final --verify` | `python scripts/upload_run4.py --final --verify` |

The outputs are unchanged: `ALREADY_RUNNING` or `STARTED`, one JSON object,
and `VERIFIED` or `NOT VERIFIED` as the last line.

## Status JSON

The schema is run 3's. The differences are as follows:

- `"run"` is `"run4"`.
- The phase names and their order are run 3's: `smoke`, `screen`, `select`,
  `rft`, `pre_eval`, `grpo`, `post_eval`, `compare`, and `report`. In run 4,
  `rft` only records the start adapter and takes no time.
- `grpo.max_steps` comes from `configs/grpo_run4.yaml` (400).
- `grpo.step` is the newest row with `"kind": "train"` (see the next
  section).
- The overrides are `RUN4_RESULTS` and `RUN4_CKPT`.

`python3 -m cua.run_status --run run4` prints the same object.
`python3 -m cua.run3_status` still reports run 3.

## metrics.jsonl

Every row has a `kind` field:

- `"kind": "train"`: one row per GRPO step, with the keys of run 3 plus the
  ones in the following list.
- `"kind": "eval"`: a mid-training eval, `{"step", "kind", "eval/success"}`.
  Its `step` repeats the step of the training row before it.

Keep the last row for each `(step, kind)` pair, not for each step. The resume
rule is unchanged: rows with a step above the resumed checkpoint are dropped.

Run 4 adds these training keys:

- `grad_norm_max`, `train/sequences`
- `train/ratio_capped_frac`, `train/ratio_over_cap_frac`
- `train/tis_weight_mean`, `train/tis_capped_frac`
- `sampler/dropped_episodes`, `sampler/dropped_unaligned`,
  `sampler/dropped_over_threshold`, `sampler/drop_capped`, and
  `sampler/max_abs_logp_diff`

## Pod layout

The paths are run 3's with `run4`. Run 4 adds the following:

| Path on the pod | Meaning |
|---|---|
| `/workspace/inputs/run3-adapter` | The run-3 adapter, restored from `checkpoints/grpo-4b-run3/final`. |
| `/workspace/ckpt/grpo-4b-run4/best/` | The adapter with the best mid-training eval, with `best.json` (`{"step", "eval_success", "seed_range", "num_seeds"}`). |
| `results/run4/pool_size.txt` | The browser count that the smoke check passed. |
| `results/run4/turn_budgets.json` | Per-task turn limits, written by `select`. |
| `results/run4/full_suite.json` | The full-suite task list, written by `smoke`. |
| `results/run4/adapter_choice.json` | Which adapter `post_eval` reported, final or best, and why. |
| `results/run3/`, `results/run2/` | Run-3 and run-2 summaries for the report's journey table, restored from the Hub when missing. |

The variables that `run4_pod_start.sh` passes through are run 3's, with
`RUN3_ADAPTER` and `RUN3_HF_PATH` in place of the run-2 ones, plus
`BASE_EVAL`, `TURN_BUDGETS`, and `CUA_POOL_SIZE`. `PRICE_PER_HR` defaults to
2.69, and `GRPO_MAX_HOURS` to 10. For a Secure pod at $4.39 per hour, pass
`GRPO_MAX_HOURS=6.5`.

## Hub layout

Run 4 adds `checkpoints/grpo-4b-run4/best/`. The sync loop uploads it each
time `best.json` changes, restore brings it back on a new pod, and the final
upload includes it. Commit messages start with `run4:`.

## Eval names

The `evals.json` keys, and the eval folders under `results/run4/eval/`, are
the following:

| Name | Model | Tasks and seeds |
|---|---|---|
| `baseline` | Run-3 adapter, sampled | Run-4 candidates, 16 seeds (the screen) |
| `run3-heldout4`, `run4-heldout4`, `base-heldout4` | Run 3, run 4, base | Held-out v2, 100 seeds |
| `run3-full4`, `run4-full4`, `base-full4` | Run 3, run 4, base | Full suite, 20 seeds |
| `select-final` | Final run-4 adapter | Run-4 pool, baseline seeds 100-119 |
| `grpo-stepN-train` | Mid-training | Run-4 pool, baseline seeds 100-119 |

`base-full4` runs only when `BASE_EVAL` resolves to `full`. Each
`summary.json` also holds `context_window_turns`, `track_b_extra_tools`, and
`turn_budgets`.

## Samples

`episode.json` is unchanged. Its `action` field can also show
`double_click(coordinate=[x, y])` and
`left_click_drag(start_coordinate=[x, y], end_coordinate=[x, y])`.
