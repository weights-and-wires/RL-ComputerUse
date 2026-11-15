# RL for computer use: one-day training plan

This plan describes a learning project: take a small open-weights vision-language
model (VLM) and use reinforcement learning (RL) to make it better at operating a
web browser. You train it on one rented Runpod GPU in one day and finish with a
before-and-after comparison you can see: success-rate charts and side-by-side
replays of the agent clicking through tasks.

- Planned on: 2026-11-14.
- Training day: Saturday, 2026-11-21.
- Budget: about $50 of GPU time, all smoke tests included.
- Experiment tracking: Weights & Biases (W&B).

## Goal and success criteria

The project succeeds when all of the following are true:

1. The trained model's success rate on the training-task pool rises by at least
   15 percentage points over the base model, measured on held-out random seeds
   with greedy decoding.
2. The success rate on a held-out task set (tasks never used in training) does
   not drop, and ideally rises.
3. The repository produces an HTML report with per-task bar charts, training
   curves, and GIF replays of base and trained episodes on the same seeds.
4. The LoRA adapter is saved on the network volume and, if the token allows it,
   in a private Hugging Face repository.

## Key decisions

The following table summarizes each decision and the reason for it.

| Area | Decision | Reason |
|---|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0, strong for its size, native GUI grounding (coordinates normalized to 0-1000), first-class vLLM support, and used in a 2026 MiniWoB GRPO study (arXiv 2607.12640). |
| Fallback model | `Qwen/Qwen3-VL-2B-Instruct` | Same code path, half the cost per rollout. Switch if 4B throughput is too low. |
| Environment | MiniWoB++ through BrowserGym (`browsergym-miniwob`), run locally with headless Playwright Chromium | Runs in a plain container without KVM, resets in milliseconds, and returns a programmatic reward. OSWorld and AndroidWorld need VMs, which Runpod pods can't run. |
| Observation | Screenshot (long side 512 px) + task goal + accessibility tree (AXTree) truncated to about 2,000 characters | Multimodal like a real computer-use agent, but the AXTree element IDs keep the action space learnable in one day. |
| Action space | Track A (main): BrowserGym element-ID actions such as `click("12")` and `fill("7", "hello")`. Track B (stretch): pure coordinate actions such as `mouse_click(x, y)` with screenshot only. | Track A gives reliable gains in one day. Track B is the "true" pixel-level computer use and is a stretch goal. |
| RL algorithm | GRPO (group relative policy optimization), multi-turn, episode-level reward | No value network, which saves memory. Fits a sparse 0/1 success reward. |
| Framework | TRL `GRPOTrainer` with `environment_factory`, vLLM in colocate mode, PEFT LoRA | Easiest to read and debug on one GPU. TRL ships a BrowserGym GRPO example to start from. |
| Fallback framework | A small hand-written GRPO loop (`cua/nano_grpo/`) that uses vLLM for rollouts and PyTorch for the loss | Used only if TRL's multi-turn VLM path breaks with Qwen3-VL. It's also the best way to learn what GRPO does. |
| Warm start | Rejection-sampling fine-tuning (RFT): sample the base model at temperature 1.0, keep only successful episodes, and run one short SFT pass | Fixes output-format errors and raises low success rates into GRPO's useful range. Needs no stronger teacher model. |
| GPU | 1x H100 80GB (PCIe or SXM), Secure Cloud, at least 16 vCPUs | Enough memory for vLLM and LoRA training on one card. 16+ vCPUs run 16-32 browsers. Two GPUs aren't worth it for one day. |
| Storage | 100 GB Runpod network volume at `/workspace` | Checkpoints survive a pod crash or termination. |

### Why task selection matters most

GRPO learns only from groups whose episodes have different outcomes. If all
eight samples for a prompt fail, or all succeed, every advantage is zero and the
model learns nothing from that prompt. The 2026 MiniWoB study found that GRPO
gave up to +22 points on tasks where the base model sometimes succeeded, and no
gain on tasks it had already mastered. Therefore, the first measured step of the
day is a sampled baseline that keeps only tasks with a base success rate between
about 10% and 80%.

## Architecture

The training loop runs entirely on one pod. The following diagram shows how the
pieces connect:

```text
            +----------------------------- H100 80GB ------------------------------+
            |  vLLM engine (colocate, ~30-40% memory)   PyTorch trainer (LoRA)       |
            |        ^  generate actions                    ^  GRPO loss, backward    |
            +--------|-------------------------------------|--------------------------+
                     |                                     |
       prompts (screenshot + goal + AXTree)      token log-probs + rewards
                     |                                     |
        +------------+-------------------------------------+-----------+
        |  Rollout manager: G episodes per task seed, up to 8 turns each |
        +------------+---------------------------------------------------+
                     |  BrowserGym actions / observations
        +------------v---------------------------------------------------+
        |  Pool of 16-32 headless Chromium pages running MiniWoB++ tasks  |
        |  (file:// URLs from a local clone of miniwob-plusplus)          |
        +----------------------------------------------------------------+
```

## Repository layout

The implementing session creates the following layout:

```text
RL-ComputerUse/
  README.md                 # Quickstart and links to the docs
  pyproject.toml            # uv-managed dependencies, pinned versions
  .env.example              # Names of required variables, no values
  docs/
    PLAN.md                 # This file
    LEARNING.md             # Concepts to study before Saturday
    RUNBOOK.md              # Hour-by-hour Saturday checklist with exact commands
  configs/
    tasks.yaml              # Candidate task pool, held-out tasks, seeds
    eval.yaml               # Eval settings (k, temperature, max turns)
    rft.yaml                # Warm-start SFT settings
    grpo_qwen3vl4b.yaml     # Main GRPO settings
    grpo_qwen3vl2b.yaml     # Fallback settings
  cua/
    env.py                  # BrowserGym MiniWoB wrapper: reset, step, obs formatting
    prompts.py              # System prompt and per-turn message builder
    actions.py              # Parse model output into a BrowserGym action string
    rewards.py              # Success reward plus shaping terms
    rollout.py              # Multi-turn episode runner over a generate() callable
    policy.py               # vLLM and HF loaders for base model plus optional LoRA
    eval.py                 # pass@k and greedy eval, per-task CSV, GIF recording
    select_tasks.py         # Pick the headroom tasks from baseline results
    rft.py                  # Collect successful episodes and run SFT
    train_grpo.py           # TRL GRPOTrainer entry point (main path)
    nano_grpo/              # Hand-written GRPO fallback
      loop.py
      loss.py
    report.py               # Build the before-and-after HTML report
  scripts/
    runpod_up.sh            # Create network volume (once) and pod via runpodctl or REST
    runpod_down.sh          # Stop or delete the pod; keep the volume
    pod_bootstrap.sh        # Run on the pod: apt deps, uv env, Playwright, clone MiniWoB
    sync_results.sh         # rsync results and reports back to the Mac
    saturday.sh             # Runs each phase in order, resumable, with logs
  tests/
    test_actions.py         # Parser cases, including malformed outputs
    test_prompts.py         # Message shapes and token-budget checks
    test_rewards.py
```

## Component specifications

### Environment wrapper (`cua/env.py`)

- Create environments with `gym.make("browsergym/miniwob.TASK_NAME")` and set
  `MINIWOB_URL` to the `file://` path of `miniwob-plusplus/miniwob/html/miniwob/`
  (keep the trailing slash).
- Launch Chromium headless with `--disable-dev-shm-usage` and `--no-sandbox`.
- Return an observation object with: goal text, screenshot as a PIL image
  resized so the long side is 512 px, AXTree text with BrowserGym element IDs
  truncated to `max_axtree_chars`, the last action error (if any), and the step
  number.
- Support a pool of N environments that the rollout manager reuses. Close and
  recreate a browser after M episodes or on any exception, to avoid leaked
  Chromium processes over a long run.
- Record frames per step when `record=True`, for GIF replays.

### Prompt and action format (`cua/prompts.py`, `cua/actions.py`)

- The system prompt describes the allowed actions and requires this output
  format: a short `<think>` block (at most about 60 words) followed by exactly one
  `<action>` block containing one BrowserGym action call.
- Track A actions: `click(bid)`, `fill(bid, text)`, `select_option(bid, option)`,
  `press(bid, key)`, `scroll(dx, dy)`, `noop()`.
- Track B actions: `mouse_click(x, y)`, `keyboard_type(text)`,
  `keyboard_press(key)`, `scroll(dx, dy)`. The model emits coordinates in its native
  0-1000 normalized space; the parser maps them to page pixels.
- History: keep the text of all previous actions, but only the most recent
  screenshot, to bound the token count.
- The parser never raises. It returns `(action_string | None, parse_error)`.
  An unparseable output becomes `noop()` and counts as a format error.

### Rewards (`cua/rewards.py`)

The episode reward is defined as follows:

- `1.0` if MiniWoB reports success (its raw reward is greater than 0), else `0.0`.
- `-0.1` per format error, capped at `-0.3` per episode.
- A small efficiency bonus applied only to successful episodes:
  `0.1 * (1 - turns_used / max_turns)`.

Log each component separately to W&B so reward hacking is visible.

### Evaluation (`cua/eval.py`)

- **Sampled baseline (for task selection):** for each candidate task, 16 seeds x
  1 sample at temperature 1.0. Output per-task success rate.
- **Greedy eval (headline number):** 50 held-out seeds per task, temperature 0.
  Run on the training pool and on the held-out task set, for both the base model
  and each checkpoint.
- Write `results/eval/RUN_NAME/per_task.csv`, `summary.json`, and GIFs for the
  first 3 seeds of every task.
- Training seeds and eval seeds must not overlap. Fix them in `configs/tasks.yaml`.

### Task pool (`configs/tasks.yaml`)

Start with about 25 candidate tasks, keep 10-15 after the baseline, and hold out
5 for generalization. Candidates:

- Training candidates: `click-button`, `click-button-sequence`, `click-checkboxes`,
  `click-checkboxes-soft`, `click-collapsible`, `click-collapsible-2`,
  `click-dialog`, `click-dialog-2`, `click-link`, `click-option`, `click-tab`,
  `click-tab-2`, `click-widget`, `choose-list`, `enter-date`, `enter-password`,
  `enter-text`, `enter-text-dynamic`, `focus-text-2`, `login-user`,
  `navigate-tree`, `search-engine`, `social-media`, `use-autocomplete`,
  `email-inbox`.
- Held-out tasks: `click-test-2`, `enter-text-2`, `login-user-popup`,
  `click-checkboxes-large`, `email-inbox-forward-nl`.

The implementing session must confirm each name exists in the installed
`browsergym-miniwob` version and replace any that don't.

### Warm start (`cua/rft.py`)

1. Sample 32 episodes per selected task at temperature 1.0 with the base model.
2. Keep successful episodes with no format errors; cap at 64 per task so easy
   tasks don't dominate.
3. Run LoRA SFT for 1 epoch, learning rate `1e-4`, on the assistant turns only.
4. Skip this phase if the baseline already shows a format-error rate below 5%
   and a mean success rate above 30% on the selected pool.

### GRPO training (`cua/train_grpo.py`)

Start from TRL's `examples/grpo_browsergym/grpo_browsergym.py`. Replace its
OpenEnv Space client with the local environment wrapper. Starting settings:

| Setting | Value |
|---|---|
| LoRA | rank 32, alpha 64, dropout 0.05, language-model attention and MLP projections only; vision tower frozen |
| Learning rate | `1e-5` (LoRA), constant with 10 warmup steps. The 2026 study saw collapse at `3e-6` and higher for full fine-tuning; LoRA tolerates more, but stay low. |
| Group size (`num_generations`) | 8 |
| Prompts per step | 8 task seeds, so 64 episodes per optimizer step |
| Max turns per episode | 8 (Track A), 10 (Track B) |
| Max new tokens per turn | 160 |
| KL coefficient (beta) | 0.0 |
| Clipping | epsilon low 0.2, epsilon high 0.28 ("clip-higher") |
| Loss | `dr_grpo` or token-level mean if available in the installed TRL |
| vLLM | `use_vllm=True`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization` 0.35 to start |
| Precision | bf16, gradient checkpointing on |
| Steps | 150-250, as time allows |
| Checkpoints | every 25 steps to `/workspace/ckpt`, keep the last 4 |
| Mid-training eval | every 50 steps, greedy, 10 seeds per task, logged to W&B |

Log to W&B: mean reward, success rate per task, fraction of groups with zero
reward variance, format-error rate, mean turns, KL estimate, entropy, gradient
norm, and episode throughput.

### Fallback: `cua/nano_grpo/`

A loop of about 300 lines with no TRL dependency:

1. Load the base model plus LoRA in PyTorch and the same weights in vLLM.
2. For each step, run G episodes per seed with vLLM, recording the token IDs and
   sampling log-probabilities for every assistant turn.
3. Compute group-normalized advantages: `(r - mean(group)) / (std(group) + 1e-4)`.
4. Compute the clipped policy-gradient loss over assistant tokens only, with
   every turn of an episode sharing that episode's advantage.
5. Update LoRA, then load the new LoRA weights into vLLM (`LoRARequest` with a
   new adapter ID each sync).

Keep the code heavily commented; it doubles as study material.

### Report (`cua/report.py`)

Generates `results/report/index.html` with:

- Bar chart: base versus RFT versus GRPO success rate per task, training pool and
  held-out pool shown separately.
- Training curves exported from W&B or the local JSONL log.
- A grid of GIF pairs (base and trained) on identical seeds.
- A table of failure modes: format errors, wrong element, ran out of turns.

## Runpod operations

- Read `RUNPOD_API_KEY`, `HF_TOKEN`, and `WANDB_API_KEY` from `~/.api_keys` on the
  Mac. Pass them to the pod as environment variables. Never write them to the
  repository or print them in logs.
- Use `runpodctl` (`brew install runpod/runpodctl/runpodctl`) or the REST API at
  `https://rest.runpod.io/v1/pods`. GPU ID: `NVIDIA H100 80GB HBM3` (SXM) or
  `NVIDIA H100 PCIe`. Set `minVCPUPerGPU` to 16 and `cloudType` to `SECURE`.
- Base image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. Install a
  separate `uv` virtual environment and let vLLM choose its own PyTorch version.
- Create the network volume in a datacenter that has H100s in stock, and attach
  it at pod creation.
- Use SSH over exposed TCP (not the proxy) so `rsync` and port forwarding work.
- `scripts/runpod_up.sh` prints the hourly price and asks for confirmation before
  creating anything. `scripts/runpod_down.sh` stops the pod and never deletes the
  network volume unless passed `--delete-volume`.

## Saturday schedule and budget

The following timeline assumes an H100 at about $2.90-$3.00 per hour. Prices
change; `runpod_up.sh` shows the live price.

| Time | Phase | Output | GPU hours |
|---|---|---|---|
| 09:00 | Create pod, run `pod_bootstrap.sh`, smoke test one episode | Working environment | 0.5 |
| 09:30 | Sampled baseline on 25 candidate tasks | `select_tasks.py` picks the pool | 0.75 |
| 10:15 | Greedy baseline on selected and held-out pools | Base numbers and GIFs | 0.5 |
| 10:45 | RFT warm start (optional) plus greedy eval | RFT numbers | 1.0 |
| 11:45 | GRPO run, Track A | Checkpoints, W&B curves | 5.0 |
| 16:45 | Final greedy eval of best checkpoint | Trained numbers and GIFs | 0.75 |
| 17:30 | Report, push adapter, stop pod | `results/report/index.html` | 0.25 |
| Optional | Track B (coordinate-only) short GRPO run | Stretch result | 2.0 |

Core total: about 9 GPU hours, roughly $27. With Track B and slack for
mistakes, the total stays under $50.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| TRL `environment_factory` doesn't handle Qwen3-VL image tool outputs | Switch to `cua/nano_grpo/`. Decide within the first 30 minutes of the GRPO phase. |
| Rollout throughput too low | Drop to Qwen3-VL-2B, cut screenshot to 384 px, cut max turns to 6, raise browser pool size. |
| No gain after 100 steps | Check the zero-variance group fraction. If it's above 60%, drop mastered and impossible tasks and resample the pool. |
| Training collapse (reward drops, entropy spikes) | Lower learning rate to `5e-6`, resume from the last good checkpoint. |
| vLLM out of memory in colocate mode | Lower `vllm_gpu_memory_utilization`, shorten context, reduce prompts per step. |
| Chromium leaks or zombie processes | Recycle browsers every M episodes; a watchdog logs process counts. |
| Pod failure | Network volume plus checkpoints every 25 steps; `saturday.sh` resumes from the last completed phase. |
| HF token is read-only | Keep the adapter on the volume and `rsync` it to the Mac. |

## Prerequisites for you

Before Saturday, complete the following:

1. Add `export WANDB_API_KEY=...` to `~/.api_keys`.
2. Add your SSH public key in the Runpod console under **Settings > SSH Public Keys**.
3. Add credit to your Runpod account (at least $50).
4. Accept the model license on Hugging Face if one is shown (Qwen3-VL is
   Apache-2.0, so normally there's nothing to accept).
5. Read `docs/LEARNING.md`.

## References

- [TRL GRPO trainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL BrowserGym GRPO example](https://github.com/huggingface/trl/tree/main/examples/grpo_browsergym)
- [BrowserGym action space](https://browsergym.readthedocs.io/latest/core/action_space.html)
- [MiniWoB++ repository](https://github.com/Farama-Foundation/miniwob-plusplus)
- [GRPO on MiniWoB with Qwen3-VL (arXiv 2607.12640)](https://arxiv.org/abs/2607.12640)
- [WebAgent-R1: multi-turn RL for web agents](https://arxiv.org/abs/2505.16421)
- [UI-R1: rule-based RL for GUI grounding](https://arxiv.org/abs/2503.21620)
- [SFT and RL compute allocation for web agents (arXiv 2507.04103)](https://arxiv.org/abs/2507.04103)
- [Runpod pod creation API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [Runpod storage types](https://docs.runpod.io/pods/storage/types)

## Decisions log

The implementing session checked the plan against library source and docs on
2026-11-14, by reading only, with nothing installed or run. The following
entries record where the implementation deviates from or refines the earlier
sections. Where an entry conflicts with an earlier section, the entry wins.

### Versions

The project pins the following stack in `pyproject.toml`:

| Package | Pin | Reason |
|---|---|---|
| Python | 3.12 | vLLM 0.33.0 needs 3.11-3.14; TRL 1.18.0 needs 3.11 or later. |
| `vllm` | 0.33.0 | Latest release. Supports Qwen3-VL (since 0.11.0) and LoRA on its language model. Pulls `torch==2.14.0`. |
| `transformers` | 5.20.2 | vLLM 0.33.0 requires `>=5.16.1,<5.21.0`. TRL's `environment_factory` needs 5.2.0 or later. |
| `trl` | 1.18.0 | Latest release; accepts `vllm>=0.23,<0.34`. |
| `peft` | 0.21.3 | Latest release. |
| `browsergym-miniwob` | 0.14.3 | Latest release; pins `playwright==1.44`. |
| `miniwob-plusplus` | commit `7fd85d71a4b60325c6585396ec4f48377d049838` | The commit BrowserGym's own setup uses. |

- The base image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` exists. Its
  PyTorch is unused: `pod_bootstrap.sh` builds a separate `uv` environment.
- Open risk: the PyPI build of vLLM 0.33.0 might target CUDA 13, which needs
  host driver R580 or later. `pod_bootstrap.sh` prints the driver version and
  `docs/RUNBOOK.md` gives the fallback (`--torch-backend=cu129`).

### Environment

- All 30 task names in `configs/tasks.yaml` exist in `browsergym-miniwob`
  0.14.3. No replacements were needed.
- MiniWoB screenshots are about 498x321 pixels (a 332x214 viewport at 1.5x
  zoom), not larger pages that need downscaling. A screenshot costs about 160
  Qwen3-VL image tokens.
- BrowserGym rewards are already binary (`1.0` if MiniWoB's raw reward is
  above 0). The raw value is in `info["task_info"]["RAW_REWARD_GLOBAL"]`.
- BrowserGym disables MiniWoB's 10-second timer and removes the goal text from
  the page, so slow inference doesn't fail episodes, and the goal must be in
  the prompt.
- BrowserGym doesn't let you add Chromium flags (`args` in
  `pw_chromium_kwargs` raises `TypeError`). Playwright adds `--no-sandbox`
  and `--disable-dev-shm-usage` by itself in headless mode; the smoke test
  confirms this on the pod.
- Coordinate actions take screenshot pixels; BrowserGym divides by its zoom
  factor. Track B maps the model's 0-1000 coordinates to screenshot pixels.
- Playwright's synchronous API is bound to the thread that created it, so
  parallel browsers run in worker processes (`RemoteEnv` and `EnvPool` in
  `cua/env.py`), not threads.

### Action format: Qwen tool calls instead of `<think>` and `<action>`

TRL's `environment_factory` turns the environment's methods into tools, and
the model calls them in Qwen's native format:

```text
The OK button has bid 12.
<tool_call>
{"name": "click", "arguments": {"bid": "12"}}
</tool_call>
```

To keep training and evaluation identical, every path uses this format: TRL,
`nano_grpo`, RFT, and eval. Qwen3-VL-Instruct is also trained on this format,
so it's likely to have fewer format errors than a custom one. Consequences:

- Tools mirror BrowserGym. Track A: `click`, `fill`, `select_option`,
  `press`, `scroll(direction)`, and `noop`. Track B: `mouse_click(x, y)`,
  `keyboard_type`, `keyboard_press`, `scroll(direction)`, and `noop`.
- An episode is one growing multi-turn conversation in which each tool result
  carries the new screenshot and page text. All screenshots stay in the
  conversation, as TRL does. At about 160 tokens each, 8 turns fit in 8,192
  tokens.
- A response without a tool call ends the episode (TRL does the same) and
  counts as a format error. A tool call with bad arguments returns the error
  to the model and also counts as a format error.
- Reasoning is optional plain text before the tool call. Qwen3-VL-Instruct has
  no thinking mode, so there's no `<think>` block.

### Default trainer: `nano_grpo`, not TRL

TRL 1.18.0 calls each environment's `reset()` and tool methods serially across
all 64 rollouts in a batch. Async tools run concurrently only within one
completion. At about 1.5 seconds per reset and 0.7 seconds per step, that's
about 4 minutes per GRPO step on browser time alone, or fewer than 80 steps in
the 5-hour slot. `cua/nano_grpo/` steps 32 browsers in parallel, so its
estimated step time is about 1 minute.

Therefore, `trainer: nano` is the default in the GRPO configs, and the TRL
path (`cua/train_grpo.py`) is the alternative. Neither path has been run. The
runbook decides between them after a short smoke run on Saturday.

### Rendering and tokenization in one place

`cua/policy.py` renders every conversation with the Hugging Face chat template
(`apply_chat_template(..., tools=...)`), the same call TRL makes. vLLM receives
the rendered prompt and images through `llm.generate(...)`, rather than parsing
chat messages itself. `tokenize_episode` builds the assistant-token loss mask
that both `nano_grpo` and RFT use.

### RFT uses a small custom SFT loop

RFT trains with a short explicit loop over `tokenize_episode`, not TRL's
`SFTTrainer`. That gives an identical loss mask to `nano_grpo` and avoids
untested interactions between `SFTTrainer`, tools, and images.

### Hyperparameter and config notes

- vLLM returns log-probs before temperature scaling by default. The configs set
  `logprobs_mode: processed_logprobs`.
- vLLM accepts only LoRA ranks 8, 16, 32, 64, 128, and higher powers of two,
  so rank 32 fits.
- LoRA targets match only `language_model` projections with a regex, so the
  vision tower stays frozen.
- In `nano_grpo`, the "old" log-probs for the PPO ratio are recomputed by the
  training model just before the update, rather than taken from vLLM. vLLM's
  values are logged only as a mismatch diagnostic. With `updates_per_step: 1`,
  the ratio is 1 and the update is plain group-baselined REINFORCE. Raising it
  to 2 makes clipping active.
- TRL's `max_prompt_length` no longer exists. `max_completion_length` covers
  the whole episode, including tool text and image tokens, and is set to
  6,144. `max_tool_calling_iterations` caps the turns.
- TRL's `loss_type` defaults to `dapo`; the configs set `dr_grpo` as planned.
- Configs inherit from `configs/base.yaml` through an `extends:` key, and every
  command accepts `--set key=value` overrides. `configs/grpo_trackB.yaml` holds
  the stretch run.
- Seed ranges are disjoint by construction: eval 0-99, baseline 100-199, RFT
  200-999, and train from 1000. The config loader refuses overlapping ranges.

### Runpod

- `runpodctl` is at v3 with noun-verb commands (`runpodctl pod create
  --gpu-id ...`). The legacy camelCase flags are deprecated. The scripts use
  the REST API (`https://rest.runpod.io/v1`) with `curl` and `jq`, and
  `runpodctl` only for optional price lookup.
- REST field names in the plan (`gpuTypeIds`, `cloudType`, `minVCPUPerGPU`,
  `networkVolumeId`, `ports`, and `env`) are correct. The hourly price is the
  pod's `costPerHr`. The SSH port is in `portMappings["22"]`.
- The container's SSH setup reads `PUBLIC_KEY`. Runpod's docs also mention
  `SSH_PUBLIC_KEY`. The scripts set `PUBLIC_KEY`, and the runbook also has you
  add the key to your Runpod account.

### Layout additions

The implementation adds these files to the repository layout:

- `cua/types.py`: the shared dataclasses (`Observation`, `Generation`,
  `ParsedAction`, `Turn`, `Episode`, `RewardBreakdown`, and others).
- `cua/config.py`: the one config loader, with `extends:`, `--set`
  overrides, task lists, and seed ranges.
- `configs/base.yaml` and `configs/grpo_trackB.yaml`.
- `scripts/lib_runpod.sh`: shared helpers for the Mac-side Runpod scripts.

### Consistency rules found in review

- The opening user message starts with `prompts.USER_PREAMBLE`, followed by
  the first observation blocks. The TRL dataset prompt holds the same
  sentence, and TRL appends `reset()`'s blocks after it, so all paths render
  identical text.
- TRL builds tool schemas from the environment's method type hints. The TRL
  tool methods are annotated `-> str` like the stubs in `cua/actions.py`, so
  the rendered schema matches the other paths.
- `saturday.sh` stops GRPO at the time budget with `SIGINT`, which
  `nano_grpo` turns into a final checkpoint.
- `RemoteEnv` starts its worker processes without re-importing the trainer's
  `__main__` module, so 32 browser workers don't each load PyTorch.

### Run 2

Run 2 continues GRPO from the run-1 adapter. Its plan, results criteria, and
launch commands are in [PLAN_RUN2.md](PLAN_RUN2.md). The code changes keep
run 1 reproducible: `configs/grpo_qwen3vl4b.yaml` still uses the round-robin
sampler with dynamic sampling off. One change applies to both runs: the
trainer computes logits only at assistant-token positions
(`grpo.assistant_only_logits`, default on), which gives the same loss with far
less memory.
