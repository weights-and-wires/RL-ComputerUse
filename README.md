# Qwen3-VL (4B) Computer Use (RL, GRPO)

A one-day learning project: RL post-training of `Qwen/Qwen3-VL-4B-Instruct` with
GRPO to make it better at browser tasks in MiniWoB++, trained on one Runpod H100.

- [Plan](docs/PLAN.md): design decisions, component specifications, schedule, and budget.
- [Learning guide](docs/LEARNING.md): the concepts to study before training day.
- [Runbook](docs/RUNBOOK.md): the hour-by-hour checklist for training day, with
  decision points and recovery steps.

## Quickstart

Keep your keys in `~/.api_keys` as `export NAME=value` lines (`RUNPOD_API_KEY`,
`HF_TOKEN`, `WANDB_API_KEY`). The scripts never print or log the values. Follow
the [runbook](docs/RUNBOOK.md) for the full procedure. The main commands are:

```bash
# Mac
scripts/runpod_up.sh --datacenter DATACENTER_ID --dry-run   # preview the request
scripts/runpod_up.sh --datacenter DATACENTER_ID             # create volume and pod
scripts/sync_results.sh --push-code                         # copy the repo to the pod
scripts/sync_results.sh --adapters                          # pull results and adapters
scripts/runpod_down.sh --stop                               # stop billing, keep the volume
```

```bash
# Pod, inside tmux
bash /workspace/RL-ComputerUse/scripts/pod_bootstrap.sh     # one-time setup
source /workspace/env.sh && cd /workspace/RL-ComputerUse
bash scripts/saturday.sh                                    # all phases, resumable
bash scripts/saturday.sh --only grpo                        # one phase
bash scripts/saturday.sh --status                           # phase status and spend estimate
```

The dependencies include vLLM, which installs only on Linux with an NVIDIA GPU, so run
the tests and smoke tests on the pod (`pytest -q tests`, `python -m cua.env --task click-button --seed 0`).

The pod bills every hour until you stop or delete it.

## Watch the model do tasks

The demo runs on your laptop, with no pod. It loads the model with Hugging Face
transformers (Apple silicon GPU, CUDA, or CPU), drives a headless Chromium
browser, and serves a page where you pick a task and watch each turn: the
screenshot, the model's reasoning, its action, and the reward.

To start it, run the following command from the repository root:

```bash
scripts/demo.sh
```

The first run installs everything into `.demo/`, downloads the run-1 and run-2
adapters from the private Hugging Face repository (it needs `HF_TOKEN`), and
downloads the 9 GB base model. Then it opens `http://127.0.0.1:8000/`.

- To also watch a visible, slowed-down Chromium window, add `--show-browser`.
- To compare models, pick **Original model**, **Run 1**, or **Run 2** in the UI.
- Pages come from the eval seeds (0-99), which no training run used.

Each turn takes several seconds on a laptop. Results can differ a little from
the published scores, because evaluation ran in vLLM on a GPU.
