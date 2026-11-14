# RL-ComputerUse

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
