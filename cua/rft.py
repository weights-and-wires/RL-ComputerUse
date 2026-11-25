"""Rejection-sampling fine-tuning (RFT), the warm start before GRPO.

Three subcommands:

    python -m cua.rft should_skip --config configs/rft.yaml   # exit 0 = skip RFT, 1 = run it
    python -m cua.rft collect     --config configs/rft.yaml   # sample episodes, keep the good ones
    python -m cua.rft train       --config configs/rft.yaml   # supervised fine-tuning on them

Why RFT first? The base model sometimes writes malformed tool calls or gets
zero reward on most tasks. GRPO learns only from groups whose rewards differ,
so a policy that always scores zero gives no gradient. RFT is the simplest
fix: sample from the base model, keep only the episodes that succeeded with a
clean format, and fine-tune on those. It is "behavior cloning on your own best
samples".

Output layout:

    results/rft/<run_name>/episodes.jsonl      one summary per sampled episode
    results/rft/<run_name>/stats.json          kept counts and success rates per task
    results/rft/<run_name>/dataset/conversations.jsonl
    results/rft/<run_name>/dataset/images/*.png
    results/train/<run_name>/metrics.jsonl     SFT metrics
    <sft.output_dir>/final/                    the LoRA adapter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from cua.config import (
    Config,
    add_config_args,
    config_from_args,
    load_selected_tasks,
    load_tasks,
    resolve_task_list,
    results_dir,
)

DEFAULT_CONFIG = "configs/rft.yaml"


def _rft_dir(cfg: Config) -> Path:
    return results_dir(cfg) / "rft" / cfg.run_name


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def _lookup(summary: dict, *keys: str) -> Any:
    """Find the first of `keys` at the top level or one level down.

    The baseline summary's exact layout belongs to cua/eval.py, so this reads
    tolerantly instead of assuming one shape.
    """
    for key in keys:
        if key in summary:
            return summary[key]
    for value in summary.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    return value[key]
    return None


def _per_task_success(summary: dict) -> dict[str, float]:
    """Return {task: success_rate} from `summary["per_task"]` (a dict or a list of dicts)."""
    per_task = summary.get("per_task", {})
    out: dict[str, float] = {}
    if isinstance(per_task, dict):
        for task, value in per_task.items():
            rate = value.get("success_rate", value.get("success")) if isinstance(value, dict) else value
            if rate is not None:
                out[task] = float(rate)
    else:
        for row in per_task:
            rate = row.get("success_rate", row.get("success"))
            if rate is not None:
                out[row["task"]] = float(rate)
    return out


def cmd_should_skip(cfg: Config) -> int:
    """Exit code 0 means skip RFT, 1 means run it. Prints the reason either way."""
    path = results_dir(cfg) / "eval" / "baseline" / "summary.json"
    if not path.exists():
        print(f"RUN: {path} not found, so there is no baseline to judge. Running RFT.")
        return 1
    summary = json.loads(path.read_text())

    fmt_rate = _lookup(summary, "format_error_rate")
    selected = load_selected_tasks(cfg)
    rates = _per_task_success(summary)
    chosen = [rates[t] for t in selected if t in rates]
    if fmt_rate is None or not chosen:
        print("RUN: the baseline summary lacks format_error_rate or per-task success for the selected tasks.")
        return 1

    mean_success = sum(chosen) / len(chosen)
    max_fmt = float(cfg.rft.skip_if_format_error_rate_below)
    min_success = float(cfg.rft.skip_if_mean_success_above)
    detail = (
        f"format_error_rate={fmt_rate:.3f} (skip below {max_fmt}), "
        f"mean success on {len(chosen)} selected tasks={mean_success:.3f} (skip above {min_success})"
    )
    if fmt_rate < max_fmt and mean_success > min_success:
        print(f"SKIP: {detail}. The base model is already good enough.")
        return 0
    print(f"RUN: {detail}.")
    return 1


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def _serialize_messages(messages: list[dict], image_dir: Path, dataset_dir: Path, prefix: str) -> list[dict]:
    """Return a JSON-safe copy of `messages`, saving PIL images as PNG files.

    Each image block `{"type": "image", "image": <PIL>}` becomes
    `{"type": "image", "image": "images/<name>.png"}`. The path is relative to
    the dataset directory, so you can move the folder without breaking it.
    """
    out: list[dict] = []
    counter = 0
    for message in messages:
        new_message = {k: v for k, v in message.items() if k != "content"}
        content = message["content"]
        if isinstance(content, str):
            new_message["content"] = content
        else:
            blocks = []
            for block in content:
                if block.get("type") == "image" and not isinstance(block["image"], str):
                    image_dir.mkdir(parents=True, exist_ok=True)
                    rel = Path("images") / f"{prefix}_{counter}.png"
                    block["image"].save(dataset_dir / rel)
                    counter += 1
                    blocks.append({"type": "image", "image": str(rel)})
                else:
                    blocks.append(dict(block))
            new_message["content"] = blocks
        out.append(new_message)
    return out


def cmd_collect(cfg: Config) -> None:
    """Sample episodes with the current model and keep the clean successes."""
    # Heavy imports live here so `should_skip` stays fast and GPU-free.
    from cua import policy, rollout
    from cua.env import EnvPool

    track = cfg.env.track
    tasks = resolve_task_list(cfg.rft.tasks, cfg)
    # Seeds come from the dedicated "rft" range, which never overlaps eval or GRPO seeds.
    seeds = load_tasks().seed_range("rft").seeds(int(cfg.rft.episodes_per_task))
    # make_group_specs takes parallel (task, seed) lists, so pair every task
    # with every seed: len(tasks) x len(seeds) episodes.
    pairs = [(task, seed) for task in tasks for seed in seeds]
    specs = rollout.make_group_specs([t for t, _ in pairs], [s for _, s in pairs], group_size=1)
    print(f"[rft] sampling {len(specs)} episodes ({len(tasks)} tasks x {len(seeds)} seeds)")

    adapter = cfg.model.get("adapter")
    vllm_policy = policy.VLLMPolicy(cfg, enable_lora=bool(adapter))
    if adapter:
        vllm_policy.set_adapter(adapter)
    # Temperature > 0 is the point: we need varied samples so that some succeed.
    sampling = policy.sampling_from_cfg(cfg, greedy=False)

    with EnvPool(cfg) as pool:
        episodes = rollout.run_episodes(specs, vllm_policy.generate, pool, cfg, sampling, track=track)

    out_dir = _rft_dir(cfg)
    dataset_dir = out_dir / "dataset"
    image_dir = dataset_dir / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    require_clean = bool(cfg.rft.require_no_format_errors)
    cap = int(cfg.rft.max_kept_per_task)
    attempted: dict[str, int] = defaultdict(int)
    solved: dict[str, int] = defaultdict(int)
    kept: dict[str, int] = defaultdict(int)

    with open(out_dir / "episodes.jsonl", "w") as ep_file, open(dataset_dir / "conversations.jsonl", "w") as ds_file:
        for ep in episodes:
            ep_file.write(json.dumps(ep.summary(), default=str) + "\n")
            attempted[ep.task] += 1
            if ep.crashed:
                continue
            if ep.success:
                solved[ep.task] += 1
            clean = ep.success and (not require_clean or ep.num_format_errors == 0)
            # The cap stops easy tasks from dominating the dataset.
            if not clean or kept[ep.task] >= cap:
                continue
            # One row per conversation: an episode with a history window (run 4)
            # has several segments, and each is a complete training sample.
            for ci, conv in enumerate(ep.conversations()):
                messages = _serialize_messages(
                    conv, image_dir, dataset_dir, f"{ep.task}_seed{ep.seed}_s{ep.sample_idx}_c{ci}"
                )
                ds_file.write(
                    json.dumps({"task": ep.task, "seed": ep.seed, "track": ep.track, "messages": messages}) + "\n"
                )
            kept[ep.task] += 1

    stats = {
        "run_name": cfg.run_name,
        "attempted": dict(attempted),
        "success_rate": {t: solved[t] / attempted[t] for t in attempted},
        "kept": {t: kept[t] for t in attempted},
        "total_kept": sum(kept.values()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[rft] kept {stats['total_kept']} episodes. Stats: {out_dir / 'stats.json'}")
    if stats["total_kept"] == 0:
        print("[rft] warning: nothing kept. Check the baseline success rates before training.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _load_conversations(dataset_dir: Path) -> list[dict]:
    rows = []
    with open(dataset_dir / "conversations.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _reopen_images(messages: list[dict], dataset_dir: Path) -> list[dict]:
    """Replace image path strings with PIL images, the form `tokenize_episode` expects."""
    from PIL import Image

    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "image" and isinstance(block["image"], str):
                    with Image.open(dataset_dir / block["image"]) as img:
                        blocks.append({"type": "image", "image": img.convert("RGB")})
                else:
                    blocks.append(block)
            content = blocks
        out.append({**message, "content": content})
    return out


def _batched(tensor, ndim_without_batch: int):
    """Add a batch dimension if the tensor lacks one."""
    return tensor.unsqueeze(0) if tensor.dim() == ndim_without_batch else tensor


def cmd_train(cfg: Config, dataset_arg: str | None = None) -> None:
    """Fine-tune a LoRA adapter on the collected episodes, assistant tokens only."""
    import torch
    import torch.nn.functional as F
    from transformers import get_scheduler

    from cua import actions, policy

    sft = cfg.sft
    dataset_dir = Path(dataset_arg) if dataset_arg else _rft_dir(cfg) / "dataset"
    rows = _load_conversations(dataset_dir)
    if not rows:
        raise SystemExit(f"{dataset_dir} holds no episodes. Run `collect` first.")
    track = cfg.env.track
    tools = actions.tool_schemas(track, track == "B" and actions.extended_tools(cfg))

    # Run 3 continues the run-2 adapter (model.adapter); runs 1 and 2 left it
    # null, which starts a fresh LoRA on the base model.
    start_adapter = cfg.model.get("adapter")
    model, processor = policy.load_hf_model(cfg, adapter=start_adapter, trainable=True)
    # `trainable=True` should return a model with LoRA attached. If it returns
    # a bare base model, attach LoRA here so the optimizer has something to
    # train and the base weights stay frozen.
    if not any(p.requires_grad for p in model.parameters()):
        from peft import get_peft_model

        model = get_peft_model(model, policy.make_lora_config(cfg))
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"[rft] {len(rows)} episodes, {n_trainable / 1e6:.1f}M trainable parameters")

    model.config.use_cache = False  # the KV cache is for generation, and it conflicts with checkpointing
    if sft.get("gradient_checkpointing", True):
        # use_reentrant=False matches load_hf_model; it works when only the
        # LoRA weights require gradients.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # With frozen embeddings, checkpointed blocks need inputs that require
        # grad, or no gradient flows back to the LoRA layers.
        model.enable_input_require_grads()
    model.train()

    # Every episode is one forward/backward pass, because images make sequence
    # lengths differ and padding would waste memory. Gradient accumulation then
    # builds the effective batch: per_device_train_batch_size * gradient_accumulation_steps episodes.
    batch_episodes = int(sft.per_device_train_batch_size) * int(sft.gradient_accumulation_steps)
    epochs = int(sft.num_train_epochs)
    steps_per_epoch = math.ceil(len(rows) / batch_episodes)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(params, lr=float(sft.learning_rate), weight_decay=0.0)
    scheduler = get_scheduler(
        sft.lr_scheduler_type,
        optimizer,
        num_warmup_steps=math.ceil(float(sft.warmup_ratio) * total_steps),
        num_training_steps=total_steps,
    )

    metrics_path = results_dir(cfg) / "train" / cfg.run_name / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(cfg)

    device = next(model.parameters()).device
    rng = random.Random(0)
    step = 0
    skipped = 0
    t0 = time.time()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_episodes):
            chunk = order[start : start + batch_episodes]
            running_loss, used = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for idx in chunk:
                messages = _reopen_images(rows[idx]["messages"], dataset_dir)
                enc = policy.tokenize_episode(processor, messages, tools)
                input_ids = _batched(torch.as_tensor(enc["input_ids"]), 1).to(device)
                # We skip over-long episodes instead of truncating, because
                # truncation can cut image tokens and break the vision inputs.
                if input_ids.shape[1] > int(sft.max_length):
                    skipped += 1
                    continue
                attention_mask = _batched(torch.as_tensor(enc["attention_mask"]), 1).to(device)
                assistant_mask = _batched(torch.as_tensor(enc["assistant_mask"]), 1).to(device).bool()
                kwargs = {}
                if enc.get("pixel_values") is not None:
                    kwargs["pixel_values"] = torch.as_tensor(enc["pixel_values"]).to(device)
                    kwargs["image_grid_thw"] = torch.as_tensor(enc["image_grid_thw"]).to(device)
                if enc.get("mm_token_type_ids") is not None:
                    kwargs["mm_token_type_ids"] = _batched(torch.as_tensor(enc["mm_token_type_ids"]), 1).to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
                # Next-token prediction: the logits at position t predict the
                # token at t+1, so we drop the last logit and the first label.
                # Keeping only the assistant positions BEFORE the loss means we
                # never compute cross-entropy on prompt, tool, or image tokens.
                shift_mask = assistant_mask[0, 1:]
                selected = logits[0, :-1][shift_mask].float()
                labels = input_ids[0, 1:][shift_mask]
                loss = F.cross_entropy(selected, labels)  # mean over this episode's assistant tokens
                # Dividing by the number of episodes makes the accumulated
                # gradient the mean over the batch. (Each episode counts equally,
                # not each token.)
                (loss / len(chunk)).backward()
                running_loss += loss.item()
                used += 1
                del logits, selected, loss

            if used == 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % int(sft.logging_steps) == 0 or step == total_steps:
                record = {
                    "step": step,
                    "epoch": epoch + start / len(order),
                    "loss": running_loss / used,
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "skipped_long": skipped,
                    "elapsed_s": time.time() - t0,
                }
                print(f"[rft] {record}", flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if wandb_run is not None:
                    wandb_run.log(record, step=step)

    final_dir = Path(sft.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[rft] saved adapter to {final_dir}")
    if wandb_run is not None:
        wandb_run.finish()


def _init_wandb(cfg: Config):
    """Start a W&B run, or return None if W&B is disabled or unavailable."""
    mode = str(cfg.wandb.get("mode", "online"))
    if mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            mode=mode,
            name=cfg.run_name,
            config=cfg.to_dict(),
        )
    except Exception as exc:  # noqa: BLE001 - logging must never stop training
        print(f"[rft] W&B disabled ({type(exc).__name__}): {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rejection-sampling fine-tuning (warm start).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("should_skip", "Exit 0 if the baseline is good enough to skip RFT, 1 otherwise."),
        ("collect", "Sample episodes and keep clean successes."),
        ("train", "Fine-tune a LoRA adapter on the kept episodes."),
    ]:
        p = sub.add_parser(name, help=help_text)
        add_config_args(p, DEFAULT_CONFIG)
        if name == "train":
            p.add_argument("--dataset", default=None, help="Dataset directory. Default: results/rft/<run_name>/dataset.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    if args.command == "should_skip":
        return cmd_should_skip(cfg)
    if args.command == "collect":
        cmd_collect(cfg)
    else:
        cmd_train(cfg, args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
