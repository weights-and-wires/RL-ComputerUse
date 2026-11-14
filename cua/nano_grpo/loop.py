"""The nano GRPO training loop: one file, one iteration per `for step` pass.

Run it with:

    python -m cua.nano_grpo.loop --config configs/grpo_qwen3vl4b.yaml [--set KEY=VALUE] [--resume]

One GRPO iteration (the numbers match the `# --- N.` section markers in
`main`):

1. Pick prompts. Choose `prompts_per_step` (task, seed) pairs, round-robin over
   the selected tasks so every task gets equal attention.
2. Roll out. For each prompt, sample G episodes with vLLM, stepping up to
   `env.pool_size` browsers in parallel. vLLM records the log-prob of each
   token it samples.
3. Score and group. Rewards come from `cua.rewards` and already sit on each
   episode. Drop crashed episodes, then turn each group's rewards into
   advantages with `loss.group_advantages`.
4. Filter. Skip groups where every reward is equal: their advantages are all 0,
   so they would add 0 to the gradient and only cost compute.
5. Train. Re-run the full conversation of each remaining episode through the
   PyTorch model, compute log-probs of the assistant tokens, and take
   `updates_per_step` optimizer steps on the clipped policy-gradient loss.
6. Sync. Save the LoRA adapter and hand it to vLLM, so the next rollout
   samples from the updated policy.
7. Log. Write metrics to `metrics.jsonl` and W&B.
8. Checkpoint and evaluate on schedule.

Why this loop exists: TRL steps its browsers one at a time. Here, step 2 keeps
dozens of browsers busy at once, which is the main source of speed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers.optimization import get_constant_schedule_with_warmup

from cua import actions, rewards, rollout
from cua.config import (
    Config,
    add_config_args,
    config_from_args,
    load_config,
    load_tasks,
    resolve_task_list,
    results_dir,
)
from cua.env import EnvPool
from cua.nano_grpo import loss as L
from cua.policy import VLLMPolicy, load_hf_model, tokenize_episode
from cua.types import Episode, SamplingSpec


def log(msg: str) -> None:
    print(f"[nano_grpo] {msg}", flush=True)


def chunks(items: list, n: int) -> list[list]:
    return [items[i : i + n] for i in range(0, len(items), n)]


# ---------------------------------------------------------------------------
# 1. Choosing prompts
# ---------------------------------------------------------------------------


def next_prompts(pos: int, n: int, tasks: list[str], seed_start: int) -> list[tuple[str, int]]:
    """Returns n (task, seed) pairs starting at sampler position `pos`.

    The k-th prompt is task `tasks[k % len(tasks)]` with seed index
    `k // len(tasks)`. Tasks therefore rotate evenly and a (task, seed) pair is
    never reused. `pos` is the only sampler state, so a checkpoint resumes
    exactly where it stopped.
    """
    return [(tasks[k % len(tasks)], seed_start + k // len(tasks)) for k in range(pos, pos + n)]


# ---------------------------------------------------------------------------
# 3 and 4. Rewards, advantages, and filtering
# ---------------------------------------------------------------------------


def build_training_items(
    episodes: list[Episode], scale_rewards: bool
) -> tuple[list[tuple[Episode, float]], list[list[float]], dict[str, float]]:
    """Groups episodes, computes advantages, and drops groups with no signal.

    Returns the (episode, advantage) pairs to train on, the rewards of every
    surviving group (for `rewards.group_stats`), and counters for logging.
    """
    groups: dict[tuple[str, int], list[Episode]] = {}
    num_crashed = 0
    for ep in episodes:
        if ep.crashed:  # a browser failure says nothing about the policy
            num_crashed += 1
            continue
        groups.setdefault((ep.task, ep.seed), []).append(ep)

    items: list[tuple[Episode, float]] = []
    rewards_by_group: list[list[float]] = []
    zero_variance = 0
    for group in groups.values():
        if len(group) < 2:  # a lone episode has no baseline to compare against
            continue
        r = torch.tensor([ep.reward.total for ep in group])
        rewards_by_group.append(r.tolist())
        if r.max() - r.min() < 1e-8:
            # Every episode scored the same, so every advantage is 0 and the
            # loss gradient is exactly 0. Skipping saves a forward/backward pass.
            zero_variance += 1
            continue
        adv = L.group_advantages(r.unsqueeze(0), scale=scale_rewards)[0]
        items.extend(zip(group, adv.tolist()))
    counters = {
        "episodes/crashed": float(num_crashed),
        "groups/used": float(len(rewards_by_group) - zero_variance),
        "episodes/trained": float(len(items)),
    }
    return items, rewards_by_group, counters


# ---------------------------------------------------------------------------
# 5. The training phase
# ---------------------------------------------------------------------------


def collate(token_dicts: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Pads one or more `tokenize_episode` outputs into a single batch.

    Text fields are right-padded to the longest sequence. Image fields have no
    batch dimension (all patches of all images are concatenated), so they are
    concatenated in sample order, which matches the order of image pads.
    """
    def seq(d: dict, key: str) -> torch.Tensor:
        t = torch.as_tensor(d[key])
        return t[0] if t.dim() == 2 else t  # accept both [T] and [1, T]

    out: dict[str, torch.Tensor] = {}
    max_len = max(seq(d, "input_ids").shape[0] for d in token_dicts)
    for key in ("input_ids", "attention_mask", "assistant_mask"):
        padded = [torch.nn.functional.pad(seq(d, key), (0, max_len - seq(d, key).shape[0])) for d in token_dicts]
        out[key] = torch.stack(padded).to(device)
    # Per-token modality IDs (0 text, 1 image) that Qwen3-VL needs for its
    # multimodal RoPE. Padding uses 0, which marks the pad positions as text.
    if all("mm_token_type_ids" in d for d in token_dicts):
        padded = [torch.nn.functional.pad(seq(d, "mm_token_type_ids"), (0, max_len - seq(d, "mm_token_type_ids").shape[0]))
                  for d in token_dicts]
        out["mm_token_type_ids"] = torch.stack(padded).to(device)
    # An episode can have no image at all (first_obs_image=false and the model
    # stopped before its first tool result). tokenize_episode then omits the
    # image keys, so only concatenate what exists.
    with_images = [d for d in token_dicts if "pixel_values" in d]
    if with_images:
        out["pixel_values"] = torch.cat([torch.as_tensor(d["pixel_values"]) for d in with_images]).to(device)
        out["image_grid_thw"] = torch.cat([torch.as_tensor(d["image_grid_thw"]) for d in with_images]).to(device)
    return out


def disable_dropout(model: torch.nn.Module) -> None:
    """Turns off every dropout layer, including LoRA dropout.

    GRPO compares log-probs from two forward passes of the same weights (old and
    new). Dropout would make the two passes differ by random noise, so the ratio
    would not be 1 even before any update. It also keeps the loss deterministic
    for a given batch.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def train_phase(
    model, processor, optimizer, scheduler, items: list[tuple[Episode, float]],
    cfg: Config, tools: list[dict], micro_bs: int,
) -> dict[str, float]:
    """Runs `updates_per_step` optimizer steps over the training items.

    Returns averaged loss metrics, the mean gradient norm, and the sampler
    mismatch diagnostic.
    """
    g = cfg.grpo
    device = next(p.device for p in model.parameters())
    updates = int(g.updates_per_step)
    # The most assistant tokens one episode can contain: every turn at its cap.
    max_tokens_norm = cfg.env.max_turns * cfg.generation.max_new_tokens
    params = optimizer.param_groups[0]["params"]
    model.train()  # needed for gradient checkpointing; dropout is already off

    token_cache: dict[int, dict] = {}  # tokenizing renders chat text and resizes images: do it once

    def tok(ep: Episode) -> dict:
        if id(ep) not in token_cache:
            token_cache[id(ep)] = tokenize_episode(processor, ep.messages, tools)
        return token_cache[id(ep)]

    def forward(batch: list[tuple[Episode, float]]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """One forward pass over whole conversations; returns log-probs and the loss mask."""
        b = collate([tok(ep) for ep, _ in batch], device)
        out = model(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"],
            pixel_values=b.get("pixel_values"), image_grid_thw=b.get("image_grid_thw"),
            mm_token_type_ids=b.get("mm_token_type_ids"), use_cache=False,
        )
        # The mask is 1 only on assistant tokens: the model did not write the
        # screenshots or page text, so training on them would be meaningless.
        return L.token_logprobs(out.logits, b["input_ids"]), b["assistant_mask"].float(), b

    # --- Old log-probs. With one update per batch the policy that samples and
    # the policy that trains are the same weights, so "old" equals "new" and we
    # use new_logp.detach() inside the loss (ratio is exactly 1, gradient is
    # still correct). With more updates, the second minibatch is trained on
    # weights that already moved once, so we need a frozen snapshot first.
    #
    # Why HF log-probs and not vLLM's? vLLM and PyTorch use different kernels and
    # batching, so their log-probs differ by small numerical noise even for the
    # same weights. A ratio built from one vLLM number and one PyTorch number
    # would be noisy, and clipping would trigger on noise. Using the same code
    # for both sides cancels that noise. We still log the vLLM difference below.
    old: dict[int, torch.Tensor] = {}
    if updates > 1:
        with torch.no_grad():
            for mb in chunks(items, micro_bs):
                lp, _, b = forward(mb)
                for j, (ep, _) in enumerate(mb):
                    n = int(b["attention_mask"][j].sum())
                    old[id(ep)] = lp[j, :n].detach()

    items = list(items)
    random.shuffle(items)
    minibatches = [items[i::updates] for i in range(updates)]  # round-robin split

    all_metrics: list[dict[str, float]] = []
    grad_norms: list[float] = []
    loss_values: list[float] = []
    mismatches: list[float] = []
    num_length_skips = 0
    first = True

    for shard in minibatches:
        if not shard:
            continue
        optimizer.zero_grad(set_to_none=True)
        total_seqs = len(shard)
        total_tokens = float(sum(int(torch.as_tensor(tok(ep)["assistant_mask"]).sum()) for ep, _ in shard))
        step_loss = 0.0
        for mb in chunks(shard, micro_bs):
            new_lp, mask, b = forward(mb)
            adv = torch.tensor([a for _, a in mb], device=device, dtype=torch.float32)
            if updates > 1:
                old_lp = torch.zeros_like(new_lp)
                for j, (ep, _) in enumerate(mb):
                    o = old[id(ep)]
                    old_lp[j, : o.shape[0]] = o
            else:
                old_lp = new_lp.detach()
            loss, m = L.clipped_pg_loss(
                new_lp, old_lp, adv, mask, g.epsilon, g.epsilon_high, g.loss_type, max_tokens_norm
            )
            # Gradient accumulation: scale so the sum over micro-batches equals
            # the loss over the whole minibatch (see loss.microbatch_weight).
            weight = L.microbatch_weight(g.loss_type, mask.sum().item(), total_tokens, len(mb), total_seqs)
            (loss * weight).backward()
            step_loss += loss.item() * weight
            all_metrics.append(m)

            if first:  # sampler-mismatch diagnostic, first micro-batch only
                first = False
                for j, (ep, _) in enumerate(mb):
                    mine = new_lp[j][mask[j] > 0].detach().float().cpu()
                    theirs = torch.tensor([x for t in ep.turns for x in t.generation.logprobs])
                    # Both lists hold assistant tokens in conversation order. They
                    # only line up when re-tokenizing the text gave the sampled ids.
                    if mine.numel() == theirs.numel() and mine.numel() > 0:
                        mismatches.append((mine - theirs).abs().mean().item())
                    else:
                        num_length_skips += 1
            del new_lp, old_lp, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(params, g.max_grad_norm)
        optimizer.step()
        scheduler.step()
        grad_norms.append(float(grad_norm))
        loss_values.append(step_loss)

    optimizer.zero_grad(set_to_none=True)
    token_cache.clear()
    torch.cuda.empty_cache()  # return fragmented blocks so vLLM's next rollout is unaffected

    def avg(key: str) -> float:
        return sum(m[key] for m in all_metrics) / max(len(all_metrics), 1)

    return {
        "loss": sum(loss_values) / max(len(loss_values), 1),
        "grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "train/clip_frac": avg("clip_frac"),
        "train/approx_kl": avg("approx_kl"),
        "train/ratio_mean": avg("ratio_mean"),
        "train/ratio_max": max((m["ratio_max"] for m in all_metrics), default=1.0),
        "train/entropy_proxy": avg("entropy_proxy"),
        "sampler/mean_abs_logp_diff": sum(mismatches) / len(mismatches) if mismatches else float("nan"),
        "sampler/length_mismatch_skips": float(num_length_skips),
    }


def train_with_oom_retry(*args: Any, micro_bs: int) -> tuple[dict[str, float], int]:
    """Runs `train_phase`; on CUDA OOM halves the micro-batch and retries once."""
    try:
        return train_phase(*args, micro_bs=micro_bs), micro_bs
    except torch.cuda.OutOfMemoryError:
        if micro_bs <= 1:
            raise
        log(f"CUDA OOM in the train phase; retrying with micro_batch_episodes={micro_bs // 2}")
        args[2].zero_grad(set_to_none=True)  # args[2] is the optimizer
        torch.cuda.empty_cache()
        micro_bs = max(1, micro_bs // 2)
        return train_phase(*args, micro_bs=micro_bs), micro_bs


# ---------------------------------------------------------------------------
# 6. Syncing weights to vLLM
# ---------------------------------------------------------------------------


def sync_to_vllm(model, policy: VLLMPolicy, out_dir: Path, step: int, keep: int = 2) -> None:
    """Saves the LoRA adapter and loads it into vLLM.

    LoRA makes this cheap. The base weights never change, so vLLM keeps them
    (and its KV cache) in memory. Only the adapter moves: the low-rank A and B
    matrices, roughly 100-250 MB at rank 32 instead of the 8 GB of the full
    model. Saving and loading take about a second. With full fine-tuning you
    would have to copy every weight into the engine on each step.
    """
    path = out_dir / "_sync" / f"step-{step}"
    shutil.rmtree(path, ignore_errors=True)
    model.save_pretrained(str(path))
    policy.set_adapter(str(path))  # a fresh LoRARequest id each call, so vLLM reloads
    old = sorted((out_dir / "_sync").glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    for stale in old[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. Checkpoints
# ---------------------------------------------------------------------------


def latest_checkpoint(out_dir: Path) -> Path | None:
    found = [p for p in out_dir.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[1].isdigit()]
    return max(found, key=lambda p: int(p.name.split("-")[1]), default=None)


def save_checkpoint(model, processor, optimizer, scheduler, out_dir: Path, state: dict, keep: int) -> None:
    """Writes everything needed to resume: adapter, optimizer, RNG, and sampler position."""
    final = out_dir / f"checkpoint-{state['step']}"
    tmp = final.with_name(final.name + ".tmp")  # write aside, then rename, so a crash never leaves a half checkpoint
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp))
    processor.save_pretrained(str(tmp))
    (tmp / "trainer_state.json").write_text(json.dumps(state, indent=2))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, tmp / "optimizer.pt")
    torch.save(
        {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        tmp / "rng_state.pt",
    )
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    ckpts = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.name.split("-")[1].isdigit()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    for stale in ckpts[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)
    log(f"saved {final}")


def load_training_state(ckpt: Path, optimizer, scheduler) -> dict:
    blob = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    rng = torch.load(ckpt / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((ckpt / "trainer_state.json").read_text())


# ---------------------------------------------------------------------------
# 7. Mid-training evaluation
# ---------------------------------------------------------------------------


def run_midtrain_eval(cfg: Config, tasks: list[str], step: int, policy: VLLMPolicy, pool: EnvPool) -> float | None:
    """Runs the eval.yaml `midtrain` preset in-process. Never raises."""
    try:
        from cua.eval import evaluate  # imported lazily so a broken eval cannot stop training

        ecfg = load_config(
            "configs/eval.yaml",
            # Copy the settings that change what the model sees, so the eval
            # prompt matches training (Track B uses more turns, for example).
            [
                "eval.mode=midtrain",
                # report.py looks for exactly this name pattern.
                f"run_name=grpo-step{step}-train",
                f"env.track={cfg.env.track}",
                f"env.max_turns={cfg.env.max_turns}",
                f"env.first_obs_image={str(bool(cfg.env.first_obs_image)).lower()}",
                f"env.screenshot_long_side={cfg.env.screenshot_long_side}",
                f"model.name={cfg.model.name}",
            ],
        )
        for key, value in ecfg.presets["midtrain"].items():  # apply the preset ourselves; it is idempotent
            ecfg.eval[key] = value
        ecfg.eval["tasks"] = list(tasks)
        summary = evaluate(ecfg, policy=policy, pool=pool, log_wandb=False)
        for key in ("success", "overall_success", "success_rate"):
            value = summary.get(key)
            if isinstance(value, dict):
                value = value.get("overall")
            if isinstance(value, (int, float)):
                return float(value)
        log(f"eval finished but no success key found in summary keys {list(summary)}")
    except Exception as exc:  # noqa: BLE001 - eval must never kill a training run
        log(f"midtrain eval failed ({type(exc).__name__}: {exc}); continuing")
    return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def init_wandb(cfg: Config, run_id: str | None):
    try:
        import wandb

        return wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
            name=cfg.run_name, config=cfg.to_dict(), id=run_id, resume="allow",
        )
    except Exception as exc:  # noqa: BLE001 - metrics.jsonl still works without W&B
        log(f"W&B disabled ({type(exc).__name__}: {exc})")
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with the hand-written GRPO loop.")
    add_config_args(parser, "configs/grpo_qwen3vl4b.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint in grpo.output_dir.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    g = cfg.grpo

    tasks = resolve_task_list(cfg.tasks, cfg)
    seed_start = load_tasks().seed_range("train").start
    out_dir = Path(g.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track = cfg.env.track
    tools = actions.tool_schemas(track)
    random.seed(g.seed)
    torch.manual_seed(g.seed)
    if cfg.generation.temperature != 1.0:
        log("warning: generation.temperature != 1.0; the loss uses temperature-1 log-probs, so training is slightly off-policy")

    resume_from = latest_checkpoint(out_dir) if args.resume else None
    adapter = str(resume_from) if resume_from else cfg.model.adapter

    # --- Models. vLLM goes first: it measures free GPU memory at start-up and
    # reserves its share (vllm_gpu_memory_utilization). If PyTorch allocated
    # first, its cached blocks would shrink what vLLM sees.
    policy = VLLMPolicy(cfg, gpu_memory_utilization=g.vllm_gpu_memory_utilization)
    if adapter:
        policy.set_adapter(adapter)
    model, processor = load_hf_model(cfg, adapter=adapter, trainable=True)
    disable_dropout(model)
    if g.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=g.learning_rate, weight_decay=0.0
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, g.warmup_steps)

    state = {"step": 0, "sampler_pos": 0, "micro_batch_episodes": int(g.micro_batch_episodes), "wandb_run_id": None}
    if resume_from:
        state = load_training_state(resume_from, optimizer, scheduler)
        log(f"resumed from {resume_from} at step {state['step']}")
    micro_bs = int(state["micro_batch_episodes"])

    run = init_wandb(cfg, state.get("wandb_run_id"))
    state["wandb_run_id"] = run.id if run else None
    metrics_path = results_dir(cfg) / "train" / cfg.run_name / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    sampling = SamplingSpec(
        temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
        max_new_tokens=cfg.generation.max_new_tokens, return_logprobs=True,
    )
    step = int(state["step"])
    checkpoint_args = (model, processor, optimizer, scheduler, out_dir)

    def checkpoint() -> None:
        state.update(step=step, micro_batch_episodes=micro_bs)
        save_checkpoint(*checkpoint_args, state, int(g.save_total_limit))

    with EnvPool(cfg) as pool:
        step_done = True
        try:
            while step < g.max_steps:
                step += 1
                step_done = False  # lets the Ctrl-C handler undo a half-finished step
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                # --- 1. Pick prompts and build G episode specs for each.
                prompts = next_prompts(state["sampler_pos"], g.prompts_per_step, tasks, seed_start)
                state["sampler_pos"] += g.prompts_per_step
                specs = []
                for task, seed in prompts:
                    specs += rollout.make_group_specs([task], [seed], g.num_generations)

                # --- 2. Roll out with vLLM (all browsers in parallel).
                episodes = rollout.run_episodes(
                    specs, policy.generate, pool, cfg, sampling, track=track, progress=False
                )
                t_rollout = time.perf_counter() - t0

                # --- 3 and 4. Advantages, and skipping groups with no signal.
                items, rewards_by_group, counters = build_training_items(
                    episodes, scale_rewards=bool(g.get("scale_rewards", True))
                )
                alive = [ep for ep in episodes if not ep.crashed]

                # --- 5. Train.
                t1 = time.perf_counter()
                train_metrics: dict[str, float] = {}
                if items:
                    train_metrics, micro_bs = train_with_oom_retry(
                        model, processor, optimizer, scheduler, items, cfg, tools, micro_bs=micro_bs
                    )
                else:
                    log(f"step {step}: no group had reward variance; skipping the update")
                t_train = time.perf_counter() - t1

                # --- 6. Sync the new adapter to vLLM (skipped if no update happened).
                t2 = time.perf_counter()
                if items:
                    sync_to_vllm(model, policy, out_dir, step)
                t_sync = time.perf_counter() - t2

                # --- 7. Log.
                free, total = torch.cuda.mem_get_info()
                record: dict[str, Any] = {
                    "step": step,
                    **(rewards.episode_stats(alive) if alive else {}),
                    **(rewards.group_stats(rewards_by_group) if rewards_by_group else {}),
                    **counters,
                    **train_metrics,
                    "lr": scheduler.get_last_lr()[0],
                    "micro_batch_episodes": micro_bs,
                    "time/rollout_s": t_rollout,
                    "time/train_s": t_train,
                    "time/sync_s": t_sync,
                    "time/episodes_per_s": len(episodes) / max(t_rollout, 1e-6),
                    # This process only: vLLM's engine runs in a child process.
                    "gpu/peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "gpu/device_used_gb": (total - free) / 2**30,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                if run:
                    import wandb

                    run.log(record, step=step)
                    if step % 10 == 0 and alive:
                        table = wandb.Table(columns=["step", "task", "seed", "reward", "success", "responses"])
                        for ep in alive[:6]:
                            table.add_data(step, ep.task, ep.seed, ep.reward.total, ep.success,
                                           "\n---\n".join(t.generation.text for t in ep.turns)[:1500])
                        run.log({"samples": table}, step=step)
                log(
                    f"step {step}: reward={record.get('reward/total', float('nan')):.3f} "
                    f"loss={record.get('loss', float('nan')):.4f} rollout={t_rollout:.0f}s train={t_train:.0f}s"
                )

                # --- 8. Checkpoint and evaluate on schedule.
                if step % g.save_steps == 0:
                    checkpoint()
                if g.eval_every and step % g.eval_every == 0:
                    success = run_midtrain_eval(cfg, tasks, step, policy, pool)
                    if success is not None:
                        extra = {"step": step, "eval/success": success}
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps(extra) + "\n")
                        if run:
                            run.log({"eval/success": success}, step=step)
                step_done = True
        except KeyboardInterrupt:
            log("Ctrl-C: saving a checkpoint before exit")
            if not step_done:
                # The interrupted step changed no saved state; replay it on resume.
                step -= 1
                state["sampler_pos"] -= g.prompts_per_step
            checkpoint()
            sys.exit(130)

    final = out_dir / "final"
    model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    checkpoint()
    log(f"done; final adapter at {final}")
    if run:
        run.finish()


if __name__ == "__main__":  # the guard matters: vLLM may spawn worker processes that re-import this file
    main()
