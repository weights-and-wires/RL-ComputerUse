"""GRPO training with TRL's `GRPOTrainer` and `environment_factory` (the TRL path).

Run it with:

    python -m cua.train_grpo --config configs/grpo_qwen3vl4b.yaml [--set KEY=VALUE] [--resume]

How the pieces fit together
---------------------------
1. TRL builds one environment object per rollout by calling a zero-argument
   factory. We pass an environment *class* (`TrackAEnv` or `TrackBEnv`) as the
   factory.
2. TRL turns every public method of that class, except `reset` and
   `get_reward`, into a tool. It renders the tool schemas into the prompt, parses
   the model's `<tool_call>` JSON, and calls the matching method. So the class
   methods ARE the model's action space. Helpers therefore start with an
   underscore, and each track has its own class so the model sees only that
   track's tools.
3. `reset(task, seed)` runs once per rollout before generation. Its return value
   (the first screenshot and page text) is appended to the last user message.
4. After each episode, `episode_reward` reads the counters stored on the
   environment objects and calls `rewards.reward_from_counts`. The nano GRPO
   loop calls the same function, so both trainers optimize the same reward.

Known limits of this path (see docs/PLAN.md, "Risks and mitigations"):
TRL 1.18 calls `reset` and every tool call serially across rollouts, so it is
slower than `cua/nano_grpo`. Use it as the library reference implementation.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from cua import actions, prompts, rewards
from cua.config import (
    Config,
    add_config_args,
    config_from_args,
    load_tasks,
    resolve_task_list,
    results_dir,
)
from cua.env import EnvCrash, RemoteEnv
from cua.types import Track

# ---------------------------------------------------------------------------
# Config hand-off to the environment class
# ---------------------------------------------------------------------------

# TRL creates environments by calling `EnvClass()` with NO arguments, and it
# does so inside the trainer, where we can't pass our own objects. The class
# therefore reads its settings from this module-level variable. `main()` sets
# it BEFORE it constructs the trainer. Never read `_CFG` at import time.
_CFG: Optional[Config] = None

# TRL never closes environments (it has no close hook), so each environment
# registers its browser subprocess here and `atexit` closes them all. Without
# this, Chromium processes outlive the training run.
_LIVE_REMOTES: list[RemoteEnv] = []


def _close_all_remotes() -> None:
    for remote in _LIVE_REMOTES:
        try:
            remote.close()
        except Exception:  # noqa: BLE001 - best effort during interpreter shutdown
            pass


atexit.register(_close_all_remotes)


def _cfg() -> Config:
    if _CFG is None:
        raise RuntimeError("cua.train_grpo._CFG is not set. Call main() or set _CFG before building the trainer.")
    return _CFG


# ---------------------------------------------------------------------------
# Environment classes
# ---------------------------------------------------------------------------


class _BaseMiniWoBEnv:
    """Shared episode bookkeeping. Exposes no tools by itself.

    Every attribute and method here starts with an underscore, except `reset`.
    That matters: TRL turns each public method into a tool, and a stray public
    helper would show up in the model's prompt.
    """

    _track: Track = "A"

    def __init__(self) -> None:
        cfg = _cfg()
        self._cfg_local = cfg
        self._track = cfg.env.track
        # RemoteEnv starts its subprocess lazily on the first call, so the
        # probe instance TRL builds at init time costs nothing.
        self._remote = RemoteEnv(cfg)
        _LIVE_REMOTES.append(self._remote)
        self._clear()

    # --- state -------------------------------------------------------------

    def _clear(self) -> None:
        self._task = ""
        self._seed = -1
        self._success = False
        self._terminated = False  # the page reported the episode is over
        self._done = False  # terminated OR out of turns OR crashed
        self._turns = 0  # tool calls the model made (valid or not)
        self._format_errors = 0
        self._crashed = ""
        self._screen_size: tuple[int, int] = (0, 0)

    def _summary(self) -> dict[str, Any]:
        """Counters the reward function reads. Plain data, no env objects."""
        return {
            "task": self._task,
            "seed": self._seed,
            "success": self._success,
            "terminated": self._terminated,
            "done": self._done,
            "turns": self._turns,
            "format_errors": self._format_errors,
            "crashed": self._crashed,
        }

    # --- TRL entry point ---------------------------------------------------

    def reset(self, task: str, seed: int, **kwargs: Any) -> list[dict] | str:
        """Start an episode. TRL passes every dataset column as a keyword.

        This method MUST stay synchronous: TRL calls it in a plain loop and
        never awaits it. `**kwargs` absorbs the `prompt` column.
        """
        cfg = self._cfg_local
        self._clear()
        self._task, self._seed = task, int(seed)

        obs = None
        for attempt in range(2):
            try:
                obs = self._remote.reset(task, int(seed))
                break
            except EnvCrash as exc:
                # RemoteEnv restarts its subprocess after a crash, so one retry
                # usually works. An exception here would kill the whole training
                # run, so we never let it escape.
                self._crashed = f"reset failed (attempt {attempt + 1}): {exc}"
        if obs is None:
            self._done = True
            return [{"type": "text", "text": "The browser failed to start. Call noop()."}]
        self._crashed = ""
        self._screen_size = obs.screen_size
        # Whether the first screenshot goes into the opening message is a
        # config flag, so train and eval show the model identical prompts.
        # TRL appends these blocks after the dataset's user text
        # (prompts.USER_PREAMBLE), which reproduces prompts.initial_messages.
        # We always return a list, so TRL takes the same path with or without
        # the image.
        return prompts.first_observation_blocks(obs, self._track, bool(cfg.env.first_obs_image))

    # --- the one place actions execute --------------------------------------

    def _act(self, name: str, kwargs: dict[str, Any]) -> list[dict]:
        """Render a tool call into a BrowserGym action, run it, return the observation.

        Raising `ValueError` is the supported way to report a problem: TRL
        catches the exception and sends its message back to the model as the
        tool result.
        """
        cfg = self._cfg_local
        if self._done or self._turns >= cfg.env.max_turns:
            self._done = True
            raise ValueError("Episode is done.")

        # Each tool call uses one turn, valid or not, because each one cost a
        # model generation. This also matches `max_tool_calling_iterations`.
        self._turns += 1
        try:
            action = actions.render_action(name, kwargs, self._track, self._screen_size)
        except actions.ActionError as exc:
            self._format_errors += 1
            if self._turns >= cfg.env.max_turns:
                self._done = True
            raise ValueError(str(exc)) from exc

        try:
            result = self._remote.step(action)
        except EnvCrash as exc:
            self._crashed = f"step failed: {exc}"
            self._done = True
            raise ValueError("The browser crashed. The episode is over.") from exc

        obs = result.observation
        self._screen_size = obs.screen_size
        if result.success:
            self._success = True
        if result.terminated or result.success:
            self._terminated = True
            self._done = True
        if self._turns >= cfg.env.max_turns:
            self._done = True
        # `obs.last_action_error` is already part of the formatted observation,
        # so we don't pass a separate `error` string.
        return prompts.tool_result_content(obs, self._track)


class _CommonToolsMixin:
    """Tools that both tracks share. Mixed into each track's class.

    The stubs in `cua/actions.py` hold the docstrings. We copy them below so
    the schema TRL builds is identical to the one the other code paths use.

    The methods really return a list of content blocks, but they are annotated
    `-> str` like the stubs: transformers builds the tool schema from the type
    hints, and the schema is rendered into the prompt. A different return type
    would make the TRL prompt differ from the eval and nano_grpo prompts.
    """

    def scroll(self, direction: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("scroll", {"direction": direction})  # type: ignore[attr-defined]

    def noop(self) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("noop", {})  # type: ignore[attr-defined]


class TrackAEnv(_CommonToolsMixin, _BaseMiniWoBEnv):
    """Track A: element-ID actions. Public methods here are the model's tools."""

    def click(self, bid: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("click", {"bid": bid})

    def fill(self, bid: str, text: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("fill", {"bid": bid, "text": text})

    def select_option(self, bid: str, option: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("select_option", {"bid": bid, "option": option})

    def press(self, bid: str, key: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("press", {"bid": bid, "key": key})


class TrackBEnv(_CommonToolsMixin, _BaseMiniWoBEnv):
    """Track B: coordinate actions (0-1000 normalized), screenshot only."""

    def mouse_click(self, coordinate: list[int]) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("mouse_click", {"coordinate": coordinate})

    def keyboard_type(self, text: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("keyboard_type", {"text": text})

    def keyboard_press(self, key: str) -> str:  # schema-only annotation, see _CommonToolsMixin
        return self._act("keyboard_press", {"key": key})


def _copy_stub_docs(cls: type, names: list[str]) -> None:
    """Give each tool method the docstring of its stub in `cua/actions.py`.

    `transformers.utils.get_json_schema` builds the tool description from the
    docstring and the type hints. Copying keeps one source of truth, so the
    model sees the same tool text in TRL, eval, and RFT.
    """
    for name in names:
        method = getattr(cls, name)
        method.__doc__ = getattr(actions, name).__doc__


# `scroll` and `noop` live on the mixin, so copy their docs once through TrackAEnv.
_copy_stub_docs(TrackAEnv, ["click", "fill", "select_option", "press", "scroll", "noop"])
_copy_stub_docs(TrackBEnv, ["mouse_click", "keyboard_type", "keyboard_press", "scroll", "noop"])

ENV_CLASSES: dict[str, type] = {"A": TrackAEnv, "B": TrackBEnv}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(cfg: Config, num_prompts: int):
    """Build the prompt dataset: one row per (task, seed), conversational format.

    TRL repeats each row `num_generations` times, so rows are *prompts* (task
    seeds), not episodes. The order is fixed by a seeded shuffle, so a resumed
    run sees the same data stream.

    The prompt is [system, user]. The user text is deliberately generic: the
    real goal arrives through `reset()`, which TRL appends to the last user
    message.
    """
    from datasets import Dataset

    track: Track = cfg.env.track
    tasks = resolve_task_list(cfg.tasks, cfg)
    train_seeds = load_tasks().seed_range("train")

    # Cycle tasks round-robin and give each full pass over the tasks a fresh seed.
    # That keeps every task equally represented and never repeats a (task, seed).
    n_seed_rounds = -(-num_prompts // len(tasks))  # ceiling division
    seeds = train_seeds.seeds(n_seed_rounds)
    pairs = [(tasks[i % len(tasks)], seeds[i // len(tasks)]) for i in range(num_prompts)]
    random.Random(cfg.grpo.seed).shuffle(pairs)

    system_message = {"role": "system", "content": prompts.system_prompt(track)}
    # Must equal the first text block of prompts.initial_messages, so the TRL
    # prompt renders exactly like the eval and nano_grpo prompts.
    user_message = {"role": "user", "content": prompts.USER_PREAMBLE}
    rows = [
        {"prompt": [system_message, user_message], "task": task, "seed": int(seed)}
        for task, seed in pairs
    ]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def episode_reward(completions, environments, log_metric=None, log_extra=None, **kwargs) -> list[float]:
    """Score each finished episode and log reward components.

    TRL passes one environment object per completion, in batch order. Prompts
    repeated `num_generations` times sit next to each other, so consecutive
    chunks of that size form GRPO groups.
    """
    cfg = _cfg()
    max_turns = cfg.env.max_turns

    totals: list[float] = []
    components: list[dict[str, float]] = []
    summaries: list[dict[str, Any]] = []
    for env in environments:
        s = env._summary()  # noqa: SLF001 - same module, deliberate
        format_errors = s["format_errors"]
        # The model stopped calling tools before the page ended and before it
        # ran out of turns. TRL ends such a rollout silently, so the missing or
        # malformed `<tool_call>` leaves no trace in the env. Count it here.
        if not s["done"] and s["turns"] < max_turns and not s["crashed"]:
            format_errors += 1
        breakdown = rewards.reward_from_counts(
            success=s["success"],
            num_format_errors=format_errors,
            turns_used=s["turns"],
            cfg=cfg,
        )
        s = {**s, "format_errors": format_errors}
        summaries.append(s)
        components.append(breakdown.to_dict())
        # A crashed browser says nothing about the policy. We still return a
        # number (not None) so TRL's reward aggregation never sees an all-None
        # column, but score it 0 and log the count so you can spot it.
        totals.append(0.0 if s["crashed"] else breakdown.total)

    if log_metric is not None and summaries:
        n = len(summaries)
        for key in components[0]:
            log_metric(key, sum(c[key] for c in components) / n)
        log_metric("success/overall", sum(s["success"] for s in summaries) / n)
        log_metric("episode/format_error_rate", sum(s["format_errors"] > 0 for s in summaries) / n)
        log_metric("episode/mean_turns", sum(s["turns"] for s in summaries) / n)
        log_metric("episode/crash_rate", sum(bool(s["crashed"]) for s in summaries) / n)
        per_task: dict[str, list[bool]] = {}
        for s in summaries:
            per_task.setdefault(s["task"], []).append(bool(s["success"]))
        for task, flags in per_task.items():
            log_metric(f"success/{task}", sum(flags) / len(flags))
        group = int(cfg.grpo.num_generations)
        if len(totals) % group == 0:
            chunks = [totals[i : i + group] for i in range(0, len(totals), group)]
            for key, value in rewards.group_stats(chunks).items():
                log_metric(key, value)
    if log_extra is not None and summaries:
        log_extra("task", [s["task"] for s in summaries])
        log_extra("seed", [s["seed"] for s in summaries])
        log_extra("success", [bool(s["success"]) for s in summaries])
        log_extra("turns", [s["turns"] for s in summaries])
    return totals


# TRL uses the function name to label the reward in logs.
episode_reward.__name__ = "episode_reward"


# ---------------------------------------------------------------------------
# Callback: JSONL metrics and optional mid-training eval
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback
except ImportError:  # keeps `import cua.train_grpo` usable on machines without transformers
    TrainerCallback = object  # type: ignore[assignment,misc]


class MetricsAndEvalCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Appends every log line to metrics.jsonl and optionally runs a mid-training eval.

    The eval runs in a SUBPROCESS because vLLM wants its own process for GPU
    memory, and colocated training already holds most of the GPU. It is OFF by
    default (`grpo.midtrain_eval_subprocess: false`). `scripts/saturday.sh`
    evaluates the saved `checkpoint-*` directories after training instead.
    """

    def __init__(self, cfg: Config, metrics_path: Path) -> None:
        self.cfg = cfg
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
        if not logs or not state.is_world_process_zero:
            return
        record = {"step": state.global_step, "time": time.time(), **logs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: D102
        cfg = self.cfg
        if not cfg.grpo.get("midtrain_eval_subprocess", False):
            return
        every = int(cfg.grpo.eval_every)
        if every <= 0 or state.global_step == 0 or state.global_step % every != 0:
            return
        if not state.is_world_process_zero or model is None:
            return
        step = state.global_step
        adapter_dir = Path(cfg.grpo.output_dir) / "midtrain" / f"step-{step}"
        model.save_pretrained(str(adapter_dir))
        cmd = [
            sys.executable, "-m", "cua.eval",
            "--config", "configs/eval.yaml",
            "--set", "eval.mode=midtrain",
            "--set", f"model.name={cfg.model.name}",
            "--set", f"model.adapter={adapter_dir}",
            "--set", f"run_name=grpo-step{step}-train",
            "--set", "vllm.gpu_memory_utilization=0.2",
        ]
        print(f"[train_grpo] mid-training eval at step {step}: {' '.join(cmd)}", flush=True)
        # check=False: a failed eval must not stop training.
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[train_grpo] mid-training eval exited with code {proc.returncode}", flush=True)


# ---------------------------------------------------------------------------
# Helpers for main()
# ---------------------------------------------------------------------------


def _latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Return the newest `checkpoint-<step>` directory, or None."""
    best: tuple[int, Path] | None = None
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return str(best[1]) if best else None


def _setup_wandb_env(cfg: Config) -> str:
    """Set the W&B env vars the Hugging Face integration reads. Returns `report_to`."""
    mode = str(cfg.wandb.get("mode", "online"))
    if mode == "disabled":
        return "none"
    os.environ["WANDB_PROJECT"] = str(cfg.wandb.project)
    os.environ["WANDB_MODE"] = mode
    if cfg.wandb.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg.wandb.entity)
    return "wandb"


def _push_to_hub(final_dir: Path) -> None:
    """Upload the final adapter to a private HF repo, if credentials are set.

    Failures are logged and swallowed: training already finished, and the adapter
    is safe on disk. We never print the token.
    """
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_ADAPTER_REPO")
    if not token or not repo_id:
        print("[train_grpo] HF_TOKEN or HF_ADAPTER_REPO not set; skipping Hub upload.")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(final_dir), repo_id=repo_id)
        print(f"[train_grpo] uploaded adapter to private repo {repo_id}")
    except Exception as exc:  # noqa: BLE001
        message = str(exc).replace(token, "***")
        print(f"[train_grpo] Hub upload failed ({type(exc).__name__}): {message}")


def build_trainer(cfg: Config):
    """Construct the `GRPOTrainer`. `_CFG` must already be set."""
    from trl import GRPOConfig, GRPOTrainer

    from cua import policy

    g = cfg.grpo
    out_dir = Path(g.output_dir)

    # Check that the batch arithmetic matches the config's intent. TRL's
    # generation batch = per_device_train_batch_size * gradient_accumulation_steps
    # episodes on one GPU, and each prompt is repeated num_generations times.
    episodes_per_step = int(g.per_device_train_batch_size) * int(g.gradient_accumulation_steps)
    if episodes_per_step % int(g.num_generations) != 0:
        raise ValueError(
            f"per_device_train_batch_size * gradient_accumulation_steps ({episodes_per_step}) "
            f"must be divisible by num_generations ({g.num_generations})."
        )
    prompts_per_step = episodes_per_step // int(g.num_generations)
    if prompts_per_step != int(g.prompts_per_step):
        print(
            f"[train_grpo] warning: grpo.prompts_per_step={g.prompts_per_step} but the TRL batch "
            f"settings give {prompts_per_step} prompts per step. Using {prompts_per_step}."
        )
    dataset = build_dataset(cfg, prompts_per_step * int(g.max_steps))

    # Load the model ourselves instead of passing a name string. That way the
    # TRL path uses the same model class, dtype, and processor pixel limits
    # (cfg.model.min_pixels / max_pixels) as every other path.
    #
    # Trade-off for the RFT warm start: if `model.adapter` is set, we load the
    # RFT adapter as a trainable PeftModel and pass NO peft_config, so GRPO
    # continues training the SAME LoRA weights. The alternative is to merge the
    # RFT adapter into the base weights and train a fresh LoRA on top. That
    # gives GRPO a clean, zero-initialized adapter, but you then need the
    # merged weights (not just the adapter) to evaluate or serve the result.
    # Continuing the same adapter keeps a single small artifact, which is why
    # this script does it. The cost is that vLLM must also load the RFT
    # weights; check that the trainer's vLLM sync picks them up (see the
    # smoke-test notes in the final report).
    adapter = cfg.model.get("adapter")
    if adapter:
        model, processor = policy.load_hf_model(cfg, adapter=adapter, trainable=True)
        peft_config = None
    else:
        model, processor = policy.load_hf_model(cfg)
        peft_config = policy.make_lora_config(cfg)

    report_to = _setup_wandb_env(cfg)
    args = GRPOConfig(
        output_dir=str(out_dir),
        run_name=cfg.run_name,
        report_to=[report_to] if report_to != "none" else "none",
        seed=int(g.seed),
        # --- optimization ---
        learning_rate=float(g.learning_rate),
        lr_scheduler_type=g.lr_scheduler_type,
        warmup_steps=int(g.warmup_steps),
        max_steps=int(g.max_steps),
        max_grad_norm=float(g.max_grad_norm),
        bf16=bool(g.bf16),
        gradient_checkpointing=bool(g.gradient_checkpointing),
        # --- GRPO objective ---
        num_generations=int(g.num_generations),
        per_device_train_batch_size=int(g.per_device_train_batch_size),
        gradient_accumulation_steps=int(g.gradient_accumulation_steps),
        loss_type=g.loss_type,
        beta=float(g.beta),
        epsilon=float(g.epsilon),
        epsilon_high=float(g.epsilon_high),
        mask_truncated_completions=bool(g.mask_truncated_completions),
        # --- generation ---
        temperature=float(cfg.generation.temperature),
        top_p=float(cfg.generation.top_p),
        # max_completion_length budgets the WHOLE multi-turn episode: model text,
        # tool text, and image tokens. There is no max_prompt_length in TRL 1.18.
        max_completion_length=int(g.max_completion_length),
        max_tool_calling_iterations=int(cfg.env.max_turns),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(g.vllm_gpu_memory_utilization),
        vllm_max_model_length=int(g.vllm_max_model_length),
        # --- logging and checkpoints ---
        log_completions=bool(g.log_completions),
        logging_steps=int(g.logging_steps),
        save_steps=int(g.save_steps),
        save_total_limit=int(g.save_total_limit),
    )

    metrics_path = results_dir(cfg) / "train" / cfg.run_name / "metrics.jsonl"
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=episode_reward,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=[MetricsAndEvalCallback(cfg, metrics_path)],
        environment_factory=ENV_CLASSES[cfg.env.track],
    )
    return trainer


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="GRPO training with TRL environment_factory.")
    add_config_args(parser, "configs/grpo_qwen3vl4b.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint-* directory in grpo.output_dir.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    # Set the module-level config BEFORE the trainer exists: TRL creates a probe
    # environment during GRPOTrainer.__init__, and it needs _CFG.
    global _CFG
    _CFG = cfg

    out_dir = Path(cfg.grpo.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume:
        resume_path = _latest_checkpoint(out_dir)
        print(f"[train_grpo] resuming from {resume_path}" if resume_path else "[train_grpo] no checkpoint found; starting fresh.")

    trainer = build_trainer(cfg)
    trainer.train(resume_from_checkpoint=resume_path)

    # Save the final adapter. For a PEFT model, save_model writes only the
    # adapter files, plus the processor when it is the trainer's processing_class.
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    try:
        trainer.processing_class.save_pretrained(str(final_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"[train_grpo] could not save the processor: {exc}")
    print(f"[train_grpo] saved final adapter to {final_dir}")

    _push_to_hub(final_dir)
    end = getattr(trainer, "end", None)
    if callable(end):
        end()
    _close_all_remotes()


if __name__ == "__main__":
    main()
