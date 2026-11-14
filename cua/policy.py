"""Model access: prompt rendering, tokenization, vLLM sampling, and HF loading.

This module is the only place that turns a chat conversation into model input,
so every code path (eval, RFT sampling, nano GRPO, TRL) shows the model the
same tokens. The pieces:

    render_prompt      conversation -> chat-template text (images become pad tokens)
    collect_images     conversation -> list of PIL images, in order
    tokenize_episode   conversation -> tensors plus a mask of assistant tokens
    VLLMPolicy         fast batched sampling (vLLM), with optional LoRA adapter
    load_hf_model      Hugging Face model + processor for training or scoring
    make_lora_config   the one LoRA recipe (language model only)
    sampling_from_cfg  config -> SamplingSpec

torch, transformers, peft, and vllm are imported inside functions, so you can
import this module (and run unit tests that fake the policy) on a machine
without a GPU.
"""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from cua.config import Config
from cua.types import CONTEXT_OVERFLOW, Generation, SamplingSpec

__all__ = [
    "render_prompt",
    "collect_images",
    "tokenize_episode",
    "VLLMPolicy",
    "load_hf_model",
    "make_lora_config",
    "sampling_from_cfg",
]


# ---------------------------------------------------------------------------
# Rendering and tokenization (shared by every path)
# ---------------------------------------------------------------------------


def render_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
    """Render a conversation to text with the model's chat template.

    The template inserts the tool schemas into the system prompt and writes each
    image as `<|vision_start|><|image_pad|><|vision_end|>`. The pad token is
    expanded to the real number of image tokens later, by the processor (HF) or
    by vLLM's own multimodal processor.

    Set `add_generation_prompt=True` to end the text with `<|im_start|>assistant\\n`
    when you want the model to continue. Set it to False when the conversation
    already ends with an assistant message (training).
    """
    return processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def collect_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Return every image in a conversation, in the order the template emits them.

    The order matters: the processor matches the i-th `<|image_pad|>` in the text
    to the i-th image you pass. Images live in `{"type": "image", "image": PIL}`
    blocks inside a message's `content` list (user and tool messages).
    """
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue  # assistant and system messages carry plain strings
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                img = block["image"]
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                # The vision processor expects 3 channels. Screenshots from
                # Playwright are already RGB, but this makes it explicit.
                images.append(img.convert("RGB"))
    return images


def _find_assistant_spans(
    ids: list[int], im_start: int, im_end: int, header: list[int]
) -> list[tuple[int, int]]:
    """Find [start, end) token spans of assistant content, including `<|im_end|>`.

    A span starts right after the sequence `<|im_start|> assistant \\n` and ends
    just after the next `<|im_end|>`. The trailing newline the template adds
    after `<|im_end|>` is not part of the span, because the model never has to
    generate it: generation stops at `<|im_end|>`.
    """
    spans: list[tuple[int, int]] = []
    n, h = len(ids), len(header)
    i = 0
    while i < n:
        if ids[i] == im_start and ids[i + 1 : i + 1 + h] == header:
            start = i + 1 + h
            end = start
            while end < n and ids[end] != im_end:
                end += 1
            spans.append((start, min(end + 1, n)))  # +1 keeps <|im_end|> in the span
            i = end + 1
        else:
            i += 1
    return spans


def tokenize_episode(
    processor: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Tokenize a whole conversation and mark which tokens the model wrote.

    Approach (the same one TRL-style tool training uses, kept explicit here):

    1. Render the full conversation to text with the chat template.
    2. Tokenize text and images together with the HF processor, which expands
       each `<|image_pad|>` into the right number of image tokens.
    3. Scan the token IDs for the sequence `<|im_start|>assistant\\n`. The
       tokens after it, up to and including the next `<|im_end|>`, are one
       assistant span. Everything else (system, user, tool, images) stays 0.

    Why scan token IDs instead of tokenizing each message separately: tokenizing
    pieces separately can split BPE merges differently than the full text, and
    the image-token expansion needs the full text anyway. Scanning the final IDs
    guarantees the mask lines up with what the model actually sees.

    One caveat: this re-tokenizes the assistant text. Usually that matches the
    tokens vLLM sampled, but BPE is not always round-trip stable. nano GRPO
    therefore scores the re-tokenized text with the training model (it does not
    reuse the vLLM token IDs), so the loss is consistent with this mask.

    Returns a dict of tensors, all with a leading batch dimension of 1:
        input_ids        (1, L) int64
        attention_mask   (1, L) int64
        assistant_mask   (1, L) int64, 1 on assistant content + its <|im_end|>
        pixel_values     (num_patches, dim) float, only when the episode has images
        image_grid_thw   (num_images, 3) int64, only when the episode has images
        mm_token_type_ids (1, L) int64, when the processor returns it
    """
    import torch  # lazy: keeps the module importable without torch

    text = render_prompt(processor, messages, tools, add_generation_prompt=False)
    images = collect_images(messages)
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    )

    tok = processor.tokenizer
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    header = tok.encode("assistant\n", add_special_tokens=False)

    ids = inputs["input_ids"][0].tolist()
    spans = _find_assistant_spans(ids, im_start, im_end, header)
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if has_assistant and not spans:
        # A silent all-zero mask would train on nothing. Fail loudly instead.
        raise ValueError("No assistant span found; check the chat template and tokenizer.")

    mask = torch.zeros(len(ids), dtype=torch.long)
    for start, end in spans:
        mask[start:end] = 1

    out: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "assistant_mask": mask.unsqueeze(0),
    }
    if "pixel_values" in inputs:
        out["pixel_values"] = inputs["pixel_values"]
        out["image_grid_thw"] = inputs["image_grid_thw"]
    # transformers 5.x computes Qwen3-VL's multimodal RoPE positions from
    # mm_token_type_ids (1, L), and raises if images are present without it.
    if inputs.get("mm_token_type_ids") is not None:
        out["mm_token_type_ids"] = inputs["mm_token_type_ids"]
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def sampling_from_cfg(cfg: Config, greedy: bool = False) -> SamplingSpec:
    """Build a SamplingSpec from `cfg.generation`.

    Set `greedy=True` for evaluation: temperature 0 makes vLLM pick the argmax
    token at every step, so the same seed gives the same episode.
    """
    gen = cfg.generation
    return SamplingSpec(
        temperature=0.0 if greedy else float(gen.temperature),
        top_p=float(gen.get("top_p", 1.0)),
        max_new_tokens=int(gen.max_new_tokens),
        return_logprobs=False,
        seed=None,
    )


def make_lora_config(cfg: Config) -> Any:
    """Return the LoRA recipe: adapters on the language model's projections only.

    The target is a regex string. PEFT applies `re.fullmatch` to each module's
    full name, so `.*language_model.*` keeps the vision tower out even if it
    ever grows a layer named `q_proj`. The vision tower stays frozen, which
    also matches what vLLM can serve (it applies LoRA to the language model by
    default).
    """
    from peft import LoraConfig  # lazy

    lora = cfg.get("lora", {}) or {}
    names = lora.get("target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    pattern = r".*language_model.*\.(" + "|".join(names) + ")"
    return LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=pattern,
        task_type="CAUSAL_LM",
    )


def _configure_processor(processor: Any, cfg: Config) -> None:
    """Bound the number of image tokens per screenshot.

    Qwen3-VL's HF image processor reads `size["shortest_edge"]` as the minimum
    pixel count and `size["longest_edge"]` as the maximum. vLLM gets the same
    numbers through `mm_processor_kwargs`, so both sides produce the same
    number of image tokens for the same screenshot.
    """
    processor.image_processor.size = {
        "shortest_edge": int(cfg.model.min_pixels),
        "longest_edge": int(cfg.model.max_pixels),
    }
    # Some versions also keep the values as plain attributes. Setting them is
    # harmless when they do not exist.
    for attr, key in (("min_pixels", "min_pixels"), ("max_pixels", "max_pixels")):
        try:
            setattr(processor.image_processor, attr, int(cfg.model[key]))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# vLLM sampling
# ---------------------------------------------------------------------------


class VLLMPolicy:
    """Batched sampling with vLLM, optionally with a LoRA adapter.

    Construct it once and reuse it: loading the engine takes minutes. Call
    `set_adapter(path)` to switch to a new LoRA checkpoint without reloading.
    """

    def __init__(
        self,
        cfg: Config,
        gpu_memory_utilization: Optional[float] = None,
        enable_lora: bool = True,
    ) -> None:
        from transformers import AutoProcessor  # lazy
        from vllm import LLM  # lazy

        self.cfg = cfg
        self.enable_lora = enable_lora
        self.adapter_path: Optional[str] = None
        self._lora_request: Any = None
        self._lora_counter = 0  # source of fresh adapter IDs, see set_adapter

        # We render prompts to text ourselves (render_prompt), so we need the
        # processor here. vLLM tokenizes and handles the images itself.
        self.processor = AutoProcessor.from_pretrained(cfg.model.name)
        _configure_processor(self.processor, cfg)

        v = cfg.vllm
        kwargs: dict[str, Any] = dict(
            model=cfg.model.name,
            dtype=cfg.model.get("dtype", "bfloat16"),
            max_model_len=int(v.max_model_len),
            max_num_seqs=int(v.get("max_num_seqs", 64)),
            gpu_memory_utilization=float(
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else v.gpu_memory_utilization
            ),
            # Every turn adds one screenshot, plus the first one. Without this
            # limit vLLM rejects prompts that hold more than one image.
            limit_mm_per_prompt={"image": int(cfg.env.max_turns) + 1},
            mm_processor_kwargs={
                "max_pixels": int(cfg.model.max_pixels),
                "min_pixels": int(cfg.model.min_pixels),
            },
            # "processed_logprobs" reports log-probs after temperature scaling,
            # which is the distribution the sampler used. RL ratios need that.
            logprobs_mode=v.get("logprobs_mode", "processed_logprobs"),
            seed=int(v.get("seed", 0)),
        )
        if enable_lora:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = int(v.get("max_lora_rank", 64))
        if v.get("enable_sleep_mode", False):
            kwargs["enable_sleep_mode"] = True
        if v.get("enforce_eager", False):
            kwargs["enforce_eager"] = True

        self.llm = LLM(**kwargs)

        # Start from the adapter named in the config (for example the RFT adapter).
        if cfg.model.get("adapter"):
            self.set_adapter(str(cfg.model.adapter))

    # -- adapters -----------------------------------------------------------

    def set_adapter(self, path: Optional[str]) -> None:
        """Use the LoRA adapter at `path` for later generate() calls (None = base model).

        vLLM caches adapters by integer ID. If you reuse an ID after the files
        on disk changed, vLLM serves the stale weights. A fresh ID per call
        forces a reload, which is what you want after each training step.
        """
        if path is None:
            self._lora_request = None
            self.adapter_path = None
            return
        if not self.enable_lora:
            raise RuntimeError("This policy was built with enable_lora=False.")
        from vllm.lora.request import LoRARequest  # lazy

        self._lora_counter += 1
        self._lora_request = LoRARequest(
            lora_name=f"adapter-{self._lora_counter}",
            lora_int_id=self._lora_counter,  # must be >= 1 and unique
            lora_path=str(path),
        )
        self.adapter_path = str(path)

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        batch: list[list[dict[str, Any]]],
        sampling: SamplingSpec,
        tools: list[dict[str, Any]],
    ) -> list[Generation]:
        """Sample one response per conversation in `batch`.

        We call `llm.generate` with prompt text plus raw images instead of
        `llm.chat`. vLLM's chat parser does not reliably handle images inside
        `tool` messages, while our own template rendering handles them the same
        way as training does.
        """
        if not batch:
            return []
        from vllm import SamplingParams  # lazy

        prompts = []
        for messages in batch:
            text = render_prompt(self.processor, messages, tools, add_generation_prompt=True)
            prompt: dict[str, Any] = {"prompt": text}
            images = collect_images(messages)
            if images:
                prompt["multi_modal_data"] = {"image": images}
            prompts.append(prompt)

        params = SamplingParams(
            # Temperature 0 means greedy decoding in vLLM. top_p is ignored then.
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
            max_tokens=int(sampling.max_new_tokens),
            # logprobs=0 returns only the sampled token's log-prob per step.
            logprobs=0 if sampling.return_logprobs else None,
            seed=sampling.seed,
        )
        try:
            outputs = self.llm.generate(
                prompts, params, lora_request=self._lora_request, use_tqdm=False
            )
        except Exception as exc:  # noqa: BLE001
            # vLLM validates every prompt first and rejects the whole batch if
            # one is longer than max_model_len. Long pages plus several turns
            # can get there. Retry one prompt at a time and mark the over-long
            # ones, so the rollout can end just those episodes as truncated.
            if "maximum model length" not in str(exc) and "max_model_len" not in str(exc):
                raise
            outputs = []
            for prompt in prompts:
                try:
                    outputs.extend(self.llm.generate(
                        [prompt], params, lora_request=self._lora_request, use_tqdm=False
                    ))
                except Exception as one_exc:  # noqa: BLE001
                    if "maximum model length" not in str(one_exc) and "max_model_len" not in str(one_exc):
                        raise
                    outputs.append(None)

        results: list[Generation] = []
        for request_output in outputs:  # vLLM returns outputs in input order
            if request_output is None:
                results.append(Generation(text="", finish_reason=CONTEXT_OVERFLOW))
                continue
            out = request_output.outputs[0]
            token_ids = list(out.token_ids)
            logprobs: list[float] = []
            if sampling.return_logprobs and out.logprobs is not None:
                # out.logprobs[t] maps token_id -> Logprob for step t.
                logprobs = [
                    float(out.logprobs[t][tid].logprob) for t, tid in enumerate(token_ids)
                ]
            results.append(
                Generation(
                    text=out.text,
                    token_ids=token_ids,
                    logprobs=logprobs,
                    prompt_token_ids=list(request_output.prompt_token_ids or []),
                    finish_reason=out.finish_reason or "",
                )
            )
        return results

    # -- memory management --------------------------------------------------

    def sleep(self, level: int = 1) -> None:
        """Offload weights and drop the KV cache so training can use the GPU.

        Needs `vllm.enable_sleep_mode: true` in the config.
        """
        self.llm.sleep(level=level)

    def wake_up(self) -> None:
        """Reload what `sleep()` offloaded."""
        self.llm.wake_up()

    def close(self) -> None:
        """Free GPU memory, best effort.

        vLLM has no single "unload" call, so this stops the engine, tears down
        its distributed state, and clears the CUDA cache. Each step is wrapped
        because a failure in one must not stop the others.
        """
        llm = getattr(self, "llm", None)
        self.llm = None
        try:
            core = llm.llm_engine.engine_core  # type: ignore[union-attr]
            core.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        try:
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hugging Face model (training and scoring)
# ---------------------------------------------------------------------------


def load_hf_model(
    cfg: Config, adapter: Optional[str] = None, trainable: bool = False
) -> tuple[Any, Any]:
    """Load Qwen3-VL and its processor for training or log-prob scoring.

    - `adapter`: path to a saved PEFT adapter to start from.
    - `trainable=True`: the LoRA weights get gradients. Without an adapter path,
      this creates a fresh LoRA (`make_lora_config`).
    - `trainable=False`: inference mode.

    Returns `(model, processor)`.
    """
    import torch  # lazy
    from peft import PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # FlashAttention is faster and uses less memory when installed. sdpa (PyTorch's
    # built-in attention) is the safe default. We feed one episode per forward
    # pass (no padding), so both give the same results.
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    dtype = getattr(torch, str(cfg.model.get("dtype", "bfloat16")))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model.name,
        dtype=dtype,
        attn_implementation=attn,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(cfg.model.name)
    _configure_processor(processor, cfg)

    if trainable:
        # The KV cache is useless in training and conflicts with checkpointing.
        model.config.use_cache = False
        # Gradient checkpointing recomputes activations in the backward pass to
        # save memory. use_reentrant=False is the variant that works when only
        # some parameters (the LoRA weights) require gradients.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The frozen embedding layer produces outputs with no grad history. This
        # makes them require grad so the checkpointed blocks have a graph to
        # backpropagate through. Call it before wrapping with PEFT.
        model.enable_input_require_grads()

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, make_lora_config(cfg))

    if not trainable:
        model.eval()
    return model, processor
