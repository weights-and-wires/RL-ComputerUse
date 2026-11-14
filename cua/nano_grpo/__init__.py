"""A hand-written GRPO trainer for Qwen3-VL browser agents .

This package is the default trainer for the project (`trainer: nano` in the
config) and doubles as study material. It has no TRL dependency.

- `loss.py` holds the math as pure PyTorch functions: group advantages,
  token log-probabilities, and the clipped policy-gradient loss. Start there.
  The module docstring works through the exercise in docs/LEARNING.md by hand.
- `loop.py` holds the training loop: roll out with vLLM, compute advantages,
  update the LoRA adapter with PyTorch, and sync the new adapter back to vLLM.

Run it with:

    python -m cua.nano_grpo.loop --config configs/grpo_qwen3vl4b.yaml [--set KEY=VALUE] [--resume]
"""
