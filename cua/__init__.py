"""RL post-training of Qwen3-VL-4B for browser tasks (BrowserGym MiniWoB++).

This package keeps its imports light on purpose. Importing `cua` never loads
BrowserGym, vLLM, PyTorch, or transformers, so unit tests and CLIs start fast
and run on a laptop without a GPU. Each module imports its heavy dependencies
inside the functions that need them.

Module map:

- `types`: dataclasses shared by every module.
- `config`: the one YAML config loader.
- `env`: BrowserGym MiniWoB++ wrapper, subprocess workers, and the env pool.
- `actions`: tool definitions, tool-call parsing, and BrowserGym action strings.
- `prompts`: system prompt and chat-message construction.
- `rewards`: episode reward and logging statistics.
"""
