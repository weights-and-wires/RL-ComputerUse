"""Watch the model do MiniWoB++ tasks in a small local web UI.

The demo loads Qwen3-VL-4B with Hugging Face transformers (CUDA, Apple MPS, or
CPU), attaches the run-1 and run-2 LoRA adapters, and drives one headless
Chromium browser through BrowserGym. A page served on localhost lets you pick a
task and a model, then shows each turn as it happens: the screenshot, the
model's reasoning, the action it chose, and the reward.

The episode loop matches `cua.rollout`: the same prompts, the same tool-call
parser, greedy decoding, and at most `env.max_turns` turns. Scores can still
differ a little from the published evaluation, because the evaluation ran in
vLLM on a GPU and small numeric differences can change a greedy choice.

Run it through `scripts/demo.sh`, which installs everything first. To run it
directly from the repository root:

    python -m demo.app --miniwob-html PATH --adapter-run2 PATH [--adapter-run1 PATH]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import random
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from cua import actions, prompts
from cua.config import load_config
from cua.env import MiniWoBEnv, normalize_miniwob_url
from cua.policy import _configure_processor, collect_images, render_prompt

HERE = Path(__file__).resolve().parent

# Greedy success rates on 50 or 100 unseen eval seeds, from results/run2/compare.
# "good": run 2 solves 90% or more. "moderate": run 2 solves 40% to 80%.
SAMPLE_TASKS: list[dict[str, Any]] = [
    # -- the model is good at these ---------------------------------------
    {"task": "enter-date", "group": "good", "base": 0, "run1": 98, "run2": 98,
     "about": "Type a date into a date field and submit. The original model never solved it."},
    {"task": "copy-paste-2", "group": "good", "base": 18, "run1": 10, "run2": 100,
     "about": "Copy text from one box into another and submit."},
    {"task": "email-inbox-important", "group": "good", "base": 18, "run1": 8, "run2": 100,
     "about": "Find an email from a given sender and mark it important."},
    {"task": "guess-number", "group": "good", "base": 100, "run1": 18, "run2": 100,
     "about": "Guess a number using higher or lower hints. Run 1 forgot this one."},
    {"task": "use-spinner", "group": "good", "base": 86, "run1": 52, "run2": 96,
     "about": "Set a number spinner to a target value and submit."},
    {"task": "find-word", "group": "good", "base": 32, "run1": 100, "run2": 96,
     "about": "Find the nth word in a paragraph and type it."},
    {"task": "login-user-popup", "group": "good", "base": 78, "run1": 94, "run2": 94,
     "held_out": True,
     "about": "Log in with a username and password, closing a popup if one appears."},
    # -- moderately difficult -----------------------------------------------
    {"task": "email-inbox-star-reply", "group": "moderate", "base": 10, "run1": 22, "run2": 72,
     "about": "Find an email, star it, or reply to it, as the goal asks."},
    {"task": "text-transform", "group": "moderate", "base": 62, "run1": 60, "run2": 76,
     "about": "Read distorted text in a picture and type it."},
    {"task": "email-inbox-delete", "group": "moderate", "base": 60, "run1": 64, "run2": 65,
     "held_out": True, "about": "Find an email from a given sender and delete it."},
    {"task": "multi-layouts", "group": "moderate", "base": 58, "run1": 58, "run2": 58,
     "about": "Fill in a form whose layout changes from page to page."},
    {"task": "email-inbox", "group": "moderate", "base": 6, "run1": 30, "run2": 54,
     "about": "Do one of several email jobs: forward, reply, star, or delete."},
    {"task": "click-scroll-list", "group": "moderate", "base": 38, "run1": 54, "run2": 52,
     "about": "Select several items in a scrolling list and submit."},
    {"task": "choose-date", "group": "moderate", "base": 32, "run1": 44, "run2": 46,
     "about": "Pick a date in a calendar widget, moving between months."},
    {"task": "social-media", "group": "moderate", "base": 6, "run1": 32, "run2": 40,
     "about": "Do an action, such as like or share, on a given user's post."},
]
TASK_NAMES = {t["task"] for t in SAMPLE_TASKS}

# Seeds 0-99 are the eval range in configs/tasks.yaml: no training run ever saw them.
EVAL_SEEDS = range(0, 100)

MODELS = {
    "base": "Original model",
    "run1": "Run 1",
    "run2": "Run 2",
}


def png_data_uri(image: Optional[Image.Image]) -> Optional[str]:
    """Encode a PIL image as a PNG data URI, or return None."""
    if image is None:
        return None
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HFPolicy:
    """Qwen3-VL with up to two LoRA adapters, for greedy, one-at-a-time decoding."""

    def __init__(self, cfg: Any, adapters: dict[str, str], device: str):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_new_tokens = int(cfg.generation.max_new_tokens)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model.name, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(cfg.model.name)
        _configure_processor(self.processor, cfg)

        self.adapters: list[str] = []
        for name, path in adapters.items():
            if not path:
                continue
            if not self.adapters:
                model = PeftModel.from_pretrained(model, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            self.adapters.append(name)
        model.eval()
        self.model = model
        self.tools = actions.tool_schemas("A")

    def available(self) -> list[str]:
        return ["base", *self.adapters]

    def generate(self, messages: list[dict[str, Any]], which: str) -> str:
        """Return the model's reply to a conversation, decoded greedily."""
        torch = self.torch
        text = render_prompt(self.processor, messages, self.tools, add_generation_prompt=True)
        images = collect_images(messages)
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self.device)

        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        with torch.inference_mode():
            if which == "base" and self.adapters:
                with self.model.disable_adapter():
                    out = self.model.generate(**inputs, **kwargs)
            else:
                if self.adapters:
                    self.model.set_adapter(which)
                out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Episode runner (one worker thread owns the model and the browser)
# ---------------------------------------------------------------------------


@dataclass
class Run:
    id: int
    task: str
    seed: int
    model: str
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    stop: bool = False


class Runner:
    """Runs one episode at a time on a dedicated thread.

    Playwright's sync API works only on the thread that started it, so the
    browser and the model both live on this one worker thread.
    """

    def __init__(self, cfg: Any, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.status = "starting"
        self.status_detail = ""
        self.policy: Optional[HFPolicy] = None
        self.runs: dict[int, Run] = {}
        self.current: Optional[Run] = None
        self.lock = threading.Lock()
        self.jobs: "queue.Queue[Run]" = queue.Queue()
        self._next_id = 1
        threading.Thread(target=self._worker, daemon=True).start()

    # -- called from HTTP threads ---------------------------------------------

    def submit(self, task: str, seed: Optional[int], model: str) -> Run:
        with self.lock:
            if self.current is not None and not self.current.done:
                raise RuntimeError("An episode is already running. Stop it or wait for it to end.")
            if seed is None:
                seed = random.choice(EVAL_SEEDS)
            run = Run(self._next_id, task, int(seed), model)
            self._next_id += 1
            self.runs[run.id] = run
            self.current = run
        self.jobs.put(run)
        return run

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.status_detail,
            "models": [
                {"id": m, "label": MODELS[m]}
                for m in (self.policy.available() if self.policy else [])
            ],
            "tasks": SAMPLE_TASKS,
            "max_turns": int(self.cfg.env.max_turns),
            "busy": bool(self.current and not self.current.done),
        }

    # -- worker thread ----------------------------------------------------------

    def _worker(self) -> None:
        try:
            self.status, self.status_detail = "loading", "Loading the model. The first start downloads about 9 GB."
            device = pick_device(self.args.device)
            adapters = {"run1": self.args.adapter_run1, "run2": self.args.adapter_run2}
            self.policy = HFPolicy(self.cfg, adapters, device)
            self.status_detail = f"Starting the browser (model on {device})."
            self.env = MiniWoBEnv(self.cfg)
            self.status, self.status_detail = "ready", f"Model on {device}."
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.status, self.status_detail = "error", f"Startup failed: {exc}"
            return

        while True:
            run = self.jobs.get()
            try:
                self._play(run)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                run.events.append({"type": "error", "message": str(exc)})
                self.env.close()  # Start the next episode with a fresh browser.
                self.env = MiniWoBEnv(self.cfg)
            finally:
                run.done = True

    def _play(self, run: Run) -> None:
        assert self.policy is not None
        track = "A"
        max_turns = int(self.cfg.env.max_turns)
        obs = self.env.reset(run.task, run.seed, record=True)
        run.events.append({
            "type": "start", "task": run.task, "seed": run.seed, "model": run.model,
            "goal": obs.goal, "screenshot": png_data_uri(self.env.reset_frame),
            "elements": obs.axtree,
        })
        messages = prompts.initial_messages(obs, track, with_image=bool(self.cfg.env.first_obs_image))
        success = False
        for turn in range(1, max_turns + 1):
            if run.stop:
                run.events.append({"type": "end", "success": False, "reason": "Stopped."})
                return
            t0 = time.perf_counter()
            reply = self.policy.generate(messages, run.model)
            think_s = time.perf_counter() - t0
            parsed = actions.parse_response(reply, track, obs.screen_size)
            event: dict[str, Any] = {
                "type": "turn", "turn": turn, "reply": reply, "reasoning": parsed.reasoning,
                "tool": parsed.tool_name, "args": parsed.tool_args, "action": parsed.action,
                "format_error": parsed.format_error, "think_s": round(think_s, 1),
            }

            if parsed.ends_episode:
                event["note"] = "The reply had no tool call, so the episode ends."
                run.events.append(event)
                break
            if parsed.action is None:
                error_text = str({"error": parsed.format_error or "invalid tool call"})
                event["note"] = "The tool call was invalid; the model gets the error and tries again."
                run.events.append(event)
                content = prompts.tool_result_content(obs, track, error=error_text)
                prompts.append_turn(messages, reply, content, parsed.tool_name or "unknown")
                continue

            res = self.env.step(parsed.action)
            obs = res.observation
            event.update({
                "screenshot": png_data_uri(res.frame or obs.screenshot),
                "elements": obs.axtree,
                "page_error": obs.last_action_error,
                "success": bool(res.success),
            })
            run.events.append(event)
            content = prompts.tool_result_content(obs, track, error=obs.last_action_error)
            prompts.append_turn(messages, reply, content, parsed.tool_name)
            success = bool(res.success)
            if res.success or res.terminated:
                break

        if success:
            reason = "The page reports the task as done."
        elif len([e for e in run.events if e["type"] == "turn"]) >= max_turns:
            reason = f"The model used all {max_turns} turns."
        else:
            reason = "The episode ended without success."
        run.events.append({"type": "end", "success": success, "reward": int(success), "reason": reason})


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def make_handler(runner: Runner) -> type[BaseHTTPRequestHandler]:
    index_html = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index_html)))
                self.end_headers()
                self.wfile.write(index_html)
            elif self.path == "/api/info":
                self._json(runner.info())
            elif self.path.startswith("/api/run/"):
                # /api/run/<id>?since=<n> returns events from index n onward.
                path, _, query = self.path.partition("?")
                try:
                    run = runner.runs[int(path.rsplit("/", 1)[1])]
                except (ValueError, KeyError):
                    self._json({"error": "Unknown run."}, 404)
                    return
                since = 0
                for part in query.split("&"):
                    if part.startswith("since="):
                        since = int(part[6:] or 0)
                self._json({"events": run.events[since:], "done": run.done})
            else:
                self._json({"error": "Not found."}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "The request body must be JSON."}, 400)
                return
            if self.path == "/api/run":
                if runner.status != "ready":
                    self._json({"error": "The model is still loading."}, 409)
                    return
                task, model = body.get("task"), body.get("model", "run2")
                if task not in TASK_NAMES:
                    self._json({"error": f"Unknown task {task!r}."}, 400)
                    return
                if runner.policy is None or model not in runner.policy.available():
                    self._json({"error": f"Model {model!r} isn't loaded."}, 400)
                    return
                seed = body.get("seed")
                try:
                    seed = None if seed in (None, "") else int(seed)
                    run = runner.submit(task, seed, model)
                except (TypeError, ValueError):
                    self._json({"error": "The seed must be a whole number."}, 400)
                    return
                except RuntimeError as exc:
                    self._json({"error": str(exc)}, 409)
                    return
                self._json({"id": run.id, "seed": run.seed})
            elif self.path == "/api/stop":
                if runner.current is not None:
                    runner.current.stop = True
                self._json({"ok": True})
            else:
                self._json({"error": "Not found."}, 404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch Qwen3-VL-4B do MiniWoB++ tasks in a local web UI.")
    parser.add_argument("--miniwob-html", required=True,
                        help="Path to miniwob-plusplus/miniwob/html/miniwob/.")
    parser.add_argument("--adapter-run2", default="", help="Directory of the run-2 LoRA adapter.")
    parser.add_argument("--adapter-run1", default="", help="Directory of the run-1 LoRA adapter.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--show-browser", action="store_true",
                        help="Also open a visible Chromium window and slow its actions down.")
    parser.add_argument("--no-open", action="store_true", help="Don't open the UI in your browser.")
    args = parser.parse_args()

    overrides = [f"paths.miniwob_html={args.miniwob_html}", "env.recycle_every=0"]
    if args.show_browser:
        overrides += ["env.headless=false", "env.slow_mo_ms=400"]
    cfg = load_config("configs/base.yaml", overrides)
    # Point BrowserGym at the local MiniWoB++ pages before any env starts.
    import os
    os.environ["MINIWOB_URL"] = normalize_miniwob_url(args.miniwob_html)

    runner = Runner(cfg, args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Demo UI: {url}  (press Ctrl+C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
