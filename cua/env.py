"""BrowserGym MiniWoB++ environment wrapper.

This module has three layers. Read them from the bottom of the stack up:

1. `MiniWoBEnv` runs one BrowserGym environment inside the current process.
   It is not thread-safe. Playwright's sync API binds every object to the
   thread that created it, so you must never share one `MiniWoBEnv` between
   threads.
2. `RemoteEnv` has the same API but runs a `MiniWoBEnv` in a separate spawned
   process and talks to it over a pipe. A crash or hang in Chromium then kills
   the worker, not your training run, and `RemoteEnv` restarts the worker on
   the next call.
3. `EnvPool` holds N `RemoteEnv` objects. The rollout code drives them from a
   thread pool. Threads are safe here because each thread only waits on a pipe
   while the browser work happens in a different process.

Heavy imports (`gymnasium`, `browsergym`, `numpy`) happen lazily inside
`MiniWoBEnv`, so `format_observation` and the other pure helpers stay
importable on a machine without BrowserGym.

Smoke test, run after installing BrowserGym and downloading MiniWoB++:

    python -m cua.env --task click-button --seed 0
"""

from __future__ import annotations

import argparse
import atexit
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
import weakref
from typing import Any, Optional

from PIL import Image

from cua.config import Config, add_config_args, config_from_args
from cua.types import Observation, StepResult

# Appended to a truncated AXTree so the model knows the page text is incomplete.
TRUNCATION_MARKER = "[truncated]"


class EnvCrash(RuntimeError):
    """Raised when the browser or the worker process fails or times out.

    The rollout code catches this, marks the episode as crashed, and drops it
    from the training batch. `RemoteEnv` has already scheduled a restart by the
    time you see this exception.
    """


# ---------------------------------------------------------------------------
# Pure helpers (no BrowserGym needed)
# ---------------------------------------------------------------------------


def normalize_miniwob_url(path_or_url: str) -> str:
    """Turn a directory path into the `MINIWOB_URL` value BrowserGym expects.

    BrowserGym builds the page URL as `MINIWOB_URL + "<task>.html"` with plain
    string concatenation. That is why the value needs a trailing slash. It also
    needs a scheme, because Chromium cannot open a bare path.

    Args:
        path_or_url: A local directory such as `/workspace/miniwob-plusplus/miniwob/html/miniwob`,
            or a URL that already starts with `file://` or `http(s)://`.

    Returns:
        The same location with a scheme and exactly one trailing slash.
    """
    value = path_or_url.strip()
    if not value.startswith(("file://", "http://", "https://")):
        value = "file://" + os.path.abspath(os.path.expanduser(value))
    return value.rstrip("/") + "/"


def truncate_axtree(text: str, max_chars: int) -> str:
    """Cut `text` to at most `max_chars` characters and mark the cut.

    The result, marker included, never exceeds `max_chars`. The function cuts
    at a line boundary when one exists in the second half of the allowed text,
    because a half-cut line such as `[12] button 'Sub` confuses the model.

    Args:
        text: The flattened accessibility tree.
        max_chars: The character budget for the whole result.

    Returns:
        `text` unchanged when it fits, else a shortened copy ending in
        `[truncated]`.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n" + TRUNCATION_MARKER
    keep = max(max_chars - len(suffix), 0)
    head = text[:keep]
    last_newline = head.rfind("\n")
    if last_newline > keep // 2:
        head = head[:last_newline]
    return (head.rstrip() + suffix)[:max_chars]


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    """Shrink `image` so its longer side is at most `long_side` pixels.

    The function never upscales. Upscaling adds image tokens without adding
    information.
    """
    width, height = image.size
    longest = max(width, height)
    if long_side <= 0 or longest <= long_side:
        return image
    scale = long_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _to_pil(screenshot: Any) -> Image.Image:
    """Convert BrowserGym's uint8 HxWx3 array to an RGB PIL image."""
    if hasattr(screenshot, "dtype") and str(screenshot.dtype) != "uint8":
        screenshot = screenshot.astype("uint8")
    return Image.fromarray(screenshot).convert("RGB")


def _goal_text(raw_obs: dict) -> str:
    """Read the goal from `goal`, falling back to the OpenAI-style `goal_object`."""
    goal = raw_obs.get("goal")
    if goal:
        return str(goal)
    parts = []
    for item in raw_obs.get("goal_object") or ():
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(parts)


def format_observation(raw_obs: dict, cfg: Config, step: int = 0) -> Observation:
    """Convert a raw BrowserGym observation dict into an `Observation`.

    This is a pure function so you can unit-test it with a hand-built dict.

    Args:
        raw_obs: The observation dict from `env.reset()` or `env.step()`.
        cfg: The run config. This function reads `env.track`,
            `env.screenshot_long_side`, and `env.max_axtree_chars`.
        step: The step counter to store in the result (0 after reset).

    Returns:
        An `Observation`. `screenshot` is resized for the model, but
        `screen_size` keeps the RAW size, because BrowserGym's coordinate
        actions work in raw screenshot pixels.
    """
    screenshot = None
    screen_size = (0, 0)
    if raw_obs.get("screenshot") is not None:
        raw_image = _to_pil(raw_obs["screenshot"])
        screen_size = raw_image.size
        screenshot = resize_long_side(raw_image, int(cfg.env.screenshot_long_side))

    axtree = ""
    if cfg.env.track == "A":
        # Lazy import: this is the only place that needs BrowserGym's helpers.
        from browsergym.utils.obs import flatten_axtree_to_str

        # filter_visible_only drops off-screen nodes, and skip_generic drops
        # anonymous wrapper nodes. Both shrink the text a lot on MiniWoB pages.
        axtree = flatten_axtree_to_str(
            raw_obs["axtree_object"],
            extra_properties=raw_obs.get("extra_element_properties"),
            filter_visible_only=True,
            skip_generic=True,
        )
        axtree = truncate_axtree(axtree, int(cfg.env.max_axtree_chars))

    return Observation(
        goal=_goal_text(raw_obs),
        screenshot=screenshot,
        axtree=axtree,
        last_action_error=str(raw_obs.get("last_action_error") or ""),
        step=step,
        screen_size=screen_size,
        focused_bid=str(raw_obs.get("focused_element_bid") or ""),
    )


# ---------------------------------------------------------------------------
# In-process environment
# ---------------------------------------------------------------------------


class MiniWoBEnv:
    """One BrowserGym MiniWoB++ environment in the current process.

    The wrapper recreates the underlying gym environment when the task
    changes, after `cfg.env.recycle_every` resets, and after any exception.
    Recycling keeps leaked Chromium processes from piling up over a long run.

    Attributes:
        reset_frame: The full-size screenshot from the last `reset()` when you
            passed `record=True`, else None. `Observation` only carries the
            resized image, so the full-size reset frame lives here.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset_frame: Optional[Image.Image] = None
        self._env: Any = None
        self._task: Optional[str] = None
        self._resets_since_make = 0
        self._record = False
        self._step_count = 0

    # -- lifecycle ----------------------------------------------------------

    def _make(self, task: str) -> None:
        """Create the gym environment for `task`."""
        # MINIWOB_URL must exist before gym.make, because BrowserGym checks it
        # while it builds the task. Respect a value the user already exported.
        if not os.environ.get("MINIWOB_URL"):
            os.environ["MINIWOB_URL"] = normalize_miniwob_url(self.cfg.paths.miniwob_html)

        import gymnasium as gym
        import browsergym.miniwob  # noqa: F401  (importing registers the tasks)
        from browsergym.core.action.highlevel import HighLevelActionSet

        # "bid" gives click/fill/select_option/press/scroll; "coord" gives
        # mouse_click/keyboard_type/keyboard_press. We leave out "infeas"
        # because report_infeasible() would end MiniWoB episodes early.
        # multiaction=False allows exactly one call per action string.
        action_set = HighLevelActionSet(subsets=["bid", "coord"], multiaction=False, strict=False)

        # Do not pass `args` through pw_chromium_kwargs: BrowserGym already
        # sets it and raises a TypeError on a duplicate.
        self._env = gym.make(
            f"browsergym/miniwob.{task}",
            headless=bool(self.cfg.env.get("headless", True)),
            action_mapping=action_set.to_python_code,
            slow_mo=self.cfg.env.slow_mo_ms,
            pre_observation_delay=self.cfg.env.pre_observation_delay,
        )
        self._task = task
        self._resets_since_make = 0

    def _close_gym(self) -> None:
        """Close the gym environment and forget it. Never raises."""
        env, self._env, self._task = self._env, None, None
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001  (a broken browser can fail to close)
                pass

    def close(self) -> None:
        """Close the browser. Call this when you finish with the env."""
        self._close_gym()

    # -- gym-like API -------------------------------------------------------

    def reset(self, task: str, seed: int, record: bool = False) -> Observation:
        """Start a new episode.

        Args:
            task: MiniWoB task name without prefix, for example `click-button`.
            seed: The seed BrowserGym passes to the task generator.
            record: Keep full-size frames. The reset frame goes to
                `self.reset_frame` and each `step()` returns its frame.

        Returns:
            The first observation.
        """
        recycle_every = int(self.cfg.env.recycle_every)
        needs_new_env = (
            self._env is None
            or task != self._task
            or (recycle_every > 0 and self._resets_since_make >= recycle_every)
        )
        if needs_new_env:
            self._close_gym()
            self._make(task)
        try:
            raw_obs, _info = self._env.reset(seed=seed)
        except Exception:
            self._close_gym()  # Do not reuse a browser that failed to reset.
            raise
        self._resets_since_make += 1
        self._record = record
        self._step_count = 0
        self.reset_frame = _to_pil(raw_obs["screenshot"]) if record else None
        return format_observation(raw_obs, self.cfg, step=0)

    def step(self, action: str) -> StepResult:
        """Run one BrowserGym action string, such as `click('12')`."""
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        try:
            raw_obs, reward, terminated, truncated, info = self._env.step(action)
        except Exception:
            self._close_gym()
            raise
        self._step_count += 1

        # BrowserGym's reward is 1.0 or 0.0. MiniWoB's own score hides in
        # info["task_info"]. Fall back to the BrowserGym reward if it is absent.
        task_info = (info or {}).get("task_info") or {}
        raw_reward = task_info.get("RAW_REWARD_GLOBAL", reward)
        try:
            raw_reward = float(raw_reward)
        except (TypeError, ValueError):
            raw_reward = float(reward)

        frame = _to_pil(raw_obs["screenshot"]) if self._record else None
        return StepResult(
            observation=format_observation(raw_obs, self.cfg, step=self._step_count),
            success=float(reward) > 0,
            raw_env_reward=raw_reward,
            terminated=bool(terminated or truncated),
            frame=frame,
        )


# ---------------------------------------------------------------------------
# Subprocess environment
# ---------------------------------------------------------------------------


def _worker_main(conn: Any, cfg_dict: dict) -> None:
    """Command loop that runs inside the spawned worker process.

    The parent sends tuples: ("reset", task, seed, record), ("step", action),
    or ("close",). The worker replies with ("ok", payload) or ("err", text).
    """
    # Debug aid: with CUA_DEBUG_WORKERS=1, each worker dumps every thread's
    # stack to /tmp/cua-worker-PID.trace every 20 s. py-spy can't attach inside
    # Runpod containers (no SYS_PTRACE), so this is how you see where a worker
    # is stuck.
    if os.environ.get("CUA_DEBUG_WORKERS"):
        import faulthandler

        trace_file = open(f"/tmp/cua-worker-{os.getpid()}.trace", "w")  # noqa: SIM115
        trace_file.write(f"worker started {time.time():.1f}\n")
        trace_file.flush()
        faulthandler.dump_traceback_later(20, repeat=True, file=trace_file)

    # Start a new session so the worker and every Chromium it launches share a
    # process group. The parent can then kill the whole group in one call.
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass

    # Run the worker (and the Chromium it launches) from a local directory.
    # On Runpod, the repo's directory is on a network filesystem, and with vLLM
    # loaded in the parent, Chromium's start-up processes blocked for over 60 s
    # in kernel path lookups (wchan d_alloc_parallel) under that cwd.
    try:
        os.chdir(os.environ.get("CUA_WORKER_CWD", "/tmp"))
    except OSError:
        pass

    env = MiniWoBEnv(Config(cfg_dict))
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:  # The parent exited.
                break
            command = msg[0]
            if command == "close":
                break
            try:
                if command == "reset":
                    obs = env.reset(msg[1], msg[2], msg[3])
                    conn.send(("ok", (obs, env.reset_frame)))
                elif command == "step":
                    conn.send(("ok", env.step(msg[1])))
                else:
                    conn.send(("err", f"Unknown command {command!r}."))
            except BaseException as exc:  # noqa: BLE001  (report everything to the parent)
                conn.send(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        env.close()
        conn.close()


# TRL never closes the environments it creates, so we track every live
# RemoteEnv and close them when the interpreter exits. A WeakSet lets garbage
# collection still free envs you drop on purpose.
_LIVE_REMOTE_ENVS: "weakref.WeakSet[RemoteEnv]" = weakref.WeakSet()

# Serializes worker start-up, because RemoteEnv._start briefly edits __main__.
_SPAWN_LOCK = threading.Lock()


@atexit.register
def _close_all_remote_envs() -> None:
    for env in list(_LIVE_REMOTE_ENVS):
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


class RemoteEnv:
    """A `MiniWoBEnv` that runs in its own spawned process.

    The process starts lazily on the first call. If the worker crashes, hangs
    for longer than `cfg.env.step_timeout_s`, or reports an error, `RemoteEnv`
    kills it, raises `EnvCrash`, and starts a fresh worker on the next call.
    After a restart, call `reset()` before `step()`.

    One thread at a time can use a `RemoteEnv`. A lock enforces that.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset_frame: Optional[Image.Image] = None
        # Send a plain dict across the process boundary. The child rebuilds Config.
        self._cfg_dict = cfg.to_dict()
        self._proc: Any = None
        self._conn: Any = None
        self._lock = threading.Lock()
        _LIVE_REMOTE_ENVS.add(self)

    # -- process management -------------------------------------------------

    def _start(self) -> None:
        # "spawn" starts a clean interpreter. Forking a process that already
        # holds CUDA or vLLM state corrupts that state, so we never fork.
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker_main, args=(child_conn, self._cfg_dict), daemon=True)
        # By default a spawned child re-imports the parent's __main__ module
        # (for example cua.nano_grpo.loop, which imports torch and transformers).
        # Thirty-two browser workers would each load those for nothing: seconds of
        # start-up and hundreds of MB of RAM apiece. The worker only needs
        # cua.env, so hide __main__'s identity while the child starts.
        main = sys.modules.get("__main__")
        saved_spec = getattr(main, "__spec__", None)
        saved_file = getattr(main, "__file__", None)
        with _SPAWN_LOCK:
            try:
                if main is not None:
                    main.__spec__ = None
                    if saved_file is not None:
                        del main.__file__
                proc.start()
            finally:
                if main is not None:
                    main.__spec__ = saved_spec
                    if saved_file is not None:
                        main.__file__ = saved_file
        child_conn.close()  # The parent must not keep the child's end open.
        self._proc, self._conn = proc, parent_conn

    def _kill(self) -> None:
        """Kill the worker and its browser children. Never raises."""
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is not None:
            try:
                # The worker called setsid(), so its pid is also its group id.
                os.killpg(proc.pid, signal.SIGKILL)
            except (AttributeError, OSError):
                pass
            try:
                proc.kill()
                proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _call(self, *msg: Any) -> Any:
        """Send one command and wait for the reply, restarting on failure."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._kill()
                self._start()
            timeout = float(self.cfg.env.step_timeout_s)
            try:
                self._conn.send(msg)
                if not self._conn.poll(timeout):
                    self._kill()
                    raise EnvCrash(f"Browser worker timed out after {timeout:.0f} s on {msg[0]}.")
                status, payload = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._kill()
                raise EnvCrash(f"Browser worker died during {msg[0]}: {exc}") from exc
            if status == "err":
                # The worker is alive but its browser state is suspect, so restart.
                self._kill()
                raise EnvCrash(str(payload))
            return payload

    # -- same API as MiniWoBEnv ----------------------------------------------

    def reset(self, task: str, seed: int, record: bool = False) -> Observation:
        """Start a new episode in the worker. See `MiniWoBEnv.reset`."""
        obs, frame = self._call("reset", task, seed, record)
        self.reset_frame = frame
        return obs

    def step(self, action: str) -> StepResult:
        """Run one action in the worker. See `MiniWoBEnv.step`."""
        return self._call("step", action)

    def close(self) -> None:
        """Ask the worker to exit, then kill it if it does not."""
        proc, conn = self._proc, self._conn
        if proc is not None and conn is not None:
            try:
                conn.send(("close",))
                proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._kill()


class EnvPool:
    """A fixed set of `RemoteEnv` workers that the rollout code reuses.

    Use it as a context manager so the workers always shut down:

        with EnvPool(cfg, size=4) as pool:
            obs = pool.envs[0].reset("click-button", seed=0)

    Attributes:
        envs: The `RemoteEnv` objects. Each one is one Chromium in its own
            process, and the processes start lazily.
    """

    def __init__(self, cfg: Config, size: Optional[int] = None):
        self.cfg = cfg
        size = int(size if size is not None else cfg.env.pool_size)
        self.envs: list[RemoteEnv] = [RemoteEnv(cfg) for _ in range(size)]

    def close(self) -> None:
        """Close every worker."""
        for env in self.envs:
            env.close()

    def __enter__(self) -> "EnvPool":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cua.env",
        description="Reset one MiniWoB++ task, print the observation, run noop(), and close.",
    )
    parser.add_argument("--task", default="click-button", help="MiniWoB task name without prefix.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the task generator.")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run the env in a spawned worker process, as the rollout code does.",
    )
    add_config_args(parser, "configs/base.yaml")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    # Imported here so `import cua.env` never depends on prompts.
    from cua.prompts import observation_text

    env = RemoteEnv(cfg) if args.remote else MiniWoBEnv(cfg)
    try:
        obs = env.reset(args.task, args.seed)
        print(f"== reset: task={args.task} seed={args.seed} track={cfg.env.track}")
        print(observation_text(obs, cfg.env.track))
        if obs.screenshot is not None:
            print(f"screenshot (model view): {obs.screenshot.size}  raw: {obs.screen_size}")
        result = env.step("noop()")
        print(
            f"== step noop(): success={result.success} raw_reward={result.raw_env_reward} "
            f"terminated={result.terminated} error={result.observation.last_action_error!r}"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
