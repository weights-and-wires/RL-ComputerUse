"""Shared helpers for the run monitor: paths, atomic writes, scrubbing, evals.

Standard library only, so the monitor runs on the system Python on Bazzite and
the tests run with plain `python3` on macOS.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import re
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def infer_run(install_dir: str | os.PathLike | None = None) -> tuple[str, Path | None]:
    """Work out the run name and home from where the monitor code is installed.

    Code at `~/cua-run3/monitor/` belongs to run 3 and uses `~/cua-run3` as its
    home. Code anywhere else (for example your checkout) returns no home.
    """
    here = Path(install_dir or Path(os.path.abspath(__file__)).parent)  # abspath, not resolve: keep ~ spelling
    m = re.fullmatch(r"cua-(run\d+[a-z]?)", here.parent.name)
    if here.name == "monitor" and m:
        return m.group(1), here.parent
    return DEFAULT_RUN, None


DEFAULT_RUN = "run4"


class RunNames:
    """Every name that depends on the run, derived from the one setting `run`.

    The pod side uses the same pattern for each run: `scripts/<run>.sh`,
    `scripts/<run>_pod_start.sh`, `scripts/upload_<run>.py`, tmux sessions
    `<run>` and `<run>-sync`, and `results/<run>/`.
    """

    def __init__(self, run: str = DEFAULT_RUN):
        if not re.fullmatch(r"run\d+[a-z]?", run):
            raise ValueError(f"run name {run!r} must look like run4")
        self.run = run
        m = re.fullmatch(r"run(\d+)([a-z]?)", run)
        self.label = f"Run {m.group(1)}{m.group(2)}"
        self.results = f"results/{run}"
        self.metrics = f"{self.results}/train/grpo-4b-{run}/metrics.jsonl"
        self.stop_flag = f"{self.results}/.stop_grpo"
        self.samples = f"{self.results}/samples"
        self.run_sh = f"scripts/{run}.sh"
        self.pod_start = f"scripts/{run}_pod_start.sh"
        self.upload = f"scripts/upload_{run}.py"
        self.tmux = run
        self.tmux_sync = f"{run}-sync"
        self.unit = f"cua-{run}-monitor"
        self.home_name = f"cua-{run}"
        self.hf_ckpt = f"checkpoints/grpo-4b-{run}"
        self.hf_results = f"results/{run}"

    def required_repo_files(self) -> list[str]:
        return [self.pod_start, self.run_sh, self.upload]

    def pod_name(self, started_ts: float, gen: int) -> str:
        return f"cua-{self.run}-{int(started_ts or 0)}-g{gen}"


class Paths:
    """Every file the monitor reads or writes, rooted at the run's home folder.

    Order of precedence for the run name: the `run` argument, `CUA_RUN`, the
    install folder (`~/cua-run3/monitor` means run 3), then `run4`. The home is
    the `home` argument, `CUA_RUN_HOME`, the install folder's parent, then
    `~/cua-<run>`.
    """

    def __init__(self, home: str | os.PathLike | None = None, run: str | None = None):
        inferred_run, inferred_home = infer_run()
        self.run = run or os.environ.get("CUA_RUN") or inferred_run
        self.names = RunNames(self.run)
        default_home = inferred_home if (inferred_home and self.run == inferred_run) else Path.home() / self.names.home_name
        self.home = Path(home or os.environ.get("CUA_RUN_HOME") or default_home)
        self.env = self.home / ".env"
        self.repo = self.home / "repo"
        self.state = self.home / "state.json"
        self.data = self.home / "data"
        self.armed = self.home / "ARMED"
        self.command = self.home / "command.json"
        self.log = self.home / "monitor.log"
        self.lock = self.home / "monitor.lock"
        self.extra_keys = self.home / "authorized_keys.extra"
        self.cache = self.home / "cache"

    def d(self, name: str) -> Path:
        return self.data / name


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def now_iso(ts: float | None = None) -> str:
    t = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc) if ts is not None else _dt.datetime.now(_dt.timezone.utc)
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        s = s.strip().replace("Z", "+00:00")
        t = _dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return t.timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n", mode)


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def append_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Append rows by rewriting the file through a temp file and a rename.

    Readers never see a half-written line. The files stay small (thousands of
    lines), so the rewrite is cheap.
    """
    if not rows:
        return
    path = Path(path)
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        old = ""
    if old and not old.endswith("\n"):
        old += "\n"
    new = "".join(json.dumps(r, default=str) + "\n" for r in rows)
    atomic_write_text(path, old + new)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

SECRET_NAMES = ("RUNPOD_API_KEY", "HF_TOKEN", "WANDB_API_KEY")
_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"rpa_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{40}(?![0-9a-fA-F])"),
]
REDACTED = "[redacted]"


def load_env(path: Path) -> dict[str, str]:
    """Parse NAME=value lines (optionally `export`, optionally quoted)."""
    env: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return env
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            continue
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            v = v.split(" #", 1)[0].strip()
        env[k] = v
    return env


class Scrubber:
    """Replaces known secret values and secret-shaped strings with [redacted]."""

    def __init__(self, env: dict[str, str] | None = None):
        self.values = sorted(
            {v for k, v in (env or {}).items() if k in SECRET_NAMES and v and len(v) >= 6},
            key=len,
            reverse=True,
        )

    def __call__(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        for v in self.values:
            if v in text:
                text = text.replace(v, REDACTED)
        for pat in _SECRET_PATTERNS:
            text = pat.sub(REDACTED, text)
        return text

    def obj(self, o):
        """Scrub every string inside a JSON-like object."""
        if isinstance(o, str):
            return self(o)
        if isinstance(o, list):
            return [self.obj(x) for x in o]
        if isinstance(o, dict):
            return {self(k) if isinstance(k, str) else k: self.obj(v) for k, v in o.items()}
        return o


# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------


def wilson_ci(successes: float, n: float, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_eval_dir(path: Path) -> dict | None:
    """Turn one eval folder (per_task.csv and/or summary.json) into the evals.json entry.

    `mean` is the pooled success rate over all episodes, and `ci95` is the
    Wilson score interval on that pooled rate.
    """
    path = Path(path)
    summary = read_json(path / "summary.json", default=None) or {}
    tasks: dict[str, dict] = {}
    csv_path = path / "per_task.csv"
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = (row.get("task") or "").strip()
                    if not name or name.upper() in ("OVERALL", "ALL", "MEAN"):
                        continue
                    sr = _f(row.get("success_rate"))
                    n = _f(row.get("n"))
                    if sr is None:
                        continue
                    tasks[name] = {"success": round(sr, 4), "n": int(n) if n is not None else None}
        except (OSError, csv.Error):
            pass
    if not tasks and isinstance(summary.get("per_task"), dict):
        for name, t in summary["per_task"].items():
            sr = _f((t or {}).get("success_rate"))
            if sr is None:
                continue
            n = _f(t.get("n"))
            tasks[name] = {"success": round(sr, 4), "n": int(n) if n is not None else None}
    if not tasks and not summary:
        return None
    succ = sum(t["success"] * (t["n"] or 0) for t in tasks.values())
    n_tot = sum((t["n"] or 0) for t in tasks.values())
    mean = _f(summary.get("overall_success_pooled")) or _f(summary.get("overall_success"))
    if mean is None and n_tot:
        mean = succ / n_tot
    if n_tot == 0:
        n_tot = int(_f(summary.get("n_episodes")) or 0)
        succ = (mean or 0) * n_tot
    return {
        "mode": summary.get("mode"),
        "track": summary.get("track"),
        "tasks": tasks,
        "mean": round(mean, 4) if mean is not None else None,
        "ci95": wilson_ci(succ, n_tot) if n_tot else None,
        "n": n_tot,
    }


def build_evals(eval_root: Path) -> dict:
    out = {}
    root = Path(eval_root)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir():
            e = parse_eval_dir(d)
            if e:
                out[d.name] = e
    return out


def clean_log_text(text: str, scrub: Scrubber, max_lines: int = 2000) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n")]
    # Collapse the empty lines that progress bars leave behind.
    lines = [ln for ln in lines if ln.strip() != ""]
    lines = lines[-max_lines:]
    return scrub("\n".join(lines) + "\n")


def sample_index(samples_dir: Path) -> list[dict]:
    rows = []
    sd = Path(samples_dir)
    if not sd.is_dir():
        return rows
    for d in sd.iterdir():
        if not d.is_dir() or not d.name.startswith("step-"):
            continue
        ep = read_json(d / "episode.json", default={}) or {}
        try:
            step = int(ep.get("step", d.name.split("-", 1)[1]))
        except (ValueError, TypeError):
            step = None
        rows.append({
            "step": step,
            "task": ep.get("task"),
            "success": ep.get("success"),
            "reward": ep.get("reward"),
            "dir": d.name,
        })
    rows.sort(key=lambda r: (r["step"] is not None, r["step"] or 0, r["dir"]), reverse=True)
    return rows
