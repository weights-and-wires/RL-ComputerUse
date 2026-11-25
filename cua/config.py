"""The one config loader every entry point uses.

Configs are plain YAML files in `configs/`. A file can inherit from another
with a top-level `extends:` key (a path relative to the file). Values in the
child override the parent, merged key by key.

Every CLI entry point accepts the same two flags:

    --config configs/grpo_qwen3vl4b.yaml
    --set grpo.learning_rate=5e-6 --set env.pool_size=24

`--set` values are parsed as YAML, so `5e-6`, `true`, and `[1, 2]` become a
float, a bool, and a list.

Example:

    cfg = load_config("configs/eval.yaml", ["eval.num_seeds=10"])
    cfg.env.max_turns        # attribute access
    cfg["env"]["max_turns"]  # dict access also works
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class Config(dict):
    """A dict that also allows attribute access (cfg.env.max_turns).

    Nested dicts are converted to Config on access, so the whole tree
    supports dots. Missing keys raise AttributeError with the full key name,
    which makes typos in configs easy to find.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key {key!r}") from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy, for W&B `config=` and JSON dumps."""
        return json.loads(json.dumps(self, default=str))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_with_extends(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_data = _load_yaml_with_extends((path.parent / parent).resolve())
        data = _deep_merge(parent_data, data)
    return data


def _set_dotted(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load a YAML config, resolve `extends:`, and apply `key=value` overrides."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    data = _load_yaml_with_extends(path.resolve())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    cfg = Config(data)
    cfg["_config_path"] = str(path)
    return cfg


def add_config_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    """Add the shared --config and --set flags to an argparse parser."""
    parser.add_argument("--config", default=default_config, help="Path to a YAML config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, for example --set env.max_turns=6. Repeatable.",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


# ---------------------------------------------------------------------------
# Task pool and seeds (configs/tasks.yaml)
# ---------------------------------------------------------------------------


@dataclass
class SeedRange:
    """A half-open range of integer seeds: [start, start + count)."""

    start: int
    count: int

    def seeds(self, n: Optional[int] = None) -> list[int]:
        n = self.count if n is None else min(n, self.count)
        return list(range(self.start, self.start + n))


@dataclass
class TaskSpec:
    train_candidates: list[str]
    heldout: list[str]
    seed_ranges: dict[str, SeedRange]
    train_candidates_run2_extra: list[str] = field(default_factory=list)
    run1_pool: list[str] = field(default_factory=list)
    heldout_v2_candidates: list[str] = field(default_factory=list)
    train_candidates_run3_extra: list[str] = field(default_factory=list)
    run2_pool: list[str] = field(default_factory=list)
    heldout_v2: list[str] = field(default_factory=list)
    """Run 2's held-out v2 list, frozen in tasks.yaml (results/ isn't in git)."""
    train_candidates_run4_new_action: list[str] = field(default_factory=list)
    train_candidates_run4_extra: list[str] = field(default_factory=list)
    run3_pool: list[str] = field(default_factory=list)
    run4_long_horizon: list[str] = field(default_factory=list)
    run4_turn_limited: list[str] = field(default_factory=list)
    run4_exclude: dict[str, str] = field(default_factory=dict)
    full_suite_exclude: dict[str, str] = field(default_factory=dict)

    @property
    def train_candidates_run2(self) -> list[str]:
        return self.train_candidates + [t for t in self.train_candidates_run2_extra if t not in self.train_candidates]

    @property
    def train_candidates_run3(self) -> list[str]:
        base = self.train_candidates_run2
        return base + [t for t in self.train_candidates_run3_extra if t not in base]

    @property
    def train_candidates_run4(self) -> list[str]:
        """Run 3's candidates without `run4_exclude`, plus the drag tasks and the unscreened extras."""
        out = [t for t in self.train_candidates_run3 if t not in self.run4_exclude]
        for t in self.train_candidates_run4_new_action + self.train_candidates_run4_extra:
            if t not in out and t not in self.run4_exclude:
                out.append(t)
        return out

    def seed_range(self, name: str) -> SeedRange:
        """Seed range by purpose: 'eval', 'baseline', 'rft', or 'train'."""
        return self.seed_ranges[name]


def load_tasks(path: str | os.PathLike = CONFIG_DIR / "tasks.yaml") -> TaskSpec:
    data = _load_yaml_with_extends(Path(path).resolve())
    ranges = {k: SeedRange(**v) for k, v in data["seeds"].items()}
    _check_seed_ranges_disjoint(ranges)
    spec = TaskSpec(
        train_candidates=list(data["train_candidates"]),
        heldout=list(data["heldout_tasks"]),
        seed_ranges=ranges,
        train_candidates_run2_extra=list(data.get("train_candidates_run2_extra") or []),
        run1_pool=list(data.get("run1_pool") or []),
        heldout_v2_candidates=list(data.get("heldout_v2_candidates") or []),
        train_candidates_run3_extra=list(data.get("train_candidates_run3_extra") or []),
        run2_pool=list(data.get("run2_pool") or []),
        heldout_v2=list(data.get("heldout_v2") or []),
        train_candidates_run4_new_action=list(data.get("train_candidates_run4_new_action") or []),
        train_candidates_run4_extra=list(data.get("train_candidates_run4_extra") or []),
        run3_pool=list(data.get("run3_pool") or []),
        run4_long_horizon=list(data.get("run4_long_horizon") or []),
        run4_turn_limited=list(data.get("run4_turn_limited") or []),
        run4_exclude=dict(data.get("run4_exclude") or {}),
        full_suite_exclude=dict(data.get("full_suite_exclude") or {}),
    )
    _check_heldout_disjoint(spec)
    return spec


def _check_heldout_disjoint(spec: TaskSpec) -> None:
    """Fail fast if a held-out task could ever be trained on."""
    train = set(spec.train_candidates_run3) | set(spec.train_candidates_run4)
    leaked = train & (set(spec.heldout) | set(spec.heldout_v2_candidates) | set(spec.heldout_v2))
    if leaked:
        raise ValueError(f"held-out tasks also listed as training candidates: {sorted(leaked)}; fix configs/tasks.yaml")


def _check_seed_ranges_disjoint(ranges: dict[str, SeedRange]) -> None:
    """Fail fast if training seeds could leak into evaluation seeds."""
    items = sorted(ranges.items(), key=lambda kv: kv[1].start)
    for (name_a, a), (name_b, b) in zip(items, items[1:]):
        if a.start + a.count > b.start:
            raise ValueError(f"seed ranges {name_a!r} and {name_b!r} overlap; fix configs/tasks.yaml")


def results_dir(cfg: Optional[Config] = None) -> Path:
    """Root for all outputs. On the pod the repo lives on the network volume,
    so `results/` survives pod termination."""
    root = os.environ.get("CUA_RESULTS_DIR") or (cfg.get("paths", {}).get("results_dir") if cfg else None) or "results"
    p = Path(root)
    return p if p.is_absolute() else REPO_ROOT / p


def selected_tasks_path(cfg: Optional[Config] = None) -> Path:
    return results_dir(cfg) / "selected_tasks.json"


def load_selected_tasks(cfg: Optional[Config] = None) -> list[str]:
    """Training pool chosen by `cua.select_tasks`. Raises if that step hasn't run."""
    path = selected_tasks_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m cua.select_tasks` after the sampled baseline."
        )
    return list(json.loads(path.read_text())["selected"])


def heldout_v2_path(cfg: Optional[Config] = None) -> Path:
    return results_dir(cfg) / "heldout_v2.json"


_BUDGET_CACHE: dict[str, dict[str, int]] = {}


def turn_budgets(cfg: Config) -> dict[str, int]:
    """Per-task turn limits from `env.turn_budgets` (run 4), or {} when unset.

    The value is either a mapping {task: turns} or the path of a JSON file
    (relative to the repository root) that holds such a mapping, or an object
    with a "budgets" key, as `cua.run_pool budgets` writes it. A path that
    doesn't exist raises, so a run never silently uses other limits than the
    ones its config names.
    """
    raw = (cfg.get("env") or {}).get("turn_budgets")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = str(path)
    if key not in _BUDGET_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"env.turn_budgets names {path}, which doesn't exist.")
        data = json.loads(path.read_text())
        data = data.get("budgets", data) if isinstance(data, dict) else {}
        _BUDGET_CACHE[key] = {str(k): int(v) for k, v in data.items()}
    return dict(_BUDGET_CACHE[key])


def turn_limit(cfg: Config, task: str) -> int:
    """The turn limit for one task: its budget from `env.turn_budgets`, capped at `env.max_turns`."""
    max_turns = int(cfg.env.max_turns)
    budget = turn_budgets(cfg).get(task, max_turns)
    return max(1, min(max_turns, int(budget)))


def resolve_task_list(names: list[str] | str, cfg: Optional[Config] = None) -> list[str]:
    """Expand the symbolic task-list names used in configs and CLIs.

    - "candidates": all training candidates from tasks.yaml
    - "candidates_run2": the candidates plus run 2's extra tasks
    - "candidates_run3": the run-2 candidates plus run 3's Track-B extras
    - "run2_pool": run 2's training pool (frozen in tasks.yaml)
    - "selected": the pool picked by select_tasks.py
    - "heldout": the held-out generalization tasks
    - "heldout_v2_candidates": tasks screened for the run-2 held-out extension
    - "heldout_v2": the held-out tasks plus the screened extension
      (results/heldout_v2.json, written by select_tasks.py --heldout-extra);
      without that file, the list frozen in tasks.yaml
    - "run1_pool": run 1's training pool
    - "candidates_run4": run 4's screening candidates
    - "run3_pool": run 3's training pool (frozen in tasks.yaml)
    - "run4_turn_limited", "run4_new_action": run 4's task groups
    - "full_suite": every task of run 4's full-suite eval
      (results/<run>/full_suite.json, written by `cua.run_pool registered`)
    - anything else: a literal comma-separated list or a YAML list of names
    """
    spec = load_tasks()
    if isinstance(names, str):
        if names == "candidates":
            return spec.train_candidates
        if names == "candidates_run2":
            return spec.train_candidates_run2
        if names == "candidates_run3":
            return spec.train_candidates_run3
        if names == "run2_pool":
            return spec.run2_pool
        if names == "selected":
            return load_selected_tasks(cfg)
        if names == "heldout":
            return spec.heldout
        if names == "heldout_v2_candidates":
            return spec.heldout_v2_candidates
        if names == "run1_pool":
            return spec.run1_pool
        if names == "candidates_run4":
            return spec.train_candidates_run4
        if names == "run3_pool":
            return spec.run3_pool
        if names == "run4_turn_limited":
            return spec.run4_turn_limited
        if names == "run4_new_action":
            return spec.train_candidates_run4_new_action
        if names == "full_suite":
            path = results_dir(cfg) / "full_suite.json"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run `python -m cua.run_pool registered --run run4` first.")
            return list(json.loads(path.read_text())["tasks"])
        if names == "heldout_v2":
            path = heldout_v2_path(cfg)
            if not path.exists() and spec.heldout_v2:
                return list(spec.heldout_v2)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run `python -m cua.select_tasks --heldout-extra 5` first.")
            return list(json.loads(path.read_text())["heldout_v2"])
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)
