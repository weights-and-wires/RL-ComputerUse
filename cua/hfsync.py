"""Run 3's link to Hugging Face: the sync loop, restore on a new pod, and the final upload.

Command line (on the pod, with HF_TOKEN and HF_ADAPTER_REPO set):

    python -m cua.hfsync loop        # runs in tmux session run3-sync
    python -m cua.hfsync once        # one sync pass
    python -m cua.hfsync restore     # on a new pod, before run3.sh starts
    python -m cua.hfsync final [--verify]   # what scripts/upload_run3.py calls

Remote layout in HF_ADAPTER_REPO:

    checkpoints/grpo-4b-run3/checkpoint-N/   GRPO checkpoints (adapter, optimizer, RNG, sampler)
    checkpoints/grpo-4b-run3/final/          the final adapter
    checkpoints/rft-4b-run3/final/           the RFT adapter, only if the rft phase ran
    results/run3/                            markers, logs, metrics, evals, samples, W&B files

The sync loop never blocks training: it runs in its own process, reads only
finished files (checkpoints appear by an atomic rename), and catches every
error. It uploads:

1. The newest finished checkpoint, if it's newer than `.synced_checkpoint`.
   So a checkpoint is never uploaded twice, even across pods, because
   `.synced_checkpoint` is synced and restored too.
2. The final and RFT adapters, once their phase is done.
3. Every small file under results/run3 whose size or modification time changed.

Adapters go up before the markers, so a `grpo.done` on the Hub implies the
final adapter is there too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# A checkpoint counts as finished only with all of these. save_checkpoint
# writes them into checkpoint-N.tmp and renames the folder at the end.
CKPT_FILES = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng_state.pt")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

# Never synced: per-pod process state and the sync bookkeeping itself.
EXCLUDE = {".phases/.pid", ".phases/.current", ".sync_state.json"}
SMALL_FILE_MAX = 50 * 2**20  # bigger files wait for the final upload

# Run-2 inputs that run 3 needs. results/ isn't in git, so a fresh pod gets them from the Hub.
RUN2_REMOTE_CKPT = "checkpoints/grpo-4b-run2"
RUN2_RESULT_FILES = (
    "results/run2/heldout_v2.json",
    "results/run2/selected_tasks.json",
    "results/run2/eval/run2-pool2/episodes.jsonl",
    "results/run2/eval/run2-pool2/summary.json",
    "results/run2/eval/run2-pool2/per_task.csv",
)


def log(msg: str) -> None:
    print(f"[hfsync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Layout:
    """Where run 3 keeps things, locally and on the Hub."""

    repo_root: Path = REPO_ROOT
    results: Path = REPO_ROOT / "results" / "run3"
    ckpt_dir: Path = Path("/workspace/ckpt/grpo-4b-run3")
    rft_dir: Path = Path("/workspace/ckpt/rft-4b-run3/final")
    run2_adapter_dir: Path = Path("/workspace/inputs/run2-adapter")
    wandb_dir: Path = REPO_ROOT / "wandb"
    remote_ckpt: str = "checkpoints/grpo-4b-run3"
    remote_rft: str = "checkpoints/rft-4b-run3/final"
    remote_results: str = "results/run3"

    @classmethod
    def from_env(cls) -> "Layout":
        lay = cls()
        env = os.environ
        if env.get("RUN3_RESULTS"):
            lay.results = Path(env["RUN3_RESULTS"])
        if env.get("RUN3_CKPT"):
            lay.ckpt_dir = Path(env["RUN3_CKPT"])
        if env.get("RUN2_ADAPTER"):
            lay.run2_adapter_dir = Path(env["RUN2_ADAPTER"])
        return lay


# ---------------------------------------------------------------------------
# Local bookkeeping (pure, tested on a laptop)
# ---------------------------------------------------------------------------


def complete_checkpoints(ckpt_dir: Path) -> list[int]:
    """Sorted steps N of finished `checkpoint-N` folders."""
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if re.fullmatch(r"checkpoint-\d+", name) and all((ckpt_dir / name / f).is_file() for f in CKPT_FILES):
            out.append(int(name.split("-")[1]))
    return sorted(out)


def has_adapter(path: Path) -> bool:
    return all((Path(path) / f).is_file() for f in ADAPTER_FILES)


def read_synced(results: Path) -> Optional[int]:
    try:
        text = (Path(results) / ".synced_checkpoint").read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def write_synced(results: Path, step: int) -> None:
    path = Path(results) / ".synced_checkpoint"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"{int(step)}\n")
    os.replace(tmp, path)


def checkpoint_to_sync(local: list[int], synced: Optional[int]) -> Optional[int]:
    """The checkpoint to upload now: the newest local one, if it's newer than the synced one."""
    if not local:
        return None
    newest = max(local)
    return newest if synced is None or newest > synced else None


def excluded(rel: str) -> bool:
    parts = rel.split("/")
    return (
        rel in EXCLUDE
        or any(p.endswith(".tmp") or p == "__pycache__" for p in parts)
    )


def scan_files(root: Path, max_bytes: Optional[int] = SMALL_FILE_MAX) -> dict[str, tuple[int, int]]:
    """{relative path: (size, mtime_ns)} for the files under root that sync may upload."""
    out: dict[str, tuple[int, int]] = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith(".tmp") and d != "__pycache__"]
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            rel = path.relative_to(root).as_posix()
            if excluded(rel):
                continue
            try:
                st = path.stat()
            except OSError:
                continue  # deleted while we walked
            if max_bytes is not None and st.st_size > max_bytes:
                continue
            out[rel] = (st.st_size, st.st_mtime_ns)
    return out


def changed_files(scan: dict[str, tuple[int, int]], manifest: dict[str, Any]) -> list[str]:
    """Paths whose (size, mtime) differ from what the last sync uploaded."""
    return sorted(rel for rel, sig in scan.items() if tuple(manifest.get(rel, ())) != tuple(sig))


def done_markers(results: Path) -> frozenset[str]:
    try:
        return frozenset(n for n in os.listdir(Path(results) / ".phases") if n.endswith(".done"))
    except OSError:
        return frozenset()


@dataclass
class SyncState:
    """What this pod has uploaded. Lives in results/run3/.sync_state.json, never synced."""

    manifest: dict[str, list[int]] = field(default_factory=dict)
    adapters: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, results: Path) -> "SyncState":
        try:
            data = json.loads((Path(results) / ".sync_state.json").read_text())
            return cls(manifest=dict(data.get("manifest", {})), adapters=list(data.get("adapters", [])))
        except (OSError, ValueError):
            return cls()

    def save(self, results: Path) -> None:
        path = Path(results) / ".sync_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"manifest": self.manifest, "adapters": self.adapters}))
        os.replace(tmp, path)


def group_remote_checkpoints(remote: dict[str, int], prefix: str) -> list[int]:
    """Sorted steps N of finished checkpoints in a remote listing {path: size}."""
    files: dict[int, set[str]] = {}
    for path in remote:
        m = re.fullmatch(re.escape(prefix) + r"/checkpoint-(\d+)/(.+)", path)
        if m:
            files.setdefault(int(m.group(1)), set()).add(m.group(2))
    return sorted(n for n, names in files.items() if all(f in names for f in CKPT_FILES))


def pick_run2_adapter(remote: dict[str, int], prefix: str = RUN2_REMOTE_CKPT) -> Optional[str]:
    """Remote folder of the run-2 adapter: `final` if it exists, else the newest checkpoint with an adapter.

    Run 2 stopped at its time cap, so it has no `final`; its adapter is
    `checkpoint-52`, the one its post-training evals used.
    """
    def has(folder: str) -> bool:
        return all(f"{folder}/{f}" in remote for f in ADAPTER_FILES)

    if has(f"{prefix}/final"):
        return f"{prefix}/final"
    steps = sorted({int(m.group(1)) for p in remote
                    if (m := re.fullmatch(re.escape(prefix) + r"/checkpoint-(\d+)/.+", p))}, reverse=True)
    for n in steps:
        if has(f"{prefix}/checkpoint-{n}"):
            return f"{prefix}/checkpoint-{n}"
    return None


def verify(local: dict[str, Path], remote: dict[str, int]) -> tuple[list[str], list[str]]:
    """Returns (missing, size mismatches): remote paths in `local` that aren't on the Hub as they are here."""
    missing, mismatched = [], []
    for rpath, lpath in sorted(local.items()):
        if rpath not in remote:
            missing.append(rpath)
        elif remote[rpath] != Path(lpath).stat().st_size:
            mismatched.append(rpath)
    return missing, mismatched


def wandb_run_dirs(wandb_dir: Path, run_id: Optional[str]) -> list[Path]:
    """Local W&B folders of the GRPO run (one per start, all ending in -<id>)."""
    if not run_id or not Path(wandb_dir).is_dir():
        return []
    return sorted(p for p in Path(wandb_dir).iterdir()
                  if p.is_dir() and not p.is_symlink() and p.name.endswith(f"-{run_id}"))


def final_upload_map(lay: Layout) -> dict[str, Path]:
    """{remote path: local file} for everything the final upload must put on the Hub."""
    out: dict[str, Path] = {}

    def add_tree(local_root: Path, remote_root: str, max_bytes: Optional[int] = None) -> None:
        for rel in scan_files(local_root, max_bytes=max_bytes):
            out[f"{remote_root}/{rel}"] = Path(local_root) / rel

    for n in complete_checkpoints(lay.ckpt_dir):
        add_tree(lay.ckpt_dir / f"checkpoint-{n}", f"{lay.remote_ckpt}/checkpoint-{n}")
    if has_adapter(lay.ckpt_dir / "final"):
        add_tree(lay.ckpt_dir / "final", f"{lay.remote_ckpt}/final")
    if has_adapter(lay.rft_dir):
        add_tree(lay.rft_dir, lay.remote_rft)
    add_tree(lay.results, lay.remote_results)
    run_id = None
    try:
        run_id = json.loads((lay.results / "wandb_run.json").read_text()).get("id")
    except (OSError, ValueError):
        pass
    for d in wandb_run_dirs(lay.wandb_dir, run_id):
        add_tree(d, f"{lay.remote_results}/wandb/{d.name}")
    return out


# ---------------------------------------------------------------------------
# The Hub (a thin wrapper, so tests can pass a fake with the same methods)
# ---------------------------------------------------------------------------


class Hub:
    """The few Hugging Face operations run 3 needs, on one model repository."""

    def __init__(self, repo_id: str, token: Optional[str] = None):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)

    def list_files(self, prefix: str) -> dict[str, int]:
        """{path: size} of every file under prefix; empty when prefix doesn't exist."""
        from huggingface_hub.hf_api import RepoFile

        try:
            items = self.api.list_repo_tree(self.repo_id, path_in_repo=prefix, recursive=True)
            return {it.path: int(it.size) for it in items if isinstance(it, RepoFile)}
        except Exception as exc:  # noqa: BLE001 - the error class moved between huggingface_hub versions
            if "404" in str(exc) or "NotFound" in type(exc).__name__:
                return {}
            raise

    def upload_folder(self, local: Path, remote: str, message: str) -> None:
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(local), path_in_repo=remote,
                               commit_message=message)

    def upload_files(self, files: list[tuple[str, Any]], message: str, chunk: int = 200) -> None:
        """Uploads (remote path, local path or bytes) pairs, `chunk` files per commit."""
        from huggingface_hub import CommitOperationAdd

        for i in range(0, len(files), chunk):
            ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=(str(d) if isinstance(d, Path) else d))
                   for r, d in files[i : i + chunk]]
            self.api.create_commit(repo_id=self.repo_id, operations=ops, commit_message=message)

    def download(self, remote_paths: Iterable[str], dest: Path) -> None:
        """Downloads files to dest/<remote path>."""
        from huggingface_hub import hf_hub_download

        for rpath in remote_paths:
            hf_hub_download(self.repo_id, rpath, local_dir=str(dest), token=self.token)


def hub_from_env() -> Hub:
    repo = os.environ.get("HF_ADAPTER_REPO")
    if not repo or not os.environ.get("HF_TOKEN"):
        raise SystemExit("Set HF_TOKEN and HF_ADAPTER_REPO (source /workspace/env.sh).")
    return Hub(repo, os.environ["HF_TOKEN"])


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def sync_once(hub: Any, lay: Layout, state: SyncState) -> dict[str, Any]:
    """One pass: newest checkpoint, finished adapters, then changed small files."""
    did: dict[str, Any] = {"checkpoint": None, "adapters": [], "files": 0}

    n = checkpoint_to_sync(complete_checkpoints(lay.ckpt_dir), read_synced(lay.results))
    if n is not None:
        log(f"uploading checkpoint-{n}")
        hub.upload_folder(lay.ckpt_dir / f"checkpoint-{n}", f"{lay.remote_ckpt}/checkpoint-{n}",
                          f"run3: checkpoint-{n}")
        write_synced(lay.results, n)
        did["checkpoint"] = n

    markers = done_markers(lay.results)
    adapters = [("grpo-final", "grpo.done", lay.ckpt_dir / "final", f"{lay.remote_ckpt}/final"),
                ("rft-final", "rft.done", lay.rft_dir, lay.remote_rft)]
    for name, marker, local, remote in adapters:
        if name in state.adapters or marker not in markers or not has_adapter(local):
            continue
        on_hub = hub.list_files(remote)
        if not all(f"{remote}/{f}" in on_hub for f in ADAPTER_FILES):
            log(f"uploading {name}")
            hub.upload_folder(local, remote, f"run3: {name}")
        state.adapters.append(name)
        did["adapters"].append(name)

    scan = scan_files(lay.results)
    changed = changed_files(scan, state.manifest)
    if changed:
        payload = []
        for rel in changed:
            try:
                payload.append((f"{lay.remote_results}/{rel}", (lay.results / rel).read_bytes()))
            except OSError:
                continue  # removed since the scan
        if payload:
            hub.upload_files(payload, f"run3: sync {len(payload)} files")
            for rel in changed:
                state.manifest[rel] = list(scan[rel])
            did["files"] = len(payload)
    state.save(lay.results)
    return did


def all_done(results: Path) -> bool:
    from cua.run3_status import last_line

    return last_line(Path(results) / "run.log") == "ALL DONE"


def sync_loop(hub: Any, lay: Layout, interval: float = 300.0, poll: float = 20.0,
              sleep: Callable[[float], None] = time.sleep, max_ticks: Optional[int] = None) -> None:
    """Syncs every `interval` seconds, and right away after a phase finishes or a checkpoint appears.

    Ends after one last pass once run.log ends with ALL DONE. Errors are
    logged and retried at the next poll; they never end the loop.
    """
    state = SyncState.load(lay.results)
    last_t = -1e18
    last_markers: Optional[frozenset] = None
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        markers = done_markers(lay.results)
        new_ckpt = checkpoint_to_sync(complete_checkpoints(lay.ckpt_dir), read_synced(lay.results)) is not None
        finished = all_done(lay.results)
        due = time.monotonic() - last_t >= interval
        if due or new_ckpt or markers != last_markers or finished:
            try:
                did = sync_once(hub, lay, state)
                last_t = time.monotonic()
                last_markers = markers
                if did["checkpoint"] is not None or did["adapters"] or did["files"]:
                    log(f"synced: {did}")
                if finished:
                    log("run.log ends with ALL DONE; last sync done, exiting")
                    return
            except Exception as exc:  # noqa: BLE001 - keep syncing; the next pass retries
                log(f"sync failed ({type(exc).__name__}: {exc}); retrying")
        sleep(poll)


# ---------------------------------------------------------------------------
# Restore on a new pod
# ---------------------------------------------------------------------------


def _download_tree(hub: Any, remote_files: Iterable[str], remote_root: str, local_root: Path,
                   staging: Path) -> list[str]:
    """Downloads remote files under remote_root into local_root; returns the relative paths."""
    files = [p for p in remote_files if p.startswith(remote_root + "/")]
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    hub.download(files, staging)
    rels = []
    for rpath in files:
        rel = rpath[len(remote_root) + 1 :]
        dst = Path(local_root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / rpath), str(dst))  # staging can be on another filesystem
        rels.append(rel)
    shutil.rmtree(staging, ignore_errors=True)
    return rels


def restore(hub: Any, lay: Layout) -> dict[str, Any]:
    """Brings a new pod to where the last synced state left off. Safe to run again.

    - run-2 inputs: the adapter (required) and run 2's result files (optional);
    - results/run3/, only when it has no `.phases` folder here;
    - the newest synced GRPO checkpoint, only when there is no local one;
    - the final and RFT adapters, when their phases are done.

    With nothing on the Hub yet, it restores nothing for run 3, and GRPO starts
    fresh from the run-2 adapter.
    """
    staging_root = Path(lay.ckpt_dir).parent / ".restore-tmp"
    did: dict[str, Any] = {}

    # Run-2 adapter.
    if not has_adapter(lay.run2_adapter_dir):
        remote2 = hub.list_files(RUN2_REMOTE_CKPT)
        folder = os.environ.get("RUN2_HF_PATH") or pick_run2_adapter(remote2)
        if not folder:
            raise SystemExit(f"No run-2 adapter under {RUN2_REMOTE_CKPT} on the Hub. Set RUN2_HF_PATH.")
        log(f"downloading the run-2 adapter from {folder}")
        _download_tree(hub, [f"{folder}/{f}" for f in ADAPTER_FILES], folder, lay.run2_adapter_dir,
                       staging_root / "run2")
        did["run2_adapter"] = folder
    # Run-2 result files: optional inputs with fallbacks in run3.sh.
    remote_r2 = None
    for rpath in RUN2_RESULT_FILES:
        local = lay.repo_root / rpath
        if local.exists():
            continue
        if remote_r2 is None:
            remote_r2 = hub.list_files("results/run2")
        if rpath in remote_r2:
            _download_tree(hub, [rpath], "results/run2", lay.repo_root / "results" / "run2", staging_root / "r2")
            did.setdefault("run2_results", []).append(rpath)
        else:
            log(f"note: {rpath} isn't on the Hub; run3.sh falls back")

    # results/run3.
    if not (lay.results / ".phases").is_dir():
        remote = hub.list_files(lay.remote_results)
        files = [p for p in remote if not excluded(p[len(lay.remote_results) + 1 :])]
        if files:
            log(f"restoring {len(files)} files of {lay.remote_results}")
            rels = _download_tree(hub, files, lay.remote_results, lay.results, staging_root / "results")
            # Record them as synced, so the loop doesn't upload them straight back.
            state = SyncState.load(lay.results)
            for rel, sig in scan_files(lay.results).items():
                if rel in rels:
                    state.manifest[rel] = list(sig)
            state.save(lay.results)
            did["results_files"] = len(rels)
        else:
            log(f"nothing under {lay.remote_results} on the Hub yet; starting run 3 from scratch")

    # The newest GRPO checkpoint.
    remote_ck = None
    if not complete_checkpoints(lay.ckpt_dir):
        remote_ck = hub.list_files(lay.remote_ckpt)
        steps = group_remote_checkpoints(remote_ck, lay.remote_ckpt)
        if steps:
            n = steps[-1]
            folder = f"{lay.remote_ckpt}/checkpoint-{n}"
            log(f"restoring {folder}")
            tmp_dir = Path(lay.ckpt_dir) / f"checkpoint-{n}.tmp"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _download_tree(hub, [p for p in remote_ck if p.startswith(folder + "/")], folder, tmp_dir,
                           staging_root / "ckpt")
            os.replace(tmp_dir, Path(lay.ckpt_dir) / f"checkpoint-{n}")
            synced = read_synced(lay.results)
            if synced is None or n > synced:
                write_synced(lay.results, n)
            did["checkpoint"] = n
        else:
            log("no synced GRPO checkpoint on the Hub; GRPO starts from its start adapter")

    # Final and RFT adapters, when their phase is done.
    markers = done_markers(lay.results)
    for marker, local, remote in (("grpo.done", lay.ckpt_dir / "final", f"{lay.remote_ckpt}/final"),
                                  ("rft.done", lay.rft_dir, lay.remote_rft)):
        if marker in markers and not has_adapter(local):
            files = hub.list_files(remote)
            if files:
                log(f"restoring {remote}")
                _download_tree(hub, list(files), remote, local, staging_root / "adapter")
                did.setdefault("adapters", []).append(remote)
    return did


# ---------------------------------------------------------------------------
# Final upload
# ---------------------------------------------------------------------------


def remote_listing(hub: Any, lay: Layout) -> dict[str, int]:
    out: dict[str, int] = {}
    for prefix in (lay.remote_ckpt, lay.remote_rft, lay.remote_results):
        out.update(hub.list_files(prefix))
    return out


def upload_final(hub: Any, lay: Layout, do_verify: bool = True, attempts: int = 3) -> bool:
    """Uploads every file that isn't on the Hub with the same size, then verifies. True when verified."""
    for attempt in range(1, attempts + 1):
        local = final_upload_map(lay)
        missing, mismatched = verify(local, remote_listing(hub, lay))
        todo = missing + mismatched
        log(f"attempt {attempt}: {len(local)} local files, {len(todo)} to upload")
        if todo:
            hub.upload_files([(r, local[r]) for r in todo], "run3: final upload")
        if not do_verify:
            return True
        missing, mismatched = verify(final_upload_map(lay), remote_listing(hub, lay))
        if not missing and not mismatched:
            log(f"verified: all {len(local)} files are on the Hub with matching sizes")
            return True
        log(f"not verified yet: {len(missing)} missing, {len(mismatched)} with another size, "
            f"for example {(missing + mismatched)[:3]}")
    return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync run 3 with Hugging Face.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop", help="Sync until run.log ends with ALL DONE.")
    lp.add_argument("--interval", type=float, default=float(os.environ.get("SYNC_INTERVAL_S", "300")))
    lp.add_argument("--poll", type=float, default=20.0)
    sub.add_parser("once", help="One sync pass.")
    sub.add_parser("restore", help="Restore run 3 on a new pod.")
    fp = sub.add_parser("final", help="Upload everything, optionally verify.")
    fp.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    lay = Layout.from_env()
    hub = hub_from_env()
    if args.cmd == "loop":
        sync_loop(hub, lay, interval=args.interval, poll=args.poll)
    elif args.cmd == "once":
        log(f"synced: {sync_once(hub, lay, SyncState.load(lay.results))}")
    elif args.cmd == "restore":
        log(f"restored: {restore(hub, lay)}")
    elif args.cmd == "final":
        return 0 if upload_final(hub, lay, do_verify=args.verify) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
