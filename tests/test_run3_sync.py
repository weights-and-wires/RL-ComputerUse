"""Tests for the run-3 Hugging Face sync, restore, and final upload (cua/hfsync.py).

A fake Hub stands in for Hugging Face, so nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cua import hfsync as hs


class FakeHub:
    """In-memory stand-in for hfsync.Hub. `ops` records every write, in order."""

    def __init__(self, files: dict[str, bytes] | None = None):
        self.files: dict[str, bytes] = dict(files or {})
        self.ops: list[tuple[str, str]] = []

    def list_files(self, prefix: str) -> dict[str, int]:
        return {p: len(b) for p, b in self.files.items() if p.startswith(prefix.rstrip("/") + "/")}

    def upload_folder(self, local: Path, remote: str, message: str) -> None:
        self.ops.append(("folder", remote))
        for path in Path(local).rglob("*"):
            if path.is_file():
                self.files[f"{remote}/{path.relative_to(local).as_posix()}"] = path.read_bytes()

    def upload_files(self, files, message: str) -> None:
        for rpath, data in files:
            self.ops.append(("file", rpath))
            self.files[rpath] = Path(data).read_bytes() if isinstance(data, Path) else bytes(data)

    def download(self, remote_paths, dest: Path) -> None:
        for rpath in remote_paths:
            out = Path(dest) / rpath
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.files[rpath])


def layout(tmp_path: Path) -> hs.Layout:
    return hs.Layout(
        repo_root=tmp_path / "repo",
        results=tmp_path / "repo" / "results" / "run3",
        ckpt_dir=tmp_path / "ws" / "ckpt" / "grpo-4b-run3",
        rft_dir=tmp_path / "ws" / "ckpt" / "rft-4b-run3" / "final",
        run2_adapter_dir=tmp_path / "ws" / "inputs" / "run2-adapter",
        wandb_dir=tmp_path / "repo" / "wandb",
    )


def make_ckpt(ckpt_dir: Path, n: int, complete: bool = True, name: str | None = None) -> Path:
    d = ckpt_dir / (name or f"checkpoint-{n}")
    d.mkdir(parents=True)
    files = hs.CKPT_FILES if complete else hs.CKPT_FILES[:2]
    for f in files:
        (d / f).write_text(json.dumps({"step": n}) if f == "trainer_state.json" else f"{f}-{n}")
    return d


def make_adapter(d: Path, tag: str = "x") -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in hs.ADAPTER_FILES:
        (d / f).write_text(f"{f}-{tag}")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_checkpoint_to_sync_never_repeats():
    assert hs.checkpoint_to_sync([], None) is None
    assert hs.checkpoint_to_sync([10, 20], None) == 20
    assert hs.checkpoint_to_sync([10, 20], 20) is None
    assert hs.checkpoint_to_sync([10, 20], 30) is None
    assert hs.checkpoint_to_sync([10, 20, 30], 20) == 30


def test_complete_checkpoints_skip_tmp_and_partial(tmp_path):
    d = tmp_path / "ck"
    make_ckpt(d, 10)
    make_ckpt(d, 20, complete=False)
    make_ckpt(d, 30, name="checkpoint-30.tmp")
    (d / "final").mkdir()
    assert hs.complete_checkpoints(d) == [10]
    assert hs.complete_checkpoints(tmp_path / "missing") == []


def test_scan_and_changed_files_respect_exclusions(tmp_path):
    root = tmp_path / "r"
    write(root / "run.log", "a")
    write(root / ".phases/grpo.done", "t")
    write(root / ".phases/.pid", "1")
    write(root / ".phases/.current", "grpo")
    write(root / ".sync_state.json", "{}")
    write(root / "samples/step-000005.tmp/episode.json", "{}")
    write(root / "big.bin", "x" * 100)
    scan = hs.scan_files(root, max_bytes=50)
    assert sorted(scan) == [".phases/grpo.done", "run.log"]
    assert hs.changed_files(scan, {}) == [".phases/grpo.done", "run.log"]
    manifest = {k: list(v) for k, v in scan.items()}
    assert hs.changed_files(scan, manifest) == []
    write(root / "run.log", "ab")
    assert hs.changed_files(hs.scan_files(root, max_bytes=50), manifest) == ["run.log"]


def test_pick_run2_adapter_prefers_final_then_newest_checkpoint():
    p = hs.RUN2_REMOTE_CKPT
    listing = {f"{p}/checkpoint-25/adapter_config.json": 1, f"{p}/checkpoint-25/adapter_model.safetensors": 9,
               f"{p}/checkpoint-52/adapter_config.json": 1, f"{p}/checkpoint-52/adapter_model.safetensors": 9,
               f"{p}/checkpoint-60/adapter_config.json": 1}  # incomplete
    assert hs.pick_run2_adapter(listing) == f"{p}/checkpoint-52"
    listing[f"{p}/final/adapter_config.json"] = 1
    listing[f"{p}/final/adapter_model.safetensors"] = 9
    assert hs.pick_run2_adapter(listing) == f"{p}/final"
    assert hs.pick_run2_adapter({}) is None


# ---------------------------------------------------------------------------
# sync_once
# ---------------------------------------------------------------------------


def test_sync_uploads_each_checkpoint_once_and_only_changed_files(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub()
    make_ckpt(lay.ckpt_dir, 10)
    make_ckpt(lay.ckpt_dir, 20)
    write(lay.results / "run.log", "line 1\n")
    write(lay.results / ".phases/.pid", "123")
    state = hs.SyncState.load(lay.results)

    did = hs.sync_once(hub, lay, state)
    assert did["checkpoint"] == 20  # only the newest
    assert hs.read_synced(lay.results) == 20
    assert f"{lay.remote_ckpt}/checkpoint-20/optimizer.pt" in hub.files
    assert not any("checkpoint-10" in p for p in hub.files)
    assert "results/run3/run.log" in hub.files
    assert "results/run3/.synced_checkpoint" in hub.files
    assert "results/run3/.phases/.pid" not in hub.files
    assert "results/run3/.sync_state.json" not in hub.files

    n_ops = len(hub.ops)
    did = hs.sync_once(hub, lay, hs.SyncState.load(lay.results))  # a fresh process reads the state file
    assert did == {"checkpoint": None, "adapters": [], "files": 0}
    assert len(hub.ops) == n_ops

    write(lay.results / "run.log", "line 1\nline 2\n")
    make_ckpt(lay.ckpt_dir, 30)
    did = hs.sync_once(hub, lay, state)
    assert did["checkpoint"] == 30 and did["files"] == 2  # run.log and .synced_checkpoint
    assert hub.files["results/run3/run.log"] == b"line 1\nline 2\n"
    assert [op for op in hub.ops if op[0] == "folder"] == [
        ("folder", f"{lay.remote_ckpt}/checkpoint-20"), ("folder", f"{lay.remote_ckpt}/checkpoint-30")]


def test_final_adapter_goes_up_before_the_done_marker(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub()
    make_adapter(lay.ckpt_dir / "final")
    write(lay.results / ".phases/grpo.done", "2026-11-25T10:00:00Z")
    state = hs.SyncState()
    did = hs.sync_once(hub, lay, state)
    assert did["adapters"] == ["grpo-final"]
    order = [p for _, p in hub.ops]
    assert order.index(f"{lay.remote_ckpt}/final") < order.index("results/run3/.phases/grpo.done")
    # Not twice, and not at all on a pod whose state is fresh but whose Hub has it.
    hs.sync_once(hub, lay, state)
    fresh = hs.SyncState()
    n_ops = len(hub.ops)
    hs.sync_once(hub, lay, fresh)
    assert ("folder", f"{lay.remote_ckpt}/final") not in hub.ops[n_ops:]


def test_adapter_waits_for_its_marker(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub()
    make_adapter(lay.rft_dir)
    assert hs.sync_once(hub, lay, hs.SyncState())["adapters"] == []
    write(lay.results / ".phases/rft.done", "t")
    assert hs.sync_once(hub, lay, hs.SyncState())["adapters"] == ["rft-final"]


def test_sync_loop_exits_after_all_done(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub()
    write(lay.results / "run.log", "x\nALL DONE\n")
    sleeps = []
    hs.sync_loop(hub, lay, interval=300, poll=1, sleep=sleeps.append, max_ticks=5)
    assert sleeps == []  # synced once and returned
    assert "results/run3/run.log" in hub.files


def test_sync_loop_survives_hub_errors(tmp_path):
    lay = layout(tmp_path)
    write(lay.results / "run.log", "x\n")

    class Broken(FakeHub):
        def upload_files(self, files, message):
            raise ConnectionError("hub down")

    sleeps = []
    hs.sync_loop(Broken(), lay, interval=0, poll=1, sleep=sleeps.append, max_ticks=3)
    assert len(sleeps) == 3


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def run2_files() -> dict[str, bytes]:
    p = f"{hs.RUN2_REMOTE_CKPT}/checkpoint-52"
    return {f"{p}/adapter_config.json": b"{}", f"{p}/adapter_model.safetensors": b"weights",
            "results/run2/heldout_v2.json": b'{"heldout_v2": []}'}


def test_restore_with_nothing_synced_starts_fresh(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub(run2_files())
    did = hs.restore(hub, lay)
    assert did["run2_adapter"] == f"{hs.RUN2_REMOTE_CKPT}/checkpoint-52"
    assert (lay.run2_adapter_dir / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (lay.repo_root / "results/run2/heldout_v2.json").exists()
    assert "checkpoint" not in did and "results_files" not in did
    assert hs.complete_checkpoints(lay.ckpt_dir) == []
    assert hs.read_synced(lay.results) is None
    assert hub.ops == []  # restore never writes to the Hub
    assert hs.restore(hub, lay) == {}  # a second run has nothing to do


def test_restore_without_run2_adapter_fails(tmp_path):
    with pytest.raises(SystemExit):
        hs.restore(FakeHub(), layout(tmp_path))


def test_restore_brings_back_results_and_newest_checkpoint_then_sync_is_quiet(tmp_path):
    # Pod 1 trains, syncs, and dies.
    lay1 = layout(tmp_path / "pod1")
    hub = FakeHub(run2_files())
    for name in ("smoke", "screen", "select", "rft", "pre_eval"):
        write(lay1.results / f".phases/{name}.done", "t")
    write(lay1.results / "train/grpo-4b-run3/metrics.jsonl", '{"step": 1}\n')
    write(lay1.results / "wandb_run.json", '{"id": "abc"}')
    make_ckpt(lay1.ckpt_dir, 10)
    hs.sync_once(hub, lay1, hs.SyncState())
    make_ckpt(lay1.ckpt_dir, 20)
    hs.sync_once(hub, lay1, hs.SyncState.load(lay1.results))
    hub.files[f"{lay1.remote_ckpt}/checkpoint-30/trainer_state.json"] = b"{}"  # a half upload: ignored

    # Pod 2 starts empty.
    lay2 = layout(tmp_path / "pod2")
    did = hs.restore(hub, lay2)
    assert did["checkpoint"] == 20
    assert hs.complete_checkpoints(lay2.ckpt_dir) == [20]
    assert not (lay2.ckpt_dir / "checkpoint-20.tmp").exists()
    assert hs.read_synced(lay2.results) == 20
    assert (lay2.results / ".phases/pre_eval.done").exists()
    assert json.loads((lay2.results / "wandb_run.json").read_text())["id"] == "abc"

    n_ops = len(hub.ops)
    did = hs.sync_once(hub, lay2, hs.SyncState.load(lay2.results))
    assert did["checkpoint"] is None and did["files"] == 0
    assert len(hub.ops) == n_ops

    # Local progress wins: a second restore changes nothing.
    write(lay2.results / "run.log", "local\n")
    assert hs.restore(hub, lay2) == {}
    assert (lay2.results / "run.log").read_text() == "local\n"


def test_restore_fetches_final_adapter_when_grpo_is_done(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub(run2_files())
    hub.files["results/run3/.phases/grpo.done"] = b"t"
    for f in hs.ADAPTER_FILES:
        hub.files[f"{lay.remote_ckpt}/final/{f}"] = b"final"
    did = hs.restore(hub, lay)
    assert did["adapters"] == [f"{lay.remote_ckpt}/final"]
    assert hs.has_adapter(lay.ckpt_dir / "final")


# ---------------------------------------------------------------------------
# Final upload and verify
# ---------------------------------------------------------------------------


def test_final_upload_covers_everything_and_verifies(tmp_path):
    lay = layout(tmp_path)
    hub = FakeHub()
    make_ckpt(lay.ckpt_dir, 10)
    make_ckpt(lay.ckpt_dir, 20)
    make_adapter(lay.ckpt_dir / "final")
    make_adapter(lay.rft_dir)
    write(lay.results / "run.log", "ALL DONE\n")
    write(lay.results / "eval/run3-pool3/episodes.jsonl", "{}\n")
    write(lay.results / "big/blob.bin", "x" * 10)
    write(lay.results / "wandb_run.json", json.dumps({"id": "r1"}))
    write(lay.wandb_dir / "run-20261125_010101-r1/files/output.log", "hello")
    write(lay.wandb_dir / "run-20261125_090909-other/files/output.log", "not ours")
    os.symlink(lay.wandb_dir / "run-20261125_010101-r1", lay.wandb_dir / "latest-run")
    hs.sync_once(hub, lay, hs.SyncState())  # checkpoint-20 already synced

    assert hs.upload_final(hub, lay, do_verify=True)
    remote = set(hub.files)
    assert f"{lay.remote_ckpt}/checkpoint-10/optimizer.pt" in remote
    assert f"{lay.remote_ckpt}/final/adapter_model.safetensors" in remote
    assert f"{lay.remote_rft}/adapter_model.safetensors" in remote
    assert "results/run3/eval/run3-pool3/episodes.jsonl" in remote
    assert "results/run3/wandb/run-20261125_010101-r1/files/output.log" in remote
    assert not any("other" in p or "latest-run" in p for p in remote)
    # checkpoint-20 was already there, so the final upload didn't send it again.
    assert ("file", f"{lay.remote_ckpt}/checkpoint-20/optimizer.pt") not in hub.ops


def test_verify_reports_missing_and_size_mismatch(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("12345")
    b.write_text("12")
    missing, mismatched = hs.verify({"r/a.txt": a, "r/b.txt": b, "r/c.txt": a}, {"r/a.txt": 5, "r/b.txt": 3})
    assert missing == ["r/c.txt"] and mismatched == ["r/b.txt"]


def test_upload_final_fails_when_the_hub_drops_files(tmp_path):
    lay = layout(tmp_path)
    write(lay.results / "run.log", "x")

    class Lossy(FakeHub):
        def upload_files(self, files, message):
            pass  # accepts the call, stores nothing

    assert hs.upload_final(Lossy(), lay, do_verify=True, attempts=2) is False
