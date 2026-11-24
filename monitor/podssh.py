"""Run commands on a pod over SSH. Secrets go through stdin, never argv."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_REPO = "/root/RL-ComputerUse"

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=4",
]


@dataclass
class Result:
    rc: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    @property
    def ssh_failed(self) -> bool:
        # ssh exits 255 on connection errors; -1 is our timeout marker.
        return self.rc in (255, -1)


class SSHPod:
    def __init__(self, host: str, port: int, key: str | os.PathLike | None = None, user: str = "root"):
        self.host = host
        self.port = int(port)
        self.key = str(key or Path.home() / ".ssh" / "id_ed25519")
        self.user = user

    def argv(self, remote_cmd: str) -> list[str]:
        return ["ssh", *SSH_OPTS, "-i", self.key, "-p", str(self.port), f"{self.user}@{self.host}", remote_cmd]

    def run(self, cmd: str, stdin: bytes | None = None, timeout: float = 120, in_repo: bool = True,
            binary_out: bool = False) -> Result:
        remote = f"cd {REMOTE_REPO} && {cmd}" if in_repo else cmd
        try:
            p = subprocess.run(self.argv(remote), input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result(-1, "", f"timed out after {timeout:.0f} s")
        except OSError as e:
            return Result(-1, "", f"ssh failed to start: {e}")
        out = p.stdout if binary_out else p.stdout.decode("utf-8", errors="replace")
        return Result(p.returncode, out, p.stderr.decode("utf-8", errors="replace")[-4000:])

    def pipe_from(self, local_argv: list[str], remote_cmd: str, timeout: float = 900, in_repo: bool = False) -> Result:
        """Run local_argv and stream its stdout into remote_cmd on the pod (for tar)."""
        remote = f"cd {REMOTE_REPO} && {remote_cmd}" if in_repo else remote_cmd
        try:
            src = subprocess.Popen(local_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dst = subprocess.run(self.argv(remote), stdin=src.stdout, capture_output=True, timeout=timeout)
            src.stdout.close()
            src_rc = src.wait(timeout=60)
            err = dst.stderr.decode(errors="replace") + src.stderr.read().decode(errors="replace")
            rc = dst.returncode if dst.returncode != 0 else src_rc
            return Result(rc, dst.stdout.decode(errors="replace"), err[-4000:])
        except subprocess.TimeoutExpired:
            return Result(-1, "", f"timed out after {timeout:.0f} s")
        except OSError as e:
            return Result(-1, "", str(e))

    def pipe_to(self, remote_cmd: str, local_argv: list[str], timeout: float = 600, in_repo: bool = True) -> Result:
        """Stream remote_cmd's stdout into local_argv (for pulling a tar)."""
        remote = f"cd {REMOTE_REPO} && {remote_cmd}" if in_repo else remote_cmd
        try:
            src = subprocess.Popen(self.argv(remote), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dst = subprocess.run(local_argv, stdin=src.stdout, capture_output=True, timeout=timeout)
            src.stdout.close()
            src_rc = src.wait(timeout=60)
            err = src.stderr.read().decode(errors="replace") + dst.stderr.decode(errors="replace")
            rc = src_rc if src_rc != 0 else dst.returncode
            return Result(rc, dst.stdout.decode(errors="replace"), err[-4000:])
        except subprocess.TimeoutExpired:
            return Result(-1, "", f"timed out after {timeout:.0f} s")
        except OSError as e:
            return Result(-1, "", str(e))
