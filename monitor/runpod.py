"""Minimal Runpod API client (REST v2 at https://api.runpod.io/v2), standard library only.

Runpod retired REST v1 (rest.runpod.io no longer resolves), so this client uses
v2. `normalize_pod` also accepts v1-shaped objects, which keeps the code working
if a field moves back.

The API key travels only in an HTTP header. Nothing here logs it, and pod
objects pass through `strip_env` before anything prints them, because a pod
echoes its env block.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"

GONE_STATUSES = {"TERMINATED", "EXITED", "ERROR", "DEAD"}


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def strip_env(obj):
    if isinstance(obj, dict):
        return {k: ("<env hidden>" if k == "env" else strip_env(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_env(x) for x in obj]
    return obj


def normalize_pod(p: dict) -> dict:
    """Pull the fields the monitor needs out of a v2 (or v1) pod object."""
    if not isinstance(p, dict):
        return {}
    status = p.get("status") or p.get("desiredStatus")
    host = port = None
    ssh = p.get("ssh") or {}
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    if isinstance(direct, dict) and direct.get("host") and direct.get("port"):
        host, port = direct["host"], int(direct["port"])
    runtime = p.get("runtime") or {}
    if not host and isinstance(runtime, dict):
        for m in runtime.get("ports") or []:
            if int(m.get("private") or m.get("privatePort") or 0) == 22 and (m.get("public") or m.get("publicPort")):
                host = m.get("ip") or p.get("publicIp")
                port = int(m.get("public") or m.get("publicPort"))
                break
    if not host and p.get("publicIp") and isinstance(p.get("portMappings"), dict):
        pm = p["portMappings"].get("22")
        if pm:
            host, port = p["publicIp"], int(pm)
    cost = p.get("cost")
    if cost is None:
        cost = p.get("adjustedCostPerHr", p.get("costPerHr"))
    gpu = p.get("gpu") or {}
    gpu_id = gpu.get("id") if isinstance(gpu, dict) else None
    if not gpu_id:
        machine = p.get("machine") or {}
        gpu_id = (machine.get("gpuType") or {}).get("id") if isinstance(machine, dict) else None
    uptime = runtime.get("uptime") if isinstance(runtime, dict) else None
    if uptime is None and isinstance(runtime, dict):
        uptime = runtime.get("uptimeInSeconds")
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "status": status,
        "ssh_host": host,
        "ssh_port": port,
        "cost_per_hr": float(cost) if cost is not None else None,
        "gpu": gpu_id,
        "cloud": p.get("cloud") or p.get("cloudType"),
        "created_at": p.get("createdAt"),
        "started_at": p.get("startedAt") or p.get("lastStartedAt"),
        "uptime_s": uptime,
    }


class RunpodAPI:
    def __init__(self, api_key: str, base: str = API, timeout: float = 60):
        self._key = api_key
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, body=None, url: str | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url or (self.base + path), data=data, method=method,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "User-Agent": "cua-run-monitor/1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw.strip() else None)
        except urllib.error.HTTPError as e:
            raw = e.read() or b""
            try:
                j = json.loads(raw)
                detail = j.get("detail") or j.get("message") or j.get("error") or raw.decode()[:300]
                if j.get("errors"):
                    detail = f"{detail} {j['errors']}"
            except ValueError:
                detail = raw.decode(errors="replace")[:300]
            raise ApiError(e.code, str(detail)[:500]) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ApiError(0, f"network error: {e}") from None

    # --- pods -------------------------------------------------------------------

    def get_pod(self, pod_id: str) -> dict | None:
        """Return the raw pod, or None when the API says 404."""
        try:
            _, j = self._req("GET", f"/pods/{pod_id}")
            return j
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def list_pods(self) -> list[dict]:
        _, j = self._req("GET", "/pods")
        if isinstance(j, list):
            return j
        if isinstance(j, dict):
            return j.get("pods") or j.get("items") or []
        return []

    def create_pod(self, body: dict) -> dict:
        _, j = self._req("POST", "/pods", body)
        return j or {}

    def delete_pod(self, pod_id: str) -> int:
        try:
            status, _ = self._req("DELETE", f"/pods/{pod_id}")
            return status
        except ApiError as e:
            if e.status == 404:
                return 404
            raise

    # --- prices (GraphQL; read-only, best effort) ------------------------------------

    def gpu_prices(self) -> dict[str, dict]:
        q = {"query": "query { gpuTypes { id displayName memoryInGb securePrice communityPrice } }"}
        try:
            _, j = self._req("POST", "", q, url=GRAPHQL)
            return {g["id"]: g for g in (j or {}).get("data", {}).get("gpuTypes", []) or []}
        except (ApiError, KeyError, TypeError, AttributeError):
            return {}

    def client_balance(self) -> float | None:
        """The account's credit balance in dollars, or None when the query fails.

        Uses the GraphQL query `myself { clientBalance }`. The key goes only in
        the request header; nothing here prints it.
        """
        q = {"query": "query { myself { clientBalance } }"}
        try:
            _, j = self._req("POST", "", q, url=GRAPHQL)
            v = ((j or {}).get("data") or {}).get("myself", {}).get("clientBalance")
            return float(v) if v is not None else None
        except (ApiError, KeyError, TypeError, ValueError, AttributeError):
            return None
