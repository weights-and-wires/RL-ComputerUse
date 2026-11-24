#!/usr/bin/env python3
"""Read-only web server for the run 3 live dashboard.

Serves three things:

* ``/``            the static page in ``dashboard/static/``.
* ``/data/...``    the monitor's data directory (default ``~/cua-run3/data``),
                   always with ``Cache-Control: no-cache``. Supports single-range
                   requests so the page can fetch only the new bytes of a
                   ``.jsonl`` file.
* ``/api/log/stream``  server-sent events that tail ``log.txt``.

Python 3 standard library only. The server never writes, never lists
directories, and refuses any path that resolves outside its two roots.
"""

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
LOG_POLL_S = 1.0
HEARTBEAT_S = 15.0
BACKLOG_LINES = 400
MAX_LINE_CHARS = 2000
MAX_SSE_CLIENTS = 64

EXTRA_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_sse_lock = threading.Lock()
_sse_clients = 0


def content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXTRA_TYPES:
        return EXTRA_TYPES[ext]
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def safe_resolve(root: Path, rel: str):
    """Return the resolved file under `root` for URL path `rel`, or None.

    Refuses empty segments, "..", dotfiles, NUL bytes, and symlinks that
    leave the root.
    """
    if "\x00" in rel or "\\" in rel:
        return None
    parts = [p for p in rel.split("/") if p != ""]
    for part in parts:
        if part in (".", "..") or part.startswith("."):
            return None
    try:
        candidate = (root.joinpath(*parts)).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return None
    return candidate


def read_lines(path: Path):
    """Read a log file as a list of lines (tolerates a missing file)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return []
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln[:MAX_LINE_CHARS] for ln in lines]


def new_lines(prev, cur):
    """Return (mode, lines): what to send to a client that has seen `prev`.

    The monitor rewrites log.txt atomically with the last N lines, so the
    window slides. Find the longest suffix of `prev` that is a prefix of
    `cur`; everything after it is new. With no overlap and different content
    the log was replaced, so the client should reset.
    """
    if prev == cur:
        return "append", []
    if not prev:
        return "append", cur
    if not cur:
        return "reset", []
    first = cur[0]
    for i, line in enumerate(prev):
        if line != first:
            continue
        tail = prev[i:]
        if cur[: len(tail)] == tail:
            return "append", cur[len(tail):]
    return "reset", cur


class Handler(BaseHTTPRequestHandler):
    server_version = "cua-run3-dashboard/1"
    protocol_version = "HTTP/1.1"
    timeout = 30

    # Set from main().
    static_root: Path
    data_root: Path

    def log_message(self, fmt, *args):  # quiet, no client addresses in logs
        if os.environ.get("DASH_VERBOSE"):
            sys.stderr.write("%s\n" % (fmt % args))

    # -- helpers ------------------------------------------------------------
    def _headers(self, status, ctype, length=None, extra=None, cache="no-cache"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _text(self, status, message):
        body = message.encode()
        self._headers(status, "text/plain; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve_file(self, path: Path, allow_range: bool):
        try:
            if not path.is_file():
                return self._text(HTTPStatus.NOT_FOUND, "not found\n")
            size = path.stat().st_size
        except OSError:
            return self._text(HTTPStatus.NOT_FOUND, "not found\n")
        start, end, status = 0, size - 1, HTTPStatus.OK
        extra = {"Accept-Ranges": "bytes" if allow_range else "none"}
        rng = self.headers.get("Range")
        if allow_range and rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                end = min(end, size - 1)
                if start == size and size > 0 and m.group(1):
                    # Nothing new since the client's offset: 204, not an error.
                    self._headers(HTTPStatus.NO_CONTENT, "text/plain; charset=utf-8", None,
                                  {"Content-Range": "bytes */%d" % size})
                    return
                if start >= size or start > end:
                    extra["Content-Range"] = "bytes */%d" % size
                    self._headers(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                                  "text/plain; charset=utf-8", 0, extra)
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                extra["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
        length = end - start + 1 if size else 0
        self._headers(status, content_type(path), length, extra)
        if self.command == "HEAD" or not size:
            return
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except OSError:
            self.close_connection = True

    # -- routing ------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/log/stream":
            return self._log_stream()
        if path == "/healthz":
            return self._text(HTTPStatus.OK, "ok\n")
        if path == "/data" or path.startswith("/data/"):
            target = safe_resolve(self.data_root, path[len("/data"):])
            if target is None:
                return self._text(HTTPStatus.NOT_FOUND, "not found\n")
            return self._serve_file(target, allow_range=True)
        rel = "index.html" if path in ("", "/") else path
        target = safe_resolve(self.static_root, rel)
        if target is None:
            return self._text(HTTPStatus.NOT_FOUND, "not found\n")
        return self._serve_file(target, allow_range=False)

    def _method_not_allowed(self):
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed

    # -- server-sent events -------------------------------------------------
    def _sse(self, event, payload):
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(("event: %s\ndata: %s\n\n" % (event, data)).encode())
        self.wfile.flush()

    def _log_stream(self):
        global _sse_clients
        with _sse_lock:
            if _sse_clients >= MAX_SSE_CLIENTS:
                return self._text(HTTPStatus.SERVICE_UNAVAILABLE, "too many streams\n")
            _sse_clients += 1
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            log_path = self.data_root / "log.txt"
            self.wfile.write(b"retry: 5000\n\n")
            prev = read_lines(log_path)
            self._sse("init", {"lines": prev[-BACKLOG_LINES:]})
            sig = _signature(log_path)
            last_beat = time.monotonic()
            while True:
                time.sleep(LOG_POLL_S)
                now_sig = _signature(log_path)
                if now_sig != sig:
                    sig = now_sig
                    cur = read_lines(log_path)
                    mode, lines = new_lines(prev, cur)
                    prev = cur
                    if mode == "reset":
                        self._sse("reset", {"lines": lines[-BACKLOG_LINES:]})
                        last_beat = time.monotonic()
                    elif lines:
                        self._sse("append", {"lines": lines})
                        last_beat = time.monotonic()
                if time.monotonic() - last_beat >= HEARTBEAT_S:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    last_beat = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            with _sse_lock:
                _sse_clients -= 1


def _signature(path: Path):
    try:
        st = path.stat()
        return (st.st_ino, st.st_size, st.st_mtime_ns)
    except OSError:
        return None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="~/cua-run3/data", help="data directory to serve at /data/")
    ap.add_argument("--static", default=str(HERE / "static"), help="static files directory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)
    Handler.static_root = Path(args.static).expanduser().resolve()
    Handler.data_root = Path(args.data).expanduser().resolve()
    srv = Server((args.host, args.port), Handler)
    print("serving %s and %s on %s:%d" % (Handler.static_root, Handler.data_root, args.host, args.port),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
