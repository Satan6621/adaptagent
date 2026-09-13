"""In-process fake HTTP servers for tests (stdlib only, threaded)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

Router = Callable[[str, str, dict, bytes], tuple[int, str, bytes]]  # method, path, headers, body


class _Handler(BaseHTTPRequestHandler):
    router: Router | None = None

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        status, ctype, out = type(self).router(self.command, self.path, dict(self.headers), body)  # noqa: B010
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class FakeHTTPServer:
    """Context manager: runs a ThreadingHTTPServer on an ephemeral port."""

    def __init__(self, router: Router) -> None:
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _Handler.router = router

    def __enter__(self) -> "FakeHTTPServer":
        import threading

        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def mcp_router(method: str, path: str, headers: dict, body: bytes) -> tuple[int, str, bytes]:
    """Minimal MCP streamable-HTTP server: initialize, tools/list, tools/call."""
    msg = json.loads(body or b"{}")
    m = msg.get("method")
    rid = msg.get("id")
    if m == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0"},
        }
    elif m == "notifications/initialized":
        return 202, "application/json", b""
    elif m == "tools/list":
        result = {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo the message back",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                }
            ]
        }
    elif m == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        result = {"content": [{"type": "text", "text": f"echo:{args.get('msg')}"}]}
    else:
        result = {}
    out = {"jsonrpc": "2.0", "id": rid, "result": result}
    return 200, "application/json", json.dumps(out).encode()


def echo_router(method: str, path: str, headers: dict, body: bytes) -> tuple[int, str, bytes]:
    """Echoes back URL query (GET) or JSON body (POST) as JSON."""
    parsed = urlparse(path)
    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    if method == "GET":
        out = {"method": "GET", "path": parsed.path, "query": query}
    else:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            payload = {}
        out = {"method": method, "path": parsed.path, "body": payload}
    return 200, "application/json", json.dumps(out).encode()
