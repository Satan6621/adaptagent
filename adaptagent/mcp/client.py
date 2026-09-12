"""MCP (Model Context Protocol) client: stdio JSON-RPC 2.0, no SDK required.

Spawns a server as a subprocess, performs the initialize handshake and the
tools/list discovery, then exposes each remote tool as a local Tool whose
schema is converted from MCP inputSchema to OpenAI function-calling format.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from ..llm.schemas import get_tool_schema
from ..tools.base import Tool

INITIALIZE_PARAMS = {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "adaptagent", "version": "0.2.0"},
}


@dataclass
class _PendingRequest:
    future: asyncio.Future
    method: str


class MCPConnection:
    """One stdio connection to an MCP server."""

    def __init__(self, command: str, args: list[str] | None = None, timeout: float = 30.0) -> None:
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, _PendingRequest] = {}
        self._next_id = 0
        self._reader_task: asyncio.Task | None = None
        self.server_info: dict[str, Any] = {}

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in message and message["id"] in self._pending:
                pending = self._pending.pop(message["id"])
                if not pending.future.done():
                    if "error" in message:
                        pending.future.set_exception(RuntimeError(str(message["error"])))
                    else:
                        pending.future.set_result(message.get("result", {}))
            # notifications (no id) are ignored for now

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        request = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        assert self._proc and self._proc.stdin
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _PendingRequest(future=fut, method=method)
        self._proc.stdin.write((json.dumps(request) + "\n").encode())
        await self._proc.stdin.drain()
        return await asyncio.wait_for(fut, self.timeout)

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        notification = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notification["params"] = params
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(notification) + "\n").encode())
        await self._proc.stdin.drain()

    async def initialize(self) -> dict[str, Any]:
        self.server_info = await self._request("initialize", INITIALIZE_PARAMS)
        await self._notify("notifications/initialized")
        return self.server_info

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else json.dumps(result)

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            self._proc.kill()
            await self._proc.wait()


def _mcp_schema_to_openai(name: str, description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    """MCP inputSchema is already JSON-Schema; pass through with light cleanup."""
    schema = dict(input_schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return {
        "type": "function",
        "function": {"name": name, "description": description or f"MCP tool {name}", "parameters": schema},
    }


def mcp_tool_from_spec(conn: MCPConnection, spec: dict[str, Any]) -> Tool:
    """Wrap one remote MCP tool as a local Tool with an explicit schema."""

    name = spec["name"]
    description = spec.get("description", "")
    input_schema = spec.get("inputSchema", {})
    openai_schema = _mcp_schema_to_openai(name, description, input_schema)

    async def _fn(**kwargs: Any) -> str:
        return await conn.call_tool(name, kwargs)

    _fn.__name__ = name.replace("-", "_").replace(" ", "_")
    _fn.__doc__ = description or f"MCP tool {name}"
    tool = Tool(name=name, description=description or f"MCP tool {name}", fn=_fn)
    tool.openai_schema = openai_schema  # consumed by get_tool_schemas_for
    return tool


def get_tool_schemas_for(tools: list[Tool]) -> list[dict[str, Any]]:
    """Schema list respecting explicit overrides (e.g. MCP passthrough)."""
    schemas = []
    for t in tools:
        if getattr(t, "openai_schema", None):
            schemas.append(t.openai_schema)
        else:
            schemas.append(get_tool_schema(t))
    return schemas


class MCPClient:
    """High-level client: connect to servers, collect tools.

    Usage:
        client = MCPClient()
        tools = await client.connect("npx", ["-y", "@some/mcp-server"])
        # tools: list[Tool] usable by any Agent
        ...
        await client.close()
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.connections: list[MCPConnection] = []
        self.timeout = timeout

    async def connect(self, command: str, args: list[str] | None = None) -> list[Tool]:
        conn = MCPConnection(command, args, timeout=self.timeout)
        await conn.start()
        await conn.initialize()
        specs = await conn.list_tools()
        self.connections.append(conn)
        return [mcp_tool_from_spec(conn, s) for s in specs]

    async def close(self) -> None:
        for conn in self.connections:
            try:
                await conn.stop()
            except Exception:  # noqa: BLE001
                pass
        self.connections = []
