"""Tests for the MCP streamable-HTTP connection and connect_spec."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_http import FakeHTTPServer, mcp_router  # noqa: E402


def test_http_connection_discovers_and_calls_tools():
    result = {}

    async def go():
        from adaptagent.mcp.client import MCPClient

        with FakeHTTPServer(mcp_router) as server:
            client = MCPClient(timeout=5.0)
            try:
                tools = await client.connect_spec({"type": "http", "url": server.base_url + "/mcp"})
                result["tools"] = tools
                result["out"] = await tools[0].fn(msg="hola")
            finally:
                await client.close()

    asyncio.run(go())
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool.name == "echo"
    schema = tool.openai_schema["function"]
    assert schema["parameters"]["properties"]["msg"]["type"] == "string"
    assert result["out"] == "echo:hola"


def test_http_spec_requires_url():
    async def go():
        from adaptagent.mcp.client import MCPClient

        client = MCPClient(timeout=2.0)
        try:
            await client.connect_spec({"type": "http"})
        finally:
            await client.close()

    with pytest.raises(ValueError):
        asyncio.run(go())


def test_stdio_spec_requires_command():
    async def go():
        from adaptagent.mcp.client import MCPClient

        client = MCPClient(timeout=2.0)
        try:
            await client.connect_spec({"type": "stdio"})
        finally:
            await client.close()

    with pytest.raises(ValueError):
        asyncio.run(go())


def test_connect_spec_reports_server_errors():
    async def go():
        from adaptagent.mcp.client import MCPClient

        client = MCPClient(timeout=2.0)
        try:
            await client.connect_spec({"type": "http", "url": "http://127.0.0.1:1/mcp"})
        finally:
            await client.close()

    with pytest.raises(Exception):
        asyncio.run(go())


def test_http_connection_sse_response():
    """Servers that reply text/event-stream must also work."""

    def sse_router(method, path, headers, body):
        msg = json.loads(body or b"{}")
        m = msg.get("method")
        rid = msg.get("id")
        if m == "tools/list":
            result = {"tools": [{"name": "sse_tool", "description": "d", "inputSchema": {"type": "object"}}]}
        elif m == "tools/call":
            result = {"content": [{"type": "text", "text": "via-sse"}]}
        else:
            result = {"ok": True}
        frame = {"jsonrpc": "2.0", "id": rid, "result": result}
        out = f"data: {json.dumps(frame)}\n\n".encode()
        return 200, "text/event-stream", out

    result = {}

    async def go():
        from adaptagent.mcp.client import MCPClient

        with FakeHTTPServer(sse_router) as server:
            client = MCPClient(timeout=5.0)
            try:
                tools = await client.connect_spec({"type": "http", "url": server.base_url + "/mcp"})
                result["names"] = [t.name for t in tools]
                result["out"] = await tools[0].fn()
            finally:
                await client.close()

    asyncio.run(go())
    assert result["names"] == ["sse_tool"]
    assert result["out"] == "via-sse"
