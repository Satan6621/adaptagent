"""Tests for user-defined HTTP tools (make_custom_http_tool)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_http import FakeHTTPServer, echo_router  # noqa: E402

from adaptagent.tools.custom import make_custom_http_tool  # noqa: E402


def test_custom_tool_schema_from_placeholders():
    tool = make_custom_http_tool(
        {"name": "weather", "description": "Get the weather", "method": "GET", "url": "https://api.example.com/w={city}"}
    )
    schema = tool.openai_schema["function"]
    assert schema["name"] == "weather"
    assert schema["parameters"]["properties"]["city"]["type"] == "string"
    assert "city" in schema["parameters"]["required"]


def test_custom_tool_call_real_http():
    with FakeHTTPServer(echo_router) as server:
        tool = make_custom_http_tool(
            {"name": "q", "description": "query", "method": "GET", "url": server.base_url + "/we?city={city}"}
        )
        out = asyncio.run(tool.fn(city="Madrid"))
        data = json.loads(out)
        assert data["method"] == "GET"
        assert data["query"]["city"] == "Madrid"


def test_custom_tool_post_body():
    with FakeHTTPServer(echo_router) as server:
        tool = make_custom_http_tool({"name": "send", "description": "post", "method": "POST", "url": server.base_url + "/post"})
        out = json.loads(asyncio.run(tool.fn(body={"a": 1})))
        assert out["method"] == "POST"
        assert out["body"]["a"] == 1


def test_custom_tool_custom_params_schema():
    tool = make_custom_http_tool(
        {
            "name": "rich",
            "description": "d",
            "method": "GET",
            "url": "https://api.example.com/x",
            "params": [{"name": "count", "type": "integer", "required": True, "description": "how many"}],
        }
    )
    schema = tool.openai_schema["function"]["parameters"]
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["count"]["description"] == "how many"
    assert schema["required"] == ["count"]


def test_custom_tool_requires_url():
    with pytest.raises(ValueError):
        make_custom_http_tool({"name": "x", "url": ""})
