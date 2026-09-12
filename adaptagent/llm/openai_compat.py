from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable

import httpx

from .base import (
    DEFAULT_BASE_URLS,
    ENV_KEYS,
    LLM,
    PROVIDER_ALIASES,
    LLMConfig,
    LLMResponse,
    ToolCall,
    register_llm,
)


class _ProviderBase:
    """Shared plumbing: headers, client lifecycle, env resolution."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = self._finalize(config)

    def _finalize(self, config: LLMConfig) -> LLMConfig:
        provider = PROVIDER_ALIASES.get(config.provider.lower(), config.provider.lower())
        api_key = config.api_key or os.getenv(ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY"), "")
        base_url = config.base_url or os.getenv(f"{provider.upper()}_BASE_URL") or DEFAULT_BASE_URLS.get(
            provider, DEFAULT_BASE_URLS["openai"]
        )
        return config.with_overrides(provider=provider, api_key=api_key, base_url=base_url)

    async def close(self) -> None:
        if getattr(self, "client", None):
            await self.client.aclose()


@register_llm("openai")
class OpenAICompatLLM(_ProviderBase, LLM):
    """Works with any OpenAI-compatible /chat/completions endpoint."""

    def __post_init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout, headers=self._headers()
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/adaptagent"
            headers["X-Title"] = "AdaptAgent"
        return headers

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout, headers=self._headers()
        )

    def tool_prompt(self, schemas: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tools": schemas,
            "tool_choice": "auto",
        }

    RETRY_STATUS = {429, 500, 502, 503, 504}
    MAX_RETRIES = 3

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions with exponential backoff on 429/5xx."""
        import asyncio as _asyncio

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = await self.client.post("/chat/completions", json=payload)
                if resp.status_code in self.RETRY_STATUS and attempt < self.MAX_RETRIES:
                    wait = float(2**attempt)  # exponential backoff
                    if resp.headers.get("retry-after"):
                        try:
                            wait = max(wait, float(resp.headers["retry-after"]))
                        except ValueError:
                            pass
                    await _asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in self.RETRY_STATUS and attempt < self.MAX_RETRIES:
                    await _asyncio.sleep(2**attempt)
                    continue
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    await _asyncio.sleep(2**attempt)
                    continue
                raise
        raise last_error or RuntimeError("unreachable")

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update({k: v for k, v in kwargs.items() if k != "stream"})
        if stream and not tools:
            payload["stream"] = True
            return await self._generate_stream(payload, on_token)
        data = await self._post_with_retry(payload)
        return self._parse(data)

    async def _generate_stream(
        self, payload: dict[str, Any], on_token: Callable[[str], None] | None
    ) -> LLMResponse:
        """Server-sent-events streaming; falls back to non-streaming on error."""
        chunks: list[str] = []
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    delta = (
                        json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
                    )
                    if delta:
                        chunks.append(delta)
                        if on_token:
                            result = on_token(delta)
                            if asyncio.iscoroutine(result):
                                await result
        except httpx.HTTPError:
            payload.pop("stream", None)
            resp = await self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return self._parse(resp.json())
        return LLMResponse(text="".join(chunks))

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = [ToolCall.from_openai(tc) for tc in message.get("tool_calls", []) or []]
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            raw=data,
            usage=data.get("usage", {}) or {},
        )


@register_llm("anthropic")
class AnthropicLLM(_ProviderBase, LLM):
    """Anthropic Messages API bridge with OpenAI-style tool loop."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
                **self.config.extra_headers,
            },
        )

    def tool_prompt(self, schemas: list[dict[str, Any]]) -> dict[str, Any]:
        return {"tools": schemas}

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        converted: list[dict[str, Any]] = []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                continue
            converted.append({"role": role if role in ("user", "assistant") else "user", "content": content})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature,
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        resp = await self.client.post("/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        text_parts, tool_calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("input", {}))
                )
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            raw=data,
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
        )
