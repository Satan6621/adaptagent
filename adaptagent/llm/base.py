"""Multi-provider LLM layer.

A single HTTP client talks to any OpenAI-compatible endpoint (OpenAI, DeepSeek,
OpenRouter, SiliconFlow, Ollama, vLLM, LM Studio, ...). Anthropic is bridged
through its Messages API. No heavyweight SDKs are required.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import httpx

from ..config import _load_env_file

_load_env_file()


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    text: str = ""
    tool_calls: list["ToolCall"] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai(cls, call: dict[str, Any]) -> "ToolCall":
        args = call.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"raw": args}
        return cls(id=call.get("id", ""), name=call.get("function", {}).get("name", ""), arguments=args)


@runtime_checkable
class LLM(Protocol):
    """Minimal interface every provider implementation must satisfy."""

    config: "LLMConfig"

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse: ...

    def tool_prompt(self, schemas: list[dict[str, Any]]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


@dataclass
class LLMConfig:
    """Provider-agnostic configuration."""

    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "LLMConfig":
        data = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        data.update({k: v for k, v in kwargs.items() if k in data})
        return LLMConfig(**data)


PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "siliconflow": "siliconflow",
    "gemini": "gemini",
    "google": "gemini",
    "ollama": "ollama",
    "vllm": "vllm",
    "lmstudio": "lmstudio",
    "anthropic": "anthropic",
    "claude": "anthropic",
}

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "siliconflow": "https://api.siliconflow.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_LLM_REGISTRY: dict[str, Callable[[LLMConfig], LLM]] = {}


def register_llm(provider: str) -> Callable:
    """Class decorator to register a provider implementation."""

    def wrapper(cls: type) -> type:
        _LLM_REGISTRY[provider.lower()] = cls
        return cls

    return wrapper


def resolve_llm(config: LLMConfig | None = None, **kwargs: Any) -> LLM:
    """Build an LLM instance from config (falls back to env: ADAPTAGENT_*)."""
    from .openai_compat import AnthropicLLM, OpenAICompatLLM

    if config is None:
        config = LLMConfig()
    provider = PROVIDER_ALIASES.get(config.provider.lower(), config.provider.lower())
    if provider in _LLM_REGISTRY:
        return _LLM_REGISTRY[provider](config.with_overrides(**kwargs))
    if provider == "anthropic":
        return AnthropicLLM(config.with_overrides(**kwargs))
    return OpenAICompatLLM(config.with_overrides(**kwargs))
