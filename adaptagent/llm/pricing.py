"""Token pricing table (USD per 1M tokens) — best-effort, updates welcome."""

from __future__ import annotations

PRICING: dict[str, tuple[float, float]] = {
    # model prefix -> (input $/1M, output $/1M)
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/o4-mini": (1.10, 4.40),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "Qwen/Qwen2.5-72B-Instruct": (0.35, 0.40),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost for a call. Unknown models -> 0.0 (not billed)."""
    price = PRICING.get(model)
    if price is None:  # try prefix match (versions like -2024-xx)
        for key, p in PRICING.items():
            if model.startswith(key):
                price = p
                break
    if price is None:
        return 0.0
    return prompt_tokens * price[0] / 1_000_000 + completion_tokens * price[1] / 1_000_000


def parse_usage(usage: dict, provider: str = "openai") -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a provider usage blob."""
    if not usage:
        return 0, 0
    if provider == "anthropic":
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
