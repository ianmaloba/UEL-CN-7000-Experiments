"""Conservative text-token price lookup for preflight budget reservations.

Rates are USD per million tokens. Current provider prices change, so each frozen
campaign stores its own registry snapshot and this file records the lookup policy;
actual token usage is also retained per response for post-run reconciliation.
"""
from __future__ import annotations

import re
from typing import Any


def rates_for(provider: str, model_id: str, metadata: dict[str, Any] | None = None) -> tuple[float, float] | None:
    model = model_id.lower()
    if provider == "openai":
        if model.endswith("chat-latest"):
            return 5.0, 30.0
        if "o1-pro" in model:
            return 150.0, 600.0
        if "gpt-5.5-pro" in model or "gpt-5.4-pro" in model:
            return 30.0, 180.0
        if "gpt-5.2-pro" in model:
            return 21.0, 168.0
        if "gpt-5-pro" in model:
            return 15.0, 120.0
        if model.startswith("gpt-6-astra"):
            return 10.0, 50.0
        if model.startswith("gpt-6-sol"):
            return 2.0, 10.0
        if model.startswith("gpt-6-luna"):
            return 0.10, 0.50
        if model.startswith("gpt-5.6-sol"):
            return 4.0, 20.0
        if model.startswith("gpt-5.6-terra"):
            return 2.0, 12.0
        if model.startswith("gpt-5.6-luna"):
            return 0.20, 1.20
        if model.startswith("gpt-5.5") or model == "chat-latest":
            return 5.0, 30.0
        if model.startswith("gpt-5.4-mini"):
            return 0.40, 1.60
        if model.startswith("gpt-5.4-nano"):
            return 0.10, 0.40
        if model.startswith("gpt-5.4"):
            return 2.50, 15.0
        if model.startswith("gpt-5.3-codex"):
            return 3.50, 28.0
        if model.startswith("gpt-5.2-codex"):
            return 1.75, 14.0
        if model.startswith("gpt-5.1-codex") or model.startswith("gpt-5-codex"):
            return 1.25, 10.0
        if model.startswith("gpt-5.2"):
            return 1.75, 14.0
        if model.startswith("gpt-5.1") or model.startswith("gpt-5-") or model == "gpt-5":
            if "nano" in model:
                return 0.05, 0.40
            if "mini" in model:
                return 0.25, 2.0
            return 1.25, 10.0
        if model.startswith("gpt-4.1-mini"):
            return 0.40, 1.60
        if model.startswith("gpt-4.1-nano"):
            return 0.10, 0.40
        if model.startswith("gpt-4.1"):
            return 2.0, 8.0
        if model.startswith("gpt-4o-mini"):
            return 0.15, 0.60
        if model.startswith("gpt-4o"):
            return 2.50, 10.0
        if model.startswith("gpt-4-turbo"):
            return 10.0, 30.0
        if model.startswith("gpt-4"):
            return 30.0, 60.0
        if model.startswith("gpt-3.5-turbo-16k"):
            return 3.0, 4.0
        if model.startswith("gpt-3.5-turbo-instruct"):
            return 1.50, 2.0
        if model.startswith("gpt-3.5-turbo"):
            return 0.50, 1.50
        if model.startswith("babbage-002"):
            return 0.40, 0.40
        if model.startswith("davinci-002"):
            return 2.0, 2.0
        if model.startswith("o1"):
            return 15.0, 60.0
        if model.startswith("o3-mini") or model.startswith("o4-mini"):
            return 1.10, 4.40
        if model.startswith("o3"):
            return 2.0, 8.0
        return None

    if provider == "xai" and metadata:
        prompt_price = metadata.get("prompt_text_token_price")
        completion_price = metadata.get("completion_text_token_price")
        if isinstance(prompt_price, (int, float)) and isinstance(completion_price, (int, float)):
            # xAI's model-list fields are USD cents per 100 million text tokens.
            return float(prompt_price) / 10_000, float(completion_price) / 10_000
        return None
    if provider == "xai":
        if model.startswith(("grok-4.7", "grok-4.6", "grok-4.5")):
            return 2.0, 6.0
        if model.startswith(("grok-4.20", "grok-4.3")):
            return 1.25, 2.50
        if model.startswith(("grok-build", "grok-code-fast")):
            return 1.0, 2.0
        return None

    if provider == "deepseek":
        if "pro" in model:
            return 1.32, 3.96
        return 0.30, 1.20

    if provider == "glm":
        if model == "glm-5":
            return 1.0, 3.20
        if model == "glm-5-turbo":
            return 10.0, 10.0
        if "5.3-flashx" in model:
            return 0.37, 1.25
        if "5.3-flash" in model:
            return 0.15, 0.50
        if "5.3" in model or "5.2" in model or "5.1" in model:
            return 1.40, 4.40
        if "4.5-airx" in model:
            return 1.10, 4.50
        if "4.5-air" in model:
            return 0.20, 1.10
        if "4.5-x" in model:
            return 2.20, 8.90
        if "4.7-flashx" in model:
            return 0.07, 0.40
        if "flash" in model:
            return 0.0, 0.0
        if "4-32b" in model:
            return 0.10, 0.10
        if re.search(r"glm-4\.[567]", model) or "glm-4.5" in model:
            return 0.60, 2.20
        return None

    if provider == "mistral":
        if "codestral" in model:
            return 0.30, 0.90
        if "medium" in model:
            return 1.50, 7.50
        if "large" in model:
            return 0.50, 1.50
        if "ministral-14b" in model:
            return 0.20, 0.20
        if "ministral-8b" in model:
            return 0.15, 0.15
        if "ministral-3b" in model:
            return 0.10, 0.10
        if "small" in model:
            return 0.15, 0.60
        # Use the highest published general text-model tier as a safe upper bound
        # for new names, and label the estimate conservative in the manifest.
        return 1.50, 7.50

    return None


def token_cost(input_tokens: int, output_tokens: int, rates: tuple[float, float]) -> float:
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def output_tokens_from_usage(
    provider: str, output_tokens: int, usage: dict[str, Any],
) -> int:
    """Include separately reported hidden reasoning tokens in xAI usage totals."""
    if provider != "xai":
        return output_tokens
    completion_details = usage.get("completion_tokens_details", {})
    reasoning_tokens = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict) else None
    )
    return output_tokens + reasoning_tokens if isinstance(reasoning_tokens, int) else output_tokens


def output_token_reserve(provider: str, max_tokens: int) -> int:
    """Reserve extra output headroom for xAI's separately billed reasoning tokens.

    Observed xAI responses can report reasoning usage in addition to the requested
    visible-token cap. Two caps cover the observed range; runtime budget checks still
    reconcile provider-reported spend before continuing with another request.
    """
    return max_tokens * 2 if provider == "xai" else max_tokens
