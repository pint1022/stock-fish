"""Helpers for OpenAI-compatible chat completion parameters."""

from __future__ import annotations

from typing import Dict, Optional


def temperature_kwargs(default_temperature: Optional[float]) -> Dict[str, float]:
    """Return a temperature kwarg unless the configured value is the API default.

    Some newer models reject explicit non-default temperature values and only
    support the default value. Omitting temperature lets those models use their
    required default while preserving non-default temperatures for models that
    support them.
    """
    from config import settings

    configured = getattr(settings, "LLM_TEMPERATURE", None)
    value = configured if configured is not None else default_temperature
    if value is None:
        return {}

    temperature = float(value)
    if temperature == 1.0:
        return {}
    return {"temperature": temperature}


def token_limit_kwargs(model: str, max_tokens: int) -> Dict[str, int]:
    """Return the token-limit parameter supported by the selected model."""
    normalized = (model or "").lower()
    if normalized.startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}
