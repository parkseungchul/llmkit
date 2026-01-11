from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.errors import ValidationError


@dataclass
class LlmRequest:
    env: str
    provider: str
    model: str
    system_prompt: str = ""
    user_prompt: str = ""
    rag_id: str = ""
    rag_text: str = ""
    return_json: bool = False
    notice: str = ""


def normalize_provider(p: Any) -> str:
    s = (p or "").strip().lower()
    if s in ("openai", "ytl", "gemini"):
        return s
    raise ValidationError(f"Unsupported provider: {p}")


def validate_nonempty(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} is required")
    return value.strip()
