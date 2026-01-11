"""Shared types and lightweight data containers.

We avoid heavy frameworks here. The goal is:
- keep Layer1 portable
- keep typing clear but not over-abstract
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

Provider = Literal["openai", "ytl", "gemini"]

@dataclass
class CallOptions:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False

@dataclass
class InputSpec:
    provider: Provider
    model: str
    system_prompt: Optional[str]
    user_prompt: Optional[str]
    rag_id: Optional[str]
    rag_text: Optional[str]
    return_json: bool
    strict: bool
    options: CallOptions
    # Advanced pass-through: if caller already has messages (OpenAI-like),
    # they can provide them. We still sanitize roles to system/developer/user/assistant only.
    messages: Optional[List[Dict[str, Any]]] = None

@dataclass
class PreparedRequest:
    provider: Provider
    model: str
    messages: List[Dict[str, Any]]
    return_json: bool
    options: CallOptions
