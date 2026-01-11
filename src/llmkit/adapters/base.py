"""Adapter interface for LLM providers."""
from __future__ import annotations

from typing import Any, Dict, List

class LlmAdapterError(Exception):
    pass

class BaseChatAdapter:
    def generate(self, messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
