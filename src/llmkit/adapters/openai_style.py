"""OpenAI-style chat.completions adapter."""
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List

from .base import BaseChatAdapter, LlmAdapterError

def _read_env(key: str) -> str:
    v = (os.getenv(key) or "").strip()
    if not v:
        raise LlmAdapterError(f"Missing environment variable: {key}")
    return v

class OpenAIStyleAdapter(BaseChatAdapter):
    def __init__(self, endpoint: str, api_key_env: str, timeout: int = 30):
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
        api_key = _read_env(self.api_key_env)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": cfg.get("model"),
            "messages": messages,
            "max_tokens": cfg.get("max_tokens", 1024),
            "temperature": cfg.get("temperature", 0.7),
            "top_p": cfg.get("top_p", 1.0),
            "stream": cfg.get("stream", False),
        }

        if cfg.get("return_json"):
            payload["response_format"] = {"type": "json_object"}

        try:
            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise LlmAdapterError(f"request failed: {e}")

        if r.status_code != 200:
            raise LlmAdapterError(f"http {r.status_code}: {r.text[:800]}")

        return r.json()
