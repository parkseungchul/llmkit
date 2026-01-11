"""Gemini REST adapter (generateContent)."""
from __future__ import annotations

import hashlib
import json
import os
import requests
from typing import Any, Dict, List, Tuple

from .base import BaseChatAdapter, LlmAdapterError

def _sanitize_api_key(raw: str) -> str:
    k = (raw or "").strip()
    k = k.strip(' "\'`')
    k = k.strip("“”‘’")
    return k

def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)

def _messages_to_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for m in messages or []:
        role = (m.get("role") or "user").strip().lower()
        text = _to_text(m.get("content", ""))

        if role in ("system", "developer"):
            if text.strip():
                system_parts.append(text.strip())
            continue

        gem_role = "model" if role == "assistant" else "user"
        contents.append({"role": gem_role, "parts": [{"text": text}]})

    system_text = "\n\n".join(system_parts).strip()

    payload: Dict[str, Any] = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload

class GeminiAdapter(BaseChatAdapter):
    def __init__(self, timeout: int = 30, api_key_env: str = "GEMINI_API_KEY"):
        self.timeout = timeout
        self.api_key_env = api_key_env

    def generate(self, messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
        raw_key = os.getenv(self.api_key_env) or ""
        api_key = _sanitize_api_key(raw_key)
        if not api_key:
            raise LlmAdapterError(f"Missing environment variable: {self.api_key_env}")

        model = cfg.get("model", "gemini-2.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        payload = _messages_to_payload(messages)

        generation_config: Dict[str, Any] = {
            "temperature": cfg.get("temperature", 0.7),
            "maxOutputTokens": cfg.get("max_tokens", 1024),
            "topP": cfg.get("top_p", 1.0),
        }
        if cfg.get("return_json"):
            generation_config["responseMimeType"] = "application/json"

        payload["generationConfig"] = generation_config

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise LlmAdapterError(f"request failed: {e}")

        if r.status_code != 200:
            raise LlmAdapterError(f"http {r.status_code}: {r.text[:800]}")

        data = r.json()

        generated_text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )

        usage = data.get("usageMetadata", {}) or {}
        prompt_tokens = usage.get("promptTokenCount", 0) or 0
        completion_tokens = usage.get("candidatesTokenCount", 0) or 0
        total_tokens = usage.get("totalTokenCount", 0) or 0

        return {
            "choices": [{"message": {"role": "assistant", "content": generated_text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "gemini_raw": data,
        }
