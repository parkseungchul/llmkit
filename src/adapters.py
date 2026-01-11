from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

import requests

from src.errors import AdapterError
from src.logging_config import logger
from src.utils import sanitize_api_key


class BaseAdapter:
    def generate(self, messages: List[Dict[str, Any]], model: str, return_json: bool) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIStyleAdapter(BaseAdapter):
    """OpenAI Chat Completions compatible."""

    def __init__(self, endpoint: str, api_key_env: str, timeout: int = 30):
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _api_key(self) -> str:
        key = (os.getenv(self.api_key_env) or "").strip()
        if not key:
            raise AdapterError(f"{self.api_key_env} is not set")
        return key

    def generate(self, messages: List[Dict[str, Any]], model: str, return_json: bool) -> Dict[str, Any]:
        key = self._api_key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1000,
            "stream": False,
        }
        if return_json:
            payload["response_format"] = {"type": "json_object"}

        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise AdapterError(f"HTTP {r.status_code}: {r.text[:600]}")
        return r.json()


class GeminiAdapter(BaseAdapter):
    """Google Generative Language API generateContent."""

    def __init__(self, timeout: int = 30, api_key_env: str = "GEMINI_API_KEY"):
        self.timeout = timeout
        self.api_key_env = api_key_env

    def _api_key(self) -> str:
        key = sanitize_api_key(os.getenv(self.api_key_env) or "")
        if not key:
            raise AdapterError(f"{self.api_key_env} is not set")
        return key

    @staticmethod
    def _openai_msgs_to_gemini(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        def to_text(c: Any) -> str:
            if isinstance(c, str):
                return c
            return json.dumps(c, ensure_ascii=False)

        for m in messages:
            role = (m.get("role") or "user").strip().lower()
            text = to_text(m.get("content", ""))

            if role in ("system", "developer"):
                if text.strip():
                    system_parts.append(text.strip())
                continue

            gem_role = "model" if role == "assistant" else "user"
            contents.append({"role": gem_role, "parts": [{"text": text}]})

        payload: Dict[str, Any] = {"contents": contents}
        system_text = "\n\n".join(system_parts).strip()
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        return payload

    def generate(self, messages: List[Dict[str, Any]], model: str, return_json: bool) -> Dict[str, Any]:
        key = self._api_key()
        sha8 = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        logger.info("[GEMINI_KEY] len=%d sha8=%s", len(key), sha8)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = self._openai_msgs_to_gemini(messages)

        gen_cfg: Dict[str, Any] = {"temperature": 0.0, "topP": 1.0, "maxOutputTokens": 1000}
        if return_json:
            gen_cfg["responseMimeType"] = "application/json"
        payload["generationConfig"] = gen_cfg

        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise AdapterError(f"HTTP {r.status_code}: {r.text[:600]}")

        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )

        usage = data.get("usageMetadata", {}) or {}
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0) or 0,
                "completion_tokens": usage.get("candidatesTokenCount", 0) or 0,
                "total_tokens": usage.get("totalTokenCount", 0) or 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "_raw_provider": "gemini",
            "_raw": data,
        }


def build_adapter(provider: str) -> BaseAdapter:
    if provider == "openai":
        return OpenAIStyleAdapter("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY")
    if provider == "ytl":
        return OpenAIStyleAdapter("https://api.ytlailabs.tech/v1/chat/completions", "YTL_API_KEY")
    if provider == "gemini":
        return GeminiAdapter()
    raise AdapterError(f"Unsupported provider: {provider}")
