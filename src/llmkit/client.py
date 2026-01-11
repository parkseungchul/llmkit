"""LLMClient: Layer1 orchestrator."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import GeminiAdapter, OpenAIStyleAdapter
from .coerce import coerce_json_object_text
from .input_spec import parse_input
from .prompt_layers import prepare_request
from .registry import is_allowed
from .logging_util import get_logger, log_step

logger = get_logger(__name__)

class LLMClient:
    def __init__(self, project_root: Optional[Path] = None):
        # Auto-detect root:
        # <root>/src/llmkit/client.py -> parents[2] == <root>
        self.project_root = project_root or Path(__file__).resolve().parents[2]

        openai_endpoint = (os.environ.get("OPENAI_ENDPOINT") or "https://api.openai.com/v1/chat/completions").strip()
        ytl_endpoint = (os.environ.get("YTL_ENDPOINT") or "https://api.ytlailabs.tech/v1/chat/completions").strip()

        self._adapters = {
            "openai": OpenAIStyleAdapter(endpoint=openai_endpoint, api_key_env="OPENAI_API_KEY", timeout=30),
            "ytl": OpenAIStyleAdapter(endpoint=ytl_endpoint, api_key_env="YTL_API_KEY", timeout=30),
            "gemini": GeminiAdapter(timeout=30),
        }

    def run(self, req: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"request_id": request_id, "steps": {}, "where": None}
        t0 = time.time()

        try:
            log_step(logger, "1", "parse input")
            spec = parse_input(req, project_root=self.project_root)
            meta["steps"]["parse_input_ms"] = int((time.time() - t0) * 1000)

            log_step(logger, "2", "strict allowlist check")
            ok, reason = is_allowed(self.project_root, spec.provider, spec.model, spec.strict)
            meta["steps"]["allowlist_reason"] = reason
            if not ok:
                meta["where"] = "allowlist"
                meta["steps"]["total_ms"] = int((time.time() - t0) * 1000)
                return {"raw": None, "view": None, "parse_error": reason, "meta": meta}

            log_step(logger, "3", "prepare prompt layers")
            prepared = prepare_request(self.project_root, spec)
            meta["steps"]["prepare_ms"] = int((time.time() - t0) * 1000)

            log_step(logger, "4", "call provider")
            adapter = self._adapters.get(prepared.provider)
            if not adapter:
                meta["where"] = "adapter_select"
                meta["steps"]["total_ms"] = int((time.time() - t0) * 1000)
                return {"raw": None, "view": None, "parse_error": f"unsupported provider: {prepared.provider}", "meta": meta}

            cfg = {
                "provider": prepared.provider,
                "model": prepared.model,
                "max_tokens": prepared.options.max_tokens,
                "temperature": prepared.options.temperature,
                "top_p": prepared.options.top_p,
                "stream": prepared.options.stream,
                "return_json": prepared.return_json,
            }

            t_call = time.time()
            raw = adapter.generate(prepared.messages, cfg)
            meta["steps"]["call_ms"] = int((time.time() - t_call) * 1000)

            log_step(logger, "5", "build view and optional coerce")
            text = self._extract_text(raw)
            parsed_json = None
            parse_error = None
            if prepared.return_json:
                parsed_json, parse_error = coerce_json_object_text(text)

            view = {
                "provider": prepared.provider,
                "model": prepared.model,
                "text": text,
                "json": parsed_json,
                "meta": {"return_json": prepared.return_json},
            }

            meta["steps"]["total_ms"] = int((time.time() - t0) * 1000)
            return {"raw": raw, "view": view, "parse_error": parse_error, "meta": meta}

        except Exception as e:
            logger.exception("LLMClient.run failed: %s", e)
            meta["where"] = meta.get("where") or "LLMClient.run"
            meta["steps"]["total_ms"] = int((time.time() - t0) * 1000)
            return {"raw": None, "view": None, "parse_error": str(e), "meta": meta}

    @staticmethod
    def _extract_text(raw: Dict[str, Any]) -> str:
        try:
            choices = raw.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            return str(msg.get("content") or "")
        except Exception:
            return ""
