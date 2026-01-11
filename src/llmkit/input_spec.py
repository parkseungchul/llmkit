"""Input specification parsing.

Goals:
- Accept a single JSON dict as the input for Layer1.
- Support file references using '@path/to/file.txt' for prompt fields.
- Auto-detect "mode":
  - If 'messages' exists and is a list -> advanced mode (caller provides OpenAI-like messages)
  - Else -> our simplified mode (system_prompt + user_prompt + optional rag_id/rag_text)

We ignore unsupported roles in messages:
- Only keep system, developer, user, assistant
- Any other roles are dropped (special role inputs are ignored)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import CallOptions, InputSpec, Provider
from .logging_util import get_logger

logger = get_logger(__name__)

def _read_text_ref(v: Optional[str], project_root: Path) -> Optional[str]:
    if not v:
        return None
    if isinstance(v, str) and v.startswith("@"):
        path = Path(v[1:])
        if not path.is_absolute():
            path = (project_root / path).resolve()
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read file reference: {v} ({e})")
    return v

def _normalize_provider(v: Any) -> Provider:
    s = (v or "").strip().lower()
    if s in ("openai", "ytl", "gemini"):
        return s  # type: ignore
    # Default provider: ytl
    return "ytl"  # type: ignore

def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "y", "yes"):
        return True
    if s in ("0", "false", "n", "no"):
        return False
    return default

def _to_int(v: Any, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _to_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _sanitize_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(messages, list):
        return None

    allowed = {"system", "developer", "user", "assistant"}
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        if role not in allowed:
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out

def parse_input(req: Dict[str, Any], project_root: Path) -> InputSpec:
    provider = _normalize_provider(req.get("provider"))
    model = (req.get("model") or "").strip()

    if not model:
        model = {
            "openai": "gpt-4o-mini",
            "ytl": "ILMU-text",
            "gemini": "gemini-2.5-flash-lite",
        }.get(provider, "gpt-4o-mini")

    options_in = req.get("options") or {}
    if not isinstance(options_in, dict):
        options_in = {}

    options = CallOptions(
        max_tokens=_to_int(options_in.get("max_tokens"), 1024),
        temperature=_to_float(options_in.get("temperature"), 0.7),
        top_p=_to_float(options_in.get("top_p"), 1.0),
        stream=_to_bool(options_in.get("stream"), False),
    )

    system_prompt = _read_text_ref(req.get("system_prompt"), project_root)
    user_prompt = _read_text_ref(req.get("user_prompt"), project_root)

    rag_id = (req.get("rag_id") or "").strip() or None
    rag_text = _read_text_ref(req.get("rag_text"), project_root)

    return_json = _to_bool(req.get("return_json"), False)
    strict = _to_bool(req.get("strict"), False)

    messages = _sanitize_messages(req.get("messages"))

    spec = InputSpec(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        rag_id=rag_id,
        rag_text=rag_text,
        return_json=return_json,
        strict=strict,
        options=options,
        messages=messages,
    )

    logger.debug("Parsed InputSpec: %s", asdict(spec))
    return spec
