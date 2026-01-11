"""Prompt layer assembly.

Rules:
- We only use system/developer/user layers.
- Unknown/special roles are ignored.
- System prompt should be stable and not easily polluted.
- RAG injection:
  - OpenAI provider: inject as developer message
  - Other providers: merge RAG text into the last user message with tags
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import InputSpec, PreparedRequest
from .logging_util import get_logger

logger = get_logger(__name__)

def _load_default_system(project_root: Path) -> str:
    p = project_root / "src" / "prompts" / "system_default.txt"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return "You are a helpful assistant."

def _load_json_contract(project_root: Path) -> str:
    p = project_root / "src" / "prompts" / "system_json_contract.txt"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return "Return a valid JSON object only."

def _resolve_rag_text(project_root: Path, rag_id: Optional[str], rag_text: Optional[str]) -> Optional[str]:
    if rag_text and rag_text.strip():
        return rag_text.strip()

    if not rag_id:
        return None

    rag_dir = os.environ.get("LLMKIT_RAG_DIR", "").strip()
    candidates = []
    if rag_dir:
        candidates.append(Path(rag_dir) / f"{rag_id}.txt")
    candidates.append(project_root / "src" / "rag" / f"{rag_id}.txt")

    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return None

def _truncate_rag(text: str) -> str:
    lim = os.environ.get("LLMKIT_RAG_MAX_CHARS", "").strip()
    if not lim:
        return text
    try:
        n = int(lim)
        if n > 0 and len(text) > n:
            return text[:n]
    except Exception:
        pass
    return text

def _build_messages_from_spec(project_root: Path, spec: InputSpec) -> List[Dict[str, Any]]:
    system_text = (spec.system_prompt or "").strip() or _load_default_system(project_root)
    user_text = (spec.user_prompt or "").strip()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]
    messages.append({"role": "user", "content": user_text})

    # Optional JSON contract append (opt-in)
    if spec.return_json and os.environ.get("LLMKIT_APPEND_JSON_CONTRACT", "").strip() == "1":
        messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + _load_json_contract(project_root).strip()

    return messages

def _sanitize_advanced_messages(project_root: Path, spec: InputSpec) -> List[Dict[str, Any]]:
    msgs = list(spec.messages or [])
    if not msgs:
        return _build_messages_from_spec(project_root, spec)

    sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if sys_idx is None:
        msgs.insert(0, {"role": "system", "content": _load_default_system(project_root)})
        sys_idx = 0

    if spec.system_prompt and spec.system_prompt.strip():
        msgs[sys_idx]["content"] = spec.system_prompt.strip()

    if spec.return_json and os.environ.get("LLMKIT_APPEND_JSON_CONTRACT", "").strip() == "1":
        msgs[sys_idx]["content"] = (msgs[sys_idx].get("content") or "").rstrip() + "\n\n" + _load_json_contract(project_root).strip()

    return msgs

def apply_rag_injection(messages: List[Dict[str, Any]], provider: str, rag_text: str) -> List[Dict[str, Any]]:
    rag_text = _truncate_rag(rag_text)

    if provider == "openai":
        sys_indexes = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        insert_pos = (sys_indexes[-1] + 1) if sys_indexes else 1
        out = list(messages)
        out.insert(insert_pos, {"role": "developer", "content": rag_text})
        return out

    user_indexes = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    out = list(messages)
    if user_indexes:
        last = user_indexes[-1]
        q = str(out[last].get("content") or "")
        out[last]["content"] = (
            "### Reference Context:\n"
            f"{rag_text}\n\n"
            "### User Question:\n"
            f"{q}"
        )
    else:
        out.append({"role": "user", "content": f"### Reference Context:\n{rag_text}"})
    return out

def prepare_request(project_root: Path, spec: InputSpec) -> PreparedRequest:
    if spec.messages:
        messages = _sanitize_advanced_messages(project_root, spec)
    else:
        messages = _build_messages_from_spec(project_root, spec)

    rag_text = _resolve_rag_text(project_root, spec.rag_id, spec.rag_text)
    if rag_text:
        messages = apply_rag_injection(messages, provider=spec.provider, rag_text=rag_text)

    return PreparedRequest(
        provider=spec.provider,
        model=spec.model,
        messages=messages,
        return_json=spec.return_json,
        options=spec.options,
    )
