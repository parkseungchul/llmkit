from __future__ import annotations

from typing import Any, Dict, List

from src.schema import LlmRequest


def build_messages(req: LlmRequest) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with your chosen RAG injection policy."""
    messages: List[Dict[str, Any]] = []

    if req.system_prompt.strip():
        messages.append({"role": "system", "content": req.system_prompt})

    if req.rag_text.strip():
        if req.provider == "openai":
            messages.append({"role": "developer", "content": req.rag_text})
            messages.append({"role": "user", "content": req.user_prompt})
        else:
            tagged = (
                "### Reference Context:\n"
                f"{req.rag_text}\n\n"
                "### User Question:\n"
                f"{req.user_prompt}"
            )
            messages.append({"role": "user", "content": tagged})
    else:
        messages.append({"role": "user", "content": req.user_prompt})

    return messages
