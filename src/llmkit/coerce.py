"""Optional output coercion.

Your decisions:
- If return_json=true, try to interpret the model output as JSON.
- If parsing/coercion fails, still return the raw output and attach parse_error metadata.
- Only coerce when possible (do not aggressively rewrite valid outputs).

This file implements:
- coerce_json_object_text(text) -> (parsed_json_or_none, error_or_none)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()

def coerce_json_object_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if text is None:
        return None, "empty response"

    s = str(text).strip()
    if not s:
        return None, "empty response"

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "parsed JSON is not an object"
    except Exception:
        pass

    extracted = _extract_first_json_object(s)
    if not extracted:
        return None, "no JSON object found in text"

    try:
        obj = json.loads(extracted)
        if isinstance(obj, dict):
            return obj, None
        return None, "extracted JSON is not an object"
    except Exception as e:
        return None, f"json parse failed after extraction: {e}"
