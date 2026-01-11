"""Provider/model registry and strict allowlist.

Design:
- Strict mode means: "Only allowed provider+model combinations are callable."
- If allowlist file is missing or empty and strict mode is on -> block.
- Non-strict mode: allow any provider+model (but still requires API key and correct adapter).

allowlist.yaml supports:
- providers: ["openai", "ytl", "gemini"]
- models:
    openai:
      - gpt-4o-mini
    gemini:
      - gemini-2.5-flash-lite
    ytl:
      - ILMU-text
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from .logging_util import get_logger

logger = get_logger(__name__)

def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.error("Failed to load YAML: %s (%s)", path, e)
        return {}

def load_allowlist(project_root: Path) -> Dict:
    path = project_root / "src" / "configs" / "allowlist.yaml"
    return _load_yaml(path)

def is_allowed(project_root: Path, provider: str, model: str, strict: bool) -> Tuple[bool, str]:
    if not strict:
        return True, "strict=false"

    allow = load_allowlist(project_root)
    if not allow:
        return False, "strict=true but allowlist missing or empty"

    providers = set((allow.get("providers") or []))
    if providers and provider not in providers:
        return False, f"provider not allowed: {provider}"

    models = allow.get("models") or {}
    allowed_models = set(models.get(provider) or [])
    if allowed_models and model not in allowed_models:
        return False, f"model not allowed for provider={provider}: {model}"

    return True, "allowed"
