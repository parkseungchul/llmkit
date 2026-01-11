from pathlib import Path
from src.llmkit.registry import is_allowed

def test_allowlist_strict_blocks_when_missing(tmp_path):
    (tmp_path / "src" / "configs").mkdir(parents=True, exist_ok=True)
    ok, reason = is_allowed(tmp_path, "openai", "gpt-4o-mini", strict=True)
    assert ok is False
