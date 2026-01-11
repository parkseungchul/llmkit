from pathlib import Path
from src.llmkit.input_spec import parse_input

def test_parse_minimal():
    project_root = Path(__file__).resolve().parents[1]
    spec = parse_input({"provider":"openai","model":"gpt-4o-mini","user_prompt":"hi"}, project_root)
    assert spec.provider == "openai"
    assert spec.model == "gpt-4o-mini"
    assert spec.user_prompt == "hi"
