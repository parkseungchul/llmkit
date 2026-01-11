from pathlib import Path
from src.llmkit.input_spec import parse_input
from src.llmkit.prompt_layers import prepare_request

def test_rag_injection_non_openai_merges_into_user():
    project_root = Path(__file__).resolve().parents[1]
    spec = parse_input({
        "provider":"gemini",
        "model":"gemini-2.5-flash-lite",
        "system_prompt":"sys",
        "user_prompt":"question",
        "rag_text":"ctx",
        "return_json": False
    }, project_root)

    pr = prepare_request(project_root, spec)
    assert pr.messages[-1]["role"] == "user"
    assert "Reference Context" in pr.messages[-1]["content"]

def test_rag_injection_openai_developer():
    project_root = Path(__file__).resolve().parents[1]
    spec = parse_input({
        "provider":"openai",
        "model":"gpt-4o-mini",
        "system_prompt":"sys",
        "user_prompt":"question",
        "rag_text":"ctx",
        "return_json": False
    }, project_root)

    pr = prepare_request(project_root, spec)
    roles = [m["role"] for m in pr.messages]
    assert "developer" in roles
