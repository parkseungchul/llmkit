"""Simple CLI for Layer1.

Usage examples:
- JSON string input:
  python cli.py "{\"provider\":\"openai\",\"model\":\"gpt-4o-mini\",\"user_prompt\":\"hi\"}"

- JSON file input (prefix with @):
  python cli.py @request.json

- Pretty print:
  python cli.py @request.json --pretty

Notes:
- This CLI does not manage multi-turn or fallback. It is strictly a single call executor.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.llmkit.client import LLMClient
from src.llmkit.logging_util import get_logger

logger = get_logger(__name__)

def _load_input(spec: str) -> Dict[str, Any]:
    if spec.startswith("@"):
        p = Path(spec[1:])
        data = p.read_text(encoding="utf-8")
        return json.loads(data)

    return json.loads(spec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSON string or @path/to/json")
    ap.add_argument("--pretty", action="store_true", help="Pretty print the output JSON")
    args = ap.parse_args()

    try:
        req = _load_input(args.input)
    except Exception as e:
        logger.error("Failed to parse input: %s", e)
        sys.exit(2)

    client = LLMClient()
    out = client.run(req, request_id="CLI")

    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
