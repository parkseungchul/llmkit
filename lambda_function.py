"""AWS Lambda entrypoint.

Design goals:
- Keep this file small and stable.
- Delegate all real logic to src/llmkit so that:
  - The same codebase can be used from CLI and from Lambda.
  - Layer2(RAG DB) / Layer3(multi-turn) can be added later without touching Lambda glue.

Expected event shapes (minimal):
1) API Gateway (body is a JSON string):
   {"body": "{\"provider\":\"openai\",\"model\":\"gpt-4o-mini\",\"user_prompt\":\"hi\"}"}

2) Direct invoke / local test (event itself is the JSON dict):
   {"provider":"openai","model":"gpt-4o-mini","user_prompt":"hi"}

Return:
- statusCode: 200 unless input parsing fails badly
- body: JSON string of {"raw":..., "view":..., "parse_error":..., "meta":...}
"""
import json
from typing import Any, Dict

from src.llmkit.client import LLMClient
from src.llmkit.logging_util import get_logger

logger = get_logger(__name__)

_client = LLMClient()

def _safe_json_loads(s: Any):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def lambda_handler(event: Dict[str, Any], context: Any):
    try:
        body = event.get("body", event)
        req = _safe_json_loads(body)

        result = _client.run(req, request_id=getattr(context, "aws_request_id", None))
        return {"statusCode": 200, "body": json.dumps(result, ensure_ascii=False)}

    except Exception as e:
        logger.exception("lambda_handler fatal error: %s", e)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {"raw": None, "view": None, "parse_error": str(e), "meta": {"where": "lambda_handler"}},
                ensure_ascii=False,
            ),
        }
