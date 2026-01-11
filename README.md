# my_llmkit (Layer1)

This project provides a minimal, reusable Python module to call multiple LLM providers
(OpenAI-style, Gemini REST) using a single internal input format.

Scope:
- Layer1: single request -> single provider call
- Optional: RAG context injection if `rag_id` / `rag_text` is provided (file-based resolver only)
- No fallback logic (planned later)
- No multi-turn state management (caller owns conversation history)

## Environment variables

### OpenAI-style
- OPENAI_API_KEY
- YTL_API_KEY (if you use provider=ytl)

Optional endpoint overrides:
- OPENAI_ENDPOINT (default: https://api.openai.com/v1/chat/completions)
- YTL_ENDPOINT (default: https://api.ytlailabs.tech/v1/chat/completions)

### Gemini
- GEMINI_API_KEY

## Input JSON format (recommended)

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "system_prompt": "@src/prompts/system_default.txt",
  "user_prompt": "hi",
  "rag_id": "",
  "rag_text": "",
  "return_json": true,
  "strict": false,
  "options": {
    "max_tokens": 1000,
    "temperature": 0.0,
    "top_p": 1.0
  }
}
```

### system_prompt / user_prompt values
- Plain string: "You are ..."
- File reference: "@path/to/file.txt"

## RAG injection rules (Layer1)
- If provider=openai:
  - inject RAG context as a `developer` message (right after system message)
- Else:
  - merge RAG context into the last user message with tags:
    "### Reference Context:" and "### User Question:"

RAG text resolution (Layer1):
- If `rag_text` is provided, use it directly
- Else if `rag_id` is provided:
  - try `LLMKIT_RAG_DIR/<rag_id>.txt`
  - else try `src/rag/<rag_id>.txt`

## Strict mode
- If `strict=true`, the caller must be allowed by `src/configs/allowlist.yaml`
- If the allowlist file is missing or empty, strict mode blocks the call.

## Output
The module returns a dict:
- raw: provider-specific response (or normalized raw for Gemini)
- view: a lightweight normalized view {provider, model, text, json, meta}
- parse_error: error message if JSON parsing/coercion failed (when return_json=true)
- meta: internal meta including step timings and where errors occurred
