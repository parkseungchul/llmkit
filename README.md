# llmkit (layer1) + minimal test runner

Goal (layer1)
- Provider-agnostic call with a single JSON case format.
- Minimal ON/OFF tests where ONLY ONE option is enabled at a time:
  - system_prompt ON (otherwise empty)
  - rag ON (otherwise empty)
  - return_json ON (otherwise false)

Providers
- openai (OpenAI Chat Completions)
- ytl (OpenAI-compatible endpoint)
- gemini (Google Generative Language API generateContent)

Run (Windows Git Bash)
1) Create venv and install
   python -m venv .venv
   source .venv/Scripts/activate
   pip install .

2) Set env vars
   export OPENAI_API_KEY="..."
   export YTL_API_KEY="..."
   export GEMINI_API_KEY="..."

3) Run a case file
   ./run.sh tests/cases/openai_one_option.json
   ./run.sh tests/cases/ytl_one_option.json
   ./run.sh tests/cases/gemini_one_option.json

Notes
- Fails fast if the required API key is missing.
- If return_json=true, it tries to parse JSON; if it fails, it returns raw_text + parse_error.
