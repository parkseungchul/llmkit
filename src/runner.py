from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from src.adapters import build_adapter
from src.errors import ConfigError, ValidationError
from src.logging_config import logger
from src.prompt_builder import build_messages
from src.schema import LlmRequest, normalize_provider, validate_nonempty
from src.utils import coerce_json_object_text, read_text_maybe_file


# ----------------------------------------------------------------------
# Config / Allowlist
# ----------------------------------------------------------------------
def _load_allowlist(_repo_root: Path | None = None) -> Dict[str, Any]:
    """
    allowlist는 프로젝트 표준으로 ./src/allowlist.json 에만 둔다.
    - 호출부 호환을 위해 _repo_root 인자를 받지만, 실제 경로 계산에는 사용하지 않는다.
    """
    p = Path(__file__).resolve().parent / "allowlist.json"  # == ./src/allowlist.json
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ConfigError(f"Failed to load allowlist.json: {e}")


def _check_allowlist(provider: str, model: str, allowlist: Dict[str, Any]) -> None:
    providers = (allowlist or {}).get("providers") or {}
    allowed_models = providers.get(provider) or []
    if model not in allowed_models:
        raise ValidationError(f"Model not allowed: provider={provider} model={model}")


def _required_env(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "ytl":
        return "YTL_API_KEY"
    if provider == "gemini":
        return "GEMINI_API_KEY"
    return ""


def _fail_fast_env(case_id: str, provider: str) -> None:
    import os

    env = _required_env(provider)
    if env and not (os.getenv(env) or "").strip():
        raise ValidationError(f"[{case_id}] Missing required env var: {env}")


# ----------------------------------------------------------------------
# Output (console + file)
# ----------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    import os

    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "y", "yes", "on")


def _safe_filename(s: str) -> str:
    """
    파일명 안전화:
    - Windows/Unix 모두에서 문제 될만한 문자 제거/치환
    """
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]  # 너무 길면 잘라냄


def _get_output_path(repo_root: Path, provider: str, model: str, run_ts: str) -> Path:
    """
    output/{provider}_{model}_{YYYYMMDD_HHMMSS}.jsonl
    """
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{_safe_filename(provider)}_{_safe_filename(model)}_{run_ts}.jsonl"
    return out_dir / fn


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

def _one_line_preview(s: str, limit: int = 200) -> str:
    """
    콘솔용 1줄 요약:
    - 개행/탭은 공백으로
    - 연속 공백 정리
    - limit 초과시 ... 처리
    """
    if s is None:
        return ""
    t = str(s).replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        return t[:limit].rstrip() + " ..."
    return t

def _pretty_print(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------
# Case parsing
# ----------------------------------------------------------------------
def _parse_case(defaults: Dict[str, Any], c: Dict[str, Any], base_dir: str) -> LlmRequest:
    def pick(key: str) -> Any:
        v = c.get(key, "")
        if v == "" or v is None:
            v = defaults.get(key, "")
        return v

    env = str(pick("env") or "").strip()
    provider = normalize_provider(pick("provider"))
    model = str(pick("model") or "").strip()

    system_prompt = read_text_maybe_file(str(pick("system_prompt") or ""), base_dir=base_dir)
    user_prompt = read_text_maybe_file(str(pick("user_prompt") or ""), base_dir=base_dir)

    rag_id = str(pick("rag_id") or "").strip()
    rag_text = read_text_maybe_file(str(pick("rag_text") or ""), base_dir=base_dir)

    return_json = bool(pick("return_json"))
    notice = str(pick("notice") or "")

    validate_nonempty("env", env)
    validate_nonempty("model", model)
    validate_nonempty("user_prompt", user_prompt)

    return LlmRequest(
        env=env,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        rag_id=rag_id,
        rag_text=rag_text,
        return_json=return_json,
        notice=notice,
    )


def _extract_text_and_usage(resp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    resp는 OpenAI-style로 normalize 된 형태를 기대한다.
    (Gemini adapter도 여기 맞춰서 choices/message/content로 만들어 주는 구조)
    """
    text = ""
    try:
        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        text = ""
    usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return text, usage


# ----------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------
def run_case_file(case_file: str, continue_on_error: bool = False) -> None:
    case_path = Path(case_file)
    base_dir = str(case_path.parent)

    # repo root: ./src/runner.py 기준 2단계 위
    repo_root = Path(__file__).resolve().parents[1]

    data = json.loads(case_path.read_text(encoding="utf-8"))
    defaults = data.get("defaults") or {}
    cases = data.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValidationError("cases must be a non-empty list")

    # allowlist는 무조건 ./src/allowlist.json에서 로드
    allowlist = _load_allowlist(repo_root)

    # 실행 단위 timestamp (한 번만 생성해서 파일명에 사용)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # provider_raw 저장 여부
    save_provider_raw = _env_flag("LLMKIT_SAVE_PROVIDER_RAW")

    logger.info("CASE FILE: %s", str(case_path))
    logger.info("TOTAL CASES: %d", len(cases))
    logger.info("SAVE_PROVIDER_RAW: %s (env LLMKIT_SAVE_PROVIDER_RAW=1 to enable)", save_provider_raw)
    logger.info("OUTPUT_DIR: %s", str((repo_root / "output").resolve()))

    for idx, c in enumerate(cases, start=1):
        case_id = c.get("id") or f"case_{idx:02d}"
        logger.info("------------------------------------------------------------")
        logger.info("[CASE %s] start", case_id)

        t0 = time.perf_counter()

        # 케이스 실행 중간에 provider/model을 알아야 파일 경로 결정 가능하므로,
        # req 파싱 후 output path를 잡는다.
        out_path: Path | None = None

        try:
            req = _parse_case(defaults, c, base_dir=base_dir)

            _fail_fast_env(case_id, req.provider)
            _check_allowlist(req.provider, req.model, allowlist)

            messages = build_messages(req)

            logger.info("[STEP 1] build request provider=%s model=%s env=%s", req.provider, req.model, req.env)
            logger.info(
                "[STEP 2] messages count=%d system=%s rag=%s return_json=%s",
                len(messages),
                bool((req.system_prompt or "").strip()),
                bool((req.rag_text or "").strip()),
                req.return_json,
            )

            adapter = build_adapter(req.provider)

            logger.info("[STEP 3] call provider")
            resp = adapter.generate(messages=messages, model=req.model, return_json=req.return_json)

            text, usage = _extract_text_and_usage(resp)

            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            envelope: Dict[str, Any] = {
                "meta": {
                    "case_id": case_id,
                    "env": req.env,
                    "provider": req.provider,
                    "model": req.model,
                    "notice": req.notice,
                    "return_json": req.return_json,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_ms": elapsed_ms,
                },
                "input": {
                    "env": req.env,
                    "provider": req.provider,
                    "model": req.model,
                    "system_prompt": req.system_prompt,
                    "rag_id": req.rag_id,
                    "rag_text": req.rag_text,
                    "user_prompt": req.user_prompt,
                    "messages": messages,  # 실제로 호출된 최종 messages 전체
                },
                "output": {
                    "raw_text": text,
                    "usage": usage,
                },
            }

            # return_json이면 parse 결과를 같이 넣음 (실패해도 raw_text는 그대로 유지)
            if req.return_json:
                ok, obj, err = coerce_json_object_text(text)
                if ok:
                    envelope["output"]["parsed_json"] = obj
                    envelope["output"]["parse_error"] = ""
                else:
                    envelope["output"]["parsed_json"] = None
                    envelope["output"]["parse_error"] = err or "unknown parse error"

            # provider_raw는 기본 OFF. 켜면 전체 저장/출력에 포함.
            if save_provider_raw:
                # adapters 쪽에서 _provider_raw를 붙여주면 그걸 우선 사용
                provider_raw = None
                if isinstance(resp, dict) and "_provider_raw" in resp:
                    provider_raw = resp.get("_provider_raw")
                else:
                    # 현재 구조상 최소한 resp 전체라도 보관(성능/로그 폭발은 사용자가 감수)
                    provider_raw = resp
                envelope["output"]["provider_raw"] = provider_raw

            logger.info("[STEP 4] result (console pretty print)")
            logger.info(f"[INPUT] {_one_line_preview(req.user_prompt, 200)}")
            logger.info(f"[OUTPUT] {_one_line_preview(text, 200)}")

            
            _pretty_print(envelope)

            # 파일 저장 (provider+model 별 jsonl)
            out_path = _get_output_path(repo_root, req.provider, req.model, run_ts)
            _append_jsonl(out_path, envelope)

            logger.info("[CASE %s] success (saved: %s)", case_id, str(out_path))


        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            # 에러도 envelope로 저장/출력 (나중에 분석/파싱 편하게)
            err_env: Dict[str, Any] = {
                "meta": {
                    "case_id": case_id,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_ms": elapsed_ms,
                },
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                },
            }

            logger.error("[CASE %s] failed: %s", case_id, e)
            _pretty_print(err_env)

            # provider/model이 아직 없을 수 있으니 기본 파일로 저장
            # (가능하면 케이스 json의 provider/model을 힌트로 파일을 고르게 함)
            hinted_provider = normalize_provider(c.get("provider") or defaults.get("provider") or "unknown")
            hinted_model = str(c.get("model") or defaults.get("model") or "unknown")
            out_path = _get_output_path(repo_root, hinted_provider, hinted_model, run_ts)
            _append_jsonl(out_path, err_env)

            if not continue_on_error:
                raise

    logger.info("DONE")
