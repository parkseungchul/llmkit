# src/utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

def coerce_json_object_text(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    LLM이 준 텍스트에서 JSON object(dict)를 최대한 파싱해본다.

    Returns:
      (ok, obj, err)
        - ok: 파싱 성공 여부
        - obj: 성공 시 dict, 실패 시 None
        - err: 실패 시 에러 메시지, 성공 시 None

    정책:
      - 이미 dict면 그대로 성공 처리
      - 문자열이면:
          1) json.loads 그대로 시도
          2) ```json ... ``` 코드펜스 제거 후 재시도
          3) 첫 '{' ~ 마지막 '}' 잘라내서 재시도
      - 그래도 실패하면 ok=False로 반환 (raw는 caller가 보관)
    """
    if text is None:
        return False, None, "empty text"

    if isinstance(text, dict):
        return True, text, None

    s = str(text).strip()
    if not s:
        return False, None, "empty text"

    # 1) direct
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return True, obj, None
        return False, None, f"json is not object: {type(obj).__name__}"
    except Exception as e1:
        pass

    # 2) code fence 제거
    t = s
    if t.startswith("```"):
        # ```json 또는 ``` 로 시작하는 경우
        t = t.lstrip("`")
        # 첫 줄 제거
        parts = s.splitlines()
        if len(parts) >= 2:
            t = "\n".join(parts[1:])
        # 끝의 ``` 제거
        if t.rstrip().endswith("```"):
            t = t.rstrip()
            t = t[: -3]
        t = t.strip()

        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return True, obj, None
            return False, None, f"json is not object after fence strip: {type(obj).__name__}"
        except Exception:
            pass

    # 3) 중괄호 구간만 추출
    i = s.find("{")
    j = s.rfind("}")
    if 0 <= i < j:
        candidate = s[i : j + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return True, obj, None
            return False, None, f"json is not object after brace slice: {type(obj).__name__}"
        except Exception as e3:
            return False, None, f"json parse failed: {e3}"

    return False, None, "no json object braces found"


def sanitize_api_key(raw: str) -> str:
    """
    헤더 인코딩(latin-1) 문제/복사 실수로 들어오는 따옴표 문제 방지용.
    - 앞뒤 공백 제거
    - 일반 따옴표/백틱 제거
    - 스마트 따옴표(“ ” ‘ ’) 제거
    """
    k = (raw or "").strip()
    k = k.strip(' "\'`')
    k = k.strip("“”‘’")
    return k

def project_root() -> Path:
    """
    프로젝트 루트 디렉토리를 반환한다.

    전제:
      - 이 파일은 <repo_root>/src/utils.py 에 위치한다.
      - 따라서 repo_root == src 폴더의 부모 폴더

    예:
      repo_root/
        src/
          utils.py   <- this file
        tests/
        run.py
    """
    return Path(__file__).resolve().parents[1]


def read_text_maybe_file(
    value: str,
    base_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
) -> str:
    """
    value가 '@...' 형태면 파일로 읽고, 아니면 value 자체를 텍스트로 반환한다.

    - '@relative/or/abs/path.txt' 형태를 지원한다.
    - 상대경로인 경우 탐색 순서:
        1) base_dir / path
        2) root_dir / path  (root_dir 미지정이면 project_root() 사용)
        3) path 그대로 (현재 작업 디렉토리 기준)

    왜 이런 순서냐:
      - 테스트 케이스 JSON이 있는 위치(base_dir) 기준 상대경로를 우선 지원
      - 그 다음, repo 전체 기준(root_dir) 상대경로도 지원 (예: @tests/assets/..)
      - 마지막으로, 사용자가 cwd 기준으로 실행하는 경우도 흡수

    반환:
      - '@'가 아니면 원문 문자열(value)
      - '@'면 파일 텍스트(utf-8)

    예외:
      - 파일이 없으면 FileNotFoundError
    """
    v = (value or "").strip()
    if not v.startswith("@"):
        return v

    path_str = v[1:].strip()
    if not path_str:
        return ""

    p = Path(path_str)
    tried: list[str] = []

    # 0) absolute path면 그대로
    if p.is_absolute():
        tried.append(str(p))
        if p.exists():
            return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No such file: {p}")

    # 1) base_dir 기준
    if base_dir is not None:
        cand = (Path(base_dir) / p).resolve()
        tried.append(str(cand))
        if cand.exists():
            return cand.read_text(encoding="utf-8")

    # 2) repo root 기준
    if root_dir is None:
        root_dir = project_root()

    cand = (Path(root_dir) / p).resolve()
    tried.append(str(cand))
    if cand.exists():
        return cand.read_text(encoding="utf-8")

    # 3) cwd 기준(그대로)
    cand = p.resolve()
    tried.append(str(cand))
    if cand.exists():
        return cand.read_text(encoding="utf-8")

    raise FileNotFoundError(f"No such file: {path_str}. Tried: {tried}")
