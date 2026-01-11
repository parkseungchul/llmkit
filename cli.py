"""Repo-root convenience entrypoint.

설치(pip install -e .) 없이도 아래처럼 실행할 수 있게 해준다.

  python cli.py run tests/cases/test01_openai_basic.json

실제 구현은 패키지 모듈 `llmkit.cli`에 있다.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # src/ 레이아웃이므로, repo-root에서 실행할 때 src를 sys.path에 추가
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from llmkit.cli import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
