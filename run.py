import sys
from src.runner import run_case_file

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run.py <case_file.json> [--continue]")
        return 2
    case_file = sys.argv[1]
    cont = "--continue" in sys.argv[2:]
    run_case_file(case_file, continue_on_error=cont)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
