"""예측 프로그램을 실행하고 생성된 산출물을 바로 검증합니다.

실행 예:
    .venv/Scripts/python.exe run_and_validate.py
    .venv/Scripts/python.exe run_and_validate.py --validation-start 2024-07-01
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="드론 예측을 새로 실행하고 결과를 검증합니다.")
    parser.add_argument("--data-dir", default="data", help="입력 CSV 폴더")
    parser.add_argument("--output-dir", default="outputs", help="결과 저장 폴더")
    parser.add_argument("--validation-start", default="2024-07-01", help="검증 기간 시작일")
    args = parser.parse_args()

    shared_args = [
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--validation-start", args.validation_start,
    ]
    commands = [
        ([sys.executable, str(ROOT / "predict_maintenance.py"), *shared_args], "예측 산출물 생성"),
        ([sys.executable, str(ROOT / "validate_outputs.py"), *shared_args], "산출물 검증"),
        ([sys.executable, str(ROOT / "write_run_log.py"), "--output-dir", args.output_dir], "log.md 기록"),
    ]

    for command, step in commands:
        print(f"\n=== {step} ===", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            print(f"실패: {step} (종료 코드 {completed.returncode})", file=sys.stderr)
            return completed.returncode

    print("\n전체 실행 및 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
