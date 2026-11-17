"""드론 학습 CSV의 결측치를 열별로 집계하고 결과를 출력·저장한다.

실행:
    .venv/Scripts/python.exe analyze_missing.py
    .venv/Scripts/python.exe analyze_missing.py --input data/drone_train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 데이터 결측치 요약")
    parser.add_argument("--input", default="data/drone_train.csv", help="학습 CSV 경로")
    parser.add_argument(
        "--output", default="outputs/missing_values_train.csv", help="요약 CSV 저장 경로"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    data = pd.read_csv(input_path)
    missing_count = data.isna().sum()
    summary = pd.DataFrame({
        "column": data.columns,
        "dtype": [str(dtype) for dtype in data.dtypes],
        "missing_count": missing_count.to_numpy(dtype=int),
        "missing_pct": (missing_count / len(data) * 100).to_numpy(),
        "non_missing_count": (len(data) - missing_count).to_numpy(dtype=int),
    }).sort_values(["missing_count", "column"], ascending=[False, True])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    missing_rows = int(data.isna().any(axis=1).sum())
    total_cells = int(data.shape[0] * data.shape[1])
    total_missing = int(missing_count.sum())
    print(f"파일: {input_path}")
    print(f"데이터 크기: {len(data):,}행 × {len(data.columns):,}열")
    print(f"결측 셀이 하나 이상 있는 행: {missing_rows:,}행 ({missing_rows / len(data) * 100:.2f}%)")
    print(f"전체 결측 셀: {total_missing:,}개 / {total_cells:,}개 ({total_missing / total_cells * 100:.2f}%)")
    print("\n열별 결측 현황:")
    print(summary[["column", "missing_count", "missing_pct"]].to_string(
        index=False,
        formatters={"missing_pct": lambda value: f"{value:.2f}%"},
    ))
    print(f"\nCSV 저장: {output_path}")


if __name__ == "__main__":
    main()
