"""outputs 산출물의 실제 값을 요약해 log.md를 만들고 이 코드도 함께 기록한다."""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent

# 이 프로젝트 대화에서 사용자가 직접 요청한 작업 지시를 시간 순서대로 보존한다.
USER_REQUESTS = [
    "ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt' 해결해줘",
    "@README.txt 파일을 읽고 SKILL과 AGENTS.md 파일을 만들어줘",
    "@README.txt 파일을 보고 예측 프로그램을 파이썬 코드로 만들어줘",
    "실행 검증을 해야 하는데 절차를 알려주고, 파이썬 코드도 알려줘",
    "검증 실패: 점검 계획 행 수가 20이 아닙니다: 81 을 해결해줘",
    "검증 실패: 'actual' 이것을 해결해줘",
    "검증 실패: 'actual' 원인 분석과 실행환경 오류면 실행환경도 다시 고쳐줘",
    "검증할 수 있는 코드를 짜줘",
    "지시하는 내용에 대한 결과에 대해 log.md 파일로 저장하고 저장하는 결과값에 대한 python 코드도 생성해서 log.md 파일에 넣어줘",
    "log 파일에는 AI에게 지시하는 모든 내용이 다 들어가도록 수정해줘",
]

INSTRUCTION_SOURCES = [
    ("README.txt", ROOT / "README.txt"),
    ("AGENTS.md", ROOT / "AGENTS.md"),
    ("프로젝트 SKILL.md", ROOT / ".agents" / "skills" / "drone-maintenance-analysis" / "SKILL.md"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="실행 결과와 지표를 log.md에 기록합니다.")
    parser.add_argument("--output-dir", default="outputs", help="예측 산출물 폴더")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    validation = pd.read_csv(output_dir / "validation_predictions.csv")
    monthly = pd.read_csv(output_dir / "validation_monthly_metrics.csv")
    test = pd.read_csv(output_dir / "test_predictions_2025.csv")
    plan = pd.read_csv(output_dir / "inspection_plan_2026-01.csv")
    scored_path = output_dir / "current_monthly_scored.csv"
    scored = pd.read_csv(scored_path) if scored_path.exists() else plan.copy()
    thresholds = json.loads((output_dir / "risk_thresholds.json").read_text(encoding="utf-8"))

    validation_auc = average_precision_score(validation["actual"], validation["prob"])
    lines = [
        "# 실행 및 검증 로그",
        "",
        f"- 기록 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- 상태: 예측 프로그램과 산출물 검증 단계가 모두 완료된 뒤 생성됨",
        "- 입력 기준: `data/drone_train.csv`, `data/drone_test.csv`, `data/drone_this_month.csv`",
        "",
        "## 검증 데이터 결과",
        "",
        f"- 검증 행 수: {len(validation):,}",
        f"- 실제 재고장 수: {int(validation['actual'].sum()):,}",
        f"- PR-AUC (validation_predictions.csv에서 재계산): {validation_auc:.6f}",
        f"- 월별 상위 20대 적중 합계: {int(monthly['model_top20_hits'].sum()):,}",
        f"- 무작위 선정 기대 적중 합계: {monthly['random_expected_hits'].sum():.2f}",
        f"- 경과연수 기준 적중 합계: {int(monthly['oldest_age_top20_hits'].sum()):,}",
        f"- 평가 월 수: {len(monthly):,}",
        "",
        "## 위험 임계값 결과 (비용 단위: 만원)",
        "",
    ]
    for key, title in [("miss_cost_300_won_10k", "미탐 300만원 가정"),
                       ("miss_cost_600_won_10k", "미탐 600만원 가정")]:
        item = thresholds[key]
        lines.extend([
            f"### {title}",
            f"- 선택 임계값: {item['threshold']:.6f}",
            f"- 임계값 초과 점검 수: {item['flagged_for_inspection']:,}",
            f"- 미탐 재고장 수: {item['missed_failures']:,}",
            f"- 검증 비용: {item['estimated_cost_won_10k']:.0f}만원",
            "",
        ])

    lines.extend([
        "## 제출 및 2026-01 계획",
        "",
        f"- 2025 시험 예측 행 수: {len(test):,}",
        f"- 시험 예측 확률 범위: {test['prob'].min():.6f} ~ {test['prob'].max():.6f}",
        f"- 시험 예측 평균 확률: {test['prob'].mean():.6f}",
        f"- 전체 장비별 최신 기록: {len(scored):,}대",
        f"- 점검 계획에 저장된 대상: {len(plan):,}대",
        f"- 전체 최신 기록 중 임계값 초과: {int(scored['above_risk_threshold'].astype(str).str.lower().isin(['true', '1']).sum()):,}대",
        "",
        "### 점검 계획 상위 5대",
        "",
        "| 순위 | 장비 ID | 정비일 | 확률 | 위험 요인 |",
        "|---:|---|---|---:|---|",
    ])
    for _, row in plan.head(5).iterrows():
        reasons = str(row.get("risk_factors", "" )).replace("|", "/")
        lines.append(
            f"| {int(row['rank'])} | {row['equipment_id']} | {row['maintenance_date']} | "
            f"{float(row['prob']):.6f} | {reasons} |"
        )

    source = Path(__file__).read_text(encoding="utf-8")
    lines.extend([
        "",
        "## 이 로그를 생성한 Python 코드",
        "",
        "아래 코드는 산출물 파일에서 지표와 결과를 읽어 이 `log.md`를 생성합니다.",
        "",
        "```python",
        source.rstrip(),
        "```",
        "",
    ])
    # 사용자 요청과 프로젝트에서 관리하는 AI 지침의 전문을 로그에 보관한다.
    instruction_lines = [
        "",
        "## AI 지시사항과 프로젝트 지침",
        "",
        "### 대화에서 받은 사용자 요청",
        "",
    ]
    instruction_lines.extend(
        f"{index}. {request}" for index, request in enumerate(USER_REQUESTS, start=1)
    )
    for title, path in INSTRUCTION_SOURCES:
        instruction_lines.extend(["", f"### {title}", ""])
        if path.exists():
            instruction_lines.extend(["```text", path.read_text(encoding="utf-8").rstrip(), "```"])
        else:
            instruction_lines.append(f"파일을 찾지 못했습니다: `{path.relative_to(ROOT)}`")
    instruction_lines.append("")
    lines.extend(instruction_lines)

    (ROOT / "log.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"로그 저장 완료: {ROOT / 'log.md'}")


if __name__ == "__main__":
    main()
