# 실행 및 검증 로그

- 기록 시각: 2026-11-17T15:18:44+09:00
- 상태: 예측 프로그램과 산출물 검증 단계가 모두 완료된 뒤 생성됨
- 입력 기준: `data/drone_train.csv`, `data/drone_test.csv`, `data/drone_this_month.csv`

## 검증 데이터 결과

- 검증 행 수: 643
- 실제 재고장 수: 176
- PR-AUC (validation_predictions.csv에서 재계산): 0.799272
- 월별 상위 20대 적중 합계: 94
- 무작위 선정 기대 적중 합계: 32.54
- 경과연수 기준 적중 합계: 50
- 평가 월 수: 6

## 위험 임계값 결과 (비용 단위: 만원)

### 미탐 300만원 가정
- 선택 임계값: 0.484503
- 임계값 초과 점검 수: 237
- 미탐 재고장 수: 30
- 검증 비용: 23220만원

### 미탐 600만원 가정
- 선택 임계값: 0.229905
- 임계값 초과 점검 수: 365
- 미탐 재고장 수: 13
- 검증 비용: 29700만원

## 제출 및 2026-01 계획

- 2025 시험 예측 행 수: 1,241
- 시험 예측 확률 범위: 0.031311 ~ 0.983207
- 시험 예측 평균 확률: 0.386039
- 전체 장비별 최신 기록: 81대
- 점검 계획에 저장된 대상: 20대
- 전체 최신 기록 중 임계값 초과: 27대

### 점검 계획 상위 5대

| 순위 | 장비 ID | 정비일 | 확률 | 위험 요인 |
|---:|---|---|---:|---|
| 1 | DR-0115 | 2026-01-24 | 0.973800 | 고장 심각도 5; 정비 성공 여부가 실패; 최근 365일 이전 정비 54건; 직전 정비와 간격 1일; 고장 유형 모터; 이전 고장 유형 센서 |
| 2 | DR-0046 | 2026-01-30 | 0.854148 | 고장 심각도 2; 최근 365일 이전 정비 13건; 직전 정비와 간격 4일; 고장 유형 배터리; 이전 고장 유형 배터리 |
| 3 | DR-0018 | 2026-01-18 | 0.836596 | 고장 심각도 2; 최근 365일 이전 정비 10건; 직전 정비와 간격 3일; 고장 유형 모터; 이전 고장 유형 배터리 |
| 4 | DR-0038 | 2026-01-22 | 0.828976 | 고장 심각도 2; 정비 성공 여부가 실패; 최근 365일 이전 정비 13건; 직전 정비와 간격 12일; 고장 유형 센서; 이전 고장 유형 배터리 |
| 5 | DR-0061 | 2026-01-16 | 0.777954 | 고장 심각도 4; 최근 365일 이전 정비 11건; 직전 정비와 간격 22일; 고장 유형 기체; 이전 고장 유형 모터 |

## 이 로그를 생성한 Python 코드

아래 코드는 산출물 파일에서 지표와 결과를 읽어 이 `log.md`를 생성합니다.

```python
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
```


## AI 지시사항과 프로젝트 지침

### 대화에서 받은 사용자 요청

1. ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt' 해결해줘
2. @README.txt 파일을 읽고 SKILL과 AGENTS.md 파일을 만들어줘
3. @README.txt 파일을 보고 예측 프로그램을 파이썬 코드로 만들어줘
4. 실행 검증을 해야 하는데 절차를 알려주고, 파이썬 코드도 알려줘
5. 검증 실패: 점검 계획 행 수가 20이 아닙니다: 81 을 해결해줘
6. 검증 실패: 'actual' 이것을 해결해줘
7. 검증 실패: 'actual' 원인 분석과 실행환경 오류면 실행환경도 다시 고쳐줘
8. 검증할 수 있는 코드를 짜줘
9. 지시하는 내용에 대한 결과에 대해 log.md 파일로 저장하고 저장하는 결과값에 대한 python 코드도 생성해서 log.md 파일에 넣어줘
10. log 파일에는 AI에게 지시하는 모든 내용이 다 들어가도록 수정해줘

### README.txt

```text
드론 예측정비 실습 안내
========================================================================

※ 이 데이터는 교육용 가상 데이터입니다. 실제 부대 · 기종 정보는 없습니다.


1. 분석 목적
------------------------------------------------------------------------
정비를 마친 드론 가운데 30일 안에 다시 고장 날 드론을 미리 골라,
이번 달(2026-01) 재점검할 드론 20대를 정한다.

  - 사용자     : 드론 정비반장
  - 예측 대상  : failure_within_30days
                 (정비 뒤 30일 안에 사후정비 · 긴급정비가 있었으면 1, 없으면 0)
  - 예측 시점  : 정비를 마친 직후
  - 업무 조건  : 한 달에 재점검할 수 있는 드론은 20대다.
                 재고장을 놓치면 1건에 300만원, 재점검은 1대에 60만원이 든다(가정).


2. 제공 데이터
------------------------------------------------------------------------
  drone_train.csv       2023-01 ~ 2024-12   2,346건   정답 있음   학습용
  drone_test.csv        2025-01 ~ 2025-12   1,241건   정답 없음   시험용(채점)
  drone_this_month.csv  2026-01               106건   정답 없음   이번 달 재점검용

  - 드론 150대(정찰 · 감시 · 수송 · 교육용), 가상 부대 12곳
  - 한 행이 정비 1건이다. 같은 드론의 정비 기록이 날짜순으로 여러 행 있다.
  - 시험 데이터의 정답은 강사가 갖고 있으며, 제출한 예측으로 채점한다.

  컬럼
  equipment_id                 드론 번호
  equipment_type               드론 유형
  unit_code                    부대 코드
  equipment_age_years          경과연수(년)
  maintenance_date             정비일
  maintenance_type             정비 구분(예방정비 · 사후정비 · 긴급정비)
  fault_code                   고장 코드
  fault_category               고장 유형(배터리 · 모터 · 통신 · 기체 · 센서)
  severity                     고장 심각도(0 ~ 5)
  downtime_hours               운용중단 시간(시간)
  repair_duration_hours        수리 시간(시간)
  parts_replaced_count         교체 부품 수
  maintenance_cost             정비 비용(만원)
  maintenance_success          정비 성공 여부(1 성공, 0 실패)
  cumulative_operating_hours   누적 비행시간(시간)
  days_since_last_maintenance  직전 정비 뒤 경과일
  availability_rate            그 달 가동률
  failure_within_30days        정답(학습 데이터에만 있음)


3. 실습에서 할 일
------------------------------------------------------------------------
[1] 데이터 살펴보기
    - 컬럼별 분포, 이상치, 결측을 확인한다.
    - 결측이 왜 생겼는지(입력 실수인지, 원래 값이 없는 것인지) 판단한다.
    - 드론 유형 · 계절 · 고장 유형 사이에 눈에 띄는 패턴을 찾는다.

[2] 데이터 처리 기준 세우기
    - 이상치 · 결측을 어떻게 처리할지 기준을 정하고, 그 근거를 적는다.
    - 행을 지우는 처리는 이유를 분명히 밝힌다.

[3] 피처 만들기
    - 드론별 지난 정비 기록으로 새 피처를 만든다.
      (예: 최근 일정 기간의 고장 수, 직전 고장 뒤 경과일, 예방정비 뒤 비행시간)
    - 그 정비일 이후의 기록은 쓰지 않는다.
    - 시험 · 이번 달 데이터의 피처는 앞 기간 기록과 이어서 계산해야 한다.

[4] 쓸 컬럼 고르기
    - 예측 시점(정비 직후)에 알 수 없는 값은 모델에 넣지 않는다.
    - 넣지 않기로 한 컬럼과 그 이유를 적는다.

[5] 모델 학습과 검증
    - 무작위로 나누지 말고 기간으로 나눈다.
      학습 데이터 안에서 뒤쪽 기간(예: 2024-07 ~ 12)을 검증 기간으로 쓴다.
    - 성능은 PR-AUC와 '매달 20대를 골랐을 때 실제 재고장 수'로 본다.
    - 반드시 기준선과 비교한다.
        · 무작위로 20대 고르기
        · 경과연수가 오래된 순으로 20대 고르기
    - 정확도 하나만으로 성능을 말하지 않는다.

[6] 위험 표시 기준 정하기
    - 검증 기간에서 '놓친 재고장 수 × 300 + 기준 이상 건수 × 60'(만원)이
      가장 작은 확률 기준을 고른다.
    - 놓친 재고장 비용이 600만원이면 기준이 어떻게 바뀌는지도 확인한다.
    - 시험 데이터(2025)로 기준을 고르지 않는다.

[7] 시험 기간 예측 제출
    - 최종 모델로 drone_test.csv 1,241건 모두에 재고장 확률을 매긴다.

[8] 이번 달 재점검 20대 선정
    - drone_this_month.csv로 2026-01 재점검 대상 20대를 확률 순으로 고른다.
    - 한 드론이 이번 달에 여러 번 정비했으면 가장 최근 기록으로 판단한다.
    - 상위 드론마다 위험을 올린 요인을 설명한다.


4. 제출물
------------------------------------------------------------------------
  (1) test_predictions_2025.csv  시험 기간 예측 (채점용)
        열: equipment_id, maintenance_date(YYYY-MM-DD), prob(0 ~ 1)
        drone_test.csv의 모든 행(1,241건)에 예측이 있어야 한다.

  (2) 분석 보고서
        - 데이터 처리 기준과 근거
        - 넣지 않은 컬럼과 이유
        - 나눈 기간, 성능(PR-AUC · 월 20대 적중)과 기준선 비교
        - 위험 표시 기준과 고른 방법(300만원 · 600만원 비교 포함)

  (3) 2026-01 드론 재점검 계획
        - 재점검 대상 20대: 순위, 드론, 유형, 부대, 확률, 위험을 올린 요인, 권고
        - 위험 표시 기준을 넘은 드론 수와 20대를 넘는 드론의 처리 방안
        - 모델의 한계(최종 판단은 정비 담당자가 한다는 점 포함)

  ※ 보고서의 모든 숫자는 직접 실행한 결과에서 가져오고, 어느 코드 · 파일에서
    나온 숫자인지 적는다.


5. 평가
------------------------------------------------------------------------
  - 시험 기간 PR-AUC, 매달 20대 적중 수(12개월 합)를 기준선과 비교해 평가한다.
  - 데이터 처리 · 컬럼 선택 · 기간 분할의 근거가 타당한지 본다.
  - 보고서 숫자가 제출한 결과와 일치하는지 대조한다.
```

### AGENTS.md

```text
# 프로젝트 지침

## 목표와 데이터

- 이 저장소는 README.txt에 명시된 드론 예측정비 실습을 위한 프로젝트다. 작업 전에 README.txt와 필요한 데이터 파일을 확인한다.
- 목표는 정비 직후 시점에서 `failure_within_30days`를 예측하고, 2026-01 데이터에서 월간 재점검 한도인 20대를 우선순위로 정하는 것이다.
- 데이터 파일은 `data/drone_train.csv`, `data/drone_test.csv`, `data/drone_this_month.csv`에 있다. 학습 데이터에만 정답이 있고, 테스트 및 이번 달 데이터에는 정답이 없다.
- 교육용 가상 데이터이므로 결과를 실제 장비·부대에 대한 사실로 표현하지 않는다.

## 분석 규칙

- 정비일 뒤에야 알 수 있는 값이나 이후 정비 기록을 예측 입력에 포함하지 않는다. 드론별 이력 피처는 각 예측 행의 정비일까지 관측 가능한 기록만 사용한다.
- 학습·검증은 시간 순서로 나눈다. 무작위 분할이나 테스트 연도(2025)를 이용한 모델·임계값 선택은 하지 않는다.
- 검증은 PR-AUC와 월 20대 선정 시 실제 재고장 수를 함께 보고한다. 무작위 선정과 경과연수 기준선을 비교하며, 정확도만으로 결론내리지 않는다.
- 위험 임계값은 검증 기간에서 `놓친 재고장 수 × 300 + 기준 이상 점검 수 × 60`(만원)을 기준으로 선택하고, 놓친 재고장 비용 600만원 가정도 비교한다. 테스트 데이터는 최종 평가에 남겨 둔다.
- 보고서의 수치는 직접 실행한 산출물에서 가져오고, 산출에 사용한 코드와 파일을 함께 밝힌다. 실행하지 않은 분석의 결과를 만들어내지 않는다.
- 2026-01 계획은 드론별 가장 최근 정비 기록을 사용하고, 상위 20대의 근거와 한계를 설명한다. 최종 정비 판단은 담당자가 한다.

## 산출물

- 제출용 예측 파일은 `test_predictions_2025.csv`로 만들고, 열은 `equipment_id`, `maintenance_date`(YYYY-MM-DD), `prob`(0~1) 순서로 한다. `drone_test.csv`의 모든 행을 포함한다.
- README.txt에 요구된 분석 보고서와 2026-01 재점검 계획을 제공한다. 별도 요청이 없으면 기존 데이터 파일을 수정하지 않는다.
- 분석 요청에는 프로젝트 스킬 `.agents/skills/drone-maintenance-analysis/SKILL.md`를 참고한다.

## Streamlit 앱 (`app.py`)

- 이 프로젝트의 대시보드 진입 파일은 저장소 루트의 `app.py`다. 파일이 없으면 예측정비 분석 화면을 만들고, 실행할 수 있도록 한다.
- 경로는 현재 작업 디렉터리에 의존하지 말고 `Path(__file__).resolve().parent`를 기준으로 `data/`와 `outputs/`를 찾는다.
- 화면에는 학습 데이터 결측 요약, 검증 성능과 기준선, 위험 임계값, 2025 예측 결과, 2026-01 점검 계획을 표시한다. 해당 산출물이 없으면 없는 상태를 안내하고 임의의 숫자를 만들지 않는다.
- 앱을 여는 것만으로 모델 학습이나 CSV 덮어쓰기를 실행하지 않는다. 데이터 분석·학습 작업은 사용자가 명시적으로 실행할 때만 수행한다.
- 화면 데이터는 CSV/JSON 산출물에서 읽고, 보고서 숫자와 일치하도록 한다. 교육용 가상 데이터이며 최종 정비 판단은 담당자가 한다는 점을 표시한다.
- Streamlit은 `requirements.txt`에서 관리한다. 권장 실행은 `python -m streamlit run app.py`이며, 프로젝트 가상환경의 Python으로 실행한다.
- 앱 제작·수정 시 `.agents/skills/drone-maintenance-analysis/SKILL.md`의 Streamlit 대시보드 지침을 따른다.
```

### 프로젝트 SKILL.md

```text
---
name: drone-maintenance-analysis
description: Analyze this project's drone maintenance data, build leakage-safe failure predictions, evaluate time-based models, prepare the submission and inspection plan, or create the Streamlit dashboard in app.py.
---

# 드론 예측정비 분석

이 스킬은 저장소의 드론 정비 데이터 분석, 재고장 예측, 결과 제출 작업에 사용한다. 작업 목표와 제출 기준은 저장소 루트의 `README.txt`를 따른다.

## Streamlit 대시보드 (`app.py`)

앱을 만들거나 수정할 때 프로젝트 루트의 `AGENTS.md`와 이 문서의 분석 기준을 따른다. 화면에서 분석 결과를 읽기 쉽게 확인하도록 만들며, 앱 로딩 중 모델 학습이나 결과 파일 생성을 자동 실행하지 않는다.

- 프로젝트 경로는 `Path(__file__).resolve().parent` 기준으로 해석한다. 입력은 `data/`, 산출물은 `outputs/`에서 찾는다.
- 결측치 화면은 `data/drone_train.csv`에서 결측 개수와 열별 비율을 계산하거나 `outputs/missing_values_train.csv`를 사용한다. 계산한 경우 원자료 행 수를 함께 보여준다.
- 검증 화면은 PR-AUC와 월별 상위 20대 적중 수를 무작위 기대 및 경과연수 기준선과 나란히 보여준다.
- 임계값 화면은 검증에서 정한 300만원·600만원 미탐 비용 가정의 기준, 추정 비용, 점검 건수를 표시한다.
- 제출 화면은 2025 예측 확률 분포와 2026-01 점검 대상 순위·확률·위험 요인을 보여준다. 점검 대상은 장비별 최신 정비 기록을 기준으로 한다.
- 파일이 없거나 열 구성이 맞지 않으면 해당 화면에서 필요한 파일 또는 재실행 명령을 안내한다. 결과 수치를 하드코딩하거나 없는 값을 추정하지 않는다.
- CSV에서 정비일을 읽을 때 날짜 형식을 확인하고, 표시할 때 ID·확률·날짜를 명확히 구분한다. 그래프는 표를 보완할 때만 사용한다.
- 실행 명령은 프로젝트 가상환경을 기준으로 `python -m streamlit run app.py`를 안내한다. Streamlit 의존성은 `requirements.txt`에 선언한다.
- 교육용 가상 데이터라는 점을 표시하고, 모델이 최종 정비 결정을 대신하지 않는다고 알린다.

## 데이터와 예측 시점

- `data/drone_train.csv`: 2023-01~2024-12, 타깃 `failure_within_30days` 포함.
- `data/drone_test.csv`: 2025년, 제출 대상이며 타깃 없음.
- `data/drone_this_month.csv`: 2026-01, 재점검 우선순위 대상이며 타깃 없음.
- 한 행은 정비 한 건이며 같은 `equipment_id`가 반복될 수 있다. 날짜순으로 정렬하고 키 중복과 날짜 범위를 점검한다.
- 예측 시점은 정비 직후다. 그 시점에 확인할 수 없는 사후 정보와 예측 행 날짜 뒤의 이력을 쓰지 않는다. 과거 이력 피처를 만들 때 학습·검증·시험·이번 달 기록을 날짜순으로 연결하되, 각 행은 그 시점 이전 정보만 참조하게 한다.

## 분석 흐름

1. 열 이름, 자료형, 행 수, 기간, 결측, 중복, 타깃 비율을 확인한다. 결측·이상치 처리 기준은 데이터 의미에 근거해 정하고 보고서에 이유를 남긴다.
2. 예측 시점에 알 수 없는 열은 제외하고 이유를 기록한다. 전처리와 피처 선택은 검증 데이터를 미리 보지 않도록 학습 기간 안에서 정한다.
3. 무작위 분할 대신 시간 순서로 학습과 검증을 나눈다. README.txt의 예시인 2024-07~12 검증 구간을 고려하고, 실제 사용한 경계를 명시한다.
4. 기준선(무작위 월 20대, 경과연수 내림차순 월 20대)과 모델을 비교한다. PR-AUC와 월별 상위 20대의 실제 재고장 수를 보고하고, 집계 방식과 표본 수를 적는다.
5. 검증 구간에서 비용을 계산해 위험 임계값을 정한다: `미탐 재고장 수 × 300 + 기준 이상 점검 수 × 60`만원. 미탐 비용을 600만원으로 바꾼 비교도 제공한다. 검증에서 정한 뒤 테스트에 맞춰 임계값을 조정하지 않는다.
6. 최종 모델을 학습해 테스트 각 행의 확률을 산출한다. `test_predictions_2025.csv`는 `equipment_id, maintenance_date, prob` 열로 구성하고 모든 테스트 행을 포함한다.
7. 2026-01 계획은 드론별 가장 최근 기록으로 중복을 정리한 뒤 확률 상위 20대를 제시한다. 순위, 드론, 유형, 부대, 확률, 위험 요인, 권고를 포함하고 임계값 초과 수와 20대 초과 시 대응을 설명한다.

## 보고와 해석

- 보고서에는 전처리 근거, 제외 열과 이유, 기간 분할, PR-AUC, 월 20대 적중, 기준선, 두 비용 가정의 임계값 비교를 포함한다.
- 수치는 실행 산출물에서만 가져온다. 표와 설명이 제출 CSV 및 분석 결과와 일치하는지 확인하고, 수치가 나온 코드·파일을 표시한다.
- 결과는 교육용 가상 데이터에 대한 실습 결과로 한정한다. 모델의 불확실성과 한계를 설명하며 최종 정비 결정은 담당자가 내린다고 명시한다.
```
