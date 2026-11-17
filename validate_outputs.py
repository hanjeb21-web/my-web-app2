"""예측 파이프라인 산출물을 데이터와 대조하는 간단한 실행 검증기.

사용법:
    python validate_outputs.py
    python validate_outputs.py --validation-start 2024-07-01
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> pd.DataFrame:
    require(path.exists(), f"파일이 없습니다: {path}")
    return pd.read_csv(path)


def normalize_key(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame[["equipment_id", "maintenance_date"]].copy()
    normalized["equipment_id"] = normalized["equipment_id"].astype("string").fillna("UNKNOWN")
    normalized["maintenance_date"] = pd.to_datetime(
        normalized["maintenance_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    require(normalized["maintenance_date"].notna().all(), "날짜를 해석할 수 없는 키가 있습니다.")
    return normalized


def compare_key_counts(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> None:
    expected_counts = normalize_key(expected).value_counts().rename("expected")
    actual_counts = normalize_key(actual).value_counts().rename("actual")
    joined = pd.concat([expected_counts, actual_counts], axis=1).fillna(0)
    require((joined["expected"] == joined["actual"]).all(),
            f"{label}: (equipment_id, maintenance_date) 행이 원본과 일치하지 않습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="드론 예측 산출물 검증")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--validation-start", default="2024-07-01")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    train = read_csv(data_dir / "drone_train.csv")
    test = read_csv(data_dir / "drone_test.csv")
    current = read_csv(data_dir / "drone_this_month.csv")
    train["maintenance_date"] = pd.to_datetime(train["maintenance_date"], errors="coerce")
    require(train["maintenance_date"].notna().all(), "학습 데이터 날짜 오류")
    require(train["failure_within_30days"].dropna().isin([0, 1]).all(), "학습 타깃에 0/1 외 값이 있습니다.")

    # 시험 제출 파일: 스키마, 행 전체 포함 여부, 확률 유효성
    pred = read_csv(output_dir / "test_predictions_2025.csv")
    require(pred.columns.tolist() == ["equipment_id", "maintenance_date", "prob"],
            "test_predictions_2025.csv 열 또는 순서가 README 규격과 다릅니다.")
    require(len(pred) == len(test), f"시험 예측 행 수가 다릅니다: {len(pred)} != {len(test)}")
    compare_key_counts(test, pred, "시험 예측")
    pred["prob"] = pd.to_numeric(pred["prob"], errors="coerce")
    require(np.isfinite(pred["prob"]).all(), "예측 확률에 비어 있거나 무한한 값이 있습니다.")
    require(pred["prob"].between(0, 1).all(), "예측 확률이 0~1 범위를 벗어났습니다.")

    # 2026-01 계획: 장비별 최신 행 하나, 20대(데이터가 적으면 가능한 수), 점수순 정렬
    plan = read_csv(output_dir / "inspection_plan_2026-01.csv")
    needed = {"rank", "equipment_id", "maintenance_date", "prob", "risk_threshold", "above_risk_threshold"}
    require(needed.issubset(plan.columns), "점검 계획에 필수 열이 없습니다.")
    current["maintenance_date"] = pd.to_datetime(current["maintenance_date"], errors="coerce")
    current["equipment_id"] = current["equipment_id"].astype("string").fillna("UNKNOWN")
    expected_latest = current.sort_values(["equipment_id", "maintenance_date"]).drop_duplicates(
        "equipment_id", keep="last"
    )
    expected_n = min(20, expected_latest["equipment_id"].nunique())
    require(len(plan) == expected_n, f"점검 계획 행 수가 {expected_n}이 아닙니다: {len(plan)}")
    require(plan["equipment_id"].astype("string").nunique() == len(plan), "점검 계획에 장비 중복이 있습니다.")
    require(plan["rank"].tolist() == list(range(1, len(plan) + 1)), "순위가 1부터 연속되지 않습니다.")
    plan_prob = pd.to_numeric(plan["prob"], errors="coerce")
    require(np.isfinite(plan_prob).all() and plan_prob.between(0, 1).all(), "점검 계획 확률이 유효하지 않습니다.")
    require(plan_prob.is_monotonic_decreasing, "점검 계획이 확률 내림차순이 아닙니다.")
    # 계획 CSV는 20대만 포함하고, 전체 순위 파일은 장비별 최신 행을 모두 담는다.
    scored = read_csv(output_dir / "current_monthly_scored.csv")
    require(len(scored) == expected_latest["equipment_id"].nunique(),
            "전체 2026-01 점수 파일에 장비별 행이 하나씩 있지 않습니다.")
    compare_key_counts(expected_latest, scored, "2026-01 전체 장비 최신 기록")
    require(pd.to_numeric(scored["rank"], errors="coerce").tolist() == list(range(1, len(scored) + 1)),
            "전체 장비 점수 파일의 순위가 연속적이지 않습니다.")
    require(pd.to_numeric(scored["prob"], errors="coerce").is_monotonic_decreasing,
            "전체 장비 점수 파일이 확률 내림차순이 아닙니다.")
    compare_key_counts(scored.head(expected_n), plan, "상위 20대 점검 계획")
    threshold = pd.to_numeric(plan["risk_threshold"], errors="coerce")
    require(threshold.notna().all() and threshold.between(0, 1).all(), "점검 계획 임계값이 유효하지 않습니다.")
    expected_flag = plan_prob >= threshold
    actual_flag = plan["above_risk_threshold"].astype(str).str.lower().isin(["true", "1"])
    require((expected_flag.to_numpy() == actual_flag.to_numpy()).all(), "임계값 초과 표시가 확률과 일치하지 않습니다.")

    # 검증 지표의 월별 행 수와 양성 수를 학습 정답에서 독립적으로 대조
    metrics = read_csv(output_dir / "validation_monthly_metrics.csv")
    metric_cols = {"month", "rows", "actual_failures", "model_top20_hits", "random_expected_hits", "oldest_age_top20_hits"}
    require(metric_cols.issubset(metrics.columns), "월별 검증 지표 열이 부족합니다.")
    start = pd.Timestamp(args.validation_start)
    valid = train.loc[train["maintenance_date"] >= start].copy()
    valid["month"] = valid["maintenance_date"].dt.to_period("M").astype(str)
    expected_metrics = valid.groupby("month").agg(
        rows=("failure_within_30days", "size"),
        actual_failures=("failure_within_30days", "sum"),
    ).reset_index()
    checked = expected_metrics.merge(metrics, on="month", suffixes=("_expected", "_actual"), how="outer")
    require(not checked.isna().any().any(), "월별 검증 결과의 기간이 학습 자료와 다릅니다.")
    require((checked["rows_expected"] == checked["rows_actual"]).all(), "검증 월별 행 수가 학습 데이터와 다릅니다.")
    require((checked["actual_failures_expected"] == checked["actual_failures_actual"]).all(), "검증 월별 정답 수가 학습 데이터와 다릅니다.")
    require((metrics["model_top20_hits"].between(0, np.minimum(20, metrics["rows"]))).all(), "월별 상위 20대 적중 수가 범위를 벗어났습니다.")
    expected_random = metrics["actual_failures"] * np.minimum(20, metrics["rows"]) / metrics["rows"]
    require(np.allclose(metrics["random_expected_hits"], expected_random), "무작위 기준선 계산이 맞지 않습니다.")

    # 저장된 검증 예측으로 모델 점수와 월별 상위 20대 결과를 다시 계산
    valid_pred = read_csv(output_dir / "validation_predictions.csv")
    # Excel/수동 편집으로 헤더에 BOM이나 공백이 붙은 경우를 정규화한다.
    valid_pred.columns = [str(column).strip().lstrip("\ufeff") for column in valid_pred.columns]
    required_valid_columns = {"equipment_id", "maintenance_date", "prob"}
    require(required_valid_columns.issubset(valid_pred.columns),
            "validation_predictions.csv에 equipment_id, maintenance_date, prob 열이 필요합니다.")
    expected_valid = train.loc[train["maintenance_date"] >= start].copy()
    expected_valid = expected_valid.loc[expected_valid["failure_within_30days"].isin([0, 1])]
    require(len(valid_pred) == len(expected_valid), "검증 예측 행 수가 검증 구간 학습 행 수와 다릅니다.")
    compare_key_counts(expected_valid, valid_pred, "검증 예측")
    # 예전 실행에서 actual 열 없이 저장된 파일도 원본 학습 라벨로 보완한다.
    # 같은 장비·날짜가 중복이면 행별 정답을 안전하게 짝지을 수 없으므로 재생성을 안내한다.
    if "actual" not in valid_pred.columns:
        expected_keys = normalize_key(expected_valid)
        prediction_keys = normalize_key(valid_pred)
        require(not expected_keys.duplicated().any() and not prediction_keys.duplicated().any(),
                "actual 열이 없고 장비·정비일 키가 중복됩니다. predict_maintenance.py를 다시 실행하세요.")
        label_lookup = expected_keys.copy()
        label_lookup["actual"] = expected_valid["failure_within_30days"].astype(int).to_numpy()
        label_lookup = label_lookup.set_index(["equipment_id", "maintenance_date"])["actual"]
        valid_tuples = list(map(tuple, prediction_keys.to_numpy()))
        require(all(key in label_lookup.index for key in valid_tuples),
                "검증 예측을 원본 정답과 연결할 수 없습니다. predict_maintenance.py를 다시 실행하세요.")
        valid_pred["actual"] = [int(label_lookup.loc[key]) for key in valid_tuples]
    valid_pred["actual"] = pd.to_numeric(valid_pred["actual"], errors="coerce")
    valid_pred["prob"] = pd.to_numeric(valid_pred["prob"], errors="coerce")
    require(valid_pred["actual"].isin([0, 1]).all(), "검증 예측의 정답 값이 0/1이 아닙니다.")
    require(np.isfinite(valid_pred["prob"]).all() and valid_pred["prob"].between(0, 1).all(),
            "검증 예측 확률이 유효하지 않습니다.")
    valid_pred["maintenance_date"] = pd.to_datetime(valid_pred["maintenance_date"], errors="coerce")
    require(valid_pred["maintenance_date"].notna().all(), "검증 예측 날짜가 올바르지 않습니다.")
    # CSV 행 순서에 의존하지 않고 (장비, 날짜, 정답) 조합의 개수를 비교
    expected_labels = expected_valid[["equipment_id", "maintenance_date", "failure_within_30days"]].copy()
    expected_labels["equipment_id"] = expected_labels["equipment_id"].astype("string").fillna("UNKNOWN")
    expected_labels["maintenance_date"] = pd.to_datetime(expected_labels["maintenance_date"]).dt.strftime("%Y-%m-%d")
    expected_labels["failure_within_30days"] = expected_labels["failure_within_30days"].astype(int)
    actual_labels = valid_pred[["equipment_id", "maintenance_date", "actual"]].copy()
    actual_labels = actual_labels.rename(columns={"actual": "failure_within_30days"})
    actual_labels["equipment_id"] = actual_labels["equipment_id"].astype("string").fillna("UNKNOWN")
    actual_labels["maintenance_date"] = actual_labels["maintenance_date"].dt.strftime("%Y-%m-%d")
    actual_labels["failure_within_30days"] = actual_labels["failure_within_30days"].astype(int)
    expected_label_counts = expected_labels.value_counts().sort_index()
    actual_label_counts = actual_labels.value_counts().sort_index()
    require(expected_label_counts.equals(actual_label_counts), "검증 예측의 정답이 원본 학습 정답과 다릅니다.")
    independently_computed_auc = average_precision_score(valid_pred["actual"], valid_pred["prob"])
    require(valid_pred["actual"].nunique() == 2, "검증 구간 정답이 한 클래스뿐이라 PR-AUC를 계산할 수 없습니다.")
    monthly_hits = {}
    for month, group in valid_pred.groupby(valid_pred["maintenance_date"].dt.to_period("M")):
        chosen = group.nlargest(min(20, len(group)), "prob", keep="first")
        monthly_hits[str(month)] = int(chosen["actual"].sum())
    recorded_hits = dict(zip(metrics["month"].astype(str), metrics["model_top20_hits"].astype(int)))
    require(monthly_hits == recorded_hits, "모델 월별 상위 20대 적중 수를 재계산한 결과와 다릅니다.")

    # 임계값 비용 산식과 보고서의 PR-AUC 표기 범위를 점검
    threshold_file = output_dir / "risk_thresholds.json"
    require(threshold_file.exists(), f"파일이 없습니다: {threshold_file}")
    threshold_data = json.loads(threshold_file.read_text(encoding="utf-8"))
    for miss_cost, key in [(300, "miss_cost_300_won_10k"), (600, "miss_cost_600_won_10k")]:
        item = threshold_data[key]
        require(0 <= item["threshold"] <= 1, f"{key}: 임계값 범위 오류")
        calculated = item["missed_failures"] * miss_cost + item["flagged_for_inspection"] * 60
        require(np.isclose(item["estimated_cost_won_10k"], calculated), f"{key}: 비용 산식 오류")
        best = None
        y = valid_pred["actual"].to_numpy(dtype=int)
        probabilities = valid_pred["prob"].to_numpy(dtype=float)
        for candidate_threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
            flagged = probabilities >= candidate_threshold
            missed = int(np.sum((y == 1) & ~flagged))
            inspected = int(flagged.sum())
            candidate = (missed * miss_cost + inspected * 60, inspected, -float(candidate_threshold))
            if best is None or candidate < best:
                best = candidate
        require(best is not None and np.isclose(item["threshold"], -best[2]),
                f"{key}: 검증 확률에서 비용 최소 임계값을 재현하지 못했습니다.")
    report_path = output_dir / "analysis_report.md"
    require(report_path.exists(), "analysis_report.md가 없습니다.")
    report = report_path.read_text(encoding="utf-8")
    auc_match = re.search(r"PR-AUC:\s*([0-9.]+)", report)
    require(auc_match is not None and 0 <= float(auc_match.group(1)) <= 1,
            "보고서 PR-AUC를 찾지 못했거나 0~1 범위가 아닙니다.")
    require(np.isclose(float(auc_match.group(1)), independently_computed_auc, atol=0.000051),
            "보고서 PR-AUC가 검증 예측으로 재계산한 값과 다릅니다.")

    print("검증 통과")
    print(f"- 2025 예측: {len(pred):,}/{len(test):,}행, 확률 0~1")
    print(f"- 2026-01 점검 계획: 장비별 최신 기록 {len(plan)}대")
    print(f"- 검증 월: {len(metrics)}개월, 재계산 PR-AUC {independently_computed_auc:.4f}")
    print("- 월별 top-20, 비용 최소 임계값, 보고서 지표 일치")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(f"검증 실패: {exc}") from exc
