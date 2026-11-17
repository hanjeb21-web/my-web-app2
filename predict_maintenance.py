"""드론 예측정비 실습 파이프라인.

실행: python predict_maintenance.py
입력: data/drone_{train,test,this_month}.csv
출력: outputs/ 아래 예측, 검증 지표, 점검 계획, 분석 보고서
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
LABEL = "failure_within_30days"
DATE = "maintenance_date"
ID = "equipment_id"
AGE = "equipment_age_years"
OUTPUTS = ROOT / "outputs"
HISTORY_FIELDS = [
    "maintenance_type",
    "fault_category",
    "severity",
    "maintenance_success",
]


def read_csv(path: Path, labeled: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    frame = pd.read_csv(path)
    required = {ID, DATE}
    if labeled:
        required.add(LABEL)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}에 필수 열이 없습니다: {', '.join(missing)}")
    frame[ID] = frame[ID].astype("string").fillna("UNKNOWN")
    frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce")
    if frame[DATE].isna().any():
        raise ValueError(f"{path.name}의 {DATE}에 날짜로 변환할 수 없는 값이 있습니다.")
    if labeled:
        frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce")
        frame = frame.loc[frame[LABEL].isin([0, 1])].copy()
        frame[LABEL] = frame[LABEL].astype(int)
    return frame


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """각 행의 날짜보다 앞선 날짜에 기록된 정비 이력만 계산한다.

    같은 드론의 같은 날짜 행들은 서로의 이력으로 사용하지 않는다.
    정답 열은 이력 계산에서 전혀 사용하지 않는다.
    """
    result = frame.copy()
    result["maintenance_year"] = result[DATE].dt.year
    result["maintenance_month"] = result[DATE].dt.month
    result["maintenance_quarter"] = result[DATE].dt.quarter
    result["maintenance_dayofyear"] = result[DATE].dt.dayofyear

    numeric_history = ["days_since_previous_record", "records_previous_90d",
                       "records_previous_365d", "records_before_date"]
    for col in numeric_history:
        result[col] = np.nan
    for col in HISTORY_FIELDS:
        result[f"previous_{col}"] = pd.Series(index=result.index, dtype=object)

    # 원본 인덱스와 무관하게 결과를 원래 행에 기록한다.
    for _, group in result.groupby(ID, sort=False, dropna=False):
        ordered = group.sort_values(DATE, kind="stable")
        past_dates: deque[pd.Timestamp] = deque()
        past_values: dict[str, Any] = {}
        n_past = 0
        last_date: pd.Timestamp | None = None

        for event_date, same_day in ordered.groupby(DATE, sort=True):
            while past_dates and past_dates[0] < event_date - pd.Timedelta(days=365):
                past_dates.popleft()
            count_90 = sum(d >= event_date - pd.Timedelta(days=90) for d in past_dates)
            count_365 = len(past_dates)
            row_ids = same_day.index
            result.loc[row_ids, "days_since_previous_record"] = (
                (event_date - last_date).days if last_date is not None else np.nan
            )
            result.loc[row_ids, "records_previous_90d"] = count_90
            result.loc[row_ids, "records_previous_365d"] = count_365
            result.loc[row_ids, "records_before_date"] = n_past
            for col in HISTORY_FIELDS:
                if col in past_values:
                    result.loc[row_ids, f"previous_{col}"] = past_values[col]

            # 현재 날짜의 행은 같은 날짜의 다른 행을 위한 과거값이 되지 않는다.
            for _, current in same_day.iterrows():
                past_dates.append(event_date)
                n_past += 1
                for col in HISTORY_FIELDS:
                    if col in current.index:
                        past_values[col] = current[col]
            last_date = event_date

    return result


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    # availability_rate는 월 전체의 가동률이라 정비 직후에는 아직 알 수 없을 수 있어 제외.
    excluded = {DATE, LABEL, "availability_rate", "_source"}
    return frame.drop(columns=[c for c in excluded if c in frame.columns])


def build_exploratory_outputs(raw_train: pd.DataFrame, usable_train: pd.DataFrame,
                              out_dir: Path) -> dict[str, Any]:
    """학습 자료의 결측, 수치 분포, IQR 이상치, 유형·계절 패턴을 요약한다."""
    missing_count = raw_train.isna().sum()
    missing_summary = pd.DataFrame({
        "column": raw_train.columns,
        "dtype": [str(dtype) for dtype in raw_train.dtypes],
        "missing_count": missing_count.to_numpy(dtype=int),
        "missing_pct": (missing_count / len(raw_train) * 100).to_numpy(),
        "non_missing_count": (len(raw_train) - missing_count).to_numpy(dtype=int),
    }).sort_values(["missing_count", "column"], ascending=[False, True])
    missing_summary.to_csv(out_dir / "eda_missing_by_column.csv", index=False, encoding="utf-8-sig")

    numeric_rows = []
    numeric_columns = raw_train.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        values = pd.to_numeric(raw_train[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (values < low_fence) | (values > high_fence)
        numeric_rows.append({
            "column": column,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()) if values.count() > 1 else 0.0,
            "min": float(values.min()),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(values.max()),
            "iqr_low_fence": float(low_fence),
            "iqr_high_fence": float(high_fence),
            "iqr_outlier_count": int(outlier_mask.sum()),
            "iqr_outlier_pct": float(outlier_mask.mean() * 100),
        })
    numeric_summary = pd.DataFrame(numeric_rows)
    numeric_summary.to_csv(out_dir / "eda_numeric_summary.csv", index=False, encoding="utf-8-sig")

    categorical_rows = []
    for column in raw_train.select_dtypes(include=["object", "category", "string"]).columns:
        if column == ID:
            continue
        counts = raw_train[column].fillna("(결측)").astype(str).value_counts().head(10)
        for value, count in counts.items():
            categorical_rows.append({
                "column": column,
                "value": value,
                "count": int(count),
                "pct": float(count / len(raw_train) * 100),
            })
    categorical_summary = pd.DataFrame(categorical_rows)
    categorical_summary.to_csv(out_dir / "eda_categorical_summary.csv", index=False, encoding="utf-8-sig")

    pattern_base = usable_train.copy()
    pattern_base["season"] = pattern_base[DATE].dt.month.map({
        12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을",
    })
    pattern_rows = []
    for dimension in ["equipment_type", "season", "fault_category"]:
        if dimension not in pattern_base:
            continue
        grouped = pattern_base.groupby(dimension, dropna=False)[LABEL].agg(rows="size", failures="sum", failure_rate="mean")
        for value, stats in grouped.iterrows():
            pattern_rows.append({
                "dimension": dimension,
                "value": "(결측)" if pd.isna(value) else str(value),
                "rows": int(stats["rows"]),
                "failures": int(stats["failures"]),
                "failure_rate": float(stats["failure_rate"]),
            })
    if {"equipment_type", "fault_category"}.issubset(pattern_base.columns):
        grouped = pattern_base.groupby(["equipment_type", "fault_category"], dropna=False)[LABEL].agg(
            rows="size", failures="sum", failure_rate="mean"
        )
        for (equipment_type, fault_category), stats in grouped.iterrows():
            pattern_rows.append({
                "dimension": "equipment_type × fault_category",
                "value": f"{equipment_type} × {fault_category}",
                "rows": int(stats["rows"]),
                "failures": int(stats["failures"]),
                "failure_rate": float(stats["failure_rate"]),
            })
    patterns = pd.DataFrame(pattern_rows)
    patterns.to_csv(out_dir / "eda_failure_patterns.csv", index=False, encoding="utf-8-sig")

    # 결측이 정비 유형에 따라 달라지는지 확인할 수 있도록 유형별 결측률을 저장한다.
    missing_by_type_rows = []
    if "maintenance_type" in raw_train:
        for column in raw_train.columns:
            if not raw_train[column].isna().any():
                continue
            rates = raw_train.assign(_missing=raw_train[column].isna()).groupby(
                "maintenance_type", dropna=False
            )["_missing"].agg(rows="size", missing_count="sum", missing_rate="mean")
            for maintenance_type, stats in rates.iterrows():
                missing_by_type_rows.append({
                    "column": column,
                    "maintenance_type": "(결측)" if pd.isna(maintenance_type) else str(maintenance_type),
                    "rows": int(stats["rows"]),
                    "missing_count": int(stats["missing_count"]),
                    "missing_pct": float(stats["missing_rate"] * 100),
                })
    pd.DataFrame(missing_by_type_rows).to_csv(
        out_dir / "eda_missing_by_maintenance_type.csv", index=False, encoding="utf-8-sig"
    )

    target = pd.to_numeric(raw_train.get(LABEL, pd.Series(dtype=float)), errors="coerce")
    dropped_target_rows = int((~target.isin([0, 1])).sum())
    rows_with_missing = int(raw_train.isna().any(axis=1).sum())
    return {
        "raw_rows": len(raw_train),
        "usable_rows": len(usable_train),
        "dropped_target_rows": dropped_target_rows,
        "missing_cells": int(missing_count.sum()),
        "missing_columns": int((missing_count > 0).sum()),
        "rows_with_missing": rows_with_missing,
        "numeric_summary": numeric_summary,
        "patterns": patterns,
    }


def make_model(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def top_k_mask(scores: pd.Series, dates: pd.Series, k: int = 20) -> pd.Series:
    selected = pd.Series(False, index=scores.index)
    months = dates.dt.to_period("M")
    for _, idx in months.groupby(months).groups.items():
        ranked = scores.loc[idx].sort_values(ascending=False, kind="stable")
        selected.loc[ranked.index[:k]] = True
    return selected


def evaluate(validation: pd.DataFrame, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = validation[LABEL].astype(int).reset_index(drop=True)
    dates = validation[DATE].reset_index(drop=True)
    scores = pd.Series(probabilities)
    hits_model = int((y * top_k_mask(scores, dates)).sum())
    if AGE in validation:
        age = pd.to_numeric(validation[AGE], errors="coerce").fillna(-np.inf).reset_index(drop=True)
    else:
        age = pd.Series(np.zeros(len(validation)))
    hits_age = int((y * top_k_mask(age, dates)).sum())

    monthly_rows = []
    for month, idx in dates.dt.to_period("M").groupby(dates.dt.to_period("M")).groups.items():
        n = len(idx)
        positives = int(y.loc[idx].sum())
        model_hits = int((y.loc[idx] * top_k_mask(scores.loc[idx], dates.loc[idx])).sum())
        age_hits = int((y.loc[idx] * top_k_mask(age.loc[idx], dates.loc[idx])).sum())
        monthly_rows.append({
            "month": str(month), "rows": n, "actual_failures": positives,
            "model_top20_hits": model_hits,
            "random_expected_hits": positives * min(20, n) / n if n else 0.0,
            "oldest_age_top20_hits": age_hits,
        })
    pr_auc = float(average_precision_score(y, scores)) if y.nunique() == 2 else float("nan")
    summary = {
        "pr_auc": pr_auc,
        "model_monthly_top20_hits": hits_model,
        "random_monthly_expected_hits": float(sum(r["random_expected_hits"] for r in monthly_rows)),
        "oldest_age_monthly_top20_hits": hits_age,
        "validation_rows": len(validation),
        "validation_failures": int(y.sum()),
    }
    return pd.DataFrame(monthly_rows), summary


def choose_threshold(y: pd.Series, probabilities: np.ndarray, miss_cost: float) -> dict[str, float | int]:
    y = y.astype(int).to_numpy()
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best: tuple[float, int, float] | None = None
    for threshold in candidates:
        flagged = probabilities >= threshold
        missed = int(np.sum((y == 1) & ~flagged))
        inspected = int(flagged.sum())
        cost = missed * miss_cost + inspected * 60.0
        # 같은 비용이면 더 적은 점검을 선택하고, 그래도 같으면 높은 기준을 선택한다.
        candidate = (cost, inspected, -float(threshold))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    threshold = -best[2]
    flagged = probabilities >= threshold
    missed = int(np.sum((y == 1) & ~flagged))
    inspected = int(flagged.sum())
    return {"threshold": float(threshold), "missed_failures": missed,
            "flagged_for_inspection": inspected,
            "estimated_cost_won_10k": float(missed * miss_cost + inspected * 60.0)}


def make_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if pd.notna(row.get("severity")):
        reasons.append(f"고장 심각도 {row['severity']}")
    if pd.notna(row.get("maintenance_success")) and row["maintenance_success"] == 0:
        reasons.append("정비 성공 여부가 실패")
    if pd.notna(row.get("records_previous_365d")) and row["records_previous_365d"] >= 2:
        reasons.append(f"최근 365일 이전 정비 {int(row['records_previous_365d'])}건")
    if pd.notna(row.get("days_since_previous_record")) and row["days_since_previous_record"] <= 30:
        reasons.append(f"직전 정비와 간격 {int(row['days_since_previous_record'])}일")
    if pd.notna(row.get("fault_category")):
        reasons.append(f"고장 유형 {row['fault_category']}")
    if pd.notna(row.get("previous_fault_category")):
        reasons.append(f"이전 고장 유형 {row['previous_fault_category']}")
    return "; ".join(reasons) if reasons else "모델 확률 순위가 높음(개별 요인 설명은 참고용)"


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(data_dir / "drone_train.csv")
    if DATE not in raw_train.columns:
        raise ValueError(f"drone_train.csv에 필수 열 {DATE}이 없습니다.")
    raw_train[DATE] = pd.to_datetime(raw_train[DATE], errors="coerce")
    train = read_csv(data_dir / "drone_train.csv", labeled=True)
    test = read_csv(data_dir / "drone_test.csv")
    current = read_csv(data_dir / "drone_this_month.csv")
    quality_rows = []
    for name, source in [("train", raw_train), ("test", test), ("this_month", current)]:
        duplicate_key_count = int(source.duplicated([ID, DATE]).sum())
        missing = source.isna().sum()
        quality_rows.append({
            "dataset": name,
            "rows": len(source),
            "columns": len(source.columns),
            "date_min": source[DATE].min().strftime("%Y-%m-%d"),
            "date_max": source[DATE].max().strftime("%Y-%m-%d"),
            "duplicate_equipment_date_rows": duplicate_key_count,
            "missing_cells": int(missing.sum()),
            "columns_with_missing": int((missing > 0).sum()),
            "target_positive_rate": float(source[LABEL].mean()) if LABEL in source else np.nan,
        })
    pd.DataFrame(quality_rows).to_csv(
        out_dir / "data_quality_summary.csv", index=False, encoding="utf-8-sig"
    )
    eda = build_exploratory_outputs(raw_train, train, out_dir)
    train["_source"] = "train"
    test["_source"] = "test"
    current["_source"] = "current"
    all_rows = pd.concat([train, test, current], ignore_index=True, sort=False)
    engineered = add_history_features(all_rows)
    train_e = engineered.loc[engineered["_source"] == "train"].copy()
    test_e = engineered.loc[engineered["_source"] == "test"].copy()
    current_e = engineered.loc[engineered["_source"] == "current"].copy()

    validation_start = pd.Timestamp(args.validation_start)
    # 직전 30일의 학습 라벨은 검증 시작 시점까지 결과가 확정되지 않았을 수 있다.
    train_end = validation_start - pd.Timedelta(days=30)
    fit_mask = train_e[DATE] < train_end
    valid_mask = train_e[DATE] >= validation_start
    fit_data = train_e.loc[fit_mask].copy()
    valid_data = train_e.loc[valid_mask].copy()
    if fit_data.empty or valid_data.empty:
        raise ValueError("기간 검증 데이터가 비었습니다. --validation-start 값을 확인하세요.")
    if fit_data[LABEL].nunique() < 2:
        raise ValueError("검증 전 학습 구간에 타깃 클래스가 하나뿐입니다.")

    x_fit = make_features(fit_data)
    x_valid = make_features(valid_data)
    validation_model = make_model(x_fit)
    validation_model.fit(x_fit, fit_data[LABEL])
    valid_prob = validation_model.predict_proba(x_valid)[:, 1]
    monthly_metrics, metric_summary = evaluate(valid_data.reset_index(drop=True), valid_prob)
    monthly_metrics.to_csv(out_dir / "validation_monthly_metrics.csv", index=False, encoding="utf-8-sig")
    validation_predictions = pd.DataFrame({
        ID: valid_data[ID].to_numpy(),
        DATE: valid_data[DATE].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual": valid_data[LABEL].to_numpy(),
        "prob": valid_prob,
    })
    validation_predictions.to_csv(
        out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    threshold_300 = choose_threshold(valid_data[LABEL], valid_prob, 300.0)
    threshold_600 = choose_threshold(valid_data[LABEL], valid_prob, 600.0)
    threshold_info = {"miss_cost_300_won_10k": threshold_300,
                      "miss_cost_600_won_10k": threshold_600}
    (out_dir / "risk_thresholds.json").write_text(
        json.dumps(threshold_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 최종 모델은 라벨이 있는 전체 학습 자료로 재학습한다.
    final_x = make_features(train_e)
    final_model = make_model(final_x)
    final_model.fit(final_x, train_e[LABEL])
    test_prob = final_model.predict_proba(make_features(test_e))[:, 1]
    predictions = pd.DataFrame({
        ID: test_e[ID].to_numpy(),
        DATE: test_e[DATE].dt.strftime("%Y-%m-%d").to_numpy(),
        "prob": test_prob,
    })
    predictions.to_csv(out_dir / "test_predictions_2025.csv", index=False, encoding="utf-8-sig")

    current_prob = final_model.predict_proba(make_features(current_e))[:, 1]
    plan = current_e[[ID, DATE]].copy()
    plan["prob"] = current_prob
    plan["equipment_type"] = current_e.get("equipment_type", pd.Series(index=current_e.index, dtype=object))
    plan["unit_code"] = current_e.get("unit_code", pd.Series(index=current_e.index, dtype=object))
    plan["risk_factors"] = current_e.apply(make_reasons, axis=1).to_numpy()
    plan = plan.sort_values([ID, DATE], ascending=[True, False], kind="stable").drop_duplicates(ID)
    plan["risk_threshold"] = threshold_300["threshold"]
    plan["above_risk_threshold"] = plan["prob"] >= threshold_300["threshold"]
    plan = plan.sort_values("prob", ascending=False, kind="stable").reset_index(drop=True)
    plan.insert(0, "rank", np.arange(1, len(plan) + 1))
    plan["recommendation"] = np.where(
        plan.index < 20,
        "재점검 우선 대상(정비 담당자 확인)",
        np.where(plan["above_risk_threshold"], "임계값 초과(20대 한도 초과 시 담당자 검토)", "일반 모니터링"),
    )
    # 전체 장비별 최신 기록은 검증용으로 보존하고, 제출용 점검 계획은 상위 20대로 제한한다.
    plan.to_csv(out_dir / "current_monthly_scored.csv", index=False, encoding="utf-8-sig")
    inspection_plan = plan.head(20).copy()
    inspection_plan.to_csv(
        out_dir / "inspection_plan_2026-01.csv", index=False, encoding="utf-8-sig"
    )
    threshold_exceed_count = int(plan["above_risk_threshold"].sum())
    over_capacity_count = max(0, threshold_exceed_count - 20)

    missing_table = pd.read_csv(out_dir / "eda_missing_by_column.csv")
    missing_top = missing_table.loc[missing_table["missing_count"] > 0].head(5)
    missing_finding = "; ".join(
        f"{row.column} {int(row.missing_count)}건({row.missing_pct:.1f}%)"
        for row in missing_top.itertuples()
    ) or "결측 없음"
    outlier_top = eda["numeric_summary"].sort_values("iqr_outlier_count", ascending=False)
    outlier_top = outlier_top.loc[outlier_top["iqr_outlier_count"] > 0].head(5)
    outlier_finding = "; ".join(
        f"{row.column} {int(row.iqr_outlier_count)}건" for row in outlier_top.itertuples()
    ) or "IQR 기준 이상치 없음"
    pattern_findings = []
    if len(eda["patterns"]):
        for dimension, group in eda["patterns"].groupby("dimension"):
            comparable = group.loc[group["rows"] >= 10].sort_values("failure_rate")
            if len(comparable) >= 2:
                low, high = comparable.iloc[0], comparable.iloc[-1]
                pattern_findings.append(
                    f"{dimension}: 낮은 {low['value']} {low['failure_rate']:.1%}(n={int(low['rows'])}), "
                    f"높은 {high['value']} {high['failure_rate']:.1%}(n={int(high['rows'])})"
                )
    pattern_finding = "; ".join(pattern_findings) or "표본 10건 이상인 비교 가능한 그룹이 부족합니다."

    report = [
        "# 드론 예측정비 분석 결과",
        "",
        "> 교육용 가상 데이터의 분석 결과입니다. 실제 정비 결정은 담당자가 확인해야 합니다.",
        "",
        "## 데이터 처리 및 입력 기준",
        "",
        f"- 학습 원본 {eda['raw_rows']:,}행 중 타깃이 없거나 0/1이 아닌 {eda['dropped_target_rows']:,}행을 학습에서 제외했습니다. 사용 학습 {len(train):,}행, 검증 {len(valid_data):,}행, 시험 {len(test):,}행, 2026-01 {len(current):,}행입니다.",
        "- 날짜 범위, 중복 장비·정비일 행, 결측 셀·열 수, 타깃 양성률은 `data_quality_summary.csv`에 기록했습니다.",
        "- 날짜와 장비 ID를 기준으로 이력을 연결했습니다. 같은 장비의 같은 날짜 행은 서로의 과거 이력으로 보지 않습니다.",
        "- 예측 시점 뒤의 데이터는 피처에 사용하지 않습니다. `maintenance_date`는 연·월·분기·연중 일자로 변환하며, 타깃·내부 데이터 구분 열은 입력에서 제외합니다. `availability_rate`는 월 전체를 요약해 정비 직후 알 수 없을 수 있어 제외했습니다.",
        "- 숫자형 결측은 중앙값으로 대체하고 결측 표시 피처를 추가합니다. 범주형 결측은 최빈값으로 대체하고 결측 표시 피처를 추가합니다. 중앙값은 극단값의 영향을 평균보다 덜 받고, 최빈값은 범주형 자료에 적용할 수 있어 사용했습니다. 이 처리의 효과는 원본 결측률과 함께 검토해야 합니다.",
        "- IQR 1.5배 기준 이상치는 탐색 목적으로 표시할 뿐 행 삭제나 윈저라이징을 하지 않았습니다. 트리 모델은 값의 크기 구간을 분할하므로 일괄 제거로 유효한 고심각도 기록을 잃지 않도록 했습니다. 실제 입력 오류로 확인된 값은 별도 근거를 갖고 정정해야 합니다.",
        "- 검증 기간 경계: " + validation_start.strftime("%Y-%m-%d") + "부터. 검증 시작 전 30일의 학습 행은 라벨 성숙 기간으로 제외했습니다.",
        "",
        "## 탐색 분석 결과",
        "",
        f"- 학습 원본 {eda['raw_rows']:,}행 중 결측 셀 {eda['missing_cells']:,}개, 결측 열 {eda['missing_columns']:,}개, 결측 포함 행 {eda['rows_with_missing']:,}개입니다. 열별 값은 `eda_missing_by_column.csv`에 있습니다.",
        f"- 결측이 많은 열: {missing_finding}.",
        "- 수치형 변수의 평균·사분위수·범위와 IQR 이상치 개수는 `eda_numeric_summary.csv`에 있습니다. 이상치는 데이터 검토 지표이며 자동 제외하지 않았습니다.",
        f"- IQR 기준 이상치 상위 열: {outlier_finding}.",
        "- 범주형 상위 빈도는 `eda_categorical_summary.csv`, 정비 유형별 결측률은 `eda_missing_by_maintenance_type.csv`에 저장했습니다.",
        "- 장비 유형·계절·고장 유형별 재고장률과 장비 유형×고장 유형 조합은 `eda_failure_patterns.csv`에 있습니다. 표본 수가 작은 그룹의 비율은 불안정하므로 원인으로 단정하지 않습니다.",
        f"- 학습 자료에서 관찰된 그룹별 재고장률 범위: {pattern_finding}.",
        "- 결측 원인은 결측률만으로 자동 판정할 수 없습니다. 정비 유형별 발생 양상과 해당 열의 의미를 함께 보고 업무상 미기록인지 입력 누락인지 확인해야 합니다.",
        "- 탐색 패턴은 학습 데이터의 기술통계입니다. 변수 간 연관은 인과관계를 뜻하지 않으며, 검증 구간 성능과 별도로 해석합니다.",
        "",
        "## 검증 성능",
        "",
        f"- PR-AUC: {metric_summary['pr_auc']:.4f}",
        f"- 월별 상위 20대 실제 재고장 합계: 모델 {metric_summary['model_monthly_top20_hits']}건, 무작위 기대값 {metric_summary['random_monthly_expected_hits']:.2f}건, 경과연수 기준 {metric_summary['oldest_age_monthly_top20_hits']}건.",
        f"- 검증 행 {metric_summary['validation_rows']:,}건, 실제 재고장 {metric_summary['validation_failures']:,}건.",
        "- 월별 결과는 `validation_monthly_metrics.csv`에, 행별 검증 정답과 확률은 `validation_predictions.csv`에 있습니다. 무작위 기준은 월별 실제 양성 수에 20/N을 곱한 기대 적중 수입니다.",
        "",
        "## 검증에서 선택한 위험 기준",
        "",
        f"- 미탐 비용 300만원 가정: 확률 기준 {threshold_300['threshold']:.6f}, 기준 초과 {threshold_300['flagged_for_inspection']}건, 미탐 {threshold_300['missed_failures']}건, 추정 비용 {threshold_300['estimated_cost_won_10k']:.0f}만원.",
        f"- 미탐 비용 600만원 가정: 확률 기준 {threshold_600['threshold']:.6f}, 기준 초과 {threshold_600['flagged_for_inspection']}건, 미탐 {threshold_600['missed_failures']}건, 추정 비용 {threshold_600['estimated_cost_won_10k']:.0f}만원.",
        "- 임계값은 검증 자료에서만 고르고 테스트 자료를 이용해 조정하지 않았습니다. 산출 세부는 `risk_thresholds.json`에 있습니다.",
        "",
        "## 산출물 및 한계",
        "",
        "- `test_predictions_2025.csv`: 시험 자료 전체에 대한 재고장 확률.",
        "- `inspection_plan_2026-01.csv`: 장비별 가장 최근 기록을 사용해 확률 상위 20대를 우선 표시한 계획. 임계값 초과 여부와 참고용 위험 요인을 함께 제공합니다.",
        f"- 임계값 초과 장비는 전체 {threshold_exceed_count}대이며, 월간 20대 한도를 넘는 수는 {over_capacity_count}대입니다. 초과 시 상위 20대부터 점검하고 나머지는 확률 순위, 안전상 우선순위, 정비 여력을 담당자가 검토해야 합니다.",
        "- 검증 구간이 짧고 가상 데이터에 학습했으며 실제 환경의 분포 변화와 장비별 반복 관측 영향을 완전히 반영하지 못합니다. 개별 위험 요인은 모델 설명이 아니라 원자료에 기반한 참고 정보입니다.",
        "",
        "## 결과 파일",
        "",
        "이 보고서의 실행 수치는 `predict_maintenance.py`가 생성한 `data_quality_summary.csv`, `validation_predictions.csv`, `validation_monthly_metrics.csv`, `risk_thresholds.json`, `test_predictions_2025.csv`, `inspection_plan_2026-01.csv`에서 확인할 수 있습니다.",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"완료: {out_dir}")
    print(f"검증 PR-AUC: {metric_summary['pr_auc']:.4f}")
    print(f"2025 예측 행: {len(predictions):,}")
    print(f"2026-01 점검 후보: {min(20, len(plan))}대")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="드론 정비 후 30일 내 재고장 확률을 예측합니다.")
    parser.add_argument("--data-dir", default="data", help="CSV 폴더 (기본값: data)")
    parser.add_argument("--output-dir", default="outputs", help="결과 폴더 (기본값: outputs)")
    parser.add_argument("--validation-start", default="2024-07-01",
                        help="검증 구간 시작일 (기본값: 2024-07-01)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
