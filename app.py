"""드론 예측정비 결과를 조회하는 Streamlit 대시보드."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
NAV_ITEMS = [
    ("overview", "개요", ":material/dashboard:"),
    ("missingness", "결측치", ":material/water_drop:"),
    ("validation", "검증 결과", ":material/query_stats:"),
    ("thresholds", "위험 임계값", ":material/tune:"),
    ("test_predictions", "2025 시험 예측", ":material/monitoring:"),
    ("inspection_plan", "2026-01 점검 계획", ":material/build:"),
]


def apply_pastel_style() -> None:
    """Apply a small set of page-level pastel CSS accents over the native theme."""
    st.html(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(221, 213, 255, .48), transparent 34%),
                radial-gradient(ellipse at 94% 8%, rgba(255, 225, 207, .42), transparent 28%),
                linear-gradient(145deg, #fcfbff 0%, #f8f9ff 48%, #fffaf7 100%);
        }
        header[data-testid="stHeader"] { background: transparent; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f3efff 0%, #f7f3ff 56%, #fff5f0 100%);
            border-right: 1px solid rgba(154, 139, 199, .18);
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, .78);
            border: 1px solid rgba(160, 148, 198, .22);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 8px 24px rgba(74, 62, 111, .055);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(74, 62, 111, .045);
        }
        [data-testid="stAlert"] { border-radius: 14px; }
        [data-testid="stSidebar"] .st-key-screen_nav_overview button,
        [data-testid="stSidebar"] .st-key-screen_nav_missingness button,
        [data-testid="stSidebar"] .st-key-screen_nav_validation button,
        [data-testid="stSidebar"] .st-key-screen_nav_thresholds button,
        [data-testid="stSidebar"] .st-key-screen_nav_test_predictions button,
        [data-testid="stSidebar"] .st-key-screen_nav_inspection_plan button {
            min-height: 2.8rem;
            justify-content: flex-start;
            text-align: left;
            border-radius: 12px;
            font-weight: 700;
            font-size: calc(1em + 4pt);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        [data-testid="stSidebar"] .st-key-screen_nav_overview button:hover,
        [data-testid="stSidebar"] .st-key-screen_nav_missingness button:hover,
        [data-testid="stSidebar"] .st-key-screen_nav_validation button:hover,
        [data-testid="stSidebar"] .st-key-screen_nav_thresholds button:hover,
        [data-testid="stSidebar"] .st-key-screen_nav_test_predictions button:hover,
        [data-testid="stSidebar"] .st-key-screen_nav_inspection_plan button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 14px rgba(76, 65, 112, .12);
        }
        .pastel-hero {
            position: relative;
            overflow: hidden;
            margin: .15rem 0 1.5rem;
            padding: 1.55rem 1.8rem;
            border: 1px solid rgba(255, 255, 255, .78);
            border-radius: 24px;
            background: linear-gradient(112deg, rgba(232, 226, 255, .96), rgba(232, 244, 255, .94) 53%, rgba(255, 237, 226, .94));
            box-shadow: 0 14px 36px rgba(75, 64, 115, .09);
        }
        .pastel-hero::after {
            content: "";
            position: absolute;
            z-index: 0;
            width: 190px;
            height: 190px;
            right: -55px;
            top: -92px;
            border-radius: 50%;
            background: rgba(255, 255, 255, .3);
        }
        .hero-drone-watermark {
            position: absolute;
            z-index: 0;
            top: 50%;
            right: 66px;
            width: 300px;
            height: 100%;
            transform: translateY(-50%);
            opacity: .15;
            pointer-events: none;
        }
        .hero-drone-watermark svg { width: 100%; height: 100%; }
        .home-drone-motifs {
            position: absolute;
            z-index: 1;
            inset: 0;
            overflow: hidden;
            pointer-events: none;
        }
        .home-drone-motifs .drone-motif {
            position: absolute;
            width: 54px;
            height: 54px;
            opacity: .27;
            filter: drop-shadow(0 3px 5px rgba(76, 105, 91, .08));
        }
        .home-drone-motifs .motif-mint { left: 43%; top: 17%; color: #53987F; transform: rotate(-12deg); }
        .home-drone-motifs .motif-blue { left: 62%; bottom: 8%; width: 42px; height: 42px; color: #718FC7; transform: rotate(10deg); }
        .home-drone-motifs .motif-peach { right: 24%; top: 12%; width: 38px; height: 38px; color: #C88F75; transform: rotate(14deg); }
        .home-drone-motifs .motif-lilac { left: 76%; bottom: 13%; width: 34px; height: 34px; color: #8A7BE0; transform: rotate(-8deg); }
        .home-drone-motifs .motif-sage { left: 31%; bottom: 9%; width: 40px; height: 40px; color: #78A98F; transform: rotate(9deg); }
        .hero-copy {
            position: relative;
            z-index: 2;
            max-width: calc(100% - 112px);
        }
        .pastel-hero .eyebrow {
            margin: 0 0 .4rem;
            color: #716795;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .pastel-hero h1 {
            margin: 0;
            color: #367A62;
            font-size: clamp(1.65rem, 3vw, 2.25rem);
            line-height: 1.2;
            letter-spacing: -.035em;
        }
        .pastel-hero p {
            margin: .55rem 0 0;
            color: #655f7d;
            font-size: .96rem;
        }
        .hero-drone-icon {
            position: absolute;
            z-index: 1;
            top: 50%;
            right: 1.65rem;
            display: grid;
            width: 76px;
            height: 76px;
            place-items: center;
            transform: translateY(-50%);
            border: 1px solid rgba(255, 255, 255, .9);
            border-radius: 22px;
            background: rgba(255, 255, 255, .58);
            box-shadow: 0 8px 22px rgba(78, 104, 95, .1);
        }
        .hero-drone-icon svg { width: 48px; height: 48px; }
        .hero-copy h1, .hero-copy p { max-width: 100%; }
        @media (max-width: 640px) {
            .pastel-hero { padding: 1.2rem; border-radius: 18px; }
            .hero-drone-icon { right: 1rem; width: 58px; height: 58px; border-radius: 18px; }
            .hero-drone-icon svg { width: 38px; height: 38px; }
            .hero-drone-watermark { right: 30px; width: 190px; opacity: .11; }
            .home-drone-motifs .motif-mint { left: 46%; top: 8%; width: 40px; height: 40px; }
            .home-drone-motifs .motif-blue { left: 60%; bottom: 5%; width: 32px; height: 32px; }
            .home-drone-motifs .motif-peach { right: 26%; top: 7%; width: 30px; height: 30px; }
            .home-drone-motifs .motif-lilac { left: 75%; bottom: 6%; width: 28px; height: 28px; }
            .home-drone-motifs .motif-sage { left: 34%; bottom: 5%; width: 30px; height: 30px; }
            .hero-copy { max-width: calc(100% - 72px); }
        }
        </style>
        """
    )


def show_hero(show_background_drones: bool = False) -> None:
    background_drones = ""
    if show_background_drones:
        background_drones = """
        <div class="home-drone-motifs" aria-hidden="true">
            <svg class="drone-motif motif-mint" viewBox="0 0 64 64" fill="none">
                <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9M18 22h8m12 0h8M18 42h8m12 0h8M20 32h24M32 22v20"/>
                    <rect x="24" y="25" width="16" height="14" rx="5" fill="#DDF1E8"/>
                </g>
                <circle cx="15" cy="15" r="4" fill="#B9E3D2"/><circle cx="49" cy="15" r="4" fill="#B9E3D2"/>
                <circle cx="15" cy="49" r="4" fill="#B9E3D2"/><circle cx="49" cy="49" r="4" fill="#B9E3D2"/>
            </svg>
            <svg class="drone-motif motif-blue" viewBox="0 0 64 64" fill="none">
                <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9M18 22h8m12 0h8M18 42h8m12 0h8M20 32h24M32 22v20"/>
                    <rect x="24" y="25" width="16" height="14" rx="5" fill="#E4EBFA"/>
                </g>
                <circle cx="15" cy="15" r="4" fill="#CEDAF3"/><circle cx="49" cy="15" r="4" fill="#CEDAF3"/>
                <circle cx="15" cy="49" r="4" fill="#CEDAF3"/><circle cx="49" cy="49" r="4" fill="#CEDAF3"/>
            </svg>
            <svg class="drone-motif motif-peach" viewBox="0 0 64 64" fill="none">
                <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9M18 22h8m12 0h8M18 42h8m12 0h8M20 32h24M32 22v20"/>
                    <rect x="24" y="25" width="16" height="14" rx="5" fill="#F8E7DE"/>
                </g>
                <circle cx="15" cy="15" r="4" fill="#F0D2C3"/><circle cx="49" cy="15" r="4" fill="#F0D2C3"/>
                <circle cx="15" cy="49" r="4" fill="#F0D2C3"/><circle cx="49" cy="49" r="4" fill="#F0D2C3"/>
            </svg>
            <svg class="drone-motif motif-lilac" viewBox="0 0 64 64" fill="none">
                <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9M20 32h24M32 22v20"/>
                    <rect x="24" y="25" width="16" height="14" rx="5" fill="#E9E4FB"/>
                </g>
                <circle cx="15" cy="15" r="4" fill="#D9D1F6"/><circle cx="49" cy="15" r="4" fill="#D9D1F6"/>
                <circle cx="15" cy="49" r="4" fill="#D9D1F6"/><circle cx="49" cy="49" r="4" fill="#D9D1F6"/>
            </svg>
            <svg class="drone-motif motif-sage" viewBox="0 0 64 64" fill="none">
                <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9M20 32h24M32 22v20"/>
                    <rect x="24" y="25" width="16" height="14" rx="5" fill="#E3F1E7"/>
                </g>
                <circle cx="15" cy="15" r="4" fill="#CDE5D4"/><circle cx="49" cy="15" r="4" fill="#CDE5D4"/>
                <circle cx="15" cy="49" r="4" fill="#CDE5D4"/><circle cx="49" cy="49" r="4" fill="#CDE5D4"/>
            </svg>
        </div>
        """
    st.html(
        f"""
        <section class="pastel-hero">
            {background_drones}
            <div class="hero-drone-watermark" aria-hidden="true">
                <svg viewBox="0 0 360 180" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M17 138C67 138 72 92 117 92" stroke="#5A9A83" stroke-width="3" stroke-linecap="round" stroke-dasharray="2 11"/>
                    <path d="M246 116C292 116 289 58 343 58" stroke="#8A7BE0" stroke-width="3" stroke-linecap="round" stroke-dasharray="2 11"/>
                    <g stroke="#4B8D76" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M148 83 103 45M212 83l45-38M148 108l-45 38m109-38 45 38"/>
                        <path d="M91 38h24m132 0h24M91 153h24m132 0h24"/>
                        <path d="M142 94h76M180 74v42"/>
                    </g>
                    <rect x="157" y="82" width="46" height="29" rx="11" fill="#D8EFE5" stroke="#4B8D76" stroke-width="4"/>
                    <path d="M169 111v12h22v-12" stroke="#4B8D76" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="103" cy="39" r="13" fill="#D9F0E6" stroke="#4B8D76" stroke-width="4"/>
                    <circle cx="257" cy="39" r="13" fill="#D9F0E6" stroke="#4B8D76" stroke-width="4"/>
                    <circle cx="103" cy="153" r="13" fill="#D9F0E6" stroke="#4B8D76" stroke-width="4"/>
                    <circle cx="257" cy="153" r="13" fill="#D9F0E6" stroke="#4B8D76" stroke-width="4"/>
                    <circle cx="180" cy="96" r="4" fill="#8A7BE0"/>
                </svg>
            </div>
            <div class="hero-copy">
                <p class="eyebrow">Maintenance intelligence · 2026.01</p>
                <h1>드론 예측정비 대시보드</h1>
                <p>정비 기록과 검증 결과를 살펴보고, 재점검 우선순위를 확인하세요.</p>
            </div>
            <div class="hero-drone-icon">
                <svg viewBox="0 0 64 64" role="img" aria-label="드론 아이콘">
                    <g fill="none" stroke="#43866F" stroke-linecap="round" stroke-linejoin="round" stroke-width="3">
                        <path d="M24 25 15 16M40 25l9-9M24 39l-9 9m25-9 9 9" />
                        <path d="M18 22h8m12 0h8M18 42h8m12 0h8" />
                        <path d="M20 32h24M32 22v20" />
                        <rect x="24" y="25" width="16" height="14" rx="5" fill="#DDF1E8" />
                        <path d="M28 39v5h8v-5" />
                    </g>
                    <circle cx="15" cy="15" r="4" fill="#9CD5BF" />
                    <circle cx="49" cy="15" r="4" fill="#9CD5BF" />
                    <circle cx="15" cy="49" r="4" fill="#9CD5BF" />
                    <circle cx="49" cy="49" r="4" fill="#9CD5BF" />
                </svg>
            </div>
        </section>
        """
    )


def load_csv(path: Path) -> pd.DataFrame | None:
    """CSV가 존재하고 읽을 수 있으면 반환한다. 없거나 잘못됐으면 None."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"파일을 읽지 못했습니다: {path.relative_to(ROOT)} ({exc})")
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"파일을 읽지 못했습니다: {path.relative_to(ROOT)} ({exc})")
        return None


def show_missingness() -> None:
    st.header("학습 데이터 결측치")
    saved = load_csv(OUTPUT_DIR / "missing_values_train.csv")
    if saved is not None and {"column", "missing_count", "missing_pct"}.issubset(saved.columns):
        summary = saved.copy()
        source_rows = None
    else:
        train = load_csv(DATA_DIR / "drone_train.csv")
        if train is None:
            st.warning("`data/drone_train.csv` 또는 `outputs/missing_values_train.csv`가 없습니다.")
            st.code(".venv\\Scripts\\python.exe analyze_missing.py", language="powershell")
            return
        missing = train.isna().sum()
        summary = pd.DataFrame({
            "column": train.columns,
            "missing_count": missing.to_numpy(dtype=int),
            "missing_pct": (missing / len(train) * 100).to_numpy(),
        })
        source_rows = len(train)

    summary["missing_count"] = pd.to_numeric(summary["missing_count"], errors="coerce").fillna(0)
    summary["missing_pct"] = pd.to_numeric(summary["missing_pct"], errors="coerce").fillna(0)
    total_missing = int(summary["missing_count"].sum())
    columns_with_missing = int((summary["missing_count"] > 0).sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("결측 셀", f"{total_missing:,}")
    col2.metric("결측이 있는 열", f"{columns_with_missing:,}")
    if source_rows is not None:
        col3.metric("학습 행 수", f"{source_rows:,}")
    else:
        col3.caption("결과 파일: missing_values_train.csv")

    shown = summary.sort_values(["missing_count", "column"], ascending=[False, True])
    st.dataframe(shown, width="stretch", hide_index=True, alt="학습 데이터 열별 결측 개수와 비율")
    if total_missing == 0:
        st.success("학습 데이터에서 결측값을 찾지 못했습니다.")
    else:
        st.info("결측값 처리 기준은 분석 보고서에서 확인하세요. 이 화면은 결측치를 임의로 대체하거나 삭제하지 않습니다.")


def show_validation() -> None:
    st.header("기간 검증 결과")
    predictions = load_csv(OUTPUT_DIR / "validation_predictions.csv")
    monthly = load_csv(OUTPUT_DIR / "validation_monthly_metrics.csv")
    if predictions is None or not {"actual", "prob"}.issubset(predictions.columns):
        st.warning("검증 예측 파일이 없습니다. 먼저 예측과 검증을 실행하세요.")
        st.code(".venv\\Scripts\\python.exe run_and_validate.py", language="powershell")
        return

    actual = pd.to_numeric(predictions["actual"], errors="coerce")
    probability = pd.to_numeric(predictions["prob"], errors="coerce")
    valid = actual.isin([0, 1]) & probability.between(0, 1)
    y = actual.loc[valid].astype(int)
    p = probability.loc[valid]
    pr_auc = None
    if len(y) and y.nunique() == 2:
        try:
            from sklearn.metrics import average_precision_score
            pr_auc = float(average_precision_score(y, p))
        except (ImportError, ValueError):
            pr_auc = None

    c1, c2, c3 = st.columns(3)
    c1.metric("검증 행", f"{len(predictions):,}")
    c2.metric("실제 재고장", f"{int(y.sum()):,}" if len(y) else "확인 불가")
    c3.metric("PR-AUC", f"{pr_auc:.4f}" if pr_auc is not None else "계산 불가")

    if monthly is None:
        st.warning("월별 기준선 결과 파일 `validation_monthly_metrics.csv`가 없습니다.")
        return
    display_cols = [
        col for col in ["month", "rows", "actual_failures", "model_top20_hits",
                        "random_expected_hits", "oldest_age_top20_hits"]
        if col in monthly.columns
    ]
    if display_cols:
        st.subheader("월별 상위 20대 적중 비교")
        st.dataframe(monthly[display_cols], width="stretch", hide_index=True,
                     alt="월별 모델 및 기준선의 상위 20대 재고장 적중 수")
    hit_cols = [col for col in ["model_top20_hits", "random_expected_hits", "oldest_age_top20_hits"]
                if col in monthly.columns]
    if hit_cols:
        totals = monthly[hit_cols].sum().rename({
            "model_top20_hits": "모델",
            "random_expected_hits": "무작위 기대값",
            "oldest_age_top20_hits": "경과연수 기준",
        })
        st.bar_chart(totals, alt="검증 기간 전체의 모델 및 기준선 적중 수")


def show_thresholds() -> None:
    st.header("위험 임계값")
    results = load_json(OUTPUT_DIR / "risk_thresholds.json")
    if results is None:
        st.warning("`outputs/risk_thresholds.json`가 없습니다. 먼저 예측과 검증을 실행하세요.")
        return
    rows = []
    for key, label in [("miss_cost_300_won_10k", "미탐 300만원"),
                       ("miss_cost_600_won_10k", "미탐 600만원")]:
        item = results.get(key)
        if isinstance(item, dict):
            rows.append({
                "비용 가정": label,
                "확률 임계값": item.get("threshold"),
                "점검 기준 초과 수": item.get("flagged_for_inspection"),
                "미탐 재고장 수": item.get("missed_failures"),
                "검증 추정 비용(만원)": item.get("estimated_cost_won_10k"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     alt="미탐 비용 가정별 선택된 위험 임계값과 검증 비용")
        st.caption("임계값은 검증 구간에서 정한 값입니다. 테스트 구간에 맞춰 조정하지 않습니다.")
    else:
        st.warning("임계값 파일에 예상한 비용 시나리오가 없습니다.")


def show_test_predictions() -> None:
    st.header("2025 시험 예측")
    test = load_csv(OUTPUT_DIR / "test_predictions_2025.csv")
    if test is None or "prob" not in test.columns:
        st.warning("`outputs/test_predictions_2025.csv`가 없습니다. 먼저 예측과 검증을 실행하세요.")
        return
    test["prob"] = pd.to_numeric(test["prob"], errors="coerce")
    valid_prob = test["prob"].dropna()
    a, b, c = st.columns(3)
    a.metric("예측 행", f"{len(test):,}")
    b.metric("평균 재고장 확률", f"{valid_prob.mean():.3f}" if len(valid_prob) else "확인 불가")
    c.metric("확률 범위", f"{valid_prob.min():.3f}–{valid_prob.max():.3f}" if len(valid_prob) else "확인 불가")
    if len(valid_prob):
        bins = pd.cut(valid_prob, bins=[-0.000001, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.000001],
                      include_lowest=True).value_counts(sort=False)
        st.subheader("확률 분포")
        st.bar_chart(bins, alt="2025년 시험 예측 확률의 구간별 건수")
    st.dataframe(test, width="stretch", hide_index=True, alt="2025년 시험 데이터의 재고장 예측 확률")


def show_inspection_plan() -> None:
    st.header("2026-01 재점검 계획")
    plan = load_csv(OUTPUT_DIR / "inspection_plan_2026-01.csv")
    scored = load_csv(OUTPUT_DIR / "current_monthly_scored.csv")
    if plan is None:
        st.warning("`outputs/inspection_plan_2026-01.csv`가 없습니다. 먼저 예측과 검증을 실행하세요.")
        return
    if scored is not None and "above_risk_threshold" in scored.columns:
        above = scored["above_risk_threshold"].astype(str).str.lower().isin(["true", "1"])
        c1, c2 = st.columns(2)
        c1.metric("전체 장비 최신 기록", f"{len(scored):,}")
        c2.metric("임계값 초과 장비", f"{int(above.sum()):,}")
        if int(above.sum()) > 20:
            st.warning(f"임계값 초과 장비가 20대보다 {int(above.sum()) - 20}대 많습니다. 정비 담당자가 우선순위와 여력을 검토해야 합니다.")

    preferred = ["rank", "equipment_id", "maintenance_date", "equipment_type", "unit_code",
                 "prob", "above_risk_threshold", "risk_factors", "recommendation"]
    columns = [col for col in preferred if col in plan.columns]
    st.dataframe(plan[columns] if columns else plan, width="stretch", hide_index=True,
                 alt="2026년 1월 드론 재점검 우선순위와 위험 정보")
    st.caption("계획은 모델 점수에 따른 참고 우선순위입니다. 최종 점검 여부는 담당자가 결정합니다.")


def render_sidebar_navigation() -> str:
    """Render each screen as a regular button with a clear active state."""
    if "dashboard_page" not in st.session_state:
        st.session_state.dashboard_page = "overview"

    st.sidebar.markdown("#### 화면 선택")
    for key, label, icon in NAV_ITEMS:
        if st.sidebar.button(
            label,
            key=f"screen_nav_{key}",
            type="primary" if st.session_state.dashboard_page == key else "secondary",
            icon=icon,
            width="stretch",
        ):
            st.session_state.dashboard_page = key
            st.rerun()
    return st.session_state.dashboard_page


def main() -> None:
    st.set_page_config(page_title="드론 예측정비", page_icon=":material/monitoring:", layout="wide")
    apply_pastel_style()
    page = render_sidebar_navigation()
    show_hero(show_background_drones=True)
    st.caption("교육용 가상 데이터 · 모델 결과는 참고용이며 최종 정비 판단은 담당자가 합니다.")

    if page == "overview":
        st.subheader("분석 결과 빠른 보기")
        report_path = OUTPUT_DIR / "analysis_report.md"
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info("분석 보고서가 아직 없습니다. 아래 명령으로 예측과 검증 결과를 생성하세요.")
            st.code(".venv\\Scripts\\python.exe run_and_validate.py", language="powershell")
        st.markdown("결측치와 모델 평가, 예측 및 점검 계획은 왼쪽 메뉴에서 각각 확인할 수 있습니다.")
    elif page == "missingness":
        show_missingness()
    elif page == "validation":
        show_validation()
    elif page == "thresholds":
        show_thresholds()
    elif page == "test_predictions":
        show_test_predictions()
    elif page == "inspection_plan":
        show_inspection_plan()


if __name__ == "__main__":
    main()
