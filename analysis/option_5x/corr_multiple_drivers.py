# -*- coding: utf-8 -*-
"""
2배 vs 10배 차이 원인 규명 - 상관관계(corr) 분석

데이터
  - opportunities.csv            : 계약(expiry_date+option_code)별 결과(max_multiple, 5x/10x) + 구조 피처
  - s3_breakout_event_details.csv: previous_vix(전일 VKOSPI) 병합용

미래정보(look-ahead) 원칙
  - 피처는 "진입 시점 또는 그 이전"에 알 수 있어야 corr 해석이 의미가 있음.
  - 당일 누적치(total_volume, bar_count)는 결과와 동시에 쌓이므로 '사후/참고용'으로 분리 표기.

출력
  - output/corr_multiple_drivers.csv          : 피처별 상관계수 표
  - output/corr_multiple_drivers_matrix.csv   : 피처 간 상관행렬
  - output/corr_multiple_drivers_summary.json : 요약(JSON)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
# 입력 파일/출력 접미사: argv[1]=입력 csv명(기본 opportunities.csv), argv[2]=출력 접미사
OPP_NAME = sys.argv[1] if len(sys.argv) > 1 else "opportunities.csv"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OPP = OUT / OPP_NAME
S3 = OUT / "s3_breakout_event_details.csv"


def outp(stem: str) -> Path:
    return OUT / f"{stem}{SUFFIX}.csv"

# 기울기/추세 스냅샷 재사용 (직전일까지만 사용 = leak-free)
sys.path.insert(0, str(BASE))


def vix_trend_by_expiry() -> pd.DataFrame:
    """만기일별 '직전일까지' VKOSPI 레벨/기울기/추세 스냅샷."""
    import strategy6_v02_vix_ma5_rising_backtest as v02  # noqa: WPS433

    dates, snapshots = v02.vix_snapshots()
    rows = []
    for d in dates:
        snap = v02.previous_snapshot(d, dates, snapshots)  # 직전 거래일 스냅샷
        if snap is None:
            continue
        dev = (snap.close - snap.ma5) if math.isfinite(snap.ma5) else math.nan
        rows.append(
            {
                "expiry_date": int(d),
                "vix_level": snap.close,          # 전일 VKOSPI 종가(레벨)
                "vix_ma5_dev": dev,               # 전일 종가 - MA5 (이격; +면 5선 위)
                "vix_above_ma5": 1.0 if (math.isfinite(dev) and dev > 0) else 0.0,
                "vix_slope5": snap.slope5,        # 전일까지 5일 정규화 기울기
                "vix_slope120": snap.slope120,    # 전일까지 120일 정규화 기울기
                "vix_daily_return": snap.daily_return,  # 전일 1일 변화율
                "vix_uptrend": 1.0 if snap.strict_uptrend else 0.0,  # 상승추세 게이트
            }
        )
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    df = pd.read_csv(OPP)
    # 유효 계약만: 진입 가능 + max_multiple 존재
    df = df[df["eligible"] == True].copy()
    df = df[df["max_multiple"].notna()].copy()

    # 전일 VKOSPI 레벨/기울기/추세 병합 (만기일 기준, 전일까지만 = leak-free)
    try:
        trend = vix_trend_by_expiry()
        df = df.merge(trend, on="expiry_date", how="left")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] VKOSPI 기울기 병합 실패: {exc}")
        for col in ("vix_level", "vix_ma5_dev", "vix_above_ma5",
                    "vix_slope5", "vix_slope120", "vix_daily_return", "vix_uptrend"):
            df[col] = np.nan

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # --- 진입 시점 이전/당시에 알 수 있는 피처 (leak-free) ---
    f["entry_premium"] = df["first_eligible_low"].astype(float)          # 진입 프리미엄
    f["log_entry_premium"] = np.log(f["entry_premium"].clip(lower=1e-6))
    f["is_put"] = (df["call_put"] == "PUT").astype(float)               # 콜/풋
    f["entry_time"] = df["first_eligible_time"].astype(float)           # 진입 시각(HHMM)
    f["strike"] = df["strike"].astype(float)

    # 전일 VKOSPI 레벨 + 기울기/추세 (전일까지만 사용 = leak-free)
    f["vix_level"] = df["vix_level"].astype(float)                      # 전일 VKOSPI 레벨
    f["vix_ma5_dev"] = df["vix_ma5_dev"].astype(float)                  # MA5 이격
    f["vix_above_ma5"] = df["vix_above_ma5"].astype(float)             # 5선 위/아래
    f["vix_slope5"] = df["vix_slope5"].astype(float)                   # 5일 기울기
    f["vix_slope120"] = df["vix_slope120"].astype(float)              # 120일 기울기
    f["vix_daily_return"] = df["vix_daily_return"].astype(float)       # 전일 변화율
    f["vix_uptrend"] = df["vix_uptrend"].astype(float)               # 상승추세 게이트

    # --- 당일 누적 = 결과와 동시 발생 (사후/참고용, corr 해석 주의) ---
    f["total_volume_post"] = df["total_volume"].astype(float)
    f["log_total_volume_post"] = np.log(f["total_volume_post"].clip(lower=1))
    f["bar_count_post"] = df["bar_count"].astype(float)

    # --- 결과(outcome) ---
    # 연도(국면 통제용)
    f["year"] = (df["expiry_date"].astype(int) // 10000).astype(int)

    mm = df["max_multiple"].astype(float)
    f["max_multiple"] = mm
    f["log_max_multiple"] = np.log(mm.clip(lower=1e-6))
    f["reached_2x"] = (mm >= 2.0).astype(float)
    f["reached_5x"] = (mm >= 5.0).astype(float)
    f["reached_10x"] = (mm >= 10.0).astype(float)
    return f


LEAKFREE = [
    "entry_premium", "log_entry_premium", "is_put", "entry_time", "strike",
    "vix_level", "vix_ma5_dev", "vix_above_ma5",
    "vix_slope5", "vix_slope120", "vix_daily_return", "vix_uptrend",
]
POST = ["log_total_volume_post", "bar_count_post"]
OUTCOMES = ["log_max_multiple", "reached_2x", "reached_5x", "reached_10x"]


def corr_table(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in LEAKFREE + POST:
        kind = "leak-free" if feat in LEAKFREE else "post(사후)"
        for out in OUTCOMES:
            sub = f[[feat, out]].dropna()
            n = len(sub)
            if n < 30 or sub[feat].nunique() < 2:
                pear = spear = float("nan")
            else:
                pear = sub[feat].corr(sub[out], method="pearson")
                spear = sub[feat].corr(sub[out], method="spearman")
            rows.append(
                {
                    "feature": feat,
                    "type": kind,
                    "outcome": out,
                    "n": n,
                    "pearson": round(pear, 4) if pear == pear else None,
                    "spearman": round(spear, 4) if spear == spear else None,
                }
            )
    return pd.DataFrame(rows)


def _residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """y를 통제변수 Z(절편 포함)로 회귀한 잔차."""
    X = np.column_stack([np.ones(len(y)), Z]) if Z.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def partial_corr(
    f: pd.DataFrame,
    feat: str,
    outcome: str,
    num_controls: list[str],
    use_year: bool,
    method: str = "spearman",
) -> tuple[float, int]:
    """feat vs outcome 편상관 (num_controls + 연도더미 통제)."""
    cols = [feat, outcome, *num_controls]
    if use_year:
        cols.append("year")
    sub = f[cols].dropna()
    if len(sub) < 50 or sub[feat].nunique() < 2:
        return float("nan"), len(sub)

    a = sub[feat].to_numpy(dtype=float)
    b = sub[outcome].to_numpy(dtype=float)
    if method == "spearman":
        a, b = _rank(a), _rank(b)

    z_parts: list[np.ndarray] = []
    for c in num_controls:
        col = sub[c].to_numpy(dtype=float)
        z_parts.append(_rank(col) if method == "spearman" else col)
    if use_year:
        # 연도 더미 (첫 범주 제외) - 더미는 순위변환 안 함
        dummies = pd.get_dummies(sub["year"], drop_first=True).to_numpy(dtype=float)
        if dummies.size:
            z_parts.append(dummies)

    Z = np.column_stack(z_parts) if z_parts else np.empty((len(sub), 0))
    ra = _residualize(a, Z)
    rb = _residualize(b, Z)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan"), len(sub)
    return float(np.corrcoef(ra, rb)[0, 1]), len(sub)


# 편상관 대상 피처 (레벨/행사가는 통제변수로 쓰므로 피처에서 제외)
PARTIAL_FEATURES = [
    "log_entry_premium", "is_put", "entry_time",
    "vix_ma5_dev", "vix_above_ma5", "vix_slope5", "vix_slope120",
    "vix_daily_return", "vix_uptrend",
]

# 통제 시나리오: (라벨, 연속통제, 연도통제여부)
CONTROL_SETS = [
    ("연도통제", [], True),
    ("레벨통제", ["vix_level"], False),
    ("연도+레벨+행사가", ["vix_level", "strike"], True),
]


def partial_corr_table(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in PARTIAL_FEATURES:
        # 원시 상관(통제 없음)
        base = f[[feat, "log_max_multiple"]].dropna()
        raw = (
            base[feat].corr(base["log_max_multiple"], method="spearman")
            if len(base) >= 50 else float("nan")
        )
        row = {"feature": feat, "raw_spearman": round(raw, 4) if raw == raw else None}
        for label, num_c, use_year in CONTROL_SETS:
            pc, n = partial_corr(f, feat, "log_max_multiple", num_c, use_year)
            row[label] = round(pc, 4) if pc == pc else None
        rows.append(row)
    return pd.DataFrame(rows)


def partial_corr_table_10x(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in PARTIAL_FEATURES:
        base = f[[feat, "reached_10x"]].dropna()
        raw = (
            base[feat].corr(base["reached_10x"], method="spearman")
            if len(base) >= 50 else float("nan")
        )
        row = {"feature": feat, "raw_spearman": round(raw, 4) if raw == raw else None}
        for label, num_c, use_year in CONTROL_SETS:
            pc, n = partial_corr(f, feat, "reached_10x", num_c, use_year)
            row[label] = round(pc, 4) if pc == pc else None
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_corr(f: pd.DataFrame) -> pd.DataFrame:
    """연도별 10배도달율 + 주요 피처 Spearman 상관."""
    feats = ["log_entry_premium", "vix_level", "vix_slope120", "vix_ma5_dev", "is_put"]
    rows = []
    for year, sub in f.groupby("year"):
        row = {
            "year": int(year),
            "n": int(len(sub)),
            "rate_10x_pct": round(float(sub["reached_10x"].mean() * 100), 2),
            "mean_vix_level": round(float(sub["vix_level"].mean()), 2),
        }
        for feat in feats:
            s = sub[[feat, "log_max_multiple"]].dropna()
            row[f"corr_{feat}"] = (
                round(float(s[feat].corr(s["log_max_multiple"], method="spearman")), 3)
                if len(s) >= 30 else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def group_contrast(f: pd.DataFrame) -> list[dict]:
    """2배 도달(2x~) vs 10배 도달(10x~) 그룹의 피처 평균 대비."""
    reached2 = f[f["reached_2x"] == 1]
    reached10 = f[f["reached_10x"] == 1]
    out = []
    for feat in LEAKFREE + POST:
        a = reached2[feat].dropna()
        b = reached10[feat].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        out.append(
            {
                "feature": feat,
                "mean_reached2x": round(float(a.mean()), 4),
                "mean_reached10x": round(float(b.mean()), 4),
                "diff(10x-2x)": round(float(b.mean() - a.mean()), 4),
            }
        )
    return out


def premium_band_rates(f: pd.DataFrame) -> list[dict]:
    bins = [0, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.01]
    labels = ["0.4-0.59", "0.6-0.79", "0.8-0.99", "1.0-1.49", "1.5-1.99",
              "2.0-2.49", "2.5-2.99", "3.0-3.49", "3.5-3.99", "4.0-5.0"]
    f = f.copy()
    f["band"] = pd.cut(f["entry_premium"], bins=bins, labels=labels, right=False)
    g = f.groupby("band", observed=True)
    rows = []
    for band, sub in g:
        rows.append(
            {
                "premium_band": str(band),
                "n": int(len(sub)),
                "rate_2x_pct": round(float(sub["reached_2x"].mean() * 100), 2),
                "rate_5x_pct": round(float(sub["reached_5x"].mean() * 100), 2),
                "rate_10x_pct": round(float(sub["reached_10x"].mean() * 100), 2),
                "mean_max_multiple": round(float(sub["max_multiple"].mean()), 3),
            }
        )
    return rows


def main() -> None:
    df = load()
    f = build_features(df)

    n_total = len(f)
    n_vix = int(f["vix_slope5"].notna().sum())
    print(f"유효 계약 수: {n_total:,} (VKOSPI 기울기 병합: {n_vix:,}건, {n_vix/n_total*100:.1f}%)")
    print(f"2x 도달율: {f['reached_2x'].mean()*100:.1f}%  "
          f"5x: {f['reached_5x'].mean()*100:.1f}%  10x: {f['reached_10x'].mean()*100:.1f}%")

    ct = corr_table(f)
    ct.to_csv(outp("corr_multiple_drivers"), index=False, encoding="utf-8-sig")

    # 피처 간 상관행렬 (주요 leak-free + 결과)
    mat_cols = [
        "log_entry_premium", "is_put", "entry_time", "strike",
        "vix_level", "vix_ma5_dev", "vix_slope5", "vix_slope120", "vix_uptrend",
        "log_max_multiple", "reached_10x",
    ]
    mat = f[mat_cols].corr(method="spearman").round(3)
    mat.to_csv(outp("corr_multiple_drivers_matrix"), encoding="utf-8-sig")

    contrast = group_contrast(f)
    bands = premium_band_rates(f)

    # 편상관 (교란 통제)
    pct_mm = partial_corr_table(f)
    pct_10x = partial_corr_table_10x(f)
    pct_mm.to_csv(outp("corr_multiple_drivers_partial"), index=False, encoding="utf-8-sig")
    pct_10x.to_csv(outp("corr_multiple_drivers_partial_10x"), index=False, encoding="utf-8-sig")
    ycorr = yearly_corr(f)
    ycorr.to_csv(outp("corr_multiple_drivers_yearly"), index=False, encoding="utf-8-sig")

    summary = {
        "n_contracts": n_total,
        "n_with_vix_slope": n_vix,
        "rate_2x_pct": round(float(f["reached_2x"].mean() * 100), 2),
        "rate_5x_pct": round(float(f["reached_5x"].mean() * 100), 2),
        "rate_10x_pct": round(float(f["reached_10x"].mean() * 100), 2),
        "corr_vs_log_max_multiple": ct[ct["outcome"] == "log_max_multiple"]
        .sort_values("spearman", key=lambda s: s.abs(), ascending=False)
        .to_dict(orient="records"),
        "corr_vs_reached_10x": ct[ct["outcome"] == "reached_10x"]
        .sort_values("spearman", key=lambda s: s.abs(), ascending=False)
        .to_dict(orient="records"),
        "group_contrast_2x_vs_10x": contrast,
        "premium_band_rates": bands,
        "partial_corr_vs_log_max_multiple": pct_mm.to_dict(orient="records"),
        "partial_corr_vs_reached_10x": pct_10x.to_dict(orient="records"),
        "yearly_corr": ycorr.to_dict(orient="records"),
    }
    (OUT / f"corr_multiple_drivers_summary{SUFFIX}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 콘솔 출력 ----
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("\n[1] log(max_multiple) 대비 상관계수 (|spearman| 내림차순)")
    v = ct[ct["outcome"] == "log_max_multiple"].copy()
    v = v.reindex(v["spearman"].abs().sort_values(ascending=False).index)
    print(v[["feature", "type", "n", "pearson", "spearman"]].to_string(index=False))

    print("\n[2] reached_10x 대비 상관계수 (|spearman| 내림차순)")
    v = ct[ct["outcome"] == "reached_10x"].copy()
    v = v.reindex(v["spearman"].abs().sort_values(ascending=False).index)
    print(v[["feature", "type", "n", "pearson", "spearman"]].to_string(index=False))

    print("\n[3] 2배도달 그룹 vs 10배도달 그룹 평균 대비")
    print(pd.DataFrame(contrast).to_string(index=False))

    print("\n[4] 진입 프리미엄 밴드별 도달율")
    print(pd.DataFrame(bands).to_string(index=False))

    print("\n[5] 피처 간 Spearman 상관행렬")
    print(mat.to_string())

    print("\n[6] 편상관 - log(max_multiple) 대비 (교란 통제 전후)")
    print(pct_mm.to_string(index=False))

    print("\n[7] 편상관 - reached_10x 대비 (교란 통제 전후)")
    print(pct_10x.to_string(index=False))

    print("\n[8] 연도별 10배율 + 피처 상관")
    print(ycorr.to_string(index=False))

    print(f"\n저장: output/corr_multiple_drivers*{SUFFIX}.csv / _summary{SUFFIX}.json")


if __name__ == "__main__":
    main()
