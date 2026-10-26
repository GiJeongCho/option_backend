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
OPP = OUT / "opportunities.csv"
S3 = OUT / "s3_breakout_event_details.csv"

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
    bins = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.01]
    labels = ["1.0-1.49", "1.5-1.99", "2.0-2.49", "2.5-2.99", "3.0-3.49", "3.5-3.99", "4.0-5.0"]
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
    ct.to_csv(OUT / "corr_multiple_drivers.csv", index=False, encoding="utf-8-sig")

    # 피처 간 상관행렬 (주요 leak-free + 결과)
    mat_cols = [
        "log_entry_premium", "is_put", "entry_time", "strike",
        "vix_level", "vix_ma5_dev", "vix_slope5", "vix_slope120", "vix_uptrend",
        "log_max_multiple", "reached_10x",
    ]
    mat = f[mat_cols].corr(method="spearman").round(3)
    mat.to_csv(OUT / "corr_multiple_drivers_matrix.csv", encoding="utf-8-sig")

    contrast = group_contrast(f)
    bands = premium_band_rates(f)

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
    }
    (OUT / "corr_multiple_drivers_summary.json").write_text(
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

    print("\n저장: output/corr_multiple_drivers.csv / _matrix.csv / _summary.json")


if __name__ == "__main__":
    main()
