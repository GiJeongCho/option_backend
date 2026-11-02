# -*- coding: utf-8 -*-
"""
10배 vs 저배(低倍) 차이 집중 분석 — 2025-03 ~ 현재 구간 한정

목적: 국면 전략으로 넘어가기 전에, 최근 구간에서 '10배로 터진 계약'과
      '저배(2배도 못 가고 소멸/저조)' 계약을 가르는 진입시점 요인을 규명.

그룹 (최대배수 기준):
  - LOW (저배) : max_multiple < 2        (소멸/저조)
  - MID        : 2 ≤ max_multiple < 10
  - TEN (10배) : max_multiple ≥ 10

모든 피처는 진입 시점/그 이전에 알 수 있는 값만 사용(미래정보 차단).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import corr_multiple_drivers as cmd  # vix_trend_by_expiry 재사용

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OPP = OUT / "opportunities_lowband.csv"
START = 20250301  # 2025년 3월부터
SUFFIX = "_2025h1_on"

ENTRY_FEATURES = [
    "entry_premium", "log_entry_premium", "is_put", "entry_time",
    "vix_level", "vix_ma5_dev", "vix_above_ma5",
    "vix_slope5", "vix_slope120", "vix_daily_return", "vix_uptrend",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(OPP)
    df = df[(df["eligible"] == True) & (df["max_multiple"].notna())].copy()
    df = df[df["expiry_date"].astype(int) >= START].copy()
    trend = cmd.vix_trend_by_expiry()
    df = df.merge(trend, on="expiry_date", how="left")

    f = pd.DataFrame(index=df.index)
    f["entry_premium"] = df["first_eligible_low"].astype(float)
    f["log_entry_premium"] = np.log(f["entry_premium"].clip(lower=1e-6))
    f["is_put"] = (df["call_put"] == "PUT").astype(float)
    f["entry_time"] = df["first_eligible_time"].astype(float)
    for c in ["vix_level", "vix_ma5_dev", "vix_above_ma5", "vix_slope5",
              "vix_slope120", "vix_daily_return", "vix_uptrend"]:
        f[c] = df[c].astype(float)
    mm = df["max_multiple"].astype(float)
    f["max_multiple"] = mm
    f["log_max_multiple"] = np.log(mm.clip(lower=1e-6))
    f["reached_10x"] = (mm >= 10.0).astype(float)
    f["is_low"] = (mm < 2.0).astype(float)
    f["group"] = np.where(mm >= 10, "TEN", np.where(mm >= 2, "MID", "LOW"))
    f["year"] = (df["expiry_date"].astype(int) // 10000).astype(int)
    f["expiry_date"] = df["expiry_date"].astype(int)
    return f


def group_profile(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in ENTRY_FEATURES:
        g = f.groupby("group")[feat].mean()
        low = g.get("LOW", float("nan"))
        mid = g.get("MID", float("nan"))
        ten = g.get("TEN", float("nan"))
        rows.append(
            {
                "feature": feat,
                "LOW(저배)": round(float(low), 4),
                "MID": round(float(mid), 4),
                "TEN(10배)": round(float(ten), 4),
                "TEN-LOW": round(float(ten - low), 4),
            }
        )
    return pd.DataFrame(rows)


def discriminative(f: pd.DataFrame) -> pd.DataFrame:
    """TEN vs LOW 만 대상으로 한 단변량 판별력 (MID 제외해 10배 vs 저배 대비 선명화)."""
    sub = f[f["group"].isin(["TEN", "LOW"])].copy()
    y = (sub["group"] == "TEN").astype(float)
    rows = []
    for feat in ENTRY_FEATURES:
        x = sub[feat]
        m = x.notna()
        if m.sum() < 30 or x[m].nunique() < 2:
            corr = auc = float("nan")
        else:
            corr = x[m].corr(y[m], method="spearman")
            auc = roc_auc(x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float))
        rows.append(
            {
                "feature": feat,
                "spearman(10x vs 저배)": round(corr, 4) if corr == corr else None,
                "AUC": round(auc, 4) if auc == auc else None,
            }
        )
    out = pd.DataFrame(rows)
    out["|AUC-0.5|"] = (out["AUC"] - 0.5).abs()
    return out.sort_values("|AUC-0.5|", ascending=False).drop(columns="|AUC-0.5|")


def roc_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney 기반 AUC (x가 클수록 y=1 경향이면 >0.5)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    r = 1
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg
        r += (j - i + 1)
        i = j + 1
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = float(np.sum(ranks[y == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def premium_band_rates(f: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.01]
    labels = ["0.4-0.59", "0.6-0.79", "0.8-0.99", "1.0-1.49", "1.5-1.99",
              "2.0-2.49", "2.5-2.99", "3.0-3.99", "4.0-5.0"]
    f = f.copy()
    f["band"] = pd.cut(f["entry_premium"], bins=bins, labels=labels, right=False)
    rows = []
    for band, sub in f.groupby("band", observed=True):
        rows.append(
            {
                "premium_band": str(band),
                "n": int(len(sub)),
                "저배율%": round(float(sub["is_low"].mean() * 100), 1),
                "10배율%": round(float(sub["reached_10x"].mean() * 100), 1),
                "평균배수": round(float(sub["max_multiple"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def time_rates(f: pd.DataFrame) -> pd.DataFrame:
    bins = [800, 930, 1000, 1030, 1100, 1200, 1300, 1600]
    labels = ["~9:30", "9:30-10", "10-10:30", "10:30-11", "11-12", "12-13", "13+"]
    f = f.copy()
    f["tb"] = pd.cut(f["entry_time"], bins=bins, labels=labels, right=False)
    rows = []
    for tb, sub in f.groupby("tb", observed=True):
        rows.append(
            {
                "진입시간대": str(tb),
                "n": int(len(sub)),
                "저배율%": round(float(sub["is_low"].mean() * 100), 1),
                "10배율%": round(float(sub["reached_10x"].mean() * 100), 1),
                "평균배수": round(float(sub["max_multiple"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def put_call_rates(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, val in [("CALL", 0.0), ("PUT", 1.0)]:
        sub = f[f["is_put"] == val]
        rows.append(
            {
                "방향": name,
                "n": int(len(sub)),
                "저배율%": round(float(sub["is_low"].mean() * 100), 1),
                "10배율%": round(float(sub["reached_10x"].mean() * 100), 1),
                "평균배수": round(float(sub["max_multiple"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    f = load()
    n = len(f)
    counts = f["group"].value_counts()
    print(f"구간: {START} ~ 현재  유효계약 {n:,}건")
    print(f"그룹: LOW(저배<2x) {int(counts.get('LOW',0)):,}  "
          f"MID(2~10x) {int(counts.get('MID',0)):,}  "
          f"TEN(10x+) {int(counts.get('TEN',0)):,}")
    print(f"10배율 {f['reached_10x'].mean()*100:.1f}%  저배율 {f['is_low'].mean()*100:.1f}%\n")

    prof = group_profile(f)
    disc = discriminative(f)
    bands = premium_band_rates(f)
    times = time_rates(f)
    pc = put_call_rates(f)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("[1] 그룹별 피처 평균 (저배 vs 10배)")
    print(prof.to_string(index=False))
    print("\n[2] 10배 vs 저배 단변량 판별력 (|AUC-0.5| 내림차순)")
    print(disc.to_string(index=False))
    print("\n[3] 진입 프리미엄 밴드별")
    print(bands.to_string(index=False))
    print("\n[4] 진입 시간대별")
    print(times.to_string(index=False))
    print("\n[5] 방향(콜/풋)별")
    print(pc.to_string(index=False))

    summary = {
        "period_start": START,
        "n_contracts": n,
        "group_counts": {k: int(v) for k, v in counts.items()},
        "rate_10x_pct": round(float(f["reached_10x"].mean() * 100), 2),
        "rate_low_pct": round(float(f["is_low"].mean() * 100), 2),
        "group_profile": prof.to_dict(orient="records"),
        "discriminative_10x_vs_low": disc.to_dict(orient="records"),
        "premium_band": bands.to_dict(orient="records"),
        "time_of_day": times.to_dict(orient="records"),
        "put_call": pc.to_dict(orient="records"),
    }
    (OUT / f"analyze_10x_vs_low{SUFFIX}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    disc.to_csv(OUT / f"analyze_10x_vs_low_discriminative{SUFFIX}.csv",
                index=False, encoding="utf-8-sig")
    print(f"\n저장: output/analyze_10x_vs_low{SUFFIX}.json")


if __name__ == "__main__":
    main()
