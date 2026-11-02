# -*- coding: utf-8 -*-
"""
2배 vs 10배 차이 분석 — '이미 2배는 간 계약' 안에서의 구분

핵심 질문: 진입 시점(또는 그 이전)에 알 수 있는 정보만으로,
           "2배에서 멈추는 계약(MID)"과 "10배까지 가는 계약(TEN)"을
           구분할 수 있는가?

대상: max_multiple >= 2 인 계약만 (즉 최소 2배는 간 계약).
  - MID : 2 <= max_multiple < 10
  - TEN : max_multiple >= 10
  y = 1(TEN) / 0(MID).  AUC>0.5 면 그 피처가 클수록 10배 경향.

모든 피처는 진입 시점/이전 값만 사용(미래정보 차단).
추가로, 참고용(진입시점엔 모르는) 경로 관측치도 descriptive 로만 출력.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import corr_multiple_drivers as cmd  # vix_trend_by_expiry 재사용
from analyze_10x_vs_low import roc_auc  # Mann-Whitney AUC 재사용

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OPP = OUT / "opportunities_lowband.csv"

ENTRY_FEATURES = [
    "entry_premium", "log_entry_premium", "is_put", "entry_time",
    "vix_level", "vix_ma5_dev", "vix_above_ma5",
    "vix_slope5", "vix_slope120", "vix_daily_return", "vix_uptrend",
]

# 진입 때는 모르지만 '2배를 찍는 순간'에는 알 수 있는 경로 신호
PATH_FEATURES = ["t2x_clock", "mins_to_2x"]


def to_minutes(hhmm):
    """HHMM 정수를 '장중 분'으로 변환 (846 -> 8*60+46)."""
    v = pd.to_numeric(hhmm, errors="coerce")
    return (v // 100) * 60 + (v % 100)


def load() -> pd.DataFrame:
    df = pd.read_csv(OPP)
    df = df[(df["eligible"] == True) & (df["max_multiple"].notna())].copy()
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
    f["reached_10x"] = (mm >= 10.0).astype(float)
    f["group"] = np.where(mm >= 10, "TEN", np.where(mm >= 2, "MID", "LOW"))
    f["year"] = (df["expiry_date"].astype(int) // 10000).astype(int)
    f["expiry_date"] = df["expiry_date"].astype(int)
    # 참고용 경로 관측치 (진입시점엔 알 수 없음)
    f["peak_time"] = pd.to_numeric(df["max_multiple_time"], errors="coerce")
    f["t10_time"] = pd.to_numeric(df["target_time_10x"], errors="coerce")
    # 경로 신호: 2배를 찍은 순간에 알 수 있는 값
    f["t2x_clock"] = to_minutes(df["target_time_2x"])        # 2배 도달 시각(장중 분)
    entry_min = to_minutes(df["first_eligible_time"])
    f["mins_to_2x"] = f["t2x_clock"] - entry_min             # 진입→2배 소요(분)
    return f


def group_profile(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in ENTRY_FEATURES + PATH_FEATURES + ["peak_time"]:
        g = f.groupby("group")[feat].mean()
        mid = g.get("MID", float("nan"))
        ten = g.get("TEN", float("nan"))
        rows.append({
            "feature": feat,
            "MID(2~10배)": round(float(mid), 4),
            "TEN(10배)": round(float(ten), 4),
            "TEN-MID": round(float(ten - mid), 4),
        })
    return pd.DataFrame(rows)


def discriminative(f: pd.DataFrame, features) -> pd.DataFrame:
    """MID vs TEN 단변량 판별력 (2배는 간 계약 안에서 10배 구분)."""
    sub = f[f["group"].isin(["TEN", "MID"])].copy()
    y = (sub["group"] == "TEN").astype(float)
    rows = []
    for feat in features:
        x = sub[feat]
        m = x.notna()
        if m.sum() < 30 or x[m].nunique() < 2:
            corr = auc = float("nan")
        else:
            corr = x[m].corr(y[m], method="spearman")
            auc = roc_auc(x[m].to_numpy(float), y[m].to_numpy(float))
        rows.append({
            "feature": feat,
            "spearman": round(corr, 4) if corr == corr else None,
            "AUC": round(auc, 4) if auc == auc else None,
        })
    out = pd.DataFrame(rows)
    out["|AUC-0.5|"] = (out["AUC"] - 0.5).abs()
    return out.sort_values("|AUC-0.5|", ascending=False).drop(columns="|AUC-0.5|")


def continuation_by(f: pd.DataFrame, key: str, binner) -> pd.DataFrame:
    """2배 이상 간 계약 중 10배로 이어진 비율 (= 조건부 연속확률)."""
    g = f[f["max_multiple"] >= 2.0].copy()
    g["bin"] = binner(g)
    rows = []
    for b, sub in g.groupby("bin", observed=True):
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            key: str(b),
            "n(2배+)": n,
            "그중_10배": int(sub["reached_10x"].sum()),
            "연속확률%": round(float(sub["reached_10x"].mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def report(f: pd.DataFrame, title: str) -> dict:
    counts = f["group"].value_counts()
    n_mid = int(counts.get("MID", 0))
    n_ten = int(counts.get("TEN", 0))
    cont_overall = n_ten / (n_mid + n_ten) * 100 if (n_mid + n_ten) else 0.0

    prof = group_profile(f)
    disc = discriminative(f, ENTRY_FEATURES)
    disc_path = discriminative(f, PATH_FEATURES)
    band = continuation_by(
        f, "프리미엄밴드",
        lambda g: pd.cut(
            g["entry_premium"],
            bins=[0, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.01],
            labels=["0.4-0.59", "0.6-0.79", "0.8-0.99", "1.0-1.49",
                    "1.5-1.99", "2.0-2.99", "3.0-5.0"],
            right=False),
    )
    tod = continuation_by(
        f, "진입시간대",
        lambda g: pd.cut(
            g["entry_time"], bins=[800, 930, 1030, 1200, 1300, 1600],
            labels=["~9:30", "9:30-10:30", "10:30-12", "12-13", "13+"],
            right=False),
    )
    pc = continuation_by(f, "방향",
                         lambda g: g["is_put"].map({1.0: "PUT", 0.0: "CALL"}))
    reg = continuation_by(
        f, "전일VKOSPI일간수익",
        lambda g: pd.cut(
            g["vix_daily_return"], bins=[-1, 0, 0.02, 0.05, 1],
            labels=["하락", "0~2%", "2~5%", "5%+"], right=False),
    )
    t2x = continuation_by(
        f, "2배도달시각",
        lambda g: pd.cut(
            g["t2x_clock"], bins=[0, 600, 630, 660, 720, 780, 1000],
            labels=["~10:00", "10:00-10:30", "10:30-11:00", "11:00-12:00",
                    "12:00-13:00", "13:00+"], right=False),
    )
    spd = continuation_by(
        f, "진입→2배소요(분)",
        lambda g: pd.cut(
            g["mins_to_2x"], bins=[-1, 15, 30, 60, 120, 1000],
            labels=["~15분", "15-30분", "30-60분", "60-120분", "120분+"],
            right=False),
    )

    print(f"\n===== {title} =====")
    print(f"2배+ 계약: MID {n_mid:,}  TEN {n_ten:,}  "
          f"→ 2배 도달 후 10배 연속확률 {cont_overall:.1f}%")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n[1] MID vs TEN 피처 평균")
    print(prof.to_string(index=False))
    print("\n[2] MID vs TEN 진입시점 피처 판별력 (|AUC-0.5| 내림차순, 0.5면 무력)")
    print(disc.to_string(index=False))
    print("\n[2b] ★경로신호 판별력 (2배 찍는 순간에 알 수 있는 값)")
    print(disc_path.to_string(index=False))
    print("\n[3] 진입프리미엄 밴드별 10배 연속확률")
    print(band.to_string(index=False))
    print("\n[4] 진입시간대별 10배 연속확률")
    print(tod.to_string(index=False))
    print("\n[5] 방향별 10배 연속확률")
    print(pc.to_string(index=False))
    print("\n[6] 전일 VKOSPI 일간수익별 10배 연속확률")
    print(reg.to_string(index=False))
    print("\n[7] ★2배 도달 시각별 10배 연속확률")
    print(t2x.to_string(index=False))
    print("\n[8] ★진입→2배 소요시간별 10배 연속확률")
    print(spd.to_string(index=False))

    return {
        "title": title,
        "n_mid": n_mid, "n_ten": n_ten,
        "continuation_pct_overall": round(cont_overall, 2),
        "group_profile": prof.to_dict(orient="records"),
        "discriminative_mid_vs_ten": disc.to_dict(orient="records"),
        "discriminative_path_signals": disc_path.to_dict(orient="records"),
        "continuation_by_premium": band.to_dict(orient="records"),
        "continuation_by_time": tod.to_dict(orient="records"),
        "continuation_by_direction": pc.to_dict(orient="records"),
        "continuation_by_vix_return": reg.to_dict(orient="records"),
        "continuation_by_t2x_clock": t2x.to_dict(orient="records"),
        "continuation_by_mins_to_2x": spd.to_dict(orient="records"),
    }


def main() -> None:
    f = load()
    full = report(f, "전체기간")
    recent = report(f[f["expiry_date"] >= 20250301], "2025-03~")
    (OUT / "analyze_2x_vs_10x.json").write_text(
        json.dumps({"full": full, "recent_2025_03_on": recent},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n저장: output/analyze_2x_vs_10x.json")


if __name__ == "__main__":
    main()
