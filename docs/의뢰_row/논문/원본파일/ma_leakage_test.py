# -*- coding: utf-8 -*-
"""
목적: 2023_12_10목종목시.ipynb 의 이동평균 지표 평가 로직을 재현하고
      후행이동평균 look-ahead(미래정보) 편향을 제거(shift(1))했을 때
      어떤 지표가 실제로 더 정확한지 비교한다.

- 원본(BIASED)   : MA[i] (i주차 종가 포함) 를 i주차 고가/저가 범위와 비교  -> 미래정보 누출
- 보정(SHIFTED)  : MA[i-1] (직전 주까지의 종가만 사용) 를 i주차 고가/저가 범위와 비교
"""

import numpy as np
import pandas as pd
from datetime import timedelta

pd.options.mode.chained_assignment = None

BASE = r"c:\project\option\backend\docs\의뢰_row\논문\원본파일"
DAILY_CSV = BASE + r"\코스피200 선물 (F) 선물 과거 데이터.csv"
WINDOW = 3

# ---------------------------------------------------------------------------
# 1) 일봉 데이터 로딩 & 전처리 (노트북과 동일)
# ---------------------------------------------------------------------------
def pretreatment(df):
    df["날짜"] = [df["날짜"][i][0:4] + "-" + df["날짜"][i][6:8] + "-" + df["날짜"][i][10:12]
                for i in range(len(df["날짜"]))]
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d", errors="raise")
    df = (df.sort_values(["날짜"], ascending=True)
            .drop_duplicates("날짜")
            .reset_index(drop=True)
            .rename(columns={"날짜": "time"}))
    return df

try:
    raw = pd.read_csv(DAILY_CSV, encoding="cp949")
except UnicodeDecodeError:
    raw = pd.read_csv(DAILY_CSV, encoding="utf-8")
raw = raw.rename(columns={"시가": "오픈"})
df = pretreatment(raw)
df = df[["time", "종가", "오픈", "고가", "저가"]]
df["day_of_year"] = df.time.dt.day_of_year
df["weekofyear"] = df.time.dt.isocalendar().week.astype(int)
df["year"] = df.time.dt.year
df["day_of_week"] = df.time.dt.day_of_week
df["year_of_month"] = df.time.dt.month

# 12월인데 isoweek==1 인 경우 53주로 보정 (노트북과 동일)
df.loc[(df.year_of_month == 12) & (df.weekofyear == 1), "weekofyear"] = 53


def weekofyear_(date):
    wk = int(date.isocalendar()[1])
    if (date.month == 12) and (wk == 1):
        return 53
    return wk


# ---------------------------------------------------------------------------
# 2) 주봉(만기 목요일 기준) 생성 : out_put() 재현
# ---------------------------------------------------------------------------
def out_put(df, start_dayofweek=3, end_dayofweek=3):
    info = {"time": 0, "오픈": 0, "저가": 0, "고가": 0, "종가": None, "시작일": 0}
    info_dataFrame = pd.DataFrame(info, index=[0])
    df1 = df[["year", "weekofyear"]].drop_duplicates()

    for i in range(len(df1)):
        df_strat_day = df.loc[(df1.iloc[i, :].year == df.year) &
                              (df1.iloc[i, :].weekofyear == df.weekofyear)].time.iloc[0]
        while True:
            new_df_1 = df.loc[(df.year == df_strat_day.year) &
                              (df.weekofyear == weekofyear_(df_strat_day))]
            if len(new_df_1) == 0:
                df_strat_day = df_strat_day - timedelta(weeks=1)
            else:
                break

        ww = start_dayofweek
        while True:
            new_df = new_df_1.loc[(new_df_1.day_of_week == ww)]
            if len(new_df) == 0:
                ww = ww - 1
            if len(new_df) == 1:
                break
            if ww == -1:
                break

        if (ww == -1) and len(new_df_1) >= 1:
            wk = 6
            while True:
                new_df = new_df_1.loc[(new_df_1.day_of_week == wk)]
                if len(new_df) == 0:
                    wk = wk - 1
                if len(new_df) == 1:
                    break
                if wk == -1:
                    break

        if len(new_df_1) == 0:
            continue

        df_end_day = df_strat_day = new_df.time.iloc[0]
        df_end_day = df_end_day + timedelta(weeks=1)

        while True:
            new_df_1 = df.loc[(df.year == df_end_day.year) &
                              (df.weekofyear == weekofyear_(df_end_day))]
            if len(new_df_1) == 0:
                df_end_day = df_end_day + timedelta(weeks=1)
            if df.iloc[len(df) - 1, :].time < df_end_day:
                return info_dataFrame[1:].sort_values(by=["time"]).reset_index(drop=True)
            elif len(new_df_1) >= 1:
                if new_df_1.time.iloc[0] < df_strat_day:
                    df_end_day = df_end_day + timedelta(weeks=1)
                    new_df_1 = df.loc[(df.year == df_end_day.year) &
                                      (df.weekofyear == weekofyear_(df_end_day))]
                break

        wk = end_dayofweek
        while True:
            new_df = new_df_1.loc[(new_df_1.day_of_week == wk)]
            if len(new_df) == 0:
                wk = wk + 1
            if len(new_df) == 1:
                break
            if wk == 7:
                break

        if (wk == 7) and len(new_df_1) >= 1:
            while True:
                df_end_day = df_end_day + timedelta(weeks=1)
                new_df_1 = df.loc[(df.year == df_end_day.year) &
                                  (df.weekofyear == weekofyear_(df_end_day))]
                if len(new_df_1) >= 1:
                    break
            wk = 0
            while True:
                new_df = new_df_1.loc[(new_df_1.day_of_week == wk)]
                if len(new_df) == 0:
                    wk = wk + 1
                if len(new_df) == 1:
                    break

        if len(new_df_1) == 0 or len(new_df) == 0:
            continue

        df_end_day = new_df.time.iloc[0]
        df_table = df.loc[(df.time.between(df_strat_day, df_end_day))]
        info = {"time": df_end_day,
                "오픈": df_table.iloc[0, ].오픈,
                "저가": min(df_table.저가.values),
                "고가": max(df_table.고가.values),
                "종가": df_table[df_table.time == df_end_day].오픈.iloc[0],
                "시작일": df_strat_day}
        info_dataFrame = pd.concat([info_dataFrame, pd.DataFrame(info, index=[0])], axis=0)

    return info_dataFrame[1:].sort_values(by=["time"]).reset_index(drop=True)


info_dataFrame = out_put(df, start_dayofweek=3, end_dayofweek=3)
info_dataFrame["시작일"] = pd.to_datetime(info_dataFrame["시작일"], format="%Y-%m-%d", errors="raise")


# ---------------------------------------------------------------------------
# 3) 이동평균 함수 (노트북과 동일한 후행 이동평균)
# ---------------------------------------------------------------------------
def SMA(data, period=3, column="종가"):
    return data[column].rolling(window=period).mean()


def EMA(data, period=3, column="종가"):
    return data[column].ewm(span=period, adjust=False).mean()


def WMA(data, period=3, column="종가"):
    val = [None] * WINDOW
    for i in range(len(data) - period):
        d = data[[column]].loc[i + 1:i + period].reset_index(drop=True)
        s1 = s2 = 0.0
        for j in range(period):
            s1 += (j + 1) * float(d.iloc[j].iloc[0])
            s2 += (j + 1)
        val.append(s1 / s2)
    return val


def LSMA(data, period=3, column="종가"):
    val = [None] * WINDOW
    for i in range(len(data) - period):
        d = data[[column]].loc[i + 1:i + period].reset_index(drop=True)
        X = [k + 1 for k in range(period)]
        Y = [float(d.iloc[k].iloc[0]) for k in range(period)]
        LRS = (len(X) * sum(X[k] * Y[k] for k in range(len(X))) - sum(X) * sum(Y)) / \
              (len(X) * sum(X[k] * X[k] for k in range(len(X))) - sum(X) * sum(X))
        LRT = (sum(Y) - LRS * sum(X)) / len(X)
        LRI = LRT + LRS * len(X)
        val.append(LRI)
    return val


def TRIMA(data, period=3, column="종가"):
    val = [None] * WINDOW
    for i in range(len(data) - period):
        d = data[[column]].loc[i + 1:i + period].reset_index(drop=True)
        A = int(np.ceil((period + 0.1) / 2))
        B = float(d.iloc[0:A].mean().iloc[0])
        for_Trima = list(d[column].values)
        for_Trima.insert(0, B)
        val.append(float(np.mean(for_Trima)))
    return val


data = info_dataFrame.reset_index(drop=True).copy()
data["단순이동평균_종가"] = SMA(data, period=WINDOW, column="종가")
data["지수이동평균_종가"] = EMA(data, period=WINDOW, column="종가")
data["가중이동평균_종가"] = WMA(data, period=WINDOW, column="종가")
data["최소제곱평균_종가"] = LSMA(data, period=WINDOW, column="종가")
data["삼각이동평균_종가"] = TRIMA(data, period=WINDOW, column="종가")

data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d", errors="raise")
data = data.fillna(0)

ma_cols = ["단순이동평균_종가", "지수이동평균_종가", "가중이동평균_종가",
           "최소제곱평균_종가", "삼각이동평균_종가"]


# ---------------------------------------------------------------------------
# 4) 평가 루프
# ---------------------------------------------------------------------------
def evaluate(data, col, use_shift):
    """use_shift=False -> 원본(후행편향), True -> 직전주 MA로 예측(보정)"""
    series = data[col].shift(1) if use_shift else data[col]
    hits, total = 0, 0
    for i in range(len(data)):
        ma = series.iloc[i]
        if pd.isna(ma) or ma == 0:          # 값이 없는(초기) 주는 제외
            continue
        new_df = df.loc[df.time.between(data.iloc[i].시작일, data.iloc[i].time)]
        if len(new_df) < 2:
            continue
        if new_df.iloc[1].오픈 <= ma:        # Long
            ok = ma <= max(new_df.고가)
        else:                                # Short
            ok = ma >= min(new_df.저가)
        total += 1
        hits += bool(ok)
    return hits, total


print("=" * 70)
print(f"주봉 개수: {len(data)}   (기간 {data.time.min().date()} ~ {data.time.max().date()})")
print("=" * 70)

rows = []
for col in ma_cols:
    hb, tb = evaluate(data, col, use_shift=False)
    hs, ts = evaluate(data, col, use_shift=True)
    rows.append((col, hb / tb, hs / ts, hb / tb - hs / ts))

print(f"\n{'지표':<16}{'원본(편향)':>12}{'보정 shift(1)':>16}{'차이(과대평가)':>16}")
print("-" * 60)
for name, b, s, d_ in rows:
    print(f"{name:<16}{b:>12.4f}{s:>16.4f}{d_:>16.4f}")

print("\n[보정 기준 정확도 순위]")
for rank, (name, b, s, d_) in enumerate(sorted(rows, key=lambda x: -x[2]), 1):
    print(f"  {rank}. {name:<16} {s:.4f}")
