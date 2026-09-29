from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FUTURES_PATH = ROOT / "rowdata" / "코스피200 선물 (F) 선물 과거 데이터.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WINDOW = 3
HIGH_VOL_START = pd.Timestamp("2025-10-01")
HIGH_VOL_END = pd.Timestamp("2026-08-31")
PRICE_COLUMNS = ("종가", "시가", "고가", "저가")


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, str(path))


def load_daily(path: Path = FUTURES_PATH) -> pd.DataFrame:
    raw = read_csv(path)
    required = {"날짜", *PRICE_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"선물 CSV 필수 열 누락: {sorted(missing)}")
    frame = raw.copy()
    frame["time"] = pd.to_datetime(
        frame["날짜"].astype(str).str.replace(" ", "", regex=False),
        format="%Y-%m-%d",
        errors="raise",
    )
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="raise",
        )
    source_descending = bool(frame["time"].is_monotonic_decreasing)
    duplicate_dates = int(frame["time"].duplicated().sum())
    frame = (
        frame.sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
        .rename(columns={"시가": "오픈"})
    )
    invalid = frame[
        (frame["저가"] > frame[["오픈", "종가"]].min(axis=1))
        | (frame["고가"] < frame[["오픈", "종가"]].max(axis=1))
        | (frame[list(("오픈", "종가", "고가", "저가"))] <= 0).any(
            axis=1
        )
    ]
    if len(invalid) > max(10, round(len(frame) * 0.01)):
        raise ValueError(f"비정상 OHLC 과다: {len(invalid)}행")
    invalid_rows = [
        {
            "date": row.time.strftime("%Y-%m-%d"),
            "open": float(row.오픈),
            "high": float(row.고가),
            "low": float(row.저가),
            "close": float(row.종가),
        }
        for _, row in invalid.iterrows()
    ]
    frame = frame.drop(index=invalid.index).reset_index(drop=True)
    frame.attrs["source_descending"] = source_descending
    frame.attrs["source_duplicate_dates"] = duplicate_dates
    frame.attrs["excluded_ohlc_rows"] = invalid_rows
    frame["weekofyear"] = frame.time.dt.isocalendar().week.astype(int)
    frame["year"] = frame.time.dt.year
    frame["day_of_week"] = frame.time.dt.day_of_week
    frame["month"] = frame.time.dt.month
    frame.loc[
        (frame.month == 12) & (frame.weekofyear == 1), "weekofyear"
    ] = 53
    return frame


def weekofyear(date_value: pd.Timestamp) -> int:
    week = int(date_value.isocalendar()[1])
    if date_value.month == 12 and week == 1:
        return 53
    return week


def weekly_like_source(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    year_weeks = daily[["year", "weekofyear"]].drop_duplicates()
    for _, year_week in year_weeks.iterrows():
        start_day = daily.loc[
            (daily.year == year_week.year)
            & (daily.weekofyear == year_week.weekofyear),
            "time",
        ].iloc[0]
        while True:
            current_week = daily.loc[
                (daily.year == start_day.year)
                & (daily.weekofyear == weekofyear(start_day))
            ]
            if current_week.empty:
                start_day -= timedelta(weeks=1)
            else:
                break

        weekday = 3
        selected = current_week.loc[current_week.day_of_week == weekday]
        while selected.empty and weekday >= 0:
            weekday -= 1
            selected = current_week.loc[
                current_week.day_of_week == weekday
            ]
        if selected.empty:
            for weekday in range(6, -1, -1):
                selected = current_week.loc[
                    current_week.day_of_week == weekday
                ]
                if not selected.empty:
                    break
        if selected.empty:
            continue
        start_day = selected.time.iloc[0]
        end_day = start_day + timedelta(weeks=1)

        while True:
            next_week = daily.loc[
                (daily.year == end_day.year)
                & (daily.weekofyear == weekofyear(end_day))
            ]
            if next_week.empty:
                end_day += timedelta(weeks=1)
            if daily.time.iloc[-1] < end_day:
                return pd.DataFrame(rows)
            if not next_week.empty:
                if next_week.time.iloc[0] < start_day:
                    end_day += timedelta(weeks=1)
                    next_week = daily.loc[
                        (daily.year == end_day.year)
                        & (daily.weekofyear == weekofyear(end_day))
                    ]
                break

        weekday = 3
        selected = next_week.loc[next_week.day_of_week == weekday]
        while selected.empty and weekday <= 6:
            weekday += 1
            selected = next_week.loc[
                next_week.day_of_week == weekday
            ]
        if selected.empty:
            while selected.empty:
                end_day += timedelta(weeks=1)
                next_week = daily.loc[
                    (daily.year == end_day.year)
                    & (daily.weekofyear == weekofyear(end_day))
                ]
                if next_week.empty:
                    if daily.time.iloc[-1] < end_day:
                        return pd.DataFrame(rows)
                    continue
                for weekday in range(0, 7):
                    selected = next_week.loc[
                        next_week.day_of_week == weekday
                    ]
                    if not selected.empty:
                        break
        if selected.empty:
            continue
        end_day = selected.time.iloc[0]
        interval = daily.loc[daily.time.between(start_day, end_day)]
        if interval.empty:
            continue
        end_row = daily.loc[daily.time == end_day].iloc[0]
        rows.append(
            {
                "time": end_day,
                "오픈": float(interval.iloc[0].오픈),
                "저가": float(interval.저가.min()),
                "고가": float(interval.고가.max()),
                # 원 논문 코드와 동일하게 종료일 종가가 아니라 시가 사용
                "종가": float(end_row.오픈),
                "시작일": start_day,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def source_wma(data: pd.DataFrame, period: int = WINDOW) -> list[float | None]:
    values: list[float | None] = [None] * WINDOW
    for index in range(len(data) - period):
        window = data[["종가"]].loc[
            index + 1 : index + period
        ].reset_index(drop=True)
        numerator = sum(
            (position + 1) * float(window.iloc[position].iloc[0])
            for position in range(period)
        )
        denominator = sum(range(1, period + 1))
        values.append(numerator / denominator)
    return values


def source_lsma(
    data: pd.DataFrame, period: int = WINDOW
) -> list[float | None]:
    values: list[float | None] = [None] * WINDOW
    for index in range(len(data) - period):
        window = data[["종가"]].loc[
            index + 1 : index + period
        ].reset_index(drop=True)
        x_values = [position + 1 for position in range(period)]
        y_values = [
            float(window.iloc[position].iloc[0])
            for position in range(period)
        ]
        slope = (
            len(x_values)
            * sum(
                x_values[position] * y_values[position]
                for position in range(len(x_values))
            )
            - sum(x_values) * sum(y_values)
        ) / (
            len(x_values)
            * sum(value * value for value in x_values)
            - sum(x_values) ** 2
        )
        intercept = (
            sum(y_values) - slope * sum(x_values)
        ) / len(x_values)
        values.append(intercept + slope * len(x_values))
    return values


def source_trima(
    data: pd.DataFrame, period: int = WINDOW
) -> list[float | None]:
    values: list[float | None] = [None] * WINDOW
    for index in range(len(data) - period):
        window = data[["종가"]].loc[
            index + 1 : index + period
        ].reset_index(drop=True)
        first_length = int(np.ceil((period + 0.1) / 2))
        first_average = float(
            window.iloc[0:first_length].mean().iloc[0]
        )
        inputs = list(window["종가"].values)
        inputs.insert(0, first_average)
        values.append(float(np.mean(inputs)))
    return values


def add_indicators(weekly: pd.DataFrame) -> pd.DataFrame:
    result = weekly.copy()
    result["SMA"] = result["종가"].rolling(WINDOW).mean()
    result["EMA"] = result["종가"].ewm(span=WINDOW, adjust=False).mean()
    result["WMA"] = source_wma(result)
    result["LSMA"] = source_lsma(result)
    result["TRIMA"] = source_trima(result)
    return result


def evaluate(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    column: str,
    use_shift: bool,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, object]:
    series = weekly[column].shift(1) if use_shift else weekly[column]
    touch_hits = 0
    direction_hits = 0
    total = 0
    for index, row in weekly.iterrows():
        if start is not None and row.time < start:
            continue
        if end is not None and row.time > end:
            continue
        value = series.iloc[index]
        if pd.isna(value) or value == 0:
            continue
        interval = daily.loc[
            daily.time.between(row.시작일, row.time)
        ]
        if len(interval) < 2:
            continue
        signal_open = float(interval.iloc[1].오픈)
        final_close = float(interval.iloc[-1].종가)
        if signal_open <= value:
            touch_hit = value <= float(interval.고가.max())
            direction_hit = final_close >= signal_open
        else:
            touch_hit = value >= float(interval.저가.min())
            direction_hit = final_close <= signal_open
        total += 1
        touch_hits += bool(touch_hit)
        direction_hits += bool(direction_hit)
    return {
        "hits": touch_hits,
        "total": total,
        "touch_accuracy": round(touch_hits / total, 6) if total else None,
        "endpoint_direction_hits": direction_hits,
        "endpoint_direction_accuracy": (
            round(direction_hits / total, 6) if total else None
        ),
    }


def profile(daily: pd.DataFrame) -> dict[str, object]:
    missing_business_days = len(
        pd.bdate_range(daily.time.min(), daily.time.max()).difference(
            pd.DatetimeIndex(daily.time)
        )
    )
    return {
        "path": str(FUTURES_PATH),
        "rows": len(daily),
        "start": daily.time.min().strftime("%Y-%m-%d"),
        "end": daily.time.max().strftime("%Y-%m-%d"),
        "descending_in_source": bool(
            daily.attrs["source_descending"]
        ),
        "duplicate_dates_in_source": int(
            daily.attrs["source_duplicate_dates"]
        ),
        "duplicate_dates_after_cleanup": int(daily.time.duplicated().sum()),
        "excluded_ohlc_rows": daily.attrs["excluded_ohlc_rows"],
        "missing_weekdays_including_holidays": missing_business_days,
        "minimum_close": float(daily.종가.min()),
        "maximum_close": float(daily.종가.max()),
    }


def run() -> dict[str, object]:
    daily = load_daily()
    weekly = add_indicators(weekly_like_source(daily))
    indicators: dict[str, object] = {}
    for column in ("SMA", "EMA", "WMA", "LSMA", "TRIMA"):
        biased = evaluate(weekly, daily, column, False)
        shifted = evaluate(weekly, daily, column, True)
        shifted_high_vol = evaluate(
            weekly,
            daily,
            column,
            True,
            HIGH_VOL_START,
            HIGH_VOL_END,
        )
        indicators[column] = {
            "biased": biased,
            "shifted": shifted,
            "shifted_high_volatility_period": shifted_high_vol,
            "touch_accuracy_overstatement": round(
                float(biased["touch_accuracy"])
                - float(shifted["touch_accuracy"]),
                6,
            ),
        }

    references = weekly[
        ["time", "시작일", "오픈", "고가", "저가", "종가", "LSMA"]
    ].copy()
    references["shifted_lsma"] = references["LSMA"].shift(1)
    references.to_csv(
        OUTPUT_DIR / "futures_lsma_weekly_references.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "status": "research_complete",
        "source_method": (
            "ma_leakage_test.py weekly construction and indicators"
        ),
        "data_profile": profile(daily),
        "weekly_rows": len(weekly),
        "weekly_start": weekly.time.min().strftime("%Y-%m-%d"),
        "weekly_end": weekly.time.max().strftime("%Y-%m-%d"),
        "window": WINDOW,
        "indicators": indicators,
        "interpretation_boundary": {
            "reported_accuracy_is": (
                "whether the following weekly range touches the shifted "
                "indicator level"
            ),
            "reported_accuracy_is_not": (
                "option profitability or final futures direction accuracy"
            ),
            "lsma_formula": (
                "linear-regression fitted value at the final input point; "
                "shift(1) supplies the prior completed value"
            ),
            "safe_option_alignment": (
                "use only shifted LSMA from the latest weekly row whose "
                "date is not later than the option trading date"
            ),
        },
    }
    (OUTPUT_DIR / "futures_lsma_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
