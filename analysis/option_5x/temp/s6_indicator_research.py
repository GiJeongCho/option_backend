from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT.parents[1]
ROWDATA_DIR = BACKEND_DIR / "rowdata"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

import minimal_baseline_backtest as base  # noqa: E402
import strategy1_backtest as strategy1  # noqa: E402
from app.index_minute_catalog import IndexMinuteCatalog  # noqa: E402
from inventory import ANALYSIS_START_DATE  # noqa: E402


INDEX_MINUTE_CATALOG = IndexMinuteCatalog(ROWDATA_DIR)
SPOT_PATH = INDEX_MINUTE_CATALOG.normalized_path("spot")
FUTURES_PATH = INDEX_MINUTE_CATALOG.normalized_path("futures")
COMMON_START = ANALYSIS_START_DATE
DEVELOPMENT_END = "20260430"
ENTRY_START = 930
ENTRY_CUTOFF = 1300
OUTCOME_END = 1515
MAX_ENTRY_WAIT = 5
PRICE_EPSILON = 1e-9


@dataclass(frozen=True)
class PremiumBand:
    key: str
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower - PRICE_EPSILON <= value <= self.upper + PRICE_EPSILON


BANDS = (
    PremiumBand("LOW_040_050", 0.40, 0.50),
    PremiumBand("HIGH_100_250", 1.00, 2.50),
)


@dataclass(frozen=True)
class EntryEvent:
    signal_index: int
    fill_index: int
    signal_minute: int
    fill_minute: int
    signal_close: float
    raw_fill_open: float
    fill_price: float


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return math.nan
    return numerator / denominator


def normalized_slope(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3 or abs(array[-1]) <= 1e-12:
        return math.nan
    x = np.arange(len(array), dtype=float)
    return float(np.polyfit(x, array, 1)[0] / array[-1])


def rsi(values: Iterable[float], period: int) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < period + 1:
        return math.nan
    delta = np.diff(array[-(period + 1) :])
    gain = float(np.clip(delta, 0, None).mean())
    loss = float((-np.clip(delta, None, 0)).mean())
    if loss <= 1e-12:
        return 100.0 if gain > 0 else 50.0
    relative = gain / loss
    return 100.0 - 100.0 / (1.0 + relative)


def realized_volatility(values: Iterable[float], period: int) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < period + 1 or np.any(array[-(period + 1) :] <= 0):
        return math.nan
    returns = np.diff(np.log(array[-(period + 1) :]))
    return float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0


def moving_average(values: np.ndarray, period: int) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < period:
        return math.nan
    return float(valid[-period:].mean())


def rolling_average(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    series = pd.Series(values, dtype=float)
    return series.rolling(period, min_periods=period).mean().to_numpy()


def episode_count(condition: np.ndarray) -> int:
    if not len(condition):
        return 0
    values = np.asarray(condition, dtype=bool)
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def ma_structure_features(
    values: Iterable[float],
    prefix: str,
    touch_threshold: float,
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    result: dict[str, float | int] = {}
    if not len(array):
        return result
    current = array[-1]
    averages: dict[int, float] = {}
    rolling: dict[int, np.ndarray] = {}
    for period in (5, 20, 60, 120):
        averages[period] = moving_average(array, period)
        rolling[period] = rolling_average(array, period)
        result[f"{prefix}_ma{period}_distance"] = (
            safe_ratio(current - averages[period], current)
            if math.isfinite(averages[period])
            else math.nan
        )
    result[f"{prefix}_bull_stack_5_20_60"] = int(
        all(math.isfinite(averages[item]) for item in (5, 20, 60))
        and current > averages[5] > averages[20] > averages[60]
    )
    result[f"{prefix}_bear_stack_5_20_60"] = int(
        all(math.isfinite(averages[item]) for item in (5, 20, 60))
        and current < averages[5] < averages[20] < averages[60]
    )
    result[f"{prefix}_bull_stack_5_20_60_120"] = int(
        all(math.isfinite(averages[item]) for item in (5, 20, 60, 120))
        and current
        > averages[5]
        > averages[20]
        > averages[60]
        > averages[120]
    )
    lookback = min(30, len(array))
    for short, long in ((5, 20), (5, 60), (20, 60), (20, 120)):
        short_ma = rolling[short][-lookback:]
        long_ma = rolling[long][-lookback:]
        price = array[-lookback:]
        valid = np.isfinite(short_ma) & np.isfinite(long_ma) & (price != 0)
        if valid.sum() < 2:
            result[f"{prefix}_cross_{short}_{long}_count30"] = math.nan
            result[f"{prefix}_touch_{short}_{long}_count30"] = math.nan
            continue
        difference = short_ma[valid] - long_ma[valid]
        signs = np.sign(difference)
        result[f"{prefix}_cross_{short}_{long}_count30"] = int(
            np.sum(signs[1:] * signs[:-1] < 0)
        )
        near = np.abs(difference) / np.abs(price[valid]) <= touch_threshold
        result[f"{prefix}_touch_{short}_{long}_count30"] = episode_count(near)
    return result


def path_features(
    values: Iterable[float],
    prefix: str,
    windows: tuple[int, ...] = (3, 5, 10, 20, 30, 60, 120),
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    result: dict[str, float | int] = {}
    if not len(array):
        return result
    current = float(array[-1])
    result[f"{prefix}_value"] = current
    for window in windows:
        if len(array) >= window + 1:
            previous = float(array[-window - 1])
            result[f"{prefix}_return_{window}"] = (
                current / previous - 1.0
                if abs(previous) > 1e-12
                else math.nan
            )
        else:
            result[f"{prefix}_return_{window}"] = math.nan
        result[f"{prefix}_slope_{window}"] = (
            normalized_slope(array[-window:]) if len(array) >= window else math.nan
        )
    if len(array) >= 10:
        recent = normalized_slope(array[-5:])
        previous = normalized_slope(array[-10:-5])
        result[f"{prefix}_slope_acceleration_5"] = recent - previous
    else:
        result[f"{prefix}_slope_acceleration_5"] = math.nan
    for period in (5, 14, 21):
        result[f"{prefix}_rsi_{period}"] = rsi(array, period)
    for period in (5, 20, 60):
        result[f"{prefix}_realized_volatility_{period}"] = realized_volatility(
            array, period
        )
    result.update(ma_structure_features(array, prefix, touch_threshold=0.005))
    return result


def load_market_minutes(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        skiprows=1,
        encoding="utf-8-sig",
        thousands=",",
        header=0,
    )
    if frame.shape[1] != 5:
        raise RuntimeError(f"{path.name}: 예상하지 못한 열 수 {frame.shape[1]}")
    frame.columns = ["timestamp", "open", "high", "low", "close"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns}
    )


def market_days(spot: pd.DataFrame, futures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    joined = spot.join(futures, how="inner")
    joined = joined.between_time("09:01", "15:30")
    joined["minute"] = joined.index.hour * 100 + joined.index.minute
    return {
        day.strftime("%Y%m%d"): group.copy()
        for day, group in joined.groupby(joined.index.normalize())
    }


def market_snapshot(
    day_frame: pd.DataFrame | None,
    minute: int,
) -> dict[str, float | int]:
    if day_frame is None:
        return {}
    history = day_frame[day_frame["minute"] <= minute]
    if history.empty:
        return {}
    result: dict[str, float | int] = {
        "market_snapshot_minute": int(history.iloc[-1]["minute"]),
        "market_snapshot_lag": minute - int(history.iloc[-1]["minute"]),
    }
    for prefix in ("spot", "futures"):
        closes = history[f"{prefix}_close"].to_numpy(dtype=float)
        result.update(path_features(closes, f"intraday_{prefix}"))
        highs = history[f"{prefix}_high"].to_numpy(dtype=float)
        lows = history[f"{prefix}_low"].to_numpy(dtype=float)
        opening = float(history.iloc[0][f"{prefix}_open"])
        result[f"intraday_{prefix}_range_from_open"] = (
            safe_ratio(float(np.nanmax(highs)) - float(np.nanmin(lows)), opening)
        )
    basis = (
        history["futures_close"].to_numpy(dtype=float)
        - history["spot_close"].to_numpy(dtype=float)
    )
    result.update(path_features(basis, "intraday_basis", (3, 5, 10, 20, 30, 60)))
    spot_returns = history["spot_close"].pct_change()
    futures_returns = history["futures_close"].pct_change()
    for window in (5, 10, 20, 30, 60):
        paired = pd.concat(
            [spot_returns.tail(window), futures_returns.tail(window)], axis=1
        ).dropna()
        result[f"intraday_spot_futures_corr_{window}"] = (
            float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            if len(paired) >= 4
            else math.nan
        )
        lead = pd.concat(
            [
                futures_returns.shift(1).tail(window),
                spot_returns.tail(window),
            ],
            axis=1,
        ).dropna()
        result[f"intraday_futures_leads_spot_corr_{window}"] = (
            float(lead.iloc[:, 0].corr(lead.iloc[:, 1]))
            if len(lead) >= 4
            else math.nan
        )
    return result


def daily_ohlc(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grouped = frame.groupby(frame.index.normalize()).agg(
        {
            f"{prefix}_open": "first",
            f"{prefix}_high": "max",
            f"{prefix}_low": "min",
            f"{prefix}_close": "last",
        }
    )
    grouped.index = pd.DatetimeIndex(grouped.index)
    return grouped


def weekly_ohlc(daily: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return daily.resample("W-FRI").agg(
        {
            f"{prefix}_open": "first",
            f"{prefix}_high": "max",
            f"{prefix}_low": "min",
            f"{prefix}_close": "last",
        }
    ).dropna()


def historical_snapshot(
    frame: pd.DataFrame,
    event_date: str,
    source_column: str,
    feature_prefix: str,
) -> dict[str, float | int]:
    history = frame[frame.index < pd.Timestamp(event_date)]
    if history.empty:
        return {}
    return path_features(
        history[source_column].to_numpy(dtype=float),
        feature_prefix,
        (1, 3, 5, 10, 20, 60, 120),
    )


def vix_frame() -> pd.DataFrame:
    dates, values = strategy1.volatility_closes()
    return pd.DataFrame(
        {"vix_close": values},
        index=pd.to_datetime(dates, format="%Y%m%d"),
    ).sort_index()


def quarterly_expiry_distance(value: str) -> tuple[int, int]:
    current = datetime.strptime(value, "%Y%m%d").date()
    candidates: list[date] = []
    for year in range(current.year, current.year + 2):
        for month in (3, 6, 9, 12):
            first = date(year, month, 1)
            offset = (3 - first.weekday()) % 7
            second_thursday = first + timedelta(days=offset + 7)
            if second_thursday >= current:
                candidates.append(second_thursday)
    selected = min(candidates)
    return (selected - current).days, selected.month


def load_contracts_with_metadata(
    path: Path,
    group: dict[str, str],
) -> tuple[dict[str, base.ContractGrid], dict[str, dict[str, object]]]:
    raw_by_code: dict[str, dict[int, base.RawBar]] = defaultdict(dict)
    metadata: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="cp949", newline="") as source:
        for row in csv.DictReader(source):
            if row["시장ID"] != base.DAY_MARKET_ID:
                continue
            if base.product_kind(row["종목명"]) != group["product_kind"]:
                continue
            if base.extract_series(row["종목명"]) != group["series"]:
                continue
            try:
                minute = int(row["기준시각"].strip().zfill(4))
                raw = base.RawBar(
                    minute=minute,
                    open=float(row["시가"]),
                    high=float(row["고가"]),
                    low=float(row["저가"]),
                    close=float(row["종가"]),
                    volume=int(float(row["거래량"])),
                )
            except (TypeError, ValueError):
                continue
            if minute not in base.MINUTE_INDEX:
                continue
            code = row["종목코드"]
            existing = raw_by_code[code].get(minute)
            if existing is None:
                raw_by_code[code][minute] = raw
            else:
                raw_by_code[code][minute] = base.RawBar(
                    minute=minute,
                    open=existing.open,
                    high=max(existing.high, raw.high),
                    low=min(existing.low, raw.low),
                    close=raw.close,
                    volume=existing.volume + raw.volume,
                )
            metadata.setdefault(
                code,
                {
                    "option_name": row["종목명"],
                    "call_put": row["콜풋구분"],
                    "strike": finite(str(row["행사가격"]).replace(",", "")),
                },
            )
    contracts = {
        code: base.build_contract(
            code,
            {
                "name": str(metadata[code]["option_name"]),
                "call_put": str(metadata[code]["call_put"]),
            },
            raw,
        )
        for code, raw in raw_by_code.items()
        if len(raw) >= 2
    }
    return contracts, metadata


def find_entry(contract: base.ContractGrid, band: PremiumBand) -> EntryEvent | None:
    start = base.MINUTE_INDEX[ENTRY_START]
    for signal_index in range(start, len(contract.bars)):
        signal_bar = contract.bars[signal_index]
        if signal_bar.minute >= ENTRY_CUTOFF:
            break
        if (
            not signal_bar.traded
            or signal_bar.close is None
            or not band.contains(float(signal_bar.close))
        ):
            continue
        fill = base.next_traded_bar(contract, signal_index, MAX_ENTRY_WAIT)
        if fill is None:
            continue
        fill_index, fill_bar = fill
        if (
            fill_bar.minute >= ENTRY_CUTOFF
            or fill_bar.open is None
            or not band.contains(float(fill_bar.open))
        ):
            continue
        return EntryEvent(
            signal_index=signal_index,
            fill_index=fill_index,
            signal_minute=signal_bar.minute,
            fill_minute=fill_bar.minute,
            signal_close=float(signal_bar.close),
            raw_fill_open=float(fill_bar.open),
            fill_price=round(float(fill_bar.open) + base.SLIPPAGE, 4),
        )
    return None


def is_exact_tenth(value: float | None) -> bool:
    return (
        value is not None
        and math.isfinite(float(value))
        and abs(float(value) * 10.0 - round(float(value) * 10.0)) <= 1e-8
    )


def maximum_no_trade_run(bars: list[base.MinuteBar]) -> int:
    longest = 0
    current = 0
    for bar in bars:
        if bar.traded:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def option_features(
    contract: base.ContractGrid,
    snapshot_index: int,
    prefix: str = "option",
) -> dict[str, float | int]:
    closes = [
        float(bar.close)
        for bar in contract.bars[: snapshot_index + 1]
        if bar.close is not None
    ]
    result = path_features(
        closes,
        prefix,
        (1, 3, 5, 10, 20, 30, 60, 120),
    )
    volumes = np.asarray(
        [bar.volume for bar in contract.bars[: snapshot_index + 1]],
        dtype=float,
    )
    current_volume = float(volumes[-1])
    for period in (5, 20, 60):
        previous = volumes[-period - 1 : -1]
        result[f"{prefix}_volume_ratio_{period}"] = (
            current_volume / float(previous.mean())
            if len(previous) == period and previous.mean() > 0
            else math.nan
        )
    history = contract.bars[: snapshot_index + 1]
    recent = history[-20:]
    traded = [bar for bar in recent if bar.traded]
    total_volume = sum(bar.volume for bar in traded)
    round_volume = sum(
        bar.volume
        for bar in traded
        if is_exact_tenth(float(bar.close) if bar.close is not None else None)
    )
    ranges = [
        (float(bar.high) - float(bar.low)) / float(bar.close)
        for bar in traded
        if (
            bar.high is not None
            and bar.low is not None
            and bar.close is not None
            and float(bar.close) > 0
        )
    ]
    result.update(
        {
            f"{prefix}_traded_bars": sum(bar.traded for bar in history),
            f"{prefix}_recent20_traded_ratio": (
                len(traded) / len(recent) if recent else math.nan
            ),
            f"{prefix}_recent20_total_volume": total_volume,
            f"{prefix}_recent20_round_010_volume_share": (
                round_volume / total_volume
                if total_volume > 0
                else math.nan
            ),
            f"{prefix}_recent20_round_010_bar_share": (
                sum(
                    is_exact_tenth(
                        float(bar.close) if bar.close is not None else None
                    )
                    for bar in traded
                )
                / len(traded)
                if traded
                else math.nan
            ),
            f"{prefix}_recent20_max_no_trade_run": maximum_no_trade_run(
                recent
            ),
            f"{prefix}_recent20_mean_range_pct": (
                float(np.mean(ranges)) if ranges else math.nan
            ),
        }
    )
    if prefix == "option":
        result["option_traded_bars_before_entry"] = result[
            "option_traded_bars"
        ]
    return result


def entry_liquidity_features(
    contract: base.ContractGrid,
    event: EntryEvent,
) -> dict[str, float | int]:
    signal_bar = contract.bars[event.signal_index]
    fill_bar = contract.bars[event.fill_index]
    return {
        "entry_wait_minutes": minute_distance(
            event.signal_minute,
            event.fill_minute,
        ),
        "entry_immediate_next_minute": int(
            minute_distance(event.signal_minute, event.fill_minute) == 1
        ),
        "signal_bar_volume": signal_bar.volume,
        "fill_bar_volume": fill_bar.volume,
        "fill_capacity_10pct_contracts": math.floor(
            fill_bar.volume * base.ENTRY_PARTICIPATION + 1e-12
        ),
        "signal_exact_010": int(
            is_exact_tenth(
                float(signal_bar.close)
                if signal_bar.close is not None
                else None
            )
        ),
        "fill_exact_010": int(is_exact_tenth(event.raw_fill_open)),
        "fill_bar_range_pct": (
            (float(fill_bar.high) - float(fill_bar.low))
            / float(fill_bar.open)
            if (
                fill_bar.high is not None
                and fill_bar.low is not None
                and fill_bar.open is not None
                and float(fill_bar.open) > 0
            )
            else math.nan
        ),
    }


def at_2x_landmark_features(
    contract: base.ContractGrid,
    event: EntryEvent,
    outcome: dict[str, float | int | None],
    day_market: pd.DataFrame | None,
) -> dict[str, float | int]:
    raw_minute = outcome.get("target_2x_minute")
    if raw_minute is None:
        return {}
    target_minute = int(raw_minute)
    target_index = base.MINUTE_INDEX[target_minute]
    target_bar = contract.bars[target_index]
    result: dict[str, float | int] = {
        "at2x_minute": target_minute,
        "at2x_minutes_since_entry": minute_distance(
            event.fill_minute,
            target_minute,
        ),
        "at2x_close_multiple": (
            float(target_bar.close) / event.fill_price
            if target_bar.close is not None
            else math.nan
        ),
        "at2x_high_multiple": (
            float(target_bar.high) / event.fill_price
            if target_bar.high is not None
            else math.nan
        ),
        "at2x_bar_volume": target_bar.volume,
    }
    result.update(
        option_features(
            contract,
            target_index,
            prefix="at2x_option",
        )
    )
    market_values = market_snapshot(day_market, target_minute)
    result.update(
        {
            f"at2x_{key}": value
            for key, value in market_values.items()
        }
    )

    path = [
        bar
        for bar in contract.bars[event.fill_index + 1 : target_index + 1]
        if bar.traded and bar.close is not None
    ]
    path_closes = np.asarray(
        [float(bar.close) for bar in path],
        dtype=float,
    )
    absolute_move = (
        float(np.abs(np.diff(path_closes)).sum())
        if len(path_closes) >= 2
        else 0.0
    )
    result["at2x_path_efficiency"] = (
        abs(float(path_closes[-1]) - float(path_closes[0])) / absolute_move
        if len(path_closes) >= 2 and absolute_move > 1e-12
        else math.nan
    )
    result["at2x_path_traded_bars"] = len(path)
    result["at2x_path_min_multiple"] = (
        min(
            float(bar.low) / event.fill_price
            for bar in path
            if bar.low is not None
        )
        if any(bar.low is not None for bar in path)
        else math.nan
    )

    later = [
        bar
        for bar in contract.bars[target_index + 1 :]
        if (
            bar.minute <= OUTCOME_END
            and bar.traded
            and bar.high is not None
        )
    ]
    future_maximum = max(
        (float(bar.high) for bar in later),
        default=event.fill_price,
    )
    target_10x_minute = outcome.get("target_10x_minute")
    result.update(
        {
            "at2x_future_max_multiple": future_maximum / event.fill_price,
            "at2x_reached_5x_later": int(
                future_maximum + PRICE_EPSILON >= event.fill_price * 5
            ),
            "at2x_reached_10x_later": int(
                future_maximum + PRICE_EPSILON >= event.fill_price * 10
            ),
            "at2x_same_bar_10x": int(
                target_10x_minute is not None
                and int(target_10x_minute) == target_minute
            ),
        }
    )
    return result


def future_outcome(
    contract: base.ContractGrid,
    event: EntryEvent,
) -> dict[str, float | int | None]:
    later = [
        (index, bar)
        for index, bar in enumerate(
            contract.bars[event.fill_index + 1 :], start=event.fill_index + 1
        )
        if bar.minute <= OUTCOME_END and bar.traded
    ]
    highs = [
        float(bar.high)
        for _, bar in later
        if bar.high is not None
    ]
    lows = [
        float(bar.low)
        for _, bar in later
        if bar.low is not None
    ]
    maximum = max(highs, default=event.fill_price)
    minimum = min(lows, default=event.fill_price)
    result: dict[str, float | int | None] = {
        "maximum_future_high": maximum,
        "minimum_future_low": minimum,
        "max_multiple": maximum / event.fill_price,
        "mae_pct": (minimum / event.fill_price - 1.0) * 100.0,
    }
    for multiple in (2, 5, 10):
        target = event.fill_price * multiple
        target_minute = next(
            (
                bar.minute
                for _, bar in later
                if (
                    bar.high is not None
                    and float(bar.high) + 1e-9 >= target
                )
            ),
            None,
        )
        result[f"reached_{multiple}x"] = int(target_minute is not None)
        result[f"target_{multiple}x_minute"] = target_minute
        result[f"minutes_to_{multiple}x"] = (
            minute_distance(event.fill_minute, target_minute)
            if target_minute is not None
            else None
        )
    return result


def minute_distance(start: int, end: int) -> int:
    start_minutes = start // 100 * 60 + start % 100
    end_minutes = end // 100 * 60 + end % 100
    return end_minutes - start_minutes


def is_call(value: str) -> bool:
    upper = value.upper()
    return "CALL" in upper or "콜" in value


def add_directional_features(
    row: dict[str, object],
    direction: int,
) -> None:
    for key, value in list(row.items()):
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            continue
        if (
            ("intraday_spot_" in key or "intraday_futures_" in key)
            and ("return_" in key or "slope_" in key or "ma" in key)
            and "volatility" not in key
        ):
            row[f"signed_{key}"] = float(value) * direction


def feature_cache_for_date(
    date_value: str,
    daily_spot: pd.DataFrame,
    daily_futures: pd.DataFrame,
    weekly_spot: pd.DataFrame,
    weekly_futures: pd.DataFrame,
    vix: pd.DataFrame,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    result.update(
        historical_snapshot(
            daily_spot,
            date_value,
            "spot_close",
            "daily_spot",
        )
    )
    result.update(
        historical_snapshot(
            daily_futures,
            date_value,
            "futures_close",
            "daily_futures",
        )
    )
    result.update(
        historical_snapshot(
            weekly_spot,
            date_value,
            "spot_close",
            "weekly_spot",
        )
    )
    result.update(
        historical_snapshot(
            weekly_futures,
            date_value,
            "futures_close",
            "weekly_futures",
        )
    )
    result.update(
        historical_snapshot(vix, date_value, "vix_close", "vix")
    )
    dte, front_month = quarterly_expiry_distance(date_value)
    parsed = datetime.strptime(date_value, "%Y%m%d")
    result.update(
        {
            "calendar_month": parsed.month,
            "calendar_weekday": parsed.weekday(),
            "quarterly_futures_dte": dte,
            "quarterly_front_month": front_month,
            "quarterly_expiry_near_2d": int(dte <= 2),
            "quarterly_expiry_near_10d": int(dte <= 10),
        }
    )
    return result


def build_events() -> tuple[pd.DataFrame, dict[str, object]]:
    print("[1/4] 현물·선물 분봉 로딩", flush=True)
    spot = load_market_minutes(SPOT_PATH, "spot")
    futures = load_market_minutes(FUTURES_PATH, "futures")
    day_market = market_days(spot, futures)
    daily_spot = daily_ohlc(spot, "spot")
    daily_futures = daily_ohlc(futures, "futures")
    weekly_spot = weekly_ohlc(daily_spot, "spot")
    weekly_futures = weekly_ohlc(daily_futures, "futures")
    vix = vix_frame()

    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= COMMON_START
    ]
    files = base.target_file_map()
    daily_cache: dict[str, dict[str, float | int]] = {}
    market_cache: dict[tuple[str, int], dict[str, float | int]] = {}
    events: list[dict[str, object]] = []

    print(f"[2/4] 옵션 만기 {len(groups)}개 사건 생성", flush=True)
    for group_index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        if date_value not in files:
            continue
        contracts, metadata = load_contracts_with_metadata(
            files[date_value], group
        )
        daily_values = daily_cache.setdefault(
            date_value,
            feature_cache_for_date(
                date_value,
                daily_spot,
                daily_futures,
                weekly_spot,
                weekly_futures,
                vix,
            ),
        )
        for code, contract in contracts.items():
            contract_metadata = metadata[code]
            call = is_call(str(contract_metadata["call_put"]))
            direction = 1 if call else -1
            for band in BANDS:
                entry = find_entry(contract, band)
                if entry is None:
                    continue
                current_day_market = day_market.get(date_value)
                cache_key = (date_value, entry.signal_minute)
                market_values = market_cache.setdefault(
                    cache_key,
                    market_snapshot(
                        current_day_market,
                        entry.signal_minute,
                    ),
                )
                outcome = future_outcome(contract, entry)
                entry_option_values = option_features(
                    contract,
                    entry.signal_index,
                )
                liquidity_values = entry_liquidity_features(contract, entry)
                landmark_values = at_2x_landmark_features(
                    contract,
                    entry,
                    outcome,
                    current_day_market,
                )
                row: dict[str, object] = {
                    "expiry_date": date_value,
                    "year": int(date_value[:4]),
                    "product_kind": group["product_kind"],
                    "series": group["series"],
                    "option_code": code,
                    "option_name": contract_metadata["option_name"],
                    "call_put": contract_metadata["call_put"],
                    "is_call": int(call),
                    "direction_sign": direction,
                    "strike": contract_metadata["strike"],
                    "premium_band": band.key,
                    "band_low": int(band.key == "LOW_040_050"),
                    "signal_minute": entry.signal_minute,
                    "fill_minute": entry.fill_minute,
                    "signal_close": entry.signal_close,
                    "raw_fill_open": entry.raw_fill_open,
                    "fill_price": entry.fill_price,
                    **daily_values,
                    **market_values,
                    **entry_option_values,
                    **liquidity_values,
                    **outcome,
                    **landmark_values,
                }
                spot_value = finite(row.get("intraday_spot_value"))
                futures_value = finite(row.get("intraday_futures_value"))
                strike = finite(row.get("strike"))
                if strike is not None and spot_value is not None:
                    row["signed_spot_moneyness"] = (
                        (spot_value - strike) / spot_value * direction
                    )
                if strike is not None and futures_value is not None:
                    row["signed_futures_moneyness"] = (
                        (futures_value - strike) / futures_value * direction
                    )
                add_directional_features(row, direction)
                events.append(row)
        if group_index % 20 == 0 or group_index == len(groups):
            print(
                f"  {group_index}/{len(groups)} 만기, 사건 {len(events):,}건",
                flush=True,
            )

    frame = pd.DataFrame(events).sort_values(
        ["expiry_date", "premium_band", "option_code"]
    )
    profile_summary = {
        "common_period_start": (
            groups[0]["expiry_date"] if groups else COMMON_START
        ),
        "common_period_end": groups[-1]["expiry_date"] if groups else None,
        "expiry_groups": len(groups),
        "event_rows": len(frame),
        "spot_minute_start": spot.index.min().isoformat(),
        "spot_minute_end": spot.index.max().isoformat(),
        "futures_minute_start": futures.index.min().isoformat(),
        "futures_minute_end": futures.index.max().isoformat(),
        "entry_window": [ENTRY_START, ENTRY_CUTOFF],
        "outcome_end": OUTCOME_END,
        "definition": (
            "09:30~12:59 첫 완료봉 종가가 구간에 들어오고, 5분 안의 "
            "다음 실거래봉 시가도 같은 구간일 때 시가+0.01 진입. "
            "진입 다음 봉부터 15:15까지의 고가를 체결가와 비교."
        ),
    }
    return frame, profile_summary


def percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100.0, 4) if denominator else 0.0


def outcome_group(rows: pd.DataFrame) -> dict[str, object]:
    multiples = rows["max_multiple"].astype(float)
    return {
        "events": len(rows),
        "expiry_days": int(rows["expiry_date"].nunique()),
        "reached_2x": int(rows["reached_2x"].sum()),
        "reached_2x_pct": percentage(int(rows["reached_2x"].sum()), len(rows)),
        "reached_5x": int(rows["reached_5x"].sum()),
        "reached_5x_pct": percentage(int(rows["reached_5x"].sum()), len(rows)),
        "reached_10x": int(rows["reached_10x"].sum()),
        "reached_10x_pct": percentage(int(rows["reached_10x"].sum()), len(rows)),
        "median_max_multiple": round(float(multiples.median()), 4),
        "p90_max_multiple": round(float(multiples.quantile(0.90)), 4),
        "p99_max_multiple": round(float(multiples.quantile(0.99)), 4),
        "median_mae_pct": round(float(rows["mae_pct"].median()), 4),
    }


def descriptive_summary(frame: pd.DataFrame) -> dict[str, object]:
    by_band = {
        band: outcome_group(rows)
        for band, rows in frame.groupby("premium_band")
    }
    by_band_year = [
        {
            "premium_band": band,
            "year": int(year),
            **outcome_group(rows),
        }
        for (band, year), rows in frame.groupby(["premium_band", "year"])
    ]
    by_call_put = [
        {
            "premium_band": band,
            "call_put": str(call_put),
            **outcome_group(rows),
        }
        for (band, call_put), rows in frame.groupby(
            ["premium_band", "call_put"]
        )
    ]
    expiry_oracle = (
        frame.groupby(["expiry_date", "premium_band"], as_index=False)[
            "max_multiple"
        ]
        .max()
        .rename(columns={"max_multiple": "best_contract_multiple"})
    )
    oracle_by_band = {}
    for band, rows in expiry_oracle.groupby("premium_band"):
        values = rows["best_contract_multiple"]
        oracle_by_band[band] = {
            "warning": (
                "같은 만기일의 최종 최고배수 계약을 사후 선택한 상한이며 "
                "실행 전략 수익률이 아님"
            ),
            "expiry_days": len(rows),
            "days_2x_pct": percentage(int((values >= 2).sum()), len(rows)),
            "days_5x_pct": percentage(int((values >= 5).sum()), len(rows)),
            "days_10x_pct": percentage(int((values >= 10).sum()), len(rows)),
        }
    return {
        "by_band": by_band,
        "by_band_year": by_band_year,
        "by_call_put": by_call_put,
        "expiry_oracle_upper_bound": oracle_by_band,
    }


IDENTIFIER_COLUMNS = {
    "expiry_date",
    "year",
    "product_kind",
    "series",
    "option_code",
    "option_name",
    "call_put",
    "premium_band",
}
OUTCOME_COLUMNS = {
    "maximum_future_high",
    "minimum_future_low",
    "max_multiple",
    "mae_pct",
    "reached_2x",
    "reached_5x",
    "reached_10x",
    "target_2x_minute",
    "target_5x_minute",
    "target_10x_minute",
    "minutes_to_2x",
    "minutes_to_5x",
    "minutes_to_10x",
}


def false_discovery_rate(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        candidate = values[original_index] * len(values) / rank
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def top_fraction_lift(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0 or float(y_true.mean()) <= 0:
        return math.nan
    count = max(1, math.ceil(len(y_true) * 0.20))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].mean() / y_true.mean())


def metric_document(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    if len(np.unique(y_true)) < 2:
        return {
            "rows": len(y_true),
            "positives": int(y_true.sum()),
            "auc": None,
            "average_precision": None,
            "balanced_accuracy": None,
            "top_20pct_lift": None,
        }
    prediction = (scores >= 0.5).astype(int)
    return {
        "rows": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence_pct": round(float(y_true.mean() * 100.0), 4),
        "auc": round(float(roc_auc_score(y_true, scores)), 4),
        "average_precision": round(
            float(average_precision_score(y_true, scores)), 4
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, prediction)), 4
        ),
        "top_20pct_lift": round(top_fraction_lift(y_true, scores), 4),
    }


def analyze_binary_task(
    frame: pd.DataFrame,
    task_name: str,
    mask: pd.Series,
    target: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    task = frame.loc[mask].copy()
    task["target"] = target.loc[mask].astype(int)
    development = task[task["expiry_date"] <= DEVELOPMENT_END]
    validation = task[task["expiry_date"] > DEVELOPMENT_END]
    numeric_columns = [
        column
        for column in task.select_dtypes(include=[np.number]).columns
        if column not in OUTCOME_COLUMNS
        and column not in IDENTIFIER_COLUMNS
        and column != "target"
        and task[column].notna().mean() >= 0.35
        and task[column].nunique(dropna=True) > 1
    ]
    univariate: list[dict[str, object]] = []
    for column in numeric_columns:
        train = development[[column, "target"]].dropna()
        test = validation[[column, "target"]].dropna()
        if (
            train["target"].nunique() < 2
            or test["target"].nunique() < 2
            or train.groupby("target").size().min() < 8
            or test.groupby("target").size().min() < 3
        ):
            continue
        negative = train.loc[train["target"] == 0, column].to_numpy()
        positive = train.loc[train["target"] == 1, column].to_numpy()
        statistic = mannwhitneyu(
            positive, negative, alternative="two-sided"
        )
        raw_train_auc = roc_auc_score(train["target"], train[column])
        direction = 1.0 if raw_train_auc >= 0.5 else -1.0
        train_auc = roc_auc_score(
            train["target"], train[column] * direction
        )
        validation_auc = roc_auc_score(
            test["target"], test[column] * direction
        )
        correlation, correlation_p = spearmanr(
            train[column], train["target"]
        )
        univariate.append(
            {
                "task": task_name,
                "feature": column,
                "development_rows": len(train),
                "validation_rows": len(test),
                "development_negative_median": float(np.median(negative)),
                "development_positive_median": float(np.median(positive)),
                "development_auc_oriented": float(train_auc),
                "validation_auc_oriented": float(validation_auc),
                "direction": int(direction),
                "mann_whitney_p": float(statistic.pvalue),
                "spearman": float(correlation),
                "spearman_p": float(correlation_p),
            }
        )
    q_values = false_discovery_rate(
        [float(item["mann_whitney_p"]) for item in univariate]
    )
    for item, q_value in zip(univariate, q_values):
        item["mann_whitney_q"] = q_value
    univariate.sort(
        key=lambda item: (
            -float(item["validation_auc_oriented"]),
            float(item["mann_whitney_q"]),
        )
    )

    candidates = sorted(
        univariate,
        key=lambda item: (
            -float(item["development_auc_oriented"]),
            float(item["mann_whitney_q"]),
        ),
    )
    selected: list[str] = []
    for item in candidates:
        feature = str(item["feature"])
        if feature in selected:
            continue
        selected.append(feature)
        if len(selected) >= 12:
            break
    model_document: dict[str, object] = {
        "selected_features": selected,
        "development_rows": len(development),
        "development_positives": int(development["target"].sum()),
        "validation_rows": len(validation),
        "validation_positives": int(validation["target"].sum()),
    }
    if (
        selected
        and development["target"].nunique() == 2
        and validation["target"].nunique() == 2
    ):
        transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    selected,
                )
            ],
            remainder="drop",
        )
        logistic = Pipeline(
            [
                ("features", transformer),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )
        logistic.fit(development[selected], development["target"])
        development_scores = logistic.predict_proba(
            development[selected]
        )[:, 1]
        validation_scores = logistic.predict_proba(validation[selected])[:, 1]
        model_document["logistic_development"] = metric_document(
            development["target"].to_numpy(), development_scores
        )
        model_document["logistic_validation"] = metric_document(
            validation["target"].to_numpy(), validation_scores
        )

        imputer = SimpleImputer(strategy="median")
        development_values = imputer.fit_transform(development[selected])
        validation_values = imputer.transform(validation[selected])
        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=max(12, int(len(development) * 0.04)),
            class_weight="balanced",
            random_state=42,
        )
        tree.fit(development_values, development["target"])
        tree_development_scores = tree.predict_proba(development_values)[:, 1]
        tree_validation_scores = tree.predict_proba(validation_values)[:, 1]
        model_document["tree_rules"] = export_text(
            tree, feature_names=selected
        )
        model_document["tree_development"] = metric_document(
            development["target"].to_numpy(), tree_development_scores
        )
        model_document["tree_validation"] = metric_document(
            validation["target"].to_numpy(), tree_validation_scores
        )
        validation_copy = validation[
            ["expiry_date", "option_code", "premium_band", "target"]
        ].copy()
        validation_copy["score"] = validation_scores
        selected_per_expiry = validation_copy.loc[
            validation_copy.groupby("expiry_date")["score"].idxmax()
        ]
        if selected_per_expiry["target"].nunique() == 2:
            model_document["logistic_validation_best_event_per_expiry"] = (
                metric_document(
                    selected_per_expiry["target"].to_numpy(),
                    selected_per_expiry["score"].to_numpy(),
                )
            )
    validation_logistic = model_document.get("logistic_validation", {})
    validation_tree = model_document.get("tree_validation", {})
    model_document["meaningful_for_s6"] = bool(
        isinstance(validation_logistic, dict)
        and isinstance(validation_tree, dict)
        and (validation_logistic.get("positives") or 0) >= 10
        and (validation_logistic.get("auc") or 0) >= 0.60
        and (validation_logistic.get("top_20pct_lift") or 0) >= 1.35
        and (validation_tree.get("auc") or 0) >= 0.56
    )
    return model_document, univariate


def indicator_analysis(
    frame: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    tasks: dict[str, tuple[pd.Series, pd.Series]] = {
        "TEN_PLUS_VS_TWO_TO_FIVE": (
            (frame["max_multiple"] >= 2)
            & ((frame["max_multiple"] < 5) | (frame["max_multiple"] >= 10)),
            frame["max_multiple"] >= 10,
        ),
        "FIVE_PLUS_VS_TWO_TO_FIVE": (
            frame["max_multiple"] >= 2,
            frame["max_multiple"] >= 5,
        ),
        "TEN_PLUS_VS_ALL_LOWER": (
            pd.Series(True, index=frame.index),
            frame["max_multiple"] >= 10,
        ),
    }
    documents: dict[str, object] = {}
    all_univariate: list[dict[str, object]] = []
    for name, (mask, target) in tasks.items():
        document, univariate = analyze_binary_task(
            frame, name, mask, target
        )
        documents[name] = document
        all_univariate.extend(univariate)
    return documents, pd.DataFrame(all_univariate)


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"JSON 변환 불가: {type(value)!r}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, data_profile = build_events()
    if frame.empty:
        raise RuntimeError("분석할 옵션 진입 사건이 없습니다.")
    print("[3/4] 지표 차이와 시계열 분할 검증", flush=True)
    descriptive = descriptive_summary(frame)
    models, univariate = indicator_analysis(frame)
    meaningful_tasks = [
        name
        for name, document in models.items()
        if bool(document.get("meaningful_for_s6"))
    ]
    result = {
        "status": "temporary_indicator_research",
        "generated_at": datetime.now().astimezone().isoformat(),
        "data": data_profile,
        "guardrails": {
            "label": (
                "실제 다음 봉 진입가 이후 15:15까지의 미래 고가 배수"
            ),
            "same_entry_bar_excluded": True,
            "features_end_at": "진입 신호 완료봉 또는 그 이전",
            "development_end": DEVELOPMENT_END,
            "validation_is_independent": False,
            "validation_note": (
                "시간순 후반 안정성 확인용이며 연구자가 전체 기간의 "
                "맥락을 알고 있어 완전한 독립 전진검증은 아님"
            ),
            "multiple_testing": "Mann-Whitney p에 BH-FDR q 적용",
            "expiry_clustering": (
                "분할은 만기일 기준이며, 동일 만기 계약 간 상관 때문에 "
                "행 단위 p값을 단독 채택 근거로 사용하지 않음"
            ),
        },
        "descriptive": descriptive,
        "models": models,
        "meaningful_tasks": meaningful_tasks,
        "s6_decision": (
            "S6 후보 백테스트 진행"
            if meaningful_tasks
            else "후반 구간에서 유지되는 지표가 없어 S6 승격 보류"
        ),
    }
    print("[4/4] 임시 산출물 저장", flush=True)
    frame.to_csv(
        OUTPUT_DIR / "s6_event_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    univariate.to_csv(
        OUTPUT_DIR / "s6_univariate_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "s6_indicator_research_summary.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "events": len(frame),
                "by_band": descriptive["by_band"],
                "meaningful_tasks": meaningful_tasks,
                "s6_decision": result["s6_decision"],
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
