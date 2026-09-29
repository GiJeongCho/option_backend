from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import minimal_baseline_backtest as base


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DAILY_PATH = OUTPUT_DIR / "strategy2_daily.csv"
DEVELOPMENT_START = "20251001"
DEVELOPMENT_END = "20260430"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def period(date_value: str) -> str:
    if date_value < DEVELOPMENT_START:
        return "historical_backcast"
    if date_value <= DEVELOPMENT_END:
        return "development"
    return "validation"


def value_at(values: list[float | None], index: int) -> float | None:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def rising(values: list[float | None], index: int, lookback: int = 5) -> bool:
    current = value_at(values, index)
    previous = value_at(values, index - lookback)
    return (
        current is not None
        and previous is not None
        and current > previous
    )


def short_state(
    ma5: float | None, ma10: float | None, ma20: float | None
) -> str:
    if ma5 is None or ma10 is None or ma20 is None:
        return "unavailable"
    if ma5 > ma10 > ma20:
        return "bull_5_10_20"
    if ma10 >= ma5 > ma20:
        return "ma10_above_ma5"
    if ma5 > ma20 >= ma10:
        return "ma10_below_ma20"
    return "other"


def long_state(
    ma20: float | None, ma40: float | None, ma60: float | None
) -> str:
    if ma20 is None or ma40 is None or ma60 is None:
        return "unavailable"
    if ma20 > ma40 > ma60:
        return "bull_20_40_60"
    if ma20 < ma40 < ma60:
        return "bear_20_40_60"
    return "mixed_20_40_60"


def bucket_volume(value: float) -> str:
    if value < 1.5:
        return "1.00_1.49"
    if value < 2.0:
        return "1.50_1.99"
    if value < 3.0:
        return "2.00_2.99"
    return "3.00_plus"


def bucket_price(value: float) -> str:
    if value < 3.25:
        return "2.50_3.24"
    if value < 4.0:
        return "3.25_3.99"
    return "4.00_5.00"


def bucket_time(value: int) -> str:
    if value < 1000:
        return "before_1000"
    if value < 1200:
        return "1000_1159"
    return "1200_onward"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    pnls = [int(row["pnl"]) for row in rows]
    gains = sum(value for value in pnls if value > 0)
    losses = -sum(value for value in pnls if value < 0)
    positives = sorted((value for value in pnls if value > 0), reverse=True)
    return {
        "count": len(rows),
        "profitable": sum(value > 0 for value in pnls),
        "losing": sum(value < 0 for value in pnls),
        "total_pnl": sum(pnls),
        "average_pnl": round(sum(pnls) / len(pnls)) if pnls else 0,
        "profit_factor": (
            round(gains / losses, 4) if losses else None
        ),
        "pnl_excluding_best_3": sum(pnls) - sum(positives[:3]),
    }


def grouped(
    rows: list[dict[str, object]], field: str
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = []
    for name, items in sorted(groups.items()):
        result.append(
            {
                "group": name,
                "all": summarize(items),
                "development": summarize(
                    [row for row in items if row["period"] == "development"]
                ),
                "validation": summarize(
                    [row for row in items if row["period"] == "validation"]
                ),
            }
        )
    return result


def analyze() -> dict[str, object]:
    daily = [
        row
        for row in read_csv(DAILY_PATH)
        if row["entry_filled"].lower() == "true"
    ]
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = {
        item["expiry_date"]: item
        for item in base.derive_expiry_groups(profile)
    }
    files = base.target_file_map()
    details: list[dict[str, object]] = []

    for row in daily:
        date_value = row["date"]
        contracts = base.load_expiry_day(
            files[date_value], groups[date_value]
        )
        contract = contracts[row["code"]]
        signal_index = base.MINUTE_INDEX[int(row["signal_minute"])]
        entry_index = base.MINUTE_INDEX[int(row["entry_minute"])]
        closes = [bar.close for bar in contract.bars]
        ma10_values = base.rolling_mean(closes, 10)
        ma40_values = base.rolling_mean(closes, 40)
        ma60_values = base.rolling_mean(closes, 60)
        ma5 = value_at(contract.ma5, signal_index)
        ma10 = value_at(ma10_values, signal_index)
        ma20 = value_at(contract.ma20, signal_index)
        ma40 = value_at(ma40_values, signal_index)
        ma60 = value_at(ma60_values, signal_index)
        short = short_state(ma5, ma10, ma20)
        long = long_state(ma20, ma40, ma60)
        bar = contract.bars[signal_index]
        prior_volume = sum(
            item.volume
            for item in contract.bars[signal_index - 20 : signal_index]
        ) / 20
        volume_ratio = bar.volume / prior_volume if prior_volume else 0.0
        first_20_closes = [
            float(item.close)
            for item in contract.bars[
                entry_index : min(entry_index + 21, len(contract.bars))
            ]
            if item.traded and item.close is not None
        ]
        initial_price = float(row["initial_price"])
        first_20_max = (
            max(first_20_closes) / initial_price
            if first_20_closes
            else 0.0
        )
        first_20_end = (
            first_20_closes[-1] / initial_price
            if first_20_closes
            else 0.0
        )
        details.append(
            {
                "date": date_value,
                "period": period(date_value),
                "code": row["code"],
                "call_put": row["call_put"],
                "signal_minute": int(row["signal_minute"]),
                "entry_minute": int(row["entry_minute"]),
                "initial_price": initial_price,
                "pnl": int(row["pnl"]),
                "profitable": int(row["pnl"]) > 0,
                "confirmed": row["confirmed"].lower() == "true",
                "exit_reason": row["exit_reason"],
                "short_state": short,
                "long_state": long,
                "full_bull_alignment": (
                    ma5 is not None
                    and ma10 is not None
                    and ma20 is not None
                    and ma40 is not None
                    and ma60 is not None
                    and ma5 > ma10 > ma20 > ma40 > ma60
                ),
                "short_bull_long_bear_reversal": (
                    short == "bull_5_10_20"
                    and long == "bear_20_40_60"
                ),
                "ma5_rising_5m": rising(
                    contract.ma5, signal_index
                ),
                "ma20_rising_5m": rising(
                    contract.ma20, signal_index
                ),
                "ma40_rising_5m": rising(ma40_values, signal_index),
                "ma60_rising_5m": rising(ma60_values, signal_index),
                "ma5_ma20_both_rising": (
                    rising(contract.ma5, signal_index)
                    and rising(contract.ma20, signal_index)
                ),
                "volume_ratio": round(volume_ratio, 4),
                "volume_ratio_bucket": bucket_volume(volume_ratio),
                "price_bucket": bucket_price(initial_price),
                "entry_time_bucket": bucket_time(
                    int(row["entry_minute"])
                ),
                "first_20m_max_multiple": round(first_20_max, 4),
                "first_20m_end_multiple": round(first_20_end, 4),
            }
        )

    dimensions = {
        field: grouped(details, field)
        for field in (
            "short_state",
            "long_state",
            "full_bull_alignment",
            "short_bull_long_bear_reversal",
            "ma5_rising_5m",
            "ma20_rising_5m",
            "ma40_rising_5m",
            "ma60_rising_5m",
            "ma5_ma20_both_rising",
            "volume_ratio_bucket",
            "price_bucket",
            "entry_time_bucket",
            "call_put",
            "confirmed",
            "exit_reason",
        )
    }
    result = {
        "strategy": "S2-v0.1",
        "definition": {
            "development": (
                f"{DEVELOPMENT_START}_{DEVELOPMENT_END}"
            ),
            "validation": "20260501_onward",
            "moving_averages": [5, 10, 20, 40, 60],
            "feature_time": "completed signal minute only",
        },
        "overall": summarize(details),
        "development": summarize(
            [row for row in details if row["period"] == "development"]
        ),
        "validation": summarize(
            [row for row in details if row["period"] == "validation"]
        ),
        "dimensions": dimensions,
    }
    write_csv(OUTPUT_DIR / "strategy2_feature_details.csv", details)
    (OUTPUT_DIR / "strategy2_feature_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
