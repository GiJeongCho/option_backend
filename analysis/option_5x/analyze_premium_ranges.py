from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from count_opportunities import (
    DAY_MARKET_ID,
    PRICE_EPSILON,
    PROFILE_PATH,
    derive_expiry_groups,
    extract_series,
    product_kind,
    target_file_map,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RANGES = ("below_1", "from_1_to_5", "above_5")


def range_name(low: float) -> str | None:
    if 0 < low < 1.0:
        return "below_1"
    if 1.0 <= low <= 5.0:
        return "from_1_to_5"
    if low > 5.0:
        return "above_5"
    return None


def classify(bars: list[tuple[int, float, float]]) -> dict[str, tuple[bool, bool]]:
    ordered = sorted(bars)
    suffix_high_after = [float("-inf")] * len(ordered)
    running_high = float("-inf")
    for index in range(len(ordered) - 1, -1, -1):
        suffix_high_after[index] = running_high
        running_high = max(running_high, ordered[index][1])

    result = {name: [False, False] for name in RANGES}
    for index, (_, _, low) in enumerate(ordered):
        name = range_name(low)
        if name is None:
            continue
        result[name][0] = True
        if suffix_high_after[index] + PRICE_EPSILON >= 5.0 * low:
            result[name][1] = True
    return {name: (values[0], values[1]) for name, values in result.items()}


def summarize(
    keys_by_range: dict[str, set[tuple[str, str]]],
    eligible_by_range: dict[str, set[tuple[str, str]]],
    expiry_dates: set[str],
) -> dict[str, object]:
    def range_summary(name: str, keys: set[tuple[str, str]]) -> dict[str, object]:
        eligible = eligible_by_range[name]
        event_dates = {date for date, _ in keys}
        return {
            "eligible_contract_days": len(eligible),
            "success_5x_contract_days": len(keys),
            "success_5x_contract_rate_pct": round(
                len(keys) / len(eligible) * 100, 4
            )
            if eligible
            else 0.0,
            "expiry_days_with_5x": len(event_dates),
            "expiry_days_with_5x_rate_pct": round(
                len(event_dates) / len(expiry_dates) * 100, 4
            ),
        }

    below = keys_by_range["below_1"]
    middle = keys_by_range["from_1_to_5"]
    above = keys_by_range["above_5"]
    outside = below | above
    all_prices = outside | middle
    outside_only_contracts = outside - middle

    middle_dates = {date for date, _ in middle}
    outside_dates = {date for date, _ in outside}
    all_dates = {date for date, _ in all_prices}
    outside_only_dates = outside_dates - middle_dates

    return {
        "ranges": {
            name: range_summary(name, keys_by_range[name]) for name in RANGES
        },
        "unique_5x_contract_days_all_positive_premiums": len(all_prices),
        "unique_5x_contract_days_outside_1_to_5": len(outside),
        "unique_5x_contract_days_only_outside_1_to_5": len(
            outside_only_contracts
        ),
        "expiry_days_with_5x_all_positive_premiums": len(all_dates),
        "expiry_days_with_5x_outside_1_to_5": len(outside_dates),
        "expiry_days_with_5x_only_outside_1_to_5": len(outside_only_dates),
        "expiry_days_without_1_to_5_but_with_outside_5x": sorted(
            outside_only_dates
        ),
        "expiry_days_without_any_5x": sorted(expiry_dates - all_dates),
    }


def analyze() -> dict[str, object]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = target_file_map()
    keys_by_range = {name: set() for name in RANGES}
    eligible_by_range = {name: set() for name in RANGES}

    for group in groups:
        bars_by_code: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        with files[group["expiry_date"]].open(
            "r", encoding="cp949", newline=""
        ) as source:
            for row in csv.DictReader(source):
                if row["시장ID"] != DAY_MARKET_ID:
                    continue
                if product_kind(row["종목명"]) != group["product_kind"]:
                    continue
                if extract_series(row["종목명"]) != group["series"]:
                    continue
                try:
                    minute = int(row["기준시각"].strip().zfill(4))
                    high = float(row["고가"])
                    low = float(row["저가"])
                except (TypeError, ValueError):
                    continue
                code = row["종목코드"]
                if minute in bars_by_code[code]:
                    old_high, old_low = bars_by_code[code][minute]
                    high = max(high, old_high)
                    low = min(low, old_low)
                bars_by_code[code][minute] = (high, low)

        for code, values in bars_by_code.items():
            key = (group["expiry_date"], code)
            bars = [
                (minute, high, low)
                for minute, (high, low) in values.items()
            ]
            outcomes = classify(bars)
            for name, (eligible, success) in outcomes.items():
                if eligible:
                    eligible_by_range[name].add(key)
                if success:
                    keys_by_range[name].add(key)

    expiry_dates = {group["expiry_date"] for group in groups}
    full = summarize(keys_by_range, eligible_by_range, expiry_dates)

    dates_2026 = {date for date in expiry_dates if date >= "20260101"}
    keys_2026 = {
        name: {key for key in keys if key[0] >= "20260101"}
        for name, keys in keys_by_range.items()
    }
    eligible_2026 = {
        name: {key for key in keys if key[0] >= "20260101"}
        for name, keys in eligible_by_range.items()
    }

    result = {
        "definition": {
            "same_bar_low_high_allowed": False,
            "unit": "expiry_date + option_code",
            "ranges": {
                "below_1": "0 < low < 1.00",
                "from_1_to_5": "1.00 <= low <= 5.00",
                "above_5": "low > 5.00",
            },
            "overlap_note": "한 종목이 여러 가격대를 통과하면 가격대별 결과에는 중복될 수 있음",
        },
        "all_period": full,
        "year_2026": summarize(keys_2026, eligible_2026, dates_2026),
    }
    (OUTPUT_DIR / "premium_range_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
