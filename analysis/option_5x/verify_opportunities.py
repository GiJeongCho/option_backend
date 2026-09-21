from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from inventory import DATASET_NAMES, DATE_RE, ROWDATA_DIR


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SERIES_RE = re.compile(r"\b(\d{4}W\d)\b")
PRICE_EPSILON = 1e-9
PREMIUM_BANDS: tuple[tuple[float, float, str], ...] = (
    (1.00, 1.50, "1.00-1.49"),
    (1.50, 2.00, "1.50-1.99"),
    (2.00, 2.50, "2.00-2.49"),
    (2.50, 3.00, "2.50-2.99"),
    (3.00, 3.50, "3.00-3.49"),
    (3.50, 4.00, "3.50-3.99"),
    (4.00, 4.50, "4.00-4.49"),
    (4.50, 5.01, "4.50-5.00"),
)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def file_map() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for dataset_name in DATASET_NAMES:
        for path in (ROWDATA_DIR / dataset_name).glob("*.csv"):
            match = DATE_RE.search(path.name)
            if match:
                result[match.group(1)] = path
    return result


def load_expected() -> tuple[
    dict[tuple[str, str], tuple[bool, bool]],
    dict[tuple[str, str, str], tuple[bool, bool]],
]:
    overall: dict[tuple[str, str], tuple[bool, bool]] = {}
    with (OUTPUT_DIR / "opportunities.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            overall[(row["expiry_date"], row["option_code"])] = (
                as_bool(row["success_5x"]),
                as_bool(row["success_10x"]),
            )

    bands: dict[tuple[str, str, str], tuple[bool, bool]] = {}
    with (OUTPUT_DIR / "premium_band_details.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            bands[
                (row["expiry_date"], row["option_code"], row["premium_band"])
            ] = (
                as_bool(row["success_5x"]),
                as_bool(row["success_10x"]),
            )
    return overall, bands


def classify(
    bars: list[tuple[int, float, float]]
) -> tuple[tuple[bool, bool] | None, dict[str, tuple[bool, bool]]]:
    ordered = sorted(bars)
    suffix_high_after = [float("-inf")] * len(ordered)
    running_high = float("-inf")

    for index in range(len(ordered) - 1, -1, -1):
        suffix_high_after[index] = running_high
        running_high = max(running_high, ordered[index][1])

    eligible = False
    success_5x = False
    success_10x = False
    band_results: dict[str, tuple[bool, bool]] = {}

    for lower, upper, label in PREMIUM_BANDS:
        band_lows = [
            (low, suffix_high_after[index])
            for index, (_, _, low) in enumerate(ordered)
            if lower <= low < upper
        ]
        if band_lows:
            band_results[label] = (
                any(
                    later_high + PRICE_EPSILON >= 5.0 * low
                    for low, later_high in band_lows
                ),
                any(
                    later_high + PRICE_EPSILON >= 10.0 * low
                    for low, later_high in band_lows
                ),
            )

    for index, (_, _, low) in enumerate(ordered):
        if not 1.0 <= low <= 5.0:
            continue
        eligible = True
        later_high = suffix_high_after[index]
        success_5x = (
            success_5x or later_high + PRICE_EPSILON >= 5.0 * low
        )
        success_10x = (
            success_10x or later_high + PRICE_EPSILON >= 10.0 * low
        )

    return ((success_5x, success_10x) if eligible else None), band_results


def verify() -> dict[str, object]:
    groups = json.loads((OUTPUT_DIR / "expiry_groups.json").read_text("utf-8"))
    paths = file_map()
    expected_overall, expected_bands = load_expected()
    actual_overall: dict[tuple[str, str], tuple[bool, bool]] = {}
    actual_bands: dict[tuple[str, str, str], tuple[bool, bool]] = {}
    invalid_rows = 0

    for group in groups:
        date = group["expiry_date"]
        bars_by_code: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)

        with paths[date].open("r", encoding="cp949", newline="") as source:
            for row in csv.DictReader(source):
                is_monday = "위클리M" in row["종목명"]
                kind = "MONDAY" if is_monday else "THURSDAY"
                series_match = SERIES_RE.search(row["종목명"])

                if row["시장ID"] != "DRV" or kind != group["product_kind"]:
                    continue
                if series_match is None or series_match.group(1) != group["series"]:
                    continue

                try:
                    time = int(row["기준시각"].strip().zfill(4))
                    high = float(row["고가"])
                    low = float(row["저가"])
                except (TypeError, ValueError):
                    invalid_rows += 1
                    continue

                code = row["종목코드"]
                if time in bars_by_code[code]:
                    old_high, old_low = bars_by_code[code][time]
                    high = max(high, old_high)
                    low = min(low, old_low)
                bars_by_code[code][time] = (high, low)

        for code, by_time in bars_by_code.items():
            bars = [
                (time, high, low) for time, (high, low) in by_time.items()
            ]
            overall, bands = classify(bars)
            if overall is not None:
                actual_overall[(date, code)] = overall
            for label, outcome in bands.items():
                actual_bands[(date, code, label)] = outcome

    overall_missing = sorted(set(expected_overall) - set(actual_overall))
    overall_extra = sorted(set(actual_overall) - set(expected_overall))
    overall_mismatch = sorted(
        key
        for key in set(expected_overall) & set(actual_overall)
        if expected_overall[key] != actual_overall[key]
    )
    band_missing = sorted(set(expected_bands) - set(actual_bands))
    band_extra = sorted(set(actual_bands) - set(expected_bands))
    band_mismatch = sorted(
        key
        for key in set(expected_bands) & set(actual_bands)
        if expected_bands[key] != actual_bands[key]
    )

    invariant_errors: list[str] = []
    with (OUTPUT_DIR / "opportunities.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            for multiple in ("5", "10"):
                if not as_bool(row[f"success_{multiple}x"]):
                    continue
                base = float(row[f"base_{multiple}x"])
                base_time = int(row[f"base_time_{multiple}x"])
                target_time = int(row[f"target_time_{multiple}x"])
                target_high = float(row[f"target_high_{multiple}x"])
                if not 1.0 <= base <= 5.0:
                    invariant_errors.append(
                        f"{row['expiry_date']} {row['option_code']}: base {base}"
                    )
                if base_time >= target_time:
                    invariant_errors.append(
                        f"{row['expiry_date']} {row['option_code']}: time order"
                    )
                if target_high + 1e-12 < int(multiple) * base:
                    invariant_errors.append(
                        f"{row['expiry_date']} {row['option_code']}: ratio"
                    )

    passed = not any(
        (
            overall_missing,
            overall_extra,
            overall_mismatch,
            band_missing,
            band_extra,
            band_mismatch,
            invariant_errors,
        )
    )
    return {
        "passed": passed,
        "verified_expiry_groups": len(groups),
        "verified_overall_records": len(actual_overall),
        "verified_band_records": len(actual_bands),
        "invalid_rows_seen": invalid_rows,
        "overall_missing_count": len(overall_missing),
        "overall_extra_count": len(overall_extra),
        "overall_mismatch_count": len(overall_mismatch),
        "band_missing_count": len(band_missing),
        "band_extra_count": len(band_extra),
        "band_mismatch_count": len(band_mismatch),
        "invariant_error_count": len(invariant_errors),
        "samples": {
            "overall_missing": overall_missing[:5],
            "overall_extra": overall_extra[:5],
            "overall_mismatch": overall_mismatch[:5],
            "band_missing": band_missing[:5],
            "band_extra": band_extra[:5],
            "band_mismatch": band_mismatch[:5],
            "invariant_errors": invariant_errors[:5],
        },
    }


if __name__ == "__main__":
    result = verify()
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
