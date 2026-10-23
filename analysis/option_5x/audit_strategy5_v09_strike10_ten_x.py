from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import audit_strategy5_v05 as base_audit
import minimal_baseline_backtest as raw


STRATEGY_VERSION = "S5-v0.9-STRIKE10-10X"
OUTPUT_STEM = "strategy5_v09_strike10_ten_x"


def source_strikes(
    path: Path,
    codes: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", encoding="cp949", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["종목코드"]
            if code not in codes or code in result:
                continue
            try:
                result[code] = float(
                    row["행사가격"].replace(",", "")
                )
            except (TypeError, ValueError):
                continue
            if len(result) == len(codes):
                break
    return result


def run() -> dict[str, object]:
    base_audit.OUTPUT_STEM = OUTPUT_STEM
    base_audit.SUMMARY_PATH = (
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
    )
    base_audit.AUDIT_PATH = (
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_independent_audit.json"
    )
    base_audit.STRATEGY_VERSION = STRATEGY_VERSION
    base_audit.EXPECTED_INITIAL_CASH = 40_000_000
    base_audit.EXPECTED_ENTRY_RAW_MIN = 0.40
    base_audit.EXPECTED_ENTRY_RAW_MAX = 0.49
    base_audit.EXPECTED_TARGET = 10.0
    base_audit.EXPECTED_TARGET_REASON = "TARGET_10X_ALL"
    result = base_audit.run()

    trades = base_audit.read_csv(
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv"
    )
    buys_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        if trade["side"] == "BUY":
            buys_by_date[trade["date"]].append(trade)

    files = raw.target_file_map()
    extra_errors: list[str] = []
    observed_strikes: set[float] = set()

    for date_value, buys in buys_by_date.items():
        codes = {row["code"] for row in buys}
        strikes = source_strikes(files[date_value], codes)
        if set(strikes) != codes:
            extra_errors.append(f"{date_value}: source strike missing")
        for code, strike in strikes.items():
            observed_strikes.add(strike)
            if not math.isclose(
                strike / 10.0,
                round(strike / 10.0),
                abs_tol=1e-9,
            ):
                extra_errors.append(
                    f"{date_value} {code}: invalid strike {strike}"
                )

        ordered = sorted(
            buys,
            key=lambda row: base_audit.integer(row["campaign"]),
        )
        raw_prices = [
            round(base_audit.number(row["price"]) - raw.SLIPPAGE, 8)
            for row in ordered
        ]
        if raw_prices != sorted(raw_prices):
            extra_errors.append(
                f"{date_value}: raw premium priority {raw_prices}"
            )
        if (
            sum(base_audit.integer(row["principal"]) for row in buys)
            > base_audit.INITIAL_BUDGET
        ):
            extra_errors.append(f"{date_value}: principal limit")

    result["error_samples"] = [
        *result["error_samples"],
        *extra_errors,
    ][:60]
    result["error_count"] = (
        int(result["error_count"]) + len(extra_errors)
    )
    result["passed"] = not result["error_count"]
    result["checks"].extend(
        [
            "source 행사가격 is independently restricted to 10-point steps",
            "BUY campaigns are ordered from raw premium 0.40 through 0.49",
            "daily new-buy principal remains at or below 5,000,000 won",
        ]
    )
    result["observed_strikes"] = sorted(observed_strikes)
    base_audit.AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
