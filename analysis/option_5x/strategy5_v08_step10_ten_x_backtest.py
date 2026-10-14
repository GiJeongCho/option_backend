from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import minimal_baseline_backtest as base
import strategy5_v04_basket_backtest as v04
import strategy5_v06_ten_x_backtest as v06


STRATEGY_VERSION = "S5-v0.8-STEP10-10X"
OUTPUT_STEM = "strategy5_v08_step10_ten_x"
INITIAL_CASH = 40_000_000
ALLOWED_CENTS = tuple(range(40, 121, 10))
PRICE_TOLERANCE_CENTS = 1e-7

_ORIGINAL_PREMIUM_BIN = v04.premium_bin
_ORIGINAL_OBSERVATION = v04.observation


def exact_quote_cents(price: float) -> int | None:
    """Return the quote in cents only for 0.40, 0.50, ..., 1.20."""
    cents = round(price * 100)
    if (
        cents in ALLOWED_CENTS
        and abs(price * 100 - cents) <= PRICE_TOLERANCE_CENTS
    ):
        return cents
    return None


def exact_quote_level(price: float) -> str:
    cents = exact_quote_cents(price)
    if cents is None:
        return "INELIGIBLE"
    return f"{cents / 100:.1f}"


def exact_quote_observation(
    contract: base.ContractGrid,
) -> v04.Setup | None:
    setup = _ORIGINAL_OBSERVATION(contract)
    if setup is None:
        return None
    fill = v04.setup_fill(contract)
    if fill is None:
        return None
    raw_entry_price = fill[2] - base.SLIPPAGE
    cents = exact_quote_cents(raw_entry_price)
    if cents is None:
        return None
    return replace(setup, premium_bin=f"{cents / 100:.1f}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_quote_audit() -> dict[str, object]:
    trades_path = v04.OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv"
    daily_path = v04.OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv"
    trades = _read_csv(trades_path)
    daily = _read_csv(daily_path)
    errors: list[str] = []

    buys = [row for row in trades if row["side"] == "BUY"]
    principal_by_day: dict[str, int] = {}
    levels_by_day: dict[str, set[int]] = {}
    for buy in buys:
        raw_price = float(buy["price"]) - base.SLIPPAGE
        cents = exact_quote_cents(raw_price)
        if cents is None:
            errors.append(
                f"{buy['date']} {buy['code']}: invalid raw entry "
                f"{raw_price:.8f}"
            )
            continue
        principal_by_day[buy["date"]] = (
            principal_by_day.get(buy["date"], 0)
            + int(float(buy["principal"]))
        )
        levels_by_day.setdefault(buy["date"], set()).add(cents)

    for date_value, principal in principal_by_day.items():
        if principal > v04.INITIAL_BUDGET:
            errors.append(f"{date_value}: budget {principal:,}")

    previous_end: int | None = None
    for row in daily:
        start_cash = int(float(row["start_cash"]))
        end_cash = int(float(row["end_cash"]))
        if previous_end is not None and start_cash != previous_end:
            errors.append(f"{row['date']}: cash continuity")
        if start_cash < 0 or end_cash < 0:
            errors.append(f"{row['date']}: negative cash")
        executed_levels = len(levels_by_day.get(row["date"], set()))
        actual_levels = int(float(row["active_premium_bins"]))
        if not executed_levels <= actual_levels <= len(ALLOWED_CENTS):
            errors.append(
                f"{row['date']}: active/executed levels "
                f"{actual_levels}/{executed_levels}"
            )
        required_cash = sum(
            -int(float(trade["cash_flow"]))
            for trade in trades
            if trade["date"] == row["date"] and trade["side"] == "BUY"
        )
        if required_cash > start_cash:
            errors.append(f"{row['date']}: insufficient cash")
        previous_end = end_cash

    if daily and int(float(daily[0]["start_cash"])) != INITIAL_CASH:
        errors.append("initial cash")
    if any(row["strategy"] != STRATEGY_VERSION for row in trades):
        errors.append("trade strategy version")

    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "buy_rows": len(buys),
        "traded_days": len(principal_by_day),
        "observed_entry_levels": [
            f"{cents / 100:.1f}"
            for cents in sorted(
                {
                    cents
                    for day_levels in levels_by_day.values()
                    for cents in day_levels
                }
            )
        ],
        "checks": [
            "every raw entry open is exactly 0.4, 0.5, ..., 1.2",
            "daily buy principal does not exceed 5,000,000 won",
            "cash is continuous, sufficient at entry, and never negative",
            "eligible level count covers all executed exact quote levels",
        ],
    }


def run() -> dict[str, object]:
    original_initial_cash = base.INITIAL_CASH
    original_premium_bin = v04.premium_bin
    original_observation = v04.observation
    original_version = v06.STRATEGY_VERSION
    original_output_stem = v06.OUTPUT_STEM

    try:
        base.INITIAL_CASH = INITIAL_CASH
        v04.premium_bin = exact_quote_level
        v04.observation = exact_quote_observation
        v06.STRATEGY_VERSION = STRATEGY_VERSION
        v06.OUTPUT_STEM = OUTPUT_STEM
        result = v06.run()
    finally:
        base.INITIAL_CASH = original_initial_cash
        v04.premium_bin = original_premium_bin
        v04.observation = original_observation
        v06.STRATEGY_VERSION = original_version
        v06.OUTPUT_STEM = original_output_stem

    quote_audit = exact_quote_audit()
    result["status"] = "experimental_exact_step_quote_logic_check"
    result["rules"]["entry"] = (
        "from the 09:10 decision, accept only a first executable raw "
        "open exactly equal to 0.4, 0.5, ..., 1.2; split at most "
        "5,000,000 won equally across active exact quote levels"
    )
    result["rules"]["entry_levels"] = [
        f"{cents / 100:.1f}" for cents in ALLOWED_CENTS
    ]
    result["rules"]["entry_level_interpretation"] = (
        "The requested 0.11 and 0.12 are interpreted as 1.1 and 1.2."
    )
    result["rules"]["initial_cash"] = INITIAL_CASH
    result["exact_quote_audit"] = quote_audit
    result["audit"]["error_count"] = (
        int(result["audit"]["error_count"])
        + int(quote_audit["error_count"])
    )
    result["audit"]["error_samples"] = [
        *result["audit"]["error_samples"],
        *quote_audit["error_samples"],
    ][:60]
    result["audit"]["checks"] = [
        *result["audit"]["checks"],
        *quote_audit["checks"],
    ]
    result["audit"]["passed"] = (
        bool(result["audit"]["passed"]) and quote_audit["passed"]
    )
    result["limitations"].append(
        "If the exact quote is absent, a nearby price such as 0.41 or "
        "1.19 is not substituted and that contract is not bought."
    )

    summary_path = v04.OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (v04.OUTPUT_DIR / f"{OUTPUT_STEM}_exact_quote_audit.json").write_text(
        json.dumps(quote_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (v04.OUTPUT_DIR / f"{OUTPUT_STEM}_audit.json").write_text(
        json.dumps(result["audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not quote_audit["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
