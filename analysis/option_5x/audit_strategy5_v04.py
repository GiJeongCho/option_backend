from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_PATH = OUTPUT_DIR / "strategy5_v04_summary.json"
AUDIT_PATH = OUTPUT_DIR / "strategy5_v04_independent_audit.json"

INITIAL_CASH = 80_000_000
INITIAL_BUDGET = 5_000_000
PPT_COUNTS = {
    "eligible": 1320,
    "early_1_5x": 633,
    "same_bar_ambiguous": 37,
    "failed_1_5x": 650,
}
PPT_BUCKETS = {
    "1_5_TO_2": 153,
    "2_TO_5": 262,
    "5_TO_10": 92,
    "10_PLUS": 126,
}
PPT_FLOORS = {
    "1_5_TO_2": 1.5,
    "2_TO_5": 2.0,
    "5_TO_10": 5.0,
    "10_PLUS": 10.0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def integer(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def number(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def boolean(value: object) -> bool:
    return str(value).lower() == "true"


def raw_event_counts() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if legacy.PPT_START
        <= group["expiry_date"]
        <= legacy.PPT_END
    ]
    files = base.target_file_map()
    rows: list[tuple[bool, bool, float]] = []
    for group in groups:
        contracts = base.load_expiry_day(
            files[group["expiry_date"]],
            group,
        )
        for contract in contracts.values():
            observed: tuple[int, float] | None = None
            for index in range(
                base.MINUTE_INDEX[legacy.MORNING_START],
                base.MINUTE_INDEX[legacy.MORNING_END] + 1,
            ):
                bar = contract.bars[index]
                if not bar.traded:
                    continue
                values = legacy.observed_values(bar)
                if values:
                    observed = (index, max(values))
                    break
            if observed is None:
                continue
            index, price = observed
            bar = contract.bars[index]
            ambiguous = (
                bar.high is not None
                and float(bar.high)
                >= price * legacy.CONFIRMATION_MULTIPLE
            )
            deadline = min(
                len(contract.bars) - 1,
                index + legacy.CONFIRMATION_WINDOW_MINUTES,
            )
            confirmed = any(
                current.traded
                and current.high is not None
                and float(current.high)
                >= price * legacy.CONFIRMATION_MULTIPLE
                for current in contract.bars[index + 1 : deadline + 1]
            )
            peaks = [
                float(current.high)
                for current in contract.bars[index + 1 :]
                if current.traded and current.high is not None
            ]
            peak = max(peaks, default=price) / price
            rows.append((ambiguous, confirmed, peak))
    ordered = [row for row in rows if not row[0]]
    confirmed = [row for row in ordered if row[1]]
    peaks = [row[2] for row in confirmed]
    buckets = {
        "1_5_TO_2": sum(1.5 <= value < 2.0 for value in peaks),
        "2_TO_5": sum(2.0 <= value < 5.0 for value in peaks),
        "5_TO_10": sum(5.0 <= value < 10.0 for value in peaks),
        "10_PLUS": sum(value >= 10.0 for value in peaks),
    }
    return {
        "definition": (
            "first qualifying completed OHLC bar; maximum qualifying "
            "OHLC is P0"
        ),
        "eligible": len(rows),
        "same_bar_ambiguous": sum(row[0] for row in rows),
        "ordered_confirmed": len(confirmed),
        "ordered_failed": len(ordered) - len(confirmed),
        "confirmed_peak_buckets": buckets,
        "confirmed_peak_median": round(median(peaks), 4),
        "maximum_peak_multiple": round(max(peaks), 4),
    }


def audit_portfolio(
    name: str,
    document: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    suffix = name.lower()
    daily = read_csv(
        OUTPUT_DIR / f"strategy5_v04_{suffix}_daily.csv"
    )
    trades = read_csv(
        OUTPUT_DIR / f"strategy5_v04_{suffix}_trades.csv"
    )
    positions = read_csv(
        OUTPUT_DIR / f"strategy5_v04_{suffix}_positions.csv"
    )
    portfolio = dict(document["portfolios"])[name]  # type: ignore[arg-type]
    summary = portfolio["summary"]
    version = str(portfolio["strategy_version"])
    if len(daily) != integer(document["scope"]["expiry_days"]):
        errors.append(f"daily rows {len(daily)}")
    if len({row["date"] for row in daily}) != len(daily):
        errors.append("duplicate dates")

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        by_date[trade["date"]].append(trade)
    positions_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for position in positions:
        positions_by_date[position["date"]].append(position)
    expected_cash = INITIAL_CASH
    for row in daily:
        date_value = row["date"]
        day_trades = by_date[date_value]
        pnl = integer(row["pnl"])
        if row["strategy"] != version:
            errors.append(f"{date_value}: daily strategy")
        if integer(row["start_cash"]) != expected_cash:
            errors.append(f"{date_value}: cash chain")
        if integer(row["end_cash"]) != expected_cash + pnl:
            errors.append(f"{date_value}: pnl equation")
        if sum(integer(item["cash_flow"]) for item in day_trades) != pnl:
            errors.append(f"{date_value}: trade cash flow")
        initial_buys = [
            item
            for item in day_trades
            if item["side"] == "BUY"
            and item["reason"] == "09_10_PREMIUM_BIN_BASKET"
        ]
        reentry_buys = [
            item
            for item in day_trades
            if item["side"] == "BUY"
            and item["reason"] == "LATE_1_5X_REENTRY"
        ]
        initial_principal = sum(
            integer(item["principal"]) for item in initial_buys
        )
        reentry_principal = sum(
            integer(item["principal"]) for item in reentry_buys
        )
        if initial_principal != integer(
            row["initial_buy_principal"]
        ):
            errors.append(f"{date_value}: initial principal")
        if reentry_principal != integer(row["late_buy_principal"]):
            errors.append(f"{date_value}: reentry principal")
        if initial_principal > INITIAL_BUDGET:
            errors.append(f"{date_value}: initial budget")
        if integer(row["maximum_deployed_principal"]) > INITIAL_BUDGET:
            errors.append(f"{date_value}: simultaneous budget")
        if reentry_principal > integer(row["failed_sale_proceeds"]):
            errors.append(f"{date_value}: proceeds recycling")
        if any(
            not 911 <= integer(item["minute"]) <= 915
            for item in initial_buys
        ):
            errors.append(f"{date_value}: initial entry time")
        if any(
            not 0.41 - 1e-9
            <= number(item["price"])
            <= 1.21 + 1e-9
            for item in initial_buys
        ):
            errors.append(f"{date_value}: initial premium range")
        if any(
            integer(item["minute"]) <= 930 for item in reentry_buys
        ):
            errors.append(f"{date_value}: reentry time")

        day_positions = positions_by_date[date_value]
        position_by_key = {
            (item["code"], item["campaign"]): item
            for item in day_positions
        }
        failed_keys = {
            key
            for key, item in position_by_key.items()
            if item["branch"] == "EARLY_FAILED"
        }
        for reentry in reentry_buys:
            realized_before_entry = sum(
                integer(item["cash_flow"])
                for item in day_trades
                if item["side"] == "SELL"
                and (item["code"], item["campaign"]) in failed_keys
                and integer(item["minute"])
                < integer(reentry["minute"])
            )
            if integer(reentry["principal"]) > max(
                0, realized_before_entry
            ):
                errors.append(
                    f"{date_value}: future reentry proceeds"
                )

        quantities: dict[tuple[str, str], int] = defaultdict(int)
        sells_by_bar: dict[
            tuple[str, int], list[dict[str, str]]
        ] = defaultdict(list)
        for trade in day_trades:
            if trade["strategy"] != version:
                errors.append(f"{date_value}: trade strategy")
            quantity = integer(trade["quantity"])
            principal = integer(trade["principal"])
            fee = integer(trade["fee"])
            cash_flow = integer(trade["cash_flow"])
            key = (trade["code"], trade["campaign"])
            if trade["side"] == "BUY":
                quantities[key] += quantity
                if quantity > math.floor(
                    integer(trade["bar_volume"])
                    * base.ENTRY_PARTICIPATION
                ):
                    errors.append(f"{date_value}: entry participation")
                if fee != quantity * base.COMMISSION_PER_CONTRACT:
                    errors.append(f"{date_value}: buy fee")
                if principal != round(
                    quantity
                    * number(trade["price"])
                    * base.MULTIPLIER
                ):
                    errors.append(f"{date_value}: buy principal")
                if cash_flow != -(principal + fee):
                    errors.append(f"{date_value}: buy cash flow")
            elif trade["side"] == "SELL":
                quantities[key] -= quantity
                sells_by_bar[
                    (trade["code"], integer(trade["minute"]))
                ].append(trade)
                if fee != quantity * base.COMMISSION_PER_CONTRACT:
                    errors.append(f"{date_value}: sell fee")
                if principal != round(
                    quantity
                    * number(trade["price"])
                    * base.MULTIPLIER
                ):
                    errors.append(f"{date_value}: sell principal")
                if cash_flow != principal - fee:
                    errors.append(f"{date_value}: sell cash flow")
            elif trade["side"] == "EXPIRE":
                quantities[key] -= quantity
                if principal or fee or cash_flow:
                    errors.append(f"{date_value}: expiry cash flow")
            else:
                errors.append(f"{date_value}: unknown side")
        if any(value != 0 for value in quantities.values()):
            errors.append(f"{date_value}: open quantity")
        for key, bar_sells in sells_by_bar.items():
            maximum = math.floor(
                integer(bar_sells[0]["bar_volume"])
                * base.EXIT_PARTICIPATION
            )
            if sum(integer(item["quantity"]) for item in bar_sells) > maximum:
                errors.append(
                    f"{date_value}: exit participation {key}"
                )

        confirmed_principal = sum(
            integer(item["buy_principal"])
            for item in day_positions
            if item["branch"] == "EARLY_CONFIRMED"
        )
        position_failed_proceeds = sum(
            integer(item["net_sale_proceeds"])
            for item in day_positions
            if item["branch"] == "EARLY_FAILED"
        )
        if confirmed_principal + reentry_principal != integer(
            row["maximum_deployed_principal"]
        ):
            errors.append(f"{date_value}: deployed reconciliation")
        if position_failed_proceeds != integer(
            row["failed_sale_proceeds"]
        ):
            errors.append(f"{date_value}: failed proceeds")
        for position in day_positions:
            confirmation = integer(position["confirmation_minute"])
            entry_minute = integer(position["entry_minute"])
            if (
                position["branch"] == "EARLY_CONFIRMED"
                and confirmation < entry_minute
            ):
                errors.append(
                    f"{date_value}: pre-entry confirmation"
                )
            if (
                position["branch"] == "EARLY_FAILED"
                and confirmation
            ):
                errors.append(
                    f"{date_value}: failed confirmation"
                )
            key = (position["code"], position["campaign"])
            campaign_trades = [
                item
                for item in day_trades
                if (item["code"], item["campaign"]) == key
            ]
            if sum(
                integer(item["cash_flow"])
                for item in campaign_trades
            ) != integer(position["pnl"]):
                errors.append(
                    f"{date_value}: position pnl {key}"
                )
        expected_cash = integer(row["end_cash"])

    if integer(summary["initial_cash"]) != INITIAL_CASH:
        errors.append("summary initial cash")
    if integer(summary["ending_cash"]) != expected_cash:
        errors.append("summary ending cash")
    if integer(summary["total_pnl"]) != expected_cash - INITIAL_CASH:
        errors.append("summary total pnl")
    if integer(summary["profitable_days"]) != sum(
        integer(row["pnl"]) > 0 for row in daily
    ):
        errors.append("summary profitable days")
    if integer(summary["losing_days"]) != sum(
        integer(row["pnl"]) < 0 for row in daily
    ):
        errors.append("summary losing days")
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:40],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "position_rows": len(positions),
    }


def run() -> dict[str, object]:
    document = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    if document.get("strategy_version") != "S5-v0.4":
        errors.append("strategy version")
    claim = dict(document["ppt_claim"])
    for name, value in PPT_COUNTS.items():
        if integer(claim[name]) != value:
            errors.append(f"PPT claim {name}")
    if {
        key: integer(value)
        for key, value in dict(claim["early_peak_buckets"]).items()
    } != PPT_BUCKETS:
        errors.append("PPT peak buckets")

    weighted = sum(
        PPT_BUCKETS[name] * PPT_FLOORS[name]
        for name in PPT_BUCKETS
    )
    expectation = dict(document["expectation"])
    if not math.isclose(
        number(expectation["conditional_early_peak_floor_multiple"]),
        weighted / PPT_COUNTS["early_1_5x"],
        abs_tol=1e-6,
    ):
        errors.append("conditional expectation")
    if not math.isclose(
        number(expectation["all_1320_oracle_floor_multiple"]),
        weighted / PPT_COUNTS["eligible"],
        abs_tol=1e-6,
    ):
        errors.append("all-event expectation")

    raw = raw_event_counts()
    reported_raw = dict(document["raw_validation"])
    comparable = {
        "eligible",
        "same_bar_ambiguous",
        "ordered_confirmed",
        "ordered_failed",
        "confirmed_peak_buckets",
        "confirmed_peak_median",
        "maximum_peak_multiple",
    }
    if any(raw[key] != reported_raw[key] for key in comparable):
        errors.append("raw event recomputation")
    portfolio_results = {
        name: audit_portfolio(name, document)
        for name in ("EV", "STAGED")
    }
    for name, result in portfolio_results.items():
        if not result["passed"]:
            errors.extend(
                f"{name}: {message}"
                for message in result["error_samples"]
            )

    result = {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "ppt_claim": {
            **PPT_COUNTS,
            "early_peak_buckets": PPT_BUCKETS,
        },
        "raw_recomputation": raw,
        "raw_matches_ppt": (
            integer(raw["eligible"]) == PPT_COUNTS["eligible"]
            and integer(raw["same_bar_ambiguous"])
            == PPT_COUNTS["same_bar_ambiguous"]
            and integer(raw["ordered_confirmed"])
            == PPT_COUNTS["early_1_5x"]
            and integer(raw["ordered_failed"])
            == PPT_COUNTS["failed_1_5x"]
            and raw["confirmed_peak_buckets"] == PPT_BUCKETS
        ),
        "portfolios": portfolio_results,
        "checks": [
            "PPT arithmetic and weighted peak-floor expectation",
            "independent current-raw event recomputation",
            "cash chain and trade cash-flow reconciliation",
            "5,000,000 won initial and simultaneous principal caps",
            "realized-before-entry failed proceeds for late reentry",
            "post-fill confirmation and post-09:30 reentry ordering",
            "integer quantity closure, multiplier, fees, volume caps",
        ],
    }
    AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
