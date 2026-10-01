from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_STEM = "strategy5_v05"
SUMMARY_PATH = OUTPUT_DIR / "strategy5_v05_summary.json"
AUDIT_PATH = OUTPUT_DIR / "strategy5_v05_independent_audit.json"
STRATEGY_VERSION = "S5-v0.5"
INITIAL_BUDGET = 5_000_000
EXPECTED_INITIAL_CASH = base.INITIAL_CASH
EXPECTED_ENTRY_RAW_MIN = 0.40
EXPECTED_ENTRY_RAW_MAX = 1.20
EXPECTED_TARGET = 3.907583
EXPECTED_TARGET_REASON = "EXPECTED_PEAK_3_9076X_ALL"


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


def run() -> dict[str, object]:
    document = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    daily = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv")
    trades = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv")
    positions = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv")
    errors: list[str] = []

    if document["strategy_version"] != STRATEGY_VERSION:
        errors.append("strategy version")
    if document["rules"]["reentry"] != "disabled":
        errors.append("reentry rule")
    if any(
        row["reason"] == "LATE_1_5X_REENTRY"
        or (
            row["side"] == "BUY"
            and row["reason"] != "09_10_PREMIUM_BIN_BASKET"
        )
        for row in trades
    ):
        errors.append("reentry trade exists")
    if any(row["branch"] == "LATE_REENTRY" for row in positions):
        errors.append("reentry position exists")

    trades_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    trades_by_position: dict[
        tuple[str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    positions_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[trade["date"]].append(trade)
        trades_by_position[
            (trade["date"], trade["code"], trade["campaign"])
        ].append(trade)
    for position in positions:
        positions_by_date[position["date"]].append(position)

    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = {
        group["expiry_date"]: group
        for group in base.derive_expiry_groups(profile)
        if legacy.PPT_START
        <= group["expiry_date"]
        <= legacy.PPT_END
    }
    files = base.target_file_map()
    expected_cash = EXPECTED_INITIAL_CASH

    for daily_row in daily:
        date_value = daily_row["date"]
        day_trades = trades_by_date[date_value]
        pnl = integer(daily_row["pnl"])
        if integer(daily_row["start_cash"]) != expected_cash:
            errors.append(f"{date_value}: cash chain")
        if integer(daily_row["end_cash"]) != expected_cash + pnl:
            errors.append(f"{date_value}: pnl equation")
        if sum(integer(row["cash_flow"]) for row in day_trades) != pnl:
            errors.append(f"{date_value}: trade cash flow")
        initial_principal = sum(
            integer(row["principal"])
            for row in day_trades
            if row["side"] == "BUY"
        )
        if initial_principal > INITIAL_BUDGET:
            errors.append(f"{date_value}: initial budget")
        buy_cash_required = sum(
            -integer(row["cash_flow"])
            for row in day_trades
            if row["side"] == "BUY"
        )
        if buy_cash_required > integer(daily_row["start_cash"]):
            errors.append(f"{date_value}: insufficient cash")
        if integer(daily_row["late_buy_principal"]) != 0:
            errors.append(f"{date_value}: late principal")
        if (
            str(daily_row["late_signal_found"]).lower() == "true"
            or str(daily_row["late_reentry_filled"]).lower() == "true"
        ):
            errors.append(f"{date_value}: late state")

        quantities: dict[tuple[str, str], int] = defaultdict(int)
        sells_by_bar: dict[
            tuple[str, int], list[dict[str, str]]
        ] = defaultdict(list)
        for trade in day_trades:
            quantity = integer(trade["quantity"])
            key = (trade["code"], trade["campaign"])
            fee = integer(trade["fee"])
            principal = integer(trade["principal"])
            cash_flow = integer(trade["cash_flow"])
            if trade["side"] == "BUY":
                quantities[key] += quantity
                if not 911 <= integer(trade["minute"]) <= 915:
                    errors.append(f"{date_value}: entry minute")
                if quantity > math.floor(
                    integer(trade["bar_volume"])
                    * base.ENTRY_PARTICIPATION
                ):
                    errors.append(f"{date_value}: entry volume")
                if cash_flow != -(principal + fee):
                    errors.append(f"{date_value}: buy cash")
            elif trade["side"] == "SELL":
                quantities[key] -= quantity
                sells_by_bar[
                    (trade["code"], integer(trade["minute"]))
                ].append(trade)
                if cash_flow != principal - fee:
                    errors.append(f"{date_value}: sell cash")
            elif trade["side"] == "EXPIRE":
                quantities[key] -= quantity
                if principal or fee or cash_flow:
                    errors.append(f"{date_value}: expiry cash")
            else:
                errors.append(f"{date_value}: side")
        if any(quantity != 0 for quantity in quantities.values()):
            errors.append(f"{date_value}: open quantity")
        for key, sale_rows in sells_by_bar.items():
            if all(
                row["reason"] == "NO_1_5X_09_30_EXIT"
                for row in sale_rows
            ):
                continue
            maximum = math.floor(
                integer(sale_rows[0]["bar_volume"])
                * base.EXIT_PARTICIPATION
            )
            if (
                sum(integer(row["quantity"]) for row in sale_rows)
                > maximum
            ):
                errors.append(f"{date_value}: exit volume {key}")

        contracts = base.load_expiry_day(
            files[date_value],
            groups[date_value],
        )
        for position in positions_by_date[date_value]:
            contract = contracts[position["code"]]
            campaign_rows = trades_by_position[
                (
                    date_value,
                    position["code"],
                    position["campaign"],
                )
            ]
            observed_index: int | None = None
            observed_price = 0.0
            for index in range(
                base.MINUTE_INDEX[legacy.MORNING_START],
                base.MINUTE_INDEX[legacy.MORNING_END] + 1,
            ):
                bar = contract.bars[index]
                if not bar.traded:
                    continue
                values = legacy.observed_values(bar)
                if values:
                    observed_index = index
                    observed_price = max(values)
                    break
            if (
                observed_index is None
                or contract.bars[observed_index].minute
                != integer(position["observation_minute"])
                or not math.isclose(
                    observed_price,
                    number(position["observation_price"]),
                    abs_tol=1e-9,
                )
            ):
                errors.append(f"{date_value}: P0 {position['code']}")
                continue

            entry_minute = integer(position["entry_minute"])
            entry_index = base.MINUTE_INDEX[entry_minute]
            entry_bar = contract.bars[entry_index]
            if (
                not entry_bar.traded
                or entry_bar.open is None
                or not EXPECTED_ENTRY_RAW_MIN
                <= float(entry_bar.open)
                <= EXPECTED_ENTRY_RAW_MAX
                or not math.isclose(
                    float(entry_bar.open) + base.SLIPPAGE,
                    number(position["entry_price"]),
                    abs_tol=1e-9,
                )
            ):
                errors.append(f"{date_value}: entry price")

            confirmation_index: int | None = None
            threshold = observed_price * legacy.CONFIRMATION_MULTIPLE
            for index in range(
                entry_index,
                base.MINUTE_INDEX[929] + 1,
            ):
                bar = contract.bars[index]
                if (
                    bar.traded
                    and bar.high is not None
                    and float(bar.high) >= threshold
                ):
                    confirmation_index = index
                    break
            reported_confirmation = integer(
                position["confirmation_minute"]
            )
            if confirmation_index is None:
                if (
                    position["branch"] != "EARLY_FAILED"
                    or reported_confirmation
                ):
                    errors.append(f"{date_value}: failed branch")
                forced_sales = [
                    row
                    for row in campaign_rows
                    if row["side"] == "SELL"
                    and row["reason"] == "NO_1_5X_09_30_EXIT"
                ]
                zero_closures = [
                    row
                    for row in campaign_rows
                    if row["side"] == "EXPIRE"
                    and row["reason"] == "NO_09_30_LIQUIDITY_ZERO"
                ]
                exit_bar = contract.bars[base.MINUTE_INDEX[930]]
                if exit_bar.traded and exit_bar.open is not None:
                    expected_exit_price = max(
                        0.0, float(exit_bar.open) - base.SLIPPAGE
                    )
                    if (
                        len(forced_sales) != 1
                        or zero_closures
                        or integer(forced_sales[0]["minute"]) != 930
                        or integer(forced_sales[0]["quantity"])
                        != integer(position["quantity"])
                        or not math.isclose(
                            number(forced_sales[0]["price"]),
                            expected_exit_price,
                            abs_tol=1e-9,
                        )
                    ):
                        errors.append(f"{date_value}: 09:30 full exit")
                elif (
                    forced_sales
                    or len(zero_closures) != 1
                    or integer(zero_closures[0]["minute"]) != 930
                    or integer(zero_closures[0]["quantity"])
                    != integer(position["quantity"])
                    or integer(zero_closures[0]["cash_flow"]) != 0
                ):
                    errors.append(f"{date_value}: 09:30 zero close")
            else:
                if (
                    position["branch"] != "EARLY_CONFIRMED"
                    or reported_confirmation
                    != contract.bars[confirmation_index].minute
                ):
                    errors.append(f"{date_value}: confirmed branch")

            target = observed_price * EXPECTED_TARGET
            for sale in campaign_rows:
                if sale["reason"] != EXPECTED_TARGET_REASON:
                    continue
                sale_index = base.MINUTE_INDEX[integer(sale["minute"])]
                if not any(
                    bar.traded
                    and bar.high is not None
                    and float(bar.high) >= target
                    for bar in contract.bars[entry_index:sale_index]
                ):
                    errors.append(f"{date_value}: target without signal")

            if (
                sum(integer(row["cash_flow"]) for row in campaign_rows)
                != integer(position["pnl"])
            ):
                errors.append(f"{date_value}: position pnl")
        expected_cash = integer(daily_row["end_cash"])

    summary = document["summary"]
    if integer(summary["ending_cash"]) != expected_cash:
        errors.append("summary ending cash")
    if (
        integer(summary["total_pnl"])
        != expected_cash - EXPECTED_INITIAL_CASH
    ):
        errors.append("summary total pnl")

    result = {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "position_rows": len(positions),
        "checks": [
            "independent P0 and actual entry-price reconstruction",
            "post-entry 1.5x confirmation only through 09:29",
            "no reentry state, trade, position, or recycled principal",
            (
                "full 09:30 open liquidation and "
                f"{EXPECTED_TARGET:g}x target ordering"
            ),
            "cash, quantity, fees, and normal-order volume limits",
        ],
    }
    AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
