from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy
import strategy5_v04_basket_backtest as v04


STRATEGY_VERSION = "S5-v0.5"
OUTPUT_DIR = v04.OUTPUT_DIR
CONFIRMATION_END = 929
FORCED_EXIT_MINUTE = 930


def force_exit_at_0930(
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    quantity: int,
) -> tuple[list[dict[str, object]], int]:
    if quantity <= 0:
        return [], 0
    bar = contract.bars[base.MINUTE_INDEX[FORCED_EXIT_MINUTE]]
    if not bar.traded or bar.open is None:
        return (
            [
                v04.trade_row(
                    STRATEGY_VERSION,
                    date_value,
                    campaign,
                    FORCED_EXIT_MINUTE,
                    contract,
                    "EXPIRE",
                    "NO_09_30_LIQUIDITY_ZERO",
                    9,
                    quantity,
                    0.0,
                    0,
                )
            ],
            quantity,
        )
    return (
        [
            v04.trade_row(
                STRATEGY_VERSION,
                date_value,
                campaign,
                bar.minute,
                contract,
                "SELL",
                "NO_1_5X_09_30_EXIT",
                3,
                quantity,
                max(0.0, float(bar.open) - base.SLIPPAGE),
                bar.volume,
            )
        ],
        0,
    )


def simulate_initial_position(
    date_value: str,
    contract: base.ContractGrid,
    entry: v04.PlannedEntry,
    campaign: int,
    used_capacity: dict[int, int],
) -> v04.PositionOutcome:
    buy = v04.trade_row(
        STRATEGY_VERSION,
        date_value,
        campaign,
        entry.fill_minute,
        contract,
        "BUY",
        "09_10_PREMIUM_BIN_BASKET",
        1,
        entry.quantity,
        entry.fill_price,
        entry.fill_volume,
    )
    confirmation = v04.first_high_touch(
        contract,
        entry.fill_index,
        CONFIRMATION_END,
        entry.setup.observation_price * legacy.CONFIRMATION_MULTIPLE,
    )
    if confirmation is not None:
        remaining, exit_index, sells = v04.confirmed_exit(
            "EV",
            STRATEGY_VERSION,
            date_value,
            campaign,
            contract,
            entry.setup,
            entry.fill_index,
            entry.quantity,
            used_capacity,
        )
    else:
        sells, remaining = force_exit_at_0930(
            date_value,
            campaign,
            contract,
            entry.quantity,
        )
        exit_index = base.MINUTE_INDEX[FORCED_EXIT_MINUTE]
    sell_rows = [row for row in sells if row["side"] == "SELL"]
    return v04.PositionOutcome(
        setup=entry.setup,
        early_confirmed=confirmation is not None,
        early_confirmation_index=confirmation,
        entry_quantity=entry.quantity,
        entry_principal=int(buy["principal"]),
        entry_fee=int(buy["fee"]),
        sold_quantity=sum(int(row["quantity"]) for row in sell_rows),
        net_sale_proceeds=sum(
            int(row["cash_flow"]) for row in sell_rows
        ),
        exit_index=exit_index,
        expired_quantity=remaining,
        trades=[buy, *sells],
    )


def simulate_day(
    date_value: str,
    contracts: dict[str, base.ContractGrid],
    setups: list[v04.Setup],
    start_cash: int,
) -> tuple[
    v04.DayResult,
    list[dict[str, object]],
    list[v04.PositionRecord],
]:
    entries = v04.allocate_entries(contracts, setups)
    used_capacity: dict[str, dict[int, int]] = defaultdict(dict)
    outcomes: list[v04.PositionOutcome] = []
    trades: list[dict[str, object]] = []
    positions: list[v04.PositionRecord] = []
    for campaign, entry in enumerate(entries, start=1):
        contract = contracts[entry.setup.code]
        outcome = simulate_initial_position(
            date_value,
            contract,
            entry,
            campaign,
            used_capacity[contract.code],
        )
        outcomes.append(outcome)
        trades.extend(outcome.trades)
        positions.append(
            v04.PositionRecord(
                strategy=STRATEGY_VERSION,
                date=date_value,
                campaign=campaign,
                code=entry.setup.code,
                call_put=entry.setup.call_put,
                branch=(
                    "EARLY_CONFIRMED"
                    if outcome.early_confirmed
                    else "EARLY_FAILED"
                ),
                observation_minute=entry.setup.observation_minute,
                observation_price=round(
                    entry.setup.observation_price, 4
                ),
                confirmation_minute=(
                    contract.bars[
                        outcome.early_confirmation_index
                    ].minute
                    if outcome.early_confirmation_index is not None
                    else None
                ),
                entry_minute=entry.fill_minute,
                entry_price=round(entry.fill_price, 4),
                quantity=entry.quantity,
                buy_principal=outcome.entry_principal,
                buy_fee=outcome.entry_fee,
                net_sale_proceeds=outcome.net_sale_proceeds,
                pnl=sum(
                    int(item["cash_flow"])
                    for item in outcome.trades
                ),
                expired_quantity=outcome.expired_quantity,
            )
        )

    confirmed = [row for row in outcomes if row.early_confirmed]
    failed = [row for row in outcomes if not row.early_confirmed]
    initial_principal = sum(row.entry_principal for row in outcomes)
    pnl = sum(int(row["cash_flow"]) for row in trades)
    return (
        v04.DayResult(
            strategy=STRATEGY_VERSION,
            date=date_value,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            eligible_contracts=len(setups),
            active_premium_bins=len(
                {setup.premium_bin for setup in setups}
            ),
            initial_positions=len(entries),
            initial_buy_principal=initial_principal,
            early_confirmed_positions=len(confirmed),
            early_failed_positions=len(failed),
            failed_sale_proceeds=sum(
                row.net_sale_proceeds for row in failed
            ),
            late_signal_found=False,
            late_reentry_filled=False,
            late_code="",
            late_call_put="",
            late_confirmation_minute=None,
            late_entry_minute=None,
            late_buy_principal=0,
            total_buy_principal=initial_principal,
            maximum_deployed_principal=initial_principal,
            total_fees=sum(int(row["fee"]) for row in trades),
            expired_quantity=sum(
                int(row["quantity"])
                for row in trades
                if row["side"] == "EXPIRE"
            ),
        ),
        trades,
        positions,
    )


def audit(
    daily: list[v04.DayResult],
    trades: list[dict[str, object]],
    positions: list[v04.PositionRecord],
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_date: dict[
        str, list[dict[str, object]]
    ] = defaultdict(list)
    positions_by_date: dict[str, list[v04.PositionRecord]] = defaultdict(
        list
    )
    for trade in trades:
        trades_by_date[str(trade["date"])].append(trade)
    for position in positions:
        positions_by_date[position.date].append(position)

    expected_cash = base.INITIAL_CASH
    for row in daily:
        day_trades = trades_by_date[row.date]
        if row.strategy != STRATEGY_VERSION:
            errors.append(f"{row.date}: strategy")
        if row.start_cash != expected_cash:
            errors.append(f"{row.date}: cash chain")
        if row.end_cash != row.start_cash + row.pnl:
            errors.append(f"{row.date}: pnl equation")
        if sum(int(item["cash_flow"]) for item in day_trades) != row.pnl:
            errors.append(f"{row.date}: trade cash flow")
        if row.initial_buy_principal > v04.INITIAL_BUDGET:
            errors.append(f"{row.date}: initial budget")
        if row.maximum_deployed_principal != row.initial_buy_principal:
            errors.append(f"{row.date}: deployed principal")
        if (
            row.late_signal_found
            or row.late_reentry_filled
            or row.late_buy_principal
        ):
            errors.append(f"{row.date}: reentry state")
        if any(
            item["reason"] == "LATE_1_5X_REENTRY"
            for item in day_trades
        ):
            errors.append(f"{row.date}: reentry trade")

        quantities: dict[tuple[str, int], int] = defaultdict(int)
        sold_by_bar: dict[
            tuple[str, int], list[dict[str, object]]
        ] = defaultdict(list)
        for trade in day_trades:
            code = str(trade["code"])
            campaign = int(trade["campaign"])
            key = (code, campaign)
            quantity = int(trade["quantity"])
            if trade["side"] == "BUY":
                quantities[key] += quantity
                if trade["reason"] != "09_10_PREMIUM_BIN_BASKET":
                    errors.append(f"{row.date}: unknown buy")
                if not 911 <= int(trade["minute"]) <= 915:
                    errors.append(f"{row.date}: entry time")
                if quantity > math.floor(
                    int(trade["bar_volume"])
                    * base.ENTRY_PARTICIPATION
                ):
                    errors.append(f"{row.date}: entry volume")
            elif trade["side"] == "SELL":
                quantities[key] -= quantity
                sold_by_bar[(code, int(trade["minute"]))].append(trade)
            elif trade["side"] == "EXPIRE":
                quantities[key] -= quantity
            else:
                errors.append(f"{row.date}: side")
        if any(quantity != 0 for quantity in quantities.values()):
            errors.append(f"{row.date}: open quantity")
        for key, rows in sold_by_bar.items():
            if all(
                row["reason"] == "NO_1_5X_09_30_EXIT"
                for row in rows
            ):
                continue
            maximum = math.floor(
                int(rows[0]["bar_volume"]) * base.EXIT_PARTICIPATION
            )
            if sum(int(item["quantity"]) for item in rows) > maximum:
                errors.append(f"{row.date}: exit volume {key}")

        for position in positions_by_date[row.date]:
            campaign_rows = [
                item
                for item in day_trades
                if str(item["code"]) == position.code
                and int(item["campaign"]) == position.campaign
            ]
            if position.branch not in {
                "EARLY_CONFIRMED",
                "EARLY_FAILED",
            }:
                errors.append(f"{row.date}: branch")
            if (
                position.branch == "EARLY_CONFIRMED"
                and (
                    position.confirmation_minute is None
                    or position.confirmation_minute
                    < position.entry_minute
                    or position.confirmation_minute > CONFIRMATION_END
                )
            ):
                errors.append(f"{row.date}: confirmation time")
            if (
                position.branch == "EARLY_FAILED"
                and position.confirmation_minute is not None
            ):
                errors.append(f"{row.date}: failed confirmation")
            if position.branch == "EARLY_FAILED":
                forced_sales = [
                    item
                    for item in campaign_rows
                    if item["side"] == "SELL"
                    and item["reason"] == "NO_1_5X_09_30_EXIT"
                ]
                zero_closures = [
                    item
                    for item in campaign_rows
                    if item["side"] == "EXPIRE"
                    and item["reason"] == "NO_09_30_LIQUIDITY_ZERO"
                ]
                forced_rows = [*forced_sales, *zero_closures]
                if (
                    len(forced_rows) != 1
                    or int(forced_rows[0]["minute"])
                    != FORCED_EXIT_MINUTE
                    or int(forced_rows[0]["quantity"])
                    != position.quantity
                ):
                    errors.append(f"{row.date}: 09:30 full exit")
                allowed = {
                    "09_10_PREMIUM_BIN_BASKET",
                    "NO_1_5X_09_30_EXIT",
                    "NO_09_30_LIQUIDITY_ZERO",
                    "UNFILLED_ZERO_VALUE",
                }
            else:
                allowed = {
                    "09_10_PREMIUM_BIN_BASKET",
                    "EXPECTED_PEAK_3_9076X_ALL",
                    "TIME",
                    "UNFILLED_ZERO_VALUE",
                }
            if any(
                str(item["reason"]) not in allowed
                for item in campaign_rows
            ):
                errors.append(f"{row.date}: exit reason")
            if (
                sum(int(item["cash_flow"]) for item in campaign_rows)
                != position.pnl
            ):
                errors.append(f"{row.date}: position pnl")
        expected_cash = row.end_cash

    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:50],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "position_rows": len(positions),
        "checks": [
            "no late reentry signal, buy, position, or capital",
            "post-fill confirmation through the completed 09:29 bar",
            "full 09:30 open liquidation for every unconfirmed position",
            "3.907583x P0 target only for confirmed positions",
            "cash, quantity, fees, and normal-order volume limits",
        ],
    }


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if legacy.PPT_START
        <= group["expiry_date"]
        <= legacy.PPT_END
    ]
    files = base.target_file_map()
    cash = base.INITIAL_CASH
    daily: list[v04.DayResult] = []
    trades: list[dict[str, object]] = []
    positions: list[v04.PositionRecord] = []

    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        setups = [
            setup
            for contract in contracts.values()
            if (setup := v04.observation(contract)) is not None
        ]
        row, day_trades, day_positions = simulate_day(
            date_value,
            contracts,
            setups,
            cash,
        )
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)
        positions.extend(day_positions)

    development = [
        row for row in daily if row.date <= legacy.DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > legacy.DEVELOPMENT_END
    ]
    audit_result = audit(daily, trades, positions)
    summary = {
        **v04.summarize(daily),
        "initial_cash": base.INITIAL_CASH,
        "ending_cash": cash,
    }
    result = {
        "status": "experimental_no_reentry_expected_peak_exit",
        "strategy_version": STRATEGY_VERSION,
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": legacy.DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "expectation": {
            "source": "PPT 633 confirmed peak buckets",
            "conditional_peak_floor_multiple": round(
                v04.PPT_EXPECTED_PEAK_FLOOR, 6
            ),
            "warning": (
                "3.907583x is an ex-post lower-bound average of intraday "
                "peaks, not a guaranteed executable expected sale price."
            ),
        },
        "rules": {
            "entry": (
                "after 09:10, buy actual 0.40~1.20 opens by equally "
                "funded 0.1 premium bins, maximum 5,000,000 won"
            ),
            "confirmation": (
                "post-fill 1.5x P0 high touch through the completed "
                "09:29 bar"
            ),
            "confirmed_exit": "all at 3.907583x P0, otherwise 15:15",
            "unconfirmed_exit": (
                "sell every remaining contract at the 09:30 open "
                "regardless of profit, loss, or the 10% volume cap"
            ),
            "reentry": "disabled",
            "slippage": base.SLIPPAGE,
            "commission_per_contract": base.COMMISSION_PER_CONTRACT,
            "volume_participation": base.ENTRY_PARTICIPATION,
            "forced_09_30_exit_volume_cap": "disabled",
        },
        "forced_exit": {
            "minute": FORCED_EXIT_MINUTE,
            "full_fill_positions": sum(
                row["reason"] == "NO_1_5X_09_30_EXIT"
                for row in trades
            ),
            "full_fill_quantity": sum(
                int(row["quantity"])
                for row in trades
                if row["reason"] == "NO_1_5X_09_30_EXIT"
            ),
            "no_trade_zero_positions": sum(
                row["reason"] == "NO_09_30_LIQUIDITY_ZERO"
                for row in trades
            ),
            "no_trade_zero_quantity": sum(
                int(row["quantity"])
                for row in trades
                if row["reason"] == "NO_09_30_LIQUIDITY_ZERO"
            ),
        },
        "summary": summary,
        "development": v04.summarize(development),
        "validation": v04.summarize(validation),
        "position_expectation": v04.summarize_positions(positions),
        "worst_days": [
            asdict(row)
            for row in sorted(daily, key=lambda item: item.pnl)[:15]
        ],
        "audit": audit_result,
        "limitations": [
            (
                "The PPT 1,320/633/37/650 counts are not reproduced "
                "exactly by the current explicit raw-data definition."
            ),
            (
                "The 3.907583x target was derived after observing the "
                "full PPT period and is not an independent parameter."
            ),
            (
                "The development/validation split is diagnostic only; "
                "a new expiry period is required."
            ),
            (
                "The requested exact 09:30 liquidation disables the 10% "
                "exit-volume cap; a contract with no actual 09:30 trade "
                "is conservatively closed at zero instead of using a "
                "later bar."
            ),
        ],
    }

    v04.write_csv(
        OUTPUT_DIR / "strategy5_v05_daily.csv",
        [asdict(row) for row in daily],
    )
    v04.write_csv(
        OUTPUT_DIR / "strategy5_v05_trades.csv",
        trades,
    )
    v04.write_csv(
        OUTPUT_DIR / "strategy5_v05_positions.csv",
        [asdict(row) for row in positions],
    )
    (OUTPUT_DIR / "strategy5_v05_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy5_v05_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
