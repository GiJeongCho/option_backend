from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy
import strategy5_v04_basket_backtest as v04


STRATEGY_VERSION = "S5-v0.6-10X"
OUTPUT_DIR = v04.OUTPUT_DIR
OUTPUT_STEM = "strategy5_v06_ten_x"
CONFIRMATION_END = 929
FORCED_EXIT_MINUTE = 930
TARGET_MULTIPLE = 10.0


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


def confirmed_ten_x_exit(
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    setup: v04.Setup,
    entry_index: int,
    quantity: int,
    used_capacity: dict[int, int],
) -> tuple[int, int, list[dict[str, object]]]:
    target_index = v04.first_high_touch(
        contract,
        entry_index,
        v04.FORCE_EXIT_TIME,
        setup.observation_price * TARGET_MULTIPLE,
    )
    signal_index = (
        target_index
        if target_index is not None
        else base.MINUTE_INDEX[v04.FORCE_EXIT_TIME]
    )
    reason = "TARGET_10X_ALL" if target_index is not None else "TIME"
    stage = 1 if target_index is not None else 8
    filled, last_index, trades = v04.execute_sale(
        STRATEGY_VERSION,
        date_value,
        campaign,
        contract,
        signal_index,
        quantity,
        reason,
        stage,
        used_capacity,
    )
    remaining = quantity - filled
    if remaining > 0:
        trades.append(
            v04.expire_trade(
                STRATEGY_VERSION,
                date_value,
                campaign,
                contract,
                remaining,
            )
        )
        last_index = len(contract.bars) - 1
    return remaining, last_index, trades


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
        remaining, exit_index, sells = confirmed_ten_x_exit(
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
    baseline = v04.audit(daily, trades)
    errors = [
        f"baseline: {error}"
        for error in baseline["error_samples"]
    ]
    trades_by_position: dict[
        tuple[str, str, int], list[dict[str, object]]
    ] = defaultdict(list)
    sold_by_bar: dict[
        tuple[str, str, int], list[dict[str, object]]
    ] = defaultdict(list)
    for trade in trades:
        trades_by_position[
            (
                str(trade["date"]),
                str(trade["code"]),
                int(trade["campaign"]),
            )
        ].append(trade)
        if (
            trade["side"] == "SELL"
            and trade["reason"] != "NO_1_5X_09_30_EXIT"
        ):
            sold_by_bar[
                (
                    str(trade["date"]),
                    str(trade["code"]),
                    int(trade["minute"]),
                )
            ].append(trade)

    if any(row.strategy != STRATEGY_VERSION for row in daily):
        errors.append("daily strategy version")
    if any(position.strategy != STRATEGY_VERSION for position in positions):
        errors.append("position strategy version")
    if any(str(row["strategy"]) != STRATEGY_VERSION for row in trades):
        errors.append("trade strategy version")
    if any(
        row.late_signal_found
        or row.late_reentry_filled
        or row.late_buy_principal
        for row in daily
    ):
        errors.append("reentry state")

    for key, sale_rows in sold_by_bar.items():
        maximum = math.floor(
            int(sale_rows[0]["bar_volume"])
            * base.EXIT_PARTICIPATION
        )
        if sum(int(row["quantity"]) for row in sale_rows) > maximum:
            errors.append(f"exit volume {key}")

    for position in positions:
        key = (position.date, position.code, position.campaign)
        campaign_rows = trades_by_position[key]
        allowed = {
            "09_10_PREMIUM_BIN_BASKET",
            "TARGET_10X_ALL",
            "TIME",
            "UNFILLED_ZERO_VALUE",
        }
        if position.branch == "EARLY_FAILED":
            allowed = {
                "09_10_PREMIUM_BIN_BASKET",
                "NO_1_5X_09_30_EXIT",
                "NO_09_30_LIQUIDITY_ZERO",
            }
            closing_rows = [
                row
                for row in campaign_rows
                if row["side"] in {"SELL", "EXPIRE"}
            ]
            if (
                position.confirmation_minute is not None
                or len(closing_rows) != 1
                or int(closing_rows[0]["minute"])
                != FORCED_EXIT_MINUTE
                or int(closing_rows[0]["quantity"]) != position.quantity
            ):
                errors.append(f"{position.date}: failed exit {key}")
        elif (
            position.branch != "EARLY_CONFIRMED"
            or position.confirmation_minute is None
            or position.confirmation_minute > CONFIRMATION_END
        ):
            errors.append(f"{position.date}: confirmed state {key}")

        if any(str(row["reason"]) not in allowed for row in campaign_rows):
            errors.append(f"{position.date}: exit reason {key}")
        if (
            sum(int(row["cash_flow"]) for row in campaign_rows)
            != position.pnl
        ):
            errors.append(f"{position.date}: position pnl {key}")

        contract_rows = [
            row
            for row in campaign_rows
            if row["reason"] == "TARGET_10X_ALL"
        ]
        for sale in contract_rows:
            sale_index = base.MINUTE_INDEX[int(sale["minute"])]
            entry_index = base.MINUTE_INDEX[position.entry_minute]
            # TARGET_10X_ALL is always filled after a completed signal bar.
            if sale_index <= entry_index:
                errors.append(f"{position.date}: target order {key}")

    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "position_rows": len(positions),
        "checks": [
            "cash continuity, 5,000,000 won entry cap, and closed quantity",
            "post-entry 1.5x confirmation only through 09:29",
            "full 09:30 liquidation for every unconfirmed position",
            "10x all-sale or 15:15 time exit for confirmed positions",
            "no reentry and normal-order 10% volume limits",
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
    target_signal_positions: set[tuple[str, str, int]] = set()

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
        for position in day_positions:
            if position.branch != "EARLY_CONFIRMED":
                continue
            contract = contracts[position.code]
            if (
                v04.first_high_touch(
                    contract,
                    base.MINUTE_INDEX[position.entry_minute],
                    v04.FORCE_EXIT_TIME,
                    position.observation_price * TARGET_MULTIPLE,
                )
                is not None
            ):
                target_signal_positions.add(
                    (position.date, position.code, position.campaign)
                )

    development = [
        row for row in daily if row.date <= legacy.DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > legacy.DEVELOPMENT_END
    ]
    audit_result = audit(daily, trades, positions)
    target_positions = {
        (str(row["date"]), str(row["code"]), int(row["campaign"]))
        for row in trades
        if row["reason"] == "TARGET_10X_ALL"
    }
    time_exit_positions = {
        (str(row["date"]), str(row["code"]), int(row["campaign"]))
        for row in trades
        if row["reason"] == "TIME"
    }
    confirmed_positions = [
        position
        for position in positions
        if position.branch == "EARLY_CONFIRMED"
    ]
    summary = {
        **v04.summarize(daily),
        "initial_cash": base.INITIAL_CASH,
        "ending_cash": cash,
    }
    result = {
        "status": "experimental_logic_check_10x_all",
        "strategy_version": STRATEGY_VERSION,
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": legacy.DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "ppt_claim": {
            "total": 1_320,
            "confirmed_30m": 633,
            "same_bar_ambiguous": 37,
            "strict_unconfirmed_30m": 650,
            "confirmed_reached_2x": 480,
            "confirmed_reached_2x_pct": 75.83,
            "strict_unconfirmed_reached_003": 615,
            "strict_unconfirmed_reached_003_pct": 94.62,
            "combined_unconfirmed_including_ambiguous": 687,
            "combined_reached_003": 646,
            "combined_reached_003_pct": 94.03,
            "combined_not_reached_003": 41,
            "non_003_reached_10x": 33,
            "non_003_reached_10x_pct": 80.49,
            "interpretation": (
                "41 is the remainder of 687 after including the 37 "
                "same-bar ambiguous cases, not 4% of the strict 650. "
                "Of those 41, 33 reached 10x. A 0.03 touch is not the "
                "same as a zero-valued close."
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
            "confirmed_exit": (
                "hold the full position until 10x P0; otherwise exit "
                "after the 15:15 signal"
            ),
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
        "confirmed_outcomes": {
            "positions": len(confirmed_positions),
            "reached_10x_signal_positions": len(
                target_signal_positions
            ),
            "target_sale_filled_positions": len(target_positions),
            "time_exit_signal_positions": (
                len(confirmed_positions) - len(target_signal_positions)
            ),
            "time_sale_filled_positions": len(time_exit_positions),
            "positions_with_expired_quantity": sum(
                position.expired_quantity > 0
                for position in confirmed_positions
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
                "This is a requested logic-check variant, not a "
                "recommended or independently validated strategy."
            ),
            (
                "The PPT 1,320/633/37/650 counts are not reproduced "
                "exactly by the current explicit raw-data definition."
            ),
            (
                "The development/validation split is diagnostic only; "
                "the full PPT period was already observed."
            ),
            (
                "The 09:30 forced exit bypasses the 10% volume cap; no "
                "09:30 trade is conservatively valued at zero."
            ),
        ],
    }

    v04.write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv",
        [asdict(row) for row in daily],
    )
    v04.write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv",
        trades,
    )
    v04.write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv",
        [asdict(row) for row in positions],
    )
    (OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{OUTPUT_STEM}_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
