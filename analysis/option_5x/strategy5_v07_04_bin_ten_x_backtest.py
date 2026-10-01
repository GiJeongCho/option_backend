from __future__ import annotations

import json
from dataclasses import asdict

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy
import strategy5_v04_basket_backtest as v04
import strategy5_v06_ten_x_backtest as v06


STRATEGY_VERSION = "S5-v0.7-04BIN-10X"
OUTPUT_DIR = v04.OUTPUT_DIR
OUTPUT_STEM = "strategy5_v07_04_bin_ten_x"
INITIAL_CASH = 40_000_000
ENTRY_BIN = "0.4-0.5"
ENTRY_RAW_MIN = 0.40
ENTRY_RAW_MAX = 0.49


def entry_bin_setups(
    contracts: dict[str, base.ContractGrid],
) -> list[v04.Setup]:
    setups: list[v04.Setup] = []
    for contract in contracts.values():
        setup = v04.observation(contract)
        if setup is None:
            continue
        fill = v04.setup_fill(contract)
        if fill is None or fill[4] != ENTRY_BIN:
            continue
        setups.append(setup)
    return setups


def audit_variant(
    daily: list[v04.DayResult],
    trades: list[dict[str, object]],
    positions: list[v04.PositionRecord],
) -> dict[str, object]:
    result = v06.audit(daily, trades, positions)
    errors = list(result["error_samples"])
    trades_by_date: dict[str, list[dict[str, object]]] = {}
    for trade in trades:
        trades_by_date.setdefault(str(trade["date"]), []).append(trade)

    for row in daily:
        day_trades = trades_by_date.get(row.date, [])
        buys = [trade for trade in day_trades if trade["side"] == "BUY"]
        required_cash = sum(
            int(trade["principal"]) + int(trade["fee"])
            for trade in buys
        )
        if required_cash > row.start_cash:
            errors.append(f"{row.date}: insufficient cash")
        if row.end_cash < 0:
            errors.append(f"{row.date}: negative cash")
        if row.active_premium_bins not in {0, 1}:
            errors.append(f"{row.date}: multiple premium bins")
        for trade in buys:
            raw_price = float(trade["price"]) - base.SLIPPAGE
            if not ENTRY_RAW_MIN <= raw_price <= ENTRY_RAW_MAX:
                errors.append(
                    f"{row.date}: entry outside 0.40-0.49"
                )

    return {
        **result,
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "checks": [
            *list(result["checks"]),
            "every buy raw open is within the 0.40-0.49 premium bin",
            "40,000,000 won initial cash and no borrowing",
        ],
    }


def run() -> dict[str, object]:
    original_cash = base.INITIAL_CASH
    original_version = v06.STRATEGY_VERSION
    base.INITIAL_CASH = INITIAL_CASH
    v06.STRATEGY_VERSION = STRATEGY_VERSION
    try:
        profile = json.loads(
            base.PROFILE_PATH.read_text(encoding="utf-8")
        )
        groups = [
            group
            for group in base.derive_expiry_groups(profile)
            if legacy.PPT_START
            <= group["expiry_date"]
            <= legacy.PPT_END
        ]
        files = base.target_file_map()
        cash = INITIAL_CASH
        daily: list[v04.DayResult] = []
        trades: list[dict[str, object]] = []
        positions: list[v04.PositionRecord] = []
        target_signal_positions: set[tuple[str, str, int]] = set()

        for group in groups:
            date_value = group["expiry_date"]
            contracts = base.load_expiry_day(
                files[date_value],
                group,
            )
            setups = entry_bin_setups(contracts)
            row, day_trades, day_positions = v06.simulate_day(
                date_value,
                contracts,
                setups,
                cash,
            )
            row.active_premium_bins = 1 if row.initial_positions else 0
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
                        position.observation_price
                        * v06.TARGET_MULTIPLE,
                    )
                    is not None
                ):
                    target_signal_positions.add(
                        (
                            position.date,
                            position.code,
                            position.campaign,
                        )
                    )

        development = [
            row for row in daily if row.date <= legacy.DEVELOPMENT_END
        ]
        validation = [
            row for row in daily if row.date > legacy.DEVELOPMENT_END
        ]
        audit_result = audit_variant(daily, trades, positions)
        target_sale_positions = {
            (
                str(trade["date"]),
                str(trade["code"]),
                int(trade["campaign"]),
            )
            for trade in trades
            if trade["reason"] == "TARGET_10X_ALL"
        }
        time_sale_positions = {
            (
                str(trade["date"]),
                str(trade["code"]),
                int(trade["campaign"]),
            )
            for trade in trades
            if trade["reason"] == "TIME"
        }
        confirmed_positions = [
            position
            for position in positions
            if position.branch == "EARLY_CONFIRMED"
        ]
        summary = {
            **v04.summarize(daily),
            "initial_cash": INITIAL_CASH,
            "ending_cash": cash,
        }
        result = {
            "status": "experimental_04_bin_10x_logic_check",
            "strategy_version": STRATEGY_VERSION,
            "scope": {
                "start": groups[0]["expiry_date"],
                "end": groups[-1]["expiry_date"],
                "expiry_days": len(groups),
                "development_end": legacy.DEVELOPMENT_END,
                "validation_is_independent": False,
            },
            "rules": {
                "initial_cash": INITIAL_CASH,
                "daily_principal_limit": v04.INITIAL_BUDGET,
                "entry": (
                    "after 09:10, invest only in contracts whose next "
                    "actual raw open is 0.40-0.49"
                ),
                "allocation": (
                    "use the full daily budget only within the 0.40-0.49 "
                    "bin; balance integer contracts inside that bin"
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
                    "sell every remaining contract at the 09:30 open; "
                    "no 09:30 trade is valued at zero"
                ),
                "reentry": "disabled",
                "slippage": base.SLIPPAGE,
                "commission_per_contract": (
                    base.COMMISSION_PER_CONTRACT
                ),
                "volume_participation": base.ENTRY_PARTICIPATION,
            },
            "confirmed_outcomes": {
                "positions": len(confirmed_positions),
                "reached_10x_signal_positions": len(
                    target_signal_positions
                ),
                "target_sale_filled_positions": len(
                    target_sale_positions
                ),
                "time_exit_signal_positions": (
                    len(confirmed_positions)
                    - len(target_signal_positions)
                ),
                "time_sale_filled_positions": len(
                    time_sale_positions
                ),
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
                for row in sorted(
                    daily,
                    key=lambda item: item.pnl,
                )[:15]
            ],
            "audit": audit_result,
            "comparison": {
                "baseline": "S5-v0.6-10X",
                "changed": (
                    "initial cash 80,000,000 -> 40,000,000 and entry "
                    "universe 0.40-1.20 multi-bin -> 0.40-0.49 only"
                ),
            },
            "limitations": [
                (
                    "This is a requested logic-check variant, not an "
                    "independently validated strategy."
                ),
                (
                    "The current P0 and same-bar branch rules do not "
                    "reproduce the PPT 1,320/633/37/650 counts."
                ),
                (
                    "The development/validation split is diagnostic only."
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
    finally:
        base.INITIAL_CASH = original_cash
        v06.STRATEGY_VERSION = original_version


if __name__ == "__main__":
    run()
