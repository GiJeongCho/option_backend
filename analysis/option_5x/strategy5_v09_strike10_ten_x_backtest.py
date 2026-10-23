from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy
import strategy5_v04_basket_backtest as v04
import strategy5_v06_ten_x_backtest as v06
import strategy5_v07_04_bin_ten_x_backtest as v07


STRATEGY_VERSION = "S5-v0.9-STRIKE10-10X"
OUTPUT_STEM = "strategy5_v09_strike10_ten_x"
INITIAL_CASH = 40_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
ENTRY_RAW_MIN = 0.40
ENTRY_RAW_MAX = 0.49
STRIKE_STEP = 10.0
MINIMUM_RUNNER_SAMPLE = 30


def contract_strike(contract: base.ContractGrid) -> float | None:
    """Read the strike printed at the end of the option name."""
    try:
        return float(contract.name.rsplit(maxsplit=1)[-1].replace(",", ""))
    except (IndexError, ValueError):
        return None


def is_ten_point_strike(strike: float | None) -> bool:
    if strike is None:
        return False
    return math.isclose(
        strike / STRIKE_STEP,
        round(strike / STRIKE_STEP),
        abs_tol=1e-9,
    )


def strike10_setups(
    contracts: dict[str, base.ContractGrid],
) -> list[v04.Setup]:
    return [
        setup
        for setup in v07.entry_bin_setups(contracts)
        if is_ten_point_strike(contract_strike(contracts[setup.code]))
    ]


def cheapest_first_entries(
    contracts: dict[str, base.ContractGrid],
    setups: list[v04.Setup],
) -> list[v04.PlannedEntry]:
    """Allocate principal to 0.40, then 0.41, ... under volume limits."""
    candidates: list[
        tuple[v04.Setup, int, int, float, int, str, float]
    ] = []
    for setup in setups:
        fill = v04.setup_fill(contracts[setup.code])
        if fill is None:
            continue
        index, minute, price, volume, entry_bin = fill
        raw_price = price - base.SLIPPAGE
        if not ENTRY_RAW_MIN <= raw_price <= ENTRY_RAW_MAX:
            continue
        candidates.append(
            (
                setup,
                index,
                minute,
                price,
                volume,
                entry_bin,
                raw_price,
            )
        )

    candidates.sort(
        key=lambda item: (
            round(item[6], 8),
            item[1],
            -item[0].cumulative_volume,
            item[0].code,
        )
    )
    remaining = DAILY_PRINCIPAL_LIMIT
    entries: list[v04.PlannedEntry] = []
    for setup, index, minute, price, volume, entry_bin, _ in candidates:
        unit_principal = round(price * base.MULTIPLIER)
        maximum_quantity = math.floor(
            volume * base.ENTRY_PARTICIPATION
        )
        quantity = min(
            maximum_quantity,
            remaining // unit_principal,
        )
        if quantity <= 0:
            continue
        entries.append(
            v04.PlannedEntry(
                setup=setup,
                fill_index=index,
                fill_minute=minute,
                fill_price=price,
                fill_volume=volume,
                premium_bin=entry_bin,
                quantity=quantity,
            )
        )
        remaining -= quantity * unit_principal
    return entries


def peak_bucket(multiple: float) -> str:
    if multiple >= 10.0:
        return "10_PLUS"
    if multiple >= 5.0:
        return "5_TO_10"
    if multiple >= 2.0:
        return "2_TO_5"
    return "1_5_TO_2"


def confirmed_runner_diagnostics(
    positions: list[v04.PositionRecord],
    contracts_by_date: dict[str, dict[str, base.ContractGrid]],
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for position in positions:
        if (
            position.branch != "EARLY_CONFIRMED"
            or position.confirmation_minute is None
        ):
            continue
        contract = contracts_by_date[position.date][position.code]
        confirmation_index = base.MINUTE_INDEX[
            position.confirmation_minute
        ]
        highs = [
            float(bar.high)
            for bar in contract.bars[
                confirmation_index : base.MINUTE_INDEX[v04.FORCE_EXIT_TIME] + 1
            ]
            if bar.traded and bar.high is not None
        ]
        peak_multiple = (
            max(highs, default=position.observation_price)
            / position.observation_price
        )
        cases.append(
            {
                "date": position.date,
                "code": position.code,
                "call_put": position.call_put,
                "strike": contract_strike(contract),
                "entry_raw_price": round(
                    position.entry_price - base.SLIPPAGE,
                    4,
                ),
                "confirmation_minute": position.confirmation_minute,
                "peak_multiple": round(peak_multiple, 4),
                "peak_bucket": peak_bucket(peak_multiple),
            }
        )

    buckets = {
        name: sum(case["peak_bucket"] == name for case in cases)
        for name in ("1_5_TO_2", "2_TO_5", "5_TO_10", "10_PLUS")
    }
    enough_sample = len(cases) >= MINIMUM_RUNNER_SAMPLE
    return {
        "confirmed_positions": len(cases),
        "minimum_sample_for_runner_classifier": MINIMUM_RUNNER_SAMPLE,
        "runner_classifier_applied": False,
        "classifier_status": (
            "not_applied_requires_locked_forward_validation"
            if enough_sample
            else "not_applied_insufficient_sample"
        ),
        "peak_buckets": buckets,
        "cases": cases,
        "decision": (
            "The requested 1.5x confirmation and 10x full-position exit "
            "remain unchanged. No 2x/5x/10x classifier is fitted to this "
            "sample because doing so would be post-selection overfitting."
        ),
    }


def audit_variant(
    daily: list[v04.DayResult],
    trades: list[dict[str, object]],
    positions: list[v04.PositionRecord],
    contracts_by_date: dict[str, dict[str, base.ContractGrid]],
) -> dict[str, object]:
    result = v07.audit_variant(daily, trades, positions)
    errors = list(result["error_samples"])
    trades_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[str(trade["date"])].append(trade)

    for row in daily:
        buys = sorted(
            (
                trade
                for trade in trades_by_date[row.date]
                if trade["side"] == "BUY"
            ),
            key=lambda trade: int(trade["campaign"]),
        )
        raw_prices: list[float] = []
        for trade in buys:
            contract = contracts_by_date[row.date][str(trade["code"])]
            strike = contract_strike(contract)
            if not is_ten_point_strike(strike):
                errors.append(
                    f"{row.date} {trade['code']}: non-10-point strike "
                    f"{strike}"
                )
            raw_prices.append(
                round(float(trade["price"]) - base.SLIPPAGE, 8)
            )
        if raw_prices != sorted(raw_prices):
            errors.append(f"{row.date}: entry price priority")
        if row.initial_buy_principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{row.date}: daily principal limit")

    return {
        **result,
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "checks": [
            *list(result["checks"]),
            "every bought contract has a strike divisible by 10 points",
            "campaign allocation follows raw premium 0.40 through 0.49",
            "daily new-buy principal does not exceed 5,000,000 won",
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
        groups = legacy.backtest_groups(profile)
        files = base.target_file_map()
        cash = INITIAL_CASH
        daily: list[v04.DayResult] = []
        trades: list[dict[str, object]] = []
        positions: list[v04.PositionRecord] = []
        contracts_by_date: dict[
            str, dict[str, base.ContractGrid]
        ] = {}

        for group in groups:
            date_value = group["expiry_date"]
            contracts = base.load_expiry_day(
                files[date_value],
                group,
            )
            contracts_by_date[date_value] = contracts
            setups = strike10_setups(contracts)
            row, day_trades, day_positions = v06.simulate_day(
                date_value,
                contracts,
                setups,
                cash,
                entry_allocator=cheapest_first_entries,
            )
            row.active_premium_bins = 1 if row.initial_positions else 0
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
        audit_result = audit_variant(
            daily,
            trades,
            positions,
            contracts_by_date,
        )
        runner_diagnostics = confirmed_runner_diagnostics(
            positions,
            contracts_by_date,
        )
        summary = {
            **v04.summarize(daily),
            "initial_cash": INITIAL_CASH,
            "ending_cash": cash,
        }
        baseline_path = (
            v04.OUTPUT_DIR / "strategy5_v07_04_bin_ten_x_summary.json"
        )
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_path.exists()
            else None
        )
        baseline_summary = (
            dict(baseline["summary"]) if baseline is not None else {}
        )
        result = {
            "status": "experimental_strike10_cheapest_first_logic_check",
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
                "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
                "entry": (
                    "after 09:10, accept only contracts with a raw next "
                    "open of 0.40-0.49 and a strike divisible by 10 points"
                ),
                "allocation": (
                    "allocate to the cheapest raw premium first: 0.40, "
                    "then 0.41, through 0.49; within equal prices prefer "
                    "earlier fills and higher observed volume"
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
                    "sell every remaining contract at the 09:30 open"
                ),
                "reentry": "disabled",
                "slippage": base.SLIPPAGE,
                "commission_per_contract": base.COMMISSION_PER_CONTRACT,
                "volume_participation": base.ENTRY_PARTICIPATION,
            },
            "summary": summary,
            "development": v04.summarize(development),
            "validation": v04.summarize(validation),
            "position_expectation": v04.summarize_positions(positions),
            "runner_diagnostics": runner_diagnostics,
            "worst_days": [
                asdict(row)
                for row in sorted(daily, key=lambda item: item.pnl)[:15]
            ],
            "audit": audit_result,
            "comparison": {
                "baseline": "S5-v0.7-04BIN-10X",
                "baseline_total_pnl": baseline_summary.get("total_pnl"),
                "total_pnl_difference": (
                    summary["total_pnl"]
                    - int(baseline_summary["total_pnl"])
                    if baseline_summary.get("total_pnl") is not None
                    else None
                ),
                "baseline_average_initial_buy_principal": (
                    baseline_summary.get(
                        "average_initial_buy_principal"
                    )
                ),
                "average_initial_buy_principal_difference": (
                    summary["average_initial_buy_principal"]
                    - int(
                        baseline_summary[
                            "average_initial_buy_principal"
                        ]
                    )
                    if baseline_summary.get(
                        "average_initial_buy_principal"
                    )
                    is not None
                    else None
                ),
            },
            "limitations": [
                (
                    "A 10-point strike filter reduces the eligible universe; "
                    "it does not increase deployed principal under the same "
                    "5,000,000 won daily cap."
                ),
                (
                    "The 09:10 model compares next executable opens within "
                    "the existing five-minute fill window; order-book queue "
                    "depth is not available."
                ),
                (
                    "A 2x/5x/10x predictor is not fitted when the confirmed "
                    "sample is too small, and the 10x exit remains unchanged."
                ),
                (
                    "The development/validation split is diagnostic only."
                ),
            ],
        }

        v04.write_csv(
            v04.OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv",
            [asdict(row) for row in daily],
        )
        v04.write_csv(
            v04.OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv",
            trades,
        )
        v04.write_csv(
            v04.OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv",
            [asdict(row) for row in positions],
        )
        (
            v04.OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
        ).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (
            v04.OUTPUT_DIR / f"{OUTPUT_STEM}_audit.json"
        ).write_text(
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
