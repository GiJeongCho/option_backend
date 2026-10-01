from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy
import strategy5_v03_event_research as events


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S5-v0.3"
OBSERVATION_MODE: events.ObservationMode = "first_qualifying_open"
LATE_ENTRY_CUTOFF = 1430
TARGET_MULTIPLE = 2.0
FORCE_EXIT_TIME = 1515
BRANCHES = ("EARLY", "LATE")


def portfolio_version(branch: str) -> str:
    return f"{STRATEGY_VERSION}-{branch}"


@dataclass(frozen=True)
class Setup:
    code: str
    call_put: str
    observation_index: int
    observation_minute: int
    observation_price: float
    confirmation_index: int
    confirmation_minute: int
    confirmation_volume: int
    cumulative_volume: int
    branch: str


@dataclass
class DayResult:
    strategy: str
    date: str
    start_cash: int
    end_cash: int
    pnl: int
    signal_found: bool
    entry_filled: bool
    code: str
    call_put: str
    confirmation_branch: str
    observation_minute: int | None
    observation_price: float
    confirmation_minute: int | None
    entry_minute: int | None
    entry_price: float
    entry_quantity: int
    buy_principal: int
    buy_fee: int
    reached_2x: bool
    stop_triggered: bool
    gross_sales: int
    sell_fee: int
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float


def setup_for_contract(
    contract: base.ContractGrid,
) -> Setup | None:
    observed = events.observation(contract, OBSERVATION_MODE)
    if observed is None:
        return None
    first_bar = contract.bars[observed.index]
    if (
        first_bar.high is not None
        and float(first_bar.high)
        >= observed.price * events.CONFIRMATION_MULTIPLE
    ):
        return None
    confirmation_index = events.first_touch(
        contract,
        observed.index + 1,
        events.EARLY_END,
        observed.price,
    )
    branch = "EARLY"
    if confirmation_index is None:
        confirmation_index = events.first_touch(
            contract,
            base.MINUTE_INDEX[events.EARLY_END] + 1,
            LATE_ENTRY_CUTOFF,
            observed.price,
        )
        branch = "LATE"
    if confirmation_index is None:
        return None
    confirmation_bar = contract.bars[confirmation_index]
    return Setup(
        code=contract.code,
        call_put=contract.call_put,
        observation_index=observed.index,
        observation_minute=observed.minute,
        observation_price=observed.price,
        confirmation_index=confirmation_index,
        confirmation_minute=confirmation_bar.minute,
        confirmation_volume=confirmation_bar.volume,
        cumulative_volume=sum(
            bar.volume
            for bar in contract.bars[
                observed.index : confirmation_index + 1
            ]
        ),
        branch=branch,
    )


def select_setup(
    contracts: dict[str, base.ContractGrid],
    branch: str,
) -> Setup | None:
    candidates = [
        setup
        for contract in contracts.values()
        if (setup := setup_for_contract(contract)) is not None
        and setup.branch == branch
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item.confirmation_index,
            -item.confirmation_volume,
            -item.cumulative_volume,
            item.code,
        ),
    )[0]


def buy_after_confirmation(
    date_value: str,
    contract: base.ContractGrid,
    setup: Setup,
    available_cash: int,
    strategy_version: str,
) -> tuple[int, int, int, int, dict[str, object]] | None:
    fill = base.next_traded_bar(
        contract,
        setup.confirmation_index,
        base.ENTRY_WAIT_MINUTES,
    )
    if fill is None:
        return None
    fill_index, bar = fill
    assert bar.open is not None
    fill_price = float(bar.open) + base.SLIPPAGE
    by_principal = math.floor(
        base.DAILY_PRINCIPAL_LIMIT
        / (fill_price * base.MULTIPLIER)
    )
    by_cash = math.floor(
        available_cash
        / (
            fill_price * base.MULTIPLIER
            + 2 * base.COMMISSION_PER_CONTRACT
        )
    )
    by_volume = math.floor(bar.volume * base.ENTRY_PARTICIPATION)
    quantity = min(by_principal, by_cash, by_volume)
    if quantity <= 0:
        return None
    principal = round(quantity * fill_price * base.MULTIPLIER)
    fee = quantity * base.COMMISSION_PER_CONTRACT
    return (
        fill_index,
        quantity,
        principal,
        fee,
        {
            "strategy": strategy_version,
            "date": date_value,
            "campaign": 1,
            "minute": bar.minute,
            "code": contract.code,
            "call_put": contract.call_put,
            "side": "BUY",
            "reason": f"{setup.branch}_1_5X_CONFIRM_ENTRY",
            "stage": 1,
            "quantity": quantity,
            "price": fill_price,
            "principal": principal,
            "fee": fee,
            "cash_flow": -(principal + fee),
            "bar_volume": bar.volume,
        },
    )


def sell_after_signal(
    date_value: str,
    contract: base.ContractGrid,
    signal_index: int,
    requested_quantity: int,
    reason: str,
    strategy_version: str,
) -> tuple[int, int, int, int, list[dict[str, object]]]:
    remaining = requested_quantity
    gross_sales = 0
    fees = 0
    capacity = 0.0
    last_fill_index = signal_index
    trades: list[dict[str, object]] = []
    for index in range(signal_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if not bar.traded:
            continue
        assert bar.open is not None
        capacity += bar.volume * base.EXIT_PARTICIPATION
        fillable = min(remaining, math.floor(capacity + 1e-12))
        if fillable <= 0:
            continue
        capacity -= fillable
        fill_price = max(0.0, float(bar.open) - base.SLIPPAGE)
        principal = round(
            fillable * fill_price * base.MULTIPLIER
        )
        fee = fillable * base.COMMISSION_PER_CONTRACT
        remaining -= fillable
        gross_sales += principal
        fees += fee
        last_fill_index = index
        trades.append(
            {
                "strategy": strategy_version,
                "date": date_value,
                "campaign": 1,
                "minute": bar.minute,
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "SELL",
                "reason": reason,
                "stage": 1,
                "quantity": fillable,
                "price": fill_price,
                "principal": principal,
                "fee": fee,
                "cash_flow": principal - fee,
                "bar_volume": bar.volume,
            }
        )
        if remaining <= 0:
            break
    return (
        requested_quantity - remaining,
        gross_sales,
        fees,
        last_fill_index,
        trades,
    )


def empty_day(
    date_value: str,
    start_cash: int,
    branch: str,
    setup: Setup | None = None,
) -> DayResult:
    return DayResult(
        strategy=portfolio_version(branch),
        date=date_value,
        start_cash=start_cash,
        end_cash=start_cash,
        pnl=0,
        signal_found=setup is not None,
        entry_filled=False,
        code=setup.code if setup else "",
        call_put=setup.call_put if setup else "",
        confirmation_branch=setup.branch if setup else "",
        observation_minute=(
            setup.observation_minute if setup else None
        ),
        observation_price=(
            round(setup.observation_price, 4) if setup else 0.0
        ),
        confirmation_minute=(
            setup.confirmation_minute if setup else None
        ),
        entry_minute=None,
        entry_price=0.0,
        entry_quantity=0,
        buy_principal=0,
        buy_fee=0,
        reached_2x=False,
        stop_triggered=False,
        gross_sales=0,
        sell_fee=0,
        exit_reason="",
        exit_minute=None,
        expired_quantity=0,
        mfe_multiple=0.0,
        mae_pct=0.0,
    )


def excursion(
    contract: base.ContractGrid,
    entry_index: int,
    entry_price: float,
) -> tuple[float, float]:
    highs = [
        float(bar.high)
        for bar in contract.bars[entry_index:]
        if bar.traded and bar.high is not None
    ]
    lows = [
        float(bar.low)
        for bar in contract.bars[entry_index:]
        if bar.traded and bar.low is not None
    ]
    maximum = max(highs, default=entry_price) / entry_price
    minimum_pct = (
        min(lows, default=entry_price) / entry_price - 1.0
    ) * 100
    return maximum, minimum_pct


def simulate_day(
    date_value: str,
    contracts: dict[str, base.ContractGrid],
    setup: Setup | None,
    start_cash: int,
    branch: str,
) -> tuple[DayResult, list[dict[str, object]]]:
    strategy_version = portfolio_version(branch)
    if setup is None:
        return empty_day(date_value, start_cash, branch), []
    contract = contracts[setup.code]
    fill = buy_after_confirmation(
        date_value,
        contract,
        setup,
        start_cash,
        strategy_version,
    )
    if fill is None:
        return empty_day(date_value, start_cash, branch, setup), []
    (
        entry_index,
        quantity,
        buy_principal,
        buy_fee,
        buy_trade,
    ) = fill
    entry_price = float(buy_trade["price"])
    trades = [buy_trade]
    exit_reason = ""
    exit_index = entry_index
    reached_2x = False
    stop_triggered = False
    gross_sales = 0
    sell_fee = 0
    remaining = quantity

    for index in range(entry_index, len(contract.bars)):
        bar = contract.bars[index]
        reason = ""
        if bar.traded and bar.close is not None:
            close = float(bar.close)
            if close >= setup.observation_price * TARGET_MULTIPLE:
                reached_2x = True
                reason = "P0_TARGET_2X"
            elif close <= setup.observation_price:
                stop_triggered = True
                reason = "P0_FAILURE_STOP"
        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
        if not reason:
            continue
        (
            filled,
            gross_sales,
            sell_fee,
            exit_index,
            sell_trades,
        ) = sell_after_signal(
            date_value,
            contract,
            index,
            remaining,
            reason,
            strategy_version,
        )
        remaining -= filled
        trades.extend(sell_trades)
        exit_reason = (
            reason if remaining == 0 else "UNFILLED_ZERO_VALUE"
        )
        break

    if remaining > 0:
        trades.append(
            {
                "strategy": strategy_version,
                "date": date_value,
                "campaign": 1,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "stage": 1,
                "quantity": remaining,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
        exit_reason = "UNFILLED_ZERO_VALUE"
        exit_index = len(contract.bars) - 1

    pnl = gross_sales - sell_fee - buy_principal - buy_fee
    mfe_multiple, mae_pct = excursion(
        contract,
        entry_index,
        entry_price,
    )
    return (
        DayResult(
            strategy=strategy_version,
            date=date_value,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=contract.code,
            call_put=contract.call_put,
            confirmation_branch=setup.branch,
            observation_minute=setup.observation_minute,
            observation_price=round(setup.observation_price, 4),
            confirmation_minute=setup.confirmation_minute,
            entry_minute=contract.bars[entry_index].minute,
            entry_price=round(entry_price, 4),
            entry_quantity=quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            reached_2x=reached_2x,
            stop_triggered=stop_triggered,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_reason=exit_reason,
            exit_minute=contract.bars[exit_index].minute,
            expired_quantity=remaining,
            mfe_multiple=round(mfe_multiple, 6),
            mae_pct=round(mae_pct, 4),
        ),
        trades,
    )


def max_drawdown(rows: list[DayResult]) -> int:
    equity = base.INITIAL_CASH
    peak = equity
    result = 0
    for row in rows:
        equity += row.pnl
        peak = max(peak, equity)
        result = min(result, equity - peak)
    return result


def summarize(rows: list[DayResult]) -> dict[str, object]:
    traded = [row for row in rows if row.entry_filled]
    profits = [row.pnl for row in traded]
    ordered = sorted(profits, reverse=True)
    gains = sum(value for value in profits if value > 0)
    losses = -sum(value for value in profits if value < 0)
    return {
        "days": len(rows),
        "signal_days": sum(row.signal_found for row in rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in profits),
        "losing_days": sum(value < 0 for value in profits),
        "break_even_days": sum(value == 0 for value in profits),
        "total_pnl": sum(profits),
        "average_trade_day_pnl": (
            round(sum(profits) / len(profits)) if profits else 0
        ),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(profits, default=0),
        "maximum_day_pnl": max(profits, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": (
            sum(profits) - sum(ordered[:3])
        ),
        "pnl_excluding_best_5_days": (
            sum(profits) - sum(ordered[:5])
        ),
        "total_buy_principal": sum(
            row.buy_principal for row in traded
        ),
        "average_buy_principal": (
            round(
                sum(row.buy_principal for row in traded) / len(traded)
            )
            if traded
            else 0
        ),
        "expired_contracts": sum(
            row.expired_quantity for row in traded
        ),
        "early_trade_days": sum(
            row.confirmation_branch == "EARLY" for row in traded
        ),
        "late_trade_days": sum(
            row.confirmation_branch == "LATE" for row in traded
        ),
        "target_exit_days": sum(row.reached_2x for row in traded),
        "stop_exit_days": sum(row.stop_triggered for row in traded),
    }


def grouped(
    rows: list[DayResult],
    field: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field))].append(row.pnl)
    return [
        {
            "group": name,
            "count": len(values),
            "profitable_count": sum(value > 0 for value in values),
            "losing_count": sum(value < 0 for value in values),
            "total_pnl": sum(values),
            "average_pnl": round(sum(values) / len(values)),
        }
        for name, values in sorted(groups.items())
    ]


def execution_metrics(rows: list[DayResult]) -> dict[str, object]:
    traded = [row for row in rows if row.entry_filled]
    multiples = [
        row.entry_price / row.observation_price for row in traded
    ]
    premiums = [
        (multiple / events.CONFIRMATION_MULTIPLE - 1.0) * 100
        for multiple in multiples
    ]
    return {
        "average_entry_to_p0_multiple": (
            round(sum(multiples) / len(multiples), 4)
            if multiples
            else 0.0
        ),
        "median_entry_to_p0_multiple": (
            round(median(multiples), 4) if multiples else 0.0
        ),
        "average_premium_over_1_5x_pct": (
            round(sum(premiums) / len(premiums), 2)
            if premiums
            else 0.0
        ),
        "maximum_premium_over_1_5x_pct": (
            round(max(premiums), 2) if premiums else 0.0
        ),
    }


def audit(
    daily: list[DayResult],
    trades: list[dict[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[str(trade["date"])].append(trade)
    expected_cash = base.INITIAL_CASH
    for row in daily:
        day_trades = trades_by_date[row.date]
        if row.start_cash != expected_cash:
            errors.append(f"{row.date}: cash chain")
        if row.end_cash != row.start_cash + row.pnl:
            errors.append(f"{row.date}: pnl equation")
        if row.buy_principal > base.DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{row.date}: principal limit")
        if sum(int(item["cash_flow"]) for item in day_trades) != row.pnl:
            errors.append(f"{row.date}: trade cash flow")
        buys = [item for item in day_trades if item["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{row.date}: multiple buys")
        quantity = 0
        for trade in day_trades:
            amount = int(trade["quantity"])
            if trade["side"] == "BUY":
                quantity += amount
                if amount > math.floor(
                    int(trade["bar_volume"])
                    * base.ENTRY_PARTICIPATION
                ):
                    errors.append(f"{row.date}: entry participation")
            else:
                quantity -= amount
        if quantity != 0:
            errors.append(f"{row.date}: open quantity")
        if row.entry_filled:
            if not (
                row.observation_minute is not None
                and row.confirmation_minute is not None
                and row.entry_minute is not None
                and row.observation_minute
                < row.confirmation_minute
                < row.entry_minute
            ):
                errors.append(f"{row.date}: causal order")
            if (
                row.confirmation_branch == "EARLY"
                and row.confirmation_minute > events.EARLY_END
            ):
                errors.append(f"{row.date}: early branch")
            if (
                row.confirmation_branch == "LATE"
                and not (
                    events.EARLY_END
                    < row.confirmation_minute
                    <= LATE_ENTRY_CUTOFF
                )
            ):
                errors.append(f"{row.date}: late branch")
        expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:30],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    cash = {branch: base.INITIAL_CASH for branch in BRANCHES}
    daily: dict[str, list[DayResult]] = {
        branch: [] for branch in BRANCHES
    }
    trades: dict[str, list[dict[str, object]]] = {
        branch: [] for branch in BRANCHES
    }
    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        for branch in BRANCHES:
            setup = select_setup(contracts, branch)
            row, day_trades = simulate_day(
                date_value,
                contracts,
                setup,
                cash[branch],
                branch,
            )
            cash[branch] = row.end_cash
            daily[branch].append(row)
            trades[branch].extend(day_trades)

    event_document = json.loads(
        (
            OUTPUT_DIR / "strategy5_v03_event_research_summary.json"
        ).read_text(encoding="utf-8")
    )
    parent_document = json.loads(
        (OUTPUT_DIR / "strategy5_summary.json").read_text(
            encoding="utf-8"
        )
    )
    portfolios: dict[str, dict[str, object]] = {}
    audits: dict[str, dict[str, object]] = {}
    for branch in BRANCHES:
        branch_daily = daily[branch]
        branch_trades = trades[branch]
        development_rows = [
            row
            for row in branch_daily
            if row.date <= legacy.DEVELOPMENT_END
        ]
        validation_rows = [
            row
            for row in branch_daily
            if row.date > legacy.DEVELOPMENT_END
        ]
        summary = summarize(branch_daily)
        summary.update(
            {
                "initial_cash": base.INITIAL_CASH,
                "ending_cash": cash[branch],
            }
        )
        traded = [row for row in branch_daily if row.entry_filled]
        branch_audit = audit(branch_daily, branch_trades)
        audits[branch] = branch_audit
        portfolios[branch] = {
            "strategy_version": portfolio_version(branch),
            "summary": summary,
            "development": summarize(development_rows),
            "validation": summarize(validation_rows),
            "comparison_to_s5_v02": {
                "total_pnl_difference": (
                    int(summary["total_pnl"])
                    - int(parent_document["summary"]["total_pnl"])
                ),
                "max_drawdown_difference": (
                    int(summary["max_drawdown"])
                    - int(parent_document["summary"]["max_drawdown"])
                ),
            },
            "analysis": {
                "by_call_put": grouped(traded, "call_put"),
                "by_exit_reason": grouped(traded, "exit_reason"),
                "entry_execution": execution_metrics(traded),
                "worst_days": [
                    asdict(row)
                    for row in sorted(
                        traded, key=lambda item: item.pnl
                    )[:15]
                ],
            },
            "audit": branch_audit,
        }

    early_summary = dict(portfolios["EARLY"]["summary"])
    late_summary = dict(portfolios["LATE"]["summary"])
    audit_result = {
        "passed": all(
            bool(result["passed"]) for result in audits.values()
        ),
        "error_count": sum(
            int(result["error_count"]) for result in audits.values()
        ),
        "branches": audits,
    }
    result = {
        "status": "experimental_dual_confirmation_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S5-v0.2",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": legacy.DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "source_intent": {
            "recording": "docs/의뢰_row/녹음3.txt",
            "early_branch": (
                "1.5x confirmation by 09:30 after a 09:00~09:10 "
                "reference price"
            ),
            "late_branch": (
                "if early-unconfirmed, continue watching for the first "
                "1.5x confirmation after 09:30"
            ),
            "entry_exit_separated": True,
        },
        "rules": {
            "observation": (
                "first actual traded-bar open within 09:00~09:10 "
                "whose price is 0.40~1.20"
            ),
            "same_observation_bar_touch": "exclude as ambiguous",
            "early_confirmation": "first 1.5x high touch by 09:30",
            "late_confirmation": (
                "first 1.5x high touch after 09:30 and by 14:30, "
                "only when no early confirmation exists"
            ),
            "daily_selection": (
                "independent EARLY and LATE portfolios; earliest "
                "confirmation within each branch, then volume and code "
                "tie-breakers"
            ),
            "portfolio_budget": (
                "each branch is an independent comparison portfolio "
                "with a 5,000,000 won daily cap; amounts are not added"
            ),
            "entry": (
                "next actual traded bar within 5 minutes, maximum "
                "5,000,000 won"
            ),
            "temporary_entry_test_exit": {
                "target": "completed close at or above 2x P0",
                "failure_stop": "completed close at or below P0",
                "time": "15:15",
            },
            "execution": {
                "slippage": base.SLIPPAGE,
                "commission_per_contract": (
                    base.COMMISSION_PER_CONTRACT
                ),
                "entry_volume_participation": (
                    base.ENTRY_PARTICIPATION
                ),
                "exit_volume_participation": (
                    base.EXIT_PARTICIPATION
                ),
            },
        },
        "event_study": event_document["modes"][OBSERVATION_MODE],
        "portfolios": portfolios,
        "branch_comparison": {
            "late_minus_early_total_pnl": (
                int(late_summary["total_pnl"])
                - int(early_summary["total_pnl"])
            ),
            "late_minus_early_max_drawdown": (
                int(late_summary["max_drawdown"])
                - int(early_summary["max_drawdown"])
            ),
            "late_minus_early_trade_days": (
                int(late_summary["trade_days"])
                - int(early_summary["trade_days"])
            ),
        },
        "audit": audit_result,
        "limitations": [
            (
                "The recording's 80% late-ten-bagger statement is not "
                "used. All late confirmations form the denominator."
            ),
            (
                "The first qualifying open is executable but does not "
                "reproduce the PPT's 1,320-contract universe exactly."
            ),
            (
                "The P0 stop and 2x-P0 target are temporary common exits "
                "for entry comparison, not optimized final exits."
            ),
            (
                "EARLY and LATE are separate comparison portfolios. "
                "Their principals and profits must not be added as one "
                "5,000,000-won-per-day strategy."
            ),
            (
                "Contract observations on the same expiry date are "
                "correlated and are not independent samples."
            ),
            (
                "S5-v0.3 was designed after reading the full-period PPT "
                "and recording, so the temporal split is diagnostic."
            ),
        ],
    }
    for branch in BRANCHES:
        suffix = branch.lower()
        write_csv(
            OUTPUT_DIR / f"strategy5_v03_{suffix}_daily.csv",
            [asdict(row) for row in daily[branch]],
        )
        write_csv(
            OUTPUT_DIR / f"strategy5_v03_{suffix}_trades.csv",
            trades[branch],
        )
    (OUTPUT_DIR / "strategy5_v03_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy5_v03_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
