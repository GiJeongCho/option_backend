from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as strategy


CANDIDATE_VERSION = "S5-research-confirm-before-buy"


@dataclass(frozen=True)
class ConfirmedSetup:
    morning: strategy.MorningSignal
    confirmation_index: int
    confirmation_volume: int


def first_observation(
    contract: base.ContractGrid,
) -> strategy.MorningSignal | None:
    start_index = base.MINUTE_INDEX[strategy.MORNING_START]
    end_index = base.MINUTE_INDEX[strategy.MORNING_END]
    for index in range(start_index, end_index + 1):
        bar = contract.bars[index]
        if not bar.traded:
            continue
        values = strategy.observed_values(bar)
        if not values:
            continue
        return strategy.MorningSignal(
            code=contract.code,
            index=index,
            minute=bar.minute,
            observed_price=max(values),
            minute_volume=bar.volume,
            cumulative_volume=sum(
                item.volume
                for item in contract.bars[start_index : index + 1]
            ),
        )
    return None


def first_confirmed_setup(
    contracts: dict[str, base.ContractGrid],
) -> ConfirmedSetup | None:
    candidates: list[ConfirmedSetup] = []
    for contract in contracts.values():
        morning = first_observation(contract)
        if morning is None:
            continue
        deadline = min(
            len(contract.bars) - 1,
            morning.index + strategy.CONFIRMATION_WINDOW_MINUTES,
        )
        for index in range(morning.index + 1, deadline + 1):
            bar = contract.bars[index]
            if (
                bar.traded
                and bar.high is not None
                and float(bar.high)
                >= morning.observed_price
                * strategy.CONFIRMATION_MULTIPLE
            ):
                candidates.append(
                    ConfirmedSetup(
                        morning=morning,
                        confirmation_index=index,
                        confirmation_volume=bar.volume,
                    )
                )
                break
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item.confirmation_index,
            -item.confirmation_volume,
            -item.morning.cumulative_volume,
            item.morning.code,
        ),
    )[0]


def buy_after_confirmation(
    date_value: str,
    contract: base.ContractGrid,
    setup: ConfirmedSetup,
    available_cash: int,
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
            "strategy": CANDIDATE_VERSION,
            "date": date_value,
            "campaign": 1,
            "minute": bar.minute,
            "code": contract.code,
            "call_put": contract.call_put,
            "side": "BUY",
            "reason": "MORNING_1_5X_CONFIRM_ENTRY",
            "stage": 1,
            "quantity": quantity,
            "price": fill_price,
            "principal": principal,
            "fee": fee,
            "cash_flow": -(principal + fee),
            "bar_volume": bar.volume,
        },
    )


def simulate_day(
    date_value: str,
    contracts: dict[str, base.ContractGrid],
    setup: ConfirmedSetup | None,
    start_cash: int,
) -> tuple[strategy.DayResult, list[dict[str, object]]]:
    if setup is None:
        row = strategy.empty_day(date_value, start_cash)
        row.strategy = CANDIDATE_VERSION
        return row, []
    contract = contracts[setup.morning.code]
    fill = buy_after_confirmation(
        date_value,
        contract,
        setup,
        start_cash,
    )
    if fill is None:
        row = strategy.empty_day(
            date_value,
            start_cash,
            setup.morning,
            contract.call_put,
        )
        row.strategy = CANDIDATE_VERSION
        return row, []
    (
        entry_index,
        quantity,
        buy_principal,
        buy_fee,
        buy_trade,
    ) = fill
    initial_price = float(buy_trade["price"])
    (
        remaining,
        gross_sales,
        sell_fee,
        exit_reason,
        exit_index,
        reached_2x,
        reached_5x,
        reached_10x,
        sell_trades,
    ) = strategy.simulate_confirmed_position(
        date_value,
        contract,
        entry_index,
        setup.morning.observed_price,
        quantity,
    )
    trades = [buy_trade, *sell_trades]
    for trade in trades:
        trade["strategy"] = CANDIDATE_VERSION
    if remaining > 0:
        trades.append(
            {
                "strategy": CANDIDATE_VERSION,
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
    mfe_multiple, mae_pct = strategy.excursion(
        contract,
        entry_index,
        initial_price,
    )
    return (
        strategy.DayResult(
            strategy=CANDIDATE_VERSION,
            date=date_value,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=contract.code,
            call_put=contract.call_put,
            signal_minute=setup.morning.minute,
            observed_price=round(setup.morning.observed_price, 4),
            entry_minute=contract.bars[entry_index].minute,
            initial_price=round(initial_price, 4),
            entry_quantity=quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            confirmed=True,
            confirmation_minute=contract.bars[
                setup.confirmation_index
            ].minute,
            reached_2x=reached_2x,
            reached_5x=reached_5x,
            reached_10x=reached_10x,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_reason=exit_reason,
            exit_minute=contract.bars[exit_index].minute,
            expired_quantity=remaining,
            mfe_multiple=mfe_multiple,
            mae_pct=mae_pct,
        ),
        trades,
    )


def audit_confirmation_entry(
    daily: list[strategy.DayResult],
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
            errors.append(f"{row.date}: cash flow")
        buys = [item for item in day_trades if item["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{row.date}: multiple entries")
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
        if row.signal_minute is not None and not (
            strategy.MORNING_START
            <= row.signal_minute
            <= strategy.MORNING_END
        ):
            errors.append(f"{row.date}: signal time")
        if row.entry_filled:
            if (
                row.signal_minute is None
                or row.confirmation_minute is None
                or row.entry_minute is None
            ):
                errors.append(f"{row.date}: missing causal timestamp")
            elif not (
                row.signal_minute
                < row.confirmation_minute
                < row.entry_minute
            ):
                errors.append(f"{row.date}: causal order")
        expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:20],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
    }


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if strategy.PPT_START
        <= group["expiry_date"]
        <= strategy.PPT_END
    ]
    files = base.target_file_map()
    cash = base.INITIAL_CASH
    daily: list[strategy.DayResult] = []
    trades: list[dict[str, object]] = []
    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        setup = first_confirmed_setup(contracts)
        row, day_trades = simulate_day(
            date_value,
            contracts,
            setup,
            cash,
        )
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)

    development = [
        row for row in daily if row.date <= strategy.DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > strategy.DEVELOPMENT_END
    ]
    result = {
        "status": "ppt_confirmation_entry_research",
        "candidate": CANDIDATE_VERSION,
        "rule": (
            "after each completed 09:00~09:10 bar, use the maximum "
            "qualifying 0.40~1.20 OHLC value; buy only after the first "
            "1.5x confirmation; use the same 2x/5x/10x ladder"
        ),
        "all": strategy.summarize(daily),
        "development": strategy.summarize(development),
        "validation": strategy.summarize(validation),
        "audit": audit_confirmation_entry(daily, trades),
        "daily": [asdict(row) for row in daily],
    }
    (strategy.OUTPUT_DIR / "strategy5_research_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
