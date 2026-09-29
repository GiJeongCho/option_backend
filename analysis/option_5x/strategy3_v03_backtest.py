from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy2_probe_research as execution
import strategy3_v02_backtest as parent
from strategy1_backtest import (
    COMMISSION,
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    MULTIPLIER,
    SLIPPAGE,
    VOLUME_PARTICIPATION,
    previous_volatility,
    volatility_closes,
)
from strategy2_backtest import audit


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S3-v0.3"
HIGH_VOL_START = "20251001"
DEVELOPMENT_END = "20260430"
PATTERN_CUTOFF = 1430
FIRST_TO_LOW_MINUTES = 30
LOW_TO_SECOND_CROSS_MAX_MINUTES = 15
SECOND_LOW_RATIO_MIN = 0.90
SECOND_LOW_RATIO_MAX = 0.95
STRENGTH_HOLD_MINUTES = 3
STRENGTH_WAIT_MINUTES = 5
IMPROVED_BUDGETS = (1_000_000, 2_000_000, 2_000_000)
BASELINE_BUDGETS = parent.TRANCHE_BUDGETS
TARGET_MULTIPLE = 2.0

EntryMode = Literal["v02_falling_buy", "v03_confirmed_add"]
ExitMode = Literal["v02_second_cross", "v03_recover_trail"]


@dataclass(frozen=True)
class Config:
    name: str
    entry_mode: EntryMode
    exit_mode: ExitMode

    @property
    def key(self) -> str:
        return self.name


CONFIGS = (
    Config("S3-v0.3-entry-only", "v03_confirmed_add", "v02_second_cross"),
    Config("S3-v0.3-exit-only", "v02_falling_buy", "v03_recover_trail"),
    Config(STRATEGY_VERSION, "v03_confirmed_add", "v03_recover_trail"),
)
FIXED_CONFIG = CONFIGS[-1]


@dataclass(frozen=True)
class Pattern:
    lower_low_index: int | None
    lower_low_ratio: float
    first_death: parent.CrossEvent | None
    second_golden: parent.CrossEvent | None
    strength_index: int | None
    failure_index: int
    rejection_reason: str


@dataclass
class DayResult:
    strategy: str
    date: str
    previous_vix: float
    start_cash: int
    end_cash: int
    pnl: int
    signal_found: bool
    entry_filled: bool
    code: str
    call_put: str
    first_low: float
    second_low_ratio: float
    first_golden_cross_minute: int | None
    lower_low_minute: int | None
    second_golden_cross_minute: int | None
    strength_minute: int | None
    pattern_qualified: bool
    pattern_rejection_reason: str
    entry_stages: int
    entry_quantity: int
    average_entry_price: float
    buy_principal: int
    buy_fee: int
    reached_2x: bool
    principal_recovered: bool
    recovered_net_sales: int
    runner_initial_quantity: int
    gross_sales: int
    sell_fee: int
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float


@dataclass
class SaleResult:
    filled: int
    gross_sales: int
    fees: int
    last_fill_index: int
    trades: list[dict[str, object]]
    reason: str
    principal_recovered: bool = False
    recovered_net_sales: int = 0
    runner_initial_quantity: int = 0


def minute(index: int) -> int:
    return base.SESSION_MINUTES[index]


def index_at_or_before(minute_value: int) -> int:
    return base.MINUTE_INDEX[minute_value]


def quality_lower_low(
    contract: base.ContractGrid,
    setup: parent.Setup,
    first_fill_index: int,
) -> tuple[int | None, float, int, str]:
    cutoff = index_at_or_before(PATTERN_CUTOFF)
    for index in range(first_fill_index + 1, cutoff + 1):
        bar = contract.bars[index]
        if not bar.traded or bar.low is None:
            continue
        ratio = float(bar.low) / setup.first_low
        if ratio < SECOND_LOW_RATIO_MIN:
            return None, ratio, index, "SECOND_LOW_TOO_DEEP"
        if ratio > SECOND_LOW_RATIO_MAX:
            continue
        if index - setup.event.raw_index < FIRST_TO_LOW_MINUTES:
            return None, ratio, index, "SECOND_LOW_TOO_EARLY"
        if (
            bar.close is None
            or not parent.PREMIUM_MIN
            <= float(bar.close)
            <= parent.PREMIUM_MAX
        ):
            continue
        return index, ratio, index, ""
    return None, 0.0, cutoff, "NO_QUALIFIED_SECOND_LOW"


def strength_confirmation(
    contract: base.ContractGrid,
    second_golden: parent.CrossEvent,
) -> int | None:
    start = second_golden.confirmation_index
    end = min(
        len(contract.bars) - 1,
        start + STRENGTH_WAIT_MINUTES,
        index_at_or_before(PATTERN_CUTOFF),
    )
    for index in range(start + STRENGTH_HOLD_MINUTES - 1, end + 1):
        if any(
            not parent.relation(contract, position, "up")
            for position in range(start, index + 1)
        ):
            return None
        bar = contract.bars[index]
        current_ma20 = contract.ma20[index]
        start_ma20 = contract.ma20[start]
        if (
            bar.traded
            and bar.close is not None
            and parent.PREMIUM_MIN
            <= float(bar.close)
            <= parent.PREMIUM_MAX
            and current_ma20 is not None
            and start_ma20 is not None
            and float(current_ma20) > float(start_ma20)
        ):
            return index
    return None


def build_pattern(
    contract: base.ContractGrid,
    setup: parent.Setup,
    first_fill_index: int,
    entry_mode: EntryMode,
) -> Pattern:
    up_events = parent.cross_events(contract, "up")
    down_events = parent.cross_events(contract, "down")
    first_death = parent.first_event_after(
        down_events,
        first_fill_index,
        raw_after=setup.event.raw_index,
    )
    if entry_mode == "v02_falling_buy":
        lower = parent.first_lower_low(
            contract, first_fill_index, setup.first_low
        )
        lower_index = lower[0] if lower else None
        ratio = (
            float(lower[1]) / setup.first_low if lower is not None else 0.0
        )
        second_golden = None
        if lower_index is not None and first_death is not None:
            second_golden = parent.first_event_after(
                up_events,
                max(lower_index, first_death.confirmation_index),
                raw_after=first_death.raw_index,
                raw_gap_from=setup.event.raw_index,
            )
        failure_index = (
            min(
                len(contract.bars) - 1,
                lower_index + LOW_TO_SECOND_CROSS_MAX_MINUTES,
            )
            if lower_index is not None and second_golden is None
            else index_at_or_before(PATTERN_CUTOFF)
        )
        return Pattern(
            lower_low_index=lower_index,
            lower_low_ratio=ratio,
            first_death=first_death,
            second_golden=second_golden,
            strength_index=(
                second_golden.confirmation_index
                if second_golden is not None
                else None
            ),
            failure_index=failure_index,
            rejection_reason=(
                ""
                if second_golden is not None
                else "NO_SECOND_GOLDEN_CROSS"
            ),
        )

    lower_index, ratio, failure_index, rejection = quality_lower_low(
        contract, setup, first_fill_index
    )
    second_golden = None
    if lower_index is not None and first_death is not None:
        candidate = parent.first_event_after(
            up_events,
            max(lower_index, first_death.confirmation_index),
            raw_after=first_death.raw_index,
            raw_gap_from=setup.event.raw_index,
        )
        if (
            candidate is not None
            and candidate.confirmation_index - lower_index
            <= LOW_TO_SECOND_CROSS_MAX_MINUTES
        ):
            second_golden = candidate
        else:
            failure_index = min(
                len(contract.bars) - 1,
                lower_index + LOW_TO_SECOND_CROSS_MAX_MINUTES,
            )
            rejection = "SECOND_GOLDEN_CROSS_TOO_LATE"
    elif lower_index is not None:
        rejection = "NO_CONFIRMED_FIRST_DEATH_CROSS"

    strength_index = (
        strength_confirmation(contract, second_golden)
        if second_golden is not None
        else None
    )
    if second_golden is not None and strength_index is None:
        rejection = "STRENGTH_NOT_CONFIRMED"
        failure_index = min(
            len(contract.bars) - 1,
            second_golden.confirmation_index + STRENGTH_WAIT_MINUTES,
        )
    return Pattern(
        lower_low_index=lower_index,
        lower_low_ratio=ratio,
        first_death=first_death,
        second_golden=second_golden,
        strength_index=strength_index,
        failure_index=failure_index,
        rejection_reason=rejection,
    )


def buy_stage(
    date_value: str,
    config: Config,
    contract: base.ContractGrid,
    signal_index: int,
    stage: int,
    budget: int,
    start_cash: int,
    buy_principal: int,
    buy_fee: int,
) -> tuple[int, int, int, int, dict[str, object]] | None:
    bar = contract.bars[signal_index]
    if bar.close is None:
        return None
    allowed_budget = min(
        budget, DAILY_PRINCIPAL_LIMIT - buy_principal
    )
    if allowed_budget <= 0:
        return None
    fill = execution.buy(
        date_value,
        config,  # type: ignore[arg-type]
        contract,
        signal_index,
        float(bar.close),
        allowed_budget,
        start_cash - buy_principal - buy_fee,
        f"STAGE_{stage}",
    )
    if fill is None:
        return None
    fill_index, quantity, principal, fee, trade = fill
    trade["stage"] = stage
    trade["call_put"] = contract.call_put
    return fill_index, quantity, principal, fee, trade


def target_index(
    contract: base.ContractGrid,
    start_index: int,
    average_entry_price: float,
) -> int | None:
    cutoff = index_at_or_before(FORCE_EXIT_TIME)
    for index in range(start_index + 1, cutoff + 1):
        bar = contract.bars[index]
        if (
            bar.traded
            and bar.close is not None
            and float(bar.close)
            >= TARGET_MULTIPLE * average_entry_price
        ):
            return index
    return None


def sell_to_cash_target(
    date_value: str,
    config: Config,
    contract: base.ContractGrid,
    signal_index: int,
    quantity: int,
    cash_target: int,
) -> tuple[int, int, int, int, list[dict[str, object]]]:
    remaining = quantity
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
        capacity += bar.volume * VOLUME_PARTICIPATION
        available = math.floor(capacity + 1e-12)
        price = max(0.0, bar.open - SLIPPAGE)
        net_per_contract = round(price * MULTIPLIER) - COMMISSION
        if available <= 0 or net_per_contract <= 0:
            continue
        needed = max(
            1,
            math.ceil(
                (cash_target - (gross_sales - fees))
                / net_per_contract
            ),
        )
        fillable = min(remaining, available, needed)
        if fillable <= 0:
            continue
        capacity -= fillable
        principal = round(fillable * price * MULTIPLIER)
        fee = fillable * COMMISSION
        remaining -= fillable
        gross_sales += principal
        fees += fee
        last_fill_index = index
        trades.append(
            {
                "strategy": config.key,
                "date": date_value,
                "minute": bar.minute,
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "SELL",
                "reason": "PRINCIPAL_RECOVERY_2X",
                "stage": 0,
                "quantity": fillable,
                "price": price,
                "principal": principal,
                "fee": fee,
                "cash_flow": principal - fee,
                "bar_volume": bar.volume,
            }
        )
        if gross_sales - fees >= cash_target or remaining <= 0:
            break
    return (
        quantity - remaining,
        gross_sales,
        fees,
        last_fill_index,
        trades,
    )


def trailing_signal(
    contract: base.ContractGrid,
    target_signal_index: int,
    start_index: int,
    average_entry_price: float,
) -> tuple[int | None, str]:
    peak_close = TARGET_MULTIPLE * average_entry_price
    cutoff = index_at_or_before(FORCE_EXIT_TIME)
    for index in range(target_signal_index, cutoff + 1):
        bar = contract.bars[index]
        if not bar.traded or bar.close is None:
            continue
        peak_close = max(peak_close, float(bar.close))
        if index <= start_index:
            continue
        peak_multiple = peak_close / average_entry_price
        if peak_multiple >= 10.0:
            drawdown = 0.15
        elif peak_multiple >= 5.0:
            drawdown = 0.20
        else:
            drawdown = 0.25
        if float(bar.close) <= peak_close * (1.0 - drawdown):
            return index, f"TRAIL_{round(drawdown * 100)}PCT"
    return None, "TIME"


def add_trade_metadata(
    trades: list[dict[str, object]],
    call_put: str,
    stage: int,
) -> None:
    for trade in trades:
        trade["call_put"] = call_put
        trade["stage"] = stage


def recover_and_trail(
    date_value: str,
    config: Config,
    contract: base.ContractGrid,
    signal_index: int,
    quantity: int,
    cash_target: int,
    average_entry_price: float,
    stage: int,
) -> SaleResult:
    (
        recovery_filled,
        recovery_gross,
        recovery_fees,
        recovery_last_index,
        recovery_trades,
    ) = sell_to_cash_target(
        date_value,
        config,
        contract,
        signal_index,
        quantity,
        cash_target,
    )
    remaining = quantity - recovery_filled
    recovered_net = recovery_gross - recovery_fees
    if remaining <= 0:
        return SaleResult(
            filled=recovery_filled,
            gross_sales=recovery_gross,
            fees=recovery_fees,
            last_fill_index=recovery_last_index,
            trades=recovery_trades,
            reason="PRINCIPAL_RECOVERY_2X",
            principal_recovered=recovered_net >= cash_target,
            recovered_net_sales=recovered_net,
            runner_initial_quantity=0,
        )

    trail_index, trail_reason = trailing_signal(
        contract,
        signal_index,
        recovery_last_index,
        average_entry_price,
    )
    runner_signal = (
        trail_index
        if trail_index is not None
        else max(
            recovery_last_index,
            index_at_or_before(FORCE_EXIT_TIME),
        )
    )
    (
        runner_filled,
        runner_gross,
        runner_fees,
        runner_last_index,
        runner_trades,
    ) = execution.sell_all(
        date_value,
        config,  # type: ignore[arg-type]
        contract,
        runner_signal,
        remaining,
        f"RECOVER_THEN_{trail_reason}",
    )
    add_trade_metadata(runner_trades, contract.call_put, stage)
    return SaleResult(
        filled=recovery_filled + runner_filled,
        gross_sales=recovery_gross + runner_gross,
        fees=recovery_fees + runner_fees,
        last_fill_index=max(recovery_last_index, runner_last_index),
        trades=recovery_trades + runner_trades,
        reason=f"RECOVER_THEN_{trail_reason}",
        principal_recovered=recovered_net >= cash_target,
        recovered_net_sales=recovered_net,
        runner_initial_quantity=remaining,
    )


def regular_sale(
    date_value: str,
    config: Config,
    contract: base.ContractGrid,
    signal_index: int,
    quantity: int,
    reason: str,
    stage: int,
) -> SaleResult:
    filled, gross, fees, last_index, trades = execution.sell_all(
        date_value,
        config,  # type: ignore[arg-type]
        contract,
        signal_index,
        quantity,
        reason,
    )
    add_trade_metadata(trades, contract.call_put, stage)
    return SaleResult(
        filled=filled,
        gross_sales=gross,
        fees=fees,
        last_fill_index=last_index,
        trades=trades,
        reason=reason,
    )


def failure_exit_event(
    contract: base.ContractGrid,
    setup: parent.Setup,
    pattern: Pattern,
    last_entry_index: int,
) -> parent.CrossEvent | None:
    down_events = parent.cross_events(contract, "down")
    if pattern.second_golden is not None:
        return parent.first_event_after(
            down_events,
            max(
                last_entry_index,
                pattern.second_golden.confirmation_index,
            ),
            raw_after=pattern.second_golden.raw_index,
            raw_gap_from=(
                pattern.first_death.raw_index
                if pattern.first_death is not None
                else setup.event.raw_index
            ),
        )
    return parent.first_event_after(
        down_events,
        max(last_entry_index, pattern.failure_index),
        raw_after=(
            pattern.first_death.raw_index
            if pattern.first_death is not None
            else setup.event.raw_index
        ),
    )


def empty_day(
    config: Config,
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool,
) -> DayResult:
    return DayResult(
        strategy=config.key,
        date=date_value,
        previous_vix=previous_vix,
        start_cash=start_cash,
        end_cash=start_cash,
        pnl=0,
        signal_found=signal_found,
        entry_filled=False,
        code="",
        call_put="",
        first_low=0.0,
        second_low_ratio=0.0,
        first_golden_cross_minute=None,
        lower_low_minute=None,
        second_golden_cross_minute=None,
        strength_minute=None,
        pattern_qualified=False,
        pattern_rejection_reason="",
        entry_stages=0,
        entry_quantity=0,
        average_entry_price=0.0,
        buy_principal=0,
        buy_fee=0,
        reached_2x=False,
        principal_recovered=False,
        recovered_net_sales=0,
        runner_initial_quantity=0,
        gross_sales=0,
        sell_fee=0,
        exit_reason="",
        exit_minute=None,
        expired_quantity=0,
        mfe_multiple=0.0,
        mae_pct=0.0,
    )


def simulate_day(
    date_value: str,
    previous_vix: float,
    config: Config,
    contracts: dict[str, base.ContractGrid],
    setup: parent.Setup | None,
    start_cash: int,
) -> tuple[DayResult, list[dict[str, object]]]:
    if setup is None:
        return empty_day(
            config, date_value, previous_vix, start_cash, False
        ), []
    contract = contracts[setup.code]
    budgets = (
        IMPROVED_BUDGETS
        if config.entry_mode == "v03_confirmed_add"
        else BASELINE_BUDGETS
    )
    first_fill = buy_stage(
        date_value,
        config,
        contract,
        setup.event.confirmation_index,
        1,
        budgets[0],
        start_cash,
        0,
        0,
    )
    if first_fill is None:
        return empty_day(
            config, date_value, previous_vix, start_cash, True
        ), []
    (
        first_fill_index,
        quantity,
        buy_principal,
        buy_fee,
        first_trade,
    ) = first_fill
    trades = [first_trade]
    stages = 1
    last_entry_index = first_fill_index
    pattern = build_pattern(
        contract, setup, first_fill_index, config.entry_mode
    )

    entry_signals: list[tuple[int, int, int]] = []
    if config.entry_mode == "v02_falling_buy":
        if pattern.lower_low_index is not None:
            entry_signals.append(
                (2, pattern.lower_low_index, budgets[1])
            )
        if pattern.second_golden is not None:
            entry_signals.append(
                (
                    3,
                    pattern.second_golden.confirmation_index,
                    budgets[2],
                )
            )
    else:
        if pattern.second_golden is not None:
            entry_signals.append(
                (
                    2,
                    pattern.second_golden.confirmation_index,
                    budgets[1],
                )
            )
        if pattern.strength_index is not None:
            entry_signals.append((3, pattern.strength_index, budgets[2]))

    for stage, signal_index, budget in entry_signals:
        fill = buy_stage(
            date_value,
            config,
            contract,
            signal_index,
            stage,
            budget,
            start_cash,
            buy_principal,
            buy_fee,
        )
        if fill is None:
            continue
        fill_index, added, principal, fee, trade = fill
        quantity += added
        buy_principal += principal
        buy_fee += fee
        stages += 1
        last_entry_index = max(last_entry_index, fill_index)
        trades.append(trade)

    average_entry_price = buy_principal / quantity / MULTIPLIER
    failure_event = failure_exit_event(
        contract, setup, pattern, last_entry_index
    )
    failure_index = (
        failure_event.confirmation_index
        if failure_event is not None
        and contract.bars[failure_event.confirmation_index].minute
        <= FORCE_EXIT_TIME
        else None
    )

    reached_2x_index = target_index(
        contract, last_entry_index, average_entry_price
    )
    if config.exit_mode == "v03_recover_trail":
        if (
            reached_2x_index is not None
            and (
                failure_index is None
                or reached_2x_index <= failure_index
            )
        ):
            sale = recover_and_trail(
                date_value,
                config,
                contract,
                reached_2x_index,
                quantity,
                buy_principal + buy_fee,
                average_entry_price,
                stages,
            )
        else:
            sale = regular_sale(
                date_value,
                config,
                contract,
                (
                    failure_index
                    if failure_index is not None
                    else index_at_or_before(FORCE_EXIT_TIME)
                ),
                quantity,
                (
                    "PATTERN_FAILURE_DEATH_CROSS"
                    if failure_index is not None
                    else "TIME"
                ),
                stages,
            )
    else:
        baseline_failure_index = (
            failure_index
            if failure_index is not None
            and pattern.second_golden is not None
            else None
        )
        sale = regular_sale(
            date_value,
            config,
            contract,
            (
                baseline_failure_index
                if baseline_failure_index is not None
                else index_at_or_before(FORCE_EXIT_TIME)
            ),
            quantity,
            (
                "SECOND_DEATH_CROSS"
                if baseline_failure_index is not None
                else "TIME"
            ),
            stages,
        )
    trades.extend(sale.trades)
    expired_quantity = quantity - sale.filled
    if expired_quantity > 0:
        trades.append(
            {
                "strategy": config.key,
                "date": date_value,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "stage": stages,
                "quantity": expired_quantity,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
        exit_reason = "UNFILLED_ZERO_VALUE"
        exit_minute = base.SESSION_MINUTES[-1]
    else:
        exit_reason = sale.reason
        exit_minute = contract.bars[sale.last_fill_index].minute

    mfe_multiple, mae_pct = parent.excursion(
        contract, first_fill_index, average_entry_price
    )
    pnl = (
        sale.gross_sales - sale.fees - buy_principal - buy_fee
    )
    return (
        DayResult(
            strategy=config.key,
            date=date_value,
            previous_vix=previous_vix,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=contract.code,
            call_put=contract.call_put,
            first_low=round(setup.first_low, 4),
            second_low_ratio=round(pattern.lower_low_ratio, 6),
            first_golden_cross_minute=minute(setup.event.raw_index),
            lower_low_minute=(
                minute(pattern.lower_low_index)
                if pattern.lower_low_index is not None
                else None
            ),
            second_golden_cross_minute=(
                minute(pattern.second_golden.raw_index)
                if pattern.second_golden is not None
                else None
            ),
            strength_minute=(
                minute(pattern.strength_index)
                if pattern.strength_index is not None
                else None
            ),
            pattern_qualified=pattern.second_golden is not None,
            pattern_rejection_reason=pattern.rejection_reason,
            entry_stages=stages,
            entry_quantity=quantity,
            average_entry_price=round(average_entry_price, 6),
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            reached_2x=reached_2x_index is not None,
            principal_recovered=sale.principal_recovered,
            recovered_net_sales=sale.recovered_net_sales,
            runner_initial_quantity=sale.runner_initial_quantity,
            gross_sales=sale.gross_sales,
            sell_fee=sale.fees,
            exit_reason=exit_reason,
            exit_minute=exit_minute,
            expired_quantity=expired_quantity,
            mfe_multiple=round(mfe_multiple, 6),
            mae_pct=round(mae_pct, 4),
        ),
        trades,
    )


def max_drawdown(rows: list[DayResult]) -> int:
    equity = INITIAL_CASH
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
        "total_pnl": sum(profits),
        "average_trade_day_pnl": (
            round(sum(profits) / len(profits)) if profits else 0
        ),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(profits, default=0),
        "maximum_day_pnl": max(profits, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": sum(profits) - sum(ordered[:3]),
        "pnl_excluding_best_5_days": sum(profits) - sum(ordered[:5]),
        "total_buy_principal": sum(
            row.buy_principal for row in traded
        ),
        "expired_contracts": sum(
            row.expired_quantity for row in traded
        ),
        "one_stage_days": sum(row.entry_stages == 1 for row in traded),
        "two_stage_days": sum(row.entry_stages == 2 for row in traded),
        "three_stage_days": sum(row.entry_stages == 3 for row in traded),
        "qualified_pattern_days": sum(
            row.pattern_qualified for row in traded
        ),
        "reached_2x_days": sum(row.reached_2x for row in traded),
        "principal_recovered_days": sum(
            row.principal_recovered for row in traded
        ),
    }


def grouped(
    rows: list[DayResult], field: str
) -> list[dict[str, object]]:
    values: dict[str, list[int]] = {}
    for row in rows:
        values.setdefault(str(getattr(row, field)), []).append(row.pnl)
    return [
        {
            "group": name,
            "count": len(profits),
            "profitable_count": sum(value > 0 for value in profits),
            "losing_count": sum(value < 0 for value in profits),
            "total_pnl": sum(profits),
            "average_pnl": round(sum(profits) / len(profits)),
        }
        for name, profits in sorted(values.items())
    ]


def candidate(
    config: Config, rows: list[DayResult]
) -> dict[str, object]:
    return {
        "strategy": config.key,
        "entry_mode": config.entry_mode,
        "exit_mode": config.exit_mode,
        "all": summarize(rows),
        "development": summarize(
            [row for row in rows if row.date <= DEVELOPMENT_END]
        ),
        "validation": summarize(
            [row for row in rows if row.date > DEVELOPMENT_END]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        if group["expiry_date"] >= HIGH_VOL_START
    ]
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    states = {
        config.key: {
            "config": config,
            "cash": INITIAL_CASH,
            "daily": [],
            "trades": [],
        }
        for config in CONFIGS
    }

    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        setup = parent.first_setup(contracts)
        for state in states.values():
            config = state["config"]
            row, trades = simulate_day(
                date_value,
                previous_vix,
                config,  # type: ignore[arg-type]
                contracts,
                setup,
                int(state["cash"]),
            )
            state["cash"] = row.end_cash
            state["daily"].append(row)  # type: ignore[union-attr]
            state["trades"].extend(trades)  # type: ignore[union-attr]

    candidates = [
        candidate(
            state["config"],  # type: ignore[arg-type]
            state["daily"],  # type: ignore[arg-type]
        )
        for state in states.values()
    ]
    audits = {
        key: audit(
            state["daily"],  # type: ignore[arg-type]
            state["trades"],  # type: ignore[arg-type]
        )
        for key, state in states.items()
    }
    fixed = states[STRATEGY_VERSION]
    fixed_daily: list[DayResult] = fixed["daily"]  # type: ignore[assignment]
    fixed_trades: list[dict[str, object]] = fixed["trades"]  # type: ignore[assignment]
    fixed_candidate = next(
        row for row in candidates if row["strategy"] == STRATEGY_VERSION
    )
    parent_v02 = json.loads(
        (OUTPUT_DIR / "strategy3_v02_summary.json").read_text(
            encoding="utf-8"
        )
    )
    parent_v01 = json.loads(
        (OUTPUT_DIR / "strategy3_summary.json").read_text(encoding="utf-8")
    )
    summary = dict(fixed_candidate["all"])
    summary.update(
        {
            "initial_cash": INITIAL_CASH,
            "ending_cash": int(fixed["cash"]),
        }
    )
    traded = [row for row in fixed_daily if row.entry_filled]
    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S3-v0.2",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": DEVELOPMENT_END,
        },
        "rules": {
            "probe_budget": IMPROVED_BUDGETS[0],
            "lower_low_action": "observe_only",
            "second_low_ratio": [
                SECOND_LOW_RATIO_MIN,
                SECOND_LOW_RATIO_MAX,
            ],
            "first_cross_to_lower_low_min_minutes": (
                FIRST_TO_LOW_MINUTES
            ),
            "lower_low_to_second_cross_max_minutes": (
                LOW_TO_SECOND_CROSS_MAX_MINUTES
            ),
            "second_cross_budget": IMPROVED_BUDGETS[1],
            "strength_confirmation": (
                "MA5 above MA20 held for 3 minutes and MA20 rising"
            ),
            "strength_budget": IMPROVED_BUDGETS[2],
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "target": "recover invested cash at 2x average entry",
            "runner_trailing_drawdown": {
                "2x_to_below_5x": 0.25,
                "5x_to_below_10x": 0.20,
                "10x_or_more": 0.15,
            },
            "failure_exit": "next confirmed bearish MA cross",
        },
        "summary": summary,
        "development": fixed_candidate["development"],
        "validation": fixed_candidate["validation"],
        "research_candidates": [
            {
                "strategy": "S3-v0.2",
                "entry_mode": "v02_falling_buy",
                "exit_mode": "v02_second_cross",
                "all": parent_v02["summary"],
                "development": parent_v02["development"],
                "validation": parent_v02["validation"],
            },
            *candidates,
        ],
        "comparison": {
            "to_s3_v02_total_pnl": (
                int(summary["total_pnl"])
                - int(parent_v02["summary"]["total_pnl"])
            ),
            "to_s3_v02_development_pnl": (
                int(fixed_candidate["development"]["total_pnl"])
                - int(parent_v02["development"]["total_pnl"])
            ),
            "to_s3_v02_validation_pnl": (
                int(fixed_candidate["validation"]["total_pnl"])
                - int(parent_v02["validation"]["total_pnl"])
            ),
            "to_s3_v01_total_pnl": (
                int(summary["total_pnl"])
                - int(parent_v01["summary"]["total_pnl"])
            ),
        },
        "analysis": {
            "by_entry_stages": grouped(traded, "entry_stages"),
            "by_exit_reason": grouped(traded, "exit_reason"),
            "by_pattern_rejection": grouped(
                traded, "pattern_rejection_reason"
            ),
            "worst_days": [
                asdict(row)
                for row in sorted(traded, key=lambda item: item.pnl)[:15]
            ],
        },
        "audit": audits[STRATEGY_VERSION],
        "candidate_audits": audits,
        "limitations": [
            "0.90~0.95 저점 깊이와 시간 조건은 동일 표본의 탐색 결과이므로 후속 전진검증이 필요합니다.",
            "2배 보호는 마지막 체결된 분할매수 뒤부터 활성화해 주문 대기 중 상태 충돌을 피했습니다.",
            "원금회수와 잔량청산은 각각 별도 거래량 참여 한도를 적용하며 앞 주문 체결 이후에만 다음 주문을 냅니다.",
            "미체결 만기 잔량은 0원으로 평가했습니다.",
        ],
    }
    write_csv(
        OUTPUT_DIR / "strategy3_v03_daily.csv",
        [asdict(row) for row in fixed_daily],
    )
    write_csv(OUTPUT_DIR / "strategy3_v03_trades.csv", fixed_trades)
    write_csv(
        OUTPUT_DIR / "strategy3_v03_research_daily.csv",
        [
            asdict(row)
            for state in states.values()
            for row in state["daily"]  # type: ignore[union-attr]
        ],
    )
    write_csv(
        OUTPUT_DIR / "strategy3_v03_research_trades.csv",
        [
            row
            for state in states.values()
            for row in state["trades"]  # type: ignore[union-attr]
        ],
    )
    (OUTPUT_DIR / "strategy3_v03_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy3_v03_audit.json").write_text(
        json.dumps(audits[STRATEGY_VERSION], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(row["passed"] for row in audits.values()):
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
