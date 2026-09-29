from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy2_probe_research as execution
from strategy1_backtest import (
    COMMISSION,
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    MULTIPLIER,
    SLIPPAGE,
    previous_volatility,
    volatility_closes,
)
from strategy2_backtest import audit


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
HIGH_VOL_START = "20251001"
DEVELOPMENT_END = "20260430"
ENTRY_CUTOFF = 1430
ENTRY_PRICE_MIN = 1.00
ENTRY_PRICE_MAX = 5.00
MAX_REBOUND = 1.80

ExitMode = Literal[
    "time_only",
    "two_x_all",
    "two_x_half_five_all",
    "two_x_recover_five_all",
    "ladder_2_5_10",
    "recover_trail",
    "recover_trail_balanced",
    "recover_trail_tight",
    "trailing_only",
]


@dataclass(frozen=True)
class EntryConfig:
    name: str
    low_lookback: int
    rebound: float
    breakout_lookback: int
    minimum_volume_ratio: float
    require_ma20_rising: bool
    previous_vix_min: float | None = None

    @property
    def key(self) -> str:
        return f"S3-entry|{self.name}"


ENTRY_CONFIGS = (
    EntryConfig("L40_R20_B5", 40, 1.20, 5, 0.0, False),
    EntryConfig("L20_R20_B5_MA20", 20, 1.20, 5, 0.0, True),
    EntryConfig("L40_R20_B5_MA20", 40, 1.20, 5, 0.0, True),
    EntryConfig("L60_R20_B5_MA20", 60, 1.20, 5, 0.0, True),
    EntryConfig("L40_R20_B5_V1_MA20", 40, 1.20, 5, 1.0, True),
    EntryConfig("L40_R30_B5_V1_MA20", 40, 1.30, 5, 1.0, True),
    EntryConfig("L40_R20_B10_V1_MA20", 40, 1.20, 10, 1.0, True),
    EntryConfig(
        "L40_R20_B5_V1_MA20_VIX30",
        40,
        1.20,
        5,
        1.0,
        True,
        30.0,
    ),
    EntryConfig(
        "L40_R30_B5_V1_MA20_VIX30",
        40,
        1.30,
        5,
        1.0,
        True,
        30.0,
    ),
)

EXIT_MODES: tuple[ExitMode, ...] = (
    "time_only",
    "two_x_all",
    "two_x_half_five_all",
    "two_x_recover_five_all",
    "ladder_2_5_10",
    "recover_trail",
    "recover_trail_balanced",
    "recover_trail_tight",
    "trailing_only",
)


@dataclass(frozen=True)
class RuntimeConfig:
    key_value: str
    exit_mode: ExitMode

    @property
    def key(self) -> str:
        return self.key_value


@dataclass(frozen=True)
class S3Signal:
    code: str
    index: int
    minute: int
    close: float
    anchor_low: float
    low_age_minutes: int
    rebound_multiple: float
    volume_ratio: float
    recent_volume: int


@dataclass
class S3Day:
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
    signal_minute: int | None
    anchor_low: float
    low_age_minutes: int
    signal_rebound_multiple: float
    signal_volume_ratio: float
    entry_minute: int | None
    entry_price: float
    entry_quantity: int
    buy_principal: int
    buy_fee: int
    gross_sales: int
    sell_fee: int
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float
    reached_2x: bool
    reached_5x: bool
    reached_10x: bool


@dataclass
class State:
    runtime: RuntimeConfig
    entry: EntryConfig
    cash: int = INITIAL_CASH
    daily: list[S3Day] | None = None
    trades: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []
        if self.trades is None:
            self.trades = []


def volume_ratio(contract: base.ContractGrid, index: int) -> float:
    if index < 20:
        return 0.0
    average = (
        sum(bar.volume for bar in contract.bars[index - 20 : index]) / 20
    )
    if average <= 0:
        return float("inf") if contract.bars[index].volume > 0 else 0.0
    return contract.bars[index].volume / average


def anchor_low(
    contract: base.ContractGrid,
    index: int,
    lookback: int,
) -> tuple[float, int] | None:
    start = max(0, index - lookback)
    candidates = [
        (float(bar.low), position)
        for position, bar in enumerate(
            contract.bars[start:index], start=start
        )
        if bar.traded
        and bar.low is not None
        and ENTRY_PRICE_MIN <= float(bar.low) <= ENTRY_PRICE_MAX
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], -item[1]))


def is_signal(
    contract: base.ContractGrid,
    index: int,
    config: EntryConfig,
) -> S3Signal | None:
    if index < max(20, config.breakout_lookback, config.low_lookback):
        return None
    bar = contract.bars[index]
    if (
        not bar.traded
        or bar.minute > ENTRY_CUTOFF
        or bar.close is None
        or not ENTRY_PRICE_MIN <= float(bar.close) <= ENTRY_PRICE_MAX
    ):
        return None
    anchor = anchor_low(contract, index, config.low_lookback)
    if anchor is None:
        return None
    low, low_index = anchor
    rebound_multiple = float(bar.close) / low
    if (
        rebound_multiple < config.rebound
        or rebound_multiple > MAX_REBOUND
    ):
        return None
    previous_highs = [
        item.high
        for item in contract.bars[
            index - config.breakout_lookback : index
        ]
    ]
    if (
        any(value is None for value in previous_highs)
        or float(bar.close)
        <= max(float(value) for value in previous_highs)
    ):
        return None
    current_volume_ratio = volume_ratio(contract, index)
    if current_volume_ratio <= config.minimum_volume_ratio:
        return None
    if config.require_ma20_rising:
        if (
            index < 5
            or contract.ma20[index] is None
            or contract.ma20[index - 5] is None
            or float(contract.ma20[index])
            <= float(contract.ma20[index - 5])
        ):
            return None
    return S3Signal(
        code=contract.code,
        index=index,
        minute=bar.minute,
        close=float(bar.close),
        anchor_low=low,
        low_age_minutes=index - low_index,
        rebound_multiple=rebound_multiple,
        volume_ratio=current_volume_ratio,
        recent_volume=sum(
            item.volume for item in contract.bars[index - 19 : index + 1]
        ),
    )


def first_signal(
    contracts: dict[str, base.ContractGrid],
    config: EntryConfig,
    previous_vix: float,
) -> S3Signal | None:
    if (
        config.previous_vix_min is not None
        and previous_vix < config.previous_vix_min
    ):
        return None
    for index in range(20, len(base.SESSION_MINUTES)):
        candidates = [
            signal
            for contract in contracts.values()
            if (signal := is_signal(contract, index, config)) is not None
        ]
        if candidates:
            return sorted(
                candidates,
                key=lambda item: (
                    -item.volume_ratio,
                    -item.recent_volume,
                    item.code,
                ),
            )[0]
    return None


def empty_day(
    runtime: RuntimeConfig,
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool,
) -> S3Day:
    return S3Day(
        strategy=runtime.key,
        date=date_value,
        previous_vix=previous_vix,
        start_cash=start_cash,
        end_cash=start_cash,
        pnl=0,
        signal_found=signal_found,
        entry_filled=False,
        code="",
        call_put="",
        signal_minute=None,
        anchor_low=0.0,
        low_age_minutes=0,
        signal_rebound_multiple=0.0,
        signal_volume_ratio=0.0,
        entry_minute=None,
        entry_price=0.0,
        entry_quantity=0,
        buy_principal=0,
        buy_fee=0,
        gross_sales=0,
        sell_fee=0,
        exit_reason="",
        exit_minute=None,
        expired_quantity=0,
        mfe_multiple=0.0,
        mae_pct=0.0,
        reached_2x=False,
        reached_5x=False,
        reached_10x=False,
    )


def recovery_quantity(
    quantity: int,
    close: float,
    buy_principal: int,
    buy_fee: int,
) -> int:
    estimated_net = max(0.0, close - SLIPPAGE) * MULTIPLIER - COMMISSION
    if estimated_net <= 0:
        return quantity
    return min(
        quantity,
        math.ceil((buy_principal + buy_fee) / estimated_net),
    )


def leave_one_half(quantity: int) -> int:
    if quantity <= 1:
        return 0
    return min(quantity - 1, max(1, math.ceil(quantity / 2)))


def full_session_excursion(
    contract: base.ContractGrid,
    entry_index: int,
    entry_price: float,
) -> tuple[float, float]:
    maximum = 1.0
    minimum_move = 0.0
    for bar in contract.bars[entry_index:]:
        if not bar.traded:
            continue
        if bar.high is not None:
            maximum = max(maximum, float(bar.high) / entry_price)
        if bar.low is not None:
            minimum_move = min(
                minimum_move,
                (float(bar.low) / entry_price - 1.0) * 100,
            )
    return maximum, minimum_move


def simulate_day(
    date_value: str,
    previous_vix: float,
    runtime: RuntimeConfig,
    contracts: dict[str, base.ContractGrid],
    signal: S3Signal | None,
    start_cash: int,
) -> tuple[S3Day, list[dict[str, object]]]:
    if signal is None:
        return (
            empty_day(
                runtime,
                date_value,
                previous_vix,
                start_cash,
                signal_found=False,
            ),
            [],
        )
    contract = contracts[signal.code]
    fill = execution.buy(
        date_value,
        runtime,  # type: ignore[arg-type]
        contract,
        signal.index,
        signal.close,
        DAILY_PRINCIPAL_LIMIT,
        start_cash,
        "S3_BREAKOUT_ENTRY",
    )
    if fill is None:
        return (
            empty_day(
                runtime,
                date_value,
                previous_vix,
                start_cash,
                signal_found=True,
            ),
            [],
        )
    entry_index, entry_quantity, buy_principal, buy_fee, buy_trade = fill
    entry_price = float(buy_trade["price"])
    mfe_multiple, mae_pct = full_session_excursion(
        contract, entry_index, entry_price
    )
    quantity = entry_quantity
    gross_sales = 0
    sell_fee = 0
    exit_reason = ""
    exit_minute: int | None = None
    trades = [buy_trade]
    stage_2 = False
    stage_5 = False
    stage_10 = False
    trail_active = False
    high_water_close = entry_price
    index = entry_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        reason = ""
        request = 0
        if bar.traded and bar.close is not None:
            close = float(bar.close)
            multiple = close / entry_price
            high_water_close = max(high_water_close, close)
            mode = runtime.exit_mode

            if mode == "two_x_all" and multiple >= 2.0:
                reason, request = "TARGET_2X_ALL", quantity
            elif mode == "two_x_half_five_all":
                if not stage_2 and multiple >= 2.0:
                    stage_2 = True
                    reason = "TARGET_2X_HALF"
                    request = max(1, math.ceil(quantity / 2))
                elif stage_2 and multiple >= 5.0:
                    reason, request = "TARGET_5X_REMAINDER", quantity
            elif mode == "two_x_recover_five_all":
                if not stage_2 and multiple >= 2.0:
                    stage_2 = True
                    reason = "TARGET_2X_RECOVER"
                    request = recovery_quantity(
                        quantity,
                        close,
                        buy_principal,
                        buy_fee,
                    )
                elif stage_2 and multiple >= 5.0:
                    reason, request = "TARGET_5X_REMAINDER", quantity
            elif mode == "ladder_2_5_10":
                if not stage_2 and multiple >= 2.0:
                    stage_2 = True
                    reason = "LADDER_2X"
                    request = leave_one_half(quantity)
                elif stage_2 and not stage_5 and multiple >= 5.0:
                    stage_5 = True
                    reason = "LADDER_5X"
                    request = leave_one_half(quantity)
                elif stage_5 and multiple >= 10.0:
                    stage_10 = True
                    reason, request = "LADDER_10X_ALL", quantity
            elif mode in {
                "recover_trail",
                "recover_trail_balanced",
                "recover_trail_tight",
            }:
                if not stage_2 and multiple >= 2.0:
                    stage_2 = True
                    trail_active = True
                    reason = "TRAIL_2X_RECOVER"
                    request = recovery_quantity(
                        quantity,
                        close,
                        buy_principal,
                        buy_fee,
                    )
                elif trail_active:
                    high_multiple = high_water_close / entry_price
                    if mode == "recover_trail_tight":
                        drawdown = (
                            0.15 if high_multiple >= 10.0 else 0.20
                        )
                    elif mode == "recover_trail_balanced":
                        drawdown = (
                            0.20
                            if high_multiple >= 10.0
                            else 0.25
                            if high_multiple >= 5.0
                            else 0.30
                        )
                    else:
                        drawdown = (
                            0.20
                            if high_multiple >= 10.0
                            else 0.30
                            if high_multiple >= 5.0
                            else 0.40
                        )
                    if close <= high_water_close * (1.0 - drawdown):
                        reason = "DYNAMIC_TRAIL"
                        request = quantity
            elif mode == "trailing_only":
                if not trail_active and multiple >= 2.0:
                    trail_active = True
                if trail_active:
                    high_multiple = high_water_close / entry_price
                    drawdown = (
                        0.20
                        if high_multiple >= 10.0
                        else 0.30
                        if high_multiple >= 5.0
                        else 0.40
                    )
                    if close <= high_water_close * (1.0 - drawdown):
                        reason = "DYNAMIC_TRAIL_ALL"
                        request = quantity

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason, request = "TIME", quantity
        if not reason or request <= 0:
            index += 1
            continue

        filled, sales, fees, fill_index, sell_trades = execution.sell_all(
            date_value,
            runtime,  # type: ignore[arg-type]
            contract,
            index,
            request,
            reason,
        )
        quantity -= filled
        gross_sales += sales
        sell_fee += fees
        trades.extend(sell_trades)
        if filled < request:
            exit_reason = "UNFILLED_ZERO_VALUE"
            exit_minute = base.SESSION_MINUTES[-1]
            break
        if quantity <= 0:
            exit_reason = reason
            exit_minute = contract.bars[fill_index].minute
            break
        index = fill_index

    expired_quantity = quantity
    if expired_quantity > 0:
        trades.append(
            {
                "strategy": runtime.key,
                "date": date_value,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
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

    pnl = gross_sales - sell_fee - buy_principal - buy_fee
    return (
        S3Day(
            strategy=runtime.key,
            date=date_value,
            previous_vix=previous_vix,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=contract.code,
            call_put=contract.call_put,
            signal_minute=signal.minute,
            anchor_low=round(signal.anchor_low, 4),
            low_age_minutes=signal.low_age_minutes,
            signal_rebound_multiple=round(
                signal.rebound_multiple, 6
            ),
            signal_volume_ratio=round(signal.volume_ratio, 6),
            entry_minute=int(buy_trade["minute"]),
            entry_price=entry_price,
            entry_quantity=entry_quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_reason=exit_reason,
            exit_minute=exit_minute,
            expired_quantity=expired_quantity,
            mfe_multiple=round(mfe_multiple, 6),
            mae_pct=round(mae_pct, 4),
            reached_2x=mfe_multiple >= 2.0,
            reached_5x=mfe_multiple >= 5.0,
            reached_10x=mfe_multiple >= 10.0,
        ),
        trades,
    )


def max_drawdown(rows: list[S3Day]) -> int:
    equity = INITIAL_CASH
    peak = equity
    worst = 0
    for row in rows:
        equity += row.pnl
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def summarize(rows: list[S3Day]) -> dict[str, object]:
    trades = [row for row in rows if row.entry_filled]
    profits = [row.pnl for row in trades]
    positives = sum(value for value in profits if value > 0)
    negatives = -sum(value for value in profits if value < 0)
    sorted_profits = sorted(profits, reverse=True)
    return {
        "days": len(rows),
        "signal_days": sum(row.signal_found for row in rows),
        "trade_days": len(trades),
        "profitable_days": sum(value > 0 for value in profits),
        "losing_days": sum(value < 0 for value in profits),
        "total_pnl": sum(profits),
        "average_trade_day_pnl": (
            round(sum(profits) / len(profits)) if profits else 0
        ),
        "profit_factor": (
            round(positives / negatives, 4) if negatives else None
        ),
        "minimum_day_pnl": min(profits, default=0),
        "maximum_day_pnl": max(profits, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": (
            sum(profits) - sum(sorted_profits[:3])
        ),
        "pnl_excluding_best_5_days": (
            sum(profits) - sum(sorted_profits[:5])
        ),
        "total_buy_principal": sum(
            row.buy_principal for row in trades
        ),
        "expired_contracts": sum(
            row.expired_quantity for row in trades
        ),
        "entry_reached_2x_days": sum(row.reached_2x for row in trades),
        "entry_reached_5x_days": sum(row.reached_5x for row in trades),
        "entry_reached_10x_days": sum(
            row.reached_10x for row in trades
        ),
    }


def periods(rows: list[S3Day]) -> dict[str, dict[str, object]]:
    development = [row for row in rows if row.date <= DEVELOPMENT_END]
    validation = [row for row in rows if row.date > DEVELOPMENT_END]
    return {
        "all": summarize(rows),
        "development": summarize(development),
        "validation": summarize(validation),
    }


def run_state_day(
    state: State,
    date_value: str,
    previous_vix: float,
    contracts: dict[str, base.ContractGrid],
    signal: S3Signal | None,
) -> None:
    row, trades = simulate_day(
        date_value,
        previous_vix,
        state.runtime,
        contracts,
        signal,
        state.cash,
    )
    state.cash = row.end_cash
    assert state.daily is not None
    assert state.trades is not None
    state.daily.append(row)
    state.trades.extend(trades)


def audit_state(state: State) -> dict[str, object]:
    assert state.daily is not None
    assert state.trades is not None
    return audit(state.daily, state.trades)  # type: ignore[arg-type]


def state_document(state: State) -> dict[str, object]:
    assert state.daily is not None
    return {
        "strategy": state.runtime.key,
        "entry": asdict(state.entry),
        "exit_mode": state.runtime.exit_mode,
        **periods(state.daily),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_entry(candidates: list[dict[str, object]]) -> str:
    eligible = [
        row
        for row in candidates
        if int(row["development"]["trade_days"]) >= 15  # type: ignore[index]
    ]
    if not eligible:
        raise RuntimeError("개발 구간 15거래 이상 진입 후보가 없습니다.")
    chosen = max(
        eligible,
        key=lambda row: (
            int(
                row["development"]["pnl_excluding_best_3_days"]  # type: ignore[index]
            ),
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(row["development"]["entry_reached_5x_days"]),  # type: ignore[index]
        ),
    )
    return str(chosen["entry"]["name"])  # type: ignore[index]


def selected_exit(candidates: list[dict[str, object]]) -> str:
    chosen = max(
        candidates,
        key=lambda row: (
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(
                row["development"]["pnl_excluding_best_3_days"]  # type: ignore[index]
            ),
            int(row["development"]["max_drawdown"]),  # type: ignore[index]
        ),
    )
    return str(chosen["exit_mode"])


def same_entry_audit(states: list[State]) -> dict[str, object]:
    comparable = [
        state
        for state in states
        if state.runtime.exit_mode != "time_only"
    ]
    reference_state = next(
        state
        for state in comparable
        if state.runtime.exit_mode == "two_x_all"
    )
    reference = reference_state.daily or []
    reference_entries = {
        row.date: (row.code, row.entry_minute, row.entry_quantity)
        for row in reference
        if row.entry_filled
    }
    mismatches: list[dict[str, object]] = []
    for state in comparable:
        if state is reference_state:
            continue
        entries = {
            row.date: (row.code, row.entry_minute, row.entry_quantity)
            for row in (state.daily or [])
            if row.entry_filled
        }
        if entries != reference_entries:
            different_dates = sorted(
                date
                for date in set(reference_entries) | set(entries)
                if reference_entries.get(date) != entries.get(date)
            )
            mismatches.append(
                {
                    "exit_mode": state.runtime.exit_mode,
                    "different_dates": different_dates,
                }
            )
    return {
        "passed": not mismatches,
        "reference_exit_mode": reference_state.runtime.exit_mode,
        "entry_days": len(reference_entries),
        "mismatches": mismatches,
        "excluded_controls": [
            {
                "exit_mode": "time_only",
                "reason": (
                    "continuous account exhausted before period end, "
                    "so later fills are not comparable"
                ),
            }
        ],
    }


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= HIGH_VOL_START
    ]
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()

    entry_states = [
        State(
            runtime=RuntimeConfig(
                f"S3-entry|{config.name}|two_x_all",
                "two_x_all",
            ),
            entry=config,
        )
        for config in ENTRY_CONFIGS
    ]
    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        for state in entry_states:
            signal = first_signal(
                contracts, state.entry, previous_vix
            )
            run_state_day(
                state,
                date_value,
                previous_vix,
                contracts,
                signal,
            )

    entry_candidates = [state_document(state) for state in entry_states]
    entry_choice_name = selected_entry(entry_candidates)
    entry_choice = next(
        config for config in ENTRY_CONFIGS if config.name == entry_choice_name
    )

    exit_states = [
        State(
            runtime=RuntimeConfig(
                f"S3-exit|{entry_choice.name}|{exit_mode}",
                exit_mode,
            ),
            entry=entry_choice,
        )
        for exit_mode in EXIT_MODES
    ]
    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        signal = first_signal(contracts, entry_choice, previous_vix)
        for state in exit_states:
            run_state_day(
                state,
                date_value,
                previous_vix,
                contracts,
                signal,
            )

    exit_candidates = [state_document(state) for state in exit_states]
    exit_choice_name = selected_exit(exit_candidates)
    fixed_state = next(
        state
        for state in exit_states
        if state.runtime.exit_mode == exit_choice_name
    )
    fixed_daily = [
        S3Day(**{**asdict(row), "strategy": "S3-v0.1"})
        for row in (fixed_state.daily or [])
    ]
    fixed_trades = [
        {**trade, "strategy": "S3-v0.1"}
        for trade in (fixed_state.trades or [])
    ]
    fixed_audit = audit(
        fixed_daily, fixed_trades  # type: ignore[arg-type]
    )
    entry_audits = {
        state.entry.name: audit_state(state) for state in entry_states
    }
    exit_audits = {
        state.runtime.exit_mode: audit_state(state)
        for state in exit_states
    }
    same_entries = same_entry_audit(exit_states)
    all_passed = (
        all(item["passed"] for item in entry_audits.values())
        and all(item["passed"] for item in exit_audits.values())
        and fixed_audit["passed"]
        and same_entries["passed"]
    )

    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": "S3-v0.1",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": DEVELOPMENT_END,
            "validation_used_for_selection": False,
            "historical_low_volatility_backcast": False,
        },
        "event_study_source": "s3_breakout_event_summary.json",
        "separation": {
            "entry_stage_exit_fixed_to": "two_x_all",
            "entry_selection": (
                "development pnl excluding best 3 days, minimum 15 "
                "trades"
            ),
            "exit_stage_entry_fixed_to": entry_choice.name,
            "exit_selection": "development total pnl",
        },
        "rules": {
            "selected_entry": asdict(entry_choice),
            "selected_exit": exit_choice_name,
            "entry_price_range": [
                ENTRY_PRICE_MIN,
                ENTRY_PRICE_MAX,
            ],
            "maximum_signal_rebound": MAX_REBOUND,
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "entry_execution": (
                "next actual bar within 5 minutes, 10% participation, "
                "0.01 slippage"
            ),
            "exit_execution": (
                "close signal then later actual bars, 10% participation, "
                "0.01 slippage"
            ),
            "dynamic_trail": {
                "wide": "40% / 30% / 20% after 2x / 5x / 10x",
                "balanced": "30% / 25% / 20%",
                "tight": "20% / 20% / 15%",
            },
        },
        "summary": periods(fixed_daily)["all"],
        "development": periods(fixed_daily)["development"],
        "validation": periods(fixed_daily)["validation"],
        "entry_candidates": entry_candidates,
        "exit_candidates": exit_candidates,
        "audit": {
            "passed": all_passed,
            "fixed": fixed_audit,
            "same_entries_across_exit_candidates": same_entries,
            "entry_candidates_passed": all(
                item["passed"] for item in entry_audits.values()
            ),
            "exit_candidates_passed": all(
                item["passed"] for item in exit_audits.values()
            ),
        },
        "limitations": [
            "진입 후보와 청산 후보 모두 개발 구간 비교의 영향을 받았습니다.",
            "후반 검증은 선택에 사용하지 않았지만 완전한 신규 만기일 전진검증은 아닙니다.",
            "분봉 OHLC 내부 경로를 가정하지 않기 위해 신호는 종가, 체결은 다음 실제 봉 시가를 사용했습니다.",
            "미체결 만기 잔량은 0원으로 평가했습니다.",
        ],
    }

    write_csv(
        OUTPUT_DIR / "strategy3_entry_research_daily.csv",
        [
            asdict(row)
            for state in entry_states
            for row in (state.daily or [])
        ],
    )
    write_csv(
        OUTPUT_DIR / "strategy3_entry_research_trades.csv",
        [
            trade
            for state in entry_states
            for trade in (state.trades or [])
        ],
    )
    write_csv(
        OUTPUT_DIR / "strategy3_exit_research_daily.csv",
        [
            asdict(row)
            for state in exit_states
            for row in (state.daily or [])
        ],
    )
    write_csv(
        OUTPUT_DIR / "strategy3_exit_research_trades.csv",
        [
            trade
            for state in exit_states
            for trade in (state.trades or [])
        ],
    )
    write_csv(
        OUTPUT_DIR / "strategy3_daily.csv",
        [asdict(row) for row in fixed_daily],
    )
    write_csv(OUTPUT_DIR / "strategy3_trades.csv", fixed_trades)
    (OUTPUT_DIR / "strategy3_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy3_audit.json").write_text(
        json.dumps(result["audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
