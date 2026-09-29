from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from count_opportunities import (
    DAY_MARKET_ID,
    PROFILE_PATH,
    derive_expiry_groups,
    extract_series,
    product_kind,
    target_file_map,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

INITIAL_CASH = 80_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
MULTIPLIER = 250_000
COMMISSION_PER_CONTRACT = 500
SLIPPAGE = 0.01
ENTRY_PARTICIPATION = 0.10
EXIT_PARTICIPATION = 0.10
ENTRY_WAIT_MINUTES = 5
ENTRY_CUTOFF = 1430
FORCE_EXIT_TIME = 1515

ExitMode = Literal["time_only", "two_x_all", "runner"]
LiquidityMode = Literal["next_bar", "volume_10pct"]
EntryMode = Literal["technical", "fixed_0930"]


@dataclass(frozen=True)
class PremiumBand:
    name: str
    lower: float
    upper: float

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


PREMIUM_BANDS = (
    PremiumBand("1.00-5.00", 1.00, 5.00),
    PremiumBand("0.05-0.49", 0.05, 0.49),
    PremiumBand("0.50-0.99", 0.50, 0.99),
    PremiumBand("1.00-1.49", 1.00, 1.49),
    PremiumBand("1.50-2.49", 1.50, 2.49),
    PremiumBand("2.50-5.00", 2.50, 5.00),
    PremiumBand("5.01-10.00", 5.01, 10.00),
)
EXIT_MODES: tuple[ExitMode, ...] = (
    "time_only",
    "two_x_all",
    "runner",
)
LIQUIDITY_MODES: tuple[LiquidityMode, ...] = (
    "next_bar",
    "volume_10pct",
)


@dataclass(frozen=True)
class RawBar:
    minute: int
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class MinuteBar:
    minute: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int
    traded: bool


@dataclass
class ContractGrid:
    code: str
    name: str
    call_put: str
    bars: list[MinuteBar]
    ma5: list[float | None]
    ma20: list[float | None]


@dataclass(frozen=True)
class Signal:
    code: str
    index: int
    minute: int
    close: float
    recent_volume: int


@dataclass(frozen=True)
class Config:
    premium: PremiumBand
    exit_mode: ExitMode
    liquidity_mode: LiquidityMode
    entry_mode: EntryMode = "technical"

    @property
    def key(self) -> str:
        return (
            f"{self.premium.name}|{self.entry_mode}|"
            f"{self.exit_mode}|{self.liquidity_mode}"
        )


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
    signal_minute: int | None
    entry_minute: int | None
    entry_quantity: int
    buy_principal: int
    buy_fee: int
    gross_sales: int
    sell_fee: int
    expired_quantity: int


@dataclass
class PositionResult:
    gross_sales: int
    sell_fee: int
    expired_quantity: int
    trades: list[dict[str, object]]


@dataclass
class State:
    config: Config
    cash: int = INITIAL_CASH
    daily: list[DayResult] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


def minute_values() -> list[int]:
    values: list[int] = []
    for total in range(8 * 60 + 45, 15 * 60 + 20):
        hour, minute = divmod(total, 60)
        values.append(hour * 100 + minute)
    return values


SESSION_MINUTES = minute_values()
MINUTE_INDEX = {minute: index for index, minute in enumerate(SESSION_MINUTES)}


def rolling_mean(
    values: list[float | None], period: int
) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        if any(value is None for value in window):
            continue
        result[index] = sum(float(value) for value in window) / period
    return result


def build_contract(
    code: str,
    metadata: dict[str, str],
    raw_by_minute: dict[int, RawBar],
) -> ContractGrid:
    bars: list[MinuteBar] = []
    previous_close: float | None = None
    for minute in SESSION_MINUTES:
        raw = raw_by_minute.get(minute)
        if raw is not None:
            previous_close = raw.close
            bars.append(
                MinuteBar(
                    minute=minute,
                    open=raw.open,
                    high=raw.high,
                    low=raw.low,
                    close=raw.close,
                    volume=raw.volume,
                    traded=True,
                )
            )
        elif previous_close is None:
            bars.append(
                MinuteBar(minute, None, None, None, None, 0, False)
            )
        else:
            bars.append(
                MinuteBar(
                    minute,
                    previous_close,
                    previous_close,
                    previous_close,
                    previous_close,
                    0,
                    False,
                )
            )
    closes = [bar.close for bar in bars]
    return ContractGrid(
        code=code,
        name=metadata["name"],
        call_put=metadata["call_put"],
        bars=bars,
        ma5=rolling_mean(closes, 5),
        ma20=rolling_mean(closes, 20),
    )


def load_expiry_day(
    path: Path, group: dict[str, str]
) -> dict[str, ContractGrid]:
    raw_by_code: dict[str, dict[int, RawBar]] = defaultdict(dict)
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="cp949", newline="") as source:
        for row in csv.DictReader(source):
            if row["시장ID"] != DAY_MARKET_ID:
                continue
            if product_kind(row["종목명"]) != group["product_kind"]:
                continue
            if extract_series(row["종목명"]) != group["series"]:
                continue
            try:
                minute = int(row["기준시각"].strip().zfill(4))
                raw = RawBar(
                    minute=minute,
                    open=float(row["시가"]),
                    high=float(row["고가"]),
                    low=float(row["저가"]),
                    close=float(row["종가"]),
                    volume=int(float(row["거래량"])),
                )
            except (TypeError, ValueError):
                continue
            if minute not in MINUTE_INDEX:
                continue
            code = row["종목코드"]
            existing = raw_by_code[code].get(minute)
            if existing is None:
                raw_by_code[code][minute] = raw
            else:
                raw_by_code[code][minute] = RawBar(
                    minute=minute,
                    open=existing.open,
                    high=max(existing.high, raw.high),
                    low=min(existing.low, raw.low),
                    close=raw.close,
                    volume=existing.volume + raw.volume,
                )
            metadata.setdefault(
                code,
                {
                    "name": row["종목명"],
                    "call_put": row["콜풋구분"],
                },
            )
    return {
        code: build_contract(code, metadata[code], raw)
        for code, raw in raw_by_code.items()
        if len(raw) >= 2
    }


def is_signal(contract: ContractGrid, index: int, band: PremiumBand) -> bool:
    if index < 20:
        return False
    bar = contract.bars[index]
    previous = contract.bars[index - 1]
    ma5 = contract.ma5[index]
    previous_ma5 = contract.ma5[index - 1]
    ma20 = contract.ma20[index]
    if (
        not bar.traded
        or bar.minute > ENTRY_CUTOFF
        or bar.close is None
        or previous.close is None
        or ma5 is None
        or previous_ma5 is None
        or ma20 is None
        or not band.contains(bar.close)
    ):
        return False
    previous_highs = [
        contract.bars[position].high
        for position in range(index - 5, index)
    ]
    if any(value is None for value in previous_highs):
        return False
    previous_volumes = [
        contract.bars[position].volume
        for position in range(index - 20, index)
    ]
    average_volume = sum(previous_volumes) / 20
    return (
        previous.close <= previous_ma5
        and bar.close > ma5
        and ma5 > ma20
        and bar.close > max(float(value) for value in previous_highs)
        and bar.volume > average_volume
    )


def first_signal(
    contracts: dict[str, ContractGrid], band: PremiumBand
) -> Signal | None:
    for index in range(20, len(SESSION_MINUTES)):
        candidates: list[Signal] = []
        for contract in contracts.values():
            if not is_signal(contract, index, band):
                continue
            close = contract.bars[index].close
            assert close is not None
            recent_volume = sum(
                bar.volume for bar in contract.bars[index - 19 : index + 1]
            )
            candidates.append(
                Signal(
                    code=contract.code,
                    index=index,
                    minute=contract.bars[index].minute,
                    close=close,
                    recent_volume=recent_volume,
                )
            )
        if candidates:
            return sorted(
                candidates,
                key=lambda item: (-item.recent_volume, item.code),
            )[0]
    return None


def fixed_time_signal(
    contracts: dict[str, ContractGrid],
    band: PremiumBand,
    minute: int = 930,
) -> Signal | None:
    index = MINUTE_INDEX[minute]
    candidates: list[Signal] = []
    for contract in contracts.values():
        bar = contract.bars[index]
        if (
            not bar.traded
            or bar.close is None
            or not band.contains(bar.close)
        ):
            continue
        recent_volume = sum(
            item.volume for item in contract.bars[index - 19 : index + 1]
        )
        candidates.append(
            Signal(
                code=contract.code,
                index=index,
                minute=minute,
                close=bar.close,
                recent_volume=recent_volume,
            )
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (-item.recent_volume, item.code),
    )[0]


def next_traded_bar(
    contract: ContractGrid, signal_index: int, max_wait: int | None
) -> tuple[int, MinuteBar] | None:
    last_index = len(contract.bars) - 1
    if max_wait is not None:
        last_index = min(last_index, signal_index + max_wait)
    for index in range(signal_index + 1, last_index + 1):
        if contract.bars[index].traded:
            return index, contract.bars[index]
    return None


def entry_fill(
    contract: ContractGrid,
    signal: Signal,
    available_cash: int,
) -> tuple[int, int, MinuteBar, float] | None:
    next_bar_result = next_traded_bar(
        contract, signal.index, ENTRY_WAIT_MINUTES
    )
    if next_bar_result is None:
        return None
    fill_index, bar = next_bar_result
    assert bar.open is not None
    fill_price = bar.open + SLIPPAGE
    requested_quantity = math.floor(
        DAILY_PRINCIPAL_LIMIT
        / ((signal.close + SLIPPAGE) * MULTIPLIER)
    )
    by_actual_principal = math.floor(
        DAILY_PRINCIPAL_LIMIT / (fill_price * MULTIPLIER)
    )
    by_cash = math.floor(
        available_cash
        / (fill_price * MULTIPLIER + 2 * COMMISSION_PER_CONTRACT)
    )
    by_volume = math.floor(bar.volume * ENTRY_PARTICIPATION)
    quantity = min(
        requested_quantity, by_actual_principal, by_cash, by_volume
    )
    if quantity <= 0:
        return None
    return fill_index, quantity, bar, fill_price


def execute_sell(
    date: str,
    config: Config,
    contract: ContractGrid,
    signal_index: int,
    requested_quantity: int,
    reason: str,
) -> tuple[int, int, int, int, list[dict[str, object]]]:
    remaining = requested_quantity
    gross_sales = 0
    fees = 0
    last_fill_index = signal_index
    capacity = 0.0
    trades: list[dict[str, object]] = []
    for index in range(signal_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if not bar.traded:
            continue
        assert bar.open is not None
        if config.liquidity_mode == "next_bar":
            fillable = remaining
        else:
            capacity += bar.volume * EXIT_PARTICIPATION
            fillable = min(remaining, math.floor(capacity + 1e-12))
        if fillable <= 0:
            continue
        if config.liquidity_mode == "volume_10pct":
            capacity -= fillable
        fill_price = max(0.0, bar.open - SLIPPAGE)
        gross = round(fillable * fill_price * MULTIPLIER)
        fee = fillable * COMMISSION_PER_CONTRACT
        gross_sales += gross
        fees += fee
        remaining -= fillable
        last_fill_index = index
        trades.append(
            {
                "strategy": config.key,
                "date": date,
                "minute": bar.minute,
                "code": contract.code,
                "side": "SELL",
                "reason": reason,
                "quantity": fillable,
                "price": fill_price,
                "principal": gross,
                "fee": fee,
                "cash_flow": gross - fee,
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


def simulate_position(
    date: str,
    config: Config,
    contract: ContractGrid,
    entry_index: int,
    entry_quantity: int,
    entry_price: float,
) -> PositionResult:
    quantity = entry_quantity
    first_target_done = False
    second_target_done = False
    high_water = entry_price
    gross_sales = 0
    sell_fee = 0
    trades: list[dict[str, object]] = []
    index = entry_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        if bar.traded and bar.close is not None:
            high_water = max(high_water, bar.close)
            reason: str | None = None
            request = 0
            if config.exit_mode == "two_x_all" and bar.close >= entry_price * 2:
                reason = "TWO_X_ALL"
                request = quantity
            elif config.exit_mode == "runner":
                if not first_target_done and bar.close >= entry_price * 2:
                    reason = "TWO_X_HALF"
                    request = max(1, math.ceil(quantity / 2))
                    first_target_done = True
                elif (
                    first_target_done
                    and not second_target_done
                    and bar.close >= entry_price * 5
                ):
                    second_target_done = True
                    if quantity > 1:
                        reason = "FIVE_X_PARTIAL"
                        request = min(
                            quantity - 1, max(1, math.ceil(quantity / 2))
                        )
                elif (
                    first_target_done
                    and bar.close <= high_water * 0.70
                ):
                    reason = "TRAIL_30_PERCENT"
                    request = quantity
            if reason is not None and request > 0:
                (
                    filled,
                    gross,
                    fees,
                    fill_index,
                    fill_trades,
                ) = execute_sell(
                    date,
                    config,
                    contract,
                    index,
                    request,
                    reason,
                )
                quantity -= filled
                gross_sales += gross
                sell_fee += fees
                trades.extend(fill_trades)
                if filled < request:
                    break
                index = fill_index
                continue

        if bar.minute >= FORCE_EXIT_TIME:
            (
                filled,
                gross,
                fees,
                fill_index,
                fill_trades,
            ) = execute_sell(
                date,
                config,
                contract,
                index,
                quantity,
                "TIME",
            )
            quantity -= filled
            gross_sales += gross
            sell_fee += fees
            trades.extend(fill_trades)
            index = fill_index
            break
        index += 1

    if quantity > 0:
        trades.append(
            {
                "strategy": config.key,
                "date": date,
                "minute": SESSION_MINUTES[-1],
                "code": contract.code,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "quantity": quantity,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
    return PositionResult(
        gross_sales=gross_sales,
        sell_fee=sell_fee,
        expired_quantity=quantity,
        trades=trades,
    )


def simulate_day(
    date: str,
    config: Config,
    contracts: dict[str, ContractGrid],
    signal: Signal | None,
    start_cash: int,
) -> tuple[DayResult, list[dict[str, object]]]:
    if signal is None:
        return (
            DayResult(
                config.key,
                date,
                start_cash,
                start_cash,
                0,
                False,
                False,
                "",
                None,
                None,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            [],
        )
    contract = contracts[signal.code]
    fill = entry_fill(contract, signal, start_cash)
    if fill is None:
        return (
            DayResult(
                config.key,
                date,
                start_cash,
                start_cash,
                0,
                True,
                False,
                signal.code,
                signal.minute,
                None,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            [],
        )

    entry_index, quantity, entry_bar, entry_price = fill
    buy_principal = round(quantity * entry_price * MULTIPLIER)
    buy_fee = quantity * COMMISSION_PER_CONTRACT
    buy_trade = {
        "strategy": config.key,
        "date": date,
        "minute": entry_bar.minute,
        "code": contract.code,
        "side": "BUY",
        "reason": "FIRST_AND_SIGNAL",
        "quantity": quantity,
        "price": entry_price,
        "principal": buy_principal,
        "fee": buy_fee,
        "cash_flow": -(buy_principal + buy_fee),
        "bar_volume": entry_bar.volume,
    }
    position = simulate_position(
        date,
        config,
        contract,
        entry_index,
        quantity,
        entry_price,
    )
    pnl = (
        position.gross_sales
        - position.sell_fee
        - buy_principal
        - buy_fee
    )
    end_cash = start_cash + pnl
    return (
        DayResult(
            strategy=config.key,
            date=date,
            start_cash=start_cash,
            end_cash=end_cash,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=signal.code,
            signal_minute=signal.minute,
            entry_minute=entry_bar.minute,
            entry_quantity=quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            gross_sales=position.gross_sales,
            sell_fee=position.sell_fee,
            expired_quantity=position.expired_quantity,
        ),
        [buy_trade, *position.trades],
    )


def max_drawdown(values: list[int]) -> int:
    peak = values[0]
    worst = 0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def summarize(state: State) -> dict[str, object]:
    assert state.daily is not None
    profits = [day.pnl for day in state.daily]
    sorted_profits = sorted(profits, reverse=True)
    values = [INITIAL_CASH] + [day.end_cash for day in state.daily]
    late = [day.pnl for day in state.daily if day.date >= "20260501"]
    return {
        "strategy": state.config.key,
        "premium_range": state.config.premium.name,
        "entry_mode": state.config.entry_mode,
        "exit_mode": state.config.exit_mode,
        "liquidity_mode": state.config.liquidity_mode,
        "expiry_days": len(state.daily),
        "signal_days": sum(day.signal_found for day in state.daily),
        "trade_days": sum(day.entry_filled for day in state.daily),
        "profitable_days": sum(value > 0 for value in profits),
        "losing_days": sum(value < 0 for value in profits),
        "total_pnl": sum(profits),
        "ending_cash": state.cash,
        "maximum_day_pnl": max(profits),
        "minimum_day_pnl": min(profits),
        "max_drawdown": max_drawdown(values),
        "pnl_excluding_best_1_day": sum(profits) - sum(sorted_profits[:1]),
        "pnl_excluding_best_3_days": sum(profits) - sum(sorted_profits[:3]),
        "pnl_excluding_best_5_days": sum(profits) - sum(sorted_profits[:5]),
        "pnl_2026_05_to_08": sum(late),
        "days_at_least_2_5m_profit": sum(
            value >= 2_500_000 for value in profits
        ),
        "total_buy_principal": sum(
            day.buy_principal for day in state.daily
        ),
        "total_fees": sum(
            day.buy_fee + day.sell_fee for day in state.daily
        ),
        "expired_contracts": sum(
            day.expired_quantity for day in state.daily
        ),
        "expiry_residual_days": sum(
            day.expired_quantity > 0 for day in state.daily
        ),
        "maximum_daily_principal": max(
            day.buy_principal for day in state.daily
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(
    daily_rows: list[dict[str, object]],
    trades: list[dict[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for trade in trades:
        trades_by_key[(str(trade["strategy"]), str(trade["date"]))].append(
            trade
        )

    previous_end: dict[str, int] = {}
    for row in daily_rows:
        strategy = str(row["strategy"])
        date = str(row["date"])
        start_cash = int(row["start_cash"])
        end_cash = int(row["end_cash"])
        pnl = int(row["pnl"])
        principal = int(row["buy_principal"])
        if principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{strategy} {date}: principal limit")
        expected_start = previous_end.get(strategy, INITIAL_CASH)
        if start_cash != expected_start:
            errors.append(f"{strategy} {date}: cash chain")
        if end_cash != start_cash + pnl:
            errors.append(f"{strategy} {date}: pnl equation")

        day_trades = trades_by_key[(strategy, date)]
        cash_flow = sum(int(trade["cash_flow"]) for trade in day_trades)
        if cash_flow != pnl:
            errors.append(f"{strategy} {date}: cash flow")
        quantity = 0
        for trade in day_trades:
            side = str(trade["side"])
            trade_quantity = int(trade["quantity"])
            if side == "BUY":
                quantity += trade_quantity
                if trade_quantity > math.floor(
                    int(trade["bar_volume"]) * ENTRY_PARTICIPATION
                ):
                    errors.append(f"{strategy} {date}: entry participation")
            elif side in {"SELL", "EXPIRE"}:
                quantity -= trade_quantity
        if quantity != 0:
            errors.append(f"{strategy} {date}: open quantity {quantity}")
        previous_end[strategy] = end_cash

    return {
        "passed": not errors,
        "daily_rows": len(daily_rows),
        "trade_rows": len(trades),
        "error_count": len(errors),
        "error_samples": errors[:20],
    }


def run() -> dict[str, object]:
    configs = [
        Config(
            premium=band,
            exit_mode=exit_mode,
            liquidity_mode=liquidity_mode,
        )
        for band in PREMIUM_BANDS
        for exit_mode in EXIT_MODES
        for liquidity_mode in LIQUIDITY_MODES
    ]
    configs.append(
        Config(
            premium=PREMIUM_BANDS[0],
            entry_mode="fixed_0930",
            exit_mode="time_only",
            liquidity_mode="next_bar",
        )
    )
    states = {config.key: State(config) for config in configs}
    all_daily: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = target_file_map()

    for group in groups:
        date = group["expiry_date"]
        contracts = load_expiry_day(files[date], group)
        signal_cache = {
            band.name: first_signal(contracts, band)
            for band in PREMIUM_BANDS
        }
        fixed_signal = fixed_time_signal(contracts, PREMIUM_BANDS[0])
        for state in states.values():
            signal = (
                fixed_signal
                if state.config.entry_mode == "fixed_0930"
                else signal_cache[state.config.premium.name]
            )
            day, trades = simulate_day(
                date,
                state.config,
                contracts,
                signal,
                state.cash,
            )
            assert state.daily is not None
            state.daily.append(day)
            state.cash = day.end_cash
            all_daily.append(asdict(day))
            all_trades.extend(trades)

    summaries = [summarize(state) for state in states.values()]
    audit_result = audit(all_daily, all_trades)
    result = {
        "status": "minimal_fixed_rules_baseline",
        "not_an_optimized_strategy": True,
        "analysis_period": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
        },
        "no_trade_baseline": {
            "initial_cash": INITIAL_CASH,
            "ending_cash": INITIAL_CASH,
            "total_pnl": 0,
        },
        "rules": {
            "signal": "same completed minute: MA5 recovery AND MA5>MA20 AND previous-5-minute-high breakout AND volume above prior-20-minute average",
            "selection": "first signal only; tie by recent 20-minute volume then code",
            "entry": "next actual traded bar within 5 minutes; no fallback candidate or re-entry",
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "compounding": False,
            "minute_grid": "missing minutes forward-filled close with zero volume",
            "entry_participation": ENTRY_PARTICIPATION,
            "exit_liquidity_modes": {
                "next_bar": "entire order at next actual bar open",
                "volume_10pct": EXIT_PARTICIPATION,
            },
            "unfilled_expiry_value": 0,
        },
        "execution": {
            "slippage_per_side": SLIPPAGE,
            "commission_per_contract_per_side": COMMISSION_PER_CONTRACT,
            "multiplier": MULTIPLIER,
        },
        "summaries": summaries,
        "audit": audit_result,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "minimal_baseline_grid.csv", summaries)
    write_csv(OUTPUT_DIR / "minimal_baseline_daily.csv", all_daily)
    write_csv(OUTPUT_DIR / "minimal_baseline_trades.csv", all_trades)
    (OUTPUT_DIR / "minimal_baseline_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "minimal_baseline_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
