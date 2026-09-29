"""Exploratory grid only.

This file intentionally preserves the first 81-combination experiment.  Its
OR signals, add-on entries, trade-bar moving averages, and full-period model
selection make it ineligible as validation evidence.  Use
minimal_baseline_backtest.py for the corrected simple baselines.
"""

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

INITIAL_ACCOUNT = 80_000_000
DAILY_BUY_LIMIT = 5_000_000
CONTRACT_MULTIPLIER = 250_000
COMMISSION_PER_CONTRACT = 500
SLIPPAGE = 0.01
ENTRY_VOLUME_PARTICIPATION = 0.10
EXTRA_EXECUTION_BARS = 0
FIRST_ENTRY_BUDGET = 2_500_000
NEW_ENTRY_CUTOFF = 1430
FORCE_EXIT_SIGNAL_TIME = 1515

SignalMode = Literal["recovery", "breakout", "either"]


@dataclass(frozen=True)
class PremiumRange:
    name: str
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


PREMIUM_RANGES = (
    PremiumRange("0.01-0.09", 0.01, 0.09),
    PremiumRange("0.10-0.24", 0.10, 0.24),
    PremiumRange("0.25-0.49", 0.25, 0.49),
    PremiumRange("0.50-0.74", 0.50, 0.74),
    PremiumRange("0.75-0.99", 0.75, 0.99),
    PremiumRange("1.00-1.49", 1.00, 1.49),
    PremiumRange("1.50-2.49", 1.50, 2.49),
    PremiumRange("2.50-5.00", 2.50, 5.00),
    PremiumRange("5.01-10.00", 5.01, 10.00),
)


@dataclass(frozen=True)
class ExitProfile:
    name: str
    stop_multiple: float
    first_target: float
    second_target: float
    trailing_keep: float


EXIT_PROFILES = (
    ExitProfile("tight", 0.70, 1.50, 3.00, 0.75),
    ExitProfile("balanced", 0.50, 2.00, 5.00, 0.70),
    ExitProfile("wide", 0.25, 2.00, 10.00, 0.50),
)


@dataclass(frozen=True)
class StrategyConfig:
    premium: PremiumRange
    signal_mode: SignalMode
    exit_profile: ExitProfile

    @property
    def key(self) -> str:
        return (
            f"{self.premium.name}|{self.signal_mode}|{self.exit_profile.name}"
        )


@dataclass(frozen=True)
class Bar:
    minute: int
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class ContractDay:
    code: str
    name: str
    call_put: str
    bars: list[Bar]
    ma5: list[float | None]
    ma20: list[float | None]
    volume_ma20: list[float | None]
    previous_high5: list[float | None]


@dataclass(frozen=True)
class EntryEvent:
    code: str
    signal_index: int
    fill_index: int
    signal_minute: int
    fill_minute: int
    score: float


@dataclass
class PositionResult:
    pnl: int
    entry_spend: int
    exit_minute: int
    buy_count: int
    sell_count: int
    trades: list[dict[str, object]]


@dataclass
class DayResult:
    date: str
    start_cash: int
    pnl: int
    end_cash: int
    entry_spend: int
    campaigns: int
    buy_count: int
    sell_count: int


@dataclass
class StrategyState:
    config: StrategyConfig
    cash: int = INITIAL_ACCOUNT
    daily: list[DayResult] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


def rolling_average(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def build_contract(
    code: str,
    metadata: dict[str, str],
    bars_by_minute: dict[int, Bar],
) -> ContractDay:
    bars = sorted(bars_by_minute.values(), key=lambda item: item.minute)
    closes = [bar.close for bar in bars]
    volumes = [float(bar.volume) for bar in bars]
    previous_high5: list[float | None] = [None] * len(bars)
    for index in range(5, len(bars)):
        previous_high5[index] = max(bar.high for bar in bars[index - 5 : index])
    return ContractDay(
        code=code,
        name=metadata["name"],
        call_put=metadata["call_put"],
        bars=bars,
        ma5=rolling_average(closes, 5),
        ma20=rolling_average(closes, 20),
        volume_ma20=rolling_average(volumes, 20),
        previous_high5=previous_high5,
    )


def load_expiry_day(
    path: Path, group: dict[str, str]
) -> dict[str, ContractDay]:
    bars_by_code: dict[str, dict[int, Bar]] = defaultdict(dict)
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
                open_price = float(row["시가"])
                high = float(row["고가"])
                low = float(row["저가"])
                close = float(row["종가"])
                volume = int(float(row["거래량"]))
            except (TypeError, ValueError):
                continue

            code = row["종목코드"]
            existing = bars_by_code[code].get(minute)
            if existing is None:
                bars_by_code[code][minute] = Bar(
                    minute, open_price, high, low, close, volume
                )
            else:
                bars_by_code[code][minute] = Bar(
                    minute=minute,
                    open=existing.open,
                    high=max(existing.high, high),
                    low=min(existing.low, low),
                    close=close,
                    volume=existing.volume + volume,
                )
            metadata.setdefault(
                code,
                {
                    "name": row["종목명"],
                    "call_put": row["콜풋구분"],
                },
            )

    return {
        code: build_contract(code, metadata[code], bars)
        for code, bars in bars_by_code.items()
        if len(bars) >= 21
    }


def signal_flags(contract: ContractDay, index: int) -> tuple[bool, bool, float]:
    if index < 20:
        return False, False, 0.0
    ma5 = contract.ma5[index]
    ma20 = contract.ma20[index]
    previous_ma5 = contract.ma5[index - 1]
    previous_close = contract.bars[index - 1].close
    volume_ma20 = contract.volume_ma20[index]
    previous_high = contract.previous_high5[index]
    if None in (ma5, ma20, previous_ma5, volume_ma20, previous_high):
        return False, False, 0.0

    bar = contract.bars[index]
    assert ma5 is not None
    assert ma20 is not None
    assert previous_ma5 is not None
    assert volume_ma20 is not None
    assert previous_high is not None

    trend = ma5 > ma20 and ma5 > previous_ma5 and bar.close > ma20
    volume_ok = bar.volume >= max(1.0, volume_ma20 * 0.5)
    recovery = (
        trend
        and volume_ok
        and previous_close <= previous_ma5
        and bar.close > ma5
    )
    breakout = trend and volume_ok and bar.close > previous_high
    score = (bar.close / ma20 - 1.0) * 100 + min(
        bar.volume / max(volume_ma20, 1.0), 5.0
    )
    return recovery, breakout, score


def build_entry_events(
    contracts: dict[str, ContractDay],
    premium: PremiumRange,
    signal_mode: SignalMode,
) -> list[EntryEvent]:
    events: list[EntryEvent] = []
    for contract in contracts.values():
        last_signal_index = len(contract.bars) - 1 - EXTRA_EXECUTION_BARS
        for index in range(20, last_signal_index):
            bar = contract.bars[index]
            fill_index = index + 1 + EXTRA_EXECUTION_BARS
            fill_bar = contract.bars[fill_index]
            if bar.minute > NEW_ENTRY_CUTOFF:
                continue
            if not premium.contains(bar.close):
                continue
            recovery, breakout, score = signal_flags(contract, index)
            signal = (
                recovery
                if signal_mode == "recovery"
                else breakout
                if signal_mode == "breakout"
                else recovery or breakout
            )
            if not signal:
                continue
            events.append(
                EntryEvent(
                    code=contract.code,
                    signal_index=index,
                    fill_index=fill_index,
                    signal_minute=bar.minute,
                    fill_minute=fill_bar.minute,
                    score=score,
                )
            )
    return sorted(
        events,
        key=lambda item: (
            item.signal_minute,
            -item.score,
            item.fill_minute,
            item.code,
        ),
    )


def affordable_quantity(budget: int, price: float, fill_volume: int) -> int:
    per_contract = price * CONTRACT_MULTIPLIER + COMMISSION_PER_CONTRACT
    by_cash = math.floor(budget / per_contract) if per_contract > 0 else 0
    by_volume = math.floor(fill_volume * ENTRY_VOLUME_PARTICIPATION)
    return max(0, min(by_cash, by_volume))


def buy(
    date: str,
    contract: ContractDay,
    bar: Bar,
    quantity: int,
    reason: str,
) -> tuple[int, dict[str, object]]:
    fill_price = bar.open + SLIPPAGE
    cash = round(
        quantity
        * (fill_price * CONTRACT_MULTIPLIER + COMMISSION_PER_CONTRACT)
    )
    return cash, {
        "date": date,
        "minute": bar.minute,
        "code": contract.code,
        "name": contract.name,
        "call_put": contract.call_put,
        "side": "BUY",
        "reason": reason,
        "quantity": quantity,
        "price": fill_price,
        "bar_volume": bar.volume,
        "cash_flow": -cash,
    }


def sell(
    date: str,
    contract: ContractDay,
    price: float,
    minute: int,
    quantity: int,
    reason: str,
) -> tuple[int, dict[str, object]]:
    fill_price = max(0.0, price - SLIPPAGE)
    cash = round(
        quantity
        * (fill_price * CONTRACT_MULTIPLIER - COMMISSION_PER_CONTRACT)
    )
    return cash, {
        "date": date,
        "minute": minute,
        "code": contract.code,
        "name": contract.name,
        "call_put": contract.call_put,
        "side": "SELL",
        "reason": reason,
        "quantity": quantity,
        "price": fill_price,
        "bar_volume": None,
        "cash_flow": cash,
    }


def simulate_position(
    date: str,
    contract: ContractDay,
    entry: EntryEvent,
    config: StrategyConfig,
    initial_budget: int,
    add_budget: int,
) -> PositionResult | None:
    entry_bar = contract.bars[entry.fill_index]
    entry_price = entry_bar.open + SLIPPAGE
    initial_quantity = affordable_quantity(
        initial_budget, entry_price, entry_bar.volume
    )
    if initial_quantity <= 0:
        return None

    initial_cash, initial_trade = buy(
        date, contract, entry_bar, initial_quantity, "INITIAL"
    )
    quantity = initial_quantity
    total_bought_quantity = initial_quantity
    weighted_entry_value = entry_price * initial_quantity
    entry_spend = initial_cash
    cash_flow = -initial_cash
    trades = [initial_trade]
    buy_count = 1
    sell_count = 0
    added = False
    first_target_done = False
    second_target_done = False
    high_water = entry_price
    exit_minute = entry_bar.minute

    index = entry.fill_index
    while index < len(contract.bars):
        bar = contract.bars[index]
        average_entry = weighted_entry_value / total_bought_quantity
        high_water = max(high_water, bar.close)
        execution_index = index + 1 + EXTRA_EXECUTION_BARS
        execution_bar = (
            contract.bars[execution_index]
            if execution_index < len(contract.bars)
            else None
        )

        reason: str | None = None
        sell_quantity = 0
        profile = config.exit_profile
        if bar.close <= average_entry * profile.stop_multiple:
            reason = "STOP"
            sell_quantity = quantity
        elif not first_target_done and bar.close >= average_entry * profile.first_target:
            reason = "TARGET_1"
            sell_quantity = max(1, math.ceil(quantity / 2))
            first_target_done = True
        elif (
            first_target_done
            and not second_target_done
            and bar.close >= average_entry * profile.second_target
        ):
            reason = "TARGET_2"
            sell_quantity = max(1, math.ceil(quantity / 2))
            second_target_done = True
        elif first_target_done and bar.close <= high_water * profile.trailing_keep:
            reason = "TRAIL"
            sell_quantity = quantity
        elif bar.minute >= FORCE_EXIT_SIGNAL_TIME:
            reason = "TIME"
            sell_quantity = quantity

        if reason is not None:
            if execution_bar is None:
                proceeds, trade = sell(
                    date,
                    contract,
                    bar.close,
                    bar.minute,
                    sell_quantity,
                    f"{reason}_LAST_CLOSE",
                )
                exit_minute = bar.minute
            else:
                proceeds, trade = sell(
                    date,
                    contract,
                    execution_bar.open,
                    execution_bar.minute,
                    sell_quantity,
                    reason,
                )
                exit_minute = execution_bar.minute
            cash_flow += proceeds
            quantity -= sell_quantity
            sell_count += 1
            trades.append(trade)
            if quantity <= 0:
                return PositionResult(
                    pnl=cash_flow,
                    entry_spend=entry_spend,
                    exit_minute=exit_minute,
                    buy_count=buy_count,
                    sell_count=sell_count,
                    trades=trades,
                )
            index = execution_index
            continue

        ma5 = contract.ma5[index]
        ma20 = contract.ma20[index]
        previous_high = contract.previous_high5[index]
        can_add = (
            not added
            and add_budget > 0
            and execution_bar is not None
            and not first_target_done
            and ma5 is not None
            and ma20 is not None
            and previous_high is not None
            and ma5 > ma20
            and bar.close >= average_entry * 1.20
            and bar.close < average_entry * profile.first_target
            and bar.close > previous_high
        )
        if can_add:
            assert execution_bar is not None
            add_price = execution_bar.open + SLIPPAGE
            add_quantity = affordable_quantity(
                add_budget, add_price, execution_bar.volume
            )
            if add_quantity > 0:
                add_cash, add_trade = buy(
                    date, contract, execution_bar, add_quantity, "ADD"
                )
                quantity += add_quantity
                total_bought_quantity += add_quantity
                weighted_entry_value += add_price * add_quantity
                entry_spend += add_cash
                cash_flow -= add_cash
                buy_count += 1
                trades.append(add_trade)
                added = True
                index = execution_index - 1

        index += 1

    last_bar = contract.bars[-1]
    proceeds, trade = sell(
        date, contract, last_bar.close, last_bar.minute, quantity, "DATA_END"
    )
    trades.append(trade)
    return PositionResult(
        pnl=cash_flow + proceeds,
        entry_spend=entry_spend,
        exit_minute=last_bar.minute,
        buy_count=buy_count,
        sell_count=sell_count + 1,
        trades=trades,
    )


def simulate_day(
    date: str,
    contracts: dict[str, ContractDay],
    events: list[EntryEvent],
    config: StrategyConfig,
    start_cash: int,
    collect_trades: bool = False,
) -> tuple[DayResult, list[dict[str, object]]]:
    daily_limit = min(DAILY_BUY_LIMIT, start_cash)
    spent = 0
    pnl = 0
    cursor = 0
    campaigns = 0
    buy_count = 0
    sell_count = 0
    trade_log: list[dict[str, object]] = []
    consumed_events: set[tuple[str, int]] = set()

    while daily_limit - spent > 0:
        candidates = [
            event
            for event in events
            if event.signal_minute >= cursor
            and (event.code, event.signal_index) not in consumed_events
        ]
        if not candidates:
            break
        first_minute = candidates[0].signal_minute
        same_minute = [
            event for event in candidates if event.signal_minute == first_minute
        ]
        event = max(same_minute, key=lambda item: item.score)
        consumed_events.add((event.code, event.signal_index))

        remaining = daily_limit - spent
        initial_budget = min(FIRST_ENTRY_BUDGET, remaining)
        add_budget = max(0, remaining - initial_budget)
        result = simulate_position(
            date,
            contracts[event.code],
            event,
            config,
            initial_budget,
            add_budget,
        )
        if result is None:
            cursor = event.signal_minute
            continue

        campaigns += 1
        spent += result.entry_spend
        pnl += result.pnl
        buy_count += result.buy_count
        sell_count += result.sell_count
        cursor = result.exit_minute
        if collect_trades:
            trade_log.extend(result.trades)

    day = DayResult(
        date=date,
        start_cash=start_cash,
        pnl=pnl,
        end_cash=start_cash + pnl,
        entry_spend=spent,
        campaigns=campaigns,
        buy_count=buy_count,
        sell_count=sell_count,
    )
    return day, trade_log


def strategy_configs() -> list[StrategyConfig]:
    return [
        StrategyConfig(premium, signal_mode, exit_profile)
        for premium in PREMIUM_RANGES
        for signal_mode in ("recovery", "breakout", "either")
        for exit_profile in EXIT_PROFILES
    ]


def max_drawdown(values: list[int]) -> int:
    peak = values[0]
    worst = 0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def summarize_state(state: StrategyState) -> dict[str, object]:
    assert state.daily is not None
    traded = [item for item in state.daily if item.campaigns > 0]
    profits = [item.pnl for item in state.daily]
    values = [INITIAL_ACCOUNT] + [item.end_cash for item in state.daily]
    profits_2026 = [
        item.pnl for item in state.daily if item.date >= "20260101"
    ]
    profits_development = [
        item.pnl for item in state.daily if item.date <= "20260430"
    ]
    profits_late_period = [
        item.pnl for item in state.daily if item.date >= "20260501"
    ]
    best_days = sorted(profits, reverse=True)
    return {
        "strategy": state.config.key,
        "premium_range": state.config.premium.name,
        "signal_mode": state.config.signal_mode,
        "exit_profile": state.config.exit_profile.name,
        "expiry_days": len(state.daily),
        "trade_days": len(traded),
        "no_trade_days": len(state.daily) - len(traded),
        "profitable_days": sum(value > 0 for value in profits),
        "losing_days": sum(value < 0 for value in profits),
        "total_pnl": sum(profits),
        "ending_cash": state.cash,
        "average_expiry_pnl": round(sum(profits) / len(profits), 2),
        "maximum_day_pnl": max(profits),
        "minimum_day_pnl": min(profits),
        "max_drawdown": max_drawdown(values),
        "minimum_end_of_day_cash": min(values),
        "pnl_excluding_best_1_day": sum(profits) - sum(best_days[:1]),
        "pnl_excluding_best_3_days": sum(profits) - sum(best_days[:3]),
        "pnl_excluding_best_5_days": sum(profits) - sum(best_days[:5]),
        "days_at_least_2_5m_profit": sum(value >= 2_500_000 for value in profits),
        "maximum_daily_entry_spend": max(
            item.entry_spend for item in state.daily
        ),
        "total_entry_spend": sum(item.entry_spend for item in state.daily),
        "total_campaigns": sum(item.campaigns for item in state.daily),
        "total_pnl_through_2026_04": sum(profits_development),
        "total_pnl_2026_05_to_08": sum(profits_late_period),
        "total_pnl_2026": sum(profits_2026),
        "profitable_days_2026": sum(value > 0 for value in profits_2026),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_grid() -> tuple[list[dict[str, object]], StrategyConfig]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = target_file_map()
    states = {config.key: StrategyState(config=config) for config in strategy_configs()}

    for group in groups:
        date = group["expiry_date"]
        contracts = load_expiry_day(files[date], group)
        event_cache: dict[tuple[str, SignalMode], list[EntryEvent]] = {}
        for premium in PREMIUM_RANGES:
            for signal_mode in ("recovery", "breakout", "either"):
                event_cache[(premium.name, signal_mode)] = build_entry_events(
                    contracts, premium, signal_mode
                )

        for state in states.values():
            events = event_cache[
                (state.config.premium.name, state.config.signal_mode)
            ]
            day, _ = simulate_day(
                date,
                contracts,
                events,
                state.config,
                state.cash,
            )
            assert state.daily is not None
            state.daily.append(day)
            state.cash = day.end_cash

    summaries = sorted(
        (summarize_state(state) for state in states.values()),
        key=lambda item: (
            int(item["total_pnl"]),
            int(item["max_drawdown"]),
        ),
        reverse=True,
    )
    best_key = str(summaries[0]["strategy"])
    return summaries, states[best_key].config


def rerun_best(
    config: StrategyConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = target_file_map()
    cash = INITIAL_ACCOUNT
    daily_rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for group in groups:
        date = group["expiry_date"]
        contracts = load_expiry_day(files[date], group)
        events = build_entry_events(
            contracts, config.premium, config.signal_mode
        )
        day, day_trades = simulate_day(
            date,
            contracts,
            events,
            config,
            cash,
            collect_trades=True,
        )
        cash = day.end_cash
        daily_rows.append(asdict(day))
        trades.extend(day_trades)
    return daily_rows, trades


def main() -> None:
    summaries, best_config = run_grid()
    daily_rows, trades = rerun_best(best_config)
    best_summary = summaries[0]
    development_selected = max(
        summaries,
        key=lambda item: int(item["total_pnl_through_2026_04"]),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "baseline_grid.csv", summaries)
    write_csv(OUTPUT_DIR / "baseline_best_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "baseline_best_trades.csv", trades)
    result = {
        "status": "discarded_exploratory_in_sample",
        "validation_eligible": False,
        "account": {
            "initial_cash": INITIAL_ACCOUNT,
            "daily_buy_limit": DAILY_BUY_LIMIT,
            "compounding_position_size": False,
            "same_day_exit_cash_reused": False,
        },
        "execution": {
            "multiplier": CONTRACT_MULTIPLIER,
            "commission_per_contract_per_side": COMMISSION_PER_CONTRACT,
            "slippage_per_side": SLIPPAGE,
            "entry_volume_participation": ENTRY_VOLUME_PARTICIPATION,
            "signal_on_close_fill": "next actual bar open",
        },
        "grid_count": len(summaries),
        "best": best_summary,
        "development_period_selected": development_selected,
        "top_10": summaries[:10],
    }
    (OUTPUT_DIR / "baseline_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
