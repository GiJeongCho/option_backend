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


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S5-v0.4"
DECISION_MINUTE = 910
EARLY_END = 930
LATE_END = 1430
FORCE_EXIT_TIME = 1515
INITIAL_BUDGET = 5_000_000

PPT_TOTAL = 1_320
PPT_EARLY = 633
PPT_AMBIGUOUS = 37
PPT_FAILED = 650
PPT_BUCKETS = {
    "1_5_TO_2": 153,
    "2_TO_5": 262,
    "5_TO_10": 92,
    "10_PLUS": 126,
}
PPT_BUCKET_FLOORS = {
    "1_5_TO_2": 1.5,
    "2_TO_5": 2.0,
    "5_TO_10": 5.0,
    "10_PLUS": 10.0,
}
PPT_EXPECTED_PEAK_FLOOR = sum(
    PPT_BUCKETS[name] * PPT_BUCKET_FLOORS[name]
    for name in PPT_BUCKETS
) / PPT_EARLY

CONFIGS = {
    "EV": {
        "strategy": "S5-v0.4-EV",
        "confirmed_exit": "EXPECTED_PEAK_3_9076X_ALL",
        "failure_exit": "09:30 all",
    },
    "STAGED": {
        "strategy": "S5-v0.4-STAGED",
        "confirmed_exit": "10/20/30/40% at 1.5/2/5/10x",
        "failure_exit": (
            "before confirmation: half at 0.15/0.20 and rest at "
            "0.03/0.04; stop-first if same-bar order is unknown"
        ),
    },
}


@dataclass(frozen=True)
class Setup:
    code: str
    call_put: str
    observation_index: int
    observation_minute: int
    observation_price: float
    premium_bin: str
    cumulative_volume: int


@dataclass(frozen=True)
class PlannedEntry:
    setup: Setup
    fill_index: int
    fill_minute: int
    fill_price: float
    fill_volume: int
    premium_bin: str
    quantity: int


@dataclass
class PositionOutcome:
    setup: Setup
    early_confirmed: bool
    early_confirmation_index: int | None
    entry_quantity: int
    entry_principal: int
    entry_fee: int
    sold_quantity: int
    net_sale_proceeds: int
    exit_index: int
    expired_quantity: int
    trades: list[dict[str, object]]


@dataclass
class DayResult:
    strategy: str
    date: str
    start_cash: int
    end_cash: int
    pnl: int
    eligible_contracts: int
    active_premium_bins: int
    initial_positions: int
    initial_buy_principal: int
    early_confirmed_positions: int
    early_failed_positions: int
    failed_sale_proceeds: int
    late_signal_found: bool
    late_reentry_filled: bool
    late_code: str
    late_call_put: str
    late_confirmation_minute: int | None
    late_entry_minute: int | None
    late_buy_principal: int
    total_buy_principal: int
    maximum_deployed_principal: int
    total_fees: int
    expired_quantity: int


@dataclass
class PositionRecord:
    strategy: str
    date: str
    campaign: int
    code: str
    call_put: str
    branch: str
    observation_minute: int
    observation_price: float
    confirmation_minute: int | None
    entry_minute: int
    entry_price: float
    quantity: int
    buy_principal: int
    buy_fee: int
    net_sale_proceeds: int
    pnl: int
    expired_quantity: int


def premium_bin(price: float) -> str:
    index = min(7, max(0, math.floor((price - 0.4) / 0.1 + 1e-9)))
    lower = 0.4 + index * 0.1
    upper = lower + 0.1
    return f"{lower:.1f}-{upper:.1f}"


def observation(contract: base.ContractGrid) -> Setup | None:
    start_index = base.MINUTE_INDEX[legacy.MORNING_START]
    end_index = base.MINUTE_INDEX[legacy.MORNING_END]
    for index in range(start_index, end_index + 1):
        bar = contract.bars[index]
        if not bar.traded:
            continue
        values = legacy.observed_values(bar)
        if not values:
            continue
        price = max(values)
        return Setup(
            code=contract.code,
            call_put=contract.call_put,
            observation_index=index,
            observation_minute=bar.minute,
            observation_price=price,
            premium_bin=premium_bin(price),
            cumulative_volume=sum(
                item.volume
                for item in contract.bars[start_index : index + 1]
            ),
        )
    return None


def first_high_touch(
    contract: base.ContractGrid,
    start_index: int,
    end_minute: int,
    threshold: float,
) -> int | None:
    for index in range(start_index, len(contract.bars)):
        bar = contract.bars[index]
        if bar.minute > end_minute:
            return None
        if (
            bar.traded
            and bar.high is not None
            and float(bar.high) >= threshold
        ):
            return index
    return None


def setup_fill(
    contract: base.ContractGrid,
) -> tuple[int, int, float, int, str] | None:
    fill = base.next_traded_bar(
        contract,
        base.MINUTE_INDEX[DECISION_MINUTE],
        base.ENTRY_WAIT_MINUTES,
    )
    if fill is None:
        return None
    index, bar = fill
    assert bar.open is not None
    raw_price = float(bar.open)
    if not legacy.PREMIUM_MIN <= raw_price <= legacy.PREMIUM_MAX:
        return None
    return (
        index,
        bar.minute,
        raw_price + base.SLIPPAGE,
        bar.volume,
        premium_bin(raw_price),
    )


def allocate_entries(
    contracts: dict[str, base.ContractGrid],
    setups: list[Setup],
) -> list[PlannedEntry]:
    fillable: dict[
        str, tuple[Setup, int, int, float, int, str]
    ] = {}
    for setup in setups:
        fill = setup_fill(contracts[setup.code])
        if fill is None:
            continue
        fillable[setup.code] = (setup, *fill)
    by_bin: dict[
        str, list[tuple[Setup, int, int, float, int, str]]
    ] = defaultdict(list)
    for candidate in fillable.values():
        by_bin[candidate[5]].append(candidate)
    if not by_bin:
        return []

    bin_names = sorted(by_bin)
    base_budget, remainder = divmod(INITIAL_BUDGET, len(bin_names))
    result: list[PlannedEntry] = []
    for bin_position, name in enumerate(bin_names):
        budget = base_budget + (
            remainder if bin_position == len(bin_names) - 1 else 0
        )
        candidates = sorted(
            by_bin[name],
            key=lambda item: (
                item[1],
                -item[0].cumulative_volume,
                item[0].code,
            ),
        )
        allocated = {item[0].code: 0 for item in candidates}
        maximum = {
            item[0].code: math.floor(
                item[4] * base.ENTRY_PARTICIPATION
            )
            for item in candidates
        }
        units = {
            item[0].code: round(
                item[3] * base.MULTIPLIER
            )
            for item in candidates
        }
        remaining = budget
        while True:
            available = [
                item
                for item in candidates
                if allocated[item[0].code] < maximum[item[0].code]
                and units[item[0].code] <= remaining
            ]
            if not available:
                break
            selected = min(
                available,
                key=lambda item: (
                    allocated[item[0].code]
                    * units[item[0].code],
                    -item[0].cumulative_volume,
                    item[0].code,
                ),
            )
            code = selected[0].code
            allocated[code] += 1
            remaining -= units[code]
        for setup, index, minute, price, volume, entry_bin in candidates:
            quantity = allocated[setup.code]
            if quantity <= 0:
                continue
            result.append(
                PlannedEntry(
                    setup=setup,
                    fill_index=index,
                    fill_minute=minute,
                    fill_price=price,
                    fill_volume=volume,
                    premium_bin=entry_bin,
                    quantity=quantity,
                )
            )
    allocated_by_code = {
        code: 0 for code in fillable
    }
    for entry in result:
        allocated_by_code[entry.setup.code] = entry.quantity
    maximum_by_code = {
        code: math.floor(candidate[4] * base.ENTRY_PARTICIPATION)
        for code, candidate in fillable.items()
    }
    unit_by_code = {
        code: round(candidate[3] * base.MULTIPLIER)
        for code, candidate in fillable.items()
    }
    invested_by_bin = {
        name: sum(
            allocated_by_code[item[0].code]
            * unit_by_code[item[0].code]
            for item in candidates
        )
        for name, candidates in by_bin.items()
    }
    remaining = INITIAL_BUDGET - sum(invested_by_bin.values())
    while True:
        available = [
            candidate
            for candidate in fillable.values()
            if (
                allocated_by_code[candidate[0].code]
                < maximum_by_code[candidate[0].code]
                and unit_by_code[candidate[0].code] <= remaining
            )
        ]
        if not available:
            break
        selected = min(
            available,
            key=lambda item: (
                invested_by_bin[item[5]],
                allocated_by_code[item[0].code]
                * unit_by_code[item[0].code],
                -item[0].cumulative_volume,
                item[0].code,
            ),
        )
        code = selected[0].code
        allocated_by_code[code] += 1
        invested_by_bin[selected[5]] += unit_by_code[code]
        remaining -= unit_by_code[code]
    result = [
        PlannedEntry(
            setup=setup,
            fill_index=index,
            fill_minute=minute,
            fill_price=price,
            fill_volume=volume,
            premium_bin=entry_bin,
            quantity=allocated_by_code[setup.code],
        )
        for setup, index, minute, price, volume, entry_bin in fillable.values()
        if allocated_by_code[setup.code] > 0
    ]
    return sorted(
        result,
        key=lambda item: (
            item.fill_index,
            item.premium_bin,
            item.setup.code,
        ),
    )


def trade_row(
    strategy_version: str,
    date_value: str,
    campaign: int,
    minute: int,
    contract: base.ContractGrid,
    side: str,
    reason: str,
    stage: int,
    quantity: int,
    price: float,
    bar_volume: int,
) -> dict[str, object]:
    principal = round(quantity * price * base.MULTIPLIER)
    fee = (
        quantity * base.COMMISSION_PER_CONTRACT
        if side in {"BUY", "SELL"}
        else 0
    )
    cash_flow = (
        -(principal + fee)
        if side == "BUY"
        else principal - fee
        if side == "SELL"
        else 0
    )
    return {
        "strategy": strategy_version,
        "date": date_value,
        "campaign": campaign,
        "minute": minute,
        "code": contract.code,
        "call_put": contract.call_put,
        "side": side,
        "reason": reason,
        "stage": stage,
        "quantity": quantity,
        "price": price,
        "principal": principal,
        "fee": fee,
        "cash_flow": cash_flow,
        "bar_volume": bar_volume,
    }


def execute_sale(
    strategy_version: str,
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    signal_index: int,
    requested_quantity: int,
    reason: str,
    stage: int,
    used_capacity: dict[int, int],
) -> tuple[int, int, list[dict[str, object]]]:
    remaining = requested_quantity
    last_index = signal_index
    trades: list[dict[str, object]] = []
    for index in range(signal_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if not bar.traded:
            continue
        assert bar.open is not None
        maximum = math.floor(
            bar.volume * base.EXIT_PARTICIPATION
        )
        fillable = min(
            remaining,
            max(0, maximum - used_capacity.get(index, 0)),
        )
        if fillable <= 0:
            continue
        used_capacity[index] = used_capacity.get(index, 0) + fillable
        fill_price = max(0.0, float(bar.open) - base.SLIPPAGE)
        trades.append(
            trade_row(
                strategy_version,
                date_value,
                campaign,
                bar.minute,
                contract,
                "SELL",
                reason,
                stage,
                fillable,
                fill_price,
                bar.volume,
            )
        )
        remaining -= fillable
        last_index = index
        if remaining <= 0:
            break
    return requested_quantity - remaining, last_index, trades


def expire_trade(
    strategy_version: str,
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    quantity: int,
) -> dict[str, object]:
    return trade_row(
        strategy_version,
        date_value,
        campaign,
        base.SESSION_MINUTES[-1],
        contract,
        "EXPIRE",
        "UNFILLED_ZERO_VALUE",
        9,
        quantity,
        0.0,
        0,
    )


def confirmed_exit(
    config_name: str,
    strategy_version: str,
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    setup: Setup,
    entry_index: int,
    quantity: int,
    used_capacity: dict[int, int],
    minimum_multiple: float = 0.0,
) -> tuple[int, int, list[dict[str, object]]]:
    remaining = quantity
    sold = 0
    last_index = entry_index
    trades: list[dict[str, object]] = []
    if config_name == "EV":
        thresholds = (
            (
                PPT_EXPECTED_PEAK_FLOOR,
                1.0,
                "EXPECTED_PEAK_3_9076X_ALL",
                1,
            ),
        )
    else:
        thresholds = (
            (1.5, 0.10, "TARGET_1_5X_10PCT", 1),
            (2.0, 0.30, "TARGET_2X_CUMULATIVE_30PCT", 2),
            (5.0, 0.60, "TARGET_5X_CUMULATIVE_60PCT", 3),
            (10.0, 1.00, "TARGET_10X_ALL", 4),
        )
    achieved = 0.0
    index = entry_index
    while index < len(contract.bars) and remaining > 0:
        bar = contract.bars[index]
        desired_cumulative = None
        reason = ""
        stage = 0
        if bar.traded and bar.high is not None:
            multiple = max(
                minimum_multiple,
                float(bar.high) / setup.observation_price,
            )
            for threshold, cumulative, name, number in thresholds:
                if multiple >= threshold and cumulative > achieved:
                    desired_cumulative = cumulative
                    reason = name
                    stage = number
        if desired_cumulative is not None:
            target_sold = (
                quantity
                if desired_cumulative >= 1.0
                else math.ceil(quantity * desired_cumulative)
            )
            request = min(remaining, max(0, target_sold - sold))
            achieved = desired_cumulative
            if request > 0:
                filled, fill_index, sale_trades = execute_sale(
                    strategy_version,
                    date_value,
                    campaign,
                    contract,
                    index,
                    request,
                    reason,
                    stage,
                    used_capacity,
                )
                remaining -= filled
                sold += filled
                last_index = max(last_index, fill_index)
                trades.extend(sale_trades)
                index = max(index + 1, fill_index)
                continue
        if bar.minute >= FORCE_EXIT_TIME:
            filled, fill_index, sale_trades = execute_sale(
                strategy_version,
                date_value,
                campaign,
                contract,
                index,
                remaining,
                "TIME",
                8,
                used_capacity,
            )
            remaining -= filled
            sold += filled
            last_index = max(last_index, fill_index)
            trades.extend(sale_trades)
            break
        index += 1
    if remaining > 0:
        trades.append(
            expire_trade(
                strategy_version,
                date_value,
                campaign,
                contract,
                remaining,
            )
        )
        last_index = len(contract.bars) - 1
    return remaining, last_index, trades


def staged_waiting_stops(
    strategy_version: str,
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    setup: Setup,
    entry_index: int,
    confirmation_index: int | None,
    quantity: int,
    used_capacity: dict[int, int],
) -> tuple[int, int, list[dict[str, object]]]:
    """Apply decay stops while the post-entry 1.5x signal is unconfirmed.

    If a minute contains both a stop low and the confirming high, the stop is
    processed first.  Minute OHLC cannot reveal the intrabar order, so this
    fixed conservative convention avoids selecting the favorable ordering.
    """
    remaining = quantity
    sold = 0
    last_index = entry_index
    trades: list[dict[str, object]] = []
    signal_end = (
        confirmation_index
        if confirmation_index is not None
        else base.MINUTE_INDEX[EARLY_END]
    )
    middle = 0.15 if setup.observation_price < 0.8 else 0.20
    final = 0.03 if setup.observation_price < 0.8 else 0.04
    middle_done = False
    index = entry_index
    while index <= signal_end and remaining > 0:
        bar = contract.bars[index]
        if bar.traded and bar.low is not None:
            low = float(bar.low)
            if low <= final:
                request = remaining
                reason = "DECAY_FINAL_STOP"
                stage = 2
            elif low <= middle and not middle_done:
                target_sold = math.ceil(quantity / 2)
                request = min(remaining, max(0, target_sold - sold))
                reason = "DECAY_HALF_STOP"
                stage = 1
                middle_done = True
            else:
                request = 0
                reason = ""
                stage = 0
            if request > 0:
                filled, fill_index, sale_trades = execute_sale(
                    strategy_version,
                    date_value,
                    campaign,
                    contract,
                    index,
                    request,
                    reason,
                    stage,
                    used_capacity,
                )
                remaining -= filled
                sold += filled
                last_index = max(last_index, fill_index)
                trades.extend(sale_trades)
                index = max(index + 1, fill_index)
                continue
        index += 1
    return remaining, last_index, trades


def failed_exit(
    strategy_version: str,
    date_value: str,
    campaign: int,
    contract: base.ContractGrid,
    entry_index: int,
    quantity: int,
    used_capacity: dict[int, int],
) -> tuple[int, int, list[dict[str, object]]]:
    remaining = quantity
    sold = 0
    last_index = entry_index
    trades: list[dict[str, object]] = []
    end_index = base.MINUTE_INDEX[EARLY_END]
    if remaining > 0:
        filled, fill_index, sale_trades = execute_sale(
            strategy_version,
            date_value,
            campaign,
            contract,
            end_index,
            remaining,
            "NO_1_5X_09_30_EXIT",
            3,
            used_capacity,
        )
        remaining -= filled
        sold += filled
        last_index = max(last_index, fill_index)
        trades.extend(sale_trades)
    if remaining > 0:
        trades.append(
            expire_trade(
                strategy_version,
                date_value,
                campaign,
                contract,
                remaining,
            )
        )
        last_index = len(contract.bars) - 1
    return remaining, last_index, trades


def simulate_initial_position(
    config_name: str,
    strategy_version: str,
    date_value: str,
    contract: base.ContractGrid,
    entry: PlannedEntry,
    campaign: int,
    used_capacity: dict[int, int],
) -> PositionOutcome:
    buy = trade_row(
        strategy_version,
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
    confirmation = first_high_touch(
        contract,
        entry.fill_index,
        EARLY_END,
        entry.setup.observation_price * legacy.CONFIRMATION_MULTIPLE,
    )
    if config_name == "STAGED":
        remaining, stop_exit_index, stop_sells = staged_waiting_stops(
            strategy_version,
            date_value,
            campaign,
            contract,
            entry.setup,
            entry.fill_index,
            confirmation,
            entry.quantity,
            used_capacity,
        )
        if confirmation is not None and remaining > 0:
            remaining, target_exit_index, target_sells = confirmed_exit(
                config_name,
                strategy_version,
                date_value,
                campaign,
                contract,
                entry.setup,
                max(confirmation, stop_exit_index),
                remaining,
                used_capacity,
                minimum_multiple=legacy.CONFIRMATION_MULTIPLE,
            )
            exit_index = max(stop_exit_index, target_exit_index)
            sells = [*stop_sells, *target_sells]
        elif confirmation is None and remaining > 0:
            remaining, failed_exit_index, failed_sells = failed_exit(
                strategy_version,
                date_value,
                campaign,
                contract,
                max(entry.fill_index, stop_exit_index),
                remaining,
                used_capacity,
            )
            exit_index = max(stop_exit_index, failed_exit_index)
            sells = [*stop_sells, *failed_sells]
        else:
            exit_index = stop_exit_index
            sells = stop_sells
    elif confirmation is not None:
        remaining, exit_index, sells = confirmed_exit(
            config_name,
            strategy_version,
            date_value,
            campaign,
            contract,
            entry.setup,
            entry.fill_index,
            entry.quantity,
            used_capacity,
        )
    else:
        remaining, exit_index, sells = failed_exit(
            strategy_version,
            date_value,
            campaign,
            contract,
            entry.fill_index,
            entry.quantity,
            used_capacity,
        )
    sell_rows = [row for row in sells if row["side"] == "SELL"]
    return PositionOutcome(
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


def late_candidate(
    contracts: dict[str, base.ContractGrid],
    failed: list[PositionOutcome],
) -> tuple[PositionOutcome, int] | None:
    candidates: list[tuple[PositionOutcome, int, int]] = []
    start_after_930 = base.MINUTE_INDEX[EARLY_END] + 1
    for outcome in failed:
        if outcome.expired_quantity > 0:
            continue
        contract = contracts[outcome.setup.code]
        start = max(start_after_930, outcome.exit_index + 1)
        confirmation = first_high_touch(
            contract,
            start,
            LATE_END,
            outcome.setup.observation_price
            * legacy.CONFIRMATION_MULTIPLE,
        )
        if confirmation is None:
            continue
        candidates.append(
            (
                outcome,
                confirmation,
                contract.bars[confirmation].volume,
            )
        )
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: (
            item[1],
            -item[2],
            item[0].setup.code,
        ),
    )
    return selected[0], selected[1]


def simulate_late_reentry(
    strategy_version: str,
    date_value: str,
    contract: base.ContractGrid,
    outcome: PositionOutcome,
    confirmation_index: int,
    budget: int,
    campaign: int,
    used_capacity: dict[int, int],
) -> tuple[list[dict[str, object]], int, int, int]:
    fill = base.next_traded_bar(
        contract,
        confirmation_index,
        base.ENTRY_WAIT_MINUTES,
    )
    if fill is None or budget <= 0:
        return [], 0, 0, 0
    entry_index, bar = fill
    assert bar.open is not None
    price = float(bar.open) + base.SLIPPAGE
    quantity = min(
        math.floor(budget / (price * base.MULTIPLIER)),
        math.floor(bar.volume * base.ENTRY_PARTICIPATION),
    )
    if quantity <= 0:
        return [], 0, 0, 0
    buy = trade_row(
        strategy_version,
        date_value,
        campaign,
        bar.minute,
        contract,
        "BUY",
        "LATE_1_5X_REENTRY",
        1,
        quantity,
        price,
        bar.volume,
    )
    remaining = quantity
    sold = 0
    trades: list[dict[str, object]] = [buy]
    thresholds = (
        (2.0, 0.50, "TARGET_2X_HALF", 2),
        (5.0, 0.75, "TARGET_5X_HALF_REMAINDER", 3),
        (10.0, 1.00, "TARGET_10X_ALL", 4),
    )
    achieved = 0.0
    index = entry_index
    while index < len(contract.bars) and remaining > 0:
        current = contract.bars[index]
        desired = None
        reason = ""
        stage = 0
        if current.traded and current.high is not None:
            multiple = (
                float(current.high) / outcome.setup.observation_price
            )
            for threshold, cumulative, name, number in thresholds:
                if multiple >= threshold and cumulative > achieved:
                    desired = cumulative
                    reason = name
                    stage = number
        if desired is not None:
            target_sold = (
                quantity
                if desired >= 1.0
                else math.ceil(quantity * desired)
            )
            request = min(remaining, max(0, target_sold - sold))
            achieved = desired
            if request > 0:
                filled, fill_index, sale_rows = execute_sale(
                    strategy_version,
                    date_value,
                    campaign,
                    contract,
                    index,
                    request,
                    reason,
                    stage,
                    used_capacity,
                )
                remaining -= filled
                sold += filled
                trades.extend(sale_rows)
                index = max(index + 1, fill_index)
                continue
        if current.minute >= FORCE_EXIT_TIME:
            filled, _, sale_rows = execute_sale(
                strategy_version,
                date_value,
                campaign,
                contract,
                index,
                remaining,
                "TIME",
                8,
                used_capacity,
            )
            remaining -= filled
            trades.extend(sale_rows)
            break
        index += 1
    if remaining > 0:
        trades.append(
            expire_trade(
                strategy_version,
                date_value,
                campaign,
                contract,
                remaining,
            )
        )
    return trades, int(buy["principal"]), bar.minute, remaining


def simulate_day(
    config_name: str,
    date_value: str,
    contracts: dict[str, base.ContractGrid],
    setups: list[Setup],
    start_cash: int,
    strategy_version_override: str | None = None,
    allow_reentry: bool = True,
) -> tuple[
    DayResult,
    list[dict[str, object]],
    list[PositionRecord],
]:
    strategy_version = (
        strategy_version_override
        or str(CONFIGS[config_name]["strategy"])
    )
    entries = allocate_entries(contracts, setups)
    used_capacity: dict[str, dict[int, int]] = defaultdict(dict)
    outcomes: list[PositionOutcome] = []
    trades: list[dict[str, object]] = []
    positions: list[PositionRecord] = []
    for campaign, entry in enumerate(entries, start=1):
        contract = contracts[entry.setup.code]
        outcome = simulate_initial_position(
            config_name,
            strategy_version,
            date_value,
            contract,
            entry,
            campaign,
            used_capacity[contract.code],
        )
        outcomes.append(outcome)
        trades.extend(outcome.trades)
        positions.append(
            PositionRecord(
                strategy=strategy_version,
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
                    contracts[entry.setup.code]
                    .bars[outcome.early_confirmation_index]
                    .minute
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
    failed_proceeds = sum(row.net_sale_proceeds for row in failed)
    confirmed_principal = sum(row.entry_principal for row in confirmed)
    late = late_candidate(contracts, failed) if allow_reentry else None
    late_trades: list[dict[str, object]] = []
    late_principal = 0
    late_entry_minute = 0
    if late is not None:
        outcome, confirmation_index = late
        planned_fill = base.next_traded_bar(
            contracts[outcome.setup.code],
            confirmation_index,
            base.ENTRY_WAIT_MINUTES,
        )
        planned_entry_minute = (
            planned_fill[1].minute if planned_fill is not None else 0
        )
        realized_failed_proceeds = sum(
            int(trade["cash_flow"])
            for failed_outcome in failed
            for trade in failed_outcome.trades
            if trade["side"] == "SELL"
            and int(trade["minute"]) < planned_entry_minute
        )
        reentry_budget = max(
            0,
            min(
                realized_failed_proceeds,
                INITIAL_BUDGET - confirmed_principal,
            ),
        )
        late_trades, late_principal, late_entry_minute, _ = (
            simulate_late_reentry(
                strategy_version,
                date_value,
                contracts[outcome.setup.code],
                outcome,
                confirmation_index,
                reentry_budget,
                len(entries) + 1,
                used_capacity[outcome.setup.code],
            )
        )
        trades.extend(late_trades)
        if late_trades:
            buy = next(
                row for row in late_trades if row["side"] == "BUY"
            )
            positions.append(
                PositionRecord(
                    strategy=strategy_version,
                    date=date_value,
                    campaign=len(entries) + 1,
                    code=outcome.setup.code,
                    call_put=outcome.setup.call_put,
                    branch="LATE_REENTRY",
                    observation_minute=(
                        outcome.setup.observation_minute
                    ),
                    observation_price=round(
                        outcome.setup.observation_price, 4
                    ),
                    confirmation_minute=contracts[
                        outcome.setup.code
                    ].bars[confirmation_index].minute,
                    entry_minute=int(buy["minute"]),
                    entry_price=round(float(buy["price"]), 4),
                    quantity=int(buy["quantity"]),
                    buy_principal=int(buy["principal"]),
                    buy_fee=int(buy["fee"]),
                    net_sale_proceeds=sum(
                        int(row["cash_flow"])
                        for row in late_trades
                        if row["side"] == "SELL"
                    ),
                    pnl=sum(
                        int(row["cash_flow"]) for row in late_trades
                    ),
                    expired_quantity=sum(
                        int(row["quantity"])
                        for row in late_trades
                        if row["side"] == "EXPIRE"
                    ),
                )
            )

    pnl = sum(int(row["cash_flow"]) for row in trades)
    total_fees = sum(int(row["fee"]) for row in trades)
    total_expired = sum(
        int(row["quantity"])
        for row in trades
        if row["side"] == "EXPIRE"
    )
    late_outcome = late[0] if late is not None else None
    late_confirmation = late[1] if late is not None else None
    return (
        DayResult(
            strategy=strategy_version,
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
            failed_sale_proceeds=failed_proceeds,
            late_signal_found=late is not None,
            late_reentry_filled=late_principal > 0,
            late_code=(
                late_outcome.setup.code if late_outcome else ""
            ),
            late_call_put=(
                late_outcome.setup.call_put if late_outcome else ""
            ),
            late_confirmation_minute=(
                contracts[late_outcome.setup.code]
                .bars[late_confirmation]
                .minute
                if late_outcome is not None
                and late_confirmation is not None
                else None
            ),
            late_entry_minute=late_entry_minute or None,
            late_buy_principal=late_principal,
            total_buy_principal=initial_principal + late_principal,
            maximum_deployed_principal=(
                confirmed_principal + late_principal
            ),
            total_fees=total_fees,
            expired_quantity=total_expired,
        ),
        trades,
        positions,
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
    traded = [row for row in rows if row.initial_positions > 0]
    pnl = [row.pnl for row in traded]
    ordered = sorted(pnl, reverse=True)
    gains = sum(value for value in pnl if value > 0)
    losses = -sum(value for value in pnl if value < 0)
    return {
        "days": len(rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in pnl),
        "losing_days": sum(value < 0 for value in pnl),
        "break_even_days": sum(value == 0 for value in pnl),
        "win_rate_pct": (
            round(sum(value > 0 for value in pnl) / len(pnl) * 100, 2)
            if pnl
            else 0.0
        ),
        "total_pnl": sum(pnl),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(pnl, default=0),
        "maximum_day_pnl": max(pnl, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": sum(pnl) - sum(ordered[:3]),
        "pnl_excluding_best_5_days": sum(pnl) - sum(ordered[:5]),
        "initial_buy_principal": sum(
            row.initial_buy_principal for row in traded
        ),
        "average_initial_buy_principal": (
            round(
                sum(row.initial_buy_principal for row in traded)
                / len(traded)
            )
            if traded
            else 0
        ),
        "early_confirmed_positions": sum(
            row.early_confirmed_positions for row in traded
        ),
        "early_failed_positions": sum(
            row.early_failed_positions for row in traded
        ),
        "late_signal_days": sum(row.late_signal_found for row in traded),
        "late_reentry_days": sum(
            row.late_reentry_filled for row in traded
        ),
        "late_buy_principal": sum(
            row.late_buy_principal for row in traded
        ),
        "expired_quantity": sum(
            row.expired_quantity for row in traded
        ),
    }


def summarize_positions(
    rows: list[PositionRecord],
) -> list[dict[str, object]]:
    grouped: dict[str, list[PositionRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.branch].append(row)
    result: list[dict[str, object]] = []
    for branch, branch_rows in sorted(grouped.items()):
        principal = sum(row.buy_principal for row in branch_rows)
        pnl = sum(row.pnl for row in branch_rows)
        result.append(
            {
                "group": branch,
                "count": len(branch_rows),
                "profitable_count": sum(
                    row.pnl > 0 for row in branch_rows
                ),
                "losing_count": sum(
                    row.pnl < 0 for row in branch_rows
                ),
                "total_buy_principal": principal,
                "total_pnl": pnl,
                "expected_net_return_pct": (
                    round(pnl / principal * 100, 2)
                    if principal
                    else 0.0
                ),
                "average_pnl": (
                    round(pnl / len(branch_rows))
                    if branch_rows
                    else 0
                ),
            }
        )
    return result


def raw_event_validation(
    rows: list[tuple[str, Setup, base.ContractGrid]],
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    for date_value, setup, contract in rows:
        first_bar = contract.bars[setup.observation_index]
        ambiguous = (
            first_bar.high is not None
            and float(first_bar.high)
            >= setup.observation_price * legacy.CONFIRMATION_MULTIPLE
        )
        deadline = min(
            len(contract.bars) - 1,
            setup.observation_index
            + legacy.CONFIRMATION_WINDOW_MINUTES,
        )
        confirmation = first_high_touch(
            contract,
            setup.observation_index + 1,
            contract.bars[deadline].minute,
            setup.observation_price * legacy.CONFIRMATION_MULTIPLE,
        )
        highs = [
            float(bar.high)
            for bar in contract.bars[setup.observation_index + 1 :]
            if bar.traded and bar.high is not None
        ]
        peak = (
            max(highs, default=setup.observation_price)
            / setup.observation_price
        )
        events.append(
            {
                "date": date_value,
                "code": setup.code,
                "ambiguous": ambiguous,
                "confirmed": confirmation is not None,
                "peak_multiple": peak,
            }
        )
    ordered = [row for row in events if not bool(row["ambiguous"])]
    confirmed = [row for row in ordered if bool(row["confirmed"])]
    peaks = [float(row["peak_multiple"]) for row in confirmed]
    buckets = {
        "1_5_TO_2": sum(1.5 <= value < 2.0 for value in peaks),
        "2_TO_5": sum(2.0 <= value < 5.0 for value in peaks),
        "5_TO_10": sum(5.0 <= value < 10.0 for value in peaks),
        "10_PLUS": sum(value >= 10.0 for value in peaks),
    }
    return {
        "definition": (
            "first completed 09:00~09:10 bar with a qualifying OHLC; "
            "maximum qualifying OHLC is P0"
        ),
        "eligible": len(events),
        "same_bar_ambiguous": sum(
            bool(row["ambiguous"]) for row in events
        ),
        "ordered_confirmed": len(confirmed),
        "ordered_failed": len(ordered) - len(confirmed),
        "confirmed_peak_buckets": buckets,
        "confirmed_peak_median": (
            round(median(peaks), 4) if peaks else 0.0
        ),
        "maximum_peak_multiple": round(max(peaks, default=0.0), 4),
        "matches_ppt_exactly": (
            len(events) == PPT_TOTAL
            and len(confirmed) == PPT_EARLY
            and sum(bool(row["ambiguous"]) for row in events)
            == PPT_AMBIGUOUS
            and len(ordered) - len(confirmed) == PPT_FAILED
            and buckets == PPT_BUCKETS
        ),
    }


def audit(
    daily: list[DayResult],
    trades: list[dict[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        by_date[str(trade["date"])].append(trade)
    expected_cash = base.INITIAL_CASH
    for row in daily:
        day_trades = by_date[row.date]
        if row.start_cash != expected_cash:
            errors.append(f"{row.date}: cash chain")
        if row.end_cash != row.start_cash + row.pnl:
            errors.append(f"{row.date}: pnl equation")
        if sum(int(item["cash_flow"]) for item in day_trades) != row.pnl:
            errors.append(f"{row.date}: trade cash flow")
        initial = [
            item
            for item in day_trades
            if item["side"] == "BUY"
            and item["reason"] == "09_10_PREMIUM_BIN_BASKET"
        ]
        if sum(int(item["principal"]) for item in initial) != (
            row.initial_buy_principal
        ):
            errors.append(f"{row.date}: initial principal")
        if row.initial_buy_principal > INITIAL_BUDGET:
            errors.append(f"{row.date}: initial budget")
        if row.maximum_deployed_principal > INITIAL_BUDGET:
            errors.append(f"{row.date}: deployed budget")
        if row.late_buy_principal > row.failed_sale_proceeds:
            errors.append(f"{row.date}: recycled proceeds")
        quantities: dict[tuple[str, int], int] = defaultdict(int)
        for trade in day_trades:
            quantity = int(trade["quantity"])
            key = (str(trade["code"]), int(trade["campaign"]))
            if trade["side"] == "BUY":
                quantities[key] += quantity
                if quantity > math.floor(
                    int(trade["bar_volume"])
                    * base.ENTRY_PARTICIPATION
                ):
                    errors.append(f"{row.date}: entry participation")
            elif trade["side"] in {"SELL", "EXPIRE"}:
                quantities[key] -= quantity
            else:
                errors.append(f"{row.date}: side")
        if any(quantity != 0 for quantity in quantities.values()):
            errors.append(f"{row.date}: open quantity")
        expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:40],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
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
        if legacy.PPT_START
        <= group["expiry_date"]
        <= legacy.PPT_END
    ]
    files = base.target_file_map()
    cash = {name: base.INITIAL_CASH for name in CONFIGS}
    daily: dict[str, list[DayResult]] = {
        name: [] for name in CONFIGS
    }
    trades: dict[str, list[dict[str, object]]] = {
        name: [] for name in CONFIGS
    }
    positions: dict[str, list[PositionRecord]] = {
        name: [] for name in CONFIGS
    }
    event_rows: list[tuple[str, Setup, base.ContractGrid]] = []

    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        setups = [
            setup
            for contract in contracts.values()
            if (setup := observation(contract)) is not None
        ]
        event_rows.extend(
            (date_value, setup, contracts[setup.code])
            for setup in setups
        )
        for config_name in CONFIGS:
            row, day_trades, day_positions = simulate_day(
                config_name,
                date_value,
                contracts,
                setups,
                cash[config_name],
            )
            cash[config_name] = row.end_cash
            daily[config_name].append(row)
            trades[config_name].extend(day_trades)
            positions[config_name].extend(day_positions)

    raw_validation = raw_event_validation(event_rows)
    ppt_weighted_floor_sum = sum(
        PPT_BUCKETS[name] * PPT_BUCKET_FLOORS[name]
        for name in PPT_BUCKETS
    )
    expectation = {
        "warning": (
            "PPT buckets describe ex-post intraday maxima, not "
            "executable sale prices."
        ),
        "conditional_early_peak_floor_multiple": round(
            PPT_EXPECTED_PEAK_FLOOR, 6
        ),
        "all_1320_oracle_floor_multiple": round(
            ppt_weighted_floor_sum / PPT_TOTAL,
            6,
        ),
        "ordered_1283_oracle_floor_multiple": round(
            ppt_weighted_floor_sum
            / (PPT_TOTAL - PPT_AMBIGUOUS),
            6,
        ),
        "all_or_zero_target_multiples_on_ordered_1283": {
            "1_5x": round(PPT_EARLY * 1.5 / 1_283, 6),
            "2x": round(
                (
                    PPT_BUCKETS["2_TO_5"]
                    + PPT_BUCKETS["5_TO_10"]
                    + PPT_BUCKETS["10_PLUS"]
                )
                * 2.0
                / 1_283,
                6,
            ),
            "5x": round(
                (
                    PPT_BUCKETS["5_TO_10"]
                    + PPT_BUCKETS["10_PLUS"]
                )
                * 5.0
                / 1_283,
                6,
            ),
            "10x": round(
                PPT_BUCKETS["10_PLUS"] * 10.0 / 1_283,
                6,
            ),
        },
    }

    portfolios: dict[str, object] = {}
    audits: dict[str, object] = {}
    for name, config in CONFIGS.items():
        all_rows = daily[name]
        development = [
            row
            for row in all_rows
            if row.date <= legacy.DEVELOPMENT_END
        ]
        validation = [
            row
            for row in all_rows
            if row.date > legacy.DEVELOPMENT_END
        ]
        audit_result = audit(all_rows, trades[name])
        audits[name] = audit_result
        portfolios[name] = {
            "strategy_version": config["strategy"],
            "rules": config,
            "summary": {
                **summarize(all_rows),
                "initial_cash": base.INITIAL_CASH,
                "ending_cash": cash[name],
            },
            "development": summarize(development),
            "validation": summarize(validation),
            "worst_days": [
                asdict(row)
                for row in sorted(
                    all_rows, key=lambda item: item.pnl
                )[:15]
            ],
            "position_expectation": summarize_positions(
                positions[name]
            ),
            "audit": audit_result,
        }
        suffix = name.lower()
        write_csv(
            OUTPUT_DIR / f"strategy5_v04_{suffix}_daily.csv",
            [asdict(row) for row in all_rows],
        )
        write_csv(
            OUTPUT_DIR / f"strategy5_v04_{suffix}_trades.csv",
            trades[name],
        )
        write_csv(
            OUTPUT_DIR / f"strategy5_v04_{suffix}_positions.csv",
            [asdict(row) for row in positions[name]],
        )

    result = {
        "status": "experimental_09_10_basket_and_reentry",
        "strategy_version": STRATEGY_VERSION,
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": legacy.DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "ppt_claim": {
            "eligible": PPT_TOTAL,
            "early_1_5x": PPT_EARLY,
            "same_bar_ambiguous": PPT_AMBIGUOUS,
            "failed_1_5x": PPT_FAILED,
            "early_peak_buckets": PPT_BUCKETS,
        },
        "expectation": expectation,
        "raw_validation": raw_validation,
        "rules": {
            "observation": (
                "first completed 09:00~09:10 bar containing a 0.40~1.20 "
                "OHLC; maximum qualifying OHLC is P0"
            ),
            "entry": (
                "after 09:10, split 5,000,000 won equally across active "
                "0.1 premium bins and as equally as integer contracts allow"
            ),
            "early_confirmation": (
                "post-fill 1.5x P0 high touch by 09:30"
            ),
            "failed_action": (
                "sell the remaining initial position after 09:30"
            ),
            "reentry": (
                "reuse only failed-position net proceeds realized before "
                "the reentry fill; choose the earliest post-exit 1.5x "
                "confirmation through 14:30"
            ),
            "reentry_exit": (
                "existing S5-v0.2 2x half, 5x half remainder, 10x all"
            ),
            "maximum_simultaneous_principal": INITIAL_BUDGET,
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
        "portfolios": portfolios,
        "audit": {
            "passed": all(
                bool(dict(value)["passed"])
                for value in audits.values()
            ),
            "error_count": sum(
                int(dict(value)["error_count"])
                for value in audits.values()
            ),
            "portfolios": audits,
        },
        "limitations": [
            (
                "The current raw-data parser does not reproduce the PPT "
                "1,320/633/37/650 event counts exactly."
            ),
            (
                "The 3.9076x value is a lower-end weighted ex-post peak, "
                "not an executable expected sale price."
            ),
            (
                "Both exit candidates and the reentry rule were designed "
                "after reading the full PPT period."
            ),
            (
                "The 09:10 basket may underinvest because quantities are "
                "integer and each fill is capped at 10% of bar volume."
            ),
        ],
    }
    (OUTPUT_DIR / "strategy5_v04_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy5_v04_audit.json").write_text(
        json.dumps(result["audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["audit"]["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
