from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import minimal_baseline_backtest as base


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S5-v0.1"
PPT_START = "20260105"
PPT_END = "20260831"
DEVELOPMENT_END = "20260430"

MORNING_START = 900
MORNING_END = 910
PREMIUM_MIN = 0.40
PREMIUM_MAX = 1.20
CONFIRMATION_MULTIPLE = 1.5
CONFIRMATION_WINDOW_MINUTES = 30
TARGET_MULTIPLES = (2.0, 5.0, 10.0)
FORCE_EXIT_TIME = 1515


@dataclass(frozen=True)
class MorningSignal:
    code: str
    index: int
    minute: int
    observed_price: float
    minute_volume: int
    cumulative_volume: int


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
    signal_minute: int | None
    observed_price: float
    entry_minute: int | None
    initial_price: float
    entry_quantity: int
    buy_principal: int
    buy_fee: int
    confirmed: bool
    confirmation_minute: int | None
    reached_2x: bool
    reached_5x: bool
    reached_10x: bool
    gross_sales: int
    sell_fee: int
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float


@dataclass(frozen=True)
class Sale:
    filled: int
    gross_sales: int
    fees: int
    last_fill_index: int
    trades: list[dict[str, object]]


def observed_values(bar: base.MinuteBar) -> list[float]:
    values = (bar.open, bar.high, bar.low, bar.close)
    return [
        float(value)
        for value in values
        if value is not None and PREMIUM_MIN <= float(value) <= PREMIUM_MAX
    ]


def first_morning_signal(
    contracts: dict[str, base.ContractGrid],
) -> MorningSignal | None:
    start_index = base.MINUTE_INDEX[MORNING_START]
    end_index = base.MINUTE_INDEX[MORNING_END]
    for index in range(start_index, end_index + 1):
        candidates: list[MorningSignal] = []
        for contract in contracts.values():
            bar = contract.bars[index]
            if not bar.traded:
                continue
            values = observed_values(bar)
            if not values:
                continue
            cumulative_volume = sum(
                item.volume
                for item in contract.bars[start_index : index + 1]
            )
            candidates.append(
                MorningSignal(
                    code=contract.code,
                    index=index,
                    minute=bar.minute,
                    observed_price=max(values),
                    minute_volume=bar.volume,
                    cumulative_volume=cumulative_volume,
                )
            )
        if candidates:
            return sorted(
                candidates,
                key=lambda item: (
                    -item.minute_volume,
                    -item.cumulative_volume,
                    item.code,
                ),
            )[0]
    return None


def buy_after_signal(
    date_value: str,
    contract: base.ContractGrid,
    signal: MorningSignal,
    available_cash: int,
) -> tuple[int, int, int, int, dict[str, object]] | None:
    fill = base.next_traded_bar(
        contract,
        signal.index,
        base.ENTRY_WAIT_MINUTES,
    )
    if fill is None:
        return None
    fill_index, bar = fill
    assert bar.open is not None
    if not PREMIUM_MIN <= float(bar.open) <= PREMIUM_MAX:
        return None
    fill_price = float(bar.open) + base.SLIPPAGE
    requested = math.floor(
        base.DAILY_PRINCIPAL_LIMIT
        / ((signal.observed_price + base.SLIPPAGE) * base.MULTIPLIER)
    )
    by_actual = math.floor(
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
    quantity = min(requested, by_actual, by_cash, by_volume)
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
            "strategy": STRATEGY_VERSION,
            "date": date_value,
            "campaign": 1,
            "minute": bar.minute,
            "code": contract.code,
            "call_put": contract.call_put,
            "side": "BUY",
            "reason": "MORNING_RANGE_ENTRY",
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
) -> Sale:
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
                "strategy": STRATEGY_VERSION,
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
    return Sale(
        filled=requested_quantity - remaining,
        gross_sales=gross_sales,
        fees=fees,
        last_fill_index=last_fill_index,
        trades=trades,
    )


def empty_day(
    date_value: str,
    start_cash: int,
    signal: MorningSignal | None = None,
    call_put: str = "",
) -> DayResult:
    return DayResult(
        strategy=STRATEGY_VERSION,
        date=date_value,
        start_cash=start_cash,
        end_cash=start_cash,
        pnl=0,
        signal_found=signal is not None,
        entry_filled=False,
        code=signal.code if signal else "",
        call_put=call_put,
        signal_minute=signal.minute if signal else None,
        observed_price=signal.observed_price if signal else 0.0,
        entry_minute=None,
        initial_price=0.0,
        entry_quantity=0,
        buy_principal=0,
        buy_fee=0,
        confirmed=False,
        confirmation_minute=None,
        reached_2x=False,
        reached_5x=False,
        reached_10x=False,
        gross_sales=0,
        sell_fee=0,
        exit_reason="",
        exit_minute=None,
        expired_quantity=0,
        mfe_multiple=0.0,
        mae_pct=0.0,
    )


def confirmation_index(
    contract: base.ContractGrid,
    signal_index: int,
    entry_index: int,
    observed_price: float,
) -> int | None:
    deadline = min(
        len(contract.bars) - 1,
        signal_index + CONFIRMATION_WINDOW_MINUTES,
    )
    for index in range(entry_index + 1, deadline + 1):
        bar = contract.bars[index]
        if (
            bar.traded
            and bar.high is not None
            and float(bar.high)
            >= observed_price * CONFIRMATION_MULTIPLE
        ):
            return index
    return None


def excursion(
    contract: base.ContractGrid,
    entry_index: int,
    initial_price: float,
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
    mfe = max(highs, default=initial_price) / initial_price
    mae = (
        (min(lows, default=initial_price) / initial_price - 1.0)
        * 100
    )
    return round(mfe, 6), round(mae, 4)


def simulate_confirmed_position(
    date_value: str,
    contract: base.ContractGrid,
    confirmed_index: int,
    observed_price: float,
    initial_quantity: int,
) -> tuple[
    int,
    int,
    int,
    str,
    int,
    bool,
    bool,
    bool,
    list[dict[str, object]],
]:
    quantity = initial_quantity
    gross_sales = 0
    sell_fee = 0
    trades: list[dict[str, object]] = []
    reached_2x = False
    reached_5x = False
    reached_10x = False
    exit_reason = ""
    exit_index = confirmed_index
    index = confirmed_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        reason = ""
        request = 0
        if bar.traded and bar.high is not None:
            high = float(bar.high)
            if not reached_2x and high >= observed_price * 2.0:
                reached_2x = True
                reason = "TARGET_2X_HALF"
                request = max(1, math.ceil(quantity / 2))
            elif (
                reached_2x
                and not reached_5x
                and high >= observed_price * 5.0
            ):
                reached_5x = True
                reason = "TARGET_5X_HALF_REMAINDER"
                request = max(1, math.ceil(quantity / 2))
            elif (
                reached_5x
                and not reached_10x
                and high >= observed_price * 10.0
            ):
                reached_10x = True
                reason = "TARGET_10X_ALL"
                request = quantity

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
            request = quantity
        if not reason:
            index += 1
            continue

        sale = sell_after_signal(
            date_value,
            contract,
            index,
            request,
            reason,
        )
        quantity -= sale.filled
        gross_sales += sale.gross_sales
        sell_fee += sale.fees
        trades.extend(sale.trades)
        exit_index = sale.last_fill_index
        exit_reason = reason
        if sale.filled < request:
            break
        index = sale.last_fill_index

    return (
        quantity,
        gross_sales,
        sell_fee,
        exit_reason,
        exit_index,
        reached_2x,
        reached_5x,
        reached_10x,
        trades,
    )


def simulate_day(
    date_value: str,
    contracts: dict[str, base.ContractGrid],
    signal: MorningSignal | None,
    start_cash: int,
) -> tuple[DayResult, list[dict[str, object]]]:
    if signal is None:
        return empty_day(date_value, start_cash), []
    contract = contracts[signal.code]
    fill = buy_after_signal(
        date_value,
        contract,
        signal,
        start_cash,
    )
    if fill is None:
        return (
            empty_day(
                date_value,
                start_cash,
                signal,
                contract.call_put,
            ),
            [],
        )
    (
        entry_index,
        initial_quantity,
        buy_principal,
        buy_fee,
        buy_trade,
    ) = fill
    initial_price = float(buy_trade["price"])
    trades = [buy_trade]
    confirmed_index = confirmation_index(
        contract,
        signal.index,
        entry_index,
        signal.observed_price,
    )

    if confirmed_index is None:
        deadline = min(
            len(contract.bars) - 1,
            signal.index + CONFIRMATION_WINDOW_MINUTES,
        )
        sale = sell_after_signal(
            date_value,
            contract,
            deadline,
            initial_quantity,
            "NO_1_5X_30M",
        )
        remaining = initial_quantity - sale.filled
        gross_sales = sale.gross_sales
        sell_fee = sale.fees
        trades.extend(sale.trades)
        exit_reason = "NO_1_5X_30M"
        exit_index = sale.last_fill_index
        reached_2x = False
        reached_5x = False
        reached_10x = False
    else:
        (
            remaining,
            gross_sales,
            sell_fee,
            exit_reason,
            exit_index,
            reached_2x,
            reached_5x,
            reached_10x,
            position_trades,
        ) = simulate_confirmed_position(
            date_value,
            contract,
            confirmed_index,
            signal.observed_price,
            initial_quantity,
        )
        trades.extend(position_trades)

    if remaining > 0:
        trades.append(
            {
                "strategy": STRATEGY_VERSION,
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
        initial_price,
    )
    return (
        DayResult(
            strategy=STRATEGY_VERSION,
            date=date_value,
            start_cash=start_cash,
            end_cash=start_cash + pnl,
            pnl=pnl,
            signal_found=True,
            entry_filled=True,
            code=contract.code,
            call_put=contract.call_put,
            signal_minute=signal.minute,
            observed_price=round(signal.observed_price, 4),
            entry_minute=contract.bars[entry_index].minute,
            initial_price=round(initial_price, 4),
            entry_quantity=initial_quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            confirmed=confirmed_index is not None,
            confirmation_minute=(
                contract.bars[confirmed_index].minute
                if confirmed_index is not None
                else None
            ),
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
    pnl = [row.pnl for row in traded]
    ordered = sorted(pnl, reverse=True)
    gains = sum(value for value in pnl if value > 0)
    losses = -sum(value for value in pnl if value < 0)
    return {
        "days": len(rows),
        "signal_days": sum(row.signal_found for row in rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in pnl),
        "losing_days": sum(value < 0 for value in pnl),
        "break_even_days": sum(value == 0 for value in pnl),
        "total_pnl": sum(pnl),
        "average_trade_day_pnl": (
            round(sum(pnl) / len(pnl)) if pnl else 0
        ),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(pnl, default=0),
        "maximum_day_pnl": max(pnl, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": sum(pnl) - sum(ordered[:3]),
        "pnl_excluding_best_5_days": sum(pnl) - sum(ordered[:5]),
        "total_buy_principal": sum(
            row.buy_principal for row in traded
        ),
        "expired_contracts": sum(
            row.expired_quantity for row in traded
        ),
        "confirmed_days": sum(row.confirmed for row in traded),
        "reached_2x_days": sum(row.reached_2x for row in traded),
        "reached_5x_days": sum(row.reached_5x for row in traded),
        "reached_10x_days": sum(row.reached_10x for row in traded),
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
            errors.append(f"{row.date}: cash flow")
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
            errors.append(f"{row.date}: quantity")
        if row.signal_minute is not None and not (
            MORNING_START <= row.signal_minute <= MORNING_END
        ):
            errors.append(f"{row.date}: signal time")
        if (
            row.entry_minute is not None
            and row.signal_minute is not None
            and row.entry_minute <= row.signal_minute
        ):
            errors.append(f"{row.date}: non-causal entry")
        if (
            row.confirmation_minute is not None
            and row.entry_minute is not None
            and row.confirmation_minute <= row.entry_minute
        ):
            errors.append(f"{row.date}: non-causal confirmation")
        expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:20],
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
        if PPT_START <= group["expiry_date"] <= PPT_END
    ]
    if not groups:
        raise RuntimeError("PPT 기간 만기일을 찾을 수 없습니다.")
    files = base.target_file_map()
    cash = base.INITIAL_CASH
    daily: list[DayResult] = []
    trades: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        signal = first_morning_signal(contracts)
        row, day_trades = simulate_day(
            date_value,
            contracts,
            signal,
            cash,
        )
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)

    development = [
        row for row in daily if row.date <= DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > DEVELOPMENT_END
    ]
    summary = summarize(daily)
    summary.update(
        {
            "initial_cash": base.INITIAL_CASH,
            "ending_cash": cash,
        }
    )
    traded = [row for row in daily if row.entry_filled]
    audit_result = audit(daily, trades)
    result = {
        "status": "experimental_ppt_derived_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "source": {
            "document": (
                "위클리옵션_만기일_가격경로_연구결과_"
                "2026_최종보완본2.pptx"
            ),
            "signal_inputs": [
                "09:00~09:10 option OHLC",
                "30-minute 1.5x high touch",
                "2x, 5x, 10x high touches",
            ],
            "excluded_inputs": [
                "LSMA",
                "moving averages",
                "VIX",
                "futures direction",
            ],
        },
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "rules": {
            "morning_observation_window": [
                MORNING_START,
                MORNING_END,
            ],
            "premium_range": [PREMIUM_MIN, PREMIUM_MAX],
            "within_bar_observed_price": (
                "maximum qualifying OHLC value after the bar completes"
            ),
            "contract_selection": (
                "earliest qualifying minute, then highest minute volume"
            ),
            "maximum_campaigns_per_day": 1,
            "daily_principal_limit": base.DAILY_PRINCIPAL_LIMIT,
            "entry": "next actual traded bar after completed signal bar",
            "confirmation_multiple": CONFIRMATION_MULTIPLE,
            "confirmation_window_minutes": (
                CONFIRMATION_WINDOW_MINUTES
            ),
            "unconfirmed_exit": (
                "sell all after 30-minute confirmation deadline"
            ),
            "confirmed_exit": {
                "2x": "sell half",
                "5x": "sell half of remainder",
                "10x": "sell all remainder",
                "otherwise": "sell all after 15:15",
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
        "summary": summary,
        "development": summarize(development),
        "validation": summarize(validation),
        "analysis": {
            "by_confirmation": grouped(traded, "confirmed"),
            "by_call_put": grouped(traded, "call_put"),
            "by_exit_reason": grouped(traded, "exit_reason"),
            "worst_days": [
                asdict(row)
                for row in sorted(traded, key=lambda item: item.pnl)[:15]
            ],
        },
        "audit": audit_result,
        "limitations": [
            (
                "The PPT used all 2026-01~08 observations to derive the "
                "rule, so the displayed temporal split is not an "
                "independent holdout."
            ),
            (
                "The PPT did not define one tradable contract per day; "
                "earliest time and highest volume are deterministic "
                "execution tie-breakers."
            ),
            (
                "The PPT counted OHLC touches. This backtest acts only on "
                "the next actual traded bar with slippage and 10% volume "
                "participation."
            ),
            (
                "No LSMA, moving average, VIX, or futures direction is "
                "used."
            ),
        ],
    }

    write_csv(
        OUTPUT_DIR / "strategy5_v01_daily.csv",
        [asdict(row) for row in daily],
    )
    write_csv(OUTPUT_DIR / "strategy5_v01_trades.csv", trades)
    (OUTPUT_DIR / "strategy5_v01_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy5_v01_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
