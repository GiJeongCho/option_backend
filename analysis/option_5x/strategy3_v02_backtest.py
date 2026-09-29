from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy2_probe_research as execution
from strategy1_backtest import (
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    previous_volatility,
    volatility_closes,
)
from strategy2_backtest import audit


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S3-v0.2"
HIGH_VOL_START = "20251001"
DEVELOPMENT_END = "20260430"
FIRST_ENTRY_CUTOFF = 1400
LATER_ENTRY_CUTOFF = 1430
PREMIUM_MIN = 1.00
PREMIUM_MAX = 5.00
LOW_LOOKBACK = 60
CROSS_CONFIRMATION_WAIT = 5
MINIMUM_CROSS_GAP = 8
LOW_BREAK_TICK = 0.01
TRANCHE_BUDGETS = (1_666_666, 1_666_667, 1_666_667)

Direction = Literal["up", "down"]


@dataclass(frozen=True)
class RuntimeConfig:
    @property
    def key(self) -> str:
        return STRATEGY_VERSION


CONFIG = RuntimeConfig()


@dataclass(frozen=True)
class CrossEvent:
    raw_index: int
    confirmation_index: int


@dataclass(frozen=True)
class Setup:
    code: str
    event: CrossEvent
    first_low: float
    first_low_index: int
    volume_ratio: float
    recent_volume: int


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
    second_low: float
    first_golden_cross_minute: int | None
    lower_low_minute: int | None
    second_golden_cross_minute: int | None
    first_death_cross_minute: int | None
    second_death_cross_minute: int | None
    entry_stages: int
    entry_quantity: int
    average_entry_price: float
    buy_principal: int
    buy_fee: int
    gross_sales: int
    sell_fee: int
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float


def relation(
    contract: base.ContractGrid,
    index: int,
    direction: Direction,
) -> bool:
    ma5 = contract.ma5[index]
    ma20 = contract.ma20[index]
    if ma5 is None or ma20 is None:
        return False
    return (
        float(ma5) > float(ma20)
        if direction == "up"
        else float(ma5) < float(ma20)
    )


def cross_events(
    contract: base.ContractGrid, direction: Direction
) -> list[CrossEvent]:
    events: list[CrossEvent] = []
    for raw_index in range(20, len(contract.bars) - 1):
        if not contract.bars[raw_index].traded:
            continue
        previous_relation = relation(contract, raw_index - 1, direction)
        current_relation = relation(contract, raw_index, direction)
        if previous_relation or not current_relation:
            continue
        end = min(
            len(contract.bars) - 1,
            raw_index + CROSS_CONFIRMATION_WAIT,
        )
        for confirmation_index in range(raw_index + 1, end + 1):
            if not contract.bars[confirmation_index].traded:
                continue
            if relation(contract, confirmation_index, direction):
                events.append(
                    CrossEvent(raw_index, confirmation_index)
                )
            break
    return events


def prior_low(
    contract: base.ContractGrid, index: int
) -> tuple[float, int] | None:
    start = max(0, index - LOW_LOOKBACK)
    candidates = [
        (float(bar.low), position)
        for position, bar in enumerate(
            contract.bars[start:index], start=start
        )
        if bar.traded
        and bar.low is not None
        and PREMIUM_MIN <= float(bar.low) <= PREMIUM_MAX
    ]
    return min(candidates, default=None, key=lambda item: (item[0], -item[1]))


def volume_ratio(contract: base.ContractGrid, index: int) -> float:
    average = (
        sum(bar.volume for bar in contract.bars[index - 20 : index]) / 20
    )
    if average <= 0:
        return float("inf") if contract.bars[index].volume > 0 else 0.0
    return contract.bars[index].volume / average


def first_setup(
    contracts: dict[str, base.ContractGrid],
) -> Setup | None:
    candidates: list[Setup] = []
    for contract in contracts.values():
        for event in cross_events(contract, "up"):
            bar = contract.bars[event.confirmation_index]
            if event.confirmation_index < LOW_LOOKBACK:
                continue
            if bar.minute > FIRST_ENTRY_CUTOFF or bar.close is None:
                break
            if not PREMIUM_MIN <= float(bar.close) <= PREMIUM_MAX:
                continue
            low = prior_low(contract, event.confirmation_index)
            if low is None:
                continue
            candidates.append(
                Setup(
                    code=contract.code,
                    event=event,
                    first_low=low[0],
                    first_low_index=low[1],
                    volume_ratio=volume_ratio(
                        contract, event.confirmation_index
                    ),
                    recent_volume=sum(
                        item.volume
                        for item in contract.bars[
                            event.confirmation_index
                            - 19 : event.confirmation_index
                            + 1
                        ]
                    ),
                )
            )
            break
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item.event.confirmation_index,
            -item.volume_ratio,
            -item.recent_volume,
            item.code,
        ),
    )[0]


def first_lower_low(
    contract: base.ContractGrid,
    start_index: int,
    first_low: float,
) -> tuple[int, float] | None:
    for index in range(start_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if bar.minute > LATER_ENTRY_CUTOFF:
            return None
        if (
            bar.traded
            and bar.low is not None
            and bar.close is not None
            and float(bar.low) <= first_low - LOW_BREAK_TICK
            and PREMIUM_MIN <= float(bar.close) <= PREMIUM_MAX
        ):
            return index, float(bar.low)
    return None


def first_event_after(
    events: list[CrossEvent],
    confirmation_after: int,
    raw_after: int | None = None,
    raw_gap_from: int | None = None,
) -> CrossEvent | None:
    for event in events:
        if event.confirmation_index <= confirmation_after:
            continue
        if raw_after is not None and event.raw_index <= raw_after:
            continue
        if (
            raw_gap_from is not None
            and event.raw_index - raw_gap_from < MINIMUM_CROSS_GAP
        ):
            continue
        return event
    return None


def buy_stage(
    date_value: str,
    contract: base.ContractGrid,
    signal_index: int,
    stage: int,
    budget: int,
    start_cash: int,
    buy_principal: int,
    buy_fee: int,
) -> tuple[
    int,
    int,
    int,
    int,
    dict[str, object],
] | None:
    bar = contract.bars[signal_index]
    if bar.close is None:
        return None
    remaining_principal = DAILY_PRINCIPAL_LIMIT - buy_principal
    allowed_budget = min(budget, remaining_principal)
    if allowed_budget <= 0:
        return None
    fill = execution.buy(
        date_value,
        CONFIG,  # type: ignore[arg-type]
        contract,
        signal_index,
        float(bar.close),
        allowed_budget,
        start_cash - buy_principal - buy_fee,
        f"DOUBLE_BOTTOM_STAGE_{stage}",
    )
    if fill is None:
        return None
    fill_index, quantity, principal, fee, trade = fill
    trade["stage"] = stage
    trade["call_put"] = contract.call_put
    return fill_index, quantity, principal, fee, trade


def empty_day(
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool,
) -> DayResult:
    return DayResult(
        strategy=STRATEGY_VERSION,
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
        second_low=0.0,
        first_golden_cross_minute=None,
        lower_low_minute=None,
        second_golden_cross_minute=None,
        first_death_cross_minute=None,
        second_death_cross_minute=None,
        entry_stages=0,
        entry_quantity=0,
        average_entry_price=0.0,
        buy_principal=0,
        buy_fee=0,
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
    start_index: int,
    average_entry_price: float,
) -> tuple[float, float]:
    maximum = 1.0
    minimum_pct = 0.0
    for bar in contract.bars[start_index:]:
        if not bar.traded:
            continue
        if bar.high is not None:
            maximum = max(
                maximum, float(bar.high) / average_entry_price
            )
        if bar.low is not None:
            minimum_pct = min(
                minimum_pct,
                (float(bar.low) / average_entry_price - 1.0) * 100,
            )
    return maximum, minimum_pct


def simulate_day(
    date_value: str,
    previous_vix: float,
    contracts: dict[str, base.ContractGrid],
    setup: Setup | None,
    start_cash: int,
) -> tuple[DayResult, list[dict[str, object]]]:
    if setup is None:
        return empty_day(
            date_value, previous_vix, start_cash, False
        ), []
    contract = contracts[setup.code]
    first_fill = buy_stage(
        date_value,
        contract,
        setup.event.confirmation_index,
        1,
        TRANCHE_BUDGETS[0],
        start_cash,
        0,
        0,
    )
    if first_fill is None:
        return empty_day(
            date_value, previous_vix, start_cash, True
        ), []

    (
        first_fill_index,
        first_quantity,
        first_principal,
        first_fee,
        first_trade,
    ) = first_fill
    trades = [first_trade]
    quantity = first_quantity
    buy_principal = first_principal
    buy_fee = first_fee
    stages = 1
    last_entry_index = first_fill_index

    lower_low = first_lower_low(
        contract, first_fill_index, setup.first_low
    )
    lower_low_index = lower_low[0] if lower_low else None
    second_low = lower_low[1] if lower_low else 0.0
    if lower_low is not None:
        second_fill = buy_stage(
            date_value,
            contract,
            lower_low_index,
            2,
            TRANCHE_BUDGETS[1],
            start_cash,
            buy_principal,
            buy_fee,
        )
        if second_fill is not None:
            (
                fill_index,
                fill_quantity,
                principal,
                fee,
                trade,
            ) = second_fill
            quantity += fill_quantity
            buy_principal += principal
            buy_fee += fee
            stages += 1
            last_entry_index = max(last_entry_index, fill_index)
            trades.append(trade)

    up_events = cross_events(contract, "up")
    down_events = cross_events(contract, "down")
    first_death = first_event_after(
        down_events,
        first_fill_index,
        raw_after=setup.event.raw_index,
    )
    second_golden: CrossEvent | None = None
    if lower_low_index is not None and first_death is not None:
        second_golden = first_event_after(
            up_events,
            max(lower_low_index, first_death.confirmation_index),
            raw_after=first_death.raw_index,
            raw_gap_from=setup.event.raw_index,
        )
        if second_golden is not None:
            signal_bar = contract.bars[
                second_golden.confirmation_index
            ]
            if (
                signal_bar.minute <= LATER_ENTRY_CUTOFF
                and signal_bar.close is not None
                and PREMIUM_MIN
                <= float(signal_bar.close)
                <= PREMIUM_MAX
            ):
                third_fill = buy_stage(
                    date_value,
                    contract,
                    second_golden.confirmation_index,
                    3,
                    TRANCHE_BUDGETS[2],
                    start_cash,
                    buy_principal,
                    buy_fee,
                )
                if third_fill is not None:
                    (
                        fill_index,
                        fill_quantity,
                        principal,
                        fee,
                        trade,
                    ) = third_fill
                    quantity += fill_quantity
                    buy_principal += principal
                    buy_fee += fee
                    stages += 1
                    last_entry_index = max(last_entry_index, fill_index)
                    trades.append(trade)

    second_death: CrossEvent | None = None
    if second_golden is not None and first_death is not None:
        second_death = first_event_after(
            down_events,
            max(
                last_entry_index,
                second_golden.confirmation_index,
            ),
            raw_after=second_golden.raw_index,
            raw_gap_from=first_death.raw_index,
        )
        if (
            second_death is not None
            and contract.bars[
                second_death.confirmation_index
            ].minute
            > FORCE_EXIT_TIME
        ):
            second_death = None

    exit_signal_index = (
        second_death.confirmation_index
        if second_death is not None
        else base.MINUTE_INDEX[FORCE_EXIT_TIME]
    )
    exit_reason = (
        "SECOND_DEATH_CROSS"
        if second_death is not None
        else "TIME"
    )
    filled, gross_sales, sell_fee, fill_index, sell_trades = (
        execution.sell_all(
            date_value,
            CONFIG,  # type: ignore[arg-type]
            contract,
            exit_signal_index,
            quantity,
            exit_reason,
        )
    )
    for trade in sell_trades:
        trade["stage"] = stages
        trade["call_put"] = contract.call_put
    trades.extend(sell_trades)
    expired_quantity = quantity - filled
    if expired_quantity > 0:
        trades.append(
            {
                "strategy": STRATEGY_VERSION,
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
        exit_minute = contract.bars[fill_index].minute

    average_entry_price = (
        buy_principal / quantity / 250_000 if quantity else 0.0
    )
    mfe_multiple, mae_pct = excursion(
        contract,
        first_fill_index,
        average_entry_price,
    )
    pnl = gross_sales - sell_fee - buy_principal - buy_fee
    return (
        DayResult(
            strategy=STRATEGY_VERSION,
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
            second_low=round(second_low, 4),
            first_golden_cross_minute=contract.bars[
                setup.event.raw_index
            ].minute,
            lower_low_minute=(
                contract.bars[lower_low_index].minute
                if lower_low_index is not None
                else None
            ),
            second_golden_cross_minute=(
                contract.bars[second_golden.raw_index].minute
                if second_golden is not None
                else None
            ),
            first_death_cross_minute=(
                contract.bars[first_death.raw_index].minute
                if first_death is not None
                else None
            ),
            second_death_cross_minute=(
                contract.bars[second_death.raw_index].minute
                if second_death is not None
                else None
            ),
            entry_stages=stages,
            entry_quantity=quantity,
            average_entry_price=round(average_entry_price, 6),
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
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
    sorted_profits = sorted(profits, reverse=True)
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
        "pnl_excluding_best_3_days": (
            sum(profits) - sum(sorted_profits[:3])
        ),
        "pnl_excluding_best_5_days": (
            sum(profits) - sum(sorted_profits[:5])
        ),
        "total_buy_principal": sum(
            row.buy_principal for row in traded
        ),
        "expired_contracts": sum(
            row.expired_quantity for row in traded
        ),
        "one_stage_days": sum(row.entry_stages == 1 for row in traded),
        "two_stage_days": sum(row.entry_stages == 2 for row in traded),
        "three_stage_days": sum(row.entry_stages == 3 for row in traded),
        "lower_low_days": sum(
            row.lower_low_minute is not None for row in traded
        ),
        "second_golden_cross_days": sum(
            row.second_golden_cross_minute is not None for row in traded
        ),
        "second_death_cross_exit_days": sum(
            row.exit_reason == "SECOND_DEATH_CROSS" for row in traded
        ),
    }


def period_summaries(
    rows: list[DayResult],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    development = [row for row in rows if row.date <= DEVELOPMENT_END]
    validation = [row for row in rows if row.date > DEVELOPMENT_END]
    return summarize(rows), summarize(development), summarize(validation)


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    cash = INITIAL_CASH
    daily: list[DayResult] = []
    trades: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        setup = first_setup(contracts)
        row, day_trades = simulate_day(
            date_value,
            previous_vix,
            contracts,
            setup,
            cash,
        )
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)

    summary, development, validation = period_summaries(daily)
    summary["initial_cash"] = INITIAL_CASH
    summary["ending_cash"] = cash
    audit_result = audit(daily, trades)  # type: ignore[arg-type]
    traded = [row for row in daily if row.entry_filled]
    losing = [row for row in traded if row.pnl < 0]
    s3_v01 = json.loads(
        (OUTPUT_DIR / "strategy3_summary.json").read_text(encoding="utf-8")
    )
    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S3-v0.1",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": DEVELOPMENT_END,
            "historical_low_volatility_backcast": False,
        },
        "rules": {
            "first_entry": (
                "first confirmed MA5 above MA20 cross; confirmation is "
                "the next actual traded bar within 5 minutes"
            ),
            "second_entry": (
                "first later completed bar whose low is at least 0.01 "
                "below the first 60-minute low"
            ),
            "third_entry": (
                "second confirmed MA5 above MA20 cross after the lower "
                "low and a confirmed bearish cross"
            ),
            "minimum_cross_gap_minutes": MINIMUM_CROSS_GAP,
            "tranche_budgets": list(TRANCHE_BUDGETS),
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "exit": (
                "sell all after the second confirmed MA5 below MA20 "
                "cross; otherwise signal at 15:15"
            ),
            "entry_execution": (
                "next actual bar within 5 minutes, 10% volume "
                "participation, 0.01 slippage"
            ),
            "exit_execution": (
                "later actual bars, 10% volume participation, "
                "0.01 slippage"
            ),
        },
        "summary": summary,
        "development": development,
        "validation": validation,
        "comparison_to_s3_v01": {
            "total_pnl_difference": (
                int(summary["total_pnl"])
                - int(s3_v01["summary"]["total_pnl"])
            ),
            "development_pnl_difference": (
                int(development["total_pnl"])
                - int(s3_v01["development"]["total_pnl"])
            ),
            "validation_pnl_difference": (
                int(validation["total_pnl"])
                - int(s3_v01["validation"]["total_pnl"])
            ),
        },
        "analysis": {
            "by_entry_stages": grouped(traded, "entry_stages"),
            "by_exit_reason": grouped(traded, "exit_reason"),
            "by_call_put": grouped(traded, "call_put"),
            "losing_days_mfe_at_least_1_5x": sum(
                row.mfe_multiple >= 1.5 for row in losing
            ),
            "losing_days_mfe_at_least_2x": sum(
                row.mfe_multiple >= 2.0 for row in losing
            ),
            "worst_days": [
                asdict(row)
                for row in sorted(traded, key=lambda item: item.pnl)[:15]
            ],
        },
        "audit": audit_result,
        "limitations": [
            "쌍바닥·쌍고점은 가격 모양을 사후 판독하지 않고 확정된 MA 교차와 더 낮은 저점으로 근사했습니다.",
            "두 번째 저점 매수는 하락 중 매수이므로 세 번째 골든크로스가 나오지 않는 위험을 포함합니다.",
            "별도 손절은 넣지 않았고 두 번째 데드크로스가 없으면 15:15 청산합니다.",
            "미체결 만기 잔량은 0원으로 평가했습니다.",
        ],
    }
    write_csv(
        OUTPUT_DIR / "strategy3_v02_daily.csv",
        [asdict(row) for row in daily],
    )
    write_csv(OUTPUT_DIR / "strategy3_v02_trades.csv", trades)
    (OUTPUT_DIR / "strategy3_v02_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy3_v02_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
