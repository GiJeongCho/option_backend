from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import minimal_baseline_backtest as base
from strategy1_backtest import (
    COMMISSION,
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    MULTIPLIER,
    SLIPPAGE,
    VIX_THRESHOLD,
    VOLUME_PARTICIPATION,
    previous_volatility,
    volatility_closes,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEVELOPMENT_END = "20260430"
PREMIUM = base.PremiumBand("2.50-5.00", 2.50, 5.00)


@dataclass(frozen=True)
class ResearchConfig:
    target_multiple: float
    stop_loss_pct: float | None
    ma_reversal_stop: bool

    @property
    def key(self) -> str:
        stop = (
            "none"
            if self.stop_loss_pct is None
            else f"{round(self.stop_loss_pct * 100)}pct"
        )
        ma_stop = "ma2" if self.ma_reversal_stop else "no_ma"
        return (
            f"target_{self.target_multiple:g}x|stop_{stop}|{ma_stop}"
        )


CONFIGS = tuple(
    ResearchConfig(target, stop, ma_stop)
    for target in (1.5, 2.0)
    for stop in (0.20, 0.30, 0.40, None)
    for ma_stop in (False, True)
)


@dataclass
class ResearchDay:
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
    mfe_pct: float
    mae_pct: float


@dataclass
class State:
    config: ResearchConfig
    cash: int = INITIAL_CASH
    daily: list[ResearchDay] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


def sell_all(
    date_value: str,
    config: ResearchConfig,
    contract: base.ContractGrid,
    signal_index: int,
    quantity: int,
    reason: str,
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
        fillable = min(remaining, math.floor(capacity + 1e-12))
        if fillable <= 0:
            continue
        capacity -= fillable
        fill_price = max(0.0, bar.open - SLIPPAGE)
        principal = round(fillable * fill_price * MULTIPLIER)
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
                "side": "SELL",
                "reason": reason,
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
    return quantity - remaining, gross_sales, fees, last_fill_index, trades


def empty_day(
    config: ResearchConfig,
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool = False,
) -> ResearchDay:
    return ResearchDay(
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
        signal_minute=None,
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
        mfe_pct=0.0,
        mae_pct=0.0,
    )


def simulate_day(
    date_value: str,
    previous_vix: float,
    config: ResearchConfig,
    contracts: dict[str, base.ContractGrid],
    signal: base.Signal | None,
    start_cash: int,
) -> tuple[ResearchDay, list[dict[str, object]]]:
    if previous_vix < VIX_THRESHOLD or signal is None:
        return (
            empty_day(
                config,
                date_value,
                previous_vix,
                start_cash,
                signal_found=signal is not None,
            ),
            [],
        )
    contract = contracts[signal.code]
    fill = base.entry_fill(contract, signal, start_cash)
    if fill is None:
        return (
            empty_day(
                config,
                date_value,
                previous_vix,
                start_cash,
                signal_found=True,
            ),
            [],
        )

    entry_index, quantity, entry_bar, entry_price = fill
    principal = round(quantity * entry_price * MULTIPLIER)
    buy_fee = quantity * COMMISSION
    trades: list[dict[str, object]] = [
        {
            "strategy": config.key,
            "date": date_value,
            "minute": entry_bar.minute,
            "code": contract.code,
            "side": "BUY",
            "reason": "STRICT_AND_ENTRY",
            "quantity": quantity,
            "price": entry_price,
            "principal": principal,
            "fee": buy_fee,
            "cash_flow": -(principal + buy_fee),
            "bar_volume": entry_bar.volume,
        }
    ]
    remaining = quantity
    gross_sales = 0
    sell_fee = 0
    exit_reason = ""
    exit_minute: int | None = None
    below_ma_count = 0
    mfe_pct = 0.0
    mae_pct = 0.0

    for index in range(entry_index, len(contract.bars)):
        bar = contract.bars[index]
        reason = ""
        if bar.traded and bar.close is not None:
            move = bar.close / entry_price - 1.0
            mfe_pct = max(mfe_pct, move * 100)
            mae_pct = min(mae_pct, move * 100)
            ma5 = contract.ma5[index]
            ma20 = contract.ma20[index]
            if ma5 is not None and ma20 is not None and ma5 < ma20:
                below_ma_count += 1
            else:
                below_ma_count = 0
            if (
                config.stop_loss_pct is not None
                and bar.close
                <= entry_price * (1.0 - config.stop_loss_pct)
            ):
                reason = f"STOP_{round(config.stop_loss_pct * 100)}"
            elif config.ma_reversal_stop and below_ma_count >= 2:
                reason = "STOP_MA_REVERSAL"
            elif bar.close >= entry_price * config.target_multiple:
                reason = f"TARGET_{config.target_multiple:g}X"

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
        if not reason:
            continue

        (
            filled,
            sales,
            fees,
            fill_index,
            sell_trades,
        ) = sell_all(
            date_value,
            config,
            contract,
            index,
            remaining,
            reason,
        )
        remaining -= filled
        gross_sales += sales
        sell_fee += fees
        trades.extend(sell_trades)
        if remaining > 0:
            exit_reason = "UNFILLED_ZERO_VALUE"
            exit_minute = base.SESSION_MINUTES[-1]
        else:
            exit_reason = reason
            exit_minute = contract.bars[fill_index].minute
        break

    if remaining > 0:
        trades.append(
            {
                "strategy": config.key,
                "date": date_value,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "quantity": remaining,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
        exit_reason = "UNFILLED_ZERO_VALUE"
        exit_minute = base.SESSION_MINUTES[-1]

    pnl = gross_sales - sell_fee - principal - buy_fee
    return (
        ResearchDay(
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
            signal_minute=signal.minute,
            entry_minute=entry_bar.minute,
            entry_price=entry_price,
            entry_quantity=quantity,
            buy_principal=principal,
            buy_fee=buy_fee,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_reason=exit_reason,
            exit_minute=exit_minute,
            expired_quantity=remaining,
            mfe_pct=round(mfe_pct, 4),
            mae_pct=round(mae_pct, 4),
        ),
        trades,
    )


def period_summary(rows: list[ResearchDay]) -> dict[str, object]:
    profits = [row.pnl for row in rows]
    traded = [value for value in profits if value != 0]
    sorted_profits = sorted(profits, reverse=True)
    running = 0
    peak = 0
    drawdown = 0
    for value in profits:
        running += value
        peak = max(peak, running)
        drawdown = min(drawdown, running - peak)
    gains = sum(value for value in traded if value > 0)
    losses = -sum(value for value in traded if value < 0)
    return {
        "days": len(rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in traded),
        "losing_days": sum(value < 0 for value in traded),
        "total_pnl": sum(profits),
        "average_trade_day_pnl": round(
            sum(traded) / len(traded) if traded else 0
        ),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(profits, default=0),
        "maximum_day_pnl": max(profits, default=0),
        "max_drawdown": drawdown,
        "pnl_excluding_best_3_days": (
            sum(profits) - sum(sorted_profits[:3])
        ),
        "pnl_excluding_best_5_days": (
            sum(profits) - sum(sorted_profits[:5])
        ),
        "expired_contracts": sum(row.expired_quantity for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(
    states: list[State], trades: list[dict[str, object]]
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for trade in trades:
        key = (str(trade["strategy"]), str(trade["date"]))
        trades_by_key.setdefault(key, []).append(trade)
    for state in states:
        expected_cash = INITIAL_CASH
        assert state.daily is not None
        for row in state.daily:
            if row.start_cash != expected_cash:
                errors.append(f"{row.strategy}/{row.date}: cash chain")
            if row.end_cash != row.start_cash + row.pnl:
                errors.append(f"{row.strategy}/{row.date}: pnl equation")
            if row.buy_principal > DAILY_PRINCIPAL_LIMIT:
                errors.append(f"{row.strategy}/{row.date}: principal limit")
            cash_flow = sum(
                int(item["cash_flow"])
                for item in trades_by_key.get(
                    (row.strategy, row.date), []
                )
            )
            if cash_flow != row.pnl:
                errors.append(f"{row.strategy}/{row.date}: cash flow")
            expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:20],
        "strategy_count": len(states),
        "daily_row_count": sum(
            len(state.daily or []) for state in states
        ),
        "trade_row_count": len(trades),
    }


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    states = [State(config) for config in CONFIGS]
    all_trades: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        signal = base.first_signal(contracts, PREMIUM)
        for state in states:
            day, trades = simulate_day(
                date_value,
                previous_vix,
                state.config,
                contracts,
                signal,
                state.cash,
            )
            state.cash = day.end_cash
            assert state.daily is not None
            state.daily.append(day)
            all_trades.extend(trades)

    summaries: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for state in states:
        assert state.daily is not None
        development = [
            row for row in state.daily if row.date <= DEVELOPMENT_END
        ]
        validation = [
            row for row in state.daily if row.date > DEVELOPMENT_END
        ]
        summaries.append(
            {
                "strategy": state.config.key,
                "target_multiple": state.config.target_multiple,
                "stop_loss_pct": state.config.stop_loss_pct,
                "ma_reversal_stop": state.config.ma_reversal_stop,
                "all": period_summary(state.daily),
                "development": period_summary(development),
                "validation": period_summary(validation),
            }
        )
        daily_rows.extend(asdict(row) for row in state.daily)

    eligible = [
        row
        for row in summaries
        if int(row["development"]["total_pnl"]) > 0  # type: ignore[index]
        and int(
            row["development"]["pnl_excluding_best_5_days"]  # type: ignore[index]
        )
        > 0
    ]
    selected = max(
        eligible or summaries,
        key=lambda row: (
            int(
                row["development"]["pnl_excluding_best_5_days"]  # type: ignore[index]
            ),
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(row["development"]["max_drawdown"]),  # type: ignore[index]
        ),
    )
    audit_result = audit(states, all_trades)
    result = {
        "status": "strategy_2_exit_research",
        "premium_range": PREMIUM.name,
        "volatility_filter": f"previous_close >= {VIX_THRESHOLD:g}",
        "development_period": f"through_{DEVELOPMENT_END}",
        "selection_rule": (
            "development pnl and pnl excluding best 5 days positive, "
            "then maximize development pnl excluding best 5 days"
        ),
        "selected_on_development": selected,
        "candidates": summaries,
        "audit": audit_result,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "strategy2_research_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "strategy2_research_trades.csv", all_trades)
    (OUTPUT_DIR / "strategy2_research_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
