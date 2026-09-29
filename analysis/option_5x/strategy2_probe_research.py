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
from strategy2_research import period_summary


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEVELOPMENT_END = "20260430"
PREMIUM = base.PremiumBand("2.50-5.00", 2.50, 5.00)
ENTRY_WAIT_MINUTES = 5


@dataclass(frozen=True)
class ProbeConfig:
    probe_budget: int
    confirmation_multiple: float
    confirmation_window: int
    post_confirmation_floor: bool

    @property
    def key(self) -> str:
        floor = "entry_floor" if self.post_confirmation_floor else "no_floor"
        return (
            f"probe_{self.probe_budget // 10_000}man|"
            f"confirm_{self.confirmation_multiple:g}x_"
            f"{self.confirmation_window}m|{floor}"
        )


CONFIGS = tuple(
    ProbeConfig(budget, multiple, window, floor)
    for budget in (1_250_000, 2_500_000)
    for multiple in (1.3, 1.5)
    for window in (20, 30, 60)
    for floor in (False, True)
)


@dataclass
class ProbeDay:
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
    initial_price: float
    initial_quantity: int
    confirmed: bool
    confirmation_minute: int | None
    added_quantity: int
    total_quantity: int
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
    config: ProbeConfig
    cash: int = INITIAL_CASH
    daily: list[ProbeDay] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


def next_traded(
    contract: base.ContractGrid, signal_index: int
) -> tuple[int, base.MinuteBar] | None:
    end = min(
        len(contract.bars) - 1, signal_index + ENTRY_WAIT_MINUTES
    )
    for index in range(signal_index + 1, end + 1):
        if contract.bars[index].traded:
            return index, contract.bars[index]
    return None


def buy(
    date_value: str,
    config: ProbeConfig,
    contract: base.ContractGrid,
    signal_index: int,
    signal_close: float,
    budget: int,
    available_cash: int,
    reason: str,
) -> tuple[int, int, int, int, dict[str, object]] | None:
    fill = next_traded(contract, signal_index)
    if fill is None:
        return None
    fill_index, bar = fill
    assert bar.open is not None
    fill_price = bar.open + SLIPPAGE
    requested = math.floor(
        budget / ((signal_close + SLIPPAGE) * MULTIPLIER)
    )
    by_actual = math.floor(budget / (fill_price * MULTIPLIER))
    by_cash = math.floor(
        available_cash / (fill_price * MULTIPLIER + 2 * COMMISSION)
    )
    by_volume = math.floor(bar.volume * VOLUME_PARTICIPATION)
    quantity = min(requested, by_actual, by_cash, by_volume)
    if quantity <= 0:
        return None
    principal = round(quantity * fill_price * MULTIPLIER)
    fee = quantity * COMMISSION
    trade = {
        "strategy": config.key,
        "date": date_value,
        "minute": bar.minute,
        "code": contract.code,
        "side": "BUY",
        "reason": reason,
        "quantity": quantity,
        "price": fill_price,
        "principal": principal,
        "fee": fee,
        "cash_flow": -(principal + fee),
        "bar_volume": bar.volume,
    }
    return fill_index, quantity, principal, fee, trade


def sell_all(
    date_value: str,
    config: ProbeConfig,
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
        price = max(0.0, bar.open - SLIPPAGE)
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
                "side": "SELL",
                "reason": reason,
                "quantity": fillable,
                "price": price,
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
    config: ProbeConfig,
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool = False,
) -> ProbeDay:
    return ProbeDay(
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
        initial_price=0.0,
        initial_quantity=0,
        confirmed=False,
        confirmation_minute=None,
        added_quantity=0,
        total_quantity=0,
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
    config: ProbeConfig,
    contracts: dict[str, base.ContractGrid],
    signal: base.Signal | None,
    start_cash: int,
) -> tuple[ProbeDay, list[dict[str, object]]]:
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
    initial_fill = buy(
        date_value,
        config,
        contract,
        signal.index,
        signal.close,
        config.probe_budget,
        start_cash,
        "PROBE_ENTRY",
    )
    if initial_fill is None:
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

    (
        entry_index,
        initial_quantity,
        initial_principal,
        initial_fee,
        initial_trade,
    ) = initial_fill
    initial_price = float(initial_trade["price"])
    trades = [initial_trade]
    quantity = initial_quantity
    added_quantity = 0
    buy_principal = initial_principal
    buy_fee = initial_fee
    confirmed = False
    confirmation_minute: int | None = None
    gross_sales = 0
    sell_fee = 0
    exit_reason = ""
    exit_minute: int | None = None
    mfe_pct = 0.0
    mae_pct = 0.0
    index = entry_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        reason = ""
        if bar.traded and bar.close is not None:
            move = bar.close / initial_price - 1.0
            mfe_pct = max(mfe_pct, move * 100)
            mae_pct = min(mae_pct, move * 100)

            if bar.close >= initial_price * 2.0:
                reason = "TARGET_2X_INITIAL"
            elif (
                confirmed
                and config.post_confirmation_floor
                and bar.close <= initial_price
            ):
                reason = "POST_CONFIRMATION_ENTRY_FLOOR"
            elif (
                not confirmed
                and index - entry_index <= config.confirmation_window
                and bar.close
                >= initial_price * config.confirmation_multiple
            ):
                remaining_budget = DAILY_PRINCIPAL_LIMIT - buy_principal
                addition = buy(
                    date_value,
                    config,
                    contract,
                    index,
                    bar.close,
                    remaining_budget,
                    start_cash - buy_principal - buy_fee,
                    "CONFIRMATION_ADD",
                )
                confirmed = True
                confirmation_minute = bar.minute
                if addition is not None:
                    (
                        fill_index,
                        fill_quantity,
                        principal,
                        fee,
                        trade,
                    ) = addition
                    quantity += fill_quantity
                    added_quantity += fill_quantity
                    buy_principal += principal
                    buy_fee += fee
                    trades.append(trade)
                    index = fill_index
                    continue

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
        if not reason:
            index += 1
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
            quantity,
            reason,
        )
        quantity -= filled
        gross_sales += sales
        sell_fee += fees
        trades.extend(sell_trades)
        if quantity > 0:
            exit_reason = "UNFILLED_ZERO_VALUE"
            exit_minute = base.SESSION_MINUTES[-1]
        else:
            exit_reason = reason
            exit_minute = contract.bars[fill_index].minute
        break

    expired_quantity = quantity
    if expired_quantity > 0:
        trades.append(
            {
                "strategy": config.key,
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
        ProbeDay(
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
            entry_minute=int(initial_trade["minute"]),
            initial_price=initial_price,
            initial_quantity=initial_quantity,
            confirmed=confirmed,
            confirmation_minute=confirmation_minute,
            added_quantity=added_quantity,
            total_quantity=initial_quantity + added_quantity,
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_reason=exit_reason,
            exit_minute=exit_minute,
            expired_quantity=expired_quantity,
            mfe_pct=round(mfe_pct, 4),
            mae_pct=round(mae_pct, 4),
        ),
        trades,
    )


def summarize(rows: list[ProbeDay]) -> dict[str, object]:
    base_summary = period_summary(rows)  # type: ignore[arg-type]
    traded = [row for row in rows if row.entry_filled]
    base_summary.update(
        {
            "confirmed_days": sum(row.confirmed for row in traded),
            "added_days": sum(row.added_quantity > 0 for row in traded),
            "total_buy_principal": sum(
                row.buy_principal for row in traded
            ),
        }
    )
    return base_summary


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
            day_trades = trades_by_key.get((row.strategy, row.date), [])
            if sum(int(item["cash_flow"]) for item in day_trades) != row.pnl:
                errors.append(f"{row.strategy}/{row.date}: cash flow")
            quantity = 0
            for trade in day_trades:
                amount = int(trade["quantity"])
                quantity += amount if trade["side"] == "BUY" else -amount
            if quantity != 0:
                errors.append(f"{row.strategy}/{row.date}: quantity")
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
                "probe_budget": state.config.probe_budget,
                "confirmation_multiple": (
                    state.config.confirmation_multiple
                ),
                "confirmation_window": (
                    state.config.confirmation_window
                ),
                "post_confirmation_floor": (
                    state.config.post_confirmation_floor
                ),
                "all": summarize(state.daily),
                "development": summarize(development),
                "validation": summarize(validation),
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
        "status": "strategy_2_probe_research",
        "premium_range": PREMIUM.name,
        "volatility_filter": f"previous_close >= {VIX_THRESHOLD:g}",
        "target": "2x initial probe entry price",
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
    write_csv(OUTPUT_DIR / "strategy2_probe_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "strategy2_probe_trades.csv", all_trades)
    (OUTPUT_DIR / "strategy2_probe_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
