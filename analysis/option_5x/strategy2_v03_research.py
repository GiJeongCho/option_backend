from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy2_improvement_research as s2
import strategy2_probe_research as probe
from strategy1_backtest import (
    COMMISSION,
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    MULTIPLIER,
    SLIPPAGE,
    VIX_THRESHOLD,
    previous_volatility,
    volatility_closes,
)
from strategy2_backtest import audit


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PREMIUM = base.PremiumBand("2.50-5.00", 2.50, 5.00)
LateExit = Literal["baseline_2x", "late_1_5_all", "late_1_5_recover"]


@dataclass(frozen=True)
class Config:
    late_exit: LateExit

    @property
    def key(self) -> str:
        return f"S2-v0.3-candidate|{self.late_exit}"


CONFIGS = (
    Config("baseline_2x"),
    Config("late_1_5_all"),
    Config("late_1_5_recover"),
)


@dataclass
class State:
    config: Config
    cash: int = INITIAL_CASH
    daily: list[probe.ProbeDay] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


def recovery_quantity(
    quantity: int,
    bar_close: float,
    principal: int,
    buy_fee: int,
) -> int:
    estimated_net = (
        max(0.0, bar_close - SLIPPAGE) * MULTIPLIER - COMMISSION
    )
    if estimated_net <= 0:
        return quantity
    return min(quantity, math.ceil((principal + buy_fee) / estimated_net))


def simulate_day(
    date_value: str,
    previous_vix: float,
    config: Config,
    contracts: dict[str, base.ContractGrid],
    signal: base.Signal | None,
    start_cash: int,
) -> tuple[probe.ProbeDay, list[dict[str, object]]]:
    if previous_vix < VIX_THRESHOLD or signal is None:
        return (
            probe.empty_day(
                config,  # type: ignore[arg-type]
                date_value,
                previous_vix,
                start_cash,
                signal_found=signal is not None,
            ),
            [],
        )
    contract = contracts[signal.code]
    initial_fill = probe.buy(
        date_value,
        config,  # type: ignore[arg-type]
        contract,
        signal.index,
        signal.close,
        2_500_000,
        start_cash,
        "PROBE_ENTRY",
    )
    if initial_fill is None:
        return (
            probe.empty_day(
                config,  # type: ignore[arg-type]
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
    quantity = initial_quantity
    added_quantity = 0
    buy_principal = initial_principal
    buy_fee = initial_fee
    gross_sales = 0
    sell_fee = 0
    confirmed = False
    confirmation_minute: int | None = None
    late_exit_done = False
    exit_reason = ""
    exit_minute: int | None = None
    mfe_pct = 0.0
    mae_pct = 0.0
    trades = [initial_trade]
    index = entry_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        reason = ""
        request = 0
        if bar.traded and bar.close is not None:
            multiple = bar.close / initial_price
            mfe_pct = max(mfe_pct, (multiple - 1.0) * 100)
            mae_pct = min(mae_pct, (multiple - 1.0) * 100)
            elapsed = index - entry_index

            if multiple >= 2.0:
                reason = "TARGET_2X_INITIAL"
                request = quantity
            elif (
                not confirmed
                and elapsed <= 20
                and multiple >= 1.5
            ):
                remaining_budget = DAILY_PRINCIPAL_LIMIT - buy_principal
                addition = probe.buy(
                    date_value,
                    config,  # type: ignore[arg-type]
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
            elif (
                not confirmed
                and not late_exit_done
                and elapsed > 20
                and multiple >= 1.5
                and config.late_exit != "baseline_2x"
            ):
                late_exit_done = True
                if config.late_exit == "late_1_5_all":
                    reason = "LATE_1_5_ALL"
                    request = quantity
                else:
                    reason = "LATE_1_5_RECOVER"
                    request = recovery_quantity(
                        quantity,
                        bar.close,
                        initial_principal,
                        initial_fee,
                    )

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
            request = quantity
        if not reason:
            index += 1
            continue

        filled, sales, fees, fill_index, sell_trades = probe.sell_all(
            date_value,
            config,  # type: ignore[arg-type]
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
        probe.ProbeDay(
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


def period_rows(
    rows: list[probe.ProbeDay],
) -> tuple[list[probe.ProbeDay], list[probe.ProbeDay], list[probe.ProbeDay]]:
    historical = [
        row for row in rows if row.date < s2.DEVELOPMENT_START
    ]
    development = [
        row
        for row in rows
        if s2.DEVELOPMENT_START <= row.date <= s2.DEVELOPMENT_END
    ]
    validation = [
        row for row in rows if row.date > s2.DEVELOPMENT_END
    ]
    return historical, development, validation


def document(state: State) -> dict[str, object]:
    assert state.daily is not None
    historical, development, validation = period_rows(state.daily)
    return {
        "strategy": state.config.key,
        "late_exit": state.config.late_exit,
        "all": probe.summarize(state.daily),
        "historical_backcast": probe.summarize(historical),
        "development": probe.summarize(development),
        "validation": probe.summarize(validation),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    states = [State(config) for config in CONFIGS]
    trades: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        indicators = s2.build_indicators(contracts)
        signal = s2.first_signal(
            contracts, indicators, "ma20_rising"
        )
        for state in states:
            day, day_trades = simulate_day(
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
            trades.extend(day_trades)

    candidates = [document(state) for state in states]
    alternatives = [
        row for row in candidates if row["late_exit"] != "baseline_2x"
    ]
    selected = max(
        alternatives,
        key=lambda row: (
            int(
                row["development"]["pnl_excluding_best_5_days"]  # type: ignore[index]
            ),
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(row["development"]["max_drawdown"]),  # type: ignore[index]
        ),
    )
    daily_rows = [
        asdict(row)
        for state in states
        for row in (state.daily or [])
    ]
    audits = {}
    for state in states:
        assert state.daily is not None
        state_trades = [
            trade
            for trade in trades
            if trade["strategy"] == state.config.key
        ]
        audits[state.config.late_exit] = audit(
            state.daily,
            state_trades,
        )
    audit_result = {
        "passed": all(item["passed"] for item in audits.values()),
        "candidates": audits,
    }
    reference = json.loads(
        (OUTPUT_DIR / "strategy2_v02_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = next(
        row for row in candidates if row["late_exit"] == "baseline_2x"
    )
    reconciliation = {
        "total_pnl": (
            int(baseline["all"]["total_pnl"])  # type: ignore[index]
            == int(reference["summary"]["total_pnl"])
        ),
        "development_pnl": (
            int(baseline["development"]["total_pnl"])  # type: ignore[index]
            == int(reference["development"]["total_pnl"])
        ),
        "validation_pnl": (
            int(baseline["validation"]["total_pnl"])  # type: ignore[index]
            == int(reference["validation"]["total_pnl"])
        ),
    }
    if not all(reconciliation.values()):
        raise RuntimeError(f"S2-v0.2 재현 실패: {reconciliation}")

    result = {
        "status": "strategy_2_v03_research",
        "method": {
            "development_period": (
                f"{s2.DEVELOPMENT_START}_{s2.DEVELOPMENT_END}"
            ),
            "validation_period": "20260501_onward",
            "validation_used_for_selection": False,
            "entry": "S2-v0.2 fixed entry",
            "confirmed_exit": "confirmation add then 2x all",
            "selection_rule": (
                "among two adaptive late-exit alternatives, maximize "
                "development pnl excluding best 5 days"
            ),
        },
        "baseline": baseline,
        "selected": selected,
        "candidates": candidates,
        "baseline_reconciliation": {
            "passed": all(reconciliation.values()),
            **reconciliation,
        },
        "audit": audit_result,
    }
    write_csv(OUTPUT_DIR / "strategy2_v03_research_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "strategy2_v03_research_trades.csv", trades)
    (OUTPUT_DIR / "strategy2_v03_research_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
