from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy2_probe_research as probe
from strategy1_backtest import (
    DAILY_PRINCIPAL_LIMIT,
    FORCE_EXIT_TIME,
    INITIAL_CASH,
    VIX_THRESHOLD,
    previous_volatility,
    volatility_closes,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEVELOPMENT_START = "20251001"
DEVELOPMENT_END = "20260430"
PREMIUM = base.PremiumBand("2.50-5.00", 2.50, 5.00)
MIN_DEVELOPMENT_TRADES = 15

EntryFilter = Literal[
    "baseline",
    "short_bull",
    "long_bull",
    "full_bull",
    "short_bull_long_bear",
    "ma20_rising",
    "ma40_rising",
    "ma60_rising",
    "ma5_ma20_rising",
    "ma20_ma40_rising",
]
FailureExit = Literal[
    "hold",
    "exit_unconfirmed",
    "exit_if_below_entry",
    "exit_if_below_ma20",
    "exit_if_weak_1_3",
]


@dataclass(frozen=True)
class ImprovementConfig:
    entry_filter: EntryFilter
    confirmation_window: int = 20
    failure_exit: FailureExit = "hold"

    @property
    def key(self) -> str:
        return (
            f"entry_{self.entry_filter}|confirm_1.5x_"
            f"{self.confirmation_window}m|{self.failure_exit}"
        )


@dataclass
class State:
    config: ImprovementConfig
    cash: int = INITIAL_CASH
    daily: list[probe.ProbeDay] | None = None

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []


@dataclass(frozen=True)
class Indicators:
    ma10: list[float | None]
    ma40: list[float | None]
    ma60: list[float | None]


ENTRY_CONFIGS = tuple(
    ImprovementConfig(entry_filter=name)
    for name in (
        "baseline",
        "short_bull",
        "long_bull",
        "full_bull",
        "short_bull_long_bear",
        "ma20_rising",
        "ma40_rising",
        "ma60_rising",
        "ma5_ma20_rising",
        "ma20_ma40_rising",
    )
)


def build_indicators(
    contracts: dict[str, base.ContractGrid],
) -> dict[str, Indicators]:
    result: dict[str, Indicators] = {}
    for code, contract in contracts.items():
        closes = [bar.close for bar in contract.bars]
        result[code] = Indicators(
            ma10=base.rolling_mean(closes, 10),
            ma40=base.rolling_mean(closes, 40),
            ma60=base.rolling_mean(closes, 60),
        )
    return result


def available(values: list[float | None], index: int) -> float | None:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def rising(
    values: list[float | None], index: int, lookback: int = 5
) -> bool:
    current = available(values, index)
    previous = available(values, index - lookback)
    return (
        current is not None
        and previous is not None
        and current > previous
    )


def passes_entry_filter(
    contract: base.ContractGrid,
    indicators: Indicators,
    index: int,
    name: EntryFilter,
) -> bool:
    if name == "baseline":
        return True
    ma5 = available(contract.ma5, index)
    ma10 = available(indicators.ma10, index)
    ma20 = available(contract.ma20, index)
    ma40 = available(indicators.ma40, index)
    ma60 = available(indicators.ma60, index)
    short_bull = (
        ma5 is not None
        and ma10 is not None
        and ma20 is not None
        and ma5 > ma10 > ma20
    )
    long_bull = (
        ma20 is not None
        and ma40 is not None
        and ma60 is not None
        and ma20 > ma40 > ma60
    )
    long_bear = (
        ma20 is not None
        and ma40 is not None
        and ma60 is not None
        and ma20 < ma40 < ma60
    )
    checks = {
        "short_bull": short_bull,
        "long_bull": long_bull,
        "full_bull": short_bull and long_bull,
        "short_bull_long_bear": short_bull and long_bear,
        "ma20_rising": rising(contract.ma20, index),
        "ma40_rising": rising(indicators.ma40, index),
        "ma60_rising": rising(indicators.ma60, index),
        "ma5_ma20_rising": (
            rising(contract.ma5, index)
            and rising(contract.ma20, index)
        ),
        "ma20_ma40_rising": (
            rising(contract.ma20, index)
            and rising(indicators.ma40, index)
        ),
    }
    return checks[name]


def first_signal(
    contracts: dict[str, base.ContractGrid],
    indicators: dict[str, Indicators],
    entry_filter: EntryFilter,
) -> base.Signal | None:
    for index in range(20, len(base.SESSION_MINUTES)):
        candidates: list[base.Signal] = []
        for code, contract in contracts.items():
            if not base.is_signal(contract, index, PREMIUM):
                continue
            if not passes_entry_filter(
                contract, indicators[code], index, entry_filter
            ):
                continue
            close = contract.bars[index].close
            assert close is not None
            candidates.append(
                base.Signal(
                    code=code,
                    index=index,
                    minute=contract.bars[index].minute,
                    close=close,
                    recent_volume=sum(
                        bar.volume
                        for bar in contract.bars[index - 19 : index + 1]
                    ),
                )
            )
        if candidates:
            return sorted(
                candidates,
                key=lambda item: (-item.recent_volume, item.code),
            )[0]
    return None


def failure_reason(
    config: ImprovementConfig,
    contract: base.ContractGrid,
    index: int,
    initial_price: float,
    maximum_multiple: float,
) -> str:
    if config.failure_exit == "hold":
        return ""
    bar = contract.bars[index]
    if bar.close is None:
        return ""
    if config.failure_exit == "exit_unconfirmed":
        return "FAILED_CONFIRMATION_TIME"
    if (
        config.failure_exit == "exit_if_below_entry"
        and bar.close < initial_price
    ):
        return "FAILED_CONFIRMATION_BELOW_ENTRY"
    if config.failure_exit == "exit_if_below_ma20":
        ma20 = contract.ma20[index]
        if ma20 is not None and bar.close < ma20:
            return "FAILED_CONFIRMATION_BELOW_MA20"
    if (
        config.failure_exit == "exit_if_weak_1_3"
        and maximum_multiple < 1.3
    ):
        return "FAILED_CONFIRMATION_WEAK_MOMENTUM"
    return ""


def simulate_day(
    date_value: str,
    previous_vix: float,
    config: ImprovementConfig,
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
    maximum_multiple = 1.0
    index = entry_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        reason = ""
        if bar.traded and bar.close is not None:
            multiple = bar.close / initial_price
            maximum_multiple = max(maximum_multiple, multiple)
            move = multiple - 1.0
            mfe_pct = max(mfe_pct, move * 100)
            mae_pct = min(mae_pct, move * 100)

            if multiple >= 2.0:
                reason = "TARGET_2X_INITIAL"
            elif (
                not confirmed
                and index - entry_index <= config.confirmation_window
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
                and index - entry_index >= config.confirmation_window
            ):
                reason = failure_reason(
                    config,
                    contract,
                    index,
                    initial_price,
                    maximum_multiple,
                )

        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"
        if not reason:
            index += 1
            continue

        filled, sales, fees, fill_index, sell_trades = probe.sell_all(
            date_value,
            config,  # type: ignore[arg-type]
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


def split_summary(rows: list[probe.ProbeDay]) -> dict[str, object]:
    historical = [row for row in rows if row.date < DEVELOPMENT_START]
    development = [
        row
        for row in rows
        if DEVELOPMENT_START <= row.date <= DEVELOPMENT_END
    ]
    validation = [row for row in rows if row.date > DEVELOPMENT_END]
    return {
        "all": probe.summarize(rows),
        "historical_backcast": probe.summarize(historical),
        "development": probe.summarize(development),
        "validation": probe.summarize(validation),
    }


def candidate_document(state: State) -> dict[str, object]:
    assert state.daily is not None
    return {
        "strategy": state.config.key,
        "entry_filter": state.config.entry_filter,
        "confirmation_window": state.config.confirmation_window,
        "failure_exit": state.config.failure_exit,
        **split_summary(state.daily),
    }


def select_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    eligible = [
        row
        for row in candidates
        if int(row["development"]["trade_days"])  # type: ignore[index]
        >= MIN_DEVELOPMENT_TRADES
        and int(row["development"]["total_pnl"]) > 0  # type: ignore[index]
        and int(
            row["development"]["pnl_excluding_best_5_days"]  # type: ignore[index]
        )
        > 0
    ]
    return max(
        eligible or candidates,
        key=lambda row: (
            int(
                row["development"]["pnl_excluding_best_5_days"]  # type: ignore[index]
            ),
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(row["development"]["max_drawdown"]),  # type: ignore[index]
        ),
    )


def run_states(
    configs: tuple[ImprovementConfig, ...],
    groups: list[dict[str, str]],
    files: dict[str, Path],
    volatility_dates: list[str],
    volatility_values: list[float],
) -> tuple[
    list[State], list[dict[str, object]], list[dict[str, object]]
]:
    states = [State(config) for config in configs]
    all_trades: list[dict[str, object]] = []
    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        indicators = build_indicators(contracts)
        signals = {
            config.entry_filter: first_signal(
                contracts, indicators, config.entry_filter
            )
            for config in configs
        }
        for state in states:
            day, trades = simulate_day(
                date_value,
                previous_vix,
                state.config,
                contracts,
                signals[state.config.entry_filter],
                state.cash,
            )
            state.cash = day.end_cash
            assert state.daily is not None
            state.daily.append(day)
            all_trades.extend(trades)
    daily_rows = [
        asdict(row)
        for state in states
        for row in (state.daily or [])
    ]
    return states, daily_rows, all_trades


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

    entry_states, entry_daily, entry_trades = run_states(
        ENTRY_CONFIGS,
        groups,
        files,
        volatility_dates,
        volatility_values,
    )
    entry_candidates = [
        candidate_document(state) for state in entry_states
    ]
    selected_entry = select_candidate(entry_candidates)
    selected_filter = str(selected_entry["entry_filter"])

    exit_configs = (
        ImprovementConfig(selected_filter, 20, "hold"),  # type: ignore[arg-type]
        ImprovementConfig(selected_filter, 30, "hold"),  # type: ignore[arg-type]
        ImprovementConfig(  # type: ignore[arg-type]
            selected_filter, 20, "exit_unconfirmed"
        ),
        ImprovementConfig(  # type: ignore[arg-type]
            selected_filter, 30, "exit_unconfirmed"
        ),
        ImprovementConfig(  # type: ignore[arg-type]
            selected_filter, 20, "exit_if_below_entry"
        ),
        ImprovementConfig(  # type: ignore[arg-type]
            selected_filter, 20, "exit_if_below_ma20"
        ),
        ImprovementConfig(  # type: ignore[arg-type]
            selected_filter, 20, "exit_if_weak_1_3"
        ),
    )
    exit_states, exit_daily, exit_trades = run_states(
        exit_configs,
        groups,
        files,
        volatility_dates,
        volatility_values,
    )
    exit_candidates = [
        candidate_document(state) for state in exit_states
    ]
    selected_final = select_candidate(exit_candidates)

    entry_audit = probe.audit(  # type: ignore[arg-type]
        entry_states, entry_trades
    )
    exit_audit = probe.audit(  # type: ignore[arg-type]
        exit_states, exit_trades
    )
    baseline = next(
        row for row in entry_candidates if row["entry_filter"] == "baseline"
    )
    reference = json.loads(
        (OUTPUT_DIR / "strategy2_summary.json").read_text(encoding="utf-8")
    )
    baseline_reconciliation = {
        "total_pnl": (
            int(baseline["all"]["total_pnl"])  # type: ignore[index]
            == int(reference["summary"]["total_pnl"])
        ),
        "trade_days": (
            int(baseline["all"]["trade_days"])  # type: ignore[index]
            == int(reference["summary"]["trade_days"])
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
    if not all(baseline_reconciliation.values()):
        raise RuntimeError(
            f"S2-v0.1 기준선 재현 실패: {baseline_reconciliation}"
        )

    result = {
        "status": "strategy_2_improvement_research",
        "method": {
            "development_period": (
                f"{DEVELOPMENT_START}_{DEVELOPMENT_END}"
            ),
            "validation_period": "20260501_onward",
            "validation_used_for_selection": False,
            "selection_rule": (
                "at least 15 development trades; positive development "
                "pnl and pnl excluding best 5 days; then maximize "
                "development pnl excluding best 5 days"
            ),
            "moving_averages": [5, 10, 20, 40, 60],
            "stage_order": [
                "select one entry filter with baseline S2 exit",
                "select one failure exit with the selected entry filter",
            ],
        },
        "baseline": baseline,
        "entry_stage": {
            "selected": selected_entry,
            "candidates": entry_candidates,
            "audit": entry_audit,
        },
        "exit_stage": {
            "selected": selected_final,
            "candidates": exit_candidates,
            "audit": exit_audit,
        },
        "baseline_reconciliation": {
            "passed": all(baseline_reconciliation.values()),
            **baseline_reconciliation,
        },
    }
    write_csv(
        OUTPUT_DIR / "strategy2_improvement_entry_daily.csv",
        entry_daily,
    )
    write_csv(
        OUTPUT_DIR / "strategy2_improvement_entry_trades.csv",
        entry_trades,
    )
    write_csv(
        OUTPUT_DIR / "strategy2_improvement_exit_daily.csv",
        exit_daily,
    )
    write_csv(
        OUTPUT_DIR / "strategy2_improvement_exit_trades.csv",
        exit_trades,
    )
    (OUTPUT_DIR / "strategy2_improvement_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not entry_audit["passed"] or not exit_audit["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
