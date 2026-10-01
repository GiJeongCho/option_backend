from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import minimal_baseline_backtest as base
import strategy2_probe_research as execution
import strategy3_research as parent
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
RESEARCH_VERSION = "S3-v0.4-research"
HIGH_VOL_START = "20251001"
DEVELOPMENT_END = "20260430"
LATER_ENTRY_CUTOFF = 1430
LOW_BREAK_TICK = 0.01
TARGET_MULTIPLE = 2.0
TRANCHE_COUNTS = (3, 4, 5)
STOP_RATIOS = (0.80, 0.70, 0.60, 0.50)
STOP_LOSS_AMOUNTS = (1_000_000, 1_500_000, 2_000_000, 2_500_000)
ENTRY_CONFIG = next(
    config
    for config in parent.ENTRY_CONFIGS
    if config.name == "L60_R20_B5_MA20"
)


@dataclass(frozen=True)
class RuntimeConfig:
    tranche_count: int
    stop_ratio: float | None = None
    stop_loss_amount: int | None = None

    @property
    def key(self) -> str:
        if self.stop_ratio is not None:
            stop_pct = round((1.0 - self.stop_ratio) * 100)
            stop_name = f"PRICE{stop_pct}"
        else:
            stop_name = f"LOSS{self.stop_loss_amount}"
        return f"{RESEARCH_VERSION}|T{self.tranche_count}|{stop_name}"

    @property
    def stop_mode(self) -> str:
        return (
            "initial_price_ratio"
            if self.stop_ratio is not None
            else "daily_loss_amount"
        )

    @property
    def tranche_budgets(self) -> tuple[int, ...]:
        unit = DAILY_PRINCIPAL_LIMIT // self.tranche_count
        budgets = [unit] * self.tranche_count
        budgets[-1] += DAILY_PRINCIPAL_LIMIT - sum(budgets)
        return tuple(budgets)


CONFIGS = tuple(
    RuntimeConfig(tranche_count, stop_ratio)
    for tranche_count in TRANCHE_COUNTS
    for stop_ratio in STOP_RATIOS
) + tuple(
    RuntimeConfig(
        tranche_count=tranche_count,
        stop_loss_amount=stop_loss_amount,
    )
    for tranche_count in TRANCHE_COUNTS
    for stop_loss_amount in STOP_LOSS_AMOUNTS
)


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
    signal_minute: int | None
    anchor_low: float
    initial_entry_minute: int | None
    initial_entry_price: float
    initial_quantity: int
    entry_stages: int
    lower_low_entries: int
    last_entry_minute: int | None
    last_bought_low: float
    entry_quantity: int
    average_entry_price: float
    buy_principal: int
    buy_fee: int
    planned_principal_limit: int
    deployment_pct: float
    stop_mode: str
    stop_ratio: float | None
    stop_loss_amount: int | None
    stop_price: float
    stop_triggered: bool
    reached_2x: bool
    gross_sales: int
    sell_fee: int
    exit_signal_reason: str
    exit_reason: str
    exit_minute: int | None
    expired_quantity: int
    mfe_multiple: float
    mae_pct: float


def empty_day(
    config: RuntimeConfig,
    date_value: str,
    previous_vix: float,
    start_cash: int,
    signal_found: bool,
) -> DayResult:
    return DayResult(
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
        anchor_low=0.0,
        initial_entry_minute=None,
        initial_entry_price=0.0,
        initial_quantity=0,
        entry_stages=0,
        lower_low_entries=0,
        last_entry_minute=None,
        last_bought_low=0.0,
        entry_quantity=0,
        average_entry_price=0.0,
        buy_principal=0,
        buy_fee=0,
        planned_principal_limit=DAILY_PRINCIPAL_LIMIT,
        deployment_pct=0.0,
        stop_mode=config.stop_mode,
        stop_ratio=config.stop_ratio,
        stop_loss_amount=config.stop_loss_amount,
        stop_price=0.0,
        stop_triggered=False,
        reached_2x=False,
        gross_sales=0,
        sell_fee=0,
        exit_signal_reason="",
        exit_reason="",
        exit_minute=None,
        expired_quantity=0,
        mfe_multiple=0.0,
        mae_pct=0.0,
    )


def buy_stage(
    date_value: str,
    config: RuntimeConfig,
    contract: base.ContractGrid,
    signal_index: int,
    signal_close: float,
    stage: int,
    start_cash: int,
    buy_principal: int,
    buy_fee: int,
) -> tuple[int, int, int, int, dict[str, object]] | None:
    cumulative_budget = sum(config.tranche_budgets[:stage])
    allowed_budget = min(
        cumulative_budget - buy_principal,
        DAILY_PRINCIPAL_LIMIT - buy_principal,
    )
    if allowed_budget <= 0:
        return None
    fill = execution.buy(
        date_value,
        config,  # type: ignore[arg-type]
        contract,
        signal_index,
        signal_close,
        allowed_budget,
        start_cash - buy_principal - buy_fee,
        (
            "LOW_RENEWAL_INITIAL"
            if stage == 1
            else "LOW_RENEWAL_ADD"
        ),
    )
    if fill is None:
        return None
    fill_index, quantity, principal, fee, trade = fill
    trade["stage"] = stage
    trade["campaign"] = 1
    trade["call_put"] = contract.call_put
    return fill_index, quantity, principal, fee, trade


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
                maximum,
                float(bar.high) / average_entry_price,
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
    config: RuntimeConfig,
    contracts: dict[str, base.ContractGrid],
    signal: parent.S3Signal | None,
    start_cash: int,
) -> tuple[DayResult, list[dict[str, object]]]:
    if signal is None:
        return (
            empty_day(
                config,
                date_value,
                previous_vix,
                start_cash,
                False,
            ),
            [],
        )
    contract = contracts[signal.code]
    first_fill = buy_stage(
        date_value,
        config,
        contract,
        signal.index,
        signal.close,
        1,
        start_cash,
        0,
        0,
    )
    if first_fill is None:
        return (
            empty_day(
                config,
                date_value,
                previous_vix,
                start_cash,
                True,
            ),
            [],
        )

    (
        first_fill_index,
        initial_quantity,
        first_principal,
        first_fee,
        first_trade,
    ) = first_fill
    initial_price = float(first_trade["price"])
    stop_price = (
        initial_price * config.stop_ratio
        if config.stop_ratio is not None
        else 0.0
    )
    quantity = initial_quantity
    buy_principal = first_principal
    buy_fee = first_fee
    stages = 1
    lower_low_entries = 0
    last_entry_index = first_fill_index
    last_bought_low = signal.anchor_low
    gross_sales = 0
    sell_fee = 0
    exit_signal_reason = ""
    exit_reason = ""
    exit_minute: int | None = None
    reached_2x = False
    stop_triggered = False
    trades = [first_trade]
    index = first_fill_index

    while index < len(contract.bars) and quantity > 0:
        bar = contract.bars[index]
        average_entry_price = (
            buy_principal / quantity / MULTIPLIER
            if quantity
            else 0.0
        )
        reason = ""
        close: float | None = None
        if bar.traded and bar.close is not None:
            close = float(bar.close)
            if close >= average_entry_price * TARGET_MULTIPLE:
                reached_2x = True
                reason = "TARGET_2X_ALL"

        can_add = (
            not reason
            and stages < config.tranche_count
            and bar.minute <= LATER_ENTRY_CUTOFF
            and bar.traded
            and bar.low is not None
            and bar.close is not None
            and float(bar.low) <= last_bought_low - LOW_BREAK_TICK
        )
        if can_add:
            next_stage = stages + 1
            added = buy_stage(
                date_value,
                config,
                contract,
                index,
                float(bar.close),
                next_stage,
                start_cash,
                buy_principal,
                buy_fee,
            )
            if added is not None:
                (
                    fill_index,
                    added_quantity,
                    principal,
                    fee,
                    trade,
                ) = added
                quantity += added_quantity
                buy_principal += principal
                buy_fee += fee
                stages = next_stage
                lower_low_entries += 1
                last_entry_index = fill_index
                last_bought_low = float(bar.low)
                trades.append(trade)
                index = fill_index
                continue

        if not reason and close is not None:
            if (
                config.stop_ratio is not None
                and close <= stop_price
            ):
                stop_triggered = True
                reason = "INITIAL_PRICE_STOP"
            elif config.stop_loss_amount is not None:
                estimated_net_sales = (
                    round(
                        quantity
                        * max(0.0, close - SLIPPAGE)
                        * MULTIPLIER
                    )
                    - quantity * COMMISSION
                )
                marked_loss = (
                    buy_principal + buy_fee - estimated_net_sales
                )
                if marked_loss >= config.stop_loss_amount:
                    stop_triggered = True
                    reason = "DAILY_LOSS_STOP"
        if not reason and bar.minute >= FORCE_EXIT_TIME:
            reason = "TIME"

        if reason:
            (
                filled,
                gross_sales,
                sell_fee,
                fill_index,
                sell_trades,
            ) = execution.sell_all(
                date_value,
                config,  # type: ignore[arg-type]
                contract,
                index,
                quantity,
                reason,
            )
            for trade in sell_trades:
                trade["stage"] = stages
                trade["campaign"] = 1
                trade["call_put"] = contract.call_put
            trades.extend(sell_trades)
            quantity -= filled
            exit_signal_reason = reason
            exit_reason = (
                reason if quantity == 0 else "UNFILLED_ZERO_VALUE"
            )
            exit_minute = (
                contract.bars[fill_index].minute
                if quantity == 0
                else base.SESSION_MINUTES[-1]
            )
            break
        index += 1

    expired_quantity = quantity
    if expired_quantity > 0:
        trades.append(
            {
                "strategy": config.key,
                "date": date_value,
                "campaign": 1,
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
        if not exit_reason:
            exit_reason = "UNFILLED_ZERO_VALUE"
            exit_signal_reason = exit_signal_reason or "TIME"
            exit_minute = base.SESSION_MINUTES[-1]

    entry_quantity = sum(
        int(trade["quantity"])
        for trade in trades
        if trade["side"] == "BUY"
    )
    average_entry_price = (
        buy_principal / entry_quantity / MULTIPLIER
        if entry_quantity
        else 0.0
    )
    mfe_multiple, mae_pct = excursion(
        contract,
        first_fill_index,
        average_entry_price,
    )
    pnl = gross_sales - sell_fee - buy_principal - buy_fee
    return (
        DayResult(
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
            anchor_low=round(signal.anchor_low, 4),
            initial_entry_minute=contract.bars[
                first_fill_index
            ].minute,
            initial_entry_price=round(initial_price, 4),
            initial_quantity=initial_quantity,
            entry_stages=stages,
            lower_low_entries=lower_low_entries,
            last_entry_minute=contract.bars[last_entry_index].minute,
            last_bought_low=round(last_bought_low, 4),
            entry_quantity=entry_quantity,
            average_entry_price=round(average_entry_price, 6),
            buy_principal=buy_principal,
            buy_fee=buy_fee,
            planned_principal_limit=DAILY_PRINCIPAL_LIMIT,
            deployment_pct=round(
                buy_principal / DAILY_PRINCIPAL_LIMIT * 100,
                4,
            ),
            stop_mode=config.stop_mode,
            stop_ratio=config.stop_ratio,
            stop_loss_amount=config.stop_loss_amount,
            stop_price=round(stop_price, 4),
            stop_triggered=stop_triggered,
            reached_2x=reached_2x,
            gross_sales=gross_sales,
            sell_fee=sell_fee,
            exit_signal_reason=exit_signal_reason,
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
    ordered = sorted(profits, reverse=True)
    gains = sum(value for value in profits if value > 0)
    losses = -sum(value for value in profits if value < 0)
    return {
        "days": len(rows),
        "signal_days": sum(row.signal_found for row in rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in profits),
        "losing_days": sum(value < 0 for value in profits),
        "break_even_days": sum(value == 0 for value in profits),
        "total_pnl": sum(profits),
        "average_trade_day_pnl": (
            round(sum(profits) / len(profits)) if profits else 0
        ),
        "profit_factor": round(gains / losses, 4) if losses else None,
        "minimum_day_pnl": min(profits, default=0),
        "maximum_day_pnl": max(profits, default=0),
        "max_drawdown": max_drawdown(rows),
        "pnl_excluding_best_3_days": (
            sum(profits) - sum(ordered[:3])
        ),
        "pnl_excluding_best_5_days": (
            sum(profits) - sum(ordered[:5])
        ),
        "total_buy_principal": sum(
            row.buy_principal for row in traded
        ),
        "average_buy_principal": (
            round(
                sum(row.buy_principal for row in traded) / len(traded)
            )
            if traded
            else 0
        ),
        "average_deployment_pct": (
            round(
                sum(row.deployment_pct for row in traded) / len(traded),
                2,
            )
            if traded
            else 0.0
        ),
        "fully_deployed_days": sum(
            row.deployment_pct >= 95.0 for row in traded
        ),
        "stop_exit_days": sum(row.stop_triggered for row in traded),
        "target_exit_days": sum(row.reached_2x for row in traded),
        "expired_contracts": sum(
            row.expired_quantity for row in traded
        ),
        "stage_counts": {
            str(stage): sum(
                row.entry_stages == stage for row in traded
            )
            for stage in range(1, 6)
        },
    }


def grouped(
    rows: list[DayResult],
    field: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, field)), []).append(row.pnl)
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


def candidate(
    config: RuntimeConfig,
    daily: list[DayResult],
    audit_result: dict[str, object],
) -> dict[str, object]:
    development_rows = [
        row for row in daily if row.date <= DEVELOPMENT_END
    ]
    validation_rows = [
        row for row in daily if row.date > DEVELOPMENT_END
    ]
    return {
        "key": config.key,
        "tranche_count": config.tranche_count,
        "tranche_budgets": list(config.tranche_budgets),
        "stop_mode": config.stop_mode,
        "stop_ratio": config.stop_ratio,
        "stop_loss_amount": config.stop_loss_amount,
        "all": summarize(daily),
        "development": summarize(development_rows),
        "validation": summarize(validation_rows),
        "audit": audit_result,
    }


def selection_key(row: dict[str, object]) -> tuple[int, int, int]:
    development = row["development"]
    assert isinstance(development, dict)
    return (
        int(development["pnl_excluding_best_3_days"]),
        int(development["total_pnl"]),
        int(development["max_drawdown"]),
    )


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
        if group["expiry_date"] >= HIGH_VOL_START
    ]
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    states: dict[str, dict[str, object]] = {
        config.key: {
            "config": config,
            "cash": INITIAL_CASH,
            "daily": [],
            "trades": [],
        }
        for config in CONFIGS
    }

    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value,
            volatility_dates,
            volatility_values,
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        signal = parent.first_signal(
            contracts,
            ENTRY_CONFIG,
            previous_vix,
        )
        for state in states.values():
            config = state["config"]
            row, day_trades = simulate_day(
                date_value,
                previous_vix,
                config,  # type: ignore[arg-type]
                contracts,
                signal,
                int(state["cash"]),
            )
            state["cash"] = row.end_cash
            state["daily"].append(row)  # type: ignore[union-attr]
            state["trades"].extend(day_trades)  # type: ignore[union-attr]

    audits = {
        key: audit(
            state["daily"],  # type: ignore[arg-type]
            state["trades"],  # type: ignore[arg-type]
        )
        for key, state in states.items()
    }
    candidates = [
        candidate(
            state["config"],  # type: ignore[arg-type]
            state["daily"],  # type: ignore[arg-type]
            audits[key],
        )
        for key, state in states.items()
    ]
    selected = max(candidates, key=selection_key)
    result = {
        "status": "development_only_stop_research",
        "research_version": RESEARCH_VERSION,
        "parent_strategy": "S3-v0.1",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": DEVELOPMENT_END,
            "validation_used_for_selection": False,
        },
        "fixed_inputs": {
            "entry": ENTRY_CONFIG.name,
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "lower_low_tick": LOW_BREAK_TICK,
            "later_entry_cutoff": LATER_ENTRY_CUTOFF,
            "target_multiple_of_weighted_average": TARGET_MULTIPLE,
            "candidate_tranche_counts": list(TRANCHE_COUNTS),
            "candidate_stop_ratios": list(STOP_RATIOS),
            "candidate_daily_loss_amounts": list(STOP_LOSS_AMOUNTS),
            "stop_definitions": {
                "initial_price_ratio": (
                    "completed close versus the first fill price"
                ),
                "daily_loss_amount": (
                    "estimated liquidation loss versus cumulative "
                    "principal and entry fees"
                ),
            },
        },
        "selection_rule": (
            "highest development pnl excluding best 3 days, then "
            "development total pnl, then development max drawdown"
        ),
        "selected": selected,
        "candidates": candidates,
        "all_audits_passed": all(
            bool(result["passed"]) for result in audits.values()
        ),
    }
    (OUTPUT_DIR / "strategy3_v04_research_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        OUTPUT_DIR / "strategy3_v04_research_candidates.csv",
        [
            {
                "key": row["key"],
                "tranche_count": row["tranche_count"],
                "stop_mode": row["stop_mode"],
                "stop_ratio": row["stop_ratio"],
                "stop_loss_amount": row["stop_loss_amount"],
                **{
                    f"all_{name}": value
                    for name, value in row["all"].items()
                    if not isinstance(value, dict)
                },
                **{
                    f"development_{name}": value
                    for name, value in row["development"].items()
                    if not isinstance(value, dict)
                },
                **{
                    f"validation_{name}": value
                    for name, value in row["validation"].items()
                    if not isinstance(value, dict)
                },
            }
            for row in candidates
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_audits_passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
