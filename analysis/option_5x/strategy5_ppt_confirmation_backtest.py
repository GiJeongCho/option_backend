from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as literal
import strategy5_ppt_momentum_research as research


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S5-v0.2"


def audit(
    daily: list[literal.DayResult],
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
        buys = [item for item in day_trades if item["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{row.date}: multiple entries")
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
            errors.append(f"{row.date}: open quantity")
        if row.signal_minute is not None and not (
            literal.MORNING_START
            <= row.signal_minute
            <= literal.MORNING_END
        ):
            errors.append(f"{row.date}: signal time")
        if row.entry_filled:
            if row.confirmation_minute is None:
                errors.append(f"{row.date}: missing confirmation")
            elif row.entry_minute is None:
                errors.append(f"{row.date}: missing entry")
            elif not (
                row.signal_minute
                < row.confirmation_minute
                < row.entry_minute
            ):
                errors.append(f"{row.date}: causal order")
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


def preserve_literal_outputs() -> dict[str, object] | None:
    source_summary = OUTPUT_DIR / "strategy5_summary.json"
    if not source_summary.exists():
        return None
    document = json.loads(source_summary.read_text(encoding="utf-8"))
    if document.get("strategy_version") != literal.STRATEGY_VERSION:
        return None
    copies = (
        ("strategy5_summary.json", "strategy5_v01_summary.json"),
        ("strategy5_daily.csv", "strategy5_v01_daily.csv"),
        ("strategy5_trades.csv", "strategy5_v01_trades.csv"),
        ("strategy5_audit.json", "strategy5_v01_audit.json"),
    )
    for source_name, target_name in copies:
        source = OUTPUT_DIR / source_name
        if source.exists():
            (OUTPUT_DIR / target_name).write_bytes(source.read_bytes())
    return document


def run() -> dict[str, object]:
    literal_document = preserve_literal_outputs()
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if literal.PPT_START
        <= group["expiry_date"]
        <= literal.PPT_END
    ]
    if not groups:
        raise RuntimeError("PPT 기간 만기일을 찾을 수 없습니다.")
    files = base.target_file_map()
    cash = base.INITIAL_CASH
    daily: list[literal.DayResult] = []
    trades: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        setup = research.first_confirmed_setup(contracts)
        row, day_trades = research.simulate_day(
            date_value,
            contracts,
            setup,
            cash,
        )
        row.strategy = STRATEGY_VERSION
        for trade in day_trades:
            trade["strategy"] = STRATEGY_VERSION
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)

    development = [
        row for row in daily if row.date <= literal.DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > literal.DEVELOPMENT_END
    ]
    summary = literal.summarize(daily)
    summary.update(
        {
            "initial_cash": base.INITIAL_CASH,
            "ending_cash": cash,
        }
    )
    traded = [row for row in daily if row.entry_filled]
    audit_result = audit(daily, trades)
    comparison: dict[str, object] = {}
    if literal_document is not None:
        literal_summary = dict(literal_document["summary"])
        comparison = {
            "literal_strategy": literal.STRATEGY_VERSION,
            "literal_total_pnl": int(literal_summary["total_pnl"]),
            "literal_max_drawdown": int(
                literal_summary["max_drawdown"]
            ),
            "total_pnl_difference": (
                int(summary["total_pnl"])
                - int(literal_summary["total_pnl"])
            ),
            "max_drawdown_difference": (
                int(summary["max_drawdown"])
                - int(literal_summary["max_drawdown"])
            ),
        }

    result = {
        "status": "experimental_ppt_confirmation_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": literal.STRATEGY_VERSION,
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
            "development_end": literal.DEVELOPMENT_END,
            "validation_is_independent": False,
        },
        "rules": {
            "morning_observation_window": [
                literal.MORNING_START,
                literal.MORNING_END,
            ],
            "premium_range": [
                literal.PREMIUM_MIN,
                literal.PREMIUM_MAX,
            ],
            "within_bar_observed_price": (
                "maximum qualifying OHLC value after the bar completes"
            ),
            "contract_selection": (
                "first completed 1.5x confirmation across all observed "
                "contracts, then highest confirmation-bar volume"
            ),
            "maximum_campaigns_per_day": 1,
            "daily_principal_limit": base.DAILY_PRINCIPAL_LIMIT,
            "entry": (
                "buy only on the next actual traded bar after a completed "
                "1.5x confirmation bar"
            ),
            "confirmation_multiple": literal.CONFIRMATION_MULTIPLE,
            "confirmation_window_minutes": (
                literal.CONFIRMATION_WINDOW_MINUTES
            ),
            "unconfirmed_action": "do not buy",
            "confirmed_exit": {
                "2x_observed_price": "sell half",
                "5x_observed_price": "sell half of remainder",
                "10x_observed_price": "sell all remainder",
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
        "development": literal.summarize(development),
        "validation": literal.summarize(validation),
        "comparison_to_literal_entry": comparison,
        "analysis": {
            "by_confirmation": literal.grouped(traded, "confirmed"),
            "by_call_put": literal.grouped(traded, "call_put"),
            "by_exit_reason": literal.grouped(traded, "exit_reason"),
            "worst_days": [
                asdict(row)
                for row in sorted(traded, key=lambda item: item.pnl)[:15]
            ],
        },
        "audit": audit_result,
        "limitations": [
            (
                "Both S5 versions were derived after reading aggregate "
                "results from the full PPT period. The temporal split is "
                "diagnostic, not an independent holdout."
            ),
            (
                "S5-v0.2 was introduced after the literal entry-first "
                "S5-v0.1 showed large unconfirmed losses, so its comparison "
                "is iterative research rather than pre-registered evidence."
            ),
            (
                "The PPT did not define a one-contract portfolio. First "
                "confirmation and volume are deterministic execution "
                "tie-breakers."
            ),
            (
                "OHLC touches are converted to next-actual-bar execution "
                "with slippage, commission, and 10% volume participation."
            ),
            (
                "No LSMA, moving average, VIX, or futures direction is "
                "used."
            ),
        ],
    }

    write_csv(
        OUTPUT_DIR / "strategy5_daily.csv",
        [asdict(row) for row in daily],
    )
    write_csv(OUTPUT_DIR / "strategy5_trades.csv", trades)
    (OUTPUT_DIR / "strategy5_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy5_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
