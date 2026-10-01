from __future__ import annotations

import json
from dataclasses import asdict

import minimal_baseline_backtest as base
import strategy3_v04_research as research
from strategy1_backtest import (
    DAILY_PRINCIPAL_LIMIT,
    INITIAL_CASH,
    previous_volatility,
    volatility_closes,
)
from strategy2_backtest import audit


STRATEGY_VERSION = "S3-v0.4"
FIXED_CONFIG = research.RuntimeConfig(
    tranche_count=4,
    stop_loss_amount=2_000_000,
)


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= research.HIGH_VOL_START
    ]
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    cash = INITIAL_CASH
    daily: list[research.DayResult] = []
    trades: list[dict[str, object]] = []

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
        signal = research.parent.first_signal(
            contracts,
            research.ENTRY_CONFIG,
            previous_vix,
        )
        row, day_trades = research.simulate_day(
            date_value,
            previous_vix,
            FIXED_CONFIG,
            contracts,
            signal,
            cash,
        )
        row.strategy = STRATEGY_VERSION
        for trade in day_trades:
            trade["strategy"] = STRATEGY_VERSION
        cash = row.end_cash
        daily.append(row)
        trades.extend(day_trades)

    development_rows = [
        row for row in daily if row.date <= research.DEVELOPMENT_END
    ]
    validation_rows = [
        row for row in daily if row.date > research.DEVELOPMENT_END
    ]
    summary = research.summarize(daily)
    summary.update(
        {
            "initial_cash": INITIAL_CASH,
            "ending_cash": cash,
        }
    )
    development = research.summarize(development_rows)
    validation = research.summarize(validation_rows)
    audit_result = audit(daily, trades)  # type: ignore[arg-type]
    traded = [row for row in daily if row.entry_filled]
    parent_document = json.loads(
        (research.OUTPUT_DIR / "strategy3_summary.json").read_text(
            encoding="utf-8"
        )
    )
    research_document = json.loads(
        (
            research.OUTPUT_DIR
            / "strategy3_v04_research_summary.json"
        ).read_text(encoding="utf-8")
    )
    selected = research_document["selected"]
    if selected["key"] != FIXED_CONFIG.key:
        raise RuntimeError(
            "개발 구간 선택 결과와 S3-v0.4 고정값이 다릅니다."
        )

    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S3-v0.1",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
            "development_end": research.DEVELOPMENT_END,
            "validation_used_for_selection": False,
            "historical_low_volatility_backcast": False,
        },
        "selection": {
            "candidate_count": len(research.CONFIGS),
            "selected_key": FIXED_CONFIG.key,
            "criterion": research_document["selection_rule"],
            "inputs": research_document["fixed_inputs"],
        },
        "rules": {
            "initial_entry": (
                "S3-v0.1 L60_R20_B5_MA20 signal, next actual traded "
                "bar within 5 minutes"
            ),
            "planned_daily_principal": DAILY_PRINCIPAL_LIMIT,
            "tranche_budgets": list(FIXED_CONFIG.tranche_budgets),
            "maximum_entry_stages": FIXED_CONFIG.tranche_count,
            "lower_low_entry": (
                "after a completed traded bar breaks the last bought "
                "low by at least 0.01, buy on the next actual bar"
            ),
            "entry_cutoff": research.LATER_ENTRY_CUTOFF,
            "entry_priority": (
                "2x target, then remaining lower-low tranche, then stop"
            ),
            "stop": {
                "mode": "daily_loss_amount",
                "amount": FIXED_CONFIG.stop_loss_amount,
                "mark": (
                    "cumulative principal plus entry fees minus "
                    "estimated net liquidation value at completed close"
                ),
                "execution": "sell from the next actual traded bar",
            },
            "target": (
                "sell all after completed close reaches 2x the current "
                "weighted average entry price"
            ),
            "time_exit": "signal at 15:15",
            "execution": {
                "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
                "entry_volume_participation": 0.10,
                "exit_volume_participation": 0.10,
                "slippage": 0.01,
                "commission_per_contract": 500,
            },
        },
        "summary": summary,
        "development": development,
        "validation": validation,
        "comparison_to_s3_v01": {
            "total_pnl_difference": (
                int(summary["total_pnl"])
                - int(parent_document["summary"]["total_pnl"])
            ),
            "development_pnl_difference": (
                int(development["total_pnl"])
                - int(parent_document["development"]["total_pnl"])
            ),
            "validation_pnl_difference": (
                int(validation["total_pnl"])
                - int(parent_document["validation"]["total_pnl"])
            ),
            "max_drawdown_difference": (
                int(summary["max_drawdown"])
                - int(parent_document["summary"]["max_drawdown"])
            ),
        },
        "analysis": {
            "by_entry_stages": research.grouped(
                traded,
                "entry_stages",
            ),
            "by_exit_reason": research.grouped(
                traded,
                "exit_signal_reason",
            ),
            "by_call_put": research.grouped(traded, "call_put"),
            "worst_days": [
                asdict(row)
                for row in sorted(traded, key=lambda item: item.pnl)[:15]
            ],
        },
        "audit": audit_result,
        "limitations": [
            (
                "The 4-tranche and 2,000,000 won stop were selected on "
                "data through 2026-04 only; later results were not used "
                "for ranking."
            ),
            (
                "Five million won is a planned daily cap, not a forced "
                "fill. Integer contracts, volume limits, missing lower "
                "lows, targets, and stops can leave capital unused."
            ),
            (
                "Lower-low buying deliberately adds during weakness and "
                "can fill immediately before the stop on the next bar."
            ),
            (
                "The validation period remains positive but its pnl "
                "excluding the best three days is negative."
            ),
            "Unfilled expiry residuals are valued at zero.",
        ],
    }

    research.write_csv(
        research.OUTPUT_DIR / "strategy3_v04_daily.csv",
        [asdict(row) for row in daily],
    )
    research.write_csv(
        research.OUTPUT_DIR / "strategy3_v04_trades.csv",
        trades,
    )
    (
        research.OUTPUT_DIR / "strategy3_v04_summary.json"
    ).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (
        research.OUTPUT_DIR / "strategy3_v04_audit.json"
    ).write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
