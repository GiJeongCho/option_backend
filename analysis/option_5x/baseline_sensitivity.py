from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import baseline_long_backtest as engine


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class Scenario:
    name: str
    slippage: float
    commission: int
    participation: float
    extra_execution_bars: int


SCENARIOS = (
    Scenario("baseline", 0.01, 500, 0.10, 0),
    Scenario("double_cost", 0.02, 1_000, 0.10, 0),
    Scenario("entry_participation_2pct", 0.01, 500, 0.02, 0),
    Scenario("one_extra_bar_delay", 0.01, 500, 0.10, 1),
    Scenario("combined_strict", 0.02, 1_000, 0.02, 1),
)


def max_drawdown(values: list[int]) -> int:
    peak = values[0]
    worst = 0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def summarize(
    scenario: Scenario, daily: list[dict[str, object]]
) -> dict[str, object]:
    profits = [int(item["pnl"]) for item in daily]
    best_days = sorted(profits, reverse=True)
    values = [engine.INITIAL_ACCOUNT] + [
        int(item["end_cash"]) for item in daily
    ]
    late = [
        int(item["pnl"])
        for item in daily
        if str(item["date"]) >= "20260501"
    ]
    return {
        "scenario": scenario.name,
        "slippage": scenario.slippage,
        "commission": scenario.commission,
        "entry_participation": scenario.participation,
        "extra_execution_bars": scenario.extra_execution_bars,
        "trade_days": sum(int(item["campaigns"]) > 0 for item in daily),
        "profitable_days": sum(value > 0 for value in profits),
        "total_pnl": sum(profits),
        "ending_cash": int(daily[-1]["end_cash"]),
        "maximum_day_pnl": max(profits),
        "minimum_day_pnl": min(profits),
        "max_drawdown": max_drawdown(values),
        "pnl_excluding_best_3_days": sum(profits) - sum(best_days[:3]),
        "pnl_excluding_best_5_days": sum(profits) - sum(best_days[:5]),
        "pnl_2026_05_to_08": sum(late),
        "maximum_daily_entry_spend": max(
            int(item["entry_spend"]) for item in daily
        ),
    }


def selected_config() -> engine.StrategyConfig:
    premium = next(
        item for item in engine.PREMIUM_RANGES if item.name == "5.01-10.00"
    )
    exit_profile = next(
        item for item in engine.EXIT_PROFILES if item.name == "wide"
    )
    return engine.StrategyConfig(premium, "recovery", exit_profile)


def run() -> dict[str, object]:
    config = selected_config()
    results: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        engine.SLIPPAGE = scenario.slippage
        engine.COMMISSION_PER_CONTRACT = scenario.commission
        engine.ENTRY_VOLUME_PARTICIPATION = scenario.participation
        engine.EXTRA_EXECUTION_BARS = scenario.extra_execution_bars
        daily, _ = engine.rerun_best(config)
        results.append(summarize(scenario, daily))

    result = {
        "strategy": config.key,
        "selection_rule": "2025-10 through 2026-04 highest total PnL",
        "limitations": [
            "exit quantity is not capped by minute volume",
            "all dates have been used in earlier exploratory research",
            "missing transaction minutes are not reconstructed from quotes",
        ],
        "scenarios": results,
    }
    (OUTPUT_DIR / "baseline_sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
