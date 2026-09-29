from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import minimal_baseline_backtest as base
import strategy2_improvement_research as s2
import strategy2_v03_research as research
from strategy1_backtest import INITIAL_CASH, previous_volatility, volatility_closes
from strategy2_backtest import audit
from strategy2_probe_research import ProbeDay, summarize


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S2-v0.3"
CONFIG = research.Config("late_1_5_all")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def period_summary(rows: list[ProbeDay]) -> dict[str, object]:
    result = summarize(rows)
    result["expiry_residual_days"] = sum(
        row.expired_quantity > 0 for row in rows
    )
    return result


def grouped(
    rows: list[ProbeDay], field: str
) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field))].append(row.pnl)
    return sorted(
        (
            {
                "group": name,
                "count": len(values),
                "profitable_count": sum(value > 0 for value in values),
                "losing_count": sum(value < 0 for value in values),
                "total_pnl": sum(values),
                "average_pnl": round(sum(values) / len(values)),
            }
            for name, values in groups.items()
        ),
        key=lambda item: int(item["total_pnl"]),
    )


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    cash = INITIAL_CASH
    daily: list[ProbeDay] = []
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
        signal = s2.first_signal(contracts, indicators, "ma20_rising")
        row, day_trades = research.simulate_day(
            date_value,
            previous_vix,
            CONFIG,
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

    historical = [row for row in daily if row.date < s2.DEVELOPMENT_START]
    development = [
        row
        for row in daily
        if s2.DEVELOPMENT_START <= row.date <= s2.DEVELOPMENT_END
    ]
    validation = [row for row in daily if row.date > s2.DEVELOPMENT_END]
    summary = period_summary(daily)
    summary["initial_cash"] = INITIAL_CASH
    summary["ending_cash"] = cash
    audit_result = audit(daily, trades)
    parent = json.loads(
        (OUTPUT_DIR / "strategy2_v02_summary.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "status": "experimental_fixed_strategy_not_promoted",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S2-v0.2",
        "selection_source": (
            "two adaptive exits compared only on 2025-10 through "
            "2026-04; validation was not used"
        ),
        "rules": {
            "entry": "same as S2-v0.2",
            "probe_budget": 2_500_000,
            "confirmation": "1.5x within 20 minutes then add",
            "confirmed_exit": "2x initial entry price",
            "unconfirmed_exit": (
                "after 20 minutes, first 1.5x close exits all"
            ),
            "daily_principal_limit": 5_000_000,
        },
        "summary": summary,
        "historical_backcast": period_summary(historical),
        "development": period_summary(development),
        "validation": period_summary(validation),
        "comparison_to_parent": {
            "all_pnl_difference": (
                int(summary["total_pnl"])
                - int(parent["summary"]["total_pnl"])
            ),
            "development_pnl_difference": (
                int(period_summary(development)["total_pnl"])
                - int(parent["development"]["total_pnl"])
            ),
            "validation_pnl_difference": (
                int(period_summary(validation)["total_pnl"])
                - int(parent["validation"]["total_pnl"])
            ),
            "promoted_over_parent": False,
        },
        "analysis": {
            "by_exit_reason": grouped(
                [row for row in daily if row.entry_filled],
                "exit_reason",
            ),
            "by_confirmation": grouped(
                [row for row in daily if row.entry_filled],
                "confirmed",
            ),
        },
        "audit": audit_result,
        "limitations": [
            "승률은 높아졌지만 S2-v0.2보다 개발·검증·전체 수익이 모두 낮았습니다.",
            "따라서 버전명은 유지하되 권장 전략으로 승격하지 않습니다.",
            "15:19까지 미체결된 잔량은 0원으로 평가했습니다.",
        ],
    }
    write_csv(
        OUTPUT_DIR / "strategy2_v03_daily.csv",
        [asdict(row) for row in daily],
    )
    write_csv(OUTPUT_DIR / "strategy2_v03_trades.csv", trades)
    (OUTPUT_DIR / "strategy2_v03_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy2_v03_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
