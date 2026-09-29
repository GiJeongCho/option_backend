from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import minimal_baseline_backtest as base
import strategy2_improvement_research as research
from strategy1_backtest import INITIAL_CASH, volatility_closes
from strategy2_backtest import audit
from strategy2_probe_research import ProbeDay, summarize


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S2-v0.2"
CONFIG = research.ImprovementConfig(
    entry_filter="ma20_rising",
    confirmation_window=20,
    failure_exit="hold",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def period_summary(rows: list[ProbeDay]) -> dict[str, object]:
    result = summarize(rows)
    result["expiry_residual_days"] = sum(
        row.expired_quantity > 0 for row in rows
    )
    return result


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    states, _, trades = research.run_states(
        (CONFIG,),
        groups,
        files,
        volatility_dates,
        volatility_values,
    )
    state = states[0]
    assert state.daily is not None
    daily = state.daily
    for row in daily:
        row.strategy = STRATEGY_VERSION
    for trade in trades:
        trade["strategy"] = STRATEGY_VERSION

    historical = [
        row for row in daily if row.date < research.DEVELOPMENT_START
    ]
    development = [
        row
        for row in daily
        if research.DEVELOPMENT_START
        <= row.date
        <= research.DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > research.DEVELOPMENT_END
    ]
    summary = period_summary(daily)
    summary["initial_cash"] = INITIAL_CASH
    summary["ending_cash"] = INITIAL_CASH + int(summary["total_pnl"])
    audit_result = audit(daily, trades)
    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "parent_strategy": "S2-v0.1",
        "selection_source": (
            "10 entry filters selected only on 2025-10 through "
            "2026-04 by pnl excluding best 5 days; 7 failure-exit "
            "candidates tested after entry selection"
        ),
        "rules": {
            "previous_vix_min": 30.0,
            "premium_range": "2.50-5.00",
            "base_signal": (
                "MA5 recovery AND MA5>MA20 AND previous-5-minute-high "
                "breakout AND volume above prior-20-minute average"
            ),
            "added_entry_filter": (
                "signal-minute MA20 > MA20 from 5 minutes earlier"
            ),
            "moving_averages_used": [5, 20],
            "moving_averages_researched": [5, 10, 20, 40, 60],
            "probe_budget": 2_500_000,
            "confirmation_multiple": 1.5,
            "confirmation_window_minutes": 20,
            "daily_principal_limit": 5_000_000,
            "target_multiple_of_initial_price": 2.0,
            "fixed_price_stop": None,
            "failed_confirmation_exit": None,
            "maximum_campaigns_per_day": 1,
        },
        "summary": summary,
        "historical_backcast": period_summary(historical),
        "development": period_summary(development),
        "validation": period_summary(validation),
        "analysis": {
            "by_exit_reason": grouped(
                [row for row in daily if row.entry_filled],
                "exit_reason",
            ),
            "by_confirmation": grouped(
                [row for row in daily if row.entry_filled],
                "confirmed",
            ),
            "by_call_put": grouped(
                [row for row in daily if row.entry_filled],
                "call_put",
            ),
            "worst_days": [
                asdict(row)
                for row in sorted(daily, key=lambda item: item.pnl)[:15]
                if row.pnl < 0
            ],
        },
        "audit": audit_result,
        "limitations": [
            "MA20 기울기 조건은 10개 진입 후보를 개발 구간에서 비교해 선택한 탐색 결과입니다.",
            "후반 구간은 선택에 사용하지 않았지만 독립 신규 만기일 전진검증은 아닙니다.",
            "고정 손절을 쓰지 않아 확인 전 최대 약 250만원, 확인 후 최대 500만원과 비용을 잃을 수 있습니다.",
            "15:19까지 미체결된 잔량은 0원으로 평가했습니다.",
        ],
    }
    daily_rows = [asdict(row) for row in daily]
    write_csv(OUTPUT_DIR / "strategy2_v02_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "strategy2_v02_trades.csv", trades)
    (OUTPUT_DIR / "strategy2_v02_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy2_v02_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
