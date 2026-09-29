from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import minimal_baseline_backtest as base
from strategy1_backtest import (
    DAILY_PRINCIPAL_LIMIT,
    INITIAL_CASH,
    VIX_THRESHOLD,
    VOLUME_PARTICIPATION,
    previous_volatility,
    volatility_closes,
)
from strategy2_probe_research import (
    DEVELOPMENT_END,
    PREMIUM,
    ProbeConfig,
    ProbeDay,
    simulate_day,
    summarize,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S2-v0.1"
CONFIG = ProbeConfig(
    probe_budget=2_500_000,
    confirmation_multiple=1.5,
    confirmation_window=20,
    post_confirmation_floor=False,
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


def audit(
    daily: list[ProbeDay], trades: list[dict[str, object]]
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[str(trade["date"])].append(trade)
    expected_cash = INITIAL_CASH
    for row in daily:
        day_trades = trades_by_date[row.date]
        if row.start_cash != expected_cash:
            errors.append(f"{row.date}: cash chain")
        if row.end_cash != row.start_cash + row.pnl:
            errors.append(f"{row.date}: pnl equation")
        if row.buy_principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{row.date}: principal limit")
        if sum(int(item["cash_flow"]) for item in day_trades) != row.pnl:
            errors.append(f"{row.date}: trade cash flow")

        quantity = 0
        for trade in day_trades:
            amount = int(trade["quantity"])
            if trade["side"] == "BUY":
                quantity += amount
                if amount > math.floor(
                    int(trade["bar_volume"]) * VOLUME_PARTICIPATION
                ):
                    errors.append(f"{row.date}: entry participation")
            else:
                quantity -= amount
        if quantity != 0:
            errors.append(f"{row.date}: open quantity")
        expected_cash = row.end_cash
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:20],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
    }


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
        signal = base.first_signal(contracts, PREMIUM)
        day, day_trades = simulate_day(
            date_value,
            previous_vix,
            CONFIG,
            contracts,
            signal,
            cash,
        )
        day.strategy = STRATEGY_VERSION
        for trade in day_trades:
            trade["strategy"] = STRATEGY_VERSION
        cash = day.end_cash
        daily.append(day)
        trades.extend(day_trades)

    development = [
        row for row in daily if row.date <= DEVELOPMENT_END
    ]
    validation = [
        row for row in daily if row.date > DEVELOPMENT_END
    ]
    all_summary = summarize(daily)
    all_summary["initial_cash"] = INITIAL_CASH
    all_summary["ending_cash"] = cash
    all_summary["expiry_residual_days"] = sum(
        row.expired_quantity > 0 for row in daily
    )
    exits = grouped(
        [row for row in daily if row.entry_filled], "exit_reason"
    )
    confirmation = grouped(
        [row for row in daily if row.entry_filled], "confirmed"
    )
    worst_days = [
        asdict(row)
        for row in sorted(daily, key=lambda item: item.pnl)[:15]
        if row.pnl < 0
    ]
    audit_result = audit(daily, trades)
    result = {
        "status": "experimental_fixed_strategy",
        "strategy_version": STRATEGY_VERSION,
        "selection_source": (
            "24 probe/confirmation candidates selected only on "
            "data through 2026-04"
        ),
        "rules": {
            "previous_vix_min": VIX_THRESHOLD,
            "premium_range": PREMIUM.name,
            "probe_budget": CONFIG.probe_budget,
            "confirmation_multiple": CONFIG.confirmation_multiple,
            "confirmation_window_minutes": CONFIG.confirmation_window,
            "daily_principal_limit": DAILY_PRINCIPAL_LIMIT,
            "target_multiple_of_initial_price": 2.0,
            "fixed_price_stop": None,
            "maximum_campaigns_per_day": 1,
        },
        "summary": all_summary,
        "development": summarize(development),
        "validation": summarize(validation),
        "analysis": {
            "by_exit_reason": exits,
            "by_confirmation": confirmation,
            "worst_days": worst_days,
        },
        "audit": audit_result,
        "limitations": [
            "프리미엄 2.50~5.00은 이전 전체 자료 탐색의 영향을 받았습니다.",
            "후반 구간은 양수지만 최고 수익 3일을 제외하면 음수입니다.",
            "고정 손절 대신 최초 원금을 250만원으로 제한하므로 확인 후 증액된 거래는 최대 500만원을 잃을 수 있습니다.",
            "15:19까지 미체결된 잔량은 0원으로 평가했습니다.",
        ],
    }
    daily_rows = [asdict(row) for row in daily]
    write_csv(OUTPUT_DIR / "strategy2_daily.csv", daily_rows)
    write_csv(OUTPUT_DIR / "strategy2_trades.csv", trades)
    (OUTPUT_DIR / "strategy2_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy2_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not audit_result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
