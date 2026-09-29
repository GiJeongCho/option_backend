from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from baseline_long_backtest import (
    COMMISSION_PER_CONTRACT,
    CONTRACT_MULTIPLIER,
    DAILY_BUY_LIMIT,
    ENTRY_VOLUME_PARTICIPATION,
    INITIAL_ACCOUNT,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def audit() -> dict[str, object]:
    daily = read_csv(OUTPUT_DIR / "baseline_best_daily.csv")
    trades = read_csv(OUTPUT_DIR / "baseline_best_trades.csv")
    errors: list[str] = []

    trades_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[trade["date"]].append(trade)

    previous_end = INITIAL_ACCOUNT
    for row in daily:
        date = row["date"]
        start_cash = int(row["start_cash"])
        pnl = int(row["pnl"])
        end_cash = int(row["end_cash"])
        entry_spend = int(row["entry_spend"])
        day_trades = trades_by_date[date]

        if start_cash != previous_end:
            errors.append(f"{date}: 이전 종료자금과 시작자금 불일치")
        if end_cash != start_cash + pnl:
            errors.append(f"{date}: 종료자금 산식 불일치")
        if entry_spend > DAILY_BUY_LIMIT:
            errors.append(f"{date}: 일일 매수한도 초과 {entry_spend}")

        cash_flow_sum = sum(int(item["cash_flow"]) for item in day_trades)
        buy_outflow = -sum(
            int(item["cash_flow"])
            for item in day_trades
            if item["side"] == "BUY"
        )
        if cash_flow_sum != pnl:
            errors.append(f"{date}: 거래현금흐름과 손익 불일치")
        if buy_outflow != entry_spend:
            errors.append(f"{date}: 매수현금흐름과 투자금 불일치")

        quantity_by_code: dict[str, int] = defaultdict(int)
        for item in day_trades:
            quantity = int(item["quantity"])
            price = float(item["price"])
            cash_flow = int(item["cash_flow"])
            if item["side"] == "BUY":
                expected = -round(
                    quantity
                    * (
                        price * CONTRACT_MULTIPLIER
                        + COMMISSION_PER_CONTRACT
                    )
                )
                quantity_by_code[item["code"]] += quantity
                bar_volume = int(item["bar_volume"])
                if quantity > math.floor(
                    bar_volume * ENTRY_VOLUME_PARTICIPATION
                ):
                    errors.append(
                        f"{date} {item['code']}: 진입 거래량 한도 초과"
                    )
            else:
                expected = round(
                    quantity
                    * (
                        price * CONTRACT_MULTIPLIER
                        - COMMISSION_PER_CONTRACT
                    )
                )
                quantity_by_code[item["code"]] -= quantity
            if expected != cash_flow:
                errors.append(
                    f"{date} {item['code']}: 체결 현금흐름 산식 불일치"
                )

        for code, quantity in quantity_by_code.items():
            if quantity != 0:
                errors.append(f"{date} {code}: 미청산 수량 {quantity}")
        previous_end = end_cash

    profits = [int(row["pnl"]) for row in daily]
    sorted_profits = sorted(profits, reverse=True)
    result = {
        "passed": not errors,
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "initial_cash": int(daily[0]["start_cash"]) if daily else None,
        "ending_cash": int(daily[-1]["end_cash"]) if daily else None,
        "total_pnl": sum(profits),
        "maximum_daily_entry_spend": max(
            int(row["entry_spend"]) for row in daily
        )
        if daily
        else 0,
        "pnl_excluding_best_1_day": sum(profits) - sum(sorted_profits[:1]),
        "pnl_excluding_best_3_days": sum(profits) - sum(sorted_profits[:3]),
        "pnl_excluding_best_5_days": sum(profits) - sum(sorted_profits[:5]),
        "error_count": len(errors),
        "error_samples": errors[:20],
    }
    (OUTPUT_DIR / "baseline_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
