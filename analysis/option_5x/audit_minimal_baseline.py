from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
INITIAL_CASH = 80_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
MULTIPLIER = 250_000
COMMISSION_PER_CONTRACT = 500
ENTRY_PARTICIPATION = 0.10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def audit() -> dict[str, object]:
    daily = read_csv(OUTPUT_DIR / "minimal_baseline_daily.csv")
    trades = read_csv(OUTPUT_DIR / "minimal_baseline_trades.csv")
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        by_key[(trade["strategy"], trade["date"])].append(trade)

    errors: list[str] = []
    previous_cash: dict[str, int] = {}
    for row in daily:
        strategy = row["strategy"]
        date = row["date"]
        start = int(row["start_cash"])
        end = int(row["end_cash"])
        pnl = int(row["pnl"])
        principal = int(row["buy_principal"])
        expected_start = previous_cash.get(strategy, INITIAL_CASH)
        if start != expected_start:
            errors.append(f"{strategy} {date}: cash chain")
        if end != start + pnl:
            errors.append(f"{strategy} {date}: pnl equation")
        if principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{strategy} {date}: principal limit")

        cash_flow = 0
        open_quantity = 0
        buy_principal = 0
        buy_fee = 0
        gross_sales = 0
        sell_fee = 0
        for trade in by_key[(strategy, date)]:
            quantity = int(trade["quantity"])
            price = float(trade["price"])
            reported_principal = int(trade["principal"])
            fee = int(trade["fee"])
            cash_flow += int(trade["cash_flow"])
            if trade["side"] == "BUY":
                open_quantity += quantity
                buy_principal += reported_principal
                buy_fee += fee
                if reported_principal != round(
                    quantity * price * MULTIPLIER
                ):
                    errors.append(f"{strategy} {date}: buy value")
                if fee != quantity * COMMISSION_PER_CONTRACT:
                    errors.append(f"{strategy} {date}: buy fee")
                if quantity > math.floor(
                    int(trade["bar_volume"]) * ENTRY_PARTICIPATION
                ):
                    errors.append(
                        f"{strategy} {date}: entry participation"
                    )
            elif trade["side"] == "SELL":
                open_quantity -= quantity
                gross_sales += reported_principal
                sell_fee += fee
                if reported_principal != round(
                    quantity * price * MULTIPLIER
                ):
                    errors.append(f"{strategy} {date}: sell value")
                if fee != quantity * COMMISSION_PER_CONTRACT:
                    errors.append(f"{strategy} {date}: sell fee")
            elif trade["side"] == "EXPIRE":
                open_quantity -= quantity
                if reported_principal != 0 or fee != 0:
                    errors.append(f"{strategy} {date}: zero mark")

        if cash_flow != pnl:
            errors.append(f"{strategy} {date}: cash flow")
        if open_quantity != 0:
            errors.append(f"{strategy} {date}: open quantity")
        if buy_principal != principal:
            errors.append(f"{strategy} {date}: daily principal")
        if buy_fee != int(row["buy_fee"]):
            errors.append(f"{strategy} {date}: daily buy fee")
        if gross_sales != int(row["gross_sales"]):
            errors.append(f"{strategy} {date}: daily sales")
        if sell_fee != int(row["sell_fee"]):
            errors.append(f"{strategy} {date}: daily sell fee")
        previous_cash[strategy] = end

    result = {
        "passed": not errors,
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "strategy_count": len(previous_cash),
        "error_count": len(errors),
        "error_samples": errors[:20],
    }
    (OUTPUT_DIR / "minimal_baseline_independent_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
