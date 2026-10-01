from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_PATH = OUTPUT_DIR / "strategy3_v04_summary.json"
DAILY_PATH = OUTPUT_DIR / "strategy3_v04_daily.csv"
TRADES_PATH = OUTPUT_DIR / "strategy3_v04_trades.csv"
AUDIT_PATH = OUTPUT_DIR / "strategy3_v04_independent_audit.json"

INITIAL_CASH = 80_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
MULTIPLIER = 250_000
COMMISSION = 500
MAXIMUM_ENTRY_STAGES = 4


def integer(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def number(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def boolean(value: object) -> bool:
    return str(value).lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def run() -> dict[str, object]:
    errors: list[str] = []
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    daily = read_csv(DAILY_PATH)
    trades = read_csv(TRADES_PATH)

    if summary.get("strategy_version") != "S3-v0.4":
        errors.append("strategy version")
    if summary.get("parent_strategy") != "S3-v0.1":
        errors.append("parent strategy")
    if (
        summary.get("rules", {})
        .get("stop", {})
        .get("amount")
        != 2_000_000
    ):
        errors.append("fixed stop amount")
    if len(daily) != 83:
        errors.append(f"daily rows {len(daily)}")
    if len({row["date"] for row in daily}) != len(daily):
        errors.append("duplicate daily dates")

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        by_date[trade["date"]].append(trade)

    expected_cash = INITIAL_CASH
    for row in daily:
        date_value = row["date"]
        day_trades = sorted(
            by_date[date_value],
            key=lambda item: (
                integer(item["minute"]),
                0 if item["side"] == "BUY" else 1,
            ),
        )
        pnl = integer(row["pnl"])
        if integer(row["start_cash"]) != expected_cash:
            errors.append(f"{date_value}: cash chain")
        if integer(row["end_cash"]) != expected_cash + pnl:
            errors.append(f"{date_value}: pnl equation")
        if sum(integer(trade["cash_flow"]) for trade in day_trades) != pnl:
            errors.append(f"{date_value}: trade cash flow")

        buys = [trade for trade in day_trades if trade["side"] == "BUY"]
        if len(buys) > MAXIMUM_ENTRY_STAGES:
            errors.append(f"{date_value}: entry stage count")
        if buys:
            expected_stages = list(range(1, len(buys) + 1))
            actual_stages = [integer(trade["stage"]) for trade in buys]
            if actual_stages != expected_stages:
                errors.append(f"{date_value}: entry stage order")
        buy_principal = sum(
            integer(trade["principal"]) for trade in buys
        )
        if buy_principal != integer(row["buy_principal"]):
            errors.append(f"{date_value}: buy principal reconciliation")
        if buy_principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{date_value}: principal limit")

        quantity = 0
        for trade in day_trades:
            side = trade["side"]
            amount = integer(trade["quantity"])
            principal = integer(trade["principal"])
            fee = integer(trade["fee"])
            if side == "BUY":
                quantity += amount
                if fee != amount * COMMISSION:
                    errors.append(f"{date_value}: buy fee")
                if principal != round(
                    amount * number(trade["price"]) * MULTIPLIER
                ):
                    errors.append(f"{date_value}: buy principal")
                if integer(trade["cash_flow"]) != -(principal + fee):
                    errors.append(f"{date_value}: buy cash flow")
            elif side == "SELL":
                quantity -= amount
                if fee != amount * COMMISSION:
                    errors.append(f"{date_value}: sell fee")
                if principal != round(
                    amount * number(trade["price"]) * MULTIPLIER
                ):
                    errors.append(f"{date_value}: sell principal")
                if integer(trade["cash_flow"]) != principal - fee:
                    errors.append(f"{date_value}: sell cash flow")
            elif side == "EXPIRE":
                quantity -= amount
                if principal or fee or integer(trade["cash_flow"]):
                    errors.append(f"{date_value}: expiry cash flow")
            else:
                errors.append(f"{date_value}: unknown side {side}")
        if quantity != 0:
            errors.append(f"{date_value}: open quantity {quantity}")
        if boolean(row["stop_triggered"]) and not any(
            trade["reason"] == "DAILY_LOSS_STOP"
            for trade in day_trades
        ):
            errors.append(f"{date_value}: stop flag")
        expected_cash = integer(row["end_cash"])

    reported = summary["summary"]
    if integer(reported["ending_cash"]) != expected_cash:
        errors.append("ending cash")
    if integer(reported["total_pnl"]) != expected_cash - INITIAL_CASH:
        errors.append("total pnl")
    if integer(reported["stop_exit_days"]) != sum(
        boolean(row["stop_triggered"]) for row in daily
    ):
        errors.append("stop day count")

    result = {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:30],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "checks": [
            "cash chain and pnl equation",
            "trade cash-flow reconciliation",
            "integer quantity closure",
            "four-stage maximum and stage order",
            "5,000,000 won principal cap",
            "contract multiplier and commission",
            "2,000,000 won stop ledger consistency",
        ],
    }
    AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
