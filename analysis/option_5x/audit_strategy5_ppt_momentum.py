from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_PATH = OUTPUT_DIR / "strategy5_summary.json"
DAILY_PATH = OUTPUT_DIR / "strategy5_daily.csv"
TRADES_PATH = OUTPUT_DIR / "strategy5_trades.csv"
AUDIT_PATH = OUTPUT_DIR / "strategy5_independent_audit.json"

INITIAL_CASH = 80_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
MORNING_START = 900
MORNING_END = 910
PPT_START = "20260105"
PPT_END = "20260831"


def integer(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def boolean(value: object) -> bool:
    return str(value).lower() == "true"


def total_minutes(value: int) -> int:
    return value // 100 * 60 + value % 100


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def ppt_fingerprint() -> dict[str, object]:
    matches = list(
        (BACKEND_DIR / "docs").rglob(
            "위클리옵션_만기일_가격경로_연구결과_"
            "2026_최종보완본2.pptx"
        )
    )
    if len(matches) != 1:
        return {"found": False, "matches": len(matches)}
    path = matches[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "found": True,
        "path": str(path.relative_to(BACKEND_DIR)),
        "size": path.stat().st_size,
        "sha256": digest,
    }


def run() -> dict[str, object]:
    errors: list[str] = []
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    daily = read_csv(DAILY_PATH)
    trades = read_csv(TRADES_PATH)

    if summary.get("strategy_version") != "S5-v0.2":
        errors.append("summary strategy version")
    if summary.get("scope", {}).get("start") != PPT_START:
        errors.append("summary start")
    if summary.get("scope", {}).get("end") != PPT_END:
        errors.append("summary end")
    if len(daily) != 60:
        errors.append(f"daily count {len(daily)}")
    if len({row["date"] for row in daily}) != len(daily):
        errors.append("duplicate daily date")
    if any(
        not PPT_START <= row["date"] <= PPT_END for row in daily
    ):
        errors.append("daily outside PPT period")
    forbidden = {"lsma", "moving_average", "ma5", "ma20", "vix"}
    headers = {key.lower() for key in daily[0]} if daily else set()
    if headers & forbidden:
        errors.append(f"forbidden daily fields {headers & forbidden}")

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        by_date[trade["date"]].append(trade)

    expected_cash = INITIAL_CASH
    for row in daily:
        date_value = row["date"]
        day_trades = by_date[date_value]
        pnl = integer(row["pnl"])
        if integer(row["start_cash"]) != expected_cash:
            errors.append(f"{date_value}: cash chain")
        if integer(row["end_cash"]) != expected_cash + pnl:
            errors.append(f"{date_value}: pnl equation")
        if sum(integer(trade["cash_flow"]) for trade in day_trades) != pnl:
            errors.append(f"{date_value}: trade cash flow")
        if integer(row["buy_principal"]) > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{date_value}: principal limit")

        buys = [trade for trade in day_trades if trade["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{date_value}: multiple buys")
        quantity = 0
        for trade in day_trades:
            amount = integer(trade["quantity"])
            if trade["side"] == "BUY":
                quantity += amount
            elif trade["side"] in {"SELL", "EXPIRE"}:
                quantity -= amount
            else:
                errors.append(f"{date_value}: unknown side")
        if quantity != 0:
            errors.append(f"{date_value}: open quantity {quantity}")

        signal_minute = integer(row["signal_minute"])
        entry_minute = integer(row["entry_minute"])
        confirmation_minute = integer(row["confirmation_minute"])
        if boolean(row["signal_found"]) and not (
            MORNING_START <= signal_minute <= MORNING_END
        ):
            errors.append(f"{date_value}: signal time")
        if boolean(row["entry_filled"]) and entry_minute <= signal_minute:
            errors.append(f"{date_value}: non-causal entry")
        if boolean(row["confirmed"]):
            if not (
                signal_minute < confirmation_minute < entry_minute
            ):
                errors.append(f"{date_value}: confirmation order")
            if (
                total_minutes(confirmation_minute)
                - total_minutes(signal_minute)
                > 30
            ):
                errors.append(f"{date_value}: late confirmation")
        expected_cash = integer(row["end_cash"])

    reported = summary.get("summary", {})
    if integer(reported.get("ending_cash")) != expected_cash:
        errors.append("summary ending cash")
    if integer(reported.get("total_pnl")) != (
        expected_cash - INITIAL_CASH
    ):
        errors.append("summary total pnl")

    result = {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:30],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "ppt": ppt_fingerprint(),
        "checks": [
            "cash chain and pnl equation",
            "trade cash-flow reconciliation",
            "closed quantity",
            "one buy per day",
            "5,000,000 won daily principal cap",
            "09:00~09:10 signal time",
            "30-minute confirmation then next-bar entry order",
            "PPT period and forbidden external indicator fields",
        ],
    }
    AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
