from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_PATH = OUTPUT_DIR / "strategy5_v03_summary.json"
EVENT_SUMMARY_PATH = (
    OUTPUT_DIR / "strategy5_v03_event_research_summary.json"
)
EVENTS_PATH = OUTPUT_DIR / "strategy5_v03_contract_events.csv"
AUDIT_PATH = OUTPUT_DIR / "strategy5_v03_independent_audit.json"

BRANCHES = ("EARLY", "LATE")
INITIAL_CASH = 80_000_000
DAILY_PRINCIPAL_LIMIT = 5_000_000
MULTIPLIER = 250_000
COMMISSION = 500
ENTRY_PARTICIPATION = 0.10
EARLY_END = 930
LATE_END = 1430


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


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


def event_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    ambiguous = [
        row for row in rows if boolean(row["initial_bar_ambiguous"])
    ]
    ordered = [
        row for row in rows if not boolean(row["initial_bar_ambiguous"])
    ]
    early = [row for row in ordered if row["branch"] == "EARLY"]
    late = [row for row in ordered if row["branch"] == "LATE"]
    never = [row for row in ordered if row["branch"] == "NEVER"]
    unconfirmed = late + never
    return {
        "eligible_contracts": len(rows),
        "ambiguous_initial_bar": len(ambiguous),
        "ordered_contracts": len(ordered),
        "early": len(early),
        "early_reached_2x": sum(
            boolean(row["reached_2x_including_confirmation"])
            for row in early
        ),
        "early_unconfirmed": len(unconfirmed),
        "early_unconfirmed_reached_003": sum(
            boolean(row["reached_003_after_early_window"])
            for row in unconfirmed
        ),
        "late": len(late),
        "late_reached_2x": sum(
            boolean(row["reached_2x_after_confirmation"])
            for row in late
        ),
        "late_reached_5x": sum(
            boolean(row["reached_5x_after_confirmation"])
            for row in late
        ),
        "late_reached_10x": sum(
            boolean(row["reached_10x_after_confirmation"])
            for row in late
        ),
        "never": len(never),
        "never_reached_003": sum(
            boolean(row["reached_003_after_early_window"])
            for row in never
        ),
    }


def audit_events(
    summary: dict[str, object],
    event_summary: dict[str, object],
    rows: list[dict[str, str]],
    errors: list[str],
) -> dict[str, int]:
    counts = event_counts(rows)
    expected = {
        "eligible_contracts": 1219,
        "ambiguous_initial_bar": 82,
        "ordered_contracts": 1137,
        "early": 476,
        "early_reached_2x": 358,
        "early_unconfirmed": 661,
        "early_unconfirmed_reached_003": 619,
        "late": 186,
        "late_reached_2x": 142,
        "late_reached_5x": 72,
        "late_reached_10x": 40,
        "never": 475,
        "never_reached_003": 475,
    }
    for name, value in expected.items():
        if counts[name] != value:
            errors.append(
                f"events {name}: {counts[name]} != {value}"
            )

    selected = dict(event_summary["modes"])[  # type: ignore[arg-type]
        "first_qualifying_open"
    ]
    selected_counts = {
        "eligible_contracts": selected["eligible_contracts"],
        "ambiguous_initial_bar": selected["ambiguous_initial_bar"],
        "ordered_contracts": selected["ordered_contracts"],
        "early": selected["early"]["count"],
        "early_reached_2x": selected["early"][
            "reached_2x_including_confirmation"
        ],
        "early_unconfirmed": selected["early_unconfirmed"]["count"],
        "early_unconfirmed_reached_003": selected[
            "early_unconfirmed"
        ]["reached_003"],
        "late": selected["late"]["count"],
        "late_reached_2x": selected["late"][
            "reached_2x_after_confirmation"
        ],
        "late_reached_5x": selected["late"][
            "reached_5x_after_confirmation"
        ],
        "late_reached_10x": selected["late"][
            "reached_10x_after_confirmation"
        ],
        "never": selected["never"]["count"],
        "never_reached_003": selected["never"]["reached_003"],
    }
    if counts != {
        key: integer(value) for key, value in selected_counts.items()
    }:
        errors.append("event CSV and event summary mismatch")
    if summary["event_study"] != selected:
        errors.append("strategy and event summary mismatch")
    return counts


def candidates_by_date(
    event_rows: list[dict[str, str]],
    branch: str,
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        if row["branch"] != branch:
            continue
        confirmation = integer(
            row[
                "early_confirmation_minute"
                if branch == "EARLY"
                else "late_confirmation_minute"
            ]
        )
        if branch == "LATE" and confirmation > LATE_END:
            continue
        result[row["date"]].append(row)
    return result


def audit_branch(
    branch: str,
    document: dict[str, object],
    event_rows: list[dict[str, str]],
) -> dict[str, object]:
    errors: list[str] = []
    suffix = branch.lower()
    daily = read_csv(
        OUTPUT_DIR / f"strategy5_v03_{suffix}_daily.csv"
    )
    trades = read_csv(
        OUTPUT_DIR / f"strategy5_v03_{suffix}_trades.csv"
    )
    version = f"S5-v0.3-{branch}"
    portfolio = dict(document["portfolios"])[branch]  # type: ignore[arg-type]
    summary = portfolio["summary"]

    if portfolio["strategy_version"] != version:
        errors.append("portfolio strategy version")
    if len(daily) != integer(document["scope"]["expiry_days"]):
        errors.append(f"daily rows {len(daily)}")
    if len({row["date"] for row in daily}) != len(daily):
        errors.append("duplicate daily dates")

    candidates = candidates_by_date(event_rows, branch)
    trades_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[trade["date"]].append(trade)

    expected_cash = INITIAL_CASH
    for row in daily:
        date_value = row["date"]
        day_trades = sorted(
            trades_by_date[date_value],
            key=lambda item: (
                integer(item["minute"]),
                0 if item["side"] == "BUY" else 1,
            ),
        )
        pnl = integer(row["pnl"])
        if row["strategy"] != version:
            errors.append(f"{date_value}: daily strategy")
        if integer(row["start_cash"]) != expected_cash:
            errors.append(f"{date_value}: cash chain")
        if integer(row["end_cash"]) != expected_cash + pnl:
            errors.append(f"{date_value}: pnl equation")
        if sum(integer(item["cash_flow"]) for item in day_trades) != pnl:
            errors.append(f"{date_value}: trade cash flow")

        date_candidates = candidates.get(date_value, [])
        has_signal = bool(date_candidates)
        if boolean(row["signal_found"]) != has_signal:
            errors.append(f"{date_value}: event signal mismatch")
        if has_signal:
            field = (
                "early_confirmation_minute"
                if branch == "EARLY"
                else "late_confirmation_minute"
            )
            earliest = min(
                integer(item[field]) for item in date_candidates
            )
            if integer(row["confirmation_minute"]) != earliest:
                errors.append(f"{date_value}: not earliest confirmation")

        buys = [item for item in day_trades if item["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{date_value}: multiple buys")
        if boolean(row["entry_filled"]) != bool(buys):
            errors.append(f"{date_value}: entry flag")
        buy_principal = sum(integer(item["principal"]) for item in buys)
        if buy_principal != integer(row["buy_principal"]):
            errors.append(f"{date_value}: buy principal reconciliation")
        if buy_principal > DAILY_PRINCIPAL_LIMIT:
            errors.append(f"{date_value}: principal cap")
        if buys and (
            buy_principal
            + 2 * sum(integer(item["fee"]) for item in buys)
            > integer(row["start_cash"])
        ):
            errors.append(f"{date_value}: available cash")

        quantity = 0
        for trade in day_trades:
            if trade["strategy"] != version:
                errors.append(f"{date_value}: trade strategy")
            amount = integer(trade["quantity"])
            principal = integer(trade["principal"])
            fee = integer(trade["fee"])
            cash_flow = integer(trade["cash_flow"])
            side = trade["side"]
            if side == "BUY":
                quantity += amount
                if trade["reason"] != f"{branch}_1_5X_CONFIRM_ENTRY":
                    errors.append(f"{date_value}: buy reason")
                if amount > math.floor(
                    integer(trade["bar_volume"])
                    * ENTRY_PARTICIPATION
                ):
                    errors.append(f"{date_value}: entry participation")
                if fee != amount * COMMISSION:
                    errors.append(f"{date_value}: buy fee")
                if principal != round(
                    amount * number(trade["price"]) * MULTIPLIER
                ):
                    errors.append(f"{date_value}: buy principal")
                if cash_flow != -(principal + fee):
                    errors.append(f"{date_value}: buy cash flow")
            elif side == "SELL":
                quantity -= amount
                if fee != amount * COMMISSION:
                    errors.append(f"{date_value}: sell fee")
                if principal != round(
                    amount * number(trade["price"]) * MULTIPLIER
                ):
                    errors.append(f"{date_value}: sell principal")
                if cash_flow != principal - fee:
                    errors.append(f"{date_value}: sell cash flow")
            elif side == "EXPIRE":
                quantity -= amount
                if principal or fee or cash_flow:
                    errors.append(f"{date_value}: expiry cash flow")
            else:
                errors.append(f"{date_value}: unknown side {side}")
        if quantity != 0:
            errors.append(f"{date_value}: open quantity {quantity}")

        if boolean(row["entry_filled"]):
            observation = integer(row["observation_minute"])
            confirmation = integer(row["confirmation_minute"])
            entry = integer(row["entry_minute"])
            if not observation < confirmation < entry:
                errors.append(f"{date_value}: causal order")
            if branch == "EARLY" and confirmation > EARLY_END:
                errors.append(f"{date_value}: early window")
            if branch == "LATE" and not (
                EARLY_END < confirmation <= LATE_END
            ):
                errors.append(f"{date_value}: late window")
        expected_cash = integer(row["end_cash"])

    if integer(summary["initial_cash"]) != INITIAL_CASH:
        errors.append("summary initial cash")
    if integer(summary["ending_cash"]) != expected_cash:
        errors.append("summary ending cash")
    if integer(summary["total_pnl"]) != expected_cash - INITIAL_CASH:
        errors.append("summary total pnl")
    if integer(summary["signal_days"]) != sum(
        boolean(row["signal_found"]) for row in daily
    ):
        errors.append("summary signal days")
    if integer(summary["trade_days"]) != sum(
        boolean(row["entry_filled"]) for row in daily
    ):
        errors.append("summary trade days")
    if integer(summary["profitable_days"]) != sum(
        integer(row["pnl"]) > 0 for row in daily
    ):
        errors.append("summary profitable days")
    if integer(summary["losing_days"]) != sum(
        integer(row["pnl"]) < 0 for row in daily
    ):
        errors.append("summary losing days")

    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:30],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
    }


def run() -> dict[str, object]:
    document = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    event_summary = json.loads(
        EVENT_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    event_rows = read_csv(EVENTS_PATH)
    errors: list[str] = []

    if document.get("strategy_version") != "S5-v0.3":
        errors.append("strategy version")
    if document.get("parent_strategy") != "S5-v0.2":
        errors.append("parent strategy")
    counts = audit_events(
        document,
        event_summary,
        event_rows,
        errors,
    )
    branch_results = {
        branch: audit_branch(branch, document, event_rows)
        for branch in BRANCHES
    }
    for branch, result in branch_results.items():
        if not result["passed"]:
            errors.extend(
                f"{branch}: {message}"
                for message in result["error_samples"]
            )

    result = {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:50],
        "event_denominators": counts,
        "branches": branch_results,
        "checks": [
            "contract-event denominators recomputed from CSV",
            "661 early-unconfirmed contracts kept as the late denominator",
            "cash chain and trade cash-flow reconciliation",
            "integer quantity closure and one buy per branch per day",
            "5,000,000 won independent branch principal cap",
            "contract multiplier, commission, and entry participation",
            "P0 observation < confirmation < next-bar entry",
            "EARLY <=09:30 and LATE 09:31~14:30",
            "earliest confirmation per branch and expiry date",
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
