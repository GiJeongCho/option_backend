from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = TEMP_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "s6_risk_sensitivity_summary.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEMP_ROOT))

import minimal_baseline_backtest as base  # noqa: E402
import s6_candidate_backtest as candidate  # noqa: E402
import s6_ladder_backtest as ladder  # noqa: E402


DEVELOPMENT_END = "20260430"
SELECTOR = "ANY_VIXPUT_RV20"
RISK_LEVELS = (0.05, 0.075, 0.10)
PLAN_NAMES = (
    "FULL_2X",
    "HALF_2X_HALF_10X",
    "LADDER_2X_5X_10X",
)


def main() -> None:
    events = pd.read_csv(candidate.EVENT_PATH, encoding="utf-8-sig")
    events["expiry_date"] = events["expiry_date"].astype(str).str.zfill(8)
    selected = candidate.selected_events(events)[SELECTOR]
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if str(group["expiry_date"]) >= str(events["expiry_date"].min())
    ]
    files = base.target_file_map()
    accounts = {
        f"S6-RISK-{round(risk * 1000):03d}-{plan_name}": (
            ladder.LadderAccount(
                key=f"S6-RISK-{round(risk * 1000):03d}-{plan_name}",
                selector=SELECTOR,
                plan_name=plan_name,
                risk_fraction=risk,
            )
        )
        for risk in RISK_LEVELS
        for plan_name in PLAN_NAMES
    }
    for index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        event = selected.get(date_value)
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if event is not None and date_value in files
            else {}
        )
        for account in accounts.values():
            start_cash = account.cash
            if event is None:
                row = candidate.empty_day(account, date_value, start_cash)
                row["exit_plan"] = account.plan_name
                trades: list[dict[str, object]] = []
            else:
                contract = contracts.get(str(event["option_code"]))
                if contract is None:
                    row = candidate.empty_day(
                        account,
                        date_value,
                        start_cash,
                        signal=True,
                        skip_reason="계약 원자료 없음",
                    )
                    row["exit_plan"] = account.plan_name
                    trades = []
                else:
                    row, trades = ladder.simulate_position(
                        account,
                        date_value,
                        contract,
                        event,
                    )
            account.cash = int(row["end_cash"])
            assert account.daily is not None
            assert account.trades is not None
            account.daily.append(row)
            account.trades.extend(trades)
        if index % 20 == 0 or index == len(groups):
            print(f"{index}/{len(groups)} 만기 처리", flush=True)

    strategies: dict[str, object] = {}
    for key, account in accounts.items():
        assert account.daily is not None
        assert account.trades is not None
        development = [
            row for row in account.daily if str(row["date"]) <= DEVELOPMENT_END
        ]
        validation = [
            row for row in account.daily if str(row["date"]) > DEVELOPMENT_END
        ]
        audit = candidate.audit(account)
        strategies[key] = {
            "risk_fraction": account.risk_fraction,
            "exit_plan": account.plan_name,
            "summary": ladder.summarize(account.daily),
            "development": ladder.summarize(development),
            "validation": ladder.summarize(validation),
            "audit": audit,
        }
        stem = key.lower().replace("-", "_")
        candidate.write_csv(OUTPUT_DIR / f"{stem}_daily.csv", account.daily)
        candidate.write_csv(OUTPUT_DIR / f"{stem}_trades.csv", account.trades)

    result = {
        "status": "temporary_s6_risk_sensitivity",
        "generated_at": datetime.now().astimezone().isoformat(),
        "selector": {
            "name": SELECTOR,
            "vix_slope_120": "> 0",
            "call_put": "PUT",
            "option_realized_volatility_20": (
                f"> {candidate.RV20_THRESHOLD}"
            ),
            "premium_bands": ["0.40~0.50", "1.00~2.50"],
            "selection": "09:30~12:59 첫 통과 신호, 재진입 없음",
        },
        "risk_levels": RISK_LEVELS,
        "strategies": strategies,
        "audit": {
            "passed": all(
                bool(strategy["audit"]["passed"])
                for strategy in strategies.values()
            ),
            "error_count": sum(
                int(strategy["audit"]["error_count"])
                for strategy in strategies.values()
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value["summary"]
                for key, value in strategies.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
