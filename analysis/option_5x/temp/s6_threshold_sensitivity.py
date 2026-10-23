from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = TEMP_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "s6_threshold_sensitivity_summary.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEMP_ROOT))

import minimal_baseline_backtest as base  # noqa: E402
import s6_candidate_backtest as candidate  # noqa: E402
import s6_ladder_backtest as ladder  # noqa: E402


DEVELOPMENT_END = "20260430"
THRESHOLDS = (0.09, 0.11, 0.13, 0.15, 0.17)
RISK_FRACTION = 0.05
PLAN_NAME = "HALF_2X_HALF_10X"


def select(
    events: pd.DataFrame,
    threshold: float,
) -> dict[str, pd.Series]:
    rows = events[
        (events["vix_slope_120"] > 0)
        & (events["is_call"] == 0)
        & (events["option_realized_volatility_20"] > threshold)
    ].sort_values(["expiry_date", "signal_minute", "option_code"])
    return {
        str(date_value): row
        for date_value, row in rows.groupby("expiry_date", sort=True).first().iterrows()
    }


def main() -> None:
    events = pd.read_csv(candidate.EVENT_PATH, encoding="utf-8-sig")
    events["expiry_date"] = events["expiry_date"].astype(str).str.zfill(8)
    selected = {
        threshold: select(events, threshold) for threshold in THRESHOLDS
    }
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if str(group["expiry_date"]) >= str(events["expiry_date"].min())
    ]
    files = base.target_file_map()
    accounts = {
        threshold: ladder.LadderAccount(
            key=f"S6-THRESHOLD-{threshold:.2f}",
            selector="ANY_VIXPUT_RV20",
            plan_name=PLAN_NAME,
            risk_fraction=RISK_FRACTION,
        )
        for threshold in THRESHOLDS
    }
    for index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        day_events = {
            threshold: rows.get(date_value)
            for threshold, rows in selected.items()
        }
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if any(event is not None for event in day_events.values())
            and date_value in files
            else {}
        )
        for threshold, account in accounts.items():
            start_cash = account.cash
            event = day_events[threshold]
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
    for threshold, account in accounts.items():
        assert account.daily is not None
        assert account.trades is not None
        development = [
            row for row in account.daily if str(row["date"]) <= DEVELOPMENT_END
        ]
        validation = [
            row for row in account.daily if str(row["date"]) > DEVELOPMENT_END
        ]
        strategies[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "summary": ladder.summarize(account.daily),
            "development": ladder.summarize(development),
            "validation": ladder.summarize(validation),
            "audit": candidate.audit(account),
        }
    result = {
        "status": "temporary_s6_threshold_sensitivity",
        "generated_at": datetime.now().astimezone().isoformat(),
        "thresholds": THRESHOLDS,
        "risk_fraction": RISK_FRACTION,
        "exit_plan": PLAN_NAME,
        "strategies": strategies,
        "audit": {
            "passed": all(
                bool(strategy["audit"]["passed"])
                for strategy in strategies.values()
            )
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(strategies, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
