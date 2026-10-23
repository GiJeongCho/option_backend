from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = TEMP_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "s6_vix_trend_gate_comparison.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEMP_ROOT))

import minimal_baseline_backtest as base  # noqa: E402
import s6_candidate_backtest as candidate  # noqa: E402
import s6_ladder_backtest as ladder  # noqa: E402


DEVELOPMENT_END = "20260430"
RV20_THRESHOLD = 0.13
RISK_FRACTION = 0.05
PLAN_NAME = "HALF_2X_HALF_10X"


def gate_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    above_ma5 = events["vix_ma5_distance"] > 0
    daily_up = events["vix_return_1"] > 0
    slope5_up = events["vix_slope_5"] > 0
    return {
        "OLD_SLOPE120": events["vix_slope_120"] > 0,
        "ABOVE_MA5": above_ma5,
        "ABOVE_MA5_DAILY_UP": above_ma5 & daily_up,
        "SLOPE5_DAILY_UP": slope5_up & daily_up,
        "STRICT_UPTREND": above_ma5 & slope5_up & daily_up,
    }


def selections(events: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    common = (
        (events["is_call"] == 0)
        & (
            events["option_realized_volatility_20"]
            > RV20_THRESHOLD
        )
    )
    result: dict[str, dict[str, pd.Series]] = {}
    for name, gate in gate_masks(events).items():
        rows = events[common & gate].sort_values(
            ["expiry_date", "signal_minute", "option_code"]
        )
        first = rows.groupby("expiry_date", sort=True).first()
        result[name] = {
            str(date_value): row
            for date_value, row in first.iterrows()
        }
    return result


def main() -> None:
    events = pd.read_csv(candidate.EVENT_PATH, encoding="utf-8-sig")
    events["expiry_date"] = events["expiry_date"].astype(str).str.zfill(8)
    selected = selections(events)
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if str(group["expiry_date"]) >= str(events["expiry_date"].min())
    ]
    files = base.target_file_map()
    accounts = {
        name: ladder.LadderAccount(
            key=f"S6-VIX-GATE-{name}",
            selector=name,
            plan_name=PLAN_NAME,
            risk_fraction=RISK_FRACTION,
        )
        for name in selected
    }
    for index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        day_events = {
            name: rows.get(date_value)
            for name, rows in selected.items()
        }
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if any(event is not None for event in day_events.values())
            and date_value in files
            else {}
        )
        for name, account in accounts.items():
            start_cash = account.cash
            event = day_events[name]
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
    for name, account in accounts.items():
        assert account.daily is not None
        development = [
            row for row in account.daily if str(row["date"]) <= DEVELOPMENT_END
        ]
        validation = [
            row for row in account.daily if str(row["date"]) > DEVELOPMENT_END
        ]
        strategies[name] = {
            "summary": ladder.summarize(account.daily),
            "development": ladder.summarize(development),
            "validation": ladder.summarize(validation),
            "audit": candidate.audit(account),
        }
    result = {
        "status": "temporary_vix_trend_gate_comparison",
        "generated_at": datetime.now().astimezone().isoformat(),
        "common_rules": {
            "put_only": True,
            "option_rv20_threshold": RV20_THRESHOLD,
            "risk_fraction": RISK_FRACTION,
            "exit_plan": PLAN_NAME,
            "features_use": "직전 VIX 관측일까지",
        },
        "gate_definitions": {
            "OLD_SLOPE120": "VIX 120일 정규화 기울기 > 0",
            "ABOVE_MA5": "VIX 종가 > VIX SMA5",
            "ABOVE_MA5_DAILY_UP": "종가 > SMA5, 전일 대비 상승",
            "SLOPE5_DAILY_UP": "VIX 5일 회귀기울기 > 0, 전일 대비 상승",
            "STRICT_UPTREND": (
                "종가 > SMA5, 5일 회귀기울기 > 0, 전일 대비 상승"
            ),
        },
        "strategies": strategies,
        "audit": {
            "passed": all(
                bool(item["audit"]["passed"])
                for item in strategies.values()
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
