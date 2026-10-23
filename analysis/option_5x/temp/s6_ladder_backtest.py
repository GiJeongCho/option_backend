from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = TEMP_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "s6_ladder_backtest_summary.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEMP_ROOT))

import minimal_baseline_backtest as base  # noqa: E402
import s6_candidate_backtest as candidate  # noqa: E402


INITIAL_CASH = 20_000_000
MAX_DAILY_BUDGET = 5_000_000
RISK_FRACTION = 0.10
FORCE_EXIT_TIME = 1515
DEVELOPMENT_END = "20260430"

EXIT_PLANS: dict[str, tuple[tuple[int, float], ...]] = {
    "FULL_2X": ((2, 1.0),),
    "HALF_2X_HALF_5X": ((2, 0.50), (5, 0.50)),
    "HALF_2X_HALF_10X": ((2, 0.50), (10, 0.50)),
    "LADDER_2X_5X_10X": ((2, 0.50), (5, 0.25), (10, 0.25)),
}


@dataclass
class LadderAccount:
    key: str
    selector: str
    plan_name: str
    risk_fraction: float = RISK_FRACTION
    target: int = 0
    cash: int = INITIAL_CASH
    daily: list[dict[str, object]] | None = None
    trades: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.daily = [] if self.daily is None else self.daily
        self.trades = [] if self.trades is None else self.trades


@dataclass
class Tranche:
    stage: int
    target: int
    quantity: int
    remaining: int
    triggered: bool = False
    eligible_index: int | None = None
    reason: str = ""


def split_quantity(
    quantity: int,
    plan: tuple[tuple[int, float], ...],
) -> list[Tranche]:
    allocated = 0
    result: list[Tranche] = []
    for index, (target, weight) in enumerate(plan, start=1):
        if index == len(plan):
            amount = quantity - allocated
        else:
            amount = min(
                quantity - allocated,
                math.ceil(quantity * weight),
            )
            allocated += amount
        if amount <= 0:
            continue
        result.append(
            Tranche(
                stage=index,
                target=target,
                quantity=amount,
                remaining=amount,
            )
        )
    return result


def exit_positions(
    account: LadderAccount,
    date_value: str,
    contract: base.ContractGrid,
    fill_index: int,
    fill_price: float,
    tranches: list[Tranche],
) -> tuple[int, int, list[dict[str, object]], dict[int, bool]]:
    gross_sales = 0
    fees = 0
    trades: list[dict[str, object]] = []
    target_reached = {tranche.target: False for tranche in tranches}
    capacity = 0.0
    for index in range(fill_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        pending = [
            tranche
            for tranche in tranches
            if tranche.remaining > 0
            and tranche.eligible_index is not None
            and tranche.eligible_index <= index
        ]
        if pending and bar.traded and bar.open is not None:
            capacity += bar.volume * base.EXIT_PARTICIPATION
            for tranche in sorted(pending, key=lambda item: item.stage):
                fillable = min(
                    tranche.remaining,
                    math.floor(capacity + 1e-12),
                )
                if fillable <= 0:
                    continue
                capacity -= fillable
                price = max(0.0, float(bar.open) - base.SLIPPAGE)
                principal = round(fillable * price * base.MULTIPLIER)
                fee = fillable * base.COMMISSION_PER_CONTRACT
                gross_sales += principal
                fees += fee
                tranche.remaining -= fillable
                trades.append(
                    {
                        "strategy": account.key,
                        "date": date_value,
                        "campaign": 1,
                        "minute": bar.minute,
                        "code": contract.code,
                        "call_put": contract.call_put,
                        "side": "SELL",
                        "reason": tranche.reason,
                        "stage": tranche.stage,
                        "quantity": fillable,
                        "price": round(price, 4),
                        "principal": principal,
                        "fee": fee,
                        "cash_flow": principal - fee,
                        "bar_volume": bar.volume,
                    }
                )
        still_pending = any(
            tranche.remaining > 0
            and tranche.eligible_index is not None
            and tranche.eligible_index <= index
            for tranche in tranches
        )
        if not still_pending:
            capacity = 0.0
        if bar.minute <= FORCE_EXIT_TIME and bar.traded and bar.high is not None:
            for tranche in tranches:
                if (
                    tranche.remaining > 0
                    and not tranche.triggered
                    and float(bar.high) + 1e-9
                    >= fill_price * tranche.target
                ):
                    tranche.triggered = True
                    tranche.eligible_index = index + 1
                    tranche.reason = f"TARGET_{tranche.target}X"
                    target_reached[tranche.target] = True
        if bar.minute == FORCE_EXIT_TIME:
            for tranche in tranches:
                if tranche.remaining > 0 and not tranche.triggered:
                    tranche.triggered = True
                    tranche.eligible_index = index + 1
                    tranche.reason = "TIME"
        if (
            bar.minute > FORCE_EXIT_TIME
            and not any(tranche.remaining > 0 for tranche in tranches)
        ):
            break
    for tranche in tranches:
        if tranche.remaining <= 0:
            continue
        trades.append(
            {
                "strategy": account.key,
                "date": date_value,
                "campaign": 1,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "stage": tranche.stage,
                "quantity": tranche.remaining,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
    return gross_sales, fees, trades, target_reached


def simulate_position(
    account: LadderAccount,
    date_value: str,
    contract: base.ContractGrid,
    event: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    start_cash = account.cash
    fill_minute = int(event["fill_minute"])
    fill_index = base.MINUTE_INDEX[fill_minute]
    fill_bar = contract.bars[fill_index]
    fill_price = float(event["fill_price"])
    budget = min(
        MAX_DAILY_BUDGET,
        math.floor(start_cash * account.risk_fraction),
    )
    quantity = min(
        math.floor(budget / (fill_price * base.MULTIPLIER)),
        math.floor(
            start_cash
            / (
                fill_price * base.MULTIPLIER
                + 2 * base.COMMISSION_PER_CONTRACT
            )
        ),
        math.floor(fill_bar.volume * base.ENTRY_PARTICIPATION),
    )
    if quantity <= 0:
        row = candidate.empty_day(
            account,
            date_value,
            start_cash,
            signal=True,
            skip_reason="현금 또는 거래량 부족",
        )
        row["exit_plan"] = account.plan_name
        return row, []
    principal = round(quantity * fill_price * base.MULTIPLIER)
    buy_fee = quantity * base.COMMISSION_PER_CONTRACT
    plan = EXIT_PLANS[account.plan_name]
    tranches = split_quantity(quantity, plan)
    buy = {
        "strategy": account.key,
        "date": date_value,
        "campaign": 1,
        "minute": fill_minute,
        "code": contract.code,
        "call_put": contract.call_put,
        "side": "BUY",
        "reason": f"{account.selector}_ENTRY",
        "stage": 1,
        "quantity": quantity,
        "price": round(fill_price, 4),
        "principal": principal,
        "fee": buy_fee,
        "cash_flow": -(principal + buy_fee),
        "bar_volume": fill_bar.volume,
    }
    gross_sales, sell_fees, sells, targets = exit_positions(
        account,
        date_value,
        contract,
        fill_index,
        fill_price,
        tranches,
    )
    pnl = gross_sales - sell_fees - principal - buy_fee
    end_cash = start_cash + pnl
    if end_cash < 0:
        raise RuntimeError(f"{account.key} {date_value}: 음수 현금")
    expired_quantity = sum(tranche.remaining for tranche in tranches)
    row = {
        "strategy": account.key,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": True,
        "signal_found": True,
        "entry_filled": True,
        "selector": account.selector,
        "target_multiple": 0,
        "risk_fraction": account.risk_fraction,
        "exit_plan": account.plan_name,
        "code": contract.code,
        "call_put": contract.call_put,
        "signal_minute": int(event["signal_minute"]),
        "entry_minute": fill_minute,
        "entry_price": round(fill_price, 4),
        "entry_quantity": quantity,
        "buy_principal": principal,
        "fees": buy_fee + sell_fees,
        "target_reached": bool(targets.get(2)),
        "reached_2x": bool(targets.get(2)),
        "reached_5x": bool(targets.get(5)),
        "reached_10x": bool(targets.get(10)),
        "exit_reason": account.plan_name,
        "expired_quantity": expired_quantity,
        "skip_reason": "",
    }
    return row, [buy, *sells]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result = candidate.summarize(rows)
    traded = [row for row in rows if bool(row["trade_day"])]
    principal = sum(int(row["buy_principal"]) for row in traded)
    result.update(
        {
            "buy_principal_total": principal,
            "pnl_on_principal_pct": (
                round(int(result["total_pnl"]) / principal * 100, 4)
                if principal
                else 0.0
            ),
            "reached_2x_days": sum(
                bool(row.get("reached_2x")) for row in traded
            ),
            "reached_5x_days": sum(
                bool(row.get("reached_5x")) for row in traded
            ),
            "reached_10x_days": sum(
                bool(row.get("reached_10x")) for row in traded
            ),
        }
    )
    return result


def main() -> None:
    events = pd.read_csv(candidate.EVENT_PATH, encoding="utf-8-sig")
    events["expiry_date"] = events["expiry_date"].astype(str).str.zfill(8)
    all_selected = candidate.selected_events(events)
    selectors = (
        "LOW_BASE",
        "HIGH_BASE",
        "LOW_VIXPUT",
        "HIGH_VIXPUT",
        "LOW_VIXPUT_RV20",
        "ANY_VIXPUT_RV20",
    )
    selected = {name: all_selected[name] for name in selectors}
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if str(group["expiry_date"]) >= str(events["expiry_date"].min())
    ]
    files = base.target_file_map()
    accounts = {
        f"S6-LADDER-{selector}-{plan_name}": LadderAccount(
            key=f"S6-LADDER-{selector}-{plan_name}",
            selector=selector,
            plan_name=plan_name,
        )
        for selector in selectors
        for plan_name in EXIT_PLANS
    }
    for index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        day_events = {
            selector: rows.get(date_value)
            for selector, rows in selected.items()
        }
        needs_data = any(event is not None for event in day_events.values())
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if needs_data and date_value in files
            else {}
        )
        for account in accounts.values():
            start_cash = account.cash
            event = day_events[account.selector]
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
                    row, trades = simulate_position(
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
            "selector": account.selector,
            "exit_plan": account.plan_name,
            "summary": summarize(account.daily),
            "development": summarize(development),
            "validation": summarize(validation),
            "audit": audit,
        }
        stem = key.lower().replace("-", "_")
        candidate.write_csv(OUTPUT_DIR / f"{stem}_daily.csv", account.daily)
        candidate.write_csv(OUTPUT_DIR / f"{stem}_trades.csv", account.trades)

    result = {
        "status": "temporary_s6_ladder_backtest",
        "generated_at": datetime.now().astimezone().isoformat(),
        "rules": {
            "initial_cash": INITIAL_CASH,
            "risk": "당일 시작자금 10%, 최대 500만원",
            "exit_plans": EXIT_PLANS,
            "execution": (
                "매수·매도 거래량 10%, 수수료 계약당 편도 500원, "
                "슬리피지 ±0.01, 목표 확인 다음 실제 거래봉 체결"
            ),
            "time_exit": FORCE_EXIT_TIME,
            "reentry": False,
        },
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
