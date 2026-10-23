from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
EVENT_PATH = OUTPUT_DIR / "s6_event_features.csv"
SUMMARY_PATH = OUTPUT_DIR / "s6_candidate_backtest_summary.json"
sys.path.insert(0, str(ROOT))

import minimal_baseline_backtest as base  # noqa: E402


INITIAL_CASH = 20_000_000
DAILY_BUDGET = 5_000_000
FORCE_EXIT_TIME = 1515
DEVELOPMENT_END = "20260430"
TARGETS = (2, 5, 10)
RV20_THRESHOLD = 0.13


@dataclass
class Account:
    key: str
    selector: str
    target: int
    risk_fraction: float | None
    cash: int = INITIAL_CASH
    daily: list[dict[str, object]] | None = None
    trades: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.daily = [] if self.daily is None else self.daily
        self.trades = [] if self.trades is None else self.trades


def selector_rows(events: pd.DataFrame, selector: str) -> pd.DataFrame:
    if selector.startswith("LOW"):
        rows = events[events["premium_band"] == "LOW_040_050"]
    elif selector.startswith("HIGH"):
        rows = events[events["premium_band"] == "HIGH_100_250"]
    else:
        rows = events
    if "VIXPUT" in selector:
        rows = rows[(rows["vix_slope_120"] > 0) & (rows["is_call"] == 0)]
    if "RV20" in selector:
        rows = rows[
            rows["option_realized_volatility_20"] > RV20_THRESHOLD
        ]
    return rows.sort_values(
        ["expiry_date", "signal_minute", "option_code"]
    )


def selected_events(events: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    result: dict[str, dict[str, pd.Series]] = {}
    for selector in (
        "LOW_BASE",
        "HIGH_BASE",
        "LOW_VIXPUT",
        "HIGH_VIXPUT",
        "LOW_VIXPUT_RV20",
        "ANY_VIXPUT_RV20",
    ):
        rows = selector_rows(events, selector)
        first = rows.groupby("expiry_date", sort=True).first()
        result[selector] = {
            str(date_value): row for date_value, row in first.iterrows()
        }
    return result


def sell_after_signal(
    date_value: str,
    contract: base.ContractGrid,
    signal_index: int,
    quantity: int,
    reason: str,
    strategy: str,
) -> tuple[int, int, int, list[dict[str, object]]]:
    remaining = quantity
    gross_sales = 0
    fees = 0
    capacity = 0.0
    trades: list[dict[str, object]] = []
    for bar in contract.bars[signal_index + 1 :]:
        if not bar.traded or bar.open is None:
            continue
        capacity += bar.volume * base.EXIT_PARTICIPATION
        fillable = min(remaining, math.floor(capacity + 1e-12))
        if fillable <= 0:
            continue
        capacity -= fillable
        fill_price = max(0.0, float(bar.open) - base.SLIPPAGE)
        principal = round(fillable * fill_price * base.MULTIPLIER)
        fee = fillable * base.COMMISSION_PER_CONTRACT
        gross_sales += principal
        fees += fee
        remaining -= fillable
        trades.append(
            {
                "strategy": strategy,
                "date": date_value,
                "campaign": 1,
                "minute": bar.minute,
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "SELL",
                "reason": reason,
                "stage": 1,
                "quantity": fillable,
                "price": round(fill_price, 4),
                "principal": principal,
                "fee": fee,
                "cash_flow": principal - fee,
                "bar_volume": bar.volume,
            }
        )
        if remaining <= 0:
            break
    return remaining, gross_sales, fees, trades


def simulate_position(
    account: Account,
    date_value: str,
    contract: base.ContractGrid,
    event: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    start_cash = account.cash
    fill_minute = int(event["fill_minute"])
    fill_index = base.MINUTE_INDEX[fill_minute]
    fill_bar = contract.bars[fill_index]
    fill_price = float(event["fill_price"])
    account_budget = DAILY_BUDGET
    if account.risk_fraction is not None:
        account_budget = min(
            account_budget,
            math.floor(start_cash * account.risk_fraction),
        )
    quantity_by_budget = math.floor(
        account_budget / (fill_price * base.MULTIPLIER)
    )
    quantity_by_cash = math.floor(
        start_cash
        / (
            fill_price * base.MULTIPLIER
            + 2 * base.COMMISSION_PER_CONTRACT
        )
    )
    quantity_by_volume = math.floor(
        fill_bar.volume * base.ENTRY_PARTICIPATION
    )
    quantity = min(
        quantity_by_budget,
        quantity_by_cash,
        quantity_by_volume,
    )
    if quantity <= 0:
        return (
            empty_day(
                account,
                date_value,
                start_cash,
                signal=True,
                skip_reason="현금 또는 거래량 부족",
            ),
            [],
        )
    principal = round(quantity * fill_price * base.MULTIPLIER)
    buy_fee = quantity * base.COMMISSION_PER_CONTRACT
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
    target_price = fill_price * account.target
    target_signal_index: int | None = None
    time_signal_index: int | None = None
    for index in range(fill_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if bar.minute > FORCE_EXIT_TIME:
            break
        if (
            target_signal_index is None
            and bar.traded
            and bar.high is not None
            and float(bar.high) + 1e-9 >= target_price
        ):
            target_signal_index = index
            break
        if bar.minute >= FORCE_EXIT_TIME:
            time_signal_index = index
            break
    if target_signal_index is not None:
        signal_index = target_signal_index
        reason = f"TARGET_{account.target}X_ALL"
        target_reached = True
    else:
        signal_index = (
            time_signal_index
            if time_signal_index is not None
            else base.MINUTE_INDEX[FORCE_EXIT_TIME]
        )
        reason = "TIME"
        target_reached = False
    remaining, gross_sales, sell_fee, sells = sell_after_signal(
        date_value,
        contract,
        signal_index,
        quantity,
        reason,
        account.key,
    )
    expiry = []
    if remaining > 0:
        expiry.append(
            {
                "strategy": account.key,
                "date": date_value,
                "campaign": 1,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "stage": 1,
                "quantity": remaining,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
    pnl = gross_sales - sell_fee - principal - buy_fee
    end_cash = start_cash + pnl
    if end_cash < 0:
        raise RuntimeError(f"{account.key} {date_value}: 음수 현금")
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
        "target_multiple": account.target,
        "risk_fraction": account.risk_fraction,
        "code": contract.code,
        "call_put": contract.call_put,
        "signal_minute": int(event["signal_minute"]),
        "entry_minute": fill_minute,
        "entry_price": round(fill_price, 4),
        "entry_quantity": quantity,
        "buy_principal": principal,
        "fees": buy_fee + sell_fee,
        "target_reached": target_reached,
        "exit_reason": reason if not remaining else "UNFILLED_ZERO_VALUE",
        "expired_quantity": remaining,
        "skip_reason": "",
    }
    return row, [buy, *sells, *expiry]


def empty_day(
    account: Account,
    date_value: str,
    start_cash: int,
    signal: bool = False,
    skip_reason: str = "",
) -> dict[str, object]:
    return {
        "strategy": account.key,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": start_cash,
        "pnl": 0,
        "trade_day": False,
        "signal_found": signal,
        "entry_filled": False,
        "selector": account.selector,
        "target_multiple": account.target,
        "risk_fraction": account.risk_fraction,
        "code": "",
        "call_put": "",
        "signal_minute": None,
        "entry_minute": None,
        "entry_price": 0.0,
        "entry_quantity": 0,
        "buy_principal": 0,
        "fees": 0,
        "target_reached": False,
        "exit_reason": "",
        "expired_quantity": 0,
        "skip_reason": skip_reason,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    traded = [row for row in rows if bool(row["trade_day"])]
    pnls = [int(row["pnl"]) for row in traded]
    segment_start_cash = (
        int(rows[0]["start_cash"]) if rows else INITIAL_CASH
    )
    equity = [segment_start_cash] + [int(row["end_cash"]) for row in rows]
    running_peak = equity[0]
    drawdowns: list[int] = []
    for value in equity:
        running_peak = max(running_peak, value)
        drawdowns.append(value - running_peak)
    profits = sorted((value for value in pnls if value > 0), reverse=True)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    return {
        "days": len(rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in pnls),
        "losing_days": sum(value < 0 for value in pnls),
        "break_even_days": sum(value == 0 for value in pnls),
        "win_rate_pct": (
            round(sum(value > 0 for value in pnls) / len(traded) * 100, 2)
            if traded
            else 0.0
        ),
        "total_pnl": sum(pnls),
        "ending_cash": int(rows[-1]["end_cash"]) if rows else INITIAL_CASH,
        "max_drawdown": min(drawdowns, default=0),
        "profit_factor": (
            round(gross_profit / gross_loss, 4)
            if gross_loss
            else None
        ),
        "target_days": sum(bool(row["target_reached"]) for row in traded),
        "expired_quantity": sum(int(row["expired_quantity"]) for row in traded),
        "average_buy_principal": (
            round(
                sum(int(row["buy_principal"]) for row in traded) / len(traded)
            )
            if traded
            else 0
        ),
        "pnl_excluding_best_1_day": sum(pnls) - sum(profits[:1]),
        "pnl_excluding_best_3_days": sum(pnls) - sum(profits[:3]),
        "pnl_excluding_best_5_days": sum(pnls) - sum(profits[:5]),
        "cash_or_volume_skips": sum(
            bool(row["signal_found"]) and not bool(row["entry_filled"])
            for row in rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        fieldnames = list(
            dict.fromkeys(
                field
                for row in rows
                for field in row
            )
        )
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(account: Account) -> dict[str, object]:
    errors: list[str] = []
    previous_end = INITIAL_CASH
    trades_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in account.trades or []:
        trades_by_date[str(trade["date"])].append(trade)
    for row in account.daily or []:
        date_value = str(row["date"])
        if int(row["start_cash"]) != previous_end:
            errors.append(f"{date_value}: 현금 연속성")
        if int(row["end_cash"]) < 0:
            errors.append(f"{date_value}: 음수 현금")
        day_trades = trades_by_date.get(date_value, [])
        buys = [trade for trade in day_trades if trade["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{date_value}: 하루 다중 매수")
        if sum(int(trade["principal"]) for trade in buys) > DAILY_BUDGET:
            errors.append(f"{date_value}: 원금 한도")
        quantity = 0
        for trade in day_trades:
            side = str(trade["side"])
            amount = int(trade["quantity"])
            quantity += amount if side == "BUY" else -amount
        if quantity != 0:
            errors.append(f"{date_value}: 미폐쇄 수량 {quantity}")
        previous_end = int(row["end_cash"])
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:50],
    }


def main() -> None:
    events = pd.read_csv(EVENT_PATH, encoding="utf-8-sig")
    events["expiry_date"] = events["expiry_date"].astype(str).str.zfill(8)
    selected = selected_events(events)
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if str(group["expiry_date"]) >= str(events["expiry_date"].min())
    ]
    files = base.target_file_map()
    risk_modes = {
        "FULL500": None,
        "RISK10": 0.10,
    }
    accounts = {
        f"S6-CANDIDATE-{risk_mode}-{selector}-{target}X": Account(
            key=f"S6-CANDIDATE-{risk_mode}-{selector}-{target}X",
            selector=selector,
            target=target,
            risk_fraction=risk_fraction,
        )
        for selector in selected
        for target in TARGETS
        for risk_mode, risk_fraction in risk_modes.items()
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
                row = empty_day(account, date_value, start_cash)
                trades: list[dict[str, object]] = []
            else:
                code = str(event["option_code"])
                contract = contracts.get(code)
                if contract is None:
                    row = empty_day(
                        account,
                        date_value,
                        start_cash,
                        signal=True,
                        skip_reason="계약 원자료 없음",
                    )
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

    summaries: dict[str, object] = {}
    audits: dict[str, object] = {}
    for key, account in accounts.items():
        assert account.daily is not None
        assert account.trades is not None
        development = [
            row for row in account.daily if str(row["date"]) <= DEVELOPMENT_END
        ]
        validation = [
            row for row in account.daily if str(row["date"]) > DEVELOPMENT_END
        ]
        audit_result = audit(account)
        summaries[key] = {
            "selector": account.selector,
            "target_multiple": account.target,
            "risk_fraction": account.risk_fraction,
            "summary": summarize(account.daily),
            "development": summarize(development),
            "validation": summarize(validation),
            "audit": audit_result,
        }
        audits[key] = audit_result
        stem = key.lower().replace("-", "_")
        write_csv(OUTPUT_DIR / f"{stem}_daily.csv", account.daily)
        write_csv(OUTPUT_DIR / f"{stem}_trades.csv", account.trades)

    result = {
        "status": "temporary_s6_candidate_backtest",
        "generated_at": datetime.now().astimezone().isoformat(),
        "rules": {
            "initial_cash": INITIAL_CASH,
            "daily_budget": DAILY_BUDGET,
            "risk_modes": {
                "FULL500": "매일 최대 500만원",
                "RISK10": "당일 시작자금의 10%, 단 최대 500만원",
            },
            "entry": (
                "09:30~12:59 첫 완료봉 종가와 다음 5분 내 실거래봉 시가가 "
                "각 selector 가격대에 속할 때 시가+0.01 매수"
            ),
            "vix_put_filter": (
                "VIX 직전일까지 120일 정규화 기울기 > 0, PUT만"
            ),
            "exits": [2, 5, 10],
            "time_exit": FORCE_EXIT_TIME,
            "reentry": False,
            "execution": (
                "매수·매도 거래량 10%, 수수료 계약당 편도 500원, "
                "슬리피지 ±0.01, 미체결 잔량 0원"
            ),
        },
        "limitations": [
            "후반 결과를 확인한 뒤 VIX상승+PUT 단순 규칙을 진단했으므로 독립 검증이 아니다.",
            "현물 1분봉이 2024-09-24부터라 그 이전 만기일은 포함하지 않는다.",
            "목표 고가 확인 다음 실제 거래봉에서 매도해 고가 터치 가격 체결을 가정하지 않는다.",
        ],
        "strategies": summaries,
        "audit": {
            "passed": all(bool(item["passed"]) for item in audits.values()),
            "error_count": sum(int(item["error_count"]) for item in audits.values()),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    compact = {
        key: value["summary"]
        for key, value in summaries.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
