from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import minimal_baseline_backtest as base
import strategy1_backtest as strategy1
from inventory import ANALYSIS_START_DATE


STRATEGY_VERSION = "S6-v0.1"
OUTPUT_STEM = "strategy6_v01"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

INITIAL_CASH = 20_000_000
MAX_DAILY_PRINCIPAL = 5_000_000
RISK_FRACTION = 0.05
COMMON_START = ANALYSIS_START_DATE
DEVELOPMENT_END = "20260430"
ENTRY_START = 930
ENTRY_CUTOFF = 1300
MAX_ENTRY_WAIT = 5
FORCE_EXIT_TIME = 1515
RV20_THRESHOLD = 0.13
PREMIUM_BANDS = (
    base.PremiumBand("LOW_040_050", 0.40, 0.50),
    base.PremiumBand("HIGH_100_250", 1.00, 2.50),
)


@dataclass(frozen=True)
class EntryEvent:
    code: str
    band: str
    signal_index: int
    fill_index: int
    signal_minute: int
    fill_minute: int
    signal_close: float
    raw_fill_open: float
    fill_price: float
    option_rv20: float
    vix_slope_120: float


@dataclass
class Tranche:
    stage: int
    target_multiple: int
    quantity: int
    remaining: int
    triggered: bool = False
    eligible_index: int | None = None
    reason: str = ""


@dataclass
class Account:
    cash: int = INITIAL_CASH
    daily: list[dict[str, object]] | None = None
    trades: list[dict[str, object]] | None = None
    positions: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.daily = [] if self.daily is None else self.daily
        self.trades = [] if self.trades is None else self.trades
        self.positions = [] if self.positions is None else self.positions


def normalized_slope(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3 or abs(array[-1]) <= 1e-12:
        return math.nan
    x = np.arange(len(array), dtype=float)
    return float(np.polyfit(x, array, 1)[0] / array[-1])


def realized_volatility(values: list[float], period: int = 20) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < period + 1 or np.any(array[-(period + 1) :] <= 0):
        return math.nan
    returns = np.diff(np.log(array[-(period + 1) :]))
    return float(np.std(returns, ddof=1))


def vix_slope_by_date() -> dict[str, float]:
    dates, closes = strategy1.volatility_closes()
    ordered = sorted(zip(dates, closes), key=lambda item: item[0])
    result: dict[str, float] = {}
    history: list[float] = []
    previous_date = ""
    for date_value, close in ordered:
        if date_value == previous_date:
            history[-1] = float(close)
        else:
            history.append(float(close))
            previous_date = date_value
        result[date_value] = normalized_slope(history[-120:])
    return result


def previous_vix_slope(
    date_value: str,
    dates: list[str],
    slopes: dict[str, float],
) -> float:
    previous = next(
        (item for item in reversed(dates) if item < date_value),
        "",
    )
    return slopes.get(previous, math.nan)


def is_put(contract: base.ContractGrid) -> bool:
    return "PUT" in contract.call_put.upper() or "풋" in contract.call_put


def first_band_event(
    contract: base.ContractGrid,
    band: base.PremiumBand,
    vix_slope: float,
) -> EntryEvent | None:
    start_index = base.MINUTE_INDEX[ENTRY_START]
    for signal_index in range(start_index, len(contract.bars)):
        signal_bar = contract.bars[signal_index]
        if signal_bar.minute >= ENTRY_CUTOFF:
            break
        if (
            not signal_bar.traded
            or signal_bar.close is None
            or not band.contains(float(signal_bar.close))
        ):
            continue
        fill = base.next_traded_bar(
            contract,
            signal_index,
            MAX_ENTRY_WAIT,
        )
        if fill is None:
            continue
        fill_index, fill_bar = fill
        if (
            fill_bar.minute >= ENTRY_CUTOFF
            or fill_bar.open is None
            or not band.contains(float(fill_bar.open))
        ):
            continue
        closes = [
            float(bar.close)
            for bar in contract.bars[: signal_index + 1]
            if bar.close is not None
        ]
        return EntryEvent(
            code=contract.code,
            band=band.name,
            signal_index=signal_index,
            fill_index=fill_index,
            signal_minute=signal_bar.minute,
            fill_minute=fill_bar.minute,
            signal_close=float(signal_bar.close),
            raw_fill_open=float(fill_bar.open),
            fill_price=round(float(fill_bar.open) + base.SLIPPAGE, 4),
            option_rv20=realized_volatility(closes),
            vix_slope_120=vix_slope,
        )
    return None


def select_event(
    contracts: dict[str, base.ContractGrid],
    vix_slope: float,
) -> EntryEvent | None:
    if not math.isfinite(vix_slope) or vix_slope <= 0:
        return None
    candidates: list[EntryEvent] = []
    for contract in contracts.values():
        if not is_put(contract):
            continue
        for band in PREMIUM_BANDS:
            event = first_band_event(contract, band, vix_slope)
            if (
                event is not None
                and math.isfinite(event.option_rv20)
                and event.option_rv20 > RV20_THRESHOLD
            ):
                candidates.append(event)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda event: (
            event.signal_minute,
            event.code,
            event.band,
        ),
    )


def empty_day(
    date_value: str,
    start_cash: int,
    signal_found: bool = False,
    event: EntryEvent | None = None,
    skip_reason: str = "",
) -> dict[str, object]:
    return {
        "strategy": STRATEGY_VERSION,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": start_cash,
        "pnl": 0,
        "trade_day": False,
        "signal_found": signal_found,
        "entry_filled": False,
        "code": event.code if event else "",
        "call_put": "PUT" if event else "",
        "premium_band": event.band if event else "",
        "signal_minute": event.signal_minute if event else None,
        "entry_minute": None,
        "signal_close": event.signal_close if event else 0.0,
        "raw_entry_open": event.raw_fill_open if event else 0.0,
        "entry_price": 0.0,
        "entry_quantity": 0,
        "risk_budget": 0,
        "buy_principal": 0,
        "gross_sales": 0,
        "fees": 0,
        "expired_quantity": 0,
        "vix_slope_120": event.vix_slope_120 if event else None,
        "option_realized_volatility_20": (
            event.option_rv20 if event else None
        ),
        "reached_2x": False,
        "reached_10x": False,
        "exit_reasons": "",
        "skip_reason": skip_reason,
    }


def tranches(quantity: int) -> list[Tranche]:
    two_x_quantity = math.ceil(quantity / 2)
    result = [
        Tranche(
            stage=1,
            target_multiple=2,
            quantity=two_x_quantity,
            remaining=two_x_quantity,
        )
    ]
    runner_quantity = quantity - two_x_quantity
    if runner_quantity > 0:
        result.append(
            Tranche(
                stage=2,
                target_multiple=10,
                quantity=runner_quantity,
                remaining=runner_quantity,
            )
        )
    return result


def execute_exits(
    date_value: str,
    contract: base.ContractGrid,
    event: EntryEvent,
    positions: list[Tranche],
) -> tuple[int, int, list[dict[str, object]], dict[int, bool]]:
    gross_sales = 0
    fees = 0
    trades: list[dict[str, object]] = []
    reached = {2: False, 10: False}
    capacity = 0.0
    for index in range(event.fill_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        pending = [
            position
            for position in positions
            if position.remaining > 0
            and position.eligible_index is not None
            and position.eligible_index <= index
        ]
        if pending and bar.traded and bar.open is not None:
            capacity += bar.volume * base.EXIT_PARTICIPATION
            for position in sorted(
                pending,
                key=lambda item: item.stage,
            ):
                fillable = min(
                    position.remaining,
                    math.floor(capacity + 1e-12),
                )
                if fillable <= 0:
                    continue
                capacity -= fillable
                price = max(0.0, float(bar.open) - base.SLIPPAGE)
                principal = round(
                    fillable * price * base.MULTIPLIER
                )
                fee = fillable * base.COMMISSION_PER_CONTRACT
                gross_sales += principal
                fees += fee
                position.remaining -= fillable
                trades.append(
                    {
                        "strategy": STRATEGY_VERSION,
                        "date": date_value,
                        "campaign": 1,
                        "minute": bar.minute,
                        "code": contract.code,
                        "call_put": contract.call_put,
                        "side": "SELL",
                        "reason": position.reason,
                        "stage": position.stage,
                        "quantity": fillable,
                        "price": round(price, 4),
                        "principal": principal,
                        "fee": fee,
                        "cash_flow": principal - fee,
                        "bar_volume": bar.volume,
                    }
                )
        still_pending = any(
            position.remaining > 0
            and position.eligible_index is not None
            and position.eligible_index <= index
            for position in positions
        )
        if not still_pending:
            capacity = 0.0

        if (
            bar.minute <= FORCE_EXIT_TIME
            and bar.traded
            and bar.high is not None
        ):
            for position in positions:
                if (
                    position.remaining > 0
                    and not position.triggered
                    and float(bar.high) + 1e-9
                    >= event.fill_price * position.target_multiple
                ):
                    position.triggered = True
                    position.eligible_index = index + 1
                    position.reason = (
                        f"TARGET_{position.target_multiple}X"
                    )
                    reached[position.target_multiple] = True
        if bar.minute == FORCE_EXIT_TIME:
            for position in positions:
                if position.remaining > 0 and not position.triggered:
                    position.triggered = True
                    position.eligible_index = index + 1
                    position.reason = "TIME"
        if (
            bar.minute > FORCE_EXIT_TIME
            and not any(
                position.remaining > 0 for position in positions
            )
        ):
            break

    for position in positions:
        if position.remaining <= 0:
            continue
        trades.append(
            {
                "strategy": STRATEGY_VERSION,
                "date": date_value,
                "campaign": 1,
                "minute": base.SESSION_MINUTES[-1],
                "code": contract.code,
                "call_put": contract.call_put,
                "side": "EXPIRE",
                "reason": "UNFILLED_ZERO_VALUE",
                "stage": position.stage,
                "quantity": position.remaining,
                "price": 0.0,
                "principal": 0,
                "fee": 0,
                "cash_flow": 0,
                "bar_volume": 0,
            }
        )
    return gross_sales, fees, trades, reached


def simulate_day(
    date_value: str,
    account: Account,
    contracts: dict[str, base.ContractGrid],
    event: EntryEvent | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    start_cash = account.cash
    if event is None:
        return empty_day(date_value, start_cash), [], None
    contract = contracts[event.code]
    fill_bar = contract.bars[event.fill_index]
    risk_budget = min(
        MAX_DAILY_PRINCIPAL,
        math.floor(start_cash * RISK_FRACTION),
    )
    quantity = min(
        math.floor(
            risk_budget / (event.fill_price * base.MULTIPLIER)
        ),
        math.floor(
            start_cash
            / (
                event.fill_price * base.MULTIPLIER
                + 2 * base.COMMISSION_PER_CONTRACT
            )
        ),
        math.floor(fill_bar.volume * base.ENTRY_PARTICIPATION),
    )
    if quantity <= 0:
        return (
            empty_day(
                date_value,
                start_cash,
                signal_found=True,
                event=event,
                skip_reason="현금 또는 진입봉 거래량 부족",
            ),
            [],
            None,
        )

    buy_principal = round(
        quantity * event.fill_price * base.MULTIPLIER
    )
    buy_fee = quantity * base.COMMISSION_PER_CONTRACT
    buy_trade = {
        "strategy": STRATEGY_VERSION,
        "date": date_value,
        "campaign": 1,
        "minute": event.fill_minute,
        "code": contract.code,
        "call_put": contract.call_put,
        "side": "BUY",
        "reason": "VIX_RV_PUT_ENTRY",
        "stage": 1,
        "quantity": quantity,
        "price": round(event.fill_price, 4),
        "principal": buy_principal,
        "fee": buy_fee,
        "cash_flow": -(buy_principal + buy_fee),
        "bar_volume": fill_bar.volume,
    }
    positions = tranches(quantity)
    gross_sales, sell_fees, sell_trades, reached = execute_exits(
        date_value,
        contract,
        event,
        positions,
    )
    expired_quantity = sum(
        position.remaining for position in positions
    )
    pnl = gross_sales - sell_fees - buy_principal - buy_fee
    end_cash = start_cash + pnl
    if end_cash < 0:
        raise RuntimeError(f"{date_value}: 현금이 음수가 됐습니다.")
    exit_reasons = list(
        dict.fromkeys(
            str(trade["reason"])
            for trade in sell_trades
            if trade["side"] in {"SELL", "EXPIRE"}
        )
    )
    daily = {
        "strategy": STRATEGY_VERSION,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": True,
        "signal_found": True,
        "entry_filled": True,
        "code": event.code,
        "call_put": contract.call_put,
        "premium_band": event.band,
        "signal_minute": event.signal_minute,
        "entry_minute": event.fill_minute,
        "signal_close": round(event.signal_close, 4),
        "raw_entry_open": round(event.raw_fill_open, 4),
        "entry_price": round(event.fill_price, 4),
        "entry_quantity": quantity,
        "risk_budget": risk_budget,
        "buy_principal": buy_principal,
        "gross_sales": gross_sales,
        "fees": buy_fee + sell_fees,
        "expired_quantity": expired_quantity,
        "vix_slope_120": event.vix_slope_120,
        "option_realized_volatility_20": event.option_rv20,
        "reached_2x": reached[2],
        "reached_10x": reached[10],
        "exit_reasons": ",".join(exit_reasons),
        "skip_reason": "",
    }
    position = {
        "strategy": STRATEGY_VERSION,
        "date": date_value,
        "campaign": 1,
        "code": event.code,
        "call_put": contract.call_put,
        "premium_band": event.band,
        "signal_minute": event.signal_minute,
        "entry_minute": event.fill_minute,
        "entry_price": round(event.fill_price, 4),
        "entry_quantity": quantity,
        "target_2x_quantity": math.ceil(quantity / 2),
        "target_10x_quantity": quantity - math.ceil(quantity / 2),
        "vix_slope_120": event.vix_slope_120,
        "option_realized_volatility_20": event.option_rv20,
    }
    return daily, [buy_trade, *sell_trades], position


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    traded = [row for row in rows if bool(row["trade_day"])]
    pnls = [int(row["pnl"]) for row in traded]
    start = int(rows[0]["start_cash"]) if rows else INITIAL_CASH
    equity = [start, *[int(row["end_cash"]) for row in rows]]
    peak = equity[0]
    maximum_drawdown = 0
    maximum_drawdown_pct = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = value - peak
        maximum_drawdown = min(maximum_drawdown, drawdown)
        if peak:
            maximum_drawdown_pct = min(
                maximum_drawdown_pct,
                drawdown / peak * 100,
            )
    profits = sorted(
        (value for value in pnls if value > 0),
        reverse=True,
    )
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    return {
        "period_start": rows[0]["date"] if rows else None,
        "period_end": rows[-1]["date"] if rows else None,
        "days": len(rows),
        "signal_days": sum(bool(row["signal_found"]) for row in rows),
        "trade_days": len(traded),
        "profitable_days": sum(value > 0 for value in pnls),
        "losing_days": sum(value < 0 for value in pnls),
        "break_even_days": sum(value == 0 for value in pnls),
        "win_rate_pct": (
            round(sum(value > 0 for value in pnls) / len(traded) * 100, 2)
            if traded
            else 0.0
        ),
        "initial_cash": start,
        "ending_cash": int(rows[-1]["end_cash"]) if rows else start,
        "total_pnl": sum(pnls),
        "total_return_pct": (
            round(sum(pnls) / start * 100, 4) if start else 0.0
        ),
        "profit_factor": (
            round(gross_profit / gross_loss, 4)
            if gross_loss
            else None
        ),
        "max_drawdown": maximum_drawdown,
        "max_drawdown_pct": round(maximum_drawdown_pct, 4),
        "reached_2x_days": sum(
            bool(row["reached_2x"]) for row in traded
        ),
        "reached_10x_days": sum(
            bool(row["reached_10x"]) for row in traded
        ),
        "expired_quantity": sum(
            int(row["expired_quantity"]) for row in traded
        ),
        "pnl_excluding_best_1_day": sum(pnls) - sum(profits[:1]),
        "pnl_excluding_best_3_days": sum(pnls) - sum(profits[:3]),
        "pnl_excluding_best_5_days": sum(pnls) - sum(profits[:5]),
    }


def internal_audit(
    daily: list[dict[str, object]],
    trades: list[dict[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    trades_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[str(trade["date"])].append(trade)
    previous_end = INITIAL_CASH
    for row in daily:
        date_value = str(row["date"])
        if int(row["start_cash"]) != previous_end:
            errors.append(f"{date_value}: 현금 연속성")
        if int(row["end_cash"]) < 0:
            errors.append(f"{date_value}: 음수 현금")
        day_trades = trades_by_date.get(date_value, [])
        cash_flow = sum(int(trade["cash_flow"]) for trade in day_trades)
        if cash_flow != int(row["pnl"]):
            errors.append(f"{date_value}: 거래장부 손익")
        buys = [trade for trade in day_trades if trade["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{date_value}: 다중 진입")
        principal = sum(int(trade["principal"]) for trade in buys)
        if principal > int(row["risk_budget"]):
            errors.append(f"{date_value}: 위험예산 초과")
        quantity = 0
        for trade in day_trades:
            amount = int(trade["quantity"])
            quantity += amount if trade["side"] == "BUY" else -amount
        if quantity != 0:
            errors.append(f"{date_value}: 미폐쇄 수량 {quantity}")
        previous_end = int(row["end_cash"])
    return {
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:60],
        "checks": [
            "현금 연속성·비음수",
            "일별 거래장부 손익 일치",
            "하루 한 계약·한 번만 진입",
            "당일 시작자금 5%·500만원 상한",
            "BUY·SELL·EXPIRE 수량 보존",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(
        dict.fromkeys(field for row in rows for field in row)
    )
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= COMMON_START
    ]
    files = base.target_file_map()
    slopes = vix_slope_by_date()
    vix_dates = sorted(slopes)
    account = Account()
    assert account.daily is not None
    assert account.trades is not None
    assert account.positions is not None

    for index, group in enumerate(groups, start=1):
        date_value = group["expiry_date"]
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if date_value in files
            else {}
        )
        slope = previous_vix_slope(date_value, vix_dates, slopes)
        event = select_event(contracts, slope)
        daily, trades, position = simulate_day(
            date_value,
            account,
            contracts,
            event,
        )
        account.cash = int(daily["end_cash"])
        account.daily.append(daily)
        account.trades.extend(trades)
        if position is not None:
            account.positions.append(position)
        if index % 20 == 0 or index == len(groups):
            print(f"{index}/{len(groups)} 만기 처리", flush=True)

    development = [
        row for row in account.daily if str(row["date"]) <= DEVELOPMENT_END
    ]
    validation = [
        row for row in account.daily if str(row["date"]) > DEVELOPMENT_END
    ]
    audit = internal_audit(account.daily, account.trades)
    result = {
        "status": "provisional_time_split_validated",
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "initial_cash": INITIAL_CASH,
            "data_start": COMMON_START,
            "entry_window": [ENTRY_START, ENTRY_CUTOFF],
            "premium_bands": [
                [band.lower, band.upper] for band in PREMIUM_BANDS
            ],
            "market_gate": "직전일까지 VIX 120일 정규화 기울기 > 0",
            "direction": "PUT only",
            "option_gate": (
                f"신호 완료봉까지 20분 로그수익률 표준편차 > "
                f"{RV20_THRESHOLD}"
            ),
            "selection": (
                "계약·가격대별 첫 적격 사건 중 가장 이른 하나; "
                "당일 재진입 없음"
            ),
            "risk": (
                "당일 시작자금의 5%, 최대 500만원; "
                "진입·청산봉 거래량 10%"
            ),
            "exit": (
                "계약수 절반 올림을 2배, 나머지를 10배에서 청산; "
                "15:15 미도달 잔량 시간청산"
            ),
            "costs": (
                "매수 +0.01·매도 -0.01 슬리피지, "
                "계약당 편도 수수료 500원"
            ),
        },
        "summary": summarize(account.daily),
        "development": summarize(development),
        "validation": summarize(validation),
        "audit": audit,
        "limitations": [
            "RV20 0.13은 개발구간에서 선택해 완전한 독립 검증이 아니다.",
            "상위 수익일을 제거하면 누적손익이 음수인 꼬리수익 전략이다.",
            "현물·선물 MA·RSI·basis·상관 지표는 후반 재현성이 낮아 "
            "최종 진입 규칙에서 제외했다.",
            "실거래 전에 이후 만기일을 잠근 전진검증이 필요하다.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv",
        account.daily,
    )
    write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv",
        account.trades,
    )
    write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv",
        account.positions,
    )
    (OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{OUTPUT_STEM}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
