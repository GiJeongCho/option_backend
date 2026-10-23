from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import minimal_baseline_backtest as base
import strategy1_backtest as strategy1


STRATEGY_VERSION = "S6-v0.2"
OUTPUT_STEM = "strategy6_v02"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
AUDIT_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_independent_audit.json"

EXPECTED_INITIAL_CASH = 20_000_000
MAX_DAILY_PRINCIPAL = 5_000_000
RISK_FRACTION = 0.05
ENTRY_CUTOFF = 1300
RV20_THRESHOLD = 0.13
BANDS = ((0.40, 0.50), (1.00, 2.50))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def integer(value: str | None) -> int:
    return int(float(value or 0))


def number(value: str | None) -> float:
    return float(value or 0)


def normalized_slope(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3 or abs(array[-1]) <= 1e-12:
        return math.nan
    x = np.arange(len(array), dtype=float)
    return float(np.polyfit(x, array, 1)[0] / array[-1])


def realized_volatility(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 21 or np.any(array[-21:] <= 0):
        return math.nan
    returns = np.diff(np.log(array[-21:]))
    return float(np.std(returns, ddof=1))


def independent_vix_features() -> tuple[list[str], dict[str, dict[str, float]]]:
    dates, closes = strategy1.volatility_closes()
    ordered = sorted(zip(dates, closes), key=lambda item: item[0])
    unique_dates: list[str] = []
    unique_closes: list[float] = []
    for date_value, close in ordered:
        if unique_dates and date_value == unique_dates[-1]:
            unique_closes[-1] = float(close)
        else:
            unique_dates.append(date_value)
            unique_closes.append(float(close))

    features: dict[str, dict[str, float]] = {}
    for index, date_value in enumerate(unique_dates):
        history = unique_closes[: index + 1]
        close = history[-1]
        previous_close = history[-2] if len(history) >= 2 else math.nan
        features[date_value] = {
            "close": close,
            "previous_close": previous_close,
            "ma5": (
                float(np.mean(history[-5:]))
                if len(history) >= 5
                else math.nan
            ),
            "slope5": (
                normalized_slope(history[-5:])
                if len(history) >= 5
                else math.nan
            ),
            "daily_return_1": (
                close / previous_close - 1.0
                if previous_close > 0
                else math.nan
            ),
        }
    return unique_dates, features


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def run() -> dict[str, object]:
    daily = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv")
    trades = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv")
    positions = read_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = {
        group["expiry_date"]: group
        for group in base.derive_expiry_groups(profile)
    }
    files = base.target_file_map()
    vix_dates, vix_features = independent_vix_features()

    trades_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trade in trades:
        trades_by_date[trade["date"]].append(trade)
        if trade["strategy"] != STRATEGY_VERSION:
            errors.append(f"{trade['date']}: 잘못된 전략명")

    previous_end = EXPECTED_INITIAL_CASH
    for row in daily:
        date_value = row["date"]
        start_cash = integer(row["start_cash"])
        end_cash = integer(row["end_cash"])
        pnl = integer(row["pnl"])
        day_trades = trades_by_date.get(date_value, [])
        if start_cash != previous_end:
            errors.append(f"{date_value}: 현금 연속성")
        if end_cash != start_cash + pnl or end_cash < 0:
            errors.append(f"{date_value}: 일별 현금")
        if sum(integer(item["cash_flow"]) for item in day_trades) != pnl:
            errors.append(f"{date_value}: 거래장부 손익")

        buys = [item for item in day_trades if item["side"] == "BUY"]
        if len(buys) > 1:
            errors.append(f"{date_value}: 하루 다중 진입")
        buy_principal = sum(integer(item["principal"]) for item in buys)
        budget = min(
            MAX_DAILY_PRINCIPAL,
            math.floor(start_cash * RISK_FRACTION),
        )
        if buy_principal > budget:
            errors.append(f"{date_value}: 위험예산 초과")

        inventory = 0
        for trade in sorted(
            day_trades,
            key=lambda item: (
                integer(item["minute"]),
                0 if item["side"] == "BUY" else 1,
            ),
        ):
            quantity = integer(trade["quantity"])
            inventory += quantity if trade["side"] == "BUY" else -quantity
            if inventory < 0:
                errors.append(f"{date_value}: 음수 보유수량")
        if inventory:
            errors.append(f"{date_value}: 미폐쇄 수량 {inventory}")
        previous_end = end_cash

    contracts_by_date: dict[str, dict[str, base.ContractGrid]] = {}
    for position in positions:
        date_value = position["date"]
        if position["strategy"] != STRATEGY_VERSION:
            errors.append(f"{date_value}: 포지션 전략명")
        if date_value not in groups or date_value not in files:
            errors.append(f"{date_value}: 원자료 누락")
            continue
        if date_value not in contracts_by_date:
            contracts_by_date[date_value] = base.load_expiry_day(
                files[date_value],
                groups[date_value],
            )
        contract = contracts_by_date[date_value].get(position["code"])
        if contract is None:
            errors.append(f"{date_value}: 계약 누락")
            continue
        if "PUT" not in contract.call_put.upper() and "풋" not in contract.call_put:
            errors.append(f"{date_value}: PUT 아님")

        signal_minute = integer(position["signal_minute"])
        entry_minute = integer(position["entry_minute"])
        if not 930 <= signal_minute < ENTRY_CUTOFF:
            errors.append(f"{date_value}: 신호 시간")
        signal_index = base.MINUTE_INDEX[signal_minute]
        entry_index = base.MINUTE_INDEX[entry_minute]
        signal_bar = contract.bars[signal_index]
        entry_bar = contract.bars[entry_index]
        if not signal_bar.traded or signal_bar.close is None:
            errors.append(f"{date_value}: 신호봉")
            continue
        if not entry_bar.traded or entry_bar.open is None:
            errors.append(f"{date_value}: 진입봉")
            continue
        if entry_index <= signal_index or entry_index - signal_index > 5:
            errors.append(f"{date_value}: 다음 봉 진입")

        raw_open = float(entry_bar.open)
        if not any(lower <= raw_open <= upper for lower, upper in BANDS):
            errors.append(f"{date_value}: 진입 가격대")
        if not close_enough(
            number(position["entry_price"]),
            raw_open + base.SLIPPAGE,
        ):
            errors.append(f"{date_value}: 매수 슬리피지")

        quantity = integer(position["entry_quantity"])
        if quantity > math.floor(
            entry_bar.volume * base.ENTRY_PARTICIPATION + 1e-12
        ):
            errors.append(f"{date_value}: 진입 거래량")
        first_quantity = math.ceil(quantity / 2)
        if integer(position["target_2x_quantity"]) != first_quantity:
            errors.append(f"{date_value}: 2배 배분")
        if integer(position["target_10x_quantity"]) != quantity - first_quantity:
            errors.append(f"{date_value}: 10배 배분")

        closes = [
            float(bar.close)
            for bar in contract.bars[: signal_index + 1]
            if bar.close is not None
        ]
        rv20 = realized_volatility(closes)
        if (
            not rv20 > RV20_THRESHOLD
            or not close_enough(
                rv20,
                number(position["option_realized_volatility_20"]),
            )
        ):
            errors.append(f"{date_value}: RV20 게이트")

        reference_date = next(
            (item for item in reversed(vix_dates) if item < date_value),
            "",
        )
        feature = vix_features.get(reference_date)
        if feature is None:
            errors.append(f"{date_value}: VIX 기준일")
        else:
            strict_uptrend = (
                feature["close"] > feature["ma5"]
                and feature["slope5"] > 0
                and feature["close"] > feature["previous_close"]
            )
            if position["vix_reference_date"] != reference_date:
                errors.append(f"{date_value}: VIX 미래정보 차단")
            for column, key in (
                ("vix_close", "close"),
                ("vix_previous_close", "previous_close"),
                ("vix_ma5", "ma5"),
                ("vix_slope_5", "slope5"),
                ("vix_daily_return_1", "daily_return_1"),
            ):
                if not close_enough(number(position[column]), feature[key]):
                    errors.append(f"{date_value}: {column} 재계산")
            if not strict_uptrend or position["vix_strict_uptrend"] != "True":
                errors.append(f"{date_value}: VIX 단기 상승추세 게이트")

        entry_price = number(position["entry_price"])
        for trade in trades_by_date.get(date_value, []):
            if trade["side"] != "SELL":
                continue
            minute = integer(trade["minute"])
            sell_bar = contract.bars[base.MINUTE_INDEX[minute]]
            if (
                not sell_bar.traded
                or sell_bar.open is None
                or not close_enough(
                    number(trade["price"]),
                    max(0.0, float(sell_bar.open) - base.SLIPPAGE),
                )
            ):
                errors.append(f"{date_value}: 매도 체결가")
            reason = trade["reason"]
            if reason in {"TARGET_2X", "TARGET_10X"}:
                target = 2 if reason == "TARGET_2X" else 10
                sell_index = base.MINUTE_INDEX[minute]
                prior_hit = any(
                    bar.traded
                    and bar.high is not None
                    and float(bar.high) + 1e-9 >= entry_price * target
                    for bar in contract.bars[entry_index + 1 : sell_index]
                )
                if not prior_hit:
                    errors.append(f"{date_value}: {reason} 선행신호")
            elif reason == "TIME" and minute <= 1515:
                errors.append(f"{date_value}: 시간청산 시각")

    expected_total_pnl = sum(integer(row["pnl"]) for row in daily)
    if expected_total_pnl != int(summary["summary"]["total_pnl"]):
        errors.append("요약 누적손익")
    if daily and integer(daily[0]["start_cash"]) != EXPECTED_INITIAL_CASH:
        errors.append("시작자금")

    result = {
        "status": "independent_ledger_and_raw_replay_audit",
        "strategy_version": STRATEGY_VERSION,
        "passed": not errors,
        "error_count": len(errors),
        "error_samples": errors[:80],
        "daily_rows": len(daily),
        "trade_rows": len(trades),
        "position_rows": len(positions),
        "checks": [
            "현금 연속성·일별 손익·거래장부 현금흐름",
            "하루 한 번 진입·위험예산·수량 보존",
            "거래일 이전 VIX 종가·SMA5·5일 기울기·전일 상승 원자료 재계산",
            "PUT·옵션 RV20·가격대·다음 실제 거래봉 원자료 재계산",
            "거래량 10%·슬리피지·2배/10배 수량과 목표 선행 고가",
        ],
    }
    AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
