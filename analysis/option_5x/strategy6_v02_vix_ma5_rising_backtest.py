from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import minimal_baseline_backtest as base
import strategy1_backtest as strategy1
import strategy6_v01_vix_rv_put_backtest as v01


STRATEGY_VERSION = "S6-v0.2"
OUTPUT_STEM = "strategy6_v02"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class VixSnapshot:
    date: str
    close: float
    previous_close: float
    ma5: float
    slope5: float
    slope120: float

    @property
    def daily_return(self) -> float:
        if abs(self.previous_close) <= 1e-12:
            return math.nan
        return self.close / self.previous_close - 1.0

    @property
    def strict_uptrend(self) -> bool:
        return bool(
            math.isfinite(self.ma5)
            and math.isfinite(self.slope5)
            and self.close > self.ma5
            and self.slope5 > 0
            and self.close > self.previous_close
        )


def vix_snapshots() -> tuple[list[str], dict[str, VixSnapshot]]:
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
    result: dict[str, VixSnapshot] = {}
    for index, date_value in enumerate(unique_dates):
        history = unique_closes[: index + 1]
        previous_close = history[-2] if len(history) >= 2 else math.nan
        ma5 = (
            float(np.mean(history[-5:]))
            if len(history) >= 5
            else math.nan
        )
        result[date_value] = VixSnapshot(
            date=date_value,
            close=history[-1],
            previous_close=previous_close,
            ma5=ma5,
            slope5=v01.normalized_slope(history[-5:]),
            slope120=v01.normalized_slope(history[-120:]),
        )
    return unique_dates, result


def previous_snapshot(
    date_value: str,
    dates: list[str],
    snapshots: dict[str, VixSnapshot],
) -> VixSnapshot | None:
    previous_date = next(
        (item for item in reversed(dates) if item < date_value),
        "",
    )
    return snapshots.get(previous_date)


def add_gate_fields(
    row: dict[str, object],
    snapshot: VixSnapshot | None,
) -> None:
    row.update(
        {
            "vix_reference_date": snapshot.date if snapshot else "",
            "vix_close": snapshot.close if snapshot else None,
            "vix_previous_close": (
                snapshot.previous_close if snapshot else None
            ),
            "vix_ma5": snapshot.ma5 if snapshot else None,
            "vix_slope_5": snapshot.slope5 if snapshot else None,
            "vix_daily_return_1": (
                snapshot.daily_return if snapshot else None
            ),
            "vix_strict_uptrend": (
                snapshot.strict_uptrend if snapshot else False
            ),
        }
    )


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= v01.COMMON_START
    ]
    files = base.target_file_map()
    dates, snapshots = vix_snapshots()
    account = v01.Account()
    assert account.daily is not None
    assert account.trades is not None
    assert account.positions is not None

    original_version = v01.STRATEGY_VERSION
    v01.STRATEGY_VERSION = STRATEGY_VERSION
    try:
        for index, group in enumerate(groups, start=1):
            date_value = group["expiry_date"]
            contracts = (
                base.load_expiry_day(files[date_value], group)
                if date_value in files
                else {}
            )
            snapshot = previous_snapshot(date_value, dates, snapshots)
            gate_value = (
                1.0
                if snapshot is not None and snapshot.strict_uptrend
                else -1.0
            )
            event = v01.select_event(contracts, gate_value)
            if event is not None and snapshot is not None:
                event = replace(
                    event,
                    vix_slope_120=snapshot.slope120,
                )
            daily, trades, position = v01.simulate_day(
                date_value,
                account,
                contracts,
                event,
            )
            add_gate_fields(daily, snapshot)
            if position is not None:
                add_gate_fields(position, snapshot)
            account.cash = int(daily["end_cash"])
            account.daily.append(daily)
            account.trades.extend(trades)
            if position is not None:
                account.positions.append(position)
            if index % 20 == 0 or index == len(groups):
                print(f"{index}/{len(groups)} 만기 처리", flush=True)
    finally:
        v01.STRATEGY_VERSION = original_version

    development = [
        row
        for row in account.daily
        if str(row["date"]) <= v01.DEVELOPMENT_END
    ]
    validation = [
        row
        for row in account.daily
        if str(row["date"]) > v01.DEVELOPMENT_END
    ]
    audit = v01.internal_audit(account.daily, account.trades)
    result = {
        "status": "rejected_time_split_comparison",
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "initial_cash": v01.INITIAL_CASH,
            "data_start": v01.COMMON_START,
            "date_gate": (
                "직전 VIX 종가 > 직전 VIX SMA5, VIX 5일 정규화 "
                "회귀기울기 > 0, 직전 VIX 종가 > 그 전 VIX 종가"
            ),
            "date_gate_lookahead": "거래일보다 이전 VIX 값만 사용",
            "direction": "PUT only",
            "premium_bands": [
                [band.lower, band.upper] for band in v01.PREMIUM_BANDS
            ],
            "option_gate": (
                f"신호 완료봉까지 옵션 RV20 > "
                f"{v01.RV20_THRESHOLD}"
            ),
            "risk": (
                "당일 시작자금의 5%, 최대 500만원; "
                "진입·청산봉 거래량 10%"
            ),
            "exit": (
                "계약수 절반 올림을 2배, 나머지를 10배에서 청산; "
                "15:15 시간청산"
            ),
        },
        "summary": v01.summarize(account.daily),
        "development": v01.summarize(development),
        "validation": v01.summarize(validation),
        "audit": audit,
        "decision": (
            "요청한 단기 상승추세 일자 필터를 정확히 적용했지만 "
            "시간순 후반 손익이 음수여서 S6-v0.1을 대체하지 않음"
        ),
        "limitations": [
            "후반 실제 거래일이 적어 추가 만기일 잠금 검증이 필요하다.",
            "최고 수익일 의존도가 높고 최고 3일 제외 손익이 음수다.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    v01.write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_daily.csv",
        account.daily,
    )
    v01.write_csv(
        OUTPUT_DIR / f"{OUTPUT_STEM}_trades.csv",
        account.trades,
    )
    v01.write_csv(
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
