from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMP_OUTPUT = Path(__file__).resolve().parent / "output"
sys.path.insert(0, str(ROOT))

import minimal_baseline_backtest as base  # noqa: E402
import strategy6_v01_vix_rv_put_backtest as v01  # noqa: E402


STRATEGY_VERSION = "S6-v0.3-RV60-CANDIDATE"
OUTPUT_STEM = "s6_v03_rv60_runner_candidate"
RV60_THRESHOLD = 0.14116165
SLOPE10_THRESHOLD = 0.04306970


def runner_rule(
    contract: base.ContractGrid,
    index: int,
) -> tuple[bool, float, float]:
    closes = [
        float(bar.close)
        for bar in contract.bars[: index + 1]
        if bar.close is not None
    ]
    rv60 = v01.realized_volatility(closes, period=60)
    slope10 = v01.normalized_slope(closes[-10:])
    passed = (
        math.isfinite(rv60)
        and math.isfinite(slope10)
        and rv60 >= RV60_THRESHOLD
        and slope10 >= SLOPE10_THRESHOLD
    )
    return passed, rv60, slope10


def execute_exits(
    date_value: str,
    contract: base.ContractGrid,
    event: v01.EntryEvent,
    positions: list[v01.Tranche],
) -> tuple[int, int, list[dict[str, object]], dict[int, bool]]:
    gross_sales = 0
    fees = 0
    trades: list[dict[str, object]] = []
    reached = {2: False, 10: False}
    capacity = 0.0
    runner_decided = False
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
            for position in sorted(pending, key=lambda item: item.stage):
                fillable = min(
                    position.remaining,
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
            bar.minute <= v01.FORCE_EXIT_TIME
            and bar.traded
            and bar.high is not None
        ):
            two_x_hit = (
                not reached[2]
                and float(bar.high) + 1e-9 >= event.fill_price * 2
            )
            if two_x_hit:
                reached[2] = True
                for position in positions:
                    if position.stage == 1 and not position.triggered:
                        position.triggered = True
                        position.eligible_index = index + 1
                        position.reason = "TARGET_2X"
                runner = next(
                    (
                        position
                        for position in positions
                        if position.stage == 2 and position.remaining > 0
                    ),
                    None,
                )
                if runner is not None and not runner_decided:
                    runner_decided = True
                    passed, rv60, slope10 = runner_rule(contract, index)
                    trades.append(
                        {
                            "strategy": STRATEGY_VERSION,
                            "date": date_value,
                            "campaign": 1,
                            "minute": bar.minute,
                            "code": contract.code,
                            "call_put": contract.call_put,
                            "side": "RULE",
                            "reason": (
                                "RV60_SLOPE10_KEEP"
                                if passed
                                else "RV60_SLOPE10_REJECT"
                            ),
                            "stage": 2,
                            "quantity": 0,
                            "price": 0.0,
                            "principal": 0,
                            "fee": 0,
                            "cash_flow": 0,
                            "bar_volume": bar.volume,
                            "rv60": rv60,
                            "slope10": slope10,
                        }
                    )
                    if not passed:
                        runner.triggered = True
                        runner.eligible_index = index + 1
                        runner.reason = "RV60_SLOPE10_REJECT_2X_EXIT"

            for position in positions:
                if (
                    position.stage == 2
                    and position.remaining > 0
                    and not position.triggered
                    and float(bar.high) + 1e-9 >= event.fill_price * 10
                ):
                    position.triggered = True
                    position.eligible_index = index + 1
                    position.reason = "TARGET_10X"
                    reached[10] = True

        if bar.minute == v01.FORCE_EXIT_TIME:
            for position in positions:
                if position.remaining > 0 and not position.triggered:
                    position.triggered = True
                    position.eligible_index = index + 1
                    position.reason = "TIME"
        if (
            bar.minute > v01.FORCE_EXIT_TIME
            and not any(position.remaining > 0 for position in positions)
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


def main() -> None:
    TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    v01.STRATEGY_VERSION = STRATEGY_VERSION
    v01.OUTPUT_STEM = OUTPUT_STEM
    v01.OUTPUT_DIR = TEMP_OUTPUT
    v01.execute_exits = execute_exits
    result = v01.run()
    trades_path = TEMP_OUTPUT / f"{OUTPUT_STEM}_trades.csv"
    trade_text = trades_path.read_text(encoding="utf-8-sig")
    result["status"] = "exploratory_post_selection_candidate"
    result["rules"]["runner_filter"] = (
        "첫 2배 확인봉 완료시 RV60 >= 0.14116165 AND "
        "10분 정규화 기울기 >= 0.04306970이면 잔여 절반을 10배까지 "
        "보유, 아니면 다음 실제 거래봉에서 청산"
    )
    result["runner_diagnostics"] = {
        "kept": trade_text.count("RV60_SLOPE10_KEEP"),
        "rejected": trade_text.count("RV60_SLOPE10_REJECT,"),
    }
    result["limitations"].insert(
        0,
        (
            "RV60·기울기 후보를 후반 단변량 결과 확인 뒤 선택했으므로 "
            "독립 검증이 아니다. S6-v0.3 연구 후보로 화면 비교는 "
            "허용하되 실거래 채택 전략으로 승격하지 않는다."
        ),
    )
    summary_path = TEMP_OUTPUT / f"{OUTPUT_STEM}_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "development": result["development"],
                "validation": result["validation"],
                "runner_diagnostics": result["runner_diagnostics"],
                "audit": result["audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
