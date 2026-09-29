from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import minimal_baseline_backtest as base


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEVELOPMENT_END = "20260430"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def stats(rows: list[dict[str, object]]) -> dict[str, object]:
    values = [int(row["pnl"]) for row in rows]
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return {
        "count": len(rows),
        "profitable": sum(value > 0 for value in values),
        "losing": sum(value < 0 for value in values),
        "total_pnl": sum(values),
        "average_pnl": round(sum(values) / len(values)) if values else 0,
        "profit_factor": round(gains / losses, 4) if losses else None,
    }


def grouped(
    rows: list[dict[str, object]], field: str
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result: list[dict[str, object]] = []
    for name, group_rows in sorted(groups.items()):
        development = [
            row
            for row in group_rows
            if str(row["date"]) <= DEVELOPMENT_END
        ]
        validation = [
            row
            for row in group_rows
            if str(row["date"]) > DEVELOPMENT_END
        ]
        result.append(
            {
                "group": name,
                "all": stats(group_rows),
                "development": stats(development),
                "validation": stats(validation),
            }
        )
    return result


def entry_time_bucket(minute: int) -> str:
    if minute < 930:
        return "before_0930"
    if minute < 1000:
        return "0930_0959"
    return "after_1000"


def vix_bucket(value: float) -> str:
    if value < 40:
        return "30_39"
    if value < 50:
        return "40_49"
    return "50_plus"


def price_bucket(value: float) -> str:
    if value < 1.20:
        return "1.00_1.19"
    if value < 1.35:
        return "1.20_1.34"
    return "1.35_1.49"


def setup_bucket(value: float) -> str:
    if value < 0.20:
        return "under_20pct"
    if value < 0.35:
        return "20_35pct"
    return "35pct_plus"


def volume_bucket(value: float) -> str:
    if value < 1.5:
        return "1.00_1.49"
    if value < 2.0:
        return "1.50_1.99"
    if value < 3.0:
        return "2.00_2.99"
    return "3.00_plus"


def run() -> dict[str, object]:
    campaign_rows = read_csv(OUTPUT_DIR / "strategy1_campaigns.csv")
    daily_rows = {
        row["date"]: row
        for row in read_csv(OUTPUT_DIR / "strategy1_daily.csv")
    }
    trades = read_csv(OUTPUT_DIR / "strategy1_trades.csv")
    first_buys = {
        (row["date"], row["campaign"]): row
        for row in trades
        if row["side"] == "BUY" and row["reason"] == "ENTRY_STAGE_1"
    }

    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = {
        group["expiry_date"]: group
        for group in base.derive_expiry_groups(profile)
    }
    files = base.target_file_map()
    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in campaign_rows:
        by_date[row["date"]].append(row)

    enriched: list[dict[str, object]] = []
    for date_value, rows in by_date.items():
        contracts = base.load_expiry_day(
            files[date_value], groups[date_value]
        )
        for row in rows:
            first_buy = first_buys[(date_value, row["campaign"])]
            initial_price = float(first_buy["price"])
            contract = contracts[row["code"]]
            signal_index = base.MINUTE_INDEX[
                int(row["first_signal_minute"])
            ]
            bar = contract.bars[signal_index]
            ma5 = contract.ma5[signal_index]
            ma20 = contract.ma20[signal_index]
            assert bar.close is not None and ma5 is not None and ma20 is not None
            previous_high = max(
                float(contract.bars[index].high)
                for index in range(signal_index - 5, signal_index)
                if contract.bars[index].high is not None
            )
            average_volume = (
                sum(
                    contract.bars[index].volume
                    for index in range(signal_index - 20, signal_index)
                )
                / 20
            )
            setup_distance = (
                initial_price - float(row["setup_low"])
            ) / initial_price
            ma_above = ma5 > ma20
            breakout = bar.close > previous_high
            if ma_above and breakout:
                signal_structure = "ma_above_and_breakout"
            elif ma_above:
                signal_structure = "ma_above_only"
            elif breakout:
                signal_structure = "breakout_only"
            else:
                signal_structure = "neither"
            enriched.append(
                {
                    **row,
                    "pnl": int(row["pnl"]),
                    "campaign_order": row["campaign"],
                    "initial_price": initial_price,
                    "initial_price_bucket": price_bucket(initial_price),
                    "entry_time_bucket": entry_time_bucket(
                        int(row["first_entry_minute"])
                    ),
                    "previous_vix": float(
                        daily_rows[date_value]["previous_vix"]
                    ),
                    "vix_bucket": vix_bucket(
                        float(daily_rows[date_value]["previous_vix"])
                    ),
                    "setup_distance_pct": setup_distance * 100,
                    "setup_distance_bucket": setup_bucket(setup_distance),
                    "weekday": datetime.strptime(
                        date_value, "%Y%m%d"
                    ).strftime("%a"),
                    "signal_structure": signal_structure,
                    "volume_ratio": (
                        bar.volume / average_volume
                        if average_volume
                        else 0.0
                    ),
                    "volume_ratio_bucket": volume_bucket(
                        bar.volume / average_volume
                        if average_volume
                        else 0.0
                    ),
                }
            )

    dimensions = {
        name: grouped(enriched, name)
        for name in (
            "campaign_order",
            "tranche_count",
            "call_put",
            "entry_time_bucket",
            "vix_bucket",
            "initial_price_bucket",
            "setup_distance_bucket",
            "weekday",
            "exit_reason",
            "signal_structure",
            "volume_ratio_bucket",
        )
    }
    result = {
        "strategy": "S1-v0.1",
        "development_period": f"through_{DEVELOPMENT_END}",
        "validation_period": "20260501_onward",
        "overall": stats(enriched),
        "dimensions": dimensions,
        "findings": [
            {
                "pattern": "initial_price_1.20_1.34",
                "interpretation": (
                    "유일하게 개발·후반 구간 모두 양수였지만 "
                    "사후 구간 선택이므로 새 전략의 단독 근거로 사용하지 않음"
                ),
            },
            {
                "pattern": "tranche_2_only",
                "interpretation": (
                    "2차 진입 후 3차 확인에 이르지 못한 8건은 "
                    "모두 손실이어서 이른 증액이 손실을 키움"
                ),
            },
            {
                "pattern": "strict_ma_and_breakout",
                "interpretation": (
                    "MA5>MA20과 5분 돌파를 함께 만족한 표본은 "
                    "양 구간 양수지만 11건뿐이라 별도 대조군 필요"
                ),
            },
            {
                "pattern": "fixed_stop_instability",
                "interpretation": (
                    "별도 2.50~5.00 연구에서 -20~-40% 가격 손절은 "
                    "2배 성공 전 큰 역행을 견딘 거래까지 제거함"
                ),
            },
        ],
    }
    (OUTPUT_DIR / "strategy1_pattern_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
