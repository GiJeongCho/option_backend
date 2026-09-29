from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import minimal_baseline_backtest as base
from strategy1_backtest import previous_volatility, volatility_closes


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OPPORTUNITIES_PATH = OUTPUT_DIR / "opportunities.csv"
HIGH_VOL_START = "20251001"


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def opportunity_rows() -> dict[tuple[str, str], dict[str, str]]:
    with OPPORTUNITIES_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        return {
            (row["expiry_date"], row["option_code"]): row
            for row in csv.DictReader(source)
            if truth(row["eligible"]) and row["expiry_date"] >= HIGH_VOL_START
        }


def minute_index(contract: base.ContractGrid, minute: int) -> int | None:
    index = base.MINUTE_INDEX.get(minute)
    if index is None or index >= len(contract.bars):
        return None
    return index


def first_time(
    contract: base.ContractGrid,
    start_index: int,
    end_index: int,
    condition: Callable[[int], bool],
) -> int | None:
    for index in range(start_index + 1, end_index):
        if condition(index):
            return contract.bars[index].minute
    return None


def ratio_to_previous_volume(
    contract: base.ContractGrid, index: int
) -> float:
    if index < 20:
        return 0.0
    average = (
        sum(bar.volume for bar in contract.bars[index - 20 : index]) / 20
    )
    if average <= 0:
        return float("inf") if contract.bars[index].volume > 0 else 0.0
    return contract.bars[index].volume / average


def breakout(
    contract: base.ContractGrid, index: int, lookback: int
) -> bool:
    if index < lookback:
        return False
    bar = contract.bars[index]
    if not bar.traded or bar.close is None:
        return False
    previous_highs = [
        item.high for item in contract.bars[index - lookback : index]
    ]
    return (
        all(value is not None for value in previous_highs)
        and bar.close > max(float(value) for value in previous_highs)
    )


def rising(
    values: list[float | None], index: int, lookback: int = 5
) -> bool:
    if index < lookback:
        return False
    return (
        values[index] is not None
        and values[index - lookback] is not None
        and float(values[index]) > float(values[index - lookback])
    )


def analyze_contract(
    date_value: str,
    contract: base.ContractGrid,
    opportunity: dict[str, str],
    previous_vix: float,
) -> dict[str, object] | None:
    success_5x = truth(opportunity["success_5x"])
    success_10x = truth(opportunity["success_10x"])
    if success_10x:
        anchor_price = optional_float(opportunity["base_10x"])
        anchor_minute = optional_int(opportunity["base_time_10x"])
    elif success_5x:
        anchor_price = optional_float(opportunity["base_5x"])
        anchor_minute = optional_int(opportunity["base_time_5x"])
    else:
        anchor_price = optional_float(opportunity["max_multiple_base"])
        anchor_minute = optional_int(opportunity["max_multiple_base_time"])
    if anchor_price is None or anchor_minute is None:
        return None
    anchor_index = minute_index(contract, anchor_minute)
    if anchor_index is None:
        return None

    target_minute = (
        optional_int(opportunity["target_time_10x"])
        if success_10x
        else optional_int(opportunity["target_time_5x"])
    )
    target_index = (
        minute_index(contract, target_minute)
        if target_minute is not None
        else None
    )
    # 목표 배수가 처음 나온 봉은 봉 내부 고가와 종가의 선후를 알 수 없다.
    # 따라서 성공 사례의 공통점은 목표 봉 직전까지만 관찰한다.
    end_index = (
        target_index
        if target_index is not None
        else len(contract.bars)
    )
    if end_index <= anchor_index + 1:
        return None

    def traded(index: int) -> bool:
        bar = contract.bars[index]
        return bar.traded and bar.close is not None

    def rebound(index: int, multiple: float) -> bool:
        return (
            traded(index)
            and float(contract.bars[index].close) >= anchor_price * multiple
        )

    def ma5_reclaim(index: int) -> bool:
        if index <= 0 or not traded(index):
            return False
        previous = contract.bars[index - 1]
        return (
            previous.close is not None
            and contract.ma5[index - 1] is not None
            and contract.ma5[index] is not None
            and previous.close <= float(contract.ma5[index - 1])
            and float(contract.bars[index].close)
            > float(contract.ma5[index])
        )

    def ma5_above_ma20(index: int) -> bool:
        return (
            contract.ma5[index] is not None
            and contract.ma20[index] is not None
            and float(contract.ma5[index]) > float(contract.ma20[index])
        )

    def volume_ratio(index: int, threshold: float) -> bool:
        return traded(index) and ratio_to_previous_volume(
            contract, index
        ) > threshold

    conditions: dict[str, Callable[[int], bool]] = {
        "rebound_1_10": lambda i: rebound(i, 1.10),
        "rebound_1_20": lambda i: rebound(i, 1.20),
        "rebound_1_30": lambda i: rebound(i, 1.30),
        "rebound_1_50": lambda i: rebound(i, 1.50),
        "breakout_3": lambda i: breakout(contract, i, 3),
        "breakout_5": lambda i: breakout(contract, i, 5),
        "breakout_10": lambda i: breakout(contract, i, 10),
        "ma5_reclaim": ma5_reclaim,
        "ma5_above_ma20": ma5_above_ma20,
        "ma20_rising_5m": lambda i: rising(contract.ma20, i),
        "volume_gt_avg20": lambda i: volume_ratio(i, 1.0),
        "volume_gt_1_5_avg20": lambda i: volume_ratio(i, 1.5),
        "r20_break5": lambda i: rebound(i, 1.20)
        and breakout(contract, i, 5),
        "r20_break5_volume": lambda i: rebound(i, 1.20)
        and breakout(contract, i, 5)
        and volume_ratio(i, 1.0),
        "r20_break5_ma20_up": lambda i: rebound(i, 1.20)
        and breakout(contract, i, 5)
        and rising(contract.ma20, i),
        "r20_break5_volume_ma20_up": lambda i: rebound(i, 1.20)
        and breakout(contract, i, 5)
        and volume_ratio(i, 1.0)
        and rising(contract.ma20, i),
        "r30_break5_volume": lambda i: rebound(i, 1.30)
        and breakout(contract, i, 5)
        and volume_ratio(i, 1.0),
        "ma5_reclaim_break5_volume": lambda i: ma5_reclaim(i)
        and breakout(contract, i, 5)
        and volume_ratio(i, 1.0),
    }
    times = {
        name: first_time(
            contract, anchor_index, end_index, condition
        )
        for name, condition in conditions.items()
    }
    elapsed = {
        name: (
            None
            if minute is None
            else base.MINUTE_INDEX[minute] - anchor_index
        )
        for name, minute in times.items()
    }
    rebound_20_time = times["rebound_1_20"]
    minutes_to_rebound_20 = (
        None
        if rebound_20_time is None
        else base.MINUTE_INDEX[rebound_20_time] - anchor_index
    )
    return {
        "expiry_date": date_value,
        "option_code": contract.code,
        "call_put": contract.call_put,
        "previous_vix": previous_vix,
        "success_5x": success_5x,
        "success_10x": success_10x,
        "max_multiple": float(opportunity["max_multiple"]),
        "anchor_price": anchor_price,
        "anchor_minute": anchor_minute,
        "target_minute": target_minute,
        "minutes_to_rebound_20": minutes_to_rebound_20,
        **{
            f"{name}_before_target": minute is not None
            for name, minute in times.items()
        },
        **{
            f"rebound_1_20_within_{window}m_before_target": (
                elapsed["rebound_1_20"] is not None
                and int(elapsed["rebound_1_20"]) <= window
            )
            for window in (5, 10, 20, 40, 60)
        },
        **{
            f"r20_break5_within_{window}m_before_target": (
                elapsed["r20_break5"] is not None
                and int(elapsed["r20_break5"]) <= window
            )
            for window in (10, 20, 40, 60)
        },
        **{f"{name}_minute": minute for name, minute in times.items()},
    }


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    feature_names = [
        key.removesuffix("_before_target")
        for key in rows[0]
        if key.endswith("_before_target")
    ]
    result: list[dict[str, object]] = []
    for feature in feature_names:
        column = f"{feature}_before_target"
        present = [row for row in rows if bool(row[column])]
        five = [row for row in rows if bool(row["success_5x"])]
        ten = [row for row in rows if bool(row["success_10x"])]
        non_five = [row for row in rows if not bool(row["success_5x"])]
        result.append(
            {
                "feature": feature,
                "all_count": len(rows),
                "pattern_count": len(present),
                "pattern_rate_pct": rate(len(present), len(rows)),
                "five_x_count": len(five),
                "five_x_with_pattern": sum(
                    bool(row[column]) for row in five
                ),
                "five_x_commonality_pct": rate(
                    sum(bool(row[column]) for row in five), len(five)
                ),
                "ten_x_count": len(ten),
                "ten_x_with_pattern": sum(
                    bool(row[column]) for row in ten
                ),
                "ten_x_commonality_pct": rate(
                    sum(bool(row[column]) for row in ten), len(ten)
                ),
                "non_five_x_count": len(non_five),
                "non_five_x_with_pattern": sum(
                    bool(row[column]) for row in non_five
                ),
                "non_five_x_pattern_pct": rate(
                    sum(bool(row[column]) for row in non_five),
                    len(non_five),
                ),
                "five_x_rate_when_pattern_pct": rate(
                    sum(bool(row["success_5x"]) for row in present),
                    len(present),
                ),
                "ten_x_rate_when_pattern_pct": rate(
                    sum(bool(row["success_10x"]) for row in present),
                    len(present),
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -float(row["five_x_commonality_pct"]),
            -float(row["five_x_rate_when_pattern_pct"]),
        ),
    )


def by_period(rows: list[dict[str, object]]) -> dict[str, object]:
    periods = {
        "all": rows,
        "development": [
            row for row in rows if row["expiry_date"] <= "20260430"
        ],
        "validation": [
            row for row in rows if row["expiry_date"] >= "20260501"
        ],
    }
    return {
        name: {
            "eligible_contracts": len(values),
            "five_x_contracts": sum(
                bool(row["success_5x"]) for row in values
            ),
            "ten_x_contracts": sum(
                bool(row["success_10x"]) for row in values
            ),
            "features": summaries(values) if values else [],
        }
        for name, values in periods.items()
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, object]:
    opportunities = opportunity_rows()
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= HIGH_VOL_START
    ]
    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    rows: list[dict[str, object]] = []
    for group in groups:
        date_value = group["expiry_date"]
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        for code, contract in contracts.items():
            opportunity = opportunities.get((date_value, code))
            if opportunity is None:
                continue
            analyzed = analyze_contract(
                date_value, contract, opportunity, previous_vix
            )
            if analyzed is not None:
                rows.append(analyzed)

    result = {
        "status": "descriptive_event_study_not_entry_backtest",
        "scope": {
            "date_start": HIGH_VOL_START,
            "previous_vix_filter": None,
            "eligible_low_range": "1.00-5.00",
            "source_eligible_contracts": len(opportunities),
            "source_five_x_contracts": sum(
                truth(row["success_5x"])
                for row in opportunities.values()
            ),
            "source_ten_x_contracts": sum(
                truth(row["success_10x"])
                for row in opportunities.values()
            ),
            "eligible_contracts": len(rows),
            "excluded_without_pre_target_observation": (
                len(opportunities) - len(rows)
            ),
        },
        "method": {
            "winner_anchor": (
                "10x base low if available, otherwise 5x base low"
            ),
            "non_winner_anchor": "max-multiple base low",
            "same_bar_guard": (
                "target bar excluded because intrabar high/close order "
                "is unknown"
            ),
            "warning": (
                "anchors use future outcome labels; commonality counts "
                "are descriptive only and cannot be traded directly"
            ),
        },
        "periods": by_period(rows),
    }
    write_csv(OUTPUT_DIR / "s3_breakout_event_details.csv", rows)
    write_csv(
        OUTPUT_DIR / "s3_breakout_feature_summary.csv",
        result["periods"]["all"]["features"],  # type: ignore[index]
    )
    (OUTPUT_DIR / "s3_breakout_event_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
