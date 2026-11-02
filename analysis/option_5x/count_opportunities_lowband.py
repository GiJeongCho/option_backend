# -*- coding: utf-8 -*-
"""
저가 밴드(0.4~) 포함 기회 집계 - 녹음5의 '0.4~0.8이 10배의 핵심' 검증용.

원본 count_opportunities.py 는 프리미엄 하한이 1.0으로 하드코딩돼 있어
0.4~0.8 구간(전문가가 지목한 10배 핵심 밴드)이 통째로 제외된다.
여기서는 하한만 0.4로 낮춰 opportunities_lowband.csv 를 따로 생성한다.
(원본 opportunities.csv / 기존 산출물은 건드리지 않는다.)
"""
from __future__ import annotations

import json
from pathlib import Path

import count_opportunities as co

PREMIUM_FLOOR = 0.40
PREMIUM_CEIL = 5.0
OUTPUT_DIR: Path = co.OUTPUT_DIR
EPS = co.PRICE_EPSILON


def update_candidate(candidate, low, time):
    if not PREMIUM_FLOOR <= low <= PREMIUM_CEIL:
        return candidate
    if candidate is None or low < candidate[0]:
        return (low, time)
    return candidate


def analyze_bars(bars):
    """원본 analyze_bars 와 동일 로직, 프리미엄 하한만 0.4. 밴드상세는 생략([])."""
    ordered = sorted(bars, key=lambda bar: bar.time)
    candidate = None
    first_candidate = None
    event_2x = None
    event_5x = None
    event_10x = None
    max_multiple = 0.0
    max_multiple_time = None
    max_multiple_base = None
    max_multiple_base_time = None

    for bar in ordered:
        # 목표 판정은 이전 봉까지 만들어진 후보 저가로만 (미래정보 차단, 원본과 동일)
        if candidate is not None:
            base, base_time = candidate
            multiple = bar.high / base
            if multiple > max_multiple:
                max_multiple = multiple
                max_multiple_time = bar.time
                max_multiple_base = base
                max_multiple_base_time = base_time
            if event_2x is None and co.reaches_multiple(bar.high, base, 2.0):
                event_2x = {
                    "base": base, "base_time": base_time,
                    "target_time": bar.time, "target_high": bar.high,
                }
            if event_5x is None and co.reaches_multiple(bar.high, base, 5.0):
                event_5x = {
                    "base": base, "base_time": base_time,
                    "target_time": bar.time, "target_high": bar.high,
                }
            if event_10x is None and co.reaches_multiple(bar.high, base, 10.0):
                event_10x = {
                    "base": base, "base_time": base_time,
                    "target_time": bar.time, "target_high": bar.high,
                }

        if PREMIUM_FLOOR <= bar.low <= PREMIUM_CEIL:
            if first_candidate is None:
                first_candidate = (bar.low, bar.time)
            candidate = update_candidate(candidate, bar.low, bar.time)

    overall = {
        "eligible": first_candidate is not None,
        "first_eligible_low": first_candidate[0] if first_candidate else None,
        "first_eligible_time": first_candidate[1] if first_candidate else None,
        "success_2x": event_2x is not None,
        "base_2x": event_2x["base"] if event_2x else None,
        "base_time_2x": event_2x["base_time"] if event_2x else None,
        "target_time_2x": event_2x["target_time"] if event_2x else None,
        "target_high_2x": event_2x["target_high"] if event_2x else None,
        "success_5x": event_5x is not None,
        "base_5x": event_5x["base"] if event_5x else None,
        "base_time_5x": event_5x["base_time"] if event_5x else None,
        "target_time_5x": event_5x["target_time"] if event_5x else None,
        "target_high_5x": event_5x["target_high"] if event_5x else None,
        "success_10x": event_10x is not None,
        "base_10x": event_10x["base"] if event_10x else None,
        "base_time_10x": event_10x["base_time"] if event_10x else None,
        "target_time_10x": event_10x["target_time"] if event_10x else None,
        "target_high_10x": event_10x["target_high"] if event_10x else None,
        "max_multiple": max_multiple,
        "max_multiple_time": max_multiple_time,
        "max_multiple_base": max_multiple_base,
        "max_multiple_base_time": max_multiple_base_time,
        "bar_count": len(ordered),
        "first_bar_time": ordered[0].time if ordered else None,
        "last_bar_time": ordered[-1].time if ordered else None,
        "total_volume": sum(bar.volume for bar in ordered),
    }
    return overall, []  # 밴드 상세는 corr 분석에 불필요


def main() -> None:
    # 원본 read_expiry_group 이 모듈 전역 analyze_bars/update_candidate 를 참조하므로 교체
    co.analyze_bars = analyze_bars
    co.update_candidate = update_candidate

    profile = json.loads(co.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = co.derive_expiry_groups(profile)
    files = co.target_file_map()

    opportunities: list[dict[str, object]] = []
    for group in groups:
        date = group["expiry_date"]
        if date not in files:
            raise FileNotFoundError(f"만기일 파일 없음: {date}")
        day_opps, _band, _diag = co.read_expiry_group(files[date], group)
        opportunities.extend(day_opps)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    co.write_csv(OUTPUT_DIR / "opportunities_lowband.csv", opportunities)

    n = len(opportunities)
    n5 = sum(bool(o["success_5x"]) for o in opportunities)
    n10 = sum(bool(o["success_10x"]) for o in opportunities)
    print(f"저가밴드(≥{PREMIUM_FLOOR}) 유효계약: {n:,}  5배: {n5:,}({n5/n*100:.1f}%)  "
          f"10배: {n10:,}({n10/n*100:.1f}%)")
    print("저장: output/opportunities_lowband.csv")


if __name__ == "__main__":
    main()
