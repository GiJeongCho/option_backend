from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from inventory import DATASET_NAMES, DATE_RE, ROWDATA_DIR


START_DATE = "20251001"
DAY_MARKET_ID = "DRV"
PROFILE_PATH = Path(__file__).resolve().parent / "output" / "data_profile.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SERIES_RE = re.compile(r"\b(\d{4}W\d)\b")
PRICE_EPSILON = 1e-9

PREMIUM_BANDS: tuple[tuple[float, float, str], ...] = (
    (1.00, 1.50, "1.00-1.49"),
    (1.50, 2.00, "1.50-1.99"),
    (2.00, 2.50, "2.00-2.49"),
    (2.50, 3.00, "2.50-2.99"),
    (3.00, 3.50, "3.00-3.49"),
    (3.50, 4.00, "3.50-3.99"),
    (4.00, 4.50, "4.00-4.49"),
    (4.50, 5.01, "4.50-5.00"),
)


def product_kind(name: str) -> str:
    return "MONDAY" if "위클리M" in name else "THURSDAY"


def extract_series(name: str) -> str:
    match = SERIES_RE.search(name)
    if match is None:
        raise ValueError(f"주차 정보를 찾을 수 없는 종목명: {name}")
    return match.group(1)


def premium_band(price: float) -> str | None:
    for lower, upper, label in PREMIUM_BANDS:
        if lower <= price < upper:
            return label
    return None


def reaches_multiple(high: float, base: float, multiple: float) -> bool:
    return high + PRICE_EPSILON >= multiple * base


def target_file_map() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for dataset_name in DATASET_NAMES:
        for path in (ROWDATA_DIR / dataset_name).glob("*.csv"):
            match = DATE_RE.search(path.name)
            if match is None or match.group(1) < START_DATE:
                continue
            date = match.group(1)
            if date in result:
                raise ValueError(f"중복 거래일 파일: {date}")
            result[date] = path
    return result


def derive_expiry_groups(
    profile: dict[str, object],
) -> list[dict[str, str]]:
    cutoff = max(profile["per_date"])  # type: ignore[arg-type]
    group_expiry: dict[tuple[str, str], str] = {}

    for item in profile["codes"]:  # type: ignore[index]
        code_info = dict(item)
        kind = product_kind(str(code_info["name"]))
        series = str(code_info["series"])
        key = (kind, series)
        last_date = str(code_info["last_trade_date"])
        group_expiry[key] = max(group_expiry.get(key, ""), last_date)

    result: list[dict[str, str]] = []
    for (kind, series), expiry_date in sorted(
        group_expiry.items(), key=lambda item: item[1]
    ):
        if expiry_date < START_DATE:
            continue

        # 데이터 마지막 날에 아직 만기가 오지 않은 다음 달물은 제외한다.
        series_month = series[:4]
        cutoff_month = cutoff[2:6]
        if expiry_date == cutoff and series_month > cutoff_month:
            continue

        result.append(
            {
                "expiry_date": expiry_date,
                "product_kind": kind,
                "series": series,
            }
        )

    duplicate_keys = Counter(item["expiry_date"] for item in result)
    duplicates = [date for date, count in duplicate_keys.items() if count > 1]
    if duplicates:
        raise ValueError(f"같은 날짜에 중복 만기 그룹이 있습니다: {duplicates}")

    return result


@dataclass
class Bar:
    time: int
    high: float
    low: float
    volume: int


def merge_bar(existing: Bar | None, time: int, high: float, low: float, volume: int) -> Bar:
    if existing is None:
        return Bar(time=time, high=high, low=low, volume=volume)
    return Bar(
        time=time,
        high=max(existing.high, high),
        low=min(existing.low, low),
        volume=existing.volume + volume,
    )


def update_candidate(
    candidate: tuple[float, int] | None, low: float, time: int
) -> tuple[float, int] | None:
    if not 1.0 <= low <= 5.0:
        return candidate
    if candidate is None or low < candidate[0]:
        return (low, time)
    return candidate


def analyze_bars(bars: Iterable[Bar]) -> tuple[dict[str, object], list[dict[str, object]]]:
    ordered = sorted(bars, key=lambda bar: bar.time)
    candidate: tuple[float, int] | None = None
    first_candidate: tuple[float, int] | None = None
    event_5x: dict[str, object] | None = None
    event_10x: dict[str, object] | None = None
    max_multiple = 0.0
    max_multiple_time: int | None = None
    max_multiple_base: float | None = None
    max_multiple_base_time: int | None = None

    band_candidates: dict[str, tuple[float, int] | None] = {
        label: None for _, _, label in PREMIUM_BANDS
    }
    band_results: dict[str, dict[str, object]] = {
        label: {
            "eligible": False,
            "success_5x": False,
            "success_10x": False,
            "base_5x": None,
            "base_time_5x": None,
            "target_time_5x": None,
            "base_10x": None,
            "base_time_10x": None,
            "target_time_10x": None,
        }
        for _, _, label in PREMIUM_BANDS
    }

    for bar in ordered:
        # 현재 봉의 저가는 현재 봉의 고가보다 먼저 발생했다고 가정하지 않는다.
        # 따라서 목표 판정은 이전 봉까지 만들어진 후보 저가로만 수행한다.
        if candidate is not None:
            base, base_time = candidate
            multiple = bar.high / base
            if multiple > max_multiple:
                max_multiple = multiple
                max_multiple_time = bar.time
                max_multiple_base = base
                max_multiple_base_time = base_time

            if event_5x is None and reaches_multiple(bar.high, base, 5.0):
                event_5x = {
                    "base": base,
                    "base_time": base_time,
                    "target_time": bar.time,
                    "target_high": bar.high,
                }
            if event_10x is None and reaches_multiple(bar.high, base, 10.0):
                event_10x = {
                    "base": base,
                    "base_time": base_time,
                    "target_time": bar.time,
                    "target_high": bar.high,
                }

        for _, _, label in PREMIUM_BANDS:
            band_candidate = band_candidates[label]
            if band_candidate is None:
                continue
            base, base_time = band_candidate
            multiple = bar.high / base
            result = band_results[label]
            if not result["success_5x"] and reaches_multiple(bar.high, base, 5.0):
                result.update(
                    {
                        "success_5x": True,
                        "base_5x": base,
                        "base_time_5x": base_time,
                        "target_time_5x": bar.time,
                    }
                )
            if not result["success_10x"] and reaches_multiple(
                bar.high, base, 10.0
            ):
                result.update(
                    {
                        "success_10x": True,
                        "base_10x": base,
                        "base_time_10x": base_time,
                        "target_time_10x": bar.time,
                    }
                )

        if 1.0 <= bar.low <= 5.0:
            if first_candidate is None:
                first_candidate = (bar.low, bar.time)
            candidate = update_candidate(candidate, bar.low, bar.time)

            label = premium_band(bar.low)
            if label is not None:
                band_results[label]["eligible"] = True
                current = band_candidates[label]
                if current is None or bar.low < current[0]:
                    band_candidates[label] = (bar.low, bar.time)

    overall = {
        "eligible": first_candidate is not None,
        "first_eligible_low": first_candidate[0] if first_candidate else None,
        "first_eligible_time": first_candidate[1] if first_candidate else None,
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
    return overall, [
        {"premium_band": label, **result} for label, result in band_results.items()
    ]


def read_expiry_group(
    path: Path, group: dict[str, str]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    bars_by_code: dict[str, dict[int, Bar]] = defaultdict(dict)
    metadata: dict[str, dict[str, str]] = {}
    duplicate_bar_rows = 0
    invalid_rows = 0

    with path.open("r", encoding="cp949", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if row["시장ID"] != DAY_MARKET_ID:
                continue
            if product_kind(row["종목명"]) != group["product_kind"]:
                continue
            if extract_series(row["종목명"]) != group["series"]:
                continue

            try:
                time = int(row["기준시각"].strip().zfill(4))
                high = float(row["고가"])
                low = float(row["저가"])
                volume = int(float(row["거래량"]))
            except (TypeError, ValueError):
                invalid_rows += 1
                continue

            code = row["종목코드"]
            if time in bars_by_code[code]:
                duplicate_bar_rows += 1
            bars_by_code[code][time] = merge_bar(
                bars_by_code[code].get(time), time, high, low, volume
            )
            metadata.setdefault(
                code,
                {
                    "option_code": code,
                    "option_name": row["종목명"],
                    "call_put": row["콜풋구분"],
                    "strike": row["행사가격"],
                },
            )

    opportunities: list[dict[str, object]] = []
    band_details: list[dict[str, object]] = []
    for code, bars_by_time in sorted(bars_by_code.items()):
        overall, bands = analyze_bars(bars_by_time.values())
        base = {
            "expiry_date": group["expiry_date"],
            "product_kind": group["product_kind"],
            "series": group["series"],
            **metadata[code],
        }
        if overall["eligible"]:
            opportunities.append({**base, **overall})
        for band in bands:
            if band["eligible"]:
                band_details.append({**base, **band})

    diagnostic = {
        "expiry_date": group["expiry_date"],
        "product_kind": group["product_kind"],
        "series": group["series"],
        "contract_count": len(bars_by_code),
        "eligible_contract_count": len(opportunities),
        "success_5x_count": sum(bool(item["success_5x"]) for item in opportunities),
        "success_10x_count": sum(bool(item["success_10x"]) for item in opportunities),
        "duplicate_bar_rows": duplicate_bar_rows,
        "invalid_rows": invalid_rows,
        "min_time": min(
            (bar.time for values in bars_by_code.values() for bar in values.values()),
            default=None,
        ),
        "max_time": max(
            (bar.time for values in bars_by_code.values() for bar in values.values()),
            default=None,
        ),
    }
    return opportunities, band_details, diagnostic


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    groups: list[dict[str, str]],
    opportunities: list[dict[str, object]],
    band_details: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    success_5x = [item for item in opportunities if item["success_5x"]]
    success_10x = [item for item in opportunities if item["success_10x"]]
    expiry_5x = {str(item["expiry_date"]) for item in success_5x}
    expiry_10x = {str(item["expiry_date"]) for item in success_10x}

    def percent(numerator: int, denominator: int) -> float:
        return round(numerator / denominator * 100, 4) if denominator else 0.0

    band_summary: list[dict[str, object]] = []
    for _, _, label in PREMIUM_BANDS:
        eligible = [item for item in band_details if item["premium_band"] == label]
        band_5x = [item for item in eligible if item["success_5x"]]
        band_10x = [item for item in eligible if item["success_10x"]]
        band_summary.append(
            {
                "premium_band": label,
                "eligible_contract_days": len(eligible),
                "success_5x_contract_days": len(band_5x),
                "success_5x_rate_pct": percent(len(band_5x), len(eligible)),
                "success_10x_contract_days": len(band_10x),
                "success_10x_rate_pct": percent(len(band_10x), len(eligible)),
                "expiry_days_with_5x": len(
                    {str(item["expiry_date"]) for item in band_5x}
                ),
                "expiry_days_with_10x": len(
                    {str(item["expiry_date"]) for item in band_10x}
                ),
            }
        )

    call_put_summary: list[dict[str, object]] = []
    for call_put in sorted({str(item["call_put"]) for item in opportunities}):
        eligible = [item for item in opportunities if item["call_put"] == call_put]
        count_5x = sum(bool(item["success_5x"]) for item in eligible)
        count_10x = sum(bool(item["success_10x"]) for item in eligible)
        call_put_summary.append(
            {
                "call_put": call_put,
                "eligible_contract_days": len(eligible),
                "success_5x_contract_days": count_5x,
                "success_5x_rate_pct": percent(count_5x, len(eligible)),
                "success_10x_contract_days": count_10x,
                "success_10x_rate_pct": percent(count_10x, len(eligible)),
            }
        )

    return {
        "definition": {
            "period_start": START_DATE,
            "period_end": max(item["expiry_date"] for item in groups),
            "market_id": DAY_MARKET_ID,
            "premium_min": 1.0,
            "premium_max": 5.0,
            "same_bar_low_high_allowed": False,
            "primary_unit": "expiry_date + option_code",
        },
        "expiry_group_count": len(groups),
        "eligible_contract_days": len(opportunities),
        "success_5x_contract_days": len(success_5x),
        "success_5x_contract_rate_pct": percent(len(success_5x), len(opportunities)),
        "success_10x_contract_days": len(success_10x),
        "success_10x_contract_rate_pct": percent(len(success_10x), len(opportunities)),
        "expiry_days_with_5x": len(expiry_5x),
        "expiry_days_with_5x_rate_pct": percent(len(expiry_5x), len(groups)),
        "expiry_days_with_10x": len(expiry_10x),
        "expiry_days_with_10x_rate_pct": percent(len(expiry_10x), len(groups)),
        "expiry_days_without_5x": sorted(
            {item["expiry_date"] for item in groups} - expiry_5x
        ),
        "expiry_days_without_10x": sorted(
            {item["expiry_date"] for item in groups} - expiry_10x
        ),
        "premium_band_summary": band_summary,
        "call_put_summary": call_put_summary,
        "diagnostics": {
            "duplicate_bar_rows": sum(
                int(item["duplicate_bar_rows"]) for item in diagnostics
            ),
            "invalid_rows": sum(int(item["invalid_rows"]) for item in diagnostics),
            "min_time": min(
                int(item["min_time"])
                for item in diagnostics
                if item["min_time"] is not None
            ),
            "max_time": max(
                int(item["max_time"])
                for item in diagnostics
                if item["max_time"] is not None
            ),
        },
    }


def main() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = target_file_map()

    opportunities: list[dict[str, object]] = []
    band_details: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for group in groups:
        date = group["expiry_date"]
        if date not in files:
            raise FileNotFoundError(f"만기일 파일 없음: {date}")
        day_opportunities, day_bands, diagnostic = read_expiry_group(
            files[date], group
        )
        opportunities.extend(day_opportunities)
        band_details.extend(day_bands)
        diagnostics.append(diagnostic)

    summary = aggregate(groups, opportunities, band_details, diagnostics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "opportunities.csv", opportunities)
    write_csv(OUTPUT_DIR / "premium_band_details.csv", band_details)
    write_csv(OUTPUT_DIR / "expiry_diagnostics.csv", diagnostics)
    (OUTPUT_DIR / "expiry_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "opportunity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
