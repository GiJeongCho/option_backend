from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from inventory import DATASET_NAMES, DATE_RE, ROWDATA_DIR


START_DATE = "20251001"
SERIES_RE = re.compile(r"\b(\d{4}W\d)\b")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def time_bucket(value: str) -> str:
    normalized = value.strip().zfill(4)
    if not normalized.isdigit() or len(normalized) != 4:
        return "invalid"

    hhmm = int(normalized)
    if hhmm < 800:
        return "night_morning"
    if hhmm < 1800:
        return "day_candidate"
    if hhmm < 2400:
        return "night_evening"
    return "invalid"


def target_files() -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for dataset_name in DATASET_NAMES:
        for path in (ROWDATA_DIR / dataset_name).glob("*.csv"):
            match = DATE_RE.search(path.name)
            if match and match.group(1) >= START_DATE:
                result.append((match.group(1), path))
    return sorted(result)


def profile() -> dict[str, object]:
    per_date: dict[str, dict[str, object]] = {}
    series_dates: dict[str, set[str]] = defaultdict(set)
    series_rows: Counter[str] = Counter()
    market_ids: Counter[str] = Counter()
    code_info: dict[str, dict[str, object]] = {}
    market_time_ranges: dict[str, list[str | None]] = defaultdict(lambda: [None, None])
    invalid_series_rows = 0

    for file_date, path in target_files():
        buckets: Counter[str] = Counter()
        times: list[str] = []
        date_series: set[str] = set()
        row_count = 0

        with path.open("r", encoding="cp949", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                row_count += 1
                raw_time = row["기준시각"].strip().zfill(4)
                buckets[time_bucket(raw_time)] += 1
                if raw_time.isdigit() and len(raw_time) == 4:
                    times.append(raw_time)

                market_id = row["시장ID"]
                market_ids[market_id] += 1
                if raw_time.isdigit() and len(raw_time) == 4:
                    current_min, current_max = market_time_ranges[market_id]
                    market_time_ranges[market_id][0] = (
                        raw_time
                        if current_min is None
                        else min(current_min, raw_time)
                    )
                    market_time_ranges[market_id][1] = (
                        raw_time
                        if current_max is None
                        else max(current_max, raw_time)
                    )

                match = SERIES_RE.search(row["종목명"])
                if match is None:
                    invalid_series_rows += 1
                    continue

                series = match.group(1)
                series_dates[series].add(file_date)
                series_rows[series] += 1
                date_series.add(series)

                if market_id == "DRV":
                    code = row["종목코드"]
                    if code not in code_info:
                        code_info[code] = {
                            "code": code,
                            "name": row["종목명"],
                            "series": series,
                            "call_put": row["콜풋구분"],
                            "strike": row["행사가격"],
                            "first_trade_date": file_date,
                            "last_trade_date": file_date,
                            "row_count": 1,
                        }
                    else:
                        info = code_info[code]
                        info["first_trade_date"] = min(
                            str(info["first_trade_date"]), file_date
                        )
                        info["last_trade_date"] = max(
                            str(info["last_trade_date"]), file_date
                        )
                        info["row_count"] = int(info["row_count"]) + 1

        per_date[file_date] = {
            "file": str(path),
            "row_count": row_count,
            "min_time": min(times) if times else None,
            "max_time": max(times) if times else None,
            "time_buckets": dict(sorted(buckets.items())),
            "series": sorted(date_series),
        }

    series_summary: list[dict[str, object]] = []
    for series, dates in sorted(series_dates.items()):
        sorted_dates = sorted(dates)
        last_date = sorted_dates[-1]
        series_summary.append(
            {
                "series": series,
                "first_trade_date": sorted_dates[0],
                "last_trade_date": last_date,
                "last_trade_weekday": datetime.strptime(last_date, "%Y%m%d").strftime(
                    "%A"
                ),
                "observed_trade_days": len(sorted_dates),
                "row_count": series_rows[series],
            }
        )

    total_buckets: Counter[str] = Counter()
    for summary in per_date.values():
        total_buckets.update(summary["time_buckets"])  # type: ignore[arg-type]

    code_last_date_counts = Counter(
        str(info["last_trade_date"]) for info in code_info.values()
    )

    return {
        "start_date": START_DATE,
        "file_count": len(per_date),
        "row_count": sum(int(item["row_count"]) for item in per_date.values()),
        "market_ids": dict(market_ids),
        "market_time_ranges": {
            market_id: {"min_time": values[0], "max_time": values[1]}
            for market_id, values in sorted(market_time_ranges.items())
        },
        "time_buckets": dict(sorted(total_buckets.items())),
        "invalid_series_rows": invalid_series_rows,
        "series": series_summary,
        "code_count": len(code_info),
        "code_last_date_counts": dict(sorted(code_last_date_counts.items())),
        "codes": sorted(code_info.values(), key=lambda item: str(item["code"])),
        "per_date": per_date,
    }


if __name__ == "__main__":
    result = profile()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "data_profile.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_count": result["file_count"],
                "row_count": result["row_count"],
                "market_ids": result["market_ids"],
                "market_time_ranges": result["market_time_ranges"],
                "time_buckets": result["time_buckets"],
                "invalid_series_rows": result["invalid_series_rows"],
                "series": result["series"],
                "code_count": result["code_count"],
                "code_last_date_counts": result["code_last_date_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
