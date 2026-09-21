from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
ROWDATA_DIR = BACKEND_DIR / "rowdata"
DATASET_NAMES = (
    "[파생 옵션 코스피200 위클리]일중 매매정보(1분)(주문번호-4831-1)",
    "[파생 옵션 코스피200 위클리]일중 매매정보(1분)(주문번호-4437-1)",
    "[파생 옵션 코스피200 위클리]일중 매매정보(1분)(주문번호-4571-1)",
    "[파생 옵션 코스피200 위클리]일중 매매정보(1분)(주문번호-4831-2)",
)
DATE_RE = re.compile(r"_(\d{8})\.csv$")


def collect_inventory() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    date_sources: dict[str, list[str]] = defaultdict(list)
    invalid_names: list[str] = []

    for dataset_name in DATASET_NAMES:
        dataset_dir = ROWDATA_DIR / dataset_name
        dates: list[str] = []

        for path in dataset_dir.glob("*.csv"):
            match = DATE_RE.search(path.name)
            if match is None:
                invalid_names.append(str(path))
                continue
            date = match.group(1)
            dates.append(date)
            date_sources[date].append(dataset_name)

        dates.sort()
        datasets.append(
            {
                "name": dataset_name,
                "file_count": len(dates),
                "min_date": dates[0] if dates else None,
                "max_date": dates[-1] if dates else None,
                "target_period_file_count": sum(date >= "20251001" for date in dates),
            }
        )

    target_dates = sorted(date for date in date_sources if date >= "20251001")
    duplicate_dates = {
        date: sources
        for date, sources in sorted(date_sources.items())
        if date >= "20251001" and len(sources) > 1
    }

    return {
        "datasets": datasets,
        "all_unique_date_count": len(date_sources),
        "target_min_date": target_dates[0] if target_dates else None,
        "target_max_date": target_dates[-1] if target_dates else None,
        "target_unique_date_count": len(target_dates),
        "target_duplicate_dates": duplicate_dates,
        "invalid_filenames": invalid_names,
    }


if __name__ == "__main__":
    print(json.dumps(collect_inventory(), ensure_ascii=False, indent=2))
