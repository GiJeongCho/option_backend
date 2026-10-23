from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
ROWDATA_DIR = BACKEND_DIR / "rowdata"
DATASET_MARKER = "[파생 옵션 코스피200 위클리]일중 매매정보(1분)"
DATE_RE = re.compile(r"_(\d{8})\.csv$")
ANALYSIS_START_DATE = "20230801"


def dataset_directories(
    rowdata_dir: Path = ROWDATA_DIR,
) -> tuple[Path, ...]:
    if not rowdata_dir.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in rowdata_dir.iterdir()
                if path.is_dir() and DATASET_MARKER in path.name
            ),
            key=lambda path: path.name,
        )
    )


def option_file_index(
    rowdata_dir: Path = ROWDATA_DIR,
    start_date: str = ANALYSIS_START_DATE,
) -> dict[str, Path]:
    by_date: dict[str, list[Path]] = defaultdict(list)
    for dataset_dir in dataset_directories(rowdata_dir):
        for path in dataset_dir.glob("*.csv"):
            match = DATE_RE.search(path.name)
            if match is None or match.group(1) < start_date:
                continue
            by_date[match.group(1)].append(path)

    duplicates = {
        date: paths
        for date, paths in sorted(by_date.items())
        if len(paths) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{date}: {', '.join(path.parent.name for path in paths)}"
            for date, paths in duplicates.items()
        )
        raise ValueError(f"중복 거래일 옵션 파일: {detail}")
    return {
        date: paths[0]
        for date, paths in sorted(by_date.items())
    }


DATASET_NAMES = tuple(path.name for path in dataset_directories())


def collect_inventory() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    date_sources: dict[str, list[str]] = defaultdict(list)
    invalid_names: list[str] = []

    for dataset_dir in dataset_directories():
        dataset_name = dataset_dir.name
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
                "target_period_file_count": sum(
                    date >= ANALYSIS_START_DATE for date in dates
                ),
            }
        )

    target_dates = sorted(
        date for date in date_sources if date >= ANALYSIS_START_DATE
    )
    duplicate_dates = {
        date: sources
        for date, sources in sorted(date_sources.items())
        if date >= ANALYSIS_START_DATE and len(sources) > 1
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
