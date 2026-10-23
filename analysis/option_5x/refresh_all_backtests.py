from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from count_opportunities import derive_expiry_groups
from inventory import dataset_directories, option_file_index


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
LOG_DIR = OUTPUT_DIR / "refresh_logs"
PROFILE_PATH = OUTPUT_DIR / "data_profile.json"
SUMMARY_PATH = OUTPUT_DIR / "refresh_all_backtests_summary.json"

STEPS = (
    "profile_data.py",
    "count_opportunities.py",
    "verify_opportunities.py",
    "minimal_baseline_backtest.py",
    "audit_minimal_baseline.py",
    "strategy1_backtest.py",
    "analyze_strategy1_patterns.py",
    "strategy2_backtest.py",
    "strategy2_v02_backtest.py",
    "strategy2_v03_backtest.py",
    "strategy3_research.py",
    "strategy3_v02_backtest.py",
    "strategy3_v03_backtest.py",
    "strategy3_v04_research.py",
    "strategy3_v04_backtest.py",
    "audit_strategy3_v04.py",
    "strategy4_lsma_futures_backtest.py",
    "strategy5_ppt_momentum_backtest.py",
    "strategy5_ppt_momentum_research.py",
    "strategy5_ppt_confirmation_backtest.py",
    "audit_strategy5_ppt_momentum.py",
    "strategy5_v03_event_research.py",
    "strategy5_v03_dual_confirmation_backtest.py",
    "audit_strategy5_v03.py",
    "strategy5_v04_basket_backtest.py",
    "audit_strategy5_v04.py",
    "strategy5_v05_no_reentry_backtest.py",
    "audit_strategy5_v05.py",
    "strategy5_v06_ten_x_backtest.py",
    "audit_strategy5_v06_ten_x.py",
    "strategy5_v07_04_bin_ev_backtest.py",
    "audit_strategy5_v07_04_bin_ev.py",
    "strategy5_v07_04_bin_ten_x_backtest.py",
    "audit_strategy5_v07_04_bin_ten_x.py",
    "strategy5_v08_step10_ten_x_backtest.py",
    "audit_strategy5_v08_step10_ten_x.py",
    "strategy5_v09_strike10_ten_x_backtest.py",
    "audit_strategy5_v09_strike10_ten_x.py",
    "strategy6_v01_vix_rv_put_backtest.py",
    "audit_strategy6_v01.py",
    "strategy6_v02_vix_ma5_rising_backtest.py",
    "audit_strategy6_v02.py",
    "strategy6_v03_rv60_runner_backtest.py",
)

STRATEGY_DAILY_FILES = (
    "strategy1_daily.csv",
    "strategy2_daily.csv",
    "strategy2_v02_daily.csv",
    "strategy2_v03_daily.csv",
    "strategy3_daily.csv",
    "strategy3_v02_daily.csv",
    "strategy3_v03_daily.csv",
    "strategy3_v04_daily.csv",
    "strategy4_v01_daily.csv",
    "strategy4_daily.csv",
    "strategy5_daily.csv",
    "strategy5_v03_early_daily.csv",
    "strategy5_v03_late_daily.csv",
    "strategy5_v04_ev_daily.csv",
    "strategy5_v04_staged_daily.csv",
    "strategy5_v05_daily.csv",
    "strategy5_v06_ten_x_daily.csv",
    "strategy5_v07_04_bin_ev_daily.csv",
    "strategy5_v07_04_bin_ten_x_daily.csv",
    "strategy5_v08_step10_ten_x_daily.csv",
    "strategy5_v09_strike10_ten_x_daily.csv",
    "strategy6_v01_daily.csv",
    "strategy6_v02_daily.csv",
    "strategy6_v03_rv60_runner_daily.csv",
)


@dataclass(frozen=True)
class StepResult:
    script: str
    elapsed_seconds: float
    log: str


def run_step(script: str) -> StepResult:
    path = ROOT / script
    if not path.exists():
        raise FileNotFoundError(path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{path.stem}.log"
    started = time.perf_counter()
    print(f"[START] {script}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"{script} 실패(exit={completed.returncode}): {log_path}"
        )
    print(f"[ OK  ] {script} ({elapsed:.1f}s)", flush=True)
    return StepResult(
        script=script,
        elapsed_seconds=round(elapsed, 3),
        log=str(log_path),
    )


def maximum_date(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        dates = {
            str(row.get("date", ""))
            for row in csv.DictReader(source)
            if row.get("date")
        }
    if not dates:
        raise RuntimeError(f"날짜가 없는 결과 파일: {path}")
    return max(dates)


def verify_outputs(latest_expiry: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in STRATEGY_DAILY_FILES:
        path = OUTPUT_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = maximum_date(path)
    stale = {
        name: date
        for name, date in result.items()
        if date != latest_expiry
    }
    if stale:
        raise RuntimeError(
            f"최신 만기일 {latest_expiry} 미반영 결과: {stale}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="전체 옵션 원자료 프로파일과 저장 전략을 재생성합니다."
    )
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        help="지정한 스크립트부터 이어서 실행합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(STEPS)
    if args.from_step:
        selected = selected[selected.index(args.from_step) :]

    started = time.perf_counter()
    results = [run_step(script) for script in selected]
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    if not groups:
        raise RuntimeError("완료된 옵션 만기 그룹이 없습니다.")
    latest_expiry = groups[-1]["expiry_date"]
    output_dates = verify_outputs(latest_expiry)
    raw_files = option_file_index()
    document = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_directories": [
            path.name for path in dataset_directories()
        ],
        "raw_file_count": len(raw_files),
        "raw_min_date": min(raw_files),
        "raw_max_date": max(raw_files),
        "expiry_group_count": len(groups),
        "latest_completed_expiry": latest_expiry,
        "strategy_count": len(output_dates),
        "strategy_latest_dates": output_dates,
        "steps": [asdict(result) for result in results],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    SUMMARY_PATH.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
