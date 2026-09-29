from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

import minimal_baseline_backtest as base
import strategy3_research as s3
import strategy3_v02_backtest as s3_parent
import strategy3_v03_backtest as s3_v03
from futures_lsma_research import add_indicators, load_daily, weekly_like_source
from strategy1_backtest import INITIAL_CASH, previous_volatility, volatility_closes
from strategy2_backtest import audit


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PAPER_STRATEGY_VERSION = "S4-v0.1"
STRATEGY_VERSION = "S4-v0.2"
HIGH_VOL_START = "20251001"
DEVELOPMENT_END = "20260430"
SELECTED_S3_ENTRY = "L60_R20_B5_MA20"

BaseStrategy = Literal["S3-v0.1", "S3-v0.3"]
DirectionMode = Literal[
    "baseline",
    "toward_gate",
    "toward_filter",
    "away_filter",
    "slope_filter",
    "consensus_filter",
]


@dataclass(frozen=True)
class FuturesDirection:
    date: str
    open: float
    shifted_lsma: float
    distance_pct: float
    reference_week: str
    latest_input_week: str
    toward_direction: str
    away_direction: str
    slope_direction: str
    consensus_direction: str


@dataclass
class State:
    key: str
    base_strategy: BaseStrategy
    direction_mode: DirectionMode
    cash: int = INITIAL_CASH
    daily: list[object] | None = None
    trades: list[dict[str, object]] | None = None
    unfiltered_signal_days: int = 0
    direction_rejected_days: int = 0

    def __post_init__(self) -> None:
        if self.daily is None:
            self.daily = []
        if self.trades is None:
            self.trades = []


def direction_map(option_dates: list[str]) -> dict[str, FuturesDirection]:
    daily = load_daily()
    weekly = add_indicators(weekly_like_source(daily))
    weekly["shifted_lsma"] = weekly["LSMA"].shift(1)
    result: dict[str, FuturesDirection] = {}
    for date_value in option_dates:
        date = pd.Timestamp(
            f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:]}"
        )
        day = daily.loc[daily.time == date]
        references = weekly.loc[
            (weekly.time <= date) & weekly.shifted_lsma.notna()
        ]
        if day.empty or references.empty:
            continue
        reference_index = int(references.index[-1])
        reference = weekly.loc[reference_index]
        if reference_index < 2:
            continue
        latest_input = weekly.loc[reference_index - 1, "time"]
        previous_shifted = weekly.loc[
            reference_index - 1, "shifted_lsma"
        ]
        if pd.isna(previous_shifted):
            continue
        open_price = float(day.iloc[0].오픈)
        lsma = float(reference.shifted_lsma)
        toward = "CALL" if open_price < lsma else "PUT"
        away = "PUT" if toward == "CALL" else "CALL"
        slope = (
            "CALL" if lsma > float(previous_shifted) else "PUT"
        )
        consensus = toward if toward == slope else "NONE"
        if latest_input >= date:
            raise RuntimeError(
                f"{date_value}: LSMA 최신 입력일 누출 {latest_input}"
            )
        result[date_value] = FuturesDirection(
            date=date_value,
            open=round(open_price, 6),
            shifted_lsma=round(lsma, 6),
            distance_pct=round(
                abs(open_price - lsma) / open_price * 100, 6
            ),
            reference_week=reference.time.strftime("%Y-%m-%d"),
            latest_input_week=latest_input.strftime("%Y-%m-%d"),
            toward_direction=toward,
            away_direction=away,
            slope_direction=slope,
            consensus_direction=consensus,
        )
    return result


def allowed_direction(
    signal: FuturesDirection | None, mode: DirectionMode
) -> str:
    if mode == "baseline":
        return "ALL"
    if signal is None:
        return "NONE"
    if mode in {"toward_gate", "toward_filter"}:
        return signal.toward_direction
    if mode == "away_filter":
        return signal.away_direction
    if mode == "slope_filter":
        return signal.slope_direction
    if mode == "consensus_filter":
        return signal.consensus_direction
    return "ALL"


def filtered_contracts(
    contracts: dict[str, base.ContractGrid], direction: str
) -> dict[str, base.ContractGrid]:
    if direction == "ALL":
        return contracts
    if direction == "NONE":
        return {}
    return {
        code: contract
        for code, contract in contracts.items()
        if contract.call_put == direction
    }


def selected_s3_entry() -> s3.EntryConfig:
    return next(
        config
        for config in s3.ENTRY_CONFIGS
        if config.name == SELECTED_S3_ENTRY
    )


def simulate_s3_v01(
    state: State,
    date_value: str,
    previous_vix: float,
    contracts: dict[str, base.ContractGrid],
    futures: FuturesDirection | None,
) -> None:
    assert state.daily is not None
    assert state.trades is not None
    entry = selected_s3_entry()
    unfiltered = s3.first_signal(contracts, entry, previous_vix)
    state.unfiltered_signal_days += unfiltered is not None
    direction = allowed_direction(futures, state.direction_mode)
    if state.direction_mode == "toward_gate":
        signal = (
            unfiltered
            if unfiltered is not None
            and futures is not None
            and unfiltered.code in contracts
            and contracts[unfiltered.code].call_put == direction
            else None
        )
    else:
        signal = s3.first_signal(
            filtered_contracts(contracts, direction),
            entry,
            previous_vix,
        )
    if unfiltered is not None and signal is None:
        state.direction_rejected_days += 1
    runtime = s3.RuntimeConfig(state.key, "two_x_all")
    row, trades = s3.simulate_day(
        date_value,
        previous_vix,
        runtime,
        contracts,
        signal,
        state.cash,
    )
    state.cash = row.end_cash
    state.daily.append(row)
    state.trades.extend(trades)


def simulate_s3_v03(
    state: State,
    date_value: str,
    previous_vix: float,
    contracts: dict[str, base.ContractGrid],
    futures: FuturesDirection | None,
) -> None:
    assert state.daily is not None
    assert state.trades is not None
    unfiltered = s3_parent.first_setup(contracts)
    state.unfiltered_signal_days += unfiltered is not None
    direction = allowed_direction(futures, state.direction_mode)
    if state.direction_mode == "toward_gate":
        setup = (
            unfiltered
            if unfiltered is not None
            and futures is not None
            and contracts[unfiltered.code].call_put == direction
            else None
        )
    else:
        setup = s3_parent.first_setup(
            filtered_contracts(contracts, direction)
        )
    if unfiltered is not None and setup is None:
        state.direction_rejected_days += 1
    config = s3_v03.Config(
        state.key,
        s3_v03.FIXED_CONFIG.entry_mode,
        s3_v03.FIXED_CONFIG.exit_mode,
    )
    row, trades = s3_v03.simulate_day(
        date_value,
        previous_vix,
        config,
        contracts,
        setup,
        state.cash,
    )
    state.cash = row.end_cash
    state.daily.append(row)
    state.trades.extend(trades)


def summarize(state: State) -> dict[str, object]:
    assert state.daily is not None
    if state.base_strategy == "S3-v0.1":
        periods = s3.periods(state.daily)  # type: ignore[arg-type]
    else:
        periods = {
            "all": s3_v03.summarize(state.daily),  # type: ignore[arg-type]
            "development": s3_v03.summarize(  # type: ignore[arg-type]
                [
                    row
                    for row in state.daily
                    if str(getattr(row, "date")) <= DEVELOPMENT_END
                ]
            ),
            "validation": s3_v03.summarize(  # type: ignore[arg-type]
                [
                    row
                    for row in state.daily
                    if str(getattr(row, "date")) > DEVELOPMENT_END
                ]
            ),
        }
    return {
        "strategy": state.key,
        "base_strategy": state.base_strategy,
        "direction_mode": state.direction_mode,
        "unfiltered_signal_days": state.unfiltered_signal_days,
        "direction_rejected_days": state.direction_rejected_days,
        **periods,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def enriched_daily_rows(
    state: State, directions: dict[str, FuturesDirection]
) -> list[dict[str, object]]:
    assert state.daily is not None
    rows: list[dict[str, object]] = []
    for row in state.daily:
        values = asdict(row)  # type: ignore[arg-type]
        futures = directions.get(str(values["date"]))
        values.update(
            {
                "futures_open": futures.open if futures else "",
                "shifted_lsma": (
                    futures.shifted_lsma if futures else ""
                ),
                "lsma_distance_pct": (
                    futures.distance_pct if futures else ""
                ),
                "lsma_reference_week": (
                    futures.reference_week if futures else ""
                ),
                "lsma_latest_input_week": (
                    futures.latest_input_week if futures else ""
                ),
                "lsma_toward_direction": (
                    futures.toward_direction if futures else "MISSING"
                ),
                "lsma_slope_direction": (
                    futures.slope_direction if futures else "MISSING"
                ),
            }
        )
        rows.append(values)
    return rows


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if group["expiry_date"] >= HIGH_VOL_START
    ]
    option_dates = [str(group["expiry_date"]) for group in groups]
    directions = direction_map(option_dates)
    missing = sorted(set(option_dates) - set(directions))

    modes: tuple[DirectionMode, ...] = (
        "baseline",
        "toward_gate",
        "toward_filter",
        "away_filter",
        "slope_filter",
        "consensus_filter",
    )
    states: list[State] = []
    for base_strategy in ("S3-v0.1", "S3-v0.3"):
        for mode in modes:
            key = (
                PAPER_STRATEGY_VERSION
                if base_strategy == "S3-v0.1"
                and mode == "toward_gate"
                else STRATEGY_VERSION
                if base_strategy == "S3-v0.3"
                and mode == "away_filter"
                else f"S4-research|{base_strategy}|{mode}"
            )
            states.append(State(key, base_strategy, mode))

    files = base.target_file_map()
    volatility_dates, volatility_values = volatility_closes()
    alignment_rows: list[dict[str, object]] = []
    for group in groups:
        date_value = str(group["expiry_date"])
        previous_vix = previous_volatility(
            date_value, volatility_dates, volatility_values
        )
        if previous_vix is None:
            raise RuntimeError(f"{date_value}: 직전 변동성지수 없음")
        contracts = base.load_expiry_day(files[date_value], group)
        futures = directions.get(date_value)
        alignment_rows.append(
            asdict(futures)
            if futures is not None
            else {
                "date": date_value,
                "open": "",
                "shifted_lsma": "",
                "distance_pct": "",
                "reference_week": "",
                "latest_input_week": "",
                "toward_direction": "MISSING",
                "away_direction": "MISSING",
                "slope_direction": "MISSING",
                "consensus_direction": "MISSING",
            }
        )
        for state in states:
            if state.base_strategy == "S3-v0.1":
                simulate_s3_v01(
                    state,
                    date_value,
                    previous_vix,
                    contracts,
                    futures,
                )
            else:
                simulate_s3_v03(
                    state,
                    date_value,
                    previous_vix,
                    contracts,
                    futures,
                )

    candidates = [summarize(state) for state in states]
    audits = {
        state.key: audit(
            state.daily,  # type: ignore[arg-type]
            state.trades or [],
        )
        for state in states
    }
    fixed = next(state for state in states if state.key == STRATEGY_VERSION)
    fixed_document = next(
        row for row in candidates if row["strategy"] == STRATEGY_VERSION
    )
    paper = next(
        state for state in states if state.key == PAPER_STRATEGY_VERSION
    )
    paper_document = next(
        row
        for row in candidates
        if row["strategy"] == PAPER_STRATEGY_VERSION
    )
    s3_v01_summary = json.loads(
        (OUTPUT_DIR / "strategy3_summary.json").read_text(encoding="utf-8")
    )
    s3_v03_summary = json.loads(
        (OUTPUT_DIR / "strategy3_v03_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_v01 = next(
        row
        for row in candidates
        if row["base_strategy"] == "S3-v0.1"
        and row["direction_mode"] == "baseline"
    )
    baseline_v03 = next(
        row
        for row in candidates
        if row["base_strategy"] == "S3-v0.3"
        and row["direction_mode"] == "baseline"
    )
    if (
        int(baseline_v01["all"]["total_pnl"])  # type: ignore[index]
        != int(s3_v01_summary["summary"]["total_pnl"])
        or int(baseline_v03["all"]["total_pnl"])  # type: ignore[index]
        != int(s3_v03_summary["summary"]["total_pnl"])
    ):
        raise RuntimeError("부모 전략 기준선 재현 실패")
    development_choice = max(
        (
            row
            for row in candidates
            if row["base_strategy"] == "S3-v0.3"
            and row["direction_mode"] != "baseline"
        ),
        key=lambda row: (
            int(row["development"]["total_pnl"]),  # type: ignore[index]
            int(
                row["development"]["pnl_excluding_best_3_days"]  # type: ignore[index]
            ),
            int(row["development"]["max_drawdown"]),  # type: ignore[index]
        ),
    )
    if development_choice["strategy"] != STRATEGY_VERSION:
        raise RuntimeError(
            f"개발 구간 선택 불일치: {development_choice['strategy']}"
        )

    direction_counts = {
        direction: sum(
            row.toward_direction == direction
            for row in directions.values()
        )
        for direction in ("CALL", "PUT")
    }
    fixed_summary = dict(fixed_document["all"])  # type: ignore[arg-type]
    fixed_summary.update(
        {
            "initial_cash": INITIAL_CASH,
            "ending_cash": fixed.cash,
        }
    )
    result = {
        "status": "experimental_development_selected",
        "strategy_version": STRATEGY_VERSION,
        "base_strategy": "S3-v0.3",
        "scope": {
            "start": option_dates[0],
            "end": option_dates[-1],
            "expiry_days": len(option_dates),
            "futures_aligned_days": len(directions),
            "futures_missing_dates": missing,
            "development_end": DEVELOPMENT_END,
        },
        "hypothesis": {
            "source": "shifted weekly LSMA trend-alignment candidate",
            "rule": (
                "futures open above shifted weekly LSMA permits CALL; "
                "open below LSMA permits PUT"
            ),
            "application": (
                "filter the S3-v0.3 option universe before selecting its "
                "first setup"
            ),
            "selection": (
                "highest development total PnL among predefined S3-v0.3 "
                "LSMA direction candidates; validation not used"
            ),
            "missing_same_day_futures_action": "do not trade",
            "current_day_futures_fields": ["open"],
            "latest_input_strictly_before_option_date": True,
        },
        "direction_counts": direction_counts,
        "summary": fixed_summary,
        "development": fixed_document["development"],
        "validation": fixed_document["validation"],
        "paper_hypothesis_test": {
            "strategy_version": PAPER_STRATEGY_VERSION,
            "base_strategy": "S3-v0.1",
            "rule": (
                "futures open below shifted weekly LSMA permits CALL; "
                "open above LSMA permits PUT"
            ),
            "all": paper_document["all"],
            "development": paper_document["development"],
            "validation": paper_document["validation"],
        },
        "candidates": candidates,
        "comparison": {
            "to_s3_v03_total_pnl": (
                int(fixed_summary["total_pnl"])
                - int(s3_v03_summary["summary"]["total_pnl"])
            ),
            "to_s3_v03_development_pnl": (
                int(fixed_document["development"]["total_pnl"])  # type: ignore[index]
                - int(s3_v03_summary["development"]["total_pnl"])
            ),
            "to_s3_v03_validation_pnl": (
                int(fixed_document["validation"]["total_pnl"])  # type: ignore[index]
                - int(s3_v03_summary["validation"]["total_pnl"])
            ),
            "to_s3_v01_total_pnl": (
                int(fixed_summary["total_pnl"])
                - int(s3_v01_summary["summary"]["total_pnl"])
            ),
        },
        "audit": audits[STRATEGY_VERSION],
        "candidate_audits": audits,
        "limitations": [
            "0.83은 다음 주 범위가 LSMA 가격을 건드린 비율이지 종가 방향 정확도가 아닙니다.",
            "S4-v0.2의 방향은 논문식 평균회귀 방향의 반대이며 옵션 추세전략용 개발 후보입니다.",
            "현재 옵션일 선물 시가만 당일 정보로 사용하며 고가·저가·종가는 사용하지 않습니다.",
            "동일 날짜 선물 행이 없는 4개 옵션일은 이전·다음 날짜로 대체하지 않고 거래하지 않습니다.",
            "연속선물 롤오버로 OHLC가 불일치한 3개 행은 LSMA 연구에서 제외했습니다.",
            "선물 필터 성과 비교는 같은 옵션 표본에서 수행한 연구이며 새 옵션일 전진검증이 필요합니다.",
        ],
    }
    write_csv(
        OUTPUT_DIR / "strategy4_lsma_alignment.csv", alignment_rows
    )
    write_csv(
        OUTPUT_DIR / "strategy4_daily.csv",
        enriched_daily_rows(fixed, directions),
    )
    write_csv(OUTPUT_DIR / "strategy4_trades.csv", fixed.trades or [])
    write_csv(
        OUTPUT_DIR / "strategy4_v01_daily.csv",
        enriched_daily_rows(paper, directions),
    )
    write_csv(
        OUTPUT_DIR / "strategy4_v01_trades.csv", paper.trades or []
    )
    write_csv(
        OUTPUT_DIR / "strategy4_research_daily.csv",
        [
            {
                "candidate": state.key,
                "base_strategy": state.base_strategy,
                "direction_mode": state.direction_mode,
                **row,
            }
            for state in states
            for row in enriched_daily_rows(state, directions)
        ],
    )
    (OUTPUT_DIR / "strategy4_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy4_audit.json").write_text(
        json.dumps(audits[STRATEGY_VERSION], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "strategy4_v01_audit.json").write_text(
        json.dumps(
            audits[PAPER_STRATEGY_VERSION],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(row["passed"] for row in audits.values()):
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
