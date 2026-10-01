from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import minimal_baseline_backtest as base
import strategy5_ppt_momentum_backtest as legacy


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OBSERVATION_START = 900
OBSERVATION_END = 910
EARLY_END = 930
EVENT_END = 1515
PREMIUM_MIN = 0.40
PREMIUM_MAX = 1.20
CONFIRMATION_MULTIPLE = 1.5

ObservationMode = Literal[
    "first_window_trade_open",
    "first_qualifying_open",
    "legacy_qualifying_ohlc_max",
]
OBSERVATION_MODES: tuple[ObservationMode, ...] = (
    "first_window_trade_open",
    "first_qualifying_open",
    "legacy_qualifying_ohlc_max",
)


@dataclass(frozen=True)
class Observation:
    index: int
    minute: int
    price: float


@dataclass
class ContractEvent:
    mode: str
    date: str
    code: str
    call_put: str
    observation_minute: int
    observation_price: float
    initial_bar_ambiguous: bool
    branch: str
    early_confirmation_minute: int | None
    late_confirmation_minute: int | None
    late_strategy_eligible: bool
    reached_003_after_early_window: bool
    reached_2x_after_early_window: bool
    reached_5x_after_early_window: bool
    reached_10x_after_early_window: bool
    reached_2x_including_confirmation: bool
    reached_5x_including_confirmation: bool
    reached_10x_including_confirmation: bool
    reached_2x_after_confirmation: bool
    reached_5x_after_confirmation: bool
    reached_10x_after_confirmation: bool
    maximum_multiple_after_confirmation: float


def observation(
    contract: base.ContractGrid,
    mode: ObservationMode,
) -> Observation | None:
    start_index = base.MINUTE_INDEX[OBSERVATION_START]
    end_index = base.MINUTE_INDEX[OBSERVATION_END]
    for index in range(start_index, end_index + 1):
        bar = contract.bars[index]
        if not bar.traded or bar.open is None:
            continue
        if mode == "first_window_trade_open":
            price = float(bar.open)
            return (
                Observation(index, bar.minute, price)
                if PREMIUM_MIN <= price <= PREMIUM_MAX
                else None
            )
        if mode == "first_qualifying_open":
            price = float(bar.open)
            if PREMIUM_MIN <= price <= PREMIUM_MAX:
                return Observation(index, bar.minute, price)
            continue
        values = legacy.observed_values(bar)
        if values:
            return Observation(index, bar.minute, max(values))
    return None


def first_touch(
    contract: base.ContractGrid,
    start_index: int,
    end_minute: int,
    price: float,
) -> int | None:
    threshold = price * CONFIRMATION_MULTIPLE
    for index in range(start_index, len(contract.bars)):
        bar = contract.bars[index]
        if bar.minute > end_minute:
            return None
        if (
            bar.traded
            and bar.high is not None
            and float(bar.high) >= threshold
        ):
            return index
    return None


def reached(
    contract: base.ContractGrid,
    start_index: int,
    end_minute: int,
    threshold: float,
    field: str = "high",
) -> bool:
    for bar in contract.bars[start_index:]:
        if bar.minute > end_minute:
            break
        if not bar.traded:
            continue
        value = getattr(bar, field)
        if value is not None and float(value) <= threshold:
            return True
    return False


def high_reached(
    contract: base.ContractGrid,
    start_index: int,
    end_minute: int,
    threshold: float,
) -> bool:
    for bar in contract.bars[start_index:]:
        if bar.minute > end_minute:
            break
        if (
            bar.traded
            and bar.high is not None
            and float(bar.high) >= threshold
        ):
            return True
    return False


def maximum_multiple(
    contract: base.ContractGrid,
    start_index: int,
    end_minute: int,
    price: float,
) -> float:
    values = [
        float(bar.high) / price
        for bar in contract.bars[start_index:]
        if bar.minute <= end_minute
        and bar.traded
        and bar.high is not None
    ]
    return max(values, default=0.0)


def contract_event(
    date_value: str,
    contract: base.ContractGrid,
    mode: ObservationMode,
) -> ContractEvent | None:
    observed = observation(contract, mode)
    if observed is None:
        return None
    first_bar = contract.bars[observed.index]
    ambiguous = (
        first_bar.high is not None
        and float(first_bar.high)
        >= observed.price * CONFIRMATION_MULTIPLE
    )
    early_index = None
    late_index = None
    if not ambiguous:
        early_index = first_touch(
            contract,
            observed.index + 1,
            EARLY_END,
            observed.price,
        )
        if early_index is None:
            late_index = first_touch(
                contract,
                base.MINUTE_INDEX[EARLY_END] + 1,
                EVENT_END,
                observed.price,
            )
    if ambiguous:
        branch = "AMBIGUOUS_INITIAL_BAR"
    elif early_index is not None:
        branch = "EARLY"
    elif late_index is not None:
        branch = "LATE"
    else:
        branch = "NEVER"

    confirmation_index = (
        early_index if early_index is not None else late_index
    )
    including_start = (
        confirmation_index if confirmation_index is not None else 0
    )
    after_start = (
        confirmation_index + 1 if confirmation_index is not None else 0
    )
    early_after_index = base.MINUTE_INDEX[EARLY_END] + 1
    return ContractEvent(
        mode=mode,
        date=date_value,
        code=contract.code,
        call_put=contract.call_put,
        observation_minute=observed.minute,
        observation_price=round(observed.price, 4),
        initial_bar_ambiguous=ambiguous,
        branch=branch,
        early_confirmation_minute=(
            contract.bars[early_index].minute
            if early_index is not None
            else None
        ),
        late_confirmation_minute=(
            contract.bars[late_index].minute
            if late_index is not None
            else None
        ),
        late_strategy_eligible=(
            late_index is not None
            and contract.bars[late_index].minute <= base.ENTRY_CUTOFF
        ),
        reached_003_after_early_window=reached(
            contract,
            early_after_index,
            EVENT_END,
            0.03,
            field="low",
        ),
        reached_2x_after_early_window=high_reached(
            contract,
            early_after_index,
            EVENT_END,
            observed.price * 2.0,
        ),
        reached_5x_after_early_window=high_reached(
            contract,
            early_after_index,
            EVENT_END,
            observed.price * 5.0,
        ),
        reached_10x_after_early_window=high_reached(
            contract,
            early_after_index,
            EVENT_END,
            observed.price * 10.0,
        ),
        reached_2x_including_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                including_start,
                EVENT_END,
                observed.price * 2.0,
            )
        ),
        reached_5x_including_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                including_start,
                EVENT_END,
                observed.price * 5.0,
            )
        ),
        reached_10x_including_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                including_start,
                EVENT_END,
                observed.price * 10.0,
            )
        ),
        reached_2x_after_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                after_start,
                EVENT_END,
                observed.price * 2.0,
            )
        ),
        reached_5x_after_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                after_start,
                EVENT_END,
                observed.price * 5.0,
            )
        ),
        reached_10x_after_confirmation=(
            confirmation_index is not None
            and high_reached(
                contract,
                after_start,
                EVENT_END,
                observed.price * 10.0,
            )
        ),
        maximum_multiple_after_confirmation=round(
            maximum_multiple(
                contract,
                after_start,
                EVENT_END,
                observed.price,
            )
            if confirmation_index is not None
            else 0.0,
            6,
        ),
    )


def percentage(numerator: int, denominator: int) -> float:
    return (
        round(numerator / denominator * 100, 2)
        if denominator
        else 0.0
    )


def mode_summary(
    rows: list[ContractEvent],
) -> dict[str, object]:
    ambiguous = [row for row in rows if row.initial_bar_ambiguous]
    ordered = [row for row in rows if not row.initial_bar_ambiguous]
    early = [row for row in ordered if row.branch == "EARLY"]
    early_unconfirmed = [
        row for row in ordered if row.branch in {"LATE", "NEVER"}
    ]
    late = [row for row in ordered if row.branch == "LATE"]
    never = [row for row in ordered if row.branch == "NEVER"]

    def branch_summary(branch_rows: list[ContractEvent]) -> dict[str, object]:
        count = len(branch_rows)
        return {
            "count": count,
            "reached_2x_including_confirmation": sum(
                row.reached_2x_including_confirmation
                for row in branch_rows
            ),
            "reached_2x_including_confirmation_pct": percentage(
                sum(
                    row.reached_2x_including_confirmation
                    for row in branch_rows
                ),
                count,
            ),
            "reached_5x_including_confirmation": sum(
                row.reached_5x_including_confirmation
                for row in branch_rows
            ),
            "reached_10x_including_confirmation": sum(
                row.reached_10x_including_confirmation
                for row in branch_rows
            ),
            "reached_10x_including_confirmation_pct": percentage(
                sum(
                    row.reached_10x_including_confirmation
                    for row in branch_rows
                ),
                count,
            ),
            "reached_2x_after_confirmation": sum(
                row.reached_2x_after_confirmation
                for row in branch_rows
            ),
            "reached_2x_after_confirmation_pct": percentage(
                sum(
                    row.reached_2x_after_confirmation
                    for row in branch_rows
                ),
                count,
            ),
            "reached_5x_after_confirmation": sum(
                row.reached_5x_after_confirmation
                for row in branch_rows
            ),
            "reached_10x_after_confirmation": sum(
                row.reached_10x_after_confirmation
                for row in branch_rows
            ),
            "reached_10x_after_confirmation_pct": percentage(
                sum(
                    row.reached_10x_after_confirmation
                    for row in branch_rows
                ),
                count,
            ),
        }

    late_ten = sum(
        row.reached_10x_after_early_window for row in early_unconfirmed
    )
    return {
        "eligible_contracts": len(rows),
        "ambiguous_initial_bar": len(ambiguous),
        "ordered_contracts": len(ordered),
        "early": branch_summary(early),
        "early_unconfirmed": {
            "count": len(early_unconfirmed),
            "reached_003": sum(
                row.reached_003_after_early_window
                for row in early_unconfirmed
            ),
            "reached_003_pct": percentage(
                sum(
                    row.reached_003_after_early_window
                    for row in early_unconfirmed
                ),
                len(early_unconfirmed),
            ),
            "eventual_10x": late_ten,
            "eventual_10x_pct": percentage(
                late_ten,
                len(early_unconfirmed),
            ),
        },
        "late": {
            **branch_summary(late),
            "strategy_eligible_count": sum(
                row.late_strategy_eligible for row in late
            ),
        },
        "never": {
            "count": len(never),
            "reached_003": sum(
                row.reached_003_after_early_window for row in never
            ),
            "reached_003_pct": percentage(
                sum(
                    row.reached_003_after_early_window
                    for row in never
                ),
                len(never),
            ),
        },
    }


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        group
        for group in base.derive_expiry_groups(profile)
        if legacy.PPT_START
        <= group["expiry_date"]
        <= legacy.PPT_END
    ]
    files = base.target_file_map()
    rows_by_mode: dict[str, list[ContractEvent]] = {
        mode: [] for mode in OBSERVATION_MODES
    }
    for group in groups:
        date_value = group["expiry_date"]
        contracts = base.load_expiry_day(files[date_value], group)
        for mode in OBSERVATION_MODES:
            for contract in contracts.values():
                event = contract_event(date_value, contract, mode)
                if event is not None:
                    rows_by_mode[mode].append(event)

    result = {
        "status": "entry_event_denominator_research",
        "scope": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
            "expiry_days": len(groups),
        },
        "rules": {
            "observation_window": [OBSERVATION_START, OBSERVATION_END],
            "early_window_end": EARLY_END,
            "event_end": EVENT_END,
            "premium_range": [PREMIUM_MIN, PREMIUM_MAX],
            "confirmation_multiple": CONFIRMATION_MULTIPLE,
            "same_observation_bar_touch": "ambiguous and excluded",
            "late_denominator": (
                "all ordered early-unconfirmed contracts, without "
                "conditioning on 0.03 survival"
            ),
        },
        "modes": {
            mode: mode_summary(rows)
            for mode, rows in rows_by_mode.items()
        },
    }
    (
        OUTPUT_DIR / "strategy5_v03_event_research_summary.json"
    ).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        OUTPUT_DIR / "strategy5_v03_contract_events.csv",
        [
            asdict(row)
            for row in rows_by_mode["first_qualifying_open"]
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
