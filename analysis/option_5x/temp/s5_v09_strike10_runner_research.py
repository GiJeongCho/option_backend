from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import minimal_baseline_backtest as base
import strategy5_v04_basket_backtest as v04
import strategy5_v09_strike10_ten_x_backtest as v09


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "s5_v09_strike10_runner_research.json"
)


def run() -> dict[str, object]:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()
    by_year: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "expiry_days": 0,
            "eligible": 0,
            "confirmed_1_5x": 0,
            "reached_2x": 0,
            "reached_5x": 0,
            "reached_10x": 0,
        }
    )
    cases: list[dict[str, object]] = []

    for group in groups:
        date_value = group["expiry_date"]
        year = date_value[:4]
        by_year[year]["expiry_days"] += 1
        contracts = base.load_expiry_day(files[date_value], group)
        setups = v09.strike10_setups(contracts)
        by_year[year]["eligible"] += len(setups)

        for setup in setups:
            fill = v04.setup_fill(contracts[setup.code])
            if fill is None:
                continue
            fill_index, fill_minute, fill_price, _, _ = fill
            confirmation = v04.first_high_touch(
                contracts[setup.code],
                fill_index,
                929,
                setup.observation_price * 1.5,
            )
            if confirmation is None:
                continue
            highs = [
                float(bar.high)
                for bar in contracts[setup.code].bars[
                    confirmation : base.MINUTE_INDEX[1515] + 1
                ]
                if bar.traded and bar.high is not None
            ]
            peak_multiple = (
                max(highs, default=setup.observation_price)
                / setup.observation_price
            )
            by_year[year]["confirmed_1_5x"] += 1
            for key, threshold in (
                ("reached_2x", 2.0),
                ("reached_5x", 5.0),
                ("reached_10x", 10.0),
            ):
                by_year[year][key] += peak_multiple >= threshold
            cases.append(
                {
                    "date": date_value,
                    "code": setup.code,
                    "call_put": setup.call_put,
                    "strike": v09.contract_strike(
                        contracts[setup.code]
                    ),
                    "entry_minute": fill_minute,
                    "entry_raw_price": round(
                        fill_price - base.SLIPPAGE,
                        4,
                    ),
                    "confirmation_minute": (
                        contracts[setup.code].bars[confirmation].minute
                    ),
                    "peak_multiple": round(peak_multiple, 4),
                    "peak_bucket": v09.peak_bucket(peak_multiple),
                }
            )

    totals = {
        key: sum(year[key] for year in by_year.values())
        for key in (
            "expiry_days",
            "eligible",
            "confirmed_1_5x",
            "reached_2x",
            "reached_5x",
            "reached_10x",
        )
    }
    result = {
        "status": "descriptive_only_insufficient_sample",
        "period": {
            "start": groups[0]["expiry_date"],
            "end": groups[-1]["expiry_date"],
        },
        "definition": {
            "entry_raw_open": "0.40-0.49",
            "strike": "divisible by 10 points",
            "confirmation": "post-fill 1.5x P0 through 09:29",
            "peak_window": "confirmation through 15:15",
        },
        "totals": totals,
        "by_year": dict(sorted(by_year.items())),
        "cases": cases,
        "decision": {
            "classifier_applied": False,
            "reason": (
                "Only six confirmed events and one 10x event exist across "
                "the full available period. Indicator fitting would be "
                "post-selection overfitting."
            ),
            "minimum_confirmed_sample": v09.MINIMUM_RUNNER_SAMPLE,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
