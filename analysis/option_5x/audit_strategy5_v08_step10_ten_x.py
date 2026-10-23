from __future__ import annotations

import json
import math

import audit_strategy5_v05 as base_audit
import minimal_baseline_backtest as raw
import strategy5_ppt_momentum_backtest as legacy


STRATEGY_VERSION = "S5-v0.8-STEP10-10X"
OUTPUT_STEM = "strategy5_v08_step10_ten_x"
ALLOWED_CENTS = set(range(40, 121, 10))


def run() -> dict[str, object]:
    base_audit.OUTPUT_STEM = OUTPUT_STEM
    base_audit.SUMMARY_PATH = (
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
    )
    base_audit.AUDIT_PATH = (
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_independent_audit.json"
    )
    base_audit.STRATEGY_VERSION = STRATEGY_VERSION
    base_audit.EXPECTED_INITIAL_CASH = 40_000_000
    base_audit.EXPECTED_ENTRY_RAW_MIN = 0.40
    base_audit.EXPECTED_ENTRY_RAW_MAX = 1.20
    base_audit.EXPECTED_TARGET = 10.0
    base_audit.EXPECTED_TARGET_REASON = "TARGET_10X_ALL"
    result = base_audit.run()

    positions = base_audit.read_csv(
        base_audit.OUTPUT_DIR / f"{OUTPUT_STEM}_positions.csv"
    )
    profile = json.loads(raw.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = {
        group["expiry_date"]: group
        for group in legacy.backtest_groups(profile)
    }
    files = raw.target_file_map()
    contracts_by_date: dict[str, dict[str, raw.ContractGrid]] = {}
    exact_errors: list[str] = []
    observed_levels: set[int] = set()

    for position in positions:
        date_value = position["date"]
        if date_value not in contracts_by_date:
            contracts_by_date[date_value] = raw.load_expiry_day(
                files[date_value],
                groups[date_value],
            )
        contract = contracts_by_date[date_value][position["code"]]
        minute = base_audit.integer(position["entry_minute"])
        bar = contract.bars[raw.MINUTE_INDEX[minute]]
        if not bar.traded or bar.open is None:
            exact_errors.append(f"{date_value}: missing entry bar")
            continue
        raw_price = float(bar.open)
        cents = round(raw_price * 100)
        if (
            cents not in ALLOWED_CENTS
            or not math.isclose(
                raw_price * 100,
                cents,
                abs_tol=1e-7,
            )
        ):
            exact_errors.append(
                f"{date_value} {position['code']}: "
                f"non-step raw open {raw_price:.8f}"
            )
        else:
            observed_levels.add(cents)

    result["error_samples"] = [
        *result["error_samples"],
        *exact_errors,
    ][:60]
    result["error_count"] = int(result["error_count"]) + len(exact_errors)
    result["passed"] = not result["error_count"]
    result["checks"].append(
        "raw entry opens independently restricted to exact 0.1 steps"
    )
    result["observed_entry_levels"] = [
        f"{cents / 100:.1f}" for cents in sorted(observed_levels)
    ]
    base_audit.AUDIT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
