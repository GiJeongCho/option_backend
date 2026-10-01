from __future__ import annotations

import audit_strategy5_v05 as base_audit


def run() -> dict[str, object]:
    base_audit.OUTPUT_STEM = "strategy5_v07_04_bin_ev"
    base_audit.SUMMARY_PATH = (
        base_audit.OUTPUT_DIR
        / "strategy5_v07_04_bin_ev_summary.json"
    )
    base_audit.AUDIT_PATH = (
        base_audit.OUTPUT_DIR
        / "strategy5_v07_04_bin_ev_independent_audit.json"
    )
    base_audit.STRATEGY_VERSION = "S5-v0.7-04BIN-EV"
    base_audit.EXPECTED_INITIAL_CASH = 40_000_000
    base_audit.EXPECTED_ENTRY_RAW_MIN = 0.40
    base_audit.EXPECTED_ENTRY_RAW_MAX = 0.49
    base_audit.EXPECTED_TARGET = 3.907583
    base_audit.EXPECTED_TARGET_REASON = "EXPECTED_PEAK_3_9076X_ALL"
    return base_audit.run()


if __name__ == "__main__":
    run()
