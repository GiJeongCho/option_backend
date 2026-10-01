from __future__ import annotations

import audit_strategy5_v05 as base_audit


def run() -> dict[str, object]:
    base_audit.OUTPUT_STEM = "strategy5_v06_ten_x"
    base_audit.SUMMARY_PATH = (
        base_audit.OUTPUT_DIR / "strategy5_v06_ten_x_summary.json"
    )
    base_audit.AUDIT_PATH = (
        base_audit.OUTPUT_DIR
        / "strategy5_v06_ten_x_independent_audit.json"
    )
    base_audit.STRATEGY_VERSION = "S5-v0.6-10X"
    base_audit.EXPECTED_TARGET = 10.0
    base_audit.EXPECTED_TARGET_REASON = "TARGET_10X_ALL"
    return base_audit.run()


if __name__ == "__main__":
    run()
