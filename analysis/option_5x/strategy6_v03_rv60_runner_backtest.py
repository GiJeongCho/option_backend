from __future__ import annotations

from pathlib import Path

from temp import s6_v03_rv60_runner_backtest as research


ROOT = Path(__file__).resolve().parent


def main() -> None:
    research.STRATEGY_VERSION = "S6-v0.3"
    research.OUTPUT_STEM = "strategy6_v03_rv60_runner"
    research.TEMP_OUTPUT = ROOT / "output"
    research.main()


if __name__ == "__main__":
    main()
