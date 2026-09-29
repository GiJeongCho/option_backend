from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .data_service import DataNotFoundError


INITIAL_CASH = 80_000_000


def _to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _to_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _max_drawdown(values: list[int]) -> int:
    peak = values[0]
    worst = 0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


class BacktestResultService:
    def __init__(self, backend_dir: Path | None = None) -> None:
        self.backend_dir = backend_dir or Path(__file__).resolve().parents[1]
        self.output_dir = (
            self.backend_dir / "analysis" / "option_5x" / "output"
        )
        self.summary_path = self.output_dir / "minimal_baseline_summary.json"
        self.grid_path = self.output_dir / "minimal_baseline_grid.csv"
        self.daily_path = self.output_dir / "minimal_baseline_daily.csv"

    def _require_outputs(self) -> None:
        missing = [
            path
            for path in (self.summary_path, self.grid_path, self.daily_path)
            if not path.exists()
        ]
        if missing:
            raise DataNotFoundError(
                "웹 백테스트 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                "minimal_baseline_backtest.py`를 먼저 실행하세요."
            )

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def _summary_document(self) -> dict[str, object]:
        self._require_outputs()
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def _grid_rows(self) -> list[dict[str, str]]:
        self._require_outputs()
        return self._read_csv(self.grid_path)

    def _daily_rows(self) -> list[dict[str, str]]:
        self._require_outputs()
        return self._read_csv(self.daily_path)

    @staticmethod
    def _strategy_key(
        premium_range: str, exit_mode: str, liquidity_mode: str
    ) -> str:
        return (
            f"{premium_range}|technical|{exit_mode}|{liquidity_mode}"
        )

    def options(self) -> dict[str, object]:
        document = self._summary_document()
        rows = [
            row
            for row in self._grid_rows()
            if row["entry_mode"] == "technical"
        ]
        premium_ranges = list(dict.fromkeys(row["premium_range"] for row in rows))
        exit_modes = list(dict.fromkeys(row["exit_mode"] for row in rows))
        liquidity_modes = list(
            dict.fromkeys(row["liquidity_mode"] for row in rows)
        )
        combinations = [
            {
                "premiumRange": row["premium_range"],
                "exitMode": row["exit_mode"],
                "liquidityMode": row["liquidity_mode"],
            }
            for row in rows
        ]
        generated_at = datetime.fromtimestamp(
            self.summary_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        available_dates = sorted(
            {row["date"] for row in self._daily_rows()}
        )
        return {
            "status": document.get("status"),
            "initialCash": INITIAL_CASH,
            "availablePeriod": {
                "start": available_dates[0] if available_dates else None,
                "end": available_dates[-1] if available_dates else None,
            },
            "premiumRanges": premium_ranges,
            "exitModes": exit_modes,
            "liquidityModes": liquidity_modes,
            "combinations": combinations,
            "default": {
                "premiumRange": "2.50-5.00",
                "exitMode": "two_x_all",
                "liquidityMode": "next_bar",
            },
            "rules": document.get("rules", {}),
            "execution": document.get("execution", {}),
            "audit": document.get("audit", {}),
            "generatedAt": generated_at,
        }

    def result(
        self,
        premium_range: str,
        exit_mode: str,
        liquidity_mode: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        key = self._strategy_key(
            premium_range, exit_mode, liquidity_mode
        )
        combination = next(
            (row for row in self._grid_rows() if row["strategy"] == key),
            None,
        )
        if combination is None:
            raise ValueError(f"지원하지 않는 백테스트 조합입니다: {key}")

        all_daily_source = [
            row for row in self._daily_rows() if row["strategy"] == key
        ]
        if not all_daily_source:
            raise DataNotFoundError(f"백테스트 일별 결과가 없습니다: {key}")
        available_start = all_daily_source[0]["date"]
        available_end = all_daily_source[-1]["date"]
        requested_start = start_date or available_start
        requested_end = end_date or available_end
        if requested_start > requested_end:
            raise ValueError("백테스트 시작일은 종료일보다 늦을 수 없습니다.")
        daily_source = [
            row
            for row in all_daily_source
            if requested_start <= row["date"] <= requested_end
        ]
        if not daily_source:
            raise ValueError(
                "선택 기간에 백테스트 대상 만기일이 없습니다."
            )

        cash = INITIAL_CASH
        daily: list[dict[str, object]] = []
        for row in daily_source:
            pnl = _to_int(row["pnl"])
            item: dict[str, object] = {
                "date": row["date"],
                "startCash": cash,
                "endCash": cash + pnl,
                "pnl": pnl,
                "signalFound": _to_bool(row["signal_found"]),
                "entryFilled": _to_bool(row["entry_filled"]),
                "code": row["code"],
                "signalMinute": (
                    _to_int(row["signal_minute"])
                    if row["signal_minute"]
                    else None
                ),
                "entryMinute": (
                    _to_int(row["entry_minute"])
                    if row["entry_minute"]
                    else None
                ),
                "entryQuantity": _to_int(row["entry_quantity"]),
                "buyPrincipal": _to_int(row["buy_principal"]),
                "fees": _to_int(row["buy_fee"])
                + _to_int(row["sell_fee"]),
                "grossSales": _to_int(row["gross_sales"]),
                "expiredQuantity": _to_int(row["expired_quantity"]),
            }
            daily.append(item)
            cash += pnl

        profits = [int(row["pnl"]) for row in daily]
        positive_profits = sorted(
            (value for value in profits if value > 0), reverse=True
        )
        equity = [INITIAL_CASH] + [
            int(row["endCash"]) for row in daily
        ]
        total_pnl = sum(profits)
        late_period_pnl = sum(
            int(row["pnl"])
            for row in daily
            if "20260501" <= str(row["date"]) <= "20260831"
        )
        summary = {
            "initialCash": INITIAL_CASH,
            "endingCash": INITIAL_CASH + total_pnl,
            "totalPnl": total_pnl,
            "expiryDays": len(daily),
            "signalDays": sum(bool(row["signalFound"]) for row in daily),
            "tradeDays": sum(bool(row["entryFilled"]) for row in daily),
            "profitableDays": sum(value > 0 for value in profits),
            "losingDays": sum(value < 0 for value in profits),
            "maximumDayPnl": max(profits),
            "minimumDayPnl": min(profits),
            "maxDrawdown": _max_drawdown(equity),
            "pnlExcludingBest5Days": (
                total_pnl - sum(positive_profits[:5])
            ),
            "latePeriodPnl": late_period_pnl,
            "daysAtLeast2_5mProfit": sum(
                value >= 2_500_000 for value in profits
            ),
            "totalBuyPrincipal": sum(
                int(row["buyPrincipal"]) for row in daily
            ),
            "totalFees": sum(int(row["fees"]) for row in daily),
            "expiredContracts": sum(
                int(row["expiredQuantity"]) for row in daily
            ),
            "expiryResidualDays": sum(
                int(row["expiredQuantity"]) > 0 for row in daily
            ),
            "maximumDailyPrincipal": max(
                int(row["buyPrincipal"]) for row in daily
            ),
        }
        yearly = []
        for year in sorted({str(row["date"])[:4] for row in daily}):
            year_rows = [
                row for row in daily if str(row["date"]).startswith(year)
            ]
            year_profits = [int(row["pnl"]) for row in year_rows]
            yearly.append(
                {
                    "year": year,
                    "expiryDays": len(year_rows),
                    "tradeDays": sum(
                        bool(row["entryFilled"]) for row in year_rows
                    ),
                    "profitableDays": sum(
                        value > 0 for value in year_profits
                    ),
                    "losingDays": sum(value < 0 for value in year_profits),
                    "totalPnl": sum(year_profits),
                    "totalBuyPrincipal": sum(
                        int(row["buyPrincipal"]) for row in year_rows
                    ),
                }
            )
        return {
            "strategy": {
                "key": key,
                "premiumRange": premium_range,
                "entryMode": "technical",
                "exitMode": exit_mode,
                "liquidityMode": liquidity_mode,
            },
            "summary": summary,
            "yearly": yearly,
            "daily": daily,
            "source": {
                "kind": "audited-local-raw-data-result",
                "period": {
                    "start": daily[0]["date"] if daily else None,
                    "end": daily[-1]["date"] if daily else None,
                },
                "availablePeriod": {
                    "start": available_start,
                    "end": available_end,
                },
                "requestedPeriod": {
                    "start": requested_start,
                    "end": requested_end,
                },
                "limitations": [
                    "동일 전체 자료에서 가격대를 비교한 탐색 결과입니다.",
                    "미사용 홀드아웃 검증 전이므로 최종 전략이 아닙니다.",
                    "마지막까지 미체결된 잔량은 0원으로 평가했습니다.",
                    "선택 기간은 고정 8,000만원에서 다시 시작하도록 일별 손익을 재기준화합니다.",
                ],
            },
        }
