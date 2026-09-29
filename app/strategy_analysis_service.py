from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data_service import DataNotFoundError


def _integer(value: str | int | float | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _number(value: str | int | float | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _boolean(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


class StrategyAnalysisService:
    def __init__(self, backend_dir: Path | None = None) -> None:
        self.backend_dir = backend_dir or Path(__file__).resolve().parents[1]
        self.output_dir = (
            self.backend_dir / "analysis" / "option_5x" / "output"
        )
        self.summary_path = self.output_dir / "strategy1_summary.json"
        self.daily_path = self.output_dir / "strategy1_daily.csv"
        self.campaign_path = self.output_dir / "strategy1_campaigns.csv"
        self.strategy_two_summary_path = (
            self.output_dir / "strategy2_summary.json"
        )
        self.strategy_two_daily_path = self.output_dir / "strategy2_daily.csv"
        self.strategy_two_v02_summary_path = (
            self.output_dir / "strategy2_v02_summary.json"
        )
        self.strategy_two_v02_daily_path = (
            self.output_dir / "strategy2_v02_daily.csv"
        )
        self.strategy_two_v03_summary_path = (
            self.output_dir / "strategy2_v03_summary.json"
        )
        self.strategy_two_v03_daily_path = (
            self.output_dir / "strategy2_v03_daily.csv"
        )
        self.strategy_three_summary_path = (
            self.output_dir / "strategy3_summary.json"
        )
        self.strategy_three_daily_path = (
            self.output_dir / "strategy3_daily.csv"
        )
        self.strategy_three_event_path = (
            self.output_dir / "s3_breakout_event_summary.json"
        )
        self.strategy_three_v02_summary_path = (
            self.output_dir / "strategy3_v02_summary.json"
        )
        self.strategy_three_v02_daily_path = (
            self.output_dir / "strategy3_v02_daily.csv"
        )
        self.strategy_three_v02_trades_path = (
            self.output_dir / "strategy3_v02_trades.csv"
        )
        self.strategy_three_v03_summary_path = (
            self.output_dir / "strategy3_v03_summary.json"
        )
        self.strategy_three_v03_daily_path = (
            self.output_dir / "strategy3_v03_daily.csv"
        )
        self.strategy_three_v03_trades_path = (
            self.output_dir / "strategy3_v03_trades.csv"
        )
        self.futures_lsma_summary_path = (
            self.output_dir / "futures_lsma_summary.json"
        )
        self.strategy_four_summary_path = (
            self.output_dir / "strategy4_summary.json"
        )
        self.strategy_four_daily_path = (
            self.output_dir / "strategy4_daily.csv"
        )
        self.strategy_four_trades_path = (
            self.output_dir / "strategy4_trades.csv"
        )
        self.strategy_four_v01_daily_path = (
            self.output_dir / "strategy4_v01_daily.csv"
        )
        self.strategy_four_v01_trades_path = (
            self.output_dir / "strategy4_v01_trades.csv"
        )
        self.strategy_five_summary_path = (
            self.output_dir / "strategy5_summary.json"
        )
        self.strategy_five_daily_path = (
            self.output_dir / "strategy5_daily.csv"
        )
        self.strategy_five_trades_path = (
            self.output_dir / "strategy5_trades.csv"
        )

    def _require_outputs(self) -> None:
        missing = [
            path
            for path in (
                self.summary_path,
                self.daily_path,
                self.campaign_path,
            )
            if not path.exists()
        ]
        if missing:
            raise DataNotFoundError(
                "전략 1 분석 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                "strategy1_backtest.py`를 먼저 실행하세요."
            )

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def _document(self) -> dict[str, object]:
        self._require_outputs()
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def _daily(self) -> list[dict[str, str]]:
        self._require_outputs()
        return self._read_csv(self.daily_path)

    def _campaigns(self) -> list[dict[str, str]]:
        self._require_outputs()
        return self._read_csv(self.campaign_path)

    @staticmethod
    def _loss_groups(
        rows: object,
    ) -> list[dict[str, object]]:
        return [
            {
                "group": str(row["group"]),
                "lossCount": _integer(row["loss_count"]),
                "totalLoss": _integer(row["total_loss"]),
                "averageLoss": _number(row["average_loss"]),
                "worstLoss": _integer(row["worst_loss"]),
            }
            for row in list(rows)  # type: ignore[arg-type]
        ]

    def strategy_one(self) -> dict[str, object]:
        document = self._document()
        raw_summary = dict(document["summary"])  # type: ignore[arg-type]
        raw_loss = dict(document["loss_analysis"])  # type: ignore[arg-type]
        daily = [
            {
                "date": row["date"],
                "previousVix": _number(row["previous_vix"]),
                "highVolatility": _boolean(row["high_volatility"]),
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "signalFound": _boolean(row["signal_found"]),
                "tradeDay": _boolean(row["trade_day"]),
                "campaignCount": _integer(row["campaign_count"]),
                "buyPrincipal": _integer(row["buy_principal"]),
                "fees": _integer(row["fees"]),
                "expiredQuantity": _integer(row["expired_quantity"]),
                "firstEntryMinute": (
                    _integer(row["first_entry_minute"])
                    if row["first_entry_minute"]
                    else None
                ),
                "callPut": row["call_put"],
                "exitReasons": row["exit_reasons"].split(",")
                if row["exit_reasons"]
                else [],
                "mfePct": _number(row["mfe_pct"]),
                "maePct": _number(row["mae_pct"]),
            }
            for row in self._daily()
        ]

        campaigns = self._campaigns()
        exit_overview: dict[str, list[int]] = defaultdict(list)
        for row in campaigns:
            exit_overview[row["exit_reason"]].append(_integer(row["pnl"]))
        exit_reasons = sorted(
            (
                {
                    "reason": reason,
                    "count": len(values),
                    "profitableCount": sum(value > 0 for value in values),
                    "losingCount": sum(value < 0 for value in values),
                    "totalPnl": sum(values),
                    "averagePnl": round(sum(values) / len(values), 2),
                }
                for reason, values in exit_overview.items()
            ),
            key=lambda item: int(item["count"]),
            reverse=True,
        )
        worst_campaigns = [
            {
                "date": row["date"],
                "campaign": _integer(row["campaign"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "entryMinute": _integer(row["first_entry_minute"]),
                "exitMinute": _integer(row["exit_minute"]),
                "trancheCount": _integer(row["tranche_count"]),
                "buyPrincipal": _integer(row["buy_principal"]),
                "pnl": _integer(row["pnl"]),
                "mfePct": _number(row["mfe_pct"]),
                "maePct": _number(row["mae_pct"]),
                "exitReason": row["exit_reason"],
            }
            for row in sorted(
                campaigns, key=lambda item: _integer(item["pnl"])
            )[:15]
        ]

        audit = dict(document["audit"])  # type: ignore[arg-type]
        generated_at = datetime.fromtimestamp(
            self.summary_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "summary": {
                "expiryDays": _integer(raw_summary["expiry_days"]),
                "highVolatilityDays": _integer(
                    raw_summary["high_volatility_days"]
                ),
                "signalDays": _integer(raw_summary["signal_days"]),
                "tradeDays": _integer(raw_summary["trade_days"]),
                "profitableDays": _integer(raw_summary["profitable_days"]),
                "losingDays": _integer(raw_summary["losing_days"]),
                "noTradeDays": _integer(raw_summary["no_trade_days"]),
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
                "totalPnl": _integer(raw_summary["total_pnl"]),
                "maximumDayPnl": _integer(
                    raw_summary["maximum_day_pnl"]
                ),
                "minimumDayPnl": _integer(
                    raw_summary["minimum_day_pnl"]
                ),
                "maxDrawdown": _integer(raw_summary["max_drawdown"]),
                "pnlExcludingBest5Days": _integer(
                    raw_summary["pnl_excluding_best_5_days"]
                ),
                "totalCampaigns": _integer(
                    raw_summary["total_campaigns"]
                ),
                "totalBuyPrincipal": _integer(
                    raw_summary["total_buy_principal"]
                ),
                "totalFees": _integer(raw_summary["total_fees"]),
                "expiredContracts": _integer(
                    raw_summary["expired_contracts"]
                ),
            },
            "lossAnalysis": {
                "losingDayCount": _integer(raw_loss["losing_day_count"]),
                "losingCampaignCount": _integer(
                    raw_loss["losing_campaign_count"]
                ),
                "totalLosingDayLoss": _integer(
                    raw_loss["total_losing_day_loss"]
                ),
                "byExitReason": self._loss_groups(
                    raw_loss["by_exit_reason"]
                ),
                "byCallPut": self._loss_groups(raw_loss["by_call_put"]),
                "byEntryTime": self._loss_groups(
                    raw_loss["by_entry_time"]
                ),
                "byVix": self._loss_groups(raw_loss["by_vix"]),
                "byTrancheCount": self._loss_groups(
                    raw_loss["by_tranche_count"]
                ),
                "byMonth": self._loss_groups(raw_loss["by_month"]),
            },
            "exitReasonOverview": exit_reasons,
            "worstCampaigns": worst_campaigns,
            "daily": daily,
            "nextStrategies": [
                {
                    "id": "S3",
                    "name": "전략 1 진입·2배 전량청산",
                    "purpose": "손실 원인이 진입인지 5배 추적 청산인지 분리",
                },
                {
                    "id": "S4",
                    "name": "손실 회피형 워크포워드",
                    "purpose": "손실 집중 구간 제외 조건을 과적합 없이 검증",
                },
            ],
            "audit": {
                "passed": bool(audit["passed"]),
                "errorCount": _integer(audit["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": generated_at,
        }

    @staticmethod
    def _strategy_two_period(raw: object) -> dict[str, object]:
        period = dict(raw)  # type: ignore[arg-type]
        return {
            "days": _integer(period["days"]),
            "tradeDays": _integer(period["trade_days"]),
            "profitableDays": _integer(period["profitable_days"]),
            "losingDays": _integer(period["losing_days"]),
            "totalPnl": _integer(period["total_pnl"]),
            "profitFactor": _number(period["profit_factor"]),
            "minimumDayPnl": _integer(period["minimum_day_pnl"]),
            "maximumDayPnl": _integer(period["maximum_day_pnl"]),
            "maxDrawdown": _integer(period["max_drawdown"]),
            "pnlExcludingBest3Days": _integer(
                period["pnl_excluding_best_3_days"]
            ),
            "pnlExcludingBest5Days": _integer(
                period["pnl_excluding_best_5_days"]
            ),
            "expiredContracts": _integer(period["expired_contracts"]),
            "confirmedDays": _integer(period["confirmed_days"]),
            "addedDays": _integer(period["added_days"]),
            "totalBuyPrincipal": _integer(period["total_buy_principal"]),
        }

    def _strategy_two_response(
        self,
        summary_path: Path,
        daily_path: Path,
        backtest_script: str,
    ) -> dict[str, object]:
        missing = [
            path
            for path in (
                summary_path,
                daily_path,
            )
            if not path.exists()
        ]
        if missing:
            raise DataNotFoundError(
                "전략 2 분석 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                f"{backtest_script}`를 먼저 실행하세요."
            )
        document = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_two_period(raw_summary)
        summary.update(
            {
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
                "expiryResidualDays": _integer(
                    raw_summary["expiry_residual_days"]
                ),
            }
        )
        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "entryMinute": (
                    _integer(row["entry_minute"])
                    if row["entry_minute"]
                    else None
                ),
                "buyPrincipal": _integer(row["buy_principal"]),
                "confirmed": _boolean(row["confirmed"]),
                "addedQuantity": _integer(row["added_quantity"]),
                "exitReason": row["exit_reason"],
                "expiredQuantity": _integer(row["expired_quantity"]),
            }
            for row in self._read_csv(daily_path)
        ]
        analysis = dict(document["analysis"])
        audit = dict(document["audit"])
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "rules": document["rules"],
            "summary": summary,
            "development": self._strategy_two_period(
                document["development"]
            ),
            "validation": self._strategy_two_period(
                document["validation"]
            ),
            "exitReasonOverview": [
                {
                    "reason": row["group"],
                    "count": _integer(row["count"]),
                    "profitableCount": _integer(
                        row["profitable_count"]
                    ),
                    "losingCount": _integer(row["losing_count"]),
                    "totalPnl": _integer(row["total_pnl"]),
                    "averagePnl": _integer(row.get("average_pnl")),
                }
                for row in analysis["by_exit_reason"]
            ],
            "daily": daily,
            "audit": {
                "passed": bool(audit["passed"]),
                "errorCount": _integer(audit["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def strategy_two(self) -> dict[str, object]:
        return self._strategy_two_response(
            self.strategy_two_summary_path,
            self.strategy_two_daily_path,
            "strategy2_backtest.py",
        )

    def strategy_two_v02(self) -> dict[str, object]:
        return self._strategy_two_response(
            self.strategy_two_v02_summary_path,
            self.strategy_two_v02_daily_path,
            "strategy2_v02_backtest.py",
        )

    def strategy_two_v03(self) -> dict[str, object]:
        return self._strategy_two_response(
            self.strategy_two_v03_summary_path,
            self.strategy_two_v03_daily_path,
            "strategy2_v03_backtest.py",
        )

    @staticmethod
    def _strategy_three_period(raw: object) -> dict[str, object]:
        period = dict(raw)  # type: ignore[arg-type]
        return {
            "days": _integer(period["days"]),
            "signalDays": _integer(period["signal_days"]),
            "tradeDays": _integer(period["trade_days"]),
            "profitableDays": _integer(period["profitable_days"]),
            "losingDays": _integer(period["losing_days"]),
            "totalPnl": _integer(period["total_pnl"]),
            "profitFactor": _number(period["profit_factor"]),
            "minimumDayPnl": _integer(period["minimum_day_pnl"]),
            "maximumDayPnl": _integer(period["maximum_day_pnl"]),
            "maxDrawdown": _integer(period["max_drawdown"]),
            "pnlExcludingBest3Days": _integer(
                period["pnl_excluding_best_3_days"]
            ),
            "pnlExcludingBest5Days": _integer(
                period["pnl_excluding_best_5_days"]
            ),
            "totalBuyPrincipal": _integer(
                period["total_buy_principal"]
            ),
            "expiredContracts": _integer(period["expired_contracts"]),
            "reached2xDays": _integer(
                period.get(
                    "entry_reached_2x_days",
                    period.get("reached_2x_days"),
                )
            ),
            "reached5xDays": _integer(
                period.get("entry_reached_5x_days")
            ),
            "reached10xDays": _integer(
                period.get("entry_reached_10x_days")
            ),
            "oneStageDays": _integer(period.get("one_stage_days")),
            "twoStageDays": _integer(period.get("two_stage_days")),
            "threeStageDays": _integer(period.get("three_stage_days")),
            "lowerLowDays": _integer(period.get("lower_low_days")),
            "secondGoldenCrossDays": _integer(
                period.get("second_golden_cross_days")
            ),
            "secondDeathCrossExitDays": _integer(
                period.get("second_death_cross_exit_days")
            ),
            "qualifiedPatternDays": _integer(
                period.get("qualified_pattern_days")
            ),
            "principalRecoveredDays": _integer(
                period.get("principal_recovered_days")
            ),
        }

    def strategy_three(self) -> dict[str, object]:
        required = (
            self.strategy_three_summary_path,
            self.strategy_three_daily_path,
            self.strategy_three_event_path,
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise DataNotFoundError(
                "전략 3 분석 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                "analyze_breakout_opportunities.py`와 "
                "`strategy3_research.py`를 먼저 실행하세요."
            )
        document = json.loads(
            self.strategy_three_summary_path.read_text(encoding="utf-8")
        )
        event = json.loads(
            self.strategy_three_event_path.read_text(encoding="utf-8")
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_three_period(raw_summary)
        summary.update(
            {
                "initialCash": 80_000_000,
                "endingCash": (
                    80_000_000 + _integer(raw_summary["total_pnl"])
                ),
            }
        )
        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "entryMinute": (
                    _integer(row["entry_minute"])
                    if row["entry_minute"]
                    else None
                ),
                "entryPrice": _number(row["entry_price"]),
                "buyPrincipal": _integer(row["buy_principal"]),
                "exitReason": row["exit_reason"],
                "expiredQuantity": _integer(row["expired_quantity"]),
                "mfeMultiple": _number(row["mfe_multiple"]),
                "reached2x": _boolean(row["reached_2x"]),
                "reached5x": _boolean(row["reached_5x"]),
                "reached10x": _boolean(row["reached_10x"]),
            }
            for row in self._read_csv(self.strategy_three_daily_path)
        ]

        def candidate(raw: object) -> dict[str, object]:
            row = dict(raw)  # type: ignore[arg-type]
            return {
                "strategy": row["strategy"],
                "entry": row["entry"],
                "exitMode": row["exit_mode"],
                "all": self._strategy_three_period(row["all"]),
                "development": self._strategy_three_period(
                    row["development"]
                ),
                "validation": self._strategy_three_period(
                    row["validation"]
                ),
            }

        features = list(event["periods"]["all"]["features"])
        audit_document = dict(document["audit"])
        fixed_audit = dict(audit_document["fixed"])
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "scope": document["scope"],
            "rules": document["rules"],
            "summary": summary,
            "development": self._strategy_three_period(
                document["development"]
            ),
            "validation": self._strategy_three_period(
                document["validation"]
            ),
            "eventStudy": {
                "scope": event["scope"],
                "method": event["method"],
                "features": [
                    {
                        "feature": row["feature"],
                        "patternCount": _integer(row["pattern_count"]),
                        "fiveXCount": _integer(row["five_x_count"]),
                        "fiveXWithPattern": _integer(
                            row["five_x_with_pattern"]
                        ),
                        "fiveXCommonalityPct": _number(
                            row["five_x_commonality_pct"]
                        ),
                        "tenXCount": _integer(row["ten_x_count"]),
                        "tenXWithPattern": _integer(
                            row["ten_x_with_pattern"]
                        ),
                        "tenXCommonalityPct": _number(
                            row["ten_x_commonality_pct"]
                        ),
                        "nonFiveXPatternPct": _number(
                            row["non_five_x_pattern_pct"]
                        ),
                        "fiveXRateWhenPatternPct": _number(
                            row["five_x_rate_when_pattern_pct"]
                        ),
                        "tenXRateWhenPatternPct": _number(
                            row["ten_x_rate_when_pattern_pct"]
                        ),
                    }
                    for row in features
                ],
            },
            "entryCandidates": [
                candidate(row) for row in document["entry_candidates"]
            ],
            "exitCandidates": [
                candidate(row) for row in document["exit_candidates"]
            ],
            "daily": daily,
            "audit": {
                "passed": bool(audit_document["passed"]),
                "errorCount": _integer(fixed_audit["error_count"]),
                "sameEntriesPassed": bool(
                    audit_document[
                        "same_entries_across_exit_candidates"
                    ]["passed"]
                ),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                self.strategy_three_summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def strategy_three_v02(self) -> dict[str, object]:
        required = (
            self.strategy_three_v02_summary_path,
            self.strategy_three_v02_daily_path,
        )
        if any(not path.exists() for path in required):
            raise DataNotFoundError(
                "S3-v0.2 분석 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                "strategy3_v02_backtest.py`를 먼저 실행하세요."
            )
        document = json.loads(
            self.strategy_three_v02_summary_path.read_text(
                encoding="utf-8"
            )
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_three_period(raw_summary)
        summary.update(
            {
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
            }
        )
        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "buyPrincipal": _integer(row["buy_principal"]),
                "entryStages": _integer(row["entry_stages"]),
                "averageEntryPrice": _number(
                    row["average_entry_price"]
                ),
                "exitReason": row["exit_reason"],
                "expiredQuantity": _integer(row["expired_quantity"]),
                "mfeMultiple": _number(row["mfe_multiple"]),
                "firstGoldenCrossMinute": (
                    _integer(row["first_golden_cross_minute"])
                    if row["first_golden_cross_minute"]
                    else None
                ),
                "lowerLowMinute": (
                    _integer(row["lower_low_minute"])
                    if row["lower_low_minute"]
                    else None
                ),
                "secondGoldenCrossMinute": (
                    _integer(row["second_golden_cross_minute"])
                    if row["second_golden_cross_minute"]
                    else None
                ),
                "secondDeathCrossMinute": (
                    _integer(row["second_death_cross_minute"])
                    if row["second_death_cross_minute"]
                    else None
                ),
            }
            for row in self._read_csv(self.strategy_three_v02_daily_path)
        ]
        audit_document = dict(document["audit"])
        analysis = dict(document.get("analysis", {}))

        def analysis_groups(name: str) -> list[dict[str, object]]:
            return [
                {
                    "group": row["group"],
                    "count": _integer(row["count"]),
                    "profitableCount": _integer(
                        row["profitable_count"]
                    ),
                    "losingCount": _integer(row["losing_count"]),
                    "totalPnl": _integer(row["total_pnl"]),
                    "averagePnl": _integer(row["average_pnl"]),
                }
                for row in analysis.get(name, [])
            ]

        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "parentStrategy": document["parent_strategy"],
            "scope": document["scope"],
            "rules": document["rules"],
            "summary": summary,
            "development": self._strategy_three_period(
                document["development"]
            ),
            "validation": self._strategy_three_period(
                document["validation"]
            ),
            "comparisonToParent": document["comparison_to_s3_v01"],
            "analysis": {
                "byEntryStages": analysis_groups("by_entry_stages"),
                "byExitReason": analysis_groups("by_exit_reason"),
                "byCallPut": analysis_groups("by_call_put"),
                "losingDaysMfeAtLeast1_5x": _integer(
                    analysis.get("losing_days_mfe_at_least_1_5x")
                ),
                "losingDaysMfeAtLeast2x": _integer(
                    analysis.get("losing_days_mfe_at_least_2x")
                ),
            },
            "daily": daily,
            "audit": {
                "passed": bool(audit_document["passed"]),
                "errorCount": _integer(audit_document["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                self.strategy_three_v02_summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def strategy_three_v03(self) -> dict[str, object]:
        required = (
            self.strategy_three_v03_summary_path,
            self.strategy_three_v03_daily_path,
        )
        if any(not path.exists() for path in required):
            raise DataNotFoundError(
                "S3-v0.3 분석 결과가 없습니다. "
                "`python backend/analysis/option_5x/"
                "strategy3_v03_backtest.py`를 먼저 실행하세요."
            )
        document = json.loads(
            self.strategy_three_v03_summary_path.read_text(
                encoding="utf-8"
            )
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_three_period(raw_summary)
        summary.update(
            {
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
            }
        )

        def groups(name: str) -> list[dict[str, object]]:
            analysis = dict(document["analysis"])
            return [
                {
                    "group": row["group"],
                    "count": _integer(row["count"]),
                    "profitableCount": _integer(
                        row["profitable_count"]
                    ),
                    "losingCount": _integer(row["losing_count"]),
                    "totalPnl": _integer(row["total_pnl"]),
                    "averagePnl": _integer(row["average_pnl"]),
                }
                for row in analysis[name]
            ]

        def research_candidate(raw: object) -> dict[str, object]:
            row = dict(raw)  # type: ignore[arg-type]
            return {
                "strategy": row["strategy"],
                "entryMode": row["entry_mode"],
                "exitMode": row["exit_mode"],
                "all": self._strategy_three_period(row["all"]),
                "development": self._strategy_three_period(
                    row["development"]
                ),
                "validation": self._strategy_three_period(
                    row["validation"]
                ),
            }

        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "buyPrincipal": _integer(row["buy_principal"]),
                "entryStages": _integer(row["entry_stages"]),
                "averageEntryPrice": _number(
                    row["average_entry_price"]
                ),
                "patternQualified": _boolean(
                    row["pattern_qualified"]
                ),
                "patternRejectionReason": row[
                    "pattern_rejection_reason"
                ],
                "reached2x": _boolean(row["reached_2x"]),
                "principalRecovered": _boolean(
                    row["principal_recovered"]
                ),
                "runnerInitialQuantity": _integer(
                    row["runner_initial_quantity"]
                ),
                "exitReason": row["exit_reason"],
                "expiredQuantity": _integer(row["expired_quantity"]),
                "mfeMultiple": _number(row["mfe_multiple"]),
            }
            for row in self._read_csv(self.strategy_three_v03_daily_path)
        ]
        audit_document = dict(document["audit"])
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "parentStrategy": document["parent_strategy"],
            "scope": document["scope"],
            "rules": document["rules"],
            "summary": summary,
            "development": self._strategy_three_period(
                document["development"]
            ),
            "validation": self._strategy_three_period(
                document["validation"]
            ),
            "researchCandidates": [
                research_candidate(row)
                for row in document["research_candidates"]
            ],
            "comparison": document["comparison"],
            "analysis": {
                "byEntryStages": groups("by_entry_stages"),
                "byExitReason": groups("by_exit_reason"),
                "byPatternRejection": groups(
                    "by_pattern_rejection"
                ),
            },
            "daily": daily,
            "audit": {
                "passed": bool(audit_document["passed"]),
                "errorCount": _integer(audit_document["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                self.strategy_three_v03_summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def strategy_four(self) -> dict[str, object]:
        required = (
            self.futures_lsma_summary_path,
            self.strategy_four_summary_path,
            self.strategy_four_daily_path,
        )
        if any(not path.exists() for path in required):
            raise DataNotFoundError(
                "S4 선물 LSMA 분석 결과가 없습니다. "
                "`futures_lsma_research.py`와 "
                "`strategy4_lsma_futures_backtest.py`를 실행하세요."
            )
        document = json.loads(
            self.strategy_four_summary_path.read_text(encoding="utf-8")
        )
        futures_document = json.loads(
            self.futures_lsma_summary_path.read_text(encoding="utf-8")
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_three_period(raw_summary)
        summary.update(
            {
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
            }
        )

        def candidate(raw: object) -> dict[str, object]:
            row = dict(raw)  # type: ignore[arg-type]
            return {
                "strategy": row["strategy"],
                "baseStrategy": row["base_strategy"],
                "directionMode": row["direction_mode"],
                "unfilteredSignalDays": _integer(
                    row["unfiltered_signal_days"]
                ),
                "directionRejectedDays": _integer(
                    row["direction_rejected_days"]
                ),
                "all": self._strategy_three_period(row["all"]),
                "development": self._strategy_three_period(
                    row["development"]
                ),
                "validation": self._strategy_three_period(
                    row["validation"]
                ),
            }

        indicators = []
        for name, raw in futures_document["indicators"].items():
            indicators.append(
                {
                    "name": name,
                    "biasedTouchAccuracy": _number(
                        raw["biased"]["touch_accuracy"]
                    ),
                    "shiftedTouchAccuracy": _number(
                        raw["shifted"]["touch_accuracy"]
                    ),
                    "shiftedEndpointDirectionAccuracy": _number(
                        raw["shifted"][
                            "endpoint_direction_accuracy"
                        ]
                    ),
                    "highVolTouchAccuracy": _number(
                        raw["shifted_high_volatility_period"][
                            "touch_accuracy"
                        ]
                    ),
                    "highVolEndpointDirectionAccuracy": _number(
                        raw["shifted_high_volatility_period"][
                            "endpoint_direction_accuracy"
                        ]
                    ),
                    "overstatement": _number(
                        raw["touch_accuracy_overstatement"]
                    ),
                }
            )

        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "buyPrincipal": _integer(row["buy_principal"]),
                "entryStages": _integer(row["entry_stages"]),
                "exitReason": row["exit_reason"],
                "futuresOpen": _number(row["futures_open"]),
                "shiftedLsma": _number(row["shifted_lsma"]),
                "lsmaDistancePct": _number(
                    row["lsma_distance_pct"]
                ),
                "lsmaReferenceWeek": row["lsma_reference_week"],
                "lsmaLatestInputWeek": row[
                    "lsma_latest_input_week"
                ],
                "lsmaTowardDirection": row[
                    "lsma_toward_direction"
                ],
                "lsmaSlopeDirection": row[
                    "lsma_slope_direction"
                ],
            }
            for row in self._read_csv(self.strategy_four_daily_path)
        ]
        audit_document = dict(document["audit"])
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "baseStrategy": document["base_strategy"],
            "scope": document["scope"],
            "hypothesis": document["hypothesis"],
            "directionCounts": document["direction_counts"],
            "summary": summary,
            "development": self._strategy_three_period(
                document["development"]
            ),
            "validation": self._strategy_three_period(
                document["validation"]
            ),
            "paperHypothesisTest": {
                "strategyVersion": document[
                    "paper_hypothesis_test"
                ]["strategy_version"],
                "baseStrategy": document["paper_hypothesis_test"][
                    "base_strategy"
                ],
                "rule": document["paper_hypothesis_test"]["rule"],
                "all": self._strategy_three_period(
                    document["paper_hypothesis_test"]["all"]
                ),
                "development": self._strategy_three_period(
                    document["paper_hypothesis_test"]["development"]
                ),
                "validation": self._strategy_three_period(
                    document["paper_hypothesis_test"]["validation"]
                ),
            },
            "candidates": [
                candidate(row) for row in document["candidates"]
            ],
            "comparison": document["comparison"],
            "futuresStudy": {
                "dataProfile": futures_document["data_profile"],
                "weeklyRows": _integer(
                    futures_document["weekly_rows"]
                ),
                "weeklyStart": futures_document["weekly_start"],
                "weeklyEnd": futures_document["weekly_end"],
                "indicators": indicators,
                "interpretationBoundary": futures_document[
                    "interpretation_boundary"
                ],
            },
            "daily": daily,
            "audit": {
                "passed": bool(audit_document["passed"]),
                "errorCount": _integer(audit_document["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                self.strategy_four_summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    @staticmethod
    def _strategy_five_period(raw: object) -> dict[str, object]:
        period = dict(raw)  # type: ignore[arg-type]
        result = StrategyAnalysisService._strategy_three_period(period)
        result.update(
            {
                "breakEvenDays": _integer(period.get("break_even_days")),
                "confirmedDays": _integer(period.get("confirmed_days")),
                "reached2xDays": _integer(
                    period.get("reached_2x_days")
                ),
                "reached5xDays": _integer(
                    period.get("reached_5x_days")
                ),
                "reached10xDays": _integer(
                    period.get("reached_10x_days")
                ),
            }
        )
        return result

    def strategy_five(self) -> dict[str, object]:
        required = (
            self.strategy_five_summary_path,
            self.strategy_five_daily_path,
            self.strategy_five_trades_path,
        )
        if any(not path.exists() for path in required):
            raise DataNotFoundError(
                "S5 PPT 모멘텀 분석 결과가 없습니다. "
                "`strategy5_ppt_confirmation_backtest.py`를 실행하세요."
            )
        document = json.loads(
            self.strategy_five_summary_path.read_text(encoding="utf-8")
        )
        raw_summary = dict(document["summary"])
        summary = self._strategy_five_period(raw_summary)
        summary.update(
            {
                "initialCash": _integer(raw_summary["initial_cash"]),
                "endingCash": _integer(raw_summary["ending_cash"]),
            }
        )
        daily = [
            {
                "date": row["date"],
                "startCash": _integer(row["start_cash"]),
                "endCash": _integer(row["end_cash"]),
                "pnl": _integer(row["pnl"]),
                "signalFound": _boolean(row["signal_found"]),
                "entryFilled": _boolean(row["entry_filled"]),
                "code": row["code"],
                "callPut": row["call_put"],
                "signalMinute": (
                    _integer(row["signal_minute"])
                    if row["signal_minute"]
                    else None
                ),
                "observedPrice": _number(row["observed_price"]),
                "entryMinute": (
                    _integer(row["entry_minute"])
                    if row["entry_minute"]
                    else None
                ),
                "initialPrice": _number(row["initial_price"]),
                "entryQuantity": _integer(row["entry_quantity"]),
                "buyPrincipal": _integer(row["buy_principal"]),
                "confirmed": _boolean(row["confirmed"]),
                "confirmationMinute": (
                    _integer(row["confirmation_minute"])
                    if row["confirmation_minute"]
                    else None
                ),
                "reached2x": _boolean(row["reached_2x"]),
                "reached5x": _boolean(row["reached_5x"]),
                "reached10x": _boolean(row["reached_10x"]),
                "exitReason": row["exit_reason"],
                "exitMinute": (
                    _integer(row["exit_minute"])
                    if row["exit_minute"]
                    else None
                ),
                "expiredQuantity": _integer(row["expired_quantity"]),
                "mfeMultiple": _number(row["mfe_multiple"]),
                "maePct": _number(row["mae_pct"]),
            }
            for row in self._read_csv(self.strategy_five_daily_path)
        ]

        def groups(name: str) -> list[dict[str, object]]:
            return [
                {
                    "group": str(row["group"]),
                    "count": _integer(row["count"]),
                    "profitableCount": _integer(
                        row["profitable_count"]
                    ),
                    "losingCount": _integer(row["losing_count"]),
                    "totalPnl": _integer(row["total_pnl"]),
                    "averagePnl": _integer(row["average_pnl"]),
                }
                for row in document["analysis"][name]
            ]

        audit_document = dict(document["audit"])
        return {
            "status": document["status"],
            "strategyVersion": document["strategy_version"],
            "source": document["source"],
            "scope": document["scope"],
            "rules": document["rules"],
            "summary": summary,
            "development": self._strategy_five_period(
                document["development"]
            ),
            "validation": self._strategy_five_period(
                document["validation"]
            ),
            "analysis": {
                "byConfirmation": groups("by_confirmation"),
                "byCallPut": groups("by_call_put"),
                "byExitReason": groups("by_exit_reason"),
            },
            "daily": daily,
            "audit": {
                "passed": bool(audit_document["passed"]),
                "errorCount": _integer(audit_document["error_count"]),
            },
            "limitations": document["limitations"],
            "generatedAt": datetime.fromtimestamp(
                self.strategy_five_summary_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }

    def _trade_sources(self) -> dict[str, tuple[str, Path, Path]]:
        return {
            "S1-v0.1": (
                "S1-v0.1 기존 분할매수",
                self.output_dir / "strategy1_daily.csv",
                self.output_dir / "strategy1_trades.csv",
            ),
            "S2-v0.1": (
                "S2-v0.1 탐색·확인 증액",
                self.strategy_two_daily_path,
                self.output_dir / "strategy2_trades.csv",
            ),
            "S2-v0.2": (
                "S2-v0.2 MA20 상승",
                self.strategy_two_v02_daily_path,
                self.output_dir / "strategy2_v02_trades.csv",
            ),
            "S2-v0.3": (
                "S2-v0.3 확인별 목표",
                self.strategy_two_v03_daily_path,
                self.output_dir / "strategy2_v03_trades.csv",
            ),
            "S3-v0.1": (
                "S3-v0.1 저점 회복 돌파",
                self.strategy_three_daily_path,
                self.output_dir / "strategy3_trades.csv",
            ),
            "S3-v0.2": (
                "S3-v0.2 쌍바닥·쌍고점",
                self.strategy_three_v02_daily_path,
                self.strategy_three_v02_trades_path,
            ),
            "S3-v0.3": (
                "S3-v0.3 확인증액·원금회수",
                self.strategy_three_v03_daily_path,
                self.strategy_three_v03_trades_path,
            ),
            "S4-v0.1": (
                "S4-v0.1 LSMA 평균회귀 게이트",
                self.strategy_four_v01_daily_path,
                self.strategy_four_v01_trades_path,
            ),
            "S4-v0.2": (
                "S4-v0.2 LSMA 추세 정렬",
                self.strategy_four_daily_path,
                self.strategy_four_trades_path,
            ),
            "S5-v0.2": (
                "S5-v0.2 PPT 확인 후 매수",
                self.strategy_five_daily_path,
                self.strategy_five_trades_path,
            ),
        }

    def strategy_trade_records(
        self, strategy_version: str, losses_only: bool = True
    ) -> dict[str, object]:
        sources = self._trade_sources()
        if strategy_version not in sources:
            raise ValueError(f"지원하지 않는 전략: {strategy_version}")
        label, daily_path, trades_path = sources[strategy_version]
        if not daily_path.exists() or not trades_path.exists():
            raise DataNotFoundError(
                f"{strategy_version} 거래 결과 파일이 없습니다."
            )
        daily_rows = self._read_csv(daily_path)
        trade_days = [
            row
            for row in daily_rows
            if _boolean(row.get("entry_filled"))
            or _boolean(row.get("trade_day"))
        ]
        profitable_days = sum(
            _integer(row.get("pnl")) > 0 for row in trade_days
        )
        losing_days = sum(
            _integer(row.get("pnl")) < 0 for row in trade_days
        )
        break_even_days = len(trade_days) - profitable_days - losing_days
        call_put_by_key = {
            (row.get("date", ""), row.get("code", "")): row.get(
                "call_put", ""
            )
            for row in daily_rows
        }
        grouped: dict[
            tuple[str, str, str], list[dict[str, str]]
        ] = defaultdict(list)
        for row in self._read_csv(trades_path):
            key = (
                row["date"],
                row["code"],
                row.get("campaign", "1") or "1",
            )
            grouped[key].append(row)

        records: list[dict[str, object]] = []
        for (date_value, code, campaign), rows in grouped.items():
            ordered = sorted(
                rows,
                key=lambda row: (
                    _integer(row["minute"]),
                    0 if row["side"] == "BUY" else 1,
                ),
            )
            pnl = sum(_integer(row["cash_flow"]) for row in ordered)
            if losses_only and pnl >= 0:
                continue
            actionable = [
                row
                for row in ordered
                if row["side"] in {"BUY", "SELL"}
            ]
            if not actionable:
                continue
            call_put = (
                ordered[0].get("call_put")
                or call_put_by_key.get((date_value, code), "")
            )
            records.append(
                {
                    "id": (
                        f"{strategy_version}:{date_value}:"
                        f"{code}:{campaign}"
                    ),
                    "date": date_value,
                    "code": code,
                    "callPut": call_put,
                    "campaign": _integer(campaign),
                    "pnl": pnl,
                    "buyPrincipal": sum(
                        _integer(row["principal"])
                        for row in ordered
                        if row["side"] == "BUY"
                    ),
                    "fees": sum(
                        _integer(row["fee"]) for row in ordered
                    ),
                    "firstMinute": min(
                        _integer(row["minute"]) for row in actionable
                    ),
                    "lastMinute": max(
                        _integer(row["minute"]) for row in actionable
                    ),
                    "trades": [
                        {
                            "minute": _integer(row["minute"]),
                            "side": row["side"],
                            "reason": row["reason"],
                            "stage": _integer(row.get("stage")),
                            "quantity": _integer(row["quantity"]),
                            "price": _number(row["price"]),
                            "principal": _integer(row["principal"]),
                            "fee": _integer(row["fee"]),
                            "cashFlow": _integer(row["cash_flow"]),
                        }
                        for row in ordered
                    ],
                }
            )
        records.sort(
            key=lambda row: (
                str(row["date"]),
                _integer(row["firstMinute"]),
            ),
            reverse=True,
        )
        return {
            "strategyVersion": strategy_version,
            "strategyLabel": label,
            "lossesOnly": losses_only,
            "count": len(records),
            "summary": {
                "tradeDays": len(trade_days),
                "profitableDays": profitable_days,
                "losingDays": losing_days,
                "breakEvenDays": break_even_days,
                "winRatePct": round(
                    profitable_days / len(trade_days) * 100, 2
                )
                if trade_days
                else 0.0,
            },
            "records": records,
            "strategies": [
                {"version": version, "label": source[0]}
                for version, source in sources.items()
                if source[1].exists() and source[2].exists()
            ],
        }
