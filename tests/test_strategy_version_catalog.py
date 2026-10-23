from __future__ import annotations

import json
import unittest

from backend.app.strategy_analysis_service import StrategyAnalysisService


class StrategyVersionCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = StrategyAnalysisService()
        cls.catalog = cls.service.strategy_version_catalog()

    def test_highest_total_return_is_default(self) -> None:
        strategies = self.catalog["strategies"]
        highest = max(
            strategies,
            key=lambda item: item["summary"]["totalReturnPct"],
        )

        self.assertEqual(
            self.catalog["defaultStrategyVersion"],
            highest["version"],
        )
        self.assertEqual(highest["returnRank"], 1)

    def test_every_saved_version_has_distinct_detail(self) -> None:
        strategies = self.catalog["strategies"]
        self.assertGreaterEqual(len(strategies), 20)

        for item in strategies:
            with self.subTest(strategy=item["version"]):
                detail = self.service.strategy_version_detail(
                    item["version"]
                )
                self.assertEqual(
                    detail["strategyVersion"],
                    item["version"],
                )
                self.assertTrue(detail["metadata"]["summary"])
                self.assertTrue(detail["metadata"]["entry"])
                self.assertTrue(detail["metadata"]["exit"])
                self.assertEqual(
                    detail["summary"]["endingCash"],
                    detail["summary"]["initialCash"]
                    + detail["summary"]["totalPnl"],
                )

    def test_s5_v08_exact_quote_variant_is_available(self) -> None:
        version = "S5-v0.8-STEP10-10X"
        detail = self.service.strategy_version_detail(version)

        self.assertEqual(detail["strategyVersion"], version)
        self.assertEqual(detail["summary"]["initialCash"], 40_000_000)
        self.assertIn("0.4·0.5·…·1.2", detail["metadata"]["summary"])

    def test_s5_v09_strike10_variant_is_available(self) -> None:
        version = "S5-v0.9-STRIKE10-10X"
        detail = self.service.strategy_version_detail(version)

        self.assertEqual(detail["strategyVersion"], version)
        self.assertEqual(detail["summary"]["initialCash"], 40_000_000)
        self.assertIn("행사가격", detail["metadata"]["summary"])
        self.assertIn("0.40", detail["metadata"]["summary"])
        records = self.service.strategy_trade_records(
            version,
            losses_only=False,
        )
        self.assertGreater(records["count"], 0)
        levels = records["records"][0]["strategyContext"]["levels"]
        self.assertIn("TARGET_10X", {level["key"] for level in levels})

    def test_s6_v03_research_candidate_is_available(self) -> None:
        version = "S6-v0.3"
        detail = self.service.strategy_version_detail(version)

        self.assertEqual(detail["strategyVersion"], version)
        self.assertEqual(detail["summary"]["initialCash"], 20_000_000)
        self.assertIn("독립검증 전", detail["metadata"]["status"])
        self.assertIn("RV60", detail["metadata"]["summary"])

    def test_every_strategy_reaches_latest_completed_expiry(self) -> None:
        refresh = json.loads(
            (
                self.service.output_dir
                / "refresh_all_backtests_summary.json"
            ).read_text(encoding="utf-8")
        )
        latest = refresh["latest_completed_expiry"]

        for item in self.catalog["strategies"]:
            with self.subTest(strategy=item["version"]):
                detail = self.service.strategy_version_detail(
                    item["version"]
                )
                self.assertEqual(detail["summary"]["period"]["end"], latest)

    def test_comparison_normalizes_each_strategy_to_its_own_seed(self) -> None:
        versions = ["S3-v0.1", "S5-v0.7-04BIN-10X"]
        comparison = self.service.strategy_version_comparison(versions)

        self.assertEqual(comparison["selectedCount"], 2)
        self.assertEqual(
            [series["version"] for series in comparison["series"]],
            versions,
        )
        for series in comparison["series"]:
            with self.subTest(strategy=series["version"]):
                self.assertTrue(series["points"])
                self.assertAlmostEqual(
                    series["points"][-1]["returnPct"],
                    series["totalReturnPct"],
                    places=5,
                )
                self.assertEqual(
                    series["endingCash"],
                    series["initialCash"] + series["totalPnl"],
                )

    def test_every_strategy_has_a_detailed_logic_document(self) -> None:
        for item in self.catalog["strategies"]:
            version = item["version"]
            with self.subTest(strategy=version):
                logic = self.service.strategy_version_logic(version)
                document = logic["document"]

                self.assertTrue(document["available"])
                self.assertTrue(document["name"])
                self.assertTrue(document["content"])
                self.assertIn(version, document["content"])
                self.assertTrue(logic["metadata"]["entry"])
                self.assertTrue(logic["metadata"]["exit"])

    def test_seed_backtest_keeps_cash_nonnegative(self) -> None:
        result = self.service.strategy_seed_backtest(
            "S3-v0.2",
            20_000_000,
        )
        daily = result["daily"]

        self.assertEqual(result["summary"]["initialCash"], 20_000_000)
        self.assertGreater(result["execution"]["skippedTradeDays"], 0)
        self.assertTrue(all(row["endCash"] >= 0 for row in daily))
        self.assertTrue(
            all(
                left["endCash"] == right["startCash"]
                for left, right in zip(daily, daily[1:])
            )
        )
        self.assertTrue(
            all(
                row["availableCash"] < row["requiredCash"]
                for row in result["execution"]["skippedDates"]
            )
        )

    def test_seed_backtest_preserves_pnl_when_cash_is_sufficient(
        self,
    ) -> None:
        result = self.service.strategy_seed_backtest(
            "S5-v0.7-04BIN-10X",
            20_000_000,
        )
        original = self.service.strategy_version_detail(
            "S5-v0.7-04BIN-10X"
        )

        self.assertEqual(result["execution"]["skippedTradeDays"], 0)
        self.assertEqual(
            result["summary"]["totalPnl"],
            original["summary"]["totalPnl"],
        )
        self.assertEqual(
            result["summary"]["endingCash"],
            20_000_000 + original["summary"]["totalPnl"],
        )


if __name__ == "__main__":
    unittest.main()
