from __future__ import annotations

import math
import unittest

from backend.app.performance_metrics import calculate_performance_statistics


class PerformanceMetricsTest(unittest.TestCase):
    def test_paper_formulas_use_population_deviation(self) -> None:
        result = calculate_performance_statistics(
            returns=[0.1, -0.1],
            equities=[100.0, 110.0, 99.0],
            market_returns=[0.05, -0.05],
            initial_equity=100.0,
            ending_equity=99.0,
            annualization_factor=2,
        )

        self.assertAlmostEqual(result["annualized_return"], -0.01)
        self.assertAlmostEqual(result["daily_return_deviation"], 0.1)
        self.assertAlmostEqual(
            result["annualized_volatility"],
            0.1 * math.sqrt(2),
        )
        self.assertAlmostEqual(result["lpm1"], 0.05)
        self.assertAlmostEqual(result["lpm2"], 0.005)
        self.assertAlmostEqual(result["omega_ratio"], 1.0)
        self.assertAlmostEqual(result["beta"], 2.0)
        self.assertAlmostEqual(result["maximum_drawdown"], -0.1)
        self.assertAlmostEqual(
            result["sharpe_ratio"],
            -0.01 / (0.1 * math.sqrt(2)),
        )
        self.assertAlmostEqual(result["sortino_ratio"], -0.1)

    def test_undefined_ratios_are_none_not_infinity(self) -> None:
        result = calculate_performance_statistics(
            returns=[0.0, 0.0],
            equities=[100.0, 100.0, 100.0],
            market_returns=[0.0, 0.0],
            initial_equity=100.0,
            ending_equity=100.0,
        )

        self.assertIsNone(result["sharpe_ratio"])
        self.assertIsNone(result["omega_ratio"])
        self.assertIsNone(result["sortino_ratio"])
        self.assertIsNone(result["beta"])
        self.assertEqual(result["maximum_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
