from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "option_5x"))
import minimal_baseline_backtest as base
from s8_live_research import POLICIES, simulate_day


class S8LiveResearchTest(unittest.TestCase):
    def prepared(self):
        rows = [
            (900, .6, .6, .6, .6),
            (911, .5, .55, .49, .5),
            (912, .5, .6, .49, .5),
            (913, .49, .5, .3, .3),
            (914, .1, .15, .09, .1),
            (915, .1, .1, .1, .1),
        ]
        contract = base.build_contract(
            "C", {"name": "C", "call_put": "CALL"},
            {minute: base.RawBar(minute, opening, high, low, close, 1000)
             for minute, opening, high, low, close in rows},
        )
        return [{
            "code": "C", "contract": contract, "p0": .6,
            "signal_idx": base.MINUTE_INDEX[911], "own_half_idx": None,
            "ma120": [None] * len(contract.bars),
        }]

    def test_gap_below_stop_is_sold_at_next_open_not_stop_price(self):
        cash, result = simulate_day(self.prepared(), POLICIES[1], 2, 1, 20_000_000)
        # 09:12에 0.51로 2계약, 09:13 종가 손절 확인 뒤 09:14 0.09에 매도.
        self.assertEqual(result["pnl"], -212_000)
        self.assertEqual(cash, 19_788_000)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["contracts_bought"], 2)
        self.assertEqual(result["extra_roundtrip_tick_cost"], 10_000)

    def test_no_purchase_when_one_contract_exceeds_available_cash(self):
        cash, result = simulate_day(self.prepared(), POLICIES[1], 2, 1, 100)
        self.assertEqual(cash, 100)
        self.assertEqual(result["trades"], 0)

    def test_cost_stress_includes_both_buy_and_market_sell(self):
        _, normal = simulate_day(self.prepared(), POLICIES[1], 2, 1, 20_000_000)
        _, stressed = simulate_day(self.prepared(), POLICIES[1], 2, 2, 20_000_000)
        self.assertEqual(stressed["pnl"], -222_000)
        self.assertLess(stressed["pnl"], normal["pnl"])


if __name__ == "__main__":
    unittest.main()
