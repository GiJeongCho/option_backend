from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "option_5x"
sys.path.insert(0, str(ANALYSIS_DIR))
import minimal_baseline_backtest as base
import strategy6_v01_pair_half_backtest as pair


class S8P0PairingTest(unittest.TestCase):
    @staticmethod
    def candidate(code, side, p0):
        return (code, SimpleNamespace(call_put=side), p0, 0)

    @staticmethod
    def contract(code, side, prices):
        raw = {
            minute: base.RawBar(minute, opening, high, low, close, 100)
            for minute, opening, high, low, close in prices
        }
        return base.build_contract(code, {"name": code, "call_put": side}, raw)

    def test_exact_p0_has_priority_over_other_opposite_options(self):
        own = self.candidate("CALL", "CALL", 0.4)
        close = self.candidate("NEAR", "PUT", 0.41)
        exact = self.candidate("EXACT", "PUT", 0.4)
        self.assertEqual(pair.match_opposite_p0(own, [own, close, exact]), exact)

    def test_nearest_tie_is_fixed_without_using_later_signal(self):
        own = self.candidate("CALL", "CALL", 0.5)
        lower = self.candidate("LOW", "PUT", 0.49)
        higher = self.candidate("HIGH", "PUT", 0.51)
        self.assertEqual(pair.match_opposite_p0(own, [own, higher, lower]), lower)

    def test_same_direction_cannot_be_used_as_opposite(self):
        own = self.candidate("CALL", "CALL", 0.4)
        other_call = self.candidate("OTHER_CALL", "CALL", 0.4)
        self.assertIsNone(pair.match_opposite_p0(own, [own, other_call]))

    def test_matching_signal_is_saved_and_fill_is_after_confirmation(self):
        own = self.contract("CALL", "CALL", [
            (900, .4, .4, .4, .4),
            (911, .3, .32, .3, .3),
            (912, .31, .33, .3, .32),
        ])
        opposite = self.contract("PUT", "PUT", [
            (900, .42, .42, .42, .42),
            (911, .22, .22, .2, .2),
            (912, .2, .2, .19, .19),
        ])
        group = {"expiry_date": "20260112", "product_kind": "MONDAY", "series": "2601W2"}
        with patch.object(base, "load_expiry_day", return_value={"CALL": own, "PUT": opposite}):
            _, positions = pair.build_day(group, Path("unused"), None)
        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertEqual(position["code"], "CALL")
        self.assertEqual(position["signal_code"], "PUT")
        self.assertEqual(position["signal_minute"], 911)
        self.assertEqual(position["fill_minute"], 912)
        self.assertEqual(position["pair_series"], group["series"])
        self.assertAlmostEqual(position["fill_price"], .32)

    def test_unmatched_atm_signal_does_not_trigger_own_purchase(self):
        own = self.contract("CALL", "CALL", [
            (900, .4, .4, .4, .4),
            (911, .3, .32, .3, .3),
            (912, .31, .33, .3, .32),
        ])
        near = self.contract("NEAR", "PUT", [
            (900, .42, .42, .42, .42),
            (911, .42, .42, .4, .41),
            (912, .41, .42, .4, .41),
        ])
        far = self.contract("FAR", "PUT", [
            (900, 1.2, 1.2, 1.2, 1.2),
            (911, .6, .6, .5, .5),
            (912, .5, .5, .49, .49),
        ])
        group = {"expiry_date": "20260112", "product_kind": "MONDAY", "series": "2601W2"}
        with patch.object(base, "load_expiry_day", return_value={"CALL": own, "NEAR": near, "FAR": far}):
            _, positions = pair.build_day(group, Path("unused"), None)
        self.assertFalse(any(position["code"] == "CALL" for position in positions))

    def test_exact_p0_from_another_expiry_cannot_replace_same_expiry_pair(self):
        source = io.StringIO()
        fields = ["시장ID", "종목명", "기준시각", "시가", "고가", "저가", "종가", "거래량", "종목코드", "콜풋구분"]
        writer = csv.DictWriter(source, fieldnames=fields)
        writer.writeheader()
        for code, name, side, p0, later in [
            ("CALL", "코스피위클리M C 2601W2 680.0", "CALL", .4, .3),
            ("SAME_EXPIRY", "코스피위클리M P 2601W2 670.0", "PUT", .42, .2),
            ("WRONG_EXPIRY", "코스피위클리 P 2601W3 647.5", "PUT", .4, .2),
        ]:
            for minute, price in [(900, p0), (911, later), (912, later)]:
                writer.writerow({
                    "시장ID": "DRV", "종목명": name, "종목코드": code,
                    "콜풋구분": side, "기준시각": str(minute),
                    "시가": price, "고가": price, "저가": price,
                    "종가": price, "거래량": 100,
                })
        path = Mock()
        path.open.side_effect = lambda *args, **kwargs: io.StringIO(source.getvalue())
        group = {"expiry_date": "20260112", "product_kind": "MONDAY", "series": "2601W2"}
        _, positions = pair.build_day(group, path, None)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["signal_code"], "SAME_EXPIRY")
        self.assertEqual(positions[0]["pair_rule"], "nearest_p0")


class S8GapAdjustedMATest(unittest.TestCase):
    contract = staticmethod(S8P0PairingTest.contract)

    def test_previous_close_is_aligned_to_day_open_not_day_close(self):
        prior = self.contract("X", "CALL", [(900, .2, .2, .2, .2)])
        current = self.contract("X", "CALL", [(900, .4, .8, .4, .8)])
        ma = pair.ma120_for_day(current, prior)
        self.assertAlmostEqual(ma[pair.IDX[900]], (119 * .4 + .8) / 120)
        legacy = pair.ma120_for_day(current, prior, legacy_close_anchor=True)
        self.assertAlmostEqual(legacy[pair.IDX[900]], .8)
        self.assertIsNone(ma[pair.IDX[859]])

    def test_premarket_price_cannot_seed_missing_day_opening_minutes(self):
        prior = self.contract("X", "CALL", [(900, .2, .2, .2, .2)])
        current = self.contract("X", "CALL", [(845, 99, 99, 99, 99), (901, .4, .8, .4, .8)])
        ma = pair.ma120_for_day(current, prior)
        self.assertIsNone(ma[pair.IDX[900]])
        self.assertIsNone(ma[pair.IDX[901]])
        self.assertIsNone(ma[pair.IDX[1059]])
        self.assertAlmostEqual(ma[pair.IDX[1100]], .8)

    def test_crossover_uses_gap_adjusted_prior_minutes(self):
        prior = self.contract("X", "CALL", [(900, 100, 100, 100, 100)])
        current = self.contract("X", "CALL", [
            (900, .4, .4, .4, .4), (901, .3, .3, .3, .3), (902, .5, .5, .5, .5),
        ])
        ma = pair.ma120_for_day(current, prior)
        self.assertEqual(pair.ma_cross_up(current, ma, pair.IDX[901]), pair.IDX[902])


if __name__ == "__main__":
    unittest.main()
