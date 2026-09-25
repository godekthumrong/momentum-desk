import unittest

import numpy as np
import pandas as pd

from mean_reversion_short_backtest import (
    Candidate,
    _tiered_margin_interest,
    _trade_commission,
    indicators,
    select_top_orders,
    simulate,
    wilder,
)


class MeanReversionShortTests(unittest.TestCase):
    def test_wilder_uses_sma_seed(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        got = wilder(s, 3)
        self.assertTrue(np.isnan(got.iloc[1]))
        self.assertAlmostEqual(got.iloc[2], 2.0)
        self.assertAlmostEqual(got.iloc[3], (2.0 * 2 + 4.0) / 3)

    def test_top_ten_ranked_before_fill(self):
        rows = []
        for i in range(12):
            rows.append(Candidate(str(i), "2024-01-02", "2024-01-03", 100-i, 1, 10,
                                  i >= 2, 10 if i >= 2 else None, 9, "2024-01-04" if i >= 2 else None,
                                  9 if i >= 2 else None, "time_exit" if i >= 2 else None,
                                  "close" if i >= 2 else None, 9 if i >= 2 else None))
        selected, count = select_top_orders(rows)
        self.assertEqual(count, 12)
        self.assertEqual(len(selected), 10)
        self.assertFalse(selected[0].filled)
        self.assertFalse(selected[1].filled)
        self.assertNotIn("10", {x.symbol for x in selected})

    def test_simulator_respects_full_exposure_cap(self):
        rows = []
        for i in range(12):
            rows.append(Candidate(str(i), "2024-01-02", "2024-01-03", 100-i, 0.2, 10,
                                  True, 10, 9, "2024-01-04", 9, "time_exit", "close", 9.5))
        orders, _ = select_top_orders(rows)
        curve, trades, metrics = simulate(orders, ["2024-01-03", "2024-01-04"], 0)
        self.assertLessEqual(curve["max_gross"] .max() if "max_gross" in curve else metrics["max_gross_exposure"], 1.0000001)
        self.assertEqual(len(trades), 10)

    def test_ibkr_tiered_commission_minimum_and_per_share_rate(self):
        self.assertAlmostEqual(_trade_commission(100, 50, 0, True), 0.35)
        self.assertAlmostEqual(_trade_commission(1000, 50, 0, True), 3.50)

    def test_tiered_margin_interest_actual_360(self):
        expected = 100_000 * 0.0513 + 50_000 * 0.0463
        self.assertAlmostEqual(_tiered_margin_interest(150_000, 360), expected)

    def test_ibkr_costs_use_calendar_days_and_reconcile(self):
        order = Candidate(
            "XYZ", "2024-01-04", "2024-01-05", 90, 1, 10,
            True, 10, 10, "2024-01-08", 9, "time_exit", "close", 9.5,
        )
        curve, trades, metrics = simulate(
            [order], ["2024-01-05", "2024-01-08"], 0,
            ibkr_tiered=True, ibkr_margin_interest=True,
            annual_borrow_rate=0.0025,
        )
        trade = trades.iloc[0]
        overnight_notional = trade["entry_notional"]
        expected_margin = overnight_notional * 0.0513 * 3 / 360
        expected_borrow = overnight_notional * 0.0025 * 3 / 360
        self.assertAlmostEqual(trade["margin_interest"], expected_margin)
        self.assertAlmostEqual(trade["borrow_cost"], expected_borrow)
        self.assertAlmostEqual(metrics["total_commissions"], 5.6)
        self.assertAlmostEqual(
            curve["equity"].iloc[-1] - 100_000,
            trades["net_pnl"].sum(),
        )

    def test_external_long_sleeve_compounds_combined_equity(self):
        returns = pd.Series({"2024-01-02": 0.0, "2024-01-03": 0.10})
        curve, trades, metrics = simulate(
            [], ["2024-01-02", "2024-01-03"], 0,
            external_daily_returns=returns, external_label="Momentum long",
        )
        self.assertTrue(trades.empty)
        self.assertAlmostEqual(curve["equity"].iloc[-1], 110_000)
        self.assertAlmostEqual(metrics["total_external_pnl"], 10_000)
        self.assertEqual(metrics["external_label"], "Momentum long")


if __name__ == "__main__":
    unittest.main()
