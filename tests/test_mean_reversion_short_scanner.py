import unittest
from unittest.mock import patch

import pandas as pd

from mean_reversion_short_scanner import build_html, evaluate_exit, scan_frame


class DailyShortScannerTests(unittest.TestCase):
    def test_signal_uses_raw_order_levels_and_position_cap(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        adjusted = pd.DataFrame(
            {"open": 50.0, "high": 52.0, "low": 49.0, "close": 50.0, "volume": 1_000_000.0},
            index=dates,
        )
        raw = adjusted.copy()
        raw.loc[dates[-1], "close"] = 100.0
        fake_indicators = pd.DataFrame(
            {"atr10": 4.0, "atr_pct10": 8.0, "adx7": 60.0, "rsi3": 90.0, "two_up": True},
            index=dates,
        )
        with patch("mean_reversion_short_scanner.indicators", return_value=fake_indicators):
            row = scan_frame(
                "XYZ", adjusted, raw, dates[-1],
                {"shortable": True, "easy_to_borrow": True},
            )
        self.assertIsNotNone(row)
        self.assertEqual(row["limit_price"], 100.0)
        # Adjusted ATR is scaled to the current raw-price level: 4 * 100 / 50.
        self.assertEqual(row["atr10_raw"], 8.0)
        self.assertEqual(row["estimated_stop"], 120.0)
        self.assertLessEqual(row["allocation_pct"], 10.0)
        self.assertEqual(row["borrow_status"], "ready")

    def test_non_shortable_signal_is_visible_but_blocked(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        frame = pd.DataFrame(
            {"open": 20.0, "high": 22.0, "low": 19.0, "close": 20.0, "volume": 900_000.0},
            index=dates,
        )
        fake_indicators = pd.DataFrame(
            {"atr10": 1.2, "atr_pct10": 6.0, "adx7": 51.0, "rsi3": 86.0, "two_up": True},
            index=dates,
        )
        with patch("mean_reversion_short_scanner.indicators", return_value=fake_indicators):
            row = scan_frame("XYZ", frame, frame, dates[-1], {"shortable": False})
        self.assertEqual(row["borrow_status"], "blocked")

    def test_empty_page_has_clear_message(self):
        payload = {"meta": {"as_of": "2024-01-31", "portfolio_value": 100000,
                            "signals": 0, "selected": 0, "ready": 0,
                            "fresh": 5000, "universe": 5100}, "signals": []}
        page = build_html(payload)
        self.assertIn("No signals match every rule", page)
        self.assertIn("daily short scanner", page.lower())

    def test_exit_rules_match_backtest_order(self):
        profit = evaluate_exit(
            "2024-01-02", 100, 4,
            [["2024-01-02", 100, 102, 94, 95], ["2024-01-03", 94, 96, 92, 93]],
        )
        self.assertEqual(profit["label"], "Profit target · cover at open")
        self.assertEqual(profit["exit_price"], 94)

        stopped = evaluate_exit(
            "2024-01-02", 100, 4,
            [["2024-01-02", 100, 105, 98, 101], ["2024-01-03", 105, 111, 103, 106]],
        )
        self.assertEqual(stopped["label"], "Stop hit intraday")
        self.assertEqual(stopped["exit_price"], 110)

        timed = evaluate_exit(
            "2024-01-02", 100, 4,
            [["2024-01-02", 100, 105, 98, 101], ["2024-01-03", 101, 106, 97, 99]],
        )
        self.assertEqual(timed["label"], "Time exit at close")
        self.assertEqual(timed["exit_price"], 99)


if __name__ == "__main__":
    unittest.main()
