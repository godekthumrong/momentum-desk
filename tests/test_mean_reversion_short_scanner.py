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
        self.assertEqual(row["sizing_distance"], 20.0)
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

    def test_order_plan_columns_follow_ticker(self):
        payload = {"meta": {"as_of": "2024-01-31", "portfolio_value": 100000,
                            "signals": 0, "selected": 0, "ready": 0,
                            "fresh": 5000, "universe": 5100}, "signals": []}
        page = build_html(payload)
        expected = (
            "<th>Rank</th><th>Ticker</th><th>Limit</th><th>Allocation</th>"
            "<th>Shares</th><th>Est. stop</th>"
        )
        self.assertIn(expected, page)

    def test_trade_journal_is_browser_local_and_has_return_stats(self):
        payload = {"meta": {"as_of": "2024-01-31", "portfolio_value": 100000,
                            "signals": 0, "selected": 0, "ready": 0,
                            "fresh": 5000, "universe": 5100}, "signals": [], "market": {}}
        page = build_html(payload)
        self.assertIn("short-scanner-journal-v1", page)
        self.assertIn("Cumulative portfolio return", page)
        self.assertIn("Profit factor", page)
        self.assertIn("Max drawdown", page)
        self.assertIn("Export CSV", page)

    def test_watchlist_has_stop_columns(self):
        payload = {"meta": {"as_of": "2024-01-31", "portfolio_value": 100000,
                            "signals": 0, "selected": 0, "ready": 0,
                            "fresh": 5000, "universe": 5100}, "signals": [], "market": {}}
        page = build_html(payload)
        self.assertIn(
            "<th>Entry</th><th>Stop / Est. stop</th><th>Latest close</th><th>Open P&amp;L</th>", page
        )
        self.assertNotIn("<th>Stop</th>", page)
        self.assertNotIn("<th>Target</th>", page)
        self.assertIn('id="w-atr"', page)

    def test_signal_can_be_added_pending_without_opening_fill_form(self):
        payload = {"meta": {"as_of": "2024-01-31", "portfolio_value": 100000,
                            "signals": 0, "selected": 0, "ready": 0,
                            "fresh": 5000, "universe": 5100}, "signals": [], "market": {}}
        page = build_html(payload)
        self.assertIn("no form entry required", page)
        self.assertIn("shares:shares,plannedLimit:r.limit_price", page)
        self.assertIn("if(old)return", page)
        self.assertNotIn("saveWatch()}}editWatch(r.symbol)", page)
        self.assertIn(
            "shown=matching.filter(function(r){return !watch.some", page
        )
        self.assertIn(
            "All matching signals are already in the watchlist.", page
        )

    def test_stop_exit_and_time_exit(self):
        entry = ["2024-01-02", 100, 150, 90, 95]
        for day2, expected, label in [
            (["2024-01-03", 112, 115, 90, 95], 112, "Stop gap at open"),
            (["2024-01-03", 101, 110, 90, 95], 110, "Stop hit intraday"),
            (["2024-01-03", 94, 109, 90, 93], 93, "Time exit at close"),
        ]:
            result = evaluate_exit("2024-01-02", 100, [entry, day2], 4)
            self.assertEqual(result["exit_price"], expected)
            self.assertEqual(result["label"], label)
        self.assertEqual(evaluate_exit("2024-01-02", 100, [entry], 4)["state"], "hold")
        self.assertEqual(evaluate_exit("2024-01-02", 100, [entry])["state"], "waiting_data")


if __name__ == "__main__":
    unittest.main()
