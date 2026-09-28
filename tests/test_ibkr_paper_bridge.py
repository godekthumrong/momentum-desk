import http.client
import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from ibkr_paper_bridge import (
    BridgeError,
    BridgeState,
    MockPaperClient,
    build_exit_plan,
    make_handler,
)


class IBKRPaperBridgeTests(unittest.TestCase):
    def test_profit_target_builds_market_on_open_cover(self):
        plan = build_exit_plan(
            {"symbol": "TEST", "shares": 100, "entry_price": 100,
             "atr": 2, "entry_close": 95},
            "2026-09-28",
        )
        self.assertEqual(plan["reason"], "profit_target")
        self.assertEqual(plan["orders"], [
            {"action": "BUY", "type": "MKT", "tif": "OPG", "quantity": 100}
        ])

    def test_non_target_builds_stop_and_timed_market_oca_legs(self):
        plan = build_exit_plan(
            {"symbol": "TEST", "shares": 80, "entry_price": 100,
             "atr": 2, "entry_close": 98},
            "2026-09-28",
        )
        self.assertEqual(plan["reason"], "stop_or_time_exit")
        self.assertEqual(plan["stop"], 105.0)
        self.assertEqual(plan["orders"][0]["type"], "STP")
        self.assertEqual(plan["orders"][1]["good_after_time"],
                         "20260928 15:59:00 US/Eastern")

    def test_preview_requires_second_single_use_confirmation(self):
        state = BridgeState(MockPaperClient())
        body = {
            "session_date": "2026-09-28",
            "positions": [{"symbol": "TEST", "shares": 50, "entry_price": 100,
                           "atr": 2, "entry_close": 98}],
        }
        preview = state.preview(body)
        self.assertEqual(preview["mode"], "paper")
        submitted = state.confirm({"token": preview["token"], "confirmation": "PAPER"})
        self.assertEqual(len(submitted["submitted"][0]["order_ids"]), 2)
        with self.assertRaises(BridgeError):
            state.confirm({"token": preview["token"], "confirmation": "PAPER"})

    def test_refuses_oversized_cover_and_bad_confirmation(self):
        state = BridgeState(MockPaperClient())
        with self.assertRaises(BridgeError):
            state.preview({
                "session_date": "2026-09-28",
                "positions": [{"symbol": "TEST", "shares": 101, "entry_price": 100,
                               "atr": 2, "entry_close": 98}],
            })
        preview = state.preview({
            "session_date": "2026-09-28",
            "positions": [{"symbol": "TEST", "shares": 10, "entry_price": 100,
                           "atr": 2, "entry_close": 98}],
        })
        with self.assertRaises(BridgeError):
            state.confirm({"token": preview["token"], "confirmation": "LIVE"})

    def test_http_api_requires_bridge_key(self):
        key = "test-bridge-key-1234567890"
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(BridgeState(MockPaperClient()), set(), key),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()

            connection.request("GET", "/api/health", headers={"X-Bridge-Key": key})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["mode"], "paper-mock")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
