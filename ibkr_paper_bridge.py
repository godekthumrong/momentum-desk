"""Paper-only local bridge from the Short Scanner watchlist to IBKR TWS.

The bridge intentionally refuses live ports and non-paper account IDs.  A
browser must request a preview first and then explicitly confirm the returned,
single-use token before any order is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8765
TWS_PAPER_PORT = 7497
TOKEN_TTL_SECONDS = 120
ORDER_REF = "SHORT_WATCHLIST_PAPER"
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
DEFAULT_ORIGINS = {
    "http://127.0.0.1:8765",
    "http://localhost:8765",
    "https://godekthumrong.github.io",
}


class BridgeError(RuntimeError):
    pass


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"{label} must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise BridgeError(f"{label} must be greater than zero")
    return number


def _paper_account(account: str) -> bool:
    return bool(re.fullmatch(r"DU[0-9A-Z]+", account or ""))


def build_exit_plan(item: dict, session_date: str) -> dict:
    """Build the exact day-3 exit plan without touching a broker."""
    symbol = str(item.get("symbol", "")).strip().upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise BridgeError("Ticker must contain only letters, numbers, dot or dash")
    shares_number = _positive_number(item.get("shares"), "Shares")
    shares = int(shares_number)
    if shares_number != shares:
        raise BridgeError("Shares must be a whole number")
    entry_price = _positive_number(item.get("entry_price"), "Entry price")
    atr = _positive_number(item.get("atr"), "ATR")
    entry_close = _positive_number(item.get("entry_close"), "Entry-day close")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session_date):
        raise BridgeError("Session date must use YYYY-MM-DD")

    stop = round(entry_price + 2.5 * atr, 2)
    target = round(entry_price * 0.96, 4)
    base = {
        "symbol": symbol,
        "shares": shares,
        "entry_price": entry_price,
        "entry_close": entry_close,
        "atr": atr,
        "stop": stop,
        "target": target,
        "session_date": session_date,
    }
    if entry_close <= target:
        base.update({
            "reason": "profit_target",
            "orders": [{"action": "BUY", "type": "MKT", "tif": "OPG", "quantity": shares}],
        })
    else:
        good_after = f"{session_date.replace('-', '')} 15:59:00 US/Eastern"
        base.update({
            "reason": "stop_or_time_exit",
            "orders": [
                {"action": "BUY", "type": "STP", "tif": "DAY", "quantity": shares,
                 "stop_price": stop, "outside_rth": False},
                {"action": "BUY", "type": "MKT", "tif": "DAY", "quantity": shares,
                 "good_after_time": good_after, "outside_rth": False},
            ],
        })
    return base


def next_weekday(day: datetime | None = None) -> str:
    """Fallback next weekday; UI may pass the exchange-calendar session."""
    current = (day or datetime.now(ZoneInfo("America/New_York"))).date()
    while current.weekday() >= 5:
        current = current.fromordinal(current.toordinal() + 1)
    return current.isoformat()


class IBPaperClient:
    """Small synchronous facade over the official event-driven TWS API."""

    def __init__(self, host: str, port: int, client_id: int):
        if port != TWS_PAPER_PORT:
            raise BridgeError("Live trading is disabled: TWS port must be 7497")
        try:
            from ibapi.client import EClient
            from ibapi.contract import Contract
            from ibapi.order import Order
            from ibapi.wrapper import EWrapper
        except ImportError as exc:
            raise BridgeError(
                "IBKR Python API is missing. Install requirements-ibkr-bridge.txt first."
            ) from exc

        owner = self

        class App(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.ready = threading.Event()
                self.positions_done = threading.Event()
                self.accounts_done = threading.Event()
                self.next_order_id: int | None = None
                self.positions: dict[str, float] = {}
                self.accounts: list[str] = []
                self.errors: list[str] = []

            def nextValidId(self, orderId):
                self.next_order_id = int(orderId)
                self.ready.set()

            def managedAccounts(self, accountsList):
                self.accounts = [x for x in accountsList.split(",") if x]
                self.accounts_done.set()

            def position(self, account, contract, position, avgCost):
                key = str(contract.symbol).upper().replace(" ", ".")
                self.positions[key] = self.positions.get(key, 0.0) + float(position)

            def positionEnd(self):
                self.positions_done.set()

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                if int(errorCode) not in {2104, 2106, 2107, 2108, 2158}:
                    self.errors.append(f"{errorCode}: {errorString}")

        self.Contract = Contract
        self.Order = Order
        self.app = App()
        self.app.connect(host, port, clientId=client_id)
        self.thread = threading.Thread(target=self.app.run, daemon=True)
        self.thread.start()
        if not self.app.ready.wait(10):
            raise BridgeError("TWS Paper did not provide an order ID")
        self.app.accounts_done.wait(5)
        if not self.app.accounts or not all(_paper_account(a) for a in self.app.accounts):
            self.app.disconnect()
            raise BridgeError("Bridge refused connection: account is not an IBKR paper account")
        self.account = self.app.accounts[0]
        self.lock = threading.Lock()

    def health(self) -> dict:
        return {
            "connected": bool(self.app.isConnected()),
            "mode": "paper",
            "account": self.account[:2] + "••••" + self.account[-2:],
            "tws_port": TWS_PAPER_PORT,
        }

    def current_positions(self) -> dict[str, float]:
        self.app.positions = {}
        self.app.positions_done.clear()
        self.app.reqPositions()
        if not self.app.positions_done.wait(8):
            raise BridgeError("Timed out while checking IBKR positions")
        self.app.cancelPositions()
        return dict(self.app.positions)

    def validate_plans(self, plans: list[dict]) -> dict:
        positions = self.current_positions()
        for plan in plans:
            held = positions.get(plan["symbol"], 0.0)
            if held >= 0:
                raise BridgeError(f"{plan['symbol']} is not a Short position in TWS Paper")
            if plan["shares"] > abs(held) + 1e-9:
                raise BridgeError(
                    f"{plan['symbol']} requests {plan['shares']} shares but Paper position is {abs(held):g}"
                )
        return {symbol: positions.get(symbol, 0.0) for symbol in {p["symbol"] for p in plans}}

    def submit(self, plans: list[dict]) -> list[dict]:
        results: list[dict] = []
        with self.lock:
            for plan in plans:
                contract = self.Contract()
                contract.symbol = plan["symbol"].replace(".", " ")
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                group = f"SWP-{plan['symbol']}-{plan['session_date']}-{secrets.token_hex(3)}"
                order_ids = []
                for spec in plan["orders"]:
                    order = self.Order()
                    order.action = "BUY"
                    order.totalQuantity = int(spec["quantity"])
                    order.orderType = spec["type"]
                    order.tif = spec["tif"]
                    order.outsideRth = False
                    order.orderRef = ORDER_REF
                    order.account = self.account
                    if spec["type"] == "STP":
                        order.auxPrice = float(spec["stop_price"])
                    if spec.get("good_after_time"):
                        order.goodAfterTime = spec["good_after_time"]
                    if len(plan["orders"]) > 1:
                        order.ocaGroup = group
                        order.ocaType = 2
                    order.transmit = True
                    order_id = int(self.app.next_order_id)
                    self.app.next_order_id += 1
                    self.app.placeOrder(order_id, contract, order)
                    order_ids.append(order_id)
                results.append({"symbol": plan["symbol"], "order_ids": order_ids, "oca_group": group})
        time.sleep(0.5)
        recent_errors = self.app.errors[-5:]
        if recent_errors:
            raise BridgeError("IBKR reported: " + " | ".join(recent_errors))
        return results


class MockPaperClient:
    """No-network client used by tests and UI demonstrations."""

    def __init__(self):
        self.positions = {"TEST": -100.0}
        self.next_id = 9000

    def health(self) -> dict:
        return {"connected": True, "mode": "paper-mock", "account": "DU••••MO", "tws_port": 7497}

    def validate_plans(self, plans: list[dict]) -> dict:
        for p in plans:
            if p["shares"] > abs(self.positions.get(p["symbol"], 0.0)):
                raise BridgeError(f"{p['symbol']} does not have enough mock Short shares")
        return {p["symbol"]: self.positions.get(p["symbol"], 0.0) for p in plans}

    def submit(self, plans: list[dict]) -> list[dict]:
        out = []
        for p in plans:
            ids = list(range(self.next_id, self.next_id + len(p["orders"])))
            self.next_id += len(ids)
            out.append({"symbol": p["symbol"], "order_ids": ids, "oca_group": "MOCK"})
        return out


@dataclass
class Preview:
    token: str
    expires_at: float
    plans: list[dict]
    fingerprint: str


class BridgeState:
    def __init__(self, broker):
        self.broker = broker
        self.previews: dict[str, Preview] = {}
        self.submitted: set[str] = set()
        self.lock = threading.Lock()

    def preview(self, body: dict) -> dict:
        raw_items = body.get("positions")
        if not isinstance(raw_items, list) or not raw_items:
            raise BridgeError("Select at least one Watchlist position")
        if len(raw_items) > 20:
            raise BridgeError("A maximum of 20 positions may be prepared at once")
        session_date = str(body.get("session_date") or next_weekday())
        plans = [build_exit_plan(item, session_date) for item in raw_items]
        positions = self.broker.validate_plans(plans)
        canonical = json.dumps(plans, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        token = secrets.token_urlsafe(24)
        expiry = time.time() + TOKEN_TTL_SECONDS
        with self.lock:
            self.previews[token] = Preview(token, expiry, plans, fingerprint)
        return {
            "token": token,
            "expires_in": TOKEN_TTL_SECONDS,
            "mode": "paper",
            "account": self.broker.health()["account"],
            "positions": positions,
            "plans": plans,
        }

    def confirm(self, body: dict) -> dict:
        token = str(body.get("token") or "")
        if body.get("confirmation") != "PAPER":
            raise BridgeError("Confirmation text must be PAPER")
        with self.lock:
            preview = self.previews.pop(token, None)
            if preview is None:
                raise BridgeError("Preview token is invalid or was already used")
            if time.time() > preview.expires_at:
                raise BridgeError("Preview expired; prepare the orders again")
            if preview.fingerprint in self.submitted:
                raise BridgeError("These exact Paper orders were already submitted")
        self.broker.validate_plans(preview.plans)
        submitted = self.broker.submit(preview.plans)
        with self.lock:
            self.submitted.add(preview.fingerprint)
        return {"mode": "paper", "submitted": submitted}


def make_handler(state: BridgeState, allowed_origins: set[str], access_key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ShortWatchPaperBridge/1.0"

        def _origin(self) -> str:
            return self.headers.get("Origin", "")

        def _cors(self):
            origin = self._origin()
            if origin in allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Bridge-Key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, status: int, payload: dict):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized_origin(self) -> bool:
            origin = self._origin()
            return not origin or origin in allowed_origins

        def _authorized_key(self) -> bool:
            supplied = self.headers.get("X-Bridge-Key", "")
            return bool(supplied) and hmac.compare_digest(supplied, access_key)

        def do_OPTIONS(self):
            if not self._authorized_origin():
                self._json(403, {"error": "Origin is not allowed"})
                return
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if not self._authorized_origin():
                self._json(403, {"error": "Origin is not allowed"})
            elif not self._authorized_key():
                self._json(401, {"error": "Bridge key is missing or incorrect"})
            elif self.path == "/api/health":
                self._json(200, state.broker.health())
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._authorized_origin():
                self._json(403, {"error": "Origin is not allowed"})
                return
            if not self._authorized_key():
                self._json(401, {"error": "Bridge key is missing or incorrect"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64_000:
                    raise BridgeError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/preview":
                    self._json(200, state.preview(body))
                elif self.path == "/api/confirm":
                    self._json(200, state.confirm(body))
                else:
                    self._json(404, {"error": "Not found"})
            except (BridgeError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "Bridge failed; check the local terminal log"})
                raise

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-only IBKR bridge for Short Watchlist")
    parser.add_argument("--host", default=BRIDGE_HOST)
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument("--tws-host", default="127.0.0.1")
    parser.add_argument("--tws-port", type=int, default=TWS_PAPER_PORT)
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--bridge-key", default=os.environ.get("IBKR_BRIDGE_KEY", ""))
    parser.add_argument("--mock", action="store_true", help="Never connect to IBKR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Bridge must listen on localhost only")
    if args.tws_port != TWS_PAPER_PORT:
        raise SystemExit("Live trading is disabled; only TWS Paper port 7497 is accepted")
    extra = {x.strip() for x in os.environ.get("IBKR_BRIDGE_ALLOWED_ORIGINS", "").split(",") if x.strip()}
    access_key = args.bridge_key.strip() or secrets.token_urlsafe(24)
    if len(access_key) < 20:
        raise SystemExit("Bridge key must contain at least 20 characters")
    broker = MockPaperClient() if args.mock else IBPaperClient(args.tws_host, args.tws_port, args.client_id)
    state = BridgeState(broker)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(state, DEFAULT_ORIGINS | extra, access_key)
    )
    print(f"Paper bridge listening on http://{args.host}:{args.port}")
    print(f"Bridge key: {access_key}")
    print(json.dumps(broker.health(), indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
