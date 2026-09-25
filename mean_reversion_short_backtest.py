"""Alpaca SIP backtest for the book's Mean-Reversion Short strategy.

The script deliberately uses a *current* active-stock universe.  That makes the
result survivorship-biased; the report labels this limitation prominently.
Prices come from Alpaca SIP.  Nasdaq Trader symbol directories are used only to
remove current ETFs and test issues because Alpaca's asset endpoint does not
identify ETFs.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}


def _iso_day(value: str | pd.Timestamp) -> str:
    return str(pd.Timestamp(value).date())


class Alpaca:
    def __init__(self, key: str | None = None, secret: str | None = None,
                 session: requests.Session | None = None):
        key = key or os.environ.get("ALPACA_API_KEY", "").strip()
        secret = secret or os.environ.get("ALPACA_SECRET_KEY", "").strip()
        if not key or not secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        self.session = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _get(self, url: str, params: dict | None = None):
        for attempt in range(8):
            try:
                response = self.session.get(
                    url, headers=self.headers, params=params, timeout=(15, 120),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt == 7:
                    raise RuntimeError(f"Alpaca request failed: {exc}") from exc
                time.sleep(min(30, 2 ** attempt))
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code in (401, 403):
                raise RuntimeError("Alpaca authentication or SIP access failed")
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 30)))
                continue
            if response.status_code >= 500:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise RuntimeError(f"Alpaca rejected request: HTTP {response.status_code}: {response.text[:300]}")
        raise RuntimeError("Alpaca did not recover after retries")

    def active_assets(self) -> list[dict]:
        data = self._get(ASSETS_URL, {"status": "active", "asset_class": "us_equity"})
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Alpaca assets response")
        return data

    def bars(self, symbols: list[str], start: str, end: str,
             adjustment: str = "all") -> dict[str, list[dict]]:
        params = {
            "symbols": ",".join(symbols), "timeframe": "1Day",
            "start": f"{start}T00:00:00Z", "end": f"{end}T23:59:59Z",
            "adjustment": adjustment, "feed": "sip", "limit": 10000, "sort": "asc",
        }
        out: dict[str, list[dict]] = {s: [] for s in symbols}
        seen: set[str] = set()
        while True:
            payload = self._get(DATA_URL, params)
            bars = payload.get("bars")
            if not isinstance(bars, dict):
                raise RuntimeError("Unexpected Alpaca bars response")
            for symbol, rows in bars.items():
                if symbol in out:
                    out[symbol].extend(rows or [])
            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise RuntimeError("Alpaca repeated a pagination token")
            seen.add(token)
            params["page_token"] = token
        return out


def _read_pipe_table(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="|"))


def current_non_etf_symbols(session: requests.Session | None = None) -> set[str]:
    """Return current non-ETF, non-test listed symbols from Nasdaq Trader."""
    session = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 mean-reversion-research/1.0"}
    symbols: set[str] = set()
    for url, symbol_col in ((NASDAQ_LISTED_URL, "Symbol"), (OTHER_LISTED_URL, "ACT Symbol")):
        response = session.get(url, headers=headers, timeout=(15, 60))
        response.raise_for_status()
        for row in _read_pipe_table(response.text):
            if row.get("ETF") == "N" and row.get("Test Issue") == "N":
                symbol = (row.get(symbol_col) or "").strip()
                if symbol:
                    symbols.add(symbol)
    if len(symbols) < 3000:
        raise RuntimeError(f"Nasdaq Trader non-ETF list unexpectedly small ({len(symbols)})")
    return symbols


def build_universe(client: Alpaca) -> tuple[list[str], dict]:
    assets = client.active_assets()
    listed = current_non_etf_symbols()
    selected = []
    excluded = defaultdict(int)
    non_stock_pattern = re.compile(r"\b(?:WARRANTS?|RIGHTS?|UNITS?|PREFERRED)\b")
    for asset in assets:
        symbol = str(asset.get("symbol", "")).strip()
        name = f" {str(asset.get('name', '')).upper()} "
        if asset.get("exchange") not in EXCHANGES:
            excluded["exchange"] += 1
        elif not asset.get("tradable"):
            excluded["not_tradable"] += 1
        elif symbol not in listed:
            excluded["etf_test_or_not_in_directory"] += 1
        elif non_stock_pattern.search(name):
            excluded["non_common_security"] += 1
        else:
            selected.append(symbol)
    selected = sorted(set(selected))
    if len(selected) < 2500:
        raise RuntimeError(f"Current active stock universe unexpectedly small ({len(selected)})")
    return selected, {"alpaca_assets": len(assets), "selected": len(selected), "excluded": dict(excluded)}


def wilder(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing with an SMA seed, matching classic ATR/RSI/ADX."""
    arr = values.astype(float).to_numpy()
    out = np.full(len(arr), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(arr))
    if len(valid) < period:
        return pd.Series(out, index=values.index)
    start = valid[0]
    if start + period > len(arr) or not np.isfinite(arr[start:start + period]).all():
        return pd.Series(out, index=values.index)
    seed_i = start + period - 1
    out[seed_i] = arr[start:start + period].mean()
    for i in range(seed_i + 1, len(arr)):
        if np.isfinite(arr[i]) and np.isfinite(out[i - 1]):
            out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return pd.Series(out, index=values.index)


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr10 = wilder(tr, 10)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    atr7 = wilder(tr, 7)
    plus_di = 100 * wilder(plus_dm, 7) / atr7.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, 7) / atr7.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx7 = wilder(dx, 7)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = wilder(gain, 3)
    avg_loss = wilder(loss, 3)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi3 = 100 - 100 / (1 + rs)
    rsi3 = rsi3.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)

    return pd.DataFrame({
        "avg_volume20": frame["volume"].rolling(20, min_periods=20).mean(),
        "atr10": atr10,
        "atr_pct10": 100 * atr10 / close,
        "adx7": adx7,
        "rsi3": rsi3,
        "two_up": (close > close.shift(1)) & (close.shift(1) > close.shift(2)),
    }, index=frame.index)


@dataclass
class Candidate:
    symbol: str
    signal_date: str
    order_date: str
    rank_rsi3: float
    atr10: float
    limit_price: float
    filled: bool
    entry_price: float | None
    entry_close: float | None
    exit_date: str | None
    exit_price: float | None
    exit_reason: str | None
    exit_timing: str | None
    exit_open: float | None


def frame_from_bars(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    data = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
    data["date"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    data = data.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return data[["open", "high", "low", "close", "volume"]].astype(float)


def symbol_candidates(symbol: str, frame: pd.DataFrame, raw_frame: pd.DataFrame,
                      test_start: str) -> list[Candidate]:
    if len(frame) < 30:
        return []
    ind = indicators(frame)
    raw = raw_frame.reindex(frame.index)
    raw_avg_volume20 = raw["volume"].rolling(20, min_periods=20).mean()
    signal = (
        (raw["close"] >= 10.0)
        & (raw_avg_volume20 > 500_000)
        & (ind["adx7"] > 50.0)
        & (ind["atr_pct10"] > 5.0)
        & ind["two_up"]
        & (ind["rsi3"] > 85.0)
        & (frame.index >= pd.Timestamp(test_start))
    )
    hits = np.flatnonzero(signal.fillna(False).to_numpy())
    out: list[Candidate] = []
    for i in hits:
        if i + 1 >= len(frame):
            continue
        limit_price = float(frame.iloc[i]["close"])
        atr = float(ind.iloc[i]["atr10"])
        rsi = float(ind.iloc[i]["rsi3"])
        day1 = frame.iloc[i + 1]
        entry_price = None
        if day1["open"] >= limit_price:
            entry_price = float(day1["open"])
        elif day1["high"] >= limit_price:
            entry_price = limit_price
        if entry_price is None:
            out.append(Candidate(symbol, _iso_day(frame.index[i]), _iso_day(frame.index[i + 1]),
                                 rsi, atr, limit_price, False, None, None, None, None, None, None, None))
            continue

        stop = entry_price + 2.5 * atr
        entry_close = float(day1["close"])
        if i + 2 >= len(frame):
            exit_date, exit_price, reason, timing, exit_open = (
                _iso_day(frame.index[i + 1]), entry_close, "end_of_data", "close", float(day1["open"])
            )
        else:
            day2 = frame.iloc[i + 2]
            exit_date = _iso_day(frame.index[i + 2])
            exit_open = float(day2["open"])
            if entry_close <= entry_price * 0.96:
                exit_price, reason, timing = exit_open, "profit_target", "open"
            elif day2["open"] >= stop:
                exit_price, reason, timing = exit_open, "stop_loss", "open"
            elif day2["high"] >= stop:
                exit_price, reason, timing = stop, "stop_loss", "intraday"
            else:
                exit_price, reason, timing = float(day2["close"]), "time_exit", "close"
        out.append(Candidate(symbol, _iso_day(frame.index[i]), _iso_day(frame.index[i + 1]),
                             rsi, atr, limit_price, True, entry_price, entry_close,
                             exit_date, float(exit_price), reason, timing, float(exit_open)))
    return out


def select_top_orders(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.signal_date].append(candidate)
    selected = []
    signal_count = 0
    for rows in grouped.values():
        signal_count += len(rows)
        selected.extend(sorted(rows, key=lambda x: (-x.rank_rsi3, x.symbol))[:10])
    return sorted(selected, key=lambda x: (x.order_date, -x.rank_rsi3, x.symbol)), signal_count


@dataclass
class Position:
    candidate: Candidate
    shares: float
    entry_notional: float
    entry_cost: float


def _max_drawdown(series: pd.Series) -> float:
    return float((series / series.cummax() - 1.0).min()) if len(series) else 0.0


def simulate(orders: list[Candidate], trading_days: list[str], cost_bps_side: float,
             initial_cash: float = 100_000.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_entry: dict[str, list[Candidate]] = defaultdict(list)
    for order in orders:
        by_entry[order.order_date].append(order)
    cash = float(initial_cash)
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    curve: list[dict] = []
    fee_rate = cost_bps_side / 10_000.0
    blocked_exposure = blocked_duplicate = 0

    for day in trading_days:
        entering = by_entry.get(day, [])

        # Targets and stop gaps execute at the open and free capital before new limits.
        for symbol, pos in list(positions.items()):
            c = pos.candidate
            if c.exit_date == day and c.exit_timing == "open":
                exit_notional = pos.shares * c.exit_price
                exit_cost = exit_notional * fee_rate
                cash -= exit_notional + exit_cost
                pnl = pos.entry_notional - exit_notional - pos.entry_cost - exit_cost
                trades.append(_trade_row(pos, pnl, exit_cost))
                del positions[symbol]

        # Existing positions are marked at today's open for the 100% gross cap.
        liabilities_open = 0.0
        for pos in positions.values():
            c = pos.candidate
            mark = c.exit_open if c.exit_date == day and c.exit_open is not None else c.entry_close
            liabilities_open += pos.shares * float(mark)
        equity_open = cash - liabilities_open
        gross_open = liabilities_open

        for c in entering:
            if not c.filled:
                continue
            if c.symbol in positions:
                blocked_duplicate += 1
                continue
            risk_per_dollar = (2.5 * c.atr10) / c.entry_price
            desired_fraction = min(0.10, 0.02 / risk_per_dollar) if risk_per_dollar > 0 else 0.0
            desired = max(0.0, equity_open * desired_fraction)
            capacity = max(0.0, equity_open - gross_open)
            notional = min(desired, capacity / (1.0 + fee_rate))
            if notional < max(100.0, equity_open * 0.001):
                blocked_exposure += 1
                continue
            shares = notional / c.entry_price
            entry_cost = notional * fee_rate
            cash += notional - entry_cost
            gross_open += notional
            equity_open -= entry_cost
            positions[c.symbol] = Position(c, shares, notional, entry_cost)

        # Intraday stops and MOC time exits happen after entry capacity was assessed.
        for symbol, pos in list(positions.items()):
            c = pos.candidate
            if c.exit_date == day and c.exit_timing in ("intraday", "close"):
                exit_notional = pos.shares * c.exit_price
                exit_cost = exit_notional * fee_rate
                cash -= exit_notional + exit_cost
                pnl = pos.entry_notional - exit_notional - pos.entry_cost - exit_cost
                trades.append(_trade_row(pos, pnl, exit_cost))
                del positions[symbol]

        liabilities_close = sum(pos.shares * pos.candidate.entry_close for pos in positions.values())
        equity_close = cash - liabilities_close
        gross_close = liabilities_close / equity_close if equity_close > 0 else math.inf
        curve.append({"date": day, "equity": equity_close, "gross_exposure": gross_close,
                      "open_positions": len(positions)})

    curve_df = pd.DataFrame(curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    if curve_df.empty:
        raise RuntimeError("No portfolio curve was produced")
    rets = curve_df["equity"].pct_change().dropna()
    years = max((pd.Timestamp(curve_df.index[-1]) - pd.Timestamp(curve_df.index[0])).days / 365.2425, 1 / 365.2425)
    end = float(curve_df["equity"].iloc[-1])
    cagr = (end / initial_cash) ** (1 / years) - 1 if end > 0 else -1.0
    sharpe = (math.sqrt(252) * rets.mean() / rets.std(ddof=1)) if len(rets) > 2 and rets.std(ddof=1) > 0 else 0.0
    wins = int((trades_df.get("net_pnl", pd.Series(dtype=float)) > 0).sum())
    losses = int((trades_df.get("net_pnl", pd.Series(dtype=float)) < 0).sum())
    gross_profit = float(trades_df.loc[trades_df.get("net_pnl", pd.Series(dtype=float)) > 0, "net_pnl"].sum()) if len(trades_df) else 0.0
    gross_loss = float(-trades_df.loc[trades_df.get("net_pnl", pd.Series(dtype=float)) < 0, "net_pnl"].sum()) if len(trades_df) else 0.0
    metrics = {
        "cost_bps_per_side": cost_bps_side, "initial_balance": initial_cash, "ending_balance": end,
        "cagr": cagr, "max_drawdown": _max_drawdown(curve_df["equity"]), "sharpe": float(sharpe),
        "trades": len(trades_df), "wins": wins, "losses": losses,
        "win_rate": wins / len(trades_df) if len(trades_df) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "avg_trade_return": float(trades_df["net_return"].mean()) if len(trades_df) else 0.0,
        "avg_gross_exposure": float(curve_df["gross_exposure"].replace([np.inf], np.nan).fillna(0).mean()),
        "max_gross_exposure": float(curve_df["gross_exposure"].replace([np.inf], np.nan).fillna(0).max()),
        "blocked_by_exposure": blocked_exposure, "blocked_duplicate_symbol": blocked_duplicate,
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict() if len(trades_df) else {},
    }
    return curve_df, trades_df, metrics


def _trade_row(pos: Position, pnl: float, exit_cost: float) -> dict:
    c = pos.candidate
    return {
        "symbol": c.symbol, "signal_date": c.signal_date, "entry_date": c.order_date,
        "exit_date": c.exit_date, "entry_price": c.entry_price, "exit_price": c.exit_price,
        "shares": pos.shares, "entry_notional": pos.entry_notional,
        "entry_cost": pos.entry_cost, "exit_cost": exit_cost, "net_pnl": pnl,
        "net_return": pnl / pos.entry_notional if pos.entry_notional else 0.0,
        "exit_reason": c.exit_reason, "rsi3": c.rank_rsi3, "atr10": c.atr10,
    }


def benchmark_curve(frame: pd.DataFrame, start: str, initial: float = 100_000.0) -> pd.DataFrame:
    close = frame.loc[pd.Timestamp(start):, "close"].dropna()
    return pd.DataFrame({"equity": initial * close / close.iloc[0]}, index=close.index)


def svg_line(series_map: dict[str, pd.Series], width: int = 960, height: int = 360) -> str:
    all_values = pd.concat(series_map.values()).replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        return ""
    lo, hi = float(all_values.min()), float(all_values.max())
    hi = hi if hi > lo else lo + 1
    n = max(len(s) for s in series_map.values())
    colors = ["#0f766e", "#2563eb", "#d97706", "#64748b"]
    paths = []
    for (label, series), color in zip(series_map.items(), colors):
        vals = series.astype(float).to_numpy()
        points = []
        for i, value in enumerate(vals):
            if not np.isfinite(value):
                continue
            x = 50 + i * (width - 80) / max(len(vals) - 1, 1)
            y = 20 + (hi - value) * (height - 60) / (hi - lo)
            points.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/><text x="{65 + 180 * len(paths)}" y="{height - 10}" fill="{color}">{html.escape(label)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curves"><rect width="100%" height="100%" fill="#fff"/><line x1="50" y1="20" x2="50" y2="{height-40}" stroke="#cbd5e1"/><line x1="50" y1="{height-40}" x2="{width-30}" y2="{height-40}" stroke="#cbd5e1"/>{"".join(paths)}</svg>'


def write_report(outdir: Path, meta: dict, scenarios: dict, curves: dict, benchmark: pd.DataFrame,
                 base_trades: pd.DataFrame, yearly: pd.DataFrame):
    def pct(x): return f"{100*x:.2f}%"
    rows = []
    for name, m in scenarios.items():
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "—"
        rows.append(f"<tr><td>{html.escape(name)}</td><td>${m['ending_balance']:,.0f}</td><td>{pct(m['cagr'])}</td><td>{pct(m['max_drawdown'])}</td><td>{m['sharpe']:.2f}</td><td>{m['trades']:,}</td><td>{pct(m['win_rate'])}</td><td>{pf}</td></tr>")
    chart_series = {name: df["equity"] for name, df in curves.items()}
    spy_chart = benchmark.copy()
    spy_chart.index = pd.to_datetime(spy_chart.index).strftime("%Y-%m-%d")
    chart_series["SPY total return"] = spy_chart["equity"].reindex(curves[next(iter(curves))].index).ffill().bfill()
    yearly_html = yearly.to_html(float_format=lambda x: f"{100*x:.1f}%", classes="yearly", border=0)
    report = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mean-Reversion Short Backtest</title><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}main{{max-width:1080px;margin:auto;padding:28px}}.card{{background:white;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 1px 4px #cbd5e1}}h1{{margin:.1em 0}}.warn{{background:#fff7ed;border-left:5px solid #f97316}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}.small{{color:#475569;font-size:13px}}code{{background:#e2e8f0;padding:2px 5px;border-radius:4px}}
    </style></head><body><main><div class="card"><h1>Mean-Reversion Short</h1><p>Alpaca SIP · {_iso_day(meta['start'])} to {_iso_day(meta['end'])} · current active US stocks</p></div>
    <div class="card warn"><strong>Important:</strong> This run has survivorship bias because the universe is today's active stocks. Historical delistings, historical short availability and borrow fees are unavailable from Alpaca and are not modeled.</div>
    <div class="card"><h2>Results</h2><table><thead><tr><th>Scenario</th><th>Ending</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Trades</th><th>Win rate</th><th>Profit factor</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <div class="card"><h2>Equity curve</h2>{svg_line(chart_series)}</div>
    <div class="card"><h2>Yearly returns</h2>{yearly_html}</div>
    <div class="card"><h2>Rules and execution assumptions</h2><ul><li>AMEX/NASDAQ/NYSE current active stocks; ETFs/test issues removed with current Nasdaq Trader directories.</li><li>Raw point-in-time close ≥ $10 and raw 20-day average volume &gt; 500,000. Split/dividend-adjusted OHLC is used for indicators and returns.</li><li>ADX(7) &gt; 50; ATR(10)/close &gt; 5%; last two closes up; RSI(3) &gt; 85.</li><li>Top 10 signals each day by RSI(3). Next-session short limit at the signal close; fill at open when open is above the limit, otherwise at the limit if high touches it.</li><li>Size = 2% equity risk to a 2.5×ATR stop, capped at 10% per position and 100% gross short exposure at entry.</li><li>No stop on entry day. From the next session: stop gap at open or stop intraday; a 4% close profit exits next open; otherwise exit at the second session's close.</li><li>Headline/base cost is 5 bps per side. Commission is otherwise treated as zero. Borrow fees are excluded.</li></ul></div>
    <div class="card small"><h2>Data audit</h2><pre>{html.escape(json.dumps(meta, indent=2))}</pre></div></main></body></html>"""
    outdir.joinpath("mean-reversion-short-report.html").write_text(report, encoding="utf-8")
    base_trades.to_csv(outdir / "mean-reversion-short-trades.csv", index=False)
    yearly.to_csv(outdir / "mean-reversion-short-yearly.csv")


def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    client = Alpaca()
    universe, universe_audit = build_universe(client)
    print(f"Universe: {len(universe):,} current active non-ETF stocks")

    warmup = _iso_day(pd.Timestamp(args.start) - pd.Timedelta(days=120))
    end = args.end or _iso_day(pd.Timestamp.utcnow())
    all_candidates: list[Candidate] = []
    observed_days: set[str] = set()
    symbols_with_data = 0
    batch_size = args.batch_size
    for offset in range(0, len(universe), batch_size):
        batch = universe[offset:offset + batch_size]
        payload = client.bars(batch, warmup, end, adjustment="all")
        raw_payload = client.bars(batch, warmup, end, adjustment="raw")
        for symbol in batch:
            frame = frame_from_bars(payload.get(symbol, []))
            raw_frame = frame_from_bars(raw_payload.get(symbol, []))
            if not frame.empty:
                symbols_with_data += 1
                observed_days.update(_iso_day(d) for d in frame.loc[pd.Timestamp(args.start):].index)
                all_candidates.extend(symbol_candidates(symbol, frame, raw_frame, args.start))
        print(f"Bars {min(offset+batch_size, len(universe)):,}/{len(universe):,}; candidates {len(all_candidates):,}")

    orders, raw_signal_count = select_top_orders(all_candidates)
    trading_days = sorted(d for d in observed_days if args.start <= d <= end)
    if not trading_days:
        raise RuntimeError("No trading days returned")

    spy = frame_from_bars(client.bars(["SPY"], warmup, end, adjustment="all").get("SPY", []))
    benchmark = benchmark_curve(spy, args.start)
    scenario_specs = {"Gross / no costs": 0.0, "Base / 5 bps each side": 5.0,
                      "Stress / 25 bps each side": 25.0}
    scenario_metrics, curves, trades = {}, {}, {}
    for name, bps in scenario_specs.items():
        curve, trade_log, metrics = simulate(orders, trading_days, bps)
        scenario_metrics[name], curves[name], trades[name] = metrics, curve, trade_log

    base_curve = curves["Base / 5 bps each side"]
    yearly = pd.DataFrame(index=sorted(set(pd.to_datetime(base_curve.index).year)))
    for name, curve in curves.items():
        temp = curve.copy()
        temp.index = pd.to_datetime(temp.index)
        eoy = temp["equity"].resample("YE").last()
        yr = eoy.pct_change()
        if len(yr):
            yr.iloc[0] = eoy.iloc[0] / temp["equity"].iloc[0] - 1
        yearly[name] = pd.Series(yr.to_numpy(), index=eoy.index.year).reindex(yearly.index)
    b = benchmark.copy(); b.index = pd.to_datetime(b.index)
    beoy = b["equity"].resample("YE").last()
    by = beoy.pct_change()
    if len(by):
        by.iloc[0] = beoy.iloc[0] / b["equity"].iloc[0] - 1
    yearly["SPY total return"] = pd.Series(by.to_numpy(), index=beoy.index.year).reindex(yearly.index)

    meta = {
        "source": "Alpaca SIP", "adjustment": "all for signals/returns; raw for price/volume screens",
        "start": args.start, "end": trading_days[-1],
        "universe": universe_audit, "symbols_with_data": symbols_with_data,
        "raw_qualifying_signals": raw_signal_count, "selected_orders": len(orders),
        "filled_selected_orders": sum(c.filled for c in orders),
        "survivorship_bias": True, "historical_short_availability": False,
        "borrow_fees_included": False, "gross_exposure_cap": 1.0,
    }
    write_report(outdir, meta, scenario_metrics, curves, benchmark,
                 trades["Base / 5 bps each side"], yearly)
    for name, curve in curves.items():
        safe = name.lower().replace(" ", "-").replace("/", "-")
        curve.to_csv(outdir / f"equity-{safe}.csv")
    result = {"meta": meta, "scenarios": scenario_metrics}
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")
    rendered = json.dumps(result, indent=2, allow_nan=False, default=json_default)
    outdir.joinpath("mean-reversion-short-results.json").write_text(rendered, encoding="utf-8")
    print(rendered)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=os.environ.get("SHORT_BT_START", "2016-01-01"))
    parser.add_argument("--end", default=os.environ.get("SHORT_BT_END", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SHORT_BT_BATCH", "100")))
    parser.add_argument("--output", default=os.environ.get("SHORT_BT_OUTPUT", "short-backtest-output"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
