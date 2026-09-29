"""Alpaca SIP backtest for the book's long mean-reversion strategy.

The universe is Alpaca's *current* active US common stocks, so results have
survivorship bias.  Indicators/returns use split-and-dividend-adjusted SIP
daily bars; raw bars are used for price and liquidity screens.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mean_reversion_short_backtest import (
    Alpaca,
    _iso_day,
    _max_drawdown,
    _trade_commission,
    benchmark_curve,
    build_universe,
    frame_from_bars,
    svg_line,
    wilder,
)


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
    holding_days: int | None
    daily_opens: dict[str, float]
    daily_closes: dict[str, float]


@dataclass
class Position:
    candidate: Candidate
    shares: int
    entry_notional: float
    entry_cost: float


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr10 = wilder(tr, 10)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    atr7 = wilder(tr, 7)
    plus_di = 100 * wilder(plus_dm, 7) / atr7.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, 7) / atr7.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = wilder(gain, 3)
    avg_loss = wilder(loss, 3)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi3 = 100 - 100 / (1 + rs)
    rsi3 = rsi3.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)

    return pd.DataFrame({
        "sma150": close.rolling(150, min_periods=150).mean(),
        "atr10": atr10,
        "atr_pct10": 100 * atr10 / close,
        "adx7": wilder(dx, 7),
        "rsi3": rsi3,
    }, index=frame.index)


def _long_exit(frame: pd.DataFrame, entry_i: int, entry_price: float,
               atr10: float) -> tuple[str, float, str, str, float, int]:
    """Resolve exit using daily OHLC and the book's no-entry-day-stop rule."""
    stop = entry_price - 2.5 * atr10
    last_i = min(entry_i + 3, len(frame) - 1)  # entry session is holding day 1

    for i in range(entry_i + 1, last_i + 1):
        row = frame.iloc[i]
        day_open = float(row["open"])

        # A +3% close from the prior session exits at this session's open.
        prior_close = float(frame.iloc[i - 1]["close"])
        if prior_close >= entry_price * 1.03:
            return _iso_day(frame.index[i]), day_open, "profit_target", "open", day_open, i - entry_i + 1

        # The protective stop begins only after the entry session's close.
        if day_open <= stop:
            return _iso_day(frame.index[i]), day_open, "stop_loss", "open", day_open, i - entry_i + 1
        if float(row["low"]) <= stop:
            return _iso_day(frame.index[i]), stop, "stop_loss", "intraday", day_open, i - entry_i + 1

        if i == entry_i + 3:
            return _iso_day(frame.index[i]), float(row["close"]), "time_exit", "close", day_open, 4

    row = frame.iloc[last_i]
    return (_iso_day(frame.index[last_i]), float(row["close"]), "end_of_data",
            "close", float(row["open"]), last_i - entry_i + 1)


def symbol_candidates(symbol: str, frame: pd.DataFrame, raw_frame: pd.DataFrame,
                      test_start: str) -> list[Candidate]:
    if len(frame) < 155:
        return []
    ind = indicators(frame)
    raw = raw_frame.reindex(frame.index)
    avg_volume50 = raw["volume"].rolling(50, min_periods=50).mean()
    avg_dollar_volume50 = (raw["close"] * raw["volume"]).rolling(50, min_periods=50).mean()
    signal = (
        (raw["close"] > 1.0)
        & (avg_volume50 > 500_000)
        & (avg_dollar_volume50 >= 2_500_000)
        & (frame["close"] > ind["sma150"])
        & (ind["adx7"] > 45.0)
        & (ind["atr_pct10"] > 4.0)
        & (ind["rsi3"] < 30.0)
        & (frame.index >= pd.Timestamp(test_start))
    )

    out: list[Candidate] = []
    for i in np.flatnonzero(signal.fillna(False).to_numpy()):
        if i + 1 >= len(frame):
            continue
        limit_price = float(frame.iloc[i]["close"]) * 0.96
        atr10 = float(ind.iloc[i]["atr10"])
        rsi3 = float(ind.iloc[i]["rsi3"])
        entry = frame.iloc[i + 1]
        entry_price = None
        if float(entry["open"]) <= limit_price:
            entry_price = float(entry["open"])
        elif float(entry["low"]) <= limit_price:
            entry_price = limit_price

        common = (symbol, _iso_day(frame.index[i]), _iso_day(frame.index[i + 1]),
                  rsi3, atr10, limit_price)
        if entry_price is None:
            out.append(Candidate(
                *common, False, None, None, None, None, None, None, None, None, {}, {}
            ))
            continue

        exit_date, exit_price, reason, timing, exit_open, holding_days = _long_exit(
            frame, i + 1, entry_price, atr10
        )
        exit_i = frame.index.get_loc(pd.Timestamp(exit_date))
        held = frame.iloc[i + 1:exit_i + 1]
        daily_opens = {_iso_day(d): float(v) for d, v in held["open"].items()}
        daily_closes = {_iso_day(d): float(v) for d, v in held["close"].items()}
        out.append(Candidate(
            *common, True, entry_price, float(entry["close"]), exit_date,
            float(exit_price), reason, timing, exit_open, holding_days,
            daily_opens, daily_closes,
        ))
    return out


def rank_orders(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.signal_date].append(candidate)
    selected: list[Candidate] = []
    for rows in grouped.values():
        # The book places at most 10 one-day orders, lowest RSI(3) first.
        selected.extend(sorted(rows, key=lambda x: (x.rank_rsi3, x.symbol))[:10])
    return sorted(selected, key=lambda x: (x.order_date, x.rank_rsi3, x.symbol)), len(candidates)


def _trade_row(pos: Position, exit_cost: float, pnl: float) -> dict:
    c = pos.candidate
    return {
        "symbol": c.symbol, "signal_date": c.signal_date,
        "entry_date": c.order_date, "exit_date": c.exit_date,
        "entry_price": c.entry_price, "exit_price": c.exit_price,
        "shares": pos.shares, "entry_notional": pos.entry_notional,
        "entry_cost": pos.entry_cost, "exit_cost": exit_cost,
        "net_pnl": pnl, "net_return": pnl / pos.entry_notional,
        "exit_reason": c.exit_reason, "exit_timing": c.exit_timing,
        "holding_days": c.holding_days, "rsi3": c.rank_rsi3, "atr10": c.atr10,
    }


def simulate(orders: list[Candidate], trading_days: list[str],
             initial_cash: float = 100_000.0,
             ibkr_tiered: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_entry: dict[str, list[Candidate]] = defaultdict(list)
    for order in orders:
        by_entry[order.order_date].append(order)

    cash = float(initial_cash)
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    curve: list[dict] = []
    blocked_slots = blocked_cash = blocked_duplicate = 0

    for day in trading_days:
        # MOO profit exits and stop gaps free cash before new limit entries.
        for symbol, pos in list(positions.items()):
            c = pos.candidate
            if c.exit_date == day and c.exit_timing == "open":
                proceeds = pos.shares * float(c.exit_price)
                exit_cost = _trade_commission(pos.shares, float(c.exit_price), 0.0, ibkr_tiered)
                cash += proceeds - exit_cost
                pnl = proceeds - pos.entry_notional - pos.entry_cost - exit_cost
                trades.append(_trade_row(pos, exit_cost, pnl))
                del positions[symbol]

        for c in by_entry.get(day, []):
            if not c.filled:
                continue
            if c.symbol in positions:
                blocked_duplicate += 1
                continue
            if len(positions) >= 10:
                blocked_slots += 1
                continue
            equity_open = cash + sum(
                p.shares * p.candidate.daily_opens.get(day, float(p.candidate.entry_close))
                for p in positions.values()
            )
            stop_distance = 2.5 * c.atr10
            risk_shares = math.floor((0.02 * equity_open) / stop_distance) if stop_distance > 0 else 0
            cap_shares = math.floor((0.10 * equity_open) / float(c.entry_price))
            cash_shares = math.floor(cash / float(c.entry_price))
            shares = min(risk_shares, cap_shares, cash_shares)
            if shares <= 0:
                blocked_cash += 1
                continue
            # Commission must also fit within cash; reduce shares if necessary.
            while shares > 0:
                notional = shares * float(c.entry_price)
                entry_cost = _trade_commission(shares, float(c.entry_price), 0.0, ibkr_tiered)
                if notional + entry_cost <= cash + 1e-9:
                    break
                shares -= 1
            if shares <= 0:
                blocked_cash += 1
                continue
            notional = shares * float(c.entry_price)
            entry_cost = _trade_commission(shares, float(c.entry_price), 0.0, ibkr_tiered)
            cash -= notional + entry_cost
            positions[c.symbol] = Position(c, shares, notional, entry_cost)

        # Intraday stops and fourth-session MOC exits happen after entries.
        for symbol, pos in list(positions.items()):
            c = pos.candidate
            if c.exit_date == day and c.exit_timing in ("intraday", "close"):
                proceeds = pos.shares * float(c.exit_price)
                exit_cost = _trade_commission(pos.shares, float(c.exit_price), 0.0, ibkr_tiered)
                cash += proceeds - exit_cost
                pnl = proceeds - pos.entry_notional - pos.entry_cost - exit_cost
                trades.append(_trade_row(pos, exit_cost, pnl))
                del positions[symbol]

        market_value = sum(
            p.shares * p.candidate.daily_closes.get(day, float(p.candidate.entry_close))
            for p in positions.values()
        )
        equity = cash + market_value
        curve.append({
            "date": day, "equity": equity,
            "gross_exposure": market_value / equity if equity > 0 else math.inf,
            "open_positions": len(positions),
        })

    curve_df = pd.DataFrame(curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    rets = curve_df["equity"].pct_change().dropna()
    years = max((pd.Timestamp(curve_df.index[-1]) - pd.Timestamp(curve_df.index[0])).days / 365.2425,
                1 / 365.2425)
    ending = float(curve_df["equity"].iloc[-1])
    cagr = (ending / initial_cash) ** (1 / years) - 1 if ending > 0 else -1.0
    sharpe = math.sqrt(252) * rets.mean() / rets.std(ddof=1) if len(rets) > 2 and rets.std(ddof=1) > 0 else 0.0
    pnl = trades_df.get("net_pnl", pd.Series(dtype=float))
    wins = int((pnl > 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    metrics = {
        "initial_balance": initial_cash, "ending_balance": ending,
        "cagr": float(cagr), "max_drawdown": _max_drawdown(curve_df["equity"]),
        "sharpe": float(sharpe), "trades": len(trades_df), "wins": wins,
        "losses": int((pnl < 0).sum()),
        "win_rate": wins / len(trades_df) if len(trades_df) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "avg_trade_return": float(trades_df["net_return"].mean()) if len(trades_df) else 0.0,
        "total_commissions": float(trades_df.get("entry_cost", pd.Series(dtype=float)).sum()
                                   + trades_df.get("exit_cost", pd.Series(dtype=float)).sum()),
        "avg_gross_exposure": float(curve_df["gross_exposure"].replace([np.inf], np.nan).fillna(0).mean()),
        "max_open_positions": int(curve_df["open_positions"].max()),
        "blocked_by_slots": blocked_slots, "blocked_by_cash": blocked_cash,
        "blocked_duplicate_symbol": blocked_duplicate,
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict() if len(trades_df) else {},
        "ibkr_tiered_commission": ibkr_tiered,
    }
    return curve_df, trades_df, metrics


def write_report(outdir: Path, meta: dict, scenarios: dict,
                 curves: dict[str, pd.DataFrame], benchmark: pd.DataFrame,
                 trades: pd.DataFrame, yearly: pd.DataFrame) -> None:
    def pct(x): return f"{100 * x:.2f}%"
    rows = []
    for name, m in scenarios.items():
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "—"
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>${m['ending_balance']:,.0f}</td>"
            f"<td>{pct(m['cagr'])}</td><td>{pct(m['max_drawdown'])}</td>"
            f"<td>{m['sharpe']:.2f}</td><td>{m['trades']:,}</td>"
            f"<td>{pct(m['win_rate'])}</td><td>{pf}</td>"
            f"<td>${m['total_commissions']:,.0f}</td></tr>"
        )
    chart_series = {name: curve["equity"] for name, curve in curves.items()}
    spy = benchmark.copy()
    spy.index = pd.to_datetime(spy.index).strftime("%Y-%m-%d")
    first_curve = curves[next(iter(curves))]
    chart_series["SPY total return"] = spy["equity"].reindex(first_curve.index).ffill().bfill()
    yearly_html = yearly.to_html(float_format=lambda x: f"{100*x:.1f}%", border=0)
    report = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Long Mean-Reversion Backtest</title><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}main{{max-width:1080px;margin:auto;padding:28px}}.card{{background:white;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 1px 4px #cbd5e1}}.warn{{background:#fff7ed;border-left:5px solid #f97316}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap}}.small{{color:#475569;font-size:13px}}
    </style></head><body><main><div class="card"><h1>Long Mean-Reversion</h1><p>Alpaca SIP · {meta['start']} to {meta['end']} · initial capital $100,000</p></div>
    <div class="card warn"><strong>Important:</strong> Alpaca supplies the current active universe, not historical delisted stocks. Results therefore contain survivorship bias and are not a point-in-time-universe test.</div>
    <div class="card"><h2>Results</h2><table><thead><tr><th>Scenario</th><th>Ending</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Trades</th><th>Win rate</th><th>Profit factor</th><th>Commission</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <div class="card"><h2>Equity curve</h2>{svg_line(chart_series)}</div>
    <div class="card"><h2>Yearly returns</h2>{yearly_html}</div>
    <div class="card"><h2>Rules</h2><ul><li>Current active AMEX/NASDAQ/NYSE common stocks; current ETFs and test issues excluded.</li><li>Raw close &gt; $1; raw 50-day average volume &gt; 500,000; raw 50-day average dollar volume ≥ $2.5M.</li><li>Adjusted close &gt; SMA(150); ADX(7) &gt; 45; ATR(10)% &gt; 4%; RSI(3) &lt; 30.</li><li>Rank lowest RSI(3) first; place at most 10 one-session limit buys at 4% below signal close.</li><li>Fill at the open when open ≤ limit; otherwise at limit when the day's low touches it.</li><li>Risk 2% of current equity to a 2.5×ATR stop; cap each position at 10% of equity; whole shares; no leverage; maximum 10 open positions.</li><li>No stop on entry day. Thereafter: 2.5×ATR stop; a close ≥ 3% above entry exits next open; otherwise exit at the fourth session's close.</li><li>Primary costs: IBKR Pro Tiered $0.0035/share, $0.35 minimum/order, 1% maximum of trade value. Variable exchange and regulatory pass-through fees are excluded.</li></ul></div>
    <div class="card small"><h2>Audit</h2><pre>{html.escape(json.dumps(meta, indent=2))}</pre></div></main></body></html>"""
    (outdir / "mean-reversion-long-report.html").write_text(report, encoding="utf-8")
    trades.to_csv(outdir / "mean-reversion-long-trades.csv", index=False)
    yearly.to_csv(outdir / "mean-reversion-long-yearly.csv")


def run(args) -> None:
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    client = Alpaca()
    universe, universe_audit = build_universe(client)
    print(f"Universe: {len(universe):,} current active non-ETF stocks")

    warmup = _iso_day(pd.Timestamp(args.start) - pd.Timedelta(days=260))
    end = args.end or _iso_day(pd.Timestamp.now(tz="UTC"))
    candidates: list[Candidate] = []
    observed_days: set[str] = set()
    symbols_with_data = 0
    for offset in range(0, len(universe), args.batch_size):
        batch = universe[offset:offset + args.batch_size]
        adjusted = client.bars(batch, warmup, end, adjustment="all")
        raw = client.bars(batch, warmup, end, adjustment="raw")
        for symbol in batch:
            frame = frame_from_bars(adjusted.get(symbol, []))
            raw_frame = frame_from_bars(raw.get(symbol, []))
            if not frame.empty:
                symbols_with_data += 1
                observed_days.update(_iso_day(d) for d in frame.loc[pd.Timestamp(args.start):].index)
                candidates.extend(symbol_candidates(symbol, frame, raw_frame, args.start))
        print(f"Bars {min(offset + args.batch_size, len(universe)):,}/{len(universe):,}; candidates {len(candidates):,}")

    orders, raw_signal_count = rank_orders(candidates)
    trading_days = sorted(d for d in observed_days if args.start <= d <= end)
    if not trading_days:
        raise RuntimeError("No trading days returned")
    with gzip.open(outdir / "mean-reversion-long-orders.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"trading_days": trading_days, "orders": [asdict(x) for x in orders]},
                  handle, separators=(",", ":"))

    scenario_specs = {"Gross / no costs": False, "IBKR Tiered": True}
    scenarios, curves, logs = {}, {}, {}
    for name, tiered in scenario_specs.items():
        curve, log, metrics = simulate(orders, trading_days, ibkr_tiered=tiered)
        scenarios[name], curves[name], logs[name] = metrics, curve, log

    spy = frame_from_bars(client.bars(["SPY"], warmup, end, adjustment="all").get("SPY", []))
    benchmark = benchmark_curve(spy, args.start)
    yearly = pd.DataFrame(index=sorted(set(pd.to_datetime(trading_days).year)))
    for name, curve in curves.items():
        temp = curve.copy(); temp.index = pd.to_datetime(temp.index)
        eoy = temp["equity"].resample("YE").last(); yr = eoy.pct_change()
        if len(yr): yr.iloc[0] = eoy.iloc[0] / temp["equity"].iloc[0] - 1
        yearly[name] = pd.Series(yr.to_numpy(), index=eoy.index.year).reindex(yearly.index)
    b = benchmark.copy(); b.index = pd.to_datetime(b.index)
    eoy = b["equity"].resample("YE").last(); yr = eoy.pct_change()
    if len(yr): yr.iloc[0] = eoy.iloc[0] / b["equity"].iloc[0] - 1
    yearly["SPY total return"] = pd.Series(yr.to_numpy(), index=eoy.index.year).reindex(yearly.index)

    meta = {
        "source": "Alpaca SIP", "start": args.start, "end": trading_days[-1],
        "adjustment": "all for indicators/execution/returns; raw for liquidity screens",
        "universe": universe_audit, "symbols_with_data": symbols_with_data,
        "raw_qualifying_signals": raw_signal_count, "selected_orders": len(orders),
        "filled_selected_orders": sum(x.filled for x in orders),
        "survivorship_bias": True, "historical_delisted_stocks_included": False,
        "commission_model": "IBKR Pro Tiered: $0.0035/share, $0.35 minimum/order, 1% cap",
        "exchange_and_regulatory_pass_through_fees_included": False,
        "daily_bar_ambiguity": "stop has priority after an open exit is checked; no stop on entry day",
    }
    primary = "IBKR Tiered"
    write_report(outdir, meta, scenarios, curves, benchmark, logs[primary], yearly)
    for name, curve in curves.items():
        curve.to_csv(outdir / f"equity-{name.lower().replace(' ', '-').replace('/', '-')}.csv")
    result = {"meta": meta, "scenarios": scenarios}
    rendered = json.dumps(result, indent=2, allow_nan=False, default=json_default)
    (outdir / "mean-reversion-long-results.json").write_text(rendered, encoding="utf-8")
    print(rendered)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=os.environ.get("LONG_BT_START", "2016-01-01"))
    parser.add_argument("--end", default=os.environ.get("LONG_BT_END", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("LONG_BT_BATCH", "100")))
    parser.add_argument("--output", default=os.environ.get("LONG_BT_OUTPUT", "long-backtest-output"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
