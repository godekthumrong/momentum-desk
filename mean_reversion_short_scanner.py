"""Build a standalone daily scanner for the Mean-Reversion Short strategy."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca_prices import completed_sessions
from mean_reversion_short_backtest import (
    Alpaca,
    build_universe,
    frame_from_bars,
    indicators,
)


OUTPUT_HTML = os.environ.get("SHORT_SCANNER_HTML", "short-scanner.html")
OUTPUT_JSON = os.environ.get("SHORT_SCANNER_JSON", "short-scanner.json")
PORTFOLIO_VALUE = float(os.environ.get("SHORT_SCANNER_PORTFOLIO_VALUE", "100000"))
BATCH_SIZE = int(os.environ.get("SHORT_SCANNER_BATCH", "100"))


def scan_frame(symbol: str, adjusted: pd.DataFrame, raw: pd.DataFrame,
               as_of: pd.Timestamp, asset: dict | None = None) -> dict | None:
    """Return the latest-session signal, using exactly the backtest filters."""
    if len(adjusted) < 30 or as_of not in adjusted.index or as_of not in raw.index:
        return None
    ind = indicators(adjusted)
    if as_of not in ind.index:
        return None
    row = ind.loc[as_of]
    raw_close = float(raw.loc[as_of, "close"])
    avg_volume20 = float(raw["volume"].rolling(20, min_periods=20).mean().loc[as_of])
    values = [raw_close, avg_volume20, row["adx7"], row["atr_pct10"],
              row["rsi3"], row["atr10"], adjusted.loc[as_of, "close"]]
    if not all(np.isfinite(v) for v in values):
        return None
    qualifies = (
        raw_close >= 10.0
        and avg_volume20 > 500_000
        and float(row["adx7"]) > 50.0
        and float(row["atr_pct10"]) > 5.0
        and bool(row["two_up"])
        and float(row["rsi3"]) > 85.0
    )
    if not qualifies:
        return None

    adjusted_close = float(adjusted.loc[as_of, "close"])
    raw_atr = float(row["atr10"]) * raw_close / adjusted_close
    risk_per_share = 2.5 * raw_atr
    desired_fraction = min(0.10, 0.02 / (risk_per_share / raw_close))
    notional = PORTFOLIO_VALUE * desired_fraction
    shares = math.floor(notional / raw_close)
    asset = asset or {}
    shortable = bool(asset.get("shortable", False))
    easy = bool(asset.get("easy_to_borrow", False))
    status = "ready" if shortable and easy else ("htb" if shortable else "blocked")
    return {
        "symbol": symbol,
        "rsi3": round(float(row["rsi3"]), 2),
        "adx7": round(float(row["adx7"]), 2),
        "atr_pct10": round(float(row["atr_pct10"]), 2),
        "atr10_raw": round(raw_atr, 4),
        "raw_close": round(raw_close, 4),
        "avg_volume20": round(avg_volume20),
        "limit_price": round(raw_close, 4),
        "estimated_stop": round(raw_close + risk_per_share, 4),
        "allocation_pct": round(desired_fraction * 100, 3),
        "shares": shares,
        "notional": round(shares * raw_close, 2),
        "shortable": shortable,
        "easy_to_borrow": easy,
        "borrow_status": status,
    }


def build_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Short Scanner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap">
<style>
:root{{--ink:#14213d;--muted:#64748b;--line:#dbe3ed;--paper:#f4f7fb;--card:#fff;--red:#dc2626;--amber:#d97706;--green:#15803d;--blue:#1d4ed8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",sans-serif}}
.wrap{{max-width:1240px;margin:auto;padding:28px 22px 60px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:22px}}
.eyebrow{{font:600 12px "IBM Plex Mono";letter-spacing:.08em;text-transform:uppercase;color:var(--red)}}h1{{font-size:42px;line-height:1.05;margin:8px 0}}.sub{{max-width:760px;color:var(--muted);margin:0}}
.nav{{white-space:nowrap;color:var(--blue);text-decoration:none;font-weight:600}}.badge{{display:inline-flex;align-items:center;gap:7px;background:#fee2e2;color:#991b1b;border-radius:999px;padding:7px 11px;font:600 12px "IBM Plex Mono"}}
.dot{{width:8px;height:8px;background:var(--red);border-radius:50%}}.controls,.kpis,.panel{{background:var(--card);border:1px solid var(--line);border-radius:13px}}
.controls{{display:flex;gap:16px;align-items:end;padding:15px 17px;margin-bottom:14px;flex-wrap:wrap}}label{{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}}input,select{{height:38px;border:1px solid var(--line);border-radius:7px;padding:0 10px;background:white;font:500 14px "IBM Plex Mono"}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);overflow:hidden;margin-bottom:18px}}.kpi{{padding:15px;border-right:1px solid var(--line)}}.kpi:last-child{{border:0}}.kpi b{{display:block;font:600 22px "IBM Plex Mono"}}.kpi span{{font-size:12px;color:var(--muted)}}
.panel{{overflow:hidden}}.panel-head{{display:flex;justify-content:space-between;gap:16px;padding:18px;border-bottom:1px solid var(--line)}}h2{{font-size:19px;margin:0}}.note{{font-size:12px;color:var(--muted)}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1050px}}th,td{{padding:11px 12px;border-bottom:1px solid #e8edf3;text-align:right;font-size:13px;font-variant-numeric:tabular-nums}}th{{position:sticky;top:0;background:#f8fafc;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{{text-align:left}}td.mono{{font-family:"IBM Plex Mono";font-weight:600}}tr.top td:first-child{{color:var(--red);font-weight:700}}
.status{{display:inline-block;border-radius:999px;padding:4px 8px;font:600 11px "IBM Plex Mono"}}.ready{{background:#dcfce7;color:#166534}}.htb{{background:#fef3c7;color:#92400e}}.blocked{{background:#fee2e2;color:#991b1b}}
.empty{{padding:48px 20px;text-align:center;color:var(--muted)}}footer{{margin-top:18px;color:var(--muted);font-size:12px;max-width:100ch}}footer strong{{color:var(--ink)}}
@media(max-width:760px){{.wrap{{padding:20px 12px 45px}}header{{display:block}}.nav{{display:inline-block;margin-top:14px}}h1{{font-size:34px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.kpi{{border-bottom:1px solid var(--line)}}}}
</style></head><body><div class="wrap">
<header><div><div class="eyebrow">Alpaca SIP · Mean-Reversion Short</div><h1>Daily Short Scanner</h1><p class="sub">Overextended stocks that meet the strategy rules at the latest completed US close. Top 10 are tomorrow's candidate orders.</p></div><a class="nav" href="index.html">Momentum Desk →</a></header>
<div class="controls"><span class="badge"><i class="dot"></i>Snapshot · {payload['meta']['as_of']} close</span><div><label for="pv">Portfolio value (USD)</label><input id="pv" type="number" min="1" step="100" value="{payload['meta']['portfolio_value']:.0f}"></div><div><label for="filter">Show</label><select id="filter"><option value="all">All qualifying</option><option value="top">Top 10 only</option><option value="ready">Easy-to-borrow only</option></select></div></div>
<div class="kpis"><div class="kpi"><b>{payload['meta']['signals']}</b><span>Qualifying signals</span></div><div class="kpi"><b>{payload['meta']['selected']}</b><span>Top-ranked orders</span></div><div class="kpi"><b>{payload['meta']['ready']}</b><span>Easy to borrow</span></div><div class="kpi"><b>{payload['meta']['fresh']:,}</b><span>Fresh symbols</span></div><div class="kpi"><b>{payload['meta']['universe']:,}</b><span>Active-stock universe</span></div></div>
<section class="panel"><div class="panel-head"><div><h2>Signals and next-session order plan</h2><div class="note">Ranked by RSI(3). Position risk 2%, maximum 10% each and 100% total short overlay.</div></div><div class="note">Limit = latest raw close · estimated stop = limit + 2.5×ATR(10)</div></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Ticker</th><th>RSI(3)</th><th>ADX(7)</th><th>ATR%</th><th>Limit</th><th>Est. stop</th><th>20D avg vol</th><th>Allocation</th><th>Shares</th><th>Borrow</th></tr></thead><tbody id="rows"></tbody></table><div class="empty" id="empty" hidden>No signals match every rule at this close.</div></div></section>
<footer><p><strong>Signal rules:</strong> raw close ≥ $10; raw 20-day average volume &gt; 500,000; ADX(7) &gt; 50; ATR(10)/close &gt; 5%; last two adjusted closes up; RSI(3) &gt; 85.</p><p><strong>Execution:</strong> Submit the short limit for the next session. The displayed stop is an estimate at the limit price; the strategy sets the real stop from the actual fill and does not activate it on entry day. A close at least 4% below entry exits next open; otherwise cover no later than the second session. “HTB” means currently shortable but not easy-to-borrow, so the assumed 0.25% annual borrow cost may not apply. Availability can change before the order fills.</p><p>Current active stocks only; ETFs and test issues excluded. This is a research scanner, not an order router.</p></footer>
</div><script>window.__SCAN__={data};
(function(){{var D=window.__SCAN__, rows=D.signals, body=document.getElementById('rows'), empty=document.getElementById('empty'), pv=document.getElementById('pv'), filter=document.getElementById('filter');
function money(v){{return '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}}
function render(){{var capital=Math.max(1,Number(pv.value)||D.meta.portfolio_value), mode=filter.value, shown=rows.filter(function(r){{return mode==='all'||(mode==='top'&&r.selected)||(mode==='ready'&&r.borrow_status==='ready')}}); body.innerHTML=shown.map(function(r){{var risk=2.5*r.atr10_raw, frac=Math.min(.10,.02/(risk/r.limit_price)), shares=Math.floor(capital*frac/r.limit_price), cls=r.borrow_status, label=cls==='ready'?'READY / ETB':(cls==='htb'?'HTB':'NOT SHORTABLE'); return '<tr class="'+(r.selected?'top':'')+'"><td>'+(r.selected?r.rank:'—')+'</td><td class="mono">'+r.symbol+'</td><td>'+r.rsi3.toFixed(1)+'</td><td>'+r.adx7.toFixed(1)+'</td><td>'+r.atr_pct10.toFixed(1)+'%</td><td>'+money(r.limit_price)+'</td><td>'+money(r.estimated_stop)+'</td><td>'+Number(r.avg_volume20).toLocaleString()+'</td><td>'+(frac*100).toFixed(1)+'%</td><td class="mono">'+shares.toLocaleString()+'</td><td><span class="status '+cls+'">'+label+'</span></td></tr>'}}).join(''); empty.hidden=shown.length>0;}}
pv.addEventListener('input',render);filter.addEventListener('change',render);render();}})();</script></body></html>'''


def run() -> dict:
    sessions, end = completed_sessions()
    as_of = pd.Timestamp(sessions[-1]).normalize()
    history = sessions[-80:]
    start = str(pd.Timestamp(history[0]).date())
    client = Alpaca()
    universe, universe_audit = build_universe(client)
    assets = {str(a.get("symbol", "")): a for a in client.active_assets()}
    signals: list[dict] = []
    fresh = 0
    for offset in range(0, len(universe), BATCH_SIZE):
        batch = universe[offset:offset + BATCH_SIZE]
        adjusted_payload = client.bars(batch, start, end.isoformat(), adjustment="all")
        raw_payload = client.bars(batch, start, end.isoformat(), adjustment="raw")
        for symbol in batch:
            adjusted = frame_from_bars(adjusted_payload.get(symbol, []))
            raw = frame_from_bars(raw_payload.get(symbol, []))
            if as_of in adjusted.index and as_of in raw.index:
                fresh += 1
            signal = scan_frame(symbol, adjusted, raw, as_of, assets.get(symbol))
            if signal:
                signals.append(signal)
        print(f"Scanned {min(offset+BATCH_SIZE, len(universe)):,}/{len(universe):,}; signals {len(signals)}")
    if fresh < math.ceil(0.90 * len(universe)):
        raise RuntimeError(f"Only {fresh}/{len(universe)} symbols have the completed-session bar")

    signals.sort(key=lambda row: (-row["rsi3"], row["symbol"]))
    for i, row in enumerate(signals):
        row["selected"] = i < 10
        row["rank"] = i + 1 if i < 10 else None
    payload = {
        "meta": {
            "as_of": str(as_of.date()),
            "portfolio_value": PORTFOLIO_VALUE,
            "universe": len(universe),
            "fresh": fresh,
            "signals": len(signals),
            "selected": min(10, len(signals)),
            "ready": sum(row["borrow_status"] == "ready" for row in signals[:10]),
            "source": "Alpaca SIP",
            "adjustment": "all for indicators; raw for price/volume/order levels",
            "universe_audit": universe_audit,
        },
        "signals": signals,
    }
    Path(OUTPUT_JSON).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    Path(OUTPUT_HTML).write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} and {OUTPUT_JSON}")
    return payload


if __name__ == "__main__":
    run()
