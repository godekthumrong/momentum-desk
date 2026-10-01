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
        "sizing_distance": round(risk_per_share, 4),
        "allocation_pct": round(desired_fraction * 100, 3),
        "shares": shares,
        "notional": round(shares * raw_close, 2),
        "shortable": shortable,
        "easy_to_borrow": easy,
        "borrow_status": status,
    }


def evaluate_exit(entry_date: str, entry_price: float, bars: list[list]) -> dict:
    """Exit at the first completed session close after the entry session."""
    ordered = sorted((bar for bar in bars if bar[0] >= entry_date), key=lambda bar: bar[0])
    entry_rows = [bar for bar in ordered if bar[0] == entry_date]
    if not entry_rows:
        return {"state": "waiting_data", "label": "Waiting for entry-day data"}
    later = [bar for bar in ordered if bar[0] > entry_date]
    if not later:
        return {"state": "hold", "label": "Hold · exit next-session close"}
    day2 = later[0]
    return {"state": "exit", "label": "Time exit at close",
            "exit_date": day2[0], "exit_price": day2[4]}


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
.status{{display:inline-block;border-radius:999px;padding:4px 8px;font:600 11px "IBM Plex Mono"}}.ready,.hold{{background:#dcfce7;color:#166534}}.htb,.pending,.waiting_data{{background:#fef3c7;color:#92400e}}.blocked,.action,.exit{{background:#fee2e2;color:#991b1b}}
.watch-panel,.journal-panel{{margin-top:18px}}.watch-form,.journal-form{{display:flex;gap:10px;align-items:end;padding:15px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}}.watch-form input,.journal-form input{{width:135px}}.watch-form input.symbol,.journal-form input.symbol{{width:105px;text-transform:uppercase}}.journal-form input.note-input{{width:210px}}button{{height:38px;border:0;border-radius:7px;padding:0 13px;font-weight:600;cursor:pointer}}button.primary{{background:var(--ink);color:white}}button.secondary{{background:#e8eef7;color:var(--ink)}}button.danger{{background:#fee2e2;color:#991b1b;height:30px}}button.watch{{height:30px;background:#e0e7ff;color:#3730a3}}button.watch:disabled{{background:#dcfce7;color:#166534;cursor:default}}button.compact{{height:30px}}.watch-help{{padding:0 18px 14px;color:var(--muted);font-size:12px}}#watch-table{{min-width:860px}}#journal-table{{min-width:1120px}}
.trade-stats{{display:grid;grid-template-columns:repeat(6,1fr);border-bottom:1px solid var(--line)}}.trade-stat{{padding:14px 16px;border-right:1px solid var(--line)}}.trade-stat:last-child{{border:0}}.trade-stat b{{display:block;font:600 19px "IBM Plex Mono"}}.trade-stat span{{font-size:11px;color:var(--muted)}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}.chart-box{{padding:16px 18px;border-bottom:1px solid var(--line)}}.chart-title{{display:flex;justify-content:space-between;gap:12px;margin-bottom:8px}}#return-chart{{display:block;width:100%;height:250px;background:#f8fafc;border:1px solid var(--line);border-radius:9px}}.chart-empty{{height:250px;display:grid;place-items:center;background:#f8fafc;border:1px dashed var(--line);border-radius:9px;color:var(--muted);font-size:13px}}
.empty{{padding:48px 20px;text-align:center;color:var(--muted)}}footer{{margin-top:18px;color:var(--muted);font-size:12px;max-width:100ch}}footer strong{{color:var(--ink)}}
@media(max-width:760px){{.wrap{{padding:20px 12px 45px}}header{{display:block}}.nav{{display:inline-block;margin-top:14px}}h1{{font-size:34px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.kpi{{border-bottom:1px solid var(--line)}}.trade-stats{{grid-template-columns:repeat(2,1fr)}}.trade-stat{{border-bottom:1px solid var(--line)}}}}
</style></head><body><div class="wrap">
<header><div><div class="eyebrow">Alpaca SIP · Mean-Reversion Short</div><h1>Daily Short Scanner</h1><p class="sub">Overextended stocks that meet the strategy rules at the latest completed US close. Top 10 are tomorrow's candidate orders.</p></div><a class="nav" href="index.html">Momentum Desk →</a></header>
<div class="controls"><span class="badge"><i class="dot"></i>Snapshot · {payload['meta']['as_of']} close</span><div><label for="pv">Portfolio value (USD)</label><input id="pv" type="number" min="1" step="100" value="{payload['meta']['portfolio_value']:.0f}"></div><div><label for="filter">Show</label><select id="filter"><option value="all">All qualifying</option><option value="top">Top 10 only</option><option value="ready">Easy-to-borrow only</option></select></div></div>
<div class="kpis"><div class="kpi"><b>{payload['meta']['signals']}</b><span>Qualifying signals</span></div><div class="kpi"><b>{payload['meta']['selected']}</b><span>Top-ranked orders</span></div><div class="kpi"><b>{payload['meta']['ready']}</b><span>Easy to borrow</span></div><div class="kpi"><b>{payload['meta']['fresh']:,}</b><span>Fresh symbols</span></div><div class="kpi"><b>{payload['meta']['universe']:,}</b><span>Active-stock universe</span></div></div>
<section class="panel"><div class="panel-head"><div><h2>Signals and next-session order plan</h2><div class="note">Ranked by RSI(3). Position risk 2%, maximum 10% each and 100% total short overlay.</div></div><div class="note">Sizing distance = 2.5×ATR(10) · used for allocation only, not as a stop</div></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Ticker</th><th>Limit</th><th>Allocation</th><th>Shares</th><th>Sizing dist.</th><th>RSI(3)</th><th>ADX(7)</th><th>ATR%</th><th>20D avg vol</th><th>Alpaca borrow</th><th>Watch</th></tr></thead><tbody id="rows"></tbody></table><div class="empty" id="empty" hidden>No signals match every rule at this close.</div></div></section>
<section class="panel watch-panel"><div class="panel-head"><div><h2>Position watchlist</h2><div class="note">Saved only in this browser. Enter the actual fill date and price after a short fills.</div></div><div class="note" id="watch-count"></div></div>
<form class="watch-form" id="watch-form"><div><label for="w-symbol">Ticker</label><input class="symbol" id="w-symbol" maxlength="10" required></div><div><label for="w-date">Actual fill date</label><input id="w-date" type="date"></div><div><label for="w-price">Actual fill price</label><input id="w-price" type="number" min="0" step="0.0001"></div><div><label for="w-shares">Shares</label><input id="w-shares" type="number" min="0" step="1"></div><button class="primary" type="submit">Save position</button><button class="secondary" id="w-cancel" type="button">Clear form</button></form>
<div class="watch-help">Tap “Watch” to save the signal immediately with its planned limit and shares—no form entry required. It stays Pending until you use Edit to enter the actual fill date and price. After the fill, the only exit is the close of the next completed session; there is no stop loss or take profit.</div><div class="table-wrap"><table id="watch-table"><thead><tr><th>Ticker</th><th>Status / instruction</th><th>Fill date</th><th>Entry</th><th>Latest close</th><th>Open P&amp;L</th><th>Shares</th><th>Actions</th></tr></thead><tbody id="watch-rows"></tbody></table><div class="empty" id="watch-empty">No watched positions yet. Tap “Watch” beside a signal.</div></div></section>
<section class="panel journal-panel"><div class="panel-head"><div><h2>Trade journal</h2><div class="note">Closed short trades saved only in this browser. Returns and P&amp;L are net of the fees you enter.</div></div><button class="secondary" id="journal-export" type="button">Export CSV</button></div>
<div class="trade-stats"><div class="trade-stat"><b id="stat-trades">0</b><span>Closed trades</span></div><div class="trade-stat"><b id="stat-win">—</b><span>Win rate</span></div><div class="trade-stat"><b id="stat-pnl">$0.00</b><span>Net P&amp;L</span></div><div class="trade-stat"><b id="stat-avg">—</b><span>Average return</span></div><div class="trade-stat"><b id="stat-pf">—</b><span>Profit factor</span></div><div class="trade-stat"><b id="stat-dd">—</b><span>Max drawdown</span></div></div>
<div class="chart-box"><div class="chart-title"><strong>Cumulative portfolio return</strong><span class="note">Based on the portfolio value above and realized net P&amp;L</span></div><svg id="return-chart" viewBox="0 0 960 250" role="img" aria-label="Cumulative portfolio return chart"></svg><div class="chart-empty" id="chart-empty">Add a closed trade to start the return chart.</div></div>
<form class="journal-form" id="journal-form"><input id="j-id" type="hidden"><div><label for="j-symbol">Ticker</label><input class="symbol" id="j-symbol" maxlength="10" required></div><div><label for="j-entry-date">Entry date</label><input id="j-entry-date" type="date" required></div><div><label for="j-entry-price">Entry price</label><input id="j-entry-price" type="number" min="0.0001" step="0.0001" required></div><div><label for="j-exit-date">Exit date</label><input id="j-exit-date" type="date" required></div><div><label for="j-exit-price">Exit price</label><input id="j-exit-price" type="number" min="0.0001" step="0.0001" required></div><div><label for="j-shares">Shares</label><input id="j-shares" type="number" min="1" step="1" required></div><div><label for="j-fees">Total fees (USD)</label><input id="j-fees" type="number" min="0" step="0.01" value="0"></div><div><label for="j-note">Note</label><input class="note-input" id="j-note" maxlength="120"></div><button class="primary" type="submit">Save trade</button><button class="secondary" id="j-cancel" type="button">Clear form</button></form>
<div class="table-wrap"><table id="journal-table"><thead><tr><th>Exit date</th><th>Ticker</th><th>Entry</th><th>Exit</th><th>Shares</th><th>Fees</th><th>Net return</th><th>Net P&amp;L</th><th>Note</th><th>Actions</th></tr></thead><tbody id="journal-rows"></tbody></table><div class="empty" id="journal-empty">No closed trades saved yet.</div></div></section>
<footer><p><strong>Signal rules:</strong> raw close ≥ $10; raw 20-day average volume &gt; 500,000; ADX(7) &gt; 50; ATR(10)/close &gt; 5%; last two adjusted closes up; RSI(3) &gt; 85.</p><p><strong>Execution:</strong> Submit the short limit for the next session, then cover at the close of the first completed session after the fill. There is no stop loss or take profit. Position size still uses a 2.5×ATR(10) sizing distance, 2% portfolio risk and a 10% per-position cap. “HTB” means currently shortable but not easy-to-borrow, so the assumed 0.25% annual borrow cost may not apply. Availability can change before the order fills.</p><p>Current active stocks only; ETFs and test issues excluded. This is a research scanner, not an order router.</p></footer>
</div><script>window.__SCAN__={data};
(function(){{var D=window.__SCAN__, rows=D.signals, body=document.getElementById('rows'), empty=document.getElementById('empty'), pv=document.getElementById('pv'), filter=document.getElementById('filter'), STORE='short-scanner-watchlist-v1', JSTORE='short-scanner-journal-v1';
function money(v){{return '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}}
function signedMoney(v){{return (v>0?'+':'')+money(v)}}
function esc(s){{return String(s).replace(/[&<>"']/g,function(c){{return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]}})}}
function loadWatch(){{try{{var x=JSON.parse(localStorage.getItem(STORE)||'[]');return Array.isArray(x)?x:[]}}catch(e){{return []}}}}var watch=loadWatch();
function loadJournal(){{try{{var x=JSON.parse(localStorage.getItem(JSTORE)||'[]');return Array.isArray(x)?x:[]}}catch(e){{return []}}}}var journal=loadJournal();
function saveWatch(){{localStorage.setItem(STORE,JSON.stringify(watch));renderWatch();render()}}
function saveJournal(){{localStorage.setItem(JSTORE,JSON.stringify(journal));renderJournal()}}
function addWatch(r){{var old=watch.find(function(x){{return x.symbol===r.symbol}});if(old)return;var capital=Math.max(1,Number(pv.value)||D.meta.portfolio_value),risk=2.5*r.atr10_raw,frac=Math.min(.10,.02/(risk/r.limit_price)),shares=Math.floor(capital*frac/r.limit_price);watch.push({{symbol:r.symbol,entryDate:'',entryPrice:null,shares:shares,plannedLimit:r.limit_price,added:D.meta.as_of}});saveWatch()}}
function evaluate(p){{if(!p.entryDate||!(p.entryPrice>0))return {{state:'pending',label:'Pending · enter actual fill'}};var bars=(D.market[p.symbol]||[]).filter(function(b){{return b[0]>=p.entryDate}}).sort(function(a,b){{return a[0].localeCompare(b[0])}}),entry=bars.find(function(b){{return b[0]===p.entryDate}});if(!entry)return {{state:'waiting_data',label:'Waiting for entry-day data'}};var later=bars.filter(function(b){{return b[0]>p.entryDate}});if(!later.length)return {{state:'hold',label:'HOLD · exit next-session close'}};var d=later[0];return {{state:'exit',label:'Time exit at close',exitDate:d[0],exitPrice:d[4]}}}}
function render(){{var capital=Math.max(1,Number(pv.value)||D.meta.portfolio_value), mode=filter.value, shown=rows.filter(function(r){{return mode==='all'||(mode==='top'&&r.selected)||(mode==='ready'&&r.borrow_status==='ready')}}); body.innerHTML=shown.map(function(r){{var risk=2.5*r.atr10_raw, frac=Math.min(.10,.02/(risk/r.limit_price)), shares=Math.floor(capital*frac/r.limit_price), cls=r.borrow_status, label=cls==='ready'?'READY / ETB':(cls==='htb'?'HTB':'NOT SHORTABLE'),watched=watch.some(function(x){{return x.symbol===r.symbol}}); return '<tr class="'+(r.selected?'top':'')+'"><td>'+(r.selected?r.rank:'—')+'</td><td class="mono">'+r.symbol+'</td><td>'+money(r.limit_price)+'</td><td>'+(frac*100).toFixed(1)+'%</td><td class="mono">'+shares.toLocaleString()+'</td><td>'+money(risk)+'</td><td>'+r.rsi3.toFixed(1)+'</td><td>'+r.adx7.toFixed(1)+'</td><td>'+r.atr_pct10.toFixed(1)+'%</td><td>'+Number(r.avg_volume20).toLocaleString()+'</td><td><span class="status '+cls+'">'+label+'</span></td><td><button class="watch" data-watch="'+r.symbol+'"'+(watched?' disabled':'')+'>'+(watched?'Added':'Watch')+'</button></td></tr>'}}).join(''); empty.hidden=shown.length>0;body.querySelectorAll('[data-watch]:not([disabled])').forEach(function(b){{b.onclick=function(){{addWatch(rows.find(function(r){{return r.symbol===b.dataset.watch}}))}}}})}}
function renderWatch(){{var wb=document.getElementById('watch-rows'),we=document.getElementById('watch-empty');document.getElementById('watch-count').textContent=watch.length+' saved';wb.innerHTML=watch.map(function(p){{var e=evaluate(p),bars=D.market[p.symbol]||[],last=bars.length?bars[bars.length-1]:null,latest=last?last[4]:null,pnl=(p.entryPrice>0&&latest!=null)?(p.entryPrice-latest)/p.entryPrice:null,detail=e.exitDate?' · '+e.exitDate+' @ '+money(e.exitPrice):'',journalButton=e.exitPrice&&p.shares>0?' <button class="primary compact" data-journal="'+esc(p.symbol)+'">Journal</button>':'';return '<tr><td class="mono">'+esc(p.symbol)+'</td><td><span class="status '+e.state+'">'+esc(e.label)+'</span>'+detail+'</td><td>'+(p.entryDate||'—')+'</td><td>'+(p.entryPrice>0?money(p.entryPrice):(p.plannedLimit?'<span class="note">Plan '+money(p.plannedLimit)+'</span>':'—'))+'</td><td>'+(latest!=null?money(latest):'—')+'</td><td>'+(pnl==null?'—':(pnl*100).toFixed(2)+'%')+'</td><td>'+(p.shares>0?Number(p.shares).toLocaleString():'—')+'</td><td>'+journalButton+' <button class="secondary compact" data-edit="'+esc(p.symbol)+'">Edit</button> <button class="danger" data-remove="'+esc(p.symbol)+'">Remove</button></td></tr>'}}).join('');we.hidden=watch.length>0;wb.querySelectorAll('[data-journal]').forEach(function(b){{b.onclick=function(){{prefillJournal(b.dataset.journal)}}}});wb.querySelectorAll('[data-edit]').forEach(function(b){{b.onclick=function(){{editWatch(b.dataset.edit)}}}});wb.querySelectorAll('[data-remove]').forEach(function(b){{b.onclick=function(){{watch=watch.filter(function(p){{return p.symbol!==b.dataset.remove}});saveWatch()}}}})}}
function editWatch(symbol){{var p=watch.find(function(x){{return x.symbol===symbol}});if(!p)return;document.getElementById('w-symbol').value=p.symbol;document.getElementById('w-date').value=p.entryDate||'';document.getElementById('w-price').value=p.entryPrice||'';document.getElementById('w-shares').value=p.shares||'';document.getElementById('w-price').focus();document.getElementById('watch-form').scrollIntoView({{behavior:'smooth',block:'center'}})}}
function clearForm(){{document.getElementById('watch-form').reset()}}document.getElementById('watch-form').onsubmit=function(ev){{ev.preventDefault();var symbol=document.getElementById('w-symbol').value.trim().toUpperCase();if(!symbol)return;var p=watch.find(function(x){{return x.symbol===symbol}});if(!p){{p={{symbol:symbol,added:D.meta.as_of}};watch.push(p)}}p.entryDate=document.getElementById('w-date').value;p.entryPrice=Number(document.getElementById('w-price').value)||null;p.shares=Number(document.getElementById('w-shares').value)||null;saveWatch();clearForm()}};document.getElementById('w-cancel').onclick=clearForm;
function tradeCalc(t){{var shares=Number(t.shares)||0,fees=Number(t.fees)||0,gross=(Number(t.entryPrice)-Number(t.exitPrice))*shares,net=gross-fees,base=Number(t.entryPrice)*shares;return {{net:net,ret:base>0?net/base:0}}}}
function setMetric(id,text,value){{var el=document.getElementById(id);el.textContent=text;el.className=value>0?'positive':(value<0?'negative':'')}}
function renderChart(sorted){{var svg=document.getElementById('return-chart'),emptyChart=document.getElementById('chart-empty');if(!sorted.length){{svg.hidden=true;emptyChart.hidden=false;svg.innerHTML='';return}}svg.hidden=false;emptyChart.hidden=true;var capital=Math.max(1,Number(pv.value)||D.meta.portfolio_value),equity=capital,peak=capital,maxDD=0,series=[{{x:0,y:0,label:'Start'}}];sorted.forEach(function(t,i){{equity+=tradeCalc(t).net;peak=Math.max(peak,equity);maxDD=Math.min(maxDD,(equity-peak)/peak);series.push({{x:i+1,y:(equity/capital-1)*100,label:t.exitDate}})}});var ys=series.map(function(p){{return p.y}}),min=Math.min(0,Math.min.apply(null,ys)),max=Math.max(0,Math.max.apply(null,ys));if(max-min<1){{max+=.5;min-=.5}}var left=58,right=20,top=20,bottom=38,w=960-left-right,h=250-top-bottom,x=function(i){{return left+(i/(series.length-1||1))*w}},y=function(v){{return top+(max-v)/(max-min)*h}},zero=y(0),path=series.map(function(p,i){{return (i?'L':'M')+x(i).toFixed(1)+' '+y(p.y).toFixed(1)}}).join(' '),area=path+' L '+x(series.length-1).toFixed(1)+' '+zero.toFixed(1)+' L '+x(0).toFixed(1)+' '+zero.toFixed(1)+' Z',last=series[series.length-1];svg.innerHTML='<line x1="'+left+'" y1="'+zero+'" x2="'+(960-right)+'" y2="'+zero+'" stroke="#94a3b8" stroke-dasharray="4 4"/><text x="8" y="'+(top+5)+'" fill="#64748b" font-size="12">'+max.toFixed(2)+'%</text><text x="8" y="'+(top+h+4)+'" fill="#64748b" font-size="12">'+min.toFixed(2)+'%</text><path d="'+area+'" fill="rgba(29,78,216,.10)"/><path d="'+path+'" fill="none" stroke="#1d4ed8" stroke-width="3" stroke-linejoin="round"/><circle cx="'+x(series.length-1)+'" cy="'+y(last.y)+'" r="5" fill="#1d4ed8"/><text x="'+Math.max(left,x(series.length-1)-72)+'" y="235" fill="#64748b" font-size="12">'+esc(last.label)+'</text><text x="'+Math.max(left,x(series.length-1)-45)+'" y="'+Math.max(16,y(last.y)-10)+'" fill="#1d4ed8" font-size="13" font-weight="600">'+(last.y>0?'+':'')+last.y.toFixed(2)+'%</text>';return maxDD}}
function renderJournal(){{var jb=document.getElementById('journal-rows'),je=document.getElementById('journal-empty'),sorted=journal.slice().sort(function(a,b){{return (a.exitDate||'').localeCompare(b.exitDate||'')||(a.id||'').localeCompare(b.id||'')}}),calcs=sorted.map(tradeCalc),wins=calcs.filter(function(c){{return c.net>0}}).length,total=calcs.reduce(function(s,c){{return s+c.net}},0),avg=calcs.length?calcs.reduce(function(s,c){{return s+c.ret}},0)/calcs.length:0,gain=calcs.reduce(function(s,c){{return s+Math.max(0,c.net)}},0),loss=-calcs.reduce(function(s,c){{return s+Math.min(0,c.net)}},0),pf=loss>0?gain/loss:(gain>0?Infinity:null),maxDD=renderChart(sorted);setMetric('stat-trades',String(sorted.length),0);setMetric('stat-win',sorted.length?(wins/sorted.length*100).toFixed(1)+'%':'—',sorted.length?wins/sorted.length-.5:0);setMetric('stat-pnl',signedMoney(total),total);setMetric('stat-avg',sorted.length?(avg*100).toFixed(2)+'%':'—',avg);setMetric('stat-pf',pf===Infinity?'∞':(pf==null?'—':pf.toFixed(2)),pf===Infinity?1:(pf==null?0:pf-1));setMetric('stat-dd',sorted.length?(maxDD*100).toFixed(2)+'%':'—',maxDD);jb.innerHTML=sorted.slice().reverse().map(function(t){{var c=tradeCalc(t),cls=c.net>0?'positive':(c.net<0?'negative':'');return '<tr><td>'+esc(t.exitDate)+'</td><td class="mono">'+esc(t.symbol)+'</td><td>'+esc(t.entryDate)+' @ '+money(t.entryPrice)+'</td><td>'+esc(t.exitDate)+' @ '+money(t.exitPrice)+'</td><td>'+Number(t.shares).toLocaleString()+'</td><td>'+money(t.fees||0)+'</td><td class="'+cls+'">'+(c.ret>0?'+':'')+(c.ret*100).toFixed(2)+'%</td><td class="'+cls+'">'+signedMoney(c.net)+'</td><td>'+esc(t.note||'—')+'</td><td><button class="secondary compact" data-jedit="'+esc(t.id)+'">Edit</button> <button class="danger" data-jremove="'+esc(t.id)+'">Delete</button></td></tr>'}}).join('');je.hidden=sorted.length>0;jb.querySelectorAll('[data-jedit]').forEach(function(b){{b.onclick=function(){{editJournal(b.dataset.jedit)}}}});jb.querySelectorAll('[data-jremove]').forEach(function(b){{b.onclick=function(){{journal=journal.filter(function(t){{return t.id!==b.dataset.jremove}});saveJournal()}}}})}}
function journalFields(t){{document.getElementById('j-id').value=t.id||'';document.getElementById('j-symbol').value=t.symbol||'';document.getElementById('j-entry-date').value=t.entryDate||'';document.getElementById('j-entry-price').value=t.entryPrice||'';document.getElementById('j-exit-date').value=t.exitDate||'';document.getElementById('j-exit-price').value=t.exitPrice||'';document.getElementById('j-shares').value=t.shares||'';document.getElementById('j-fees').value=t.fees||0;document.getElementById('j-note').value=t.note||'';document.getElementById('journal-form').scrollIntoView({{behavior:'smooth',block:'center'}})}}
function editJournal(id){{var t=journal.find(function(x){{return x.id===id}});if(t)journalFields(t)}}
function prefillJournal(symbol){{var p=watch.find(function(x){{return x.symbol===symbol}});if(!p)return;var e=evaluate(p);journalFields({{symbol:p.symbol,entryDate:p.entryDate,entryPrice:p.entryPrice,exitDate:e.exitDate||'',exitPrice:e.exitPrice||'',shares:p.shares,fees:0,note:e.label||''}})}}
function clearJournalForm(){{document.getElementById('journal-form').reset();document.getElementById('j-id').value='';document.getElementById('j-fees').value='0'}}document.getElementById('journal-form').onsubmit=function(ev){{ev.preventDefault();var id=document.getElementById('j-id').value||('t'+Date.now().toString(36)+Math.random().toString(36).slice(2,6)),t={{id:id,symbol:document.getElementById('j-symbol').value.trim().toUpperCase(),entryDate:document.getElementById('j-entry-date').value,entryPrice:Number(document.getElementById('j-entry-price').value),exitDate:document.getElementById('j-exit-date').value,exitPrice:Number(document.getElementById('j-exit-price').value),shares:Number(document.getElementById('j-shares').value),fees:Number(document.getElementById('j-fees').value)||0,note:document.getElementById('j-note').value.trim()}};var i=journal.findIndex(function(x){{return x.id===id}});if(i>=0)journal[i]=t;else journal.push(t);saveJournal();clearJournalForm()}};document.getElementById('j-cancel').onclick=clearJournalForm;
document.getElementById('journal-export').onclick=function(){{if(!journal.length)return;var lines=[['Exit date','Ticker','Entry date','Entry price','Exit price','Shares','Fees','Net return','Net P&L','Note']].concat(journal.slice().sort(function(a,b){{return a.exitDate.localeCompare(b.exitDate)}}).map(function(t){{var c=tradeCalc(t);return [t.exitDate,t.symbol,t.entryDate,t.entryPrice,t.exitPrice,t.shares,t.fees||0,(c.ret*100).toFixed(4),c.net.toFixed(2),t.note||'']}})),csv=lines.map(function(row){{return row.map(function(v){{return '"'+String(v).replace(/"/g,'""')+'"'}}).join(',')}}).join('\\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='short-trade-journal.csv';a.click();URL.revokeObjectURL(a.href)}};
pv.addEventListener('input',function(){{render();renderJournal()}});filter.addEventListener('change',render);render();renderWatch();renderJournal();}})();</script></body></html>'''


def run() -> dict:
    sessions, end = completed_sessions()
    as_of = pd.Timestamp(sessions[-1]).normalize()
    history = sessions[-80:]
    start = str(pd.Timestamp(history[0]).date())
    client = Alpaca()
    universe, universe_audit = build_universe(client)
    assets = {str(a.get("symbol", "")): a for a in client.active_assets()}
    signals: list[dict] = []
    market: dict[str, list[list]] = {}
    fresh = 0
    for offset in range(0, len(universe), BATCH_SIZE):
        batch = universe[offset:offset + BATCH_SIZE]
        adjusted_payload = client.bars(batch, start, end.isoformat(), adjustment="all")
        raw_payload = client.bars(batch, start, end.isoformat(), adjustment="raw")
        for symbol in batch:
            adjusted = frame_from_bars(adjusted_payload.get(symbol, []))
            raw = frame_from_bars(raw_payload.get(symbol, []))
            if not raw.empty:
                market[symbol] = [
                    [str(day.date()), round(float(row.open), 4), round(float(row.high), 4),
                     round(float(row.low), 4), round(float(row.close), 4)]
                    for day, row in raw.loc[:as_of].tail(5).iterrows()
                ]
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
        "market": market,
    }
    Path(OUTPUT_JSON).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    Path(OUTPUT_HTML).write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} and {OUTPUT_JSON}")
    return payload


if __name__ == "__main__":
    run()
