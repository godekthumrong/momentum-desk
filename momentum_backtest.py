"""
Momentum Backtest — how the S&P 500 momentum strategy would have performed.

รันไฟล์นี้ไฟล์เดียวจบ แล้วจะได้ momentum-backtest.html เปิดดูในเบราว์เซอร์

ผลลัพธ์มี 2 ชุดให้เทียบกัน:
  1) ยาว 20 ปี แต่ใช้หุ้นที่อยู่ใน index "ปัจจุบัน" ทั้งช่วง -> ตัวเลขสวยเกินจริง
  2) สั้นกว่า แต่ย้อนสร้างสมาชิก index ตามจริงในแต่ละเดือน -> น่าเชื่อถือกว่า
ส่วนต่างระหว่างสองชุดคือขนาดของ survivorship bias

วิธีรัน: ดับเบิลคลิก RUN-Backtest.bat  หรือ  python momentum_backtest.py
"""

# ═══ CONFIG ══════════════════════════════════════════════════════════════════
YEARS          = 20        # ความยาวการทดสอบ (ปี)
TOP_N          = 20        # จำนวนหุ้นที่ถือแต่ละเดือน
LOOKBACK_DAYS  = 125       # ช่วงคำนวณ momentum
VOL_WINDOW     = 20        # ช่วงคำนวณ volatility
COST_BPS       = 20        # ค่าธรรมเนียม+slippage ต่อ notional ที่เทรด (basis points)

# ── กฎการซื้อขาย ─────────────────────────────────────────────────────────────
EXIT_RANK      = 100       # ขายเมื่ออันดับหลุดเกินนี้ (None = เปลี่ยนตัวใหม่ทุกเดือน)
REGIME_SMA     = 200       # ห้ามซื้อใหม่เมื่อดัชนีต่ำกว่า SMA นี้ (None = ปิดกฎนี้)
REGIME_TICKER  = "^GSPC"   # ดัชนีที่ใช้ตัดสิน regime (ราคาเปล่า ตรงกับที่คนอ้างถึง)
BENCHMARK      = "SPY"     # total return (รวมปันผล) ต่างจาก ^GSPC ที่เป็นราคาเปล่า
SHOW_BIASED    = False     # True = แสดงชุดที่ใช้สมาชิกปัจจุบัน (มี survivorship bias) เทียบด้วย

OUTPUT_HTML    = "momentum-backtest.html"
OPEN_BROWSER   = True
# ═════════════════════════════════════════════════════════════════════════════

# ── environment overrides (สำหรับรันบนคลาวด์/CI ที่ไม่มีจอ) ──────────────────
import os as _os
if _os.environ.get("MOMENTUM_HEADLESS"):
    OPEN_BROWSER = False
if _os.environ.get("MOMENTUM_YEARS"):
    YEARS = int(_os.environ["MOMENTUM_YEARS"])
if _os.environ.get("MOMENTUM_TOP_N"):
    TOP_N = int(_os.environ["MOMENTUM_TOP_N"])

import bisect
import io
import json
import os
import re
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


def _bail(title, causes, detail=""):
    print("\n  " + "-" * 58)
    print(f"  {title}")
    for c in causes:
        print(f"    - {c}")
    if detail:
        print("\n  ข้อความจากระบบ / what the system said:")
        for line in detail.strip().splitlines()[-6:]:
            print(f"    {line}")
    print("  " + "-" * 58)
    if sys.platform == "win32" and sys.stdin.isatty():
        input("\nกด Enter เพื่อปิด / press Enter to close...")
    sys.exit(1)


def _ensure(pkg, import_name=None, optional=False):
    """Import pkg, installing it if missing.

    optional=True: a package that only backs a fallback path. If it cannot be
    installed, carry on rather than killing a twenty-minute run over it.
    """
    name = import_name or pkg
    try:
        return __import__(name)
    except ImportError:
        pass
    print(f"  ติดตั้ง {pkg} ... (ครั้งแรกครั้งเดียว / first run only)")
    base = [sys.executable, "-m", "pip", "install", "-q"]
    detail = ""
    for extra in ([], ["--user"], ["--break-system-packages"]):
        try:
            r = subprocess.run(base + extra + [pkg], capture_output=True, text=True)
            if r.returncode == 0:
                break
            detail = (r.stderr or r.stdout or "").strip()
        except Exception as e:
            detail = str(e)
    else:
        if optional:
            print(f"      ({pkg} ติดตั้งไม่ได้ — ข้ามไป ใช้ทางสำรองแทน)")
            return None
        _bail(f"ติดตั้ง {pkg} ไม่สำเร็จ / could not install {pkg}",
              ["ไม่ได้ต่ออินเทอร์เน็ต / no internet connection",
               "ไฟร์วอลล์หรือ proxy บล็อก pip / a firewall or proxy blocks pip"], detail)
    import importlib, site
    try:
        us = site.getusersitepackages()
        if us and us not in sys.path:
            sys.path.append(us)
    except Exception:
        pass
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    except ImportError:
        if optional:
            return None
        raise


print("=" * 68)
print("  MOMENTUM BACKTEST")
print("=" * 68)
print("\n[1/5] ตรวจสอบ dependencies")
for pkg, mod in [("pandas", "pandas"), ("numpy", "numpy"),
                 ("requests", "requests"), ("lxml", "lxml"), ("yfinance", "yfinance")]:
    _ensure(pkg, mod)
# only back a fallback parser — never worth aborting the run for
for pkg, mod in [("beautifulsoup4", "bs4"), ("html5lib", "html5lib")]:
    _ensure(pkg, mod, optional=True)
print("      พร้อม")

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ═══ backtest engine ═══

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── signal ────────────────────────────────────────────────────────────────
def quant_scores(prices, asof, lookback):
    """Annualised log-slope x R^2 for every column with enough history.

    prices: DataFrame indexed by date. Only rows <= asof are used.
    Returns a Series indexed by ticker (NaN where history is insufficient).
    """
    window = prices.loc[:asof].tail(lookback)
    if len(window) < lookback:
        return pd.Series(dtype=float)

    out = {}
    x = np.arange(lookback)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_ss = (x_dev ** 2).sum()

    for ticker in window.columns:
        col = window[ticker]
        if col.isna().any() or (col <= 0).any():
            continue
        y = np.log(col.values)
        slope = (x_dev * (y - y.mean())).sum() / x_ss
        resid = y - (y.mean() + slope * x_dev)
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
        out[ticker] = (np.exp(slope) ** TRADING_DAYS - 1) * r2 * 100

    return pd.Series(out, dtype=float)


def inverse_vol_weights(returns, asof, tickers, vol_window):
    """Weights inversely proportional to trailing volatility, summing to 1."""
    window = returns.loc[:asof].tail(vol_window)
    vols = window[[t for t in tickers if t in window.columns]].std()
    vols = vols[vols > 0].dropna()
    if vols.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / vols
    return inv / inv.sum()


# ── portfolio construction ────────────────────────────────────────────────
def select(prices, returns, asof, universe, top_n, lookback, vol_window,
           prev_w=None, exit_rank=None, can_buy=True):
    """Choose holdings under the exit-buffer and regime rules, then size them.

    exit_rank : keep a holding while its rank stays within this many names.
                None reverts to plain top-N replacement every month.
    can_buy   : False blocks NEW positions (regime filter). Existing holdings
                are still sold when they breach exit_rank.

    Weights are scaled by filled slots / top_n, so slots that cannot be filled
    stay in cash instead of concentrating the portfolio into whatever is left.
    """
    available = [t for t in universe if t in prices.columns]
    if not available:
        return pd.Series(dtype=float)

    scores = quant_scores(prices[available], asof, lookback).dropna()
    if scores.empty:
        return pd.Series(dtype=float)

    ranked = scores.sort_values(ascending=False)
    order = list(ranked.index)
    rank_of = {t: i + 1 for i, t in enumerate(order)}

    if exit_rank is None:
        held = order[:top_n]
    else:
        # keep what still ranks inside the buffer; anything that fell outside
        # it, or lost its price data entirely, is dropped
        held = [t for t in (prev_w.index if prev_w is not None else [])
                if rank_of.get(t, 10 ** 9) <= exit_rank]
        if can_buy:
            for t in order:
                if len(held) >= top_n:
                    break
                if t not in held:
                    held.append(t)

    if not held:
        return pd.Series(dtype=float)

    w = inverse_vol_weights(returns, asof, held, vol_window)
    if w.empty:
        return w
    return w * (min(len(w), top_n) / float(top_n))


def traded(prev_w, new_w):
    """Total notional changing hands, as a fraction of the portfolio.

    Sum of |weight change| across every name. A full swap (sell all, buy all
    new) is 2.0 because both sides are traded and costs are paid on both.
    Halving this — the usual "one-way turnover" — would silently undercharge
    once the portfolio may sit partly in cash, because a move into cash has no
    offsetting purchase to pair with.
    """
    keys = set(prev_w.index) | set(new_w.index)
    total = 0.0
    for k in keys:
        total += abs(float(new_w.get(k, 0.0)) - float(prev_w.get(k, 0.0)))
    return total


def run_backtest(prices, rebalance_dates, membership, top_n, lookback,
                 vol_window, cost_bps, exit_rank=None, regime=None):
    """Walk the rebalance schedule and return a daily equity curve.

    membership: callable(date) -> iterable of tickers eligible on that date.
    cost_bps:   cost in basis points charged per unit of notional traded.
    exit_rank:  hold a name until its rank falls outside this many; None means
                straight top-N replacement each month.
    regime:     optional boolean Series; where False, no new positions open.
    Returns dict with net/gross equity, traded fractions, exposure and holdings.
    """
    returns = prices.pct_change(fill_method=None)
    idx = prices.index

    net = pd.Series(1.0, index=idx, dtype=float)
    gross = pd.Series(1.0, index=idx, dtype=float)
    exposure = pd.Series(0.0, index=idx, dtype=float)
    weights = pd.Series(dtype=float)
    history, turns, blocked = [], [], 0

    equity_n = equity_g = 1.0
    rebal = {pd.Timestamp(d) for d in rebalance_dates}

    for i in range(1, len(idx)):
        today, prev = idx[i], idx[i - 1]

        # Weights set on `prev` earn today's return — signals never see today.
        if len(weights):
            day = returns.loc[today, [t for t in weights.index if t in returns.columns]]
            r = float((day.fillna(0.0) * weights.reindex(day.index).fillna(0.0)).sum())
        else:
            r = 0.0
        equity_n *= (1 + r)
        equity_g *= (1 + r)

        if prev in rebal:
            can_buy = True
            if regime is not None:
                v = regime.get(prev)
                can_buy = bool(v) if v is not None and not pd.isna(v) else True
                if not can_buy:
                    blocked += 1
            new_w = select(prices, returns, prev, membership(prev),
                           top_n, lookback, vol_window,
                           prev_w=weights, exit_rank=exit_rank, can_buy=can_buy)
            # under the regime filter an empty result means "hold nothing",
            # which is a legitimate state rather than a failed calculation
            if len(new_w) or (exit_rank is not None and not can_buy):
                t = traded(weights, new_w)
                equity_n *= (1 - t * cost_bps / 10_000.0)
                turns.append((prev, t))
                history.append((prev, new_w))
                weights = new_w

        net.iloc[i] = equity_n
        gross.iloc[i] = equity_g
        exposure.iloc[i] = float(weights.sum()) if len(weights) else 0.0

    return {"net": net, "gross": gross, "exposure": exposure,
            "traded": pd.Series(dict(turns), dtype=float),
            "blocked_rebalances": blocked,
            "holdings": history}


# ── statistics ────────────────────────────────────────────────────────────
def drawdown(equity):
    return equity / equity.cummax() - 1.0


def metrics(equity, periods_per_year=TRADING_DAYS):
    """CAGR, volatility, Sharpe, max drawdown and friends from an equity curve."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(periods_per_year)
    dd = drawdown(equity)
    downside = rets[rets < 0].std() * np.sqrt(periods_per_year)

    return {
        "years": years,
        "total_return": float(total),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(cagr / vol) if vol > 0 else np.nan,
        "sortino": float(cagr / downside) if downside > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
        "best_day": float(rets.max()),
        "worst_day": float(rets.min()),
        "positive_days": float((rets > 0).mean()),
    }


def annual_returns(equity):
    """Calendar-year returns from a daily equity curve."""
    yearly = equity.resample("YE").last()
    first = equity.iloc[0]
    out = {}
    prev = first
    for ts, val in yearly.items():
        out[ts.year] = float(val / prev - 1)
        prev = val
    return out


def month_end_dates(index, start=None):
    """Last trading day of each month present in `index`."""
    s = pd.Series(index, index=index)
    ends = s.resample("ME").last().dropna()
    ends = [pd.Timestamp(d) for d in ends.values]
    if start is not None:
        start = pd.Timestamp(start)
        ends = [d for d in ends if d >= start]
    return ends

# ═══ report template ═══

import json

CSS = """
:root {
  color-scheme: light;
  --page:#F0EEE6; --surface:#FAF9F5; --surface-2:#EDEAE0; --surface-3:#E3DFD2;
  --ink:#141413; --ink-2:#4A4844; --muted:#6F6D66;
  --grid:#E4E0D4; --border:rgba(20,20,19,0.11);
  --accent:#C4603C; --link:#A34527;
  --strategy:#CC785C;   /* net equity curve */
  --bench:#7C46A8;      /* benchmark — CVD dE 16.5 from the strategy hue */
  --gross:#C9B79A;      /* gross, deliberately recessive */
  --ok:#3F6B45; --ok-bg:#E6EEE4;
  --warn:#8F3016; --warn-bg:#F9E5DC;
  --tag-bg:#EBDBBC; --tag-ink:#6B4F2A;
}
* { box-sizing:border-box; }
html, body { margin:0; padding:0; }
body {
  background:var(--page); color:var(--ink); line-height:1.45;
  font-family:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1180px; margin:0 auto; padding:40px 24px 80px; }

.eyebrow {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:600; font-size:12px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted);
}
header.masthead {
  padding-bottom:24px; margin-bottom:26px; border-bottom:1px solid var(--border);
}
header.masthead h1 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:650; font-size:clamp(28px,4vw,40px); letter-spacing:-.01em;
  margin:6px 0 10px; text-wrap:balance;
}
header.masthead p { color:var(--ink-2); font-size:15px; max-width:66ch; margin:0; }

/* the caveat is the most important thing on the page — it leads */
.caveat {
  border:1px solid rgba(143,48,22,0.28); background:var(--warn-bg);
  border-radius:10px; padding:16px 20px; margin-bottom:30px;
}
.caveat h2 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:14px; font-weight:650; margin:0 0 7px; color:var(--warn);
  letter-spacing:.02em;
}
.caveat p { margin:0 0 8px; font-size:13px; color:var(--ink-2); max-width:80ch; }
.caveat p:last-child { margin-bottom:0; }
.caveat b { color:var(--warn); }

.tabs { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
.tabs button {
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:13px;
  padding:9px 16px; border-radius:7px; cursor:pointer;
  background:var(--surface); border:1px solid var(--border); color:var(--ink-2);
}
.tabs button:hover { background:var(--surface-3); }
.tabs button[aria-selected="true"] {
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;
}
.tabs button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

.run-note {
  font-size:13px; color:var(--ink-2); background:var(--surface);
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:8px; padding:11px 15px; margin-bottom:22px; max-width:86ch;
}

.kpi-row {
  display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border);
  border-radius:10px; overflow:hidden; margin-bottom:30px;
}
.kpi { background:var(--surface); padding:17px 19px; display:flex; flex-direction:column; gap:5px; }
.kpi .label { font-size:11.5px; color:var(--muted); }
.kpi .value { font-weight:650; font-size:25px; letter-spacing:-.01em; }
.kpi .detail {
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:12px; color:var(--ink-2);
}
.kpi .detail.good { color:var(--ok); }
.kpi .detail.bad { color:var(--warn); }
@media (max-width:900px) { .kpi-row { grid-template-columns:repeat(2,1fr); } }

section { margin-bottom:34px; }
.section-head {
  display:flex; justify-content:space-between; align-items:baseline;
  gap:16px; flex-wrap:wrap; margin-bottom:12px;
}
.section-head h2 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:650; font-size:18px; margin:0;
}
.section-head .note { font-size:12.5px; color:var(--muted); }

.card {
  border:1px solid var(--border); border-radius:10px;
  background:var(--surface); padding:18px 20px 12px;
}
.chart-wrap { position:relative; overflow:hidden; }
.chart-wrap svg { display:block; }
.legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--ink-2); margin-bottom:10px; }
.legend i { display:inline-block; width:16px; height:0; border-top:2.5px solid; margin-right:7px; vertical-align:4px; }
.legend i.s { border-color:var(--strategy); }
.legend i.b { border-color:var(--bench); }
.legend i.g { border-color:var(--gross); border-top-style:dashed; }
.scale-toggle { margin-left:auto; display:inline-flex; }
.scale-toggle button {
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:11.5px;
  padding:4px 11px; cursor:pointer; color:var(--ink-2);
  background:var(--surface); border:1px solid var(--border);
}
.scale-toggle button:first-child { border-radius:6px 0 0 6px; }
.scale-toggle button:last-child { border-radius:0 6px 6px 0; border-left:none; }
.scale-toggle button[aria-pressed="true"] {
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;
}
.scale-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

.tip {
  position:absolute; pointer-events:none; opacity:0; transition:opacity .12s;
  background:var(--surface); border:1px solid var(--border); border-radius:7px;
  padding:8px 11px; font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:11.5px; line-height:1.6; box-shadow:0 3px 14px rgba(20,20,19,.13);
  white-space:nowrap; z-index:5; font-variant-numeric:tabular-nums;
}
.tip.on { opacity:1; }
.tip b { display:block; font-family:"IBM Plex Sans", system-ui, sans-serif;
         font-size:11.5px; font-weight:650; margin-bottom:3px; }
.tip u { text-decoration:none; color:var(--muted); display:inline-block; width:76px; }

.table-scroll { overflow-x:auto; border:1px solid var(--border); border-radius:10px; }
table { width:100%; border-collapse:collapse; background:var(--surface); font-size:13px; min-width:520px; }
thead th {
  text-align:right; padding:10px 14px; border-bottom:1px solid var(--grid);
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:600; font-size:11px; letter-spacing:.04em; text-transform:uppercase;
  color:var(--muted); white-space:nowrap;
}
thead th:first-child { text-align:left; }
tbody td {
  text-align:right; padding:9px 14px; border-bottom:1px solid var(--grid);
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap;
}
tbody td:first-child { text-align:left; font-family:"IBM Plex Sans", system-ui, sans-serif; }
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover td { background:var(--surface-2); }
td.pos { color:var(--ok); }
td.neg { color:var(--warn); }
tbody tr.better td:first-child::after {
  content:"beat benchmark"; font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:9.5px; letter-spacing:.04em; text-transform:uppercase;
  background:var(--tag-bg); color:var(--tag-ink); border-radius:4px;
  padding:2px 6px; margin-left:9px; vertical-align:middle;
}

footer {
  border-top:1px solid var(--border); padding-top:20px; margin-top:34px;
  font-size:12.5px; color:var(--muted); max-width:82ch;
}
footer h3 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:13px; color:var(--ink-2); margin:16px 0 6px;
}
footer p, footer li { margin:0 0 7px; }
footer ul { padding-left:18px; margin:0 0 10px; }
"""

BODY = """
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">S&amp;P 500 momentum &middot; inverse-volatility sizing &middot; monthly rebalance</div>
    <h1>Momentum Backtest</h1>
    <p id="intro"></p>
  </header>

  <div class="caveat">
    <h2>Read this before the numbers</h2>
    <p><b>Every month here ranks the companies that were actually in the index at the
       time.</b> That sounds obvious and is easy to get wrong: rank today's members across the
       whole history instead, and the strategy gets to pick from a pool of known survivors &mdash;
       every firm that went bankrupt, was acquired at a discount or simply dropped out is
       silently missing. This page deliberately does not show that version.</p>
    <p id="bias-note"></p>
    <p>These figures are still optimistic. Companies that were delisted cannot be priced at all
       by the free data source, so they fall out of the universe here too &mdash; the residual
       bias is smaller, not zero. Membership comes from a community-maintained dataset rather
       than an official S&amp;P feed. And a backtest assumes every fill happens at the closing
       price with no market impact, no failed orders and no taxes; monthly rebalancing in a
       taxable account would realise short-term gains all the way through. Past results, even
       carefully built ones, do not predict future returns.</p>
  </div>

  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="run-note" id="run-note"></div>

  <div class="kpi-row" id="kpi-row"></div>

  <section>
    <div class="section-head">
      <h2>Growth of $1</h2>
      <span class="note">Logarithmic scale &mdash; equal vertical distance means equal percentage change</span>
    </div>
    <div class="card">
      <div class="legend">
        <span><i class="s"></i>Strategy, after costs</span>
        <span><i class="g"></i>Strategy, before costs</span>
        <span><i class="b"></i>S&amp;P 500 total return</span>
        <span class="scale-toggle">
          <button type="button" id="scale-log" aria-pressed="true">Log</button><button
                  type="button" id="scale-lin" aria-pressed="false">Linear</button>
        </span>
      </div>
      <div class="chart-wrap"><div id="equity"></div><div class="tip" id="equity-tip"></div></div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Drawdown</h2>
      <span class="note">Decline from the highest point reached so far</span>
    </div>
    <div class="card">
      <div class="legend">
        <span><i class="s"></i>Strategy</span>
        <span><i class="b"></i>S&amp;P 500</span>
      </div>
      <div class="chart-wrap"><div id="dd"></div><div class="tip" id="dd-tip"></div></div>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Statistics</h2></div>
    <div class="table-scroll">
      <table id="stats"><thead><tr>
        <th>Measure</th><th>Strategy (net)</th><th>Strategy (gross)</th><th>S&amp;P 500</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Calendar year returns</h2>
      <span class="note">Strategy after costs, against the index</span>
    </div>
    <div class="table-scroll">
      <table id="annual"><thead><tr>
        <th>Year</th><th>Strategy</th><th>S&amp;P 500</th><th>Difference</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </section>

  <footer id="method"></footer>
</div>
"""

SCRIPT = r"""
(function () {
  var D = window.__BACKTEST__;
  var M = D.meta, RUNS = D.runs;
  var cur = 0;
  var logScale = true;

  function pct(v, dp) { return (v * 100).toFixed(dp == null ? 1 : dp) + "%"; }
  function signed(v, dp) { return (v >= 0 ? "+" : "") + pct(v, dp); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
    });
  }
  function money(v) { return "$" + v.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}); }

  var rules = "Each month it holds up to " + M.top_n + " S&P 500 names ranked by " +
    M.lookback + "-day quant momentum, weighted inversely to " + M.vol_window +
    "-day volatility.";
  if (M.exit_rank) rules += " A position is kept until its rank falls outside the top " +
    M.exit_rank + ", rather than being replaced every month.";
  if (M.regime_sma) rules += " While the S&P 500 index sits below its " + M.regime_sma +
    "-day average, no new position is opened — holdings still leave on the rank rule, " +
    "and the freed slots stay in cash.";
  rules += " Costs of " + M.cost_bps + " bps are charged on every unit of notional traded.";
  document.getElementById("intro").textContent = rules;

  var bn = document.getElementById("bias-note");
  if (M.bias_measured && M.bias_measured.cagr_gap != null) {
    var gap = M.bias_measured.cagr_gap * 100;
    var ratio = M.bias_measured.wealth_ratio;
    bn.innerHTML = "Running these exact rules on today's members instead, over the same " +
      "period, raised the annual return by <b>" + gap.toFixed(1) + " percentage points</b>" +
      (ratio ? " and multiplied the ending balance by <b>" + ratio.toFixed(1) + "x</b>" : "") +
      ". Published estimates put survivorship bias near 1-3% a year for broad equity samples; " +
      "a concentrated book of twenty names is far more exposed, because it picks from the " +
      "extremes of the pool and the extremes are exactly what hindsight distorts.";
  } else {
    bn.textContent = "Published estimates put survivorship bias near 1-3% a year for broad " +
      "equity samples. A concentrated book of twenty names is far more exposed, because it " +
      "picks from the extremes of the pool and the extremes are exactly what hindsight distorts.";
  }

  // ── tabs ──────────────────────────────────────────────────────
  var tabsEl = document.getElementById("tabs");
  if (RUNS.length < 2) {
    tabsEl.hidden = true;
  } else {
    tabsEl.innerHTML = RUNS.map(function (r, i) {
      return '<button role="tab" data-i="' + i + '" aria-selected="' + (i === 0) + '">' +
             esc(r.label) + '</button>';
    }).join("");
  }
  document.getElementById("tabs").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-i]");
    if (!b) return;
    cur = +b.dataset.i;
    [].forEach.call(this.querySelectorAll("button"), function (x) {
      x.setAttribute("aria-selected", x === b);
    });
    render();
  });

  // ── generic line chart ────────────────────────────────────────
  function niceTicks(lo, hi, want) {
    var raw = (hi - lo) / want;
    if (!(raw > 0)) return [lo];
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
    return out;
  }

  function lineChart(hostId, tipId, dates, series, opts) {
    var host = document.getElementById(hostId);
    var tip = document.getElementById(tipId);
    var W = Math.max(host.clientWidth || 900, 300);
    var H = W < 560 ? 240 : (opts.height || 320);
    var Mg = {t:12, r:56, b:26, l:8};
    var pw = W - Mg.l - Mg.r, ph = H - Mg.t - Mg.b;
    var live = series.filter(function (s) { return s.values && s.values.length; });
    if (!live.length || !dates.length) { host.innerHTML = ""; return null; }

    var lo = Infinity, hi = -Infinity;
    live.forEach(function (s) {
      s.values.forEach(function (v) {
        if (v == null || !isFinite(v)) return;
        if (v < lo) lo = v; if (v > hi) hi = v;
      });
    });

    var tf = opts.log ? function (v) { return Math.log(Math.max(v, 1e-9)); }
                      : function (v) { return v; };
    var tlo, thi;
    if (opts.log) {
      // Padding must be applied in log space. Doing it linearly pushed the
      // lower bound below zero, which then clamped to 1e-6 and squashed the
      // whole curve into the top quarter of the plot.
      lo = Math.max(lo, 1e-9);
      var a = Math.log(lo), b2 = Math.log(hi);
      var padL = (b2 - a) * 0.06 || 0.15;
      tlo = a - padL; thi = b2 + padL;
      lo = Math.exp(tlo); hi = Math.exp(thi);
    } else {
      if (opts.padZero) hi = Math.max(hi, 0);
      var span = hi - lo || Math.abs(hi) || 1;
      var loWas = lo;
      lo -= span * 0.05; hi += span * 0.05;
      if (opts.padZero) hi = Math.min(hi, 0);
      // a value that cannot go negative should not get a negative axis
      if (opts.floorZero && loWas >= 0 && lo < 0) lo = 0;
      tlo = lo; thi = hi;
    }
    var y = function (v) { return Mg.t + (thi - tf(v)) / (thi - tlo) * ph; };
    var x = function (i) { return Mg.l + (dates.length < 2 ? pw / 2 : i / (dates.length - 1) * pw); };

    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H +
             '" role="img" aria-label="' + esc(opts.aria || "chart") + '">'];

    var ticks = opts.log ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    ticks.forEach(function (v) {
      var yy = y(v); if (yy < Mg.t - 1 || yy > Mg.t + ph + 1) return;
      s.push('<line x1="' + Mg.l + '" x2="' + (Mg.l + pw) + '" y1="' + yy.toFixed(1) +
             '" y2="' + yy.toFixed(1) + '" stroke="var(--grid)" stroke-width="1"/>');
      s.push('<text x="' + (Mg.l + pw + 7) + '" y="' + yy.toFixed(1) + '" dy="3.5" ' +
             'fill="var(--muted)" font-size="10.5" font-family="IBM Plex Mono, monospace">' +
             opts.fmtAxis(v) + '</text>');
    });

    var seenYear = "";
    dates.forEach(function (d, i) {
      var yr = d.slice(0, 4);
      if (yr === seenYear) return;
      seenYear = yr;
      if (i === 0) return;
      if (dates.length > 2600 && (+yr) % 2) return;
      s.push('<text x="' + x(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" ' +
             'fill="var(--muted)" font-size="10.5" font-family="IBM Plex Mono, monospace">' +
             yr + '</text>');
    });

    if (opts.zeroLine) {
      s.push('<line x1="' + Mg.l + '" x2="' + (Mg.l + pw) + '" y1="' + y(0).toFixed(1) +
             '" y2="' + y(0).toFixed(1) + '" stroke="var(--ink-2)" stroke-width="1" opacity="0.4"/>');
    }

    live.forEach(function (ser) {
      var pts = [];
      ser.values.forEach(function (v, i) {
        if (v == null || !isFinite(v)) return;
        pts.push(x(i).toFixed(1) + "," + y(v).toFixed(1));
      });
      if (pts.length < 2) return;
      s.push('<polyline points="' + pts.join(" ") + '" fill="none" stroke="' + ser.color +
             '" stroke-width="' + (ser.width || 2) + '" stroke-linejoin="round" ' +
             'stroke-linecap="round"' + (ser.dash ? ' stroke-dasharray="5 4"' : "") + '/>');
    });

    s.push('<line id="' + hostId + '-guide" y1="' + Mg.t + '" y2="' + (Mg.t + ph) +
           '" stroke="var(--ink-2)" stroke-width="1" opacity="0"/>');
    s.push('</svg>');
    host.innerHTML = s.join("");

    function move(clientX) {
      var rect = host.getBoundingClientRect();
      var frac = (clientX - rect.left - Mg.l) / pw;
      var i = Math.round(frac * (dates.length - 1));
      i = Math.max(0, Math.min(dates.length - 1, i));
      tip.innerHTML = "<b>" + dates[i] + "</b>" + live.map(function (ser) {
        var v = ser.values[i];
        return "<u>" + esc(ser.name) + "</u>" + (v == null || !isFinite(v) ? "&mdash;" : opts.fmtTip(v));
      }).join("<br>");
      tip.classList.add("on");
      var g = document.getElementById(hostId + "-guide");
      if (g) { g.setAttribute("x1", x(i)); g.setAttribute("x2", x(i)); g.setAttribute("opacity", "0.3"); }
      var tw = tip.offsetWidth, th = tip.offsetHeight;
      var left = x(i) + 14;
      if (left + tw > W) left = x(i) - tw - 14;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, Math.min(H - th, Mg.t + 6)) + "px";
    }
    function hide() {
      tip.classList.remove("on");
      tip.style.left = "0px"; tip.style.top = "0px";
      var g = document.getElementById(hostId + "-guide");
      if (g) g.setAttribute("opacity", "0");
    }
    host.onmousemove = function (e) { move(e.clientX); };
    host.onmouseleave = hide;
    host.ontouchmove = function (e) { if (e.touches[0]) move(e.touches[0].clientX); };
    host.ontouchend = hide;
    return hide;
  }

  function logTicks(lo, hi) {
    var e0 = Math.floor(Math.log(Math.max(lo, 1e-9)) / Math.LN10);
    var e1 = Math.ceil(Math.log(hi) / Math.LN10);
    // try progressively coarser mantissa sets until the count is readable
    var sets = [[1, 2, 5], [1, 3], [1]];
    for (var si = 0; si < sets.length; si++) {
      var out = [];
      for (var e = e0; e <= e1; e++) {
        for (var mi = 0; mi < sets[si].length; mi++) {
          var v = sets[si][mi] * Math.pow(10, e);
          if (v >= lo && v <= hi) out.push(v);
        }
      }
      if (out.length <= 8) return out.length ? out : [lo, hi];
    }
    return [lo, hi];
  }

  // ── render one run ────────────────────────────────────────────
  var hides = [];
  function render() {
    var r = RUNS[cur];
    hides.forEach(function (h) { if (h) h(); });

    document.getElementById("run-note").innerHTML = esc(r.caveat);

    var m = r.metrics, bm = r.bench_metrics;
    var edge = m.cagr - bm.cagr;
    document.getElementById("kpi-row").innerHTML = [
      {l:"Strategy CAGR (after costs)", v:pct(m.cagr, 1),
       d:"turned $1 into " + money(r.net[r.net.length - 1])},
      {l:"S&P 500 CAGR", v:pct(bm.cagr, 1),
       d:"same period, total return"},
      {l:"Difference per year", v:signed(edge, 1),
       d:(edge >= 0 ? "ahead of" : "behind") + " the index", cls:edge >= 0 ? "good" : "bad"},
      {l:"Worst drawdown", v:pct(m.max_drawdown, 0),
       d:"index fell " + pct(bm.max_drawdown, 0), cls:"bad"}
    ].map(function (k) {
      return '<div class="kpi"><div class="label">' + esc(k.l) + '</div>' +
             '<div class="value">' + k.v + '</div>' +
             '<div class="detail ' + (k.cls || "") + '">' + esc(k.d) + '</div></div>';
    }).join("");

    hides = [];
    hides.push(lineChart("equity", "equity-tip", r.dates, [
      {name:"Net", values:r.net, color:"var(--strategy)", width:2},
      {name:"Gross", values:r.gross, color:"var(--gross)", width:1.5, dash:true},
      {name:"S&P 500", values:r.bench, color:"var(--bench)", width:2}
    ], {log:logScale, height:330,
        aria:"Growth of one dollar, " + (logScale ? "logarithmic" : "linear") + " scale",
        floorZero:true,
        fmtAxis:function (v) {
          if (Math.abs(v) < 1e-9) return "$0";
          if (v >= 1) return "$" + v.toLocaleString("en-US", {maximumFractionDigits:0});
          return "$" + v.toFixed(v >= 0.1 ? 2 : 3);
        },
        fmtTip:money}));

    hides.push(lineChart("dd", "dd-tip", r.dates, [
      {name:"Strategy", values:r.dd, color:"var(--strategy)", width:2},
      {name:"S&P 500", values:r.bench_dd, color:"var(--bench)", width:2}
    ], {height:210, padZero:true, zeroLine:true, aria:"Drawdown from peak",
        fmtAxis:function (v) { return pct(v, 0); },
        fmtTip:function (v) { return pct(v, 1); }}));

    var rows = [
      ["Total return", "total_return", "pct0"],
      ["Annualised return (CAGR)", "cagr", "pct"],
      ["Annualised volatility", "vol", "pct"],
      ["Return / volatility", "sharpe", "num"],
      ["Sortino", "sortino", "num"],
      ["Worst drawdown", "max_drawdown", "pct"],
      ["Return / worst drawdown", "calmar", "num"],
      ["Positive days", "positive_days", "pct"]
    ];
    document.querySelector("#stats tbody").innerHTML = rows.map(function (row) {
      function f(o) {
        var v = o[row[1]];
        if (v == null || !isFinite(v)) return "&mdash;";
        return row[2] === "num" ? v.toFixed(2) : row[2] === "pct0" ? pct(v, 0) : pct(v, 1);
      }
      return "<tr><td>" + row[0] + "</td><td>" + f(m) + "</td><td>" +
             f(r.gross_metrics) + "</td><td>" + f(bm) + "</td></tr>";
    }).join("") +
      (r.avg_exposure != null
        ? '<tr><td>Average invested (rest in cash)</td><td>' + pct(r.avg_exposure, 0) +
          '</td><td>&mdash;</td><td>100%</td></tr>'
        : '') +
      (r.blocked != null && r.n_rebalances
        ? '<tr><td>Months with buying blocked</td><td>' + r.blocked + ' of ' +
          r.n_rebalances + '</td><td>&mdash;</td><td>&mdash;</td></tr>'
        : '') +
      '<tr><td>Portfolio traded per rebalance</td><td>' + pct(r.turnover, 0) +
      '</td><td>&mdash;</td><td>&mdash;</td></tr>' +
      '<tr><td>Names ranked each month</td><td>' + r.universe_size +
      '</td><td>&mdash;</td><td>&mdash;</td></tr>';

    var years = Object.keys(r.annual).sort();
    document.querySelector("#annual tbody").innerHTML = years.map(function (yr) {
      var a = r.annual[yr][0], b = r.annual[yr][1];
      var d = a - b;
      return '<tr' + (d > 0 ? ' class="better"' : "") + '><td>' + yr + "</td>" +
             '<td class="' + (a >= 0 ? "pos" : "neg") + '">' + signed(a) + "</td>" +
             '<td class="' + (b >= 0 ? "pos" : "neg") + '">' + signed(b) + "</td>" +
             '<td class="' + (d >= 0 ? "pos" : "neg") + '">' + signed(d) + "</td></tr>";
    }).join("");

    var beat = years.filter(function (y) { return r.annual[y][0] > r.annual[y][1]; }).length;
    document.getElementById("method").innerHTML =
      "<h3>How this was run</h3><ul>" +
      "<li>Signal: annualised log-slope of the last " + M.lookback +
        " trading days multiplied by its R&sup2;, computed from closes up to and including the rebalance date.</li>" +
      "<li>Positions take effect the following trading day, so no signal uses a price it could not have seen.</li>" +
      "<li>Sizing: inverse " + M.vol_window + "-day volatility, normalised to 100%.</li>" +
      "<li>Rebalanced on the last trading day of each month. On average " +
        pct(r.turnover, 0) + " of the portfolio changed hands each time.</li>" +
      (M.exit_rank ? "<li>Exit rule: a holding is sold only once its rank falls outside " +
        "the top " + M.exit_rank + ". This buffer is what keeps trading down — replacing " +
        "the book every month costs far more in fees.</li>" : "") +
      (M.regime_sma ? "<li>Regime rule: no new position opens while the index is below its " +
        M.regime_sma + "-day average, measured on the rebalance date itself. Slots that " +
        "cannot be filled hold cash at 0%, so the portfolio de-risks as holdings age out. " +
        "It was invested " + pct(r.avg_exposure, 0) + " of the time on average, and buying " +
        "was blocked in " + r.blocked + " of " + r.n_rebalances + " months.</li>" : "") +
      "<li>Costs: " + M.cost_bps + " bps charged on turnover, covering commission and slippage together. " +
        "The dashed line shows what the same trades would have returned with no costs at all. " +
        "A full swap of the book counts as 200%, since both the sale and the purchase pay.</li>" +
      "<li>Prices are dividend and split adjusted, so both the strategy and the benchmark are total returns.</li>" +
      "<li>The strategy beat the index in " + beat + " of " + years.length + " calendar years shown.</li>" +
      "</ul>" +
      "<h3>What this cannot tell you</h3><ul>" +
      "<li>" + esc(r.missing_note) + "</li>" +
      "<li>Fills are assumed at the closing price. A 20-name portfolio rebalancing monthly would " +
        "move real money in real books; slippage on less liquid names can exceed the assumption above.</li>" +
      "<li>No taxes. In a taxable account, monthly rebalancing realises short-term gains, which are " +
        "taxed at a higher rate in most jurisdictions and would materially reduce what you keep.</li>" +
      "<li>One historical path is a single sample. It cannot tell you the range of outcomes the same " +
        "rules would produce in the future.</li>" +
      "</ul>";
  }

  var logBtn = document.getElementById("scale-log");
  var linBtn = document.getElementById("scale-lin");
  function setScale(useLog) {
    logScale = useLog;
    logBtn.setAttribute("aria-pressed", String(useLog));
    linBtn.setAttribute("aria-pressed", String(!useLog));
    document.querySelector("#equity").closest("section")
      .querySelector(".note").textContent = useLog
        ? "Logarithmic scale \u2014 equal vertical distance means equal percentage change"
        : "Linear scale \u2014 the early years are compressed against the axis";
    render();
  }
  logBtn.addEventListener("click", function () { setScale(true); });
  linBtn.addEventListener("click", function () { setScale(false); });

  render();
  // The observer watches the element whose innerHTML render() replaces, so
  // without a width guard every render retriggers it — an endless loop that
  // also wipes any open tooltip.
  var rt = null, lastW = document.getElementById("equity").clientWidth;
  if (window.ResizeObserver) {
    new ResizeObserver(function () {
      var w = document.getElementById("equity").clientWidth;
      if (w === lastW) return;
      lastW = w;
      clearTimeout(rt);
      rt = setTimeout(function () { lastW = document.getElementById("equity").clientWidth; render(); }, 140);
    }).observe(document.getElementById("equity"));
  }
})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Sans:wght@400;500;650&'
         'family=IBM+Plex+Sans+Condensed:wght@600;650&'
         'family=IBM+Plex+Mono:wght@400;500;600&display=swap">')


def build_report(payload, standalone=True):
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    head = f"<title>Momentum Backtest</title>\n{FONTS}\n<style>{CSS}</style>\n"
    body = (f"{BODY}\n<script>window.__BACKTEST__ = {data};</script>\n"
            f"<script>{SCRIPT}</script>\n")
    if not standalone:
        return head + body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{head}</head>\n<body>\n{body}</body>\n</html>\n")

# ═══ runner ═══

import bisect
import io
import json
import re
import sys

import numpy as np
import pandas as pd


WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def norm(t):
    """Yahoo ticker convention."""
    return str(t).strip().upper().replace(".", "-")


def extract_tables(html):
    """Return every table on the page as a DataFrame.

    Whole-document read_html is the fast path, but it is at the mercy of the
    parser: on pandas 3.0 it returned only two of this page's tables and
    silently dropped the index-change log. So each <table> block is also
    pulled out by hand and parsed on its own, which does not depend on the
    parser walking the whole document successfully. Results are merged and
    de-duplicated.
    """
    found = []

    def add(df):
        if df is None or df.empty:
            return
        key = (df.shape, tuple(str(c)[:40] for c in list(df.columns)[:8]))
        if key not in {k for k, _ in found}:
            found.append((key, df))

    for flavor in (None, "bs4"):
        try:
            kw = {} if flavor is None else {"flavor": flavor}
            for df in pd.read_html(io.StringIO(html), **kw):
                add(df)
        except Exception:
            continue

    # per-table parse: immune to whatever makes a full-document walk skip one
    for m in re.finditer(r"<table\b.*?</table>", html, re.S | re.I):
        try:
            for df in pd.read_html(io.StringIO(m.group(0))):
                add(df)
        except Exception:
            continue

    return [df for _, df in found]


def fetch_tables(requests):
    r = requests.get(WIKI, headers=HDRS, timeout=30)
    r.raise_for_status()
    return extract_tables(r.text)


def parse_changes(tables):
    """Pull (date, added, removed) from the index-changes table.

    The table's shape has moved around over the years, so match columns by
    name rather than position and skip anything that will not parse.
    """
    for tbl in tables:
        cols = [" ".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
                for col in tbl.columns]
        low = [c.lower() for c in cols]
        date_i = next((i for i, c in enumerate(low) if "date" in c), None)
        add_i = next((i for i, c in enumerate(low) if "added" in c and "ticker" in c), None)
        rem_i = next((i for i, c in enumerate(low) if "removed" in c and "ticker" in c), None)
        if date_i is None or add_i is None or rem_i is None:
            continue
        if "symbol" in low[0] or "security" in low[0]:
            continue        # that is the constituents table, not the change log

        out = []
        for _, row in tbl.iterrows():
            try:
                d = pd.to_datetime(row.iloc[date_i], errors="coerce")
            except Exception:
                continue
            if pd.isna(d):
                continue
            a = row.iloc[add_i]
            r = row.iloc[rem_i]
            a = norm(a) if pd.notna(a) and str(a).strip() not in ("", "nan") else None
            r = norm(r) if pd.notna(r) and str(r).strip() not in ("", "nan") else None
            if a or r:
                out.append((pd.Timestamp(d).normalize(), a, r))
        if out:
            return sorted(out, key=lambda x: x[0])
    return []


# ── historical index membership ───────────────────────────────────────────
# Wikipedia removed its "Selected changes" section, so the change log it used
# to carry is simply gone. This community dataset lists the actual constituent
# set on each date back to 1996, which is strictly better: no reconstruction,
# and it covers the whole backtest rather than the last few years.
# It is community-maintained, not an official S&P feed — worth stating plainly
# wherever the results are shown.
HISTORY_URLS = [
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(Updated).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/main/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
]


def discover_history_url(requests):
    """Ask the GitHub API for whatever the file is called today."""
    try:
        r = requests.get("https://api.github.com/repos/fja05680/sp500/contents/",
                         headers=HDRS, timeout=20)
        r.raise_for_status()
        for f in r.json():
            name = f.get("name", "")
            if "Historical Components" in name and name.lower().endswith(".csv"):
                if f.get("download_url"):
                    return f["download_url"]
    except Exception:
        pass
    return None


def fetch_membership_history(requests, log=print):
    """Download the point-in-time constituent file; returns its text or None."""
    urls = list(HISTORY_URLS)
    found = discover_history_url(requests)
    if found:
        urls.insert(0, found)
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=60)
            if r.status_code == 200 and len(r.text) > 5000:
                log(f"      membership file: {u.rsplit('/', 1)[-1][:60]} "
                    f"({len(r.text):,} bytes)")
                return r.text
        except Exception:
            continue
    return None


def parse_membership_csv(text):
    """Parse the constituent file into [(date, {tickers}), ...], ascending.

    Column names in this dataset have moved around, so the date column and the
    ticker-list column are identified by content rather than by header.
    """
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return []

    date_col = None
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9:
            date_col = c
            break
    if date_col is None:
        return []

    # the ticker list is the column whose cells hold many comma-separated words
    tick_col, best = None, 0
    for c in df.columns:
        if c == date_col:
            continue
        commas = df[c].astype(str).str.count(",").mean()
        if commas > best:
            tick_col, best = c, commas
    if tick_col is None or best < 5:
        return []

    out = []
    for _, row in df.iterrows():
        d = pd.to_datetime(row[date_col], errors="coerce", format="mixed")
        if pd.isna(d):
            continue
        parts = re.split(r"[,\s]+", str(row[tick_col]))
        ticks = {norm(t) for t in parts if t and t.lower() not in ("nan", "none", "")}
        if len(ticks) > 100:          # a real index snapshot, not a stray row
            out.append((pd.Timestamp(d).normalize(), ticks))
    out.sort(key=lambda x: x[0])
    return out


def membership_from_history(snapshots):
    """Callable(date) -> the constituent set in force on that date."""
    dates = [d for d, _ in snapshots]
    sets = [s for _, s in snapshots]

    def at(d):
        i = bisect.bisect_right(dates, pd.Timestamp(d)) - 1
        return sets[max(0, i)] if sets else set()
    return at


def build_timeline(current, changes):
    """Walk changes backwards to recover membership at each point in time.

    Returns (dates, sets) where sets[i] is the membership in force from
    dates[i] until dates[i+1].
    """
    cur = set(current)
    checkpoints = []                       # (valid_from, members)
    for d, added, removed in sorted(changes, key=lambda x: x[0], reverse=True):
        checkpoints.append((d, set(cur)))  # this set is in force from d onward
        if added:
            cur.discard(added)             # it was not a member before d
        if removed:
            cur.add(removed)               # it was a member before d
    checkpoints.append((pd.Timestamp("1900-01-01"), set(cur)))
    checkpoints.sort(key=lambda x: x[0])
    return [c[0] for c in checkpoints], [c[1] for c in checkpoints]


def make_membership(dates, sets):
    def at(d):
        i = bisect.bisect_right(dates, pd.Timestamp(d)) - 1
        return sets[max(0, i)]
    return at


def summarise(equity, bench, dates_index):
    return {
        "metrics": metrics(equity),
        "bench_metrics": metrics(bench),
    }


def to_series_list(s, dp=6):
    return [None if pd.isna(v) else round(float(v), dp) for v in s]


def run_all(prices, spy, current, changes, cfg, log=print, regime=None,
            snapshots=None):
    """Produce the payload for both runs."""
    runs = []
    all_dates = prices.index

    # ── run 1: today's constituents applied to the whole history ──────────
    biased_universe = [t for t in current if t in prices.columns]
    log(f"  [biased]  {len(biased_universe)} tickers with usable history")

    # ── run 2: point-in-time membership ───────────────────────────────────
    # Prefer the historical constituent file: it states membership directly on
    # each date, so nothing has to be reconstructed. Fall back to walking a
    # Wikipedia change log only if that file is unavailable.
    if snapshots:
        membership_at = membership_from_history(snapshots)
        pit_start = max(snapshots[0][0], all_dates[0])
        ever = set()
        for _, st in snapshots:
            ever |= st
        missing = sorted(t for t in ever if t not in prices.columns)
        pit_usable = True
        pit_source = (f"point-in-time constituent file, {len(snapshots)} dated "
                      f"snapshots from {snapshots[0][0].date()}")
        log(f"  [pit]     {len(snapshots)} snapshots "
            f"({snapshots[0][0].date()} to {snapshots[-1][0].date()}), "
            f"{len(missing)} of {len(ever)} names ever listed have no price data")
    else:
        tl_dates, tl_sets = build_timeline(current, changes)
        earliest_change = min((c[0] for c in changes), default=None)
        pit_start = earliest_change if earliest_change is not None else all_dates[0]
        membership_at = make_membership(tl_dates, tl_sets)
        ever = set()
        for st in tl_sets:
            ever |= st
        missing = sorted(t for t in ever if t not in prices.columns)
        pit_usable = bool(changes)
        pit_source = f"reconstructed from {len(changes)} published index changes"
        if not pit_usable:
            log("  [pit]     NO membership history available — the point-in-time run "
                "would be identical to the biased one, so it is omitted")
        else:
            log(f"  [pit]     reconstructed to {pit_start.date()} from "
                f"{len(changes)} changes, {len(missing)} names lack price data")

    specs = [
        dict(key="biased", label=f"{cfg['years']} years · today's members",
             universe=lambda d: biased_universe,
             start=all_dates[0],
             caveat=("Ranks only companies in the index today, across the whole period. "
                     "Firms that failed or were removed never appear, so this run flatters "
                     "the strategy. Read it as an upper bound, not an estimate."),
             usize=len(biased_universe),
             missing_note=("Companies delisted or removed from the index during this period "
                           "are entirely absent, because the ranking universe is today's "
                           "membership. This is the survivorship bias described above and it "
                           "inflates every figure on this tab.")),
        dict(key="pit", label="Point-in-time members",
             universe=membership_at,
             start=pit_start,
             caveat=("Uses the index membership actually in force on each rebalance date — "
                     + pit_source + "."),
             usize=None,
             missing_note=(f"{len(missing)} companies that were index members at some point "
                           "have no price history available from the free data source — mostly "
                           "firms acquired, taken private or wound up. They are skipped, so a "
                           "residual survivorship bias remains here too: much smaller than "
                           "ranking today's members would produce, but not zero. Membership "
                           "itself comes from a community-maintained dataset, not an official "
                           "S&P feed.")),
    ]

    # The survivor-pool run is off by default: its only purpose was to measure
    # the bias, and that measurement is now carried in the caveat. Set
    # cfg["show_biased"] to bring the comparison tab back.
    if not cfg.get("show_biased") and pit_usable:
        specs = [sp for sp in specs if sp["key"] != "biased"]

    for spec in specs:
        if spec["key"] == "pit" and not pit_usable:
            continue
        usize = spec["usize"] or len([t for t in spec["universe"](all_dates[-1])
                                      if t in prices.columns])
        rebals = month_end_dates(all_dates, start=spec["start"])
        # need a full lookback of history before the first trade
        first_ok = all_dates[min(cfg["lookback"] + 5, len(all_dates) - 1)]
        rebals = [d for d in rebals if d >= first_ok]
        if len(rebals) < 3:
            log(f"  [{spec['key']}] not enough history to rebalance — skipped")
            continue

        # An exit rank at or beyond the universe size can never trigger, which
        # silently turns the strategy into buy-and-hold-forever.
        if cfg.get("exit_rank") and usize and cfg["exit_rank"] >= usize:
            log(f"  [{spec['key']}] WARNING: exit_rank {cfg['exit_rank']} >= "
                f"{usize} ranked names — the exit rule can never fire, so nothing "
                f"is ever sold and the regime filter cannot free any slot")

        res = run_backtest(prices, rebals, spec["universe"], cfg["top_n"],
                           cfg["lookback"], cfg["vol_window"], cfg["cost_bps"],
                           exit_rank=cfg.get("exit_rank"), regime=regime)

        begin = rebals[0]
        net = res["net"].loc[begin:]
        gross = res["gross"].loc[begin:]
        net = net / net.iloc[0]
        gross = gross / gross.iloc[0]

        bench = spy.reindex(net.index).ffill()
        bench = bench / bench.iloc[0]

        ann_s = annual_returns(net)
        ann_b = annual_returns(bench)
        annual = {str(y): [ann_s[y], ann_b.get(y, float("nan"))]
                  for y in sorted(ann_s) if not pd.isna(ann_b.get(y, float("nan")))}

        usize = spec["usize"]
        if usize is None:
            sizes = [len([t for t in spec["universe"](d) if t in prices.columns])
                     for d in rebals]
            usize = int(np.median(sizes))

        # thin the daily series for transport; keep every point that matters
        step = max(1, len(net) // 1400)
        idx = list(range(0, len(net), step))
        if idx[-1] != len(net) - 1:
            idx.append(len(net) - 1)

        dd_s = drawdown(net)
        dd_b = drawdown(bench)

        runs.append({
            "key": spec["key"], "label": spec["label"], "caveat": spec["caveat"],
            "missing_note": spec["missing_note"],
            "dates": [net.index[i].strftime("%Y-%m-%d") for i in idx],
            "net": [round(float(net.iloc[i]), 5) for i in idx],
            "gross": [round(float(gross.iloc[i]), 5) for i in idx],
            "bench": [round(float(bench.iloc[i]), 5) for i in idx],
            "dd": [round(float(dd_s.iloc[i]), 5) for i in idx],
            "bench_dd": [round(float(dd_b.iloc[i]), 5) for i in idx],
            "metrics": metrics(net),
            "gross_metrics": metrics(gross),
            "bench_metrics": metrics(bench),
            "annual": annual,
            "turnover": float(res["traded"].mean()) if len(res["traded"]) else 0.0,
            "exposure": [round(float(res["exposure"].loc[net.index[i]]), 4) for i in idx],
            "avg_exposure": float(res["exposure"].loc[net.index].mean()),
            "blocked": int(res["blocked_rebalances"]),
            "n_rebalances": len(rebals),
            "universe_size": usize,
        })
        m = runs[-1]["metrics"]
        log(f"  [{spec['key']}] {net.index[0].date()} to {net.index[-1].date()}  "
            f"CAGR {m['cagr']*100:.1f}%  maxDD {m['max_drawdown']*100:.0f}%  "
            f"vs index {runs[-1]['bench_metrics']['cagr']*100:.1f}%  "
            f"avg exposure {runs[-1]['avg_exposure']*100:.0f}%  "
            f"blocked {runs[-1]['blocked']}/{len(rebals)}")

    # when both were run, record what the hindsight was worth
    if len(runs) == 2:
        by = {r["key"]: r for r in runs}
        if "biased" in by and "pit" in by:
            gap = by["biased"]["metrics"]["cagr"] - by["pit"]["metrics"]["cagr"]
            ratio = (by["biased"]["net"][-1] / by["pit"]["net"][-1]
                     if by["pit"]["net"][-1] else None)
            cfg["bias_measured"] = {"cagr_gap": gap, "wealth_ratio": ratio}

    # A duplicate pair is always a bug, never a finding.
    if len(runs) == 2 and runs[0]["dates"][0] == runs[1]["dates"][0] and \
            abs(runs[0]["metrics"]["cagr"] - runs[1]["metrics"]["cagr"]) < 1e-9:
        log("  WARNING: the two runs are identical — the point-in-time universe "
            "did not differ from today's. Dropping the duplicate.")
        runs = runs[:1]

    return {"meta": cfg, "runs": runs}

# ═══ main ════════════════════════════════════════════════════════════════════
print("\n[2/5] ดึงรายชื่อ S&P 500 และประวัติการเปลี่ยนสมาชิก")
NET_CAUSES = ["ไม่ได้ต่ออินเทอร์เน็ต / no internet connection",
              "ไฟร์วอลล์หรือ proxy บล็อกอยู่ / a firewall or proxy is blocking it"]
try:
    tables = fetch_tables(requests)
    current = [norm(t) for t in tables[0]["Symbol"].tolist()]
    changes = parse_changes(tables)
except Exception as e:
    _bail("ดึงข้อมูลจาก Wikipedia ไม่สำเร็จ / could not fetch from Wikipedia",
          NET_CAUSES, f"{type(e).__name__}: {e}")

print(f"      สมาชิกปัจจุบัน {len(current)} ตัว, ประวัติจาก Wikipedia {len(changes)} รายการ")

# ─── ประวัติสมาชิกดัชนีย้อนหลัง (แหล่งหลัก) ──────────────────────────────────
# Wikipedia เอาตารางประวัติออกไปแล้ว จึงใช้ไฟล์ constituent รายวันแทน
# ซึ่งดีกว่าเดิมด้วย เพราะย้อนถึงปี 1996 ไม่ใช่แค่ 2020
print("\n      ดึงไฟล์สมาชิกดัชนีย้อนหลัง (point-in-time)")
snapshots = []
try:
    txt = fetch_membership_history(requests, log=print)
    if txt:
        snapshots = parse_membership_csv(txt)
        if snapshots:
            print(f"      {len(snapshots)} snapshots: "
                  f"{snapshots[0][0].date()} ถึง {snapshots[-1][0].date()}")
            sizes = [len(st) for _, st in snapshots]
            print(f"      ขนาดดัชนี {min(sizes)}-{max(sizes)} ตัว")
        else:
            print("      (ไฟล์โหลดได้แต่ parse ไม่ได้ — จะใช้ประวัติจาก Wikipedia แทน)")
    else:
        print("      (โหลดไฟล์ไม่ได้ — จะใช้ประวัติจาก Wikipedia แทน)")
except Exception as e:
    print(f"      (ดึงไม่สำเร็จ: {type(e).__name__} — ใช้ Wikipedia แทน)")
if changes:
    print(f"      ประวัติย้อนไปถึง {min(c[0] for c in changes).date()}")
else:
    print("      (ไม่พบตารางประวัติ — จะได้เฉพาะชุดที่มี survivorship bias)")

ever = set(current)
for _, a, r in changes:
    if a: ever.add(a)
    if r: ever.add(r)
for _, st in snapshots:
    ever |= st
tickers = sorted(ever)

print(f"\n[3/5] ดาวน์โหลดราคา {YEARS} ปี ของ {len(tickers)} ตัว")
print("      ขั้นตอนนี้ใช้เวลาหลายนาที ปล่อยให้รันไปเรื่อย ๆ ได้")
try:
    raw = yf.download(tickers, period=f"{YEARS}y", auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(axis=1, how="all").sort_index()
except Exception as e:
    _bail("ดาวน์โหลดราคาไม่สำเร็จ / could not download prices",
          NET_CAUSES, f"{type(e).__name__}: {e}")

if prices.empty or prices.shape[1] < 50:
    _bail("ได้ข้อมูลราคาน้อยเกินไป / too little price data came back",
          ["Yahoo Finance อาจกำลังจำกัดการเรียก รอสัก 5-10 นาทีแล้วลองใหม่",
           "Yahoo Finance may be rate-limiting — wait several minutes and rerun"])
print(f"      {prices.shape[0]} วัน x {prices.shape[1]} ตัว "
      f"({prices.index[0].date()} ถึง {prices.index[-1].date()})")

print(f"\n[4/5] ดาวน์โหลด benchmark ({BENCHMARK}, total return)")
try:
    braw = yf.download(BENCHMARK, period=f"{YEARS}y", auto_adjust=True, progress=False)
    if isinstance(braw.columns, pd.MultiIndex):
        braw.columns = braw.columns.droplevel(-1)
    spy = braw["Close"].dropna()
except Exception as e:
    _bail("ดาวน์โหลด benchmark ไม่สำเร็จ / could not download the benchmark",
          NET_CAUSES, f"{type(e).__name__}: {e}")
print(f"      {len(spy)} วัน")

# ─── regime filter: ดัชนีอยู่เหนือ SMA200 หรือไม่ ─────────────────────────────
regime = None
if REGIME_SMA:
    print(f"\n      ดึง {REGIME_TICKER} สำหรับ regime filter (SMA{REGIME_SMA})")
    try:
        graw = yf.download(REGIME_TICKER, period=f"{YEARS + 2}y",
                           auto_adjust=True, progress=False)
        if isinstance(graw.columns, pd.MultiIndex):
            graw.columns = graw.columns.droplevel(-1)
        gc = graw["Close"].dropna()
        sma = gc.rolling(window=REGIME_SMA).mean()
        above = (gc > sma)
        above[sma.isna()] = True          # ก่อนมี SMA ครบ ให้ซื้อได้ตามปกติ
        regime = above.reindex(prices.index).ffill().fillna(True)
        pct_on = float(regime.mean()) * 100
        print(f"      ดัชนีอยู่เหนือ SMA{REGIME_SMA} {pct_on:.0f}% ของวันทำการ")
    except Exception as e:
        print(f"      (ดึงดัชนีไม่สำเร็จ — ปิด regime filter: {type(e).__name__})")
        regime = None

print("\n[5/5] รัน backtest ทั้งสองชุด")
cfg = {"years": YEARS, "top_n": TOP_N, "lookback": LOOKBACK_DAYS,
       "vol_window": VOL_WINDOW, "cost_bps": COST_BPS, "rebalance": "monthly",
       "benchmark": BENCHMARK, "exit_rank": EXIT_RANK, "regime_sma": REGIME_SMA,
       "show_biased": SHOW_BIASED}
payload = run_all(prices, spy, current, changes, cfg, regime=regime,
                  snapshots=snapshots)

if not payload["runs"]:
    _bail("ไม่มีข้อมูลพอสำหรับ backtest / not enough history to backtest",
          ["ลองลด YEARS หรือ LOOKBACK_DAYS ในบล็อก CONFIG"])

html = build_report(payload, standalone=True)
out = os.path.abspath(OUTPUT_HTML)
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
with open("backtest-payload.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))

print("\n" + "=" * 68)
for r in payload["runs"]:
    m, bm = r["metrics"], r["bench_metrics"]
    print(f"  {r['label']}")
    print(f"    {r['dates'][0]} to {r['dates'][-1]}")
    print(f"    strategy CAGR {m['cagr']*100:6.1f}%   index {bm['cagr']*100:6.1f}%"
          f"   diff {(m['cagr']-bm['cagr'])*100:+.1f}%")
    print(f"    worst drawdown {m['max_drawdown']*100:5.0f}%   "
          f"traded {r['turnover']*100:.0f}%/month   "
          f"invested {r.get('avg_exposure', 1)*100:.0f}% on average")
    if r.get("n_rebalances"):
        print(f"    buying blocked in {r['blocked']} of {r['n_rebalances']} months")
print("=" * 68)

if len(payload["runs"]) == 2:
    a, b = payload["runs"][0]["metrics"]["cagr"], payload["runs"][1]["metrics"]["cagr"]
    print(f"\n  ส่วนต่าง CAGR ระหว่างสองชุด: {(a-b)*100:+.1f}% ต่อปี")
    print("  ยิ่งห่างมาก แปลว่า survivorship bias ยิ่งมีผลมาก")

print(f"\nเขียนไฟล์ {out} แล้ว")
if OPEN_BROWSER:
    import webbrowser
    webbrowser.open(pathlib.Path(out).as_uri())
    print("เปิดในเบราว์เซอร์แล้ว")

if sys.platform == "win32" and sys.stdin.isatty():
    input("\nกด Enter เพื่อปิดหน้าต่าง...")
