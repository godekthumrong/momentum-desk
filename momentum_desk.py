"""
Momentum Desk — S&P 500 momentum ranking + inverse-volatility position sizing.

รันไฟล์นี้ไฟล์เดียวจบ: ดึงข้อมูล -> คำนวณ -> สร้าง momentum-desk.html -> เปิดในเบราว์เซอร์
ถ้าไม่มี yfinance จะติดตั้งให้อัตโนมัติ

วิธีรัน: ดับเบิลคลิก run_momentum_desk.bat
        หรือ  python momentum_desk.py
"""

# ═══ CONFIG — แก้ค่าตรงนี้ที่เดียว ═══════════════════════════════════════════
PORTFOLIO_VALUE = 65_897      # มูลค่าพอร์ตรวม (USD)
LOOKBACK_DAYS   = 125         # ช่วงคำนวณ momentum (วันทำการ)
VOL_WINDOW      = 20          # ช่วงคำนวณ volatility (วันทำการ)
TOP_N           = 20          # จำนวนหุ้นใน momentum portfolio
HISTORY_PERIOD  = "1y"        # ช่วงข้อมูลหุ้นรายตัวที่ดึงจาก Yahoo
SMA_WINDOW      = 200         # เส้นค่าเฉลี่ยบนกราฟดัชนี (วันทำการ)
INDEX_SHOW_BARS = 252         # จำนวนแท่งเทียนที่แสดง (~1 ปี)
INDEX_FETCH_PERIOD = "2y"     # ดึงยาวกว่าที่แสดง เพื่อให้ SMA ครบตั้งแต่แท่งแรก

MY_WATCHLIST = [
    "MU", "WDC", "STX", "SNDK", "LRCX", "FIX", "AMAT", "KLAC", "GLW", "COHR",
    "INTC", "JBHT", "MRVL", "DELL", "AMD", "HUM", "HPE", "DDOG", "FLEX", "PANW",
]

OUTPUT_HTML   = "momentum-desk.html"
OPEN_BROWSER  = True          # เปิดแดชบอร์ดในเบราว์เซอร์เมื่อเสร็จ
COPY_JSON     = True          # copy JSON ลง clipboard เพื่อส่งกลับให้ Claude
# ═════════════════════════════════════════════════════════════════════════════

# ── environment overrides (สำหรับรันบนคลาวด์/CI ที่ไม่มีจอและไม่มี clipboard) ──
import os as _os
if _os.environ.get("MOMENTUM_HEADLESS"):
    OPEN_BROWSER = False
    COPY_JSON = False
if _os.environ.get("MOMENTUM_PORTFOLIO_VALUE"):
    PORTFOLIO_VALUE = float(_os.environ["MOMENTUM_PORTFOLIO_VALUE"])
if _os.environ.get("MOMENTUM_WATCHLIST"):
    MY_WATCHLIST = [t.strip().upper() for t in
                    _os.environ["MOMENTUM_WATCHLIST"].replace(",", " ").split() if t.strip()]

import io
import json
import os
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


def _bail(title, causes, detail=""):
    print("\n  " + "-" * 58)
    print(f"  {title}")
    if causes:
        print()
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


def _ensure(pkg, import_name=None):
    """Import pkg, installing it on first run if missing.

    Plain `pip install` fails on distro and Homebrew Pythons, which mark
    themselves externally-managed, so fall back through --user and
    --break-system-packages before giving up.
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
        _bail(f"ติดตั้ง {pkg} ไม่สำเร็จ / could not install {pkg}",
              ["ไม่ได้ต่ออินเทอร์เน็ต / no internet connection",
               "ไฟร์วอลล์หรือ proxy บล็อก pip / a firewall or proxy blocks pip",
               f"ลองรันเอง / try by hand:  {sys.executable} -m pip install {pkg}"],
              detail)

    # A --user install lands outside the current process's import path.
    import importlib
    import site
    try:
        usersite = site.getusersitepackages()
        if usersite and usersite not in sys.path:
            sys.path.append(usersite)
    except Exception:
        pass
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    except ImportError:
        _bail(f"ติดตั้ง {pkg} แล้วแต่ import ไม่ได้ / installed {pkg} but cannot import it",
              ["ลองปิดหน้าต่างนี้แล้วรันใหม่อีกครั้ง / close this window and run again"],
              detail)


print("=" * 68)
print("  MOMENTUM DESK")
print("=" * 68)
print("\n[1/5] ตรวจสอบ dependencies")
for pkg, mod in [("pandas", "pandas"), ("numpy", "numpy"),
                 ("requests", "requests"), ("lxml", "lxml"),
                 ("yfinance", "yfinance")]:
    _ensure(pkg, mod)
print("      พร้อม")

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ═══ dashboard template ═══

import json

CSS = """
/* Claude palette, committed to a single light treatment — every colour is
   painted explicitly so the page holds its look on any host background.
   Contrast (WCAG, vs #FAF9F5): ink 17.5 · ink-2 8.7 · muted 4.9 · link 5.8 ·
   bar 3.1 (clears the 3:1 data-mark gate) · white-on-accent 4.1. */
:root {
  color-scheme: light;
  --page:#F0EEE6;        /* warm oat ground   */
  --surface:#FAF9F5;     /* card              */
  --surface-2:#EDEAE0;   /* input, bar track  */
  --surface-3:#E3DFD2;   /* hover             */
  --ink:#141413;
  --ink-2:#4A4844;
  --muted:#6F6D66;
  --grid:#E4E0D4;
  --border:rgba(20,20,19,0.11);
  --accent:#C4603C;      /* buttons, focus ring */
  --link:#A34527;        /* accent as text      */
  --bar:#CC785C;         /* Book Cloth — the data mark */
  --pill-bg:#F7E7E0; --pill-ink:#B0522F;
  --tag-bg:#EBDBBC;  --tag-ink:#6B4F2A;   /* manilla */
  --ok:#3F6B45;      --ok-bg:#E6EEE4;
  --warn:#8F3016;    --warn-bg:#F9E5DC;
  /* candle direction. This pair is the only green/coral combination that
     cleared CVD separation (protan ΔE 10.8; the obvious alternatives scored
     3.7–5.8). Fill carries the direction too, so colour is never alone. */
  --up:#3F6B45; --down:#CC785C;
  /* SMA overlay. Purple was the only hue family that cleared CVD against
     both candle colours; this step also clears the chroma floor, which the
     first plum I tried (#7A5B8F, chroma 0.087) did not. Trio measures:
     CVD dE 10.8, normal-vision 23.1, all three >= 3:1 on the surface. */
  --sma:#7C46A8;
}

* { box-sizing: border-box; }
html, body { margin:0; padding:0; }
body {
  background: var(--page); color: var(--ink);
  font-family:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased; line-height:1.45;
}
.wrap { max-width:1180px; margin:0 auto; padding:40px 24px 80px; }

.eyebrow {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:600; font-size:12px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted);
}

header.masthead {
  display:flex; justify-content:space-between; align-items:flex-end;
  gap:24px; flex-wrap:wrap; padding-bottom:24px; margin-bottom:28px;
  border-bottom:1px solid var(--border);
}
header.masthead h1 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:650; font-size:clamp(28px,4vw,40px); letter-spacing:-.01em;
  margin:6px 0 8px; text-wrap:balance;
}
header.masthead p.sub { color:var(--ink-2); font-size:15px; max-width:54ch; margin:0; }
.snapshot-badge {
  display:inline-flex; align-items:center; gap:8px;
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12.5px;
  color:var(--ink-2); background:var(--surface-2); border:1px solid var(--border);
  border-radius:999px; padding:6px 14px; white-space:nowrap;
}
.snapshot-badge .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); }

/* ── controls ─────────────────────────────────────────────────── */
.controls {
  display:flex; gap:22px; flex-wrap:wrap; align-items:flex-end;
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:16px 20px; margin-bottom:28px;
}
.field { display:flex; flex-direction:column; gap:5px; }
.field label {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:11px; font-weight:600; letter-spacing:.05em;
  text-transform:uppercase; color:var(--muted);
}
.field .hint { font-size:11.5px; color:var(--muted); }
input[type="text"], input[type="number"] {
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:13.5px; color:var(--ink);
  background:var(--surface-2); border:1px solid var(--border);
  border-radius:6px; padding:7px 10px; min-width:0;
}
input[type="text"]:focus-visible, input[type="number"]:focus-visible,
button:focus-visible {
  outline:2px solid var(--accent); outline-offset:2px;
}
button {
  font-family:"IBM Plex Sans", system-ui, sans-serif;
  font-size:13px; font-weight:500; color:var(--ink);
  background:var(--surface-2); border:1px solid var(--border);
  border-radius:6px; padding:7px 14px; cursor:pointer;
}
button:hover { background:var(--surface-3); }
button.primary {
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;
}
button.primary:hover { filter:brightness(1.08); }
button.link {
  background:none; border:none; padding:4px 2px; color:var(--link);
  text-decoration:underline; font-size:12.5px;
}
button.link:hover { background:none; filter:brightness(1.15); }

.msg {
  font-size:12.5px; padding:7px 12px; border-radius:6px; margin-bottom:14px;
  display:none;
}
.msg.show { display:block; }
.msg.warn { color:var(--warn); background:var(--warn-bg); }
.msg.ok   { color:var(--ok); background:var(--ok-bg); }

/* ── kpis ─────────────────────────────────────────────────────── */
.kpi-row {
  display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border);
  border-radius:10px; overflow:hidden; margin-bottom:40px;
}
.kpi { background:var(--surface); padding:18px 20px; display:flex; flex-direction:column; gap:6px; }
.kpi .label { font-size:12px; color:var(--muted); letter-spacing:.03em; }
.kpi .value { font-weight:650; font-size:26px; letter-spacing:-.01em; }
.kpi .value .unit { font-size:15px; font-weight:500; color:var(--ink-2); margin-left:2px; }
.kpi .detail {
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:12.5px; color:var(--ok);
}
.kpi .detail.neutral { color:var(--ink-2); }

section { margin-bottom:48px; }
.section-head {
  display:flex; justify-content:space-between; align-items:baseline;
  gap:16px; flex-wrap:wrap; margin-bottom:14px;
}
.section-head h2 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:650; font-size:19px; margin:0; letter-spacing:-.005em;
}
.section-head .note { font-size:13px; color:var(--muted); }

/* ── tables ───────────────────────────────────────────────────── */
.table-scroll { overflow-x:auto; border:1px solid var(--border); border-radius:10px; }
table {
  width:100%; border-collapse:collapse; background:var(--surface);
  font-size:13.5px; min-width:560px;
}
thead th {
  text-align:right;
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:600; font-size:11.5px; letter-spacing:.04em; text-transform:uppercase;
  color:var(--muted); padding:10px 14px; border-bottom:1px solid var(--grid); white-space:nowrap;
}
thead th:first-child, thead th:nth-child(2) { text-align:left; }
tbody td {
  text-align:right; padding:9px 14px; border-bottom:1px solid var(--grid);
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap;
}
tbody td:first-child, tbody td:nth-child(2) {
  text-align:left; font-family:"IBM Plex Sans", system-ui, sans-serif;
}
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover td { background:var(--surface-2); }
td.empty {
  text-align:center; color:var(--muted); padding:26px 14px;
  font-family:"IBM Plex Sans", system-ui, sans-serif;
}

.rank-pill {
  display:inline-flex; align-items:center; justify-content:center;
  min-width:22px; height:20px; padding:0 5px; border-radius:5px;
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:11.5px; font-weight:600; background:var(--surface-2); color:var(--ink-2);
}
.rank-pill.top { background:var(--pill-bg); color:var(--pill-ink); }
.ticker { font-family:"IBM Plex Mono", ui-monospace, monospace; font-weight:600; letter-spacing:.01em; }
.tag {
  display:inline-flex; align-items:center; font-size:10.5px;
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-weight:600; letter-spacing:.03em; text-transform:uppercase;
  color:var(--tag-ink); background:var(--tag-bg); border-radius:4px;
  padding:2px 6px; margin-left:8px; vertical-align:middle;
}
button.remove {
  background:none; border:none; padding:2px 7px; margin-left:6px;
  color:var(--muted); font-size:15px; line-height:1; cursor:pointer;
  border-radius:4px; vertical-align:middle;
}
button.remove:hover { background:var(--surface-3); color:var(--link); }

/* ── candlestick chart ────────────────────────────────────────── */
.chart-card {
  border:1px solid var(--border); border-radius:10px;
  background:var(--surface); padding:20px 22px 14px;
}
.chart-head {
  display:flex; justify-content:space-between; align-items:baseline;
  gap:18px; flex-wrap:wrap; margin-bottom:14px;
}
.chart-stats { display:flex; gap:22px; flex-wrap:wrap; }
.chart-stats div {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12.5px;
  color:var(--ink-2); font-variant-numeric:tabular-nums;
}
.chart-stats span.k {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:10.5px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); display:block;
}
.chart-legend { display:flex; gap:16px; align-items:center; font-size:12px; color:var(--ink-2); }
.chart-legend i { display:inline-block; width:9px; height:13px; margin-right:6px; vertical-align:-2px; }
.chart-legend i.up   { background:var(--surface); border:1.4px solid var(--up); }
.chart-legend i.down { background:var(--down); border:1.4px solid var(--down); }
.chart-legend i.sma  { height:0; border:0; border-top:2px solid var(--sma);
                       vertical-align:4px; width:16px; }

.chart-wrap { position:relative; overflow:hidden; }
#spx-chart svg { display:block; }
.chart-tip {
  position:absolute; pointer-events:none; opacity:0; transition:opacity .12s;
  background:var(--surface); border:1px solid var(--border); border-radius:7px;
  padding:8px 11px; font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:11.5px; line-height:1.55; color:var(--ink);
  box-shadow:0 3px 14px rgba(20,20,19,0.13); white-space:nowrap; z-index:5;
  font-variant-numeric:tabular-nums;
}
.chart-tip.on { opacity:1; }
.chart-tip b {
  display:block; font-family:"IBM Plex Sans", system-ui, sans-serif;
  font-size:11.5px; font-weight:650; margin-bottom:3px;
}
.chart-tip u { text-decoration:none; color:var(--muted); display:inline-block; width:34px; }

/* ── panel ────────────────────────────────────────────────────── */
@media (max-width:900px) {
  .kpi-row { grid-template-columns:repeat(2,1fr); }
}
.panel { border:1px solid var(--border); border-radius:10px; background:var(--surface); padding:22px; }
.panel .panel-head { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; gap:16px; }
.panel .panel-head h3 {
  font-family:"IBM Plex Sans Condensed", system-ui, sans-serif;
  font-size:16.5px; font-weight:650; margin:0 0 3px;
}
.panel .panel-head .stat {
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-size:13px; color:var(--ink-2); text-align:right; white-space:nowrap;
}
.panel .panel-desc { font-size:12.5px; color:var(--muted); margin:0; max-width:52ch; }

/* bars beside the table on wide screens, stacked when there is no room */
.split { display:grid; grid-template-columns:minmax(280px,0.85fr) 1.15fr; gap:26px; align-items:start; }
@media (max-width:900px) { .split { grid-template-columns:1fr; gap:20px; } }

.adder { display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; }
.adder input { flex:0 1 220px; }

.barchart { display:flex; flex-direction:column; gap:9px; margin-bottom:18px; }
.bar-row { display:grid; grid-template-columns:52px 1fr 52px; align-items:center; gap:10px; }
.bar-row .tk { font-family:"IBM Plex Mono", ui-monospace, monospace; font-weight:600; font-size:12px; }
.bar-track {
  display:block; height:14px; background:var(--surface-2);
  border-radius:4px; overflow:hidden;
}
.bar-fill {
  display:block; height:100%; min-width:3px; border-radius:4px;
  background:var(--bar-color, var(--bar));
}
.bar-row .wt {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12px;
  color:var(--ink-2); text-align:right; font-variant-numeric:tabular-nums;
}
.panel .table-scroll table { min-width:420px; font-size:12.5px; }
.panel .table-scroll td, .panel .table-scroll th { padding:8px 11px; }
.panel .barchart { margin-bottom:0; }

footer {
  border-top:1px solid var(--border); padding-top:20px; margin-top:12px;
  font-size:12.5px; color:var(--muted); max-width:78ch;
}
footer p { margin:0 0 10px; }
@media (prefers-reduced-motion: no-preference) { .bar-fill { transition:width .35s ease; } }
"""

BODY = """
<div class="wrap">
  <header class="masthead">
    <div>
      <div class="eyebrow">S&amp;P 500 &middot; Absolute momentum + inverse-volatility sizing</div>
      <h1>Momentum Desk</h1>
      <p class="sub">S&amp;P 500 names ranked by __LOOKBACK__-day quant momentum score, with a watchlist you edit yourself &mdash; sized by weighting each position inversely to its __VOLWIN__-day volatility.</p>
    </div>
    <span class="snapshot-badge"><span class="dot"></span>Snapshot &middot; __ASOF__ close</span>
  </header>

  <div class="controls">
    <div class="field">
      <label for="pv">Portfolio value (USD)</label>
      <input type="number" id="pv" min="1" step="100" style="width:150px">
    </div>
    <div class="field">
      <label for="topn">Ranking depth (top N)</label>
      <input type="number" id="topn" min="1" max="100" step="1" style="width:110px">
    </div>
    <div class="field">
      <span class="hint">Edits stay in this browser.<br>Rerun the script for fresh prices.</span>
    </div>
    <div class="field">
      <button type="button" id="reset-all" class="link">Reset everything to defaults</button>
    </div>
  </div>

  <div class="kpi-row" id="kpi-row"></div>

  <section id="chart-section" hidden>
    <div class="section-head">
      <h2>S&amp;P 500 &middot; past year</h2>
      <span class="note" id="chart-note">Daily candles &middot; hollow = close above open</span>
    </div>
    <div class="chart-card">
      <div class="chart-head">
        <div class="chart-stats" id="chart-stats"></div>
        <div class="chart-legend">
          <span><i class="up"></i>Up day</span>
          <span><i class="down"></i>Down day</span>
          <span id="sma-key" hidden><i class="sma"></i>SMA <span id="sma-n"></span></span>
        </div>
      </div>
      <div class="chart-wrap">
        <div id="spx-chart"></div>
        <div class="chart-tip" id="chart-tip"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2 id="rank-heading">Momentum ranking</h2>
      <span class="note">Ranked by Quant Score (annualized log-slope &times; R&sup2;) &middot; __UNIVERSE__ members</span>
    </div>
    <div class="table-scroll">
      <table id="rank-table">
        <thead><tr><th>Rank</th><th>Ticker</th><th>__LOOKBACK__D momentum</th><th>Quant score</th><th>Watch</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Position sizing</h2>
      <span class="note">Inverse __VOLWIN__-day volatility weighting</span>
    </div>
    <div class="panel">
      <div class="panel-head">
        <div>
          <h3>Watchlist portfolio</h3>
          <p class="panel-desc">Your own names, weighted so lower-volatility positions carry more capital.</p>
        </div>
        <div class="stat" id="stat-watch"></div>
      </div>

      <div class="adder">
        <input type="text" id="add-ticker" list="ticker-list" placeholder="Add ticker (e.g. NVDA)"
               autocomplete="off" spellcheck="false" maxlength="10" aria-label="Add ticker to watchlist">
        <datalist id="ticker-list"></datalist>
        <button type="button" id="add-btn" class="primary">Add</button>
        <button type="button" id="reset-watch" class="link">Reset list</button>
      </div>
      <div class="msg" id="add-msg" role="status"></div>

      <div class="split">
        <div class="barchart" id="bars-watch"></div>
        <div class="table-scroll">
          <table id="table-watch">
            <thead><tr><th>Ticker</th><th>Rank</th><th>Weight</th><th>Vol __VOLWIN__D</th><th>Price</th><th>Shares</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <footer>
    <p><strong>Reading this:</strong> "Quant score" is the annualized exponential-regression slope of the __LOOKBACK__-day price series, scaled by its R&sup2; &mdash; it rewards trends that are both strong and smooth. Weight is inverse to __VOLWIN__-day realized volatility, normalized to sum to 100% across the watchlist, then converted to shares at the snapshot price. A tag beside a ticker means it also lands in the top-N ranking above.</p>
    <p>Prices and volatility are a fixed snapshot from the __ASOF__ close across __UNIVERSE__ index members &mdash; editing the watchlist re-slices that snapshot, it does not fetch new prices. Rerun the script to refresh.</p>
  </footer>
</div>
"""

SCRIPT = r"""
(function () {
  var D = window.__MOMENTUM_DATA__;
  var META = D.meta;
  var STORE = "momentum-desk-v2";

  // universe rows: [ticker, rank, momentum, score, volatility, price]
  var U = {}, ALL = [];
  D.universe.forEach(function (r) {
    var o = {ticker:r[0], rank:r[1], momentum:r[2], score:r[3], vol:r[4], price:r[5]};
    U[o.ticker] = o; ALL.push(o);
  });
  ALL.sort(function (a, b) { return a.rank - b.rank; });

  var state = {
    pv: META.portfolio_value,
    topN: META.top_n,
    watchlist: D.watchlist.slice()
  };

  try {
    var saved = JSON.parse(localStorage.getItem(STORE) || "null");
    if (saved && Array.isArray(saved.watchlist)) {
      state.watchlist = saved.watchlist.filter(function (t) { return U[t]; });
      if (saved.pv > 0) state.pv = saved.pv;
      if (saved.topN > 0) state.topN = saved.topN;
    }
  } catch (e) { /* private mode or blocked storage — defaults are fine */ }

  function save() {
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {}
  }

  // ── formatting ────────────────────────────────────────────────
  function pct(n) { return n.toFixed(1) + "%"; }
  function wpct(n) { return (n * 100).toFixed(1) + "%"; }
  function money(n) {
    return "$" + n.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
  }
  function money0(n) {
    return "$" + n.toLocaleString("en-US", {maximumFractionDigits:0});
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
    });
  }

  // ── the one calculation ───────────────────────────────────────
  // inverse-volatility weights, normalized to 1, then converted to shares
  function size(tickers) {
    var rows = tickers.map(function (t) { return U[t]; })
                      .filter(function (r) { return r && r.vol > 0 && r.price > 0; });
    if (!rows.length) return [];
    var invSum = rows.reduce(function (s, r) { return s + 1 / r.vol; }, 0);
    return rows.map(function (r) {
      var w = (1 / r.vol) / invSum;
      return {ticker:r.ticker, rank:r.rank, vol:r.vol, price:r.price,
              weight:w, allocated:w * state.pv, shares:w * state.pv / r.price};
    }).sort(function (a, b) { return b.weight - a.weight; });
  }

  // ── rendering ─────────────────────────────────────────────────
  function bars(elId, rows, colorVar) {
    var el = document.getElementById(elId);
    if (!rows.length) { el.innerHTML = ""; return; }
    var max = Math.max.apply(null, rows.map(function (r) { return r.weight; }));
    el.innerHTML = rows.map(function (r) {
      return '<div class="bar-row"><span class="tk">' + esc(r.ticker) + '</span>' +
             '<span class="bar-track"><span class="bar-fill" style="width:' +
             (r.weight / max * 100).toFixed(1) + '%; --bar-color: var(' + colorVar + ');"></span></span>' +
             '<span class="wt">' + wpct(r.weight) + '</span></div>';
    }).join("");
  }

  function renderAll() {
    var watch = size(state.watchlist);
    // the top-N names by rank — used for the ranking table and the "Top N" tag
    var topSet = {};
    ALL.slice(0, state.topN).forEach(function (r) { topSet[r.ticker] = 1; });

    // KPIs
    var lead = ALL[0];
    var avgMom = ALL.slice(0, state.topN)
                    .reduce(function (s, r) { return s + r.momentum; }, 0) / state.topN;
    var overlap = watch.filter(function (r) { return topSet[r.ticker]; }).length;
    var kpis = [
      {label:"Top momentum name", value:lead.ticker,
       detail:(lead.momentum >= 0 ? "+" : "") + lead.momentum.toFixed(1) + "% · " +
              META.lookback + "D, score " + lead.score.toFixed(1)},
      {label:"Avg. " + META.lookback + "D momentum (top " + state.topN + ")",
       value:avgMom.toFixed(1), unit:"%",
       detail:"across the top " + state.topN + " by score", neutral:true},
      {label:"Portfolio value", value:money0(state.pv),
       detail:"fully allocated across the watchlist", neutral:true},
      {label:"Watchlist overlap", value:overlap + " / " + watch.length,
       detail:"watchlist names also in top " + state.topN, neutral:true}
    ];
    document.getElementById("kpi-row").innerHTML = kpis.map(function (k) {
      return '<div class="kpi"><div class="label">' + esc(k.label) + '</div>' +
             '<div class="value">' + esc(k.value) +
             '<span class="unit">' + (k.unit || "") + '</span></div>' +
             '<div class="detail' + (k.neutral ? " neutral" : "") + '">' + esc(k.detail) + '</div></div>';
    }).join("");

    // ranking table — shows the momentum portfolio's members, with a watch toggle
    document.getElementById("rank-heading").textContent =
      "Momentum ranking · top " + state.topN + " of " + ALL.length;
    var watchSet = {};
    state.watchlist.forEach(function (t) { watchSet[t] = 1; });
    document.querySelector("#rank-table tbody").innerHTML = ALL.slice(0, state.topN).map(function (r) {
      var on = watchSet[r.ticker];
      return '<tr><td><span class="rank-pill' + (r.rank <= 3 ? " top" : "") + '">' + r.rank + '</span></td>' +
             '<td class="ticker">' + esc(r.ticker) + '</td>' +
             '<td>' + pct(r.momentum) + '</td>' +
             '<td>' + r.score.toFixed(2) + '</td>' +
             '<td><button type="button" class="link watch-toggle" data-ticker="' + esc(r.ticker) + '">' +
             (on ? "Remove" : "Add") + '</button></td></tr>';
    }).join("");

    // watchlist portfolio
    document.getElementById("stat-watch").innerHTML =
      money0(state.pv) + "<br>" + watch.length + " positions";
    bars("bars-watch", watch, "--bar");
    var wbody = document.querySelector("#table-watch tbody");
    if (!watch.length) {
      wbody.innerHTML = '<tr><td class="empty" colspan="6">Watchlist is empty — add a ticker above.</td></tr>';
    } else {
      wbody.innerHTML = watch.map(function (r) {
        return '<tr><td class="ticker">' + esc(r.ticker) +
               (topSet[r.ticker] ? '<span class="tag">Top ' + state.topN + '</span>' : "") +
               '<button type="button" class="remove" data-ticker="' + esc(r.ticker) +
               '" aria-label="Remove ' + esc(r.ticker) + '" title="Remove">&times;</button></td>' +
               '<td>' + (r.rank == null ? "&mdash;" : r.rank) + '</td>' +
               '<td>' + wpct(r.weight) + '</td>' +
               '<td>' + (r.vol * 100).toFixed(1) + '%</td>' +
               '<td>' + money(r.price) + '</td>' +
               '<td>' + r.shares.toFixed(2) + '</td></tr>';
      }).join("");
    }
  }

  // ── messages ──────────────────────────────────────────────────
  var msgEl = document.getElementById("add-msg");
  var msgTimer = null;
  function flash(text, kind) {
    msgEl.textContent = text;
    msgEl.className = "msg show " + (kind || "warn");
    clearTimeout(msgTimer);
    msgTimer = setTimeout(function () { msgEl.className = "msg"; }, 4000);
  }

  // ── actions ───────────────────────────────────────────────────
  function addTicker(raw) {
    var t = String(raw || "").trim().toUpperCase().replace(/\./g, "-");
    if (!t) return;
    if (!U[t]) { flash('"' + t + '" is not in this snapshot of the index.', "warn"); return; }
    if (state.watchlist.indexOf(t) !== -1) { flash(t + " is already on the watchlist.", "warn"); return; }
    state.watchlist.push(t);
    save(); renderAll();
    flash(t + " added — rank " + U[t].rank + ", weights rebalanced.", "ok");
  }

  function removeTicker(t) {
    var i = state.watchlist.indexOf(t);
    if (i === -1) return;
    state.watchlist.splice(i, 1);
    save(); renderAll();
    flash(t + " removed — weights rebalanced.", "ok");
  }

  var input = document.getElementById("add-ticker");
  document.getElementById("add-btn").addEventListener("click", function () {
    addTicker(input.value); input.value = ""; input.focus();
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); addTicker(input.value); input.value = ""; }
  });

  document.getElementById("table-watch").addEventListener("click", function (e) {
    var b = e.target.closest("button.remove");
    if (b) removeTicker(b.dataset.ticker);
  });
  document.getElementById("rank-table").addEventListener("click", function (e) {
    var b = e.target.closest("button.watch-toggle");
    if (!b) return;
    var t = b.dataset.ticker;
    if (state.watchlist.indexOf(t) === -1) addTicker(t); else removeTicker(t);
  });

  var pvEl = document.getElementById("pv");
  var topnEl = document.getElementById("topn");
  pvEl.addEventListener("input", function () {
    var v = parseFloat(pvEl.value);
    if (v > 0) { state.pv = v; save(); renderAll(); }
  });
  topnEl.addEventListener("input", function () {
    var v = parseInt(topnEl.value, 10);
    if (v > 0) { state.topN = Math.min(v, ALL.length); save(); renderAll(); }
  });

  document.getElementById("reset-watch").addEventListener("click", function () {
    state.watchlist = D.watchlist.slice();
    save(); renderAll(); flash("Watchlist reset to the script's default list.", "ok");
  });
  document.getElementById("reset-all").addEventListener("click", function () {
    state = {pv:META.portfolio_value, topN:META.top_n, watchlist:D.watchlist.slice()};
    pvEl.value = state.pv; topnEl.value = state.topN;
    save(); renderAll(); flash("Everything reset to the script's defaults.", "ok");
  });

  // ── S&P 500 candlestick chart ─────────────────────────────────
  // rows: [date, open, high, low, close]
  var IDX = (D.index || []).map(function (r) {
    return {d:r[0], o:r[1], h:r[2], l:r[3], c:r[4], m:(r.length > 5 ? r[5] : null)};
  });
  var SMA_N = D.sma_window || 200;
  var HAS_SMA = IDX.some(function (b) { return b.m != null; });

  function niceTicks(lo, hi, want) {
    var raw = (hi - lo) / want;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return out;
  }

  var MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function monthLabel(d) { return MONTHS[+d.slice(5, 7) - 1] + " " + d.slice(2, 4); }

  // true high/low (before the axis padding) for the range readout
  var lo0 = Infinity, hi0 = -Infinity;
  IDX.forEach(function (b) { if (b.l < lo0) lo0 = b.l; if (b.h > hi0) hi0 = b.h; });

  var tipEl = document.getElementById("chart-tip");
  var chartHost = document.getElementById("spx-chart");
  var lastGeom = null;

  function drawChart() {
    if (!IDX.length || !chartHost) return;

    var W = Math.max(chartHost.clientWidth || 900, 300);
    var H = W < 560 ? 260 : 330;
    var M = {t:10, r:56, b:26, l:8};
    var pw = W - M.l - M.r, ph = H - M.t - M.b;

    var lo = Infinity, hi = -Infinity;
    IDX.forEach(function (b) {
      if (b.l < lo) lo = b.l;
      if (b.h > hi) hi = b.h;
      // the average must fit the scale too, or the line runs off the plot
      if (b.m != null) { if (b.m < lo) lo = b.m; if (b.m > hi) hi = b.m; }
    });
    var pad = (hi - lo) * 0.04 || 1;
    lo -= pad; hi += pad;

    var n = IDX.length;
    var slot = pw / n;
    var body = Math.max(1.2, Math.min(9, slot * 0.62));
    var y = function (v) { return M.t + (hi - v) / (hi - lo) * ph; };
    var xc = function (i) { return M.l + slot * (i + 0.5); };

    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H +
             '" role="img" aria-label="S&P 500 daily candlestick chart, past year">'];

    // horizontal gridlines + price axis on the right
    niceTicks(lo, hi, 5).forEach(function (v) {
      var yy = y(v).toFixed(1);
      s.push('<line x1="' + M.l + '" x2="' + (M.l + pw) + '" y1="' + yy + '" y2="' + yy +
             '" stroke="var(--grid)" stroke-width="1"/>');
      s.push('<text x="' + (M.l + pw + 7) + '" y="' + yy + '" dy="3.5" fill="var(--muted)" ' +
             'font-size="10.5" font-family="IBM Plex Mono, monospace">' +
             v.toLocaleString("en-US", {maximumFractionDigits:0}) + '</text>');
    });

    // month boundaries on the date axis
    var seenMonth = "";
    IDX.forEach(function (b, i) {
      var m = b.d.slice(0, 7);
      if (m === seenMonth) return;
      seenMonth = m;
      if (i === 0 && n > 12) return;
      if (slot < 3.2 && (+b.d.slice(5, 7)) % 2 === 0) return;
      s.push('<text x="' + xc(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" ' +
             'fill="var(--muted)" font-size="10.5" font-family="IBM Plex Mono, monospace">' +
             monthLabel(b.d) + '</text>');
    });

    // candles: wick + body. Hollow body = up, filled = down, so direction
    // survives greyscale and colour-blind viewing.
    IDX.forEach(function (b, i) {
      var up = b.c >= b.o;
      var col = up ? "var(--up)" : "var(--down)";
      var cx = xc(i);
      s.push('<line x1="' + cx.toFixed(1) + '" x2="' + cx.toFixed(1) + '" y1="' + y(b.h).toFixed(1) +
             '" y2="' + y(b.l).toFixed(1) + '" stroke="' + col + '" stroke-width="1"/>');
      var yH = y(b.h), yL = y(b.l);
      var top = y(Math.max(b.o, b.c)), bot = y(Math.min(b.o, b.c));
      // A doji (open == close) would collapse to zero height; the 1px floor
      // must not push the body past the wick it belongs to.
      var hgt = Math.max(1, bot - top);
      if (top + hgt > yL) top = yL - hgt;
      if (top < yH) top = yH;
      s.push('<rect x="' + (cx - body / 2).toFixed(1) + '" y="' + top.toFixed(1) +
             '" width="' + body.toFixed(1) + '" height="' + hgt.toFixed(1) +
             '" fill="' + (up ? "var(--surface)" : col) + '" stroke="' + col +
             '" stroke-width="' + (body > 2.5 ? 1.1 : 0.8) + '"/>');
    });

    // SMA overlay, drawn after the candles so it reads on top. A gap in the
    // series breaks the path rather than bridging across missing days.
    if (HAS_SMA) {
      var seg = [], paths = [];
      IDX.forEach(function (b, i) {
        if (b.m == null) { if (seg.length > 1) paths.push(seg); seg = []; return; }
        seg.push(xc(i).toFixed(1) + ',' + y(b.m).toFixed(1));
      });
      if (seg.length > 1) paths.push(seg);
      paths.forEach(function (pts) {
        s.push('<polyline points="' + pts.join(' ') + '" fill="none" ' +
               'stroke="var(--sma)" stroke-width="2" stroke-linejoin="round" ' +
               'stroke-linecap="round"/>');
      });
    }

    s.push('<rect id="chart-hit" x="' + M.l + '" y="' + M.t + '" width="' + pw + '" height="' + ph +
           '" fill="transparent" style="cursor:crosshair"/>');
    s.push('<line id="chart-guide" y1="' + M.t + '" y2="' + (M.t + ph) +
           '" stroke="var(--ink-2)" stroke-width="1" opacity="0"/>');
    s.push('</svg>');

    chartHost.innerHTML = s.join("");
    lastGeom = {M:M, slot:slot, n:n, W:W, H:H, ph:ph, y:y, xc:xc};

    var first = IDX[0], last = IDX[n - 1];
    var chg = (last.c / first.o - 1) * 100;
    document.getElementById("chart-stats").innerHTML =
      '<div><span class="k">Last close</span>' + last.c.toLocaleString("en-US",
        {minimumFractionDigits:2, maximumFractionDigits:2}) + '</div>' +
      '<div><span class="k">1-year change</span>' + (chg >= 0 ? "+" : "") + chg.toFixed(1) + '%</div>' +
      '<div><span class="k">Range</span>' + lo0.toLocaleString("en-US", {maximumFractionDigits:0}) +
        ' – ' + hi0.toLocaleString("en-US", {maximumFractionDigits:0}) + '</div>' +
      '<div><span class="k">Sessions</span>' + n + '</div>' +
      (last.m != null
        ? '<div><span class="k">vs SMA ' + SMA_N + '</span>' +
          ((last.c / last.m - 1) * 100 >= 0 ? '+' : '') +
          ((last.c / last.m - 1) * 100).toFixed(1) + '%</div>'
        : '');
  }

  function moveTip(clientX) {
    if (!lastGeom) return;
    var rect = chartHost.getBoundingClientRect();
    var i = Math.floor((clientX - rect.left - lastGeom.M.l) / lastGeom.slot);
    i = Math.max(0, Math.min(lastGeom.n - 1, i));
    var b = IDX[i];
    var f = function (v) { return v.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}); };
    tipEl.innerHTML = '<b>' + b.d + '</b>' +
      '<u>Open</u>' + f(b.o) + '<br><u>High</u>' + f(b.h) +
      '<br><u>Low</u>' + f(b.l) + '<br><u>Close</u>' + f(b.c) +
      (b.m != null ? '<br><u>SMA' + SMA_N + '</u>' + f(b.m) : '');
    tipEl.classList.add("on");

    var cx = lastGeom.xc(i);
    var guide = document.getElementById("chart-guide");
    if (guide) { guide.setAttribute("x1", cx); guide.setAttribute("x2", cx); guide.setAttribute("opacity", "0.28"); }

    var tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    var left = cx + 14;
    if (left + tw > lastGeom.W) left = cx - tw - 14;
    tipEl.style.left = Math.max(0, left) + "px";
    tipEl.style.top = Math.max(0, Math.min(lastGeom.H - th, lastGeom.y(b.h) - th - 8)) + "px";
  }

  function hideTip() {
    tipEl.classList.remove("on");
    // park it: an absolutely-positioned box still counts toward scrollWidth,
    // and a stale offset from a wider layout would widen the page
    tipEl.style.left = "0px";
    tipEl.style.top = "0px";
    var guide = document.getElementById("chart-guide");
    if (guide) guide.setAttribute("opacity", "0");
  }

  if (IDX.length) {
    document.getElementById("chart-section").hidden = false;
    if (HAS_SMA) {
      document.getElementById("sma-key").hidden = false;
      document.getElementById("sma-n").textContent = SMA_N;
      document.getElementById("chart-note").innerHTML =
        "Daily candles &middot; hollow = close above open &middot; " +
        SMA_N + "-day simple moving average";
    }
    chartHost.addEventListener("mousemove", function (e) { moveTip(e.clientX); });
    chartHost.addEventListener("mouseleave", hideTip);
    chartHost.addEventListener("touchmove", function (e) {
      if (e.touches[0]) moveTip(e.touches[0].clientX);
    }, {passive:true});
    chartHost.addEventListener("touchend", hideTip);

    drawChart();
    // Guard on width: drawChart replaces chartHost's own innerHTML, so an
    // unguarded observer re-fires on every render and clears the tooltip.
    var rt = null, lastW = chartHost.clientWidth;
    if (window.ResizeObserver) {
      new ResizeObserver(function () {
        if (chartHost.clientWidth === lastW) return;
        lastW = chartHost.clientWidth;
        clearTimeout(rt);
        rt = setTimeout(function () { lastW = chartHost.clientWidth; hideTip(); drawChart(); }, 120);
      }).observe(chartHost);
    }
  }

  // ── init ──────────────────────────────────────────────────────
  document.getElementById("ticker-list").innerHTML =
    ALL.map(function (r) { return '<option value="' + esc(r.ticker) + '"></option>'; }).join("");
  pvEl.value = state.pv;
  topnEl.value = state.topN;
  renderAll();
})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Sans:wght@400;500;650&'
         'family=IBM+Plex+Sans+Condensed:wght@600;650&'
         'family=IBM+Plex+Mono:wght@400;500;600&display=swap">')


def _fill(template, meta):
    return (template
            .replace("__LOOKBACK__", str(meta["lookback"]))
            .replace("__VOLWIN__", str(meta["vol_window"]))
            .replace("__UNIVERSE__", str(meta["universe"]))
            .replace("__ASOF__", meta["as_of"]))


def build_dashboard(payload, standalone=True):
    """payload: {meta, universe, watchlist} -> complete HTML string."""
    body = _fill(BODY, payload["meta"])
    # Escape "<" so a stray "</script>" inside the data can never close the tag early.
    data_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    head_part = f"<title>Momentum Desk</title>\n{FONTS}\n<style>{CSS}</style>\n"
    body_part = (
        f"{body}\n"
        f"<script>window.__MOMENTUM_DATA__ = {data_json};</script>\n"
        f"<script>{SCRIPT}</script>\n"
    )

    if not standalone:
        # Artifact publishing wraps the file in its own doctype/head/body skeleton.
        return head_part + body_part

    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head_part}</head>\n<body>\n{body_part}</body>\n</html>\n"
    )


# ─── 2. ดึงรายชื่อ S&P 500 และราคาย้อนหลัง ────────────────────────────────────
print("\n[2/5] ดึงรายชื่อ S&P 500 จาก Wikipedia")
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

NET_CAUSES = ["ไม่ได้ต่ออินเทอร์เน็ต / no internet connection",
              "ไฟร์วอลล์หรือ proxy ของที่ทำงานบล็อกอยู่ / a firewall or proxy is blocking it",
              "ลองใหม่อีกครั้งในอีกสักครู่ / the service may be busy, try again shortly"]

try:
    resp = requests.get(WIKI, headers=HDRS, timeout=30)
    resp.raise_for_status()
    sp500 = pd.read_html(io.StringIO(resp.text))[0]
    tickers = [t.replace(".", "-") for t in sp500["Symbol"].tolist()]
except Exception as e:
    _bail("ดึงรายชื่อหุ้นจาก Wikipedia ไม่สำเร็จ / could not fetch the ticker list",
          NET_CAUSES, f"{type(e).__name__}: {e}")

if not tickers:
    _bail("ไม่พบรายชื่อหุ้นในหน้า Wikipedia / the Wikipedia page returned no tickers",
          ["หน้าเว็บอาจเปลี่ยนโครงสร้าง / the page layout may have changed"])
print(f"      {len(tickers)} ตัว")

print(f"\n[3/5] ดาวน์โหลดราคาย้อนหลัง {HISTORY_PERIOD} (ใช้เวลาสักครู่)")
def _download(fn, what, attempts=4):
    """Retry with growing waits.

    Yahoo throttles by IP and shared cloud runners are hit hardest; a block
    usually clears in minutes, so backing off beats failing the whole run.
    """
    import time
    delay = 20
    last = None
    for i in range(1, attempts + 1):
        try:
            out = fn()
            if out is not None and not getattr(out, "empty", True):
                return out
            last = "empty response"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if i < attempts:
            print(f"      ลองใหม่ {i+1}/{attempts} ในอีก {delay}s ({what}: {str(last)[:60]})")
            time.sleep(delay)
            delay *= 2
    print(f"      ล้มเหลวหลังลอง {attempts} ครั้ง ({what}: {str(last)[:80]})")
    return None


raw = _download(lambda: yf.download(tickers, period=HISTORY_PERIOD,
                                    auto_adjust=True, progress=True), "prices")
if raw is None:
    _bail("ดาวน์โหลดราคาไม่สำเร็จหลังลองหลายครั้ง / prices failed after retries",
          NET_CAUSES + ["Yahoo อาจบล็อก IP ชั่วคราว ปกติหายเองใน 1 ชั่วโมง",
                        "Yahoo may have rate-limited this IP; it usually clears within an hour"])
data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
data = data.dropna(axis=1, how="all")

if data.empty or data.shape[1] == 0:
    _bail("ดาวน์โหลดราคาแล้วแต่ไม่มีข้อมูล / prices came back empty",
          ["Yahoo Finance อาจกำลังจำกัดการเรียก ลองใหม่ในอีก 2-3 นาที",
           "Yahoo Finance may be rate-limiting — wait a few minutes and rerun"])

AS_OF = data.index[-1].strftime("%Y-%m-%d")
print(f"      {data.shape[0]} แถว x {data.shape[1]} ตัว, ปิดล่าสุด {AS_OF}")

# ─── ดัชนี S&P 500 เอง (สำหรับกราฟแท่งเทียน) ──────────────────────────────────
# ล้มเหลวได้โดยไม่พังทั้งสคริปต์ — แค่ไม่มีกราฟ
# ดึงยาวกว่าที่จะแสดง เพื่อให้ SMA200 มีค่าครบตั้งแต่แท่งแรกที่โชว์
print(f"      ดึงดัชนี ^GSPC ({INDEX_FETCH_PERIOD}) สำหรับกราฟแท่งเทียน + SMA{SMA_WINDOW}")
index_rows = []
try:
    idx = _download(lambda: yf.download("^GSPC", period=INDEX_FETCH_PERIOD,
                                        auto_adjust=True, progress=False), "index", attempts=3)
    if idx is None:
        raise RuntimeError("index download failed after retries")
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.droplevel(-1)
    idx = idx[["Open", "High", "Low", "Close"]].dropna()

    sma = idx["Close"].rolling(window=SMA_WINDOW).mean()

    # แสดงแค่ช่วงท้าย แต่ SMA คำนวณจากข้อมูลเต็มความยาว
    show = idx.tail(INDEX_SHOW_BARS)
    for ts, r in show.iterrows():
        o, h, l, c = float(r.Open), float(r.High), float(r.Low), float(r.Close)
        # ทิ้งแท่งที่ข้อมูลขัดกันเอง (high ต้องคลุม open/close/low เสมอ)
        if not (h >= max(o, c, l) and l <= min(o, c, h)):
            continue
        m = sma.get(ts)
        index_rows.append([ts.strftime("%Y-%m-%d"),
                           round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                           None if m is None or pd.isna(m) else round(float(m), 2)])

    have = sum(1 for r in index_rows if r[5] is not None)
    print(f"      {len(index_rows)} แท่ง (มี SMA{SMA_WINDOW} {have} แท่ง)")
    if have == 0:
        print(f"      (ข้อมูลไม่พอคำนวณ SMA{SMA_WINDOW} — กราฟจะไม่มีเส้น)")
except Exception as e:
    print(f"      (ดึงดัชนีไม่สำเร็จ ข้ามกราฟไป: {type(e).__name__})")

# ─── 3. คำนวณ momentum ────────────────────────────────────────────────────────
print(f"\n[4/5] คำนวณ momentum ({LOOKBACK_DAYS}D) และ volatility ({VOL_WINDOW}D)")
rows = []
for ticker in data.columns:
    prices = data[ticker].dropna()
    if len(prices) < LOOKBACK_DAYS:
        continue
    recent = prices.iloc[-LOOKBACK_DAYS:]
    abs_mom = (recent.iloc[-1] / recent.iloc[0] - 1) * 100

    y = np.log(recent.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0

    rows.append({"Ticker": ticker,
                 "Abs_Momentum_Pct": round(float(abs_mom), 2),
                 "Quant_Score": round(float((np.exp(slope) ** 252 - 1) * r2 * 100), 2)})

df_rank = (pd.DataFrame(rows)
             .sort_values("Quant_Score", ascending=False)
             .reset_index(drop=True))
df_rank["Rank"] = df_rank.index + 1
print(f"      จัดอันดับได้ {len(df_rank)} ตัว")

# ─── 4. Inverse-volatility sizing ─────────────────────────────────────────────
daily_returns = data.pct_change(fill_method=None)
volatility = daily_returns.rolling(window=VOL_WINDOW).std().iloc[-1]
latest_price = data.iloc[-1]


def size_portfolio(tickers_in, label):
    usable = [t for t in tickers_in
              if t in volatility.index and pd.notna(volatility[t]) and volatility[t] > 0
              and t in latest_price.index and pd.notna(latest_price[t])]
    dropped = sorted(set(tickers_in) - set(usable))
    if dropped:
        print(f"      [{label}] ข้าม {len(dropped)} ตัว: {', '.join(dropped)}")

    vol = volatility[usable]
    weights = (1 / vol) / (1 / vol).sum()
    out = pd.DataFrame({"Ticker": usable, "Volatility": vol.values,
                        "Weight": weights.values, "Price": latest_price[usable].values})
    out["Allocated"] = out["Weight"] * PORTFOLIO_VALUE
    out["Shares"] = out["Allocated"] / out["Price"]
    return out.sort_values("Weight", ascending=False).reset_index(drop=True)


rank_lookup = dict(zip(df_rank["Ticker"], df_rank["Rank"]))
df_top = size_portfolio(df_rank["Ticker"].head(TOP_N).tolist(), "top momentum")
df_watch = size_portfolio(MY_WATCHLIST, "watchlist")
df_watch["Rank"] = df_watch["Ticker"].map(rank_lookup)

# ─── 5. สร้างแดชบอร์ด ─────────────────────────────────────────────────────────
print(f"\n[5/5] สร้างแดชบอร์ด")

# The page carries the WHOLE ranked universe so the watchlist stays editable in
# the browser — every name needs a usable volatility and price to be sizeable.
universe_rows = []
for r in df_rank.itertuples():
    t = r.Ticker
    if t not in volatility.index or pd.isna(volatility[t]) or volatility[t] <= 0:
        continue
    if t not in latest_price.index or pd.isna(latest_price[t]) or latest_price[t] <= 0:
        continue
    universe_rows.append([t, int(r.Rank),
                          round(float(r.Abs_Momentum_Pct), 2),
                          round(float(r.Quant_Score), 2),
                          round(float(volatility[t]), 6),
                          round(float(latest_price[t]), 4)])

payload = {
    "meta": {
        "as_of": AS_OF, "lookback": LOOKBACK_DAYS, "vol_window": VOL_WINDOW,
        "top_n": TOP_N, "universe": len(universe_rows),
        "portfolio_value": float(PORTFOLIO_VALUE),
    },
    "universe": universe_rows,
    "watchlist": [t for t in MY_WATCHLIST if any(u[0] == t for u in universe_rows)],
    "index": index_rows,          # [date, open, high, low, close, sma] ต่อวัน
    "sma_window": SMA_WINDOW,
}

html = build_dashboard(payload, standalone=True)
out_path = os.path.abspath(OUTPUT_HTML)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"      เขียนไฟล์ {out_path} ({len(html):,} bytes)")

# ─── สรุปผลบนหน้าจอ ───────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"  TOP {TOP_N} MOMENTUM — ณ {AS_OF}")
print("=" * 68)
print(f"  {'#':>3}  {'TICKER':<7}{'MOM %':>9}{'SCORE':>11}{'WEIGHT':>9}{'SHARES':>10}")
print("  " + "-" * 64)
share_map = dict(zip(df_top["Ticker"], df_top["Shares"]))
weight_map = dict(zip(df_top["Ticker"], df_top["Weight"]))
for r in df_rank.head(TOP_N).itertuples():
    w = weight_map.get(r.Ticker)
    s = share_map.get(r.Ticker)
    print(f"  {r.Rank:>3}  {r.Ticker:<7}{r.Abs_Momentum_Pct:>9.1f}{r.Quant_Score:>11.1f}"
          f"{(f'{w*100:.1f}%' if w else '-'):>9}{(f'{s:.2f}' if s else '-'):>10}")
print("=" * 68)

payload_json = json.dumps(payload, separators=(",", ":"))
with open("momentum-payload.json", "w", encoding="utf-8") as f:
    f.write(payload_json)

if COPY_JSON:
    # json.dumps escapes non-ASCII, so the payload is pure ASCII — no encoding traps.
    blob = payload_json.encode("ascii")
    try:
        if sys.platform == "win32":
            subprocess.run("clip", input=blob, check=True, shell=True)
        elif sys.platform == "darwin":
            subprocess.run("pbcopy", input=blob, check=True)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=blob, check=True)
        print("\n  copy JSON ลง clipboard แล้ว — บอก Claude ว่า 'รันเสร็จแล้ว' ได้เลย")
    except Exception as e:
        print(f"\n  (copy clipboard ไม่สำเร็จ: {e})")
        print("  ส่งไฟล์ momentum-payload.json ให้ Claude แทนได้")

if OPEN_BROWSER:
    import webbrowser
    webbrowser.open(pathlib.Path(out_path).as_uri())
    print("  เปิดแดชบอร์ดในเบราว์เซอร์แล้ว")

print("\nเสร็จเรียบร้อย")
if sys.platform == "win32" and sys.stdin.isatty():
    input("\nกด Enter เพื่อปิดหน้าต่าง...")
