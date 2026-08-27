"""Send the daily digest to Telegram.

Reads momentum-payload.json, posts a short summary and attaches the dashboard.
Credentials come from the environment — nothing is stored in this file.

  TELEGRAM_BOT_TOKEN        from @BotFather
  TELEGRAM_CHAT_ID          your own chat id
  NOTIFY_PORTFOLIO_VALUE    optional: report share counts against THIS amount
                            instead of whatever the page was built with

The last one exists so a public dashboard can be built at a round $100,000
while the message on your phone still uses your real balance. Weights are
identical either way, so the rescale is exact — no second download needed.
"""

import json
import os
import pathlib
import sys

import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
PV_OVERRIDE = os.environ.get("NOTIFY_PORTFOLIO_VALUE", "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"

PAYLOAD = pathlib.Path("momentum-payload.json")
DASHBOARD = pathlib.Path("momentum-desk.html")


def esc(s):
    """Escape for Telegram HTML parse mode."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_message(p):
    meta, uni, wl = p["meta"], p["universe"], p["watchlist"]
    by = {r[0]: r for r in uni}
    ranked = sorted(uni, key=lambda r: r[1])

    lines = [f"<b>Momentum Desk</b> · {esc(meta['as_of'])}",
             f"<i>{len(uni)} names ranked</i>", ""]

    # the regime line first: it is the one thing that changes what you do
    idx = p.get("index") or []
    if idx and len(idx[-1]) > 5 and idx[-1][5]:
        close, sma = idx[-1][4], idx[-1][5]
        gap = (close / sma - 1) * 100
        state = "ABOVE" if gap >= 0 else "BELOW"
        lines += [f"S&amp;P 500 <b>{state}</b> its {p.get('sma_window', 200)}-day average "
                  f"({gap:+.1f}%)",
                  f"close {close:,.2f} · avg {sma:,.2f}", ""]

    lines.append("<b>Top 10 by momentum</b>")
    lines.append("<pre>")
    lines.append(f"{'#':>2}  {'TICKER':<6}{'125D':>8}{'SCORE':>9}")
    for r in ranked[:10]:
        lines.append(f"{r[1]:>2}  {r[0]:<6}{r[2]:>7.1f}%{r[3]:>9.0f}")
    lines.append("</pre>")

    # watchlist sizing, recomputed the same way the page does it
    usable = [t for t in wl if t in by and by[t][4] > 0 and by[t][5] > 0]
    if usable:
        inv = [1 / by[t][4] for t in usable]
        tot = sum(inv)
        pv = meta["portfolio_value"]
        if PV_OVERRIDE:
            try:
                pv = float(PV_OVERRIDE)
            except ValueError:
                pass
        rows = sorted(((t, (1 / by[t][4]) / tot) for t in usable),
                      key=lambda x: -x[1])
        tag = " (your balance)" if PV_OVERRIDE else ""
        lines += ["", f"<b>Watchlist</b> · ${pv:,.0f}{tag} across {len(rows)}", "<pre>",
                  f"{'TICKER':<6}{'WT':>7}{'SHARES':>9}"]
        for t, w in rows[:12]:
            lines.append(f"{t:<6}{w*100:>6.1f}%{w*pv/by[t][5]:>9.2f}")
        if len(rows) > 12:
            lines.append(f"... {len(rows)-12} more")
        lines.append("</pre>")

    return "\n".join(lines)


def main():
    if not TOKEN or not CHAT:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping notification")
        return 0
    if not PAYLOAD.exists():
        print(f"{PAYLOAD} missing — nothing to send")
        return 1

    p = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    text = build_message(p)

    r = requests.post(f"{API}/sendMessage", timeout=30, data={
        "chat_id": CHAT, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": "true"})
    if not r.ok:
        # never print the response verbatim; it can echo the token back
        print(f"sendMessage failed: HTTP {r.status_code}")
        try:
            print("  description:", r.json().get("description", "")[:200])
        except Exception:
            pass
        return 1
    print("summary sent")

    if DASHBOARD.exists():
        with DASHBOARD.open("rb") as fh:
            r2 = requests.post(f"{API}/sendDocument", timeout=120,
                               data={"chat_id": CHAT,
                                     "caption": f"Dashboard · {p['meta']['as_of']}"},
                               files={"document": (DASHBOARD.name, fh, "text/html")})
        if r2.ok:
            print("dashboard attached")
        else:
            print(f"sendDocument failed: HTTP {r2.status_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
