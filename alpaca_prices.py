"""Read-only Alpaca SIP prices and explicit per-symbol data quality checks."""
import json
import math
import os
import time
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
# Preserve the dashboard's existing Yahoo-style share-class symbols.
SYMBOL_MAP = {"BRK-B": "BRK.B", "BF-B": "BF.B"}


def completed_sessions(now=None):
    """Use the exchange calendar, including holidays, DST and early closes.

    Wait 30 minutes after the scheduled close, then request only completed
    sessions. This also stays outside the free SIP plan's 15-minute embargo.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must have a timezone")
    now = now.tz_convert("UTC")
    day = now.tz_convert("America/New_York").date()
    start = pd.Timestamp(day) - pd.Timedelta(days=400)
    cal = xcals.get_calendar("XNYS", start=start, end=str(day))
    schedule = cal.schedule.loc[str(start.date()):str(day)]
    ready = schedule[schedule["close"] <= now - pd.Timedelta(minutes=30)]
    if ready.empty:
        raise RuntimeError("No completed US trading session in calendar")
    sessions = pd.DatetimeIndex(ready.index).tz_localize(None)
    last = sessions[-1]
    # Do not include the following day's midnight (end is inclusive).
    end = ((last + pd.Timedelta(days=1)).tz_localize("America/New_York")
           - pd.Timedelta(seconds=1))
    end = min(end.tz_convert("UTC"), now - pd.Timedelta(minutes=20))
    return sessions, end


class AlpacaPrices:
    def __init__(self, api_key=None, secret_key=None, session=None, sleep=time.sleep):
        key = api_key or os.environ.get("ALPACA_API_KEY", "").strip()
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY", "").strip()
        if not key or not secret:
            raise RuntimeError("Set both ALPACA_API_KEY and ALPACA_SECRET_KEY in GitHub Actions secrets")
        self.session = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.sleep = sleep

    def _get(self, params):
        for attempt in range(4):
            try:
                r = self.session.get(DATA_URL, headers=self.headers, params=params,
                                     timeout=(10, 60), allow_redirects=False)
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError("Alpaca request failed after retries") from None
                self.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    raise RuntimeError("Alpaca returned invalid JSON") from None
                if not isinstance(data, dict) or not isinstance(data.get("bars"), dict):
                    raise RuntimeError("Alpaca returned an invalid bars response")
                return data
            if r.status_code in (401, 403):
                raise RuntimeError("Alpaca authentication/SIP access failed; check the two repository secrets")
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"Alpaca bars request rejected (HTTP {r.status_code})")
            if attempt < 3:
                self.sleep(30 if r.status_code == 429 else 2 ** attempt)
        raise RuntimeError("Alpaca unavailable or rate limited after retries")

    def closes(self, tickers, start, end, adjustment="all"):
        """Fetch every page; a page limit covers all symbols, not each one."""
        tickers = list(dict.fromkeys(tickers))
        mapping = {SYMBOL_MAP.get(t, t): t for t in tickers}
        records = []
        for offset in range(0, len(tickers), 100):
            batch = tickers[offset:offset + 100]
            params = {"symbols": ",".join(SYMBOL_MAP.get(t, t) for t in batch),
                      "timeframe": "1Day", "start": pd.Timestamp(start).isoformat(),
                      "end": pd.Timestamp(end).isoformat(), "adjustment": adjustment,
                      "feed": "sip", "limit": 10000, "sort": "asc"}
            seen_tokens = set()
            while True:
                result = self._get(params)
                for symbol, bars in result["bars"].items():
                    if symbol not in mapping:
                        raise RuntimeError("Alpaca returned an unexpected symbol")
                    for bar in bars or []:
                        ts = pd.Timestamp(bar["t"])
                        if ts.tzinfo is None:
                            raise RuntimeError("Alpaca bar timestamp has no timezone")
                        date = ts.tz_convert("America/New_York").tz_localize(None).normalize()
                        close = float(bar["c"]) if bar.get("c") is not None else np.nan
                        if not math.isfinite(close) or close <= 0:
                            close = np.nan
                        records.append((date, mapping[symbol], close))
                token = result.get("next_page_token")
                if not token:
                    break
                if token in seen_tokens:
                    raise RuntimeError("Alpaca repeated a pagination token")
                seen_tokens.add(token)
                params["page_token"] = token
        if not records:
            return pd.DataFrame(columns=tickers, dtype=float)
        rows = pd.DataFrame(records, columns=["date", "ticker", "close"])
        if rows.duplicated(["date", "ticker"]).any():
            raise RuntimeError("Duplicate Alpaca daily bars; refusing an ambiguous snapshot")
        return rows.pivot(index="date", columns="ticker", values="close").reindex(columns=tickers).sort_index()


def assess_prices(prices, raw_latest, sessions, tickers, lookback):
    """Never invent prices. Short history and internal holes are distinct."""
    data = prices.reindex(index=sessions, columns=tickers)
    data = data.where(np.isfinite(data) & (data > 0))
    raw_latest = raw_latest.reindex(tickers)
    raw_latest = raw_latest.where(np.isfinite(raw_latest) & (raw_latest > 0))
    window = data.tail(lookback)
    details, eligible = [], []
    for ticker in tickers:
        col = data[ticker]
        valid = col.dropna()
        first, last = col.first_valid_index(), col.last_valid_index()
        active_missing = col.loc[first:].index[col.loc[first:].isna()] if first is not None else []
        reasons = []
        if valid.empty:
            reasons.append("no_data")
        elif first > window.index[0] or len(window) < lookback:
            reasons.append("insufficient_history")
        elif window[ticker].isna().any():
            reasons.append("missing_in_lookback")
        if pd.isna(col.iloc[-1]) or pd.isna(raw_latest[ticker]):
            reasons.append("missing_latest")
        if not reasons:
            eligible.append(ticker)
        details.append({"ticker": ticker, "eligible": not reasons, "reasons": reasons,
                        "first_date": None if first is None else str(first.date()),
                        "last_date": None if last is None else str(last.date()),
                        "available_bars": len(valid),
                        "missing_dates": [str(d.date()) for d in active_missing],
                        "missing_in_lookback": [str(d.date()) for d in window.index[window[ticker].isna()]],
                        "raw_close": None if pd.isna(raw_latest[ticker]) else float(raw_latest[ticker])})
    fresh = int((data.iloc[-1].notna() & raw_latest.notna()).sum())
    quality = {"source": "Alpaca", "feed": "sip", "adjustment": "all",
               "as_of": str(sessions[-1].date()), "requested": len(tickers),
               "fresh": fresh, "eligible": len(eligible), "filled_cells": 0,
               "symbols": details}
    return data, eligible, quality


def load_ranking_prices(tickers, lookback=125, report_path="momentum-data-quality.json",
                        client=None, now=None, log=print):
    sessions, end = completed_sessions(now)
    start = sessions[0].tz_localize("America/New_York").tz_convert("UTC")
    latest_start = sessions[-1].tz_localize("America/New_York").tz_convert("UTC")
    client = client or AlpacaPrices()
    prices = client.closes(tickers, start, end)
    # One retry of incomplete histories. Pre-listing dates are not data holes.
    window = prices.reindex(index=sessions, columns=tickers).tail(lookback)
    retry = []
    for t in tickers:
        col = window[t]
        first = col.first_valid_index()
        if first is None or pd.isna(col.iloc[-1]) or col.loc[first:].isna().any():
            retry.append(t)
    if retry:
        log(f"      Retry incomplete histories: {', '.join(retry)}")
        again = client.closes(retry, start, end)
        # Replace the entire series to avoid mixing adjustment vintages.
        prices = prices.reindex(index=sessions, columns=tickers)
        for t in retry:
            prices[t] = again[t].reindex(sessions)
    raw = client.closes(tickers, latest_start, end, adjustment="raw")
    raw_latest = raw.reindex(index=[sessions[-1]], columns=tickers).iloc[0]
    missing_raw = raw_latest.index[raw_latest.isna()].tolist()
    if missing_raw:
        again = client.closes(missing_raw, latest_start, end, adjustment="raw")
        raw_latest.loc[missing_raw] = again.reindex(index=[sessions[-1]], columns=missing_raw).iloc[0]
    data, eligible, quality = assess_prices(prices, raw_latest, sessions, tickers, lookback)
    Path(report_path).write_text(json.dumps(quality, indent=2, allow_nan=False), encoding="utf-8")
    log(f"      Alpaca SIP {quality['as_of']}: {quality['fresh']}/{len(tickers)} fresh; {len(eligible)} eligible; no filled prices")
    for item in quality["symbols"]:
        if item["reasons"] or item["missing_dates"]:
            log(f"      {item['ticker']}: {', '.join(item['reasons']) or 'older gaps outside lookback'}; last={item['last_date']}; missing={','.join(item['missing_dates'])}")
    if quality["fresh"] < math.ceil(0.95 * len(tickers)):
        raise RuntimeError("Less than 95% of stocks have the expected session price; refusing stale/partial publication")
    return data, raw_latest, eligible, quality
