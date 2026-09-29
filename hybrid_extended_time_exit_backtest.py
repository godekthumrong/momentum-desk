"""Compare the original hybrid exit with a one-session-longer hybrid exit."""

from dataclasses import replace
from pathlib import Path
import gzip
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mean_reversion_short_backtest as bt


_original_reprice = bt.reprice_candidate_exits
_original_hybrid_candidates = []


def extend_time_exit(candidate, frame):
    """Keep the OCA stop/target active for day 3, then exit at day-3 close."""
    if not candidate.filled or candidate.exit_reason != "time_exit":
        return candidate
    day2_index = frame.index.get_indexer([pd.Timestamp(candidate.exit_date)])[0]
    if day2_index < 0 or day2_index + 1 >= len(frame):
        return replace(candidate, exit_reason="end_of_data")

    day3 = frame.iloc[day2_index + 1]
    day3_date = frame.index[day2_index + 1].strftime("%Y-%m-%d")
    stop = float(candidate.entry_price + 2.5 * candidate.atr10)
    target = float(candidate.entry_price * 0.96)
    open_price = float(day3["open"])

    # Opening gaps have a known position before the daily high/low.
    if open_price >= stop:
        price, reason, timing = open_price, "stop_loss", "open"
    elif open_price <= target:
        price, reason, timing = open_price, "profit_target", "open"
    # If both levels occur inside one daily bar, use stop first conservatively.
    elif float(day3["high"]) >= stop:
        price, reason, timing = stop, "stop_loss", "intraday"
    elif float(day3["low"]) <= target:
        price, reason, timing = target, "profit_target", "intraday"
    else:
        price, reason, timing = float(day3["close"]), "time_exit", "close"

    return replace(
        candidate,
        exit_date=day3_date,
        exit_price=float(price),
        exit_reason=reason,
        exit_timing=timing,
        exit_open=open_price,
    )


def extended_reprice(candidates, frame, exit_rule):
    old = _original_reprice(candidates, frame, exit_rule)
    if exit_rule != "hybrid_close_and_limit":
        return old
    _original_hybrid_candidates.extend(old)
    return [extend_time_exit(candidate, frame) for candidate in old]


bt.reprice_candidate_exits = extended_reprice


def write_chart(outdir, old_curve, new_curve):
    dates = pd.to_datetime(old_curve.index)
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.plot(dates, np.log(old_curve["equity"] / 100_000), linewidth=2,
            label="Original hybrid: time exit D2 close")
    ax.plot(dates, np.log(new_curve["equity"] / 100_000), linewidth=2,
            label="Extended hybrid: time exit D3 close")
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.grid(True, color="#e2e8f0", alpha=0.8)
    ax.set_title("Hybrid exit comparison — cumulative log return")
    ax.set_ylabel("log(equity / 100,000)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "hybrid-extended-time-exit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = bt.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    bt.run(args)

    if not _original_hybrid_candidates:
        raise RuntimeError("Original hybrid candidates were not captured")
    old_orders, old_signal_count = bt.select_top_orders(_original_hybrid_candidates)
    with gzip.open(outdir / "profit-target-hybrid-orders.json.gz", "rt") as handle:
        cache = json.load(handle)
    days = cache["trading_days"]
    new_orders = [bt.Candidate(**row) for row in cache["orders"]]

    old_identity = [(c.symbol, c.signal_date, c.order_date, c.filled) for c in old_orders]
    new_identity = [(c.symbol, c.signal_date, c.order_date, c.filled) for c in new_orders]
    if old_identity != new_identity:
        raise RuntimeError("Old and extended rules do not use identical entries")

    costs = dict(
        ibkr_tiered=True,
        ibkr_margin_interest=True,
        annual_borrow_rate=0.0025,
        short_gross_cap=1.0,
    )
    old_curve, old_trades, old_metrics = bt.simulate(old_orders, days, 0.0, **costs)
    new_curve, new_trades, new_metrics = bt.simulate(new_orders, days, 0.0, **costs)

    comparison = pd.DataFrame({
        "original_hybrid_equity": old_curve["equity"],
        "extended_hybrid_equity": new_curve["equity"],
    })
    comparison.to_csv(outdir / "hybrid-extended-time-exit-equity.csv")
    old_trades.to_csv(outdir / "hybrid-original-trades.csv", index=False)
    new_trades.to_csv(outdir / "hybrid-extended-trades.csv", index=False)

    result = {
        "assumptions": {
            "original": "hybrid stop/target on D2; if neither fills, time exit at D2 close",
            "extended": "hybrid stop/target on D2 and D3; if neither fills, time exit at D3 close",
            "entry_close_profit_rule": "if D1 close <= 96% of fill, cover at D2 actual open",
            "same_bar_both_levels": "stop first",
            "signals_entries_ranking_sizing_costs": "identical",
            "selected_orders": len(new_orders),
            "filled_selected_orders": sum(c.filled for c in new_orders),
            "raw_qualifying_signals": old_signal_count,
        },
        "original_hybrid": old_metrics,
        "extended_hybrid": new_metrics,
        "delta_extended_minus_original": {
            key: new_metrics[key] - old_metrics[key]
            for key in ("ending_balance", "cagr", "max_drawdown", "sharpe",
                        "trades", "win_rate", "avg_trade_return")
        },
    }
    (outdir / "hybrid-extended-time-exit-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_chart(outdir, old_curve, new_curve)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
