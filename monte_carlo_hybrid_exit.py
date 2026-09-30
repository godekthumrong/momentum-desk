"""Monte Carlo: unranked 50% short entries with hybrid stop/target exits."""

from collections import defaultdict
from pathlib import Path
import gzip
import inspect
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mean_reversion_short_backtest as bt


def all_orders(candidates):
    return sorted(candidates, key=lambda c: (c.order_date, c.symbol)), len(candidates)


bt.select_top_orders = all_orders


def capped_simulator():
    """Add a max-position admission check to an isolated copy of bt.simulate."""
    original = inspect.getsource(bt.simulate)
    signature = "short_gross_cap: float = 1.0)"
    admission = """            if not c.filled:
                continue
            if c.symbol in positions:
"""
    capped = """            if not c.filled:
                continue
            if max_positions is not None and len(positions) >= max_positions:
                blocked_exposure += 1
                continue
            if c.symbol in positions:
"""
    if signature not in original or admission not in original:
        raise RuntimeError("Backtest simulator structure changed; cap patch not applied")
    source = original.replace(
        signature,
        "short_gross_cap: float = 1.0, max_positions: int | None = None)",
        1,
    ).replace(admission, capped, 1)
    namespace = vars(bt).copy()
    exec(source, namespace)
    return namespace["simulate"]


simulate = capped_simulator()


def percentiles(frame, columns):
    labels = {5: "p05", 25: "p25", 50: "p50", 75: "p75", 95: "p95"}
    return {
        col: {labels[p]: float(np.nanpercentile(frame[col], p)) for p in labels}
        for col in columns
    }


def write_chart(outdir, days, curves, metrics, reference=None):
    dates = pd.to_datetime(days)
    q05, q50, q95 = np.percentile(curves, [5, 50, 95], axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax in (ax1, ax2):
        ax.grid(True, color="#e2e8f0", alpha=0.8)
    ax1.fill_between(dates, np.log(q05 / 100_000), np.log(q95 / 100_000),
                     color="#99f6e4", alpha=0.5, label="Hybrid P05-P95")
    ax1.plot(dates, np.log(q50 / 100_000), color="#0f766e", lw=2,
             label="Hybrid median")
    if reference is not None:
        ref_equity = pd.read_csv(reference, parse_dates=["date"])
        ax1.plot(ref_equity["date"], np.log(ref_equity["equity_p50"] / 100_000),
                 color="#f97316", lw=1.8, label="Time-exit-only median")
    ax1.axhline(0, color="#64748b", lw=1)
    ax1.set(title="Cumulative log return", ylabel="log(equity / 100,000)")
    ax1.legend(frameon=False)
    ax2.hist(metrics["ending_balance"], bins=35, color="#0f766e", alpha=0.84)
    ax2.axvline(metrics["ending_balance"].median(), color="#f97316", lw=2,
                label="Hybrid median")
    ax2.axvline(100_000, color="#dc2626", lw=1.5, ls="--", label="Initial")
    ax2.set(title="Ending balance distribution", xlabel="USD", ylabel="Runs")
    ax2.legend(frameon=False)
    fig.suptitle("50% random entries - no ranking - hybrid stop/target - max 10 shorts")
    fig.tight_layout()
    fig.savefig(outdir / "monte-carlo-hybrid-summary.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def main():
    args = bt.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_override = os.environ.get("MC_CACHE_PATH")
    if cache_override:
        cache_path = Path(cache_override)
    else:
        bt.run(args)
        cache_path = outdir / "profit-target-hybrid-orders.json.gz"

    with gzip.open(cache_path, "rt") as handle:
        cache = json.load(handle)
    days = cache["trading_days"]
    candidates = [bt.Candidate(**row) for row in cache["orders"]]
    eligible = [c for c in candidates if c.filled]
    by_day = defaultdict(list)
    for candidate in eligible:
        by_day[candidate.order_date].append(candidate)

    runs = int(os.environ.get("MONTE_CARLO_RUNS", "1000"))
    probability = float(os.environ.get("MC_ENTRY_PROBABILITY", "0.5"))
    max_positions = int(os.environ.get("MC_MAX_POSITIONS", "10"))
    if not 0 < probability <= 1 or runs < 1 or max_positions < 1:
        raise ValueError("Invalid Monte Carlo parameters")

    curves = np.empty((runs, len(days)), dtype=np.float32)
    rows = []
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        sampled = []
        for day in days:
            pool = by_day.get(day, ())
            if not pool:
                continue
            chosen = np.flatnonzero(rng.random(len(pool)) < probability)
            rng.shuffle(chosen)
            sampled.extend(pool[int(i)] for i in chosen)
        curve, _, metrics = simulate(
            sampled, days, 0.0, ibkr_tiered=True, ibkr_margin_interest=True,
            annual_borrow_rate=0.0025, short_gross_cap=1.0,
            max_positions=max_positions,
        )
        curves[seed] = curve["equity"].to_numpy()
        rows.append({"seed": seed, "entries_drawn": len(sampled), **metrics})
        if (seed + 1) % 100 == 0 or seed + 1 == runs:
            print(f"Monte Carlo hybrid {seed + 1:,}/{runs:,}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "monte-carlo-hybrid-runs.csv", index=False)
    equity = pd.DataFrame({"date": days})
    for p in (5, 25, 50, 75, 95):
        equity[f"equity_p{p:02d}"] = np.percentile(curves, p, axis=0)
    equity.to_csv(outdir / "monte-carlo-hybrid-equity-percentiles.csv", index=False)

    cols = ["ending_balance", "cagr", "max_drawdown", "sharpe", "trades",
            "win_rate", "profit_factor", "avg_trade_return", "entries_drawn",
            "blocked_by_exposure", "blocked_duplicate_symbol"]
    summary = {
        "assumptions": {
            "runs": runs, "entry_probability": probability, "ranking": "none",
            "same_day_admission": "random order",
            "eligible_pool": "all qualifying signals whose short limit entry filled",
            "exit": {
                "entry_day": "no stop; if close <= 96% of fill, cover next open",
                "day_after_entry": "stop at entry + 2.5 ATR; otherwise 4% buy limit; stop first if both touch",
                "time_exit": "day-after-entry close if neither stop nor target fills",
                "gap_fill": "actual open",
            },
            "max_simultaneous_short_positions": max_positions,
            "short_gross_cap": 1.0,
            "position_sizing": "2% ATR risk; max 10% equity per position; whole shares",
            "commission": "IBKR Tiered $0.0035/share; $0.35 minimum; 1% cap",
            "margin_interest": "5.13% first $100k; 4.63% above; actual/360",
            "annual_borrow_rate": 0.0025,
            "survivorship_bias": True, "historical_short_availability": False,
        },
        "population": {
            "raw_qualifying_signals": len(candidates),
            "filled_eligible_signals": len(eligible),
            "trading_days": len(days), "start": days[0], "end": days[-1],
        },
        "percentiles": percentiles(metrics, cols),
        "probabilities": {
            "ending_below_initial": float((metrics.ending_balance < 100_000).mean()),
            "positive_cagr": float((metrics.cagr > 0).mean()),
            "max_drawdown_worse_than_minus_30pct": float((metrics.max_drawdown < -0.30).mean()),
            "ending_above_original_hybrid_reference_638922": float((metrics.ending_balance > 638_921.7764213928).mean()),
            "ending_above_ranked_time_exit_reference_1318155": float((metrics.ending_balance > 1_318_155.374386385).mean()),
        },
    }

    prior_summary_path = os.environ.get("MC_TIME_EXIT_SUMMARY")
    if prior_summary_path:
        prior = json.loads(Path(prior_summary_path).read_text())
        summary["comparison_to_prior_time_exit_only"] = {
            metric: {
                "hybrid_p50": summary["percentiles"][metric]["p50"],
                "time_exit_only_p50": prior["percentiles"][metric]["p50"],
                "difference": summary["percentiles"][metric]["p50"] - prior["percentiles"][metric]["p50"],
            }
            for metric in ("ending_balance", "cagr", "max_drawdown", "sharpe",
                           "win_rate", "profit_factor")
        }

    (outdir / "monte-carlo-hybrid-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    prior_equity = os.environ.get("MC_TIME_EXIT_EQUITY")
    write_chart(outdir, days, curves, metrics,
                Path(prior_equity) if prior_equity else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
