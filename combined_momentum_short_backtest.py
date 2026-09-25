"""Combine the existing 100% Momentum long sleeve with the short overlay."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mean_reversion_short_backtest import Candidate, simulate


PRIMARY = "Momentum 100% + Short overlay"


def equity_metrics(equity: pd.Series) -> dict:
    equity = equity.astype(float).dropna()
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.2425, 1 / 365.2425)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    vol = float(rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 else 0.0
    return {
        "initial_balance": float(equity.iloc[0]),
        "ending_balance": float(equity.iloc[-1]),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "volatility": vol,
        "sharpe": float(rets.mean() / rets.std(ddof=1) * math.sqrt(252))
        if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0,
    }


def yearly_returns(curves: dict[str, pd.Series]) -> pd.DataFrame:
    years = sorted(set().union(*(set(s.index.year) for s in curves.values())))
    out = pd.DataFrame(index=years)
    for name, series in curves.items():
        eoy = series.resample("YE").last()
        values = eoy.pct_change()
        if len(values):
            values.iloc[0] = eoy.iloc[0] / series.iloc[0] - 1
        out[name] = pd.Series(values.to_numpy(), index=eoy.index.year).reindex(years)
    return out


def write_chart(curves: dict[str, pd.Series], output: Path):
    import matplotlib.pyplot as plt

    colors = {"Momentum only": "#2563eb", "Short only": "#d97706", PRIMARY: "#0f766e"}
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, label=name, color=colors[name], linewidth=2.0)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value (USD, log scale)")
    ax.set_title("Momentum long + Mean-Reversion Short overlay", fontsize=16, weight="bold")
    ax.grid(True, which="both", color="#e2e8f0", linewidth=0.8)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, meta: dict, metrics: dict, yearly: pd.DataFrame):
    def pct(value):
        return f"{value * 100:.2f}%"

    rows = []
    for name, item in metrics.items():
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>${item['ending_balance']:,.0f}</td>"
            f"<td>{pct(item['cagr'])}</td><td>{pct(item['max_drawdown'])}</td>"
            f"<td>{pct(item['volatility'])}</td><td>{item['sharpe']:.2f}</td></tr>"
        )
    primary = metrics[PRIMARY]
    yearly_html = yearly.to_html(float_format=lambda x: f"{x*100:.1f}%", border=0)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Momentum + Short Overlay Backtest</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}main{{max-width:1100px;margin:auto;padding:28px}}.card{{background:white;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 1px 4px #cbd5e1}}.warn{{border-left:5px solid #f97316;background:#fff7ed}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.kpi{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:14px}}.kpi b{{display:block;font-size:20px}}.kpi span,.small{{font-size:12px;color:#64748b}}@media(max-width:700px){{main{{padding:14px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.scroll{{overflow:auto}}table{{min-width:700px}}}}
</style></head><body><main><div class="card"><h1>Momentum + Short Overlay</h1><p>{meta['start']} to {meta['end']} · initial equity ${meta['initial_balance']:,.0f}</p></div>
<div class="card warn"><strong>Research limitations:</strong> The Momentum sleeve uses point-in-time S&amp;P 500 membership but still misses some delisted-price histories. The Short sleeve uses today's active-stock universe and therefore has survivorship bias. Historical borrow availability is unavailable.</div>
<div class="card"><h2>Results</h2><div class="scroll"><table><thead><tr><th>Portfolio</th><th>Ending</th><th>CAGR</th><th>Max DD</th><th>Volatility</th><th>Sharpe</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<div class="card"><div class="kpis"><div class="kpi"><b>${primary['short_commissions']:,.0f}</b><span>Short commissions</span></div><div class="kpi"><b>${primary['short_margin_interest']:,.0f}</b><span>Short margin interest</span></div><div class="kpi"><b>${primary['short_borrow_cost']:,.0f}</b><span>Short borrow cost</span></div><div class="kpi"><b>{primary['avg_total_gross']*100:.1f}%</b><span>Average combined gross</span></div></div></div>
<div class="card"><img src="combined-momentum-short-equity.png" alt="Combined equity curves"></div>
<div class="card"><h2>Yearly returns</h2><div class="scroll">{yearly_html}</div></div>
<div class="card small"><h2>Method</h2><p>Long sleeve: existing Momentum Desk backtest rules, 100% of combined equity, including its 20 bps turnover cost. Short sleeve: up to 100% short gross at entry, IBKR Tiered commission, user-defined margin interest of 5.13% on the first $100,000 and 4.63% above it, plus 0.25% annual stock-borrow cost. Long exposure is reset to 100% of combined equity daily for sleeve accounting; short positions follow the original entry/exit rules. Combined gross can reach 200% at entry and can drift slightly above it after adverse short moves.</p><pre>{html.escape(json.dumps(meta, indent=2))}</pre></div>
</main></body></html>'''
    path.write_text(document, encoding="utf-8")


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    momentum_payload = json.loads(Path(args.momentum).read_text(encoding="utf-8"))
    runs = momentum_payload.get("runs") or []
    point_in_time = next((run for run in runs if run.get("key") == "pit"), runs[0] if runs else None)
    if not point_in_time:
        raise RuntimeError("Momentum payload has no point-in-time run")
    dates = point_in_time["dates"]
    values = point_in_time["net"]
    if not momentum_payload.get("meta", {}).get("full_daily"):
        raise RuntimeError("Momentum payload is thinned; rerun with MOMENTUM_FULL_DAILY=1")
    momentum = pd.Series(values, index=pd.to_datetime(dates), dtype=float).sort_index()

    with gzip.open(args.short_orders, "rt", encoding="utf-8") as handle:
        cached = json.load(handle)
    orders = [Candidate(**row) for row in cached["orders"]]
    short_days = cached["trading_days"]
    start = max(pd.Timestamp(args.start), momentum.index[0])
    common_days = [day for day in short_days if pd.Timestamp(day) >= start and pd.Timestamp(day) in momentum.index]
    if len(common_days) < 250:
        raise RuntimeError("Too few overlapping daily observations")

    long_index = pd.to_datetime(common_days)
    long_equity = momentum.reindex(long_index)
    if long_equity.isna().any():
        raise RuntimeError("Momentum daily curve has gaps on Alpaca trading sessions")
    long_equity = 100_000.0 * long_equity / long_equity.iloc[0]
    long_returns = long_equity.pct_change().fillna(0.0)
    long_returns.index = [str(day.date()) for day in long_returns.index]

    short_curve, short_trades, short_metrics = simulate(
        orders, common_days, 0.0, ibkr_tiered=True,
        ibkr_margin_interest=True, annual_borrow_rate=0.0025,
    )
    combined_curve, combined_trades, combined_metrics = simulate(
        orders, common_days, 0.0, ibkr_tiered=True,
        ibkr_margin_interest=True, annual_borrow_rate=0.0025,
        external_daily_returns=long_returns, external_label="Momentum 100% long",
    )
    short_equity = short_curve["equity"].copy()
    short_equity.index = pd.to_datetime(short_equity.index)
    combined_equity = combined_curve["equity"].copy()
    combined_equity.index = pd.to_datetime(combined_equity.index)
    curves = {"Momentum only": long_equity, "Short only": short_equity, PRIMARY: combined_equity}

    results = {
        "Momentum only": equity_metrics(long_equity),
        "Short only": equity_metrics(short_equity),
        PRIMARY: equity_metrics(combined_equity),
    }
    results["Short only"].update({
        "short_commissions": short_metrics["total_commissions"],
        "short_margin_interest": short_metrics["total_margin_interest"],
        "short_borrow_cost": short_metrics["total_borrow_cost"],
        "avg_short_gross": short_metrics["avg_gross_exposure"],
        "max_short_gross": short_metrics["max_gross_exposure"],
        "trades": short_metrics["trades"],
    })
    results[PRIMARY].update({
        "short_commissions": combined_metrics["total_commissions"],
        "short_margin_interest": combined_metrics["total_margin_interest"],
        "short_borrow_cost": combined_metrics["total_borrow_cost"],
        "avg_short_gross": combined_metrics["avg_gross_exposure"],
        "max_short_gross": combined_metrics["max_gross_exposure"],
        "avg_total_gross": 1.0 + combined_metrics["avg_gross_exposure"],
        "max_total_gross": 1.0 + combined_metrics["max_gross_exposure"],
        "trades": combined_metrics["trades"],
        "long_pnl_contribution": combined_metrics["total_external_pnl"],
        "short_pnl_contribution": float(combined_trades["net_pnl"].sum()),
    })
    reconciliation = (
        100_000.0 + results[PRIMARY]["long_pnl_contribution"]
        + results[PRIMARY]["short_pnl_contribution"]
    )
    if not math.isclose(reconciliation, combined_equity.iloc[-1], rel_tol=0, abs_tol=0.01):
        raise RuntimeError("Combined P&L does not reconcile")

    meta = {
        "start": str(long_index[0].date()), "end": str(long_index[-1].date()),
        "initial_balance": 100_000.0,
        "momentum_rules": momentum_payload["meta"],
        "short_cost_model": "IBKR Tiered + user-defined margin tiers + 0.25% borrow",
        "long_sleeve": "100% of combined equity",
        "short_overlay_cap": "100% of combined equity at entry",
        "daily_long_sleeve_reset": True,
        "combined_pnl_reconciled": True,
    }
    yearly = yearly_returns(curves)
    write_chart(curves, output / "combined-momentum-short-equity.png")
    write_report(output / "combined-momentum-short-report.html", meta, results, yearly)
    yearly.to_csv(output / "combined-momentum-short-yearly.csv")
    pd.DataFrame({name: curve.reindex(long_index).values for name, curve in curves.items()},
                 index=long_index).to_csv(output / "combined-momentum-short-curves.csv")
    combined_trades.to_csv(output / "combined-momentum-short-trades.csv", index=False)
    rendered = json.dumps({"meta": meta, "scenarios": results}, indent=2, allow_nan=False)
    (output / "combined-momentum-short-results.json").write_text(rendered, encoding="utf-8")
    print(rendered)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--momentum", default="backtest-payload.json")
    parser.add_argument("--short-orders", default="short-backtest-output/mean-reversion-short-orders.json.gz")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--output", default="combined-backtest-output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
