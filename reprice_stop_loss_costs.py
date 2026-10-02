"""Reprice cached stop-loss-only orders with explicit slippage and borrow costs."""
import argparse
import gzip
import json
from pathlib import Path
import mean_reversion_short_backtest as bt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="repriced-stop-loss-output")
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--borrow-rate", type=float, default=0.05)
    args = parser.parse_args()

    with gzip.open(args.cache, "rt", encoding="utf-8") as handle:
        cached = json.load(handle)
    orders = [bt.Candidate(**row) for row in cached["orders"]]
    trading_days = cached["trading_days"]
    original_commission = bt._trade_commission

    def commission_plus_slippage(shares, price, cost_bps_side, ibkr_tiered):
        ibkr = original_commission(shares, price, 0.0, ibkr_tiered)
        slippage = shares * price * cost_bps_side / 10_000.0
        return ibkr + slippage

    bt._trade_commission = commission_plus_slippage
    curve, trades, metrics = bt.simulate(
        orders, trading_days, args.slippage_bps,
        ibkr_tiered=True, ibkr_margin_interest=True,
        annual_borrow_rate=args.borrow_rate,
    )
    ibkr_commissions = sum(
        original_commission(float(row.shares), float(row.entry_price), 0.0, True)
        + original_commission(float(row.shares), float(row.exit_price), 0.0, True)
        for row in trades.itertuples(index=False)
    )
    slippage = float(
        ((trades["entry_notional"] + trades["shares"] * trades["exit_price"])
         * args.slippage_bps / 10_000.0).sum()
    )
    metrics["total_ibkr_commissions"] = float(ibkr_commissions)
    metrics["total_slippage"] = slippage
    metrics["total_transaction_costs"] = float(metrics.pop("total_commissions"))
    result = {
        "assumptions": {
            "exit": "2.5x ATR stop on day 2; otherwise day-2 close",
            "slippage_bps_per_side": args.slippage_bps,
            "annual_borrow_rate": args.borrow_rate,
            "commission": "IBKR Pro Tiered, in addition to slippage",
            "margin_interest": "5.13% first $100k, 4.63% above; actual/360",
        },
        "metrics": metrics,
    }
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "stop-loss-slippage-borrow-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    curve.to_csv(outdir / "stop-loss-slippage-borrow-equity.csv")
    trades.to_csv(outdir / "stop-loss-slippage-borrow-trades.csv", index=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
