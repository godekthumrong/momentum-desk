"""Isolated comparison: original short exits vs stop loss plus day-2 close."""
from pathlib import Path
import json

import pandas as pd

import mean_reversion_short_backtest as bt


_original_resolve_exit = bt.resolve_exit


def stop_loss_comparison(entry_price, entry_close, atr, day2, exit_rule):
    """Remove every profit exit while retaining the day-2 stop and time exit."""
    if exit_rule != "hybrid_close_and_limit":
        return _original_resolve_exit(entry_price, entry_close, atr, day2, exit_rule)

    stop = float(entry_price + 2.5 * atr)
    open_price = float(day2["open"])
    if open_price >= stop:
        return open_price, "stop_loss", "open"
    if float(day2["high"]) >= stop:
        return stop, "stop_loss", "intraday"
    return float(day2["close"]), "time_exit", "close"


bt.resolve_exit = stop_loss_comparison


if __name__ == "__main__":
    args = bt.parse_args()
    bt.run(args)
    outdir = Path(args.output)

    source = outdir / "profit-target-comparison-results.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    result["assumptions"]["new"] = (
        "no take profit; on day 2 cover at the open if it gaps above the "
        "2.5x ATR stop, otherwise at the stop if the high touches it, "
        "otherwise at the day-2 close"
    )
    result["assumptions"].pop("same_day_both_levels", None)
    result["assumptions"].pop("gap_fill", None)
    (outdir / "stop-loss-only-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )

    equity = pd.read_csv(outdir / "profit-target-comparison-equity.csv")
    equity = equity.rename(columns={"new_limit_equity": "stop_loss_only_equity"})
    equity.to_csv(outdir / "stop-loss-only-equity.csv", index=False)

    trades = outdir / "profit-target-hybrid-trades.csv"
    if trades.exists():
        trades.rename(outdir / "stop-loss-only-trades.csv")

    report = (outdir / "profit-target-comparison.html").read_text(encoding="utf-8")
    report = report.replace(
        "Short exit: old rule vs hybrid −4% Buy Limit",
        "Short exit: old rule vs stop loss only",
    )
    report = report.replace(
        "New hybrid: close/open + next-day −4% Buy Limit",
        "Stop loss + day-2 time exit",
    )
    report = report.replace(
        "preserve that close-to-next-open exit. If it does not trigger, place a next-session buy limit at 4% below entry alongside the stop; if neither fills, cover at the close.",
        "remove the take-profit exit. Keep the 2.5×ATR stop active on day 2; if it is not hit, cover at the day-2 close.",
    )
    (outdir / "stop-loss-only-report.html").write_text(report, encoding="utf-8")
