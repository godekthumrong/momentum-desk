"""Isolated comparison: original short exits vs next-session close only."""
from pathlib import Path
import json
import pandas as pd
import mean_reversion_short_backtest as bt

_original_resolve_exit = bt.resolve_exit


def time_exit_comparison(entry_price, entry_close, atr, day2, exit_rule):
    if exit_rule == "hybrid_close_and_limit":
        return float(day2["close"]), "time_exit", "close"
    return _original_resolve_exit(entry_price, entry_close, atr, day2, exit_rule)


bt.resolve_exit = time_exit_comparison

if __name__ == "__main__":
    args = bt.parse_args()
    bt.run(args)
    outdir = Path(args.output)

    source = outdir / "profit-target-comparison-results.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    result["assumptions"]["new"] = "ignore stop loss and profit target; cover at next-session close"
    result["assumptions"].pop("same_day_both_levels", None)
    result["assumptions"].pop("gap_fill", None)
    (outdir / "time-exit-comparison-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )

    equity = pd.read_csv(outdir / "profit-target-comparison-equity.csv")
    equity = equity.rename(columns={"new_limit_equity": "time_exit_only_equity"})
    equity.to_csv(outdir / "time-exit-comparison-equity.csv", index=False)

    trades = outdir / "profit-target-hybrid-trades.csv"
    if trades.exists():
        trades.rename(outdir / "time-exit-only-trades.csv")

    report = (outdir / "profit-target-comparison.html").read_text(encoding="utf-8")
    report = report.replace("Short exit: old rule vs hybrid −4% Buy Limit", "Short exit: old rule vs time exit only")
    report = report.replace("New hybrid: close/open + next-day −4% Buy Limit", "Time exit only: next-day close")
    report = report.replace(
        "preserve that close-to-next-open exit. If it does not trigger, place a next-session buy limit at 4% below entry alongside the stop; if neither fills, cover at the close.",
        "ignore both stop loss and profit target, then cover every filled short at the following session's close.",
    )
    (outdir / "time-exit-comparison.html").write_text(report, encoding="utf-8")
