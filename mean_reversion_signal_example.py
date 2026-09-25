"""Generate one actual signal chart without rerunning the full universe."""

from pathlib import Path

import pandas as pd

from mean_reversion_short_backtest import Alpaca, write_signal_example


def main():
    outdir = Path("short-signal-example")
    outdir.mkdir(parents=True, exist_ok=True)
    # This was the first executed trade in the 2016-present Alpaca backtest.
    # Entry/exit prices are split/dividend-adjusted, matching the simulation.
    trade = pd.Series({
        "symbol": "CVLT",
        "signal_date": "2016-02-01",
        "entry_date": "2016-02-02",
        "exit_date": "2016-02-03",
        "entry_price": 39.06,
        "exit_price": 38.27,
        "exit_reason": "time_exit",
        "net_return": 0.0152,
    })
    write_signal_example(Alpaca(), trade, outdir)


if __name__ == "__main__":
    main()
