# IBKR Paper bridge for the Short Watchlist

This local bridge prepares day-3 exit orders from the Daily Short Scanner and
sends them only after a second confirmation in the browser. The scanner may be
opened on an iPhone while TWS and the bridge stay on a computer.

## Safety boundaries

- TWS Paper port `7497` is hard-coded as the only accepted broker port.
- IBKR account IDs must begin with `DU`, the Paper account prefix.
- The bridge creates `BUY` exit orders only; it cannot open a Short position.
- It checks that the requested cover quantity does not exceed the current
  Paper short position.
- A preview token expires after 120 seconds and can be used only once.
- The service listens only on `127.0.0.1` and accepts the configured scanner
  origin.
- Every API request also requires a random Bridge key printed only in the
  computer terminal.

## One-time TWS setup

1. Install and open Trader Workstation on the computer that will run the
   bridge.
2. Log in using **Paper Trading**.
3. Open **Global Configuration → API → Settings**.
4. Enable **ActiveX and Socket Clients**.
5. Set **Socket port** to `7497`.
6. Add `127.0.0.1` as a trusted IP if TWS requests it.
7. Disable **Read-Only API**. The bridge still refuses live accounts itself.

## Install and run

From this project directory:

```bash
python -m pip install -r requirements-ibkr-bridge.txt
python ibkr_paper_bridge.py
```

Keep the terminal open until the orders have been accepted by IBKR Paper.
The terminal prints a new Bridge key each time the bridge starts. Do not share
the key or put it in a public URL.

## Private mobile access with Tailscale

Do not open port `8765` on the router and do not use Tailscale Funnel. Use
Tailscale Serve so the bridge remains available only to devices signed into
your tailnet.

### One-time setup

1. Install Tailscale on the computer and iPhone.
2. Sign in to the same Tailscale account on both devices.
3. On the computer, start the bridge as shown above.
4. In another terminal on the computer, run:

   ```bash
   tailscale serve --bg 8765
   tailscale serve status
   ```

5. Copy the HTTPS address shown by `tailscale serve status`. It normally looks
   like `https://computer-name.tailnet-name.ts.net`.

### Each time you use the mobile scanner

1. Keep TWS Paper, the bridge terminal and Tailscale running on the computer.
2. Turn on Tailscale on the iPhone.
3. Open the Daily Short Scanner on the iPhone.
4. Paste the Tailscale HTTPS address into **Bridge URL**.
5. Copy the current **Bridge key** from the computer terminal and paste it into
   **Bridge key** on the iPhone.
6. Tap **Check bridge**. Continue only when the result shows `PAPER`, a masked
   `DU` account and port `7497`.

The Bridge URL is remembered in that browser. The Bridge key is kept only for
the current browser session. Watchlist data is also local to each browser, so a
Watchlist created on the computer does not automatically appear on the iPhone.

## Submit from the Watchlist

1. Open the Daily Short Scanner on the computer or connected iPhone.
2. Fill in the actual entry date, price, ATR and share count.
3. Click **Check bridge** and verify that it says `PAPER`, a masked `DU`
   account and port `7497`.
4. Select the **Paper** checkbox beside each eligible exit.
5. Click **Prepare selected exits**.
6. Review every symbol, quantity, stop and activation time.
7. Check **I checked every Paper order** and click
   **Confirm Send to Paper**.
8. Verify the returned IBKR order IDs in TWS before leaving the computer.

When the entry-day close is at least 4% below entry, the bridge creates a
`BUY MKT / OPG` cover. Otherwise it creates a `BUY STP / DAY` and a
`BUY MKT / DAY` activated at `15:59:00 US/Eastern`, linked in one OCA group.

## Checking from IBKR Mobile

IBKR permits only one active brokerage session per username. Logging into
IBKR Mobile with the same username can disconnect TWS. Use Mobile read-only
access or a second IBKR username if TWS must remain connected. If switching
after submission, first confirm in TWS that every order is accepted and no
order is left in an untransmitted state.

## Mock mode

The UI and confirmation flow can be exercised without TWS:

```bash
python ibkr_paper_bridge.py --mock
```

Mock mode never connects to IBKR and recognizes only a simulated 100-share
short position in ticker `TEST`.
