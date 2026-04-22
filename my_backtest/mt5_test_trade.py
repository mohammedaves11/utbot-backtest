"""
mt5_test_trade.py
==================
Quick smoke-test for MetaTrader 5 live/demo connectivity on XAUUSD.t.

WHAT IT DOES
------------
1. Connects to a running MT5 terminal.
2. Places a MARKET BUY order on XAUUSD.t (0.01 lot).
3. Waits 2 minutes (configurable via HOLD_SECONDS).
4. Closes the trade at market.
5. Prints entry price, exit price, and gross P&L.

REQUIREMENTS
------------
    pip install MetaTrader5
    MT5 desktop terminal must be running and logged into a live/demo account.
    The symbol XAUUSD.t must be visible in the Market Watch window.

RUN
---
    python mt5_test_trade.py

CONFIGURE below in the ── USER CONFIG ── block.
"""

import time
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# ──  USER CONFIG  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL       = 'XAUUSD.t'   # ← change to match your broker's exact symbol name
LOT_SIZE     = 0.01         # keep tiny for a test  (0.01 = 1 oz)
HOLD_SECONDS = 120          # seconds to hold before closing  (120 = 2 min)
MAGIC        = 999001       # unique magic number for this test order
COMMENT      = 'AVES'

# ══════════════════════════════════════════════════════════════════════════════

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError(
        "MetaTrader5 package not found.\n"
        "  Install with:  pip install MetaTrader5\n"
        "  Then re-run this script."
    )


def _ts():
    """Timestamp prefix for console messages."""
    return datetime.now().strftime('[%H:%M:%S]')


def connect():
    """Initialise MT5 connection; raise if it fails."""
    print(f"{_ts()} Connecting to MT5 terminal ...")
    if not mt5.initialize():
        err = mt5.last_error()
        raise RuntimeError(
            f"mt5.initialize() failed: {err}\n"
            "  Make sure the MT5 terminal is open and you are logged in."
        )
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    print(f"{_ts()} Connected  ✓")
    print(f"         Terminal : {info.name}  build {info.build}")
    print(f"         Account  : #{acc.login}  {acc.server}  "
          f"Balance=${acc.balance:,.2f}  Equity=${acc.equity:,.2f}")


def symbol_ready(symbol: str):
    """Ensure the symbol is available and enabled in Market Watch."""
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        raise RuntimeError(
            f"Symbol '{symbol}' not found in MT5.\n"
            f"  Check the exact name in your broker's Market Watch."
        )
    if not sym_info.visible:
        print(f"{_ts()} Enabling {symbol} in Market Watch ...")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")
    print(f"{_ts()} Symbol OK : {symbol}  "
          f"Digits={sym_info.digits}  "
          f"ContractSize={sym_info.trade_contract_size}")
    return sym_info


def get_tick(symbol: str):
    """Return the latest tick; raise if unavailable."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")
    return tick


def open_buy(symbol: str, lot: float, sym_info) -> int:
    """
    Send a market BUY order.

    Returns the ticket (position ID) on success.
    """
    tick     = get_tick(symbol)
    price    = tick.ask
    point    = sym_info.point
    digits   = sym_info.digits

    # Minimal SL/TP offsets just so MT5 accepts the order
    # (some brokers require SL/TP to be at least freeze_level away)
    # We set them 500 points away so they won't interfere with our manual close.
    sl = round(price - 500 * point, digits)
    tp = round(price + 500 * point, digits)

    request = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         mt5.ORDER_TYPE_BUY,
        'price':        price,
        'sl':           sl,
        'tp':           tp,
        'deviation':    20,          # max slippage in points
        'magic':        MAGIC,
        'comment':      COMMENT,
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }

    print(f"\n{_ts()} Sending BUY order ...")
    print(f"         Symbol   : {symbol}")
    print(f"         Lot      : {lot}")
    print(f"         Ask      : {price}")
    print(f"         SL/TP    : {sl} / {tp}  (wide safety levels)")

    result = mt5.order_send(request)

    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(
            f"BUY order failed  retcode={result.retcode}  "
            f"comment='{result.comment}'\n"
            f"  Full result: {result}"
        )

    ticket = result.order
    entry  = result.price
    print(f"{_ts()} BUY filled  ✓   ticket={ticket}   entry={entry}")
    return ticket, entry


def find_position(ticket: int, symbol: str):
    """Return the open position dict matching ticket, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for pos in positions:
        if pos.ticket == ticket:
            return pos
    return None


def close_position(pos) -> dict:
    """
    Close an open position at market.

    Returns a dict with exit info.
    """
    symbol  = pos.symbol
    ticket  = pos.ticket
    lot     = pos.volume
    typ     = pos.type     # 0 = BUY, 1 = SELL

    tick    = get_tick(symbol)

    # To close a BUY we send a SELL; to close a SELL we send a BUY
    if typ == mt5.ORDER_TYPE_BUY:
        close_type  = mt5.ORDER_TYPE_SELL
        close_price = tick.bid
    else:
        close_type  = mt5.ORDER_TYPE_BUY
        close_price = tick.ask

    sym_info = mt5.symbol_info(symbol)
    digits   = sym_info.digits

    request = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         close_type,
        'price':        close_price,
        'deviation':    20,
        'magic':        MAGIC,
        'comment':      f'{COMMENT}_close',
        'position':     ticket,
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }

    print(f"\n{_ts()} Closing position #{ticket} ...")
    result = mt5.order_send(request)

    if result is None:
        raise RuntimeError(f"Close order returned None: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(
            f"Close order failed  retcode={result.retcode}  "
            f"comment='{result.comment}'\n"
            f"  ⚠  You may need to close the position manually in MT5!"
        )

    exit_price = result.price
    print(f"{_ts()} Position closed  ✓   exit={exit_price}")
    return dict(ticket=ticket, exit_price=exit_price)


def run_test():
    """Full test: connect → open buy → wait → close → report."""
    print()
    print('=' * 58)
    print('  MT5 Strategy-5 Connectivity Test — XAUUSD.t BUY')
    print('=' * 58)

    connect()
    sym_info = symbol_ready(SYMBOL)

    ticket, entry_price = open_buy(SYMBOL, LOT_SIZE, sym_info)

    # ── Wait loop ─────────────────────────────────────────────────────────────
    print(f"\n{_ts()} Holding for {HOLD_SECONDS} seconds "
          f"({HOLD_SECONDS//60}m {HOLD_SECONDS%60}s) ...")
    interval  = 10   # print a status line every 10 seconds
    elapsed   = 0
    while elapsed < HOLD_SECONDS:
        time.sleep(interval)
        elapsed += interval
        pos = find_position(ticket, SYMBOL)
        if pos is None:
            print(f"{_ts()} ⚠  Position #{ticket} no longer found "
                  "(SL/TP may have been hit).")
            break
        current = get_tick(SYMBOL).bid
        gross   = (current - entry_price) * LOT_SIZE * sym_info.trade_contract_size
        print(f"{_ts()} {elapsed:>4}s elapsed  |  current bid={current:.2f}  "
              f"unrealised P&L ≈ ${gross:+.2f}")
    else:
        # ── Close position ─────────────────────────────────────────────────────
        pos = find_position(ticket, SYMBOL)
        if pos is not None:
            close_info = close_position(pos)
            exit_price = close_info['exit_price']
            contract   = sym_info.trade_contract_size
            gross_pnl  = (exit_price - entry_price) * LOT_SIZE * contract
            print()
            print('=' * 58)
            print('  TEST RESULT')
            print('=' * 58)
            print(f"  Symbol       : {SYMBOL}")
            print(f"  Direction    : BUY")
            print(f"  Lot          : {LOT_SIZE}")
            print(f"  Entry price  : {entry_price:.2f}")
            print(f"  Exit price   : {exit_price:.2f}")
            print(f"  Gross P&L    : ${gross_pnl:+.2f}  "
                  f"(before spread/commission)")
            print(f"  Hold time    : {HOLD_SECONDS}s")
            status = '✓  PASS' if True else '✗  FAIL'
            print(f"  Status       : {status} — MT5 order round-trip OK")
        else:
            print(f"\n{_ts()} Position was already closed (SL/TP hit during wait).")

    mt5.shutdown()
    print(f"\n{_ts()} MT5 connection closed.")
    print()


# ── How to enter and exit a trade manually in MT5 ────────────────────────────
GUIDE = """
╔══════════════════════════════════════════════════════════╗
║  HOW TO MANUALLY ENTER & EXIT A TRADE IN MT5             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ENTRY (Strategy 5 — BUY example)                        ║
║  ──────────────────────────────────                      ║
║  1. Wait for the indicator to paint a BUY arrow on the   ║
║     current bar (signal bar).                            ║
║  2. When that bar CLOSES, note the ATR-based SL level.   ║
║  3. At the OPEN of the NEXT bar, place a MARKET BUY.     ║
║  4. Set your initial Stop Loss at:                       ║
║       entry − (sl_mult × ATR)   e.g. entry − 1.5×ATR    ║
║  5. Split your total lots into 4 tranches:               ║
║       Tranche A = 50 %  → take profit at 1:2 RR          ║
║       Tranche B = 25 %  → take profit at 1:3 RR          ║
║       Tranche C = 15 %  → take profit at 1:4 RR          ║
║       Tranche D = 10 %  → hold until opposite signal     ║
║                                                          ║
║  EXIT (LONG — step by step)                              ║
║  ─────────────────────────                               ║
║  TP1 reached  (price = entry + 2×SL_dist):               ║
║    • Close Tranche A (50 %) NOW.                         ║
║    • Move SL of remaining 3 tranches to BREAKEVEN.       ║
║                                                          ║
║  TP2 reached  (price = entry + 3×SL_dist):               ║
║    • Close Tranche B (25 %) NOW.                         ║
║    • SL stays at breakeven.                              ║
║                                                          ║
║  TP3 reached  (price = entry + 4×SL_dist):               ║
║    • Close Tranche C (15 %) NOW.                         ║
║    • SL stays at breakeven.                              ║
║                                                          ║
║  Opposite signal fires:                                  ║
║    • Wait for that bar to CLOSE.                         ║
║    • At the NEXT bar OPEN → close Tranche D (10 %).      ║
║                                                          ║
║  SL hit at any phase:                                    ║
║    • All remaining tranches exit at SL price.            ║
║    • Before TP1 → SL is at original ATR level.           ║
║    • After TP1  → SL is at breakeven.                    ║
║                                                          ║
║  MT5 QUICK STEPS (manual)                                ║
║  ────────────────────────                                ║
║  Open trade  : Right-click chart → Trade → New Order     ║
║                OR press F9                               ║
║  Partial close: Right-click position in Trade tab        ║
║                → Modify or Close → enter partial volume  ║
║  Move SL     : Right-click position → Modify             ║
║                → update Stop Loss field → OK             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


if __name__ == '__main__':
    print(GUIDE)
    run_test()
