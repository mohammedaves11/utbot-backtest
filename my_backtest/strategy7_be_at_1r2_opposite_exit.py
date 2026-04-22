"""
strategy7_be_at_1r2_opposite_exit.py
======================================
Strategy 7 — Breakeven at 1:2, Exit on Opposite Signal

EXIT RULES
----------
  Before 1:2 :  SL is at entry ± (sl_mult × ATR).
                 If hit → full position exits at SL price (loss).

  At 1:2     :  Move SL to BREAKEVEN (entry price).
                 NO partial close — hold the FULL position.

  After 1:2  :  Hold the full position.
                 • If SL (breakeven) is hit → exit everything at entry price (scratch).
                 • When an OPPOSITE signal fires on a bar close → exit at the
                   OPEN of the NEXT bar (1-bar delay, same as entry mechanic).

END OF DATA :   Any open position is closed at the last bar's close.

ENTRY
-----
  Next bar's open after the signal bar closes (market-order style).

SAME-BAR PRIORITY (before 1:2 is reached)
------------------------------------------
  If bar_low <= SL  AND  bar_high >= TP1 on the same bar → SL is treated as
  triggered first (conservative assumption).

TRADE-LOG ROWS
--------------
  One row per closed trade:
    exit_reason = 'SL hit'            — full loss before 1:2
    exit_reason = 'SL (breakeven)'    — scratched after 1:2 was reached
    exit_reason = 'Opposite signal'   — exited at next-bar open after opp. signal
    exit_reason = 'End of data'       — forced close

TRADE ENTRY / EXIT GUIDE (for live trading)
--------------------------------------------
  ENTRY
  ------
  • Wait for the indicator's BUY / SELL arrow on a bar CLOSE.
  • Place a MARKET order at the OPEN of the NEXT bar.
  • Set Stop Loss at:
        LONG  → entry − (sl_mult × ATR)
        SHORT → entry + (sl_mult × ATR)
  • Enter with your FULL calculated lot size (no tranching).

  EXIT — LONG example (mirror for SHORT)
  ----------------------------------------
  1:2 level reached  (price ≥ entry + 2 × SL_dist):
    • Do NOT close any of the position.
    • Move SL order up to entry price (breakeven).

  After that:
    • If price drops back to entry (SL hit) → accept scratch trade, exit.
    • If an opposite (SELL) signal fires on a bar close:
        → At the NEXT bar's OPEN → close the entire position at market.

  MT5 TIPS
  ---------
  Open trade  : F9 or right-click chart → Trade → New Order
  Move SL     : Right-click position in Trade tab → Modify → update Stop Loss
  Close trade : Right-click position → Close Position (full volume)

Usage
-----
from backtest_utils import load_data
from strategy7_be_at_1r2_opposite_exit import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy7_be_at_1r2_opposite_exit.py --source csv --csv XAUUSD_M15.csv
python strategy7_be_at_1r2_opposite_exit.py --source mt5 --symbol XAUUSD.t --tf M15
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_utils import (
    load_data, compute_indicators, generate_report, DEFAULTS, _make_trade
)

STRATEGY_NAME = "Strategy 7 — Breakeven at 1:2, Exit on Opposite Signal"


# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df:               pd.DataFrame,
                 key_value:        float = DEFAULTS['key_value'],
                 atr_period:       int   = DEFAULTS['atr_period'],
                 sl_mult:          float = DEFAULTS['sl_mult'],
                 risk_usd:         float = DEFAULTS['risk_usd'],
                 contract_size:    float = DEFAULTS['contract_size'],
                 adx_len:          int   = DEFAULTS['adx_len'],
                 adx_thresh:       int   = DEFAULTS['adx_thresh'],
                 filter_choppy:    bool  = DEFAULTS['filter_choppy'],
                 initial_balance:  float = DEFAULTS['initial_balance'],
                 output_dir:       str   = 'reports',
                 symbol:           str   = 'XAUUSD',
                 # trail_mult accepted for run_all.py compatibility but not used
                 trail_mult:       float = DEFAULTS['trail_mult']) -> tuple:
    """
    Run Strategy 7 backtest.

    Parameters
    ----------
    df            : OHLC DataFrame (from load_data / load_csv / load_mt5)
    key_value     : ATR multiplier for trailing-stop signal line (default 3)
    atr_period    : ATR period (default 14)
    sl_mult       : SL distance in ATR multiples (default 1.5)
    risk_usd      : USD risked per trade (default 200)
    contract_size : Contract size in oz (default 100)
    adx_len       : ADX smoothing period (default 14)
    adx_thresh    : Minimum ADX for a valid trend (default 20)
    filter_choppy : Skip trades when ADX < adx_thresh (default False)
    initial_balance : Starting equity for equity-curve display
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report
    trail_mult    : Accepted for API compatibility; not used in this strategy

    Returns
    -------
    trades        : list[dict]  — one dict per closed trade
    equity_curve  : pd.Series  — cumulative P&L indexed by bar datetime
    report_path   : str        — path to the saved HTML report
    """
    sig = compute_indicators(
        df,
        key_value=key_value, atr_period=atr_period, sl_mult=sl_mult,
        adx_len=adx_len, adx_thresh=adx_thresh, filter_choppy=filter_choppy,
        risk_usd=risk_usd, contract_size=contract_size,
    )

    warmup = atr_period * 3

    # ── Simulation state ──────────────────────────────────────────────────────
    position      = None        # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    initial_sl    = 0.0         # SL level at entry (original ATR-based level)
    sl_price      = 0.0         # active SL — moves to BE once 1:2 is reached
    sl_dist_entry = 0.0         # SL distance locked at entry
    lots          = 0.0
    tp1_price     = 0.0         # 1:2 target (triggers BE move)
    be_activated  = False       # True once SL has been moved to breakeven

    # 1-bar delay for opposite-signal exit
    opposite_pending_exit = False

    pending  = None             # next-bar entry queue
    trades   = []
    cumul    = 0.0
    eq_pts   = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row      = sig.iloc[i]
        bar_time = idx[i]
        bar_open = row['open']
        bar_high = row['high']
        bar_low  = row['low']
        bar_atr  = row['atr']

        # ── A. Fill pending ENTRY at this bar's open ──────────────────────────
        if pending is not None and position is None:
            direction      = pending['direction']
            sl_d           = pending['sl_dist']
            lots           = pending['lot_size']
            position       = direction
            entry_price    = bar_open
            entry_time     = bar_time
            sl_dist_entry  = sl_d

            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            tp1_price = (entry_price + 2.0 * sl_d) if direction == 'long' \
                        else (entry_price - 2.0 * sl_d)

            be_activated          = False
            opposite_pending_exit = False
            pending               = None

        elif pending is not None:
            pending = None   # already in a position — discard

        # ── B. Manage open LONG position ──────────────────────────────────────
        if position == 'long':
            exited = False

            # ── Phase 0: before breakeven is activated ────────────────────────
            if not be_activated:
                # Conservative: SL before TP1 on same bar
                if bar_low <= sl_price:
                    pnl = (sl_price - entry_price) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit'))
                    cumul += pnl
                    position = None
                    exited   = True

                elif bar_high >= tp1_price:
                    # 1:2 reached — move SL to breakeven, hold full position
                    sl_price     = entry_price
                    be_activated = True
                    # Fall through to Phase 1 check on the SAME bar

            # ── Phase 1: breakeven active, hold until opposite signal ──────────
            if be_activated and not exited:
                # Honour deferred opposite-signal exit (exit at THIS bar's open)
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl    = (exit_p - entry_price) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        lots, pnl, 'Opposite signal'))
                    cumul  += pnl
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # Breakeven SL check
                if not exited and bar_low <= sl_price:
                    pnl = (sl_price - entry_price) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL (breakeven)'))
                    cumul  += pnl
                    position = None
                    exited   = True

                # Opposite signal fires → queue exit for NEXT bar's open
                if not exited and row['sell']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited = False

            if not be_activated:
                if bar_high >= sl_price:
                    pnl = (entry_price - sl_price) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit'))
                    cumul += pnl
                    position = None
                    exited   = True

                elif bar_low <= tp1_price:
                    sl_price     = entry_price
                    be_activated = True

            if be_activated and not exited:
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl    = (entry_price - exit_p) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        lots, pnl, 'Opposite signal'))
                    cumul  += pnl
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                if not exited and bar_high >= sl_price:
                    pnl = (entry_price - sl_price) * lots * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL (breakeven)'))
                    cumul  += pnl
                    position = None
                    exited   = True

                if not exited and row['buy']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── D. Record equity on flat / exit bars ──────────────────────────────
        if position is None:
            eq_pts[bar_time] = cumul

        # ── E. Check for new signals (only when completely flat) ──────────────
        if position is None:
            if row['buy']:
                pending = dict(direction='long',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])
            elif row['sell']:
                pending = dict(direction='short',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])

        if bar_time not in eq_pts:
            eq_pts[bar_time] = cumul

    # ── F. Force-close any open position at end of data ───────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        if position == 'long':
            pnl = (last_cl - entry_price) * lots * contract_size
        else:
            pnl = (entry_price - last_cl) * lots * contract_size
        trades.append(_make_trade(entry_time, last_time, position,
                                  entry_price, last_cl, sl_price,
                                  lots, pnl, 'End of data'))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()

    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy7_report_{ts}.html")

    generate_report(
        trades, equity_curve, STRATEGY_NAME, report_path,
        symbol=symbol,
        initial_balance=initial_balance,
        params=dict(
            Symbol           = symbol,
            Initial_Balance  = f'${initial_balance:,.0f}',
            Key_Value        = key_value,
            ATR_Period       = atr_period,
            SL_Multiplier    = sl_mult,
            Risk_USD         = f'${risk_usd:,.0f}',
            Contract_Size    = contract_size,
            ADX_Length       = adx_len,
            ADX_Threshold    = adx_thresh,
            Filter_Choppy    = filter_choppy,
            TP_Rule          = 'No partial close — BE at 1:2, exit on opposite signal',
            Total_Bars       = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 7]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp   = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*56}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*56}")
    print(f"  Total trades   : {len(trades)}")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L        : ${tp:,.2f}")
    peak = equity.cummax()
    dd   = equity - peak
    print(f"  Max Drawdown   : ${dd.min():,.2f}")
    print(f"{'─'*56}\n")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=STRATEGY_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy7_be_at_1r2_opposite_exit.py --source csv --csv XAUUSD_M15.csv
  python strategy7_be_at_1r2_opposite_exit.py --source mt5 --symbol XAUUSD.t --tf M15
  python strategy7_be_at_1r2_opposite_exit.py --source csv --csv data.csv --sl-mult 2.0
        """)
    p.add_argument('--source',     choices=['csv', 'mt5'], default='csv')
    p.add_argument('--csv',        default='XAUUSD_M15.csv')
    p.add_argument('--symbol',     default='XAUUSD')
    p.add_argument('--tf',         default='M15')
    p.add_argument('--bars',       type=int,   default=50_000)
    p.add_argument('--outdir',     default='reports')
    p.add_argument('--key-value',  type=float, default=DEFAULTS['key_value'])
    p.add_argument('--atr',        type=int,   default=DEFAULTS['atr_period'])
    p.add_argument('--sl-mult',    type=float, default=DEFAULTS['sl_mult'])
    p.add_argument('--risk',       type=float, default=DEFAULTS['risk_usd'])
    p.add_argument('--contract',   type=float, default=DEFAULTS['contract_size'])
    p.add_argument('--adx-len',    type=int,   default=DEFAULTS['adx_len'])
    p.add_argument('--adx-thresh', type=int,   default=DEFAULTS['adx_thresh'])
    p.add_argument('--no-filter',  action='store_true')
    args = p.parse_args()

    df = load_data(source=args.source, csv_path=args.csv,
                   symbol=args.symbol, timeframe=args.tf, n_bars=args.bars)

    trades, equity, report = run_backtest(
        df,
        key_value=args.key_value, atr_period=args.atr,
        sl_mult=args.sl_mult,
        risk_usd=args.risk, contract_size=args.contract,
        adx_len=args.adx_len, adx_thresh=args.adx_thresh,
        filter_choppy=not args.no_filter,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()
