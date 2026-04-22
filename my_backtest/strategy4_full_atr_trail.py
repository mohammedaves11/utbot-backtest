"""
strategy4_full_atr_trail.py
==============================
TP Rule: Pure ATR trailing stop — no fixed TP target at all.

How it works
------------
• Enter at the open of the bar after the signal.
• The trailing stop begins at the entry SL level (entry ± sl_mult × ATR).
• Every subsequent bar the trail is updated:
    Long:   trail = max(trail, bar_close - trail_mult × ATR)   ← only moves up
    Short:  trail = min(trail, bar_close + trail_mult × ATR)   ← only moves down
• Exit when price trades through the trail (intrabar check on bar low/high).
• No partial closes — the full position runs until the trail is hit.

Why 1.5 × ATR for the trail (default)?
---------------------------------------
Using the same ATR multiplier as the SL (1.5) gives a tighter trail that
locks in profit faster as gold makes its big directional moves, while still
giving the trade enough room to breathe bar to bar.

Entry:   Next bar's open after the signal bar.
SL:      Intrabar check (bar low for longs / bar high for shorts).

Usage
-----
from backtest_utils import load_data
from strategy4_full_atr_trail import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy4_full_atr_trail.py --source csv --csv XAUUSD_M15.csv
python strategy4_full_atr_trail.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
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

STRATEGY_NAME = "Strategy 4 — Full ATR Trailing Stop (no fixed TP)"


# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df:               pd.DataFrame,
                 key_value:        float = DEFAULTS['key_value'],
                 atr_period:       int   = DEFAULTS['atr_period'],
                 sl_mult:          float = DEFAULTS['sl_mult'],
                 trail_mult:       float = DEFAULTS['trail_mult'],
                 risk_usd:         float = DEFAULTS['risk_usd'],
                 contract_size:    float = DEFAULTS['contract_size'],
                 adx_len:          int   = DEFAULTS['adx_len'],
                 adx_thresh:       int   = DEFAULTS['adx_thresh'],
                 filter_choppy:    bool  = DEFAULTS['filter_choppy'],
                 initial_balance:  float = DEFAULTS['initial_balance'],
                 output_dir:       str   = 'reports',
                 symbol:           str   = 'XAUUSD') -> tuple:
    """
    Run Strategy 4 backtest.

    Parameters
    ----------
    df            : OHLC DataFrame (from load_data / load_csv / load_mt5)
    key_value     : ATR multiplier for trailing-stop signal line (default 3)
    atr_period    : ATR period (default 14)
    sl_mult       : SL distance in ATR multiples (default 1.5)
                    Also sets the initial trail level at entry.
    trail_mult    : ATR multiplier for the trailing stop (default 1.5)
                    Lower = tighter trail = faster profit lock-in.
                    Higher = looser trail = more room, bigger wins but bigger gives-back.
    risk_usd      : USD risked per trade (default 200)
    contract_size : Contract size in oz (default 100)
    adx_len       : ADX smoothing period (default 14)
    adx_thresh    : Minimum ADX for a valid trend (default 20)
    filter_choppy : Skip trades when ADX < adx_thresh (default True)
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report

    Returns
    -------
    trades       : list[dict]   — one dict per closed trade
    equity_curve : pd.Series   — cumulative P&L indexed by bar datetime
    report_path  : str         — path to the saved HTML report
    """
    sig = compute_indicators(
        df,
        key_value=key_value, atr_period=atr_period, sl_mult=sl_mult,
        adx_len=adx_len, adx_thresh=adx_thresh, filter_choppy=filter_choppy,
        risk_usd=risk_usd, contract_size=contract_size,
    )

    warmup = atr_period * 3

    # ── Simulation state ──────────────────────────────────────────────────────
    position    = None
    entry_price = 0.0
    entry_time  = None
    initial_sl  = 0.0
    trail_sl    = 0.0
    lots        = 0.0

    pending = None
    trades  = []
    cumul   = 0.0
    eq_pts  = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row      = sig.iloc[i]
        bar_time = idx[i]
        bar_open = row['open']
        bar_high = row['high']
        bar_low  = row['low']
        bar_close= row['close']
        bar_atr  = row['atr']

        # ── A. Open pending entry ─────────────────────────────────────────────
        if pending is not None and position is None:
            direction   = pending['direction']
            sl_d        = pending['sl_dist']
            position    = direction
            entry_price = bar_open
            entry_time  = bar_time
            lots        = pending['lot_size']
            initial_sl  = (entry_price - sl_d) if direction == 'long' \
                          else (entry_price + sl_d)
            # Trail starts at same level as initial SL
            trail_sl    = initial_sl
            pending     = None

        elif pending is not None:
            pending = None

        # ── B. Manage open position ───────────────────────────────────────────
        if position == 'long':
            # Update trailing stop (only moves up)
            new_trail = bar_close - trail_mult * bar_atr
            trail_sl  = max(trail_sl, new_trail)

            # Check if trail hit intrabar
            if bar_low <= trail_sl:
                exit_p = trail_sl
                pnl    = (exit_p - entry_price) * lots * contract_size

                # Label: distinguish initial SL hit vs profit exit
                if exit_p <= initial_sl + 0.01:
                    reason = 'SL hit (ATR trail at initial level)'
                elif exit_p <= entry_price + 0.01:
                    reason = 'ATR Trail exit (near breakeven)'
                else:
                    reason = 'ATR Trail exit (profit)'

                trades.append(_make_trade(entry_time, bar_time, 'long',
                                          entry_price, exit_p, initial_sl,
                                          lots, pnl, reason))
                cumul   += pnl
                position = None
                eq_pts[bar_time] = cumul

        elif position == 'short':
            # Update trailing stop (only moves down)
            new_trail = bar_close + trail_mult * bar_atr
            trail_sl  = min(trail_sl, new_trail)

            if bar_high >= trail_sl:
                exit_p = trail_sl
                pnl    = (entry_price - exit_p) * lots * contract_size

                if exit_p >= initial_sl - 0.01:
                    reason = 'SL hit (ATR trail at initial level)'
                elif exit_p >= entry_price - 0.01:
                    reason = 'ATR Trail exit (near breakeven)'
                else:
                    reason = 'ATR Trail exit (profit)'

                trades.append(_make_trade(entry_time, bar_time, 'short',
                                          entry_price, exit_p, initial_sl,
                                          lots, pnl, reason))
                cumul   += pnl
                position = None
                eq_pts[bar_time] = cumul

        # ── C. New signals when flat ──────────────────────────────────────────
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

    # ── D. Force-close at end of data ─────────────────────────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        if position == 'long':
            pnl = (last_cl - entry_price) * lots * contract_size
            trades.append(_make_trade(entry_time, last_time, 'long',
                                      entry_price, last_cl, initial_sl,
                                      lots, pnl, 'End of data'))
        else:
            pnl = (entry_price - last_cl) * lots * contract_size
            trades.append(_make_trade(entry_time, last_time, 'short',
                                      entry_price, last_cl, initial_sl,
                                      lots, pnl, 'End of data'))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()

    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy4_report_{ts}.html")

    generate_report(
        trades, equity_curve, STRATEGY_NAME, report_path,
        symbol=symbol,
        initial_balance=initial_balance,
        params=dict(
            Symbol            = symbol,
            Initial_Balance   = f'${initial_balance:,.0f}',
            Key_Value         = key_value,
            ATR_Period        = atr_period,
            SL_Multiplier     = sl_mult,
            Trail_ATR_Mult    = trail_mult,
            Risk_USD          = f'${risk_usd:,.0f}',
            Contract_Size     = contract_size,
            ADX_Length        = adx_len,
            ADX_Threshold     = adx_thresh,
            Filter_Choppy     = filter_choppy,
            TP_Rule           = 'Pure ATR trail from entry (no fixed TP)',
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 4]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*50}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*50}")
    print(f"  Total trades : {len(trades)}")
    print(f"  Win rate     : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L      : ${tp:,.2f}")
    peak = equity.cummax(); dd = equity - peak
    print(f"  Max Drawdown : ${dd.min():,.2f}")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=STRATEGY_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy4_full_atr_trail.py --source csv --csv XAUUSD_M15.csv
  python strategy4_full_atr_trail.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
  python strategy4_full_atr_trail.py --source csv --csv data.csv --trail-mult 1.0
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
    p.add_argument('--trail-mult', type=float, default=DEFAULTS['trail_mult'],
                   help='ATR multiplier for trailing stop (default 1.5). '
                        'Lower = tighter, Higher = more room to run.')
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
        sl_mult=args.sl_mult, trail_mult=args.trail_mult,
        risk_usd=args.risk, contract_size=args.contract,
        adx_len=args.adx_len, adx_thresh=args.adx_thresh,
        filter_choppy=not args.no_filter,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()