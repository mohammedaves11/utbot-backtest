"""
strategy1_opposite_signal.py
==============================
TP Rule: Exit the trade ONLY when an opposite signal fires.
SL:      Hit intrabar (bar low for longs / bar high for shorts).
Entry:   Next bar's open after the signal bar.

Trade lifecycle
---------------
1. Signal fires on bar N  (buy or sell)
2. Enter at bar N+1 open
3. Each subsequent bar:
   a. Check SL hit (intrabar low/high)              → exit at SL price
   b. Check opposite signal                          → exit at bar close,
                                                       open new position at next bar open
4. End of data → close any open position at last close

P&L formula  (XAUUSD, 1 lot = 100 oz by default)
-----------------
Long  P&L = (exit - entry) × lots × contract_size
Short P&L = (entry - exit) × lots × contract_size

Usage
-----
# Programmatic (import as module):
from backtest_utils import load_data
from strategy1_opposite_signal import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# Command line:
python strategy1_opposite_signal.py --source csv --csv XAUUSD_M15.csv
python strategy1_opposite_signal.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_utils import (
    load_data, compute_indicators, generate_report, DEFAULTS, _make_trade
)

STRATEGY_NAME = "Strategy 1 — Exit on Opposite Signal"


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
                 symbol:           str   = 'XAUUSD') -> tuple:
    """
    Run Strategy 1 backtest.

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
    filter_choppy : Skip trades when ADX < adx_thresh (default True)
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report

    Returns
    -------
    trades       : list[dict]   — one dict per closed trade
    equity_curve : pd.Series   — cumulative P&L indexed by bar datetime
    report_path  : str         — path to the saved HTML report
    """
    # ── 1. Compute all indicator columns ──────────────────────────────────────
    sig = compute_indicators(
        df,
        key_value=key_value, atr_period=atr_period, sl_mult=sl_mult,
        adx_len=adx_len, adx_thresh=adx_thresh, filter_choppy=filter_choppy,
        risk_usd=risk_usd, contract_size=contract_size,
    )

    warmup = atr_period * 3   # bars to skip while ATR/ADX stabilise

    # ── 2. Simulation state ───────────────────────────────────────────────────
    position      = None     # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    sl_price      = 0.0
    sl_dist_entry = 0.0      # SL distance locked at entry (for RR calc)
    lots          = 0.0
    max_excursion = 0.0      # max favourable price move from entry (for highest_rr)

    # Pending entry: set on signal bar, consumed on the NEXT bar's open
    pending = None   # dict with keys: direction, sl_dist, lot_size

    trades     = []
    cumul_pnl  = 0.0
    equity_pts = {}   # { datetime : cumulative_pnl }

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row      = sig.iloc[i]
        bar_time = idx[i]
        bar_open = row['open']
        bar_high = row['high']
        bar_low  = row['low']
        bar_close= row['close']

        # ── A. Open pending entry at this bar's open ──────────────────────────
        if pending is not None and position is None:
            direction     = pending['direction']
            sl_d          = pending['sl_dist']
            position      = direction
            entry_price   = bar_open
            entry_time    = bar_time
            lots          = pending['lot_size']
            sl_dist_entry = sl_d
            max_excursion = 0.0
            sl_price      = (entry_price - sl_d) if direction == 'long' \
                            else (entry_price + sl_d)
            pending = None

        elif pending is not None:
            # Already in a position — discard the pending signal
            pending = None

        # ── B. Manage open position ───────────────────────────────────────────
        exited = False

        if position == 'long':
            # Update max favourable excursion this bar
            max_excursion = max(max_excursion, bar_high - entry_price)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            # B1. SL check (intrabar low)
            if bar_low <= sl_price:
                pnl   = (sl_price - entry_price) * lots * contract_size
                c_rr  = round((sl_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(entry_time, bar_time, 'long',
                                          entry_price, sl_price, sl_price,
                                          lots, pnl, 'SL hit',
                                          highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True

            # B2. Opposite signal → exit at bar close, queue new short
            elif row['sell']:
                pnl  = (bar_close - entry_price) * lots * contract_size
                c_rr = round((bar_close - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(entry_time, bar_time, 'long',
                                          entry_price, bar_close, sl_price,
                                          lots, pnl, 'Opposite signal',
                                          highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True
                # Queue new short entry at next bar open
                pending = dict(direction='short',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])

        elif position == 'short':
            # Update max favourable excursion this bar
            max_excursion = max(max_excursion, entry_price - bar_low)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            # B1. SL check (intrabar high)
            if bar_high >= sl_price:
                pnl  = (entry_price - sl_price) * lots * contract_size
                c_rr = round((entry_price - sl_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(entry_time, bar_time, 'short',
                                          entry_price, sl_price, sl_price,
                                          lots, pnl, 'SL hit',
                                          highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True

            # B2. Opposite signal → exit at bar close, queue new long
            elif row['buy']:
                pnl  = (entry_price - bar_close) * lots * contract_size
                c_rr = round((entry_price - bar_close) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(entry_time, bar_time, 'short',
                                          entry_price, bar_close, sl_price,
                                          lots, pnl, 'Opposite signal',
                                          highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True
                # Queue new long entry at next bar open
                pending = dict(direction='long',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])

        # ── C. Check for new signals (only when flat) ─────────────────────────
        if position is None and not exited:
            if row['buy']:
                pending = dict(direction='long',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])
            elif row['sell']:
                pending = dict(direction='short',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])

        # Update equity on non-exit bars too (flat = last value)
        if bar_time not in equity_pts:
            equity_pts[bar_time] = cumul_pnl

    # ── D. Force-close any open position at end of data ───────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        h_rr      = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
        if position == 'long':
            pnl  = (last_cl - entry_price) * lots * contract_size
            c_rr = round((last_cl - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
            trades.append(_make_trade(entry_time, last_time, 'long',
                                      entry_price, last_cl, sl_price,
                                      lots, pnl, 'End of data',
                                      highest_rr=h_rr, closed_rr=c_rr))
        else:
            pnl  = (entry_price - last_cl) * lots * contract_size
            c_rr = round((entry_price - last_cl) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
            trades.append(_make_trade(entry_time, last_time, 'short',
                                      entry_price, last_cl, sl_price,
                                      lots, pnl, 'End of data',
                                      highest_rr=h_rr, closed_rr=c_rr))
        cumul_pnl += pnl
        equity_pts[last_time] = cumul_pnl

    equity_curve = pd.Series(equity_pts).sort_index()

    # ── E. Print summary ──────────────────────────────────────────────────────
    _print_summary(trades, equity_curve)

    # ── F. Generate HTML report ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy1_report_{ts}.html")

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
            Risk_USD          = f'${risk_usd:,.0f}',
            Contract_Size     = contract_size,
            ADX_Length        = adx_len,
            ADX_Threshold     = adx_thresh,
            Filter_Choppy     = filter_choppy,
            TP_Rule           = 'Opposite signal only',
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(trades: list, equity: pd.Series):
    if not trades:
        print("[Strategy 1]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    total_pnl = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*50}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*50}")
    print(f"  Total trades : {len(trades)}")
    print(f"  Win rate     : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L      : ${total_pnl:,.2f}")
    peak = equity.cummax(); dd = equity - peak
    print(f"  Max Drawdown : ${dd.min():,.2f}")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=STRATEGY_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy1_opposite_signal.py --source csv --csv XAUUSD_M15.csv
  python strategy1_opposite_signal.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
  python strategy1_opposite_signal.py --source csv --csv data.csv --risk 100 --sl-mult 2.0
        """)
    p.add_argument('--source',   choices=['csv', 'mt5'], default='csv',
                   help="Data source: 'csv' or 'mt5'  (default: csv)")
    p.add_argument('--csv',      default='XAUUSD_M15.csv',
                   help="Path to CSV file  (used when --source csv)")
    p.add_argument('--symbol',   default='XAUUSD',
                   help="MT5 symbol  (used when --source mt5)")
    p.add_argument('--tf',       default='M15',
                   help="Timeframe: M1 M5 M15 M30 H1 H4 D1  (default: M15)")
    p.add_argument('--bars',     type=int, default=50_000,
                   help="Number of bars to load from MT5  (default: 50000)")
    p.add_argument('--outdir',   default='reports',
                   help="Output folder for the HTML report  (default: reports)")
    p.add_argument('--key-value',type=float, default=DEFAULTS['key_value'],
                   help=f"ATR sensitivity key value  (default: {DEFAULTS['key_value']})")
    p.add_argument('--atr',      type=int,   default=DEFAULTS['atr_period'],
                   help=f"ATR period  (default: {DEFAULTS['atr_period']})")
    p.add_argument('--sl-mult',  type=float, default=DEFAULTS['sl_mult'],
                   help=f"SL ATR multiplier  (default: {DEFAULTS['sl_mult']})")
    p.add_argument('--risk',     type=float, default=DEFAULTS['risk_usd'],
                   help=f"Risk per trade USD  (default: {DEFAULTS['risk_usd']})")
    p.add_argument('--contract', type=float, default=DEFAULTS['contract_size'],
                   help=f"Contract size in oz  (default: {DEFAULTS['contract_size']})")
    p.add_argument('--adx-len',  type=int,   default=DEFAULTS['adx_len'],
                   help=f"ADX period  (default: {DEFAULTS['adx_len']})")
    p.add_argument('--adx-thresh',type=int,  default=DEFAULTS['adx_thresh'],
                   help=f"ADX threshold  (default: {DEFAULTS['adx_thresh']})")
    p.add_argument('--no-filter', action='store_true',
                   help="Disable ADX choppy filter (trade all signals)")
    args = p.parse_args()

    df = load_data(source=args.source, csv_path=args.csv,
                   symbol=args.symbol, timeframe=args.tf, n_bars=args.bars)

    trades, equity, report = run_backtest(
        df,
        key_value=args.key_value, atr_period=args.atr,
        sl_mult=args.sl_mult, risk_usd=args.risk,
        contract_size=args.contract,
        adx_len=args.adx_len, adx_thresh=args.adx_thresh,
        filter_choppy=not args.no_filter,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()