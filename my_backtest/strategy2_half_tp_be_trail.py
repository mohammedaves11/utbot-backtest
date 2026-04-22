"""
strategy2_half_tp_be_trail.py
===============================
TP Rule:
  1. When price reaches 1:2 RR target  →  close 50 % of position at TP1 price
                                           move SL to breakeven (entry price)
  2. Trail the remaining 50 % with a 1.5 × ATR trailing stop
     (trail_mult configurable, default = sl_mult = 1.5)
  3. Full SL hit before TP1 → exit entire position at SL price (no partial)

Entry:   Next bar's open after the signal bar.
SL:      Intrabar check (bar low for longs / bar high for shorts).

Same-bar priority
-----------------
If both SL and TP1 can be hit on the same bar (low <= SL and high >= TP1),
the SL is treated as triggered first (conservative).

Partial close recording
-----------------------
Each partial close is recorded as a separate row in the trade log so that
both the equity curve step and the per-record P&L are clearly visible.
  Row 1 — lots/2 closed at TP1 price   (exit_reason = 'TP1 (1:2) — partial')
  Row 2 — lots/2 closed at trail or SL (exit_reason = 'ATR Trail exit' / 'SL (breakeven)')

Usage
-----
from backtest_utils import load_data
from strategy2_half_tp_be_trail import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy2_half_tp_be_trail.py --source csv --csv XAUUSD_M15.csv
python strategy2_half_tp_be_trail.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
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

STRATEGY_NAME = "Strategy 2 — 50 % TP at 1:2, Breakeven, ATR Trail"


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
    Run Strategy 2 backtest.

    Parameters
    ----------
    df            : OHLC DataFrame (from load_data / load_csv / load_mt5)
    key_value     : ATR multiplier for trailing-stop signal line (default 3)
    atr_period    : ATR period (default 14)
    sl_mult       : SL distance in ATR multiples (default 1.5)
    trail_mult    : ATR multiplier for the trailing stop (default 1.5)
                    Recommended: same as sl_mult for tighter locking of profit
    risk_usd      : USD risked per trade (default 200)
    contract_size : Contract size in oz (default 100)
    adx_len       : ADX smoothing period (default 14)
    adx_thresh    : Minimum ADX for a valid trend (default 20)
    filter_choppy : Skip trades when ADX < adx_thresh (default True)
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report

    Returns
    -------
    trades       : list[dict]   — one dict per closed partial or full trade
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
    position      = None     # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    initial_sl    = 0.0      # original SL price at entry
    sl_price      = 0.0      # current active SL (moves to BE after TP1)
    lots          = 0.0      # initial full lot size
    sl_dist_entry = 0.0      # sl_dist locked at entry
    tp1_price     = 0.0      # 1:2 target
    tp1_hit       = False
    trail_sl      = 0.0      # active trailing stop for remaining 50 %

    pending  = None
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
        bar_close= row['close']
        bar_atr  = row['atr']

        # ── A. Open pending entry ─────────────────────────────────────────────
        if pending is not None and position is None:
            direction     = pending['direction']
            sl_d          = pending['sl_dist']
            position      = direction
            entry_price   = bar_open
            entry_time    = bar_time
            lots          = pending['lot_size']
            sl_dist_entry = sl_d
            initial_sl    = (entry_price - sl_d) if direction == 'long' \
                            else (entry_price + sl_d)
            sl_price      = initial_sl
            tp1_hit       = False

            if direction == 'long':
                tp1_price = entry_price + 2.0 * sl_d
                trail_sl  = initial_sl   # initialise trail at entry SL level
            else:
                tp1_price = entry_price - 2.0 * sl_d
                trail_sl  = initial_sl
            pending = None

        elif pending is not None:
            pending = None   # already in a position

        # ── B. Manage open position ───────────────────────────────────────────
        if position == 'long':
            half  = lots / 2.0
            exited_all = False

            # ─ Phase 1: before TP1 ───────────────────────────────────────────
            if not tp1_hit:
                # Conservative: check SL before TP1
                if bar_low <= sl_price:
                    pnl = (sl_price - entry_price) * lots * contract_size
                    trades.append(_make_trade(entry_time, bar_time, 'long',
                                              entry_price, sl_price, initial_sl,
                                              lots, pnl, 'SL hit'))
                    cumul += pnl
                    position = None
                    exited_all = True

                elif bar_high >= tp1_price:
                    # Close 50 % at TP1
                    pnl_half = (tp1_price - entry_price) * half * contract_size
                    trades.append(_make_trade(entry_time, bar_time, 'long',
                                              entry_price, tp1_price, initial_sl,
                                              half, pnl_half, 'TP1 (1:2) — partial close'))
                    cumul += pnl_half
                    # Move SL to breakeven, activate trailing for remaining half
                    sl_price = entry_price        # breakeven
                    tp1_hit  = True
                    # Initialise trail at MAX of (breakeven, current ATR trail)
                    trail_sl = max(entry_price,
                                   bar_close - trail_mult * bar_atr)

            # ─ Phase 2: after TP1, trailing remaining 50 % ───────────────────
            if tp1_hit and not exited_all:
                # Update trailing stop (only moves in favourable direction)
                new_trail = bar_close - trail_mult * bar_atr
                trail_sl  = max(trail_sl, new_trail, entry_price)  # never below BE

                # Check SL/trail exit
                if bar_low <= trail_sl:
                    exit_p   = trail_sl
                    pnl_rest = (exit_p - entry_price) * half * contract_size
                    reason   = ('SL (breakeven)' if abs(exit_p - entry_price) < 0.01
                                else 'ATR Trail exit')
                    trades.append(_make_trade(entry_time, bar_time, 'long',
                                              entry_price, exit_p, sl_price,
                                              half, pnl_rest, reason))
                    cumul += pnl_rest
                    position   = None
                    exited_all = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        elif position == 'short':
            half  = lots / 2.0
            exited_all = False

            # ─ Phase 1: before TP1 ───────────────────────────────────────────
            if not tp1_hit:
                if bar_high >= sl_price:
                    pnl = (entry_price - sl_price) * lots * contract_size
                    trades.append(_make_trade(entry_time, bar_time, 'short',
                                              entry_price, sl_price, initial_sl,
                                              lots, pnl, 'SL hit'))
                    cumul += pnl
                    position = None
                    exited_all = True

                elif bar_low <= tp1_price:
                    pnl_half = (entry_price - tp1_price) * half * contract_size
                    trades.append(_make_trade(entry_time, bar_time, 'short',
                                              entry_price, tp1_price, initial_sl,
                                              half, pnl_half, 'TP1 (1:2) — partial close'))
                    cumul += pnl_half
                    sl_price = entry_price
                    tp1_hit  = True
                    trail_sl = min(entry_price,
                                   bar_close + trail_mult * bar_atr)

            # ─ Phase 2: after TP1, trailing remaining 50 % ───────────────────
            if tp1_hit and not exited_all:
                new_trail = bar_close + trail_mult * bar_atr
                trail_sl  = min(trail_sl, new_trail, entry_price)  # never above BE

                if bar_high >= trail_sl:
                    exit_p   = trail_sl
                    pnl_rest = (entry_price - exit_p) * half * contract_size
                    reason   = ('SL (breakeven)' if abs(exit_p - entry_price) < 0.01
                                else 'ATR Trail exit')
                    trades.append(_make_trade(entry_time, bar_time, 'short',
                                              entry_price, exit_p, sl_price,
                                              half, pnl_rest, reason))
                    cumul += pnl_rest
                    position   = None
                    exited_all = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        # ── C. Record equity on exit bars too ─────────────────────────────────
        if position is None:
            eq_pts[bar_time] = cumul

        # ── D. Check for new signals (only when flat) ─────────────────────────
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

    # ── E. Force-close open position at end of data ───────────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        half      = lots / 2.0
        remaining = half if tp1_hit else lots
        if position == 'long':
            pnl = (last_cl - entry_price) * remaining * contract_size
            trades.append(_make_trade(entry_time, last_time, 'long',
                                      entry_price, last_cl, sl_price,
                                      remaining, pnl, 'End of data'))
        else:
            pnl = (entry_price - last_cl) * remaining * contract_size
            trades.append(_make_trade(entry_time, last_time, 'short',
                                      entry_price, last_cl, sl_price,
                                      remaining, pnl, 'End of data'))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()

    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy2_report_{ts}.html")

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
            TP_Rule           = '50% close @ 1:2 → BE → ATR trail',
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 2]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*50}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*50}")
    print(f"  Trade records: {len(trades)}  (incl. partial closes)")
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
  python strategy2_half_tp_be_trail.py --source csv --csv XAUUSD_M15.csv
  python strategy2_half_tp_be_trail.py --source mt5 --symbol XAUUSD --tf M15 --bars 50000
  python strategy2_half_tp_be_trail.py --source csv --csv data.csv --trail-mult 2.0
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
                   help='ATR multiplier for trailing stop (default 1.5)')
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