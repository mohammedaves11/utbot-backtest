"""
strategy5_staged_partial_close.py
===================================
Strategy 5 — Four-Level Staged Partial Close

EXIT RULES
----------
  1:2  → Close 50 % of position at TP1,  move SL to Breakeven (entry price)
  1:3  → Close 25 % of position at TP2,  SL stays at Breakeven
  1:4  → Close 15 % of position at TP3,  SL stays at Breakeven
  Remaining 10 % → Hold until an OPPOSITE signal fires, then exit at that bar's open

ENTRY
-----
  Next bar's open after the signal bar fires (same as all other strategies).

SL LOGIC
---------
  • Before TP1 : SL is at entry − sl_mult × ATR  (longs) or entry + sl_mult × ATR (shorts)
  • After  TP1 : SL moves to breakeven and never changes again
  • If SL is hit before TP1 → entire remaining position exits at SL price

SAME-BAR PRIORITY
-----------------
  If both SL and TP1 can be touched on the same bar, SL is treated as hit first
  (conservative assumption).

TRADE-LOG ROWS
--------------
  Row 1  — 50 % closed at TP1   (exit_reason = 'TP1 (1:2) — 50% partial')
  Row 2  — 25 % closed at TP2   (exit_reason = 'TP2 (1:3) — 25% partial')
  Row 3  — 15 % closed at TP3   (exit_reason = 'TP3 (1:4) — 15% partial')
  Row 4  — 10 % closed at opposite signal / SL / end-of-data

TRADE ENTRY/EXIT GUIDE (for live trading)
------------------------------------------
  ENTRY
  ------
  • Wait for the strategy indicator to fire a BUY or SELL signal on a bar close.
  • Enter at the OPEN of the NEXT bar (market order or limit at open price).
  • Place initial SL at:  entry − (sl_mult × ATR)   for LONG
                          entry + (sl_mult × ATR)   for SHORT
  • Divide your position into four tranches:
        Tranche A = 50 % of total lots
        Tranche B = 25 % of total lots
        Tranche C = 15 % of total lots
        Tranche D = 10 % of total lots   ← held until opposite signal

  EXIT — LONG example (mirror for SHORT)
  ----------------------------------------
  TP1 = entry + 2 × SL_distance   → Close Tranche A (50 %), move SL to entry (breakeven)
  TP2 = entry + 3 × SL_distance   → Close Tranche B (25 %), SL remains at breakeven
  TP3 = entry + 4 × SL_distance   → Close Tranche C (15 %), SL remains at breakeven
  Tranche D (10 %)                → Close when an OPPOSITE bar-close signal appears;
                                     exit at the OPEN of the bar after that signal bar.
                                     If SL (breakeven) is hit before that → exit at SL.

Usage
-----
from backtest_utils import load_data
from strategy5_staged_partial_close import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy5_staged_partial_close.py --source csv --csv XAUUSD_M15.csv
python strategy5_staged_partial_close.py --source mt5 --symbol XAUUSD.t --tf M15
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

STRATEGY_NAME = ("Strategy 5 — Staged Partial Close: "
                 "50% @ 1:2 · 25% @ 1:3 · 15% @ 1:4 · 10% on Opposite Signal")


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
                 # trail_mult is accepted but unused – keeps call-signature
                 # compatible with run_all.py which passes it to every strategy
                 trail_mult:       float = DEFAULTS['trail_mult']) -> tuple:
    """
    Run Strategy 5 backtest.

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
    initial_balance : Starting account equity for equity-curve display
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report
    trail_mult    : Accepted for API compatibility but not used in this strategy

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
    position      = None        # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    initial_sl    = 0.0         # original SL price at entry
    sl_price      = 0.0         # active SL; moves to BE after TP1
    sl_dist_entry = 0.0         # SL distance locked at entry
    lots          = 0.0         # full initial lot size

    # Tranche sizes — computed at entry from 'lots'
    lots_a = 0.0  # 50% — closed at TP1
    lots_b = 0.0  # 25% — closed at TP2
    lots_c = 0.0  # 15% — closed at TP3
    lots_d = 0.0  # 10% — closed on opposite signal

    tp1_price  = 0.0   # 1:2 target
    tp2_price  = 0.0   # 1:3 target
    tp3_price  = 0.0   # 1:4 target

    tp1_hit  = False
    tp2_hit  = False
    tp3_hit  = False

    # For the "enter next bar" mechanic
    pending  = None

    # For opposite-signal exit: we need a 1-bar delay (enter next bar after signal)
    # So we store whether an opposite signal was seen on the *previous* bar
    opposite_pending_exit = False

    trades  = []
    cumul   = 0.0
    eq_pts  = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row       = sig.iloc[i]
        bar_time  = idx[i]
        bar_open  = row['open']
        bar_high  = row['high']
        bar_low   = row['low']
        bar_atr   = row['atr']

        # ── A. Fill pending ENTRY (open next bar after signal) ────────────────
        if pending is not None and position is None:
            direction      = pending['direction']
            sl_d           = pending['sl_dist']
            lots           = pending['lot_size']
            position       = direction
            entry_price    = bar_open
            entry_time     = bar_time
            sl_dist_entry  = sl_d

            # Initial SL
            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            # TP levels
            if direction == 'long':
                tp1_price = entry_price + 2.0 * sl_d
                tp2_price = entry_price + 3.0 * sl_d
                tp3_price = entry_price + 4.0 * sl_d
            else:
                tp1_price = entry_price - 2.0 * sl_d
                tp2_price = entry_price - 3.0 * sl_d
                tp3_price = entry_price - 4.0 * sl_d

            # Tranches
            lots_a = lots * 0.25
            lots_b = lots * 0.10
            lots_c = lots * 0.20
            lots_d = lots * 0.45

            tp1_hit = tp2_hit = tp3_hit = False
            opposite_pending_exit = False
            pending = None

        elif pending is not None:
            pending = None  # already in a position — cancel

        # ── B. Manage open LONG position ──────────────────────────────────────
        if position == 'long':
            exited_all = False

            # ── Phase 0: before TP1, check full SL ───────────────────────────
            if not tp1_hit:
                # Conservative: SL before TP1
                if bar_low <= sl_price:
                    remaining = lots_a + lots_b + lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        remaining, pnl, 'SL hit (before TP1)'))
                    cumul += pnl
                    position = None
                    exited_all = True

                elif bar_high >= tp1_price:
                    # Close Tranche A at TP1
                    pnl_a = (tp1_price - entry_price) * lots_a * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp1_price, initial_sl,
                        lots_a, pnl_a, 'TP1 (1:2) — 50% partial'))
                    cumul += pnl_a
                    sl_price = entry_price   # SL moves to breakeven
                    tp1_hit  = True
                    # Check TP2 on same bar
                    if bar_high >= tp2_price:
                        pnl_b = (tp2_price - entry_price) * lots_b * contract_size
                        trades.append(_make_trade(
                            entry_time, bar_time, 'long',
                            entry_price, tp2_price, sl_price,
                            lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                        cumul += pnl_b
                        tp2_hit = True
                        # Check TP3 on same bar
                        if bar_high >= tp3_price:
                            pnl_c = (tp3_price - entry_price) * lots_c * contract_size
                            trades.append(_make_trade(
                                entry_time, bar_time, 'long',
                                entry_price, tp3_price, sl_price,
                                lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                            cumul += pnl_c
                            tp3_hit = True

            # ── Phase 1: TP1 hit, waiting for TP2 ────────────────────────────
            if tp1_hit and not tp2_hit and not exited_all:
                # Check breakeven SL on remaining
                if bar_low <= sl_price:
                    remaining = lots_b + lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP1'))
                    cumul += pnl
                    position = None
                    exited_all = True
                elif bar_high >= tp2_price:
                    pnl_b = (tp2_price - entry_price) * lots_b * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp2_price, sl_price,
                        lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                    cumul += pnl_b
                    tp2_hit = True
                    # Check TP3 same bar
                    if bar_high >= tp3_price:
                        pnl_c = (tp3_price - entry_price) * lots_c * contract_size
                        trades.append(_make_trade(
                            entry_time, bar_time, 'long',
                            entry_price, tp3_price, sl_price,
                            lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                        cumul += pnl_c
                        tp3_hit = True

            # ── Phase 2: TP2 hit, waiting for TP3 ────────────────────────────
            if tp2_hit and not tp3_hit and not exited_all:
                if bar_low <= sl_price:
                    remaining = lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP2'))
                    cumul += pnl
                    position = None
                    exited_all = True
                elif bar_high >= tp3_price:
                    pnl_c = (tp3_price - entry_price) * lots_c * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp3_price, sl_price,
                        lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                    cumul += pnl_c
                    tp3_hit = True

            # ── Phase 3: TP3 hit, hold Tranche D until opposite signal ────────
            if tp3_hit and not exited_all:
                # First: honour deferred opposite-signal exit (1-bar delay)
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl_d  = (exit_p - entry_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        lots_d, pnl_d, 'Opposite signal exit (10%)'))
                    cumul += pnl_d
                    position = None
                    exited_all = True
                    opposite_pending_exit = False

                # Breakeven SL check for Tranche D
                if not exited_all and bar_low <= sl_price:
                    pnl_d = (sl_price - entry_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        lots_d, pnl_d, 'SL (breakeven) — Tranche D'))
                    cumul += pnl_d
                    position = None
                    exited_all = True

                # Mark opposite signal for exit on *next* bar open
                if not exited_all and row['sell']:
                    opposite_pending_exit = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited_all = False

            # ── Phase 0: before TP1, check full SL ───────────────────────────
            if not tp1_hit:
                if bar_high >= sl_price:
                    remaining = lots_a + lots_b + lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        remaining, pnl, 'SL hit (before TP1)'))
                    cumul += pnl
                    position = None
                    exited_all = True

                elif bar_low <= tp1_price:
                    pnl_a = (entry_price - tp1_price) * lots_a * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp1_price, initial_sl,
                        lots_a, pnl_a, 'TP1 (1:2) — 50% partial'))
                    cumul += pnl_a
                    sl_price = entry_price
                    tp1_hit  = True
                    if bar_low <= tp2_price:
                        pnl_b = (entry_price - tp2_price) * lots_b * contract_size
                        trades.append(_make_trade(
                            entry_time, bar_time, 'short',
                            entry_price, tp2_price, sl_price,
                            lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                        cumul += pnl_b
                        tp2_hit = True
                        if bar_low <= tp3_price:
                            pnl_c = (entry_price - tp3_price) * lots_c * contract_size
                            trades.append(_make_trade(
                                entry_time, bar_time, 'short',
                                entry_price, tp3_price, sl_price,
                                lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                            cumul += pnl_c
                            tp3_hit = True

            # ── Phase 1: TP1 hit, waiting for TP2 ────────────────────────────
            if tp1_hit and not tp2_hit and not exited_all:
                if bar_high >= sl_price:
                    remaining = lots_b + lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP1'))
                    cumul += pnl
                    position = None
                    exited_all = True
                elif bar_low <= tp2_price:
                    pnl_b = (entry_price - tp2_price) * lots_b * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp2_price, sl_price,
                        lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                    cumul += pnl_b
                    tp2_hit = True
                    if bar_low <= tp3_price:
                        pnl_c = (entry_price - tp3_price) * lots_c * contract_size
                        trades.append(_make_trade(
                            entry_time, bar_time, 'short',
                            entry_price, tp3_price, sl_price,
                            lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                        cumul += pnl_c
                        tp3_hit = True

            # ── Phase 2: TP2 hit, waiting for TP3 ────────────────────────────
            if tp2_hit and not tp3_hit and not exited_all:
                if bar_high >= sl_price:
                    remaining = lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP2'))
                    cumul += pnl
                    position = None
                    exited_all = True
                elif bar_low <= tp3_price:
                    pnl_c = (entry_price - tp3_price) * lots_c * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp3_price, sl_price,
                        lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                    cumul += pnl_c
                    tp3_hit = True

            # ── Phase 3: TP3 hit, hold Tranche D until opposite signal ────────
            if tp3_hit and not exited_all:
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl_d  = (entry_price - exit_p) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        lots_d, pnl_d, 'Opposite signal exit (10%)'))
                    cumul += pnl_d
                    position = None
                    exited_all = True
                    opposite_pending_exit = False

                if not exited_all and bar_high >= sl_price:
                    pnl_d = (entry_price - sl_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        lots_d, pnl_d, 'SL (breakeven) — Tranche D'))
                    cumul += pnl_d
                    position = None
                    exited_all = True

                if not exited_all and row['buy']:
                    opposite_pending_exit = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        # ── D. Record equity on flat bars ─────────────────────────────────────
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

        # Work out exactly which tranches are still open
        open_lots = 0.0
        if not tp1_hit:
            open_lots = lots_a + lots_b + lots_c + lots_d
        elif not tp2_hit:
            open_lots = lots_b + lots_c + lots_d
        elif not tp3_hit:
            open_lots = lots_c + lots_d
        else:
            open_lots = lots_d

        if position == 'long':
            pnl = (last_cl - entry_price) * open_lots * contract_size
        else:
            pnl = (entry_price - last_cl) * open_lots * contract_size

        trades.append(_make_trade(entry_time, last_time, position,
                                  entry_price, last_cl, sl_price,
                                  open_lots, pnl, 'End of data'))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()

    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy5_report_{ts}.html")

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
            TP_Rule           = ('50% @ 1:2 (BE) → 25% @ 1:3 → '
                                 '15% @ 1:4 → 10% on opposite signal'),
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 5]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp   = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*58}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*58}")
    print(f"  Trade records  : {len(trades)}  (incl. partial closes)")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L        : ${tp:,.2f}")
    peak = equity.cummax()
    dd   = equity - peak
    print(f"  Max Drawdown   : ${dd.min():,.2f}")
    print(f"{'─'*58}\n")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=STRATEGY_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy5_staged_partial_close.py --source csv --csv XAUUSD_M15.csv
  python strategy5_staged_partial_close.py --source mt5 --symbol XAUUSD.t --tf M15
  python strategy5_staged_partial_close.py --source csv --csv data.csv --sl-mult 2.0
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
