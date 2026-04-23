"""
strategy11_ema_staged_partial_close.py
========================================
Strategy 11 — Four-Level Staged Partial Close  +  EMA 20/50 Trend Filter

Identical exit logic to Strategy 5, but adds a dual-EMA entry filter:

  BUY  signal : price must be ABOVE both EMA-20 and EMA-50 to enter.
                If not aligned, the signal is PARKED — we wait until price
                crosses above both EMAs, then enter at the NEXT bar's open.

  SELL signal : price must be BELOW both EMA-20 and EMA-50 to enter.
                If not aligned, park and wait for price to drop below both.

EXIT RULES (same as Strategy 5)
---------------------------------
  1:2  → Close 50 % of position at TP1,  move SL to Breakeven
  1:3  → Close 25 % of position at TP2,  SL stays at Breakeven
  1:4  → Close 15 % of position at TP3,  SL stays at Breakeven
  Remaining 10 % → Hold until an OPPOSITE signal fires (exit at next bar open)

Usage
-----
from backtest_utils import load_data
from strategy11_ema_staged_partial_close import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_utils import (
    load_data, compute_indicators, generate_report, DEFAULTS, _make_trade
)

STRATEGY_NAME = ("Strategy 11 — Staged Partial Close (EMA 20/50 Filter): "
                 "50% @ 1:2 · 25% @ 1:3 · 15% @ 1:4 · 10% on Opposite Signal")


# ══════════════════════════════════════════════════════════════════════════════
def _compute_emas(df: pd.DataFrame, fast: int = 20, slow: int = 50):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    return ema_fast, ema_slow


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
                 ema_fast:         int   = 20,
                 ema_slow:         int   = 50,
                 trail_mult:       float = DEFAULTS['trail_mult']) -> tuple:
    """Run Strategy 11 backtest (Strategy 5 + EMA 20/50 filter)."""

    sig = compute_indicators(
        df,
        key_value=key_value, atr_period=atr_period, sl_mult=sl_mult,
        adx_len=adx_len, adx_thresh=adx_thresh, filter_choppy=filter_choppy,
        risk_usd=risk_usd, contract_size=contract_size,
    )

    ef, es = _compute_emas(sig, ema_fast, ema_slow)
    sig['ema_fast'] = ef
    sig['ema_slow'] = es

    warmup = max(atr_period * 3, ema_slow * 2)

    # ── Simulation state ──────────────────────────────────────────────────────
    position      = None
    entry_price   = 0.0
    entry_time    = None
    initial_sl    = 0.0
    sl_price      = 0.0
    sl_dist_entry = 0.0
    lots          = 0.0

    lots_a = lots_b = lots_c = lots_d = 0.0
    tp1_price = tp2_price = tp3_price = 0.0
    tp1_hit = tp2_hit = tp3_hit = False

    opposite_pending_exit = False

    # EMA-filter pending signal
    pending_direction = None
    pending_sl_dist   = 0.0
    pending_lot_size  = 0.0
    enter_next_bar    = None   # ready to open at THIS bar's open

    trades = []
    cumul  = 0.0
    eq_pts = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row       = sig.iloc[i]
        bar_time  = idx[i]
        bar_open  = row['open']
        bar_high  = row['high']
        bar_low   = row['low']
        bar_close = row['close']

        ema_f = row['ema_fast']
        ema_s = row['ema_slow']
        price_above_emas = bar_close > ema_f and bar_close > ema_s
        price_below_emas = bar_close < ema_f and bar_close < ema_s

        # ── A. Open entry queued from previous bar ────────────────────────────
        if enter_next_bar is not None and position is None:
            direction      = enter_next_bar['direction']
            sl_d           = enter_next_bar['sl_dist']
            lots           = enter_next_bar['lot_size']
            position       = direction
            entry_price    = bar_open
            entry_time     = bar_time
            sl_dist_entry  = sl_d

            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            if direction == 'long':
                tp1_price = entry_price + 2.0 * sl_d
                tp2_price = entry_price + 3.0 * sl_d
                tp3_price = entry_price + 4.0 * sl_d
            else:
                tp1_price = entry_price - 2.0 * sl_d
                tp2_price = entry_price - 3.0 * sl_d
                tp3_price = entry_price - 4.0 * sl_d

            lots_a = round(lots * 0.50, 4)
            lots_b = round(lots * 0.25, 4)
            lots_c = round(lots * 0.15, 4)
            lots_d = round(lots * 0.10, 4)

            tp1_hit = tp2_hit = tp3_hit = False
            opposite_pending_exit = False
            enter_next_bar    = None
            pending_direction = None

        elif enter_next_bar is not None:
            enter_next_bar    = None
            pending_direction = None

        # ── B. Manage open LONG position ──────────────────────────────────────
        if position == 'long':
            exited_all = False

            # Phase 0: before TP1 — SL or TP1
            if not tp1_hit:
                if bar_low <= sl_price:
                    remaining = lots_a + lots_b + lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL hit'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_high >= tp1_price:
                    pnl_a = (tp1_price - entry_price) * lots_a * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp1_price, sl_price,
                        lots_a, pnl_a, 'TP1 (1:2) — 50% partial'))
                    cumul += pnl_a
                    sl_price = entry_price
                    tp1_hit  = True

            # Phase 1: TP1 hit, waiting for TP2
            if tp1_hit and not tp2_hit and not exited_all:
                if bar_low <= sl_price:
                    remaining = lots_b + lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP1'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_high >= tp2_price:
                    pnl_b = (tp2_price - entry_price) * lots_b * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp2_price, sl_price,
                        lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                    cumul += pnl_b
                    tp2_hit = True

            # Phase 2: TP2 hit, waiting for TP3
            if tp2_hit and not tp3_hit and not exited_all:
                if bar_low <= sl_price:
                    remaining = lots_c + lots_d
                    pnl = (sl_price - entry_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP2'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_high >= tp3_price:
                    pnl_c = (tp3_price - entry_price) * lots_c * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp3_price, sl_price,
                        lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                    cumul += pnl_c
                    tp3_hit = True

            # Phase 3: hold Tranche D until opposite signal
            if tp3_hit and not exited_all:
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl_d  = (exit_p - entry_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        lots_d, pnl_d, 'Opposite signal exit (10%)'))
                    cumul += pnl_d
                    position = None; exited_all = True
                    opposite_pending_exit = False

                if not exited_all and bar_low <= sl_price:
                    pnl_d = (sl_price - entry_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, sl_price,
                        lots_d, pnl_d, 'SL (breakeven) — Tranche D'))
                    cumul += pnl_d
                    position = None; exited_all = True

                if not exited_all and row['sell']:
                    opposite_pending_exit = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited_all = False

            if not tp1_hit:
                if bar_high >= sl_price:
                    remaining = lots_a + lots_b + lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL hit'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_low <= tp1_price:
                    pnl_a = (entry_price - tp1_price) * lots_a * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp1_price, sl_price,
                        lots_a, pnl_a, 'TP1 (1:2) — 50% partial'))
                    cumul += pnl_a
                    sl_price = entry_price
                    tp1_hit  = True

            if tp1_hit and not tp2_hit and not exited_all:
                if bar_high >= sl_price:
                    remaining = lots_b + lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP1'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_low <= tp2_price:
                    pnl_b = (entry_price - tp2_price) * lots_b * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp2_price, sl_price,
                        lots_b, pnl_b, 'TP2 (1:3) — 25% partial'))
                    cumul += pnl_b
                    tp2_hit = True

            if tp2_hit and not tp3_hit and not exited_all:
                if bar_high >= sl_price:
                    remaining = lots_c + lots_d
                    pnl = (entry_price - sl_price) * remaining * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        remaining, pnl, 'SL (breakeven) — after TP2'))
                    cumul += pnl
                    position = None; exited_all = True
                elif bar_low <= tp3_price:
                    pnl_c = (entry_price - tp3_price) * lots_c * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp3_price, sl_price,
                        lots_c, pnl_c, 'TP3 (1:4) — 15% partial'))
                    cumul += pnl_c
                    tp3_hit = True

            if tp3_hit and not exited_all:
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl_d  = (entry_price - exit_p) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        lots_d, pnl_d, 'Opposite signal exit (10%)'))
                    cumul += pnl_d
                    position = None; exited_all = True
                    opposite_pending_exit = False

                if not exited_all and bar_high >= sl_price:
                    pnl_d = (entry_price - sl_price) * lots_d * contract_size
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, sl_price,
                        lots_d, pnl_d, 'SL (breakeven) — Tranche D'))
                    cumul += pnl_d
                    position = None; exited_all = True

                if not exited_all and row['buy']:
                    opposite_pending_exit = True

            if not exited_all:
                eq_pts[bar_time] = cumul

        # ── D. Record equity on flat bars ─────────────────────────────────────
        if position is None:
            eq_pts[bar_time] = cumul

        # ── E. Check EMA condition for parked pending signal ──────────────────
        if position is None and pending_direction is not None and enter_next_bar is None:
            if pending_direction == 'long' and price_above_emas:
                enter_next_bar    = dict(direction='long',
                                         sl_dist=pending_sl_dist,
                                         lot_size=pending_lot_size)
                pending_direction = None
            elif pending_direction == 'short' and price_below_emas:
                enter_next_bar    = dict(direction='short',
                                         sl_dist=pending_sl_dist,
                                         lot_size=pending_lot_size)
                pending_direction = None

        # ── F. New signals (flat, no queued entry) ────────────────────────────
        if position is None and enter_next_bar is None and pending_direction is None:
            if row['buy']:
                if price_above_emas:
                    enter_next_bar = dict(direction='long',
                                          sl_dist=row['sl_dist'],
                                          lot_size=row['lot_size'])
                else:
                    pending_direction = 'long'
                    pending_sl_dist   = row['sl_dist']
                    pending_lot_size  = row['lot_size']
            elif row['sell']:
                if price_below_emas:
                    enter_next_bar = dict(direction='short',
                                          sl_dist=row['sl_dist'],
                                          lot_size=row['lot_size'])
                else:
                    pending_direction = 'short'
                    pending_sl_dist   = row['sl_dist']
                    pending_lot_size  = row['lot_size']

        if bar_time not in eq_pts:
            eq_pts[bar_time] = cumul

    # ── G. Force-close at end of data ─────────────────────────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']

        if not tp1_hit:
            open_lots = lots_a + lots_b + lots_c + lots_d
        elif not tp2_hit:
            open_lots = lots_b + lots_c + lots_d
        elif not tp3_hit:
            open_lots = lots_c + lots_d
        else:
            open_lots = lots_d

        pnl = ((last_cl - entry_price) if position == 'long'
               else (entry_price - last_cl)) * open_lots * contract_size
        trades.append(_make_trade(entry_time, last_time, position,
                                  entry_price, last_cl, sl_price,
                                  open_lots, pnl, 'End of data'))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()
    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy11_report_{ts}.html")

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
            EMA_Fast          = ema_fast,
            EMA_Slow          = ema_slow,
            EMA_Filter        = 'Buy above EMA20&50 | Sell below EMA20&50',
            TP_Rule           = ('50% @ 1:2 (BE) → 25% @ 1:3 → '
                                 '15% @ 1:4 → 10% on opposite signal'),
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 11]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp   = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*60}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*60}")
    print(f"  Trade records  : {len(trades)}  (incl. partial closes)")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L        : ${tp:,.2f}")
    peak = equity.cummax(); dd = equity - peak
    print(f"  Max Drawdown   : ${dd.min():,.2f}")
    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=STRATEGY_NAME)
    p.add_argument('--source',     choices=['csv', 'mt5'], default='csv')
    p.add_argument('--csv',        default='XAUUSD_M15.csv')
    p.add_argument('--symbol',     default='XAUUSD')
    p.add_argument('--tf',         default='M15')
    p.add_argument('--outdir',     default='reports')
    p.add_argument('--key-value',  type=float, default=DEFAULTS['key_value'])
    p.add_argument('--atr',        type=int,   default=DEFAULTS['atr_period'])
    p.add_argument('--sl-mult',    type=float, default=DEFAULTS['sl_mult'])
    p.add_argument('--risk',       type=float, default=DEFAULTS['risk_usd'])
    p.add_argument('--contract',   type=float, default=DEFAULTS['contract_size'])
    p.add_argument('--adx-len',    type=int,   default=DEFAULTS['adx_len'])
    p.add_argument('--adx-thresh', type=int,   default=DEFAULTS['adx_thresh'])
    p.add_argument('--no-filter',  action='store_true')
    p.add_argument('--ema-fast',   type=int,   default=20)
    p.add_argument('--ema-slow',   type=int,   default=50)
    args = p.parse_args()

    df = load_data(source=args.source, csv_path=args.csv,
                   symbol=args.symbol, timeframe=args.tf)

    trades, equity, report = run_backtest(
        df,
        key_value=args.key_value, atr_period=args.atr,
        sl_mult=args.sl_mult, risk_usd=args.risk,
        contract_size=args.contract,
        adx_len=args.adx_len, adx_thresh=args.adx_thresh,
        filter_choppy=not args.no_filter,
        ema_fast=args.ema_fast, ema_slow=args.ema_slow,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()
