"""
strategy13_ema_fixed_points_opposite.py
=========================================
Strategy 13 — Fixed +50 / +100 pt TPs + Opposite Signal  +  EMA 20/50 Filter

Identical exit logic to Strategy 9, but adds a dual-EMA entry filter:

  BUY  signal : price must be ABOVE both EMA-20 and EMA-50 to enter.
                If not aligned, park the signal and wait until price crosses
                above both EMAs, then enter at the NEXT bar's open.

  SELL signal : price must be BELOW both EMA-20 and EMA-50 to enter.
                If not aligned, park and wait.

CLOSE SCHEDULE (same as Strategy 9)
--------------------------------------
  TP1 : entry ± 50 price points  → Close 50 %, move SL to Breakeven.
  TP2 : entry ± 100 price points → Close remaining 50 %.
  If neither TP hit and opposite signal fires → exit at NEXT bar's open.
  SL  : entry ± (sl_mult × ATR)

Usage
-----
from backtest_utils import load_data
from strategy13_ema_fixed_points_opposite import run_backtest

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

STRATEGY_NAME = ("Strategy 13 — Fixed +50/+100 pt TPs, Opposite Signal "
                 "+ EMA 20/50 Filter")

TP1_POINTS = 50.0
TP2_POINTS = 100.0


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
                 ema_slope_bars:   int   = 3,     # bars to measure EMA slope over
                 min_atr_ratio:    float = 0.8,   # ATR must be >= 80% of its 20-bar avg
                 trail_mult:       float = DEFAULTS['trail_mult'],
                 tp1_points:       float = TP1_POINTS,
                 tp2_points:       float = TP2_POINTS) -> tuple:
    """Run Strategy 13 backtest (Strategy 9 + EMA 20/50 filter)."""

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
    half          = 0.0
    tp1_price     = 0.0
    tp2_price     = 0.0
    tp1_hit       = False
    max_excursion = 0.0

    opposite_pending_exit = False

    pending_direction = None
    pending_sl_dist   = 0.0
    pending_lot_size  = 0.0
    enter_next_bar    = None

    trades = []
    cumul  = 0.0
    eq_pts = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row      = sig.iloc[i]
        bar_time = idx[i]
        bar_open = row['open']
        bar_high = row['high']
        bar_low  = row['low']
        bar_close= row['close']

        ema_f = row['ema_fast']
        ema_s = row['ema_slow']
        price_above_emas = bar_close > ema_f and bar_close > ema_s
        price_below_emas = bar_close < ema_f and bar_close < ema_s

        # ── Filter 4: EMA slope — slow EMA must slope in trade direction ────────
        slope_i      = i - ema_slope_bars
        ema_slope    = row['ema_slow'] - sig['ema_slow'].iloc[slope_i] \
                       if slope_i >= 0 else 0.0
        slope_long   = ema_slope > 0    # slow EMA rising  → allow longs
        slope_short  = ema_slope < 0    # slow EMA falling → allow shorts

        # ── Filter 5: ATR minimum — skip if market is too quiet/compressed ──────
        atr_ma_val   = sig['atr_ma'].iloc[i]
        atr_ok       = (row['atr'] >= min_atr_ratio * atr_ma_val) \
                       if atr_ma_val > 0 else True

        # ── A. Open entry queued from previous bar ────────────────────────────
        if enter_next_bar is not None and position is None:
            direction      = enter_next_bar['direction']
            sl_d           = enter_next_bar['sl_dist']
            lots           = enter_next_bar['lot_size']
            half           = lots / 2.0
            position       = direction
            entry_price    = bar_open
            entry_time     = bar_time
            sl_dist_entry  = sl_d
            max_excursion  = 0.0

            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            if direction == 'long':
                tp1_price = entry_price + tp1_points
                tp2_price = entry_price + tp2_points
            else:
                tp1_price = entry_price - tp1_points
                tp2_price = entry_price - tp2_points

            tp1_hit               = False
            opposite_pending_exit = False
            enter_next_bar        = None
            pending_direction     = None

        elif enter_next_bar is not None:
            enter_next_bar    = None
            pending_direction = None

        # ── B. Manage open LONG position ──────────────────────────────────────
        if position == 'long':
            exited = False
            max_excursion = max(max_excursion, bar_high - entry_price)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            if not tp1_hit:
                # Before TP1: SL or opposite signal (full exit) or TP1
                if bar_low <= sl_price:
                    pnl  = (sl_price - entry_price) * lots * contract_size
                    c_rr = round((sl_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl; position = None; exited = True

                elif bar_high >= tp1_price:
                    pnl_half = (tp1_price - entry_price) * half * contract_size
                    c_rr     = round((tp1_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half, f'TP1 (+{tp1_points:.0f}pts) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_half
                    sl_price = entry_price
                    tp1_hit  = True
                    opposite_pending_exit = False

                else:
                    if row['sell']:
                        opposite_pending_exit = True

            if not tp1_hit and not exited and opposite_pending_exit:
                # Exit full position at bar open (delayed 1 bar)
                exit_p = bar_open
                pnl    = (exit_p - entry_price) * lots * contract_size
                c_rr   = round((exit_p - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'long',
                    entry_price, exit_p, sl_price,
                    lots, pnl, 'Opposite signal — full exit',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul += pnl; position = None; exited = True
                opposite_pending_exit = False

            if tp1_hit and not exited:
                if opposite_pending_exit:
                    exit_p  = bar_open
                    pnl_run = (exit_p - entry_price) * half * contract_size
                    c_rr    = round((exit_p - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True
                    opposite_pending_exit = False

                if not exited and bar_low <= sl_price:
                    pnl_run = (sl_price - entry_price) * half * contract_size
                    c_rr    = round((sl_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True

                if not exited and bar_high >= tp2_price:
                    pnl_run = (tp2_price - entry_price) * half * contract_size
                    c_rr    = round((tp2_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp2_price, sl_price,
                        half, pnl_run, f'TP2 (+{tp2_points:.0f}pts) — 50% close',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True

                if not exited and row['sell']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited = False
            max_excursion = max(max_excursion, entry_price - bar_low)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            if not tp1_hit:
                if bar_high >= sl_price:
                    pnl  = (entry_price - sl_price) * lots * contract_size
                    c_rr = round((entry_price - sl_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl; position = None; exited = True

                elif bar_low <= tp1_price:
                    pnl_half = (entry_price - tp1_price) * half * contract_size
                    c_rr     = round((entry_price - tp1_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half, f'TP1 (+{tp1_points:.0f}pts) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_half
                    sl_price = entry_price
                    tp1_hit  = True
                    opposite_pending_exit = False

                else:
                    if row['buy']:
                        opposite_pending_exit = True

            if not tp1_hit and not exited and opposite_pending_exit:
                exit_p = bar_open
                pnl    = (entry_price - exit_p) * lots * contract_size
                c_rr   = round((entry_price - exit_p) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'short',
                    entry_price, exit_p, sl_price,
                    lots, pnl, 'Opposite signal — full exit',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul += pnl; position = None; exited = True
                opposite_pending_exit = False

            if tp1_hit and not exited:
                if opposite_pending_exit:
                    exit_p  = bar_open
                    pnl_run = (entry_price - exit_p) * half * contract_size
                    c_rr    = round((entry_price - exit_p) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True
                    opposite_pending_exit = False

                if not exited and bar_high >= sl_price:
                    pnl_run = (entry_price - sl_price) * half * contract_size
                    c_rr    = round((entry_price - sl_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True

                if not exited and bar_low <= tp2_price:
                    pnl_run = (entry_price - tp2_price) * half * contract_size
                    c_rr    = round((entry_price - tp2_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp2_price, sl_price,
                        half, pnl_run, f'TP2 (+{tp2_points:.0f}pts) — 50% close',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl_run; position = None; exited = True

                if not exited and row['buy']:
                    opposite_pending_exit = True

            if not exited:
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
            if row['buy'] and atr_ok and slope_long:
                if price_above_emas:
                    enter_next_bar = dict(direction='long',
                                          sl_dist=row['sl_dist'],
                                          lot_size=row['lot_size'])
                else:
                    pending_direction = 'long'
                    pending_sl_dist   = row['sl_dist']
                    pending_lot_size  = row['lot_size']
            elif row['sell'] and atr_ok and slope_short:
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

    # ── G. Force-close open position at end of data ───────────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        remaining = half if tp1_hit else lots
        h_rr      = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
        if position == 'long':
            pnl  = (last_cl - entry_price) * remaining * contract_size
            c_rr = round((last_cl - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
        else:
            pnl  = (entry_price - last_cl) * remaining * contract_size
            c_rr = round((entry_price - last_cl) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
        trades.append(_make_trade(entry_time, last_time, position,
                                  entry_price, last_cl, sl_price,
                                  remaining, pnl, 'End of data',
                                  highest_rr=h_rr, closed_rr=c_rr))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()
    _print_summary(trades, equity_curve)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy13_report_{ts}.html")

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
            EMA_Fast         = ema_fast,
            EMA_Slow         = ema_slow,
            EMA_Filter       = 'Buy above EMA20&50 | Sell below EMA20&50',
            TP1_Points       = f'+{tp1_points} price pts (50% close)',
            TP2_Points       = f'+{tp2_points} price pts (50% close)',
            TP_Rule          = ('50%@+50pts → BE → 50%@+100pts '
                                '(or opposite signal fallback)'),
            Total_Bars       = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 13]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp   = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*62}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*62}")
    print(f"  Trade records : {len(trades)}  (incl. partial closes)")
    print(f"  Win rate      : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L       : ${tp:,.2f}")
    peak = equity.cummax(); dd = equity - peak
    print(f"  Max Drawdown  : ${dd.min():,.2f}")
    print(f"{'─'*62}\n")


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
    p.add_argument('--no-filter',       action='store_true')
    p.add_argument('--ema-fast',        type=int,   default=20)
    p.add_argument('--ema-slow',        type=int,   default=50)
    p.add_argument('--tp1',             type=float, default=TP1_POINTS)
    p.add_argument('--tp2',             type=float, default=TP2_POINTS)
    p.add_argument('--ema-slope-bars',  type=int,   default=3)
    p.add_argument('--min-atr-ratio',   type=float, default=0.8)
    args = p.parse_args()

    df = load_data(source=args.source, csv_path=args.csv,
                   symbol=args.symbol, timeframe=args.tf)

    trades, equity, report = run_backtest(
        df,
        key_value       = args.key_value,
        atr_period      = args.atr,
        sl_mult         = args.sl_mult,
        risk_usd        = args.risk,
        contract_size   = args.contract,
        adx_len         = args.adx_len,
        adx_thresh      = args.adx_thresh,
        filter_choppy   = not args.no_filter,
        ema_fast        = args.ema_fast,
        ema_slow        = args.ema_slow,
        tp1_points      = args.tp1,
        tp2_points      = args.tp2,
        ema_slope_bars  = args.ema_slope_bars,
        min_atr_ratio   = args.min_atr_ratio,
        output_dir      = args.outdir,
        symbol          = args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()