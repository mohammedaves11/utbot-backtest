"""
strategy14_peak_rr_trail.py
============================
Strategy 14 — Peak RR Trailing Exit  +  EMA 21/89 Trend Filter

Exit logic:
  - Track the highest RR (reward:risk) reached during the trade.
  - Once price reaches a configurable minimum RR (default 2.0R), activate
    a trailing stop that sits exactly 1R below/above the peak RR.
  - Every time a new peak is set, the trail moves up with it — it NEVER moves down.
  - When price pulls back 1R from the peak → close the ENTIRE position at that level.

  Example (long trade):
    Peak hits 5R  → trail sits at 4R
    Peak hits 10R → trail moves up to 9R
    Peak hits 24R → trail moves up to 23R
    Price drops to 23R → close at 23R  ✓

  Fallback exits (in priority order):
    1. Peak RR trail triggered  (primary exit once peak >= min_rr_activate)
    2. SL hit                   (if trail never activated)
    3. Opposite UTBot signal    (if trail never activated and price drifts)

EMA Filter (same structure as Strategies 10–13):
  BUY  only when close > EMA_fast AND close > EMA_slow
  SELL only when close < EMA_fast AND close < EMA_slow
  Pending signals are parked and entered once EMA condition is met.

Parameters
----------
ema_fast        : Fast EMA period  (default 21)
ema_slow        : Slow EMA period  (default 89)
min_rr_activate : Trail activates only after this RR is first reached (default 2.0)
trail_pullback  : How many R to trail below/above the peak (default 1.0)
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

STRATEGY_NAME = "Strategy 14 — Peak RR Trailing Exit + EMA 21/89 Filter"


# ══════════════════════════════════════════════════════════════════════════════
def _compute_emas(df: pd.DataFrame, fast: int = 21, slow: int = 89):
    """Return EMA-fast and EMA-slow Series aligned to df.index."""
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
                 ema_fast:         int   = 21,
                 ema_slow:         int   = 89,
                 min_rr_activate:  float = 2.0,   # trail activates after this RR is reached
                 trail_pullback:   float = 1.0,   # R to trail below/above peak
                 # accepted for run_all.py compatibility
                 trail_mult:       float = DEFAULTS['trail_mult'],
                 tp1_points:       float = 0.0,
                 tp2_points:       float = 0.0) -> tuple:
    """
    Run Strategy 14 backtest.

    Parameters
    ----------
    df               : OHLC DataFrame (from load_data)
    key_value        : ATR multiplier for trailing-stop signal line (default 3)
    atr_period       : ATR period (default 14)
    sl_mult          : SL distance in ATR multiples (default 1.5)
    risk_usd         : USD risked per trade (default 200)
    contract_size    : Contract size (default 1 for indices)
    adx_len          : ADX smoothing period (default 14)
    adx_thresh       : Minimum ADX for a valid trend (default 25)
    filter_choppy    : Skip trades when ADX < adx_thresh (default False)
    initial_balance  : Starting equity for equity-curve display
    output_dir       : Folder to save the HTML report
    symbol           : Symbol label used in the report
    ema_fast         : Fast EMA period (default 21)
    ema_slow         : Slow EMA period (default 89)
    min_rr_activate  : Trail only activates once peak RR >= this value (default 2.0)
    trail_pullback   : R units to trail below/above the peak RR (default 1.0)

    Returns
    -------
    trades       : list[dict]
    equity_curve : pd.Series
    report_path  : str
    """

    # ── 1. Compute base indicators + EMA lines ────────────────────────────────
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

    # ── 2. Simulation state ───────────────────────────────────────────────────
    position      = None        # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    sl_price      = 0.0
    sl_dist_entry = 0.0
    lots          = 0.0
    peak_rr       = 0.0         # highest RR reached so far this trade
    trail_exit    = None        # None = trail not yet active; float = current trail level

    # Pending entry: signal fired, waiting for EMA condition + next bar open
    pending_direction = None
    pending_sl_dist   = 0.0
    pending_lot_size  = 0.0
    enter_next_bar    = None    # dict ready to open on THIS bar's open

    trades     = []
    cumul_pnl  = 0.0
    equity_pts = {}

    idx = sig.index
    n   = len(sig)

    for i in range(warmup, n):
        row       = sig.iloc[i]
        bar_time  = idx[i]
        bar_open  = row['open']
        bar_high  = row['high']
        bar_low   = row['low']
        bar_close = row['close']
        ema_f     = row['ema_fast']
        ema_s     = row['ema_slow']

        price_above_emas = bar_close > ema_f and bar_close > ema_s
        price_below_emas = bar_close < ema_f and bar_close < ema_s

        # ── A. Open entry queued from previous bar ────────────────────────────
        if enter_next_bar is not None and position is None:
            direction     = enter_next_bar['direction']
            sl_d          = enter_next_bar['sl_dist']
            position      = direction
            entry_price   = bar_open
            entry_time    = bar_time
            lots          = enter_next_bar['lot_size']
            sl_dist_entry = sl_d
            peak_rr       = 0.0
            trail_exit    = None
            sl_price      = (entry_price - sl_d) if direction == 'long' \
                            else (entry_price + sl_d)
            enter_next_bar    = None
            pending_direction = None

        elif enter_next_bar is not None:
            # Already in a position — discard queued entry
            enter_next_bar    = None
            pending_direction = None

        # ── B. Manage open position ───────────────────────────────────────────
        exited = False

        if position == 'long':
            # Update peak RR using bar high
            current_bar_rr = (bar_high - entry_price) / sl_dist_entry \
                             if sl_dist_entry > 0 else 0.0
            if current_bar_rr > peak_rr:
                peak_rr = current_bar_rr

            h_rr = round(peak_rr, 2)

            # Update trail exit level (only moves UP, never down)
            if peak_rr >= min_rr_activate:
                new_trail = entry_price + (peak_rr - trail_pullback) * sl_dist_entry
                if trail_exit is None:
                    trail_exit = new_trail
                else:
                    trail_exit = max(trail_exit, new_trail)

            # B1. Peak RR trail triggered (bar low touched or crossed trail level)
            if trail_exit is not None and bar_low <= trail_exit:
                exit_price = trail_exit
                pnl  = (exit_price - entry_price) * lots * contract_size
                c_rr = round((exit_price - entry_price) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'long',
                    entry_price, exit_price, sl_price,
                    lots, pnl,
                    f'Peak RR trail (peak={peak_rr:.1f}R closed={c_rr:.1f}R)',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position   = None
                exited     = True

            # B2. SL hit (trail not yet active, or price gapped through SL)
            elif bar_low <= sl_price:
                pnl  = (sl_price - entry_price) * lots * contract_size
                c_rr = round((sl_price - entry_price) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'long',
                    entry_price, sl_price, sl_price,
                    lots, pnl, 'SL hit',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True

            # B3. Opposite signal fallback (trail not yet triggered)
            elif row['sell']:
                pnl  = (bar_close - entry_price) * lots * contract_size
                c_rr = round((bar_close - entry_price) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'long',
                    entry_price, bar_close, sl_price,
                    lots, pnl, 'Opposite signal',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True
                # Queue new short — subject to EMA filter
                if price_below_emas:
                    enter_next_bar = dict(direction='short',
                                         sl_dist=row['sl_dist'],
                                         lot_size=row['lot_size'])
                else:
                    pending_direction = 'short'
                    pending_sl_dist   = row['sl_dist']
                    pending_lot_size  = row['lot_size']

        elif position == 'short':
            # Update peak RR using bar low
            current_bar_rr = (entry_price - bar_low) / sl_dist_entry \
                             if sl_dist_entry > 0 else 0.0
            if current_bar_rr > peak_rr:
                peak_rr = current_bar_rr

            h_rr = round(peak_rr, 2)

            # Update trail exit level (only moves DOWN, never up)
            if peak_rr >= min_rr_activate:
                new_trail = entry_price - (peak_rr - trail_pullback) * sl_dist_entry
                if trail_exit is None:
                    trail_exit = new_trail
                else:
                    trail_exit = min(trail_exit, new_trail)

            # B1. Peak RR trail triggered (bar high touched or crossed trail level)
            if trail_exit is not None and bar_high >= trail_exit:
                exit_price = trail_exit
                pnl  = (entry_price - exit_price) * lots * contract_size
                c_rr = round((entry_price - exit_price) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'short',
                    entry_price, exit_price, sl_price,
                    lots, pnl,
                    f'Peak RR trail (peak={peak_rr:.1f}R closed={c_rr:.1f}R)',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True

            # B2. SL hit
            elif bar_high >= sl_price:
                pnl  = (entry_price - sl_price) * lots * contract_size
                c_rr = round((entry_price - sl_price) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'short',
                    entry_price, sl_price, sl_price,
                    lots, pnl, 'SL hit',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True

            # B3. Opposite signal fallback
            elif row['buy']:
                pnl  = (entry_price - bar_close) * lots * contract_size
                c_rr = round((entry_price - bar_close) / sl_dist_entry, 2) \
                       if sl_dist_entry > 0 else 0.0
                trades.append(_make_trade(
                    entry_time, bar_time, 'short',
                    entry_price, bar_close, sl_price,
                    lots, pnl, 'Opposite signal',
                    highest_rr=h_rr, closed_rr=c_rr))
                cumul_pnl += pnl
                equity_pts[bar_time] = cumul_pnl
                position = None
                exited   = True
                # Queue new long — subject to EMA filter
                if price_above_emas:
                    enter_next_bar = dict(direction='long',
                                         sl_dist=row['sl_dist'],
                                         lot_size=row['lot_size'])
                else:
                    pending_direction = 'long'
                    pending_sl_dist   = row['sl_dist']
                    pending_lot_size  = row['lot_size']

        # ── C. Check EMA condition for pending signal (flat or just exited) ───
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

        # ── D. New signals (only when flat and no pending entry) ──────────────
        if position is None and enter_next_bar is None \
                and pending_direction is None and not exited:
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

        if bar_time not in equity_pts:
            equity_pts[bar_time] = cumul_pnl

    # ── E. Force-close any open position at end of data ──────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        h_rr      = round(peak_rr, 2)
        if position == 'long':
            pnl  = (last_cl - entry_price) * lots * contract_size
            c_rr = round((last_cl - entry_price) / sl_dist_entry, 2) \
                   if sl_dist_entry > 0 else 0.0
            trades.append(_make_trade(entry_time, last_time, 'long',
                                      entry_price, last_cl, sl_price,
                                      lots, pnl, 'End of data',
                                      highest_rr=h_rr, closed_rr=c_rr))
        else:
            pnl  = (entry_price - last_cl) * lots * contract_size
            c_rr = round((entry_price - last_cl) / sl_dist_entry, 2) \
                   if sl_dist_entry > 0 else 0.0
            trades.append(_make_trade(entry_time, last_time, 'short',
                                      entry_price, last_cl, sl_price,
                                      lots, pnl, 'End of data',
                                      highest_rr=h_rr, closed_rr=c_rr))
        cumul_pnl += pnl
        equity_pts[last_time] = cumul_pnl

    equity_curve = pd.Series(equity_pts).sort_index()

    _print_summary(trades, equity_curve, min_rr_activate, trail_pullback)

    os.makedirs(output_dir, exist_ok=True)
    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy14_report_{ts}.html")

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
            EMA_Fast         = ema_fast,
            EMA_Slow         = ema_slow,
            Min_RR_Activate  = f'{min_rr_activate}R  (trail starts here)',
            Trail_Pullback   = f'{trail_pullback}R  (closes when price pulls back this far from peak)',
            TP_Primary       = 'Peak RR trail — 1R pullback from highest RR reached',
            TP_Fallback      = 'Opposite UTBot signal',
            SL               = 'ATR-based SL from entry bar',
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity, min_rr_activate, trail_pullback):
    if not trades:
        print("[Strategy 14]  No trades generated.")
        return
    wins         = [t for t in trades if t['pnl'] > 0]
    total_pnl    = sum(t['pnl'] for t in trades)
    trail_exits  = [t for t in trades if 'Peak RR trail' in t.get('exit_reason', '')]
    opp_exits    = [t for t in trades if t.get('exit_reason') == 'Opposite signal']
    sl_exits     = [t for t in trades if t.get('exit_reason') == 'SL hit']
    peak         = equity.cummax()
    dd           = equity - peak

    print(f"\n{'─'*64}")
    print(f"  {STRATEGY_NAME}")
    print(f"  Trail activates at {min_rr_activate}R | Pullback trigger: {trail_pullback}R")
    print(f"{'─'*64}")
    print(f"  Total trades   : {len(trades)}")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L        : ${total_pnl:,.2f}")
    print(f"  Max Drawdown   : ${dd.min():,.2f}")
    print(f"  Exit breakdown :")
    print(f"    Peak RR trail  : {len(trail_exits)}  ({len(trail_exits)/len(trades)*100:.1f}%)")
    print(f"    Opposite signal: {len(opp_exits)}  ({len(opp_exits)/len(trades)*100:.1f}%)")
    print(f"    SL hit         : {len(sl_exits)}  ({len(sl_exits)/len(trades)*100:.1f}%)")
    print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=STRATEGY_NAME)
    p.add_argument('--source',           choices=['csv', 'mt5'], default='csv')
    p.add_argument('--csv',              default='XAUUSD_M15.csv')
    p.add_argument('--symbol',           default='US30')
    p.add_argument('--tf',               default='M15')
    p.add_argument('--outdir',           default='reports')
    p.add_argument('--key-value',        type=float, default=DEFAULTS['key_value'])
    p.add_argument('--atr',              type=int,   default=DEFAULTS['atr_period'])
    p.add_argument('--sl-mult',          type=float, default=DEFAULTS['sl_mult'])
    p.add_argument('--risk',             type=float, default=DEFAULTS['risk_usd'])
    p.add_argument('--contract',         type=float, default=DEFAULTS['contract_size'])
    p.add_argument('--adx-len',          type=int,   default=DEFAULTS['adx_len'])
    p.add_argument('--adx-thresh',       type=int,   default=DEFAULTS['adx_thresh'])
    p.add_argument('--no-filter',        action='store_true')
    p.add_argument('--ema-fast',         type=int,   default=21)
    p.add_argument('--ema-slow',         type=int,   default=89)
    p.add_argument('--min-rr-activate',  type=float, default=2.0)
    p.add_argument('--trail-pullback',   type=float, default=1.0)
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
        min_rr_activate=args.min_rr_activate,
        trail_pullback=args.trail_pullback,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()