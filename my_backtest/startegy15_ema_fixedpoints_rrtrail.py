"""
strategy15_tp1_then_peak_rr_trail.py
======================================
Strategy 15 — Fixed TP1 (50%) then Peak RR Trail on Runner  +  EMA 21/89 Filter

This strategy merges Strategy 13 and Strategy 14:

  PHASE 1  (S13 logic — certainty):
    • Trade enters full size (same EMA filter as S13/S14).
    • TP1 fires at entry ± tp1_points (default +50 pts).
    • 50% of the position is closed at TP1.
    • Stop-loss moves to Breakeven (entry price).
    • If SL is hit before TP1 → full loss exit.
    • If opposite signal fires before TP1 → full exit at NEXT bar open.

  PHASE 2  (S14 logic — let winners run):
    • The remaining 50% (runner) is now managed by a Peak RR trail.
    • Every bar, track the highest RR the runner has ever reached.
    • Once peak RR >= min_rr_activate (default 2.0R), a trailing stop
      activates at exactly trail_pullback R below/above the peak.
    • The trail ONLY moves in your favour — it never comes back against you.
    • When price pulls back trail_pullback R from the peak → close runner.

    Runner fallback exits (in priority order):
      1. Peak RR trail triggered   (intrabar check)
      2. Breakeven SL hit          (intrabar check — guaranteed 0 net loss on runner)
      3. Opposite UTBot signal     → exit at NEXT bar open

  Example:
    Entry long at 43,000.  SL at 42,850 (150 pts = 1R).  TP1 = +50 pts.
    Bar hits 43,050 → close 50% at TP1, SL → 43,000 (BE).
    Runner tracks peak:
      Peak hits 2R (43,300) → trail activates at 1R (43,150)
      Peak hits 10R (44,500) → trail at 9R (44,350)
      Peak hits 24R (46,600) → trail at 23R (46,450)
      Price drops to 46,450 → runner closes at 46,450  ✓
    Instead of being capped at TP2 (+100 pts), runner captured +23R.

WHY THIS IS BETTER THAN S13:
  S13 caps the runner at TP2 (+100 pts) regardless of how far the move goes.
  S15 banks the same certainty at TP1, then lets the runner trail the full move.
  The 24R trade from the backtest (closed at 12R) would now close at ~23R.

EMA Filter (identical to S13):
  BUY  only when close > EMA_fast AND close > EMA_slow
  SELL only when close < EMA_fast AND close < EMA_slow
  Unaligned signals are parked and entered once EMA condition is met.

Parameters
----------
tp1_points      : Fixed points to TP1 — 50% close (default 50)
min_rr_activate : Runner trail activates once peak RR >= this (default 2.0)
trail_pullback  : Close runner when price pulls back this many R from peak (default 1.0)
ema_fast        : Fast EMA (default 21)
ema_slow        : Slow EMA (default 89)
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

STRATEGY_NAME = "Strategy 15 — TP1 Fixed (50%) + Peak RR Trail on Runner + EMA 21/89 Filter"

TP1_POINTS_DEFAULT = 50.0


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
                 tp1_points:       float = TP1_POINTS_DEFAULT,
                 min_rr_activate:  float = 2.0,
                 trail_pullback:   float = 1.0,
                 # accepted for run_all.py compatibility
                 trail_mult:       float = DEFAULTS['trail_mult'],
                 tp2_points:       float = 0.0) -> tuple:
    """
    Run Strategy 15 backtest.

    Parameters
    ----------
    df               : OHLC DataFrame (from load_data)
    key_value        : ATR multiplier for trailing-stop signal line (default 3)
    atr_period       : ATR period (default 14)
    sl_mult          : SL distance in ATR multiples (default 1.5)
    risk_usd         : USD risked per trade (default 200)
    contract_size    : Contract size (default 1 for indices)
    adx_len          : ADX period (default 14)
    adx_thresh       : Minimum ADX (default 25)
    filter_choppy    : Skip when ADX < adx_thresh (default False)
    initial_balance  : Starting equity for curve display
    output_dir       : Folder to save HTML report
    symbol           : Symbol label in report
    ema_fast         : Fast EMA period (default 21)
    ema_slow         : Slow EMA period (default 89)
    tp1_points       : Fixed price points for TP1 — 50% close (default 50)
    min_rr_activate  : Peak RR trail activates once peak >= this value (default 2.0)
    trail_pullback   : R units to trail below/above the peak (default 1.0)

    Returns
    -------
    trades       : list[dict]
    equity_curve : pd.Series
    report_path  : str
    """

    # ── 1. Compute indicators + EMAs ─────────────────────────────────────────
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
    initial_sl    = 0.0         # original SL at entry (for trade record)
    sl_price      = 0.0         # current SL (moves to BE after TP1)
    sl_dist_entry = 0.0         # 1R distance at entry
    lots          = 0.0         # full lot size at entry
    half          = 0.0         # 50% lot size (runner after TP1)
    tp1_price     = 0.0
    tp1_hit       = False

    # Peak RR trail state (runner phase)
    peak_rr       = 0.0         # highest RR seen so far this trade
    trail_exit    = None        # None = not active yet | float = trail level

    # Delayed opposite-signal exit flag (exit at next bar open)
    opposite_pending_exit = False

    # EMA-pending entry
    pending_direction = None
    pending_sl_dist   = 0.0
    pending_lot_size  = 0.0
    enter_next_bar    = None

    trades    = []
    cumul     = 0.0
    eq_pts    = {}

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

        # ── A. Open queued entry ──────────────────────────────────────────────
        if enter_next_bar is not None and position is None:
            direction     = enter_next_bar['direction']
            sl_d          = enter_next_bar['sl_dist']
            lots          = enter_next_bar['lot_size']
            half          = lots / 2.0
            position      = direction
            entry_price   = bar_open
            entry_time    = bar_time
            sl_dist_entry = sl_d
            peak_rr       = 0.0
            trail_exit    = None

            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            tp1_price             = (entry_price + tp1_points) if direction == 'long' \
                                    else (entry_price - tp1_points)
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

            # Track peak RR using bar high (full trade perspective)
            current_bar_rr = (bar_high - entry_price) / sl_dist_entry \
                             if sl_dist_entry > 0 else 0.0
            if current_bar_rr > peak_rr:
                peak_rr = current_bar_rr

            h_rr = round(peak_rr, 2)

            # ── PHASE 1: Before TP1 ───────────────────────────────────────────
            if not tp1_hit:

                # Delayed opposite-signal exit (flagged previous bar)
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl    = (exit_p - entry_price) * lots * contract_size
                    c_rr   = round((exit_p - entry_price) / sl_dist_entry, 2) \
                             if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, initial_sl,
                        lots, pnl, 'Opposite signal — full exit (pre-TP1)',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul += pnl
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # SL hit
                if not exited and bar_low <= sl_price:
                    pnl  = (sl_price - entry_price) * lots * contract_size
                    c_rr = round((sl_price - entry_price) / sl_dist_entry, 2) \
                           if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul  += pnl
                    position = None
                    exited   = True

                # TP1 hit → close 50%, move SL to BE, enter runner phase
                if not exited and bar_high >= tp1_price:
                    pnl_half = (tp1_price - entry_price) * half * contract_size
                    c_rr     = round((tp1_price - entry_price) / sl_dist_entry, 2) \
                               if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half,
                        f'TP1 (+{tp1_points:.0f}pts) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_half
                    sl_price = entry_price          # SL → Breakeven
                    tp1_hit  = True
                    opposite_pending_exit = False   # reset flag

                # Flag opposite signal for next-bar exit
                if not exited and not tp1_hit and row['sell']:
                    opposite_pending_exit = True

            # ── PHASE 2: Runner managed by Peak RR trail ──────────────────────
            if tp1_hit and not exited:

                # Update trail exit — only moves UP
                if peak_rr >= min_rr_activate:
                    new_trail = entry_price + (peak_rr - trail_pullback) * sl_dist_entry
                    if trail_exit is None:
                        trail_exit = new_trail
                    else:
                        trail_exit = max(trail_exit, new_trail)

                # Delayed opposite-signal exit (flagged previous bar)
                if opposite_pending_exit:
                    exit_p  = bar_open
                    pnl_run = (exit_p - entry_price) * half * contract_size
                    c_rr    = round((exit_p - entry_price) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner exit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # Peak RR trail triggered
                if not exited and trail_exit is not None and bar_low <= trail_exit:
                    exit_p  = trail_exit
                    pnl_run = (exit_p - entry_price) * half * contract_size
                    c_rr    = round((exit_p - entry_price) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        half, pnl_run,
                        f'Peak RR trail — runner exit (peak={peak_rr:.1f}R → closed={c_rr:.1f}R)',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                # Breakeven SL hit (trail not yet active or price gapped)
                if not exited and bar_low <= sl_price:
                    pnl_run = (sl_price - entry_price) * half * contract_size
                    c_rr    = round((sl_price - entry_price) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner exit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                # Flag opposite signal for next-bar exit
                if not exited and row['sell']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited = False

            current_bar_rr = (entry_price - bar_low) / sl_dist_entry \
                             if sl_dist_entry > 0 else 0.0
            if current_bar_rr > peak_rr:
                peak_rr = current_bar_rr

            h_rr = round(peak_rr, 2)

            # ── PHASE 1: Before TP1 ───────────────────────────────────────────
            if not tp1_hit:

                # Delayed opposite-signal exit
                if opposite_pending_exit:
                    exit_p = bar_open
                    pnl    = (entry_price - exit_p) * lots * contract_size
                    c_rr   = round((entry_price - exit_p) / sl_dist_entry, 2) \
                             if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, initial_sl,
                        lots, pnl, 'Opposite signal — full exit (pre-TP1)',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul  += pnl
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # SL hit
                if not exited and bar_high >= sl_price:
                    pnl  = (entry_price - sl_price) * lots * contract_size
                    c_rr = round((entry_price - sl_price) / sl_dist_entry, 2) \
                           if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul  += pnl
                    position = None
                    exited   = True

                # TP1 hit → close 50%, SL → BE
                if not exited and bar_low <= tp1_price:
                    pnl_half = (entry_price - tp1_price) * half * contract_size
                    c_rr     = round((entry_price - tp1_price) / sl_dist_entry, 2) \
                               if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half,
                        f'TP1 (+{tp1_points:.0f}pts) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_half
                    sl_price = entry_price          # SL → Breakeven
                    tp1_hit  = True
                    opposite_pending_exit = False

                if not exited and not tp1_hit and row['buy']:
                    opposite_pending_exit = True

            # ── PHASE 2: Runner managed by Peak RR trail ──────────────────────
            if tp1_hit and not exited:

                # Update trail exit — only moves DOWN (short direction)
                if peak_rr >= min_rr_activate:
                    new_trail = entry_price - (peak_rr - trail_pullback) * sl_dist_entry
                    if trail_exit is None:
                        trail_exit = new_trail
                    else:
                        trail_exit = min(trail_exit, new_trail)

                # Delayed opposite-signal exit
                if opposite_pending_exit:
                    exit_p  = bar_open
                    pnl_run = (entry_price - exit_p) * half * contract_size
                    c_rr    = round((entry_price - exit_p) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner exit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # Peak RR trail triggered
                if not exited and trail_exit is not None and bar_high >= trail_exit:
                    exit_p  = trail_exit
                    pnl_run = (entry_price - exit_p) * half * contract_size
                    c_rr    = round((entry_price - exit_p) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        half, pnl_run,
                        f'Peak RR trail — runner exit (peak={peak_rr:.1f}R → closed={c_rr:.1f}R)',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                # Breakeven SL hit
                if not exited and bar_high >= sl_price:
                    pnl_run = (entry_price - sl_price) * half * contract_size
                    c_rr    = round((entry_price - sl_price) / sl_dist_entry, 2) \
                              if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner exit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                # Flag opposite signal
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

    # ── G. Force-close open position at end of data ───────────────────────────
    if position is not None:
        last_row  = sig.iloc[-1]
        last_time = sig.index[-1]
        last_cl   = last_row['close']
        remaining = half if tp1_hit else lots
        h_rr      = round(peak_rr, 2)
        if position == 'long':
            pnl  = (last_cl - entry_price) * remaining * contract_size
            c_rr = round((last_cl - entry_price) / sl_dist_entry, 2) \
                   if sl_dist_entry > 0 else 0.0
        else:
            pnl  = (entry_price - last_cl) * remaining * contract_size
            c_rr = round((entry_price - last_cl) / sl_dist_entry, 2) \
                   if sl_dist_entry > 0 else 0.0
        trades.append(_make_trade(entry_time, last_time, position,
                                  entry_price, last_cl, sl_price,
                                  remaining, pnl, 'End of data',
                                  highest_rr=h_rr, closed_rr=c_rr))
        cumul += pnl
        eq_pts[last_time] = cumul

    equity_curve = pd.Series(eq_pts).sort_index()
    _print_summary(trades, equity_curve, tp1_points, min_rr_activate, trail_pullback)

    os.makedirs(output_dir, exist_ok=True)
    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"strategy15_report_{ts}.html")

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
            TP1_Points       = f'+{tp1_points} pts → 50% close + SL to BE',
            Min_RR_Activate  = f'{min_rr_activate}R  (trail starts on runner)',
            Trail_Pullback   = f'{trail_pullback}R  (runner closes on pullback from peak)',
            Phase_1_Exit     = f'SL hit (full) | TP1 +{tp1_points}pts (50% partial)',
            Phase_2_Exit     = f'Peak RR trail | BE SL | Opposite signal',
            EMA_Filter       = f'Buy above EMA{ema_fast}&{ema_slow} | Sell below',
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity, tp1_points, min_rr_activate, trail_pullback):
    if not trades:
        print("[Strategy 15]  No trades generated.")
        return

    wins        = [t for t in trades if t['pnl'] > 0]
    total_pnl   = sum(t['pnl'] for t in trades)
    tp1_exits   = [t for t in trades if 'TP1'        in t.get('exit_reason', '')]
    trail_exits = [t for t in trades if 'Peak RR'    in t.get('exit_reason', '')]
    be_exits    = [t for t in trades if 'breakeven'  in t.get('exit_reason', '').lower()]
    opp_exits   = [t for t in trades if 'Opposite'   in t.get('exit_reason', '')]
    sl_exits    = [t for t in trades if t.get('exit_reason') == 'SL hit']

    peak  = equity.cummax()
    dd    = equity - peak

    print(f"\n{'─'*68}")
    print(f"  {STRATEGY_NAME}")
    print(f"  TP1={tp1_points}pts  |  Trail activates at {min_rr_activate}R  |  Pullback={trail_pullback}R")
    print(f"{'─'*68}")
    print(f"  Trade records  : {len(trades)}  (incl. partial closes)")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Net P&L        : ${total_pnl:,.2f}")
    print(f"  Max Drawdown   : ${dd.min():,.2f}")
    print(f"  Exit breakdown :")
    print(f"    TP1 partial      : {len(tp1_exits)}  ({len(tp1_exits)/len(trades)*100:.1f}%)")
    print(f"    Peak RR trail    : {len(trail_exits)}  ({len(trail_exits)/len(trades)*100:.1f}%)")
    print(f"    Breakeven SL     : {len(be_exits)}  ({len(be_exits)/len(trades)*100:.1f}%)")
    print(f"    Opposite signal  : {len(opp_exits)}  ({len(opp_exits)/len(trades)*100:.1f}%)")
    print(f"    SL hit (full)    : {len(sl_exits)}  ({len(sl_exits)/len(trades)*100:.1f}%)")
    print(f"{'─'*68}\n")


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
    p.add_argument('--tp1',              type=float, default=TP1_POINTS_DEFAULT)
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
        tp1_points=args.tp1,
        min_rr_activate=args.min_rr_activate,
        trail_pullback=args.trail_pullback,
        output_dir=args.outdir, symbol=args.symbol,
    )
    print(f"  Report saved : {report}\n")


if __name__ == '__main__':
    main()