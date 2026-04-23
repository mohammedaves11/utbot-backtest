"""
strategy8_half_tp_be_opposite_exit.py
=======================================
Strategy 8 — 50 % at 1:2 (BE), Hold Runner Until Opposite Signal

EXIT RULES
----------
  Before 1:2 : SL at entry ± (sl_mult × ATR).
                If hit → full position exits at SL price (full loss).

  At 1:2     : Close 50 % of position at TP1 price.
                Move SL to BREAKEVEN (entry price) on remaining 50 %.

  After 1:2  : Hold the remaining 50 % (the "runner") with NO trail cap.
                • If SL (breakeven) is hit → runner exits at entry (scratch).
                • When an OPPOSITE signal fires on a bar close → exit runner
                  at the OPEN of the NEXT bar (1-bar delay).

END OF DATA : Any open portion is closed at the last bar's close.

SAME-BAR PRIORITY (before 1:2)
--------------------------------
  If bar_low <= SL  AND  bar_high >= TP1 on the same bar → SL treated as
  triggered first (conservative).

TRADE-LOG ROWS
--------------
  Row 1 — 50 % closed at TP1   (exit_reason = 'TP1 (1:2) — 50% partial')
  Row 2 — 50 % closed at opp.  (exit_reason = 'Opposite signal — runner 50%')
        or at BE SL             (exit_reason = 'SL (breakeven) — runner 50%')
        or at end of data       (exit_reason = 'End of data')

Usage
-----
from backtest_utils import load_data
from strategy8_half_tp_be_opposite_exit import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy8_half_tp_be_opposite_exit.py --source csv --csv XAUUSD_M15.csv
python strategy8_half_tp_be_opposite_exit.py --source mt5 --symbol XAUUSD --tf M15
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

STRATEGY_NAME = ("Strategy 8 — 50% at 1:2 (BE), "
                 "Hold Runner 50% Until Opposite Signal")


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
                 trail_mult:       float = DEFAULTS['trail_mult']) -> tuple:
    """
    Run Strategy 8 backtest.

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
    initial_balance : Starting equity for equity-curve display
    output_dir    : Folder to save the HTML report
    symbol        : Symbol label used in the report
    trail_mult    : Accepted for API compatibility; not used in this strategy

    Returns
    -------
    trades        : list[dict]  — one dict per closed partial or full trade
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
    initial_sl    = 0.0
    sl_price      = 0.0         # active SL — moves to BE after TP1
    sl_dist_entry = 0.0
    lots          = 0.0
    half          = 0.0
    tp1_price     = 0.0
    tp1_hit       = False
    max_excursion = 0.0

    opposite_pending_exit = False

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
        bar_atr  = row['atr']

        # ── A. Fill pending ENTRY at this bar's open ──────────────────────────
        if pending is not None and position is None:
            direction      = pending['direction']
            sl_d           = pending['sl_dist']
            lots           = pending['lot_size']
            half           = lots / 2.0
            position       = direction
            entry_price    = bar_open
            entry_time     = bar_time
            sl_dist_entry  = sl_d
            max_excursion  = 0.0

            initial_sl = (entry_price - sl_d) if direction == 'long' \
                         else (entry_price + sl_d)
            sl_price   = initial_sl

            tp1_price = (entry_price + 2.0 * sl_d) if direction == 'long' \
                        else (entry_price - 2.0 * sl_d)

            tp1_hit               = False
            opposite_pending_exit = False
            pending               = None

        elif pending is not None:
            pending = None

        # ── B. Manage open LONG position ──────────────────────────────────────
        if position == 'long':
            exited = False
            max_excursion = max(max_excursion, bar_high - entry_price)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            # ── Phase 0: before TP1 ───────────────────────────────────────────
            if not tp1_hit:
                if bar_low <= sl_price:
                    pnl  = (sl_price - entry_price) * lots * contract_size
                    c_rr = round((sl_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl
                    position = None
                    exited   = True

                elif bar_high >= tp1_price:
                    pnl_half = (tp1_price - entry_price) * half * contract_size
                    c_rr     = round((tp1_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half, 'TP1 (1:2) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_half
                    sl_price = entry_price   # move SL to breakeven
                    tp1_hit  = True

            # ── Phase 1: TP1 hit — hold runner until opposite signal ───────────
            if tp1_hit and not exited:
                # Honour deferred opposite exit (exit at THIS bar's open)
                if opposite_pending_exit:
                    exit_p   = bar_open
                    pnl_run  = (exit_p - entry_price) * half * contract_size
                    c_rr     = round((exit_p - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                # Breakeven SL check on runner
                if not exited and bar_low <= sl_price:
                    pnl_run = (sl_price - entry_price) * half * contract_size
                    c_rr    = round((sl_price - entry_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'long',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                # Opposite signal → queue exit for NEXT bar's open
                if not exited and row['sell']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── C. Manage open SHORT position ─────────────────────────────────────
        elif position == 'short':
            exited = False
            max_excursion = max(max_excursion, entry_price - bar_low)
            h_rr = round(max_excursion / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0

            # ── Phase 0: before TP1 ───────────────────────────────────────────
            if not tp1_hit:
                if bar_high >= sl_price:
                    pnl  = (entry_price - sl_price) * lots * contract_size
                    c_rr = round((entry_price - sl_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        lots, pnl, 'SL hit',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl
                    position = None
                    exited   = True

                elif bar_low <= tp1_price:
                    pnl_half = (entry_price - tp1_price) * half * contract_size
                    c_rr     = round((entry_price - tp1_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, tp1_price, initial_sl,
                        half, pnl_half, 'TP1 (1:2) — 50% partial',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_half
                    sl_price = entry_price
                    tp1_hit  = True

            # ── Phase 1: TP1 hit — hold runner until opposite signal ───────────
            if tp1_hit and not exited:
                if opposite_pending_exit:
                    exit_p   = bar_open
                    pnl_run  = (entry_price - exit_p) * half * contract_size
                    c_rr     = round((entry_price - exit_p) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, exit_p, sl_price,
                        half, pnl_run, 'Opposite signal — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True
                    opposite_pending_exit = False

                if not exited and bar_high >= sl_price:
                    pnl_run = (entry_price - sl_price) * half * contract_size
                    c_rr    = round((entry_price - sl_price) / sl_dist_entry, 2) if sl_dist_entry > 0 else 0.0
                    trades.append(_make_trade(
                        entry_time, bar_time, 'short',
                        entry_price, sl_price, initial_sl,
                        half, pnl_run, 'SL (breakeven) — runner 50%',
                        highest_rr=h_rr, closed_rr=c_rr))
                    cumul   += pnl_run
                    position = None
                    exited   = True

                if not exited and row['buy']:
                    opposite_pending_exit = True

            if not exited:
                eq_pts[bar_time] = cumul

        # ── D. Record equity on flat bars ─────────────────────────────────────
        if position is None:
            eq_pts[bar_time] = cumul

        # ── E. New signals only when completely flat ───────────────────────────
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
    report_path = os.path.join(output_dir, f"strategy8_report_{ts}.html")

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
            TP_Rule          = ('50% close @ 1:2 → BE on runner → '
                                'hold runner until opposite signal'),
            Total_Bars       = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 8]  No trades generated.")
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
  python strategy8_half_tp_be_opposite_exit.py --source csv --csv XAUUSD_M15.csv
  python strategy8_half_tp_be_opposite_exit.py --source mt5 --symbol XAUUSD --tf M15
  python strategy8_half_tp_be_opposite_exit.py --source csv --csv data.csv --sl-mult 2.0
        """)
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
    args = p.parse_args()

    df = load_data(source=args.source, csv_path=args.csv,
                   symbol=args.symbol, timeframe=args.tf)

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