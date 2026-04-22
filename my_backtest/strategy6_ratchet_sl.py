"""
strategy6_ratchet_sl.py
=========================
Strategy 6 — Ratcheting Stop-Loss (no partial closes)

SL RATCHET RULES  (LONG example — mirror for SHORT)
----------------------------------------------------
  Price reaches 1:2  →  SL moves to Breakeven  (entry + 0 × SL_dist)
  Price reaches 1:3  →  SL moves to +1R        (entry + 1 × SL_dist)
  Price reaches 1:4  →  SL moves to +2R        (entry + 2 × SL_dist)
  Price reaches 1:5  →  SL moves to +3R        (entry + 3 × SL_dist)
  Price reaches 1:N  →  SL moves to +(N-2)R    (general rule)

  The SL only ever MOVES FORWARD — it never steps back down.
  There are NO partial closes at any level.
  The FULL position stays open until one of the two exits below.

EXIT CONDITIONS
---------------
  1. SL hit (intrabar):  bar_low  <= sl_price  (long)
                         bar_high >= sl_price  (short)
     → exit entire position at current sl_price

  2. Opposite signal fires:
     → exit at that bar's CLOSE price
     → immediately queue new trade in the opposite direction
        (enters at next bar's open, same as all other strategies)

ENTRY
-----
  Next bar's open after the signal bar fires (same as all other strategies).
  SL is set at:  entry − sl_mult × ATR   (long)
                 entry + sl_mult × ATR   (short)

SAME-BAR PRIORITY
-----------------
  On any bar where BOTH the SL level AND the next ratchet target can be
  touched, the SL is treated as hit first (conservative).

TRADE-LOG ROWS
--------------
  One row per trade (no partials):
    exit_reason = 'SL hit'              — stopped out at current ratchet SL
    exit_reason = 'Opposite signal'     — exited on signal reversal
    exit_reason = 'End of data'         — force-closed at last bar

SL RATCHET TABLE (quick reference for live trading)
----------------------------------------------------
  RR target hit   |  SL moves to
  ─────────────────────────────────────
  1:2  (2R)       |  0R  (Breakeven)
  1:3  (3R)       |  1R
  1:4  (4R)       |  2R
  1:5  (5R)       |  3R
  1:6  (6R)       |  4R
  …               |  target − 2R

Usage
-----
from backtest_utils import load_data
from strategy6_ratchet_sl import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15.csv')
trades, equity, report = run_backtest(df, output_dir='reports')

# CLI:
python strategy6_ratchet_sl.py --source csv --csv XAUUSD_M15.csv
python strategy6_ratchet_sl.py --source mt5 --symbol XAUUSD.t --tf M15
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

STRATEGY_NAME = ("Strategy 6 — Ratcheting SL: "
                 "BE @ 1:2 · +1R @ 1:3 · +2R @ 1:4 · +3R @ 1:5 · …")

# First ratchet trigger is at 2R (1:2), and the SL steps begin at 0R (BE).
# After that every additional 1R of price move shifts the SL up by 1R.
# We encode this as:  ratchet trigger = (step + 2) × sl_dist
#                     ratchet SL      =  step      × sl_dist
# step=0 → trigger=2R, SL=0R (BE)
# step=1 → trigger=3R, SL=1R
# step=2 → trigger=4R, SL=2R  … and so on indefinitely.


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
                 # accepted for API compatibility with run_all.py
                 trail_mult:       float = DEFAULTS['trail_mult']) -> tuple:
    """
    Run Strategy 6 backtest.

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
    trail_mult    : Accepted for API compatibility, not used in this strategy

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
    position      = None    # None | 'long' | 'short'
    entry_price   = 0.0
    entry_time    = None
    initial_sl    = 0.0     # original SL at entry (for trade log)
    sl_price      = 0.0     # current active SL (ratchets up over time)
    sl_dist_entry = 0.0     # SL distance locked at entry
    lots          = 0.0

    # ratchet_step tracks how many ratchet levels have already been applied.
    # step=0 means we're still waiting for price to reach 2R (first trigger).
    # Once step=k is applied, we wait for price to reach (k+3)R next.
    ratchet_step  = 0

    pending = None

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

        # ── A. Fill pending entry at this bar's open ──────────────────────────
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
            ratchet_step  = 0
            pending       = None

        elif pending is not None:
            pending = None   # already in a position — discard

        # ── B. Manage open position ───────────────────────────────────────────
        exited = False

        if position == 'long':

            # ── B1. Apply any ratchet steps triggered this bar ────────────────
            # We process ALL steps that can be hit this bar (price might gap
            # through multiple levels). We stop if the SL would also be hit.
            keep_ratcheting = True
            while keep_ratcheting:
                next_trigger = entry_price + (ratchet_step + 2) * sl_dist_entry
                if bar_high >= next_trigger:
                    new_sl = entry_price + ratchet_step * sl_dist_entry
                    # Only move SL forward (safety guard)
                    if new_sl > sl_price:
                        sl_price = new_sl
                    ratchet_step += 1
                else:
                    keep_ratcheting = False

            # ── B2. SL check AFTER ratcheting (conservative order) ────────────
            if bar_low <= sl_price:
                pnl = (sl_price - entry_price) * lots * contract_size
                reason = ('SL hit (ratcheted +{}R)'.format(ratchet_step - 1)
                          if ratchet_step > 0 else 'SL hit')
                trades.append(_make_trade(entry_time, bar_time, 'long',
                                          entry_price, sl_price, initial_sl,
                                          lots, pnl, reason))
                cumul  += pnl
                position = None
                exited   = True

            # ── B3. Opposite signal → exit at bar close, queue short ──────────
            elif row['sell']:
                pnl = (bar_close - entry_price) * lots * contract_size
                trades.append(_make_trade(entry_time, bar_time, 'long',
                                          entry_price, bar_close, sl_price,
                                          lots, pnl, 'Opposite signal'))
                cumul  += pnl
                position = None
                exited   = True
                pending  = dict(direction='short',
                                sl_dist=row['sl_dist'],
                                lot_size=row['lot_size'])

        elif position == 'short':

            # ── B1. Apply any ratchet steps triggered this bar ────────────────
            keep_ratcheting = True
            while keep_ratcheting:
                next_trigger = entry_price - (ratchet_step + 2) * sl_dist_entry
                if bar_low <= next_trigger:
                    new_sl = entry_price - ratchet_step * sl_dist_entry
                    # Only move SL forward (lower for shorts = more protective)
                    if new_sl < sl_price:
                        sl_price = new_sl
                    ratchet_step += 1
                else:
                    keep_ratcheting = False

            # ── B2. SL check AFTER ratcheting ─────────────────────────────────
            if bar_high >= sl_price:
                pnl = (entry_price - sl_price) * lots * contract_size
                reason = ('SL hit (ratcheted +{}R)'.format(ratchet_step - 1)
                          if ratchet_step > 0 else 'SL hit')
                trades.append(_make_trade(entry_time, bar_time, 'short',
                                          entry_price, sl_price, initial_sl,
                                          lots, pnl, reason))
                cumul  += pnl
                position = None
                exited   = True

            # ── B3. Opposite signal → exit at bar close, queue long ───────────
            elif row['buy']:
                pnl = (entry_price - bar_close) * lots * contract_size
                trades.append(_make_trade(entry_time, bar_time, 'short',
                                          entry_price, bar_close, sl_price,
                                          lots, pnl, 'Opposite signal'))
                cumul  += pnl
                position = None
                exited   = True
                pending  = dict(direction='long',
                                sl_dist=row['sl_dist'],
                                lot_size=row['lot_size'])

        # ── C. Check for new signals when flat ───────────────────────────────
        if position is None and not exited:
            if row['buy']:
                pending = dict(direction='long',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])
            elif row['sell']:
                pending = dict(direction='short',
                               sl_dist=row['sl_dist'],
                               lot_size=row['lot_size'])

        eq_pts.setdefault(bar_time, cumul)
        if position is None:
            eq_pts[bar_time] = cumul

    # ── D. Force-close any open position at end of data ───────────────────────
    if position is not None:
        last_time = sig.index[-1]
        last_cl   = sig.iloc[-1]['close']
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
    report_path = os.path.join(output_dir, f"strategy6_report_{ts}.html")

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
            TP_Rule           = ('No partials — ratchet SL: '
                                 'BE@1:2 · +1R@1:3 · +2R@1:4 · …'),
            Total_Bars        = len(sig),
        ))

    return trades, equity_curve, report_path


# ══════════════════════════════════════════════════════════════════════════════
def _print_summary(trades, equity):
    if not trades:
        print("[Strategy 6]  No trades generated.")
        return
    wins = [t for t in trades if t['pnl'] > 0]
    tp   = sum(t['pnl'] for t in trades)
    print(f"\n{'─'*58}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'─'*58}")
    print(f"  Total trades   : {len(trades)}")
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
  python strategy6_ratchet_sl.py --source csv --csv XAUUSD_M15.csv
  python strategy6_ratchet_sl.py --source mt5 --symbol XAUUSD.t --tf M15
  python strategy6_ratchet_sl.py --source csv --csv data.csv --sl-mult 2.0
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