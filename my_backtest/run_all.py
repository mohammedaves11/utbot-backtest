"""
run_all.py
===========
One-file launcher that runs ALL 4 strategies and produces 4 HTML reports.

SETUP
-----
1. Put all 5 .py files in the same folder.
2. Install dependencies once:
       pip install pandas numpy matplotlib
       pip install MetaTrader5   ← only needed for MT5 mode

CONFIGURE (edit the block below labelled ── USER CONFIG ──)
-----------
Choose your data source: 'csv' or 'mt5'
Set your file path or MT5 symbol / timeframe.

RUN
---
From the folder that contains these files:
    python run_all.py

Reports are saved to the  reports/  sub-folder inside the same directory.
The exact file path for every report is printed to the console.
"""

import os
import sys

# ─── Make sure Python can find backtest_utils and the strategy files ──────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backtest_utils import load_data

from strategy1_opposite_signal  import run_backtest as strategy1
from strategy2_half_tp_be_trail import run_backtest as strategy2
from strategy3_staged_sl        import run_backtest as strategy3
from strategy4_full_atr_trail   import run_backtest as strategy4
from strategy5_staged_partial_close import run_backtest as strategy5
from strategy6_ratchet_sl import run_backtest as strategy6
from strategy7_be_at_1r2_opposite_exit import run_backtest as strategy7
from strategy8_half_tp_be_opposite_exit import run_backtest as strategy8

# ══════════════════════════════════════════════════════════════════════════════
# ──  USER CONFIG  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Data source: 'csv'  OR  'mt5' ────────────────────────────────────────────
SOURCE = 'mt5'           # 'csv' or 'mt5'

# ── CSV settings (used when SOURCE = 'csv') ───────────────────────────────────
CSV_PATH = r'XAUUSD_M15.csv'   # ← put your CSV filename or full path here
                                # e.g.  r'C:\Users\You\Downloads\XAUUSD_M15.csv'
# Column names in your CSV file (edit if different)
CSV_TIME_COL  = 'time'
CSV_OPEN_COL  = 'open'
CSV_HIGH_COL  = 'high'
CSV_LOW_COL   = 'low'
CSV_CLOSE_COL = 'close'
CSV_SEP       = ','    # column separator

# ── MT5 settings (used when SOURCE = 'mt5') ───────────────────────────────────
MT5_SYMBOL    = 'XAUUSD.t'         # your broker's exact symbol (e.g. 'XAUUSD.t')
MT5_TIMEFRAME = 'M15'            # M1 M5 M15 M30 H1 H4 D1
MT5_FROM_DATE = '2026-01-01'     # start date  'YYYY-MM-DD'
MT5_TO_DATE   = '2026-04-22'     # end date    'YYYY-MM-DD'  (or 'today' for today)

# ── Risk & indicator parameters ───────────────────────────────────────────────
KEY_VALUE     = 3      # ATR sensitivity (Pine: 3)
ATR_PERIOD    = 14     # ATR period      (Pine: 14)
SL_MULT       = 1.5    # SL in ATR multiples (Pine: 1.5)
TRAIL_MULT    = 1.5    # Trail ATR mult for strategies 2/3/4  (1.5 = tighter)
RISK_USD      = 200.0  # $ risked per trade (Pine: 200)
CONTRACT_SIZE = 100.0  # Gold: 1 lot = 100 oz
ADX_LEN       = 14     # ADX period
ADX_THRESH    = 20     # ADX threshold — below = choppy, skip trade
FILTER_CHOPPY = False   # True = use ADX filter (recommended)
INITIAL_BALANCE = 100_000.0  # Starting account equity shown in equity curve ($)

# ── Output ─────────────────────────────────────────────────────────────────────
SYMBOL     = 'XAUUSD'
OUTPUT_DIR = os.path.join(HERE, 'reports')   # reports/ next to these .py files

# ══════════════════════════════════════════════════════════════════════════════
# ──  MAIN  ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print('=' * 60)
    print('  AS Alerts Gold — 4-Strategy Backtest Runner')
    print('=' * 60)
    print(f'  Source      : {SOURCE}')
    if SOURCE == 'csv':
        print(f'  File        : {CSV_PATH}')
    else:
        print(f'  Symbol      : {MT5_SYMBOL}  {MT5_TIMEFRAME}  ({MT5_FROM_DATE} → {MT5_TO_DATE})')
    print(f'  Output dir  : {OUTPUT_DIR}')
    print()

    # ── Load data once, share across all strategies ───────────────────────────
    if SOURCE == 'csv':
        df = load_data(
            source    = 'csv',
            csv_path  = CSV_PATH,
            time_col  = CSV_TIME_COL,
            open_col  = CSV_OPEN_COL,
            high_col  = CSV_HIGH_COL,
            low_col   = CSV_LOW_COL,
            close_col = CSV_CLOSE_COL,
            sep       = CSV_SEP,
        )
    else:
        df = load_data(
            source     = 'mt5',
            symbol     = MT5_SYMBOL,
            timeframe  = MT5_TIMEFRAME,
            from_date  = MT5_FROM_DATE,
            to_date    = MT5_TO_DATE,
        )

    print()

    # ── Shared kwargs for every strategy ─────────────────────────────────────
    common = dict(
        key_value       = KEY_VALUE,
        atr_period      = ATR_PERIOD,
        sl_mult         = SL_MULT,
        risk_usd        = RISK_USD,
        contract_size   = CONTRACT_SIZE,
        adx_len         = ADX_LEN,
        adx_thresh      = ADX_THRESH,
        filter_choppy   = FILTER_CHOPPY,
        initial_balance = INITIAL_BALANCE,
        output_dir      = OUTPUT_DIR,
        symbol          = SYMBOL,
    )

    reports = {}

    # ── Strategy 1: Exit on Opposite Signal ──────────────────────────────────
    print('Running Strategy 1 — Exit on Opposite Signal ...')
    trades1, equity1, rep1 = strategy1(df, **common)
    reports['Strategy 1'] = rep1

    # ── Strategy 2: 50 % TP, Breakeven, ATR Trail ────────────────────────────
    print('Running Strategy 2 — 50 % close at 1:2, BE, ATR trail ...')
    trades2, equity2, rep2 = strategy2(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 2'] = rep2

    # ── Strategy 3: Staged SL (BE → +1R → ATR trail) ─────────────────────────
    print('Running Strategy 3 — Staged SL: BE at 1:2, +1R at 1:3, ATR trail ...')
    trades3, equity3, rep3 = strategy3(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 3'] = rep3

    # ── Strategy 4: Full ATR Trail ────────────────────────────────────────────
    print('Running Strategy 4 — Full ATR trailing stop ...')
    trades4, equity4, rep4 = strategy4(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 4'] = rep4


        # ── Strategy 5: Staged Partial Close ─────────────────────────────────────
    print('Running Strategy 5 — 50% @ 1:2 · 25% @ 1:3 · 15% @ 1:4 · 10% opposite signal ...')
    trades5, equity5, rep5 = strategy5(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 5'] = rep5


    # ── Strategy 6: Ratcheting SL (no partial closes) ─────────────────────────
    print('Running Strategy 6 — Ratcheting SL: BE@1:2 · +1R@1:3 · +2R@1:4 · … (no partials) ...')
    trades6, equity6, rep6 = strategy6(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 6'] = rep6

    print('Running Strategy 7 — BE at 1:2, Exit on Opposite Signal ...')
    trades7, equity7, rep7 = strategy7(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 7'] = rep7

    print('Running Strategy 8 — 50% at 1:2 (BE), Runner until Opposite Signal ...')
    trades8, equity8, rep8 = strategy8(df, trail_mult=TRAIL_MULT, **common)
    reports['Strategy 8'] = rep8
 

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('  DONE — HTML Reports')
    print('=' * 60)
    for name, path in reports.items():
        print(f'  {name} → {path}')
    print()
    print('Open any of the above .html files in your browser to view the report.')
    print()

    return {
        'strategy1': (trades1, equity1, rep1),
        'strategy2': (trades2, equity2, rep2),
        'strategy3': (trades3, equity3, rep3),
        'strategy4': (trades4, equity4, rep4),
        'strategy5': (trades5, equity5, rep5),
        'strategy6': (trades6, equity6, rep6),
        'strategy7': (trades7, equity7, rep7),
        'strategy8': (trades8, equity8, rep8),
    }


if __name__ == '__main__':
    results = main()