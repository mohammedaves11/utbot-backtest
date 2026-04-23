"""
run_all.py
===========
One-file launcher that runs ALL 7 strategies and produces 7 HTML + 7 CSV reports.

SETUP
-----
1. Put all .py files in the same folder.
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
Both an HTML report and a CSV trade log are saved for every strategy.
"""

import os
import sys

# ─── Make sure Python can find backtest_utils and the strategy files ──────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backtest_utils import load_data

from strategy1_opposite_signal           import run_backtest as strategy1
from strategy2_half_tp_be_trail          import run_backtest as strategy2
from strategy3_staged_sl                 import run_backtest as strategy3
from strategy4_full_atr_trail            import run_backtest as strategy4
from strategy5_staged_partial_close  import run_backtest as strategy5
from strategy8_half_tp_be_opposite_exit  import run_backtest as strategy8
from strategy9_fixed_points_opposite     import run_backtest as strategy9
from strategy10_ema_opposite_signal          import run_backtest as strategy10
from strategy11_ema_staged_partial_close     import run_backtest as strategy11
from strategy12_ema_half_tp_be_opposite_exit import run_backtest as strategy12
from strategy13_ema_fixed_points_opposite    import run_backtest as strategy13
from strategy14_rr_trail import run_backtest as strategy14
from startegy15_ema_fixedpoints_rrtrail import run_backtest as strategy15
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
MT5_FROM_DATE = '2024-01-01'     # start date  'YYYY-MM-DD'
MT5_TO_DATE   = '2026-04-24'     # end date    'YYYY-MM-DD'  (or 'today' for today)

# ── Risk & indicator parameters ───────────────────────────────────────────────
KEY_VALUE     = 3     # ATR sensitivity (Pine: 3)
ATR_PERIOD    = 14     # ATR period      (Pine: 14)
SL_MULT       = 1.5    # SL in ATR multiples (Pine: 1.5)
TRAIL_MULT    = 1.5    # Trail ATR mult for strategies 2/3/4  (1.5 = tighter)
RISK_USD      = 200.0  # $ risked per trade (Pine: 200)
CONTRACT_SIZE = 100.0  # Gold: 1 lot = 100 oz
ADX_LEN       = 14     # ADX period
ADX_THRESH    = 25     # ADX threshold — below = choppy, skip trade
FILTER_CHOPPY = False   # True = use ADX filter (recommended)
INITIAL_BALANCE = 100_000.0  # Starting account equity shown in equity curve ($)

# ── Strategy 9 / 13 fixed-point TP settings ───────────────────────────────────
S9_TP1_POINTS = 75.0
S9_TP2_POINTS = 100.0
 
# ── EMA filter settings (Strategies 10–13) ───────────────────────────────────
EMA_FAST = 20   # fast EMA period
EMA_SLOW = 50   # slow EMA period

# ── Strategy 14 — Peak RR Trail settings ─────────────────────────────────────
S14_MIN_RR_ACTIVATE = 5.0   # trail only activates once peak RR reaches this value
S14_TRAIL_PULLBACK  = 1.0   # close trade when price pulls back this many R from peak
                             # e.g. peak=24R, pullback=1R → trail exit triggers at 22R

# ── Strategy 15 — TP1 fixed + Peak RR Trail runner ───────────────────────────
S15_TP1_POINTS      = 75.0  # fixed points for TP1 (50% close) — same as S13
S15_MIN_RR_ACTIVATE = 5.0   # runner trail activates once peak RR reaches this value
S15_TRAIL_PULLBACK  = 1.0   # close runner when price pulls back this many R from peak
 

# ── Which strategies to run (set False to skip) ───────────────────────────────
RUN_S1 = True
RUN_S2 = False
RUN_S3 = False
RUN_S4 = False
RUN_S5 = False
RUN_S8 = False
RUN_S9 = False
RUN_S10 = False   # Strategy 1  + EMA 20/50
RUN_S11 = False   # Strategy 5  + EMA 20/50
RUN_S12 = True   # Strategy 8  + EMA 20/50
RUN_S13 = True   # Strategy 9  + EMA 20/50
RUN_S14 = True   # Peak RR Trail       + EMA filter  ← NEW
RUN_S15 = True   # TP1 fixed + Peak RR Trail runner + EMA filter  ← NEW

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
    print('  AS Alerts -Strategy Backtest Runner')
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

       # Extra kwargs only for EMA strategies (10–13)
    ema_extra = dict(ema_fast=EMA_FAST, ema_slow=EMA_SLOW)
 

    reports = {}

    # ── Strategy 1: Exit on Opposite Signal ──────────────────────────────────
    if RUN_S1:
        print('Running Strategy 1 — Exit on Opposite Signal ...')
        trades1, equity1, rep1 = strategy1(df, **common)
        reports['Strategy 1'] = rep1

    # ── Strategy 2: 50 % TP, Breakeven, ATR Trail ────────────────────────────
    if RUN_S2:
        print('Running Strategy 2 — 50 % close at 1:2, BE, ATR trail ...')
        trades2, equity2, rep2 = strategy2(df, trail_mult=TRAIL_MULT, **common)
        reports['Strategy 2'] = rep2

    # ── Strategy 3: Staged SL (BE → +1R → ATR trail) ─────────────────────────
    if RUN_S3:
        print('Running Strategy 3 — Staged SL: BE at 1:2, +1R at 1:3, ATR trail ...')
        trades3, equity3, rep3 = strategy3(df, trail_mult=TRAIL_MULT, **common)
        reports['Strategy 3'] = rep3

    # ── Strategy 4: Full ATR Trail ────────────────────────────────────────────
    if RUN_S4:
        print('Running Strategy 4 — Full ATR trailing stop ...')
        trades4, equity4, rep4 = strategy4(df, trail_mult=TRAIL_MULT, **common)
        reports['Strategy 4'] = rep4

    # ── Strategy 5: Staged Partials (50/25/15/10%) + Opposite Signal ──────────
    if RUN_S5:
        print('Running Strategy 5 — Staged partials 50/25/15/10% at 1:2/1:3/1:4/opp ...')
        trades5, equity5, rep5 = strategy5(df, **common)
        reports['Strategy 5'] = rep5

    # ── Strategy 8: 50% at 1:2 (BE), Runner to Opposite Signal ───────────────
    if RUN_S8:
        print('Running Strategy 8 — 50% close at 1:2, BE, runner to opposite signal ...')
        trades8, equity8, rep8 = strategy8(df, **common)
        reports['Strategy 8'] = rep8

    # ── Strategy 9: Fixed +50/+100pt TPs + Opposite Signal ───────────────────
    if RUN_S9:
        print('Running Strategy 9 — Fixed +50pt/+100pt TPs, opposite signal fallback ...')
        trades9, equity9, rep9 = strategy9(
            df, tp1_points=S9_TP1_POINTS, tp2_points=S9_TP2_POINTS, **common)
        reports['Strategy 9'] = rep9

    if RUN_S10:
        print(f'Running Strategy 10 — Opposite Signal + EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy10(df, **common, **ema_extra)
        reports[f'Strategy 10 (Opposite Signal + EMA {EMA_FAST}/{EMA_SLOW})'] = rep
 
    if RUN_S11:
        print(f'Running Strategy 11 — Staged Partials + EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy11(df, **common, **ema_extra)
        reports[f'Strategy 11 (Staged Partials + EMA {EMA_FAST}/{EMA_SLOW})'] = rep
 
    if RUN_S12:
        print(f'Running Strategy 12 — 50% TP Runner + EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy12(df, **common, **ema_extra)
        reports[f'Strategy 12 (50% TP Runner + EMA {EMA_FAST}/{EMA_SLOW})'] = rep
 
    if RUN_S13:
        print(f'Running Strategy 13 — Fixed Points + EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy13(df, tp1_points=S9_TP1_POINTS,
                               tp2_points=S9_TP2_POINTS,
                               **common, **ema_extra)
        reports[f'Strategy 13 (Fixed Points + EMA {EMA_FAST}/{EMA_SLOW})'] = rep


    # ── Strategy 14: Peak RR Trail + EMA filter  ← NEW ───────────────────────
    if RUN_S14:
        print(f'Running Strategy 14 — Peak RR Trail (activate={S14_MIN_RR_ACTIVATE}R, '
              f'pullback={S14_TRAIL_PULLBACK}R) + EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy14(df,
                               min_rr_activate=S14_MIN_RR_ACTIVATE,
                               trail_pullback=S14_TRAIL_PULLBACK,
                               **common, **ema_extra)
        reports[f'Strategy 14 (Peak RR Trail + EMA {EMA_FAST}/{EMA_SLOW})'] = rep

      # ── Strategy 15: TP1 fixed (50%) + Peak RR Trail runner  ← NEW ──────────
    if RUN_S15:
        print(f'Running Strategy 15 — TP1 +{S15_TP1_POINTS}pts then Peak RR Trail '
              f'(activate={S15_MIN_RR_ACTIVATE}R, pullback={S15_TRAIL_PULLBACK}R) '
              f'+ EMA {EMA_FAST}/{EMA_SLOW} filter ...')
        _, _, rep = strategy15(df,
                               tp1_points=S15_TP1_POINTS,
                               min_rr_activate=S15_MIN_RR_ACTIVATE,
                               trail_pullback=S15_TRAIL_PULLBACK,
                               **common, **ema_extra)
        reports[f'Strategy 15 (TP1+{S15_TP1_POINTS}pts → Peak RR Trail + EMA {EMA_FAST}/{EMA_SLOW})'] = rep

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('  DONE — HTML + CSV Reports')
    print('=' * 60)
    for name, path in reports.items():
        csv_path = path.replace('.html', '.csv')
        print(f'  {name}')
        print(f'    HTML → {path}')
        if os.path.exists(csv_path):
            print(f'    CSV  → {csv_path}')
    print()
    print('Open any of the above .html files in your browser to view the report.')
    print()


if __name__ == '__main__':
    main()
