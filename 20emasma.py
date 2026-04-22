# -*- coding: utf-8 -*-
"""
=======================================================
 Backtest: 20 EMA + SMA High/Low BOS/CHoCH Trade Engine
 Data source : MetaTrader 5
 Risk        : Fixed $RISK_USD per trade (auto position size)
=======================================================
 Requirements:
   pip install MetaTrader5 pandas numpy

 How to run:
   Double-click  run_backtest.bat
   OR in PowerShell:  python backtest_mt5.py
=======================================================
"""

import sys
import os

# Fix Windows console encoding so all characters print correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.")
    print("Run:  pip install MetaTrader5 pandas numpy")
    input("Press Enter to exit...")
    sys.exit(1)

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
import json
import warnings
warnings.filterwarnings("ignore")


# =========================================================
# CONFIG  --  only edit this section
# =========================================================
SYMBOL            = "XAUUSD"
TIMEFRAME         = mt5.TIMEFRAME_M5     # M1 M5 M15 M30 H1 H4 D1 W1 MN1
START_DATE        = datetime(2026,  1,  1, tzinfo=timezone.utc)
END_DATE          = datetime(2026,  4, 19, tzinfo=timezone.utc)
ACCOUNT_SIZE      = 100_000.0            # starting balance USD
RISK_USD          = 100.0               # fixed $ risk per trade

# -- MA Band ----------------------------------------------
BAND_LEN             = 20
STRICT_CANDLE_FILTER = True   # whole candle must clear the band

# -- BOS / CHoCH ------------------------------------------
STRUCTURE_LEN = 5             # pivot lookback left & right

# -- Trade Engine -----------------------------------------
ATR_LEN           = 14
SL_LOOKBACK       = 5
ATR_BUFFER        = 0.30      # ATR multiples added beyond swing high/low
MIN_BAND_SPREAD   = 0.10      # band must be >= this x ATR  (chop filter)
RANGE_LOOKBACK    = 10
MAX_RANGE_ATR     = 1.60      # recent range must be > this x ATR (chop filter)
ONE_TRADE_AT_TIME = True

# -- SL bucket labels (informational, in price points) ----
# Gold moves in $1 increments -- adjust to your comfort
SL_TIGHT_LIMIT  =  5.0        # <= $5  SL distance  -> tight
SL_MEDIUM_LIMIT = 15.0        # <= $15 SL distance  -> medium
SL_AVOID_LIMIT  = 30.0        # >  $30 SL distance  -> avoid

# -- Take-profit R:R multipliers --------------------------
TP1_MULT = 1.0
TP2_MULT = 1.5
TP3_MULT = 2.0                # trade closes here (or at SL)


# =========================================================
# MT5 -- DATA FETCH
# =========================================================

# Common broker naming variants for XAUUSD -- tried in order until one works
XAUUSD_VARIANTS = ["XAUUSD", "XAUUSDm", "XAUUSDc", "XAUUSD.", "XAUUSDx",
                   "GOLD", "GOLDm", "XAU_USD", "XAU/USD"]

def fetch_mt5_data(symbol, timeframe, start, end):
    if not mt5.initialize():
        raise RuntimeError(
            "MT5 initialize() failed -- error: {}\n"
            "Make sure MetaTrader 5 is open and logged in.".format(mt5.last_error())
        )

    # Build list of names to try: requested symbol first, then known variants
    names_to_try = [symbol] + [v for v in XAUUSD_VARIANTS if v != symbol]

    # Strip timezone -- MT5 works best with naive UTC datetimes
    start_naive = start.replace(tzinfo=None)
    end_naive   = end.replace(tzinfo=None)

    used_symbol = None
    rates       = None

    for name in names_to_try:
        # Add to Market Watch (required before fetching)
        mt5.symbol_select(name, True)
        r = mt5.copy_rates_range(name, timeframe, start_naive, end_naive)
        if r is not None and len(r) > 0:
            used_symbol = name
            rates       = r
            break

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            "No data returned for XAUUSD.\n\n"
            "  Tried: {}\n\n"
            "  Things to check:\n"
            "    1. MetaTrader 5 is open and logged in to your broker\n"
            "    2. Open a XAUUSD chart manually in MT5 first, then re-run\n"
            "    3. Try a shorter date range e.g. START_DATE = datetime(2024, 1, 1)\n"
            "    4. Make sure XAUUSD is visible in MT5 Market Watch\n"
            "       (right-click Market Watch -> Show All)".format(
                ", ".join(names_to_try[:5]))
        )

    if used_symbol != symbol:
        print("  Note: '{}' not found -- using '{}' instead.".format(symbol, used_symbol))

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


# =========================================================
# INDICATOR HELPERS
# =========================================================

def wilder_atr(high, low, close, length):
    """Wilder smoothed ATR -- matches Pine Script ta.atr()."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def pivot_high(series, left, right):
    """
    Returns the pivot-high value at bar i when series[i] is the
    local maximum over [i-left ... i+right].  NaN otherwise.
    Replicates Pine Script  ta.pivothigh(src, left, right).
    """
    arr = series.values
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == np.max(window):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


def pivot_low(series, left, right):
    """Replicates Pine Script  ta.pivotlow(src, left, right)."""
    arr = series.values
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == np.min(window):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


# =========================================================
# COMPUTE ALL INDICATORS
# =========================================================

def compute_indicators(df):
    df = df.copy()

    # MA Band
    df["ema20"]       = df["Close"].ewm(span=BAND_LEN, adjust=False).mean()
    df["sma_high20"]  = df["High"].rolling(BAND_LEN).mean()
    df["sma_low20"]   = df["Low"].rolling(BAND_LEN).mean()
    df["band_top"]    = df[["ema20", "sma_high20", "sma_low20"]].max(axis=1)
    df["band_btm"]    = df[["ema20", "sma_high20", "sma_low20"]].min(axis=1)
    df["band_spread"] = df["band_top"] - df["band_btm"]

    # ATR
    df["atr"] = wilder_atr(df["High"], df["Low"], df["Close"], ATR_LEN)

    # BOS / CHoCH pivots
    # ph_raw[i] = price if bar i is a confirmed pivot high, else NaN
    # Pivot at bar j is detectable only after bar j + STRUCTURE_LEN
    df["ph_raw"] = pivot_high(df["High"], STRUCTURE_LEN, STRUCTURE_LEN)
    df["pl_raw"] = pivot_low( df["Low"],  STRUCTURE_LEN, STRUCTURE_LEN)

    # Chop / range
    df["recent_high"] = df["High"].rolling(RANGE_LOOKBACK).max()
    df["recent_low"]  = df["Low"].rolling(RANGE_LOOKBACK).min()
    df["range_size"]  = df["recent_high"] - df["recent_low"]

    # Pre-computed SL anchors
    df["sl_buy_anchor"]  = df["Low"].rolling(SL_LOOKBACK).min()
    df["sl_sell_anchor"] = df["High"].rolling(SL_LOOKBACK).max()

    return df


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class ActiveTrade:
    direction : int   = 0
    entry     : float = np.nan
    sl        : float = np.nan
    risk      : float = np.nan
    tp1       : float = np.nan
    tp2       : float = np.nan
    tp3       : float = np.nan
    qty       : float = np.nan
    entry_bar : int   = -1
    tp1_hit   : bool  = False
    tp2_hit   : bool  = False

    def reset(self):
        self.__init__()


@dataclass
class TradeRecord:
    bar_entry   : int
    date_entry  : object
    direction   : int
    entry       : float
    sl          : float
    tp1         : float
    tp2         : float
    tp3         : float
    qty         : float
    sl_bucket   : str    = ""
    tp1_hit     : bool   = False
    tp2_hit     : bool   = False
    exit_price  : float  = np.nan
    exit_bar    : int    = -1
    date_exit   : object = None
    exit_reason : str    = ""     # "TP3" | "SL" | "END"
    pnl         : float  = np.nan
    r_multiple  : float  = np.nan


# =========================================================
# HELPERS
# =========================================================

def sl_bucket_label(price_dist):
    """Matches Pine Script slBucket() -- raw price distance."""
    if price_dist <= SL_TIGHT_LIMIT:
        return "SL <= {}".format(SL_TIGHT_LIMIT)
    elif price_dist <= SL_MEDIUM_LIMIT:
        return "SL <= {}".format(SL_MEDIUM_LIMIT)
    elif price_dist <= SL_AVOID_LIMIT:
        return "SL {}-{}".format(SL_MEDIUM_LIMIT, SL_AVOID_LIMIT)
    else:
        return "AVOID: SL > {}".format(SL_AVOID_LIMIT)


# =========================================================
# BACKTEST ENGINE  --  bar-by-bar stateful loop
# =========================================================

def run_backtest(df):
    trades = []
    state  = ActiveTrade()

    swing_high_price = np.nan
    swing_high_bar   = -1
    swing_low_price  = np.nan
    swing_low_bar    = -1
    trend_state      = 0    # 0=neutral  1=bull  -1=bear

    warmup = max(BAND_LEN, ATR_LEN, STRUCTURE_LEN * 2,
                 RANGE_LOOKBACK, SL_LOOKBACK) + 5
    n = len(df)

    for i in range(warmup, n):
        row   = df.iloc[i]
        close = row["Close"]
        high  = row["High"]
        low   = row["Low"]
        open_ = row["Open"]
        atr   = row["atr"]

        if pd.isna(atr) or atr == 0:
            continue

        band_top    = row["band_top"]
        band_btm    = row["band_btm"]
        band_spread = row["band_spread"]
        range_size  = row["range_size"]

        # 1. Process newly-confirmed pivots
        #    Pivot at bar j is confirmed when we reach bar j + STRUCTURE_LEN.
        #    At bar i, we confirm the pivot at bar i - STRUCTURE_LEN.
        conf = i - STRUCTURE_LEN
        if conf >= 0:
            ph_val = df["ph_raw"].iloc[conf]
            pl_val = df["pl_raw"].iloc[conf]
            if not np.isnan(ph_val):
                swing_high_price = ph_val
                swing_high_bar   = conf
            if not np.isnan(pl_val):
                swing_low_price  = pl_val
                swing_low_bar    = conf

        # 2. Detect BOS / CHoCH
        bull_bos   = False
        bear_bos   = False
        bull_choch = False
        bear_choch = False

        if not np.isnan(swing_high_price) and close > swing_high_price:
            if trend_state == -1:
                bull_choch = True
            else:
                bull_bos = True
            trend_state      = 1
            swing_high_price = np.nan

        if not np.isnan(swing_low_price) and close < swing_low_price:
            if trend_state == 1:
                bear_choch = True
            else:
                bear_bos = True
            trend_state     = -1
            swing_low_price = np.nan

        # 3. Anti-chop filters
        band_spread_ok = band_spread >= atr * MIN_BAND_SPREAD
        range_ok       = range_size  >  atr * MAX_RANGE_ATR
        price_in_band  = band_btm <= close <= band_top
        is_choppy      = (not band_spread_ok) or (not range_ok) or price_in_band

        if STRICT_CANDLE_FILTER:
            candle_above = (low   > band_top and close > band_top and open_ > band_top)
            candle_below = (high  < band_btm and close < band_btm and open_ < band_btm)
        else:
            candle_above = close > band_top
            candle_below = close < band_btm

        # 4. Signal logic
        raw_buy  = bull_bos or bull_choch
        raw_sell = bear_bos or bear_choch

        allow_new  = (state.direction == 0) if ONE_TRADE_AT_TIME else True
        final_buy  = raw_buy  and candle_above and not is_choppy and allow_new
        final_sell = raw_sell and candle_below and not is_choppy and allow_new

        # 5. Open new trade
        if final_buy:
            sl_price = row["sl_buy_anchor"] - atr * ATR_BUFFER
            risk     = abs(close - sl_price)
            if risk <= 0:
                continue
            qty = RISK_USD / risk
            state.direction = 1
            state.entry     = close
            state.sl        = sl_price
            state.risk      = risk
            state.qty       = qty
            state.tp1       = close + risk * TP1_MULT
            state.tp2       = close + risk * TP2_MULT
            state.tp3       = close + risk * TP3_MULT
            state.entry_bar = i
            state.tp1_hit   = False
            state.tp2_hit   = False
            trades.append(TradeRecord(
                bar_entry  = i,
                date_entry = df.index[i],
                direction  = 1,
                entry      = state.entry,
                sl         = state.sl,
                tp1        = state.tp1,
                tp2        = state.tp2,
                tp3        = state.tp3,
                qty        = qty,
                sl_bucket  = sl_bucket_label(risk),
            ))

        elif final_sell:
            sl_price = row["sl_sell_anchor"] + atr * ATR_BUFFER
            risk     = abs(sl_price - close)
            if risk <= 0:
                continue
            qty = RISK_USD / risk
            state.direction = -1
            state.entry     = close
            state.sl        = sl_price
            state.risk      = risk
            state.qty       = qty
            state.tp1       = close - risk * TP1_MULT
            state.tp2       = close - risk * TP2_MULT
            state.tp3       = close - risk * TP3_MULT
            state.entry_bar = i
            state.tp1_hit   = False
            state.tp2_hit   = False
            trades.append(TradeRecord(
                bar_entry  = i,
                date_entry = df.index[i],
                direction  = -1,
                entry      = state.entry,
                sl         = state.sl,
                tp1        = state.tp1,
                tp2        = state.tp2,
                tp3        = state.tp3,
                qty        = qty,
                sl_bucket  = sl_bucket_label(risk),
            ))

        # 6. Manage open trade  (SL checked before TP on same candle)
        if state.direction != 0 and i > state.entry_bar and trades:
            t = trades[-1]

            if state.direction == 1:
                sl_hit  = low  <= state.sl
                tp1_now = high >= state.tp1
                tp2_now = high >= state.tp2
                tp3_now = high >= state.tp3
                if sl_hit:
                    t.tp1_hit     = state.tp1_hit
                    t.tp2_hit     = state.tp2_hit
                    t.exit_price  = state.sl
                    t.exit_bar    = i
                    t.date_exit   = df.index[i]
                    t.exit_reason = "SL"
                    t.pnl         = -RISK_USD
                    t.r_multiple  = -1.0
                    state.reset()
                else:
                    if tp1_now and not state.tp1_hit:
                        state.tp1_hit = True
                        t.tp1_hit     = True
                    if tp2_now and not state.tp2_hit:
                        state.tp2_hit = True
                        t.tp2_hit     = True
                    if tp3_now:
                        t.exit_price  = state.tp3
                        t.exit_bar    = i
                        t.date_exit   = df.index[i]
                        t.exit_reason = "TP3"
                        t.pnl         = RISK_USD * TP3_MULT
                        t.r_multiple  = TP3_MULT
                        state.reset()

            elif state.direction == -1:
                sl_hit  = high >= state.sl
                tp1_now = low  <= state.tp1
                tp2_now = low  <= state.tp2
                tp3_now = low  <= state.tp3
                if sl_hit:
                    t.tp1_hit     = state.tp1_hit
                    t.tp2_hit     = state.tp2_hit
                    t.exit_price  = state.sl
                    t.exit_bar    = i
                    t.date_exit   = df.index[i]
                    t.exit_reason = "SL"
                    t.pnl         = -RISK_USD
                    t.r_multiple  = -1.0
                    state.reset()
                else:
                    if tp1_now and not state.tp1_hit:
                        state.tp1_hit = True
                        t.tp1_hit     = True
                    if tp2_now and not state.tp2_hit:
                        state.tp2_hit = True
                        t.tp2_hit     = True
                    if tp3_now:
                        t.exit_price  = state.tp3
                        t.exit_bar    = i
                        t.date_exit   = df.index[i]
                        t.exit_reason = "TP3"
                        t.pnl         = RISK_USD * TP3_MULT
                        t.r_multiple  = TP3_MULT
                        state.reset()

    # Mark-to-market any trade still open at end of data
    if state.direction != 0 and trades:
        t          = trades[-1]
        last_close = df["Close"].iloc[-1]
        pnl = ((last_close - state.entry) if state.direction == 1
               else (state.entry - last_close)) * state.qty
        t.exit_price  = last_close
        t.exit_bar    = n - 1
        t.date_exit   = df.index[-1]
        t.exit_reason = "END"
        t.pnl         = pnl
        t.r_multiple  = pnl / RISK_USD

    return trades


# =========================================================
# REPORTING
# =========================================================

def print_results(trades, df, account_size=ACCOUNT_SIZE):
    closed = [t for t in trades if t.exit_reason in ("TP3", "SL")]
    open_  = [t for t in trades if t.exit_reason in ("END", "")]

    if not closed:
        print("\n  No closed trades found in this period.")
        return None

    pnls   = [t.pnl for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate      = len(wins) / len(closed) * 100
    total_pnl     = sum(pnls)
    avg_win       = np.mean(wins)   if wins   else 0.0
    avg_loss      = np.mean(losses) if losses else 0.0
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy    = total_pnl / len(closed)
    final_balance = account_size + total_pnl
    pct_return    = total_pnl / account_size * 100

    # Equity curve & max drawdown
    equity   = np.concatenate([[account_size], account_size + np.cumsum(pnls)])
    peak     = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd   = abs(np.min(drawdown))
    max_dd_p = max_dd / account_size * 100

    longs  = [t for t in closed if t.direction ==  1]
    shorts = [t for t in closed if t.direction == -1]
    lw     = sum(1 for t in longs  if t.pnl > 0)
    sw     = sum(1 for t in shorts if t.pnl > 0)

    tp3_cnt = sum(1 for t in closed if t.exit_reason == "TP3")
    sl_cnt  = sum(1 for t in closed if t.exit_reason == "SL")
    tp1_cnt = sum(1 for t in closed if t.tp1_hit)
    tp2_cnt = sum(1 for t in closed if t.tp2_hit)

    buckets = {}
    for t in closed:
        b = t.sl_bucket
        if b not in buckets:
            buckets[b] = {"n": 0, "wins": 0, "pnl": 0.0}
        buckets[b]["n"]    += 1
        buckets[b]["wins"] += 1 if t.pnl > 0 else 0
        buckets[b]["pnl"]  += t.pnl

    SEP  = "=" * 62
    sep  = "-" * 62

    def row(label, value):
        print("  {:<32}{}".format(label, value))

    print("")
    print("  " + SEP)
    print("  {:^62}".format("BACKTEST REPORT"))
    print("  " + SEP)
    row("Symbol / Timeframe :",
        "{}  |  TF={}  |  {} -> {}".format(
            SYMBOL, TIMEFRAME, START_DATE.date(), END_DATE.date()))
    row("Bars analysed :",    "{:,}".format(len(df)))
    row("Risk per trade :",   "${:.2f}".format(RISK_USD))
    row("Starting balance :", "${:,.2f}".format(account_size))
    print("  " + sep)

    print("  TRADE SUMMARY")
    row("  Total closed :", str(len(closed)))
    row("  Still open :",   str(len(open_)))
    row("  Wins :",         "{}  ({:.1f} %)".format(len(wins), win_rate))
    row("  Losses :",       str(len(losses)))
    print("  " + sep)

    print("  P&L")
    row("  Total P&L :",        "${:+,.2f}".format(total_pnl))
    row("  Return :",           "{:+.2f} %".format(pct_return))
    row("  Final balance :",    "${:,.2f}".format(final_balance))
    row("  Expectancy/trade :", "${:+.2f}".format(expectancy))
    row("  Avg win :",          "${:+.2f}".format(avg_win))
    row("  Avg loss :",         "${:+.2f}".format(avg_loss))
    row("  Profit factor :",    "{:.2f}".format(profit_factor))
    row("  Max drawdown :",     "${:,.2f}  ({:.1f} %)".format(max_dd, max_dd_p))
    print("  " + sep)

    print("  DIRECTION")
    row("  Longs :",
        "{} trades  |  {} wins  |  ${:+,.2f}".format(
            len(longs), lw, sum(t.pnl for t in longs)))
    row("  Shorts :",
        "{} trades  |  {} wins  |  ${:+,.2f}".format(
            len(shorts), sw, sum(t.pnl for t in shorts)))
    print("  " + sep)

    print("  TP / SL OUTCOMES")
    row("  TP3 full close :", "{}  ({:.1f} %)".format(tp3_cnt, tp3_cnt / len(closed) * 100))
    row("  SL hit :",         "{}  ({:.1f} %)".format(sl_cnt,  sl_cnt  / len(closed) * 100))
    row("  TP1 level hit :",  "{}  ({:.1f} %)".format(tp1_cnt, tp1_cnt / len(closed) * 100))
    row("  TP2 level hit :",  "{}  ({:.1f} %)".format(tp2_cnt, tp2_cnt / len(closed) * 100))
    print("  " + sep)

    print("  SL BUCKET BREAKDOWN")
    for bucket, s in sorted(buckets.items()):
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        row("  {} :".format(bucket),
            "{} trades  WR: {:.1f} %  P&L: ${:+.2f}".format(s["n"], wr, s["pnl"]))
    print("  " + SEP)

    # Trade log
    print("")
    print("  TRADE LOG  (last {} closed trades)".format(min(30, len(closed))))
    print("  {:<4} {:<22} {:<5} {:>10} {:>10} {:>10} {:<4} {:<2} {:<2} {:>9} {:>6}".format(
        "#", "Entry Date", "Dir", "Entry", "SL", "Exit", "Rsn", "T1", "T2", "PnL", "R"))
    print("  " + "-" * 82)
    for idx, t in enumerate(closed[-30:], 1):
        d    = "BUY"  if t.direction == 1 else "SELL"
        date = str(t.date_entry)[:19]
        t1   = "Y" if t.tp1_hit else " "
        t2   = "Y" if t.tp2_hit else " "
        print("  {:<4} {:<22} {:<5} {:>10.2f} {:>10.2f} {:>10.2f} {:<4} {:<2} {:<2} ${:>8.2f} {:>4.1f}R".format(
            idx, date, d,
            t.entry, t.sl, t.exit_price,
            t.exit_reason, t1, t2,
            t.pnl, t.r_multiple))
    print("  " + SEP)
    print("")

    # JSON summary
    summary = {
        "symbol"          : SYMBOL,
        "timeframe"       : str(TIMEFRAME),
        "start"           : str(START_DATE.date()),
        "end"             : str(END_DATE.date()),
        "risk_usd"        : RISK_USD,
        "account_size"    : account_size,
        "total_closed"    : len(closed),
        "open_trades"     : len(open_),
        "wins"            : len(wins),
        "losses"          : len(losses),
        "win_rate_pct"    : round(win_rate, 2),
        "total_pnl"       : round(total_pnl, 2),
        "return_pct"      : round(pct_return, 2),
        "final_balance"   : round(final_balance, 2),
        "profit_factor"   : round(profit_factor, 4),
        "expectancy"      : round(expectancy, 2),
        "max_drawdown"    : round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_p, 2),
    }
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    fname_json  = os.path.join(script_dir, "summary_{}_{}.json".format(SYMBOL, TIMEFRAME))
    fname_csv   = os.path.join(script_dir, "trades_{}_{}.csv".format(SYMBOL, TIMEFRAME))

    with open(fname_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("  Summary JSON  ->  {}".format(fname_json))

    # CSV trade log
    records = []
    for t in trades:
        records.append({
            "date_entry"  : t.date_entry,
            "date_exit"   : t.date_exit,
            "direction"   : "BUY" if t.direction == 1 else "SELL",
            "entry"       : t.entry,
            "sl"          : t.sl,
            "tp1"         : t.tp1,
            "tp2"         : t.tp2,
            "tp3"         : t.tp3,
            "qty"         : round(t.qty,       4),
            "risk_usd"    : RISK_USD,
            "sl_bucket"   : t.sl_bucket,
            "tp1_hit"     : t.tp1_hit,
            "tp2_hit"     : t.tp2_hit,
            "exit_reason" : t.exit_reason,
            "exit_price"  : t.exit_price,
            "pnl"         : round(t.pnl,       2) if not np.isnan(t.pnl)        else np.nan,
            "r_multiple"  : round(t.r_multiple, 2) if not np.isnan(t.r_multiple) else np.nan,
        })
    df_out = pd.DataFrame(records)
    df_out.to_csv(fname_csv, index=False)
    print("  Trade log CSV ->  {}".format(fname_csv))
    print("")

    return df_out


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("")
    print("  Connecting to MT5...")
    print("  Symbol    : {}".format(SYMBOL))
    print("  Timeframe : {}".format(TIMEFRAME))
    print("  Period    : {} -> {}".format(START_DATE.date(), END_DATE.date()))
    print("")

    try:
        df_raw = fetch_mt5_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except RuntimeError as e:
        print("")
        print("  ERROR: {}".format(e))
        input("\n  Press Enter to exit...")
        sys.exit(1)

    print("  {:,} bars loaded.".format(len(df_raw)))
    print("")

    print("  Computing indicators...")
    df_ind = compute_indicators(df_raw)

    print("  Running backtest...")
    trades = run_backtest(df_ind)
    print("  {} trade signals found.".format(len(trades)))
    print("")

    print_results(trades, df_ind, account_size=ACCOUNT_SIZE)

    input("  Press Enter to exit...")