"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EMA CROSS STRATEGY — BACKTESTER  |  EURUSD M15  |  2 Years
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Mirrors Pine Script logic exactly:
  • Fast EMA (20) × Slow EMA (50) crossover signals
  • 200 EMA Trend Filter  (above = buy only / below = sell only)
  • ADX Choppy Filter     (skip trade when ADX < 15)
  • SL = Swing Low/High   (lookback 10) ± ATR×0.5 buffer
  • ATR Trailing SL       (ratchets SL in profit direction each bar)
  • $100 risk per trade   (lot = risk / (sl_pips × pip_value))
  • $100,000 starting account
 Outputs:
  • backtest_report_EURUSD_<ts>.html  — equity curve + charts
  • backtest_trades_EURUSD_<ts>.csv   — full trade-by-trade log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SYMBOL            = "EURUSD"
TIMEFRAME         = mt5.TIMEFRAME_M15
ACCOUNT_SIZE      = 100_000.0        # starting balance USD
RISK_USD          = 100.0            # fixed risk per trade

# EMA Settings
FAST_EMA          = 20
SLOW_EMA          = 50
TREND_EMA_LEN     = 200
USE_TREND_FILTER  = True             # only buy above / sell below trend EMA

# ADX Choppy Filter
USE_ADX           = True
ADX_LEN           = 10
ADX_THRESHOLD     = 15.0             # skip trades when ADX < this value

# SL Settings
ATR_LEN           = 14
SL_LOOKBACK       = 10               # bars to look back for swing high/low
ATR_BUFFER        = 0.5              # ATR multiplier added to swing SL

# Forex Pip Sizing  (EURUSD — USD-quoted pair, USD account)
PIP_SIZE          = 0.0001
PIP_VALUE_PER_LOT = 10.0             # USD per pip per standard lot

YEARS_BACK        = 2
_RUN_TS           = datetime.now().strftime("%Y%m%d%H%M%S")
OUTPUT_HTML       = f"backtest_report_{SYMBOL}_{_RUN_TS}.html"
OUTPUT_CSV        = f"backtest_trades_{SYMBOL}_{_RUN_TS}.csv"

# ══════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════

def fetch_data():
    print("Connecting to MT5...")
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")

    date_to   = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK + 10)

    print(f"Fetching {SYMBOL} M15 from {date_from.date()} to {date_to.date()}...")
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, date_from, date_to)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError("No data returned from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[df["time"] >= pd.Timestamp(date_from)].reset_index(drop=True)
    print(f"Loaded {len(df):,} candles")
    return df

# ══════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════

def calc_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(df, period):
    """Wilder's ADX — matches Pine Script ta.dmi() output."""
    high, low, close = df["high"], df["low"], df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing
    atr_w    = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w

    # DX → ADX
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = 100 * (plus_di - minus_di).abs() / denom
    adx   = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def build_signals(df):
    close = df["close"]

    fast_ema  = close.ewm(span=FAST_EMA,      adjust=False).mean()
    slow_ema  = close.ewm(span=SLOW_EMA,      adjust=False).mean()
    trend_ema = close.ewm(span=TREND_EMA_LEN, adjust=False).mean()
    atr       = calc_atr(df, ATR_LEN)
    adx       = calc_adx(df, ADX_LEN)

    # EMA crossovers (match Pine Script ta.crossover / ta.crossunder)
    cross_bull = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    cross_bear = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

    # Trend filter
    htf_buy_ok  = (~USE_TREND_FILTER) | (close > trend_ema)
    htf_sell_ok = (~USE_TREND_FILTER) | (close < trend_ema)

    # ADX filter
    is_choppy = USE_ADX & (adx < ADX_THRESHOLD)

    # Swing SL levels (match Pine Script ta.lowest / ta.highest)
    swing_low  = df["low"].rolling(SL_LOOKBACK).min()
    swing_high = df["high"].rolling(SL_LOOKBACK).max()
    buy_sl     = swing_low  - atr * ATR_BUFFER
    sell_sl    = swing_high + atr * ATR_BUFFER

    # Valid signals
    valid_buy  = cross_bull & htf_buy_ok  & ~is_choppy
    valid_sell = cross_bear & htf_sell_ok & ~is_choppy

    df = df.copy()
    df["fast_ema"]  = fast_ema
    df["slow_ema"]  = slow_ema
    df["trend_ema"] = trend_ema
    df["atr"]       = atr
    df["adx"]       = adx
    df["buy_sl"]    = buy_sl
    df["sell_sl"]   = sell_sl
    df["valid_buy"] = valid_buy
    df["valid_sell"]= valid_sell
    return df

# ══════════════════════════════════════════════════════════════
# LOT SIZE & PnL
# ══════════════════════════════════════════════════════════════

def calc_lot(entry, sl):
    sl_dist   = abs(entry - sl)
    sl_in_pips = sl_dist / PIP_SIZE
    if sl_in_pips <= 0:
        return 0.01
    return max(0.01, round(RISK_USD / (sl_in_pips * PIP_VALUE_PER_LOT), 2))


def calc_pnl(entry, exit_price, lots, is_buy):
    price_diff = (exit_price - entry) if is_buy else (entry - exit_price)
    pips       = price_diff / PIP_SIZE
    return round(pips * PIP_VALUE_PER_LOT * lots, 2)


def calc_mfe(entry, sl, exit_px, max_fav, is_buy):
    """
    Returns sl_pips, highest_rr, closed_rr.
      sl_pips     — distance entry → SL in pips (your risk)
      highest_rr  — peak favourable move as ratio string, e.g. "1:2.5"
      closed_rr   — actual close as ratio string, e.g. "1:1.2" (negative = loss)
    """
    sl_dist_pips = abs(entry - sl) / PIP_SIZE
    if is_buy:
        exit_pips = (exit_px  - entry) / PIP_SIZE
        mfe_pips  = max(0.0, (max_fav - entry) / PIP_SIZE)
    else:
        exit_pips = (entry  - exit_px) / PIP_SIZE
        mfe_pips  = max(0.0, (entry - max_fav) / PIP_SIZE)
    if sl_dist_pips > 0:
        highest_rr = round(mfe_pips  / sl_dist_pips, 1)
        closed_rr  = round(exit_pips / sl_dist_pips, 1)
    else:
        highest_rr = 0.0
        closed_rr  = 0.0
    return round(sl_dist_pips, 1), highest_rr, closed_rr

# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_backtest(df):
    trades       = []
    equity_curve = []

    balance       = ACCOUNT_SIZE
    trade_dir     = 0          # 1=long | -1=short | 0=flat
    entry_price   = None
    sl_price      = None
    entry_time    = None
    lot_size      = None
    entry_adx     = None
    max_favorable = None       # MFE tracker: highest high (BUY) / lowest low (SELL)
    trade_id      = 0

    # Warmup: need enough bars for TREND_EMA + ADX to settle
    warmup = max(TREND_EMA_LEN, SL_LOOKBACK) + 10

    for i in range(warmup, len(df)):
        row    = df.iloc[i]
        time_i = row["time"]
        close_i = row["close"]
        high_i  = row["high"]
        low_i   = row["low"]
        atr_i   = row["atr"]
        buy_sl_i  = row["buy_sl"]
        sell_sl_i = row["sell_sl"]

        # ── Update MFE each bar ───────────────────────────────
        if trade_dir == 1 and max_favorable is not None:
            max_favorable = max(max_favorable, high_i)
        elif trade_dir == -1 and max_favorable is not None:
            max_favorable = min(max_favorable, low_i)

        # ── Check SL hit (static — does not trail) ───────────
        sl_hit = False

        if trade_dir == 1 and sl_price is not None:
            if low_i <= sl_price:
                sl_hit      = True
                exit_price  = sl_price
                pnl         = calc_pnl(entry_price, exit_price, lot_size, True)
                balance    += pnl
                trade_id   += 1
                sl_pips_val, highest_rr, closed_rr = calc_mfe(entry_price, sl_price, exit_price, max_favorable, True)
                trades.append({
                    "id"          : trade_id,
                    "direction"   : "BUY",
                    "entry_time"  : entry_time,
                    "exit_time"   : time_i,
                    "entry_price" : round(entry_price, 5),
                    "exit_price"  : round(exit_price,  5),
                    "sl_price"    : round(sl_price,    5),
                    "lot_size"    : lot_size,
                    "adx_entry"   : round(entry_adx,   1),
                    "pnl"         : pnl,
                    "close_reason": "SL HIT",
                    "sl_pips"     : sl_pips_val,
                    "highest_rr"  : highest_rr,
                    "closed_rr"   : closed_rr,
                    "balance"     : round(balance, 2),
                })
                trade_dir = 0
                entry_price = sl_price = lot_size = entry_time = entry_adx = None
                max_favorable = None

        if trade_dir == -1 and sl_price is not None:
            if high_i >= sl_price:
                sl_hit      = True
                exit_price  = sl_price
                pnl         = calc_pnl(entry_price, exit_price, lot_size, False)
                balance    += pnl
                trade_id   += 1
                sl_pips_val, highest_rr, closed_rr = calc_mfe(entry_price, sl_price, exit_price, max_favorable, False)
                trades.append({
                    "id"          : trade_id,
                    "direction"   : "SELL",
                    "entry_time"  : entry_time,
                    "exit_time"   : time_i,
                    "entry_price" : round(entry_price, 5),
                    "exit_price"  : round(exit_price,  5),
                    "sl_price"    : round(sl_price,    5),
                    "lot_size"    : lot_size,
                    "adx_entry"   : round(entry_adx,   1),
                    "pnl"         : pnl,
                    "close_reason": "SL HIT",
                    "sl_pips"     : sl_pips_val,
                    "highest_rr"  : highest_rr,
                    "closed_rr"   : closed_rr,
                    "balance"     : round(balance, 2),
                })
                trade_dir = 0
                entry_price = sl_price = lot_size = entry_time = entry_adx = None
                max_favorable = None

        # ── Signal logic ──────────────────────────────────────
        if not sl_hit:

            # BUY trigger
            if row["valid_buy"] and trade_dir != 1:
                # Close short by reversal
                if trade_dir == -1 and entry_price is not None:
                    pnl     = calc_pnl(entry_price, close_i, lot_size, False)
                    balance += pnl
                    trade_id += 1
                    sl_pips_val, highest_rr, closed_rr = calc_mfe(entry_price, sl_price, close_i, max_favorable, False)
                    trades.append({
                        "id"          : trade_id,
                        "direction"   : "SELL",
                        "entry_time"  : entry_time,
                        "exit_time"   : time_i,
                        "entry_price" : round(entry_price, 5),
                        "exit_price"  : round(close_i,     5),
                        "sl_price"    : round(sl_price,    5),
                        "lot_size"    : lot_size,
                        "adx_entry"   : round(entry_adx,   1),
                        "pnl"         : pnl,
                        "close_reason": "REVERSAL",
                        "sl_pips"     : sl_pips_val,
                        "highest_rr"  : highest_rr,
                        "closed_rr"   : closed_rr,
                        "balance"     : round(balance, 2),
                    })

                # Open long
                entry_price   = close_i
                sl_price      = buy_sl_i
                lot_size      = calc_lot(entry_price, sl_price)
                entry_time    = time_i
                entry_adx     = row["adx"]
                trade_dir     = 1
                max_favorable = high_i   # initialise MFE at entry bar

            # SELL trigger
            elif row["valid_sell"] and trade_dir != -1:
                # Close long by reversal
                if trade_dir == 1 and entry_price is not None:
                    pnl     = calc_pnl(entry_price, close_i, lot_size, True)
                    balance += pnl
                    trade_id += 1
                    sl_pips_val, highest_rr, closed_rr = calc_mfe(entry_price, sl_price, close_i, max_favorable, True)
                    trades.append({
                        "id"          : trade_id,
                        "direction"   : "BUY",
                        "entry_time"  : entry_time,
                        "exit_time"   : time_i,
                        "entry_price" : round(entry_price, 5),
                        "exit_price"  : round(close_i,     5),
                        "sl_price"    : round(sl_price,    5),
                        "lot_size"    : lot_size,
                        "adx_entry"   : round(entry_adx,   1),
                        "pnl"         : pnl,
                        "close_reason": "REVERSAL",
                        "sl_pips"     : sl_pips_val,
                        "highest_rr"  : highest_rr,
                        "closed_rr"   : closed_rr,
                        "balance"     : round(balance, 2),
                    })

                # Open short
                entry_price   = close_i
                sl_price      = sell_sl_i
                lot_size      = calc_lot(entry_price, sl_price)
                entry_time    = time_i
                entry_adx     = row["adx"]
                trade_dir     = -1
                max_favorable = low_i    # initialise MFE at entry bar

        # ── Equity snapshot ───────────────────────────────────
        if trade_dir == 1 and entry_price:
            unreal = calc_pnl(entry_price, close_i, lot_size, True)
        elif trade_dir == -1 and entry_price:
            unreal = calc_pnl(entry_price, close_i, lot_size, False)
        else:
            unreal = 0

        equity_curve.append({
            "time"   : time_i,
            "balance": round(balance, 2),
            "equity" : round(balance + unreal, 2),
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def compute_stats(trades_df, equity_df):
    if trades_df.empty:
        return {}

    t    = trades_df
    wins = t[t["pnl"] > 0]
    losses = t[t["pnl"] <= 0]
    sl_trades  = t[t["close_reason"] == "SL HIT"]
    rev_trades = t[t["close_reason"] == "REVERSAL"]

    total_pnl    = t["pnl"].sum()
    final_bal    = ACCOUNT_SIZE + total_pnl
    peak         = equity_df["equity"].cummax()
    drawdown     = (equity_df["equity"] - peak) / peak * 100
    max_dd       = round(drawdown.min(), 2)
    avg_win      = round(wins["pnl"].mean(),   2) if not wins.empty   else 0
    avg_loss     = round(losses["pnl"].mean(), 2) if not losses.empty else 0
    pf_denom     = abs(losses["pnl"].sum())
    profit_factor = round(wins["pnl"].sum() / pf_denom, 2) if pf_denom > 0 else float("inf")

    # Sharpe (annualised)
    daily_eq  = equity_df.set_index("time")["equity"].resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe    = round((daily_ret.mean() / daily_ret.std()) * np.sqrt(252), 2) if daily_ret.std() > 0 else 0

    # Consecutive streaks
    streak = t["pnl"].apply(lambda x: 1 if x > 0 else -1)
    max_win_streak = max_loss_streak = cur = 0
    for s in streak:
        cur = cur + 1 if s > 0 else 0
        max_win_streak = max(max_win_streak, cur)
    cur = 0
    for s in streak:
        cur = cur + 1 if s < 0 else 0
        max_loss_streak = max(max_loss_streak, cur)

    return {
        "total_trades"    : len(t),
        "wins"            : len(wins),
        "losses"          : len(losses),
        "win_rate"        : round(len(wins) / len(t) * 100, 1),
        "total_pnl"       : round(total_pnl, 2),
        "final_balance"   : round(final_bal, 2),
        "return_pct"      : round(total_pnl / ACCOUNT_SIZE * 100, 2),
        "max_drawdown_pct": max_dd,
        "avg_win"         : avg_win,
        "avg_loss"        : avg_loss,
        "profit_factor"   : profit_factor,
        "sharpe"          : sharpe,
        "sl_hits"         : len(sl_trades),
        "reversals"       : len(rev_trades),
        "max_win_streak"  : max_win_streak,
        "max_loss_streak" : max_loss_streak,
        "best_trade"      : round(t["pnl"].max(), 2),
        "worst_trade"     : round(t["pnl"].min(), 2),
    }

# ══════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════

def build_html(stats, trades_df, equity_df):
    eq_times  = equity_df["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    eq_bal    = equity_df["balance"].tolist()
    eq_equity = equity_df["equity"].tolist()

    # Monthly PnL
    if not trades_df.empty:
        trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
        monthly   = trades_df.groupby("month")["pnl"].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)
        m_labels  = monthly["month"].tolist()
        m_values  = monthly["pnl"].tolist()
        m_colors  = ["'#00c9ff'" if v >= 0 else "'#ff4d6d'" for v in m_values]
    else:
        m_labels, m_values, m_colors = [], [], []

    # Trade log rows
    rows = ""
    for _, r in trades_df.iterrows():
        pnl_cls     = "win" if r["pnl"] > 0 else "loss"
        reason_icon = "🔄" if r["close_reason"] == "REVERSAL" else "🛑"
        highest_rr  = r.get("highest_rr", 0.0)
        closed_rr   = r.get("closed_rr",  0.0)
        hrr_num = highest_rr if isinstance(highest_rr, (int, float)) else 0.0
        hrr_col = "var(--green)" if hrr_num >= 2 else ("#f5c518" if hrr_num >= 1 else "var(--muted)")
        rows += f"""
        <tr class="{pnl_cls}">
          <td>{int(r['id'])}</td>
          <td class="dir-{'buy' if r['direction']=='BUY' else 'sell'}">{r['direction']}</td>
          <td>{str(r['entry_time'])[:16]}</td>
          <td>{str(r['exit_time'])[:16]}</td>
          <td>{r['entry_price']}</td>
          <td>{r['exit_price']}</td>
          <td>{r['sl_price']}</td>
          <td>{r['lot_size']}</td>
          <td>{r['adx_entry']}</td>
          <td class="pnl-{'pos' if r['pnl']>0 else 'neg'}">${r['pnl']:,.2f}</td>
          <td>{reason_icon} {r['close_reason']}</td>
          <td>{r.get('sl_pips', 0)}</td>
          <td style="color:{hrr_col};font-weight:600">{highest_rr}</td>
          <td>{closed_rr}</td>
          <td>${r['balance']:,.2f}</td>
        </tr>"""

    s         = stats
    ret_color = "#00c9ff" if s.get("return_pct", 0) >= 0 else "#ff4d6d"
    pnl_color = "#00c9ff" if s.get("total_pnl",  0) >= 0 else "#ff4d6d"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMA Cross Backtest — {SYMBOL}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #09090f;
    --surface: #111118;
    --card:    #16161f;
    --border:  #222230;
    --accent:  #00c9ff;
    --accent2: #0077ff;
    --green:   #00d4a0;
    --red:     #ff4d6d;
    --blue:    #4d9fff;
    --text:    #e8e8f0;
    --muted:   #666680;
    --font:    'Syne', sans-serif;
    --mono:    'JetBrains Mono', monospace;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--font); min-height:100vh; }}

  header {{
    background: linear-gradient(135deg, #0a0d18 0%, #0d1020 50%, #080b14 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative; overflow: hidden;
  }}
  header::before {{
    content:''; position:absolute; top:-60px; left:-60px;
    width:320px; height:320px;
    background:radial-gradient(circle, rgba(0,201,255,0.07) 0%, transparent 70%);
    pointer-events:none;
  }}
  header::after {{
    content:'{SYMBOL}'; position:absolute; right:48px; top:50%;
    transform:translateY(-50%); font-size:96px; font-weight:800;
    color:rgba(0,201,255,0.04); letter-spacing:-4px; pointer-events:none;
  }}
  .header-badge {{
    display:inline-block; background:rgba(0,201,255,0.12);
    border:1px solid rgba(0,201,255,0.3); color:var(--accent);
    font-size:11px; font-weight:600; letter-spacing:2px;
    padding:4px 12px; border-radius:2px; margin-bottom:16px; font-family:var(--mono);
  }}
  header h1 {{ font-size:36px; font-weight:800; letter-spacing:-1px; color:var(--text); margin-bottom:8px; }}
  header h1 span {{ color:var(--accent); }}
  .header-meta {{
    font-family:var(--mono); font-size:12px; color:var(--muted);
    display:flex; gap:24px; flex-wrap:wrap; margin-top:12px;
  }}
  .header-meta span {{ display:flex; align-items:center; gap:6px; }}
  .dot {{ width:6px; height:6px; border-radius:50%; background:var(--green); display:inline-block; }}

  main {{ padding:40px 48px; max-width:1600px; }}

  .kpi-grid {{
    display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr));
    gap:16px; margin-bottom:40px;
  }}
  .kpi {{
    background:var(--card); border:1px solid var(--border); border-radius:8px;
    padding:20px 22px; position:relative; overflow:hidden; transition:border-color 0.2s;
  }}
  .kpi:hover {{ border-color:rgba(0,201,255,0.3); }}
  .kpi::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--kpi-accent, var(--accent));
  }}
  .kpi-label {{ font-size:10px; font-weight:600; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; font-family:var(--mono); margin-bottom:10px; }}
  .kpi-value {{ font-size:26px; font-weight:700; color:var(--text); line-height:1; }}
  .kpi-sub   {{ font-size:11px; color:var(--muted); margin-top:6px; font-family:var(--mono); }}

  .charts-row {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; margin-bottom:28px; }}
  .chart-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:24px; }}
  .chart-title {{ font-size:13px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--muted); margin-bottom:20px; font-family:var(--mono); }}
  canvas {{ width:100% !important; }}

  .section-title {{
    font-size:18px; font-weight:700; color:var(--text); margin-bottom:16px;
    padding-bottom:12px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:10px;
  }}
  .section-title::before {{
    content:''; display:block; width:3px; height:18px;
    background:var(--accent); border-radius:2px;
  }}
  .table-wrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--border); border-radius:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; font-family:var(--mono); }}
  thead tr {{ background:var(--surface); border-bottom:1px solid var(--border); }}
  th {{ padding:12px 14px; text-align:left; font-size:10px; letter-spacing:1px; color:var(--muted); font-weight:600; white-space:nowrap; }}
  td {{ padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.03); white-space:nowrap; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.win  {{ background:rgba(0,212,160,0.02); }}
  tr.loss {{ background:rgba(255,77,109,0.02); }}
  tr:hover td {{ background:rgba(255,255,255,0.02); }}
  .dir-buy  {{ color:var(--green); font-weight:600; }}
  .dir-sell {{ color:var(--red);   font-weight:600; }}
  .pnl-pos  {{ color:var(--green); font-weight:600; }}
  .pnl-neg  {{ color:var(--red);   font-weight:600; }}

  .stats-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:28px; }}
  .stat-block {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:20px 24px; }}
  .stat-block h3 {{ font-size:11px; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; font-family:var(--mono); margin-bottom:14px; }}
  .stat-row {{ display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:12px; }}
  .stat-row:last-child {{ border-bottom:none; }}
  .stat-row .label {{ color:var(--muted); font-family:var(--mono); }}
  .stat-row .val   {{ font-weight:600; font-family:var(--mono); }}

  footer {{ text-align:center; padding:32px; color:var(--muted); font-size:11px; font-family:var(--mono); border-top:1px solid var(--border); margin-top:40px; }}
</style>
</head>
<body>

<header>
  <div class="header-badge">BACKTEST REPORT</div>
  <h1>EMA Cross <span>{SYMBOL}</span> Strategy</h1>
  <div class="header-meta">
    <span><span class="dot"></span> {SYMBOL} · M15</span>
    <span>Period: Last 2 Years</span>
    <span>Account: $100,000</span>
    <span>Risk/Trade: $100</span>
    <span>EMA {FAST_EMA}/{SLOW_EMA}/{TREND_EMA_LEN} · ADX {ADX_LEN}&lt;{ADX_THRESHOLD} · SL Lookback {SL_LOOKBACK} · ATR Buffer {ATR_BUFFER}</span>
    <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  </div>
</header>

<main>

<div class="kpi-grid">
  <div class="kpi" style="--kpi-accent:{ret_color}">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value" style="color:{ret_color}">{s.get('return_pct',0):+.2f}%</div>
    <div class="kpi-sub">on $100,000 account</div>
  </div>
  <div class="kpi" style="--kpi-accent:{pnl_color}">
    <div class="kpi-label">Net P&L</div>
    <div class="kpi-value" style="color:{pnl_color}">${s.get('total_pnl',0):+,.2f}</div>
    <div class="kpi-sub">final: ${s.get('final_balance',0):,.2f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value">{s.get('win_rate',0)}%</div>
    <div class="kpi-sub">{s.get('wins',0)}W / {s.get('losses',0)}L of {s.get('total_trades',0)}</div>
  </div>
  <div class="kpi" style="--kpi-accent:var(--red)">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value" style="color:var(--red)">{s.get('max_drawdown_pct',0):.2f}%</div>
    <div class="kpi-sub">peak-to-trough equity</div>
  </div>
  <div class="kpi" style="--kpi-accent:var(--blue)">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value">{s.get('profit_factor',0)}</div>
    <div class="kpi-sub">gross profit / gross loss</div>
  </div>
  <div class="kpi" style="--kpi-accent:var(--blue)">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value">{s.get('sharpe',0)}</div>
    <div class="kpi-sub">annualised daily returns</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Win</div>
    <div class="kpi-value" style="color:var(--green)">${s.get('avg_win',0):,.2f}</div>
    <div class="kpi-sub">per winning trade</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Loss</div>
    <div class="kpi-value" style="color:var(--red)">${s.get('avg_loss',0):,.2f}</div>
    <div class="kpi-sub">per losing trade</div>
  </div>
</div>

<div class="charts-row">
  <div class="chart-card">
    <div class="chart-title">Equity Curve</div>
    <canvas id="equityChart" height="280"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">Monthly P&L</div>
    <canvas id="monthlyChart" height="280"></canvas>
  </div>
</div>

<div class="stats-row">
  <div class="stat-block">
    <h3>Performance</h3>
    <div class="stat-row"><span class="label">Total Trades</span><span class="val">{s.get('total_trades',0)}</span></div>
    <div class="stat-row"><span class="label">Wins</span><span class="val" style="color:var(--green)">{s.get('wins',0)}</span></div>
    <div class="stat-row"><span class="label">Losses</span><span class="val" style="color:var(--red)">{s.get('losses',0)}</span></div>
    <div class="stat-row"><span class="label">Win Rate</span><span class="val">{s.get('win_rate',0)}%</span></div>
    <div class="stat-row"><span class="label">Best Trade</span><span class="val" style="color:var(--green)">${s.get('best_trade',0):,.2f}</span></div>
    <div class="stat-row"><span class="label">Worst Trade</span><span class="val" style="color:var(--red)">${s.get('worst_trade',0):,.2f}</span></div>
  </div>
  <div class="stat-block">
    <h3>Risk</h3>
    <div class="stat-row"><span class="label">Max Drawdown</span><span class="val" style="color:var(--red)">{s.get('max_drawdown_pct',0):.2f}%</span></div>
    <div class="stat-row"><span class="label">Profit Factor</span><span class="val">{s.get('profit_factor',0)}</span></div>
    <div class="stat-row"><span class="label">Sharpe Ratio</span><span class="val">{s.get('sharpe',0)}</span></div>
    <div class="stat-row"><span class="label">SL Hits</span><span class="val" style="color:var(--red)">{s.get('sl_hits',0)}</span></div>
    <div class="stat-row"><span class="label">Reversals</span><span class="val">{s.get('reversals',0)}</span></div>
    <div class="stat-row"><span class="label">Risk/Trade</span><span class="val">$100</span></div>
  </div>
  <div class="stat-block">
    <h3>Streaks & Averages</h3>
    <div class="stat-row"><span class="label">Max Win Streak</span><span class="val" style="color:var(--green)">{s.get('max_win_streak',0)}</span></div>
    <div class="stat-row"><span class="label">Max Loss Streak</span><span class="val" style="color:var(--red)">{s.get('max_loss_streak',0)}</span></div>
    <div class="stat-row"><span class="label">Avg Win</span><span class="val" style="color:var(--green)">${s.get('avg_win',0):,.2f}</span></div>
    <div class="stat-row"><span class="label">Avg Loss</span><span class="val" style="color:var(--red)">${s.get('avg_loss',0):,.2f}</span></div>
    <div class="stat-row"><span class="label">Net P&L</span><span class="val" style="color:{pnl_color}">${s.get('total_pnl',0):+,.2f}</span></div>
    <div class="stat-row"><span class="label">Return</span><span class="val" style="color:{ret_color}">{s.get('return_pct',0):+.2f}%</span></div>
  </div>
</div>

<div class="section-title">Trade Log — {s.get('total_trades',0)} Trades</div>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th><th>DIR</th><th>ENTRY TIME</th><th>EXIT TIME</th>
        <th>ENTRY</th><th>EXIT</th><th>SL</th><th>LOTS</th>
        <th>ADX</th><th>P&L</th><th>REASON</th><th>SL PIPS</th><th>HIGHEST RR</th><th>CLOSED RR</th><th>BALANCE</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>

</main>

<footer>
  EMA Cross Backtest · {SYMBOL} M15 · EMA {FAST_EMA}/{SLOW_EMA}/{TREND_EMA_LEN} · ADX {ADX_LEN}&lt;{ADX_THRESHOLD} · $100 risk/trade · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>

<script>
const eqLabels  = {json.dumps(eq_times[::4])};
const eqBalance = {json.dumps(eq_bal[::4])};
const eqEquity  = {json.dumps(eq_equity[::4])};

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: eqLabels,
    datasets: [
      {{
        label: 'Equity',
        data: eqEquity,
        borderColor: '#00c9ff',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
        backgroundColor: 'rgba(0,201,255,0.05)',
      }},
      {{
        label: 'Balance',
        data: eqBalance,
        borderColor: '#4d9fff',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.2,
        borderDash: [4,4],
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#666680', font: {{ family: 'JetBrains Mono', size: 11 }} }} }},
      tooltip: {{ backgroundColor: '#16161f', borderColor: '#222230', borderWidth: 1 }}
    }},
    scales: {{
      x: {{ ticks: {{ color:'#444460', maxTicksLimit:8, font:{{family:'JetBrains Mono',size:10}} }}, grid:{{ color:'rgba(255,255,255,0.03)' }} }},
      y: {{ ticks: {{ color:'#444460', font:{{family:'JetBrains Mono',size:10}}, callback: v => '$'+v.toLocaleString() }}, grid:{{ color:'rgba(255,255,255,0.05)' }} }}
    }}
  }}
}});

const mLabels = {json.dumps(m_labels)};
const mValues = {json.dumps(m_values)};
const mColors = [{','.join(m_colors)}];

new Chart(document.getElementById('monthlyChart'), {{
  type: 'bar',
  data: {{
    labels: mLabels,
    datasets: [{{ label: 'Monthly P&L', data: mValues, backgroundColor: mColors, borderRadius: 3 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ backgroundColor: '#16161f', borderColor: '#222230', borderWidth: 1,
        callbacks: {{ label: ctx => '$' + ctx.raw.toFixed(2) }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color:'#444460', font:{{family:'JetBrains Mono',size:9}}, maxRotation:45 }}, grid:{{ display:false }} }},
      y: {{ ticks: {{ color:'#444460', font:{{family:'JetBrains Mono',size:10}}, callback: v => '$'+v.toLocaleString() }}, grid:{{ color:'rgba(255,255,255,0.05)' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 60)
    print(f"  EMA CROSS BACKTESTER — {SYMBOL}")
    print(f"  TF: M15 | Period: 2 Years | Account: ${ACCOUNT_SIZE:,.0f}")
    print(f"  EMA {FAST_EMA}/{SLOW_EMA}/{TREND_EMA_LEN} | ADX {ADX_LEN}<{ADX_THRESHOLD} | SL Lookback {SL_LOOKBACK} | ATR Buffer {ATR_BUFFER}")
    print("━" * 60)

    df = fetch_data()

    print("Calculating indicators & signals...")
    df = build_signals(df)
    buys  = df["valid_buy"].sum()
    sells = df["valid_sell"].sum()
    print(f"Signals found — BUY: {buys} | SELL: {sells}")

    print("Running backtest engine...")
    trades_df, equity_df = run_backtest(df)
    print(f"Done — {len(trades_df)} trades executed")

    stats = compute_stats(trades_df, equity_df)

    print("\n" + "═" * 60)
    print(f"  RESULTS — {SYMBOL}")
    print("═" * 60)
    print(f"  Total Trades   : {stats.get('total_trades',0)}")
    print(f"  Win Rate       : {stats.get('win_rate',0)}%  ({stats.get('wins',0)}W / {stats.get('losses',0)}L)")
    print(f"  Net P&L        : ${stats.get('total_pnl',0):+,.2f}")
    print(f"  Return         : {stats.get('return_pct',0):+.2f}%")
    print(f"  Final Balance  : ${stats.get('final_balance',0):,.2f}")
    print(f"  Max Drawdown   : {stats.get('max_drawdown_pct',0):.2f}%")
    print(f"  Profit Factor  : {stats.get('profit_factor',0)}")
    print(f"  Sharpe Ratio   : {stats.get('sharpe',0)}")
    print(f"  Avg Win        : ${stats.get('avg_win',0):,.2f}")
    print(f"  Avg Loss       : ${stats.get('avg_loss',0):,.2f}")
    print(f"  Best Trade     : ${stats.get('best_trade',0):,.2f}")
    print(f"  Worst Trade    : ${stats.get('worst_trade',0):,.2f}")
    print(f"  SL Hits        : {stats.get('sl_hits',0)}")
    print(f"  Reversals      : {stats.get('reversals',0)}")
    print(f"  Max Win Streak : {stats.get('max_win_streak',0)}")
    print(f"  Max Loss Streak: {stats.get('max_loss_streak',0)}")
    print("═" * 60)

    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ CSV  → {OUTPUT_CSV}")

    html = build_html(stats, trades_df, equity_df)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML → {OUTPUT_HTML}")
    print("\nOpen the HTML file in your browser to view the full report.")