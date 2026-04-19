"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 UT BOT — GOLD BACKTESTER  |  XAUUSD M15  |  2 Years
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Mirrors live bot logic exactly:
  • ATR Trailing Stop  (Key Value × ATR)
  • RSI Filter         (Buy 45-65 | Sell 35-55)
  • Strict alternating (Buy → Sell → Buy)
  • $100 auto risk     (lot = 100 / SL_dist × 100)
  • SL hit → go flat
  • $100,000 starting account
 Outputs:
  • backtest_report.html  — visual report with equity curve + charts
  • backtest_trades.csv   — full trade-by-trade log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SYMBOL        = "XAUUSD"
TIMEFRAME     = mt5.TIMEFRAME_M15
ACCOUNT_SIZE  = 100_000.0      # starting balance USD
RISK_USD      = 100.0          # fixed risk per trade
KEY_VALUE     = 3              # UT Bot sensitivity
ATR_PERIOD    = 14
USE_HA        = False
RSI_PERIOD    = 14
RSI_BUY_LO    = 45
RSI_BUY_HI    = 65
RSI_SELL_LO   = 35
RSI_SELL_HI   = 55
YEARS_BACK    = 2
_RUN_TS       = datetime.now().strftime("%Y%m%d%H%M%S")
OUTPUT_HTML   = f"backtest_report_{_RUN_TS}.html"
OUTPUT_CSV    = f"backtest_trades_{_RUN_TS}.csv"

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

def heikin_ashi_close(df):
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4

def calc_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_atr_trailing_stop(src, atr, key):
    n_loss = key * atr
    ts     = np.zeros(len(src))
    for i in range(1, len(src)):
        prev = ts[i - 1]
        s    = src.iloc[i]
        sp   = src.iloc[i - 1]
        if s > prev and sp > prev:
            ts[i] = max(prev, s - n_loss.iloc[i])
        elif s < prev and sp < prev:
            ts[i] = min(prev, s + n_loss.iloc[i])
        elif s > prev:
            ts[i] = s - n_loss.iloc[i]
        else:
            ts[i] = s + n_loss.iloc[i]
    return pd.Series(ts, index=src.index)

def calc_rsi(src, period):
    delta = src.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al
    return 100 - (100 / (1 + rs))

def build_signals(df):
    src = heikin_ashi_close(df) if USE_HA else df["close"].copy()
    atr = calc_atr(df, ATR_PERIOD)
    ts  = calc_atr_trailing_stop(src, atr, KEY_VALUE)
    rsi = calc_rsi(src, RSI_PERIOD)

    above = (src.shift(1) < ts.shift(1)) & (src >= ts)
    below = (ts.shift(1) < src.shift(1)) & (ts >= src)

    raw_buy  = (src > ts) & above
    raw_sell = (src < ts) & below

    qual_buy  = raw_buy  & (rsi >= RSI_BUY_LO)  & (rsi <= RSI_BUY_HI)
    qual_sell = raw_sell & (rsi >= RSI_SELL_LO) & (rsi <= RSI_SELL_HI)

    df = df.copy()
    df["src"]       = src
    df["ts"]        = ts
    df["rsi"]       = rsi
    df["raw_buy"]   = raw_buy
    df["raw_sell"]  = raw_sell
    df["qual_buy"]  = qual_buy
    df["qual_sell"] = qual_sell
    return df

# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def calc_lot(entry, sl):
    dist = abs(entry - sl)
    if dist == 0:
        return 0.01
    return max(0.01, round(RISK_USD / (dist * 100), 2))

def run_backtest(df):
    trades      = []
    equity_curve = []

    balance      = ACCOUNT_SIZE
    trade_dir    = 0        #  1=long | -1=short | 0=flat
    entry_price  = None
    sl_price     = None
    entry_time   = None
    lot_size     = None
    entry_rsi    = None
    trade_id     = 0

    # Start from bar 50 so ATR/RSI have enough warmup
    for i in range(50, len(df)):
        row     = df.iloc[i]
        time_i  = row["time"]
        close_i = row["src"]
        ts_i    = row["ts"]
        rsi_i   = row["rsi"]
        high_i  = row["high"]
        low_i   = row["low"]

        # ── Check SL hit on current bar (before signal) ───────
        sl_hit = False
        if trade_dir == 1 and sl_price is not None:
            if low_i <= sl_price:
                sl_hit     = True
                exit_price = sl_price
                pnl        = round((exit_price - entry_price) * lot_size * 100, 2)
                balance   += pnl
                trade_id  += 1
                trades.append({
                    "id"         : trade_id,
                    "direction"  : "BUY",
                    "entry_time" : entry_time,
                    "exit_time"  : time_i,
                    "entry_price": round(entry_price, 2),
                    "exit_price" : round(exit_price, 2),
                    "sl_price"   : round(sl_price, 2),
                    "lot_size"   : lot_size,
                    "rsi_entry"  : round(entry_rsi, 1),
                    "pnl"        : pnl,
                    "close_reason": "SL HIT",
                    "balance"    : round(balance, 2),
                })
                trade_dir = 0
                entry_price = sl_price = lot_size = entry_time = entry_rsi = None

        if trade_dir == -1 and sl_price is not None:
            if high_i >= sl_price:
                sl_hit     = True
                exit_price = sl_price
                pnl        = round((entry_price - exit_price) * lot_size * 100, 2)
                balance   += pnl
                trade_id  += 1
                trades.append({
                    "id"         : trade_id,
                    "direction"  : "SELL",
                    "entry_time" : entry_time,
                    "exit_time"  : time_i,
                    "entry_price": round(entry_price, 2),
                    "exit_price" : round(exit_price, 2),
                    "sl_price"   : round(sl_price, 2),
                    "lot_size"   : lot_size,
                    "rsi_entry"  : round(entry_rsi, 1),
                    "pnl"        : pnl,
                    "close_reason": "SL HIT",
                    "balance"    : round(balance, 2),
                })
                trade_dir = 0
                entry_price = sl_price = lot_size = entry_time = entry_rsi = None

        # ── Signal logic (only on completed bar, no same-side) ─
        if not sl_hit:
            # BUY trigger
            if row["qual_buy"] and trade_dir != 1:
                # Close short by reversal
                if trade_dir == -1 and entry_price is not None:
                    exit_price = close_i
                    pnl        = round((entry_price - exit_price) * lot_size * 100, 2)
                    balance   += pnl
                    trade_id  += 1
                    trades.append({
                        "id"         : trade_id,
                        "direction"  : "SELL",
                        "entry_time" : entry_time,
                        "exit_time"  : time_i,
                        "entry_price": round(entry_price, 2),
                        "exit_price" : round(exit_price, 2),
                        "sl_price"   : round(sl_price, 2),
                        "lot_size"   : lot_size,
                        "rsi_entry"  : round(entry_rsi, 1),
                        "pnl"        : pnl,
                        "close_reason": "REVERSAL",
                        "balance"    : round(balance, 2),
                    })

                # Open long
                entry_price = close_i
                sl_price    = ts_i
                lot_size    = calc_lot(entry_price, sl_price)
                entry_time  = time_i
                entry_rsi   = rsi_i
                trade_dir   = 1

            # SELL trigger
            elif row["qual_sell"] and trade_dir != -1:
                # Close long by reversal
                if trade_dir == 1 and entry_price is not None:
                    exit_price = close_i
                    pnl        = round((exit_price - entry_price) * lot_size * 100, 2)
                    balance   += pnl
                    trade_id  += 1
                    trades.append({
                        "id"         : trade_id,
                        "direction"  : "BUY",
                        "entry_time" : entry_time,
                        "exit_time"  : time_i,
                        "entry_price": round(entry_price, 2),
                        "exit_price" : round(exit_price, 2),
                        "sl_price"   : round(sl_price, 2),
                        "lot_size"   : lot_size,
                        "rsi_entry"  : round(entry_rsi, 1),
                        "pnl"        : pnl,
                        "close_reason": "REVERSAL",
                        "balance"    : round(balance, 2),
                    })

                # Open short
                entry_price = close_i
                sl_price    = ts_i
                lot_size    = calc_lot(entry_price, sl_price)
                entry_time  = time_i
                entry_rsi   = rsi_i
                trade_dir   = -1

        # ── Equity snapshot ────────────────────────────────────
        # Mark-to-market open position
        if trade_dir == 1 and entry_price:
            unreal = round((close_i - entry_price) * lot_size * 100, 2)
        elif trade_dir == -1 and entry_price:
            unreal = round((entry_price - close_i) * lot_size * 100, 2)
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

    t        = trades_df
    wins     = t[t["pnl"] > 0]
    losses   = t[t["pnl"] <= 0]
    sl_trades = t[t["close_reason"] == "SL HIT"]
    rev_trades = t[t["close_reason"] == "REVERSAL"]

    total_pnl    = t["pnl"].sum()
    final_bal    = ACCOUNT_SIZE + total_pnl
    peak         = equity_df["equity"].cummax()
    drawdown     = (equity_df["equity"] - peak) / peak * 100
    max_dd       = round(drawdown.min(), 2)
    avg_win      = round(wins["pnl"].mean(), 2)  if not wins.empty   else 0
    avg_loss     = round(losses["pnl"].mean(), 2) if not losses.empty else 0
    profit_factor = round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if not losses.empty and losses["pnl"].sum() != 0 else float("inf")

    # Sharpe (annualised, assuming M15 bars ~26,000/year)
    daily_eq = equity_df.set_index("time")["equity"].resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = round((daily_ret.mean() / daily_ret.std()) * np.sqrt(252), 2) if daily_ret.std() > 0 else 0

    # Consecutive wins/losses
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
        "total_trades"     : len(t),
        "wins"             : len(wins),
        "losses"           : len(losses),
        "win_rate"         : round(len(wins) / len(t) * 100, 1),
        "total_pnl"        : round(total_pnl, 2),
        "final_balance"    : round(final_bal, 2),
        "return_pct"       : round(total_pnl / ACCOUNT_SIZE * 100, 2),
        "max_drawdown_pct" : max_dd,
        "avg_win"          : avg_win,
        "avg_loss"         : avg_loss,
        "profit_factor"    : profit_factor,
        "sharpe"           : sharpe,
        "sl_hits"          : len(sl_trades),
        "reversals"        : len(rev_trades),
        "max_win_streak"   : max_win_streak,
        "max_loss_streak"  : max_loss_streak,
        "best_trade"       : round(t["pnl"].max(), 2),
        "worst_trade"      : round(t["pnl"].min(), 2),
    }

# ══════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════

def build_html(stats, trades_df, equity_df):
    eq_times  = equity_df["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    eq_bal    = equity_df["balance"].tolist()
    eq_equity = equity_df["equity"].tolist()

    # Monthly PnL bar chart
    if not trades_df.empty:
        trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
        monthly = trades_df.groupby("month")["pnl"].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)
        m_labels = monthly["month"].tolist()
        m_values = monthly["pnl"].tolist()
        m_colors = ["'#00d4a0'" if v >= 0 else "'#ff4d6d'" for v in m_values]
    else:
        m_labels, m_values, m_colors = [], [], []

    # Trade log rows
    rows = ""
    for _, r in trades_df.iterrows():
        pnl_cls = "win" if r["pnl"] > 0 else "loss"
        reason_icon = "🔄" if r["close_reason"] == "REVERSAL" else "🛑"
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
          <td>{r['rsi_entry']}</td>
          <td class="pnl-{'pos' if r['pnl']>0 else 'neg'}">${r['pnl']:,.2f}</td>
          <td>{reason_icon} {r['close_reason']}</td>
          <td>${r['balance']:,.2f}</td>
        </tr>"""

    s = stats
    ret_color  = "#00d4a0" if s.get("return_pct", 0) >= 0 else "#ff4d6d"
    pnl_color  = "#00d4a0" if s.get("total_pnl",  0) >= 0 else "#ff4d6d"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UT Bot Gold Backtest Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #09090f;
    --surface:  #111118;
    --card:     #16161f;
    --border:   #222230;
    --gold:     #f5c518;
    --gold2:    #e8a800;
    --green:    #00d4a0;
    --red:      #ff4d6d;
    --blue:     #4d9fff;
    --text:     #e8e8f0;
    --muted:    #666680;
    --font:     'Syne', sans-serif;
    --mono:     'JetBrains Mono', monospace;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
  }}

  /* ── HEADER ── */
  header {{
    background: linear-gradient(135deg, #0d0d16 0%, #12101a 50%, #0a0d14 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(245,197,24,0.07) 0%, transparent 70%);
    pointer-events: none;
  }}
  header::after {{
    content: 'XAUUSD';
    position: absolute;
    right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 96px;
    font-weight: 800;
    color: rgba(245,197,24,0.04);
    letter-spacing: -4px;
    pointer-events: none;
  }}
  .header-badge {{
    display: inline-block;
    background: rgba(245,197,24,0.12);
    border: 1px solid rgba(245,197,24,0.3);
    color: var(--gold);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 16px;
    font-family: var(--mono);
  }}
  header h1 {{
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
    color: var(--text);
    margin-bottom: 8px;
  }}
  header h1 span {{ color: var(--gold); }}
  .header-meta {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 12px;
  }}
  .header-meta span {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width:6px; height:6px; border-radius:50%; background: var(--green); display:inline-block; }}

  /* ── LAYOUT ── */
  main {{ padding: 40px 48px; max-width: 1600px; }}

  /* ── KPI GRID ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 40px;
  }}
  .kpi {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .kpi:hover {{ border-color: rgba(245,197,24,0.3); }}
  .kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, var(--gold));
  }}
  .kpi-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: var(--muted);
    text-transform: uppercase;
    font-family: var(--mono);
    margin-bottom: 10px;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }}
  .kpi-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
    font-family: var(--mono);
  }}

  /* ── CHARTS ── */
  .charts-row {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    margin-bottom: 28px;
  }}
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
  }}
  .chart-title {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    font-family: var(--mono);
  }}
  canvas {{ width: 100% !important; }}

  /* ── TRADE TABLE ── */
  .section-title {{
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title::before {{
    content: '';
    display: block;
    width: 3px; height: 18px;
    background: var(--gold);
    border-radius: 2px;
  }}
  .table-wrap {{
    overflow-x: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: var(--mono);
  }}
  thead tr {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  th {{
    padding: 12px 14px;
    text-align: left;
    font-size: 10px;
    letter-spacing: 1px;
    color: var(--muted);
    font-weight: 600;
    white-space: nowrap;
  }}
  td {{ padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.03); white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.win {{ background: rgba(0,212,160,0.02); }}
  tr.loss {{ background: rgba(255,77,109,0.02); }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .dir-buy  {{ color: var(--green); font-weight: 600; }}
  .dir-sell {{ color: var(--red);   font-weight: 600; }}
  .pnl-pos  {{ color: var(--green); font-weight: 600; }}
  .pnl-neg  {{ color: var(--red);   font-weight: 600; }}

  /* ── STATS ROW ── */
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 28px;
  }}
  .stat-block {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
  }}
  .stat-block h3 {{
    font-size: 11px;
    letter-spacing: 1.5px;
    color: var(--muted);
    text-transform: uppercase;
    font-family: var(--mono);
    margin-bottom: 14px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 12px;
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-row .label {{ color: var(--muted); font-family: var(--mono); }}
  .stat-row .val   {{ font-weight: 600; font-family: var(--mono); }}

  footer {{
    text-align: center;
    padding: 32px;
    color: var(--muted);
    font-size: 11px;
    font-family: var(--mono);
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}
</style>
</head>
<body>

<header>
  <div class="header-badge">BACKTEST REPORT</div>
  <h1>UT Bot <span>Gold</span> Strategy</h1>
  <div class="header-meta">
    <span><span class="dot"></span> XAUUSD · M15</span>
    <span>Period: Last 2 Years</span>
    <span>Account: $100,000</span>
    <span>Risk/Trade: $100</span>
    <span>Key: {KEY_VALUE} · ATR: {ATR_PERIOD} · RSI {RSI_BUY_LO}-{RSI_BUY_HI}/{RSI_SELL_LO}-{RSI_SELL_HI}</span>
    <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  </div>
</header>

<main>

<!-- KPI CARDS -->
<div class="kpi-grid">
  <div class="kpi" style="--accent:{ret_color}">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value" style="color:{ret_color}">{s.get('return_pct',0):+.2f}%</div>
    <div class="kpi-sub">on $100,000 account</div>
  </div>
  <div class="kpi" style="--accent:{pnl_color}">
    <div class="kpi-label">Net P&L</div>
    <div class="kpi-value" style="color:{pnl_color}">${s.get('total_pnl',0):+,.2f}</div>
    <div class="kpi-sub">final: ${s.get('final_balance',0):,.2f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value">{s.get('win_rate',0)}%</div>
    <div class="kpi-sub">{s.get('wins',0)}W / {s.get('losses',0)}L of {s.get('total_trades',0)}</div>
  </div>
  <div class="kpi" style="--accent:var(--red)">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value" style="color:var(--red)">{s.get('max_drawdown_pct',0):.2f}%</div>
    <div class="kpi-sub">peak-to-trough equity</div>
  </div>
  <div class="kpi" style="--accent:var(--blue)">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value">{s.get('profit_factor',0)}</div>
    <div class="kpi-sub">gross profit / gross loss</div>
  </div>
  <div class="kpi" style="--accent:var(--blue)">
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

<!-- EQUITY CURVE + MONTHLY BAR -->
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

<!-- DETAILED STATS -->
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

<!-- TRADE LOG -->
<div class="section-title">Trade Log — {s.get('total_trades',0)} Trades</div>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th><th>DIR</th><th>ENTRY TIME</th><th>EXIT TIME</th>
        <th>ENTRY $</th><th>EXIT $</th><th>SL $</th><th>LOTS</th>
        <th>RSI</th><th>P&L</th><th>REASON</th><th>BALANCE</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>

</main>

<footer>
  UT Bot Gold Backtest · XAUUSD M15 · Key={KEY_VALUE} ATR={ATR_PERIOD} · RSI {RSI_BUY_LO}-{RSI_BUY_HI}/{RSI_SELL_LO}-{RSI_SELL_HI} · $100 risk/trade · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>

<script>
// ── Equity Curve ────────────────────────────────────────────
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
        borderColor: '#f5c518',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
        backgroundColor: 'rgba(245,197,24,0.05)',
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

// ── Monthly Bar ──────────────────────────────────────────────
const mLabels = {json.dumps(m_labels)};
const mValues = {json.dumps(m_values)};
const mColors = [{','.join(m_colors)}];

new Chart(document.getElementById('monthlyChart'), {{
  type: 'bar',
  data: {{
    labels: mLabels,
    datasets: [{{
      label: 'Monthly P&L',
      data: mValues,
      backgroundColor: mColors,
      borderRadius: 3,
    }}]
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
    print("  UT BOT GOLD BACKTESTER")
    print(f"  Symbol: {SYMBOL} | TF: M15 | Period: 2 Years")
    print(f"  Account: ${ACCOUNT_SIZE:,.0f} | Risk/Trade: ${RISK_USD}")
    print(f"  Key: {KEY_VALUE} | ATR: {ATR_PERIOD} | RSI {RSI_BUY_LO}-{RSI_BUY_HI}/{RSI_SELL_LO}-{RSI_SELL_HI}")
    print("━" * 60)

    # 1. Fetch data
    df = fetch_data()

    # 2. Build signals
    print("Calculating indicators & signals...")
    df = build_signals(df)

    # 3. Run backtest
    print("Running backtest engine...")
    trades_df, equity_df = run_backtest(df)
    print(f"Done — {len(trades_df)} trades found")

    # 4. Stats
    stats = compute_stats(trades_df, equity_df)

    # 5. Print summary
    print("\n" + "═" * 60)
    print("  BACKTEST RESULTS SUMMARY")
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

    # 6. Save CSV
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ CSV saved  → {OUTPUT_CSV}")

    # 7. Save HTML
    html = build_html(stats, trades_df, equity_df)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML saved → {OUTPUT_HTML}")
    print("\nOpen backtest_report.html in your browser to view the full report.")
