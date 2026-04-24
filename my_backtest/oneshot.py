"""
================================================================================
 ONE-SHOT ALGO V2 - MULTI-PROTOCOL BACKTESTER
--------------------------------------------------------------------------------
 Replicates the 4 protocols (T1 Swing, S1 Cross, S2 Slope, S3 BBands)
 with the Universal Risk Engine (ATR-based boxes) from the TradingView script
 parameters. Pulls OHLCV data directly from MetaTrader 5 (no CSV needed).

 Builds an interactive HTML report with:
   - Equity curve + drawdown chart
   - Full trade list table
   - Stats summary (PF, WR, Sharpe, MaxDD, expectancy, consec losses)
   - Price chart with trade markers (per symbol)

 USAGE (Windows + MT5 terminal installed + logged in):
     pip install MetaTrader5 pandas numpy plotly
     python oneshot_algo_v2_backtester.py

 NOTE: The original TradingView script is closed-source / invite-only.
 This implementation is a best-effort replica based on the exposed parameters
 in the indicator's input panel (EMA crossover, EMA slope, Bollinger Bands,
 ATR risk boxes, breakeven trail, 4H displacement for T1 Swing).
 Tune parameters at the top of the file to match your live-chart settings.
================================================================================
"""

from __future__ import annotations

import os
import sys
import math
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================
# 1. USER CONFIG  -  edit this block to match your live-chart setup
# ================================================================

# --- Symbols to backtest (MT5 symbol names - adjust for your broker) ---
SYMBOLS = [
  #  "USTEC.t",    # Nasdaq 100
  # "US500.t",     # S&P 500
   "US30.t",      # Dow Jones
   # "XAUUSD.t",    # Gold
    #"EURUSD.t",
    #"GBPUSD.t",
    #"USDJPY.t",
]

# --- Timeframe (scalp: M1, M5, M15) ---
TIMEFRAME = "M15"          # one of M1, M5, M15, M30, H1, H4, D1
LOOKBACK_DAYS = 730        # how many days of history to pull per symbol (730 = 2 years)

# --- Which protocols to run ---
PROTOCOLS = ["T1", "S1", "S2", "S3"]   # subset of these four

# --- Account / risk ---
INITIAL_CAPITAL = 100000.0
RISK_PER_TRADE_PCT = 0.2       # % of equity risked per trade
COMMISSION_PER_LOT = 0.0       # round-trip commission in account currency
SLIPPAGE_POINTS = 2            # points of slippage per fill

# --- Strategy parameters (from the indicator's input panel) ---
EMA_FAST        = 9
EMA_SLOW        = 21
EMA_SLOPE_LEN   = 20
BB_LENGTH       = 20
BB_STDDEV       = 2.0
ATR_PERIOD      = 14
ATR_MULT        = 3.5
ENTRY_PCT       = 3.5          # "% of Entry / Center Point"
MAX_RR          = 2.0          # TP multiplier — e.g. 2.0 = 2:1 RR, 3.0 = 3:1 RR, 1.5 = 1.5:1 RR
BREAKEVEN_PCT   = 0.0          # 0 = disabled ; e.g. 50 = move to BE at 50 % of TP
BREAKOUT_BUFFER_TICKS  = 20
MIN_4H_DISP_TICKS      = 30

# --- Backtest constraints ---
MAX_CONSECUTIVE_LOSSES  = 5    # pause trading after this many losses in a row (set to 999 to disable)

# --- Output ---
OUTPUT_HTML = "oneshot_backtest_report.html"

# ================================================================
# 2. MT5 DATA LOADER
# ================================================================

def _mt5_timeframe(tf: str):
    import MetaTrader5 as mt5
    return {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }[tf]


def load_mt5_data(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    """Pull OHLCV bars straight from the running MT5 terminal."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol} in MT5 Market Watch")

    utc_to   = datetime.utcnow()
    utc_from = utc_to - timedelta(days=lookback_days)
    rates = mt5.copy_rates_range(symbol, _mt5_timeframe(timeframe), utc_from, utc_to)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No bars returned for {symbol} {timeframe}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].set_index("time")

    # also fetch point size / contract size so risk math is correct
    info = mt5.symbol_info(symbol)
    df.attrs["point"]         = info.point          if info else 0.0001
    df.attrs["digits"]        = info.digits         if info else 5
    df.attrs["trade_contract"] = info.trade_contract_size if info else 100_000
    df.attrs["tick_value"]    = info.trade_tick_value if info else 1.0
    df.attrs["tick_size"]     = info.trade_tick_size  if info else info.point if info else 0.0001

    return df


# ================================================================
# 3. INDICATORS
# ================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def bollinger(series: pd.Series, length: int, stddev: float):
    mid = series.rolling(length).mean()
    sd  = series.rolling(length).std(ddof=0)
    return mid, mid + stddev * sd, mid - stddev * sd


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


# ================================================================
# 4. PROTOCOL SIGNAL GENERATORS
# ================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_slow"]  = ema(df["close"], EMA_SLOW)
    df["ema_slope"] = ema(df["close"], EMA_SLOPE_LEN)
    df["slope"]     = df["ema_slope"] - df["ema_slope"].shift(3)
    df["atr"]       = atr(df, ATR_PERIOD)
    mid, up, lo = bollinger(df["close"], BB_LENGTH, BB_STDDEV)
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, up, lo

    # 4H displacement context for T1 swing
    d4 = resample_4h(df[["open", "high", "low", "close", "volume"]])
    d4["body"] = (d4["close"] - d4["open"]).abs()
    df["h4_body"]  = d4["body"].reindex(df.index, method="ffill")
    df["h4_close"] = d4["close"].reindex(df.index, method="ffill")
    df["h4_open"]  = d4["open"].reindex(df.index, method="ffill")
    return df


def signal_s1_cross(df: pd.DataFrame) -> pd.Series:
    """+1 long cross, -1 short cross, 0 no signal. Evaluated at bar close."""
    cross_up   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig


def signal_s2_slope(df: pd.DataFrame) -> pd.Series:
    """Trade the turn of the EMA slope. +1 when slope flips positive, -1 negative."""
    flip_up   = (df["slope"] > 0) & (df["slope"].shift(1) <= 0)
    flip_down = (df["slope"] < 0) & (df["slope"].shift(1) >= 0)
    sig = pd.Series(0, index=df.index)
    sig[flip_up]   = 1
    sig[flip_down] = -1
    return sig


def signal_s3_bbands(df: pd.DataFrame) -> pd.Series:
    """Mean-reversion: close pierces outside band, next bar re-enters."""
    long_setup  = (df["close"].shift(1) < df["bb_lo"].shift(1)) & (df["close"] > df["bb_lo"])
    short_setup = (df["close"].shift(1) > df["bb_up"].shift(1)) & (df["close"] < df["bb_up"])
    sig = pd.Series(0, index=df.index)
    sig[long_setup]  = 1
    sig[short_setup] = -1
    return sig


def signal_t1_swing(df: pd.DataFrame, point: float) -> pd.Series:
    """4H displacement breakout with tick buffer."""
    disp_ticks = (df["h4_body"] / point)
    big_4h     = disp_ticks >= MIN_4H_DISP_TICKS
    bullish_4h = df["h4_close"] > df["h4_open"]

    buf = BREAKOUT_BUFFER_TICKS * point
    prior_high = df["high"].rolling(20).max().shift(1)
    prior_low  = df["low"].rolling(20).min().shift(1)

    long_bo  = big_4h & bullish_4h  & (df["high"] > (prior_high + buf))
    short_bo = big_4h & ~bullish_4h & (df["low"]  < (prior_low  - buf))

    sig = pd.Series(0, index=df.index)
    sig[long_bo]  = 1
    sig[short_bo] = -1
    return sig


# ================================================================
# 5. UNIVERSAL RISK ENGINE + BACKTEST LOOP
# ================================================================

@dataclass
class Trade:
    symbol: str
    protocol: str
    side: str            # "long" / "short"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    tp_price: float
    size_lots: float
    risk_amount: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    r_multiple: float = 0.0
    bars_held: int = 0


def _position_size(equity: float, risk_pct: float, stop_dist_price: float,
                   tick_size: float, tick_value: float) -> float:
    """Risk-based lot sizing using MT5 tick_value / tick_size."""
    risk_cash   = equity * (risk_pct / 100.0)
    ticks       = stop_dist_price / tick_size if tick_size > 0 else 0
    per_lot     = ticks * tick_value
    if per_lot <= 0:
        return 0.0
    lots = risk_cash / per_lot
    return max(0.01, round(lots, 2))


def backtest_symbol_protocol(df: pd.DataFrame, symbol: str, protocol: str,
                             equity_start: float) -> Tuple[List[Trade], pd.Series]:
    """Run one protocol on one symbol. Returns (trades, equity_curve)."""
    point      = df.attrs.get("point", 0.0001)
    tick_size  = df.attrs.get("tick_size", point)
    tick_value = df.attrs.get("tick_value", 1.0)

    df = add_indicators(df)

    if protocol == "S1":
        sig = signal_s1_cross(df)
    elif protocol == "S2":
        sig = signal_s2_slope(df)
    elif protocol == "S3":
        sig = signal_s3_bbands(df)
    elif protocol == "T1":
        sig = signal_t1_swing(df, point)
    else:
        raise ValueError(f"Unknown protocol {protocol}")

    trades: List[Trade] = []
    equity = equity_start
    equity_curve = pd.Series(index=df.index, dtype=float)

    open_trade: Optional[Trade] = None
    entry_bar: Optional[int] = None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["atr"].values
    times  = df.index
    sigs   = sig.values

    consec_losses = 0

    for i in range(1, len(df)):
        # ---------- manage open trade ----------
        if open_trade is not None:
            high_i, low_i = highs[i], lows[i]
            exit_price = None
            reason = ""

            if open_trade.side == "long":
                if low_i <= open_trade.stop_price:
                    exit_price, reason = open_trade.stop_price, "SL"
                elif high_i >= open_trade.tp_price:
                    exit_price, reason = open_trade.tp_price, "TP"
            else:
                if high_i >= open_trade.stop_price:
                    exit_price, reason = open_trade.stop_price, "SL"
                elif low_i <= open_trade.tp_price:
                    exit_price, reason = open_trade.tp_price, "TP"

            # Breakeven trail
            if BREAKEVEN_PCT > 0 and exit_price is None:
                tp_dist = abs(open_trade.tp_price - open_trade.entry_price)
                trig    = open_trade.entry_price + np.sign(open_trade.tp_price - open_trade.entry_price) \
                           * tp_dist * (BREAKEVEN_PCT / 100.0)
                if open_trade.side == "long" and high_i >= trig:
                    open_trade.stop_price = max(open_trade.stop_price, open_trade.entry_price)
                elif open_trade.side == "short" and low_i <= trig:
                    open_trade.stop_price = min(open_trade.stop_price, open_trade.entry_price)

            if exit_price is not None:
                ticks = (exit_price - open_trade.entry_price) / tick_size
                if open_trade.side == "short":
                    ticks = -ticks
                pnl = ticks * tick_value * open_trade.size_lots - COMMISSION_PER_LOT * open_trade.size_lots
                open_trade.exit_time   = times[i]
                open_trade.exit_price  = exit_price
                open_trade.exit_reason = reason
                open_trade.pnl         = pnl
                open_trade.r_multiple  = pnl / open_trade.risk_amount if open_trade.risk_amount else 0
                open_trade.bars_held   = i - entry_bar
                equity += pnl
                trades.append(open_trade)
                consec_losses = consec_losses + 1 if pnl < 0 else 0
                open_trade = None
                entry_bar  = None

        # ---------- open new trade ----------
        if open_trade is None and consec_losses < MAX_CONSECUTIVE_LOSSES:
            s = sigs[i]
            if s != 0 and not math.isnan(atrs[i]) and atrs[i] > 0:
                side        = "long" if s == 1 else "short"
                entry_price = closes[i] + np.sign(s) * SLIPPAGE_POINTS * point
                stop_dist   = atrs[i] * ATR_MULT
                tp_dist     = stop_dist * MAX_RR

                if side == "long":
                    stop_price = entry_price - stop_dist
                    tp_price   = entry_price + tp_dist
                else:
                    stop_price = entry_price + stop_dist
                    tp_price   = entry_price - tp_dist

                lots = _position_size(equity, RISK_PER_TRADE_PCT, stop_dist, tick_size, tick_value)
                if lots > 0:
                    risk_amount = stop_dist / tick_size * tick_value * lots
                    open_trade = Trade(
                        symbol=symbol, protocol=protocol, side=side,
                        entry_time=times[i], entry_price=entry_price,
                        stop_price=stop_price, tp_price=tp_price,
                        size_lots=lots, risk_amount=risk_amount,
                    )
                    entry_bar = i

        equity_curve.iloc[i] = equity

    equity_curve = equity_curve.ffill().fillna(equity_start)
    return trades, equity_curve


# ================================================================
# 6. STATS
# ================================================================

def compute_stats(trades: List[Trade], equity_curve: pd.Series,
                  initial_capital: float) -> Dict[str, float]:
    if not trades:
        return {"trades": 0}

    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]

    gross_profit = wins.sum()
    gross_loss   = -losses.sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate   = 100 * len(wins) / len(pnls)
    expectancy = pnls.mean()

    rets = equity_curve.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(252 * 24 * 12) if rets.std() > 0 else 0

    running_max = equity_curve.cummax()
    dd          = (equity_curve - running_max) / running_max
    max_dd_pct  = dd.min() * 100

    # max consecutive losses
    max_cl = cl = 0
    for t in trades:
        if t.pnl < 0:
            cl += 1; max_cl = max(max_cl, cl)
        else:
            cl = 0

    return {
        "trades":           len(trades),
        "wins":             int(len(wins)),
        "losses":           int(len(losses)),
        "win_rate_pct":     round(win_rate, 2),
        "profit_factor":    round(pf, 2) if pf != float("inf") else "inf",
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "net_pnl":          round(pnls.sum(), 2),
        "expectancy":       round(expectancy, 2),
        "sharpe_ann":       round(sharpe, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_consec_losses": int(max_cl),
        "avg_bars_held":    round(np.mean([t.bars_held for t in trades]), 1),
        "final_equity":     round(equity_curve.iloc[-1], 2),
        "return_pct":       round(100 * (equity_curve.iloc[-1] / initial_capital - 1), 2),
    }


# ================================================================
# 7. HTML REPORT  (Plotly, single self-contained file)
# ================================================================

def build_html_report(
    all_trades: Dict[str, List[Trade]],
    all_equity: Dict[str, pd.Series],
    all_stats:  Dict[str, Dict],
    price_data: Dict[str, pd.DataFrame],
    out_path:   str,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    sections_html: List[str] = []

    # ----- top summary table -----
    stats_rows = ""
    for key, st in all_stats.items():
        stats_rows += "<tr>" + f"<td>{key}</td>" + "".join(
            f"<td>{st.get(c, '')}</td>" for c in STAT_COLS
        ) + "</tr>"

    sections_html.append(f"""
    <h2>Performance Summary</h2>
    <table class='tbl'>
      <thead><tr><th>Symbol / Protocol</th>{''.join(f'<th>{c}</th>' for c in STAT_COLS)}</tr></thead>
      <tbody>{stats_rows}</tbody>
    </table>
    """)

    # ----- combined equity curve -----
    eq_fig = go.Figure()
    combined = None
    for key, eq in all_equity.items():
        eq_fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=key))
        combined = eq if combined is None else combined.add(eq - INITIAL_CAPITAL, fill_value=0)
    eq_fig.update_layout(
        title="Equity Curves (per Symbol / Protocol)",
        height=450, template="plotly_white",
        xaxis_title="Time", yaxis_title="Equity",
    )

    dd_fig = go.Figure()
    for key, eq in all_equity.items():
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=key, fill="tozeroy"))
    dd_fig.update_layout(
        title="Drawdown % (per Symbol / Protocol)",
        height=350, template="plotly_white",
        xaxis_title="Time", yaxis_title="Drawdown %",
    )

    sections_html.append("<h2>Equity Curve</h2>" + pio.to_html(eq_fig, include_plotlyjs="cdn", full_html=False))
    sections_html.append("<h2>Drawdown</h2>"    + pio.to_html(dd_fig,  include_plotlyjs=False, full_html=False))

    # ----- price charts with markers -----
    sections_html.append("<h2>Price Charts with Trade Markers</h2>")
    for sym, df in price_data.items():
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=sym, showlegend=False,
        ))
        # Overlay EMAs for context
        e_fast = ema(df["close"], EMA_FAST)
        e_slow = ema(df["close"], EMA_SLOW)
        fig.add_trace(go.Scatter(x=df.index, y=e_fast, mode="lines", name=f"EMA{EMA_FAST}",
                                 line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=e_slow, mode="lines", name=f"EMA{EMA_SLOW}",
                                 line=dict(width=1)))

        # markers from every protocol for this symbol
        for key, trades in all_trades.items():
            if not key.startswith(sym + " / "):
                continue
            for t in trades:
                color = "#22c55e" if t.pnl > 0 else "#ef4444"
                fig.add_trace(go.Scatter(
                    x=[t.entry_time, t.exit_time],
                    y=[t.entry_price, t.exit_price],
                    mode="lines+markers",
                    line=dict(color=color, width=1, dash="dot"),
                    marker=dict(size=7, symbol=("triangle-up" if t.side == "long" else "triangle-down")),
                    name=f"{key} {t.side}", showlegend=False,
                    hovertext=f"{key}<br>{t.side} {t.exit_reason}<br>PnL {t.pnl:.2f}  R {t.r_multiple:.2f}",
                ))
        fig.update_layout(
            title=f"{sym} — trades",
            height=500, xaxis_rangeslider_visible=False, template="plotly_white",
        )
        sections_html.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

    # ----- full trade table -----
    rows = []
    for key, trades in all_trades.items():
        for t in trades:
            rows.append({
                "symbol":   t.symbol,
                "protocol": t.protocol,
                "side":     t.side,
                "entry_time":  t.entry_time,
                "entry_price": round(t.entry_price, 5),
                "exit_time":   t.exit_time,
                "exit_price":  round(t.exit_price or 0, 5),
                "exit_reason": t.exit_reason,
                "lots":        t.size_lots,
                "pnl":         round(t.pnl, 2),
                "R":           round(t.r_multiple, 2),
                "bars":        t.bars_held,
            })
    trade_df = pd.DataFrame(rows)
    trade_table_html = trade_df.to_html(classes="tbl trades", index=False) if len(trade_df) else "<p>No trades.</p>"
    sections_html.append("<h2>Trade List</h2>" + trade_table_html)

    # ----- wrap -----
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>One-Shot Algo v2 Backtest Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width:1400px; margin:24px auto; padding:0 16px; color:#1f2937; }}
  h1 {{ border-bottom:2px solid #1f2937; padding-bottom:8px; }}
  h2 {{ margin-top:36px; }}
  .tbl {{ border-collapse:collapse; width:100%; font-size:13px; }}
  .tbl th, .tbl td {{ border:1px solid #e5e7eb; padding:6px 8px; text-align:right; }}
  .tbl th {{ background:#f3f4f6; text-align:center; }}
  .tbl td:first-child, .tbl th:first-child {{ text-align:left; }}
  .trades tbody tr:nth-child(odd) {{ background:#fafafa; }}
  .meta {{ color:#6b7280; font-size:13px; }}
</style></head><body>
<h1>One-Shot Algo v2 — Backtest Report</h1>
<p class="meta">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp;
   TF {TIMEFRAME} &nbsp;|&nbsp; Lookback {LOOKBACK_DAYS}d &nbsp;|&nbsp;
   Capital {INITIAL_CAPITAL:,.0f} &nbsp;|&nbsp; Risk {RISK_PER_TRADE_PCT}% &nbsp;|&nbsp;
   Protocols {', '.join(PROTOCOLS)}</p>
{''.join(sections_html)}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


STAT_COLS = ["trades", "win_rate_pct", "profit_factor", "net_pnl", "return_pct",
             "max_drawdown_pct", "sharpe_ann", "expectancy", "max_consec_losses", "final_equity"]


# ================================================================
# 8. MAIN
# ================================================================

def main():
    print(f"\n>>> One-Shot Algo v2 Backtester — TF {TIMEFRAME}, {LOOKBACK_DAYS}d lookback")
    print(f"    Symbols:   {SYMBOLS}")
    print(f"    Protocols: {PROTOCOLS}\n")

    all_trades: Dict[str, List[Trade]]    = {}
    all_equity: Dict[str, pd.Series]      = {}
    all_stats:  Dict[str, Dict]           = {}
    price_data: Dict[str, pd.DataFrame]   = {}

    for sym in SYMBOLS:
        try:
            df = load_mt5_data(sym, TIMEFRAME, LOOKBACK_DAYS)
            print(f"[{sym}] loaded {len(df)} bars")
        except Exception as e:
            print(f"[{sym}] SKIP — {e}")
            continue
        price_data[sym] = df

        for proto in PROTOCOLS:
            trades, eq = backtest_symbol_protocol(df, sym, proto, INITIAL_CAPITAL)
            key = f"{sym} / {proto}"
            all_trades[key] = trades
            all_equity[key] = eq
            all_stats[key]  = compute_stats(trades, eq, INITIAL_CAPITAL)
            s = all_stats[key]
            print(f"  {proto}: {s.get('trades',0)} trades | "
                  f"WR {s.get('win_rate_pct','-')}% | PF {s.get('profit_factor','-')} | "
                  f"NetPnL {s.get('net_pnl','-')} | MaxDD {s.get('max_drawdown_pct','-')}%")

    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass

    if not all_trades:
        print("No trades generated — nothing to report.")
        return

    build_html_report(all_trades, all_equity, all_stats, price_data, OUTPUT_HTML)
    print(f"\n>>> Report written to {os.path.abspath(OUTPUT_HTML)}")


if __name__ == "__main__":
    main()