"""
===============================================================
  UT Bot Alerts – XAUUSD 15-Minute Backtest via MetaTrader 5
  EXIT MODE: Opposite Signal Only
===============================================================

Strategy Logic (mirrors Pine Script v4 UT Bot):
  • Buy  signal : Close crosses above ATR Trailing Stop (EMA crossover)
  • Sell signal : Close crosses below ATR Trailing Stop (EMA crossover)

Trade Management:
  • Stop Loss : 1.5 × ATR at entry
  • Take Profit: NO fixed TP — trade stays open until:
        (a) Opposite UT Bot signal fires  → close at market
        (b) SL is hit                     → close at SL price

Position Sizing:
  • Fixed $200 risk per trade (regardless of account size)
  • Lot size = $200 / (SL distance × contract size)

Outputs:
  • Console summary statistics
  • equity_curve.png       – equity + drawdown chart
  • trade_log.csv          – every entry/exit with PnL
  • backtest_summary.txt   – key stats saved to disk

Requirements:
  pip install MetaTrader5 pandas numpy matplotlib
  MetaTrader 5 terminal must be running and logged in.
===============================================================
"""

import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION  –  edit these to your liking
# ──────────────────────────────────────────────────────────────
SYMBOL          = "XAUUSD.t"
TIMEFRAME       = mt5.TIMEFRAME_M5
START_DATE      = datetime(2026, 1, 1)
END_DATE        = datetime(2026, 4, 22)

# UT Bot Parameters (match your Pine Script)
KEY_VALUE       = 3       # Sensitivity multiplier (a)
ATR_PERIOD      = 14      # ATR period (c)

# Trade Management
SL_ATR_MULT     = 1.5     # Stop Loss = 1.5 × ATR at entry
                           # NO fixed TP — exit only on opposite signal or SL

# Risk / Position Sizing
INITIAL_EQUITY  = 100_000.0   # Starting account balance (USD)
RISK_AMOUNT     = 200.0       # Fixed dollar risk per trade (USD)
CONTRACT_SIZE   = 100         # XAUUSD: 100 oz per standard lot
MIN_LOT         = 0.01
LOT_STEP        = 0.01

# Output folder (same directory as this script)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────
#  MT5 CONNECTION & DATA
# ──────────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialisation failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info:
        print(f"[MT5]  Connected  |  Account #{info.login}  |  "
              f"Server: {info.server}  |  Balance: ${info.balance:,.2f}")
    else:
        print("[MT5]  Connected (demo / no account info)")
    return True


def fetch_ohlcv(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"[ERROR] No data returned: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    }, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"[DATA] {len(df):,} bars  |  {df.index[0]}  ->  {df.index[-1]}")
    return df


# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────

def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ATR (RMA smoothing) — matches Pine Script atr()."""
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def ut_bot_signals(df: pd.DataFrame, key_value: int = 1, atr_period: int = 10) -> pd.DataFrame:
    """
    Reproduce the UT Bot Alerts Pine Script v4 logic.
    Adds columns: ATR, nLoss, TrailingStop, Pos, Buy, Sell
    """
    df    = df.copy()
    src   = df["Close"]
    atr   = wilder_atr(df, atr_period)
    n_loss = key_value * atr

    # ── ATR Trailing Stop (iterative) ──
    ts         = np.zeros(len(df))
    close_arr  = src.values
    nloss_arr  = n_loss.values

    for i in range(1, len(df)):
        prev_ts  = ts[i - 1]
        curr_src = close_arr[i]
        prev_src = close_arr[i - 1]
        curr_nl  = nloss_arr[i]

        if curr_src > prev_ts and prev_src > prev_ts:
            ts[i] = max(prev_ts, curr_src - curr_nl)
        elif curr_src < prev_ts and prev_src < prev_ts:
            ts[i] = min(prev_ts, curr_src + curr_nl)
        elif curr_src > prev_ts:
            ts[i] = curr_src - curr_nl
        else:
            ts[i] = curr_src + curr_nl

    trailing_stop = pd.Series(ts, index=df.index)

    # ── Position ──
    pos_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        prev_src = close_arr[i - 1]
        curr_src = close_arr[i]
        prev_ts  = ts[i - 1]

        if prev_src < prev_ts and curr_src > prev_ts:
            pos_arr[i] = 1
        elif prev_src > prev_ts and curr_src < prev_ts:
            pos_arr[i] = -1
        else:
            pos_arr[i] = pos_arr[i - 1]

    # ── EMA(1) = close; crossovers ──
    ema   = src.copy()
    ts_s  = trailing_stop
    above = (ema > ts_s) & (ema.shift(1) <= ts_s.shift(1))
    below = (ts_s > ema) & (ts_s.shift(1) <= ema.shift(1))

    df["ATR"]          = atr
    df["nLoss"]        = n_loss
    df["TrailingStop"] = trailing_stop
    df["Pos"]          = pd.Series(pos_arr, index=df.index)
    df["Buy"]          = (src > trailing_stop) & above
    df["Sell"]         = (src < trailing_stop) & below

    return df


# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def round_lot(raw: float) -> float:
    stepped = np.floor(raw / LOT_STEP) * LOT_STEP
    return round(max(MIN_LOT, stepped), 2)


def calc_lots(sl_distance: float) -> float:
    """lots = RISK_AMOUNT / (SL_distance x contract_size)"""
    if sl_distance <= 0:
        return MIN_LOT
    return round_lot(RISK_AMOUNT / (sl_distance * CONTRACT_SIZE))


# ──────────────────────────────────────────────────────────────
#  BACKTEST ENGINE  –  Signal-exit only
# ──────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Simplified backtest — no partial closes, no trailing stop.

    Entry  : Buy / Sell signal
    Exit   : (a) SL hit  OR  (b) opposite signal fires
    """
    equity       = INITIAL_EQUITY
    trades       = []
    equity_curve = [equity]
    dates_curve  = [df.index[0]]

    in_trade    = False
    direction   = None      # "long" | "short"
    entry_price = 0.0
    entry_time  = None
    sl_price    = 0.0
    lots        = 0.0
    trade_id    = 0

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df["ATR"].values
    buy_arr   = df["Buy"].values
    sell_arr  = df["Sell"].values
    times     = df.index

    for i in range(2, len(df)):
        bar_time = times[i]
        close    = close_arr[i]
        high     = high_arr[i]
        low      = low_arr[i]
        atr_now  = atr_arr[i]
        is_buy   = bool(buy_arr[i])
        is_sell  = bool(sell_arr[i])

        # ── Manage open trade ──
        if in_trade:

            if direction == "long":
                # (a) SL hit
                if low <= sl_price:
                    pnl     = (sl_price - entry_price) * lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_row(trade_id, entry_time, bar_time, "Long",
                                       entry_price, sl_price, lots, sl_price,
                                       "Stop Loss", pnl, equity))
                    in_trade = False

                # (b) Opposite signal
                elif is_sell:
                    pnl     = (close - entry_price) * lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_row(trade_id, entry_time, bar_time, "Long",
                                       entry_price, close, lots, sl_price,
                                       "Opposite Signal", pnl, equity))
                    in_trade = False

            elif direction == "short":
                # (a) SL hit
                if high >= sl_price:
                    pnl     = (entry_price - sl_price) * lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_row(trade_id, entry_time, bar_time, "Short",
                                       entry_price, sl_price, lots, sl_price,
                                       "Stop Loss", pnl, equity))
                    in_trade = False

                # (b) Opposite signal
                elif is_buy:
                    pnl     = (entry_price - close) * lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_row(trade_id, entry_time, bar_time, "Short",
                                       entry_price, close, lots, sl_price,
                                       "Opposite Signal", pnl, equity))
                    in_trade = False

        # ── Enter new trade (only when flat) ──
        if not in_trade:
            if is_buy:
                trade_id   += 1
                sl_dist     = SL_ATR_MULT * atr_now
                entry_price = close
                direction   = "long"
                sl_price    = entry_price - sl_dist
                lots        = calc_lots(sl_dist)
                entry_time  = bar_time
                in_trade    = True

            elif is_sell:
                trade_id   += 1
                sl_dist     = SL_ATR_MULT * atr_now
                entry_price = close
                direction   = "short"
                sl_price    = entry_price + sl_dist
                lots        = calc_lots(sl_dist)
                entry_time  = bar_time
                in_trade    = True

        equity_curve.append(equity)
        dates_curve.append(bar_time)

    return trades, equity_curve, dates_curve


def _row(trade_id, entry_time, exit_time, direction,
         entry_price, exit_price, lots, sl, reason, pnl, equity):
    return {
        "Trade ID":    trade_id,
        "Entry Time":  entry_time,
        "Exit Time":   exit_time,
        "Direction":   direction,
        "Entry Price": round(entry_price, 2),
        "Exit Price":  round(exit_price,  2),
        "Lots":        lots,
        "SL":          round(sl, 2),
        "Exit Reason": reason,
        "PnL ($)":     round(pnl, 2),
        "Equity ($)":  round(equity, 2),
    }


# ──────────────────────────────────────────────────────────────
#  STATISTICS
# ──────────────────────────────────────────────────────────────

def compute_stats(trades: list, equity_curve: list) -> dict:
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    pnl  = df_t["PnL ($)"]

    total_trades  = len(df_t)
    winners       = pnl[pnl > 0]
    losers        = pnl[pnl <= 0]
    win_rate      = len(winners) / total_trades * 100 if total_trades else 0
    profit_factor = (abs(winners.sum() / losers.sum())
                     if losers.sum() != 0 else float("inf"))

    eq      = np.array(equity_curve)
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / peak * 100
    max_dd  = dd.min()

    final_eq     = equity_curve[-1]
    total_return = (final_eq - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    eq_s    = pd.Series(equity_curve)
    rets    = eq_s.pct_change().dropna()
    bars_py = 252 * 24 * 4
    sharpe  = ((rets.mean() / rets.std()) * np.sqrt(bars_py)
               if rets.std() != 0 else 0.0)

    streak      = pnl.apply(lambda x: 1 if x > 0 else -1)
    max_win_s   = _max_streak(streak,  1)
    max_loss_s  = _max_streak(streak, -1)

    return {
        "Period":               f"{START_DATE.date()}  ->  {END_DATE.date()}",
        "Symbol / TF":          f"{SYMBOL}  /  15-Minute",
        "Exit Mode":            "Opposite Signal (no fixed TP)",
        "Initial Equity ($)":   INITIAL_EQUITY,
        "Final Equity ($)":     round(final_eq, 2),
        "Total Return (%)":     round(total_return, 2),
        "Total PnL ($)":        round(pnl.sum(), 2),
        "Total Trades":         total_trades,
        "Win Rate (%)":         round(win_rate, 2),
        "Avg Win ($)":          round(winners.mean(), 2) if len(winners) else 0,
        "Avg Loss ($)":         round(losers.mean(),  2) if len(losers)  else 0,
        "Best Trade ($)":       round(pnl.max(), 2),
        "Worst Trade ($)":      round(pnl.min(), 2),
        "Profit Factor":        round(profit_factor, 2),
        "Max Drawdown (%)":     round(max_dd, 2),
        "Sharpe Ratio":         round(sharpe, 2),
        "Max Win Streak":       max_win_s,
        "Max Loss Streak":      max_loss_s,
        "Risk per Trade ($)":   RISK_AMOUNT,
        "SL Multiplier (ATR)":  SL_ATR_MULT,
    }


def _max_streak(series: pd.Series, val: int) -> int:
    max_s = cur = 0
    for v in series:
        cur   = cur + 1 if v == val else 0
        max_s = max(max_s, cur)
    return max_s


# ──────────────────────────────────────────────────────────────
#  CHARTS
# ──────────────────────────────────────────────────────────────

def plot_equity_curve(equity_curve: list, dates_curve: list, stats: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    fig.suptitle(
        f"XAUUSD UT Bot  |  15-Minute  |  Signal-Exit Only  |  "
        f"{START_DATE.strftime('%b %Y')} - {END_DATE.strftime('%b %Y')}",
        fontsize=13, fontweight="bold"
    )

    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100

    # Equity curve
    ax1.plot(dates_curve, equity_curve, color="#FF9800", linewidth=1.4, label="Equity")
    ax1.fill_between(dates_curve, equity_curve, INITIAL_EQUITY,
                     where=eq >= INITIAL_EQUITY,
                     alpha=0.15, color="#FF9800", interpolate=True)
    ax1.fill_between(dates_curve, equity_curve, INITIAL_EQUITY,
                     where=eq < INITIAL_EQUITY,
                     alpha=0.15, color="#F44336", interpolate=True)
    ax1.axhline(INITIAL_EQUITY, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    ret  = stats.get("Total Return (%)", 0)
    wr   = stats.get("Win Rate (%)",     0)
    pf   = stats.get("Profit Factor",    0)
    mdd  = stats.get("Max Drawdown (%)", 0)
    sh   = stats.get("Sharpe Ratio",     0)
    box  = (f"Return: {ret:+.1f}%  |  Win Rate: {wr:.1f}%  |  "
            f"Profit Factor: {pf:.2f}  |  Max DD: {mdd:.1f}%  |  Sharpe: {sh:.2f}")
    ax1.text(0.01, 0.97, box, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

    # Drawdown
    ax2.fill_between(dates_curve, dd, 0, color="#F44336", alpha=0.4)
    ax2.plot(dates_curve, dd, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

    plt.tight_layout()
    path = os.path.join(out_dir, "equity_curve_signal_exit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CHART] Equity curve saved -> {path}")


# ──────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────

def save_trade_log(trades: list, out_dir: str):
    if not trades:
        return
    path = os.path.join(out_dir, "trade_log_signal_exit.csv")
    pd.DataFrame(trades).to_csv(path, index=False)
    print(f"[CSV]   Trade log saved   -> {path}")


def save_summary(stats: dict, out_dir: str):
    path = os.path.join(out_dir, "backtest_summary_signal_exit.txt")
    sep  = "-" * 42
    with open(path, "w", encoding="utf-8") as f:
        f.write("XAUUSD UT Bot Backtest (Signal Exit) - Summary\n")
        f.write(sep + "\n")
        for k, v in stats.items():
            f.write(f"{k:<28} {v}\n")
        f.write(sep + "\n")
        f.write("\nStrategy Parameters\n")
        f.write(f"  Key Value (sensitivity)  : {KEY_VALUE}\n")
        f.write(f"  ATR Period               : {ATR_PERIOD}\n")
        f.write(f"  SL multiplier            : {SL_ATR_MULT}x ATR\n")
        f.write(f"  Take Profit              : Opposite signal only (no fixed TP)\n")
        f.write(f"  Risk per trade           : ${RISK_AMOUNT:.0f} fixed\n")
        f.write(f"  Initial equity           : ${INITIAL_EQUITY:,.0f}\n")
    print(f"[TXT]   Summary saved     -> {path}")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  XAUUSD  UT Bot  |  15-Minute  |  Signal-Exit Backtest")
    print("=" * 60)

    if not connect_mt5():
        sys.exit(1)

    print(f"\n[DATA] Fetching {SYMBOL} M15  {START_DATE.date()} -> {END_DATE.date()} ...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if df.empty:
        mt5.shutdown()
        sys.exit(1)

    print("\n[CALC] Computing UT Bot signals ...")
    df = ut_bot_signals(df, KEY_VALUE, ATR_PERIOD)
    print(f"       Buy signals  : {df['Buy'].sum():,}")
    print(f"       Sell signals : {df['Sell'].sum():,}")

    print("\n[BACK] Running backtest ...")
    trades, equity_curve, dates_curve = run_backtest(df)
    print(f"       Trades logged : {len(trades)}")

    stats = compute_stats(trades, equity_curve)
    print("\n" + "-" * 48)
    print("  BACKTEST RESULTS  (Signal Exit)")
    print("-" * 48)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print("-" * 48)

    print()
    save_trade_log(trades, OUTPUT_DIR)
    plot_equity_curve(equity_curve, dates_curve, stats, OUTPUT_DIR)
    save_summary(stats, OUTPUT_DIR)

    mt5.shutdown()
    print("\n[DONE] MT5 disconnected. Backtest complete.")


if __name__ == "__main__":
    main()