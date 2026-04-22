"""
===============================================================
  UT Bot Alerts – XAUUSD 15-Minute Backtest via MetaTrader 5
===============================================================

Strategy Logic (mirrors Pine Script v4 UT Bot):
  • Buy  signal : Close crosses above ATR Trailing Stop (with EMA crossover)
  • Sell signal : Close crosses below ATR Trailing Stop (with EMA crossover)

Trade Management:
  • Stop Loss      : 1.5 × ATR at entry
  • Partial Close  : Close 50% of position at 1:2 RR (reward = 2 × SL distance)
  • After Partial  : Move SL to break-even, then trail remaining lot with 1× ATR
  • Full Exit      : On opposite signal (or SL/trailing stop hit)

Position Sizing:
  • Fixed $200 risk per trade (regardless of account size)
  • Lot size = risk_amount / (SL distance × contract size)

Outputs:
  • Console summary statistics
  • equity_curve.png  – equity growth chart
  • trade_log.csv     – every entry/exit with PnL
  • backtest_summary.txt – key stats saved to disk

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
TIMEFRAME       = mt5.TIMEFRAME_M15
START_DATE      = datetime(2024, 4, 21)
END_DATE        = datetime(2026, 4, 21)

# UT Bot Parameters (match your Pine Script)
KEY_VALUE       = 1       # Sensitivity multiplier (a)
ATR_PERIOD      = 10      # ATR period (c)

# Trade Management
SL_ATR_MULT     = 1.5     # Stop Loss = 1.5 × ATR at entry
PARTIAL_RR      = 2.0     # Close 50% when profit reaches 2× SL distance (1:2 RR)
PARTIAL_PCT     = 0.50    # Fraction of position to close at first TP
TRAIL_ATR_MULT  = 1.0     # Trailing stop = 1× ATR after partial close

# Risk / Position Sizing
INITIAL_EQUITY  = 10_000.0   # Starting account balance (USD)
RISK_AMOUNT     = 200.0       # Fixed dollar risk per trade (USD)
CONTRACT_SIZE   = 100         # XAUUSD: 100 oz per standard lot
MIN_LOT         = 0.01
LOT_STEP        = 0.01

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────
#  MT5 CONNECTION & DATA
# ──────────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    """Initialise MT5 and display account info."""
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialisation failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info:
        print(f"[MT5]  Connected  │  Account #{info.login}  │  "
              f"Server: {info.server}  │  Balance: ${info.balance:,.2f}")
    else:
        print("[MT5]  Connected (demo / no account info)")
    return True


def fetch_ohlcv(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    """Download OHLCV bars from MT5 and return a clean DataFrame."""
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
    print(f"[DATA] {len(df):,} bars  │  {df.index[0]}  →  {df.index[-1]}")
    return df


# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────

def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's Average True Range (RMA smoothing) – matches Pine Script atr().
    """
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1
    ).max(axis=1)
    # RMA = EMA with alpha = 1/period (same as Wilder's smoothing)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def ut_bot_signals(df: pd.DataFrame, key_value: int = 1, atr_period: int = 10) -> pd.DataFrame:
    """
    Reproduce the UT Bot Alerts Pine Script logic.

    Returns df with extra columns:
        ATR, nLoss, TrailingStop, Pos, Buy, Sell
    """
    df = df.copy()
    src   = df["Close"]
    atr   = wilder_atr(df, atr_period)
    n_loss = key_value * atr

    # ── ATR Trailing Stop (iterative, mirrors Pine's history operator) ──
    ts = np.zeros(len(df))
    close_arr  = src.values
    nloss_arr  = n_loss.values

    for i in range(1, len(df)):
        prev_ts   = ts[i - 1]
        curr_src  = close_arr[i]
        prev_src  = close_arr[i - 1]
        curr_nl   = nloss_arr[i]

        if curr_src > prev_ts and prev_src > prev_ts:
            ts[i] = max(prev_ts, curr_src - curr_nl)
        elif curr_src < prev_ts and prev_src < prev_ts:
            ts[i] = min(prev_ts, curr_src + curr_nl)
        elif curr_src > prev_ts:
            ts[i] = curr_src - curr_nl
        else:
            ts[i] = curr_src + curr_nl

    trailing_stop = pd.Series(ts, index=df.index)

    # ── Position (1 = bullish, -1 = bearish) ──
    pos_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        prev_src = close_arr[i - 1]
        curr_src = close_arr[i]
        prev_ts  = ts[i - 1]
        prev_pos = pos_arr[i - 1]

        if prev_src < prev_ts and curr_src > prev_ts:
            pos_arr[i] = 1
        elif prev_src > prev_ts and curr_src < prev_ts:
            pos_arr[i] = -1
        else:
            pos_arr[i] = prev_pos

    pos = pd.Series(pos_arr, index=df.index)

    # ── EMA(1) = close itself; crossovers ──
    ema   = src.copy()                              # EMA period 1 = identity
    ts_s  = trailing_stop

    above = (ema > ts_s) & (ema.shift(1) <= ts_s.shift(1))   # ema crosses above TS
    below = (ts_s > ema) & (ts_s.shift(1) <= ema.shift(1))   # TS  crosses above ema

    df["ATR"]          = atr
    df["nLoss"]        = n_loss
    df["TrailingStop"] = trailing_stop
    df["Pos"]          = pos
    df["Buy"]          = (src > trailing_stop) & above   # long signal
    df["Sell"]         = (src < trailing_stop) & below   # short signal

    return df


# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def round_lot(raw_lots: float) -> float:
    """Round down to nearest LOT_STEP, clamped to MIN_LOT."""
    stepped = np.floor(raw_lots / LOT_STEP) * LOT_STEP
    return round(max(MIN_LOT, stepped), 2)


def calc_lots(equity: float, sl_distance: float) -> float:
    """
    Position size based on fixed dollar risk:
        lots = RISK_AMOUNT / (SL_distance × contract_size)
    equity is kept as a parameter for signature consistency.
    """
    if sl_distance <= 0:
        return MIN_LOT
    raw = RISK_AMOUNT / (sl_distance * CONTRACT_SIZE)
    return round_lot(raw)


# ──────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Event-driven bar-by-bar backtest.

    Trade lifecycle:
        1. Entry  on Buy/Sell signal  →  record entry, compute SL & first-TP
        2. During trade:
             a. SL hit?           → close remaining lots, log trade
             b. First TP hit?     → close 50%, move SL to BE, start ATR trail
             c. Opposite signal?  → close remaining lots, log trade
             d. Trailing update   → adjust sl_price upward (long) / downward (short)
        3. New signal only entered when flat (no pyramiding)

    Returns
        trades       : list of dicts (one row per partial or final exit)
        equity_curve : list of floats (length = len(df) − 1)
        dates_curve  : list of timestamps matching equity_curve
    """
    equity       = INITIAL_EQUITY
    trades       = []
    equity_curve = [equity]
    dates_curve  = [df.index[0]]

    # ── Trade state ──
    in_trade       = False
    direction      = None    # "long" | "short"
    entry_price    = 0.0
    entry_time     = None
    sl_price       = 0.0
    first_tp       = 0.0
    full_lots      = 0.0
    remaining_lots = 0.0
    partial_done   = False
    trail_sl       = 0.0
    trade_id       = 0

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df["ATR"].values
    buy_arr   = df["Buy"].values
    sell_arr  = df["Sell"].values
    times     = df.index

    for i in range(2, len(df)):
        bar_time  = times[i]
        close     = close_arr[i]
        high      = high_arr[i]
        low       = low_arr[i]
        atr_now   = atr_arr[i]
        is_buy    = bool(buy_arr[i])
        is_sell   = bool(sell_arr[i])

        # ────────────────────────────────────────────────────
        #  MANAGE OPEN TRADE
        # ────────────────────────────────────────────────────
        if in_trade:

            if direction == "long":

                # (a) Stop-loss hit
                if low <= sl_price:
                    exit_p = sl_price
                    pnl    = (exit_p - entry_price) * remaining_lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_make_row(
                        trade_id, entry_time, bar_time, "Long",
                        entry_price, exit_p, remaining_lots, sl_price,
                        "Stop Loss", pnl, equity
                    ))
                    in_trade = False

                # (b) First TP hit (50% partial close)
                elif not partial_done and high >= first_tp:
                    partial_lots   = round_lot(full_lots * PARTIAL_PCT)
                    remaining_lots = round_lot(full_lots - partial_lots)
                    pnl            = (first_tp - entry_price) * partial_lots * CONTRACT_SIZE
                    equity        += pnl
                    # Move SL to break-even, initialise trail
                    sl_price    = entry_price
                    trail_sl    = entry_price
                    partial_done = True
                    trades.append(_make_row(
                        trade_id, entry_time, bar_time, "Long (Partial 50%)",
                        entry_price, first_tp, partial_lots, sl_price,
                        "Partial TP 1:2", pnl, equity
                    ))

                else:
                    # (c) ATR trail after partial
                    if partial_done:
                        new_trail = close - atr_now * TRAIL_ATR_MULT
                        trail_sl  = max(trail_sl, new_trail)
                        sl_price  = trail_sl

                    # (d) Opposite signal → close all
                    if is_sell:
                        pnl    = (close - entry_price) * remaining_lots * CONTRACT_SIZE
                        equity += pnl
                        trades.append(_make_row(
                            trade_id, entry_time, bar_time, "Long",
                            entry_price, close, remaining_lots, sl_price,
                            "Opposite Signal", pnl, equity
                        ))
                        in_trade = False

            elif direction == "short":

                # (a) Stop-loss hit
                if high >= sl_price:
                    exit_p = sl_price
                    pnl    = (entry_price - exit_p) * remaining_lots * CONTRACT_SIZE
                    equity += pnl
                    trades.append(_make_row(
                        trade_id, entry_time, bar_time, "Short",
                        entry_price, exit_p, remaining_lots, sl_price,
                        "Stop Loss", pnl, equity
                    ))
                    in_trade = False

                # (b) First TP hit (50% partial close)
                elif not partial_done and low <= first_tp:
                    partial_lots   = round_lot(full_lots * PARTIAL_PCT)
                    remaining_lots = round_lot(full_lots - partial_lots)
                    pnl            = (entry_price - first_tp) * partial_lots * CONTRACT_SIZE
                    equity        += pnl
                    sl_price     = entry_price
                    trail_sl     = entry_price
                    partial_done  = True
                    trades.append(_make_row(
                        trade_id, entry_time, bar_time, "Short (Partial 50%)",
                        entry_price, first_tp, partial_lots, sl_price,
                        "Partial TP 1:2", pnl, equity
                    ))

                else:
                    # (c) ATR trail after partial
                    if partial_done:
                        new_trail = close + atr_now * TRAIL_ATR_MULT
                        trail_sl  = min(trail_sl, new_trail)
                        sl_price  = trail_sl

                    # (d) Opposite signal → close all
                    if is_buy:
                        pnl    = (entry_price - close) * remaining_lots * CONTRACT_SIZE
                        equity += pnl
                        trades.append(_make_row(
                            trade_id, entry_time, bar_time, "Short",
                            entry_price, close, remaining_lots, sl_price,
                            "Opposite Signal", pnl, equity
                        ))
                        in_trade = False

        # ────────────────────────────────────────────────────
        #  ENTER NEW TRADE  (only when flat)
        # ────────────────────────────────────────────────────
        if not in_trade:
            if is_buy:
                trade_id      += 1
                sl_dist        = SL_ATR_MULT * atr_now
                entry_price    = close
                direction      = "long"
                sl_price       = entry_price - sl_dist
                first_tp       = entry_price + PARTIAL_RR * sl_dist
                full_lots      = calc_lots(equity, sl_dist)
                remaining_lots = full_lots
                partial_done   = False
                trail_sl       = sl_price
                entry_time     = bar_time
                in_trade       = True

            elif is_sell:
                trade_id      += 1
                sl_dist        = SL_ATR_MULT * atr_now
                entry_price    = close
                direction      = "short"
                sl_price       = entry_price + sl_dist
                first_tp       = entry_price - PARTIAL_RR * sl_dist
                full_lots      = calc_lots(equity, sl_dist)
                remaining_lots = full_lots
                partial_done   = False
                trail_sl       = sl_price
                entry_time     = bar_time
                in_trade       = True

        equity_curve.append(equity)
        dates_curve.append(bar_time)

    return trades, equity_curve, dates_curve


def _make_row(trade_id, entry_time, exit_time, direction,
              entry_price, exit_price, lots, sl, reason, pnl, equity):
    return {
        "Trade ID":    trade_id,
        "Entry Time":  entry_time,
        "Exit Time":   exit_time,
        "Direction":   direction,
        "Entry Price": round(entry_price, 2),
        "Exit Price":  round(exit_price,  2),
        "Lots":        lots,
        "SL at Exit":  round(sl, 2),
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

    # ── Per-trade PnL (group partial + final by Trade ID) ──
    trade_pnl = df_t.groupby("Trade ID")["PnL ($)"].sum()
    total_trades = len(trade_pnl)
    winners      = trade_pnl[trade_pnl > 0]
    losers       = trade_pnl[trade_pnl <= 0]
    win_rate     = len(winners) / total_trades * 100 if total_trades else 0

    gross_profit = winners.sum()
    gross_loss   = losers.sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # ── Drawdown ──
    eq  = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    max_dd = dd.min()

    # ── Return ──
    final_eq     = equity_curve[-1]
    total_return = (final_eq - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    # ── Sharpe (annualised, 15-min bars) ──
    eq_s    = pd.Series(equity_curve)
    rets    = eq_s.pct_change().dropna()
    bars_py = 252 * 24 * 4   # ~96 15-min bars per trading day
    sharpe  = (rets.mean() / rets.std()) * np.sqrt(bars_py) if rets.std() != 0 else 0.0

    # ── Consecutive wins/losses ──
    streak = trade_pnl.apply(lambda x: 1 if x > 0 else -1)
    max_win_streak  = _max_streak(streak,  1)
    max_loss_streak = _max_streak(streak, -1)

    return {
        "Period":               f"{START_DATE.date()}  →  {END_DATE.date()}",
        "Symbol / TF":          f"{SYMBOL}  /  15-Minute",
        "Initial Equity ($)":   INITIAL_EQUITY,
        "Final Equity ($)":     round(final_eq, 2),
        "Total Return (%)":     round(total_return, 2),
        "Total PnL ($)":        round(trade_pnl.sum(), 2),
        "Total Trades":         total_trades,
        "Win Rate (%)":         round(win_rate, 2),
        "Avg Win ($)":          round(winners.mean(), 2) if len(winners) else 0,
        "Avg Loss ($)":         round(losers.mean(),  2) if len(losers)  else 0,
        "Best Trade ($)":       round(trade_pnl.max(), 2),
        "Worst Trade ($)":      round(trade_pnl.min(), 2),
        "Profit Factor":        round(profit_factor, 2),
        "Max Drawdown (%)":     round(max_dd, 2),
        "Sharpe Ratio":         round(sharpe, 2),
        "Max Win Streak":       max_win_streak,
        "Max Loss Streak":      max_loss_streak,
        "Risk per Trade ($)":   RISK_AMOUNT,
        "SL Multiplier (ATR)":  SL_ATR_MULT,
        "Partial TP RR":        f"1:{int(PARTIAL_RR)}",
    }


def _max_streak(series: pd.Series, val: int) -> int:
    max_s = cur = 0
    for v in series:
        cur = cur + 1 if v == val else 0
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
        f"XAUUSD UT Bot Backtest  │  15-Minute  │  "
        f"{START_DATE.strftime('%b %Y')} – {END_DATE.strftime('%b %Y')}",
        fontsize=13, fontweight="bold"
    )

    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100

    # ── Equity ──
    ax1.plot(dates_curve, equity_curve, color="#2196F3", linewidth=1.4, label="Equity")
    ax1.fill_between(dates_curve, equity_curve, INITIAL_EQUITY,
                     where=np.array(equity_curve) >= INITIAL_EQUITY,
                     alpha=0.15, color="#2196F3", interpolate=True)
    ax1.fill_between(dates_curve, equity_curve, INITIAL_EQUITY,
                     where=np.array(equity_curve) < INITIAL_EQUITY,
                     alpha=0.15, color="#F44336", interpolate=True)
    ax1.axhline(INITIAL_EQUITY, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    # Stats box
    ret   = stats.get("Total Return (%)", 0)
    wr    = stats.get("Win Rate (%)",     0)
    pf    = stats.get("Profit Factor",   0)
    mdd   = stats.get("Max Drawdown (%)", 0)
    sh    = stats.get("Sharpe Ratio",    0)
    box_txt = (f"Return: {ret:+.1f}%  │  Win Rate: {wr:.1f}%  │  "
               f"Profit Factor: {pf:.2f}  │  Max DD: {mdd:.1f}%  │  Sharpe: {sh:.2f}")
    ax1.text(0.01, 0.97, box_txt, transform=ax1.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

    # ── Drawdown ──
    ax2.fill_between(dates_curve, dd, 0, color="#F44336", alpha=0.4)
    ax2.plot(dates_curve, dd, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25)

    # X-axis formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

    plt.tight_layout()
    path = os.path.join(out_dir, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CHART] Equity curve saved → {path}")


# ──────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────

def save_trade_log(trades: list, out_dir: str):
    if not trades:
        return
    path = os.path.join(out_dir, "trade_log.csv")
    pd.DataFrame(trades).to_csv(path, index=False)
    print(f"[CSV]   Trade log saved   → {path}")


def save_summary(stats: dict, out_dir: str):
    path = os.path.join(out_dir, "backtest_summary.txt")
    sep  = "─" * 42
    with open(path, "w", encoding="utf-8") as f:
        f.write("XAUUSD UT Bot Backtest – Summary\n")
        f.write(sep + "\n")
        for k, v in stats.items():
            f.write(f"{k:<28} {v}\n")
        f.write(sep + "\n")
        f.write("\nStrategy Parameters\n")
        f.write(f"  Key Value (sensitivity)  : {KEY_VALUE}\n")
        f.write(f"  ATR Period               : {ATR_PERIOD}\n")
        f.write(f"  SL multiplier            : {SL_ATR_MULT}× ATR\n")
        f.write(f"  Partial close at         : 1:{int(PARTIAL_RR)} RR  (50% of position)\n")
        f.write(f"  Trail after partial      : {TRAIL_ATR_MULT}× ATR\n")
        f.write(f"  Risk per trade           : ${RISK_AMOUNT:.0f} fixed\n")
        f.write(f"  Initial equity           : ${INITIAL_EQUITY:,.0f}\n")
    print(f"[TXT]   Summary saved     → {path}")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  XAUUSD  UT Bot  │  15-Minute  │  MT5 Backtest")
    print("=" * 60)

    # 1. Connect
    if not connect_mt5():
        sys.exit(1)

    # 2. Download data
    print(f"\n[DATA] Fetching {SYMBOL} M15  {START_DATE.date()} → {END_DATE.date()} …")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if df.empty:
        mt5.shutdown()
        sys.exit(1)

    # 3. Indicators & signals
    print("\n[CALC] Computing UT Bot signals …")
    df = ut_bot_signals(df, KEY_VALUE, ATR_PERIOD)
    print(f"       Buy signals  : {df['Buy'].sum():,}")
    print(f"       Sell signals : {df['Sell'].sum():,}")

    # 4. Backtest
    print("\n[BACK] Running backtest …")
    trades, equity_curve, dates_curve = run_backtest(df)
    print(f"       Trade records logged : {len(trades)}")

    # 5. Statistics
    stats = compute_stats(trades, equity_curve)
    print("\n" + "─" * 48)
    print("  BACKTEST RESULTS")
    print("─" * 48)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print("─" * 48)

    # 6. Save outputs
    print()
    save_trade_log(trades, OUTPUT_DIR)
    plot_equity_curve(equity_curve, dates_curve, stats, OUTPUT_DIR)
    save_summary(stats, OUTPUT_DIR)

    mt5.shutdown()
    print("\n[DONE] MT5 disconnected. Backtest complete.")


if __name__ == "__main__":
    main()