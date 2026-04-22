"""
backtest_utils.py
=================
Shared core for all four XAUUSD backtest strategies.

Faithfully reproduces the "AS Alerts Gold 200 $" Pine Script v6 indicator:
  • ATR Trailing Stop signal (UT-Bot style, key_value × ATR)
  • ADX trend filter (skip trades when ADX < threshold)
  • Auto position sizing  (risk_usd / (sl_dist × contract_size))

Dependencies:  pip install pandas numpy matplotlib
Optional:      pip install MetaTrader5    (for live MT5 data)
"""

import io
import os
import base64
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT PARAMETERS  — match Pine Script defaults exactly
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    key_value     = 3,      # ATR sensitivity multiplier (nLoss = key_value × ATR)
    atr_period    = 14,     # ATR period
    sl_mult       = 1.5,    # SL distance = sl_mult × ATR
    risk_usd      = 200.0,  # USD risked per trade
    contract_size = 100.0,  # Gold: 1 lot = 100 troy oz
    adx_len       = 14,     # ADX smoothing length
    adx_thresh    = 20,     # ADX threshold — below this = choppy, skip
    filter_choppy = True,   # True → skip signals when ADX < adx_thresh
    trail_mult    = 1.5,    # ATR multiplier for trailing stop (strategies 2/3/4)
                            # 1.5 = same as SL → tighter trail, locks in more profit
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(filepath: str,
             time_col:  str = 'time',
             open_col:  str = 'open',
             high_col:  str = 'high',
             low_col:   str = 'low',
             close_col: str = 'close',
             sep:       str = ',') -> pd.DataFrame:
    """
    Load standard OHLC CSV data.

    Expected column names (all configurable via keyword args):
        time, open, high, low, close

    The datetime column is parsed automatically by pandas.

    Parameters
    ----------
    filepath  : str  — path to .csv file
    time_col  : str  — name of the datetime / timestamp column
    open_col  : str  — name of the open-price column
    high_col  : str  — name of the high-price column
    low_col   : str  — name of the low-price column
    close_col : str  — name of the close-price column
    sep       : str  — CSV column separator (default ',')

    Returns
    -------
    pd.DataFrame  — DatetimeIndex, columns: [open, high, low, close]

    Example
    -------
    df = load_csv('XAUUSD_M15.csv')
    # Custom column names:
    df = load_csv('data.csv', time_col='Date', open_col='Open', ...)
    """
    raw = pd.read_csv(filepath, sep=sep)
    raw = raw.rename(columns={
        time_col:  'time',
        open_col:  'open',
        high_col:  'high',
        low_col:   'low',
        close_col: 'close',
    })
    raw['time'] = pd.to_datetime(raw['time'])
    df = (raw.set_index('time')
             .sort_index()[['open', 'high', 'low', 'close']]
             .astype(float)
             .dropna())
    print(f"[CSV]  {len(df):,} bars loaded  "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
          f"  from  {filepath}")
    return df


def load_mt5(symbol:        str  = 'XAUUSD',
             timeframe_str: str  = 'M15',
             start_date           = None,
             end_date             = None,
             n_bars:        int  = 50_000) -> pd.DataFrame:
    """
    Load OHLC data directly from a running MetaTrader 5 terminal.

    Requirements
    ------------
    pip install MetaTrader5
    MT5 desktop terminal must be open and logged into a live/demo account.

    Parameters
    ----------
    symbol        : str      — MT5 symbol, e.g. 'XAUUSD' or 'XAUUSD.t'
    timeframe_str : str      — '1m','5m','15m','M15','30m','1H','H1','4H','H4',
                               '1D','D1','1W'
    start_date    : datetime — optional range start
    end_date      : datetime — optional range end
    n_bars        : int      — bars to fetch when no date range is given

    Returns
    -------
    pd.DataFrame  — DatetimeIndex, columns: [open, high, low, close]

    Example
    -------
    from datetime import datetime
    df = load_mt5('XAUUSD', 'M15', n_bars=50000)
    df = load_mt5('XAUUSD', 'M15',
                  start_date=datetime(2022,1,1),
                  end_date=datetime(2024,12,31))
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not found.\n"
            "  Install:  pip install MetaTrader5\n"
            "  Then ensure the MT5 terminal is running and logged in."
        )

    TF = {
        '1m':  mt5.TIMEFRAME_M1,  'M1':  mt5.TIMEFRAME_M1,
        '5m':  mt5.TIMEFRAME_M5,  'M5':  mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15, 'M15': mt5.TIMEFRAME_M15,
        '30m': mt5.TIMEFRAME_M30, 'M30': mt5.TIMEFRAME_M30,
        '1H':  mt5.TIMEFRAME_H1,  'H1':  mt5.TIMEFRAME_H1,
        '4H':  mt5.TIMEFRAME_H4,  'H4':  mt5.TIMEFRAME_H4,
        '1D':  mt5.TIMEFRAME_D1,  'D1':  mt5.TIMEFRAME_D1,
        '1W':  mt5.TIMEFRAME_W1,  'W1':  mt5.TIMEFRAME_W1,
    }
    tf = TF.get(timeframe_str)
    if tf is None:
        raise ValueError(f"Unknown timeframe '{timeframe_str}'.  "
                         f"Valid values: {list(TF.keys())}")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    try:
        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        elif start_date:
            rates = mt5.copy_rates_from(symbol, tf, start_date, n_bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe_str}.  "
                f"MT5 error: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = (df.set_index('time')
               .sort_index()[['open', 'high', 'low', 'close']]
               .astype(float)
               .dropna())
        print(f"[MT5]  {len(df):,} bars loaded  {symbol} {timeframe_str}  "
              f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        return df
    finally:
        mt5.shutdown()


def load_data(source:     str = 'csv',
              csv_path:   str = None,
              symbol:     str = 'XAUUSD',
              timeframe:  str = 'M15',
              start_date      = None,
              end_date        = None,
              n_bars:     int = 50_000,
              **csv_kwargs) -> pd.DataFrame:
    """
    Unified data-loading entry point.  Pass to run_backtest() directly.

    Parameters
    ----------
    source     : 'csv' or 'mt5'
    csv_path   : path to CSV file   (required when source='csv')
    symbol     : MT5 symbol         (used when source='mt5')
    timeframe  : timeframe string   (used when source='mt5')
    start_date, end_date, n_bars  : MT5 date-range options
    **csv_kwargs : forwarded to load_csv()  (time_col, sep, open_col, …)

    Examples
    --------
    # From CSV
    df = load_data('csv', csv_path='XAUUSD_M15.csv')

    # From MT5
    df = load_data('mt5', symbol='XAUUSD', timeframe='M15', n_bars=50000)

    # MT5 with date range
    from datetime import datetime
    df = load_data('mt5', symbol='XAUUSD', timeframe='M15',
                   start_date=datetime(2022,1,1), end_date=datetime(2024,12,31))
    """
    source = source.lower()
    if source == 'csv':
        if not csv_path:
            raise ValueError("csv_path is required when source='csv'")
        return load_csv(csv_path, **csv_kwargs)
    elif source == 'mt5':
        return load_mt5(symbol, timeframe, start_date, end_date, n_bars)
    else:
        raise ValueError(f"source must be 'csv' or 'mt5', got '{source}'")


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR CALCULATIONS  — faithful Pine Script v6 port
# ══════════════════════════════════════════════════════════════════════════════

def _atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range smoothed via Wilder's method — identical to Pine ta.atr()."""
    prev_cl = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_cl).abs(),
        (df['low']  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _adx_wilder(df: pd.DataFrame, period: int):
    """
    ADX, +DI, -DI via Wilder smoothing — identical to Pine ta.dmi().
    Returns (plus_di, minus_di, adx)  all as pd.Series.
    """
    up   =  df['high'].diff()
    down = -df['low'].diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=df.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    prev_cl = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_cl).abs(),
        (df['low']  - prev_cl).abs(),
    ], axis=1).max(axis=1)

    a   = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr
    dx  = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=a, adjust=False).mean()
    return pdi, mdi, adx


def compute_indicators(df:            pd.DataFrame,
                        key_value:    float = DEFAULTS['key_value'],
                        atr_period:   int   = DEFAULTS['atr_period'],
                        sl_mult:      float = DEFAULTS['sl_mult'],
                        adx_len:      int   = DEFAULTS['adx_len'],
                        adx_thresh:   int   = DEFAULTS['adx_thresh'],
                        filter_choppy: bool = DEFAULTS['filter_choppy'],
                        risk_usd:     float = DEFAULTS['risk_usd'],
                        contract_size:float = DEFAULTS['contract_size'],
                        ) -> pd.DataFrame:
    """
    Compute all Pine Script indicator values and entry signals.

    Adds the following columns to a copy of df:
        atr           — Wilder ATR(14)
        n_loss        — key_value × ATR  (trailing stop distance for signals)
        trailing_stop — UTBot ATR trailing stop line
        sl_dist       — sl_mult × ATR    (SL distance from entry)
        sl_buy        — entry SL level for long  = close - sl_dist
        sl_sell       — entry SL level for short = close + sl_dist
        lot_size      — risk_usd / (sl_dist × contract_size)
        adx           — ADX value
        buy           — True on filtered long-entry bar
        sell          — True on filtered short-entry bar
        buy_choppy    — True on unfiltered long  (ADX too low, skipped)
        sell_choppy   — True on unfiltered short (ADX too low, skipped)

    Parameters
    ----------
    df            : OHLC DataFrame (from load_data / load_csv / load_mt5)
    key_value     : ATR multiplier for the trailing-stop signal line (default 3)
    atr_period    : ATR smoothing period (default 14)
    sl_mult       : SL distance in ATR multiples (default 1.5)
    adx_len       : ADX smoothing period (default 14)
    adx_thresh    : Minimum ADX to consider a trend valid (default 20)
    filter_choppy : True → only trade when ADX >= adx_thresh
    risk_usd      : USD to risk per trade (default 200)
    contract_size : Contract size in oz  (default 100 for Gold)
    """
    df  = df.copy()
    src = df['close']

    atr   = _atr_wilder(df, atr_period)
    nLoss = key_value * atr

    # ── ATR Trailing Stop (direct Python port of the Pine Script loop) ────────
    ts      = np.zeros(len(df))
    src_arr = src.values
    nl_arr  = nLoss.values

    for i in range(1, len(df)):
        s, s1, nl, p = src_arr[i], src_arr[i-1], nl_arr[i], ts[i-1]
        if   s > p and s1 > p:  ts[i] = max(p, s - nl)
        elif s < p and s1 < p:  ts[i] = min(p, s + nl)
        elif s > p:              ts[i] = s - nl
        else:                    ts[i] = s + nl

    trailing_stop = pd.Series(ts, index=df.index)

    # ── Entry signals ─────────────────────────────────────────────────────────
    # Pine: above = ta.crossover(ema1, trailing_stop)  where ema1 = close
    above    = (src.shift(1) < trailing_stop.shift(1)) & (src > trailing_stop)
    below    = (src.shift(1) > trailing_stop.shift(1)) & (src < trailing_stop)
    buy_raw  = (src > trailing_stop) & above
    sell_raw = (src < trailing_stop) & below

    # ── ADX filter ─────────────────────────────────────────────────────────────
    _, _, adx_val   = _adx_wilder(df, adx_len)
    trending        = adx_val >= adx_thresh
    buy_choppy      = buy_raw  & ~trending
    sell_choppy     = sell_raw & ~trending
    buy             = (buy_raw  & trending) if filter_choppy else buy_raw
    sell            = (sell_raw & trending) if filter_choppy else sell_raw

    # ── SL & lot sizing ────────────────────────────────────────────────────────
    sl_dist  = sl_mult * atr
    lot_size = risk_usd / (sl_dist * contract_size)
    lot_size = lot_size.where(sl_dist > 0.0001, 0).round(2).clip(lower=0.01)

    df['atr']          = atr
    df['n_loss']       = nLoss
    df['trailing_stop']= trailing_stop
    df['sl_dist']      = sl_dist
    df['sl_buy']       = src - sl_dist
    df['sl_sell']      = src + sl_dist
    df['lot_size']     = lot_size
    df['adx']          = adx_val
    df['buy']          = buy
    df['sell']         = sell
    df['buy_choppy']   = buy_choppy
    df['sell_choppy']  = sell_choppy
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(trades: list, equity_curve: pd.Series) -> dict:
    """Compute standard backtest statistics from a list of trade dicts."""
    if not trades:
        return {}

    df   = pd.DataFrame(trades)
    pnl  = df['pnl']
    wins = df[pnl > 0]
    loss = df[pnl <= 0]
    n    = len(df)

    gw = wins['pnl'].sum() if len(wins) else 0.0
    gl = loss['pnl'].abs().sum() if len(loss) else 0.0
    aw = wins['pnl'].mean() if len(wins) else 0.0
    al = loss['pnl'].mean() if len(loss) else 0.0
    wr = len(wins) / n * 100

    # Max Drawdown
    peak    = equity_curve.cummax()
    dd      = equity_curve - peak
    mdd     = dd.min()
    idx_mdd = dd.values.argmin()
    worst   = peak.iloc[idx_mdd] if len(peak) > 0 else 1
    mdd_pct = (mdd / worst * 100) if worst else 0

    sharpe = (pnl.mean() / pnl.std() * (252 ** 0.5)) if pnl.std() > 0 else 0.0

    return dict(
        n=n, wins=len(wins), losses=len(loss),
        win_rate=wr,
        total_pnl=pnl.sum(),
        gross_profit=gw, gross_loss=gl,
        profit_factor=gw / gl if gl else float('inf'),
        avg_win=aw, avg_loss=al,
        rr=abs(aw / al) if al else float('inf'),
        expectancy=wr/100 * aw + (1 - wr/100) * al,
        max_dd=mdd, max_dd_pct=mdd_pct,
        sharpe=sharpe,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(trades:        list,
                    equity_curve:  pd.Series,
                    strategy_name: str,
                    output_path:   str,
                    symbol:        str  = 'XAUUSD',
                    params:        dict = None) -> str:
    """
    Generate a self-contained HTML backtest report with embedded charts.

    Parameters
    ----------
    trades        : list of trade dicts returned by run_backtest()
    equity_curve  : pd.Series — cumulative P&L indexed by bar datetime
    strategy_name : display name shown in the report header
    output_path   : file path to save the .html file
    symbol        : symbol label (display only)
    params        : dict of strategy parameters to show in the report

    Returns
    -------
    output_path : str — same as the input argument
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtk
    except ImportError:
        raise ImportError(
            "matplotlib is required for reports.\n"
            "  Install:  pip install matplotlib"
        )

    m   = calc_metrics(trades, equity_curve)
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Colour palette ────────────────────────────────────────────────────────
    BG, CARD, PANEL = '#0d0d1a', '#14142a', '#181830'
    GRN, RED, BLUE  = '#00e5aa', '#ff4d4d', '#4da6ff'
    MUTED           = '#7779aa'

    def _b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=BG)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _style(ax, title='', dollar_y=True):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color('#252550')
        ax.tick_params(colors=MUTED, labelsize=8)
        if title:
            ax.set_title(title, color='#ccccee', fontsize=11, pad=8)
        if dollar_y:
            ax.yaxis.set_major_formatter(
                mtk.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # ── Figure 1: Equity curve + Drawdown ─────────────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6), facecolor=BG,
        gridspec_kw={'height_ratios': [3, 1]})
    fig1.subplots_adjust(hspace=0.06)

    ax1.plot(equity_curve.index, equity_curve.values, color=GRN, lw=1.5, zorder=2)
    ax1.fill_between(equity_curve.index, equity_curve.values, alpha=0.12, color=GRN)
    ax1.axhline(0, color='#ffffff20', lw=0.8, ls='--')
    _style(ax1, 'Equity Curve')
    ax1.tick_params(labelbottom=False)

    peak = equity_curve.cummax()
    dd   = equity_curve - peak
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.55)
    _style(ax2, 'Drawdown')
    eq_b64 = _b64(fig1)
    plt.close(fig1)

    # ── Figure 2: Monthly P&L ──────────────────────────────────────────────────
    monthly_html = ''
    if not tdf.empty and 'entry_time' in tdf.columns:
        tdf2 = tdf.copy()
        tdf2['_m'] = pd.to_datetime(tdf2['entry_time']).dt.to_period('M')
        mon = tdf2.groupby('_m')['pnl'].sum()
        mon.index = mon.index.astype(str)

        fig2, ax3 = plt.subplots(figsize=(13, 3), facecolor=BG)
        bars_color = [GRN if v >= 0 else RED for v in mon.values]
        ax3.bar(range(len(mon)), mon.values, color=bars_color, alpha=0.85)
        ax3.set_xticks(range(len(mon)))
        ax3.set_xticklabels(mon.index, rotation=45, ha='right', fontsize=7, color=MUTED)
        ax3.axhline(0, color='#ffffff20', lw=0.8, ls='--')
        _style(ax3, 'Monthly P&L')
        mon_b64 = _b64(fig2)
        plt.close(fig2)

        mon_rows = ''.join(
            f"<tr><td>{mi}</td>"
            f"<td style='color:{'#00e5aa' if v >= 0 else '#ff4d4d'}'>${v:,.2f}</td></tr>"
            for mi, v in zip(mon.index, mon.values))
        monthly_html = f"""
<div class="card">
  <h2>Monthly P&amp;L</h2>
  <img src="data:image/png;base64,{mon_b64}" style="width:100%;border-radius:6px"/>
  <div style="max-height:200px;overflow-y:auto;margin-top:14px">
    <table><thead><tr><th>Month</th><th>P&amp;L (USD)</th></tr></thead>
    <tbody>{mon_rows}</tbody></table>
  </div>
</div>"""

    # ── Figure 3: P&L Distribution ────────────────────────────────────────────
    fig3, ax4 = plt.subplots(figsize=(7, 3), facecolor=BG)
    if not tdf.empty:
        ax4.hist(tdf['pnl'], bins=40, color=BLUE, alpha=0.70, edgecolor=BG)
        ax4.axvline(0, color='#ffffff50', lw=1, ls='--')
    _style(ax4, 'P&L Distribution')
    dist_b64 = _b64(fig3)
    plt.close(fig3)

    # ── Trade table HTML ───────────────────────────────────────────────────────
    COLS = ['entry_time', 'exit_time', 'direction',
            'entry_price', 'exit_price', 'sl_price', 'lots', 'pnl', 'exit_reason']
    show = [c for c in COLS if c in tdf.columns]
    trade_rows_html = ''
    if not tdf.empty:
        for _, r in tdf.iterrows():
            pv  = r.get('pnl', 0)
            cls = 'win' if pv > 0 else 'loss'
            cells = ''
            for c in show:
                v = r[c]
                if c == 'pnl':
                    col = '#00e5aa' if pv > 0 else '#ff4d4d'
                    cells += f"<td style='color:{col};font-weight:600'>${v:,.2f}</td>"
                elif c in ('entry_price', 'exit_price', 'sl_price'):
                    cells += f"<td>{v:.2f}</td>"
                elif c == 'lots':
                    cells += f"<td>{v:.2f}</td>"
                else:
                    cells += f"<td>{v}</td>"
            trade_rows_html += f"<tr class='{cls}'>{cells}</tr>\n"
    thead = ''.join(f"<th>{c.replace('_', ' ').title()}</th>" for c in show)

    # ── Parameter table ────────────────────────────────────────────────────────
    param_rows = ''
    if params:
        for k, v in params.items():
            param_rows += (f"<tr><td>{k.replace('_', ' ')}</td>"
                           f"<td><b>{v}</b></td></tr>")

    # ── Summary stat cards ─────────────────────────────────────────────────────
    def _card(label, val, color=BLUE):
        return (f"<div class='sc'>"
                f"<div class='sl'>{label}</div>"
                f"<div class='sv' style='color:{color}'>{val}</div></div>")

    wr   = m.get('win_rate', 0)
    tp   = m.get('total_pnl', 0)
    pf   = m.get('profit_factor', 0)
    mdd  = m.get('max_dd', 0)
    sh   = m.get('sharpe', 0)
    exp  = m.get('expectancy', 0)
    pf_s = f"{pf:.2f}" if pf != float('inf') else "∞"
    rr_s = f"{m.get('rr', 0):.2f}" if m.get('rr', 0) != float('inf') else "∞"

    stats_html = (
        _card('Total Trades',  m.get('n', 0),                   BLUE) +
        _card('Win Rate',      f"{wr:.1f}%",                    GRN if wr >= 50 else RED) +
        _card('Net P&L',       f"${tp:,.2f}",                   GRN if tp >= 0 else RED) +
        _card('Profit Factor', pf_s,                            GRN if m.get('profit_factor',0) >= 1 else RED) +
        _card('Avg Win',       f"${m.get('avg_win', 0):,.2f}",  GRN) +
        _card('Avg Loss',      f"${m.get('avg_loss', 0):,.2f}", RED) +
        _card('R:R',           rr_s,                            BLUE) +
        _card('Max Drawdown',  f"${mdd:,.2f}",                  RED) +
        _card('Max DD %',      f"{m.get('max_dd_pct', 0):.1f}%",RED) +
        _card('Sharpe Ratio',  f"{sh:.2f}",                     GRN if sh >= 1 else BLUE) +
        _card('Expectancy',    f"${exp:,.2f}",                  GRN if exp >= 0 else RED) +
        _card('W / L',         f"{m.get('wins',0)} / {m.get('losses',0)}", MUTED)
    )

    run_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{strategy_name} — {symbol}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:{BG};color:#dde;line-height:1.4}}
.hdr{{background:linear-gradient(135deg,#1a1a40,#252568);padding:24px 36px;border-bottom:2px solid #3535a0}}
.hdr h1{{font-size:1.6rem;color:#fff}}.hdr p{{color:#888;margin-top:4px;font-size:.84rem}}
.wrap{{max-width:1380px;margin:0 auto;padding:24px 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:11px;margin-bottom:20px}}
.sc{{background:{CARD};border:1px solid #252555;border-radius:9px;padding:14px;text-align:center}}
.sl{{font-size:.7rem;color:#666;text-transform:uppercase;letter-spacing:.9px}}
.sv{{font-size:1.38rem;font-weight:700;margin-top:3px}}
.card{{background:{CARD};border:1px solid #252555;border-radius:9px;padding:18px;margin-bottom:18px}}
.card h2{{font-size:.92rem;color:#aab;margin-bottom:11px;border-bottom:1px solid #252555;padding-bottom:7px}}
table{{width:100%;border-collapse:collapse;font-size:.79rem}}
th{{background:#0f0f20;color:#6668aa;padding:8px 10px;text-align:left;border-bottom:2px solid #252555;white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid #181830}}
tr.win td{{background:rgba(0,229,170,.03)}}
tr.loss td{{background:rgba(255,77,77,.03)}}
tr:hover td{{background:#1c1c3c!important}}
.r2{{display:grid;grid-template-columns:3fr 2fr;gap:16px}}
@media(max-width:720px){{.r2{{grid-template-columns:1fr}}.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="hdr">
  <h1>📊 {strategy_name}</h1>
  <p>{symbol} &nbsp;·&nbsp; Backtest Report &nbsp;·&nbsp; {run_ts}</p>
</div>
<div class="wrap">

<div class="grid">{stats_html}</div>

<div class="card">
  <h2>Equity Curve &amp; Drawdown</h2>
  <img src="data:image/png;base64,{eq_b64}" style="width:100%;border-radius:6px"/>
</div>

{monthly_html}

<div class="r2">
  <div class="card">
    <h2>P&amp;L Distribution</h2>
    <img src="data:image/png;base64,{dist_b64}" style="width:100%;border-radius:6px"/>
  </div>
  <div class="card">
    <h2>Strategy Parameters</h2>
    <table style="max-width:360px">
      <tr><th>Parameter</th><th>Value</th></tr>
      {param_rows}
    </table>
  </div>
</div>

<div class="card">
  <h2>Trade Log ({m.get('n', 0)} records)</h2>
  <div style="overflow-x:auto">
    <table>
      <thead><tr>{thead}</tr></thead>
      <tbody>{trade_rows_html}</tbody>
    </table>
  </div>
</div>

</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[REPORT]  Saved → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — build a trade record dict
# ══════════════════════════════════════════════════════════════════════════════

def _make_trade(entry_time, exit_time, direction,
                entry_price, exit_price, sl_price,
                lots, pnl, exit_reason) -> dict:
    """Build a standardised trade record."""
    return dict(
        entry_time  = entry_time,
        exit_time   = exit_time,
        direction   = direction.upper(),
        entry_price = round(float(entry_price), 2),
        exit_price  = round(float(exit_price),  2),
        sl_price    = round(float(sl_price),     2),
        lots        = round(float(lots),         2),
        pnl         = round(float(pnl),          2),
        exit_reason = exit_reason,
    )