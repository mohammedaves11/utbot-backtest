from backtest_utils import load_data
from strategy2_half_tp_be_trail import run_backtest

df = load_data('csv', csv_path='XAUUSD_M15_TillDate.csv')
trades, equity, report_path = run_backtest(df, output_dir='reports')