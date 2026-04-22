import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= SETTINGS =================
FILE_NAME = "XAUUSD_M15.csv"
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 200  # $ risk
CONTRACT_SIZE = 100   # gold standard

ATR_PERIOD = 14
ADX_PERIOD = 14
SL_MULT = 1.5
RR = 2  # Risk Reward

ADX_THRESHOLD = 20  # filter bad trades


# ================= LOAD DATA =================
df = pd.read_csv(FILE_NAME)
df.columns = [c.lower() for c in df.columns]

# Ensure correct format
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')


# ================= INDICATORS =================
def calculate_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(period).mean()

def calculate_adx(df, period):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = calculate_atr(df, period)

    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = abs(100 * (minus_dm.rolling(period).mean() / tr))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()


df['atr'] = calculate_atr(df, ATR_PERIOD)
df['adx'] = calculate_adx(df, ADX_PERIOD)


# ================= STRATEGY =================
balance = INITIAL_BALANCE
equity_curve = []
trades = []

position = None

for i in range(50, len(df)):

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # Skip if indicators not ready
    if np.isnan(row['atr']) or np.isnan(row['adx']):
        continue

    # ================= ENTRY =================
    if position is None:

        # ADX filter
        if row['adx'] < ADX_THRESHOLD:
            continue

        # Simple breakout logic (you can replace with your Pine logic later)
        if row['close'] > prev['high']:
            direction = "BUY"
        elif row['close'] < prev['low']:
            direction = "SELL"
        else:
            continue

        entry = row['close']
        atr = row['atr']
        sl_dist = atr * SL_MULT

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + sl_dist * RR
        else:
            sl = entry + sl_dist
            tp = entry - sl_dist * RR

        # Position sizing
        lot = RISK_PER_TRADE / (sl_dist * CONTRACT_SIZE)

        position = {
            "type": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "entry_time": row['time']
        }

    # ================= EXIT =================
    else:
        high = row['high']
        low = row['low']

        exit_price = None
        result = None

        if position["type"] == "BUY":
            if low <= position["sl"]:
                exit_price = position["sl"]
                result = -RISK_PER_TRADE
            elif high >= position["tp"]:
                exit_price = position["tp"]
                result = RISK_PER_TRADE * RR

        else:
            if high >= position["sl"]:
                exit_price = position["sl"]
                result = -RISK_PER_TRADE
            elif low <= position["tp"]:
                exit_price = position["tp"]
                result = RISK_PER_TRADE * RR

        if exit_price is not None:
            balance += result

            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": row['time'],
                "type": position["type"],
                "entry": position["entry"],
                "exit": exit_price,
                "profit": result,
                "balance": balance
            })

            position = None

    equity_curve.append(balance)


# ================= RESULTS =================
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    win_rate = (trades_df['profit'] > 0).mean() * 100
    total_profit = trades_df['profit'].sum()
else:
    win_rate = 0
    total_profit = 0

print(f"Final Balance: ${balance}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total Profit: ${total_profit}")


# ================= SAVE CSV =================
trades_df.to_csv("trade_results.csv", index=False)


# ================= HTML REPORT =================
plt.figure()
plt.plot(equity_curve)
plt.title("Equity Curve")
plt.xlabel("Trades")
plt.ylabel("Balance")
plt.savefig("equity_curve.png")

html_content = f"""
<html>
<head><title>Backtest Report</title></head>
<body>
<h1>Backtest Report</h1>
<p>Final Balance: ${balance}</p>
<p>Win Rate: {win_rate:.2f}%</p>
<p>Total Profit: ${total_profit}</p>
<img src="equity_curve.png">
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(html_content)

print("✅ Done: CSV + HTML generated")