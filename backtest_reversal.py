# ============================================================
# V12.2 REAL R BACKTEST — DEBUG VERSION
# ============================================================

import json
import asyncio
from reversal_detector import ReversalDetector


LOOKAHEAD = 35
START_BALANCE = 10000
RISK_PER_TRADE = 0.01


# ============================================================
# REAL R SIMULATION
# ============================================================

def simulate_trade(signal, future_candles):

    entry = signal["entry"]
    sl = signal["stop"]
    tp = signal["take_profit"]
    side = signal["side"]

    risk = abs(entry - sl)
    if risk == 0:
        return 0.0

    for candle in future_candles:

        high = float(candle["high"])
        low = float(candle["low"])

        if side == "long":

            if low <= sl:
                return -1.0

            if high >= tp:
                reward = abs(tp - entry)
                return reward / risk

        else:

            if high >= sl:
                return -1.0

            if low <= tp:
                reward = abs(entry - tp)
                return reward / risk

    return 0.0


# ============================================================
# BACKTEST
# ============================================================

async def run_backtest(candles):

    detector = ReversalDetector()

    balance = START_BALANCE
    peak_balance = START_BALANCE

    total_trades = 0
    wins = 0
    losses = 0
    no_hits = 0

    print("Starting backtest...")
    print(f"Total candles: {len(candles)}")

    for i in range(100, len(candles) - LOOKAHEAD):

        ltf = candles[:i]
        tf15 = candles[:i]
        tf1h = candles[:i]
        tf4h = candles[:i]

        signal = detector.detect(ltf, tf15, tf1h, tf4h)

        if not signal:
            continue

        future = candles[i:i + LOOKAHEAD]

        result_R = simulate_trade(signal, future)

        risk_amount = balance * RISK_PER_TRADE
        pnl = risk_amount * result_R

        balance += pnl
        total_trades += 1

        if result_R > 0:
            wins += 1
        elif result_R < 0:
            losses += 1
        else:
            no_hits += 1

        if balance > peak_balance:
            peak_balance = balance

    print("\n==============================")
    print("REAL R BACKTEST")
    print("==============================")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"No hit: {no_hits}")

    if total_trades > 0:
        print(f"Winrate: {wins / total_trades * 100:.2f}%")

    print(f"Final balance: {balance:.2f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading candles...")

    try:
        with open("data.json", "r") as f:
            candles = json.load(f)
    except Exception as e:
        print("ERROR loading data.json:", e)
        exit()

    asyncio.run(run_backtest(candles))
