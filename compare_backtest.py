import json
import asyncio
import statistics
from reversal_detector import ReversalDetector

LOOKAHEAD = 35
RISK_PER_TRADE = 0.01
START_BALANCE = 10000


# ==============================
# TRADE SIMULATION
# ==============================

def simulate_trade(signal, future):

    entry = signal["entry"]
    sl = signal["stop"]
    tp = signal["take_profit"]
    side = signal["side"]

    risk = abs(entry - sl)
    if risk == 0:
        return 0.0

    for candle in future:

        high = float(candle["high"])
        low = float(candle["low"])

        if side == "long":
            if low <= sl:
                return -1.0
            if high >= tp:
                return abs(tp - entry) / risk

        else:
            if high >= sl:
                return -1.0
            if low <= tp:
                return abs(entry - tp) / risk

    return 0.0


# ==============================
# METRICS
# ==============================

def compute_metrics(trades, equity_curve):

    if not trades:
        return None

    wins = [r for r in trades if r > 0]
    losses = [r for r in trades if r < 0]

    expectancy = statistics.mean(trades)
    winrate = len(wins) / len(trades)

    if losses:
        pf = abs(sum(wins) / sum(losses))
    else:
        pf = float("inf")

    # Max DD
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        peak = max(peak, value)
        dd = peak - value
        max_dd = max(max_dd, dd)

    # Sharpe по R
    if len(trades) > 1:
        sharpe = expectancy / statistics.stdev(trades)
    else:
        sharpe = 0

    # Longest losing streak
    streak = 0
    max_streak = 0
    for r in trades:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trades": len(trades),
        "winrate": winrate,
        "expectancy": expectancy,
        "pf": pf,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "max_losing_streak": max_streak,
        "total_R": sum(trades)
    }


# ==============================
# BACKTEST
# ==============================

async def run_backtest(candles):

    detector = ReversalDetector()

    balance = START_BALANCE
    trades = []
    equity_curve = []

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

        trades.append(result_R)
        equity_curve.append(balance)

    return compute_metrics(trades, equity_curve)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    try:
        with open("data.json", "r") as f:
            candles = json.load(f)
    except Exception as e:
        print("ERROR loading data.json:", e)
        exit()

    metrics = asyncio.run(run_backtest(candles))

    if not metrics:
        print("No trades.")
        exit()

    print("\n==============================")
    print("BACKTEST RESULTS")
    print("==============================")

    print(f"Trades: {metrics['trades']}")
    print(f"Winrate: {metrics['winrate']*100:.2f}%")
    print(f"Expectancy: {metrics['expectancy']:.3f} R")
    print(f"Total R: {metrics['total_R']:.2f}")
    print(f"Profit Factor: {metrics['pf']:.2f}")
    print(f"Max DD: {metrics['max_dd']:.2f}")
    print(f"Sharpe (R): {metrics['sharpe']:.3f}")
    print(f"Longest Losing Streak: {metrics['max_losing_streak']}")
