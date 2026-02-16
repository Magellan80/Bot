import asyncio
import aiohttp

from data_layer import fetch_klines
from reversal_detector import ReversalDetector


SYMBOL = "BTCUSDT"
TIMEFRAME = "15"
LIMIT = 1000


async def load_data():
    async with aiohttp.ClientSession() as session:
        klines = await fetch_klines(
            session=session,
            symbol=SYMBOL,
            interval=TIMEFRAME,
            limit=LIMIT
        )

    if not klines:
        print("❌ No klines received from API")
        return [], [], [], []

    klines = list(reversed(klines))

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    return closes, highs, lows, volumes


def simulate_trade(direction, entry_price, future_closes):
    tp_pct = 0.01
    sl_pct = 0.01

    if direction == "bullish":
        tp = entry_price * (1 + tp_pct)
        sl = entry_price * (1 - sl_pct)

        for price in future_closes:
            if price >= tp:
                return 1
            if price <= sl:
                return -1

    if direction == "bearish":
        tp = entry_price * (1 - tp_pct)
        sl = entry_price * (1 + sl_pct)

        for price in future_closes:
            if price <= tp:
                return 1
            if price >= sl:
                return -1

    return 0


async def main():
    closes, highs, lows, volumes = await load_data()

    print("Loaded candles:", len(closes))

    if len(closes) < 200:
        print("❌ Not enough data")
        return

    detector = ReversalDetector()

    total = 0
    wins = 0
    losses = 0
    neutral = 0

    for i in range(120, len(closes) - 35):

        result = detector.analyze_reversal(
            closes[:i],
            highs[:i],
            lows[:i],
            volumes[:i],
            min_score=0
        )

        direction = result.get("reversal")

        if direction:
            total += 1

            entry = closes[i]
            future = closes[i + 1:i + 35]

            outcome = simulate_trade(direction, entry, future)

            if outcome == 1:
                wins += 1
            elif outcome == -1:
                losses += 1
            else:
                neutral += 1

    print("\n========== BACKTEST RESULT ==========")
    print("Symbol:", SYMBOL)
    print("Total signals:", total)
    print("Wins:", wins)
    print("Losses:", losses)
    print("No hit (35 candles):", neutral)

    if total > 0:
        overall_wr = round(wins / total * 100, 2)
        print("Overall Winrate:", overall_wr, "%")

    closed_trades = wins + losses

    if closed_trades > 0:
        closed_wr = round(wins / closed_trades * 100, 2)
        print("Closed Trades Winrate:", closed_wr, "%")

        expectancy = ((wins * 0.01) - (losses * 0.01)) / closed_trades
        print("Expectancy per trade:", round(expectancy, 5))


if __name__ == "__main__":
    asyncio.run(main())
