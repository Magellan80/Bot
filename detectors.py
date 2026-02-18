# detectors.py
# PRODUCTION STABLE DETECTOR ENGINE v5
# Совместим с screener V10.2 router

import numpy as np


# ============================================================
# UTILITIES
# ============================================================

def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)

    return float(np.mean(trs[-period:]))


# ============================================================
# MAIN CLASS
# ============================================================

class Detector:
    """
    Production-safe detector.
    Не фильтрует по min_score внутри.
    Всегда возвращает:

    {
        "reversal": str | None,
        "rating": int
    }
    """

    # ========================================================
    # REVERSAL
    # ========================================================

    def analyze_reversal(
        self,
        closes,
        highs,
        lows,
        volumes,
        htf_trend_1h=0,
        htf_trend_4h=0,
        structure_1h=None,
        structure_4h=None,
        event_1h=None,
        event_4h=None,
        market_regime="neutral",
        asset_class="mid",
        min_score=0,
    ):

        closes = np.array(closes)
        highs = np.array(highs)
        lows = np.array(lows)
        volumes = np.array(volumes)

        if len(closes) < 30:
            return {"reversal": None, "rating": 0}

        rsi = _rsi(closes)
        atr = _atr(highs, lows, closes)

        rating = 0
        direction = None

        # 1️⃣ RSI exhaustion
        if rsi > 72:
            rating += 25
            direction = "bearish"

        if rsi < 28:
            rating += 25
            direction = "bullish"

        # 2️⃣ Impulse candle
        recent_range = highs[-1] - lows[-1]
        if atr > 0 and recent_range > atr * 1.6:
            rating += 15

        # 3️⃣ Volume spike
        avg_vol = np.mean(volumes[-20:])
        if volumes[-1] > avg_vol * 1.8:
            rating += 15

        # 4️⃣ HTF conflict bonus
        if direction == "bearish" and (htf_trend_1h > 0 or htf_trend_4h > 0):
            rating += 10

        if direction == "bullish" and (htf_trend_1h < 0 or htf_trend_4h < 0):
            rating += 10

        # 5️⃣ Regime normalization
        if market_regime == "high_vol":
            rating *= 0.95

        if asset_class == "major":
            rating *= 1.03

        rating = int(max(0, min(rating, 100)))

        return {
            "reversal": direction,
            "rating": rating
        }

    # ========================================================
    # CONTINUATION
    # ========================================================

    def analyze_continuation(
        self,
        closes,
        highs,
        lows,
        volumes,
        trend_1h=0,
        trend_4h=0,
        asset_class="mid",
        market_regime="neutral",
        min_score=0,
    ):

        closes = np.array(closes)
        highs = np.array(highs)
        lows = np.array(lows)
        volumes = np.array(volumes)

        if len(closes) < 30:
            return {"direction": None, "rating": 0}

        rating = 0
        direction = None

        trend_sum = trend_1h + trend_4h

        # 1️⃣ HTF alignment
        if trend_sum >= 2:
            rating += 30
            direction = "bullish"

        if trend_sum <= -2:
            rating += 30
            direction = "bearish"

        # 2️⃣ Breakout
        if closes[-1] > max(highs[-10:-1]):
            rating += 20

        if closes[-1] < min(lows[-10:-1]):
            rating += 20

        # 3️⃣ Volume confirmation
        avg_vol = np.mean(volumes[-20:])
        if volumes[-1] > avg_vol * 1.5:
            rating += 15

        # 4️⃣ Regime multiplier
        if market_regime == "trending":
            rating *= 1.05

        if asset_class == "major":
            rating *= 1.02

        rating = int(max(0, min(rating, 100)))

        return {
            "direction": direction,
            "rating": rating
        }

    # ========================================================
    # HABR (lightweight scoring only)
    # ========================================================

    def analyze_habr(
        self,
        closes,
        highs,
        lows,
        volumes
    ):
        """
        Не блокирующий модуль.
        Возвращает дополнительный скоринг.
        """

        closes = np.array(closes)
        volumes = np.array(volumes)

        if len(closes) < 20:
            return {"score": 0}

        rating = 0

        momentum = closes[-1] - closes[-5]
        avg_vol = np.mean(volumes[-20:])

        if momentum > 0:
            rating += 5

        if volumes[-1] > avg_vol * 1.3:
            rating += 5

        return {"score": int(rating)}
