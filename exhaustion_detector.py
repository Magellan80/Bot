# exhaustion_detector.py

class ExhaustionDetector:
    """
    Pump/Dump exhaustion engine
    30% weight logic
    """

    def analyze(self, closes, highs, lows, volumes, rsi, atr, swings):

        if len(closes) < 30 or len(rsi) < 10:
            return None

        score = 0
        direction = None
        filters = []

        # ===== 1. Сильный импульс =====

        move = abs(closes[-1] - closes[-20])
        if move < atr[-1] * 3:
            return None

        score += 20
        filters.append("Strong impulse")

        # ===== 2. RSI экстремум =====

        if rsi[-1] > 75:
            direction = "bearish"
            score += 15
            filters.append("RSI overbought")

        elif rsi[-1] < 25:
            direction = "bullish"
            score += 15
            filters.append("RSI oversold")

        else:
            return None

        # ===== 3. Свечной rejection =====

        body = abs(closes[-1] - closes[-2])
        rng = highs[-1] - lows[-1]

        if rng == 0:
            return None

        if direction == "bearish" and (highs[-1] - closes[-1]) > body * 1.5:
            score += 15
            filters.append("Upper wick rejection")

        if direction == "bullish" and (closes[-1] - lows[-1]) > body * 1.5:
            score += 15
            filters.append("Lower wick rejection")

        # ===== 4. Volume spike =====

        avg_vol = sum(volumes[-20:-1]) / 19
        if volumes[-1] > avg_vol * 2:
            score += 10
            filters.append("Volume spike")

        if score < 65:
            return None

        return {
            "type": "exhaustion",
            "direction": direction,
            "rating": min(score, 100),
            "filters": filters
        }
