# continuation_detector.py

class ContinuationDetector:
    """
    Trend continuation engine
    70% weight logic
    """

    def analyze(self, closes, highs, lows, volumes, atr, swings, htf_bias=None):

        if len(swings) < 3:
            return None

        score = 0
        direction = None
        filters = []

        # ===== 1. Определяем структуру =====

        last, prev, prev2 = swings[-1], swings[-2], swings[-3]

        bullish_structure = (
            prev2["type"] == "low" and
            prev["type"] == "high" and
            last["type"] == "low"
        )

        bearish_structure = (
            prev2["type"] == "high" and
            prev["type"] == "low" and
            last["type"] == "high"
        )

        if bullish_structure:
            direction = "bullish"
            score += 25
            filters.append("Bullish structure")

        if bearish_structure:
            direction = "bearish"
            score += 25
            filters.append("Bearish structure")

        if direction is None:
            return None

        # ===== 2. BOS =====

        level = closes[last["index"]]
        buffer = level * 0.001

        if direction == "bullish" and closes[-1] > level + buffer:
            score += 20
            filters.append("Bullish BOS")

        elif direction == "bearish" and closes[-1] < level - buffer:
            score += 20
            filters.append("Bearish BOS")
        else:
            return None

        # ===== 3. Pullback ATR filter =====

        move = abs(closes[-1] - closes[-15])
        if move > atr[-1] * 1.5:
            score += 15
            filters.append("Momentum confirmed")

        # ===== 4. HTF alignment bonus =====

        if htf_bias is not None:
            if (htf_bias == "bull" and direction == "bullish") or \
               (htf_bias == "bear" and direction == "bearish"):
                score += 10
                filters.append("HTF aligned")

        if score < 55:
            return None

        return {
            "type": "continuation",
            "direction": direction,
            "rating": min(score, 100),
            "filters": filters
        }
