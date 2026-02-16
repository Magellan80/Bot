# reversal_detector.py
# V7 PRO – Regime Adaptive Elite (FIXED ZERO RATING ISSUE)

import numpy as np
from typing import Dict


class ReversalDetector:

    def __init__(self):
        self.MIN_BARS = 120

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def analyze(self, data: Dict, symbol: str, btc_regime: str = "neutral"):

        closes = np.array(data["close"])
        highs = np.array(data["high"])
        lows = np.array(data["low"])
        volumes = np.array(data["volume"])

        if len(closes) < self.MIN_BARS:
            return self._empty()

        rsi = np.array(data["rsi"])
        macd = np.array(data["macd"])
        macd_signal = np.array(data["macd_signal"])
        atr = np.array(data["atr"])

        # ==========================================================
        # SMART REGIME HANDLING
        # ==========================================================

        if btc_regime in ["ranging", "neutral"]:
            regime_mode = "range"
        else:
            regime_mode = "trend"

        trend_ok = self._trend_filter(closes, atr, regime_mode)

        if not trend_ok:
            return self._empty()

        # ==========================================================
        # EXHAUSTION (PROFESSIONAL RANGE TUNED)
        # ==========================================================

        exhaustion_score, direction = self._exhaustion(rsi, macd, macd_signal)

        if direction is None:
            return self._empty()

        # ==========================================================
        # STRUCTURE
        # ==========================================================

        structure_score = self._structure(highs, lows, direction)

        # ==========================================================
        # LIQUIDITY
        # ==========================================================

        liquidity_score = self._liquidity(highs, lows, direction)

        # ==========================================================
        # VOLUME
        # ==========================================================

        volume_score = self._volume(volumes)

        # ==========================================================
        # FINAL RATING
        # ==========================================================

        rating = (
            exhaustion_score * 0.35 +
            structure_score * 0.25 +
            liquidity_score * 0.2 +
            volume_score * 0.2
        )

        rating = round(rating, 2)

        return {
            "type": "reversal",
            "direction": direction,
            "rating": rating,
            "breakdown": {
                "exhaustion": exhaustion_score,
                "structure": structure_score,
                "liquidity": liquidity_score,
                "volume": volume_score,
                "regime": regime_mode
            }
        }

    # ==========================================================
    # TREND FILTER
    # ==========================================================

    def _trend_filter(self, closes, atr, regime_mode):

        if regime_mode == "range":
            return True

        move = abs(closes[-1] - closes[-21])
        required_move = atr[-1] * 1.8

        return move >= required_move

    # ==========================================================
    # EXHAUSTION
    # ==========================================================

    def _exhaustion(self, rsi, macd, macd_signal):

        score = 0
        direction = None

        # softer RSI for real market
        if rsi[-1] > 65:
            score += 40
            direction = "short"

        if rsi[-1] < 35:
            score += 40
            direction = "long"

        # MACD divergence signal
        if macd[-1] < macd_signal[-1] and rsi[-1] > 60:
            score += 25
            direction = "short"

        if macd[-1] > macd_signal[-1] and rsi[-1] < 40:
            score += 25
            direction = "long"

        return score, direction

    # ==========================================================
    # STRUCTURE
    # ==========================================================

    def _structure(self, highs, lows, direction):

        recent_high = np.max(highs[-15:])
        recent_low = np.min(lows[-15:])

        if direction == "short":
            if highs[-1] >= recent_high:
                return 80
            return 50

        if direction == "long":
            if lows[-1] <= recent_low:
                return 80
            return 50

        return 0

    # ==========================================================
    # LIQUIDITY
    # ==========================================================

    def _liquidity(self, highs, lows, direction):

        prev_high = np.max(highs[-30:-5])
        prev_low = np.min(lows[-30:-5])

        if direction == "short":
            if highs[-1] > prev_high:
                return 80
            return 40

        if direction == "long":
            if lows[-1] < prev_low:
                return 80
            return 40

        return 0

    # ==========================================================
    # VOLUME
    # ==========================================================

    def _volume(self, volumes):

        avg = np.mean(volumes[-20:])
        current = volumes[-1]

        if current > avg * 1.4:
            return 80

        if current > avg * 1.2:
            return 60

        return 40

    # ==========================================================
    # EMPTY
    # ==========================================================

    def _empty(self):
        return {
            "type": None,
            "direction": None,
            "rating": 0,
            "breakdown": None
        }
