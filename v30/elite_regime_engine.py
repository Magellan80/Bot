# elite_regime_engine.py
# V31.1 ELITE — Regime State Machine (5m, Structured + Chaos Filter, Stability Upgrade)

from typing import List, Dict, Optional
import math


class EliteRegimeEngine:

    def __init__(self):
        # минимальные требования
        self.min_candles = 250

        # пороги для классификации
        self.low_vol_threshold = 0.25
        self.high_vol_threshold = 0.75

        self.strong_slope = 0.0016
        self.early_slope = 0.0009
        self.range_slope = 0.0005

        self.strong_separation = 0.003
        self.early_separation = 0.002
        self.range_separation = 0.0015

        self.compression_threshold = 0.7

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def detect(self, candles: List[Dict]) -> Optional[Dict]:

        if len(candles) < self.min_candles:
            return None

        atr = self._atr(candles, 14)
        if not atr or atr <= 0:
            return None

        price = float(candles[-1]["close"])
        if price <= 0:
            return None

        atr_percentile = self._atr_percentile(candles, 200)

        ema50 = self._ema(candles, 50)
        ema200 = self._ema(candles, 200)

        slope = self._ema_slope(candles, 50)
        slope_norm = slope / price

        separation = abs(ema50 - ema200) / price

        compression_ratio = self._range_compression(candles, atr)

        impulse_freq = self._impulse_frequency(candles, atr)

        regime = self._classify(
            atr_percentile,
            slope_norm,
            separation,
            compression_ratio,
            impulse_freq
        )

        trend_confidence = self._trend_confidence(
            slope_norm,
            separation,
            impulse_freq
        )

        return {
            "regime": regime,
            "atr_percentile": atr_percentile,
            "slope_norm": slope_norm,
            "separation": separation,
            "compression": compression_ratio,
            "impulse_freq": impulse_freq,
            "trend_confidence": trend_confidence
        }

    # ==========================================================
    # REGIME CLASSIFICATION
    # ==========================================================

    def _classify(
        self,
        atr_pct: float,
        slope_norm: float,
        separation: float,
        compression: float,
        impulse_freq: float
    ) -> str:

        abs_slope = abs(slope_norm)

        # ------------------------------------------------------
        # LOW VOLATILITY ZONE
        # ------------------------------------------------------
        if atr_pct < self.low_vol_threshold:

            # сильная компрессия → COMPRESSION
            if compression > self.compression_threshold:
                return "COMPRESSION"

            # низкая вола + нет тренда → LOW_VOL_RANGE
            if abs_slope < self.range_slope and separation < self.range_separation:
                return "LOW_VOL_RANGE"

        # ------------------------------------------------------
        # STRONG TREND
        # ------------------------------------------------------
        if abs_slope > self.strong_slope and separation > self.strong_separation:
            if impulse_freq > 0.25:
                return "STRONG_TREND"

        # ------------------------------------------------------
        # EARLY TREND
        # ------------------------------------------------------
        if abs_slope > self.early_slope and separation > self.early_separation:
            return "EARLY_TREND"

        # ------------------------------------------------------
        # HIGH VOLATILITY ZONE
        # ------------------------------------------------------
        if atr_pct > self.high_vol_threshold:

            # хаос: много импульсов, но слабый тренд
            if impulse_freq > 0.6 and abs_slope < self.early_slope:
                return "CHAOS"

            # расширение: высокая вола + тренд + импульсы
            if impulse_freq > 0.4 and abs_slope >= self.early_slope:
                return "EXPANSION"

        # ------------------------------------------------------
        # RANGE (fallback)
        # ------------------------------------------------------
        if abs_slope < self.range_slope and separation < self.range_separation:
            return "RANGE"

        return "RANGE"

    # ==========================================================
    # TREND CONFIDENCE
    # ==========================================================

    def _trend_confidence(
        self,
        slope_norm: float,
        separation: float,
        impulse_freq: float
    ) -> float:

        abs_slope = abs(slope_norm)

        slope_score = min(abs_slope / 0.002, 1.0)
        sep_score = min(separation / 0.004, 1.0)
        impulse_score = min(impulse_freq / 0.5, 1.0)

        return max(0.0, min(1.0, slope_score * 0.4 + sep_score * 0.4 + impulse_score * 0.2))

    # ==========================================================
    # ATR PERCENTILE
    # ==========================================================

    def _atr_percentile(self, candles, window):

        atr_values = []

        for i in range(window, len(candles)):
            atr = self._atr(candles[:i], 14)
            if atr:
                atr_values.append(atr)

        if len(atr_values) < 20:
            return 0.5

        current_atr = atr_values[-1]
        below = sum(1 for v in atr_values if v <= current_atr)

        return below / len(atr_values)

    # ==========================================================
    # EMA SLOPE
    # ==========================================================

    def _ema_slope(self, candles, period):

        if len(candles) < period + 10:
            return 0.0

        ema_now = self._ema(candles, period)
        ema_prev = self._ema(candles[:-10], period)

        return ema_now - ema_prev

    # ==========================================================
    # RANGE COMPRESSION
    # ==========================================================

    def _range_compression(self, candles, atr):

        recent = candles[-30:]

        highs = [float(c["high"]) for c in recent]
        lows = [float(c["low"]) for c in recent]

        total_range = max(highs) - min(lows)

        if atr <= 0:
            return 0.0

        ratio = total_range / (atr * 10)
        compression = 1 - min(ratio, 1)

        return max(0.0, min(compression, 1.0))

    # ==========================================================
    # IMPULSE FREQUENCY
    # ==========================================================

    def _impulse_frequency(self, candles, atr):

        recent = candles[-40:]
        if atr <= 0:
            return 0.0

        impulses = 0

        for i in range(1, len(recent)):
            move = abs(float(recent[i]["close"]) - float(recent[i - 1]["close"]))
            if move > 0.8 * atr:
                impulses += 1

        return impulses / len(recent)

    # ==========================================================
    # EMA
    # ==========================================================

    def _ema(self, candles, period):

        closes = [float(c["close"]) for c in candles]

        if len(closes) < period:
            return closes[-1]

        k = 2 / (period + 1)
        ema = closes[0]

        for price in closes:
            ema = price * k + ema * (1 - k)

        return ema

    # ==========================================================
    # ATR
    # ==========================================================

    def _atr(self, candles, period):

        trs = []

        for i in range(1, len(candles)):
            high = float(candles[i]["high"])
            low = float(candles[i]["low"])
            prev = float(candles[i - 1]["close"])
            tr = max(high - low, abs(high - prev), abs(low - prev))
            trs.append(tr)

        if len(trs) < period:
            return None

        return sum(trs[-period:]) / period
