# asset_profile_engine.py
# Automatic Asset Classification Engine

from typing import List, Dict


class AssetProfileEngine:

    def analyze(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> Dict:

        if len(closes) < 50:
            return {
                "asset_class": "mid",
                "volatility_score": 0,
                "range_pct": 0,
            }

        atr_pct = self._calculate_atr_pct(closes, highs, lows)
        avg_range_pct = self._calculate_avg_range_pct(highs, lows)

        asset_class = self._classify(atr_pct, avg_range_pct)

        return {
            "asset_class": asset_class,
            "volatility_score": round(atr_pct, 4),
            "range_pct": round(avg_range_pct, 4),
        }

    # ================================
    # ATR %
    # ================================

    def _calculate_atr_pct(self, closes, highs, lows):

        tr_values = []

        for i in range(-30, -1):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            tr_values.append(tr)

        if not tr_values:
            return 0

        atr = sum(tr_values) / len(tr_values)
        price = closes[-1]

        return (atr / price) * 100

    # ================================
    # Average Range %
    # ================================

    def _calculate_avg_range_pct(self, highs, lows):

        ranges = []

        for i in range(-30, -1):
            r = highs[i] - lows[i]
            ranges.append(r)

        if not ranges:
            return 0

        avg_range = sum(ranges) / len(ranges)
        price = highs[-1]

        return (avg_range / price) * 100

    # ================================
    # Classification
    # ================================

    def _classify(self, atr_pct, range_pct):

        if atr_pct < 1.5 and range_pct < 2:
            return "core"

        if atr_pct > 3 or range_pct > 4:
            return "small"

        return "mid"
