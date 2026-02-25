# elite_htf_sync.py
# V31 ELITE — HTF Synchronization Engine (Directional, Exhaustion-Aware)

from typing import List, Dict, Optional


class EliteHTFSync:
    """
    HTF-анализ без lookahead:
    - для каждого TF берём только свечи с timestamp <= current_timestamp
    - все EMA/strength/exhaustion считаются только по доступной истории
    - добавлены:
        - signed_strength по каждому TF
        - htf_regime (TREND / RANGE / CONTRA)
        - более жёсткая реакция на exhaustion
    """

    def __init__(
        self,
        min_len_15m: int = 200,
        min_len_1h: int = 120,
        min_len_4h: int = 80
    ):
        self.min_len_15m = min_len_15m
        self.min_len_1h = min_len_1h
        self.min_len_4h = min_len_4h

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def analyze(
        self,
        current_timestamp: int,
        tf15: List[Dict],
        tf1h: List[Dict],
        tf4h: List[Dict]
    ) -> Optional[Dict]:

        idx15 = self._last_index_leq(tf15, current_timestamp)
        idx1h = self._last_index_leq(tf1h, current_timestamp)
        idx4h = self._last_index_leq(tf4h, current_timestamp)

        if idx15 is None or idx1h is None or idx4h is None:
            return None

        hist15 = tf15[:idx15 + 1]
        hist1h = tf1h[:idx1h + 1]
        hist4h = tf4h[:idx4h + 1]

        if (
            len(hist15) < self.min_len_15m
            or len(hist1h) < self.min_len_1h
            or len(hist4h) < self.min_len_4h
        ):
            return None

        # --- BIAS ---
        bias15 = self._bias(hist15)
        bias1h = self._bias(hist1h)
        bias4h = self._bias(hist4h)

        # --- STRENGTH (unsigned) ---
        strength15 = self._trend_strength(hist15)
        strength1h = self._trend_strength(hist1h)
        strength4h = self._trend_strength(hist4h)

        # --- SIGNED STRENGTH ---
        s15 = self._signed_strength(bias15, strength15)
        s1h = self._signed_strength(bias1h, strength1h)
        s4h = self._signed_strength(bias4h, strength4h)

        # --- EXHAUSTION ---
        exhausted15 = self._exhaustion(hist15)
        exhausted1h = self._exhaustion(hist1h)
        exhausted4h = self._exhaustion(hist4h)

        bias, alignment = self._combine_bias(bias15, bias1h, bias4h)

        exhausted_any = exhausted15 or exhausted1h or exhausted4h

        # Средняя сила тренда (unsigned)
        avg_strength = (strength15 + strength1h + strength4h) / 3

        # Средняя направленная сила
        avg_signed_strength = (s15 + s1h + s4h) / 3

        # Если есть exhaustion — режем alignment и силу
        if exhausted_any:
            alignment *= 0.6
            avg_strength *= 0.7
            avg_signed_strength *= 0.7

        htf_regime = self._htf_regime(bias, avg_strength, alignment)

        return {
            "bias": bias,
            "alignment_score": alignment,
            "trend_strength": avg_strength,
            "signed_trend_strength": avg_signed_strength,
            "exhausted": exhausted_any,
            "htf_regime": htf_regime,
            "details": {
                "bias_15m": bias15,
                "bias_1h": bias1h,
                "bias_4h": bias4h,
                "strength_15m": strength15,
                "strength_1h": strength1h,
                "strength_4h": strength4h,
                "exhausted_15m": exhausted15,
                "exhausted_1h": exhausted1h,
                "exhausted_4h": exhausted4h,
            }
        }

    # ==========================================================
    # INDEX HELPERS
    # ==========================================================

    def _last_index_leq(self, candles: List[Dict], ts: int) -> Optional[int]:
        last_idx = None
        for i, c in enumerate(candles):
            if c["timestamp"] <= ts:
                last_idx = i
            else:
                break
        return last_idx

    # ==========================================================
    # BIAS DETECTION
    # ==========================================================

    def _bias(self, candles: List[Dict]) -> str:
        closes = [float(c["close"]) for c in candles]
        if len(closes) < 200:
            return "neutral"

        ema50 = self._ema(closes, 50)
        ema200 = self._ema(closes, 200)

        if ema50 > ema200:
            return "bullish"
        elif ema50 < ema200:
            return "bearish"
        else:
            return "neutral"

    # ==========================================================
    # TREND STRENGTH
    # ==========================================================

    def _trend_strength(self, candles: List[Dict]) -> float:
        closes = [float(c["close"]) for c in candles]
        if len(closes) < 60:
            return 0.0

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)

        price = closes[-1]
        if price == 0:
            return 0.0

        return abs(ema20 - ema50) / price

    def _signed_strength(self, bias: str, strength: float) -> float:
        if bias == "bullish":
            return strength
        if bias == "bearish":
            return -strength
        return 0.0

    # ==========================================================
    # EXHAUSTION DETECTOR
    # ==========================================================

    def _exhaustion(self, candles: List[Dict]) -> bool:
        if len(candles) < 40:
            return False

        recent = candles[-20:]

        highs = [float(c["high"]) for c in recent]
        lows = [float(c["low"]) for c in recent]
        closes = [float(c["close"]) for c in recent]

        total_range = max(highs) - min(lows)
        net_move = abs(closes[-1] - closes[0])

        if total_range == 0:
            return False

        efficiency = net_move / total_range

        # low efficiency → потенциальная усталость тренда
        return efficiency < 0.3

    # ==========================================================
    # COMBINE BIAS
    # ==========================================================

    def _combine_bias(self, b15: str, b1h: str, b4h: str):
        bullish = sum([b == "bullish" for b in [b15, b1h, b4h]])
        bearish = sum([b == "bearish" for b in [b15, b1h, b4h]])

        if bullish >= 2:
            return "bullish", bullish / 3

        if bearish >= 2:
            return "bearish", bearish / 3

        return "neutral", 0.33

    # ==========================================================
    # HTF REGIME
    # ==========================================================

    def _htf_regime(self, bias: str, strength: float, alignment: float) -> str:
        """
        Грубая классификация HTF:
        - HTF_TREND: сильная направленная сила + хорошее согласование TF
        - HTF_RANGE: слабая сила, низкий alignment
        - HTF_CONTRA: bias нейтральный или слабый, но strength заметный
        """
        if alignment >= 0.66 and strength > 0.01 and bias in ("bullish", "bearish"):
            return "HTF_TREND"

        if strength < 0.004 or alignment < 0.45:
            return "HTF_RANGE"

        return "HTF_CONTRA"

    # ==========================================================
    # EMA
    # ==========================================================

    def _ema(self, closes, period):
        if len(closes) < period:
            return closes[-1]

        k = 2 / (period + 1)
        ema = closes[0]

        for price in closes:
            ema = price * k + ema * (1 - k)

        return ema
