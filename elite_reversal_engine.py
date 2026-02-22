# elite_reversal_engine.py
# V31.1 ELITE — Reversal Engine (Path B: High-Quality, Low-Noise, Stability Upgrade)

from typing import Dict, Optional


class EliteReversalEngine:

    def __init__(self):
        # реверсалы должны быть редкими и очень качественными
        self.min_quality = 0.78

        # дополнительные фильтры стабильности
        self.min_clarity = 0.55
        self.min_impulse = 0.30
        self.min_atr_percentile = 0.55

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def evaluate(self, structure: Dict, regime: Dict, htf: Dict):

        if not structure or not regime or not htf:
            return None

        direction = self._direction(structure, regime, htf)
        if not direction:
            return None

        # предварительные фильтры качества
        if structure["clarity_index"] < self.min_clarity:
            return None

        if structure["impulse_strength"] < self.min_impulse:
            return None

        if regime["atr_percentile"] < self.min_atr_percentile and not htf["exhausted"]:
            return None

        quality = self._quality_score(structure, regime, htf)

        if quality < self.min_quality:
            return None

        return {
            "signal": direction,
            "quality": quality,
            "type": "reversal"
        }

    # ==========================================================
    # DIRECTION LOGIC (Path B)
    # ==========================================================

    def _direction(self, structure, regime, htf):

        regime_name = regime["regime"]
        htf_bias = htf["bias"]
        htf_regime = htf.get("htf_regime", "HTF_RANGE")
        signed_strength = htf.get("signed_trend_strength", 0)

        # -----------------------------------------
        # 1. Запрещаем реверсалы в сильном тренде
        # -----------------------------------------
        if regime_name in ["STRONG_TREND", "EXPANSION"] and abs(signed_strength) > 0.012:
            return None

        # -----------------------------------------
        # 2. Разрешаем реверсалы только в:
        #    RANGE / EXHAUSTION / LOW_VOL_RANGE / COMPRESSION
        # -----------------------------------------
        allowed = ["RANGE", "EXHAUSTION", "LOW_VOL_RANGE", "COMPRESSION"]
        if regime_name not in allowed:
            return None

        # -----------------------------------------
        # 3. Требуем HTF exhaustion или высокий ATR-percentile
        # -----------------------------------------
        if not (htf["exhausted"] or regime["atr_percentile"] > 0.7):
            return None

        # -----------------------------------------
        # 4. Направление реверсала
        # -----------------------------------------
        # Лонг: локальная структура медвежья, HTF не медвежий
        if structure["structure"] == "bearish" and htf_bias != "bearish":
            return "long"

        # Шорт: локальная структура бычья, HTF не бычий
        if structure["structure"] == "bullish" and htf_bias != "bullish":
            return "short"

        return None

    # ==========================================================
    # QUALITY SCORE (Path B)
    # ==========================================================

    def _quality_score(self, structure, regime, htf):

        score = 0.0

        # -----------------------------------------
        # 1. Чистота структуры (важно для реверсала)
        # -----------------------------------------
        score += structure["clarity_index"] * 0.30

        # -----------------------------------------
        # 2. Импульс (сильный импульс → сильный реверс)
        # -----------------------------------------
        score += min(structure["impulse_strength"] / 2, 1) * 0.25

        # -----------------------------------------
        # 3. ATR-контекст (высокая вола → сильнее реверс)
        # -----------------------------------------
        score += regime["atr_percentile"] * 0.20

        # -----------------------------------------
        # 4. EXHAUSTION (режим рынка)
        # -----------------------------------------
        if regime["regime"] == "EXHAUSTION":
            score += 0.15

        # -----------------------------------------
        # 5. HTF exhaustion
        # -----------------------------------------
        if htf["exhausted"]:
            score += 0.10

        # -----------------------------------------
        # 6. HTF alignment (низкий alignment → выше шанс реверсала)
        # -----------------------------------------
        alignment = htf["alignment_score"]
        if alignment < 0.45:
            score += 0.10
        elif alignment < 0.60:
            score += 0.05

        # -----------------------------------------
        # 7. Анти‑шумовой штраф (новое)
        # -----------------------------------------
        noise_penalty = max(0.0, 0.20 - structure["clarity_index"])
        score -= noise_penalty * 0.25

        return max(0.0, min(score, 1.0))
