# elite_trend_engine.py
# V31.1 ELITE — Trend Engine (Path B: High-Precision Trend Activation, Stability Upgrade)

from typing import Dict, Optional


class EliteTrendEngine:

    def __init__(self):
        # базовый порог качества
        self.min_quality = 0.62

        # дополнительные фильтры стабильности
        self.min_alignment = 0.66
        self.min_trend_conf = 0.45
        self.min_clarity = 0.55
        self.min_impulse = 0.35
        self.min_pullback = 0.40

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

        if structure["pullback_quality"] < self.min_pullback:
            return None

        quality = self._quality_score(structure, regime, htf)

        if quality < self.min_quality:
            return None

        return {
            "signal": direction,
            "quality": quality,
            "type": "trend"
        }

    # ==========================================================
    # TREND DIRECTION (Path B)
    # ==========================================================

    def _direction(self, structure, regime, htf):

        regime_name = regime["regime"]
        htf_bias = htf["bias"]
        htf_regime = htf.get("htf_regime", "HTF_RANGE")
        alignment = htf["alignment_score"]
        trend_conf = regime.get("trend_confidence", 0)

        # ------------------------------------------------------
        # 1. HTF должен быть в тренде
        # ------------------------------------------------------
        if htf_regime != "HTF_TREND":
            return None

        # ------------------------------------------------------
        # 2. Сильное согласование HTF
        # ------------------------------------------------------
        if alignment < self.min_alignment:
            return None

        # ------------------------------------------------------
        # 3. Уверенность тренда на младшем ТФ
        # ------------------------------------------------------
        if trend_conf < self.min_trend_conf:
            return None

        # ------------------------------------------------------
        # 4. Запрещаем трендовые сигналы в CHAOS / COMPRESSION
        # ------------------------------------------------------
        if regime_name in ["CHAOS", "COMPRESSION", "LOW_VOL_RANGE"]:
            return None

        # ------------------------------------------------------
        # 5. Направление тренда = структура + HTF bias
        # ------------------------------------------------------
        if structure["structure"] == "bullish" and htf_bias == "bullish":
            return "long"

        if structure["structure"] == "bearish" and htf_bias == "bearish":
            return "short"

        return None

    # ==========================================================
    # QUALITY SCORE (Path B)
    # ==========================================================

    def _quality_score(self, structure, regime, htf):

        score = 0.0

        # ------------------------------------------------------
        # 1. Чистота структуры
        # ------------------------------------------------------
        score += structure["clarity_index"] * 0.28

        # ------------------------------------------------------
        # 2. Импульс
        # ------------------------------------------------------
        score += min(structure["impulse_strength"] / 2, 1) * 0.22

        # ------------------------------------------------------
        # 3. Качество отката
        # ------------------------------------------------------
        score += structure["pullback_quality"] * 0.22

        # ------------------------------------------------------
        # 4. HTF alignment
        # ------------------------------------------------------
        score += htf["alignment_score"] * 0.18

        # ------------------------------------------------------
        # 5. Режим рынка: STRONG_TREND усиливает сигнал
        # ------------------------------------------------------
        if regime["regime"] == "STRONG_TREND":
            score += 0.10

        # ------------------------------------------------------
        # 6. Доп. бонус за направленную силу HTF
        # ------------------------------------------------------
        signed_strength = abs(htf.get("signed_trend_strength", 0))
        if signed_strength > 0.012:
            score += 0.10
        elif signed_strength > 0.008:
            score += 0.05

        # ------------------------------------------------------
        # 7. Анти‑шумовой штраф (новое)
        # ------------------------------------------------------
        # если структура слишком "грязная", слегка штрафуем
        noise_penalty = max(0.0, 0.15 - structure["clarity_index"])
        score -= noise_penalty * 0.25

        return max(0.0, min(score, 1.0))
