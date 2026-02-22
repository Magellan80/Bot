# elite_signal_router.py
# V31.1 ELITE — Signal Decision Engine (Path B Institutional Router, Stability Upgrade)

from typing import Optional, Dict


class EliteSignalRouter:

    def __init__(self):
        # минимальные требования для тренда
        self.min_alignment = 0.66
        self.min_trend_conf = 0.45

        # режимы, где реверсалы разрешены
        self.reversal_allowed = ["RANGE", "EXHAUSTION", "LOW_VOL_RANGE", "COMPRESSION"]

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def route(
        self,
        trend_signal: Optional[Dict],
        reversal_signal: Optional[Dict],
        regime: Dict,
        htf: Dict
    ) -> Optional[Dict]:

        regime_type = regime["regime"]
        htf_regime = htf.get("htf_regime", "HTF_RANGE")
        alignment = htf.get("alignment_score", 0)
        exhausted = htf.get("exhausted", False)
        trend_conf = regime.get("trend_confidence", 0)

        # ------------------------------------------------------
        # 0. Нет сигналов
        # ------------------------------------------------------
        if not trend_signal and not reversal_signal:
            return None

        # ------------------------------------------------------
        # 1. Только один сигнал
        # ------------------------------------------------------
        if trend_signal and not reversal_signal:
            return self._validate_trend(trend_signal, regime_type, htf_regime, alignment, trend_conf)

        if reversal_signal and not trend_signal:
            return self._validate_reversal(reversal_signal, regime_type, exhausted)

        # ------------------------------------------------------
        # 2. Оба сигнала есть → решаем по режиму
        # ------------------------------------------------------
        return self._resolve_conflict(
            trend_signal,
            reversal_signal,
            regime_type,
            htf_regime,
            alignment,
            exhausted,
            trend_conf
        )

    # ==========================================================
    # VALIDATION HELPERS
    # ==========================================================

    def _validate_trend(self, trend_signal, regime_type, htf_regime, alignment, trend_conf):

        # Тренд разрешён только если HTF в тренде
        if htf_regime != "HTF_TREND":
            return None

        # Требуем сильное согласование TF
        if alignment < self.min_alignment:
            return None

        # Требуем уверенность в тренде на младшем ТФ
        if trend_conf < self.min_trend_conf:
            return None

        # Запрещаем тренд в CHAOS / COMPRESSION
        if regime_type in ["CHAOS", "COMPRESSION", "LOW_VOL_RANGE"]:
            return None

        return trend_signal

    def _validate_reversal(self, reversal_signal, regime_type, exhausted):

        # Реверсалы разрешены только в мягких режимах
        if regime_type not in self.reversal_allowed:
            return None

        # Если HTF exhausted → реверсал усиливается
        if exhausted:
            return reversal_signal

        return reversal_signal

    # ==========================================================
    # CONFLICT RESOLUTION (Path B)
    # ==========================================================

    def _resolve_conflict(
        self,
        trend_signal: Dict,
        reversal_signal: Dict,
        regime_type: str,
        htf_regime: str,
        alignment: float,
        exhausted: bool,
        trend_conf: float
    ) -> Optional[Dict]:

        # ------------------------------------------------------
        # 1. CHAOS → почти всегда реверсал
        # ------------------------------------------------------
        if regime_type == "CHAOS":
            return reversal_signal

        # ------------------------------------------------------
        # 2. STRONG_TREND → тренд, но только если HTF подтверждает
        # ------------------------------------------------------
        if regime_type == "STRONG_TREND":
            if htf_regime == "HTF_TREND" and alignment >= self.min_alignment and trend_conf >= self.min_trend_conf:
                return trend_signal
            return None

        # ------------------------------------------------------
        # 3. RANGE → реверсал
        # ------------------------------------------------------
        if regime_type in ["RANGE", "LOW_VOL_RANGE"]:
            return reversal_signal

        # ------------------------------------------------------
        # 4. EXHAUSTION → реверсал
        # ------------------------------------------------------
        if regime_type == "EXHAUSTION":
            return reversal_signal

        # ------------------------------------------------------
        # 5. COMPRESSION → реверсал предпочтительнее
        # ------------------------------------------------------
        if regime_type == "COMPRESSION":
            return reversal_signal

        # ------------------------------------------------------
        # 6. EXPANSION → тренд, если HTF силён
        # ------------------------------------------------------
        if regime_type == "EXPANSION":
            if htf_regime == "HTF_TREND" and alignment >= self.min_alignment and trend_conf >= 0.50:
                return trend_signal
            if exhausted:
                return reversal_signal
            return self._higher_quality(trend_signal, reversal_signal)

        # ------------------------------------------------------
        # 7. EARLY_TREND → тренд предпочтительнее, но сравниваем качество
        # ------------------------------------------------------
        if regime_type == "EARLY_TREND":
            if htf_regime == "HTF_TREND" and alignment >= self.min_alignment:
                return trend_signal
            return self._higher_quality(trend_signal, reversal_signal)

        # ------------------------------------------------------
        # 8. fallback → выбираем лучший по качеству
        # ------------------------------------------------------
        return self._higher_quality(trend_signal, reversal_signal)

    # ==========================================================
    # QUALITY COMPARISON
    # ==========================================================

    def _higher_quality(self, s1, s2):

        if s1["quality"] > s2["quality"]:
            return s1

        if s2["quality"] > s1["quality"]:
            return s2

        return None
