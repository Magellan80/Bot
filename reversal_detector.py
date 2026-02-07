# ============================================
#   reversal_detector.py — Reversal Engine v4.1 EXTENDED
#   Автор: Ярослав + Кай
# ============================================

from typing import List, Dict, Optional


class ReversalDetector:
    """
    ReversalDetector v4.1 EXTENDED

    Задача:
      - Детектировать развороты Pump → Dump и Dump → Pump
      - Оценивать силу разворота (0..100)
      - Учитывать:
          * тело свечи
          * тени
          * локальный и среднесрочный тренд
          * объём
          * положение в диапазоне
          * подтверждение следующими свечами
    """

    # ----------------------------------------
    #   Публичный метод: базовый разворот
    # ----------------------------------------
    def analyze(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> Dict[str, Optional[float]]:
        """
        Базовый детектор разворота.

        Возвращает:
            {
                "reversal": "bullish" | "bearish" | None,
                "rating": int
            }
        """
        n = len(closes)
        if n < 7:
            return {"reversal": None, "rating": 0}

        body, upper_wick, lower_wick = self._body_and_wicks(closes, highs, lows)
        c0, c1 = closes[0], closes[1]
        v0, v1 = volumes[0], volumes[1]

        # ----------------------------------------
        #   Базовые признаки
        # ----------------------------------------
        prior_trend = self._trend_strength(closes, lookback=7)
        mid_trend = self._trend_strength(closes, lookback=20) if n >= 20 else 0.0
        vol_factor_short = self._volume_factor(volumes, short=5, long=20)
        range_pos = self._range_position(closes, highs, lows)

        # ----------------------------------------
        #   Bullish reversal (Dump → Pump)
        # ----------------------------------------
        bullish_core = (
            lower_wick > body * 1.5 and
            c0 > c1 and
            prior_trend < -2.0 and
            self._volume_ok(v0, v1)
        )

        bullish_extended = (
            bullish_core or
            self._bullish_engulfing(closes, highs, lows) or
            self._bullish_double_tail(closes, highs, lows)
        )

        # ----------------------------------------
        #   Bearish reversal (Pump → Dump)
        # ----------------------------------------
        bearish_core = (
            upper_wick > body * 1.5 and
            c0 < c1 and
            prior_trend > 2.0 and
            self._volume_ok(v0, v1)
        )

        bearish_extended = (
            bearish_core or
            self._bearish_engulfing(closes, highs, lows) or
            self._bearish_double_tail(closes, highs, lows)
        )

        if not bullish_extended and not bearish_extended:
            return {"reversal": None, "rating": 0}

        # ----------------------------------------
        #   Рейтинг по компонентам
        # ----------------------------------------
        wick_strength = self._wick_strength(upper_wick, lower_wick, body)
        body_strength = self._body_strength(body, highs, lows)
        trend_strength = self._trend_component(prior_trend, mid_trend)
        volume_strength = self._volume_component(vol_factor_short)
        range_strength = self._range_component(range_pos)
        confirm_strength = self._confirmation_strength(closes, highs, lows)

        base_rating = (
            wick_strength * 0.30 +
            body_strength * 0.15 +
            trend_strength * 0.20 +
            volume_strength * 0.15 +
            range_strength * 0.10 +
            confirm_strength * 0.10
        )

        # ----------------------------------------
        #   Финальный выбор направления
        # ----------------------------------------
        if bullish_extended and not bearish_extended:
            final_rating = self._clamp(base_rating)
            return {"reversal": "bullish", "rating": final_rating}

        if bearish_extended and not bullish_extended:
            final_rating = self._clamp(base_rating)
            return {"reversal": "bearish", "rating": final_rating}

        # Если оба сработали (редко, но возможно) — выбираем по тренду
        if bullish_extended and bearish_extended:
            if prior_trend < 0:
                final_rating = self._clamp(base_rating)
                return {"reversal": "bullish", "rating": final_rating}
            else:
                final_rating = self._clamp(base_rating)
                return {"reversal": "bearish", "rating": final_rating}

        return {"reversal": None, "rating": 0}

    # ----------------------------------------
    #   Публичный метод: HABR (High Accuracy Break Reversal)
    # ----------------------------------------
    def analyze_habr(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> Dict[str, Optional[float]]:
        """
        HABR v3.1 EXTENDED — детектор разворотов на пробое.

        Возвращает:
            {
                "direction": "bullish" | "bearish" | None,
                "rating": int
            }
        """
        n = len(closes)
        if n < 6:
            return {"direction": None, "rating": 0}

        c0, c1, c2 = closes[0], closes[1], closes[2]
        h0, h1 = highs[0], highs[1]
        l0, l1 = lows[0], lows[1]
        v0, v1 = volumes[0], volumes[1]

        prior_trend = self._trend_strength(closes, lookback=10) if n >= 10 else 0.0
        vol_factor = self._volume_factor(volumes, short=5, long=20)

        # Bullish HABR (Dump → Pump)
        bullish = (
            c0 > c1 and
            c1 < c2 and
            l0 < l1 and
            v0 >= v1 * 0.7 and
            prior_trend < -1.5
        )

        # Bearish HABR (Pump → Dump)
        bearish = (
            c0 < c1 and
            c1 > c2 and
            h0 > h1 and
            v0 >= v1 * 0.7 and
            prior_trend > 1.5
        )

        if not bullish and not bearish:
            return {"direction": None, "rating": 0}

        base_rating = (
            abs(c0 - c1) * 4.0 +
            abs(c1 - c2) * 2.0 +
            (v0 / max(v1, 1e-7)) * 3.0 +
            abs(prior_trend) * 1.5 +
            vol_factor * 2.0
        )

        rating = self._clamp(base_rating)

        if bullish and not bearish:
            return {"direction": "bullish", "rating": rating}
        if bearish and not bullish:
            return {"direction": "bearish", "rating": rating}

        # Если оба — выбираем по тренду
        if bullish and bearish:
            if prior_trend < 0:
                return {"direction": "bullish", "rating": rating}
            else:
                return {"direction": "bearish", "rating": rating}

        return {"direction": None, "rating": 0}

    # ============================================
    #   INTERNAL HELPERS — базовые
    # ============================================

    def _body_and_wicks(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ):
        c0, c1 = closes[0], closes[1]
        h0, l0 = highs[0], lows[0]
        body = abs(c0 - c1)
        upper = h0 - max(c0, c1)
        lower = min(c0, c1) - l0
        return body, upper, lower

    def _trend_strength(self, closes: List[float], lookback: int = 7) -> float:
        """Сила локального тренда в % за lookback свечей."""
        if len(closes) <= lookback:
            return 0.0
        c_now = closes[1]
        c_past = closes[lookback]
        if c_past == 0:
            return 0.0
        return (c_now - c_past) / c_past * 100.0

    def _volume_factor(
        self,
        volumes: List[float],
        short: int = 5,
        long: int = 20
    ) -> float:
        """Отношение среднего короткого объёма к длинному."""
        if len(volumes) < long + 1:
            return 1.0
        short_avg = sum(volumes[0:short]) / max(short, 1)
        long_avg = sum(volumes[0:long]) / max(long, 1)
        if long_avg <= 0:
            return 1.0
        return short_avg / long_avg

    def _range_position(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        lookback: int = 30
    ) -> float:
        """
        Позиция текущей цены в диапазоне за lookback:
        0.0 — у минимума, 1.0 — у максимума.
        """
        n = len(closes)
        if n < 3:
            return 0.5
        lb = min(lookback, n)
        recent_high = max(highs[0:lb])
        recent_low = min(lows[0:lb])
        c0 = closes[0]
        rng = recent_high - recent_low
        if rng <= 0:
            return 0.5
        return (c0 - recent_low) / rng

    def _volume_ok(self, v0: float, v1: float) -> bool:
        """Объём текущей свечи не должен быть сильно слабее предыдущего."""
        return v0 >= v1 * 0.7

    def _clamp(self, x: float) -> int:
        """Нормировка рейтинга в диапазон 0..100."""
        return max(0, min(100, int(x)))

    # ============================================
    #   INTERNAL HELPERS — паттерны
    # ============================================

    def _bullish_engulfing(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> bool:
        """
        Bullish engulfing:
          - предыдущая свеча медвежья
          - текущая бычья
          - тело текущей перекрывает тело предыдущей
        """
        if len(closes) < 3:
            return False
        c0, c1, c2 = closes[0], closes[1], closes[2]
        # предыдущая медвежья, текущая бычья
        prev_bear = c1 < c2
        curr_bull = c0 > c1
        if not (prev_bear and curr_bull):
            return False
        body_curr = abs(c0 - c1)
        body_prev = abs(c1 - c2)
        return body_curr > body_prev * 1.1

    def _bearish_engulfing(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> bool:
        """
        Bearish engulfing:
          - предыдущая свеча бычья
          - текущая медвежья
          - тело текущей перекрывает тело предыдущей
        """
        if len(closes) < 3:
            return False
        c0, c1, c2 = closes[0], closes[1], closes[2]
        prev_bull = c1 > c2
        curr_bear = c0 < c1
        if not (prev_bull and curr_bear):
            return False
        body_curr = abs(c0 - c1)
        body_prev = abs(c1 - c2)
        return body_curr > body_prev * 1.1

    def _bullish_double_tail(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> bool:
        """
        Двойной нижний хвост (двойное дно локально).
        """
        if len(closes) < 4:
            return False
        body0, _, low0 = self._body_and_wicks(closes[0:], highs[0:], lows[0:])
        body1, _, low1 = self._body_and_wicks(closes[1:], highs[1:], lows[1:])
        cond0 = low0 > body0 * 1.2
        cond1 = low1 > body1 * 1.2
        return cond0 and cond1

    def _bearish_double_tail(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> bool:
        """
        Двойной верхний хвост (двойная вершина локально).
        """
        if len(closes) < 4:
            return False
        body0, up0, _ = self._body_and_wicks(closes[0:], highs[0:], lows[0:])
        body1, up1, _ = self._body_and_wicks(closes[1:], highs[1:], lows[1:])
        cond0 = up0 > body0 * 1.2
        cond1 = up1 > body1 * 1.2
        return cond0 and cond1

    # ============================================
    #   INTERNAL HELPERS — компоненты рейтинга
    # ============================================

    def _wick_strength(
        self,
        upper_wick: float,
        lower_wick: float,
        body: float
    ) -> float:
        """Сила хвостов относительно тела."""
        if body <= 0:
            return 0.0
        ratio_up = upper_wick / body
        ratio_low = lower_wick / body
        # Берём максимум — сильный хвост в одну сторону
        raw = max(ratio_up, ratio_low)
        return max(0.0, min(100.0, raw * 20.0))

    def _body_strength(
        self,
        body: float,
        highs: List[float],
        lows: List[float],
    ) -> float:
        """Сила тела относительно диапазона свечи."""
        if len(highs) < 1 or len(lows) < 1:
            return 0.0
        rng = highs[0] - lows[0]
        if rng <= 0:
            return 0.0
        ratio = body / rng
        return max(0.0, min(100.0, ratio * 100.0))

    def _trend_component(
        self,
        prior_trend: float,
        mid_trend: float
    ) -> float:
        """
        Компонент тренда:
          - сильный тренд перед разворотом усиливает сигнал
          - слабый тренд → слабый разворот
        """
        score = 0.0
        score += min(abs(prior_trend), 10.0) * 4.0
        score += min(abs(mid_trend), 10.0) * 2.0
        return max(0.0, min(100.0, score))

    def _volume_component(
        self,
        vol_factor: float
    ) -> float:
        """
        Компонент объёма:
          - vol_factor ~ 1.0 → норм
          - > 1.5 → сильный всплеск
        """
        if vol_factor <= 0:
            return 0.0
        return max(0.0, min(100.0, (vol_factor - 0.5) * 50.0))

    def _range_component(
        self,
        range_pos: float
    ) -> float:
        """
        Позиция в диапазоне:
          - разворот у границ диапазона → сильнее
          - в середине → слабее
        """
        # ближе к 0 или 1 → сильнее
        dist_edge = max(0.0, 0.5 - abs(range_pos - 0.5)) * 2.0
        # dist_edge: 0..1, где 1 — у границы
        return max(0.0, min(100.0, dist_edge * 100.0))

    def _confirmation_strength(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> float:
        """
        Подтверждение разворота следующими свечами:
          - если после сигнальной свечи цена идёт в сторону разворота → +очки
        """
        if len(closes) < 4:
            return 0.0
        c0, c1, c2, c3 = closes[0], closes[1], closes[2], closes[3]
        move1 = c0 - c1
        move2 = c1 - c2
        move3 = c2 - c3
        same_dir = 0
        if move1 * move2 > 0:
            same_dir += 1
        if move2 * move3 > 0:
            same_dir += 1
        return same_dir * 30.0
