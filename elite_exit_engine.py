# elite_exit_engine.py
# V31.2 ELITE — Adaptive Exit Engine (Path B: Stable, Low-Noise, Regime-Aware)

from typing import Dict, Optional


class EliteExitEngine:

    def __init__(self):
        # базовые параметры под путь B
        self.time_exit_range = 25        # быстрее выходим в рейндже
        self.time_exit_trend = 60        # дольше держим в тренде

        # частичные фиксации
        self.partial_r = 1.0             # первая частичная фиксация
        self.second_partial_r = 2.2      # в тренде можно брать вторую частичку

        # минимальный ATR для трейлинга (защита от микродвижений)
        self.min_atr = 1e-8

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def manage_position(
        self,
        position: Dict,
        current_price: float,
        atr: float,
        regime: str,
        bar_index: int
    ) -> Dict:

        if position["status"] == "CLOSED":
            return position

        entry = position["entry"]
        sl = position["sl"]
        direction = position["direction"]

        # защита от нулевого ATR
        atr = max(atr, self.min_atr)

        risk = abs(entry - sl)
        if risk <= 0:
            return position

        r_multiple = self._r_multiple(direction, entry, sl, current_price)

        # ------------------------------------------------------
        # TIME EXIT (адаптивный)
        # ------------------------------------------------------
        if self._time_exit_trigger(position, bar_index, regime):
            position["status"] = "CLOSED"
            position["exit_price"] = current_price
            position["reason"] = "time_exit"
            return position

        # ------------------------------------------------------
        # STOP LOSS HIT
        # ------------------------------------------------------
        if self._stop_hit(direction, current_price, sl):
            position["status"] = "CLOSED"
            position["exit_price"] = sl
            position["reason"] = "stop_loss"
            return position

        # ------------------------------------------------------
        # PARTIAL @ 1R
        # ------------------------------------------------------
        if not position["partial_taken"] and r_multiple >= self.partial_r:

            position["partial_taken"] = True
            position["partial_price"] = current_price

            # Перевод в BE + небольшой буфер
            buffer = 0.05 * atr

            if direction == "long":
                position["sl"] = entry + buffer
            else:
                position["sl"] = entry - buffer

        # ------------------------------------------------------
        # SECOND PARTIAL @ 2.2R (только в тренде)
        # ------------------------------------------------------
        if position["partial_taken"] and not position.get("second_partial"):

            if regime in ["STRONG_TREND", "EXPANSION"] and r_multiple >= self.second_partial_r:
                position["second_partial"] = True
                position["second_partial_price"] = current_price

                # подтягиваем SL ближе, но не агрессивно
                if direction == "long":
                    position["sl"] = max(position["sl"], entry + 0.6 * risk)
                else:
                    position["sl"] = min(position["sl"], entry - 0.6 * risk)

        # ------------------------------------------------------
        # TRAILING LOGIC (адаптивный)
        # ------------------------------------------------------
        if position["partial_taken"]:
            new_sl = self._adaptive_trailing(
                direction,
                current_price,
                atr,
                regime,
                position
            )

            if direction == "long":
                position["sl"] = max(position["sl"], new_sl)
            else:
                position["sl"] = min(position["sl"], new_sl)

        return position

    # ==========================================================
    # TIME EXIT LOGIC
    # ==========================================================

    def _time_exit_trigger(self, position, bar_index, regime):

        bars_in_trade = bar_index - position["open_bar"]

        # В рейндже — выходим быстрее
        if regime in ["RANGE", "LOW_VOL_RANGE", "COMPRESSION"]:
            return bars_in_trade >= self.time_exit_range

        # В тренде — держим дольше
        return bars_in_trade >= self.time_exit_trend

    # ==========================================================
    # R MULTIPLE
    # ==========================================================

    def _r_multiple(self, direction, entry, sl, price):

        risk = abs(entry - sl)
        if risk == 0:
            return 0

        if direction == "long":
            return (price - entry) / risk
        else:
            return (entry - price) / risk

    # ==========================================================
    # STOP HIT CHECK
    # ==========================================================

    def _stop_hit(self, direction, price, sl):

        if direction == "long":
            return price <= sl
        else:
            return price >= sl

    # ==========================================================
    # ADAPTIVE TRAILING STOP
    # ==========================================================

    def _adaptive_trailing(self, direction, price, atr, regime, position):

        # Базовый ATR-трейл
        if regime == "STRONG_TREND":
            trail = 0.55 * atr
        elif regime == "EXPANSION":
            trail = 0.65 * atr
        elif regime == "EARLY_TREND":
            trail = 0.75 * atr
        else:
            trail = 1.1 * atr  # в рейндже — шире, чтобы не выбивало

        # Если есть вторая частичка — трейлим агрессивнее
        if position.get("second_partial"):
            trail *= 0.75

        # Swing-трейл (дополнительная защита)
        swing_offset = 0.4 * atr

        if direction == "long":
            return min(price - trail, price - swing_offset)
        else:
            return max(price + trail, price + swing_offset)
