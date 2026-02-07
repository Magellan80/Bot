# detectors.py

class SimpleReversalDetector:
    """
    Простой и надёжный реверсал:
    - Dump → Pump: падали → сильная зелёная свеча + объём
    - Pump → Dump: росли → сильная красная свеча + объём
    """

    def analyze(self, closes, highs, lows, volumes):
        if len(closes) < 4:
            return {"reversal": None, "rating": 0}

        c0, c1, c2, c3 = closes[0], closes[1], closes[2], closes[3]
        v0, v1, v2 = volumes[0], volumes[1], volumes[2]

        rating = 0
        direction = None

        # === Dump → Pump ===
        dump_before = c3 > c2 > c1
        strong_green = c0 > c1 and (c0 - c1) / max(c1, 1e-7) * 100 >= 1.0
        vol_spike = v0 > (v1 + v2) / 2 * 1.5

        if dump_before and strong_green and vol_spike:
            direction = "bullish"
            rating = 60

        # === Pump → Dump ===
        pump_before = c3 < c2 < c1
        strong_red = c0 < c1 and (c1 - c0) / max(c1, 1e-7) * 100 >= 1.0
        vol_spike_red = v0 > (v1 + v2) / 2 * 1.5

        if pump_before and strong_red and vol_spike_red:
            if rating < 60:
                direction = "bearish"
                rating = 60

        return {
            "reversal": direction,
            "rating": rating,
        }

    def analyze_habr(self, closes, highs, lows, volumes):
        """
        Упрощённый HABR — просто усиливает реверсал.
        """
        base = self.analyze(closes, highs, lows, volumes)
        if base["reversal"] is None:
            return None

        return {
            "direction": base["reversal"],
            "rating": base["rating"] + 10,
        }


detector = SimpleReversalDetector()


def detect_signal(closes, highs, lows, volumes, mode_cfg):
    """
    Простой детектор ПАМП/ДАМП:
    - смотрит на последнюю свечу
    - проверяет % движения и объём
    """

    if len(closes) < 10:
        return None

    pump_min_move_pct = mode_cfg.get("pump_min_move_pct", 1.5)
    pump_min_volume_mult = mode_cfg.get("pump_min_volume_mult", 2.0)

    c0, c1 = closes[0], closes[1]
    v0 = volumes[0]
    v_hist = volumes[1:10]
    v_avg = sum(v_hist) / max(len(v_hist), 1)

    move_pct = (c0 - c1) / max(c1, 1e-7) * 100
    vol_mult = v0 / max(v_avg, 1e-7)

    signal_type = None
    direction_side = None
    rating = 0

    # === PUMP ===
    if move_pct >= pump_min_move_pct and vol_mult >= pump_min_volume_mult:
        signal_type = "PUMP"
        direction_side = "bullish"
        rating = 55 + min(20, move_pct) + min(10, (vol_mult - pump_min_volume_mult) * 2)

    # === DUMP ===
    if move_pct <= -pump_min_move_pct and vol_mult >= pump_min_volume_mult:
        signal_type = "DUMP"
        direction_side = "bearish"
        rating = 55 + min(20, abs(move_pct)) + min(10, (vol_mult - pump_min_volume_mult) * 2)

    if signal_type is None:
        return None

    return {
        "type": signal_type,
        "direction_side": direction_side,
        "rating": int(min(rating, 100)),
    }


def adjust_rating_with_context(
    rating,
    signal_type,
    closes,
    oi_now,
    oi_prev,
    funding_rate,
    liq_status,
    flow_status,
    delta_status,
    trend_score,
    risk_score,
):
    """
    Мягкая корректировка рейтинга.
    Ничего не душит, только слегка подстраивает.
    """

    r = float(rating)

    # тренд
    if "PUMP" in signal_type and trend_score > 0:
        r += 5
    if "DUMP" in signal_type and trend_score < 0:
        r += 5

    # OI
    if oi_now and oi_prev:
        if oi_now > oi_prev * 1.03:
            r += 3
        elif oi_now < oi_prev * 0.97:
            r -= 3

    # дельта
    if delta_status == "bullish" and "PUMP" in signal_type:
        r += 3
    if delta_status == "bearish" and "DUMP" in signal_type:
        r += 3

    # риск
    if risk_score > 80:
        r -= 5
    elif risk_score < 30:
        r += 3

    return int(max(0, min(r, 100)))
