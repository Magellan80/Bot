# detectors.py
from typing import List, Optional, Tuple

from context import (
    compute_change_percent,
    count_trend_bars,
    clamp,
    funding_bias,
)
from reversal_detector import ReversalDetector

detector = ReversalDetector()


# ============================
#   HELPERS
# ============================

def compute_volume_spike(volumes: List[float], lookback: int = 15) -> float:
    if len(volumes) <= lookback:
        return 1.0
    current = volumes[0]
    prev = volumes[1:lookback + 1]
    avg_prev = sum(prev) / len(prev) if sum(prev) > 0 else 1.0
    return current / avg_prev


def small_candle_filter(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    min_body_ratio: float = 0.20
) -> bool:
    if len(closes) < 2:
        return False
    body = abs(closes[0] - closes[1])
    rng = highs[0] - lows[0] if highs[0] != lows[0] else 1e-7
    return (body / rng) >= min_body_ratio


# ============================
#   DETECTORS
# ============================

def detect_big_pump(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    mode_cfg: dict
) -> Tuple[bool, int]:
    if len(closes) < 20:
        return False, 0

    change_5m = compute_change_percent(closes, 0, 5)
    change_24h = compute_change_percent(closes, 0, 20)
    vol_spike = compute_volume_spike(volumes, lookback=15)
    up_bars, _ = count_trend_bars(closes, lookback=8)

    if not small_candle_filter(closes, highs, lows, min_body_ratio=0.20):
        return False, 0

    is_pump = (
        change_5m >= mode_cfg["pump_5m"] and
        change_24h >= mode_cfg["change_24h"] and
        vol_spike >= mode_cfg["volume_spike"] and
        up_bars >= mode_cfg["up_bars"]
    )

    raw_rating = change_5m * 4 + change_24h * 1.5 + vol_spike * 5 + up_bars * 2
    rating = clamp(raw_rating)

    return is_pump, rating


def detect_big_dump(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    mode_cfg: dict
) -> Tuple[bool, int]:
    if len(closes) < 20:
        return False, 0

    change_5m = compute_change_percent(closes, 0, 5)
    change_24h = compute_change_percent(closes, 0, 20)
    vol_spike = compute_volume_spike(volumes, lookback=15)
    _, down_bars = count_trend_bars(closes, lookback=8)

    if not small_candle_filter(closes, highs, lows, min_body_ratio=0.20):
        return False, 0

    is_dump = (
        change_5m <= -mode_cfg["pump_5m"] and
        change_24h <= -mode_cfg["change_24h"] and
        vol_spike >= mode_cfg["volume_spike"] and
        down_bars >= mode_cfg["up_bars"]
    )

    raw_rating = abs(change_5m) * 4 + abs(change_24h) * 1.5 + vol_spike * 5 + down_bars * 2
    rating = clamp(raw_rating)

    return is_dump, rating


def detect_pump_reversal(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float]
) -> Tuple[bool, int]:
    if len(closes) < 10:
        return False, 0

    c0, c1 = closes[0], closes[1]
    h0, l0 = highs[0], lows[0]
    v0, v1 = volumes[0], volumes[1]

    body = abs(c0 - c1)
    upper_wick = h0 - max(c0, c1)

    long_upper = upper_wick > body * 1.5
    red_candle = c0 < c1
    # ослабляем фильтр объёма: допускаем и всплеск, и лёгкое падение
    volume_condition = v0 <= v1 * 1.2
    support_break = c0 < lows[2]

    is_reversal = long_upper and red_candle and volume_condition and support_break

    raw_rating = upper_wick * 6 + abs(c0 - c1) * 3
    rating = clamp(raw_rating)

    return is_reversal, rating


# ============================
#   CONTEXT RATING ADJUSTMENT
# ============================

def adjust_rating_with_context(
    base_rating: int,
    signal_type: str,
    closes_1m: List[float],
    oi_now: Optional[float],
    oi_prev: Optional[float],
    funding_rate: Optional[float],
    liq_status: str,
    flow_status: str,
    delta_status: str,
    trend_score: int,
    risk_score: int
) -> int:
    rating = base_rating

    price_change_15m = compute_change_percent(closes_1m, 0, 15) if len(closes_1m) > 15 else 0.0

    oi_bias = None
    if oi_now is not None and oi_prev is not None and oi_prev != 0:
        if oi_now > oi_prev * 1.03:
            oi_bias = "rising"
        elif oi_now < oi_prev * 0.97:
            oi_bias = "falling"
        else:
            oi_bias = "flat"

    f_bias = funding_bias(funding_rate)

    is_bullish_signal = any(x in signal_type for x in ["Dump → Pump", "PUMP"])
    is_bearish_signal = any(x in signal_type for x in ["Pump → Dump", "DUMP"])

    if is_bullish_signal:
        if price_change_15m < -1.0 and oi_bias == "rising":
            rating += 10
        if f_bias == "bullish":
            rating += 5
        if liq_status == "short_spike":
            rating += 8
        if flow_status == "aggressive_buyers":
            rating += 5
        if delta_status == "bullish":
            rating += 5
        elif delta_status == "bearish":
            rating -= 5
        if oi_bias == "falling" and f_bias == "bearish":
            rating -= 5

        if trend_score > 70:
            rating += 5
        elif trend_score < 30:
            rating -= 5

    if is_bearish_signal:
        if price_change_15m > 1.0 and oi_bias == "rising":
            rating += 10
        if f_bias == "bearish":
            rating += 5
        if liq_status == "long_spike":
            rating += 8
        if flow_status == "aggressive_sellers":
            rating += 5
        if delta_status == "bearish":
            rating += 5
        elif delta_status == "bullish":
            rating -= 5
        if oi_bias == "falling" and f_bias == "bullish":
            rating -= 5

        if trend_score < 30:
            rating += 5
        elif trend_score > 70:
            rating -= 5

    if risk_score > 80:
        rating -= 10
    elif risk_score < 30:
        rating += 5

    return clamp(rating)
