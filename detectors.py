# detectors.py
# PRO MULTI-LAYER SCORING MODEL v6
# 100% FULL BACKWARD COMPATIBILITY WITH SCREENER

import numpy as np


# ============================================================
# UTILITIES
# ============================================================

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)

    return np.mean(trs[-period:])


# ============================================================
# CORE ENGINE
# ============================================================

def _multi_layer_engine(closes, highs, lows, volume, btc_regime="ranging"):

    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)
    volume = np.array(volume)

    if len(closes) < 20:
        return 0, None

    last = closes[-1]

    structure_score = 0
    exhaustion_score = 0
    flow_score = 0
    htf_score = 0

    direction_bias = 0

    # ================= STRUCTURE =================

    if closes[-1] > closes[-3] > closes[-5]:
        structure_score += 15
        direction_bias += 1

    if closes[-1] < closes[-3] < closes[-5]:
        structure_score += 15
        direction_bias -= 1

    if last > np.max(highs[-10:-1]):
        structure_score += 10
        direction_bias += 1

    if last < np.min(lows[-10:-1]):
        structure_score += 10
        direction_bias -= 1

    structure_score = min(structure_score, 30)

    # ================= EXHAUSTION =================

    rsi_value = rsi(closes)
    atr_value = atr(highs, lows, closes)

    if rsi_value > 70:
        exhaustion_score += 15
        direction_bias -= 1

    if rsi_value < 30:
        exhaustion_score += 15
        direction_bias += 1

    if atr_value > 0 and (highs[-1] - lows[-1]) > atr_value * 1.5:
        exhaustion_score += 10

    exhaustion_score = min(exhaustion_score, 25)

    # ================= FLOW =================

    avg_vol = np.mean(volume[-20:])
    if volume[-1] > avg_vol * 1.8:
        flow_score += 15

    flow_score = min(flow_score, 25)

    # ================= BTC REGIME =================

    if btc_regime == "trend":
        htf_score += 10
    elif btc_regime == "ranging":
        htf_score += 5

    htf_score = min(htf_score, 20)

    total = structure_score + exhaustion_score + flow_score + htf_score
    total = min(total, 100)

    if direction_bias > 0:
        direction = "long"
    elif direction_bias < 0:
        direction = "short"
    else:
        direction = None

    return int(total), direction


# ============================================================
# WRAPPERS (EXACT SIGNATURE MATCH FOR SCREENER)
# ============================================================

def detect_big_pump(closes, highs, lows, volume, btc_regime="ranging"):
    rating, direction = _multi_layer_engine(closes, highs, lows, volume, btc_regime)
    if direction == "long":
        return rating, direction
    return 0, None


def detect_big_dump(closes, highs, lows, volume, btc_regime="ranging"):
    rating, direction = _multi_layer_engine(closes, highs, lows, volume, btc_regime)
    if direction == "short":
        return rating, direction
    return 0, None


def detect_reversal(closes, highs, lows, volume, btc_regime="ranging"):
    return _multi_layer_engine(closes, highs, lows, volume, btc_regime)


def detect_pump_reversal(closes, highs, lows, volume, btc_regime="ranging"):
    return _multi_layer_engine(closes, highs, lows, volume, btc_regime)


def adjust_rating_with_context(rating, context=None):
    return rating


def detector(closes, highs, lows, volume, btc_regime="ranging"):
    return _multi_layer_engine(closes, highs, lows, volume, btc_regime)
