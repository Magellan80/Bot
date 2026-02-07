# smart_filters_v3.py  → теперь SmartFilters v4.0 HyperAdaptive

from __future__ import annotations
from typing import Dict, Any, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _score_trend_alignment(direction_side, trend_score, t15, t1h, t4h):
    htf_sum = t15 + t1h + t4h
    score = 0.0

    if direction_side == "bullish":
        if trend_score > 0:
            score += 1
        if htf_sum > 0:
            score += 1
    else:
        if trend_score < 0:
            score += 1
        if htf_sum < 0:
            score += 1

    return _clamp(score / 2.0, 0.0, 1.0)


def _score_btc(direction_side, btc_ctx):
    if not btc_ctx:
        return 0.0

    regime = btc_ctx.get("regime")
    if regime == "trending":
        return 0.3
    if regime == "ranging":
        return 0.2
    if regime == "high_vol":
        return -0.3
    return 0.0


def _score_liquidity(direction_side, liq_bias):
    if not liq_bias or liq_bias == "balanced":
        return 0.0

    if direction_side == "bullish" and liq_bias == "below":
        return 0.3
    if direction_side == "bearish" and liq_bias == "above":
        return 0.3

    if direction_side == "bullish" and liq_bias == "above":
        return -0.3
    if direction_side == "bearish" and liq_bias == "below":
        return -0.3

    return 0.0


def _score_ofd(direction_side, oi_status, flow_status, delta_status):
    score = 0.0

    # OI
    if oi_status == "rising":
        score += 0.2
    elif oi_status == "falling":
        score -= 0.2

    # flow
    if flow_status == "buy" and direction_side == "bullish":
        score += 0.3
    if flow_status == "sell" and direction_side == "bearish":
        score += 0.3
    if flow_status == "buy" and direction_side == "bearish":
        score -= 0.3
    if flow_status == "sell" and direction_side == "bullish":
        score -= 0.3

    # delta
    if delta_status == "bullish" and direction_side == "bullish":
        score += 0.3
    if delta_status == "bearish" and direction_side == "bearish":
        score += 0.3
    if delta_status == "bullish" and direction_side == "bearish":
        score -= 0.3
    if delta_status == "bearish" and direction_side == "bullish":
        score -= 0.3

    return _clamp(score, -0.7, 0.7)


def _score_risk(risk_score):
    if risk_score >= 80:
        return -0.5
    if risk_score <= 30:
        return 0.3
    return 0.0


def _score_memory(profile, direction_side):
    if not profile:
        return 0.0

    regime = profile.get("regime", "neutral")
    pump_prob = float(profile.get("pump_probability") or 0.0)
    dump_prob = float(profile.get("dump_probability") or 0.0)

    score = 0.0

    if direction_side == "bullish":
        score += _clamp(pump_prob * 0.5, 0, 0.5)
        if regime == "pumpy":
            score += 0.3
        if regime == "dumpy":
            score -= 0.3
    else:
        score += _clamp(dump_prob * 0.5, 0, 0.5)
        if regime == "dumpy":
            score += 0.3
        if regime == "pumpy":
            score -= 0.3

    if regime == "chaotic":
        score -= 0.4

    return _clamp(score, -0.7, 0.7)


def apply_smartfilters_v3(
    *,
    symbol: str,
    base_rating: int,
    direction_side: str,
    closes_1m,
    klines_1m,
    trend_score: int,
    trend_15m: int,
    trend_1h: int,
    trend_4h: int,
    liquidity_bias: str,
    noise_level=None,
    btc_ctx=None,
    extra_filters_ok=None,
    global_risk_proxy=None,
    risk_score: int = 0,
    oi_status=None,
    flow_status=None,
    delta_status=None,
    symbol_memory_profile=None,
    vol_cluster=None,
    market_ctx=None,
):
    """
    SmartFilters v4.0 HyperAdaptive (встроено в интерфейс v3)
    """

    rating = float(base_rating)

    # extra filters
    if extra_filters_ok:
        if not extra_filters_ok.get("oi_not_falling", True):
            rating -= 7
        if not extra_filters_ok.get("min_score_ok", True):
            rating -= 5

    # тренд
    trend_align = _score_trend_alignment(direction_side, trend_score, trend_15m, trend_1h, trend_4h)
    rating += trend_align * 8.0

    # BTC
    btc_score = _score_btc(direction_side, btc_ctx)
    rating += btc_score * 6.0

    # ликвидность
    liq_score = _score_liquidity(direction_side, liquidity_bias)
    rating += liq_score * 6.0

    # OI/flow/delta
    ofd_score = _score_ofd(direction_side, oi_status, flow_status, delta_status)
    rating += ofd_score * 7.0

    # риск
    risk_adj = _score_risk(risk_score)
    rating += risk_adj * 6.0

    # память символа
    mem_score = _score_memory(symbol_memory_profile or {}, direction_side)
    rating += mem_score * 8.0

    # волатильность
    vol_score = 0.0
    if vol_cluster:
        cluster = vol_cluster.get("cluster")
        if cluster in ("high", "high_vol"):
            vol_score = 0.1
        elif cluster in ("extreme", "chaotic"):
            vol_score = -0.3
    rating += vol_score * 5.0

    # market ctx
    if market_ctx:
        mr = market_ctx.get("market_regime")
        risk = market_ctx.get("risk")
        if mr == "trending":
            rating += 2
        elif mr == "chaotic":
            rating -= 3
        if risk == "high":
            rating -= 2
        elif risk == "low":
            rating += 1

    rating = _clamp(rating, 0, 100)

    # confidence
    factors = [trend_align, btc_score, liq_score, ofd_score, -abs(risk_adj), mem_score]
    pos = sum(1 for x in factors if x > 0)
    total = len(factors)
    align_ratio = pos / total if total else 0.5

    conf = 0.4 * (rating / 100.0) + 0.6 * align_ratio
    conf = _clamp(conf, 0.1, 1.0)

    return {
        "final_rating": int(rating),
        "confidence": float(conf),
        "symbol_regime": symbol_memory_profile or {"regime": "unknown", "strength": 1},
        "market_ctx": market_ctx or {"market_regime": "unknown", "risk": "normal"},
        "vol_cluster": vol_cluster or {"cluster": "normal", "volatility_score": 0.5},
        "memory_ctx": symbol_memory_profile or {},
        "weights": {
            "trend_align": trend_align,
            "btc_score": btc_score,
            "liq_score": liq_score,
            "ofd_score": ofd_score,
            "risk_adj": risk_adj,
            "mem_score": mem_score,
            "vol_score": vol_score,
        },
    }
