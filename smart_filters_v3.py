# smart_filters_v3.py
"""
SmartFilters v3.7 LITE
(HTF-Structure + HTF-Momentum Divergence + HTF-Volume Profile + HTF-trend
 + Trend Consistency + Structure–Trend Alignment + Trend–Momentum Fusion)

Базируется на SmartFilters v3.6 FULL:
- Symbol Regime Detection (HTF-aware + HTF-structure)
- Volatility Clustering
- Market-Wide Context
- Symbol Memory
- Smart Weighting Engine
- Confidence Engine
- apply_smartfilters_v3(...)

Добавлено/усилено в v3.7 LITE:
- Trend Consistency (15m/1h/4h согласованность)
- Structure–Trend Alignment (HTF-structure + HTF-trend согласованность)
- Trend–Momentum Fusion (HTF-trend + HTF-momentum согласованность)
- мягкое влияние этих факторов на confidence (без агрессивного скейлинга)
"""

from __future__ import annotations

import statistics
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from htf_structure import compute_htf_structure


# ==========================
# 1. Вспомогательные функции
# ==========================

def _safe_pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / abs(b) * 100.0


def _rolling_std(values: List[float], window: int) -> float:
    if len(values) < window:
        return 0.0
    chunk = values[:window]
    if len(chunk) < 2:
        return 0.0
    return statistics.pstdev(chunk)


def _rolling_atr_from_ohlc(klines: List[List[Any]], window: int = 14) -> float:
    """
    Простейший ATR по последним N барам.
    klines: [[ts, open, high, low, close, volume, ...], ...]
    """
    if not klines or len(klines) < window + 1:
        return 0.0

    trs = []
    for i in range(window):
        o = float(klines[i][1])
        h = float(klines[i][2])
        l = float(klines[i][3])
        c_prev = float(klines[i + 1][4])

        tr = max(
            h - l,
            abs(h - c_prev),
            abs(l - c_prev),
        )
        trs.append(tr)

    if not trs:
        return 0.0
    return sum(trs) / len(trs)


def _compute_htf_trend_score(
    trend_15m: Optional[float],
    trend_1h: Optional[float],
    trend_4h: Optional[float],
) -> float:
    """
    Агрегированный HTF-тренд: взвешенное среднее по 15m/1h/4h.
    Используется как мягкий, но устойчивый контекст.
    """
    parts = []
    weights = []

    if trend_15m is not None:
        parts.append(trend_15m)
        weights.append(1.0)
    if trend_1h is not None:
        parts.append(trend_1h)
        weights.append(1.5)
    if trend_4h is not None:
        parts.append(trend_4h)
        weights.append(2.0)

    if not parts or not weights:
        return 0.0

    num = sum(p * w for p, w in zip(parts, weights))
    den = sum(weights)
    if den == 0:
        return 0.0
    return num / den


# ===== HTF Momentum (RSI + divergence) =====

def _compute_rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) <= period:
        return [50.0] * len(values)
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[i - 1] - values[i]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period if sum(losses) > 0 else 0.0

    rsi = [50.0] * len(values)
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(values)):
        diff = values[i - 1] - values[i]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _detect_htf_momentum_divergence(
    closes: List[float],
    lookback: int = 30,
    rsi_period: int = 14,
) -> Dict[str, Optional[Any]]:
    """
    Простая HTF-momentum divergence:
    - bullish: цена делает более низкий минимум, RSI делает более высокий минимум
    - bearish: цена делает более высокий максимум, RSI делает более низкий максимум
    strength: 1..3 (насколько выражена дивергенция)
    """
    if len(closes) < max(lookback, rsi_period) + 2:
        return {"divergence": None, "strength": None}

    closes = closes[:lookback]
    rsi = _compute_rsi(closes, period=rsi_period)

    window = min(10, len(closes))
    segment_prices = closes[:window]
    segment_rsi = rsi[:window]

    price_last = segment_prices[0]
    price_prev = segment_prices[1]
    price_min = min(segment_prices)
    price_max = max(segment_prices)

    rsi_last = segment_rsi[0]
    rsi_prev = segment_rsi[1]
    rsi_min = min(segment_rsi)
    rsi_max = max(segment_rsi)

    bullish = False
    bearish = False

    if price_last < price_prev and rsi_last > rsi_prev and price_last <= price_min * 1.001:
        bullish = True

    if price_last > price_prev and rsi_last < rsi_prev and price_last >= price_max * 0.999:
        bearish = True

    if not bullish and not bearish:
        return {"divergence": None, "strength": None}

    def _rel(a: float, b: float) -> float:
        if b == 0:
            return 0.0
        return abs(a - b) / abs(b) * 100.0

    price_move = _rel(price_last, price_prev)
    rsi_move = _rel(rsi_last, rsi_prev)

    diff = abs(rsi_move - price_move)
    if diff > 3:
        strength = 3
    elif diff > 1.5:
        strength = 2
    else:
        strength = 1

    return {
        "divergence": "bullish" if bullish else "bearish",
        "strength": strength,
    }


# ===== HTF Volume Profile (HVN/LVN) =====

def _build_volume_profile(
    klines: List[List[Any]],
    buckets: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Простейший volume profile по последним барам:
    - делим диапазон цен на buckets
    - суммируем объём по каждому бакету
    - находим HVN/LVN и value area
    """
    if not klines or len(klines) < 10:
        return None

    closes = [float(c[4]) for c in klines]
    highs = [float(c[2]) for c in klines]
    lows = [float(c[3]) for c in klines]
    vols = [float(c[5]) for c in klines]

    price_min = min(lows)
    price_max = max(highs)
    if price_max <= price_min:
        return None

    step = (price_max - price_min) / buckets
    if step <= 0:
        return None

    vp = [0.0] * buckets

    for i in range(len(klines)):
        mid = (highs[i] + lows[i]) / 2.0
        idx = int((mid - price_min) / step)
        if idx < 0:
            idx = 0
        if idx >= buckets:
            idx = buckets - 1
        vp[idx] += vols[i]

    total_vol = sum(vp)
    if total_vol <= 0:
        return None

    max_vol = max(vp)
    min_vol = min(v for v in vp if v > 0) if any(v > 0 for v in vp) else 0.0
    hvn_idx = vp.index(max_vol)
    lvn_idx = vp.index(min_vol) if min_vol > 0 else hvn_idx

    def _bucket_price(i: int) -> float:
        return price_min + step * (i + 0.5)

    hvn_price = _bucket_price(hvn_idx)
    lvn_price = _bucket_price(lvn_idx)

    # value area ~ 70% объёма
    sorted_pairs = sorted(enumerate(vp), key=lambda x: x[1], reverse=True)
    acc = 0.0
    used = set()
    for idx, v in sorted_pairs:
        acc += v
        used.add(idx)
        if acc / total_vol >= 0.7:
            break

    if used:
        va_low_idx = min(used)
        va_high_idx = max(used)
        va_low = price_min + step * va_low_idx
        va_high = price_min + step * (va_high_idx + 1)
    else:
        va_low = price_min
        va_high = price_max

    last_price = closes[0]
    if last_price > hvn_price:
        vol_bias = "above_hvn"
    elif last_price < hvn_price:
        vol_bias = "below_hvn"
    else:
        vol_bias = "at_hvn"

    if last_price < va_low:
        va_position = "below_va"
    elif last_price > va_high:
        va_position = "above_va"
    else:
        va_position = "inside_va"

    return {
        "hvn_price": hvn_price,
        "lvn_price": lvn_price,
        "va_low": va_low,
        "va_high": va_high,
        "vol_bias": vol_bias,
        "va_position": va_position,
        "total_vol": total_vol,
        "max_vol": max_vol,
    }


def _interpret_volume_profile(
    vp_ctx: Optional[Dict[str, Any]],
) -> Dict[str, Optional[Any]]:
    """
    Превращаем сырые HVN/LVN/VA в компактный контекст:
    - profile_regime: balanced / trending / skewed
    - profile_bias: support / resistance / neutral
    - strength: 1..3
    """
    if vp_ctx is None:
        return {
            "profile_regime": None,
            "profile_bias": None,
            "strength": None,
        }

    hvn_price = vp_ctx["hvn_price"]
    lvn_price = vp_ctx["lvn_price"]
    va_low = vp_ctx["va_low"]
    va_high = vp_ctx["va_high"]
    vol_bias = vp_ctx["vol_bias"]
    va_position = vp_ctx["va_position"]

    profile_regime = "balanced"
    profile_bias = "neutral"
    strength = 1

    # Если цена стабильно внутри VA → balanced
    if va_position == "inside_va":
        profile_regime = "balanced"
        strength = 1
    else:
        profile_regime = "trending"
        strength = 2

    # Если цена выше HVN и выше VA → сопротивление снизу, поддержка снизу слабая
    if vol_bias == "above_hvn" and va_position == "above_va":
        profile_bias = "resistance"
        strength = 3
    # Если цена ниже HVN и ниже VA → поддержка сверху, сопротивление сверху слабое
    elif vol_bias == "below_hvn" and va_position == "below_va":
        profile_bias = "support"
        strength = 3
    else:
        profile_bias = "neutral"

    return {
        "profile_regime": profile_regime,
        "profile_bias": profile_bias,
        "strength": strength,
    }


# ==========================
# 2. Конфиг и dataclass
# ==========================

@dataclass
class SmartFiltersConfig:
    max_signals_per_symbol: int = 100
    memory_half_life_sec: float = 3600.0
    factor_boost_min: float = 0.9
    factor_boost_max: float = 1.1


@dataclass
class SymbolMemory:
    signals: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: float = 0.0


# ==========================
# 3. Класс SmartFilters v3.7 LITE
# ==========================

class SmartFilters:
    def __init__(self, config: Optional[SmartFiltersConfig] = None):
        self.config = config or SmartFiltersConfig()
        self._symbol_memory: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "signals": deque(maxlen=self.config.max_signals_per_symbol),
                "last_update": 0.0,
            }
        )
        self._factor_perf = {
            "trend": 1.0,
            "momentum": 1.0,
            "liquidity": 1.0,
            "delta": 1.0,
            "micro": 1.0,
            "noise": 1.0,
            "structure": 1.0,
            "orderbook": 1.0,
            "volume_profile": 1.0,
        }

    def _decay_memory(self, symbol: str):
        mem = self._symbol_memory[symbol]
        signals = mem["signals"]
        if not signals:
            return

        now = time.time()
        half_life = self.config.memory_half_life_sec
        if half_life <= 0:
            return

        new_signals = deque(maxlen=signals.maxlen)
        for s in signals:
            ts = s.get("ts", now)
            age = now - ts
            if age < 0:
                age = 0
            decay_factor = 0.5 ** (age / half_life)
            if decay_factor > 0.2:
                new_signals.append(s)

        mem["signals"] = new_signals

    def _get_factor_boost(self) -> float:
        vals = list(self._factor_perf.values())
        if not vals:
            return 1.0
        avg_perf = sum(vals) / len(vals)
        boost = 1.0 + (avg_perf - 1.0) * 0.2
        if boost < self.config.factor_boost_min:
            boost = self.config.factor_boost_min
        if boost > self.config.factor_boost_max:
            boost = self.config.factor_boost_max
        return boost

    # ==========================
    # 4. Symbol Regime Detection (HTF-aware + HTF-structure)
    # ==========================

    def detect_symbol_regime(
        self,
        closes: List[float],
        klines_1m: List[List[Any]],
        liquidity_bias: Optional[str] = None,
        noise_level: Optional[float] = None,
        trend_15m: Optional[float] = None,
        trend_1h: Optional[float] = None,
        trend_4h: Optional[float] = None,
        htf_structure_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        regime = "range"
        strength = 0.0
        meta: Dict[str, Any] = {}

        if not closes or len(closes) < 20 or not klines_1m or len(klines_1m) < 20:
            return {
                "regime": regime,
                "strength": strength,
                "meta": meta,
            }

        closes = closes[:]  # копия
        atr = _rolling_atr_from_ohlc(klines_1m, window=14)
        last_price = closes[0]
        atr_pct = _safe_pct_change(last_price + atr, last_price)

        std_20 = _rolling_std(closes, window=20)

        c0 = closes[0]
        cN = closes[min(len(closes) - 1, 30)]
        trend_pct = _safe_pct_change(c0, cN)

        diffs = [closes[i] - closes[i + 1] for i in range(min(30, len(closes) - 1))]
        if len(diffs) >= 5:
            std_diffs = statistics.pstdev(diffs)
        else:
            std_diffs = 0.0

        low_vol_thr = 0.15
        high_vol_thr = 0.8
        strong_trend_thr = 1.5
        mid_trend_thr = 0.6

        if atr_pct < low_vol_thr and std_20 < low_vol_thr:
            regime = "squeeze"
            strength = 0.7
        elif atr_pct > high_vol_thr and std_20 > high_vol_thr:
            regime = "expansion"
            strength = 0.7

        abs_trend = abs(trend_pct)
        if abs_trend > strong_trend_thr:
            regime = "trend"
            strength = 0.9
        elif abs_trend > mid_trend_thr:
            if regime not in ("squeeze", "expansion"):
                regime = "trend"
                strength = 0.6
        else:
            if regime not in ("squeeze", "expansion"):
                regime = "range"
                strength = 0.5

        if noise_level is not None and noise_level > 0.7:
            regime = "chaos"
            strength = 0.8

        # HTF-тренд
        htf_trend_score = _compute_htf_trend_score(trend_15m, trend_1h, trend_4h)
        htf_trend_abs = abs(htf_trend_score)

        if regime == "trend":
            if htf_trend_abs > 1.5:
                strength = min(1.0, strength + 0.15)
            elif htf_trend_abs > 0.7:
                strength = min(1.0, strength + 0.05)
        else:
            if htf_trend_abs > 2.0 and regime in ("range", "squeeze", "expansion"):
                regime = "trend"
                strength = max(strength, 0.6)

        # HTF-structure
        htf_structure = None
        htf_event = None
        htf_strength = None

        if htf_structure_ctx is not None:
            htf_structure = htf_structure_ctx.get("structure")
            htf_event = htf_structure_ctx.get("event")
            htf_strength = htf_structure_ctx.get("strength")

            if htf_structure in ("bullish", "bearish") and isinstance(htf_strength, int):
                if htf_strength >= 3 and regime == "trend":
                    strength = min(1.0, strength + 0.1)
                elif htf_strength >= 4 and regime in ("range", "squeeze"):
                    regime = "trend"
                    strength = max(strength, 0.6)

        if liquidity_bias == "bullish" and regime == "trend" and trend_pct > 0:
            strength = min(1.0, strength + 0.1)
        if liquidity_bias == "bearish" and regime == "trend" and trend_pct < 0:
            strength = min(1.0, strength + 0.1)

        meta = {
            "atr": atr,
            "atr_pct": atr_pct,
            "std_20": std_20,
            "trend_pct_30": trend_pct,
            "std_diffs": std_diffs,
            "noise_level": noise_level,
            "liquidity_bias": liquidity_bias,
            "htf_trend_score": htf_trend_score,
            "htf_trend_15m": trend_15m,
            "htf_trend_1h": trend_1h,
            "htf_trend_4h": trend_4h,
            "htf_structure": htf_structure,
            "htf_event": htf_event,
            "htf_strength": htf_strength,
        }

        return {
            "regime": regime,
            "strength": strength,
            "meta": meta,
        }

    # ==========================
    # 5. Volatility Clustering
    # ==========================

    def detect_volatility_cluster(
        self,
        closes: List[float],
        klines_1m: List[List[Any]],
    ) -> Dict[str, Any]:
        if not closes or len(closes) < 20 or not klines_1m or len(klines_1m) < 20:
            return {
                "cluster": "normal",
                "volatility_score": 0.0,
                "meta": {},
            }

        atr = _rolling_atr_from_ohlc(klines_1m, window=14)
        last_price = closes[0]
        atr_pct = _safe_pct_change(last_price + atr, last_price)

        std_20 = _rolling_std(closes, window=20)
        std_10 = _rolling_std(closes, window=10)

        ratio_std = std_10 / std_20 if std_20 > 0 else 1.0

        cluster = "normal"
        vol_score = 0.0

        if atr_pct < 0.15 and std_20 < 0.15:
            cluster = "low_vol"
            vol_score = 0.2
        elif atr_pct > 1.0 and std_20 > 1.0:
            cluster = "high_vol"
            vol_score = 0.8
        else:
            cluster = "normal"
            vol_score = 0.5

        if ratio_std > 1.3:
            cluster = "expansion"
            vol_score = max(vol_score, 0.7)
        elif ratio_std < 0.7:
            cluster = "contraction"
            vol_score = min(vol_score, 0.3)

        meta = {
            "atr_pct": atr_pct,
            "std_20": std_20,
            "std_10": std_10,
            "ratio_std": ratio_std,
        }

        return {
            "cluster": cluster,
            "volatility_score": vol_score,
            "meta": meta,
        }

    # ==========================
    # 6. Market-Wide Context
    # ==========================

    def compute_market_context(
        self,
        btc_ctx: Dict[str, Any],
        btc_trend_score: Optional[float] = None,
        eth_trend_score: Optional[float] = None,
        sol_trend_score: Optional[float] = None,
        global_risk_proxy: Optional[float] = None,
    ) -> Dict[str, Any]:
        regime = "neutral"
        risk = 0.5
        meta: Dict[str, Any] = {}

        btc_regime = btc_ctx.get("regime", "neutral")
        btc_factor = btc_ctx.get("factor", 1.0)

        if btc_regime == "trending":
            regime = "bullish_impulse"
        elif btc_regime == "ranging":
            regime = "range_market"
        elif btc_regime == "high_vol":
            regime = "high_risk"
        else:
            regime = "neutral"

        if btc_factor < 0.9:
            risk = 0.7
        elif btc_factor > 1.05:
            risk = 0.4
        else:
            risk = 0.5

        trend_scores = []
        if btc_trend_score is not None:
            trend_scores.append(btc_trend_score)
        if eth_trend_score is not None:
            trend_scores.append(eth_trend_score)
        if sol_trend_score is not None:
            trend_scores.append(sol_trend_score)

        avg_trend = 0.0
        if trend_scores:
            avg_trend = sum(trend_scores) / len(trend_scores)

        if avg_trend > 3:
            regime = "broad_bullish"
            risk = min(risk, 0.45)
        elif avg_trend < -3:
            regime = "broad_bearish"
            risk = max(risk, 0.65)

        if global_risk_proxy is not None:
            risk = 0.5 * risk + 0.5 * global_risk_proxy

        meta = {
            "btc_regime": btc_regime,
            "btc_factor": btc_factor,
            "btc_trend_score": btc_trend_score,
            "eth_trend_score": eth_trend_score,
            "sol_trend_score": sol_trend_score,
            "avg_trend": avg_trend,
            "global_risk_proxy": global_risk_proxy,
        }

        return {
            "market_regime": regime,
            "risk": risk,
            "meta": meta,
        }

    # ==========================
    # 7. Symbol Memory
    # ==========================

    def update_symbol_memory(
        self,
        symbol: str,
        signal_type: str,
        rating: int,
        direction_side: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self._decay_memory(symbol)
        mem = self._symbol_memory[symbol]
        mem["signals"].appendleft(
            {
                "ts": time.time(),
                "type": signal_type,
                "rating": rating,
                "direction": direction_side,
                "ctx": context or {},
            }
        )
        mem["last_update"] = time.time()

    def get_symbol_memory(self, symbol: str) -> Dict[str, Any]:
        self._decay_memory(symbol)
        mem = self._symbol_memory[symbol]
        signals = list(mem["signals"])

        if not signals:
            return {
                "memory_bias": None,
                "recent_patterns": [],
                "meta": {
                    "count": 0,
                },
            }

        bull = 0
        bear = 0
        for s in signals:
            d = s.get("direction")
            if d == "bullish":
                bull += 1
            elif d == "bearish":
                bear += 1

        memory_bias = None
        if bull > bear * 1.5:
            memory_bias = "bullish"
        elif bear > bull * 1.5:
            memory_bias = "bearish"

        recent_patterns = [s["type"] for s in signals[:5]]

        meta = {
            "count": len(signals),
            "bull": bull,
            "bear": bear,
        }

        return {
            "memory_bias": memory_bias,
            "recent_patterns": recent_patterns,
            "meta": meta,
        }

    # ==========================
    # 8. Smart Weighting Engine
    # ==========================

    def compute_smart_weights(
        self,
        symbol_regime: Dict[str, Any],
        vol_cluster: Dict[str, Any],
        market_ctx: Dict[str, Any],
    ) -> Dict[str, float]:
        regime = symbol_regime.get("regime", "range")
        regime_strength = symbol_regime.get("strength", 0.5)
        cluster = vol_cluster.get("cluster", "normal")
        market_regime = market_ctx.get("market_regime", "neutral")
        market_risk = market_ctx.get("risk", 0.5)

        w = {
            "trend": 1.0,
            "momentum": 1.0,
            "liquidity": 1.0,
            "delta": 1.0,
            "micro": 1.0,
            "noise": 1.0,
            "structure": 1.0,
            "orderbook": 1.0,
            "volume_profile": 1.0,
        }
        if regime == "trend":
            w["trend"] += 0.4 * regime_strength
            w["momentum"] += 0.2 * regime_strength
            w["structure"] += 0.2 * regime_strength
        elif regime == "range":
            w["structure"] += 0.3 * regime_strength
            w["noise"] += 0.2 * regime_strength
            w["liquidity"] += 0.1 * regime_strength
        elif regime == "squeeze":
            w["liquidity"] += 0.3 * regime_strength
            w["momentum"] += 0.2 * regime_strength
        elif regime == "expansion":
            w["momentum"] += 0.3 * regime_strength
            w["delta"] += 0.2 * regime_strength
            w["micro"] += 0.2 * regime_strength
        elif regime == "chaos":
            w["noise"] += 0.4 * regime_strength
            w["orderbook"] += 0.2 * regime_strength

        if cluster == "low_vol":
            w["noise"] += 0.2
            w["structure"] += 0.2
        elif cluster == "high_vol":
            w["orderbook"] += 0.2
            w["liquidity"] += 0.2
            w["micro"] += 0.2
        elif cluster == "expansion":
            w["momentum"] += 0.2
            w["delta"] += 0.2
        elif cluster == "contraction":
            w["structure"] += 0.2
            w["noise"] += 0.2

        if market_regime in ("broad_bullish", "bullish_impulse"):
            w["trend"] += 0.2
            w["momentum"] += 0.1
        elif market_regime in ("broad_bearish", "high_risk"):
            w["orderbook"] += 0.2
            w["noise"] += 0.2

        if market_risk > 0.6:
            w["orderbook"] += 0.2
            w["noise"] += 0.2
            w["liquidity"] += 0.1

        max_w = max(w.values()) if w else 1.0
        if max_w <= 0:
            max_w = 1.0
        for k in w:
            w[k] = w[k] / max_w

        return w

    # ==========================
    # 9. Confidence Engine (HTF-trend + HTF-structure + HTF-momentum + Volume Profile + LITE-trend-модули)
    # ==========================

    def compute_confidence(
        self,
        base_rating: int,
        filters_ok: Dict[str, bool],
        symbol_regime: Dict[str, Any],
        market_ctx: Dict[str, Any],
        memory_ctx: Dict[str, Any],
        trend_15m: Optional[float] = None,
        trend_1h: Optional[float] = None,
        trend_4h: Optional[float] = None,
        htf_structure_ctx: Optional[Dict[str, Any]] = None,
        htf_momentum_ctx: Optional[Dict[str, Any]] = None,
        volume_profile_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not filters_ok:
            return {
                "confidence": 0.0,
                "meta": {
                    "filters_ok_ratio": 0.0,
                },
            }
        total = len(filters_ok)
        good = sum(1 for v in filters_ok.values() if v)
        filters_ok_ratio = good / total if total > 0 else 0.0

        regime_strength = symbol_regime.get("strength", 0.5)
        regime = symbol_regime.get("regime", "range")

        regime_bonus = 0.0
        if regime == "trend":
            regime_bonus = 0.1 * regime_strength
        elif regime == "range":
            regime_bonus = 0.05 * regime_strength
        elif regime == "squeeze":
            regime_bonus = 0.08 * regime_strength
        elif regime == "expansion":
            regime_bonus = 0.1 * regime_strength
        elif regime == "chaos":
            regime_bonus = -0.1 * regime_strength

        market_risk = market_ctx.get("risk", 0.5)
        market_regime = market_ctx.get("market_regime", "neutral")

        market_bonus = 0.0
        if market_regime in ("broad_bullish", "bullish_impulse"):
            market_bonus += 0.05
        elif market_regime in ("broad_bearish", "high_risk"):
            market_bonus -= 0.05

        risk_penalty = (market_risk - 0.5) * 0.3

        memory_bias = memory_ctx.get("memory_bias")
        memory_bonus = 0.0
        if memory_bias is not None:
            direction_side = "bullish" if base_rating >= 0 else "bearish"
            if direction_side == memory_bias:
                memory_bonus += 0.08
            else:
                memory_bonus -= 0.05

        # HTF-тренд
        htf_trend_score = _compute_htf_trend_score(trend_15m, trend_1h, trend_4h)
        htf_trend_bonus = 0.0
        if htf_trend_score != 0:
            direction_side = "bullish" if base_rating >= 0 else "bearish"
            same_direction = (direction_side == "bullish" and htf_trend_score > 0) or (
                direction_side == "bearish" and htf_trend_score < 0
            )
            htf_abs = abs(htf_trend_score)

            if same_direction:
                if htf_abs > 2.0:
                    htf_trend_bonus += 0.10
                elif htf_abs > 1.0:
                    htf_trend_bonus += 0.05
            else:
                if htf_abs > 2.0:
                    htf_trend_bonus -= 0.10
                elif htf_abs > 1.0:
                    htf_trend_bonus -= 0.05

        # HTF-structure
        htf_structure_bonus = 0.0
        htf_structure = None
        htf_event = None
        htf_strength = None

        if htf_structure_ctx is not None:
            htf_structure = htf_structure_ctx.get("structure")
            htf_event = htf_structure_ctx.get("event")
            htf_strength = htf_structure_ctx.get("strength")

            if htf_structure in ("bullish", "bearish") and isinstance(htf_strength, int):
                direction_side = "bullish" if base_rating >= 0 else "bearish"
                same_direction = (direction_side == "bullish" and htf_structure == "bullish") or (
                    direction_side == "bearish" and htf_structure == "bearish"
                )

                if same_direction:
                    if htf_strength == 2:
                        htf_structure_bonus += 0.02
                    elif htf_strength == 3:
                        htf_structure_bonus += 0.04
                    elif htf_strength == 4:
                        htf_structure_bonus += 0.06
                    elif htf_strength >= 5:
                        htf_structure_bonus += 0.08
                else:
                    if htf_strength == 2:
                        htf_structure_bonus -= 0.02
                    elif htf_strength == 3:
                        htf_structure_bonus -= 0.04
                    elif htf_strength == 4:
                        htf_structure_bonus -= 0.06
                    elif htf_strength >= 5:
                        htf_structure_bonus -= 0.08

        # HTF-momentum divergence
        htf_momentum_bonus = 0.0
        htf_mom_div = None
        htf_mom_strength = None

        if htf_momentum_ctx is not None:
            htf_mom_div = htf_momentum_ctx.get("divergence")
            htf_mom_strength = htf_momentum_ctx.get("strength")

            if htf_mom_div in ("bullish", "bearish") and isinstance(htf_mom_strength, int):
                direction_side = "bullish" if base_rating >= 0 else "bearish"
                same_direction = (direction_side == "bullish" and htf_mom_div == "bullish") or (
                    direction_side == "bearish" and htf_mom_div == "bearish"
                )
                if same_direction:
                    if htf_mom_strength == 1:
                        htf_momentum_bonus += 0.02
                    elif htf_mom_strength == 2:
                        htf_momentum_bonus += 0.04
                    elif htf_mom_strength >= 3:
                        htf_momentum_bonus += 0.06
                else:
                    if htf_mom_strength == 1:
                        htf_momentum_bonus -= 0.02
                    elif htf_mom_strength == 2:
                        htf_momentum_bonus -= 0.04
                    elif htf_mom_strength >= 3:
                        htf_momentum_bonus -= 0.06

        # ===== v3.7 LITE: Trend Consistency (15m/1h/4h) =====
        trend_consistency_bonus = 0.0
        trend_consistency = None

        def _sign(x: Optional[float]) -> int:
            if x is None:
                return 0
            if x > 0:
                return 1
            if x < 0:
                return -1
            return 0

        s15 = _sign(trend_15m)
        s1h = _sign(trend_1h)
        s4h = _sign(trend_4h)
        signs = [s for s in (s15, s1h, s4h) if s != 0]

        if len(signs) >= 2:
            if all(s == signs[0] for s in signs):
                trend_consistency = "strong_alignment"
                trend_consistency_bonus = 0.05
            elif s4h != 0 and s1h != 0 and s4h == s1h:
                trend_consistency = "htf_alignment"
                trend_consistency_bonus = 0.03
            elif s4h != 0 and ((s1h != 0 and s4h != s1h) or (s15 != 0 and s4h != s15)):
                trend_consistency = "htf_conflict"
                trend_consistency_bonus = -0.05

        # ===== v3.7 LITE: Structure–Trend Alignment =====
        htf_trend_structure_alignment_bonus = 0.0
        htf_trend_structure_alignment = None
        htf_trend_direction = "bullish" if htf_trend_score > 0 else "bearish" if htf_trend_score < 0 else None
        htf_structure_direction = htf_structure if htf_structure in ("bullish", "bearish") else None

        if htf_trend_direction is not None and htf_structure_direction is not None:
            if htf_trend_direction == htf_structure_direction:
                htf_trend_structure_alignment = "aligned"
                htf_trend_structure_alignment_bonus = 0.03
            else:
                htf_trend_structure_alignment = "conflict"
                htf_trend_structure_alignment_bonus = -0.03

        # ===== v3.7 LITE: Trend–Momentum Fusion =====
        htf_trend_mom_fusion_bonus = 0.0
        htf_trend_mom_fusion = None

        if htf_trend_direction is not None and htf_mom_div in ("bullish", "bearish"):
            if htf_trend_direction == htf_mom_div:
                htf_trend_mom_fusion = "aligned"
                htf_trend_mom_fusion_bonus = 0.03
            else:
                if abs(htf_trend_score) > 2.0:
                    htf_trend_mom_fusion = "strong_trend_vs_momentum"
                    htf_trend_mom_fusion_bonus = -0.02

        # Volume profile
        volume_profile_bonus = 0.0
        profile_regime = None
        profile_bias = None
        profile_strength = None

        if volume_profile_ctx is not None:
            profile_regime = volume_profile_ctx.get("profile_regime")
            profile_bias = volume_profile_ctx.get("profile_bias")
            profile_strength = volume_profile_ctx.get("strength")

            if profile_bias in ("support", "resistance") and isinstance(profile_strength, int):
                direction_side = "bullish" if base_rating >= 0 else "bearish"

                if profile_bias == "support":
                    if direction_side == "bullish":
                        if profile_strength == 1:
                            volume_profile_bonus += 0.02
                        elif profile_strength == 2:
                            volume_profile_bonus += 0.04
                        elif profile_strength >= 3:
                            volume_profile_bonus += 0.06
                    else:
                        if profile_strength == 1:
                            volume_profile_bonus -= 0.02
                        elif profile_strength == 2:
                            volume_profile_bonus -= 0.04
                        elif profile_strength >= 3:
                            volume_profile_bonus -= 0.06
                elif profile_bias == "resistance":
                    if direction_side == "bearish":
                        if profile_strength == 1:
                            volume_profile_bonus += 0.02
                        elif profile_strength == 2:
                            volume_profile_bonus += 0.04
                        elif profile_strength >= 3:
                            volume_profile_bonus += 0.06
                    else:
                        if profile_strength == 1:
                            volume_profile_bonus -= 0.02
                        elif profile_strength == 2:
                            volume_profile_bonus -= 0.04
                        elif profile_strength >= 3:
                            volume_profile_bonus -= 0.06

        confidence = filters_ok_ratio
        confidence += regime_bonus
        confidence += market_bonus
        confidence -= risk_penalty
        confidence += memory_bonus
        confidence += htf_trend_bonus
        confidence += htf_structure_bonus
        confidence += htf_momentum_bonus
        confidence += volume_profile_bonus
        confidence += trend_consistency_bonus
        confidence += htf_trend_structure_alignment_bonus
        confidence += htf_trend_mom_fusion_bonus

        factor_boost = self._get_factor_boost()
        confidence *= factor_boost

        if confidence < 0:
            confidence = 0.0
        if confidence > 1:
            confidence = 1.0

        meta = {
            "filters_ok_ratio": filters_ok_ratio,
            "regime_bonus": regime_bonus,
            "market_bonus": market_bonus,
            "risk_penalty": risk_penalty,
            "memory_bonus": memory_bonus,
            "htf_trend_bonus": htf_trend_bonus,
            "htf_trend_score": htf_trend_score,
            "htf_structure_bonus": htf_structure_bonus,
            "htf_structure": htf_structure,
            "htf_event": htf_event,
            "htf_strength": htf_strength,
            "htf_momentum_bonus": htf_momentum_bonus,
            "htf_momentum_divergence": htf_mom_div,
            "htf_momentum_strength": htf_mom_strength,
            "volume_profile_bonus": volume_profile_bonus,
            "volume_profile_regime": profile_regime,
            "volume_profile_bias": profile_bias,
            "volume_profile_strength": profile_strength,
            "market_risk": market_risk,
            "market_regime": market_regime,
            "symbol_regime": symbol_regime.get("regime"),
            "memory_bias": memory_bias,
            "factor_boost": factor_boost,
            "trend_consistency_bonus": trend_consistency_bonus,
            "trend_consistency": trend_consistency,
            "htf_trend_structure_alignment_bonus": htf_trend_structure_alignment_bonus,
            "htf_trend_structure_alignment": htf_trend_structure_alignment,
            "htf_trend_mom_fusion_bonus": htf_trend_mom_fusion_bonus,
            "htf_trend_mom_fusion": htf_trend_mom_fusion,
        }

        return {
            "confidence": confidence,
            "meta": meta,
        }

    # ==========================
    # 10. Финальная обёртка v3.7 LITE (метод)
    # ==========================

    def apply_smartfilters_v3(
        self,
        symbol: str,
        base_rating: int,
        direction_side: str,
        closes_1m: List[float],
        klines_1m: List[List[Any]],
        trend_score: float,
        trend_15m: float,
        trend_1h: float,
        trend_4h: float,
        liquidity_bias: Optional[str],
        noise_level: Optional[float],
        btc_ctx: Dict[str, Any],
        extra_filters_ok: Dict[str, bool],
        global_risk_proxy: Optional[float] = None,
    ) -> Dict[str, Any]:
        # 0) HTF-structure
        htf_structure_ctx = compute_htf_structure(klines_1m)

        # 0.1) HTF-momentum divergence
        closes_for_mom = [float(c[4]) for c in klines_1m][::-1]
        htf_momentum_ctx = _detect_htf_momentum_divergence(closes_for_mom)

        # 0.2) Volume profile
        vp_raw = _build_volume_profile(klines_1m)
        volume_profile_ctx = _interpret_volume_profile(vp_raw)

        # 1) режим инструмента
        symbol_regime = self.detect_symbol_regime(
            closes_1m,
            klines_1m,
            liquidity_bias=liquidity_bias,
            noise_level=noise_level,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            htf_structure_ctx=htf_structure_ctx,
        )

        # 2) кластер волатильности
        vol_cluster = self.detect_volatility_cluster(
            closes_1m,
            klines_1m,
        )

        # 3) рыночный контекст
        market_ctx = self.compute_market_context(
            btc_ctx=btc_ctx,
            btc_trend_score=trend_score,
            eth_trend_score=None,
            sol_trend_score=None,
            global_risk_proxy=global_risk_proxy,
        )

        # 4) память по символу
        memory_ctx = self.get_symbol_memory(symbol)

        # 5) веса
        weights = self.compute_smart_weights(
            symbol_regime=symbol_regime,
            vol_cluster=vol_cluster,
            market_ctx=market_ctx,
        )

        # 6) filters_ok
        filters_ok = dict(extra_filters_ok)
        filters_ok["symbol_regime_not_chaos"] = symbol_regime.get("regime") != "chaos"
        filters_ok["market_not_high_risk"] = market_ctx.get("market_regime") not in ("high_risk",)
        filters_ok["vol_not_extreme"] = vol_cluster.get("cluster") not in ("high_vol",)

        # 7) confidence
        conf = self.compute_confidence(
            base_rating=base_rating,
            filters_ok=filters_ok,
            symbol_regime=symbol_regime,
            market_ctx=market_ctx,
            memory_ctx=memory_ctx,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            htf_structure_ctx=htf_structure_ctx,
            htf_momentum_ctx=htf_momentum_ctx,
            volume_profile_ctx=volume_profile_ctx,
        )
        confidence = conf["confidence"]

        # 8) адаптация рейтинга
        final_rating = float(base_rating)

        if confidence < 0.4:
            final_rating *= 0.85  # v2.1: было 0.7
        elif confidence > 0.7:
            final_rating *= 1.1

        market_risk = market_ctx.get("risk", 0.5)
        if market_risk > 0.7:
            final_rating *= 0.90  # v2.1: было 0.85
        elif market_risk < 0.4:
            final_rating *= 1.05

        regime = symbol_regime.get("regime", "range")
        if regime == "chaos":
            final_rating *= 0.90  # v2.1: было 0.8
        elif regime == "trend":
            final_rating *= 1.05
        elif regime == "squeeze":
            final_rating *= 1.03

        memory_bias = memory_ctx.get("memory_bias")
        if memory_bias is not None:
            if memory_bias == direction_side:
                final_rating *= 1.05
            else:
                final_rating *= 0.95

        # HTF-тренд
        htf_trend_score = _compute_htf_trend_score(trend_15m, trend_1h, trend_4h)
        if htf_trend_score != 0:
            htf_abs = abs(htf_trend_score)
            same_direction_trend = (direction_side == "bullish" and htf_trend_score > 0) or (
                direction_side == "bearish" and htf_trend_score < 0
            )
            if same_direction_trend:
                if htf_abs > 2.0:
                    final_rating *= 1.07
                elif htf_abs > 1.0:
                    final_rating *= 1.03
            else:
                if htf_abs > 2.0:
                    final_rating *= 0.95  # v2.1: было 0.93
                elif htf_abs > 1.0:
                    final_rating *= 0.98  # v2.1: было 0.97

        # HTF-structure
        htf_structure = htf_structure_ctx.get("structure")
        htf_strength = htf_structure_ctx.get("strength")
        if htf_structure in ("bullish", "bearish") and isinstance(htf_strength, int):
            same_direction_struct = (direction_side == "bullish" and htf_structure == "bullish") or (
                direction_side == "bearish" and htf_structure == "bearish"
            )
            if same_direction_struct:
                if htf_strength == 2:
                    final_rating *= 1.02
                elif htf_strength == 3:
                    final_rating *= 1.03
                elif htf_strength == 4:
                    final_rating *= 1.05
                elif htf_strength >= 5:
                    final_rating *= 1.06
            else:
                if htf_strength == 2:
                    final_rating *= 0.98
                elif htf_strength == 3:
                    final_rating *= 0.97
                elif htf_strength == 4:
                    final_rating *= 0.97  # v2.1: было 0.95
                elif htf_strength >= 5:
                    final_rating *= 0.96  # v2.1: было 0.94

        # HTF-momentum
        htf_mom_div = htf_momentum_ctx.get("divergence")
        htf_mom_strength = htf_momentum_ctx.get("strength")
        if htf_mom_div in ("bullish", "bearish") and isinstance(htf_mom_strength, int):
            same_direction_mom = (direction_side == "bullish" and htf_mom_div == "bullish") or (
                direction_side == "bearish" and htf_mom_div == "bearish"
            )
            if same_direction_mom:
                if htf_mom_strength == 1:
                    final_rating *= 1.02
                elif htf_mom_strength == 2:
                    final_rating *= 1.03
                elif htf_mom_strength >= 3:
                    final_rating *= 1.05
            else:
                if htf_mom_strength == 1:
                    final_rating *= 0.98
                elif htf_mom_strength == 2:
                    final_rating *= 0.97
                elif htf_mom_strength >= 3:
                    final_rating *= 0.95

        # Volume profile
        profile_bias = volume_profile_ctx.get("profile_bias")
        profile_strength = volume_profile_ctx.get("strength")
        if profile_bias in ("support", "resistance") and isinstance(profile_strength, int):
            if profile_bias == "support":
                if direction_side == "bullish":
                    if profile_strength == 1:
                        final_rating *= 1.02
                    elif profile_strength == 2:
                        final_rating *= 1.03
                    elif profile_strength >= 3:
                        final_rating *= 1.05
                else:
                    if profile_strength == 1:
                        final_rating *= 0.98
                    elif profile_strength == 2:
                        final_rating *= 0.97
                    elif profile_strength >= 3:
                        final_rating *= 0.95
            elif profile_bias == "resistance":
                if direction_side == "bearish":
                    if profile_strength == 1:
                        final_rating *= 1.02
                    elif profile_strength == 2:
                        final_rating *= 1.03
                    elif profile_strength >= 3:
                        final_rating *= 1.05
                else:
                    if profile_strength == 1:
                        final_rating *= 0.98
                    elif profile_strength == 2:
                        final_rating *= 0.97
                    elif profile_strength >= 3:
                        final_rating *= 0.95

        final_rating_int = int(round(final_rating))

        self.update_symbol_memory(
            symbol=symbol,
            signal_type="generic",
            rating=final_rating_int,
            direction_side=direction_side,
            context={
                "base_rating": base_rating,
                "confidence": confidence,
                "regime": regime,
                "market_regime": market_ctx.get("market_regime"),
            },
        )

        return {
            "final_rating": final_rating_int,
            "confidence": confidence,
            "symbol_regime": symbol_regime,
            "market_ctx": market_ctx,
            "vol_cluster": vol_cluster,
            "memory_ctx": memory_ctx,
            "weights": weights,
            "filters_ok": filters_ok,
            "confidence_meta": conf["meta"],
            "htf_structure_ctx": htf_structure_ctx,
            "htf_momentum_ctx": htf_momentum_ctx,
            "volume_profile_ctx": volume_profile_ctx,
        }


# ==========================
# 11. Глобальный инстанс и совместимый API
# ==========================

_GLOBAL_FILTERS = SmartFilters()


def apply_smartfilters_v3(
    symbol: str,
    base_rating: int,
    direction_side: str,
    closes_1m: List[float],
    klines_1m: List[List[Any]],
    trend_score: float,
    trend_15m: float,
    trend_1h: float,
    trend_4h: float,
    liquidity_bias: Optional[str],
    noise_level: Optional[float],
    btc_ctx: Dict[str, Any],
    extra_filters_ok: Dict[str, bool],
    global_risk_proxy: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Совместимый с v3.0–v3.6 API.
    Внутри используется SmartFilters v3.7 LITE.
    """
    return _GLOBAL_FILTERS.apply_smartfilters_v3(
        symbol=symbol,
        base_rating=base_rating,
        direction_side=direction_side,
        closes_1m=closes_1m,
        klines_1m=klines_1m,
        trend_score=trend_score,
        trend_15m=trend_15m,
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        liquidity_bias=liquidity_bias,
        noise_level=noise_level,
        btc_ctx=btc_ctx,
        extra_filters_ok=extra_filters_ok,
        global_risk_proxy=global_risk_proxy,
    )
