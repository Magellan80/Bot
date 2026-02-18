import time
import asyncio
import aiohttp
from typing import Any, Dict
from io import BytesIO
import traceback
import pandas as pd
import mplfinance as mpf
from aiogram.types import BufferedInputFile

from config import get_current_mode, load_settings
from data_layer import (
    fetch_tickers,
    fetch_klines,
    fetch_open_interest,
    fetch_funding_rate,
    fetch_liquidations,
    fetch_recent_trades,
    fetch_orderbook,
)
from liquidity_map import build_liquidity_map

from strategy_selector import StrategySelector

strategy_selector = StrategySelector()

from asset_profile_engine import AssetProfileEngine

from symbol_memory import smooth_confidence

from context import (
    compute_trend_score,
    compute_risk_score,
    funding_bias,
    interpret_liquidations,
    analyze_flow_from_trades,
    analyze_delta_from_trades,
    format_funding_text,
    format_liq_text,
    format_flow_text,
    format_delta_text,
)
from detectors import Detector

detector = Detector()

asset_engine = AssetProfileEngine()

from microstructure import (
    build_price_buckets,
    analyze_microstructure,
)
from htf_structure import compute_htf_structure, detect_swings
from footprint import compute_footprint_zones

from smart_filters_v3 import apply_smartfilters_v3
from symbol_memory import (
    update_symbol_memory,
    get_symbol_memory,
    get_symbol_state,
    set_symbol_state,
    clear_symbol_state,
)

SYMBOL_COOLDOWN = 300
_last_signal_ts = {}

_BTC_CTX_CACHE = {
    "ts": 0.0,
    "factor": 1.0,
    "regime": "neutral",
}
# =====================================================
# GLOBAL DEBUG FLAGS
# =====================================================

DEBUG_ROUTER = True

# =====================================================
# SCREENER V3 MAX — MODE SYSTEM (Balanced Pro DEFAULT)
# =====================================================

SCREENER_MODE = "balanced"  # conservative | balanced | aggressive

MODE_PROFILES = {
    "conservative": {
        "min_score_shift": +5,
        "btc_trend_boost": 1.02,
        "btc_ranging_boost": 1.00,
        "btc_high_vol_factor": 0.85,
        "impulse_multiplier": 0.9,
        "memory_boost": 0.9,
        "reversal_bonus": 0.95,
        "elite_threshold": 88,
    },
    "balanced": {  # Balanced Pro MAX (DEFAULT)
        "min_score_shift": 0,
        "btc_trend_boost": 1.05,
        "btc_ranging_boost": 1.07,
        "btc_high_vol_factor": 0.95,
        "impulse_multiplier": 1.05,
        "memory_boost": 1.05,
        "reversal_bonus": 1.05,
        "elite_threshold": 85,
    },
    "aggressive": {
        "min_score_shift": -8,
        "btc_trend_boost": 1.10,
        "btc_ranging_boost": 1.10,
        "btc_high_vol_factor": 1.00,
        "impulse_multiplier": 1.15,
        "memory_boost": 1.10,
        "reversal_bonus": 1.15,
        "elite_threshold": 80,
    }
}


def get_mode_profile():
    return MODE_PROFILES.get(SCREENER_MODE, MODE_PROFILES["balanced"])


def set_screener_mode(mode: str):
    global SCREENER_MODE
    if mode in MODE_PROFILES:
        SCREENER_MODE = mode


def symbol_on_cooldown(symbol: str) -> bool:
    ts = _last_signal_ts.get(symbol)
    if ts is None:
        return False
    return (time.time() - ts) < SYMBOL_COOLDOWN


def mark_symbol_signal(symbol: str):
    _last_signal_ts[symbol] = time.time()


def log_signal(s: dict):
    with open("signals.log", "a", encoding="utf-8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{s['type']} | {s['symbol']} | price={s['price']:.4f} | "
            f"rating={s['rating']} | trend={s['trend_score']} | risk={s['risk_score']}\n"
        )


# v2.0: НОВАЯ ФУНКЦИЯ для диагностики
def log_blocked_signal(symbol: str, reason: str, details: str = ""):
    """Логирование блокированных сигналов для диагностики"""
    with open("blocked_signals.log", "a", encoding="utf-8") as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} | {symbol} | Blocked by: {reason}")
        if details:
            f.write(f" | {details}")
        f.write("\n")


def log_error(e: Exception):
    with open("errors.log", "a", encoding="utf-8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {repr(e)}\n"
        )
        f.write(traceback.format_exc() + "\n")


def apply_reversal_filters(signal_type, closes, highs, lows, volumes, delta_status):
    if len(closes) < 5:
        return 0

    c0, c1, c2 = closes[0], closes[1], closes[2]
    h0, h1 = highs[0], highs[1]
    l0, l1 = lows[0], lows[1]
    v0, v1 = volumes[0], volumes[1]

    adj = 0

    is_bullish = any(x in signal_type for x in ["Dump → Pump", "PUMP"])
    is_bearish = any(x in signal_type for x in ["Pump → Dump", "DUMP"])

    if is_bullish:
        if c0 > c1 and (c0 - c1) / max(c1, 1e-7) * 100 > 0.2:
            adj += 7
        if c0 < c1 and (c1 - c0) / max(c1, 1e-7) * 100 > 0.2:
            adj -= 5

    if is_bearish:
        if c0 < c1 and (c1 - c0) / max(c1, 1e-7) * 100 > 0.2:
            adj += 7
        if c0 > c1 and (c0 - c1) / max(c1, 1e-7) * 100 > 0.2:
            adj -= 5

    if is_bearish and h0 > h1:
        diff = (h0 - h1) / max(h1, 1e-7) * 100
        if 0.1 < diff < 0.4 and v0 < v1:
            adj += 5

    if is_bullish and l0 < l1:
        diff = (l1 - l0) / max(l1, 1e-7) * 100
        if 0.1 < diff < 0.4 and v0 < v1:
            adj += 5

    if is_bullish:
        if delta_status == "bullish":
            adj += 3
        elif delta_status == "bearish":
            adj -= 3

    if is_bearish:
        if delta_status == "bearish":
            adj += 3
        elif delta_status == "bullish":
            adj -= 3

    if is_bullish:
        if c2 > c1 and c0 > c1:
            diff = abs(c1 - c2) / max(c2, 1e-7) * 100
            if diff < 0.6:
                adj += 5

    if is_bearish:
        if c2 < c1 and c0 < c1:
            diff = abs(c1 - c2) / max(c2, 1e-7) * 100
            if diff < 0.6:
                adj += 5

    return adj


def generate_candle_chart(klines, symbol: str, timeframe_label: str = "15m"):
    if not klines:
        return None

    df = pd.DataFrame({
        "Open":   [float(c[1]) for c in klines],
        "High":   [float(c[2]) for c in klines],
        "Low":    [float(c[3]) for c in klines],
        "Close":  [float(c[4]) for c in klines],
        "Volume": [float(c[5]) for c in klines],
    })

    df.index = pd.to_datetime([int(c[0]) for c in klines], unit="ms")
    df = df.iloc[::-1]

    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc)

    buf = BytesIO()
    mpf.plot(df, type="candle", volume=True, style=style,
             title=f"{symbol} — {timeframe_label}", savefig=buf)
    buf.seek(0)
    return buf


def compute_htf_trend_from_klines(klines):
    """
    Bybit returns klines newest → oldest.
    Index 0 = latest closed candle.

    We compute trend as % change between:
    latest close (0)
    and N candles back (30 or last available).
    """

    if not klines or len(klines) < 20:
        return 0

    try:
        # ❗ НЕ переворачиваем массив
        closes = [float(c[4]) for c in klines]

        latest = closes[0]  # newest
        lookback_index = 30 if len(closes) > 30 else len(closes) - 1
        past = closes[lookback_index]

        if past <= 0:
            return 0

        change_pct = (latest - past) / past * 100

        if change_pct > 2:
            return 5
        if change_pct > 0.7:
            return 3
        if change_pct < -2:
            return -5
        if change_pct < -0.7:
            return -3

        return 0

    except Exception as e:
        log_error(e)
        return 0


def ema(values, period: int):
    if not values or period <= 1 or len(values) < period:
        return values[:]
    k = 2 / (period + 1)
    ema_vals = []
    prev = sum(values[:period]) / period
    ema_vals.extend(values[:period - 1])
    ema_vals.append(prev)
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        ema_vals.append(prev)
    return ema_vals


def compute_atr_from_klines(klines, period: int = 14) -> float:
    if not klines or len(klines) < period + 1:
        return 0.0

    highs = [float(c[2]) for c in klines][::-1]
    lows = [float(c[3]) for c in klines][::-1]
    closes = [float(c[4]) for c in klines][::-1]

    trs = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)

    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)

    atr = sum(trs[:period]) / period
    alpha = 1 / period
    for tr in trs[period:]:
        atr = alpha * tr + (1 - alpha) * atr
    return atr


def build_htf_liquidity_light(
    closes,
    highs,
    lows,
    structure: str,
):
    try:
        swings = detect_swings(closes, highs, lows)
        if not swings:
            return {
                "swing_highs": [],
                "swing_lows": [],
                "liquidity_bias": "balanced",
            }

        swing_highs, swing_lows = swings
        swing_highs = swing_highs[-3:]
        swing_lows = swing_lows[-3:]

        price = closes[-1] if closes else None

        bias = "balanced"
        if price is not None and swing_highs and swing_lows:
            last_high = swing_highs[-1][1]
            last_low = swing_lows[-1][1]

            dist_to_high = abs(last_high - price)
            dist_to_low = abs(price - last_low)

            if structure == "bullish":
                bias = "above" if dist_to_high < dist_to_low else "below"
            elif structure == "bearish":
                bias = "below" if dist_to_low < dist_to_high else "above"
            else:
                bias = "balanced"

        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "liquidity_bias": bias,
        }
    except Exception as e:
        log_error(e)
        return {
            "swing_highs": [],
            "swing_lows": [],
            "liquidity_bias": "balanced",
        }


def detect_momentum_divergence(closes, momentum):
    if len(closes) < 5 or len(momentum) < 5:
        return None

    price_hh = closes[-1] > closes[-2] > closes[-3]
    price_ll = closes[-1] < closes[-2] < closes[-3]

    mom_lh = momentum[-1] < momentum[-2] < momentum[-3]
    mom_hl = momentum[-1] > momentum[-2] > momentum[-3]

    if price_ll and mom_hl:
        return "bullish"
    if price_hh and mom_lh:
        return "bearish"

    return None


async def compute_htf_context(session, symbol: str):
    htf = {
        "trend_15m": 0,
        "trend_1h": 0,
        "trend_4h": 0,
        "structure_15m": "ranging",
        "structure_1h": "ranging",
        "structure_4h": "ranging",
        "event_15m": None,
        "event_1h": None,
        "event_4h": None,
        "strength_15m": 1,
        "strength_1h": 1,
        "strength_4h": 1,
        "momentum_1h": 0.0,
        "momentum_4h": 0.0,
        "momentum_strength_1h": 0.0,
        "momentum_strength_4h": 0.0,
        "momentum_div_1h": None,
        "momentum_div_4h": None,
        "vol_regime_1h": "normal",
        "vol_regime_4h": "normal",
        "htf_liquidity_1h": {
            "swing_highs": [],
            "swing_lows": [],
            "liquidity_bias": "balanced",
        },
        "htf_liquidity_4h": {
            "swing_highs": [],
            "swing_lows": [],
            "liquidity_bias": "balanced",
        },
    }
    try:
        kl_15m = await fetch_klines(session, symbol, interval="15", limit=96)
        kl_1h = await fetch_klines(session, symbol, interval="60", limit=96)
        kl_4h = await fetch_klines(session, symbol, interval="240", limit=96)

        htf["trend_15m"] = compute_htf_trend_from_klines(kl_15m)
        htf["trend_1h"] = compute_htf_trend_from_klines(kl_1h)
        htf["trend_4h"] = compute_htf_trend_from_klines(kl_4h)

        if kl_15m:
            s15 = compute_htf_structure(kl_15m)
            htf["structure_15m"] = s15.get("structure", "ranging")
            htf["event_15m"] = s15.get("event")
            htf["strength_15m"] = s15.get("strength", 1)
        if kl_1h:
            s1h = compute_htf_structure(kl_1h)
            htf["structure_1h"] = s1h.get("structure", "ranging")
            htf["event_1h"] = s1h.get("event")
            htf["strength_1h"] = s1h.get("strength", 1)
        if kl_4h:
            s4h = compute_htf_structure(kl_4h)
            htf["structure_4h"] = s4h.get("structure", "ranging")
            htf["event_4h"] = s4h.get("event")
            htf["strength_4h"] = s4h.get("strength", 1)

        def _compute_mom_vol(kl, key_prefix: str):
            if not kl or len(kl) < 30:
                return

            closes = [float(c[4]) for c in kl][::-1]
            highs = [float(c[2]) for c in kl][::-1]
            lows = [float(c[3]) for c in kl][::-1]

            ema5 = ema(closes, 5)
            ema20 = ema(closes, 20)
            if not ema5 or not ema20 or len(ema5) != len(closes) or len(ema20) != len(closes):
                return

            mom = ema5[-1] - ema20[-1]
            atr = compute_atr_from_klines(kl)
            price = closes[-1] if closes else 0.0

            if atr > 0:
                mom_strength = abs(mom) / atr
            else:
                mom_strength = 0.0

            if price > 0 and atr > 0:
                vol = atr / price * 100
            else:
                vol = 0.0

            if vol < 0.4:
                regime = "low_vol"
            elif vol < 1.2:
                regime = "normal"
            elif vol < 2.5:
                regime = "high_vol"
            else:
                regime = "chaotic"

            htf[f"momentum_{key_prefix}"] = mom
            htf[f"momentum_strength_{key_prefix}"] = mom_strength
            htf[f"vol_regime_{key_prefix}"] = regime

            div = detect_momentum_divergence(closes, ema5)
            htf[f"momentum_div_{key_prefix}"] = div

            structure = htf.get(f"structure_{key_prefix}", "ranging")
            liq = build_htf_liquidity_light(
                closes=closes,
                highs=highs,
                lows=lows,
                structure=structure,
            )
            htf[f"htf_liquidity_{key_prefix}"] = liq

        _compute_mom_vol(kl_1h, "1h")
        _compute_mom_vol(kl_4h, "4h")

    except Exception as e:
        log_error(e)

    return htf


def evaluate_orderbook_quality(orderbook: dict, last_price: float):
    try:
        bids = orderbook.get("b", []) or orderbook.get("bids", [])
        asks = orderbook.get("a", []) or orderbook.get("asks", [])
        if not bids or not asks:
            return False, {}

        bids_sorted = sorted(bids, key=lambda x: float(x[0]), reverse=True)
        asks_sorted = sorted(asks, key=lambda x: float(x[0]))

        best_bid = float(bids_sorted[0][0])
        best_ask = float(asks_sorted[0][0])

        if best_ask <= 0 or best_bid <= 0 or best_ask <= best_bid:
            return False, {}

        mid = (best_ask + best_bid) / 2
        spread_pct = (best_ask - best_bid) / mid * 100

        depth_n = 10
        bid_vol = sum(float(x[1]) for x in bids_sorted[:depth_n])
        ask_vol = sum(float(x[1]) for x in asks_sorted[:depth_n])

        total_vol = bid_vol + ask_vol

        max_spread_pct = 1.0  # v2.0: было 0.5
        if last_price > 100:
            min_total_vol = 150.0
        elif last_price > 10:
            min_total_vol = 80.0
        else:
            min_total_vol = 30.0

        ok = True
        if spread_pct > max_spread_pct:
            ok = False
        if total_vol < min_total_vol:
            ok = False

        return ok, {
            "spread_pct": spread_pct,
            "bid_vol_10": bid_vol,
            "ask_vol_10": ask_vol,
            "total_vol_10": total_vol,
        }
    except Exception as e:
        log_error(e)
        return False, {}

def compute_impulse_score(closes, volumes):
    try:
        if len(closes) < 5:
            return 0.0

        c0, c1, c2, c3 = closes[0], closes[1], closes[2], closes[3]
        v0, v1, v2 = volumes[0], volumes[1], volumes[2]

        d01 = (c0 - c1) / max(c1, 1e-7) * 100
        d12 = (c1 - c2) / max(c2, 1e-7) * 100
        d23 = (c2 - c3) / max(c3, 1e-7) * 100

        acc1 = d01 - d12
        acc2 = d12 - d23

        vol_avg = (v0 + v1 + v2) / 3
        vol_spike = (v0 - vol_avg) / max(vol_avg, 1e-7) * 100

        score = 0.0

        if abs(acc1) > 0.2 and abs(acc2) > 0.2 and vol_spike > 20:
            score += 5.0

        if abs(d01) < 0.05 and vol_spike < -20:
            score -= 3.0

        if score > 8:
            score = 8
        if score < -5:
            score = -5

        return score
    except Exception as e:
        log_error(e)
        return 0.0


def infer_direction_side(signal_type: str) -> str | None:
    if any(x in signal_type for x in ["Pump → Dump", "DUMP"]):
        return "bearish"
    if any(x in signal_type for x in ["Dump → Pump", "PUMP"]):
        return "bullish"
    return None


def apply_alignment_penalties(
    rating: float,
    direction_side: str | None,
    trend_1h: float,
    trend_4h: float,
    flow_status: str,
    impulse_score: float,
) -> float:
    adjusted = rating

    if direction_side == "bearish":
        if flow_status == "aggressive_buyers":
            adjusted *= 0.9
        if impulse_score > 2:
            adjusted *= 0.9

    elif direction_side == "bullish":
        if flow_status == "aggressive_sellers":
            adjusted *= 0.9
        if impulse_score < -2:
            adjusted *= 0.9

    return adjusted


def amplify_confidence(
    base_conf: float,
    rating: int,
    adaptive_min_score: int,
    btc_factor: float,
    trend_1h: float,
    trend_4h: float,
    direction_side: str | None,
) -> float:
    """
    Confidence Amplifier V10.3
    Усиливает или ослабляет confidence
    без изменения rating.
    """

    conf = base_conf

    # 1️⃣ Distance from threshold    
    distance = rating - adaptive_min_score

    if distance > 12:
        conf *= 1.05
    elif distance > 6:
        conf *= 1.03
    elif distance < 0:
        conf *= 0.92


    # 2️⃣ BTC regime stability
    if btc_factor > 1.08:
        conf *= 1.06
    elif btc_factor < 0.92:
        conf *= 0.94

    # 3️⃣ HTF alignment
    if direction_side == "bullish":
        if trend_1h > 0 and trend_4h > 0:
            conf *= 1.06
        elif trend_1h < 0 and trend_4h < 0:
            conf *= 0.94

    elif direction_side == "bearish":
        if trend_1h < 0 and trend_4h < 0:
            conf *= 1.06
        elif trend_1h > 0 and trend_4h > 0:
            conf *= 0.94

    # 4️⃣ Rating tier bonus
    if rating >= 85:
        conf *= 1.03
    elif rating < adaptive_min_score + 3:
        conf *= 0.95

    # =====================================
    # REALISTIC CONFIDENCE CLAMP
    # =====================================

    max_cap = 0.88
    min_cap = 0.40

    conf = max(min_cap, min(conf, max_cap))

    # если рейтинг слабый — режем
    if rating < adaptive_min_score + 5:
        conf *= 0.85

    return round(conf, 3)


def _get_reversal_state(symbol: str, now_ts: float, ttl_sec: int) -> Dict[str, Any]:
    state = get_symbol_state(symbol)
    if not state:
        return {}
    ts = state.get("ts")
    if not isinstance(ts, (int, float)):
        return {}
    if now_ts - ts > ttl_sec:
        clear_symbol_state(symbol)
        return {}
    return state


def _should_allow_reversal(
    direction: str,
    state: Dict[str, Any],
    requires_state: bool,
    min_score: int,
    rating: float,
    extra_min_bonus: int,
    min_delay_bars: int,
) -> bool:
    if not requires_state:
        return rating >= min_score
    if not state:
        return rating >= (min_score + extra_min_bonus)
    state_type = state.get("type")
    state_delay = int(state.get("delay_bars", 0))
    if state_delay < min_delay_bars:
        return False
    if direction == "Dump → Pump":
        return state_type == "dump"
    if direction == "Pump → Dump":
        return state_type == "pump"
    return False


def _passes_strict_reversal_filters(
    strictness_level: str,
    direction: str,
    flow_status: str,
    delta_status: str,
    event_1h: str | None,
    event_4h: str | None,
    structure_1h: str,
    structure_4h: str,
) -> bool:
    if strictness_level == "soft":
        return True

    require_structure = strictness_level == "strict"
    require_event = strictness_level == "strict"

    structure_ok = False
    event_ok = False

    if direction == "Dump → Pump":
        # v2.0: flow фильтры только в strict режиме
        if strictness_level == "strict" and (flow_status == "aggressive_sellers" or delta_status == "bearish"):
            return False
        structure_ok = structure_1h == "bullish" or structure_4h == "bullish"
        event_ok = event_1h in ("BOS", "CHOCH") or event_4h in ("BOS", "CHOCH")
    else:
        # v2.0: flow фильтры только в strict режиме
        if strictness_level == "strict" and (flow_status == "aggressive_buyers" or delta_status == "bullish"):
            return False
        structure_ok = structure_1h == "bearish" or structure_4h == "bearish"
        event_ok = event_1h in ("BOS", "CHOCH") or event_4h in ("BOS", "CHOCH")

    if strictness_level == "medium":
        return structure_ok or event_ok

    if require_structure and not structure_ok:
        return False
    if require_event and not event_ok:
        return False
    return True


async def compute_btc_stability(session):
    global _BTC_CTX_CACHE

    now = time.time()
    if now - _BTC_CTX_CACHE["ts"] < 30:
        return _BTC_CTX_CACHE

    try:
        kl = await fetch_klines(session, "BTCUSDT", interval="1", limit=60)

        if not kl or len(kl) < 20:
            _BTC_CTX_CACHE = {"ts": now, "factor": 1.0, "regime": "neutral"}
            return _BTC_CTX_CACHE

        # ❗ НЕ переворачиваем массив
        closes = [float(c[4]) for c in kl]

        latest = closes[0]
        lookback_index = 20 if len(closes) > 20 else len(closes) - 1
        past = closes[lookback_index]

        if past <= 0:
            _BTC_CTX_CACHE = {"ts": now, "factor": 1.0, "regime": "neutral"}
            return _BTC_CTX_CACHE

        change_pct = (latest - past) / past * 100

        # средняя внутриминутная волатильность
        diffs = []
        for i in range(0, lookback_index):
            prev = closes[i + 1]
            curr = closes[i]
            if prev > 0:
                diffs.append(abs(curr - prev) / prev * 100)

        volat = sum(diffs) / len(diffs) if diffs else 0.0

        regime = "neutral"
        factor = 1.0

        if volat > 1.5:
            factor = 0.85
            regime = "high_vol"

        if abs(change_pct) > 1.0 and volat < 2.0:
            factor = 1.05
            regime = "trending"

        if abs(change_pct) < 0.3 and volat < 0.7:
            factor = 1.15
            regime = "ranging"

        _BTC_CTX_CACHE = {
            "ts": now,
            "factor": factor,
            "regime": regime
        }

        return _BTC_CTX_CACHE

    except Exception as e:
        log_error(e)
        _BTC_CTX_CACHE = {"ts": now, "factor": 1.0, "regime": "neutral"}
        return _BTC_CTX_CACHE

async def analyze_symbol_async(session, symbol: str, min_score: int, ticker_info: dict):

    filters_ok_ratio = 0.0

    print(f"[ENTER ANALYZE] {symbol}")

    settings = load_settings()

    now_ts = time.time()

    reversal_state_ttl_sec = int(settings.get("reversal_state_ttl_sec", 7200))

    strictness_level = str(settings.get("strictness_level", "soft")).lower()
    if strictness_level not in ("soft", "medium", "strict"):
        strictness_level = "strict"

    mode_key, mode_cfg = get_current_mode()
    mode_profile = get_mode_profile()
    min_score = max(0, min_score + mode_profile["min_score_shift"])

    # ------------------------
    # COOLDOWN CHECK
    # ------------------------
    if symbol_on_cooldown(symbol):
        print(f"[CUT] {symbol} ON COOLDOWN")
        return None

    # ------------------------
    # TURNOVER CHECK
    # ------------------------
    turnover = float(ticker_info.get("turnover24h", 0))
    if turnover < mode_cfg["volume_usdt"]:
        print(f"[CUT] {symbol} LOW TURNOVER {turnover} < {mode_cfg['volume_usdt']}")
        return None

    # ------------------------
    # FETCH 1M DATA
    # ------------------------
    klines_1m = await fetch_klines(session, symbol, interval="1", limit=80)

    # 🔵 FETCH 15M DATA FOR CHART
    klines_15m = await fetch_klines(session, symbol, interval="15", limit=80)

    if not klines_15m or len(klines_15m) < 40:
        return None


    if not klines_1m:
        print(f"[CUT] {symbol} NO KLINES")
        return None

    if len(klines_1m) < 40:
        print(f"[CUT] {symbol} NOT ENOUGH CANDLES {len(klines_1m)}")
        return None

    closes_1m = [float(c[4]) for c in klines_1m]
    highs_1m = [float(c[2]) for c in klines_1m]
    lows_1m = [float(c[3]) for c in klines_1m]
    volumes_1m = [float(c[5]) for c in klines_1m]

    latest_close = closes_1m[0]
    last_price = latest_close


    # =====================================================
    # PHASE 1 — LIGHT PRECHECK (EARLY FILTER)
    # =====================================================

    light_rev = detector.analyze_reversal(
        closes_1m,
        highs_1m,
        lows_1m,
        volumes_1m,
        htf_trend_1h=0,
        htf_trend_4h=0,
        structure_1h="ranging",
        structure_4h="ranging",
        event_1h=None,
        event_4h=None,
        market_regime="neutral",
        asset_class="mid",
        min_score=0,
    )

    light_rating = 0
    if light_rev:
        light_rating = light_rev.get("rating", 0)

    # 🔒 Early rejection (если слабый сигнал — дальше не идём)
    if light_rating < 55:
        return None

    orderbook = await fetch_orderbook(session, symbol, limit=50)
    liquidity = build_liquidity_map(orderbook, last_price)

    liq_bias = liquidity.get("bias")
    liq_strongest = liquidity.get("strongest_zone")
    liq_vac_up = liquidity.get("vacuum_up", 0)
    liq_vac_down = liquidity.get("vacuum_down", 0)

    ob_ok, ob_meta = evaluate_orderbook_quality(orderbook, last_price)
    if not ob_ok:
        # v2.0: Логируем почему заблокировано
        log_blocked_signal(symbol, "orderbook_quality", 
                          f"spread={ob_meta.get('spread_pct', 0):.2f}%, vol={ob_meta.get('total_vol_10', 0):.0f}")
        return None

    oi_now, oi_prev = await fetch_open_interest(session, symbol)
    oi_status = None
    if oi_now is not None and oi_prev is not None:
        if oi_now > oi_prev * 1.03:
            oi_status = "rising"
        elif oi_now < oi_prev * 0.97:
            oi_status = "falling"
        else:
            oi_status = "flat"

    funding_rate = await fetch_funding_rate(session, symbol)

    long_liq, short_liq = await fetch_liquidations(session, symbol, minutes=15)
    liq_status = interpret_liquidations(long_liq, short_liq)

    trades = await fetch_recent_trades(session, symbol, limit=200)
    flow_status = analyze_flow_from_trades(trades)
    delta_status = analyze_delta_from_trades(trades)

    clusters = build_price_buckets(trades)
    micro = analyze_microstructure(clusters, last_price)

    current_high = highs_1m[0]
    current_low = lows_1m[0]
    footprint_zones = compute_footprint_zones(trades, current_high, current_low)

    trend_score = await compute_trend_score(session, symbol)

    # =====================================================
    # HTF OPTIMIZATION V2 — CONDITIONAL LOADING
    # =====================================================

    USE_HTF_THRESHOLD = 60

    if light_rating < USE_HTF_THRESHOLD:
        # ⚡ Skip HTF loading (экономим 3 REST запроса)
        trend_15m = trend_1h = trend_4h = 0
        structure_15m = structure_1h = structure_4h = "ranging"
        event_15m = event_1h = event_4h = None
        strength_15m = strength_1h = strength_4h = 0

        momentum_1h = momentum_4h = 0
        momentum_strength_1h = momentum_strength_4h = 0
        vol_regime_1h = vol_regime_4h = "normal"
        momentum_div_1h = momentum_div_4h = None

        htf_liq_1h = {}
        htf_liq_4h = {}

    else:
        # 🧠 Загружаем HTF только для сильных сигналов
        htf = await compute_htf_context(session, symbol)

        trend_15m = htf["trend_15m"]
        trend_1h = htf["trend_1h"]
        trend_4h = htf["trend_4h"]

        structure_15m = htf["structure_15m"]
        structure_1h = htf["structure_1h"]
        structure_4h = htf["structure_4h"]

        event_15m = htf["event_15m"]
        event_1h = htf["event_1h"]
        event_4h = htf["event_4h"]

        strength_15m = htf["strength_15m"]
        strength_1h = htf["strength_1h"]
        strength_4h = htf["strength_4h"]

        momentum_1h = htf.get("momentum_1h", 0.0)
        momentum_4h = htf.get("momentum_4h", 0.0)
        momentum_strength_1h = htf.get("momentum_strength_1h", 0.0)
        momentum_strength_4h = htf.get("momentum_strength_4h", 0.0)

        vol_regime_1h = htf.get("vol_regime_1h", "normal")
        vol_regime_4h = htf.get("vol_regime_4h", "normal")

        momentum_div_1h = htf.get("momentum_div_1h")
        momentum_div_4h = htf.get("momentum_div_4h")

        htf_liq_1h = htf.get("htf_liquidity_1h", {}) or {}
        htf_liq_4h = htf.get("htf_liquidity_4h", {}) or {}


    risk_score = compute_risk_score(
        closes_1m,
        oi_status,
        funding_rate,
        liq_status,
        flow_status,
        delta_status,
        trend_score,
    )


    habr = detector.analyze_habr(closes_1m, highs_1m, lows_1m, volumes_1m)

    impulse_score = compute_impulse_score(closes_1m, volumes_1m)


    # =====================================================
    # BTC CONTEXT
    # =====================================================

    btc_ctx = await compute_btc_stability(session)

    btc_regime = btc_ctx.get("regime", "neutral")
    btc_factor = btc_ctx.get("factor", 1.0)

    # 🔒 HARD CLAMP (critical stabilization)
    btc_factor = max(0.85, min(btc_factor, 1.15))


    # =====================================================
    # REVERSAL STATE
    # =====================================================

    reversal_state = _get_reversal_state(symbol, now_ts, reversal_state_ttl_sec)


    # =====================================================
    # ASSET PROFILE ENGINE
    # =====================================================

    
    asset_profile = asset_engine.analyze(
        closes_1m,
        highs_1m,
        lows_1m,
    )

    asset_class = asset_profile.get("asset_class", "mid")


    # =====================================================
    # STRATEGY SELECTION
    # =====================================================

    strategy = strategy_selector.choose(btc_regime, asset_class)

    base_min_score = strategy_selector.get_min_score(
        btc_regime,
        asset_class
    )

    # 🔥 Adaptive threshold (stable formula)
    adaptive_min_score = int(
        (base_min_score + mode_profile["min_score_shift"]) * (2 - btc_factor)
    )

    # 🔒 Safety clamp for threshold
    adaptive_min_score = max(54, min(adaptive_min_score, 74))

    print(
        f"[MIN SCORE] {symbol} "
        f"base={base_min_score} "
        f"btc_factor={btc_factor:.3f} "
        f"adaptive={adaptive_min_score}"
    )


    # =====================================================
    # DETECTION PHASE (NO INTERNAL FILTER)
    # =====================================================

    rev = {"reversal": None, "rating": 0}

    if strategy == "reversal":

        result = detector.analyze_reversal(
            closes_1m,
            highs_1m,
            lows_1m,
            volumes_1m,
            htf_trend_1h=trend_1h,
            htf_trend_4h=trend_4h,
            structure_1h=structure_1h,
            structure_4h=structure_4h,
            event_1h=event_1h,
            event_4h=event_4h,
            market_regime=btc_regime,
            asset_class=asset_class,
            min_score=0,  # ❗ disabled internal filter
        )

        if result:
            rev = result


    elif strategy == "continuation":

        cont = detector.analyze_continuation(
            closes_1m,
            highs_1m,
            lows_1m,
            volumes_1m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            asset_class=asset_class,
            market_regime=btc_regime,
            min_score=0,
        )

        if cont:
            rev = {
                "reversal": cont.get("direction"),
                "rating": cont.get("rating", 0),
            }


    # -----------------------------------------------------
    # SAFETY DEFAULTS (prevents UnboundLocalError)
    # -----------------------------------------------------
    filters_ok_ratio = locals().get("filters_ok_ratio", 0.0)


    # =====================================================
    # ADAPTIVE SCORING ENGINE V10.2
    # =====================================================

    raw_rating = rev.get("rating", 0)

    if DEBUG_ROUTER:
        print("Raw rating BEFORE weighting:", raw_rating)

    # 🎯 Base smoothing (уменьшаем резкие скачки)
    weighted_rating = raw_rating * 0.85 + 15

    if rev.get("reversal"):

        # 1️⃣ Regime multiplier
        if btc_regime == "trending":
            weighted_rating *= 1.07
        elif btc_regime == "ranging":
            weighted_rating *= 1.05
        elif btc_regime == "high_vol":
            weighted_rating *= 0.95

        # 2️⃣ State reinforcement
        if reversal_state:
            state_type = reversal_state.get("type")
            if strategy == "reversal" and state_type in ("pump", "dump"):
                weighted_rating *= 1.07

        # 3️⃣ Volatility normalization
        if len(highs_1m) >= 20:
            recent_range = max(highs_1m[:20]) - min(lows_1m[:20])
            avg_price = sum(closes_1m[:20]) / 20

            volatility_pct = (recent_range / avg_price) * 100 if avg_price > 0 else 0

            if volatility_pct < 0.35:
                weighted_rating *= 0.96
            elif volatility_pct > 1.3:
                weighted_rating *= 1.03

        # 4️⃣ Liquidity proximity
        direction = rev.get("reversal")

        if direction in ("Dump → Pump", "bullish") and liq_bias == "below":
            weighted_rating *= 1.04

        if direction in ("Pump → Dump", "bearish") and liq_bias == "above":
            weighted_rating *= 1.04

        # 5️⃣ Asset confidence bonus
        if asset_class == "major":
            weighted_rating *= 1.03


        # ---------------------------------------------------
        # FIX: do not over-suppress reversal in ranging
        # ---------------------------------------------------
        if btc_regime == "ranging" and strategy == "reversal":
            weighted_rating *= 1.05


        # ================================
        # SmartFilters modulation (quality damping)
        # ================================
        # Если filters слабые — уменьшаем общий рейтинг
        quality_multiplier = 0.85 + (filters_ok_ratio * 0.3)
        weighted_rating *= quality_multiplier


    # FINAL CLAMP (всегда выполняется)
    # 🔒 Финальный демпфер
    weighted_rating = int(
        max(0, min(weighted_rating * 0.95 + raw_rating * 0.05, 100))
    )

    print(f"[RAW PASS] {symbol} weighted={weighted_rating} min={adaptive_min_score}")

    # ✅ SINGLE FINAL THRESHOLD CHECK
    if weighted_rating >= adaptive_min_score:
        rev["rating"] = weighted_rating
    else:
        print(f"[CUT] {symbol} BELOW THRESHOLD weighted={weighted_rating} < {adaptive_min_score}")
        rev = {"reversal": None, "rating": weighted_rating}


    # ================================
    # DEBUG RAW PASS
    # ================================

    if rev.get("reversal"):
        print(f"[DEBUG PASS RAW] {symbol} rating={weighted_rating} "
            f"adaptive_min={adaptive_min_score}")


    # =====================================================
    # END V10.2 ROUTER
    # =====================================================
    # =====================================================
    # DEBUG V10.2 ROUTER
    # =====================================================

    if DEBUG_ROUTER:
        print("\n========== ROUTER DEBUG ==========")
        print("Symbol:", symbol)
        print("BTC regime:", btc_regime)
        print("Strategy:", strategy)
        print("Asset class:", asset_class)
        print("Base min score:", base_min_score)
        print("Adaptive min score:", adaptive_min_score)

        print("Raw rating:", raw_rating)
        print("Weighted rating:", weighted_rating)
        print("Direction:", rev.get("reversal"))

        if rev.get("reversal"):
            print("Reversal passed raw detector")
        else:
            print("No reversal signal")

        print("==================================\n")

    
    # =====================================================
    # FINAL SIGNAL BUILDING
    # =====================================================

    candidates = []

    if rev.get("reversal"):

        direction = rev.get("reversal")

        if direction in ("bullish", "Dump → Pump"):
            direction_label = "Dump → Pump"
            direction_side = "bullish"
        else:
            direction_label = "Pump → Dump"
            direction_side = "bearish"

        candidates.append({
            "symbol": symbol,
            "type": f"{strategy.upper()} {direction_label} ({mode_key})",
            "emoji": "🔵" if strategy == "reversal" else "🟢",
            "price": last_price,
            "rating": rev["rating"],
            "oi": oi_status,
            "funding": funding_rate,
            "liq": liq_status,
            "flow": flow_status,
            "delta": delta_status,
            "trend_score": trend_score,
            "risk_score": risk_score,
            "trend_15m": trend_15m,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "meta_klines_15m": klines_15m,

        })

    if not candidates:
        print(f"[CUT] {symbol} NO CANDIDATES AFTER ROUTER")
        return None


    # =====================================================
    # SMART FILTERS V3
    # =====================================================

    symbol_mem = get_symbol_memory(symbol)
    symbol_profile = symbol_mem.get("profile", {}) if symbol_mem else {}

    for c in candidates:

        sf3 = apply_smartfilters_v3(
            symbol=symbol,
            base_rating=int(c["rating"]),
            direction_side=infer_direction_side(c["type"]),
            closes_1m=closes_1m,
            klines_1m=klines_1m,
            trend_score=trend_score,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            liquidity_bias=liq_bias,
            noise_level=None,
            btc_ctx=btc_ctx,
            extra_filters_ok={"min_score_ok": c["rating"] >= adaptive_min_score},
            global_risk_proxy=None,
        )

        # ================================
        # APPLY SMART FILTER RATING
        # ================================

        c["rating"] = sf3["final_rating"]

        direction_side = infer_direction_side(c["type"])

        # ================================
        # ALIGNMENT PENALTIES
        # ================================

        aligned = apply_alignment_penalties(
            rating=c["rating"],
            direction_side=direction_side,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            flow_status=flow_status,
            impulse_score=impulse_score,
        )

        c["rating"] = max(0, min(int(aligned), 100))

        # ================================
        # CONFIDENCE PIPELINE V10.3
        # ================================

        raw_conf = sf3["confidence"]
        smoothed_conf = smooth_confidence(symbol, raw_conf)

        amplified_conf = amplify_confidence(
            base_conf=smoothed_conf,
            rating=c["rating"],
            adaptive_min_score=adaptive_min_score,
            btc_factor=btc_factor,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            direction_side=direction_side,
        )

        c["confidence"] = amplified_conf

        # ================================
        # DEBUG SMART FILTERS
        # ================================

        alignment_score = sf3.get("alignment_score", 0)
        filters_ok_ratio = sf3.get("filters_ok_ratio", 0)

        print(f"[FILTER CHECK] {symbol} filters={filters_ok_ratio}")

        print(
            f"[DEBUG] {symbol} | "
            f"light={light_rating:.1f} | "
            f"conf={amplified_conf:.2f} | "
            f"align={alignment_score:.2f} | "
            f"filters={filters_ok_ratio:.2f}"
        )

        # ================================
        # META
        # ================================

        c["symbol_regime"] = sf3["symbol_regime"]
        c["market_ctx"] = sf3["market_ctx"]
    
    # ================================
    # FINALIZE BEST SIGNAL
    # ================================

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["rating"], reverse=True)
    best = candidates[0]

    atr_1m = compute_atr_from_klines(klines_1m)

    snapshot = {
        "atr_1m": atr_1m,
        "trend_score": trend_score,
        "btc_factor": btc_factor,
    }

    updated_mem = update_symbol_memory(symbol, snapshot)
    best["symbol_memory"] = updated_mem

    log_signal(best)
    mark_symbol_signal(symbol)

    return best

async def scanner_loop(send_text, send_photo, min_score: int, engine=None):

    async with aiohttp.ClientSession() as session:
        print("🚀 scanner_loop started")

        while True:
            print("🔄 scanning iteration...")

            try:
                tickers = await fetch_tickers(session)

                # 🔥 сортировка по объёму
                tickers = sorted(
                    tickers,
                    key=lambda x: float(x.get("turnover24h", 0)),
                    reverse=True
                )

                usdt_tickers = [
                    t for t in tickers
                    if t.get("symbol", "").endswith("USDT")
                ]

                # ==========================================
                # DYNAMIC SCAN DEPTH
                # ==========================================

                if min_score >= 70:
                    scan_limit = 40
                elif min_score >= 60:
                    scan_limit = 70
                else:
                    scan_limit = 100

                # ==========================================
                # LIGHT PRE-FILTER
                # ==========================================

                filtered = []

                for t in usdt_tickers:
                    try:
                        turnover = float(t.get("turnover24h", 0))
                        price_change = abs(float(t.get("price24hPcnt", 0)))

                        if turnover < 1_500_000:
                            continue

                        if price_change < 0.002:
                            continue

                        filtered.append(t)

                    except Exception:
                        continue

                symbols = [
                    (t["symbol"], t)
                    for t in filtered[:scan_limit]
                ]

                print(f"Scan depth: {scan_limit} symbols")
                print(f"Scanning {len(symbols)} symbols")

                # ==========================================
                # SEMAPHORE (RATE LIMIT STABILITY)
                # ==========================================

                semaphore = asyncio.Semaphore(10)

                async def limited_analyze(symbol, tinfo):
                    async with semaphore:
                        return await analyze_symbol_async(
                            session,
                            symbol,
                            min_score,
                            tinfo
                        )

                tasks = [
                    limited_analyze(s, tinfo)
                    for s, tinfo in symbols
                ]

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=15
                    )
                except asyncio.TimeoutError:
                    print("⚠️ Symbol analysis timeout — skipping iteration")
                    await asyncio.sleep(1)
                    continue

                for r in results:
                    if isinstance(r, Exception):
                        print("Symbol error:", repr(r))

                # 🔒 Remove None and exceptions
                results = [
                    r for r in results
                    if isinstance(r, dict) and r.get("type")
                ]


                # ==========================================
                # QUALITY FILTER
                # ==========================================

                # 🔥 Dynamic confidence threshold
                if min_score >= 70:
                    MIN_SIGNAL_CONFIDENCE = 0.60
                elif min_score >= 60:
                    MIN_SIGNAL_CONFIDENCE = 0.55
                else:
                    MIN_SIGNAL_CONFIDENCE = 0.50

                if min_score >= 70:
                    SIGNALS_PER_ITERATION_LIMIT = 1
                elif min_score >= 60:
                    SIGNALS_PER_ITERATION_LIMIT = 2
                else:
                    SIGNALS_PER_ITERATION_LIMIT = 3

                print(f"Signal limit: {SIGNALS_PER_ITERATION_LIMIT}")
                print(f"Min confidence required: {MIN_SIGNAL_CONFIDENCE}")

                # 🔎 DEBUG CONF
                for r in results:
                    if isinstance(r, dict) and r.get("type"):
                        print(
                            f"[CONF CHECK] {r['symbol']} "
                            f"conf={r.get('confidence'):.3f} "
                            f"min_conf={MIN_SIGNAL_CONFIDENCE}"
                        )

                signals = [
                    r for r in results
                    if isinstance(r, dict)
                    and r.get("type")
                    and r.get("confidence", 0) >= MIN_SIGNAL_CONFIDENCE
                ]

                print(f"Signals after quality filter: {len(signals)}")

                if signals:
                    signals = sorted(
                        signals,
                        key=lambda x: x.get("rating", 0),
                        reverse=True
                    )[:SIGNALS_PER_ITERATION_LIMIT]

                    print(f"Signals found: {len(signals)}")

                    for s in signals:

                        # ==================================
                        # STEP 4 — CHECK 15m KLINES (FIX)
                        # ==================================

                        if not s.get("meta_klines_15m"):
                            print(f"[CHART CUT] {s['symbol']} no 15m klines")
                            continue

                        print(f"Sending signal {s['symbol']} rating={s['rating']}")

                        text = (
                            f"{s['emoji']} {s['type']} — {s['symbol']}\n\n"
                            f"💰 Цена: {s['price']:.4f} USDT\n"
                            f"📊 Рейтинг: {s['rating']}/100\n"
                            f"⚡ Confidence: {s.get('confidence','n/a')}\n\n"
                            f"📈 Trend score: {s.get('trend_score')}\n"
                            f"📉 Risk score: {s.get('risk_score')}\n\n"
                            f"🧠 Flow: {s.get('flow')}\n"
                            f"🔀 Delta: {s.get('delta')}\n"
                            f"💦 Liquidity: {s.get('liq')}\n"
                            f"🏦 OI: {s.get('oi')}\n"
                            f"💸 Funding: {s.get('funding')}\n\n"
                            f"🕒 15m: {s.get('trend_15m')}\n"
                            f"🕒 1h: {s.get('trend_1h')}\n"
                            f"🕒 4h: {s.get('trend_4h')}\n\n"
                            f"🧬 Market ctx: {s.get('market_ctx')}\n"
                            f"🧠 Memory regime: {s.get('symbol_regime')}"
                        )

                        chart = generate_candle_chart(
                            klines=s.get("meta_klines_15m"),
                            symbol=s["symbol"],
                            timeframe_label="15m"
                        )

                        if chart:
                            photo = BufferedInputFile(
                                chart.getvalue(),
                                filename=f"{s['symbol']}.png"
                            )
                            await send_photo(photo, text)
                        else:
                            await send_text(text)

                        if engine is not None:
                            await engine.on_signal(s)

                await asyncio.sleep(1)

            except Exception as e:
                print("Scanner loop error:", repr(e))
                log_error(e)
                await asyncio.sleep(5)
