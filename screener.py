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
from detectors import (
    detect_big_pump,
    detect_big_dump,
    detect_pump_reversal,
    adjust_rating_with_context,
    detector,
)
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
    if not klines or len(klines) < 20:
        return 0
    closes = [float(c[4]) for c in klines][::-1]
    c0 = closes[0]
    cN = closes[min(len(closes) - 1, 30)]
    if cN <= 0:
        return 0
    change_pct = (c0 - cN) / cN * 100
    if change_pct > 2:
        return 5
    if change_pct > 0.7:
        return 3
    if change_pct < -2:
        return -5
    if change_pct < -0.7:
        return -3
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

        max_spread_pct = 0.5
        min_total_vol = 500.0

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
        if trend_1h > 2 or trend_4h > 2:
            adjusted *= 0.9
        if flow_status == "aggressive_buyers":
            adjusted *= 0.9
        if impulse_score > 2:
            adjusted *= 0.9
    elif direction_side == "bullish":
        if trend_1h < -2 or trend_4h < -2:
            adjusted *= 0.9
        if flow_status == "aggressive_sellers":
            adjusted *= 0.9
        if impulse_score < -2:
            adjusted *= 0.9
    return adjusted


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
        if flow_status == "aggressive_sellers" or delta_status == "bearish":
            return False
        structure_ok = structure_1h == "bullish" or structure_4h == "bullish"
        event_ok = event_1h in ("BOS", "CHOCH") or event_4h in ("BOS", "CHOCH")
    else:
        if flow_status == "aggressive_buyers" or delta_status == "bullish":
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

        closes = [float(c[4]) for c in kl][::-1]
        c0 = closes[0]
        cN = closes[20]

        change_pct = (c0 - cN) / max(cN, 1e-7) * 100
        diffs = [abs(closes[i] - closes[i + 1]) / max(closes[i + 1], 1e-7) * 100 for i in range(0, 20)]
        volat = sum(diffs) / len(diffs)

        regime = "neutral"
        factor = 1.0

        if volat > 1.5:
            factor = 0.8
            regime = "high_vol"

        if abs(change_pct) > 1.0 and volat < 2.0:
            factor = 1.05
            regime = "trending"

        if abs(change_pct) < 0.3 and volat < 0.7:
            factor = 1.1
            regime = "ranging"

        _BTC_CTX_CACHE = {"ts": now, "factor": factor, "regime": regime}
        return _BTC_CTX_CACHE
    except Exception as e:
        log_error(e)
        _BTC_CTX_CACHE = {"ts": now, "factor": 1.0, "regime": "neutral"}
        return _BTC_CTX_CACHE


async def analyze_symbol_async(session, symbol: str, min_score: int, ticker_info: dict):
    settings = load_settings()
    strictness_level = str(settings.get("strictness_level", "strict")).lower()
    if strictness_level not in ("soft", "medium", "strict"):
        strictness_level = "strict"
    reversal_requires_state = bool(settings.get("reversal_requires_state", True))
    reversal_state_ttl_sec = int(settings.get("reversal_state_ttl_sec", 7200))
    reversal_min_score_bonus = int(settings.get("reversal_min_score_bonus", 10))
    reversal_min_delay_bars = int(settings.get("reversal_min_delay_bars", 3))
    mode_key, mode_cfg = get_current_mode()
    now_ts = time.time()

    if symbol_on_cooldown(symbol):
        return None

    turnover = float(ticker_info.get("turnover24h", 0))
    if turnover < mode_cfg["volume_usdt"]:
        return None

    klines_1m = await fetch_klines(session, symbol, interval="1", limit=80)
    if not klines_1m or len(klines_1m) < 40:
        return None

    closes_1m = [float(c[4]) for c in klines_1m]
    highs_1m = [float(c[2]) for c in klines_1m]
    lows_1m = [float(c[3]) for c in klines_1m]
    volumes_1m = [float(c[5]) for c in klines_1m]

    latest_close = closes_1m[1] if len(closes_1m) > 1 else closes_1m[0]
    last_price = latest_close

    orderbook = await fetch_orderbook(session, symbol, limit=50)
    liquidity = build_liquidity_map(orderbook, last_price)

    liq_bias = liquidity.get("bias")
    liq_strongest = liquidity.get("strongest_zone")
    liq_vac_up = liquidity.get("vacuum_up", 0)
    liq_vac_down = liquidity.get("vacuum_down", 0)

    ob_ok, ob_meta = evaluate_orderbook_quality(orderbook, last_price)
    if not ob_ok:
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

    is_pump, pump_rating = detect_big_pump(closes_1m, highs_1m, lows_1m, volumes_1m, mode_cfg)
    is_dump, dump_rating = detect_big_dump(closes_1m, highs_1m, lows_1m, volumes_1m, mode_cfg)
    is_rev_pump, rev_pump_rating = detect_pump_reversal(closes_1m, highs_1m, lows_1m, volumes_1m)

    rev = detector.analyze(closes_1m, highs_1m, lows_1m, volumes_1m)
    habr = detector.analyze_habr(closes_1m, highs_1m, lows_1m, volumes_1m)

    impulse_score = compute_impulse_score(closes_1m, volumes_1m)

    btc_ctx = await compute_btc_stability(session)
    btc_factor = btc_ctx["factor"]
    btc_regime = btc_ctx["regime"]
    reversal_state = _get_reversal_state(symbol, now_ts, reversal_state_ttl_sec)

    candidates = []

    # === HABR STRATEGY ===
    if habr and habr.get("rating", 0) >= min_score:
        direction = "Dump → Pump" if habr["direction"] == "bullish" else "Pump → Dump"
        habr_rating = habr["rating"]

        if habr["direction"] == "bullish" and trend_score < -3:
            habr_rating *= 0.7
        if habr["direction"] == "bearish" and trend_score > 3:
            habr_rating *= 0.7

        if habr["direction"] == "bullish":
            if trend_1h < -2 or trend_4h < -2:
                habr_rating *= 0.55
            elif trend_1h > 2 and trend_4h >= 0:
                habr_rating *= 1.15
        else:
            if trend_1h > 2 or trend_4h > 2:
                habr_rating *= 0.55
            elif trend_1h < -2 and trend_4h <= 0:
                habr_rating *= 1.15

        fbias = funding_bias(funding_rate) if funding_rate is not None else None
        if fbias == "bullish" and habr["direction"] == "bearish":
            habr_rating *= 0.85
        if fbias == "bearish" and habr["direction"] == "bullish":
            habr_rating *= 0.85

        if oi_status == "falling":
            habr_rating *= 0.9

        if habr["direction"] == "bullish" and delta_status == "bearish":
            habr_rating *= 0.9
        if habr["direction"] == "bearish" and delta_status == "bullish":
            habr_rating *= 0.9

        habr_adj = adjust_rating_with_context(
            habr_rating,
            f"HABR {direction}",
            closes_1m,
            oi_now,
            oi_prev,
            funding_rate,
            liq_status,
            flow_status,
            delta_status,
            trend_score,
            risk_score,
        )

        if (
            habr_adj >= min_score
            and rev.get("reversal")
            and rev.get("rating", 0) >= min_score
            and (
                (habr["direction"] == "bullish" and rev["reversal"] == "bullish")
                or (habr["direction"] == "bearish" and rev["reversal"] == "bearish")
            )
        ):
            rev_direction = "Dump → Pump" if rev["reversal"] == "bullish" else "Pump → Dump"
            rev_adj = adjust_rating_with_context(
                rev["rating"],
                f"REVERSAL {rev_direction}",
                closes_1m,
                oi_now,
                oi_prev,
                funding_rate,
                liq_status,
                flow_status,
                delta_status,
                trend_score,
                risk_score,
            )
            hybrid_rating = int((habr_adj + rev_adj) / 2 + 15)

            candidates.append({
                "symbol": symbol,
                "type": f"HYBRID Habr+Reversal {direction} ({mode_key})",
                "emoji": "💎",
                "price": last_price,
                "rating": hybrid_rating,
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
                "meta_closes": closes_1m,
                "meta_highs": highs_1m,
                "meta_lows": lows_1m,
                "liq_map_bias": liq_bias,
                "liq_map_strongest": liq_strongest,
                "liq_map_vac_up": liq_vac_up,
                "liq_map_vac_down": liq_vac_down,
            })

            pump_side = None
            if is_pump and pump_rating >= min_score:
                pump_side = "bullish"
            if is_dump and dump_rating >= min_score:
                pump_side = "bearish"

            if pump_side and (
                (pump_side == "bullish" and habr["direction"] == "bullish" and rev["reversal"] == "bullish")
                or (pump_side == "bearish" and habr["direction"] == "bearish" and rev["reversal"] == "bearish")
            ):
                super_rating = int((hybrid_rating + max(pump_rating, rev_adj, habr_adj)) / 2 + 20)

                candidates.append({
                    "symbol": symbol,
                    "type": f"SUPER HYBRID Habr+Reversal+Pump {direction} ({mode_key})",
                    "emoji": "💠",
                    "price": last_price,
                    "rating": super_rating,
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
                    "meta_closes": closes_1m,
                    "meta_highs": highs_1m,
                    "meta_lows": lows_1m,
                    "liq_map_bias": liq_bias,
                    "liq_map_strongest": liq_strongest,
                    "liq_map_vac_up": liq_vac_up,
                    "liq_map_vac_down": liq_vac_down,
                })

            # ULTRA HYBRID removed: focus on cleaner core signals

        if habr_adj >= min_score:
            candidates.append({
                "symbol": symbol,
                "type": f"HABR {direction} ({mode_key})",
                "emoji": "🧠",
                "price": last_price,
                "rating": habr_adj,
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
                "meta_closes": closes_1m,
                "meta_highs": highs_1m,
                "meta_lows": lows_1m,
                "liq_map_bias": liq_bias,
                "liq_map_strongest": liq_strongest,
                "liq_map_vac_up": liq_vac_up,
                "liq_map_vac_down": liq_vac_down,
            })

    if is_pump and pump_rating >= min_score:
        adj = adjust_rating_with_context(
            pump_rating,
            f"PUMP ({mode_key})",
            closes_1m,
            oi_now,
            oi_prev,
            funding_rate,
            liq_status,
            flow_status,
            delta_status,
            trend_score,
            risk_score,
        )
        adj += apply_reversal_filters("PUMP", closes_1m, highs_1m, lows_1m, volumes_1m, delta_status)

        candidates.append({
            "symbol": symbol,
            "type": f"PUMP ({mode_key})",
            "emoji": "🟢",
            "price": last_price,
            "rating": adj,
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
            "meta_closes": closes_1m,
            "meta_highs": highs_1m,
            "meta_lows": lows_1m,
            "liq_map_bias": liq_bias,
            "liq_map_strongest": liq_strongest,
            "liq_map_vac_up": liq_vac_up,
            "liq_map_vac_down": liq_vac_down,
        })
        set_symbol_state(symbol, {
            "type": "pump",
            "ts": now_ts,
            "price": last_price,
            "rating": adj,
            "delay_bars": 0,
        })

    if is_dump and dump_rating >= min_score:
        adj = adjust_rating_with_context(
            dump_rating,
            f"DUMP ({mode_key})",
            closes_1m,
            oi_now,
            oi_prev,
            funding_rate,
            liq_status,
            flow_status,
            delta_status,
            trend_score,
            risk_score,
        )
        adj += apply_reversal_filters("DUMP", closes_1m, highs_1m, lows_1m, volumes_1m, delta_status)

        candidates.append({
            "symbol": symbol,
            "type": f"DUMP ({mode_key})",
            "emoji": "🔴",
            "price": last_price,
            "rating": adj,
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
            "meta_closes": closes_1m,
            "meta_highs": highs_1m,
            "meta_lows": lows_1m,
            "liq_map_bias": liq_bias,
            "liq_map_strongest": liq_strongest,
            "liq_map_vac_up": liq_vac_up,
            "liq_map_vac_down": liq_vac_down,
        })
        set_symbol_state(symbol, {
            "type": "dump",
            "ts": now_ts,
            "price": last_price,
            "rating": adj,
            "delay_bars": 0,
        })

    if reversal_state:
        delay_bars = int(reversal_state.get("delay_bars", 0)) + 1
        reversal_state["delay_bars"] = delay_bars
        set_symbol_state(symbol, reversal_state)

    if is_rev_pump and rev_pump_rating >= min_score:
        adj = adjust_rating_with_context(
            rev_pump_rating,
            "REVERSAL Pump → Dump",
            closes_1m,
            oi_now,
            oi_prev,
            funding_rate,
            liq_status,
            flow_status,
            delta_status,
            trend_score,
            risk_score,
        )
        adj += apply_reversal_filters("Pump → Dump", closes_1m, highs_1m, lows_1m, volumes_1m, delta_status)

        if (
            _passes_strict_reversal_filters(
                strictness_level=strictness_level,
                direction="Pump → Dump",
                flow_status=flow_status,
                delta_status=delta_status,
                event_1h=event_1h,
                event_4h=event_4h,
                structure_1h=structure_1h,
                structure_4h=structure_4h,
            )
            and _should_allow_reversal(
                direction="Pump → Dump",
                state=reversal_state,
                requires_state=reversal_requires_state,
                min_score=min_score,
                rating=adj,
                extra_min_bonus=reversal_min_score_bonus,
                min_delay_bars=reversal_min_delay_bars,
            )
        ):
            candidates.append({
                "symbol": symbol,
                "type": "REVERSAL Pump → Dump",
                "emoji": "🟡",
                "price": last_price,
                "rating": adj,
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
                "meta_closes": closes_1m,
                "meta_highs": highs_1m,
                "meta_lows": lows_1m,
                "liq_map_bias": liq_bias,
                "liq_map_strongest": liq_strongest,
                "liq_map_vac_up": liq_vac_up,
                "liq_map_vac_down": liq_vac_down,
            })

    if rev.get("reversal") and rev.get("rating", 0) >= min_score:
        direction = "Dump → Pump" if rev["reversal"] == "bullish" else "Pump → Dump"
        adj = adjust_rating_with_context(
            rev["rating"],
            f"REVERSAL {direction}",
            closes_1m,
            oi_now,
            oi_prev,
            funding_rate,
            liq_status,
            flow_status,
            delta_status,
            trend_score,
            risk_score,
        )
        adj += apply_reversal_filters(direction, closes_1m, highs_1m, lows_1m, volumes_1m, delta_status)

        if (
            _passes_strict_reversal_filters(
                strictness_level=strictness_level,
                direction=direction,
                flow_status=flow_status,
                delta_status=delta_status,
                event_1h=event_1h,
                event_4h=event_4h,
                structure_1h=structure_1h,
                structure_4h=structure_4h,
            )
            and _should_allow_reversal(
                direction=direction,
                state=reversal_state,
                requires_state=reversal_requires_state,
                min_score=min_score,
                rating=adj,
                extra_min_bonus=reversal_min_score_bonus,
                min_delay_bars=reversal_min_delay_bars,
            )
        ):
            candidates.append({
                "symbol": symbol,
                "type": f"REVERSAL {direction}",
                "emoji": "🔵",
                "price": last_price,
                "rating": adj,
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
                "meta_closes": closes_1m,
                "meta_highs": highs_1m,
                "meta_lows": lows_1m,
                "liq_map_bias": liq_bias,
                "liq_map_strongest": liq_strongest,
                "liq_map_vac_up": liq_vac_up,
                "liq_map_vac_down": liq_vac_down,
            })

    if not candidates:
        return None

    symbol_mem = get_symbol_memory(symbol)
    symbol_profile = symbol_mem.get("profile", {}) if symbol_mem else {}
    symbol_regime = symbol_profile.get("regime", "neutral")

    for c in candidates:
        base = c["rating"]
        t = c["type"]

        direction_side = infer_direction_side(t)
        if direction_side is None:
            direction_side = "bullish" if trend_score >= 0 else "bearish"

        base += impulse_score
        base = base * btc_factor

        if btc_regime == "trending" and ("PUMP" in t or "DUMP" in t):
            base *= 1.05
        if btc_regime == "ranging" and "REVERSAL" in t:
            base *= 1.07
        if btc_regime == "high_vol":
            base *= 0.9

        mem_bias = 0.0
        if symbol_regime == "pumpy" and "PUMP" in t:
            mem_bias += 10.0
        if symbol_regime == "dumpy" and "DUMP" in t:
            mem_bias += 10.0
        if symbol_regime == "mean_reverting" and "REVERSAL" in t:
            mem_bias += 8.0
        if symbol_regime == "chaotic":
            mem_bias -= 12.0

        base += mem_bias

        extra_filters_ok = {
            "min_score_ok": base >= min_score,
            "oi_not_falling": oi_status != "falling",
            "liq_not_contra_bull": not (
                direction_side == "bullish" and liq_bias == "bearish"
            ),
            "liq_not_contra_bear": not (
                direction_side == "bearish" and liq_bias == "bullish"
            ),
        }

        sf3 = apply_smartfilters_v3(
            symbol=symbol,
            base_rating=int(base),
            direction_side=direction_side,
            closes_1m=closes_1m,
            klines_1m=klines_1m,
            trend_score=trend_score,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            liquidity_bias=liq_bias,
            noise_level=None,
            btc_ctx=btc_ctx,
            extra_filters_ok=extra_filters_ok,
            global_risk_proxy=None,
        )

        c["rating"] = sf3["final_rating"]
        c["confidence"] = sf3["confidence"]
        c["symbol_regime"] = sf3["symbol_regime"]
        c["market_ctx"] = sf3["market_ctx"]
        c["vol_cluster"] = sf3["vol_cluster"]
        c["memory_ctx"] = sf3["memory_ctx"]
        c["sf3_weights"] = sf3["weights"]
        c["symbol_memory_profile"] = symbol_profile
        aligned_rating = apply_alignment_penalties(
            rating=c["rating"],
            direction_side=direction_side,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            flow_status=flow_status,
            impulse_score=impulse_score,
        )
        c["rating"] = max(0, min(int(aligned_rating), 100))

    candidates.sort(key=lambda x: x["rating"], reverse=True)
    best = candidates[0]

    atr_1m = compute_atr_from_klines(klines_1m)
    snapshot = {
        "atr_1m": atr_1m,
        "trend_score": trend_score,
        "is_pump": "PUMP" in best["type"],
        "is_dump": "DUMP" in best["type"],
        "btc_factor": btc_factor,
    }
    updated_mem = update_symbol_memory(symbol, snapshot)
    best["symbol_memory"] = updated_mem

    log_signal(best)
    mark_symbol_signal(symbol)
    return best


async def scanner_loop(send_text, send_photo, min_score: int, engine=None):
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                tickers = await fetch_tickers(session)

                symbols = [
                    (t["symbol"], t)
                    for t in tickers
                    if t.get("symbol", "").endswith("USDT")
                ]

                tasks = [
                    analyze_symbol_async(session, s, min_score, tinfo)
                    for s, tinfo in symbols
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                signals = [r for r in results if isinstance(r, dict)]

                if signals:
                    signals.sort(key=lambda x: x["rating"], reverse=True)

                    for s in signals[:10]:
                        symbol_regime = s.get("symbol_regime", {}) or {}
                        market_ctx = s.get("market_ctx", {}) or {}
                        vol_cluster = s.get("vol_cluster", {}) or {}
                        mem_profile = (s.get("symbol_memory") or {}).get("profile", {}) or {}

                        text = (
                            f"{s['emoji']} {s['type']} — {s['symbol']}\n"
                            f"Цена: {s['price']:.4f} USDT\n"
                            f"Сила сигнала: {s['rating']}/100\n"
                            f"Уверенность: {s.get('confidence', 0):.2f}\n"
                            f"Trend Score: {s['trend_score']}\n"
                            f"Risk Score: {s['risk_score']}\n"
                            f"HTF 15m: {s.get('trend_15m', 0)} | 1h: {s.get('trend_1h', 0)} | 4h: {s.get('trend_4h', 0)}\n"
                            f"Symbol Regime: {symbol_regime.get('regime')} (strength={symbol_regime.get('strength')})\n"
                            f"Market Regime: {market_ctx.get('market_regime')} | Risk: {market_ctx.get('risk')}\n"
                            f"Vol Cluster: {vol_cluster.get('cluster')} | VolScore: {vol_cluster.get('volatility_score')}\n"
                            f"Symbol Memory Regime: {mem_profile.get('regime')} | PumpProb: {mem_profile.get('pump_probability'):.2f} | DumpProb: {mem_profile.get('dump_probability'):.2f}\n"
                            f"OI: {s['oi']}\n"
                            f"{format_funding_text(s['funding'])}\n"
                            f"{format_liq_text(s['liq'])}\n"
                            f"{format_flow_text(s['flow'])}\n"
                            f"{format_delta_text(s['delta'])}\n"
                            f"Liquidity Bias: {s.get('liq_map_bias')}\n"
                            f"Strongest Zone: {s.get('liq_map_strongest')}\n"
                            f"Vacuum Up: {s.get('liq_map_vac_up')} | Down: {s.get('liq_map_vac_down')}\n"
                        )

                        await send_text(text)

                        klines_15m = await fetch_klines(session, s["symbol"], interval="15", limit=96)
                        chart = generate_candle_chart(klines_15m, s["symbol"], timeframe_label="15m")

                        if chart:
                            photo = BufferedInputFile(chart.getvalue(), filename=f"{s['symbol']}.png")
                            await send_photo(photo)

                        if engine is not None:
                            await engine.on_signal(s)

                await asyncio.sleep(30)

            except Exception as e:
                log_error(e)
                try:
                    await send_text(f"⚠️ Ошибка в сканере, перезапускаю цикл...\n{repr(e)}")
                except:
                    pass
                await asyncio.sleep(5)
                continue
