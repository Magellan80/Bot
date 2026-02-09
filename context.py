# context.py
import math
from typing import List, Tuple, Optional

from data_layer import (
    fetch_klines,
    fetch_open_interest,
    fetch_funding_rate,
    fetch_liquidations,
    fetch_recent_trades
)

# ============================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================

def clamp(value, min_v=0, max_v=100):
    return max(min_v, min(int(value), max_v))


def compute_change_percent(closes: List[float], idx_now: int, idx_past: int) -> float:
    if idx_past >= len(closes):
        return 0.0
    now = closes[idx_now]
    past = closes[idx_past]
    if past == 0:
        return 0.0
    return (now - past) / past * 100.0


def count_trend_bars(closes: List[float], lookback=12) -> Tuple[int, int]:
    up = 0
    down = 0
    max_i = min(lookback, len(closes)) - 1
    for i in range(max_i):
        if closes[i] > closes[i + 1]:
            up += 1
        elif closes[i] < closes[i + 1]:
            down += 1
    return up, down


# ============================
#   FUNDING
# ============================

def funding_bias(fr: Optional[float]) -> str:
    if fr is None:
        return "neutral"
    fr_pct = fr * 100
    if fr_pct > 0.01:
        return "bearish"
    if fr_pct < -0.01:
        return "bullish"
    return "neutral"


def format_funding_text(fr: Optional[float]) -> str:
    if fr is None:
        return "Funding: n/a"
    fr_pct = fr * 100
    if fr_pct > 0.01:
        bias = "bearish pressure"
    elif fr_pct < -0.01:
        bias = "bullish pressure"
    else:
        bias = "neutral"
    return f"Funding: {fr_pct:.4f}% ({bias})"


# ============================
#   LIQUIDATIONS
# ============================

def interpret_liquidations(long_liq: float, short_liq: float, threshold=100000.0) -> str:
    if long_liq < threshold and short_liq < threshold:
        return "none"

    if long_liq > short_liq * 1.5 and long_liq > threshold:
        return "long_spike"
    if short_liq > long_liq * 1.5 and short_liq > threshold:
        return "short_spike"

    return "mixed"


def format_liq_text(status: str) -> str:
    if status == "long_spike":
        return "Liquidations: long spike (dump risk)"
    elif status == "short_spike":
        return "Liquidations: short spike (pump risk)"
    elif status == "mixed":
        return "Liquidations: mixed activity"
    else:
        return "Liquidations: calm"


# ============================
#   FLOW + DELTA
# ============================

def analyze_flow_from_trades(trades, big_trade_threshold=25000.0):
    if not trades:
        return "unknown"

    big_buy = 0.0
    big_sell = 0.0

    for t in trades:
        try:
            side = t.get("side")
            qty = float(t.get("qty", 0))
            price = float(t.get("price", 0))
            notional = qty * price
            if notional < big_trade_threshold:
                continue
            if side == "Buy":
                big_buy += notional
            elif side == "Sell":
                big_sell += notional
        except:
            continue

    if big_buy == 0 and big_sell == 0:
        return "balanced"

    if big_buy > big_sell * 1.5:
        return "aggressive_buyers"
    if big_sell > big_buy * 1.5:
        return "aggressive_sellers"

    return "balanced"


def analyze_delta_from_trades(trades):
    if not trades:
        return "neutral"

    buy_vol = 0.0
    sell_vol = 0.0

    for t in trades:
        try:
            side = t.get("side")
            qty = float(t.get("qty", 0))
            price = float(t.get("price", 0))
            notional = qty * price
            if side == "Buy":
                buy_vol += notional
            elif side == "Sell":
                sell_vol += notional
        except:
            continue

    if buy_vol == 0 and sell_vol == 0:
        return "neutral"

    if buy_vol > sell_vol * 1.3:
        return "bullish"
    if sell_vol > buy_vol * 1.3:
        return "bearish"

    return "neutral"


def format_flow_text(flow_status: str) -> str:
    if flow_status == "aggressive_buyers":
        return "Flow: aggressive buyers"
    elif flow_status == "aggressive_sellers":
        return "Flow: aggressive sellers"
    elif flow_status == "balanced":
        return "Flow: balanced"
    else:
        return "Flow: n/a"


def format_delta_text(delta_status: str) -> str:
    if delta_status == "bullish":
        return "Delta: bullish"
    elif delta_status == "bearish":
        return "Delta: bearish"
    else:
        return "Delta: neutral"


# ============================
#   TREND SCORE
# ============================

def compute_ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def trend_score_from_closes(closes: List[float]) -> int:
    if len(closes) < 30:
        return 50

    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    ema100 = compute_ema(closes, 100)

    if ema20 is None or ema50 is None or ema100 is None:
        return 50

    bullish = 0
    bearish = 0

    if ema20 > ema50 > ema100:
        bullish += 40
    if ema20 < ema50 < ema100:
        bearish += 40

    slope = compute_change_percent(closes, 0, min(10, len(closes) - 1))
    if slope > 2.0:
        bullish += 30
    elif slope < -2.0:
        bearish += 30

    up, down = count_trend_bars(closes, lookback=12)
    if up > down + 3:
        bullish += 20
    if down > up + 3:
        bearish += 20

    raw = bullish - bearish
    return clamp(50 + raw)


async def compute_trend_score(session, symbol: str) -> int:
    klines_15m = await fetch_klines(session, symbol, interval="15", limit=96)
    klines_1h = await fetch_klines(session, symbol, interval="60", limit=96)

    if not klines_15m or not klines_1h:
        return 50

    closes_15m = [float(c[4]) for c in klines_15m][::-1]
    closes_1h = [float(c[4]) for c in klines_1h][::-1]

    ts15 = trend_score_from_closes(closes_15m)
    ts1h = trend_score_from_closes(closes_1h)

    return clamp((ts15 + ts1h) // 2)


# ============================
#   RISK SCORE
# ============================

def compute_risk_score(
    closes_1m: List[float],
    oi_status: Optional[str],
    funding_rate: Optional[float],
    liq_status: str,
    flow_status: str,
    delta_status: str,
    trend_score: int
) -> int:
    risk = 50

    if len(closes_1m) > 5:
        vol_5m = abs(compute_change_percent(closes_1m, 0, 5))
        if vol_5m > 3.0:
            risk += 10
        elif vol_5m < 1.0:
            risk -= 5

    if oi_status == "rising":
        risk += 10
    elif oi_status == "falling":
        risk -= 5

    f_bias = funding_bias(funding_rate)
    if f_bias in ("bearish", "bullish"):
        risk += 5

    if liq_status in ("long_spike", "short_spike"):
        risk += 15
    elif liq_status == "mixed":
        risk += 5

    if flow_status in ("aggressive_buyers", "aggressive_sellers"):
        risk += 5

    if delta_status in ("bullish", "bearish"):
        risk += 5

    if trend_score > 80 or trend_score < 20:
        risk += 5

    return clamp(risk)
