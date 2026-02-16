import time
from typing import Dict, Any
from data_layer import fetch_klines

# ==========================================================
# REGIME ENGINE V3 — Institutional HTF Aggregator
# ==========================================================
# Глобальный рыночный режим:
#   - bull / bear / chop / panic / euphoria / neutral
#   - risk: 0..1 (0 — безопасно, 1 — крайне рискованно)
#
# Логика:
#   • BTC 1H + 4H
#   • ETH 1H
#   • HTF momentum + volatility
#   • Panic / Euphoria detection
#   • Cache TTL
# ==========================================================

_REGIME_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "regime": "neutral",
    "risk": 0.5,
    "details": {},
}


def _trend_vol(klines, lookback: int = 30):
    if not klines or len(klines) < lookback + 5:
        return 0.0, 0.0

    closes = [float(c[4]) for c in klines][::-1]

    c0 = closes[0]
    cN = closes[lookback]

    if cN <= 0:
        return 0.0, 0.0

    change_pct = (c0 - cN) / cN * 100

    diffs = [
        abs(closes[i] - closes[i + 1]) / max(closes[i + 1], 1e-9) * 100
        for i in range(lookback)
    ]

    vol = sum(diffs) / len(diffs)

    return change_pct, vol


async def compute_market_regime(session, cache_ttl: int = 60) -> Dict[str, Any]:
    global _REGIME_CACHE

    now = time.time()

    if now - _REGIME_CACHE["ts"] < cache_ttl:
        return _REGIME_CACHE

    try:
        btc_1h = await fetch_klines(session, "BTCUSDT", interval="60", limit=120)
        btc_4h = await fetch_klines(session, "BTCUSDT", interval="240", limit=120)
        eth_1h = await fetch_klines(session, "ETHUSDT", interval="60", limit=120)

        btc_trend_1h, btc_vol_1h = _trend_vol(btc_1h)
        btc_trend_4h, btc_vol_4h = _trend_vol(btc_4h)
        eth_trend_1h, eth_vol_1h = _trend_vol(eth_1h)

        # --- Institutional-style weighted aggregation
        weighted_trend = (
            btc_trend_4h * 0.45 +
            btc_trend_1h * 0.35 +
            eth_trend_1h * 0.20
        )

        weighted_vol = (
            btc_vol_4h * 0.45 +
            btc_vol_1h * 0.35 +
            eth_vol_1h * 0.20
        )

        regime = "neutral"
        risk = 0.5

        # ======================================================
        # EXTREME STATES
        # ======================================================

        if weighted_trend > 4 and weighted_vol > 2.5:
            regime = "euphoria"
            risk = 0.85

        elif weighted_trend < -4 and weighted_vol > 2.5:
            regime = "panic"
            risk = 0.95

        # ======================================================
        # TREND STATES
        # ======================================================

        elif weighted_trend > 1.5 and weighted_vol <= 2.5:
            regime = "bull"
            risk = 0.6

        elif weighted_trend < -1.5 and weighted_vol <= 2.5:
            regime = "bear"
            risk = 0.7

        # ======================================================
        # CHOP
        # ======================================================

        elif abs(weighted_trend) < 0.8 and weighted_vol > 1.2:
            regime = "chop"
            risk = 0.75

        else:
            regime = "neutral"
            risk = 0.5

        details = {
            "btc_trend_1h": btc_trend_1h,
            "btc_trend_4h": btc_trend_4h,
            "eth_trend_1h": eth_trend_1h,
            "weighted_trend": weighted_trend,
            "weighted_vol": weighted_vol,
            "btc_vol_1h": btc_vol_1h,
            "btc_vol_4h": btc_vol_4h,
            "eth_vol_1h": eth_vol_1h,
        }

        _REGIME_CACHE = {
            "ts": now,
            "regime": regime,
            "risk": max(0.0, min(risk, 1.0)),
            "details": details,
        }

        return _REGIME_CACHE

    except Exception:
        _REGIME_CACHE = {
            "ts": now,
            "regime": "neutral",
            "risk": 0.5,
            "details": {},
        }
        return _REGIME_CACHE
