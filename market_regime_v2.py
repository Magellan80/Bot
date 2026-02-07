import time
from typing import Dict, Any
from data_layer import fetch_klines

_REGIME_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "regime": "neutral",
    "risk": 0.0,
    "details": {},
}


async def compute_market_regime_v2(session, cache_ttl: int = 60) -> Dict[str, Any]:
    """
    Глобальный рыночный режим:
      - regime: "bull", "bear", "chop", "panic", "euphoria", "neutral"
      - risk: 0..1 (0 — безопасно, 1 — очень рискованно)
    """
    global _REGIME_CACHE
    now = time.time()
    if now - _REGIME_CACHE["ts"] < cache_ttl:
        return _REGIME_CACHE

    try:
        # BTC 1h
        btc_1h = await fetch_klines(session, "BTCUSDT", interval="60", limit=80)
        # BTC 15m
        btc_15m = await fetch_klines(session, "BTCUSDT", interval="15", limit=80)
        # ETH 1h
        eth_1h = await fetch_klines(session, "ETHUSDT", interval="60", limit=80)

        def _trend(kl):
            if not kl or len(kl) < 30:
                return 0.0, 0.0
            closes = [float(c[4]) for c in kl][::-1]
            c0 = closes[0]
            cN = closes[30]
            if cN <= 0:
                return 0.0, 0.0
            change = (c0 - cN) / cN * 100
            diffs = [
                abs(closes[i] - closes[i + 1]) / max(closes[i + 1], 1e-7) * 100
                for i in range(0, 30)
            ]
            vol = sum(diffs) / len(diffs)
            return change, vol

        btc_trend_1h, btc_vol_1h = _trend(btc_1h)
        btc_trend_15m, btc_vol_15m = _trend(btc_15m)
        eth_trend_1h, eth_vol_1h = _trend(eth_1h)

        # Простая агрегированная картинка
        avg_trend = (btc_trend_1h + eth_trend_1h) / 2
        avg_vol = (btc_vol_1h + eth_vol_1h) / 2

        regime = "neutral"
        risk = 0.5

        # Паника / эвфория
        if avg_trend > 3 and avg_vol > 2:
            regime = "euphoria"
            risk = 0.8
        elif avg_trend < -3 and avg_vol > 2:
            regime = "panic"
            risk = 0.9
        # Трендовый бычий / медвежий
        elif avg_trend > 1 and avg_vol <= 2:
            regime = "bull"
            risk = 0.6
        elif avg_trend < -1 and avg_vol <= 2:
            regime = "bear"
            risk = 0.7
        # Флэт/пила
        elif abs(avg_trend) < 0.5 and avg_vol > 1.5:
            regime = "chop"
            risk = 0.75
        else:
            regime = "neutral"
            risk = 0.5

        details = {
            "btc_trend_1h": btc_trend_1h,
            "btc_vol_1h": btc_vol_1h,
            "btc_trend_15m": btc_trend_15m,
            "btc_vol_15m": btc_vol_15m,
            "eth_trend_1h": eth_trend_1h,
            "eth_vol_1h": eth_vol_1h,
            "avg_trend": avg_trend,
            "avg_vol": avg_vol,
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
