# data_layer.py
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# ============================
#   ПРОСТОЙ КЭШ С TTL
# ============================

class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: Dict[Any, Tuple[float, Any]] = {}

    def get(self, key: Any):
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: Any, value: Any):
        self._store[key] = (time.time(), value)


# ============================
#   КЭШИ
# ============================

_tickers_cache = TTLCache(ttl_seconds=10)
_klines_cache = TTLCache(ttl_seconds=10)
_oi_cache = TTLCache(ttl_seconds=60)
_funding_cache = TTLCache(ttl_seconds=300)
_liq_cache = TTLCache(ttl_seconds=30)
_trades_cache = TTLCache(ttl_seconds=10)
_orderbook_cache = TTLCache(ttl_seconds=1)

BASE_URL = "https://api.bybit.com"


# ============================
#   УСИЛЕННЫЙ REQUEST LAYER
# ============================

async def _get_json(
    session: aiohttp.ClientSession,
    path: str,
    params: Dict[str, Any],
    retries: int = 3
) -> Dict[str, Any]:

    url = BASE_URL + path
    delay = 0.5

    for attempt in range(retries):
        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:

                if resp.status == 429:
                    # Rate limit
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if resp.status >= 500:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if resp.status != 200:
                    return {}

                data = await resp.json()

                # Проверка retCode Bybit
                if isinstance(data, dict):
                    if data.get("retCode", 0) != 0:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue

                return data

        except asyncio.TimeoutError:
            await asyncio.sleep(delay)
            delay *= 2

        except aiohttp.ClientError:
            await asyncio.sleep(delay)
            delay *= 2

        except Exception:
            await asyncio.sleep(delay)
            delay *= 2

    return {}


# ============================
#   TICKERS
# ============================

async def fetch_tickers(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    cache_key = "tickers_linear"
    cached = _tickers_cache.get(cache_key)
    if cached is not None:
        return cached

    data = await _get_json(session, "/v5/market/tickers", {"category": "linear"})
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []
    _tickers_cache.set(cache_key, result)
    return result


# ============================
#   KLINES
# ============================

async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "1",
    limit: int = 80
) -> List[List[Any]]:

    cache_key = (symbol, interval, limit)
    cached = _klines_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    data = await _get_json(session, "/v5/market/kline", params)
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []
    _klines_cache.set(cache_key, result)
    return result


# ============================
#   OPEN INTEREST
# ============================

async def fetch_open_interest(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "15"
) -> Tuple[Optional[float], Optional[float]]:

    cache_key = (symbol, "oi", interval)
    cached = _oi_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval
    }

    data = await _get_json(session, "/v5/market/open-interest", params)
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

    if not result or len(result) < 2:
        _oi_cache.set(cache_key, (None, None))
        return None, None

    try:
        oi_now = float(result[0]["openInterest"])
        oi_prev = float(result[1]["openInterest"])
        value = (oi_now, oi_prev)
    except Exception:
        value = (None, None)

    _oi_cache.set(cache_key, value)
    return value


# ============================
#   FUNDING RATE
# ============================

async def fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str
) -> Optional[float]:

    cache_key = (symbol, "funding")
    cached = _funding_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 1
    }

    data = await _get_json(session, "/v5/market/funding/history", params)
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

    if not result:
        _funding_cache.set(cache_key, None)
        return None

    try:
        fr = float(result[0]["fundingRate"])
    except Exception:
        fr = None

    _funding_cache.set(cache_key, fr)
    return fr


# ============================
#   LIQUIDATIONS
# ============================

async def fetch_liquidations(
    session: aiohttp.ClientSession,
    symbol: str,
    minutes: int = 15
) -> Tuple[float, float]:

    cache_key = (symbol, "liq", minutes)
    cached = _liq_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 50
    }

    data = await _get_json(session, "/v5/market/liquidation", params)
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

    if not result:
        _liq_cache.set(cache_key, (0.0, 0.0))
        return 0.0, 0.0

    now_ms = int(time.time() * 1000)
    cutoff = now_ms - minutes * 60 * 1000

    long_liq = 0.0
    short_liq = 0.0

    for item in result:
        try:
            ts = int(item.get("updatedTime", item.get("createdTime", now_ms)))
            if ts < cutoff:
                continue

            side = item.get("side")
            qty = float(item.get("qty", 0))
            price = float(item.get("price", 0))
            notional = qty * price

            if side == "Sell":
                long_liq += notional
            elif side == "Buy":
                short_liq += notional

        except Exception:
            continue

    value = (long_liq, short_liq)
    _liq_cache.set(cache_key, value)
    return value


# ============================
#   RECENT TRADES
# ============================

async def fetch_recent_trades(
    session: aiohttp.ClientSession,
    symbol: str,
    limit: int = 200
) -> List[Dict[str, Any]]:

    cache_key = (symbol, "trades", limit)
    cached = _trades_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit
    }

    data = await _get_json(session, "/v5/market/recent-trade", params)
    result = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

    _trades_cache.set(cache_key, result)
    return result


# ============================
#   ORDERBOOK
# ============================

async def fetch_orderbook(
    session: aiohttp.ClientSession,
    symbol: str,
    limit: int = 50
) -> Dict[str, Any]:

    cache_key = (symbol, "orderbook", limit)
    cached = _orderbook_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit
    }

    data = await _get_json(session, "/v5/market/orderbook", params)
    result = data.get("result", {}) if isinstance(data, dict) else {}

    if isinstance(result, dict) and "list" in result:
        lst = result.get("list")
        if isinstance(lst, list) and lst:
            result = lst[0]

    _orderbook_cache.set(cache_key, result)
    return result

# ==========================================================
#   ASYNC WRAPPER FOR BACKTEST
# ==========================================================

async def load_ohlcv(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "1",
    limit: int = 200
):
    """
    Async версия для backtest.
    Возвращает список свечей:
    [
        {
            "timestamp": int,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        }
    ]
    """

    raw_klines = await fetch_klines(session, symbol, interval, limit)

    if not raw_klines:
        return []

    candles = []

    for k in raw_klines:
        try:
            candles.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        except Exception:
            continue

    candles.reverse()  # oldest → newest
    return candles
