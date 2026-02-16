# broker_bybit_futures.py

import time
import hmac
import hashlib
from typing import Any, Dict, List, Optional

import aiohttp

from trading_engine import PositionSide


class BrokerBybitFutures:
    """
    Stable production-ready broker layer
    - Single persistent aiohttp session
    - Retry mechanism
    - Clean error logging
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bybit.com",
        recv_window: int = 5000,
    ):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.session = aiohttp.ClientSession()

    # ============================
    # INTERNAL HELPERS
    # ============================

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: Dict[str, Any]) -> str:
        ts = str(params["timestamp"])
        recv_window = str(params["recvWindow"])
        api_key = self.api_key

        sorted_items = sorted(
            (k, v) for k, v in params.items()
            if k not in ("sign", "timestamp", "api_key", "recvWindow")
        )
        param_str = "&".join(f"{k}={v}" for k, v in sorted_items)

        to_sign = ts + api_key + recv_window + param_str
        return hmac.new(self.api_secret, to_sign.encode(), hashlib.sha256).hexdigest()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> Optional[Dict[str, Any]]:

        url = self.base_url + path
        params = params or {}

        for attempt in range(retries):
            try:
                params["api_key"] = self.api_key
                params["timestamp"] = self._timestamp()
                params["recvWindow"] = self.recv_window
                params["sign"] = self._sign(params)

                if method.upper() == "GET":
                    async with self.session.get(url, params=params) as resp:
                        data = await resp.json()
                else:
                    async with self.session.post(url, data=params) as resp:
                        data = await resp.json()

                if data.get("retCode") == 0:
                    return data

                print(f"[BROKER][ERROR] {data.get('retCode')} - {data.get('retMsg')}")

            except Exception as e:
                print(f"[BROKER][EXCEPTION] Attempt {attempt + 1}: {e}")

        return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    # ============================
    # PUBLIC METHODS
    # ============================

    async def get_equity(self) -> Optional[float]:
        data = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED"},
        )

        if not data:
            return None

        try:
            for acc in data["result"]["list"]:
                for c in acc.get("coin", []):
                    if c.get("coin") == "USDT":
                        return float(c.get("equity", 0))
        except Exception:
            return None

        return None

    async def get_last_price(self, symbol: str) -> Optional[float]:
        data = await self._request(
            "GET",
            "/v5/market/tickers",
            {"category": "linear", "symbol": symbol},
        )

        if not data:
            return None

        try:
            return float(data["result"]["list"][0]["lastPrice"])
        except Exception:
            return None

    async def open_market_order(
        self, symbol: str, side: PositionSide, size: float
    ) -> Optional[str]:

        side_str = "Buy" if side == PositionSide.LONG else "Sell"

        data = await self._request(
            "POST",
            "/v5/order/create",
            {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "false",
            },
        )

        if not data:
            return None

        return data["result"].get("orderId")

    async def close_market_order(
        self, symbol: str, side: PositionSide, size: float
    ) -> Optional[str]:

        side_str = "Sell" if side == PositionSide.LONG else "Buy"

        data = await self._request(
            "POST",
            "/v5/order/create",
            {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
            },
        )

        if not data:
            return None

        return data["result"].get("orderId")

    async def place_take_profit(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float,
    ) -> Optional[str]:

        side_str = "Sell" if side == PositionSide.LONG else "Buy"

        data = await self._request(
            "POST",
            "/v5/order/create",
            {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Limit",
                "qty": str(size),
                "price": f"{price:.4f}",
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
            },
        )

        if not data:
            return None

        return data["result"].get("orderId")

    async def place_stop_loss_market(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        trigger_price: float,
    ) -> Optional[str]:

        side_str = "Sell" if side == PositionSide.LONG else "Buy"

        data = await self._request(
            "POST",
            "/v5/order/create",
            {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "reduceOnly": "true",
                "triggerPrice": f"{trigger_price:.4f}",
                "triggerDirection": 2 if side == PositionSide.LONG else 1,
                "triggerBy": "LastPrice",
            },
        )

        if not data:
            return None

        return data["result"].get("orderId")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        data = await self._request(
            "GET",
            "/v5/position/list",
            {"category": "linear"},
        )

        if not data:
            return []

        positions = []

        try:
            for p in data["result"]["list"]:
                size = float(p.get("size", 0) or 0)
                if size == 0:
                    continue
                positions.append(
                    {
                        "symbol": p.get("symbol"),
                        "side": p.get("side"),
                        "size": size,
                        "avgPrice": float(p.get("avgPrice", 0)),
                    }
                )
        except Exception:
            return []

        return positions

    async def get_open_orders(self, symbol: str | None = None) -> List[Dict[str, Any]]:

        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol

        data = await self._request(
            "GET",
            "/v5/order/realtime",
            params,
        )

        if not data:
            return []

        return data["result"].get("list", []) or []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:

        data = await self._request(
            "POST",
            "/v5/order/cancel",
            {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id,
            },
        )

        return bool(data)
