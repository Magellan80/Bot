# broker_bybit_futures.py
import time
import hmac
import hashlib
from typing import Any, Dict, List, Optional

import aiohttp

from trading_engine import PositionSide


class BrokerBybitFutures:
    """
    BrokerBybitFutures
    -------------------
    - Bybit Unified Trading (USDT Perpetual, category=linear)
    - Поддерживает:
        * get_equity
        * get_last_price
        * open_market_order
        * close_market_order
        * place_take_profit
        * place_stop_loss_market
        * get_open_positions
        * get_open_orders
        * cancel_order
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

    # ============================
    #   ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
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
    ) -> Dict[str, Any]:
        url = self.base_url + path
        params = params or {}

        params["api_key"] = self.api_key
        params["timestamp"] = self._timestamp()
        params["recvWindow"] = self.recv_window
        params["sign"] = self._sign(params)

        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            else:
                async with session.post(url, data=params) as resp:
                    data = await resp.json()

        return data

    # ============================
    #   ОБЯЗАТЕЛЬНЫЕ МЕТОДЫ
    # ============================

    async def get_equity(self) -> Optional[float]:
        try:
            params = {
                "accountType": "UNIFIED",
            }
            data = await self._request("GET", "/v5/account/wallet-balance", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            list_ = result.get("list", [])
            if not list_:
                return None

            for acc in list_:
                coin_list = acc.get("coin", [])
                for c in coin_list:
                    if c.get("coin") == "USDT":
                        equity = float(c.get("equity", 0))
                        return equity
            return None
        except Exception:
            return None

    async def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
            }
            data = await self._request("GET", "/v5/market/tickers", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            list_ = result.get("list", [])
            if not list_:
                return None

            last_price = float(list_[0].get("lastPrice", 0))
            return last_price
        except Exception:
            return None

    async def open_market_order(self, symbol: str, side: PositionSide, size: float) -> Optional[str]:
        try:
            side_str = "Buy" if side == PositionSide.LONG else "Sell"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "false",
            }
            data = await self._request("POST", "/v5/order/create", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            order_id = result.get("orderId")
            return order_id
        except Exception:
            return None

    async def close_market_order(self, symbol: str, side: PositionSide, size: float) -> Optional[str]:
        try:
            side_str = "Sell" if side == PositionSide.LONG else "Buy"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
            }
            data = await self._request("POST", "/v5/order/create", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            order_id = result.get("orderId")
            return order_id
        except Exception:
            return None

    async def place_take_profit(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float
    ) -> Optional[str]:
        try:
            side_str = "Sell" if side == PositionSide.LONG else "Buy"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Limit",
                "qty": str(size),
                "price": f"{price:.4f}",
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
            }
            data = await self._request("POST", "/v5/order/create", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            order_id = result.get("orderId")
            return order_id
        except Exception:
            return None

    async def place_stop_loss(self, symbol: str, side: PositionSide, size: float, price: float) -> Optional[str]:
        try:
            side_str = "Sell" if side == PositionSide.LONG else "Buy"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Limit",
                "qty": str(size),
                "price": f"{price:.4f}",
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
            }
            data = await self._request("POST", "/v5/order/create", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            order_id = result.get("orderId")
            return order_id
        except Exception:
            return None

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            params = {
                "category": "linear",
            }
            data = await self._request("GET", "/v5/position/list", params)
            if data.get("retCode") != 0:
                return []

            result = data.get("result", {})
            list_ = result.get("list", [])
            positions = []
            for p in list_:
                size = float(p.get("size", 0) or 0)
                if size == 0:
                    continue
                positions.append(
                    {
                        "symbol": p.get("symbol"),
                        "side": p.get("side"),  # "Buy" / "Sell"
                        "size": p.get("size"),
                        "avgPrice": p.get("avgPrice"),
                    }
                )
            return positions
        except Exception:
            return []

    async def get_open_orders(self, symbol: str | None = None) -> List[Dict[str, Any]]:
        """
        Открытые ордера (TP/SL/прочие) по category=linear.
        """
        try:
            params = {
                "category": "linear",
            }
            if symbol:
                params["symbol"] = symbol

            data = await self._request("GET", "/v5/order/realtime", params)
            if data.get("retCode") != 0:
                return []

            result = data.get("result", {})
            list_ = result.get("list", [])
            return list_ or []
        except Exception:
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id,
            }
            data = await self._request("POST", "/v5/order/cancel", params)
            if data.get("retCode") != 0:
                return False
            return True
        except Exception:
            return False

    async def place_stop_loss_market(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        trigger_price: float
    ) -> Optional[str]:
        try:
            side_str = "Sell" if side == PositionSide.LONG else "Buy"

            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side_str,
                "orderType": "Market",
                "qty": str(size),
                "timeInForce": "GoodTillCancel",
                "reduceOnly": "true",
                "triggerPrice": f"{trigger_price:.4f}",
                "triggerDirection": 2 if side == PositionSide.LONG else 1,
                "triggerBy": "LastPrice",
            }

            data = await self._request("POST", "/v5/order/create", params)
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            order_id = result.get("orderId")
            return order_id
        except Exception:
            return None
