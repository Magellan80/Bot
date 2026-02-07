# bybit_ws.py
import asyncio
import json
import time
import hmac
import hashlib
from typing import Any, Dict, Optional, Callable, Awaitable

import aiohttp


class BybitWebSocket:
    """
    BybitWebSocket (Private WS v5)
    ------------------------------
    - Подключается к приватному WebSocket:
        wss://stream.bybit.com/v5/private
    - Авторизуется по api_key / api_secret
    - Подписывается на:
        * order
        * position
        * execution
    - По любому событию вызывает engine.sync_with_exchange()
      (движок сам уже умеет:
        - сверять позиции
        - сверять ордера
        - восстанавливать SL/TP
      )

    Использование:

        ws = BybitWebSocket(
            api_key=API_KEY,
            api_secret=API_SECRET,
            engine=engine,
        )
        await ws.run()

    Запускать параллельно с price_monitor_loop и sync_loop.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        engine,
        ws_url: str = "wss://stream.bybit.com/v5/private",
        reconnect_delay: int = 5,
    ):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.engine = engine
        self.ws_url = ws_url
        self.reconnect_delay = reconnect_delay

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ---------------------------------------------------------

    def _gen_signature(self, timestamp: int) -> str:
        """
        Подпись для Private WS v5:
        sign = HMAC_SHA256(secret, str(timestamp) + api_key)
        """
        msg = str(timestamp) + self.api_key
        return hmac.new(self.api_secret, msg.encode(), hashlib.sha256).hexdigest()

    async def _send(self, payload: Dict[str, Any]):
        if self._ws is None:
            return
        await self._ws.send_str(json.dumps(payload))

    async def _auth(self):
        """
        Авторизация на Private WS.
        """
        ts = int(time.time() * 1000)
        sign = self._gen_signature(ts)

        auth_msg = {
            "op": "auth",
            "args": [self.api_key, ts, sign],
        }
        await self._send(auth_msg)

    async def _subscribe(self):
        """
        Подписка на приватные топики:
        - order
        - position
        - execution
        """
        sub_msg = {
            "op": "subscribe",
            "args": [
                "order",
                "position",
                "execution",
            ],
        }
        await self._send(sub_msg)

    # ---------------------------------------------------------
    # ОБРАБОТКА СООБЩЕНИЙ
    # ---------------------------------------------------------

    async def _handle_message(self, msg: Dict[str, Any]):
        """
        Универсальный обработчик сообщений WS.
        Здесь мы:
        - логируем важные события
        - по событиям ордеров/позиций/исполнений дёргаем engine.sync_with_exchange()
        """
        topic = msg.get("topic")
        op = msg.get("op")
        event_type = msg.get("type")
        data = msg.get("data")

        # Ответы на auth/subscribe
        if op in ("auth", "subscribe"):
            # Можно залогировать при желании
            # print("WS op response:", msg)
            return

        # Пинги/понги
        if msg.get("op") == "pong" or msg.get("event") == "pong":
            return

        # Основные приватные топики
        if topic in ("order", "position", "execution"):
            # Здесь можно при желании добавить более тонкую обработку,
            # но уже сейчас достаточно просто вызвать sync_with_exchange,
            # чтобы движок сам всё сверил и восстановил.
            try:
                await self.engine.sync_with_exchange()
            except Exception as e:
                # engine имеет _log_error
                try:
                    self.engine._log_error(
                        f"ws:sync_with_exchange_error;topic={topic};error={repr(e)}"
                    )
                except Exception:
                    pass

    # ---------------------------------------------------------
    # ОСНОВНОЙ ЦИКЛ WS
    # ---------------------------------------------------------

    async def _connect_and_listen(self):
        """
        Один цикл подключения:
        - создаёт сессию
        - коннектится к WS
        - авторизуется
        - подписывается
        - слушает сообщения
        - при разрыве — выбрасывает исключение, чтобы внешний цикл переподключился
        """
        self._session = aiohttp.ClientSession()
        async with self._session.ws_connect(self.ws_url, heartbeat=20) as ws:
            self._ws = ws

            # Авторизация
            await self._auth()
            # Подписка
            await self._subscribe()

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    # Bybit может присылать массив сообщений
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                await self._handle_message(item)
                    elif isinstance(data, dict):
                        await self._handle_message(data)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break

        # Закрываем сессию после выхода из контекста
        if self._session:
            await self._session.close()
            self._session = None
        self._ws = None

        # Выкидываем исключение, чтобы внешний цикл понял, что нужно переподключиться
        raise ConnectionError("WebSocket connection closed")

    async def run(self):
        """
        Внешний цикл:
        - пытается подключиться
        - при разрыве ждёт reconnect_delay и пробует снова
        """
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Логируем ошибку и ждём перед переподключением
                try:
                    self.engine._log_error(f"ws:connection_error;error={repr(e)}")
                except Exception:
                    pass
                await asyncio.sleep(self.reconnect_delay)

    async def stop(self):
        """
        Остановка WS‑клиента.
        """
        self._running = False
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._ws = None
