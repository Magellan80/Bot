import requests
import time


class BybitDownloader:
    """
    Универсальный загрузчик свечей с Bybit v5 API.
    Поддерживает:
      - category: linear (USDT perpetual)
      - интервалы: 1m, 3m, 5m, 15m, 30m, 1h, 4h
      - до 1000 свечей за запрос
      - скачивание истории назад до total_needed
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }

    def __init__(self, category: str = "linear", max_retries: int = 5, pause: float = 0.15):
        self.category = category
        self.max_retries = max_retries
        self.pause = pause

    def download(self, symbol: str, interval: str, total_needed: int):
        print(f"\n[Bybit] Downloading {symbol} {interval} ...")

        if interval not in self.INTERVAL_MAP:
            print(f"[Bybit] Unsupported interval: {interval}")
            return []

        bybit_tf = self.INTERVAL_MAP[interval]

        all_data = []
        end_time = None

        while len(all_data) < total_needed:

            params = {
                "category": self.category,
                "symbol": symbol.upper(),
                "interval": bybit_tf,
                "limit": 1000,
            }

            if end_time:
                params["end"] = end_time

            retries = 0

            while True:
                try:
                    r = requests.get(self.BASE_URL, params=params, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    retries += 1
                    print(f"[Bybit] Retry {retries}/{self.max_retries} due to error: {e}")
                    time.sleep(0.5)
                    if retries >= self.max_retries:
                        print("[Bybit] Max retries reached. Stopping.")
                        return []

            if data.get("retCode") != 0:
                print(f"[Bybit] API error: {data.get('retMsg')}")
                break

            result = data.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                print("[Bybit] Empty list, end of history.")
                break

            # Bybit отдаёт новые → старые, разворачиваем
            list_data = list(reversed(list_data))

            all_data = list_data + all_data

            # timestamp самой старой свечи
            oldest_ts = int(list_data[0][0])
            end_time = oldest_ts - 1

            print(f"[Bybit] {interval}: {len(all_data)} candles")

            time.sleep(self.pause)

        # Преобразуем в формат твоего бэктестера
        candles = []
        for k in all_data:
            try:
                candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                })
            except Exception:
                continue

        print(f"[Bybit] {interval} DONE. Total: {len(candles)} candles.")
        return candles
