# historical_downloader.py
# Универсальный загрузчик истории с Binance + Bybit
# - несколько монет
# - несколько таймфреймов
# - сохранение в JSON и CSV
# - простой прогресс

import os
import json
import csv
import time
from typing import List, Dict, Optional

import requests


# ==========================
# CONFIG
# ==========================

DATA_DIR = "data"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT",
    "BNBUSDT", "DOGEUSDT", "ADAUSDT", "OPUSDT", "ARBUSDT",
    "AVAXUSDT", "TONUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "INJUSDT", "RUNEUSDT"
]


# таймфреймы в "человеческом" виде
TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
]

# сколько свечей минимум хотим на каждый TF
CANDLES_PER_TF = 5000

# какие биржи качаем
EXCHANGES = ["binance", "bybit"]  # можно оставить только ["bybit"] или ["binance"]


# ==========================
# УТИЛИТЫ
# ==========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, candles: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candles, f, ensure_ascii=False)


def save_csv(path: str, candles: List[Dict]):
    if not candles:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close"])
        for c in candles:
            writer.writerow([
                c["timestamp"],
                c["open"],
                c["high"],
                c["low"],
                c["close"],
            ])


def print_progress(prefix: str, current: int, total: int):
    total = max(total, 1)
    pct = current / total * 100
    print(f"{prefix}: {current}/{total} ({pct:.1f}%)")


# ==========================
# БАЗОВЫЙ КЛАСС ЗАГРУЗЧИКА
# ==========================

class BaseDownloader:

    def __init__(self, max_retries: int = 5, pause: float = 0.15):
        self.max_retries = max_retries
        self.pause = pause

    def _request(self, method: str, url: str, params: Dict) -> Optional[Dict]:
        retries = 0
        while True:
            try:
                r = requests.request(method, url, params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                retries += 1
                print(f"Request error ({url}): {e} | retry {retries}/{self.max_retries}")
                time.sleep(0.5)
                if retries >= self.max_retries:
                    print("Max retries reached, aborting this request.")
                    return None


# ==========================
# BINANCE DOWNLOADER
# ==========================

class BinanceDownloader(BaseDownloader):

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def download(self, symbol: str, interval: str, total_needed: int) -> List[Dict]:
        print(f"\n[Binance] Downloading {symbol} {interval} ...")

        all_data = []
        end_time = None

        while len(all_data) < total_needed:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": 1000
            }
            if end_time:
                params["endTime"] = end_time

            data = self._request("GET", self.BASE_URL, params)
            if not data:
                print("[Binance] No more data or error.")
                break

            if not isinstance(data, list) or not isinstance(data[0], list):
                print("[Binance] Unexpected data format.")
                break

            all_data = data + all_data
            end_time = data[0][0] - 1

            print_progress(f"[Binance] {symbol} {interval}", len(all_data), total_needed)
            time.sleep(self.pause)

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

        print(f"[Binance] {symbol} {interval} DONE. Total: {len(candles)} candles.")
        return candles


# ==========================
# BYBIT DOWNLOADER
# ==========================

class BybitDownloader(BaseDownloader):
    """
    Bybit v5 kline:
    GET https://api.bybit.com/v5/market/kline
    params:
      category: "linear" / "inverse" / "spot"
      symbol: "BTCUSDT"
      interval: "1","3","5","15","60","240","D","W","M"
      limit: up to 1000
      end: ms timestamp (optional)
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    # маппинг "человеческих" TF -> Bybit interval
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

    def __init__(self, category: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.category = category

    def download(self, symbol: str, interval: str, total_needed: int) -> List[Dict]:
        print(f"\n[Bybit] Downloading {symbol} {interval} ...")

        if interval not in self.INTERVAL_MAP:
            print(f"[Bybit] Unsupported interval: {interval}")
            return []

        bybit_interval = self.INTERVAL_MAP[interval]

        all_data = []
        end_time = None

        while len(all_data) < total_needed:
            params = {
                "category": self.category,
                "symbol": symbol.upper(),
                "interval": bybit_interval,
                "limit": 1000,
            }
            if end_time:
                params["end"] = end_time

            data = self._request("GET", self.BASE_URL, params)
            if not data:
                print("[Bybit] No more data or error.")
                break

            if data.get("retCode") != 0:
                print(f"[Bybit] API error: {data.get('retMsg')}")
                break

            result = data.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                print("[Bybit] Empty list, end of history.")
                break

            # Bybit возвращает в порядке от новых к старым → разворачиваем
            list_data = list(reversed(list_data))

            all_data = list_data + all_data

            # end_time = самый старый timestamp - 1
            oldest_ts = int(list_data[0][0])
            end_time = oldest_ts - 1

            print_progress(f"[Bybit] {symbol} {interval}", len(all_data), total_needed)
            time.sleep(self.pause)

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

        print(f"[Bybit] {symbol} {interval} DONE. Total: {len(candles)} candles.")
        return candles


# ==========================
# HIGH-LEVEL ORCHESTRATOR
# ==========================

class HistoricalDownloader:

    def __init__(self):
        self.binance = BinanceDownloader()
        self.bybit = BybitDownloader()

    def download_for_symbol(
        self,
        symbol: str,
        timeframes: List[str],
        total_needed: int,
        exchanges: List[str]
    ):
        for tf in timeframes:
            if "binance" in exchanges:
                candles = self.binance.download(symbol, tf, total_needed)
                self._save(symbol, tf, "binance", candles)

            if "bybit" in exchanges:
                candles = self.bybit.download(symbol, tf, total_needed)
                self._save(symbol, tf, "bybit", candles)

    def _save(self, symbol: str, tf: str, exchange: str, candles: List[Dict]):
        if not candles:
            print(f"[{exchange}] No candles to save for {symbol} {tf}")
            return

        base_dir = os.path.join(DATA_DIR, exchange, symbol.upper())
        ensure_dir(base_dir)

        json_path = os.path.join(base_dir, f"{tf}.json")
        csv_path = os.path.join(base_dir, f"{tf}.csv")

        save_json(json_path, candles)
        save_csv(csv_path, candles)

        print(f"[{exchange}] Saved {symbol} {tf} → {json_path}, {csv_path}")


# ==========================
# MAIN
# ==========================

def main():
    ensure_dir(DATA_DIR)

    downloader = HistoricalDownloader()

    for symbol in SYMBOLS:
        print(f"\n==============================")
        print(f"Symbol: {symbol}")
        print(f"==============================")

        downloader.download_for_symbol(
            symbol=symbol,
            timeframes=TIMEFRAMES,
            total_needed=CANDLES_PER_TF,
            exchanges=EXCHANGES
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
