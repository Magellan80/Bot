# historical_downloader.py
# Загрузчик истории с Bybit (v5 API)
# Поддерживает:
# - 5m, 15m, 1h, 4h
# - разные лимиты для каждого TF
# - сохранение в JSON и CSV

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

# Только нужные TF
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# Правильные лимиты под твой бэктестер
CANDLES_PER_TF = {
    "5m": 20000,
    "15m": 6666,
    "1h": 1666,
    "4h": 416,
}

EXCHANGES = ["bybit"]


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
    pct = current / max(total, 1) * 100
    print(f"{prefix}: {current}/{total} ({pct:.1f}%)")


# ==========================
# БАЗОВЫЙ КЛАСС
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
                print(f"Request error: {e} | retry {retries}/{self.max_retries}")
                time.sleep(0.5)
                if retries >= self.max_retries:
                    print("Max retries reached.")
                    return None


# ==========================
# BYBIT DOWNLOADER
# ==========================

class BybitDownloader(BaseDownloader):

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    INTERVAL_MAP = {
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
    }

    def __init__(self, category: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.category = category

    def download(self, symbol: str, interval: str, total_needed: int) -> List[Dict]:
        print(f"\n[Bybit] Downloading {symbol} {interval} ...")

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

            data = self._request("GET", self.BASE_URL, params)
            if not data:
                break

            if data.get("retCode") != 0:
                print("API error:", data.get("retMsg"))
                break

            list_data = data["result"]["list"]
            if not list_data:
                break

            list_data = list(reversed(list_data))
            all_data = list_data + all_data

            oldest_ts = int(list_data[0][0])
            end_time = oldest_ts - 1

            print_progress(f"[Bybit] {symbol} {interval}", len(all_data), total_needed)
            time.sleep(self.pause)

        candles = []
        for k in all_data:
            candles.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
            })

        print(f"[Bybit] {symbol} {interval} DONE. Total: {len(candles)} candles.")
        return candles


# ==========================
# HIGH-LEVEL ORCHESTRATOR
# ==========================

class HistoricalDownloader:

    def __init__(self):
        self.bybit = BybitDownloader()

    def download_for_symbol(self, symbol: str):
        for tf in TIMEFRAMES:
            candles = self.bybit.download(symbol, tf, CANDLES_PER_TF[tf])
            self._save(symbol, tf, candles)

    def _save(self, symbol: str, tf: str, candles: List[Dict]):
        base_dir = os.path.join(DATA_DIR, "bybit", symbol.upper())
        ensure_dir(base_dir)

        json_path = os.path.join(base_dir, f"{tf}.json")
        csv_path = os.path.join(base_dir, f"{tf}.csv")

        save_json(json_path, candles)
        save_csv(csv_path, candles)

        print(f"[SAVE] {symbol} {tf} → {json_path}")


# ==========================
# MAIN
# ==========================

def main():
    ensure_dir(DATA_DIR)

    downloader = HistoricalDownloader()

    for symbol in SYMBOLS:
        print("\n==============================")
        print(f"Symbol: {symbol}")
        print("==============================")
        downloader.download_for_symbol(symbol)

    print("\nAll done.")


if __name__ == "__main__":
    main()
