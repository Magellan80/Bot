# run_v30_test.py
# V31 Institutional Backtest with Auto-Downloader (Bybit)

import os
import json

from v30.v30_backtest_engine import V30BacktestEngine
from v30.elite_structure_engine import EliteStructureEngine
from v30.elite_regime_engine import EliteRegimeEngine
from v30.elite_htf_sync import EliteHTFSync
from v30.elite_trend_engine import EliteTrendEngine
from v30.elite_reversal_engine import EliteReversalEngine
from v30.elite_signal_router import EliteSignalRouter
from v30.elite_exit_engine import EliteExitEngine
from v30.risk_engine_v31 import RiskEngineV31

from historical_downloader import BybitDownloader, ensure_dir


# ==========================================
# CONFIG
# ==========================================

SYMBOL = "UNIUSDT"
EXCHANGE = "bybit"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", EXCHANGE, SYMBOL)

# Лимиты свечей под твой бэктестер
CANDLE_LIMITS = {
    "5m": 20000,
    "15m": 6666,
    "1h": 1666,
    "4h": 416,
}

INITIAL_BALANCE = 10000


# ==========================================
# LOAD OR DOWNLOAD
# ==========================================

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_download(symbol: str, tf: str, needed: int):
    """
    1) Проверяет, есть ли локальный файл
    2) Если есть и свечей достаточно → возвращает
    3) Если нет → скачивает с Bybit и сохраняет
    """

    ensure_dir(DATA_DIR)

    json_path = os.path.join(DATA_DIR, f"{tf}.json")

    # --- Пытаемся загрузить локальный файл ---
    data = load_json(json_path)
    if data and len(data) >= needed:
        print(f"[OK] Loaded {len(data)} {tf} candles from local file.")
        return data

    # --- Если файла нет или мало свечей → скачиваем ---
    print(f"[DL] Downloading {needed} candles for {symbol} {tf} from Bybit...")

    downloader = BybitDownloader()
    candles = downloader.download(symbol, tf, needed)

    if candles:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(candles, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {tf} saved → {json_path}")

    return candles


# ==========================================
# INIT ENGINES
# ==========================================

structure_engine = EliteStructureEngine()
regime_engine = EliteRegimeEngine()
htf_sync = EliteHTFSync()

trend_engine = EliteTrendEngine()
reversal_engine = EliteReversalEngine()

router = EliteSignalRouter()
exit_engine = EliteExitEngine()

risk_engine = RiskEngineV31(
    base_risk=0.008,      # 0.8%
    max_portfolio_heat=0.25
)

engine = V30BacktestEngine()


# ==========================================
# RUN
# ==========================================

def main():
    print("\nLoading data (auto mode)...\n")

    candles_5m = load_or_download(SYMBOL, "5m", CANDLE_LIMITS["5m"])
    htf_15m    = load_or_download(SYMBOL, "15m", CANDLE_LIMITS["15m"])
    htf_1h     = load_or_download(SYMBOL, "1h",  CANDLE_LIMITS["1h"])
    htf_4h     = load_or_download(SYMBOL, "4h",  CANDLE_LIMITS["4h"])

    print(
        f"\nLoaded: 5m={len(candles_5m)}, "
        f"15m={len(htf_15m)}, 1h={len(htf_1h)}, 4h={len(htf_4h)}\n"
    )

    print("\nRunning V31 Institutional Backtest...\n")

    result = engine.run(
        candles_5m=candles_5m,
        structure_engine=structure_engine,
        regime_engine=regime_engine,
        htf_sync=htf_sync,
        trend_engine=trend_engine,
        reversal_engine=reversal_engine,
        router=router,
        exit_engine=exit_engine,
        htf_15m=htf_15m,
        htf_1h=htf_1h,
        htf_4h=htf_4h,
        risk_engine=risk_engine,
        initial_balance=INITIAL_BALANCE
    )

    print("\n==============================")
    print("V31 INSTITUTIONAL RESULTS")
    print("==============================")

    for k, v in result["stats"].items():
        print(f"{k}: {v}")

    out_path = os.path.join(os.path.dirname(__file__), f"report_{SYMBOL}_v31.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
