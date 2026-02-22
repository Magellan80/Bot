# backtester_v31_mp.py
# Мультипроцессорный бэктестер для V31/V30 стека
# - читает свечи из data/<exchange>/<symbol>/<tf>.json
# - создаёт движки внутри каждого процесса
# - вызывает V30BacktestEngine.run()
# - собирает результаты в общий отчёт
# - выводит сделки с указанием монеты

import os
import json
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any

# === ПУТИ =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# === НАСТРОЙКИ ===============================================

EXCHANGE = "bybit"
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT",
    "BNBUSDT", "DOGEUSDT", "ADAUSDT", "OPUSDT", "ARBUSDT",
    "AVAXUSDT", "TONUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "INJUSDT", "RUNEUSDT"
]

N_PROCESSES = max(2, min(cpu_count(), 8))


# === ИМПОРТ ДВИЖКОВ ==========================================

from v30.v30_backtest_engine import V30BacktestEngine
from v30.elite_structure_engine import EliteStructureEngine
from v30.elite_regime_engine import EliteRegimeEngine
from v30.elite_htf_sync import EliteHTFSync
from v30.elite_trend_engine import EliteTrendEngine
from v30.elite_reversal_engine import EliteReversalEngine
from v30.elite_signal_router import EliteSignalRouter
from v30.elite_exit_engine import EliteExitEngine
from v30.risk_engine_v31 import RiskEngineV31


# === УТИЛИТЫ ==================================================

def load_json(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def load_symbol_data(symbol: str) -> Dict[str, List[Dict]]:
    base = os.path.join(DATA_DIR, EXCHANGE, symbol.upper())
    result = {}
    for tf in TIMEFRAMES:
        path = os.path.join(base, f"{tf}.json")
        result[tf] = load_json(path)
    return result


# === РАБОЧАЯ ФУНКЦИЯ ДЛЯ ПРОЦЕССА =============================

def run_for_symbol(symbol: str) -> Dict[str, Any]:
    print(f"\n[BACKTEST] Start {symbol}")

    data = load_symbol_data(symbol)

    candles_5m = data["5m"]
    htf_15m = data["15m"]
    htf_1h = data["1h"]
    htf_4h = data["4h"]

    if len(candles_5m) < 1000:
        print(f"[BACKTEST] Skip {symbol}: not enough 5m data")
        return {"symbol": symbol, "status": "no_data", "result": None}

    # === создаём движки ===
    structure_engine = EliteStructureEngine()
    regime_engine = EliteRegimeEngine()
    htf_sync = EliteHTFSync()
    trend_engine = EliteTrendEngine()
    reversal_engine = EliteReversalEngine()
    router = EliteSignalRouter()
    exit_engine = EliteExitEngine()
    risk_engine = RiskEngineV31()

    engine = V30BacktestEngine()

    # === ПАТЧ PRINT ДЛЯ ДОБАВЛЕНИЯ НАЗВАНИЯ МОНЕТЫ ===
    import builtins
    original_print = print

    def patched_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)

        # сделки
        if text.startswith("OPEN") or text.startswith("CLOSE"):
            original_print(f"[{symbol}] {text}", **kwargs)
            return

        # прогресс
        if text.startswith("Progress"):
            original_print(f"[{symbol}] {text}", **kwargs)
            return

        # всё остальное — как есть
        original_print(text, **kwargs)

    builtins.print = patched_print

    try:
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
            initial_balance=10000
        )
        status = "ok"

    except Exception as e:
        original_print(f"[ERROR] Backtest failed for {symbol}: {e}")
        result = None
        status = "error"

    finally:
        builtins.print = original_print

    return {
        "symbol": symbol,
        "status": status,
        "result": result
    }


# === АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ====================================

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "ok": 0,
        "errors": 0,
        "no_data": 0,
        "details": {}
    }

    for r in results:
        symbol = r["symbol"]
        status = r["status"]
        summary["details"][symbol] = r["result"]

        if status == "ok":
            summary["ok"] += 1
        elif status == "no_data":
            summary["no_data"] += 1
        else:
            summary["errors"] += 1

    return summary


# === MAIN =====================================================

def main():
    print(f"Running backtest on {len(SYMBOLS)} symbols using {N_PROCESSES} processes...\n")

    with Pool(N_PROCESSES) as pool:
        results = pool.map(run_for_symbol, SYMBOLS)

    summary = aggregate(results)

    out_path = os.path.join(BASE_DIR, "backtest_summary_v31.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(f"OK:      {summary['ok']}")
    print(f"NO DATA: {summary['no_data']}")
    print(f"ERRORS:  {summary['errors']}")
    print(f"Saved:   {out_path}")


if __name__ == "__main__":
    main()
