import os
import sys
import json
import time
import queue
import signal
import traceback
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Предполагаем, что backtester лежит в корне Bot/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from context import create_base_context
from data_layer import load_klines_offline
from screener import Screener
from trading_engine import TradingEngine


# ===== НАСТРОЙКИ БЭКТЕСТА =====

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT",
    "BNBUSDT", "DOGEUSDT", "ADAUSDT", "OPUSDT", "ARBUSDT",
    "AVAXUSDT", "TONUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "INJUSDT", "RUNEUSDT",
]

TIMEFRAME = "5m"
CANDLE_LIMIT = 20000

RESULTS_CSV = "backtest_trades.csv"
SUMMARY_JSON = "backtest_summary.json"

N_PROCESSES = min(8, mp.cpu_count())


# ===== СТРУКТУРА СДЕЛКИ =====

@dataclass
class TradeResult:
    symbol: str
    side: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r: float
    signal_type: str
    filters_passed: bool
    extra: Dict[str, Any]


# ===== ЗАГРУЗКА ИСТОРИИ =====

def load_history(symbol: str, timeframe: str, limit: int):
    return load_klines_offline(symbol, timeframe, limit=limit)


# ===== БЭКТЕСТ ДЛЯ ОДНОГО СИМВОЛА =====

def run_backtest_for_symbol(symbol: str) -> List[TradeResult]:
    trades: List[TradeResult] = []

    try:
        base_ctx = create_base_context()
        base_ctx["symbol"] = symbol
        base_ctx["timeframe"] = TIMEFRAME
        base_ctx["backtest_mode"] = True

        candles = load_history(symbol, TIMEFRAME, CANDLE_LIMIT)

        screener = Screener(base_ctx)
        engine = TradingEngine(base_ctx, backtest_mode=True)

        signals = screener.generate_signals_from_history(symbol, candles)

        for sig in signals:
            result = engine.on_signal_backtest(sig)
            if not result:
                continue

            tr = TradeResult(
                symbol=symbol,
                side=result["side"],
                entry_time=result["entry_time"],
                exit_time=result["exit_time"],
                entry_price=result["entry_price"],
                exit_price=result["exit_price"],
                size=result["size"],
                pnl=result["pnl"],
                r=result["r"],
                signal_type=result.get("signal_type", "unknown"),
                filters_passed=result.get("filters_passed", True),
                extra={
                    "regime": result.get("regime"),
                    "atr": result.get("atr"),
                    "vol_cluster": result.get("vol_cluster"),
                },
            )
            trades.append(tr)

        return trades

    except Exception as e:
        print(f"[{symbol}] ERROR in backtest: {e}")
        traceback.print_exc()
        return trades


# ===== РАБОЧИЙ ПРОЦЕСС =====

def worker(symbol_queue: mp.Queue, result_queue: mp.Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            symbol = symbol_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[WORKER] Start backtest for {symbol}")
        t0 = time.time()
        trades = run_backtest_for_symbol(symbol)
        dt = time.time() - t0
        print(f"[WORKER] Done {symbol}: {len(trades)} trades in {dt:.1f}s")

        for tr in trades:
            result_queue.put(asdict(tr))


# ===== АГРЕГАЦИЯ И СТАТИСТИКА =====

def aggregate_results(result_queue: mp.Queue) -> List[Dict[str, Any]]:
    results = []
    while True:
        try:
            item = result_queue.get_nowait()
        except queue.Empty:
            break
        results.append(item)
    return results


def compute_statistics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0}

    import math
    from collections import defaultdict

    total_trades = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    winrate = len(wins) / total_trades if total_trades else 0.0
    avg_r = sum(t["r"] for t in trades) / total_trades

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss = sum(t["pnl"] for t in losses) if losses else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else math.inf

    trades_sorted = sorted(trades, key=lambda x: x["exit_time"])
    equity = 0.0
    max_equity = 0.0
    max_dd = 0.0
    equity_curve = []

    for t in trades_sorted:
        equity += t["pnl"]
        max_equity = max(max_equity, equity)
        dd = max_equity - equity
        max_dd = max(max_dd, dd)
        equity_curve.append({"time": t["exit_time"], "equity": equity})

    by_symbol = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in trades:
        s = t["symbol"]
        by_symbol[s]["trades"] += 1
        by_symbol[s]["pnl"] += t["pnl"]

    by_signal_type = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in trades:
        st = t.get("signal_type", "unknown")
        by_signal_type[st]["trades"] += 1
        by_signal_type[st]["pnl"] += t["pnl"]

    return {
        "trades": total_trades,
        "total_pnl": total_pnl,
        "winrate": winrate,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "by_symbol": {k: dict(v) for k, v in by_symbol.items()},
        "by_signal_type": {k: dict(v) for k, v in by_signal_type.items()},
        "equity_curve": equity_curve,
    }


def save_results(trades: List[Dict[str, Any]], stats: Dict[str, Any]):
    if trades:
        import csv
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
            writer.writeheader()
            writer.writerows(trades)
        print(f"[SAVE] Trades saved to {RESULTS_CSV}")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    print(f"[SAVE] Summary saved to {SUMMARY_JSON}")


# ===== ФИНАЛЬНЫЙ ОТЧЁТ =====

def print_final_report(stats):
    print("\n==============================")
    print("V31 MULTI-SYMBOL BACKTEST RESULTS")
    print("==============================")

    trades = stats.get("trades", 0)
    winrate = stats.get("winrate", 0.0)
    expectancy = stats.get("avg_r", 0.0)
    profit_factor = stats.get("profit_factor", 0.0)

    starting_balance = 10000
    final_balance = starting_balance + stats.get("total_pnl", 0.0)
    return_pct = (final_balance - starting_balance) / starting_balance * 100

    max_dd = stats.get("max_drawdown", 0.0)

    print(f"trades: {trades}")
    print(f"winrate: {winrate:.2f}")
    print(f"expectancy: {expectancy:.2f}")
    print(f"profit_factor: {profit_factor:.2f}")
    print(f"final_balance: {final_balance:.2f}")
    print(f"return_pct: {return_pct:.2f}")
    print(f"max_drawdown_pct: {max_dd:.2f}")

    print("\nFull report saved to:", SUMMARY_JSON)


# ===== MAIN =====

def main():
    mp.set_start_method("spawn", force=True)

    symbol_queue = mp.Queue()
    result_queue = mp.Queue()

    for s in SYMBOLS:
        symbol_queue.put(s)

    processes = []
    t0 = time.time()

    for _ in range(N_PROCESSES):
        p = mp.Process(target=worker, args=(symbol_queue, result_queue))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt, terminating workers...")
        for p in processes:
            p.terminate()

    dt = time.time() - t0
    print(f"[MAIN] Backtest finished in {dt/60:.1f} minutes")

    trades = aggregate_results(result_queue)
    print(f"[MAIN] Collected {len(trades)} trades")

    stats = compute_statistics(trades)
    print("[MAIN] Summary:")
    print(json.dumps(stats, ensure_ascii=False, indent=4))

    save_results(trades, stats)

    print_final_report(stats)


if __name__ == "__main__":
    main()
