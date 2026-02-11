import json
import os
import time
from typing import Dict, Any

MEMORY_FILE = "symbol_memory.json"
MAX_HISTORY = 200


def _load_raw() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # v2.1: Валидация что это dict
            if not isinstance(data, dict):
                print(f"⚠️ [SYMBOL_MEMORY] Invalid data type in {MEMORY_FILE}, resetting")
                return {}
            return data
    except json.JSONDecodeError as e:
        print(f"⚠️ [SYMBOL_MEMORY] Corrupted JSON in {MEMORY_FILE}: {e}")
        # Создаём бэкап
        backup_file = f"{MEMORY_FILE}.corrupt.{int(time.time())}"
        import shutil
        shutil.copy(MEMORY_FILE, backup_file)
        print(f"⚠️ [SYMBOL_MEMORY] Backup saved to {backup_file}")
        return {}
    except Exception as e:
        print(f"⚠️ [SYMBOL_MEMORY] Error loading {MEMORY_FILE}: {e}")
        return {}


def _save_raw(data: Dict[str, Any]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_symbol_memory(symbol: str) -> Dict[str, Any]:
    data = _load_raw()
    return data.get(symbol, {})


def get_symbol_state(symbol: str) -> Dict[str, Any]:
    data = _load_raw()
    entry = data.get(symbol, {})
    state = entry.get("state", {})
    return state if isinstance(state, dict) else {}


def set_symbol_state(symbol: str, state: Dict[str, Any]) -> None:
    data = _load_raw()
    entry = data.get(symbol, {})
    entry["state"] = state
    data[symbol] = entry
    _save_raw(data)


def clear_symbol_state(symbol: str) -> None:
    data = _load_raw()
    entry = data.get(symbol, {})
    if "state" in entry:
        entry.pop("state", None)
        data[symbol] = entry
        _save_raw(data)


def _classify_behavior(stats: Dict[str, Any]) -> Dict[str, Any]:
    pump_cnt = stats.get("pump_cnt", 0)
    dump_cnt = stats.get("dump_cnt", 0)
    total = max(stats.get("total_signals", 0), 1)

    pump_prob = pump_cnt / total
    dump_prob = dump_cnt / total

    avg_atr = stats.get("avg_atr", 0.0)
    avg_vol = stats.get("avg_volatility", 0.0)
    btc_corr = stats.get("avg_btc_factor", 1.0)

    regime = "neutral"

    if pump_prob > 0.35:
        regime = "pumpy"
    if dump_prob > 0.35:
        regime = "dumpy"
    if avg_vol > 5:
        regime = "chaotic"
    if abs(avg_vol) < 1 and avg_atr < 0.3:
        regime = "stable"
    if abs(avg_vol) > 2 and abs(avg_vol) < 5:
        regime = "trending"

    return {
        "pump_probability": pump_prob,
        "dump_probability": dump_prob,
        "avg_atr": avg_atr,
        "avg_volatility": avg_vol,
        "btc_correlation_proxy": btc_corr,
        "regime": regime,
    }


def update_symbol_memory(symbol: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    snapshot:
      {
        "atr_1m": float,
        "trend_score": float,
        "is_pump": bool,
        "is_dump": bool,
        "btc_factor": float
      }
    """
    data = _load_raw()
    now = time.time()

    entry = data.get(symbol, {
        "history": [],
        "stats": {
            "total_signals": 0,
            "pump_cnt": 0,
            "dump_cnt": 0,
            "sum_atr": 0.0,
            "sum_vol": 0.0,
            "sum_btc_factor": 0.0,
        },
    })

    hist = entry.get("history", [])
    stats = entry.get("stats", {})
    state = entry.get("state", {})

    hist.append({
        "ts": now,
        "atr_1m": snapshot.get("atr_1m", 0.0),
        "trend_score": snapshot.get("trend_score", 0.0),
        "is_pump": bool(snapshot.get("is_pump", False)),
        "is_dump": bool(snapshot.get("is_dump", False)),
        "btc_factor": snapshot.get("btc_factor", 1.0),
    })

    if len(hist) > MAX_HISTORY:
        hist = hist[-MAX_HISTORY:]

    stats["total_signals"] = stats.get("total_signals", 0) + 1
    if snapshot.get("is_pump"):
        stats["pump_cnt"] = stats.get("pump_cnt", 0) + 1
    if snapshot.get("is_dump"):
        stats["dump_cnt"] = stats.get("dump_cnt", 0) + 1

    stats["sum_atr"] = stats.get("sum_atr", 0.0) + snapshot.get("atr_1m", 0.0)
    stats["sum_vol"] = stats.get("sum_vol", 0.0) + snapshot.get("trend_score", 0.0)
    stats["sum_btc_factor"] = stats.get("sum_btc_factor", 0.0) + snapshot.get("btc_factor", 1.0)

    total = max(stats["total_signals"], 1)
    stats["avg_atr"] = stats["sum_atr"] / total
    stats["avg_volatility"] = stats["sum_vol"] / total
    stats["avg_btc_factor"] = stats["sum_btc_factor"] / total

    profile = _classify_behavior(stats)

    entry["history"] = hist
    entry["stats"] = stats
    entry["profile"] = profile
    if state:
        entry["state"] = state
    entry["updated_ts"] = now

    data[symbol] = entry
    _save_raw(data)

    return entry
