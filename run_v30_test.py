# run_v30_test.py
# V31 Institutional Single-Symbol Backtest (Local JSON Loader)

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

# ==========================================
# CONFIG
# ==========================================

SYMBOL = "ETHUSDT"
EXCHANGE = "bybit"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", EXCHANGE, SYMBOL)

INITIAL_BALANCE = 10000

# ==========================================
# LOAD LOCAL DATA
# ==========================================

def load_json(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading local data...\n")

candles_5m = load_json(os.path.join(DATA_DIR, "5m.json"))
htf_15m = load_json(os.path.join(DATA_DIR, "15m.json"))
htf_1h = load_json(os.path.join(DATA_DIR, "1h.json"))
htf_4h = load_json(os.path.join(DATA_DIR, "4h.json"))

print(
    f"Loaded: 5m={len(candles_5m)}, "
    f"15m={len(htf_15m)}, 1h={len(htf_1h)}, 4h={len(htf_4h)}\n"
)

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

# ==========================================
# OUTPUT
# ==========================================

print("\n==============================")
print("V31 INSTITUTIONAL RESULTS")
print("==============================")

for k, v in result["stats"].items():
    print(f"{k}: {v}")

# save full report
out_path = os.path.join(os.path.dirname(__file__), f"report_{SYMBOL}_v31.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nFull report saved to: {out_path}")
