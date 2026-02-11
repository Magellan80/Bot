import time
import csv
from enum import Enum
from typing import Dict, Optional, Any, List, Tuple

import asyncio

from config import get_current_mode, load_settings, save_settings
from smart_filters_v3 import _GLOBAL_FILTERS as smartfilters


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradingEngine:
    """
    Trading Engine 4.5 Adaptive Risk Edition
    ----------------------------------------
    Базируется на 4.4 Self-Healing Edition +:

    - Position Sizing 2.0:
        * размер позиции зависит от equity, stop_distance, confidence, market_risk,
          symbol_regime_strength и волатильности
    - Risk Filters 2.0:
        * глобальные фильтры по market_risk, btc_regime, symbol_regime, vol_cluster
    - Confidence Thresholds:
        * жёсткий порог на минимальную уверенность
        * мягкая шкала для размера позиции

    Всё остальное (ATR, TP1/TP2/TP3, трейлинг 4.2, kill-switch, self-healing sync)
    сохранено и совместимо с существующей архитектурой.
    """

    ENGINE_MODES = {
        "A": {
            "name": "Aggressive",
            "atr_mult": 1.5,
            "tp1_r": 1.0,
            "tp2_mult": 2.0,
            "tp3_mult": 3.0,
            "trend_filter": False,
            "risk_filter": False,
            "allow_counter_trend": True,
            "min_rating": 40,
            "max_risk_score": 90,
            "cooldown_sec": 15,
            "max_entry_slippage_pct": 0.4,
            "max_atr_pct": 4.0,
            "max_spread_pct": 0.25,
            "enable_candle_filter": False,
            "max_candle_range_pct": 4.0,
        },
        "B": {
            "name": "Balanced",
            "atr_mult": 2.0,
            "tp1_r": 1.0,
            "tp2_mult": 2.0,
            "tp3_mult": 3.0,
            "trend_filter": True,
            "risk_filter": True,
            "allow_counter_trend": False,
            "min_rating": 45,
            "max_risk_score": 80,
            "cooldown_sec": 30,
            "max_entry_slippage_pct": 0.3,
            "max_atr_pct": 3.0,
            "max_spread_pct": 0.20,
            "enable_candle_filter": True,
            "max_candle_range_pct": 3.0,
        },
        "C": {
            "name": "Conservative",
            "atr_mult": 2.8,
            "tp1_r": 1.0,
            "tp2_mult": 3.0,
            "tp3_mult": 5.0,
            "trend_filter": True,
            "risk_filter": True,
            "allow_counter_trend": False,
            "strict_trend": True,
            "min_rating": 55,
            "max_risk_score": 70,
            "cooldown_sec": 60,
            "max_entry_slippage_pct": 0.2,
            "max_atr_pct": 2.5,
            "max_spread_pct": 0.15,
            "enable_candle_filter": True,
            "max_candle_range_pct": 2.0,
        },
    }

    COMMISSION_RATE = 0.00055  # 0.055%

    # параметры идеального трейлинга
    BE_OFFSET_LONG_PCT = 0.0003
    BE_OFFSET_SHORT_PCT = 0.0003

    # минимальный шаг улучшения SL (0.05%)
    MIN_IMPROVE_PCT = 0.0005

    # 4.2: параметры умного трейлинга
    PARTIAL_BE_MFE_R = 1.2
    PARTIAL_BE_LOCK_R = 0.3
    MFE_AGGRESSIVE_R = 3.0
    LOCK_IN_MIN_R_AFTER_TP2 = 1.0
    LAST_PUSH_NEAR_TP3_PCT = 0.3
    TIME_TIGHTEN_SEC = 3600
    TIME_TIGHTEN_R = 0.3

    # 4.5: базовые параметры риск‑менеджмента на сделку
    BASE_RISK_PCT = 0.01          # 1% от equity на сделку до модификаторов
    MIN_CONFIDENCE_HARD = 0.55    # ниже — не торгуем
    MID_CONFIDENCE = 0.75         # между MIN и MID — урезаем размер

    def __init__(self, broker):
        self.broker = broker
        self.positions: Dict[str, Dict[str, Any]] = {}
        # лимит одновременных позиций
        self.max_positions = 2

        # лимит общей экспозиции (notional), например не более 5× equity
        self.max_exposure_mult = 5.0

        self.daily_loss_limit_pct = 5.0
        self.daily_start_equity: Optional[float] = None
        self.daily_loss_hit = False
        self.daily_reset_date: Optional[str] = None  # YYYY-MM-DD

        self.csv_file = "trades.csv"
        self._ensure_csv_header()

        self.last_signal_time: Dict[str, float] = {}

        # baseline из settings.json
        self._load_daily_baseline_from_settings()

        # kill-switch / watchdog
        self.kill_switch_triggered: bool = False
        self.last_price_heartbeat: float = time.time()
        self.rest_error_count: int = 0

        # soft-warning (мягкий режим)
        self.soft_warning: bool = False
        self.soft_warning_ts: Optional[float] = None

        # дополнительные heartbeat-метрики для self-healing
        self.last_signal_heartbeat: float = time.time()
        self.last_sync_heartbeat: float = time.time()
        self.last_kill_switch_heartbeat: float = time.time()

        # счётчик внутренних ошибок логики (не REST)
        self.internal_error_count: int = 0

    # ---------------------------------------------------------
    # SETTINGS: baseline
    # ---------------------------------------------------------

    def _load_daily_baseline_from_settings(self):
        try:
            settings = load_settings()
        except Exception as e:
            self._log_error(f"load_settings_error;error={repr(e)}")
            return

        daily_equity = settings.get("daily_start_equity")
        daily_date = settings.get("daily_reset_date")

        if isinstance(daily_equity, (int, float)) and daily_equity > 0:
            self.daily_start_equity = float(daily_equity)
        else:
            self.daily_start_equity = None

        if isinstance(daily_date, str) and len(daily_date) == 10:
            self.daily_reset_date = daily_date
        else:
            self.daily_reset_date = None

    def _save_daily_baseline_to_settings(self):
        try:
            settings = load_settings()
        except Exception as e:
            self._log_error(f"load_settings_for_save_error;error={repr(e)}")
            settings = {}

        if self.daily_start_equity is not None:
            settings["daily_start_equity"] = float(self.daily_start_equity)
        else:
            settings.pop("daily_start_equity", None)

        if self.daily_reset_date is not None:
            settings["daily_reset_date"] = self.daily_reset_date
        else:
            settings.pop("daily_reset_date", None)

        try:
            save_settings(settings)
        except Exception as e:
            self._log_error(f"save_settings_error;error={repr(e)}")

    # ---------------------------------------------------------
    # ЛОГИ
    # ---------------------------------------------------------

    def _log_error(self, payload: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {payload}\n"
        with open("error.log", "a", encoding="utf-8") as f:
            f.write(line)

    def log_trade(self, payload: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {payload}\n"
        with open("trades.log", "a", encoding="utf-8") as f:
            f.write(line)

    # ---------------------------------------------------------
    # WATCHDOG / KILL-SWITCH ВСПОМОГАТЕЛЬНЫЕ
    # ---------------------------------------------------------

    def _register_heartbeat(self, kind: str):
        now = time.time()
        if kind == "price":
            self.last_price_heartbeat = now
        elif kind == "signal":
            self.last_signal_heartbeat = now
        elif kind == "sync":
            self.last_sync_heartbeat = now
        elif kind == "kill":
            self.last_kill_switch_heartbeat = now

    def _register_price_heartbeat(self):
        self._register_heartbeat("price")

    def _register_rest_error(self, context: str):
        self.rest_error_count += 1
        self._log_error(f"rest_error;context={context};count={self.rest_error_count}")

    def _register_internal_error(self, context: str):
        self.internal_error_count += 1
        self._log_error(f"internal_error;context={context};count={self.internal_error_count}")

    async def emergency_flatten(self, reason: str = "KILL_SWITCH"):
        """
        Аварийное закрытие всех позиций и отмена всех ордеров.
        После срабатывания kill_switch_triggered = True и новые входы блокируются.
        """
        if self.kill_switch_triggered:
            return

        self.kill_switch_triggered = True
        self.log_trade(f"event=KILL_SWITCH_TRIGGERED;reason={reason}")

        # попытка закрыть все позиции
        try:
            real_positions = await self.broker.get_open_positions()
        except Exception as e:
            self._log_error(f"kill_switch:get_open_positions_error;error={repr(e)}")
            real_positions = []

        for p in real_positions:
            symbol = p.get("symbol")
            side_str = (p.get("side") or "").upper()
            size = float(p.get("size") or 0)
            if not symbol or size <= 0:
                continue

            side = PositionSide.LONG if side_str == "BUY" else PositionSide.SHORT
            try:
                await self.broker.close_market_order(symbol, side, size)
                self.log_trade(
                    f"event=KILL_SWITCH_CLOSE;symbol={symbol};side={side.value};size={size:.6f}"
                )
            except Exception as e:
                self._log_error(
                    f"kill_switch:close_market_error;symbol={symbol};error={repr(e)}"
                )

        # попытка отменить все ордера
        try:
            all_orders = await self.broker.get_open_orders()
        except Exception as e:
            self._log_error(f"kill_switch:get_open_orders_error;error={repr(e)}")
            all_orders = []

        for o in all_orders:
            symbol = o.get("symbol")
            order_id = o.get("orderId")
            if not symbol or not order_id:
                continue
            try:
                await self.broker.cancel_order(symbol, order_id)
                self.log_trade(
                    f"event=KILL_SWITCH_CANCEL;symbol={symbol};order_id={order_id}"
                )
            except Exception as e:
                self._log_error(
                    f"kill_switch:cancel_order_error;symbol={symbol};order_id={order_id};error={repr(e)}"
                )

        # локальное состояние тоже чистим
        self.positions.clear()

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    def _ensure_csv_header(self):
        try:
            with open(self.csv_file, "r", encoding="utf-8") as f:
                if f.readline():
                    return
        except FileNotFoundError:
            pass

        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "trade_id", "symbol", "side", "entry", "exit", "pnl_net"])

    def write_csv_trade(self, trade_id: str, symbol: str, side: str, entry: float, exit_price: float, pnl_net: float):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts, trade_id, symbol, side, f"{entry:.4f}", f"{exit_price:.4f}", f"{pnl_net:.4f}"])

    # ---------------------------------------------------------
    # ATR
    # ---------------------------------------------------------

    @staticmethod
    def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None

        trs = []
        for i in range(period):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i + 1]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)

        if not trs:
            return None
        return sum(trs) / len(trs)

    # ---------------------------------------------------------
    # РАСЧЁТ РАЗМЕРА ПОЗИЦИИ (4.5: адаптивный)
    # ---------------------------------------------------------

    async def compute_position_size(self, stop_distance: float, s: Optional[Dict[str, Any]] = None) -> float:
        """
        Адаптивный расчёт размера позиции:
        - базовый риск BASE_RISK_PCT от equity
        - модификация по market_risk
        - модификация по confidence
        - модификация по symbol_regime_strength
        - модификация по волатильности (atr_regime / vol_cluster)
        """
        try:
            equity = await self.broker.get_equity()
        except Exception as e:
            self._register_rest_error("compute_position_size:get_equity")
            return 0.0

        if not equity or equity <= 0 or stop_distance <= 0:
            return 0.0

        s = s or {}

        # --- market_risk (0..1, по умолчанию 0) ---
        market_risk = float(s.get("market_risk", 0.0) or 0.0)
        market_risk = max(0.0, min(market_risk, 1.0))

        # базовый риск на сделку
        risk_pct = self.BASE_RISK_PCT

        # при высоком market_risk уменьшаем риск
        # при market_risk = 1.0 → риск в 2 раза меньше
        risk_pct *= (1.0 - 0.5 * market_risk)

        # --- confidence (0..1, по умолчанию 1.0) ---
        confidence = float(s.get("confidence", 1.0) or 1.0)
        confidence = max(0.0, min(confidence, 1.0))

        # --- symbol_regime_strength (0.3..1.5, по умолчанию 1.0) ---
        symbol_regime_strength = s.get("symbol_regime_strength")
        if symbol_regime_strength is None:
            symbol_regime_strength = s.get("symbol_regime_score", 1.0)
        try:
            symbol_regime_strength = float(symbol_regime_strength)
        except (TypeError, ValueError):
            symbol_regime_strength = 1.0
        symbol_regime_strength = max(0.3, min(symbol_regime_strength, 1.5))

        # --- волатильность (atr_regime / vol_cluster) ---
        vol_factor = 1.0
        atr_regime = s.get("atr_regime")
        vol_cluster = s.get("vol_cluster")

        # если есть vol_cluster — при high_vol уменьшаем размер
        if isinstance(vol_cluster, str):
            vc = vol_cluster.lower()
            if "high" in vc:
                vol_factor *= 0.8
            elif "low" in vc:
                vol_factor *= 1.1

        # если есть atr_regime — лёгкая корректировка
        if isinstance(atr_regime, str):
            ar = atr_regime.lower()
            if ar == "high":
                vol_factor *= 0.9
            elif ar == "low":
                vol_factor *= 1.05

        # --- базовый размер по риску ---
        risk_amount = equity * risk_pct
        size = risk_amount / stop_distance

        # --- модификации ---
        size *= confidence
        size *= symbol_regime_strength
        size *= vol_factor

        return max(size, 0.0)

    # ---------------------------------------------------------
    # ДНЕВНОЙ РИСК + DAILY BASELINE RESET
    # ---------------------------------------------------------

    async def check_daily_risk(self) -> bool:
        try:
            equity = await self.broker.get_equity()
        except Exception as e:
            self._register_rest_error("check_daily_risk:get_equity")
            return False

        if equity is None:
            return False

        utc_now = time.gmtime()
        today_str = time.strftime("%Y-%m-%d", utc_now)

        if self.daily_reset_date != today_str:
            self.daily_start_equity = equity
            self.daily_reset_date = today_str
            self.daily_loss_hit = False

            self._save_daily_baseline_to_settings()

            self.log_trade(
                f"event=DAILY_RESET;date={today_str};baseline={equity:.4f}"
            )
            return False

        if self.daily_start_equity is None:
            self.daily_start_equity = equity
            self._save_daily_baseline_to_settings()
            return False

        drawdown_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100

        if drawdown_pct >= self.daily_loss_limit_pct:
            if not self.daily_loss_hit:
                self.daily_loss_hit = True
                self.log_trade(
                    f"event=DAILY_LIMIT;dd={drawdown_pct:.2f};baseline={self.daily_start_equity:.4f};equity={equity:.4f}"
                )
            return True

        return False

    # ---------------------------------------------------------
    # НАПРАВЛЕНИЕ СИГНАЛА
    # ---------------------------------------------------------

    @staticmethod
    def get_signal_direction(signal_type: str) -> Optional[PositionSide]:
        st = signal_type.upper()

        if "DUMP → PUMP" in st or "DUMP -> PUMP" in st:
            return PositionSide.LONG
        if "PUMP → DUMP" in st or "PUMP -> DUMP" in st:
            return PositionSide.SHORT

        if "PUMP" in st and "DUMP" not in st:
            return PositionSide.LONG
        if "DUMP" in st and "PUMP" not in st:
            return PositionSide.SHORT

        if "BULL" in st:
            return PositionSide.LONG
        if "BEAR" in st:
            return PositionSide.SHORT

        return None

    # ---------------------------------------------------------
    # ФИЛЬТРЫ РЕЖИМА
    # ---------------------------------------------------------

    def passes_filters(self, engine_mode_cfg: Dict[str, Any], s: Dict[str, Any]) -> bool:
        rating = s["rating"]
        trend = s["trend_score"]
        risk_score = s["risk_score"]
        signal_type = s["type"]

        min_rating = engine_mode_cfg.get("min_rating", 40)
        max_risk_score = engine_mode_cfg.get("max_risk_score", 100)
        trend_filter = engine_mode_cfg.get("trend_filter", False)
        risk_filter = engine_mode_cfg.get("risk_filter", False)
        allow_counter_trend = engine_mode_cfg.get("allow_counter_trend", True)
        strict_trend = engine_mode_cfg.get("strict_trend", False)

        if rating < min_rating:
            return False

        if risk_filter and risk_score > max_risk_score:
            return False

        if trend_filter:
            side = self.get_signal_direction(signal_type)
            if side is not None:
                if side == PositionSide.LONG:
                    if strict_trend:
                        if trend < 60:
                            return False
                    else:
                        if trend < 50 and not allow_counter_trend:
                            return False
                elif side == PositionSide.SHORT:
                    if strict_trend:
                        if trend > 40:
                            return False
                    else:
                        if trend > 50 and not allow_counter_trend:
                            return False

        return True

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: лимит экспозиции
    # ---------------------------------------------------------

    async def _check_exposure_limit(self, symbol: str, entry_price: float, new_size: float) -> bool:
        """
        Возвращает True, если можно открыть новую позицию с таким размером,
        False — если суммарная экспозиция превысит лимит.
        """
        if entry_price <= 0 or new_size <= 0:
            return False

        current_exposure = 0.0
        for sym, pos in self.positions.items():
            e = pos.get("entry") or 0.0
            sz = pos.get("size_remaining") or 0.0
            if e > 0 and sz > 0:
                current_exposure += e * sz

        planned_exposure = current_exposure + entry_price * new_size

        try:
            equity = await self.broker.get_equity()
        except Exception as e:
            self._register_rest_error("_check_exposure_limit:get_equity")
            return False

        if not equity or equity <= 0:
            return False

        max_exposure = equity * self.max_exposure_mult

        if planned_exposure > max_exposure:
            self.log_trade(
                f"event=EXPOSURE_LIMIT_BLOCK;symbol={symbol};entry={entry_price:.4f};"
                f"new_size={new_size:.6f};planned_exposure={planned_exposure:.2f};"
                f"max_exposure={max_exposure:.2f};equity={equity:.2f}"
            )
            return False

        return True

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: нормализация размера ордера
    # ---------------------------------------------------------

    async def _normalize_order_size(self, symbol: str, raw_size: float, price: float) -> float:
        """
        Приводит размер ордера к требованиям биржи:
        - min_qty
        - qty_step
        - min_notional
        Если информации нет — возвращает raw_size.
        """
        if raw_size <= 0 or price <= 0:
            return 0.0

        limits = None
        if hasattr(self.broker, "get_symbol_limits"):
            try:
                limits = await self.broker.get_symbol_limits(symbol)
            except Exception as e:
                self._register_rest_error("_normalize_order_size:get_symbol_limits")
                limits = None

        if not limits:
            return raw_size

        min_qty = float(limits.get("min_qty") or 0.0)
        qty_step = float(limits.get("qty_step") or 0.0)
        min_notional = float(limits.get("min_notional") or 0.0)

        size = raw_size

        if min_qty > 0 and size < min_qty:
            size = 0.0

        if qty_step > 0 and size > 0:
            steps = int(size / qty_step)
            size = steps * qty_step

        if min_notional > 0 and size > 0:
            notional = size * price
            if notional < min_notional:
                size = 0.0

        return max(size, 0.0)

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: разбиение размера на TP1/TP2/TP3
    # ---------------------------------------------------------

    async def _split_tp_sizes(self, symbol: str, total_size: float, price: float) -> Tuple[float, float, float]:
        """
        Делит общий размер на 3 части с учётом биржевых ограничений.
        Возвращает (tp1_size, tp2_size, tp3_size).
        """
        if total_size <= 0:
            return 0.0, 0.0, 0.0

        base_part = total_size / 3.0

        tp1 = await self._normalize_order_size(symbol, base_part, price)
        tp2 = await self._normalize_order_size(symbol, base_part, price)
        tp3 = await self._normalize_order_size(symbol, base_part, price)

        if tp1 <= 0 and tp2 <= 0 and tp3 <= 0:
            single = await self._normalize_order_size(symbol, total_size, price)
            return single, 0.0, 0.0

        total_tp = tp1 + tp2 + tp3
        if total_tp > total_size and total_tp > 0:
            overflow = total_tp - total_size
            tp3 = max(tp3 - overflow, 0.0)

        return tp1, tp2, tp3

    # ---------------------------------------------------------
    # ОСНОВНОЙ ВХОД СИГНАЛА
    # ---------------------------------------------------------

    async def on_signal(self, s: dict):
        # heartbeat по сигналам
        self._register_heartbeat("signal")

        # лимит одновременных позиций
        if len(self.positions) >= self.max_positions:
            self.log_trade(
                f"event=MAX_POSITIONS_LIMIT;limit={self.max_positions};symbol={s.get('symbol')}"
            )
            return

        # если kill-switch уже сработал — никаких новых входов
        if self.kill_switch_triggered:
            self.log_trade("event=KILL_SWITCH_BLOCK_SIGNAL")
            return

        # если soft-warning активен — тоже блокируем новые входы
        if self.soft_warning:
            self.log_trade("event=KILL_SWITCH_SOFT_BLOCK_SIGNAL")
            return

        if await self.check_daily_risk():
            return

        if self.daily_loss_hit:
            return

        symbol = s["symbol"]

        # v2.1: КРИТИЧНО - проверка противоположных позиций
        # Определяем сторону сигнала
        signal_type = s.get("type", "")
        if any(x in signal_type for x in ["Dump → Pump", "PUMP", "bullish"]):
            signal_side = PositionSide.LONG
            opposite_side = PositionSide.SHORT
        else:
            signal_side = PositionSide.SHORT
            opposite_side = PositionSide.LONG

        # Проверяем противоположные позиции
        for trade_id, pos in list(self.positions.items()):
            if pos["symbol"] == symbol:
                if pos["side"] == opposite_side:
                    # Закрываем противоположную позицию
                    self.log_trade(
                        f"event=OPPOSITE_POSITION_CLOSE;symbol={symbol};"
                        f"closing={opposite_side.value};opening={signal_side.value}"
                    )
                    await self._close_position_by_id(
                        trade_id,
                        reason=f"opposite_{signal_side.value}_signal"
                    )
                    await asyncio.sleep(1)  # Даём время на закрытие
                elif pos["side"] == signal_side:
                    # Уже есть позиция в ту же сторону
                    self.log_trade(
                        f"event=DUPLICATE_POSITION;symbol={symbol};"
                        f"side={signal_side.value};already_open=true"
                    )
                    return  # Игнорируем дубликат

        signal_price = s["price"]

        # --- 4.5: глобальные risk‑фильтры и confidence thresholds ---
        confidence = float(s.get("confidence", 1.0) or 1.0)
        market_risk = float(s.get("market_risk", 0.0) or 0.0)
        btc_regime = s.get("btc_regime")
        symbol_regime = s.get("symbol_regime")
        vol_cluster = s.get("vol_cluster")

        # жёсткий порог confidence
        if confidence < self.MIN_CONFIDENCE_HARD:
            self.log_trade(
                f"event=CONFIDENCE_SKIP;symbol={symbol};confidence={confidence:.3f};"
                f"min_conf={self.MIN_CONFIDENCE_HARD:.3f}"
            )
            return

        # глобальный риск рынка
        if market_risk > 0.8:
            self.log_trade(
                f"event=MARKET_RISK_SKIP;symbol={symbol};market_risk={market_risk:.3f}"
            )
            return

        # BTC high_vol режим — блокируем новые входы
        if isinstance(btc_regime, str) and btc_regime.lower() == "high_vol":
            self.log_trade(
                f"event=BTC_REGIME_SKIP;symbol={symbol};btc_regime={btc_regime}"
            )
            return

        # хаотичный режим инструмента
        if isinstance(symbol_regime, str) and symbol_regime.lower() == "chaos":
            self.log_trade(
                f"event=SYMBOL_REGIME_SKIP;symbol={symbol};symbol_regime={symbol_regime}"
            )
            return

        # экстремальная волатильность кластера
        if isinstance(vol_cluster, str) and "extreme" in vol_cluster.lower():
            self.log_trade(
                f"event=VOL_CLUSTER_SKIP;symbol={symbol};vol_cluster={vol_cluster}"
            )
            return

        mode_key, _ = get_current_mode()
        engine_mode_cfg = self.ENGINE_MODES.get(mode_key, self.ENGINE_MODES["A"])

        cooldown_sec = engine_mode_cfg.get("cooldown_sec", 0)
        now = time.time()
        last_ts = self.last_signal_time.get(symbol, 0)
        if cooldown_sec > 0 and now - last_ts < cooldown_sec:
            self.log_trade(
                f"event=COOLDOWN_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                f"cooldown={cooldown_sec};since_last={now - last_ts:.1f}"
            )
            return
        self.last_signal_time[symbol] = now

        max_slip_pct = engine_mode_cfg.get("max_entry_slippage_pct", 0.0)
        if max_slip_pct > 0:
            try:
                current_price = await self.broker.get_last_price(symbol)
            except Exception as e:
                self._register_rest_error("on_signal:get_last_price")
                current_price = None

            if current_price:
                slip_pct = abs(current_price - signal_price) / signal_price * 100
                if slip_pct > max_slip_pct:
                    self.log_trade(
                        f"event=PRICE_SLIP_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                        f"signal_price={signal_price:.4f};current_price={current_price:.4f};"
                        f"slip={slip_pct:.2f};max_slip={max_slip_pct:.2f}"
                    )
                    return

        if symbol in self.positions:
            await self.handle_existing_position(symbol, s)
            return

        if not self.passes_filters(engine_mode_cfg, s):
            return

        closes = s.get("meta_closes") or []
        highs = s.get("meta_highs") or []
        lows = s.get("meta_lows") or []

        atr = None
        if closes and highs and lows:
            atr = self.compute_atr(highs, lows, closes, period=14)

        if atr is not None and atr > 0 and signal_price > 0:
            atr_pct = atr / signal_price * 100
            max_atr_pct = engine_mode_cfg.get("max_atr_pct")
            if max_atr_pct and atr_pct > max_atr_pct:
                self.log_trade(
                    f"event=VOLATILITY_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                    f"atr_pct={atr_pct:.2f};max_atr_pct={max_atr_pct:.2f}"
                )
                return

            if engine_mode_cfg.get("enable_candle_filter", False):
                opens = s.get("meta_opens") or []
                if opens and len(opens) > 0 and len(highs) > 0 and len(lows) > 0 and len(closes) > 0:
                    o = opens[0]
                    h = highs[0]
                    l = lows[0]
                    c = closes[0]

                    if c > 0:
                        range_pct = (h - l) / c * 100 if h > l else 0.0
                    else:
                        range_pct = 0.0

                    max_candle_range_pct = engine_mode_cfg.get("max_candle_range_pct", 100.0)

                    if range_pct > max_candle_range_pct:
                        self.log_trade(
                            f"event=CANDLE_RANGE_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                            f"range_pct={range_pct:.2f};max_range={max_candle_range_pct:.2f}"
                        )
                        return

                    side = self.get_signal_direction(s["type"])
                    if side is not None:
                        body = abs(c - o)
                        rng = max(h - l, 1e-8)
                        body_ratio = body / rng

                        if side == PositionSide.LONG and c < o and body_ratio > 0.6:
                            self.log_trade(
                                f"event=CANDLE_DIR_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                                f"side=LONG;body_ratio={body_ratio:.2f}"
                            )
                            return
                        if side == PositionSide.SHORT and c > o and body_ratio > 0.6:
                            self.log_trade(
                                f"event=CANDLE_DIR_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                                f"side=SHORT;body_ratio={body_ratio:.2f}"
                            )
                            return

        max_spread_pct = engine_mode_cfg.get("max_spread_pct")
        if max_spread_pct and max_spread_pct > 0:
            bid = ask = None
            orderbook = None
            try:
                if hasattr(self.broker, "get_orderbook"):
                    orderbook = await self.broker.get_orderbook(symbol)
                elif hasattr(self.broker, "get_spread"):
                    orderbook = await self.broker.get_spread(symbol)
            except Exception as e:
                self._register_rest_error("on_signal:get_orderbook")
                orderbook = None

            if orderbook:
                if isinstance(orderbook, dict):
                    bid = orderbook.get("bid") or orderbook.get("best_bid") or orderbook.get("b")
                    ask = orderbook.get("ask") or orderbook.get("best_ask") or orderbook.get("a")
                elif isinstance(orderbook, (list, tuple)) and len(orderbook) >= 2:
                    bid, ask = orderbook[0], orderbook[1]

            if bid and ask and bid > 0 and ask > 0 and ask > bid:
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid * 100
                if spread_pct > max_spread_pct:
                    self.log_trade(
                        f"event=SPREAD_SKIP;symbol={symbol};mode={engine_mode_cfg.get('name')};"
                        f"bid={bid:.4f};ask={ask:.4f};spread_pct={spread_pct:.3f};"
                        f"max_spread={max_spread_pct:.3f}"
                    )
                    return

        await self.open_position(symbol, s, engine_mode_cfg)

    # ---------------------------------------------------------
    # ОТКРЫТИЕ ПОЗИЦИИ
    # ---------------------------------------------------------

    async def open_position(self, symbol: str, s: dict, engine_mode_cfg: Dict[str, Any]):
        signal_type = s["type"]
        price = s["price"]

        side = self.get_signal_direction(signal_type)
        if side is None:
            return

        closes = s.get("meta_closes") or []
        highs = s.get("meta_highs") or []
        lows = s.get("meta_lows") or []

        if not closes or not highs or not lows:
            return

        atr = self.compute_atr(highs, lows, closes, period=14)
        if atr is None or atr <= 0:
            return

        atr_mult = engine_mode_cfg.get("atr_mult", 2.0)
        tp1_r = engine_mode_cfg.get("tp1_r", 1.0)
        tp2_mult = engine_mode_cfg.get("tp2_mult", 2.0)
        tp3_mult = engine_mode_cfg.get("tp3_mult", 3.0)

        stop_distance = atr * atr_mult

        # --- 4.5: адаптивный размер позиции с учётом сигнала ---
        size = await self.compute_position_size(stop_distance, s)
        if size <= 0:
            return

        # дополнительная мягкая шкала по confidence
        confidence = float(s.get("confidence", 1.0) or 1.0)
        confidence = max(0.0, min(confidence, 1.0))
        if self.MIN_CONFIDENCE_HARD <= confidence < self.MID_CONFIDENCE:
            size *= 0.5

        # нормализация размера под требования биржи
        size = await self._normalize_order_size(symbol, size, price)
        if size <= 0:
            self.log_trade(
                f"event=SIZE_TOO_SMALL;symbol={symbol};price={price:.4f};"
                f"stop_distance={stop_distance:.6f}"
            )
            return

        # проверка лимита экспозиции
        can_open = await self._check_exposure_limit(symbol, price, size)
        if not can_open:
            return

        trade_id = f"{symbol}-{int(time.time())}"

        try:
            order_id = await self.broker.open_market_order(symbol, side, size)
        except Exception as e:
            self._register_rest_error("open_position:open_market_order")
            return

        if not order_id:
            return

        if side == PositionSide.LONG:
            sl_price = price - stop_distance
            tp1_price = price + stop_distance * tp1_r
            tp2_price = price + stop_distance * tp2_mult
            tp3_price = price + stop_distance * tp3_mult
        else:
            sl_price = price + stop_distance
            tp1_price = price - stop_distance * tp1_r
            tp2_price = price - stop_distance * tp2_mult
            tp3_price = price - stop_distance * tp3_mult

        try:
            sl_id = await self.broker.place_stop_loss_market(symbol, side, size, sl_price)
        except Exception as e:
            self._register_rest_error("open_position:place_sl")
            sl_id = None

        tp1_size, tp2_size, tp3_size = await self._split_tp_sizes(symbol, size, price)

        tp1_id = tp2_id = tp3_id = None
        try:
            if tp1_size > 0:
                tp1_id = await self.broker.place_take_profit(symbol, side, tp1_size, tp1_price)
            if tp2_size > 0:
                tp2_id = await self.broker.place_take_profit(symbol, side, tp2_size, tp2_price)
            if tp3_size > 0:
                tp3_id = await self.broker.place_take_profit(symbol, side, tp3_size, tp3_price)
        except Exception as e:
            self._register_rest_error("open_position:place_tp")

        self.positions[symbol] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry": price,
            "size_total": size,
            "size_remaining": size,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "tp3": tp3_price,
            "sl_id": sl_id,
            "tp1_id": tp1_id,
            "tp2_id": tp2_id,
            "tp3_id": tp3_id,
            "tp1_size": tp1_size,
            "tp2_size": tp2_size,
            "tp3_size": tp3_size,
            "mode": engine_mode_cfg,
            "opened_ts": time.time(),
            "status": "OPEN",
            "signal_type": signal_type,
            "atr": atr,
            "stop_distance": stop_distance,
            "tp1_hit": False,
            "tp2_hit": False,
            "tp3_hit": False,
            "trailing_active": False,
            "mfe": 0.0,
            "mae": 0.0,
            "price_history": [price],

            # --- факторы для execution feedback ---
            "trend_score": s.get("trend_score"),
            "btc_regime": s.get("btc_regime"),
            "symbol_regime": s.get("symbol_regime"),
            "volatility_score": s.get("volatility_score"),
            "structure_score": s.get("structure_score"),
            "liquidity_score": s.get("liquidity_score"),
            "pattern_quality": s.get("pattern_quality"),
        }

        self.log_trade(
            f"id={trade_id};event=OPEN;symbol={symbol};side={side.value};entry={price:.4f};"
            f"sl={sl_price:.4f};tp1={tp1_price:.4f};tp2={tp2_price:.4f};tp3={tp3_price:.4f};"
            f"size_total={size:.6f};mode={engine_mode_cfg.get('name', 'Unknown')};atr={atr:.4f};"
            f"confidence={float(s.get('confidence', 1.0) or 1.0):.3f};market_risk={float(s.get('market_risk', 0.0) or 0.0):.3f}"
        )

    # ---------------------------------------------------------
    # СУЩЕСТВУЮЩАЯ ПОЗИЦИЯ
    # ---------------------------------------------------------

    async def handle_existing_position(self, symbol: str, s: dict):
        pos = self.positions.get(symbol)
        if not pos:
            return

        current_side = pos["side"]
        new_side = self.get_signal_direction(s["type"])
        if new_side is None:
            return

        if new_side == current_side:
            return

        await self.close_position(symbol, reason="REVERSE_SIGNAL")

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: поиск swing‑точек
    # ---------------------------------------------------------

    @staticmethod
    def _find_last_swing(price_history: List[float], side: PositionSide, window: int) -> Optional[float]:
        if len(price_history) < window:
            return None

        tail = price_history[-window:]
        n = len(tail)

        for i in range(n - 2, 0, -1):
            p_prev = tail[i - 1]
            p = tail[i]
            p_next = tail[i + 1]

            if side == PositionSide.LONG:
                if p < p_prev and p < p_next:
                    return p
            else:
                if p > p_prev and p > p_next:
                    return p

        return None

    # ---------------------------------------------------------
    # PRICE UPDATE / КОМБИНИРОВАННЫЙ ТРЕЙЛИНГ 4.2
    # ---------------------------------------------------------

    async def on_price_update(self, symbol: str, price: float):
        # heartbeat по ценам (если вызывается из WS/монитора)
        self._register_heartbeat("price")

        pos = self.positions.get(symbol)
        if not pos or pos["status"] != "OPEN":
            return

        side: PositionSide = pos["side"]
        entry = pos["entry"]
        atr = pos.get("atr") or 0.0
        trade_id = pos["trade_id"]
        stop_distance = pos.get("stop_distance", 0.0)

        tp1 = pos["tp1"]
        tp2 = pos["tp2"]
        tp3 = pos["tp3"]

        ph = pos.get("price_history")
        if ph is None:
            ph = []
        ph.append(price)
        if len(ph) > 80:
            ph.pop(0)
        pos["price_history"] = ph

        if side == PositionSide.LONG:
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl_pct = (entry - price) / entry * 100

        pos["mfe"] = max(pos["mfe"], pnl_pct)
        pos["mae"] = min(pos["mae"], pnl_pct)

        r_pct = (stop_distance / entry * 100) if entry > 0 and stop_distance > 0 else 0.0

        # размеры для TP
        tp1_size = pos.get("tp1_size") or (pos["size_total"] / 3.0)
        tp2_size = pos.get("tp2_size") or (pos["size_total"] / 3.0)
        tp3_size = pos.get("tp3_size") or (pos["size_total"] / 3.0)

        # ---------------- Динамический ранний BE (4.2) ----------------
        if not pos["tp1_hit"] and r_pct > 0:
            if pos["mfe"] >= self.PARTIAL_BE_MFE_R * r_pct:
                if side == PositionSide.LONG:
                    target_be = entry + stop_distance * self.PARTIAL_BE_LOCK_R
                    if target_be > pos["sl"]:
                        pos["sl"] = target_be
                        self.log_trade(
                            f"id={trade_id};event=EARLY_BE;symbol={symbol};side={side.value};"
                            f"price={price:.4f};sl={pos['sl']:.4f};mfe={pos['mfe']:.2f}"
                        )
                else:
                    target_be = entry - stop_distance * self.PARTIAL_BE_LOCK_R
                    if pos["sl"] is None or target_be < pos["sl"]:
                        pos["sl"] = target_be
                        self.log_trade(
                            f"id={trade_id};event=EARLY_BE;symbol={symbol};side={side.value};"
                            f"price={price:.4f};sl={pos['sl']:.4f};mfe={pos['mfe']:.2f}"
                        )

        # ---------------- TP‑логика ----------------

        # TP1
        if not pos["tp1_hit"]:
            if (side == PositionSide.LONG and price >= tp1) or (side == PositionSide.SHORT and price <= tp1):
                pos["tp1_hit"] = True
                pos["size_remaining"] = max(pos["size_remaining"] - tp1_size, 0.0)
                pos["trailing_active"] = True

                if side == PositionSide.LONG:
                    be_price = entry * (1 + self.BE_OFFSET_LONG_PCT)
                else:
                    be_price = entry * (1 - self.BE_OFFSET_SHORT_PCT)

                pos["sl"] = be_price

                self.log_trade(
                    f"id={trade_id};event=TP1;symbol={symbol};side={side.value};price={price:.4f};"
                    f"remaining={pos['size_remaining']:.6f};sl={pos['sl']:.4f}"
                )

        # TP2
        if pos["tp1_hit"] and not pos["tp2_hit"]:
            if (side == PositionSide.LONG and price >= tp2) or (side == PositionSide.SHORT and price <= tp2):
                pos["tp2_hit"] = True
                pos["size_remaining"] = max(pos["size_remaining"] - tp2_size, 0.0)

                self.log_trade(
                    f"id={trade_id};event=TP2;symbol={symbol};side={side.value};price={price:.4f};"
                    f"remaining={pos['size_remaining']:.6f}"
                )

        # TP3
        if pos["tp2_hit"] and not pos["tp3_hit"]:
            if (side == PositionSide.LONG and price >= tp3) or (side == PositionSide.SHORT and price <= tp3):
                pos["tp3_hit"] = True
                pos["size_remaining"] = 0.0

                self.log_trade(
                    f"id={trade_id};event=TP3;symbol={symbol};side={side.value};price={price:.4f}"
                )

                await self.close_position(symbol, reason="TP3_LOGICAL")
                return

    # ---------------------------------------------------------
    # ЗАКРЫТИЕ ПОЗИЦИИ
    # ---------------------------------------------------------

    async def close_position(self, symbol: str, reason: str = "MANUAL"):
        pos = self.positions.get(symbol)
        if not pos:
            return

        side: PositionSide = pos["side"]
        size_remaining = pos.get("size_remaining", 0.0)
        trade_id = pos["trade_id"]
        entry = pos["entry"]
        stop_distance = float(pos.get("stop_distance") or 0.0)

        # --- получаем последнюю цену ---
        try:
            last_price = await self.broker.get_last_price(symbol)
        except Exception as e:
            self._register_rest_error("close_position:get_last_price")
            last_price = None

        # --- закрываем остаток позиции ---
        if size_remaining > 0:
            try:
                await self.broker.close_market_order(symbol, side, size_remaining)
            except Exception as e:
                self._register_rest_error("close_position:close_market_order")

        # --- отменяем SL/TP ---
        for key in ("sl_id", "tp1_id", "tp2_id", "tp3_id"):
            oid = pos.get(key)
            if oid:
                try:
                    await self.broker.cancel_order(symbol, oid)
                except Exception as e:
                    self._register_rest_error(f"close_position:cancel_{key}")

        # --- считаем PnL ---
        if last_price is not None and entry > 0:
            if side == PositionSide.LONG:
                pnl_pct = (last_price - entry) / entry * 100
                pnl_r = (last_price - entry) / stop_distance if stop_distance > 0 else 0.0
            else:
                pnl_pct = (entry - last_price) / entry * 100
                pnl_r = (entry - last_price) / stop_distance if stop_distance > 0 else 0.0
        else:
            pnl_pct = 0.0
            pnl_r = 0.0

        # --- определяем, был ли стоп ---
        stopped_out = reason.upper().startswith("SL")

        # --- отправляем feedback в SmartFilters ---
        try:
            smartfilters.register_execution_feedback(
                trend_score=pos.get("trend_score"),
                btc_regime=pos.get("btc_regime"),
                symbol_regime=pos.get("symbol_regime"),
                volatility_score=pos.get("volatility_score"),
                structure_score=pos.get("structure_score"),
                liquidity_score=pos.get("liquidity_score"),
                pattern_quality=pos.get("pattern_quality"),
                pnl_r=float(pnl_r),
                stopped_out=bool(stopped_out),
            )
        except Exception as e:
            self._register_internal_error(f"close_position:feedback_error:{repr(e)}")

        # --- удаляем позицию ---
        self.positions.pop(symbol, None)

        # --- лог ---
        self.log_trade(
            f"id={trade_id};event=CLOSE;symbol={symbol};side={side.value};reason={reason};"
            f"entry={entry:.4f};last_price={last_price if last_price is not None else 0.0:.4f};"
            f"pnl_pct={pnl_pct:.2f};pnl_r={pnl_r:.2f}"
        )

    # ---------------------------------------------------------
    # SYNC + АВТОВОССТАНОВЛЕНИЕ SL/TP
    # ---------------------------------------------------------

    async def sync_with_exchange(self):
        # heartbeat по sync-циклу
        self._register_heartbeat("sync")

        try:
            real_positions = await self.broker.get_open_positions()
        except Exception as e:
            self._register_rest_error("sync:get_open_positions")
            return

        real_by_symbol = {p["symbol"]: p for p in real_positions}

        try:
            all_orders = await self.broker.get_open_orders()
        except Exception as e:
            self._register_rest_error("sync:get_open_orders")
            all_orders = []

        orders_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for o in all_orders:
            sym = o.get("symbol")
            if not sym:
                continue
            orders_by_symbol.setdefault(sym, []).append(o)

        for symbol in list(self.positions.keys()):
            if symbol not in real_by_symbol:
                pos = self.positions.pop(symbol)
                trade_id = pos["trade_id"]
                self.log_trade(
                    f"id={trade_id};event=SYNC_CLOSE;symbol={symbol};reason=NO_EXCHANGE_POSITION"
                )

        for symbol, rp in real_by_symbol.items():
            if symbol not in self.positions:
                side_str = rp.get("side", "").upper()
                side = PositionSide.LONG if side_str == "BUY" else PositionSide.SHORT
                size = float(rp.get("size", 0) or 0)
                entry = float(rp.get("avgPrice", 0) or 0)

                trade_id = f"{symbol}-RESTORE-{int(time.time())}"

                self.positions[symbol] = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "size_total": size,
                    "size_remaining": size,
                    "sl": None,
                    "tp1": None,
                    "tp2": None,
                    "tp3": None,
                    "sl_id": None,
                    "tp1_id": None,
                    "tp2_id": None,
                    "tp3_id": None,
                    "tp1_size": 0.0,
                    "tp2_size": 0.0,
                    "tp3_size": 0.0,
                    "mode": None,
                    "opened_ts": time.time(),
                    "status": "OPEN",
                    "signal_type": "RESTORED",
                    "atr": 0.0,
                    "stop_distance": 0.0,
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                    "trailing_active": False,
                    "mfe": 0.0,
                    "mae": 0.0,
                    "price_history": [entry],
                }

                self.log_trade(
                    f"id={trade_id};event=SYNC_RESTORE;symbol={symbol};side={side.value};"
                    f"entry={entry:.4f};size_total={size:.6f}"
                )

        for symbol, pos in list(self.positions.items()):
            sym_orders = orders_by_symbol.get(symbol, [])
            sl_order, tp_orders = self._classify_orders_for_symbol(symbol, pos, sym_orders)

            if pos.get("sl_id") and (not sl_order or sl_order.get("orderId") != pos["sl_id"]):
                self.log_trade(
                    f"id={pos['trade_id']};event=SYNC_SL_MISSING;symbol={symbol};old_sl_id={pos['sl_id']}"
                )
                pos["sl_id"] = None
                await self._restore_sl(symbol, pos)

            for key, level in (("tp1_id", "tp1"), ("tp2_id", "tp2"), ("tp3_id", "tp3")):
                if pos.get(key) and not any(o.get("orderId") == pos[key] for o in tp_orders):
                    self.log_trade(
                        f"id={pos['trade_id']};event=SYNC_TP_MISSING;symbol={symbol};{key}={pos[key]}"
                    )
                    pos[key] = None
                    await self._restore_tp(symbol, pos, key, level)

    def _classify_orders_for_symbol(
        self,
        symbol: str,
        pos: Dict[str, Any],
        orders: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        side: PositionSide = pos["side"]
        sl_order = None
        tp_orders: List[Dict[str, Any]] = []

        close_side = "Sell" if side == PositionSide.LONG else "Buy"

        for o in orders:
            if o.get("symbol") != symbol:
                continue

            if o.get("side") != close_side:
                continue

            reduce_only = str(o.get("reduceOnly", "false")).lower() == "true"
            if not reduce_only:
                continue

            order_type = o.get("orderType", "")
            trigger_price = o.get("triggerPrice")

            if order_type.upper() == "MARKET" and trigger_price not in (None, "", "0"):
                sl_order = o
                continue

            if order_type.upper() == "LIMIT":
                tp_orders.append(o)

        return sl_order, tp_orders

    async def _restore_sl(self, symbol: str, pos: Dict[str, Any]):
        side: PositionSide = pos["side"]
        trade_id = pos["trade_id"]
        sl_price = pos.get("sl")
        size_remaining = pos.get("size_remaining", 0.0)

        if sl_price is None or size_remaining <= 0:
            return

        try:
            new_sl_id = await self.broker.place_stop_loss_market(
                symbol, side, size_remaining, sl_price
            )
        except Exception as e:
            self._register_rest_error("restore_sl:place_sl")
            new_sl_id = None

        if new_sl_id:
            pos["sl_id"] = new_sl_id
            self.log_trade(
                f"id={trade_id};event=SYNC_SL_RESTORE;symbol={symbol};side={side.value};"
                f"sl={sl_price:.4f};size={size_remaining:.6f}"
            )

    async def _restore_tp(self, symbol: str, pos: Dict[str, Any], key_id: str, level_key: str):
        side: PositionSide = pos["side"]
        trade_id = pos["trade_id"]
        price = pos.get(level_key)

        if price is None:
            return

        # определяем размер для этого уровня
        if level_key == "tp1":
            size = pos.get("tp1_size", 0.0)
            hit_flag = pos.get("tp1_hit", False)
        elif level_key == "tp2":
            size = pos.get("tp2_size", 0.0)
            hit_flag = pos.get("tp2_hit", False)
        else:  # "tp3"
            size = pos.get("tp3_size", 0.0)
            hit_flag = pos.get("tp3_hit", False)

        # если уровень уже логически исполнен или размер нулевой — не восстанавливаем
        if hit_flag or size <= 0:
            return

        # не восстанавливаем TP больше, чем size_remaining
        size = min(size, pos.get("size_remaining", 0.0))
        if size <= 0:
            return

        try:
            new_tp_id = await self.broker.place_take_profit(symbol, side, size, price)
        except Exception as e:
            self._register_rest_error("restore_tp:place_tp")
            new_tp_id = None

        if new_tp_id:
            pos[key_id] = new_tp_id
            self.log_trade(
                f"id={trade_id};event=SYNC_TP_RESTORE;symbol={symbol};side={side.value};"
                f"level={level_key};price={price:.4f};size={size:.6f}"
            )


async def sync_loop(engine: TradingEngine, interval: int = 15):
    while True:
        await engine.sync_with_exchange()
        await asyncio.sleep(interval)


async def kill_switch_loop(
    engine: TradingEngine,
    max_silence_sec: int = 60,
    max_rest_errors: int = 20,
    check_interval: int = 5,
    soft_grace_sec: int = 25,
):
    """
    Watchdog (режим C: мягкий + жёсткий):
    - если давно не было heartbeat от price_monitor_loop → сначала soft-warning, потом аварийное закрытие
    - если слишком много REST-ошибок → сначала soft-warning, потом аварийное закрытие
    """
    while True:
        now = time.time()

        # heartbeat kill-switch’а
        engine._register_heartbeat("kill")

        if not engine.kill_switch_triggered:
            problem = False
            reason = None

            silence = now - engine.last_price_heartbeat
            if silence > max_silence_sec:
                problem = True
                reason = f"NO_PRICE_HEARTBEAT_{int(silence)}s"

            if not problem and engine.rest_error_count >= max_rest_errors:
                problem = True
                reason = f"REST_ERRORS_{engine.rest_error_count}"

            if not problem:
                if engine.soft_warning:
                    engine.soft_warning = False
                    engine.soft_warning_ts = None
                    engine.log_trade("event=KILL_SWITCH_SOFT_CLEAR")
                await asyncio.sleep(check_interval)
                continue

            if problem and not engine.soft_warning:
                engine.soft_warning = True
                engine.soft_warning_ts = now
                engine.log_trade(
                    f"event=KILL_SWITCH_SOFT_WARNING;reason={reason}"
                )
                await asyncio.sleep(check_interval)
                continue

            if engine.soft_warning and engine.soft_warning_ts is not None:
                elapsed = now - engine.soft_warning_ts
                if elapsed >= soft_grace_sec:
                    await engine.emergency_flatten(reason=reason or "SOFT_TIMEOUT")

        await asyncio.sleep(check_interval)
