# risk_engine_v31.py
# V31 — Institutional Adaptive Risk Allocator (Production Grade, stability tuned)

from collections import deque


class RiskEngineV31:
    """
    Продвинутый адаптивный риск‑двигатель для бэктестера/институциональной логики.

    Цели:
    - адаптация риска к просадке
    - учёт performance (PF по последним сделкам)
    - учёт рыночного режима
    - контроль портфельного тепла и плеча
    """

    def __init__(
        self,
        base_risk=0.008,               # 0.8% per trade
        max_portfolio_heat=0.20,       # 20% total risk exposure (чуть консервативнее)
        max_positions=4,
        max_leverage=5,                # hard leverage cap
        min_stop_pct=0.0005,           # 0.05% minimum stop distance
        atr_stop_floor_mult=0.5,       # stop must be >= ATR * mult
        dd_soft=0.08,
        dd_hard=0.12,
        feedback_window=30
    ):

        # Core parameters
        self.base_risk = base_risk
        self.max_portfolio_heat = max_portfolio_heat
        self.max_positions = max_positions
        self.max_leverage = max_leverage

        self.min_stop_pct = min_stop_pct
        self.atr_stop_floor_mult = atr_stop_floor_mult

        self.dd_soft = dd_soft
        self.dd_hard = dd_hard

        # Performance tracking
        self.trades_R = deque(maxlen=feedback_window)

        # Portfolio state
        self.current_heat = 0.0
        self.open_positions = 0

        self.peak_equity = None

    # ==========================================================
    # EQUITY CONTROL
    # ==========================================================

    def update_equity(self, equity):
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def current_dd(self, equity):
        if not self.peak_equity:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    # ==========================================================
    # PERFORMANCE FEEDBACK
    # ==========================================================

    def register_trade_R(self, result_R):
        self.trades_R.append(result_R)

    def performance_multiplier(self):
        """
        Мягкая адаптация по PF последних сделок.
        """

        if len(self.trades_R) < 10:
            return 1.0

        wins = [r for r in self.trades_R if r > 0]
        losses = [r for r in self.trades_R if r < 0]

        if not losses:
            # серия только вин-результатов — слегка увеличиваем риск
            return 1.10

        pf = abs(sum(wins) / sum(losses))

        if pf < 1.0:
            # плохой PF — заметно снижаем риск
            return 0.7
        elif pf > 1.8:
            # очень хороший PF — чуть увеличиваем
            return 1.10
        else:
            # нормальный PF — без изменений
            return 1.0

    # ==========================================================
    # REGIME ADJUSTMENT
    # ==========================================================

    def regime_multiplier(self, regime):
        """
        Регим‑множитель, допускающий разные форматы строки.
        """

        if not isinstance(regime, str):
            return 1.0

        r = regime.upper()

        if r == "STRONG_TREND":
            return 1.2
        elif r == "EARLY_TREND":
            return 1.1
        elif r == "RANGE":
            return 0.85
        elif r == "CHAOS":
            return 0.0
        else:
            return 1.0

    # ==========================================================
    # FINAL RISK %
    # ==========================================================

    def compute_risk_pct(self, equity, regime):

        self.update_equity(equity)
        dd = self.current_dd(equity)

        # жёсткий стоп по просадке
        if dd >= self.dd_hard:
            return 0.0

        risk = self.base_risk

        # мягкое снижение риска при soft‑DD
        if dd >= self.dd_soft:
            risk *= 0.5

        risk *= self.performance_multiplier()
        risk *= self.regime_multiplier(regime)

        # жёсткий верхний лимит 1.5% на сделку
        return min(max(risk, 0.0), 0.015)

    # ==========================================================
    # POSITION SIZING
    # ==========================================================

    def allocate(
        self,
        equity,
        entry_price,
        stop_price,
        regime,
        atr=None  # optional ATR for volatility sanity
    ):

        risk_pct = self.compute_risk_pct(equity, regime)

        if risk_pct <= 0:
            return 0.0, 0.0

        # лимит по количеству позиций
        if self.open_positions >= self.max_positions:
            return 0.0, 0.0

        # лимит по портфельному теплу
        if self.current_heat + risk_pct > self.max_portfolio_heat:
            return 0.0, 0.0

        risk_per_unit = abs(entry_price - stop_price)

        # ------------------------------------------
        # Micro-stop protection
        # ------------------------------------------
        min_stop_distance = entry_price * self.min_stop_pct

        if risk_per_unit < min_stop_distance:
            return 0.0, 0.0

        # ------------------------------------------
        # ATR sanity floor (volatility based)
        # ------------------------------------------
        if atr is not None:
            atr_floor = atr * self.atr_stop_floor_mult
            if risk_per_unit < atr_floor:
                return 0.0, 0.0

        if risk_per_unit <= 0:
            return 0.0, 0.0

        # ------------------------------------------
        # Risk sizing
        # ------------------------------------------
        risk_amount = equity * risk_pct
        size = risk_amount / risk_per_unit

        # ------------------------------------------
        # Leverage control
        # ------------------------------------------
        position_value = size * entry_price
        max_position_value = equity * self.max_leverage

        if position_value > max_position_value:
            size = max_position_value / entry_price
            position_value = size * entry_price

        # Final sanity check
        if size <= 0:
            return 0.0, 0.0

        # Register exposure
        self.current_heat += risk_pct
        self.current_heat = max(self.current_heat, 0.0)

        self.open_positions += 1
        self.open_positions = max(self.open_positions, 0)

        return size, risk_pct

    # ==========================================================
    # CLOSE POSITION
    # ==========================================================

    def close_position(
        self,
        risk_pct,
        entry,
        stop,
        exit_price,
        direction
    ):
        """
        direction: "long" или "short"
        """

        risk_per_unit = abs(entry - stop)

        if risk_per_unit == 0:
            result_R = 0.0
        else:
            d = (direction or "").lower()
            if d == "long":
                result_R = (exit_price - entry) / risk_per_unit
            else:
                result_R = (entry - exit_price) / risk_per_unit

        self.register_trade_R(result_R)

        # Clean exposure accounting
        self.current_heat -= risk_pct
        self.current_heat = max(self.current_heat, 0.0)

        self.open_positions -= 1
        self.open_positions = max(self.open_positions, 0)
