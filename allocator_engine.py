# allocator_engine.py
# Institutional Allocator Engine v1.1 (stability tuned)

from typing import Dict, Optional


class AllocatorEngine:
    """
    Простой институциональный аллокатор риска для продакшена.

    Цели:
    - держать базовый риск на сделку около 1%
    - ограничивать суммарное "тепло" портфеля
    - динамически снижать риск при просадке
    - мягко масштабировать риск по confidence
    """

    def __init__(self):

        # ====== Config ======
        # базовый риск на сделку
        self.base_risk_pct = 0.01

        # суммарный риск портфеля (консервативнее для более ровной кривой)
        self.max_portfolio_heat = 0.05   # 5% суммарный риск

        # лимит позиций
        self.min_positions = 1
        self.max_positions = 4

        # ====== State ======
        self.equity_peak = 0.0
        self.current_drawdown = 0.0

    # ==========================================================
    # EQUITY TRACKING
    # ==========================================================

    def update_equity(self, equity: float):
        if equity > self.equity_peak:
            self.equity_peak = equity

        if self.equity_peak > 0:
            self.current_drawdown = (self.equity_peak - equity) / self.equity_peak
        else:
            self.current_drawdown = 0.0

    # ==========================================================
    # DRAWDOWN MODIFIER
    # ==========================================================

    def _drawdown_modifier(self) -> float:
        """
        Чем глубже просадка, тем сильнее душим риск.
        """

        dd = self.current_drawdown

        # 20%+ просадка — почти стоп
        if dd >= 0.20:
            return 0.2
        # 15–20% — сильно режем
        elif dd >= 0.15:
            return 0.4
        # 10–15% — заметное снижение
        elif dd >= 0.10:
            return 0.6
        # 5–10% — лёгкое снижение
        elif dd >= 0.05:
            return 0.8
        else:
            return 1.0

    # ==========================================================
    # CONFIDENCE MODIFIER
    # ==========================================================

    def _confidence_modifier(self, confidence: float) -> float:
        """
        Мягкая шкала по уверенности:
        - ниже 0.55 — урезаем риск
        - 0.55–0.7 — базовый риск
        - выше 0.7 — слегка увеличиваем
        """

        if confidence < 0.55:
            return 0.6
        elif confidence < 0.70:
            return 1.0
        else:
            return 1.3

    # ==========================================================
    # PORTFOLIO HEAT CHECK
    # ==========================================================

    def _portfolio_heat_ok(
        self,
        equity: float,
        open_positions: Dict,
        new_trade_risk: float
    ) -> bool:
        """
        Проверка суммарного риска портфеля.

        Важно: TradingEngine сейчас не хранит risk_amount в позициях,
        поэтому считаем текущий heat по количеству позиций как
        консервативную оценку.
        """

        # консервативная оценка текущего тепла:
        # считаем, что каждая открытая позиция несёт базовый риск
        current_heat = len(open_positions) * equity * self.base_risk_pct

        if current_heat + new_trade_risk > equity * self.max_portfolio_heat:
            return False

        return True

    # ==========================================================
    # POSITION COUNT CHECK
    # ==========================================================

    def _position_count_ok(self, open_positions: Dict) -> bool:

        if len(open_positions) >= self.max_positions:
            return False

        return True

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def evaluate_trade(
        self,
        equity: float,
        open_positions: Dict,
        confidence: float
    ) -> Optional[float]:
        """
        Возвращает final risk_pct или None если вход запрещён.
        """

        # 1️⃣ Update drawdown state
        self.update_equity(equity)

        # 2️⃣ Base risk
        risk_pct = self.base_risk_pct

        # 3️⃣ Apply confidence scaling
        risk_pct *= self._confidence_modifier(confidence)

        # 4️⃣ Apply drawdown throttle
        risk_pct *= self._drawdown_modifier()

        # 5️⃣ Calculate absolute risk
        risk_amount = equity * risk_pct

        # 6️⃣ Portfolio heat check
        if not self._portfolio_heat_ok(
            equity,
            open_positions,
            risk_amount
        ):
            return None

        # 7️⃣ Position count check
        if not self._position_count_ok(open_positions):
            return None

        # Жёсткий верхний лимит на риск одной сделки (на всякий случай)
        risk_pct = max(0.0, min(risk_pct, 0.02))

        return risk_pct
