# v30_backtest_engine.py
# V31 Institutional Backtest Engine (ATR Integrated + Risk Safe, Clean HTF)

from typing import List, Dict


class V30BacktestEngine:

    def __init__(self):
        self.structure_window = 400
        self.regime_window = 600
        self.atr_window = 200

    def run(
        self,
        candles_5m: List[Dict],
        structure_engine,
        regime_engine,
        htf_sync,
        trend_engine,
        reversal_engine,
        router,
        exit_engine,
        htf_15m,
        htf_1h,
        htf_4h,
        risk_engine,
        initial_balance=10000
    ):

        equity = initial_balance
        peak = equity
        max_dd = 0.0

        trades = []
        position = None

        # Логи для анализа (не тормозят бэктест)
        equity_curve = []
        trade_log = []
        regime_log = []
        signal_log = []
        htf_log = []
        atr_log = []

        total = len(candles_5m)
        history = candles_5m[:250]

        print(f"Total 5m candles: {total}")
        print("Starting Institutional loop...\n")

        for i in range(250, total):

            current_bar = candles_5m[i]
            history.append(current_bar)

            # Лёгкий прогресс, чтобы видеть, что бэктест живой
            if i % 1500 == 0:
                progress = round(i / total * 100, 1)
                print(
                    f"Progress: {progress}% | "
                    f"Trades: {len(trades)} | "
                    f"Equity: {round(equity, 2)}"
                )

            price = float(current_bar["close"])
            ts = current_bar["timestamp"]

            structure = structure_engine.analyze(history[-self.structure_window:])
            regime = regime_engine.detect(history[-self.regime_window:])
            htf = htf_sync.analyze(ts, htf_15m, htf_1h, htf_4h)

            if not structure or not regime or not htf:
                equity_curve.append(equity)
                continue

            # Лог режимов и HTF
            regime_log.append(regime["regime"])
            htf_log.append({
                "alignment": htf["alignment_score"],
                "bias": htf["bias"],
                "signed_strength": htf["signed_trend_strength"],
                "htf_regime": htf["htf_regime"]
            })

            # ATR лог
            atr_val = regime_engine._atr(history[-self.atr_window:], 14)
            atr_log.append(atr_val if atr_val else 0)

            trend_signal = trend_engine.evaluate(structure, regime, htf)
            reversal_signal = reversal_engine.evaluate(structure, regime, htf)

            signal = router.route(trend_signal, reversal_signal, regime, htf)

            # Лог сигналов
            if signal:
                signal_log.append({
                    "bar": i,
                    "type": signal["type"],
                    "direction": signal["signal"],
                    "quality": signal["quality"],
                    "regime": regime["regime"],
                    "htf_regime": htf["htf_regime"]
                })
            else:
                signal_log.append(None)

            # =====================================================
            # OPEN POSITION
            # =====================================================

            if not position and signal:

                sl = self._initial_sl(signal["signal"], history)

                atr = atr_val
                if atr is None or atr <= 0:
                    equity_curve.append(equity)
                    continue

                size, risk_pct = risk_engine.allocate(
                    equity=equity,
                    entry_price=price,
                    stop_price=sl,
                    regime=regime["regime"],
                    atr=atr
                )

                if size == 0:
                    equity_curve.append(equity)
                    continue

                position = {
                    "direction": signal["signal"],
                    "entry": price,
                    "sl": sl,
                    "sl_initial": sl,
                    "size": size,
                    "risk_pct": risk_pct,
                    "status": "OPEN",
                    "partial_taken": False,
                    "open_bar": i
                }

                print(
                    f"OPEN {signal['signal'].upper()} | "
                    f"Price: {round(price,2)} | SL: {round(sl,2)} | "
                    f"Risk: {round(risk_pct*100,2)}%"
                )

            # =====================================================
            # MANAGE POSITION
            # =====================================================

            if position:

                atr = atr_val if atr_val else 0.0

                position = exit_engine.manage_position(
                    position,
                    price,
                    atr,
                    regime["regime"],
                    i
                )

                if position["status"] == "CLOSED":

                    pnl = self._calculate_pnl(position)
                    equity += pnl

                    risk_engine.close_position(
                        risk_pct=position["risk_pct"],
                        entry=position["entry"],
                        stop=position["sl_initial"],
                        exit_price=position["exit_price"],
                        direction=position["direction"]
                    )

                    peak = max(peak, equity)
                    dd = (peak - equity) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)

                    trades.append(pnl)

                    trade_log.append({
                        "entry": position["entry"],
                        "exit": position["exit_price"],
                        "direction": position["direction"],
                        "pnl": pnl,
                        "risk_pct": position["risk_pct"],
                        "sl_initial": position["sl_initial"],
                        "sl_final": position["sl"],
                        "open_bar": position["open_bar"],
                        "close_bar": i,
                        "regime": regime["regime"],
                        "htf_regime": htf["htf_regime"]
                    })

                    print(
                        f"CLOSE | PnL: {round(pnl,2)} | "
                        f"Equity: {round(equity,2)} | "
                        f"DD: {round(max_dd*100,2)}%"
                    )

                    position = None

            equity_curve.append(equity)

        print("\nBacktest Finished.\n")

        stats = self._stats(trades, equity, initial_balance, max_dd)

        return {
            "stats": stats,
            "equity_curve": equity_curve,
            "trades": trade_log,
            "regimes": regime_log,
            "signals": signal_log,
            "htf": htf_log,
            "atr": atr_log
        }

    # =====================================================
    # UTILITIES
    # =====================================================

    def _initial_sl(self, direction, history):

        recent = history[-10:]

        if direction == "long":
            return min(float(c["low"]) for c in recent)
        else:
            return max(float(c["high"]) for c in recent)

    def _calculate_pnl(self, position):

        entry = position["entry"]
        exit_price = position.get("exit_price", entry)
        size = position["size"]
        direction = position["direction"]

        if direction == "long":
            return (exit_price - entry) * size
        else:
            return (entry - exit_price) * size

    def _stats(self, trades, final_equity, initial_balance, max_dd):

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        winrate = len(wins) / len(trades) if trades else 0

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        expectancy = winrate * avg_win + (1 - winrate) * avg_loss

        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        return {
            "trades": len(trades),
            "winrate": round(winrate, 4),
            "expectancy": round(expectancy, 2),
            "profit_factor": round(pf, 2),
            "final_balance": round(final_equity, 2),
            "return_pct": round((final_equity / initial_balance - 1) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2)
        }
