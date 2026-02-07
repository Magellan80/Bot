# reversal_detector.py

class ReversalDetector:
    """
    ReversalDetector v4.2
    - Разворот только после тренда
    - Усиленные фильтры ATR / Volume / BOS
    - Компактные дивергенции (RSI, MACD, OBV)
    - Умное финальное голосование
    - Нормальный порядок индикаторов (последнее значение = [-1])
    - Habr-стратегия (Bollinger + RSI + EMA + ATR) как отдельный модуль
    """

    def __init__(self):
        pass

    # ============================
    #   ПУБЛИЧНЫЙ ИНТЕРФЕЙС
    # ============================

    def analyze(self, closes, highs, lows, volumes, ob_imbalance=None):
        if not (len(closes) == len(highs) == len(lows) == len(volumes)):
            return {"reversal": None, "rating": 0, "filters": ["Length mismatch"]}

        if len(closes) < 60:
            return {"reversal": None, "rating": 0, "filters": ["Not enough data"]}

        # Индикаторы
        rsi = self._rsi(closes)
        macd_hist = self._macd_hist(closes)
        obv = self._obv(closes, volumes)
        atr = self._atr(highs, lows, closes)

        if min(len(rsi), len(macd_hist), len(obv), len(atr)) < 20:
            return {"reversal": None, "rating": 0, "filters": ["Not enough indicator data"]}

        # Свинги
        swings = self._swings(highs, lows)
        if len(swings) < 4:
            return {"reversal": None, "rating": 0, "filters": ["No swings"]}

        last_swings = swings[-6:]

        # Проверка тренда
        trend_ok, trend_dir = self._trend(closes, atr, last_swings)
        if not trend_ok:
            return {"reversal": None, "rating": 0, "filters": ["No trend → no reversal"]}

        filters = [f"Trend: {trend_dir}"]
        score = 0

        # Дивергенции
        rsi_div, rsi_dir = self._divergence(rsi, closes, highs, lows, last_swings)
        macd_div, macd_dir = self._divergence(macd_hist, closes, highs, lows, last_swings)
        obv_div, obv_dir = self._divergence(obv, closes, highs, lows, last_swings)

        for div, name, ddir, pts in [
            (rsi_div, "RSI", rsi_dir, 18),
            (macd_div, "MACD", macd_dir, 16),
            (obv_div, "OBV", obv_dir, 14),
        ]:
            if div:
                filters.append(f"{name} {ddir} divergence")
                score += pts

        # BOS / CHOCH
        bos, bos_dir = self._bos(closes, last_swings)
        if bos:
            filters.append(f"BOS/CHOCH {bos_dir}")
            score += 24
        else:
            score -= 8

        # Volume climax
        vol_climax, vol_dir = self._volume_climax(closes, highs, lows, volumes)
        if vol_climax:
            filters.append(f"Volume climax {vol_dir}")
            score += 14

        # Candle confirmation
        candle_conf, candle_dir = self._candle(closes, highs, lows)
        if candle_conf:
            filters.append(f"Candle confirmation {candle_dir}")
            score += 10

        # ATR filter
        atr_ok, vol_regime = self._atr_filter(atr, closes)
        if not atr_ok:
            filters.append("ATR noise filter fail")
            score -= 20
        else:
            filters.append(f"Volatility: {vol_regime}")
            score += 6

        # Orderbook
        if ob_imbalance is not None:
            if ob_imbalance < -15:
                filters.append("Sell pressure")
                score += 10
            elif ob_imbalance > 15:
                filters.append("Buy pressure")
                score += 10

        # HTF trend
        htf_ok, htf_dir = self._htf(closes)
        if htf_ok:
            filters.append(f"HTF trend {htf_dir}")
            score += 10

        # False divergence
        if (rsi_div or macd_div or obv_div) and not bos:
            filters.append("Potential false divergence")
            score -= 12

        # Confluence
        if bos and candle_conf and bos_dir == candle_dir:
            filters.append("BOS + Candle confluence")
            score += 6

        score = max(0, min(score, 100))
        if score < 30:
            return {"reversal": None, "rating": score, "filters": filters}

        # Финальное направление
        direction = self._final(
            rsi_dir, macd_dir, obv_dir, bos_dir, vol_dir, candle_dir, htf_dir, rsi, macd_hist
        )

        return {"reversal": direction, "rating": score, "filters": filters}

    # ============================
    #   Habr-стратегия (Bollinger + RSI + EMA + ATR)
    # ============================

    def analyze_habr(self, closes, highs, lows, volumes):
        """
        Habr++: Bollinger + RSI + EMA + ATR
        Возвращает:
            None  — если условий нет
            {"direction": "bullish"/"bearish", "rating": int}
        """
        if len(closes) < 80:
            return None

        # Индикаторы
        bb_upper, bb_mid, bb_lower = self._bbands(closes, period=20, mult=2.0)
        rsi = self._rsi(closes)
        ema_fast = self._ema(closes, 20)
        ema_slow = self._ema(closes, 50)
        atr = self._atr(highs, lows, closes, period=14)

        if not (bb_upper and bb_lower and rsi and ema_fast and ema_slow and atr):
            return None

        c = closes[-1]
        upper = bb_upper[-1]
        lower = bb_lower[-1]
        r = rsi[-1]
        ema_f = ema_fast[-1]
        ema_s = ema_slow[-1]
        atr_last = atr[-1]

        # ATR режим
        atr_pct = atr_last / c * 100 if c > 0 else 0
        if atr_pct < 0.2 or atr_pct > 5.0:
            return None  # слишком мёртвый или слишком бешеный рынок

        direction = None
        base_score = 0

        # Bullish mean-reversion: цена ниже нижней BB, RSI перепродан, EMA-фильтр
        if c < lower and r < 35:
            # тренд не должен быть ярко медвежьим: цена не сильно ниже EMA50
            if c > ema_s * 0.97:
                direction = "bullish"
                # чем глубже под нижней полосой и чем ниже RSI — тем выше рейтинг
                dist_band = (lower - c) / lower * 100 if lower > 0 else 0
                rsi_score = max(0, 40 - r)
                base_score = dist_band * 2 + rsi_score * 1.5

        # Bearish mean-reversion: цена выше верхней BB, RSI перекуплен, EMA-фильтр
        if c > upper and r > 65:
            if c < ema_s * 1.03:
                direction = "bearish"
                dist_band = (c - upper) / upper * 100 if upper > 0 else 0
                rsi_score = max(0, r - 60)
                base_score = dist_band * 2 + rsi_score * 1.5

        if direction is None:
            return None

        # Дополнительные фильтры по EMA-наклону (простая оценка тренда)
        ema_trend = ema_fast[-1] - ema_fast[-5] if len(ema_fast) >= 5 else 0
        if direction == "bullish" and ema_trend < 0:
            # контртренд — чуть режем рейтинг
            base_score *= 0.8
        if direction == "bearish" and ema_trend > 0:
            base_score *= 0.8

        rating = int(max(0, min(base_score, 100)))
        if rating < 30:
            return None

        return {"direction": direction, "rating": rating}

    # ============================
    #   TREND CHECK
    # ============================

    def _trend(self, closes, atr, swings):
        if len(closes) < 30 or len(atr) < 20:
            return False, None

        # Движение за последние 20 свечей
        move = abs(closes[-1] - closes[-21])
        if move < atr[-1] * 3:
            return False, None

        last, prev, prev2 = swings[-1], swings[-2], swings[-3]

        if prev2["type"] == "low" and prev["type"] == "high" and last["type"] == "low":
            return True, "bullish_trend"

        if prev2["type"] == "high" and prev["type"] == "low" and last["type"] == "high":
            return True, "bearish_trend"

        return False, None

    # ============================
    #   INDICATORS (normal order)
    # ============================

    def _rsi(self, closes, period=14):
        if len(closes) < period + 2:
            return []
        gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
        losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period or 1e-7
        rsi = [100 - 100 / (1 + avg_gain / avg_loss)]
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period or 1e-7
            rsi.append(100 - 100 / (1 + avg_gain / avg_loss))
        return rsi

    def _ema(self, data, period):
        if len(data) < period:
            return []
        k = 2 / (period + 1)
        ema = sum(data[:period]) / period
        out = [ema]
        for x in data[period:]:
            ema = x * k + ema * (1 - k)
            out.append(ema)
        return out

    def _macd_hist(self, closes):
        if len(closes) < 40:
            return []
        fast = self._ema(closes, 12)
        slow = self._ema(closes, 26)
        if not fast or not slow:
            return []
        macd = [f - s for f, s in zip(fast[-len(slow):], slow)]
        signal = self._ema(macd, 9)
        if not signal:
            return []
        hist = [m - s for m, s in zip(macd[-len(signal):], signal)]
        return hist

    def _obv(self, closes, volumes):
        if len(closes) != len(volumes):
            return []
        obv = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        return obv

    def _atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 2:
            return []
        trs = []
        for i in range(1, len(closes)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            ))
        atr = sum(trs[:period]) / period
        out = [atr]
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
            out.append(atr)
        return out

    def _bbands(self, closes, period=20, mult=2.0):
        if len(closes) < period:
            return [], [], []
        ma = []
        stds = []
        for i in range(period, len(closes) + 1):
            window = closes[i - period:i]
            m = sum(window) / period
            ma.append(m)
            var = sum((x - m) ** 2 for x in window) / period
            stds.append(var ** 0.5)
        upper = [m + mult * s for m, s in zip(ma, stds)]
        lower = [m - mult * s for m, s in zip(ma, stds)]
        return upper, ma, lower

    # ============================
    #   SWINGS
    # ============================

    def _swings(self, highs, lows, lookback=3):
        out = []
        for i in range(lookback, len(highs) - lookback):
            if all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, lookback + 1)):
                out.append({"type": "high", "index": i})
            elif all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, lookback + 1)):
                out.append({"type": "low", "index": i})
        return out

    # ============================
    #   DIVERGENCE
    # ============================

    def _divergence(self, indicator, closes, highs, lows, swings):
        if len(swings) < 2 or len(indicator) < 10:
            return False, None

        last, prev = swings[-1], swings[-2]
        p1, p2 = prev["index"], last["index"]

        if p2 >= len(indicator) or p1 >= len(indicator):
            return False, None

        # Bullish: цена делает lower low, индикатор higher low
        if last["type"] == "low" and prev["type"] == "low":
            if lows[p2] < lows[p1] and indicator[p2] > indicator[p1]:
                return True, "bullish"

        # Bearish: цена делает higher high, индикатор lower high
        if last["type"] == "high" and prev["type"] == "high":
            if highs[p2] > highs[p1] and indicator[p2] < indicator[p1]:
                return True, "bearish"

        return False, None

    # ============================
    #   BOS / CHOCH
    # ============================

    def _bos(self, closes, swings):
        last = swings[-1]
        idx = last["index"]
        level = closes[idx]
        last_close = closes[-1]

        buffer = abs(level) * 0.001

        if last["type"] == "high":
            if last_close > level + buffer:
                return True, "bullish"
        if last["type"] == "low":
            if last_close < level - buffer:
                return True, "bearish"

        return False, None

    # ============================
    #   VOLUME CLIMAX
    # ============================

    def _volume_climax(self, closes, highs, lows, volumes, lookback=20):
        if len(volumes) < lookback + 5:
            return False, None

        avg = sum(volumes[-lookback - 1:-1]) / lookback
        if avg == 0:
            return False, None

        spike = volumes[-1] / avg
        body = abs(closes[-1] - closes[-2])
        rng = highs[-1] - lows[-1] or 1e-7

        if spike > 3 and body / rng > 0.55:
            return True, "bullish" if closes[-1] > closes[-2] else "bearish"

        return False, None

    # ============================
    #   CANDLE CONFIRMATION
    # ============================

    def _candle(self, closes, highs, lows):
        if len(closes) < 4:
            return False, None

        c0, c1, c2 = closes[-1], closes[-2], closes[-3]
        h0, h1, h2 = highs[-1], highs[-2], highs[-3]
        l0, l1, l2 = lows[-1], lows[-2], lows[-3]

        # Bullish: длинный нижний хвост, закрытие выше предыдущего
        if l0 < min(l1, l2) and (c0 - l0) > (h0 - c0) * 1.3 and c0 > c1:
            return True, "bullish"

        # Bearish: длинный верхний хвост, закрытие ниже предыдущего
        if h0 > max(h1, h2) and (h0 - c0) > (c0 - l0) * 1.3 and c0 < c1:
            return True, "bearish"

        return False, None

    # ============================
    #   ATR FILTER
    # ============================

    def _atr_filter(self, atr, closes):
        atr_pct = atr[-1] / closes[-1] * 100
        if atr_pct < 0.4:
            return False, "ultra_low"
        if atr_pct < 0.9:
            return True, "low"
        if atr_pct < 2.0:
            return True, "normal"
        if atr_pct < 4.0:
            return True, "high"
        return True, "extreme"

    # ============================
    #   HTF TREND
    # ============================

    def _htf(self, closes, period=40):
        if len(closes) < period + 5:
            return False, None
        sma = sum(closes[-period - 1:-1]) / period
        price = closes[-1]
        if price > sma * 1.015:
            return True, "bullish"
        if price < sma * 0.985:
            return True, "bearish"
        return True, "flat"

    # ============================
    #   FINAL DIRECTION
    # ============================

    def _final(self, rsi_dir, macd_dir, obv_dir, bos_dir, vol_dir, candle_dir, htf_dir, rsi, macd_hist):
        votes = {"bullish": 0, "bearish": 0}

        # Базовые голоса
        for d in [rsi_dir, macd_dir, obv_dir, bos_dir, vol_dir, candle_dir, htf_dir]:
            if d in votes:
                votes[d] += 1

        # RSI зоны
        if rsi:
            if rsi[-1] < 45:
                votes["bullish"] += 1
            if rsi[-1] > 55:
                votes["bearish"] += 1

        # MACD zero-cross
        if len(macd_hist) > 3:
            if macd_hist[-1] > 0 and macd_hist[-2] < 0:
                votes["bullish"] += 1
            if macd_hist[-1] < 0 and macd_hist[-2] > 0:
                votes["bearish"] += 1

        # Усиление за согласованность HTF
        if htf_dir == "bullish":
            votes["bullish"] += 1
        elif htf_dir == "bearish":
            votes["bearish"] += 1

        # Жёсткий перевес
        if votes["bullish"] >= votes["bearish"] + 2:
            return "bullish"
        if votes["bearish"] >= votes["bullish"] + 2:
            return "bearish"

        return None

