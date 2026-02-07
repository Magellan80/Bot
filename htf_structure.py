from typing import List, Dict, Optional, Tuple


def _smooth_series(values: List[float], window: int = 3) -> List[float]:
    if len(values) <= window:
        return values[:]
    res = []
    k = window // 2
    for i in range(len(values)):
        left = max(0, i - k)
        right = min(len(values), i + k + 1)
        chunk = values[left:right]
        res.append(sum(chunk) / len(chunk))
    return res


def detect_swings(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    min_distance: int = 3,
    min_swing_pct: float = 0.2,
) -> Optional[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]:
    n = len(closes)
    if n < 15:
        return None

    s_highs = _smooth_series(highs)
    s_lows = _smooth_series(lows)

    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(2, n - 2):
        if s_highs[i] > s_highs[i - 1] and s_highs[i] > s_highs[i + 1]:
            swing_highs.append((i, highs[i]))
        if s_lows[i] < s_lows[i - 1] and s_lows[i] < s_lows[i + 1]:
            swing_lows.append((i, lows[i]))

    if not swing_highs or not swing_lows:
        return None

    def _filter_swings(swings: List[Tuple[int, float]], is_high: bool) -> List[Tuple[int, float]]:
        if len(swings) < 2:
            return swings

        filtered: List[Tuple[int, float]] = []
        last_idx = None
        last_price = None

        for idx, price in swings:
            if last_idx is None:
                filtered.append((idx, price))
                last_idx = idx
                last_price = price
                continue

            if abs(idx - last_idx) < min_distance:
                if is_high:
                    if price > last_price:
                        filtered[-1] = (idx, price)
                        last_idx = idx
                        last_price = price
                else:
                    if price < last_price:
                        filtered[-1] = (idx, price)
                        last_idx = idx
                        last_price = price
                continue

            if last_price is not None and last_price > 0:
                diff_pct = abs(price - last_price) / last_price * 100
                if diff_pct < min_swing_pct:
                    continue

            filtered.append((idx, price))
            last_idx = idx
            last_price = price

        return filtered

    swing_highs = _filter_swings(swing_highs, is_high=True)
    swing_lows = _filter_swings(swing_lows, is_high=False)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    return swing_highs, swing_lows


def detect_bos_choch(
    closes: List[float],
    highs: List[float],
    lows: List[float],
) -> Dict[str, Optional[str]]:
    swings = detect_swings(closes, highs, lows)
    if not swings:
        return {"structure": "ranging", "event": None}

    swing_highs, swing_lows = swings

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "ranging", "event": None}

    last_high_idx, last_high = swing_highs[-1]
    prev_high_idx, prev_high = swing_highs[-2]

    last_low_idx, last_low = swing_lows[-1]
    prev_low_idx, prev_low = swing_lows[-2]

    def _rel_diff(a: float, b: float) -> float:
        if b == 0:
            return 0.0
        return (a - b) / b * 100

    high_diff = _rel_diff(last_high, prev_high)
    low_diff = _rel_diff(last_low, prev_low)

    bos_threshold = 0.3
    choch_threshold = 0.2

    if high_diff > bos_threshold and last_high_idx > prev_high_idx:
        return {"structure": "bullish", "event": "BOS"}

    if low_diff < -bos_threshold and last_low_idx > prev_low_idx:
        return {"structure": "bearish", "event": "BOS"}

    if low_diff > choch_threshold:
        return {"structure": "bullish", "event": "CHOCH"}

    if high_diff < -choch_threshold:
        return {"structure": "bearish", "event": "CHOCH"}

    return {"structure": "ranging", "event": None}


def compute_structure_strength(
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    structure: str,
    event: Optional[str],
) -> int:
    if structure == "ranging":
        return 1

    score = 1

    if event == "BOS":
        score += 2
    elif event == "CHOCH":
        score += 1

    if len(swing_highs) >= 2:
        d = abs(swing_highs[-1][0] - swing_highs[-2][0])
        if d > 8:
            score += 1

    if len(swing_lows) >= 2:
        d = abs(swing_lows[-1][0] - swing_lows[-2][0])
        if d > 8:
            score += 1

    try:
        if structure == "bullish" and len(swing_highs) >= 2:
            diff = (swing_highs[-1][1] - swing_highs[-2][1]) / swing_highs[-2][1] * 100
            if diff > 0.5:
                score += 1
        elif structure == "bearish" and len(swing_lows) >= 2:
            diff = (swing_lows[-2][1] - swing_lows[-1][1]) / swing_lows[-2][1] * 100
            if diff > 0.5:
                score += 1
    except:
        pass

    return max(1, min(score, 5))


def compute_htf_structure(klines: List[List[str]]) -> Dict[str, Optional[str]]:
    if not klines or len(klines) < 30:
        return {"structure": "ranging", "event": None, "strength": 1}

    closes = [float(c[4]) for c in klines][::-1]
    highs = [float(c[2]) for c in klines][::-1]
    lows = [float(c[3]) for c in klines][::-1]

    res = detect_bos_choch(closes, highs, lows)
    structure = res["structure"]
    event = res["event"]

    swings = detect_swings(closes, highs, lows)
    if swings:
        swing_highs, swing_lows = swings
        strength = compute_structure_strength(swing_highs, swing_lows, structure, event)
    else:
        strength = 1

    return {
        "structure": structure,
        "event": event,
        "strength": strength,
    }
