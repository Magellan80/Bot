# footprint.py

from typing import List, Dict, Optional


def compute_footprint_zones(
    trades: List[Dict],
    high: float,
    low: float,
    top_ratio: float = 0.33,
    bottom_ratio: float = 0.33,
):
    """
    Делит свечу на 3 зоны:
        - верхняя треть
        - средняя треть
        - нижняя треть
    И считает дельту в каждой зоне.
    """
    if not trades or high <= low:
        return None

    zone_size = (high - low) / 3
    bottom = low + zone_size
    top = high - zone_size

    zones = {
        "bottom": {"buy": 0.0, "sell": 0.0},
        "middle": {"buy": 0.0, "sell": 0.0},
        "top": {"buy": 0.0, "sell": 0.0},
    }

    for t in trades:
        try:
            price = float(t.get("price") or t.get("p"))
            qty = float(t.get("qty") or t.get("q"))
        except:
            continue

        side_raw = str(t.get("side", "")).lower()
        side = "buy" if "buy" in side_raw else "sell" if "sell" in side_raw else None
        if not side:
            continue

        if price <= bottom:
            zones["bottom"][side] += qty
        elif price >= top:
            zones["top"][side] += qty
        else:
            zones["middle"][side] += qty

    return zones


def compute_footprint_signal(zones: Dict, direction: str):
    """
    Возвращает бонус к рейтингу:
        - бычий сигнал усиливается, если агрессия внизу
        - медвежий — если агрессия вверху
    """
    if not zones:
        return 0

    bottom_delta = zones["bottom"]["buy"] - zones["bottom"]["sell"]
    top_delta = zones["top"]["buy"] - zones["top"]["sell"]

    if direction == "bullish":
        if bottom_delta > 0:
            return 5
        if bottom_delta > top_delta:
            return 3
    else:
        if top_delta < 0:
            return 5
        if top_delta < bottom_delta:
            return 3

    return 0
