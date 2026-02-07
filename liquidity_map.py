# liquidity_map.py
from typing import Any, Dict, List, Tuple, Optional


def _normalize_side(side: List[List[Any]]) -> List[Tuple[float, float]]:
    """
    Преобразуем [["price","size"], ...] -> [(price: float, size: float), ...]
    """
    out: List[Tuple[float, float]] = []
    for level in side:
        if len(level) < 2:
            continue
        try:
            p = float(level[0])
            s = float(level[1])
            out.append((p, s))
        except Exception:
            continue
    return out


def analyze_orderbook_walls(
    orderbook: Dict[str, Any],
    current_price: float,
    min_wall_size: float = 5.0,
    max_distance_pct: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Ищем крупные лимитные стенки (walls) рядом с текущей ценой.

    :param orderbook: результат fetch_orderbook(...)
    :param current_price: текущая цена (last_price)
    :param min_wall_size: минимальный размер стенки
    :param max_distance_pct: максимальное расстояние от цены в %
    :return: (walls_up, walls_down)
    """
    bids_raw = orderbook.get("b", [])
    asks_raw = orderbook.get("a", [])

    bids = _normalize_side(bids_raw)
    asks = _normalize_side(asks_raw)

    walls_up: List[Tuple[float, float]] = []
    walls_down: List[Tuple[float, float]] = []

    if current_price <= 0:
        return walls_up, walls_down

    max_dist = current_price * max_distance_pct / 100.0

    for p, size in asks:
        if size < min_wall_size:
            continue
        if 0 < p - current_price <= max_dist:
            walls_up.append((p, size))

    for p, size in bids:
        if size < min_wall_size:
            continue
        if 0 < current_price - p <= max_dist:
            walls_down.append((p, size))

    return walls_up, walls_down


def compute_liquidity_imbalance(orderbook: Dict[str, Any]) -> float:
    """
    Считаем дисбаланс ликвидности в стакане:
    (bid_vol - ask_vol) / (bid_vol + ask_vol)
    """
    bids = _normalize_side(orderbook.get("b", []))
    asks = _normalize_side(orderbook.get("a", []))

    bid_vol = sum(s for _, s in bids)
    ask_vol = sum(s for _, s in asks)

    denom = bid_vol + ask_vol
    if denom <= 0:
        return 0.0

    return (bid_vol - ask_vol) / denom


def detect_liquidity_vacuum(
    orderbook: Dict[str, Any],
    current_price: float,
    levels_to_check: int = 15,
    thin_threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Ищем "пустоты" (vacuum) в ближайших уровнях стакана:
    считаем количество очень тонких уровней (size < thin_threshold)
    сверху и снизу.

    :return: (vacuum_up_count, vacuum_down_count)
    """
    bids = _normalize_side(orderbook.get("b", []))
    asks = _normalize_side(orderbook.get("a", []))

    vacuum_up = 0
    vacuum_down = 0

    for p, size in asks[:levels_to_check]:
        if size < thin_threshold:
            vacuum_up += 1

    for p, size in bids[:levels_to_check]:
        if size < thin_threshold:
            vacuum_down += 1

    return vacuum_up, vacuum_down


def find_strongest_liquidity_zone(
    walls_up: List[Tuple[float, float]],
    walls_down: List[Tuple[float, float]],
) -> Optional[Tuple[str, Tuple[float, float]]]:
    """
    Находим самую сильную стенку среди верхних и нижних.
    :return: ("up"|"down", (price, size)) или None
    """
    strongest: Optional[Tuple[str, Tuple[float, float]]] = None

    if walls_up:
        up_max = max(walls_up, key=lambda x: x[1])
        strongest = ("up", up_max)

    if walls_down:
        down_max = max(walls_down, key=lambda x: x[1])
        if strongest is None or down_max[1] > strongest[1][1]:
            strongest = ("down", down_max)

    return strongest


def build_liquidity_map(
    orderbook: Dict[str, Any],
    current_price: float,
) -> Dict[str, Any]:
    """
    Строим карту ликвидности вокруг текущей цены.

    Возвращаем:
    {
        "walls_up": [(price, size), ...],
        "walls_down": [(price, size), ...],
        "imbalance": float,
        "vacuum_up": int,
        "vacuum_down": int,
        "strongest_zone": ("up"/"down", (price, size)) | None,
        "bias": "bullish"/"bearish"/"neutral"
    }
    """
    walls_up, walls_down = analyze_orderbook_walls(orderbook, current_price)
    imbalance = compute_liquidity_imbalance(orderbook)
    vacuum_up, vacuum_down = detect_liquidity_vacuum(orderbook, current_price)
    strongest_zone = find_strongest_liquidity_zone(walls_up, walls_down)

    if imbalance > 0.2:
        bias = "bullish"
    elif imbalance < -0.2:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "walls_up": walls_up,
        "walls_down": walls_down,
        "imbalance": imbalance,
        "vacuum_up": vacuum_up,
        "vacuum_down": vacuum_down,
        "strongest_zone": strongest_zone,
        "bias": bias,
    }
