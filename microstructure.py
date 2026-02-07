# microstructure.py

from typing import List, Dict, Optional, Literal


TradeSide = Literal["buy", "sell", ""]


def build_price_buckets(
    trades: List[Dict],
    bucket_size_ratio: float = 0.0005,
) -> List[Dict]:
    """
    Строит кластеры (ценовые корзины) по последним трейдам.

    trades: список сделок вида:
        {
            "price": "12345.6",
            "qty": "0.123",
            "side": "Buy" / "Sell" / ...
        }

    bucket_size_ratio:
        относительный размер корзины от средней цены.
        0.0005 ≈ 0.05% от цены.
    """
    if not trades:
        return []

    prices = []
    qtys = []
    sides: List[TradeSide] = []

    for t in trades:
        try:
            p = float(t.get("price") or t.get("p") or 0.0)
            q = float(t.get("qty") or t.get("q") or 0.0)
        except (TypeError, ValueError):
            continue

        side_raw = str(t.get("side", "")).lower()
        if "buy" in side_raw:
            s: TradeSide = "buy"
        elif "sell" in side_raw:
            s = "sell"
        else:
            s = ""

        prices.append(p)
        qtys.append(q)
        sides.append(s)

    if not prices or not qtys:
        return []

    mid = sum(prices) / len(prices)
    if mid <= 0:
        return []

    bucket_size = mid * bucket_size_ratio
    if bucket_size <= 0:
        return []

    buckets: Dict[int, Dict[str, float]] = {}

    for p, q, s in zip(prices, qtys, sides):
        idx = int((p - mid) / bucket_size)
        if idx not in buckets:
            buckets[idx] = {
                "price_sum": 0.0,
                "vol": 0.0,
                "buy_vol": 0.0,
                "sell_vol": 0.0,
                "count": 0.0,
            }
        b = buckets[idx]
        b["price_sum"] += p
        b["vol"] += q
        if s == "buy":
            b["buy_vol"] += q
        elif s == "sell":
            b["sell_vol"] += q
        b["count"] += 1.0

    clusters: List[Dict] = []
    for idx, b in buckets.items():
        if b["count"] <= 0:
            continue
        avg_price = b["price_sum"] / b["count"]
        delta = b["buy_vol"] - b["sell_vol"]
        clusters.append(
            {
                "idx": idx,
                "price": avg_price,
                "vol": b["vol"],
                "delta": delta,
                "buy_vol": b["buy_vol"],
                "sell_vol": b["sell_vol"],
            }
        )

    clusters.sort(key=lambda x: x["price"])
    return clusters


def analyze_microstructure(
    clusters: List[Dict],
    last_price: float,
    around_threshold_pct: float = 0.05,
) -> Optional[Dict]:
    """
    Анализирует кластеры и возвращает:
        - poc_price: цена максимального объёма (POC)
        - poc_delta_sign: 'bullish' / 'bearish' / 'neutral'
        - zone_position: 'below_price' / 'above_price' / 'around_price'
    """
    if not clusters or last_price <= 0:
        return None

    poc = max(clusters, key=lambda x: x["vol"])
    poc_price = float(poc["price"])
    delta = float(poc.get("delta", 0.0))

    if delta > 0:
        poc_delta_sign = "bullish"
    elif delta < 0:
        poc_delta_sign = "bearish"
    else:
        poc_delta_sign = "neutral"

    rel = (poc_price - last_price) / last_price * 100.0
    if abs(rel) <= around_threshold_pct:
        zone_position = "around_price"
    elif rel > 0:
        zone_position = "above_price"
    else:
        zone_position = "below_price"

    return {
        "poc_price": poc_price,
        "poc_delta_sign": poc_delta_sign,
        "zone_position": zone_position,
        "poc_vol": float(poc.get("vol", 0.0)),
    }


def compute_micro_bonus(
    direction: Literal["bullish", "bearish"],
    micro: Optional[Dict],
    max_bonus: int = 5,
) -> int:
    """
    Возвращает бонус к рейтингу на основе микроструктуры:
    - бычий сигнал усиливается, если POC с бычьей дельтой под/рядом с ценой
    - медвежий — если POC с медвежьей дельтой над/рядом с ценой
    """
    if not micro:
        return 0

    poc_delta_sign = micro.get("poc_delta_sign")
    zone_position = micro.get("zone_position")

    bonus = 0

    if direction == "bullish":
        if poc_delta_sign == "bullish" and zone_position in ("below_price", "around_price"):
            bonus = max_bonus
        elif poc_delta_sign == "bullish":
            bonus = max_bonus // 2
    else:
        if poc_delta_sign == "bearish" and zone_position in ("above_price", "around_price"):
            bonus = max_bonus
        elif poc_delta_sign == "bearish":
            bonus = max_bonus // 2

    return int(bonus)
