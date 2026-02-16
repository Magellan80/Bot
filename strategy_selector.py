class StrategySelector:

    # ==========================================================
    # STRATEGY CHOICE
    # ==========================================================

    def choose(self, btc_regime: str, asset_class: str) -> str:
        """
        Professional regime-based routing
        """

        # Strong trend → continuation priority
        if btc_regime == "trending":
            return "continuation"

        # High volatility → reversal priority
        if btc_regime == "high_vol":
            return "reversal"

        # Ranging / Neutral → reversal bias
        if btc_regime in ("ranging", "neutral"):
            return "reversal"

        return "reversal"

    # ==========================================================
    # MIN SCORE MODEL
    # ==========================================================

    def get_min_score(self, btc_regime: str, asset_class: str) -> int:
        """
        Adaptive baseline threshold
        Compatible with multi-layer scoring
        """

        base = 58

        # BTC regime effect
        if btc_regime == "trending":
            base -= 4
        elif btc_regime == "high_vol":
            base += 4
        elif btc_regime == "ranging":
            base -= 2

        # Asset class effect
        if asset_class == "major":
            base -= 3
        elif asset_class == "lowcap":
            base += 4

        return max(52, min(base, 66))


strategy_selector = StrategySelector()
