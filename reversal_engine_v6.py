# reversal_engine_v6.py

from detectors.continuation_detector import ContinuationDetector
from detectors.exhaustion_detector import ExhaustionDetector


class ReversalEngineV6:

    def __init__(self):
        self.continuation = ContinuationDetector()
        self.exhaustion = ExhaustionDetector()

    def analyze(self, closes, highs, lows, volumes, rsi, atr, swings, htf_bias=None):

        cont = self.continuation.analyze(
            closes, highs, lows, volumes, atr, swings, htf_bias
        )

        ex = self.exhaustion.analyze(
            closes, highs, lows, volumes, rsi, atr, swings
        )

        # Приоритет continuation
        if cont and ex:
            return cont if cont["rating"] >= ex["rating"] else ex

        if cont:
            return cont

        if ex:
            return ex

        return None
