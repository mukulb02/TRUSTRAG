from collections import deque
import numpy as np


class ReliabilityMemory:
    """
    Sliding window of recent trust scores.
    Used to compute historical_reliability — a smoothed measure
    of how reliable recent pipeline outputs have been.
    """

    def __init__(self, window_size: int = 200):
        self.history = deque(maxlen=window_size)

    def update(self, trust_score: float):
        """Append the latest trust score to the window."""
        self.history.append(float(trust_score))

    def score(self) -> float:
        """
        Returns the mean trust score over the recent window.
        Returns 0.5 (neutral) if no history yet.
        """
        if len(self.history) == 0:
            return 0.5

        return float(np.mean(self.history))

    def size(self) -> int:
        return len(self.history)