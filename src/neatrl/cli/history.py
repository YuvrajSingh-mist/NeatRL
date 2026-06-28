"""Rolling history tracker for terminal chart metrics."""

from collections import deque
from typing import Optional


class HistoryTracker:
    """Tracks rolling history for multiple named metrics.

    Each metric gets a deque with a fixed maxlen, so old values
    are automatically discarded as new ones arrive.
    """

    def __init__(self, maxlen: int = 100) -> None:
        self._maxlen = maxlen
        self._series: dict[str, deque[float]] = {}

    def push(self, name: str, value: float) -> None:
        """Record a new value for a named metric."""
        if name not in self._series:
            self._series[name] = deque(maxlen=self._maxlen)
        self._series[name].append(value)

    def push_many(self, **kwargs: float) -> None:
        """Record multiple metric values at once."""
        for k, v in kwargs.items():
            self.push(k, v)

    def get(self, name: str) -> list[float]:
        """Return the history for a metric as a list (oldest-first)."""
        return list(self._series.get(name, []))

    def latest(self, name: str) -> Optional[float]:
        """Return the most recent value, or None."""
        s = self._series.get(name)
        return s[-1] if s else None

    def has(self, name: str) -> bool:
        """Check if a metric has been recorded."""
        return name in self._series and len(self._series[name]) > 0

    def clear(self) -> None:
        self._series.clear()

    def all_metrics(self) -> list[str]:
        """Return sorted list of tracked metric names."""
        return sorted(self._series.keys())
