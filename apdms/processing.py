"""Signal processing utilities: Exponential Moving Average and Sliding Window."""
from __future__ import annotations

import collections
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


class EMA:
    """Exponential Moving Average (EMA) smoothing filter."""

    def __init__(self, alpha: float = 0.15) -> None:
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        """Feed a new measurement and return the updated smoothed value."""
        if self._value is None:
            self._value = float(raw)
        else:
            self._value = self.alpha * float(raw) + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        self._value = None


class ExponentialMovingAverage(EMA):
    """Alias for backwards-compatibility."""
    pass


class SlidingWindow:
    """Time-indexed sliding window buffer for temporal metric analysis.

    Supports both add(timestamp, features) and add(features, timestamp) signatures.
    """

    def __init__(self, window_duration_seconds: float = 3.0, window_seconds: Optional[float] = None) -> None:
        self.window_seconds: float = float(window_seconds if window_seconds is not None else window_duration_seconds)
        self._buf: collections.deque = collections.deque()

    @property
    def buffer(self) -> collections.deque:
        """Expose buffer for test inspections."""
        return self._buf

    def add(
        self,
        arg1: Union[float, Dict[str, Any]],
        arg2: Optional[Union[float, Dict[str, Any]]] = None,
    ) -> None:
        """Add entry to the sliding window and evict expired records.

        Accepts:
            add(timestamp, features_dict) OR add(features_dict, timestamp=None)
        """
        if isinstance(arg1, dict):
            features = arg1
            ts = float(arg2) if arg2 is not None else time.monotonic()
        else:
            ts = float(arg1)
            features = arg2 if isinstance(arg2, dict) else {}

        self._buf.append((ts, features))
        cutoff = ts - self.window_seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        if len(self._buf) < 2:
            return False
        return (self._buf[-1][0] - self._buf[0][0]) >= self.window_seconds

    def get_values(self, key: str) -> List[float]:
        return [float(entry[1][key]) for entry in self._buf if key in entry[1]]

    def get_average(self, key: str) -> float:
        vals = self.get_values(key)
        return float(np.mean(vals)) if vals else 0.0

    def get_std(self, key: str) -> float:
        vals = self.get_values(key)
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def get_perclos(self, ear_threshold: float = 0.20, ear_key: str = "ear") -> float:
        """Calculate PERCLOS (% of frames where EAR < ear_threshold)."""
        ears = self.get_values(ear_key)
        if not ears:
            return 0.0
        closed_count = sum(1 for e in ears if e < ear_threshold)
        return float((closed_count / len(ears)) * 100.0)

    def get_max_duration(self, key: str, condition: Callable[[float], bool]) -> float:
        """Calculate the longest continuous duration (in seconds) where condition holds."""
        if not self._buf:
            return 0.0

        max_dur = 0.0
        streak_start: Optional[float] = None

        for ts, features in self._buf:
            val = features.get(key)
            if val is not None and condition(val):
                if streak_start is None:
                    streak_start = ts
                dur = ts - streak_start
                if dur > max_dur:
                    max_dur = dur
            else:
                streak_start = None

        if streak_start is not None and self._buf:
            dur = self._buf[-1][0] - streak_start
            if dur > max_dur:
                max_dur = dur

        return float(max_dur)

    def get_max_continuous_duration(self, key: str, condition: Callable[[float], bool]) -> float:
        return self.get_max_duration(key, condition)

    def count_events(self, key: str, condition: Callable[[float], bool]) -> int:
        """Count completed transitions where condition is entered and exited."""
        count = 0
        in_event = False
        for _, features in self._buf:
            val = features.get(key)
            if val is None:
                continue
            if condition(val):
                if not in_event:
                    in_event = True
            else:
                if in_event:
                    count += 1
                    in_event = False
        return count

    def count_transitions(self, key: str, condition: Callable[[float], bool]) -> int:
        return self.count_events(key, condition)
