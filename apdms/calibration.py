"""Personalized baseline calibration module."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


class Calibrator:
    """Accumulates facial metrics during calibration and computes driver baselines."""

    def __init__(self, duration_seconds: float = 45.0) -> None:
        self.duration_seconds = float(duration_seconds)
        self._start_time: Optional[float] = None

        self._ears: List[float] = []
        self._mars: List[float] = []
        self._pitches: List[float] = []
        self._rolls: List[float] = []
        self._yaws: List[float] = []

        self._blink_durations: List[float] = []
        self._blink_start: Optional[float] = None
        self._in_blink: bool = False

    def start(self) -> None:
        """Start or restart the calibration timer and clear buffers."""
        self._start_time = time.monotonic()
        self._ears.clear()
        self._mars.clear()
        self._pitches.clear()
        self._rolls.clear()
        self._yaws.clear()
        self._blink_durations.clear()
        self._blink_start = None
        self._in_blink = False

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        roll: float,
        yaw: float,
        face_diagonal: float = 0.0,
    ) -> None:
        """Feed one frame of metrics into the calibration buffer."""
        self._ears.append(float(ear))
        self._mars.append(float(mar))
        self._pitches.append(float(pitch))
        self._rolls.append(float(roll))
        self._yaws.append(float(yaw))

        # Dynamic blink threshold adapting to the user's natural EAR
        if len(self._ears) >= 30:
            m = float(np.mean(self._ears))
            s = float(np.std(self._ears))
            blink_thresh = max(0.10, min(0.25, m - 1.8 * s))
        else:
            blink_thresh = 0.20

        now = time.monotonic()
        if ear < blink_thresh:
            if not self._in_blink:
                self._in_blink = True
                self._blink_start = now
        else:
            if self._in_blink and self._blink_start is not None:
                dur = now - self._blink_start
                self._in_blink = False
                self._blink_start = None
                if 0.05 < dur < 0.60:
                    self._blink_durations.append(dur)

    def is_complete(self) -> bool:
        if self._start_time is None:
            return False
        return (time.monotonic() - self._start_time) >= self.duration_seconds

    def get_progress(self) -> float:
        if self._start_time is None:
            return 0.0
        return min(1.0, (time.monotonic() - self._start_time) / self.duration_seconds)

    def get_baseline_stats(self) -> Dict[str, float]:
        """Compute baseline statistics dict for threshold generation."""
        if len(self._ears) < 10:
            return self._fallback_baseline()

        mean_ear = float(np.mean(self._ears))
        min_ear = float(np.min(self._ears))
        max_ear = float(np.max(self._ears))
        ear_std_dev = float(np.std(self._ears))

        neutral_mar = float(np.median(self._mars))
        neutral_pitch = float(np.median(self._pitches))
        neutral_roll = float(np.median(self._rolls))
        neutral_yaw = float(np.median(self._yaws))

        avg_blink_dur = (
            float(np.mean(self._blink_durations)) if self._blink_durations else 0.20
        )
        blink_rate = (
            len(self._blink_durations) / self.duration_seconds
            if self.duration_seconds > 0
            else 0.20
        )

        return {
            "mean_ear": mean_ear,
            "min_ear": min_ear,
            "max_ear": max_ear,
            "ear_std_dev": ear_std_dev,
            "ear_std": ear_std_dev,
            "avg_blink_duration": avg_blink_dur,
            "natural_blink_rate": blink_rate,
            "neutral_mar": neutral_mar,
            "neutral_pitch": neutral_pitch,
            "neutral_roll": neutral_roll,
            "neutral_yaw": neutral_yaw,
        }

    def get_baseline(self) -> Dict[str, float]:
        return self.get_baseline_stats()

    @staticmethod
    def _fallback_baseline() -> Dict[str, float]:
        return {
            "mean_ear": 0.30,
            "min_ear": 0.18,
            "max_ear": 0.42,
            "ear_std_dev": 0.025,
            "ear_std": 0.025,
            "avg_blink_duration": 0.20,
            "natural_blink_rate": 0.25,
            "neutral_mar": 0.12,
            "neutral_pitch": 0.0,
            "neutral_roll": 0.0,
            "neutral_yaw": 0.0,
        }
