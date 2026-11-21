"""Personalized threshold generator based on driver baseline statistics."""
from __future__ import annotations

from typing import Any, Dict, Union


class ThresholdGenerator:
    """Generates personalized driver thresholds from calibration baselines."""

    def __init__(self, config_or_loader: Any = None) -> None:
        if config_or_loader is None:
            self.margins: Dict[str, Any] = {}
        elif hasattr(config_or_loader, "get"):
            self.margins = config_or_loader.get("threshold_margins", default={}) or {}
        elif isinstance(config_or_loader, dict):
            self.margins = config_or_loader.get("threshold_margins", {}) or {}
        else:
            self.margins = {}

        self.ear_closed_std_mult = float(self.margins.get("ear_closed_std_multiplier", 1.5))
        self.long_eye_closure_mult = float(self.margins.get("long_eye_closure_multiplier", 2.0))
        self.blink_warning_mult = float(self.margins.get("blink_warning_multiplier", 1.8))
        self.mar_margin = float(self.margins.get("mar_margin", 0.20))
        self.head_pose_margin = float(self.margins.get("head_pose_margin_degrees", 15.0))

    def generate(self, baseline_stats: Dict[str, float]) -> Dict[str, float]:
        """Derive personalized fatigue detection thresholds from baseline statistics."""
        mean_ear = float(baseline_stats.get("mean_ear", 0.30))
        ear_std = float(baseline_stats.get("ear_std_dev", baseline_stats.get("ear_std", 0.02)))
        avg_blink = float(baseline_stats.get("avg_blink_duration", 0.20))
        neutral_mar = float(baseline_stats.get("neutral_mar", 0.10))

        neutral_pitch = float(baseline_stats.get("neutral_pitch", 0.0))
        neutral_roll = float(baseline_stats.get("neutral_roll", 0.0))
        neutral_yaw = float(baseline_stats.get("neutral_yaw", 0.0))

        # EAR closed threshold: bounded to sensible range [0.10, 0.25]
        ear_closed_thresh = mean_ear - (self.ear_closed_std_mult * ear_std)
        ear_closed_thresh = max(0.10, min(0.25, ear_closed_thresh))

        long_eye_closure_thresh = self.long_eye_closure_mult * avg_blink
        blink_warning_thresh = self.blink_warning_mult * avg_blink

        mar_thresh = neutral_mar + self.mar_margin

        return {
            "ear_closed_threshold": float(ear_closed_thresh),
            "long_eye_closure_threshold": float(long_eye_closure_thresh),
            "blink_warning_threshold": float(blink_warning_thresh),
            "mar_threshold": float(mar_thresh),
            "pitch_min": float(neutral_pitch - self.head_pose_margin),
            "pitch_max": float(neutral_pitch + self.head_pose_margin),
            "roll_min": float(neutral_roll - self.head_pose_margin),
            "roll_max": float(neutral_roll + self.head_pose_margin),
            "yaw_min": float(neutral_yaw - self.head_pose_margin),
            "yaw_max": float(neutral_yaw + self.head_pose_margin),
        }
