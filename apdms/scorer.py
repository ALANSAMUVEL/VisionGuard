"""Multi-metric rule-based fatigue scoring engine."""
from __future__ import annotations

from typing import Any, Dict


class FatigueScorer:
    """Computes a normalized fatigue score (0.0 to 100.0) combining instant and temporal metrics."""

    def __init__(self, config_or_loader: Any = None) -> None:
        if config_or_loader is None:
            self.weights: Dict[str, Any] = {}
        elif hasattr(config_or_loader, "get"):
            self.weights = config_or_loader.get("fatigue_weights", default={}) or {}
        elif isinstance(config_or_loader, dict):
            self.weights = config_or_loader.get("fatigue_weights", {}) or {}
        else:
            self.weights = {}

        self.w_eye_closure = float(self.weights.get("eye_closure", 0.40))
        self.w_perclos = float(self.weights.get("perclos", 0.25))
        self.w_blink = float(self.weights.get("blink_behaviour", 0.15))
        self.w_yawn = float(self.weights.get("yawning", 0.10))
        self.w_head = float(self.weights.get("head_pose", 0.10))

    def calculate_score(
        self,
        current_features: Dict[str, Any],
        window_features: Dict[str, Any],
        thresholds: Dict[str, Any],
    ) -> float:
        """Calculate weighted score in [0.0, 100.0]."""
        score = 0.0

        # 1. Eye Closure (continuous duration)
        max_eye_closure = float(window_features.get("max_eye_closure_duration", 0.0))
        long_closure_thresh = float(thresholds.get("long_eye_closure_threshold", 0.50))
        if long_closure_thresh > 0:
            if max_eye_closure >= long_closure_thresh:
                score += self.w_eye_closure * 100.0
            else:
                ratio = min(1.0, max_eye_closure / long_closure_thresh)
                score += self.w_eye_closure * 100.0 * (ratio ** 2)

        # 2. PERCLOS (Percentage of eye closure over sliding window)
        perclos = float(window_features.get("perclos", 0.0))
        if perclos >= 15.0:
            score += self.w_perclos * 100.0
        else:
            score += self.w_perclos * 100.0 * (perclos / 15.0)

        # 3. Blink Behaviour (Prolonged average blink duration)
        avg_blink_dur = float(window_features.get("avg_blink_duration", 0.0))
        blink_warning_thresh = float(thresholds.get("blink_warning_threshold", 0.30))
        if blink_warning_thresh > 0:
            if avg_blink_dur >= blink_warning_thresh:
                score += self.w_blink * 100.0
            else:
                ratio = min(1.0, avg_blink_dur / blink_warning_thresh)
                score += self.w_blink * 100.0 * ratio

        # 4. Yawning (MAR above threshold accompanied by eye squinting; talking rejected)
        mar = float(current_features.get("mar", 0.0))
        ear = float(current_features.get("ear", 0.0))
        mar_thresh = float(thresholds.get("mar_threshold", 0.50))
        ear_closed_thresh = float(thresholds.get("ear_closed_threshold", 0.20))

        is_yawning = (mar > mar_thresh) and (ear < (ear_closed_thresh * 1.5))
        if is_yawning:
            score += self.w_yawn * 100.0

        # 5. Head Pose (Drooping pitch, tilting roll, averted yaw)
        pitch = float(current_features.get("pitch", 0.0))
        roll = float(current_features.get("roll", 0.0))
        yaw = float(current_features.get("yaw", 0.0))

        pose_penalty = 0.0
        if pitch < float(thresholds.get("pitch_min", -15.0)) or pitch > float(thresholds.get("pitch_max", 15.0)):
            pose_penalty += 0.5
        if roll < float(thresholds.get("roll_min", -15.0)) or roll > float(thresholds.get("roll_max", 15.0)):
            pose_penalty += 0.3
        if yaw < float(thresholds.get("yaw_min", -15.0)) or yaw > float(thresholds.get("yaw_max", 15.0)):
            pose_penalty += 0.2

        score += self.w_head * 100.0 * min(1.0, pose_penalty)

        return float(min(100.0, max(0.0, score)))
