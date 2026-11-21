"""Finite State Machine (FSM) for fatigue state progression with temporal hysteresis."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class FatigueState:
    ALERT = "ALERT"
    MILD = "MILD"
    DROWSY = "DROWSY"
    CRITICAL = "CRITICAL"


class FiniteStateMachine:
    """Manages driver fatigue state transitions with hysteresis timers."""

    def __init__(self, config_or_loader: Any = None) -> None:
        cfg: Dict[str, Any] = {}
        if config_or_loader is not None:
            if hasattr(config_or_loader, "get"):
                cfg = config_or_loader
            elif isinstance(config_or_loader, dict):
                cfg = config_or_loader

        # Handle both flat and nested configuration formats
        timers = (
            (cfg.get("fsm_timers") if hasattr(cfg, "get") else None)
            or ((cfg.get("fsm") or {}).get("timers") if isinstance(cfg, dict) or hasattr(cfg, "get") else None)
            or {}
        )
        self.time_to_mild = float(timers.get("to_mild_time", timers.get("to_mild", 3.0)))
        self.time_to_drowsy = float(timers.get("to_drowsy_time", timers.get("to_drowsy", 3.0)))
        self.time_to_critical = float(timers.get("to_critical_time", timers.get("to_critical", 2.0)))
        self.recovery_time = float(timers.get("recovery_time", 5.0))

        thresholds = (
            (cfg.get("fsm_thresholds") if hasattr(cfg, "get") else None)
            or ((cfg.get("fsm") or {}).get("thresholds") if isinstance(cfg, dict) or hasattr(cfg, "get") else None)
            or {}
        )
        self.thresh_mild = float(thresholds.get("mild", 35))
        self.thresh_drowsy = float(thresholds.get("drowsy", 60))
        self.thresh_critical = float(thresholds.get("critical", 80))

        self.current_state: str = FatigueState.ALERT
        self.target_state: Optional[str] = None
        self.condition_start_time: Optional[float] = None

    def update(self, score: float) -> str:
        """Update FSM with current fatigue score and return the active state."""
        current_time = time.time()

        if score >= self.thresh_critical:
            raw_state = FatigueState.CRITICAL
        elif score >= self.thresh_drowsy:
            raw_state = FatigueState.DROWSY
        elif score >= self.thresh_mild:
            raw_state = FatigueState.MILD
        else:
            raw_state = FatigueState.ALERT

        if raw_state != self.current_state:
            if self.target_state != raw_state:
                self.target_state = raw_state
                self.condition_start_time = current_time
            else:
                elapsed = current_time - (self.condition_start_time or current_time)
                required_time = self.recovery_time

                # Escalation conditions
                if raw_state == FatigueState.CRITICAL and self.current_state in (
                    FatigueState.ALERT,
                    FatigueState.MILD,
                    FatigueState.DROWSY,
                ):
                    required_time = self.time_to_critical
                elif raw_state == FatigueState.DROWSY and self.current_state in (
                    FatigueState.ALERT,
                    FatigueState.MILD,
                ):
                    required_time = self.time_to_drowsy
                elif raw_state == FatigueState.MILD and self.current_state == FatigueState.ALERT:
                    required_time = self.time_to_mild

                if elapsed >= required_time:
                    self.current_state = raw_state
                    self.target_state = None
                    self.condition_start_time = None
        else:
            self.target_state = None
            self.condition_start_time = None

        return self.current_state
