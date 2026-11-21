"""Audio alert manager using Pygame sound mixer."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import pygame


class AlertManager:
    """Manages audio alerts on fatigue state transitions."""

    def __init__(self, config_or_loader: Any = None) -> None:
        cfg: Dict[str, Any] = {}
        if config_or_loader is not None:
            if hasattr(config_or_loader, "get"):
                cfg = config_or_loader.get("alerts", default={}) or {}
            elif isinstance(config_or_loader, dict):
                cfg = config_or_loader.get("alerts", {}) or {}

        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except Exception:
            self.audio_enabled = False

        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        if self.audio_enabled:
            self._load_sound("MILD", cfg.get("mild_sound", "assets/mild_alert.wav"))
            self._load_sound("DROWSY", cfg.get("drowsy_sound", "assets/drowsy_alert.wav"))
            self._load_sound("CRITICAL", cfg.get("critical_sound", "assets/critical_alert.wav"))

        self.last_state: str = "ALERT"

    def _load_sound(self, state: str, filepath: str) -> None:
        if os.path.exists(filepath):
            try:
                self.sounds[state] = pygame.mixer.Sound(filepath)
            except Exception:
                pass

    def trigger(self, current_state: str) -> Optional[str]:
        """Trigger audio alert on state transition. Returns event description or None."""
        alert_event = None

        if current_state != self.last_state:
            if current_state == "MILD":
                alert_event = "MILD_WARNING_TRIGGERED"
                self._play_sound("MILD")
            elif current_state == "DROWSY":
                alert_event = "DROWSY_WARNING_TRIGGERED"
                self._play_sound("DROWSY")
            elif current_state == "CRITICAL":
                alert_event = "CRITICAL_ALARM_TRIGGERED"
                self._play_sound("CRITICAL", loop=True)
            elif current_state == "ALERT":
                alert_event = "RECOVERED_TO_ALERT"
                self._stop_all()

            self.last_state = current_state

        return alert_event

    def _play_sound(self, state: str, loop: bool = False) -> None:
        if not self.audio_enabled:
            return

        self._stop_all()
        if state in self.sounds:
            loops = -1 if loop else 0
            t = threading.Thread(target=self.sounds[state].play, args=(loops,), daemon=True)
            t.start()

    def _stop_all(self) -> None:
        if self.audio_enabled:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    def cleanup(self) -> None:
        """Release audio mixer resources."""
        if self.audio_enabled:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass
