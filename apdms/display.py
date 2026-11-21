"""Real-time display module for HUD overlay using OpenCV."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Eye and mouth landmark contour indices
_EYE_CONTOUR_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
]
_MOUTH_CONTOUR_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

_STATE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "ALERT": (60, 220, 60),      # Green
    "MILD": (0, 220, 220),       # Yellow
    "DROWSY": (0, 140, 255),     # Orange
    "CRITICAL": (40, 40, 255),   # Red
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), _FONT, scale, colour, thickness, cv2.LINE_AA)


class Display:
    """Manages HUD rendering, landmark overlays, and window lifecycle."""

    def __init__(self, window_name: str = "VisionGuard – Driver Monitoring") -> None:
        self.window_name = window_name
        self._alert_banner_text: Optional[str] = None
        self._alert_banner_frames_remaining: int = 0
        self._banner_duration_frames: int = 90

    def trigger_alert_banner(self, message: str) -> None:
        self._alert_banner_text = message
        self._alert_banner_frames_remaining = self._banner_duration_frames

    def draw_landmarks(self, frame: np.ndarray, pixel_landmarks: Optional[np.ndarray]) -> None:
        if pixel_landmarks is None:
            return

        for idx in _EYE_CONTOUR_INDICES:
            if idx < len(pixel_landmarks):
                x, y = int(pixel_landmarks[idx, 0]), int(pixel_landmarks[idx, 1])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        for idx in _MOUTH_CONTOUR_INDICES:
            if idx < len(pixel_landmarks):
                x, y = int(pixel_landmarks[idx, 0]), int(pixel_landmarks[idx, 1])
                cv2.circle(frame, (x, y), 1, (0, 200, 255), -1)

    def draw_hud(
        self,
        frame: np.ndarray,
        fsm_state: str,
        score: float,
        metrics: Optional[Dict[str, Any]] = None,
        is_calibrating: bool = False,
        calib_progress: float = 0.0,
    ) -> None:
        h, w = frame.shape[:2]
        colour = _STATE_COLOURS.get(fsm_state, (200, 200, 200))

        if is_calibrating:
            self._draw_calibration_overlay(frame, w, h, calib_progress)
        else:
            self._draw_metric_panel(frame, w, h, fsm_state, score, colour, metrics)
            self._draw_state_badge(frame, w, fsm_state, score, colour)
            self._draw_alert_banner(frame, w, h)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def check_quit(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), ord("Q"))

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def _draw_calibration_overlay(
        self, frame: np.ndarray, w: int, h: int, progress: float
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        _put_text(
            frame,
            "CALIBRATION IN PROGRESS – Look straight ahead naturally",
            10,
            25,
            (255, 230, 0),
            0.52,
            1,
        )
        pct = int(progress * 100)
        _put_text(frame, f"{pct}% complete", 10, 52, (255, 255, 255), 0.55, 1)

        bar_y1, bar_y2 = h - 12, h
        cv2.rectangle(frame, (0, bar_y1), (w, bar_y2), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, bar_y1), (int(w * progress), bar_y2), (0, 220, 220), -1)

    def _draw_metric_panel(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        fsm_state: str,
        score: float,
        colour: Tuple[int, int, int],
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        panel_width = 200
        panel_height = 215
        margin = 8

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + panel_width, margin + panel_height),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(
            frame,
            (margin, margin),
            (margin + panel_width, margin + panel_height),
            colour,
            1,
        )

        m = metrics or {}
        lines = [
            ("VisionGuard APDMS", (220, 220, 220), 0.50),
            ("─" * 22, (80, 80, 80), 0.40),
            (f"EAR:    {float(m.get('ear', 0.0)):.3f}", (200, 200, 200), 0.45),
            (f"MAR:    {float(m.get('mar', 0.0)):.3f}", (200, 200, 200), 0.45),
            (f"PERCLOS:{float(m.get('perclos', 0.0)):.1f}%", (200, 200, 200), 0.45),
            (f"Pitch:  {float(m.get('pitch', 0.0)):.1f}°", (200, 200, 200), 0.45),
            (f"Roll:   {float(m.get('roll', 0.0)):.1f}°", (200, 200, 200), 0.45),
            (f"Yaw:    {float(m.get('yaw', 0.0)):.1f}°", (200, 200, 200), 0.45),
            (f"Blinks: {int(m.get('blink_count', 0))}", (200, 200, 200), 0.45),
            (f"Yawns:  {int(m.get('yawn_count', 0))}", (200, 200, 200), 0.45),
            ("─" * 22, (80, 80, 80), 0.40),
            (f"SCORE:  {score:.1f}/100", colour, 0.52),
        ]

        y = margin + 18
        for text, col, scale in lines:
            _put_text(frame, text, margin + 6, y, col, scale, 1)
            y += int(scale * 30) + 2

    def _draw_state_badge(
        self,
        frame: np.ndarray,
        w: int,
        fsm_state: str,
        score: float,
        colour: Tuple[int, int, int],
    ) -> None:
        badge_w, badge_h = 160, 50
        bx = w - badge_w - 8
        by = 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), colour, 2)

        _put_text(frame, fsm_state, bx + 10, by + 22, colour, 0.65, 2)

        bar_x = bx + 6
        bar_y = by + 36
        bar_total = badge_w - 12
        fill = int(bar_total * (min(100.0, max(0.0, score)) / 100.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_total, bar_y + 8), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 8), colour, -1)

        if self._alert_banner_frames_remaining > 0:
            self._alert_banner_frames_remaining -= 1

    def _draw_alert_banner(self, frame: np.ndarray, w: int, h: int) -> None:
        if not self._alert_banner_text or self._alert_banner_frames_remaining <= 0:
            return

        alpha = min(1.0, self._alert_banner_frames_remaining / 30.0)
        banner_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - alpha * 0.8, 0, frame)

        text_size = cv2.getTextSize(self._alert_banner_text, _FONT, 0.55, 1)[0]
        tx = max(10, (w - text_size[0]) // 2)
        ty = h - banner_h + 27
        _put_text(frame, self._alert_banner_text, tx, ty, (255, 255, 255), 0.55, 1)
