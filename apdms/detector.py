"""Face detection, landmark extraction, and quality assessment.

Key design change from v1
--------------------------
Quality is now expressed as a **confidence score** (0.0–1.0) rather than
a binary pass/fail gate.  Blur and poor lighting REDUCE confidence but do
NOT block the monitoring pipeline.

Only two conditions hard-block processing:
  1. No face detected (MediaPipe returns nothing).
  2. Landmark extraction completely fails.

The FaceFrame dataclass carries everything the pipeline needs per frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Lazy-import mediapipe to handle both old and new package layouts.
try:
    import mediapipe.python.solutions.face_mesh as _mp_face_mesh
except (ImportError, ModuleNotFoundError):
    import mediapipe as _mp
    _mp_face_mesh = _mp.solutions.face_mesh   # type: ignore[assignment]


@dataclass
class FaceFrame:
    """All per-frame face data passed downstream through the pipeline."""
    detected:        bool
    landmarks:       Optional[np.ndarray]        # normalised (N, 3)
    pixel_landmarks: Optional[np.ndarray]        # pixel-space (N, 3)
    confidence:      float                       # 0.0 – 1.0 quality score
    diagnostic:      str                         # human-readable status
    blur_score:      float                       # Laplacian variance
    brightness:      float                       # mean pixel value (0–255)


class FaceDetector:
    """Wraps MediaPipe Face Mesh and produces a ``FaceFrame`` for every input.

    Quality thresholds are loaded from the ``face_quality`` section of config.
    Blur and brightness reduce confidence but never stop the pipeline.
    """

    def __init__(self, cfg: dict) -> None:
        q = cfg
        self._blur_thresh: float    = q.get("blur_threshold",    20.0)
        self._min_bright:  float    = q.get("min_brightness",    30.0)
        self._max_bright:  float    = q.get("max_brightness",   235.0)
        self._min_area:    float    = q.get("min_face_area_ratio", 0.015)
        self._max_area:    float    = q.get("max_face_area_ratio", 0.95)
        self._min_spread:  float    = q.get("min_landmark_spread", 0.004)

        self._mesh = _mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FaceFrame:
        """Detect face landmarks in *frame* and return a FaceFrame.

        Never returns None — always returns a populated FaceFrame.
        """
        h, w = frame.shape[:2]
        blur      = self._blur_score(frame)
        brightness = self._brightness(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return FaceFrame(
                detected=False, landmarks=None, pixel_landmarks=None,
                confidence=0.0, diagnostic="NO FACE",
                blur_score=blur, brightness=brightness,
            )

        face_lm = results.multi_face_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in face_lm.landmark], dtype=np.float32)

        # Confidence = 1.0 minus penalties from blur / brightness / face size.
        confidence, diag = self._assess_quality(lm, blur, brightness, w, h)

        px_lm = lm.copy()
        px_lm[:, 0] *= w
        px_lm[:, 1] *= h

        return FaceFrame(
            detected=True,
            landmarks=lm,
            pixel_landmarks=px_lm,
            confidence=confidence,
            diagnostic=diag,
            blur_score=blur,
            brightness=brightness,
        )

    def close(self) -> None:
        self._mesh.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _blur_score(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _brightness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _assess_quality(
        self,
        lm: np.ndarray,
        blur: float,
        brightness: float,
        w: int,
        h: int,
    ) -> tuple[float, str]:
        """Return (confidence 0–1, diagnostic_string)."""
        confidence = 1.0
        reasons: list[str] = []

        # ── 1. Landmark spread guard (collapsed / degenerate detection) ────────
        sx, sy = float(lm[:, 0].std()), float(lm[:, 1].std())
        if sx < self._min_spread or sy < self._min_spread:
            return 0.0, "LANDMARK FAILURE"

        # ── 2. Face size in frame ─────────────────────────────────────────────
        xmin, xmax = lm[:, 0].min(), lm[:, 0].max()
        ymin, ymax = lm[:, 1].min(), lm[:, 1].max()
        area_ratio = (xmax - xmin) * (ymax - ymin)  # normalised area

        if area_ratio < self._min_area:
            confidence -= 0.4
            reasons.append("FACE TOO SMALL")
        elif area_ratio > self._max_area:
            confidence -= 0.2
            reasons.append("FACE TOO CLOSE")

        # ── 3. Blur ───────────────────────────────────────────────────────────
        if blur < self._blur_thresh:
            penalty = min(0.35, (self._blur_thresh - blur) / self._blur_thresh * 0.35)
            confidence -= penalty
            reasons.append("LOW SHARPNESS")

        # ── 4. Lighting ───────────────────────────────────────────────────────
        if brightness < self._min_bright:
            penalty = min(0.35, (self._min_bright - brightness) / self._min_bright * 0.35)
            confidence -= penalty
            reasons.append("LOW LIGHT")
        elif brightness > self._max_bright:
            confidence -= 0.15
            reasons.append("OVEREXPOSED")

        confidence = max(0.0, confidence)
        diag = " | ".join(reasons) if reasons else "OK"
        return confidence, diag
