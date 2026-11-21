"""Feature extraction from MediaPipe Face Mesh landmarks.

Calculates:
- Eye Aspect Ratio (EAR) for left, right, and combined eyes.
- Mouth Aspect Ratio (MAR) using inner lip landmarks.
- Head pose (pitch, roll, yaw) via solvePnP + RQDecomp3x3.
- Stateful BlinkTracker and YawnTracker.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Landmark indices for MediaPipe Face Mesh (468/478-point model)
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

_MOUTH_H_L = 78
_MOUTH_H_R = 308
_MOUTH_V1_T = 81
_MOUTH_V1_B = 178
_MOUTH_V2_T = 13
_MOUTH_V2_B = 14
_MOUTH_V3_T = 311
_MOUTH_V3_B = 402

# 3D anatomical model points for head pose estimation (in mm)
_HEAD_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),      # Chin (landmark 152)
    (-225.0, 170.0, -135.0),   # Left eye corner (landmark 33)
    (225.0, 170.0, -135.0),    # Right eye corner (landmark 263)
    (-150.0, -150.0, -125.0),  # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0),   # Right mouth corner (landmark 291)
], dtype=np.float64)

_HEAD_LM_IDX = [1, 152, 33, 263, 61, 291]


def _eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR for a single eye given 6 landmark points.

    Formula: EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    """
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def calculate_ear(landmarks: np.ndarray) -> float:
    """Calculate mean Eye Aspect Ratio (EAR) across both eyes."""
    left = _eye_aspect_ratio(landmarks[_LEFT_EYE])
    right = _eye_aspect_ratio(landmarks[_RIGHT_EYE])
    return float((left + right) / 2.0)


class EyeMetrics:
    """Namespace for Eye metrics calculation."""
    @staticmethod
    def calculate_ear(landmarks: np.ndarray) -> float:
        return calculate_ear(landmarks)


def calculate_mar(landmarks: np.ndarray) -> float:
    """Calculate Mouth Aspect Ratio (MAR) using inner lip landmarks."""
    v1 = np.linalg.norm(landmarks[_MOUTH_V1_T] - landmarks[_MOUTH_V1_B])
    v2 = np.linalg.norm(landmarks[_MOUTH_V2_T] - landmarks[_MOUTH_V2_B])
    v3 = np.linalg.norm(landmarks[_MOUTH_V3_T] - landmarks[_MOUTH_V3_B])
    h = np.linalg.norm(landmarks[_MOUTH_H_L] - landmarks[_MOUTH_H_R])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2 + v3) / (2.0 * h))


class MouthMetrics:
    """Namespace for Mouth metrics calculation."""
    @staticmethod
    def calculate_mar(landmarks: np.ndarray) -> float:
        return calculate_mar(landmarks)


def estimate_head_pose(
    pixel_landmarks: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float]:
    """Estimate head orientation (pitch, roll, yaw) in degrees via solvePnP + RQDecomp3x3.

    Returns:
        (pitch, roll, yaw) in degrees.
    """
    img_pts = np.array(
        [[pixel_landmarks[i, 0], pixel_landmarks[i, 1]] for i in _HEAD_LM_IDX],
        dtype=np.float64,
    )

    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_mat = np.array([
        [focal, 0.0, center[0]],
        [0.0, focal, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return pitch, roll, yaw


class HeadPose:
    """Namespace for HeadPose estimation."""
    @staticmethod
    def estimate_pose(
        pixel_landmarks: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        return estimate_head_pose(pixel_landmarks, frame_w, frame_h)


def landmarks_to_pixels(landmarks: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert normalised [0, 1] landmark coordinates to pixel coordinates."""
    px = landmarks.copy().astype(np.float32)
    px[:, 0] *= frame_w
    px[:, 1] *= frame_h
    return px


class BlinkTracker:
    """Stateful tracker for blink events, durations, and eye closure duration."""

    _MIN_BLINK_S: float = 0.05
    _MAX_BLINK_S: float = 0.60

    def __init__(self, ear_closed_threshold: float = 0.20) -> None:
        self.ear_closed_threshold = ear_closed_threshold
        self.blink_count: int = 0
        self.current_closure_duration: float = 0.0
        self.blink_durations: List[float] = []

        self._closed: bool = False
        self._close_start: Optional[float] = None

    @property
    def current_closure_s(self) -> float:
        return self.current_closure_duration

    @property
    def avg_blink_duration(self) -> float:
        return float(np.mean(self.blink_durations)) if self.blink_durations else 0.0

    def get_average_blink_duration(self) -> float:
        return self.avg_blink_duration

    def set_threshold(self, ear_closed_threshold: float) -> None:
        self.ear_closed_threshold = ear_closed_threshold

    def update(self, smoothed_ear: float) -> bool:
        """Update tracker with latest smoothed EAR. Returns True if a blink finished."""
        now = time.monotonic()
        blink = False

        if smoothed_ear < self.ear_closed_threshold:
            if not self._closed:
                self._closed = True
                self._close_start = now
            self.current_closure_duration = now - self._close_start
        else:
            if self._closed and self._close_start is not None:
                dur = now - self._close_start
                self._closed = False
                self._close_start = None
                self.current_closure_duration = 0.0
                if self._MIN_BLINK_S <= dur <= self._MAX_BLINK_S:
                    self.blink_durations.append(dur)
                    self.blink_count += 1
                    blink = True
            else:
                self.current_closure_duration = 0.0

        return blink

    def reset(self) -> None:
        self.blink_count = 0
        self.current_closure_duration = 0.0
        self.blink_durations.clear()
        self._closed = False
        self._close_start = None


class YawnTracker:
    """Stateful tracker for yawn events and mouth opening duration."""

    _MIN_YAWN_S: float = 1.5
    _MAX_YAWN_S: float = 8.0

    def __init__(self, mar_threshold: float = 0.50) -> None:
        self.mar_threshold = mar_threshold
        self.yawn_count: int = 0
        self.current_open_s: float = 0.0

        self._open: bool = False
        self._open_start: Optional[float] = None

    def set_threshold(self, mar_threshold: float) -> None:
        self.mar_threshold = mar_threshold

    def update(self, smoothed_mar: float) -> bool:
        """Update tracker with latest smoothed MAR. Returns True if a yawn confirmed."""
        now = time.monotonic()
        confirmed = False

        if smoothed_mar > self.mar_threshold:
            if not self._open:
                self._open = True
                self._open_start = now
            self.current_open_s = now - self._open_start
        else:
            if self._open and self._open_start is not None:
                dur = now - self._open_start
                self._open = False
                self._open_start = None
                self.current_open_s = 0.0
                if self._MIN_YAWN_S <= dur <= self._MAX_YAWN_S:
                    self.yawn_count += 1
                    confirmed = True
            else:
                self.current_open_s = 0.0

        return confirmed

    @property
    def is_yawning(self) -> bool:
        return self._open and self.current_open_s >= self._MIN_YAWN_S

    def reset(self) -> None:
        self.yawn_count = 0
        self.current_open_s = 0.0
        self._open = False
        self._open_start = None
