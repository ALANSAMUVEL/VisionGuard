import numpy as np
import pytest
from apdms.features import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    BlinkTracker,
    YawnTracker,
)


def test_calculate_ear_open_eye():
    # 478 mock landmarks
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Left eye landmarks: [33 (outer), 160, 158, 133 (inner), 153, 144]
    # Set horizontal distance = 2.0, vertical distances = 0.6 each
    landmarks[33] = [0.0, 0.0, 0.0]
    landmarks[133] = [2.0, 0.0, 0.0]
    landmarks[160] = [0.5, 0.3, 0.0]
    landmarks[144] = [0.5, -0.3, 0.0]
    landmarks[158] = [1.5, 0.3, 0.0]
    landmarks[153] = [1.5, -0.3, 0.0]

    # Right eye landmarks: [362, 385, 387, 263, 373, 380]
    landmarks[362] = [4.0, 0.0, 0.0]
    landmarks[263] = [6.0, 0.0, 0.0]
    landmarks[385] = [4.5, 0.3, 0.0]
    landmarks[380] = [4.5, -0.3, 0.0]
    landmarks[387] = [5.5, 0.3, 0.0]
    landmarks[373] = [5.5, -0.3, 0.0]

    ear = calculate_ear(landmarks)
    # v1 = 0.6, v2 = 0.6, h = 2.0 -> (0.6 + 0.6) / (2 * 2.0) = 0.30
    assert pytest.approx(ear, rel=1e-3) == 0.30


def test_calculate_mar():
    landmarks = np.zeros((478, 3), dtype=np.float32)
    # MOUTH_H_L=78, MOUTH_H_R=308, V1=(81,178), V2=(13,14), V3=(311,402)
    landmarks[78] = [0.0, 0.0, 0.0]
    landmarks[308] = [2.0, 0.0, 0.0]

    landmarks[81] = [0.5, 0.4, 0.0]
    landmarks[178] = [0.5, -0.4, 0.0]  # v1 = 0.8

    landmarks[13] = [1.0, 0.4, 0.0]
    landmarks[14] = [1.0, -0.4, 0.0]  # v2 = 0.8

    landmarks[311] = [1.5, 0.4, 0.0]
    landmarks[402] = [1.5, -0.4, 0.0]  # v3 = 0.8

    mar = calculate_mar(landmarks)
    # (0.8 + 0.8 + 0.8) / (2 * 2.0) = 2.4 / 4.0 = 0.60
    assert pytest.approx(mar, rel=1e-3) == 0.60


def test_blink_tracker_lifecycle():
    tracker = BlinkTracker(ear_closed_threshold=0.20)
    assert tracker.blink_count == 0

    # Eyes open
    tracker.update(0.35)
    assert tracker.blink_count == 0
    assert tracker.current_closure_duration == 0.0

    # Eyes close
    tracker.update(0.15)
    assert tracker.current_closure_duration >= 0.0


def test_yawn_tracker_lifecycle():
    yt = YawnTracker(mar_threshold=0.50)
    assert yt.yawn_count == 0
    assert not yt.is_yawning

    yt.update(0.20)
    assert yt.yawn_count == 0
