import pytest
from apdms.thresholds import ThresholdGenerator


def test_threshold_generator_defaults():
    tg = ThresholdGenerator()
    baselines = {
        "mean_ear": 0.32,
        "ear_std_dev": 0.02,
        "avg_blink_duration": 0.20,
        "neutral_mar": 0.12,
        "neutral_pitch": 2.0,
        "neutral_roll": 1.0,
        "neutral_yaw": -1.0,
    }
    t = tg.generate(baselines)

    # ear_closed: mean_ear - 1.5 * 0.02 = 0.29, clamped to max 0.25
    assert pytest.approx(t["ear_closed_threshold"]) == 0.25
    assert pytest.approx(t["long_eye_closure_threshold"]) == 0.40
    assert pytest.approx(t["blink_warning_threshold"]) == 0.36
    assert pytest.approx(t["mar_threshold"]) == 0.32
    assert pytest.approx(t["pitch_min"]) == -13.0
    assert pytest.approx(t["pitch_max"]) == 17.0
