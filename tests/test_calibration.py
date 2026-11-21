import pytest
from apdms.calibration import Calibrator


def test_calibrator_fallback():
    cal = Calibrator(duration_seconds=1.0)
    stats = cal.get_baseline_stats()
    assert "mean_ear" in stats
    assert "neutral_pitch" in stats
    assert stats["mean_ear"] == 0.30


def test_calibrator_accumulation():
    cal = Calibrator(duration_seconds=10.0)
    cal.start()
    for _ in range(35):
        cal.update(ear=0.32, mar=0.12, pitch=1.0, roll=0.0, yaw=-1.0)

    stats = cal.get_baseline_stats()
    assert pytest.approx(stats["mean_ear"]) == 0.32
    assert pytest.approx(stats["neutral_mar"]) == 0.12
    assert pytest.approx(stats["neutral_pitch"]) == 1.0
    assert pytest.approx(stats["neutral_yaw"]) == -1.0
