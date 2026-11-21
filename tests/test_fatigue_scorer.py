import pytest
from apdms.scorer import FatigueScorer


class MockConfigLoader:
    def get(self, section, default=None):
        if section == "fatigue_weights":
            return {
                "eye_closure": 0.40,
                "perclos": 0.25,
                "blink_behaviour": 0.15,
                "yawning": 0.10,
                "head_pose": 0.10,
            }
        return default or {}


@pytest.fixture
def scorer():
    return FatigueScorer(MockConfigLoader())


@pytest.fixture
def thresholds():
    return {
        "long_eye_closure_threshold": 0.5,
        "blink_warning_threshold": 0.3,
        "mar_threshold": 0.5,
        "ear_closed_threshold": 0.2,
        "pitch_min": -15.0,
        "pitch_max": 15.0,
        "roll_min": -15.0,
        "roll_max": 15.0,
        "yaw_min": -15.0,
        "yaw_max": 15.0,
    }


def test_fatigue_scorer_zero_fatigue(scorer, thresholds):
    current_features = {"ear": 0.35, "mar": 0.1, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    window_features = {"max_eye_closure_duration": 0.0, "perclos": 0.0, "avg_blink_duration": 0.1}

    score = scorer.calculate_score(current_features, window_features, thresholds)
    assert pytest.approx(score) == 5.0


def test_fatigue_scorer_max_fatigue(scorer, thresholds):
    current_features = {"ear": 0.1, "mar": 0.6, "pitch": 20.0, "roll": 20.0, "yaw": 20.0}
    window_features = {"max_eye_closure_duration": 0.6, "perclos": 20.0, "avg_blink_duration": 0.4}

    score = scorer.calculate_score(current_features, window_features, thresholds)
    assert pytest.approx(score) == 100.0


def test_fatigue_scorer_partial_eye_closure(scorer, thresholds):
    current_features = {"ear": 0.35, "mar": 0.1, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    window_features = {"max_eye_closure_duration": 0.25, "perclos": 0.0, "avg_blink_duration": 0.1}

    score = scorer.calculate_score(current_features, window_features, thresholds)
    assert pytest.approx(score) == 15.0


def test_fatigue_scorer_yawning(scorer, thresholds):
    current_features = {"ear": 0.25, "mar": 0.6, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    window_features = {"max_eye_closure_duration": 0.0, "perclos": 0.0, "avg_blink_duration": 0.1}

    score = scorer.calculate_score(current_features, window_features, thresholds)
    assert pytest.approx(score) == 15.0


def test_fatigue_scorer_talking_not_yawning(scorer, thresholds):
    current_features = {"ear": 0.35, "mar": 0.6, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    window_features = {"max_eye_closure_duration": 0.0, "perclos": 0.0, "avg_blink_duration": 0.1}

    score = scorer.calculate_score(current_features, window_features, thresholds)
    assert pytest.approx(score) == 5.0
