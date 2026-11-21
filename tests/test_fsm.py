import pytest
from unittest.mock import patch
from apdms.fsm import FiniteStateMachine, FatigueState


class MockConfigLoader:
    def get(self, section, default=None):
        if section == "fsm_timers":
            return {
                "to_mild_time": 3.0,
                "to_drowsy_time": 3.0,
                "to_critical_time": 3.0,
                "recovery_time": 5.0,
            }
        elif section == "fsm_thresholds":
            return {
                "mild": 40,
                "drowsy": 70,
                "critical": 90,
            }
        return default or {}


@pytest.fixture
def fsm():
    return FiniteStateMachine(MockConfigLoader())


@patch("time.time")
def test_fsm_initial_state(mock_time, fsm):
    mock_time.return_value = 100.0
    state = fsm.update(10.0)
    assert state == FatigueState.ALERT


@patch("time.time")
def test_fsm_hysteresis_no_immediate_transition(mock_time, fsm):
    mock_time.return_value = 100.0
    fsm.update(0.0)

    state = fsm.update(95.0)
    assert state == FatigueState.ALERT
    assert fsm.target_state == FatigueState.CRITICAL
    assert fsm.condition_start_time == 100.0


@patch("time.time")
def test_fsm_transition_to_critical(mock_time, fsm):
    mock_time.return_value = 100.0
    fsm.update(0.0)

    fsm.update(95.0)

    mock_time.return_value = 102.0
    state = fsm.update(95.0)
    assert state == FatigueState.ALERT

    mock_time.return_value = 103.1
    state = fsm.update(95.0)
    assert state == FatigueState.CRITICAL


@patch("time.time")
def test_fsm_recovery(mock_time, fsm):
    mock_time.return_value = 100.0
    fsm.current_state = FatigueState.CRITICAL

    fsm.update(10.0)

    mock_time.return_value = 104.0
    state = fsm.update(10.0)
    assert state == FatigueState.CRITICAL

    mock_time.return_value = 105.1
    state = fsm.update(10.0)
    assert state == FatigueState.ALERT


@patch("time.time")
def test_fsm_abort_transition(mock_time, fsm):
    mock_time.return_value = 100.0
    fsm.update(0.0)

    fsm.update(75.0)
    assert fsm.target_state == FatigueState.DROWSY

    mock_time.return_value = 101.0
    state = fsm.update(10.0)
    assert state == FatigueState.ALERT
    assert fsm.target_state is None
    assert fsm.condition_start_time is None
