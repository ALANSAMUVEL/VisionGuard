import pytest
from apdms.processing import SlidingWindow


def test_sliding_window_add_and_clean():
    window = SlidingWindow(window_duration_seconds=3.0)
    base_time = 100.0

    window.add(base_time, {"ear": 0.3})
    window.add(base_time + 1.0, {"ear": 0.25})
    window.add(base_time + 2.0, {"ear": 0.2})
    window.add(base_time + 4.0, {"ear": 0.1})

    assert len(window.buffer) == 3
    assert window.buffer[0][0] == base_time + 1.0
    assert window.buffer[1][0] == base_time + 2.0
    assert window.buffer[2][0] == base_time + 4.0


def test_sliding_window_get_average():
    window = SlidingWindow(window_duration_seconds=3.0)
    base_time = 100.0
    window.add(base_time, {"mar": 0.1})
    window.add(base_time + 1.0, {"mar": 0.3})
    window.add(base_time + 2.0, {"mar": 0.5})

    assert pytest.approx(window.get_average("mar")) == 0.3
    assert window.get_average("nonexistent") == 0.0


def test_sliding_window_get_perclos():
    window = SlidingWindow(window_duration_seconds=3.0)
    base_time = 100.0
    window.add(base_time, {"ear": 0.3})
    window.add(base_time + 0.5, {"ear": 0.15})
    window.add(base_time + 1.0, {"ear": 0.1})
    window.add(base_time + 1.5, {"ear": 0.35})

    perclos = window.get_perclos(ear_threshold=0.2)
    assert pytest.approx(perclos) == 50.0


def test_sliding_window_get_max_duration():
    window = SlidingWindow(window_duration_seconds=5.0)
    base_time = 100.0

    window.add(base_time, {"ear": 0.3})
    window.add(base_time + 1.0, {"ear": 0.15})
    window.add(base_time + 1.5, {"ear": 0.1})
    window.add(base_time + 2.5, {"ear": 0.15})
    window.add(base_time + 2.6, {"ear": 0.25})

    window.add(base_time + 3.0, {"ear": 0.1})
    window.add(base_time + 4.9, {"ear": 0.1})

    max_dur = window.get_max_duration("ear", lambda x: x < 0.2)
    assert pytest.approx(max_dur) == 1.9


def test_sliding_window_count_events():
    window = SlidingWindow(window_duration_seconds=5.0)
    base_time = 100.0

    window.add(base_time, {"ear": 0.3})
    window.add(base_time + 1.0, {"ear": 0.1})
    window.add(base_time + 1.5, {"ear": 0.3})

    window.add(base_time + 2.0, {"ear": 0.1})
    window.add(base_time + 2.5, {"ear": 0.3})

    window.add(base_time + 3.0, {"ear": 0.1})

    count = window.count_events("ear", lambda x: x < 0.2)
    assert count == 2
