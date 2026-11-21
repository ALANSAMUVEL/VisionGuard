import pytest
import shutil
import tempfile
from pathlib import Path
from apdms.storage import ProfileStorage, SessionStorage


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_profile_storage(temp_dir):
    storage = ProfileStorage(profiles_dir=temp_dir)
    assert not storage.profile_exists("Alice")

    thresholds = {"ear_closed_threshold": 0.21, "mar_threshold": 0.35}
    baseline_stats = {"mean_ear": 0.30}

    saved_path = storage.save_profile("Alice", thresholds, baseline_stats)
    assert saved_path.exists()
    assert storage.profile_exists("Alice")

    loaded = storage.load_profile("Alice")
    assert loaded is not None
    assert loaded["ear_closed_threshold"] == 0.21
    assert loaded["mean_ear"] == 0.30


def test_session_storage(temp_dir):
    session_storage = SessionStorage(driver_name="Bob", session_id=12345, sessions_dir=temp_dir, flush_rows=2)

    session_storage.log({"timestamp": 1.0, "ear": 0.32, "fatigue_score": 10.0, "fsm_state": "ALERT"})
    session_storage.log({"timestamp": 2.0, "ear": 0.28, "fatigue_score": 25.0, "fsm_state": "MILD"})

    logs = session_storage.get_logs()
    assert len(logs) == 2
    assert logs[0]["ear"] == 0.32
    assert logs[1]["fsm_state"] == "MILD"
    assert session_storage.csv_path.exists()
