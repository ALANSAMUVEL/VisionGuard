"""Lightweight file-based storage replacing SQLite.

Provides:
- ProfileStorage: JSON-based driver profile persistence (calibration & thresholds).
- SessionStorage: CSV-based real-time telemetry logging with buffered flushes.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProfileStorage:
    """Manages driver calibration profiles saved as JSON in profiles_dir."""

    def __init__(self, profiles_dir: str = "profiles") -> None:
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, driver_name: str) -> Path:
        safe_name = "".join(c for c in driver_name if c.isalnum() or c in ("-", "_")).strip()
        if not safe_name:
            safe_name = "driver"
        return self.profiles_dir / f"{safe_name}.json"

    def save_profile(
        self,
        driver_name: str,
        thresholds: Dict[str, Any],
        baseline_stats: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save driver calibration thresholds and baseline statistics to JSON."""
        path = self._get_path(driver_name)
        data = {
            "driver_name": driver_name,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_stats": baseline_stats or {},
            "thresholds": thresholds,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def load_profile(self, driver_name: str) -> Optional[Dict[str, Any]]:
        """Load profile if it exists. Returns combined dictionary or None."""
        path = self._get_path(driver_name)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            combined = {}
            if "baseline_stats" in data and isinstance(data["baseline_stats"], dict):
                combined.update(data["baseline_stats"])
            if "thresholds" in data and isinstance(data["thresholds"], dict):
                combined.update(data["thresholds"])
            return combined
        except (json.JSONDecodeError, OSError):
            return None

    def profile_exists(self, driver_name: str) -> bool:
        return self._get_path(driver_name).is_file()


class SessionStorage:
    """Manages session telemetry logging to CSV with memory buffer flushing."""

    CSV_FIELDS = [
        "timestamp",
        "ear",
        "mar",
        "blink_count",
        "blink_duration",
        "eye_closure",
        "perclos",
        "pitch",
        "roll",
        "yaw",
        "fatigue_score",
        "fsm_state",
        "alert_event",
        "yawn_count",
    ]

    def __init__(
        self,
        driver_name: str,
        session_id: Optional[int] = None,
        sessions_dir: str = "sessions",
        flush_rows: int = 30,
    ) -> None:
        self.driver_name = driver_name
        self.session_id = session_id or int(time.time())
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.flush_rows = flush_rows

        safe_name = "".join(c for c in driver_name if c.isalnum() or c in ("-", "_")).strip()
        self.csv_path = self.sessions_dir / f"session_{safe_name}_{self.session_id}.csv"

        self._buffer: List[Dict[str, Any]] = []
        self._all_records: List[Dict[str, Any]] = []
        self._file_initialized = False

    def log(self, record: Dict[str, Any]) -> None:
        """Add record to buffer and flush if buffer limit is reached."""
        clean_record = {k: record.get(k, 0.0) for k in self.CSV_FIELDS}
        self._buffer.append(clean_record)
        self._all_records.append(clean_record)

        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        """Flush in-memory buffer to CSV file."""
        if not self._buffer:
            return

        write_header = not self._file_initialized and not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if write_header:
                writer.writeheader()
                self._file_initialized = True
            writer.writerows(self._buffer)

        self._buffer.clear()

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return all logged rows for the session (from memory or reading the file)."""
        self.flush()
        if self._all_records:
            return self._all_records

        if not self.csv_path.exists():
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed: Dict[str, Any] = {}
                for k, v in row.items():
                    if k in ("fsm_state", "alert_event"):
                        parsed[k] = v if v else None
                    elif k in ("blink_count", "yawn_count"):
                        parsed[k] = int(v) if v else 0
                    else:
                        try:
                            parsed[k] = float(v) if v else 0.0
                        except ValueError:
                            parsed[k] = v
                records.append(parsed)
        return records

    def close(self) -> None:
        self.flush()
