import shutil
import tempfile
from pathlib import Path
from apdms.reports import PDFReportGenerator


def test_pdf_report_generator():
    temp_dir = tempfile.mkdtemp()
    try:
        gen = PDFReportGenerator(output_dir=temp_dir)

        # Test empty logs
        assert gen.generate_report("Alice", []) is None

        # Test populated logs
        mock_logs = [
            {
                "timestamp": 100.0 + i,
                "ear": 0.30 - (0.01 if i > 5 else 0.0),
                "mar": 0.12 + (0.2 if i == 7 else 0.0),
                "blink_count": i // 3,
                "blink_duration": 0.2,
                "eye_closure": 0.1 if i > 5 else 0.0,
                "perclos": 10.0 if i > 5 else 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "fatigue_score": 15.0 + i * 2.0,
                "fsm_state": "MILD" if i > 5 else "ALERT",
                "alert_event": "MILD_WARNING_TRIGGERED" if i == 6 else None,
                "yawn_count": 1 if i >= 7 else 0,
            }
            for i in range(12)
        ]

        pdf_path = gen.generate_report("Alice", mock_logs, session_id=999)
        assert pdf_path is not None
        assert Path(pdf_path).is_file()
        assert Path(pdf_path).stat().st_size > 1000
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
