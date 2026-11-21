"""VisionGuard — Adaptive Personalized Driver Monitoring System (APDMS).

Main application entry point. Coordinates:
  1. Configuration loading (config.yaml).
  2. Driver profile lookup and personalized baseline calibration.
  3. Real-time monitoring loop:
     Frame acquisition -> Face detection -> Feature extraction ->
     Signal smoothing (EMA) -> Sliding window temporal analysis ->
     Rule-based fatigue scoring -> Temporal FSM -> Audio/visual alerts ->
     CSV telemetry logging -> HUD rendering.
  4. Post-session PDF report compilation.

Usage:
  python main.py                  # Prompts for driver name interactively
  python main.py --driver Alice   # Specify driver name
  python main.py --recalibrate    # Force recalibration for existing profile
  python main.py --camera 0       # Select camera index
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

from apdms import (
    AlertManager,
    BlinkTracker,
    Calibrator,
    Camera,
    Display,
    EMA,
    FaceDetector,
    FatigueScorer,
    FatigueState,
    FiniteStateMachine,
    PDFReportGenerator,
    ProfileStorage,
    SessionStorage,
    SlidingWindow,
    ThresholdGenerator,
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
)


def _setup_logger(level_str: str = "INFO", log_file: str = "logs/apdms.log") -> logging.Logger:
    """Configure rotating file logger and stream output."""
    logger = logging.getLogger("VisionGuard")
    if logger.handlers:
        return logger

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file with fallback."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        return {}
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: Failed to parse {config_path}: {exc}. Using default configuration.")
        return {}


class _ThrottledLogger:
    """Emits log messages at most once per key per interval seconds."""

    def __init__(self, logger: logging.Logger, interval: float = 4.0) -> None:
        self._logger = logger
        self._interval = interval
        self._last_logged: Dict[str, float] = {}

    def warning(self, key: str, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_logged.get(key, 0.0) >= self._interval:
            self._logger.warning(msg)
            self._last_logged[key] = now

    def info(self, key: str, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_logged.get(key, 0.0) >= self._interval:
            self._logger.info(msg)
            self._last_logged[key] = now


class VisionGuardApp:
    """Application orchestrator for VisionGuard APDMS."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = _load_config(config_path)
        log_cfg = self.config.get("logging", {})
        self.logger = _setup_logger(
            level_str=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file", "logs/apdms.log"),
        )
        self.throttled = _ThrottledLogger(self.logger)

        storage_cfg = self.config.get("storage", {})
        self.profiles_dir = storage_cfg.get("profiles_dir", "profiles")
        self.sessions_dir = storage_cfg.get("sessions_dir", "sessions")
        self.reports_dir = storage_cfg.get("reports_dir", "reports")
        self.csv_flush_rows = int(storage_cfg.get("csv_flush_rows", 30))

        self.profile_storage = ProfileStorage(self.profiles_dir)
        self.pdf_generator = PDFReportGenerator(self.reports_dir)

    def run(
        self,
        driver_name: str,
        camera_index: int = 0,
        force_recalibrate: bool = False,
    ) -> None:
        """Run complete driver session lifecycle."""
        self.logger.info("=" * 60)
        self.logger.info("VisionGuard (APDMS) starting for driver: %s", driver_name)
        self.logger.info("=" * 60)

        # 1. Profile lookup or calibration
        profile = self._get_or_calibrate(driver_name, camera_index, force_recalibrate)
        if profile is None:
            self.logger.warning("Setup cancelled or calibration failed. Exiting.")
            return

        # 2. Monitoring loop
        self._run_monitoring(driver_name, camera_index, profile)

    def _get_or_calibrate(
        self, driver_name: str, camera_index: int, force: bool
    ) -> Optional[Dict[str, Any]]:
        if not force and self.profile_storage.profile_exists(driver_name):
            profile = self.profile_storage.load_profile(driver_name)
            if profile:
                self.logger.info("Loaded existing profile for '%s'", driver_name)
                return profile

        self.logger.info("Initiating baseline calibration for '%s'", driver_name)
        return self._calibrate_driver(driver_name, camera_index)

    def _calibrate_driver(self, driver_name: str, camera_index: int) -> Optional[Dict[str, Any]]:
        calib_cfg = self.config.get("calibration", {})
        duration = float(calib_cfg.get("duration_seconds", 45.0))

        cam_cfg = dict(self.config.get("camera", {}))
        cam_cfg["index"] = camera_index

        try:
            cam = Camera(cam_cfg).start()
        except IOError as e:
            self.logger.error("Camera initialization failed: %s", e)
            return None

        detector = FaceDetector(self.config.get("face_quality", {}))
        display = Display("VisionGuard – Calibration")
        calibrator = Calibrator(duration_seconds=duration)
        calibrator.start()

        self.logger.info("Calibration in progress (%.0f s). Look straight ahead naturally.", duration)

        try:
            while not calibrator.is_complete():
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                face_frame = detector.process(frame)

                if face_frame.detected and face_frame.landmarks is not None:
                    ear = calculate_ear(face_frame.landmarks)
                    mar = calculate_mar(face_frame.landmarks)
                    if face_frame.pixel_landmarks is not None:
                        pitch, roll, yaw = estimate_head_pose(face_frame.pixel_landmarks, w, h)
                        calibrator.update(ear, mar, pitch, roll, yaw)
                        display.draw_landmarks(frame, face_frame.pixel_landmarks)

                progress = calibrator.get_progress()
                display.draw_hud(frame, "ALERT", 0.0, is_calibrating=True, calib_progress=progress)
                display.show(frame)

                if display.check_quit():
                    self.logger.info("Calibration aborted by user.")
                    return None
        finally:
            cam.stop()
            detector.close()
            display.close()

        baseline_stats = calibrator.get_baseline_stats()
        threshold_gen = ThresholdGenerator(self.config)
        thresholds = threshold_gen.generate(baseline_stats)

        self.profile_storage.save_profile(driver_name, thresholds, baseline_stats)
        self.logger.info("Calibration complete. Thresholds saved successfully.")

        combined = {**baseline_stats, **thresholds}
        return combined

    def _run_monitoring(
        self,
        driver_name: str,
        camera_index: int,
        thresholds: Dict[str, Any],
    ) -> None:
        cam_cfg = dict(self.config.get("camera", {}))
        cam_cfg["index"] = camera_index

        proc_cfg = self.config.get("processing", self.config.get("signal_processing", {}))
        ema_alpha = float(proc_cfg.get("ema_alpha", 0.15))
        window_seconds = float(proc_cfg.get("window_seconds", proc_cfg.get("window_size_seconds", 4.0)))

        try:
            cam = Camera(cam_cfg).start()
        except IOError as e:
            self.logger.error("Camera initialization failed: %s", e)
            return

        detector = FaceDetector(self.config.get("face_quality", {}))
        display = Display("VisionGuard – Driver Monitoring")
        alert_mgr = AlertManager(self.config)
        scorer = FatigueScorer(self.config)
        fsm = FiniteStateMachine(self.config)

        ema_ear = EMA(alpha=ema_alpha)
        ema_mar = EMA(alpha=ema_alpha)
        ema_pitch = EMA(alpha=ema_alpha)
        ema_roll = EMA(alpha=ema_alpha)
        ema_yaw = EMA(alpha=ema_alpha)

        window = SlidingWindow(window_seconds=window_seconds)

        ear_thresh = float(thresholds.get("ear_closed_threshold", 0.20))
        mar_thresh = float(thresholds.get("mar_threshold", 0.50))
        blink_tracker = BlinkTracker(ear_closed_threshold=ear_thresh)
        yawn_tracker = YawnTracker(mar_threshold=mar_thresh)

        session_id = int(time.time())
        session_storage = SessionStorage(
            driver_name=driver_name,
            session_id=session_id,
            sessions_dir=self.sessions_dir,
            flush_rows=self.csv_flush_rows,
        )

        current_fsm_state = FatigueState.ALERT
        current_score = 0.0
        face_lost_time: Optional[float] = None
        log_interval = 1.0
        last_log_time = 0.0

        banner_map = {
            "MILD_WARNING_TRIGGERED": "Mild fatigue detected - stay focused",
            "DROWSY_WARNING_TRIGGERED": "Drowsiness detected - take a break soon",
            "CRITICAL_ALARM_TRIGGERED": "CRITICAL: PULL OVER SAFELY NOW",
            "RECOVERED_TO_ALERT": "Driver alert - normal status resumed",
        }

        self.logger.info("Monitoring session active. Press 'q' in video window to exit.")

        try:
            while True:
                now_mono = time.monotonic()
                now_wall = time.time()

                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]
                face_frame = detector.process(frame)

                if not face_frame.detected or face_frame.landmarks is None:
                    if face_lost_time is None:
                        face_lost_time = now_mono
                        self.throttled.warning("face_lost", "Face not detected. Fatigue state frozen.")
                    display.draw_hud(frame, current_fsm_state, current_score)
                    display.show(frame)
                    if display.check_quit():
                        break
                    continue

                if face_lost_time is not None:
                    self.logger.debug("Face reacquired after %.1f s", now_mono - face_lost_time)
                    face_lost_time = None

                raw_ear = calculate_ear(face_frame.landmarks)
                raw_mar = calculate_mar(face_frame.landmarks)
                pitch, roll, yaw = (0.0, 0.0, 0.0)
                if face_frame.pixel_landmarks is not None:
                    pitch, roll, yaw = estimate_head_pose(face_frame.pixel_landmarks, w, h)

                s_ear = ema_ear.update(raw_ear)
                s_mar = ema_mar.update(raw_mar)
                s_pitch = ema_pitch.update(pitch)
                s_roll = ema_roll.update(roll)
                s_yaw = ema_yaw.update(yaw)

                blink_tracker.update(s_ear)
                yawn_tracker.update(s_mar)

                frame_features = {
                    "ear": s_ear,
                    "mar": s_mar,
                    "pitch": s_pitch,
                    "roll": s_roll,
                    "yaw": s_yaw,
                }
                window.add(now_mono, frame_features)

                perclos = window.get_perclos(ear_threshold=ear_thresh)
                max_closure = blink_tracker.current_closure_duration
                avg_blink_dur = blink_tracker.avg_blink_duration

                window_features = {
                    "perclos": perclos,
                    "max_eye_closure_duration": max_closure,
                    "avg_blink_duration": avg_blink_dur,
                    "avg_ear": window.get_average("ear"),
                    "avg_mar": window.get_average("mar"),
                }

                current_score = scorer.calculate_score(frame_features, window_features, thresholds)
                new_state = fsm.update(current_score)
                alert_event = alert_mgr.trigger(new_state)

                if alert_event:
                    self.logger.info("Event: %s | Score: %.1f | State: %s", alert_event, current_score, new_state)
                    banner = banner_map.get(alert_event, alert_event)
                    display.trigger_alert_banner(banner)

                current_fsm_state = new_state

                if now_wall - last_log_time >= log_interval:
                    record = {
                        "timestamp": now_wall,
                        "ear": float(s_ear),
                        "mar": float(s_mar),
                        "blink_count": blink_tracker.blink_count,
                        "blink_duration": float(avg_blink_dur),
                        "eye_closure": float(max_closure),
                        "perclos": float(perclos),
                        "pitch": float(s_pitch),
                        "roll": float(s_roll),
                        "yaw": float(s_yaw),
                        "fatigue_score": float(current_score),
                        "fsm_state": current_fsm_state,
                        "alert_event": alert_event,
                        "yawn_count": yawn_tracker.yawn_count,
                    }
                    session_storage.log(record)
                    last_log_time = now_wall

                hud_metrics = {
                    "ear": s_ear,
                    "mar": s_mar,
                    "perclos": perclos,
                    "pitch": s_pitch,
                    "roll": s_roll,
                    "yaw": s_yaw,
                    "blink_count": blink_tracker.blink_count,
                    "yawn_count": yawn_tracker.yawn_count,
                }
                display.draw_landmarks(frame, face_frame.pixel_landmarks)
                display.draw_hud(frame, current_fsm_state, current_score, metrics=hud_metrics)
                display.show(frame)

                if display.check_quit():
                    self.logger.info("Quit command received.")
                    break

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            cam.stop()
            detector.close()
            display.close()
            alert_mgr.cleanup()
            session_storage.close()

            # Compile session report
            logs = session_storage.get_logs()
            if logs:
                self.logger.info("Compiling session PDF report...")
                report_path = self.pdf_generator.generate_report(driver_name, logs, session_id)
                if report_path:
                    self.logger.info("Report created: %s", report_path)
            self.logger.info("Session ended cleanly.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="VisionGuard",
        description="VisionGuard (APDMS) — Adaptive Personalized Driver Monitoring System",
    )
    parser.add_argument("--driver", type=str, default=None, help="Driver profile name.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration for this driver.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    driver_name = args.driver
    if not driver_name:
        try:
            driver_name = input("Enter driver name: ").strip()
        except EOFError:
            driver_name = "DefaultDriver"

    if not driver_name:
        driver_name = "DefaultDriver"

    app = VisionGuardApp(config_path=args.config)
    app.run(
        driver_name=driver_name,
        camera_index=args.camera,
        force_recalibrate=args.recalibrate,
    )


if __name__ == "__main__":
    main()
