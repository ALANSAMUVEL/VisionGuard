"""APDMS — Adaptive Personalized Driver Monitoring System.

A modular, real-time computer vision system for driver drowsiness and distraction detection.
"""

from apdms.camera import Camera
from apdms.detector import FaceDetector, FaceFrame
from apdms.features import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    landmarks_to_pixels,
    BlinkTracker,
    YawnTracker,
    EyeMetrics,
    MouthMetrics,
    HeadPose,
)
from apdms.processing import EMA, SlidingWindow, ExponentialMovingAverage
from apdms.calibration import Calibrator
from apdms.thresholds import ThresholdGenerator
from apdms.scorer import FatigueScorer
from apdms.fsm import FiniteStateMachine, FatigueState
from apdms.alerts import AlertManager
from apdms.storage import ProfileStorage, SessionStorage
from apdms.display import Display
from apdms.reports import PDFReportGenerator, PDFGenerator

__version__ = "2.1.0"

__all__ = [
    "Camera",
    "FaceDetector",
    "FaceFrame",
    "calculate_ear",
    "calculate_mar",
    "estimate_head_pose",
    "landmarks_to_pixels",
    "BlinkTracker",
    "YawnTracker",
    "EyeMetrics",
    "MouthMetrics",
    "HeadPose",
    "EMA",
    "SlidingWindow",
    "ExponentialMovingAverage",
    "Calibrator",
    "ThresholdGenerator",
    "FatigueScorer",
    "FiniteStateMachine",
    "FatigueState",
    "AlertManager",
    "ProfileStorage",
    "SessionStorage",
    "Display",
    "PDFReportGenerator",
    "PDFGenerator",
]
