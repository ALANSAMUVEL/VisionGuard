# VisionGuard (APDMS)
### Adaptive Personalized Driver Monitoring System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()

**VisionGuard** is a real-time, deterministic computer vision system designed to monitor driver drowsiness, distraction, and fatigue. Unlike fragile black-box end-to-end deep learning models, VisionGuard operates on an **adaptive, rule-based signal processing pipeline** with **personalized driver calibration**, a temporal **Finite State Machine (FSM)** with hysteresis, and comprehensive **session reporting**.

---

## Architecture Overview

```
                                      Camera Feed (DirectShow / V4L2)
                                                     │
                                                     ▼
                                        MediaPipe Face Mesh (478 pts)
                                                     │
                                                     ▼
                                      Quality & Confidence Evaluation
                                      (Laplacian Blur / Lux / Spread)
                                                     │
                                                     ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │              Feature Extraction Engine                  │
                        ├──────────────────────────┬──────────────────────────────┤
                        │ Eye Aspect Ratio (EAR)   │ Head Pose (solvePnP + RQDec) │
                        │ Mouth Aspect Ratio (MAR) │ Blink & Yawn Event Trackers  │
                        └──────────────────────────┴──────────────────────────────┘
                                                     │
                                                     ▼
                                   Exponential Moving Average (EMA)
                                                     │
                                                     ▼
                                  Time-Indexed Sliding Window (4.0s)
                                    - PERCLOS (% closure over time)
                                    - Continuous closure duration
                                    - Rolling average blink duration
                                                     │
                                                     ▼
                                  Personalized Threshold Calibration
                                                     │
                                                     ▼
                                   Rule-Based Fatigue Scorer (0-100)
                                    - Continuous eye closure (40%)
                                    - Windowed PERCLOS (25%)
                                    - Prolonged blink duration (15%)
                                    - Confirmed yawning (10%)
                                    - Off-nominal head pose (10%)
                                                     │
                                                     ▼
                                     Temporal FSM with Hysteresis
                                   [ALERT] ──► [MILD] ──► [DROWSY] ──► [CRITICAL]
                                      ▲          ▲           ▲            │
                                      └──────────┴───────────┴────────────┘ (Recovery 5.0s)
                                                     │
                        ┌────────────────────────────┼────────────────────────────┐
                        ▼                            ▼                            ▼
                 HUD Overlay                  Audio Alerts                 Telemetry Storage
            (Landmarks, Badges,           (Pygame Multi-Tone,             (JSON Driver Profile,
             Real-Time Metrics)           Non-Blocking Audio)              Buffered CSV Logs)
                                                                                  │
                                                                                  ▼
                                                                           Analytical PDF
                                                                           Session Report
```

---

## Key Features

1. **Personalized 45-Second Baseline Calibration**:
   - Collects the driver's natural, rested facial characteristics on first launch.
   - Computes driver-specific baselines for EAR, MAR, neutral head orientation, and average blink duration.
   - Automatically derives personalized thresholds; subsequent sessions load the driver's profile instantly.
2. **Deterministic Mathematical Formulations**:
   - **Eye Aspect Ratio (EAR)**: Soukupov&aacute; &amp; &Ccaron;ech formula calculated across both left and right eyes.
   - **Mouth Aspect Ratio (MAR)**: Calculated using 8 inner-lip landmarks to reject mouth movements during normal conversation while reliably capturing true yawns.
   - **Head Pose Estimation**: 3D-to-2D correspondence solved via `cv2.solvePnP` and decomposed using `cv2.RQDecomp3x3` to avoid Euler angle flipping artefacts.
3. **Continuous Confidence Scoring**:
   - Soft confidence weighting replaces brittle binary thresholding for motion blur and varying ambient illumination.
4. **Temporal Finite State Machine (FSM)**:
   - Eliminates alert flickering via configurable state-transition timers (escalation timers vs. 5.0s recovery timer).
5. **Zero-Database File Storage**:
   - Driver calibration profiles: Lightweight JSON in `profiles/`.
   - Real-time session telemetry: High-speed buffered CSV in `sessions/`.
   - End-of-session reports: Automated publication-grade PDF in `reports/` with timeline charts and statistical summaries.

---

## Repository Structure

```
VisionGuard/
│
├── main.py              # Application orchestrator and CLI entry point
├── config.yaml          # Centralized configuration file
├── requirements.txt     # Python dependencies
├── run.bat              # Windows launcher script
├── README.md            # Project documentation
├── .gitignore           # Git ignore definitions
│
├── apdms/               # Core VisionGuard package
│   ├── __init__.py      # Package export
│   ├── camera.py        # Thread-safe, non-blocking camera capture
│   ├── detector.py      # FaceMesh wrapper and quality assessment
│   ├── features.py      # EAR, MAR, Head Pose, Blink & Yawn trackers
│   ├── processing.py    # Exponential Moving Average & Sliding Window
│   ├── calibration.py   # Baseline statistics accumulator
│   ├── thresholds.py    # Personalized threshold derivation
│   ├── scorer.py        # Multi-metric rule-based fatigue scorer
│   ├── fsm.py           # Temporal Finite State Machine
│   ├── alerts.py        # Pygame audio alert manager
│   ├── storage.py       # JSON profile and CSV session storage
│   ├── display.py       # OpenCV HUD overlay and visual banners
│   └── reports.py       # ReportLab PDF session report generator
│
├── assets/              # Audio alert tone assets
│   ├── mild_alert.wav
│   ├── drowsy_alert.wav
│   └── critical_alert.wav
│
└── tests/               # Unit and regression test suite (26 tests)
    ├── test_calibration.py
    ├── test_detector.py
    ├── test_fatigue_scorer.py
    ├── test_features.py
    ├── test_fsm.py
    ├── test_reports.py
    ├── test_sliding_window.py
    ├── test_storage.py
    └── test_thresholds.py
```

---

## Installation

### Prerequisites
- Python 3.11 or 3.12
- Webcam / video capture device

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ALANSAMUVEL/VisionGuard.git
   cd VisionGuard
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### 1. Interactive Launch
Prompts for the driver profile name on startup:
```bash
python main.py
```
Or on Windows:
```bash
run.bat
```

### 2. Specify Driver Name Directly
```bash
python main.py --driver user1
```

### 3. Force Recalibration
```bash
python main.py --driver user1 --recalibrate
```

### 4. Select Camera Device Index
```bash
python main.py --camera 1
```

### 5. Exit Monitoring & Generate Report
Press **`q`** in the OpenCV display window at any time. VisionGuard will flush session telemetry and automatically compile a detailed analytical session PDF in `reports/`.

---

## Running the Test Suite

Execute the full pytest test suite:
```bash
pytest tests/ -v
```

All 26 test cases validate:
- Dynamic threshold generation
- Exponential Moving Average calculations
- Sliding window event transitions and PERCLOS
- FSM temporal hysteresis and recovery
- Weighted fatigue score combinations
- Soft confidence quality gating
- Head pose and landmark feature extraction
- File-based JSON/CSV storage persistence
- Publication-quality PDF generation

---

## Configuration Reference (`config.yaml`)

| Section | Parameter | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `camera` | `index`, `fps`, `width`, `height` | `0`, `30`, `640`, `480` | Video capture specifications |
| `processing` | `ema_alpha` | `0.15` | Smoothing responsiveness (0.0 to 1.0) |
| `processing` | `window_seconds` | `4.0` | Rolling temporal analysis window length |
| `calibration` | `duration_seconds` | `45.0` | Initial baseline calibration duration |
| `fatigue_weights`| `eye_closure` | `0.40` | Weight for continuous eye closure |
| `fatigue_weights`| `perclos` | `0.25` | Weight for windowed PERCLOS |
| `fatigue_weights`| `blink_behaviour` | `0.15` | Weight for prolonged average blink duration |
| `fatigue_weights`| `yawning` | `0.10` | Weight for verified yawn events |
| `fatigue_weights`| `head_pose` | `0.10` | Weight for head drooping / averted gaze |
| `fsm.thresholds` | `mild`, `drowsy`, `critical` | `35`, `60`, `80` | Fatigue score transition triggers |
| `fsm.timers` | `to_mild`, `to_drowsy`, `to_critical` | `3.0`, `3.0`, `2.0` | Minimum sustained time before state change |
| `fsm.timers` | `recovery` | `5.0` | Sustained low-score time to return to ALERT |

---
