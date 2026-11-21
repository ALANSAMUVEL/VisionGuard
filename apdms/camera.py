"""Threaded webcam capture for APDMS.

Runs a background daemon thread that reads frames as fast as the camera
delivers them. The main loop accesses the most recent frame via read()
without blocking on camera I/O.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


class Camera:
    """Thread-safe, non-blocking webcam capture.

    Usage::
        cam = Camera(cfg).start()
        ok, frame = cam.read()
        cam.stop()
    """

    def __init__(
        self,
        cfg_or_index: Union[Dict[str, Any], int] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend_windows: bool = True,
    ) -> None:
        if isinstance(cfg_or_index, dict):
            self._index: int = int(cfg_or_index.get("index", 0))
            self._w: int = int(cfg_or_index.get("width", width))
            self._h: int = int(cfg_or_index.get("height", height))
            self._fps: int = int(cfg_or_index.get("fps", fps))
            backend_win: bool = bool(cfg_or_index.get("backend_windows", backend_windows))
        else:
            self._index = int(cfg_or_index)
            self._w = width
            self._h = height
            self._fps = fps
            backend_win = backend_windows

        backend = (
            cv2.CAP_DSHOW
            if (sys.platform == "win32" and backend_win)
            else cv2.CAP_ANY
        )
        self._cap = cv2.VideoCapture(self._index, backend)

        if not self._cap.isOpened():
            raise IOError(
                f"Cannot open camera at index {self._index}. "
                "Verify camera connection and permissions."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

        grabbed, frame = self._cap.read()
        self._grabbed: bool = grabbed
        self._frame: np.ndarray = (
            frame if grabbed else np.zeros((self._h, self._w, 3), dtype=np.uint8)
        )

    def start(self) -> "Camera":
        """Start the background reading thread."""
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Tuple[bool, np.ndarray]:
        """Return (success, frame) with a copy of the most recent frame."""
        with self._lock:
            if not self._grabbed or self._frame is None:
                return False, np.zeros((self._h, self._w, 3), dtype=np.uint8)
            return True, self._frame.copy()

    def is_opened(self) -> bool:
        """Check whether the camera is open and active."""
        return self._cap.isOpened() and not self._stopped

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._w

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._h

    def stop(self) -> None:
        """Signal reader thread to stop, wait for termination, and release device."""
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap.isOpened():
            self._cap.release()

    def _update(self) -> None:
        """Background thread target: continuously read frames into memory."""
        while not self._stopped:
            grabbed, frame = self._cap.read()
            with self._lock:
                self._grabbed = grabbed
                if grabbed and frame is not None:
                    self._frame = frame
            time.sleep(0.005)
