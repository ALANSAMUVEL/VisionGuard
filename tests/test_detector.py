import numpy as np
from apdms.detector import FaceDetector, FaceFrame


def test_face_detector_empty_black_frame():
    detector = FaceDetector({})
    # Solid black 640x480 frame: no face should be detected
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_frame = detector.process(black_frame)

    assert isinstance(face_frame, FaceFrame)
    assert not face_frame.detected
    assert face_frame.landmarks is None
    assert face_frame.confidence == 0.0
    assert face_frame.diagnostic == "NO FACE"
    detector.close()
