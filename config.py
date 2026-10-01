"""
detector.py — Hand Detection Module (MediaPipe 1.x Tasks API)

Wraps MediaPipe HandLandmarker to provide a clean interface for hand landmark
detection. Downloads the required hand_landmarker.task model file on first run
(~3.5 MB, cached locally in models/).

Handles webcam lifecycle (open, read, release) and detection errors gracefully.

MediaPipe 1.x changed from mp.solutions.hands → mp.tasks.vision.HandLandmarker.
This module abstracts that change so the rest of the codebase is unaffected.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model download (one-time, cached)
# ---------------------------------------------------------------------------

_MODEL_DIR  = Path("models")
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model() -> str:
    """Download the hand landmarker model if not already present."""
    if _MODEL_PATH.exists():
        logger.debug("Model found at %s.", _MODEL_PATH)
        return str(_MODEL_PATH)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading hand landmarker model (~3.5 MB)…")
    print("[detector] Downloading MediaPipe hand_landmarker.task (~3.5 MB)…", flush=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model saved to %s.", _MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the MediaPipe hand landmarker model.\n"
            f"URL: {_MODEL_URL}\nError: {exc}\n"
            "Please check your internet connection and try again."
        ) from exc
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Hand-connections constant (for visualizer)
# ---------------------------------------------------------------------------

# List of (start_idx, end_idx) tuples — same structure as old mp.solutions
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (conn.start, conn.end)
    for conn in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    """Normalised (0-1) (x, y, z) landmark coordinates."""
    x: float
    y: float
    z: float

    def pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalised coordinates to absolute pixel coordinates."""
        return int(self.x * width), int(self.y * height)


@dataclass
class HandResult:
    """
    All data returned by the detector for a single detected hand per frame.

    Attributes
    ----------
    landmarks   : list of 21 Landmark objects (MediaPipe hand model)
    handedness  : 'Left' or 'Right'
    annotated_frame : the original frame (drawing is done by Visualizer)
    """
    landmarks: List[Landmark]
    handedness: str
    annotated_frame: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# HandDetector
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Detects hands and extracts 21 landmarks using MediaPipe Tasks API (1.x).

    Usage
    -----
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    results = detector.detect(frame)          # List[HandResult]
    detector.close()
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        model_path = _ensure_model()

        BaseOptions    = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,       # VIDEO mode for webcam
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms: int = 0

        logger.info(
            "HandDetector initialised (max_hands=%d, det=%.2f, track=%.2f) [MediaPipe %s]",
            max_num_hands, min_detection_confidence, min_tracking_confidence,
            mp.__version__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Run hand detection on a single BGR frame.

        Parameters
        ----------
        frame : BGR image (numpy array) from cv2.VideoCapture.read()

        Returns
        -------
        List[HandResult] — empty list if no hands are detected.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame; skipping.")
            return []

        # MediaPipe 1.x uses mp.Image (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires monotonically increasing timestamps
        self._frame_timestamp_ms += 33   # ~30 fps
        mp_result = self._detector.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not mp_result.hand_landmarks:
            return []

        results: List[HandResult] = []
        for hand_landmarks, handedness_list in zip(
            mp_result.hand_landmarks,
            mp_result.handedness,
        ):
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
            handedness = handedness_list[0].display_name   # 'Left' or 'Right'

            results.append(
                HandResult(
                    landmarks=landmarks,
                    handedness=handedness,
                    annotated_frame=frame,
                )
            )

        logger.debug("detect() found %d hand(s).", len(results))
        return results

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
        logger.info("HandDetector closed.")


# ---------------------------------------------------------------------------
# Webcam Manager (context manager)
# ---------------------------------------------------------------------------

class WebcamCapture:
    """
    Context manager that opens a webcam and raises a clear RuntimeError if
    the camera is missing or busy.

    Example
    -------
    with WebcamCapture(index=0) as cap:
        ret, frame = cap.read()
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: int = config.FPS_TARGET,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> cv2.VideoCapture:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Ensure the webcam is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        logger.info(
            "Webcam opened at index %d (%dx%d @ %d fps target).",
            self._index, self._width, self._height, self._fps,
        )
        return self._cap

    def __exit__(self, *_) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Webcam released.")
