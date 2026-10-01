"""
utils.py — Utility helpers: logging, FPS counter, screenshot saving.

Kept separate from business logic so each module can import independently.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config


# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "gesture_app") -> logging.Logger:
    """
    Configure and return a logger that writes to both console and a log file.

    Log level is read from config.LOG_LEVEL so it can be changed without
    touching source code.
    """
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not open log file '%s': %s", config.LOG_FILE, exc)

    return logger


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """
    Computes a rolling-average FPS over the last N frames.

    Usage
    -----
    fps_counter = FPSCounter()
    while True:
        fps_counter.tick()
        fps = fps_counter.fps
    """

    def __init__(self, window: int = config.FPS_SMOOTHING_WINDOW) -> None:
        self._window = window
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record the current timestamp (call once per frame)."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling-average FPS. Returns 0 if fewer than 2 frames recorded."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


# ---------------------------------------------------------------------------
# Screenshot Saving
# ---------------------------------------------------------------------------

def save_screenshot(frame: np.ndarray, directory: str = config.SCREENSHOT_DIR) -> str:
    """
    Save a frame as a PNG screenshot with a timestamp-based filename.

    Parameters
    ----------
    frame     : BGR numpy array to save
    directory : output directory (created if it doesn't exist)

    Returns
    -------
    Full path of the saved file, or empty string on failure.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(directory, f"{config.SCREENSHOT_PREFIX}{timestamp}.png")
    try:
        cv2.imwrite(filename, frame)
        logging.getLogger("gesture_app").info("Screenshot saved: %s", filename)
        return filename
    except Exception as exc:  # pylint: disable=broad-except
        logging.getLogger("gesture_app").error("Failed to save screenshot: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))
