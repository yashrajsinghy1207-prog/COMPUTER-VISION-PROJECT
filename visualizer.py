"""
visualizer.py — Real-Time Visualization Module

Handles all OpenCV drawing: landmark overlay, gesture labels, confidence bar,
FPS counter, no-hand warning, and the info panel (keyboard shortcut legend).
Designed so that no drawing logic leaks into app.py or detector.py.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from detector import HAND_CONNECTIONS, Landmark
from gesture_recognizer import GestureResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — draw a semi-transparent rectangle
# ---------------------------------------------------------------------------

def _filled_rect_alpha(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.55,
) -> None:
    """Draw a filled rectangle with alpha blending on *frame* in-place."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """
    All drawing operations for the hand gesture application.

    Every method modifies *frame* in-place and returns it for chaining.

    Usage
    -----
    viz = Visualizer()
    frame = viz.draw_landmarks(frame, landmarks)
    frame = viz.draw_gesture_label(frame, result)
    frame = viz.draw_fps(frame, fps)
    frame = viz.draw_info_panel(frame)
    """

    def __init__(self) -> None:
        self._font = cv2.FONT_HERSHEY_DUPLEX
        logger.debug("Visualizer initialised.")

    # ------------------------------------------------------------------
    # Landmarks
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List[Landmark],
        show_indices: bool = False,
    ) -> np.ndarray:
        """
        Draw 21 hand landmark dots and connections on *frame*.

        Parameters
        ----------
        frame        : BGR frame (modified in-place)
        landmarks    : 21 Landmark objects from HandDetector
        show_indices : if True, draw landmark index numbers (debug mode)
        """
        h, w = frame.shape[:2]

        # Draw connections first (below dots)
        pixel_pts = [lm.pixel(w, h) for lm in landmarks]
        for (start_idx, end_idx) in HAND_CONNECTIONS:
            cv2.line(
                frame,
                pixel_pts[start_idx],
                pixel_pts[end_idx],
                config.COLOR_CONNECTION,
                config.CONNECTION_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        # Draw dots
        for i, pt in enumerate(pixel_pts):
            cv2.circle(frame, pt, config.LANDMARK_RADIUS, config.COLOR_PRIMARY, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, config.LANDMARK_RADIUS + 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            if show_indices:
                cv2.putText(
                    frame, str(i), (pt[0] + 5, pt[1] - 5),
                    self._font, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
                )

        return frame

    # ------------------------------------------------------------------
    # Gesture Label & Confidence Bar
    # ------------------------------------------------------------------

    def draw_gesture_label(
        self, frame: np.ndarray, result: GestureResult
    ) -> np.ndarray:
        """
        Overlay gesture name + animated confidence bar in the top-left corner.
        """
        h, w = frame.shape[:2]
        pad = config.LABEL_PADDING
        x0, y0 = 20, 20

        # ---- Gesture name ---------------------------------------------------
        label = result.name
        (lw, lh), _ = cv2.getTextSize(
            label, self._font, config.FONT_SCALE_LARGE, config.FONT_THICKNESS
        )
        _filled_rect_alpha(
            frame,
            (x0 - pad, y0 - pad),
            (x0 + lw + pad, y0 + lh + pad),
            config.COLOR_TEXT_BG,
        )
        cv2.putText(
            frame, label,
            (x0, y0 + lh),
            self._font, config.FONT_SCALE_LARGE,
            config.COLOR_TEXT_FG, config.FONT_THICKNESS, cv2.LINE_AA,
        )

        # ---- Confidence bar -------------------------------------------------
        bar_y      = y0 + lh + pad * 2 + 10
        bar_width  = 220
        bar_height = 14
        conf_width = int(bar_width * result.confidence)
        conf_pct   = int(result.confidence * 100)

        # Background track
        _filled_rect_alpha(
            frame,
            (x0, bar_y),
            (x0 + bar_width, bar_y + bar_height),
            (60, 60, 60),
            alpha=0.7,
        )
        # Filled portion
        if conf_width > 0:
            bar_color = self._confidence_color(result.confidence)
            cv2.rectangle(
                frame,
                (x0, bar_y),
                (x0 + conf_width, bar_y + bar_height),
                bar_color, -1,
            )
        # Border
        cv2.rectangle(
            frame,
            (x0, bar_y),
            (x0 + bar_width, bar_y + bar_height),
            (180, 180, 180), 1,
        )
        # Percentage text
        cv2.putText(
            frame, f"{conf_pct}%",
            (x0 + bar_width + 8, bar_y + bar_height - 1),
            self._font, config.FONT_SCALE_SMALL,
            config.COLOR_TEXT_FG, 1, cv2.LINE_AA,
        )
        # Label "Confidence"
        cv2.putText(
            frame, "Confidence",
            (x0, bar_y - 5),
            self._font, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        return frame

    @staticmethod
    def _confidence_color(conf: float) -> Tuple[int, int, int]:
        """Return a BGR color that goes red→yellow→green with confidence."""
        if conf >= 0.75:
            return (0, 220, 100)    # green
        elif conf >= 0.50:
            return (0, 190, 255)    # yellow-ish
        else:
            return (0, 80, 255)     # orange-red

    # ------------------------------------------------------------------
    # FPS Counter
    # ------------------------------------------------------------------

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS in the top-right corner."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(
            text, self._font, config.FONT_SCALE_SMALL, 1
        )
        x = w - tw - 20
        y = 30
        _filled_rect_alpha(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), 0.4)
        cv2.putText(
            frame, text, (x, y),
            self._font, config.FONT_SCALE_SMALL,
            config.COLOR_FPS, 1, cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # No Hand Warning
    # ------------------------------------------------------------------

    def draw_no_hand(self, frame: np.ndarray) -> np.ndarray:
        """Display a centered warning when no hand is detected."""
        h, w = frame.shape[:2]
        lines = ["No hand detected", "Show your hand to the camera"]
        y_start = h // 2 - 30
        for i, line in enumerate(lines):
            scale = config.FONT_SCALE_LARGE if i == 0 else config.FONT_SCALE_SMALL
            (tw, th), _ = cv2.getTextSize(line, self._font, scale, config.FONT_THICKNESS)
            x = (w - tw) // 2
            y = y_start + i * (th + 20)
            _filled_rect_alpha(
                frame,
                (x - 12, y - th - 8),
                (x + tw + 12, y + 8),
                (0, 0, 0),
                alpha=0.5,
            )
            cv2.putText(
                frame, line, (x, y),
                self._font, scale,
                config.COLOR_WARNING, config.FONT_THICKNESS, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Info Panel (bottom-right)
    # ------------------------------------------------------------------

    def draw_info_panel(
        self, frame: np.ndarray, debug_mode: bool = False
    ) -> np.ndarray:
        """Draw keyboard shortcut legend in the bottom-right corner."""
        h, w = frame.shape[:2]
        controls = [
            "Q  — Quit",
            "S  — Screenshot",
            "D  — Toggle debug",
        ]
        if debug_mode:
            controls.append("[DEBUG ON]")

        scale = 0.50
        line_height = 20
        panel_h = len(controls) * line_height + 16
        panel_w = 170
        x0 = w - panel_w - 16
        y0 = h - panel_h - 16

        _filled_rect_alpha(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), 0.5)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

        for i, ctrl in enumerate(controls):
            color = (0, 200, 255) if ctrl.startswith("[DEBUG") else (200, 200, 200)
            cv2.putText(
                frame, ctrl,
                (x0 + 8, y0 + 14 + i * line_height),
                self._font, scale, color, 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Handedness label (bottom-left)
    # ------------------------------------------------------------------

    def draw_handedness(self, frame: np.ndarray, handedness: str) -> np.ndarray:
        """Draw 'Left / Right' hand label at the bottom-left."""
        h, w = frame.shape[:2]
        text = f"Hand: {handedness}"
        cv2.putText(
            frame, text,
            (20, h - 20),
            self._font, config.FONT_SCALE_SMALL,
            (180, 180, 180), 1, cv2.LINE_AA,
        )
        return frame
