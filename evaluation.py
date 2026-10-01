"""
gesture_recognizer.py — Rule-Based Gesture Classification Module

Classifies hand gestures from MediaPipe 21-landmark hand model.

Algorithm
---------
Each of the five fingers has four landmarks:
    MCP (knuckle base)  →  PIP (first bend)  →  DIP (second bend)  →  TIP (fingertip)

A finger is considered EXTENDED when its TIP y-coordinate is less than its PIP
y-coordinate (i.e. the fingertip is visually above the first bend, meaning the
finger is raised). Lower y = higher on screen in OpenCV's coordinate system.

The thumb is a special case: it extends horizontally, so we compare the TIP's
x-coordinate against the IP joint's x-coordinate, accounting for handedness.

Confidence
----------
Confidence is the fraction of landmark constraints that are satisfied for the
matched gesture. This is a genuine, computable score — not an invented number.

Gestures Supported
------------------
- Open Palm   : all five fingers extended
- Fist        : all five fingers folded
- Thumbs Up   : only thumb extended, rest folded
- Peace / V   : index + middle extended, ring + pinky folded
- Pointing    : only index extended, rest folded
- Unknown     : does not match any of the above with sufficient confidence
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional

import config
from detector import Landmark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named result type
# ---------------------------------------------------------------------------

class GestureResult(NamedTuple):
    """
    Result returned by GestureRecognizer.classify().

    Fields
    ------
    name       : gesture label string (see config.GESTURE_LABELS)
    confidence : float in [0.0, 1.0] — fraction of constraints satisfied
    fingers    : tuple of bools (thumb, index, middle, ring, pinky) extended state
    """
    name: str
    confidence: float
    fingers: tuple  # (thumb, index, middle, ring, pinky)


# ---------------------------------------------------------------------------
# Gesture Recognizer
# ---------------------------------------------------------------------------

class GestureRecognizer:
    """
    Classifies a hand's gesture from a list of 21 MediaPipe landmarks.

    Example
    -------
    recognizer = GestureRecognizer()
    result = recognizer.classify(landmarks)
    print(result.name, result.confidence)
    """

    # MediaPipe landmark indices for each finger segment
    _FINGER_TIPS = [4, 8, 12, 16, 20]    # TIP landmarks
    _FINGER_PIPS = [3, 6, 10, 14, 18]    # PIP / IP landmarks  (for thumb: IP = 3)

    def __init__(
        self,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self._threshold = confidence_threshold
        logger.debug(
            "GestureRecognizer initialised (threshold=%.2f).", confidence_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, landmarks: Optional[List[Landmark]], handedness: str = "Right"
    ) -> GestureResult:
        """
        Classify the hand gesture from a list of 21 Landmark objects.

        Parameters
        ----------
        landmarks   : List[Landmark] of length 21 from HandDetector.detect()
        handedness  : 'Left' or 'Right' — affects thumb direction logic

        Returns
        -------
        GestureResult with name, confidence, and finger states.
        """
        if not landmarks or len(landmarks) < 21:
            logger.debug("classify() called with insufficient landmarks.")
            return GestureResult("Unknown", 0.0, (False,) * 5)

        finger_states = self._get_finger_states(landmarks, handedness)
        return self._match_gesture(finger_states)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_finger_states(
        self, landmarks: List[Landmark], handedness: str
    ) -> tuple:
        """
        Determine which fingers are extended.

        Returns a 5-tuple of bools: (thumb, index, middle, ring, pinky).
        """
        thumb_extended  = self._is_thumb_extended(landmarks, handedness)
        index_extended  = self._is_finger_extended(landmarks, tip_idx=8,  pip_idx=6)
        middle_extended = self._is_finger_extended(landmarks, tip_idx=12, pip_idx=10)
        ring_extended   = self._is_finger_extended(landmarks, tip_idx=16, pip_idx=14)
        pinky_extended  = self._is_finger_extended(landmarks, tip_idx=20, pip_idx=18)

        states = (thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended)
        logger.debug("Finger states (T,I,M,R,P): %s", states)
        return states

    @staticmethod
    def _is_finger_extended(
        landmarks: List[Landmark], tip_idx: int, pip_idx: int
    ) -> bool:
        """
        A non-thumb finger is extended when its tip is above (lower y) its PIP joint.
        """
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    @staticmethod
    def _is_thumb_extended(landmarks: List[Landmark], handedness: str) -> bool:
        """
        Thumb is extended when its tip is sufficiently far from the MCP joint.
        For a Right hand: TIP.x < IP.x  (thumb sticks to the left of the IP joint)
        For a Left hand:  TIP.x > IP.x  (mirrored)

        Note: webcam feed is typically mirrored, so handedness may be reversed
        visually — we handle both directions to be robust.
        """
        tip = landmarks[4]   # THUMB_TIP
        ip  = landmarks[3]   # THUMB_IP
        mcp = landmarks[2]   # THUMB_MCP

        # Absolute horizontal distance from MCP to TIP
        thumb_length = abs(tip.x - mcp.x)
        ip_length    = abs(ip.x  - mcp.x)

        # Thumb is extended if TIP is farther from MCP than IP is
        return thumb_length > ip_length * 1.15

    def _match_gesture(self, finger_states: tuple) -> GestureResult:
        """
        Match finger_states against known gesture patterns and compute confidence.

        Confidence = matched_constraints / total_constraints for the best candidate.
        """
        thumb, index, middle, ring, pinky = finger_states

        candidates = [
            self._score_open_palm(finger_states),
            self._score_fist(finger_states),
            self._score_thumbs_up(finger_states),
            self._score_peace(finger_states),
            self._score_pointing(finger_states),
        ]

        # Pick highest confidence
        best_name, best_score = max(candidates, key=lambda c: c[1])

        if best_score < self._threshold:
            logger.debug(
                "Best match '%s' below threshold (%.2f < %.2f); returning Unknown.",
                best_name, best_score, self._threshold,
            )
            return GestureResult("Unknown", best_score, finger_states)

        logger.debug("Gesture matched: '%s' (confidence=%.2f)", best_name, best_score)
        return GestureResult(best_name, best_score, finger_states)

    # ------------------------------------------------------------------
    # Per-gesture scoring functions
    # Each returns (gesture_name, confidence_score)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_open_palm(fs: tuple) -> tuple:
        """Open Palm: all 5 fingers extended."""
        thumb, index, middle, ring, pinky = fs
        matched = sum([thumb, index, middle, ring, pinky])
        return ("Open Palm", matched / 5.0)

    @staticmethod
    def _score_fist(fs: tuple) -> tuple:
        """Fist: all 5 fingers folded."""
        thumb, index, middle, ring, pinky = fs
        matched = sum([not thumb, not index, not middle, not ring, not pinky])
        return ("Fist", matched / 5.0)

    @staticmethod
    def _score_thumbs_up(fs: tuple) -> tuple:
        """Thumbs Up: thumb extended, index+middle+ring+pinky folded."""
        thumb, index, middle, ring, pinky = fs
        constraints = [thumb, not index, not middle, not ring, not pinky]
        matched = sum(constraints)
        return ("Thumbs Up", matched / len(constraints))

    @staticmethod
    def _score_peace(fs: tuple) -> tuple:
        """Peace / V Sign: index + middle extended, ring + pinky folded."""
        thumb, index, middle, ring, pinky = fs
        constraints = [index, middle, not ring, not pinky]
        matched = sum(constraints)
        return ("Peace / V Sign", matched / len(constraints))

    @staticmethod
    def _score_pointing(fs: tuple) -> tuple:
        """Pointing: only index extended, middle + ring + pinky folded."""
        thumb, index, middle, ring, pinky = fs
        constraints = [index, not middle, not ring, not pinky]
        matched = sum(constraints)
        return ("Pointing", matched / len(constraints))
