"""
tests/test_gestures.py — Unit Tests for GestureRecognizer

Tests run entirely from synthetic landmark data — no webcam required.
Run with:  pytest tests/ -v

Each test constructs a minimal set of 21 Landmark objects that represent
a known hand pose, then asserts that GestureRecognizer classifies it correctly.
"""

from __future__ import annotations

import sys
import os

# Allow importing parent-level modules without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from detector import Landmark
from gesture_recognizer import GestureRecognizer, GestureResult


# ---------------------------------------------------------------------------
# Shared Fixture & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def recognizer() -> GestureRecognizer:
    return GestureRecognizer(confidence_threshold=0.50)


def _lm(x: float, y: float) -> Landmark:
    return Landmark(x=x, y=y, z=0.0)


def _base_hand() -> list:
    """
    21-landmark list where all fingers are in a neutral (folded) position.
    Thumb TIP is close to MCP → NOT extended.
    All non-thumb TIPs are BELOW their PIPs → NOT extended.
    """
    lm = [_lm(0.5, 0.90)] * 21  # wrist & defaults

    # Thumb — NOT extended (TIP barely past MCP)
    lm[1] = _lm(0.46, 0.85)  # CMC
    lm[2] = _lm(0.44, 0.82)  # MCP
    lm[3] = _lm(0.42, 0.80)  # IP
    lm[4] = _lm(0.43, 0.78)  # TIP  (barely past MCP → folded)

    # Index — folded (TIP.y > PIP.y)
    lm[5] = _lm(0.50, 0.75); lm[6] = _lm(0.50, 0.68)
    lm[7] = _lm(0.50, 0.74); lm[8] = _lm(0.50, 0.80)

    # Middle — folded
    lm[9]  = _lm(0.56, 0.75); lm[10] = _lm(0.56, 0.68)
    lm[11] = _lm(0.56, 0.74); lm[12] = _lm(0.56, 0.80)

    # Ring — folded
    lm[13] = _lm(0.62, 0.75); lm[14] = _lm(0.62, 0.68)
    lm[15] = _lm(0.62, 0.74); lm[16] = _lm(0.62, 0.80)

    # Pinky — folded
    lm[17] = _lm(0.68, 0.77); lm[18] = _lm(0.68, 0.70)
    lm[19] = _lm(0.68, 0.76); lm[20] = _lm(0.68, 0.82)

    return lm


def _extend_finger(lm: list, tip: int, pip: int) -> list:
    """Raise a finger by placing TIP above (lower y) PIP."""
    lm = list(lm)
    # Set PIP mid-way, TIP well above PIP
    lm[pip] = _lm(lm[pip].x, 0.60)
    lm[tip] = _lm(lm[tip].x, 0.30)
    return lm


def _extend_thumb(lm: list) -> list:
    """Make thumb extended: TIP far horizontally from MCP."""
    lm = list(lm)
    lm[2] = _lm(0.38, 0.82)  # MCP
    lm[3] = _lm(0.33, 0.80)  # IP
    lm[4] = _lm(0.27, 0.77)  # TIP  — far from MCP
    return lm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFist:
    def test_basic_fist(self, recognizer):
        """All fingers folded → Fist."""
        lm = _base_hand()
        result = recognizer.classify(lm, "Right")
        assert result.name == "Fist", f"Expected Fist, got {result.name}"

    def test_fist_confidence_high(self, recognizer):
        """Fist confidence should be ≥ 80%."""
        result = recognizer.classify(_base_hand(), "Right")
        assert result.confidence >= 0.80, f"Confidence too low: {result.confidence:.2%}"


class TestOpenPalm:
    def test_all_fingers_extended(self, recognizer):
        """All five fingers extended → Open Palm."""
        lm = _base_hand()
        lm = _extend_thumb(lm)
        lm = _extend_finger(lm, tip=8,  pip=6)
        lm = _extend_finger(lm, tip=12, pip=10)
        lm = _extend_finger(lm, tip=16, pip=14)
        lm = _extend_finger(lm, tip=20, pip=18)
        result = recognizer.classify(lm, "Right")
        assert result.name == "Open Palm", f"Expected Open Palm, got {result.name}"

    def test_open_palm_confidence_perfect(self, recognizer):
        """Perfect open palm should score 1.0 confidence."""
        lm = _base_hand()
        lm = _extend_thumb(lm)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            lm = _extend_finger(lm, tip, pip)
        result = recognizer.classify(lm, "Right")
        assert result.confidence == pytest.approx(1.0), (
            f"Expected 1.0 confidence for perfect open palm, got {result.confidence}"
        )


class TestThumbsUp:
    def test_only_thumb_extended(self, recognizer):
        """Only thumb extended → Thumbs Up."""
        lm = _extend_thumb(_base_hand())
        result = recognizer.classify(lm, "Right")
        assert result.name == "Thumbs Up", f"Expected Thumbs Up, got {result.name}"

    def test_thumbs_up_confidence(self, recognizer):
        """Thumbs Up should have ≥ 0.80 confidence."""
        lm = _extend_thumb(_base_hand())
        result = recognizer.classify(lm, "Right")
        assert result.confidence >= 0.80


class TestPeace:
    def test_index_and_middle_extended(self, recognizer):
        """Index + middle extended, ring + pinky folded → Peace / V Sign."""
        lm = _base_hand()
        lm = _extend_finger(lm, tip=8,  pip=6)
        lm = _extend_finger(lm, tip=12, pip=10)
        result = recognizer.classify(lm, "Right")
        assert result.name == "Peace / V Sign", f"Expected Peace / V Sign, got {result.name}"

    def test_peace_full_confidence(self, recognizer):
        """4/4 peace constraints satisfied → confidence = 1.0."""
        lm = _base_hand()
        lm = _extend_finger(lm, tip=8,  pip=6)
        lm = _extend_finger(lm, tip=12, pip=10)
        result = recognizer.classify(lm, "Right")
        assert result.confidence == pytest.approx(1.0)


class TestPointing:
    def test_only_index_extended(self, recognizer):
        """Only index extended → Pointing."""
        lm = _base_hand()
        lm = _extend_finger(lm, tip=8, pip=6)
        result = recognizer.classify(lm, "Right")
        assert result.name == "Pointing", f"Expected Pointing, got {result.name}"

    def test_pointing_confidence(self, recognizer):
        """Pointing (3/4 constraints) should be ≥ 0.75."""
        lm = _base_hand()
        lm = _extend_finger(lm, tip=8, pip=6)
        result = recognizer.classify(lm, "Right")
        assert result.confidence >= 0.75


class TestEdgeCases:
    def test_empty_landmarks(self, recognizer):
        """Empty landmark list → Unknown with confidence 0."""
        result = recognizer.classify([], "Right")
        assert result.name == "Unknown"
        assert result.confidence == 0.0

    def test_none_landmarks(self, recognizer):
        """None landmark list → Unknown with confidence 0."""
        result = recognizer.classify(None, "Right")
        assert result.name == "Unknown"
        assert result.confidence == 0.0

    def test_incomplete_landmarks(self, recognizer):
        """Fewer than 21 landmarks → Unknown."""
        result = recognizer.classify([_lm(0.5, 0.5)] * 10, "Right")
        assert result.name == "Unknown"

    def test_gesture_result_is_named_tuple(self, recognizer):
        """GestureResult should be a NamedTuple with expected fields."""
        result = recognizer.classify(_base_hand(), "Right")
        assert hasattr(result, "name")
        assert hasattr(result, "confidence")
        assert hasattr(result, "fingers")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_finger_states_are_tuple_of_bools(self, recognizer):
        """fingers field should be a 5-tuple of booleans."""
        result = recognizer.classify(_base_hand(), "Right")
        assert len(result.fingers) == 5
        assert all(isinstance(f, bool) for f in result.fingers)

    def test_left_hand_thumbs_up(self, recognizer):
        """Thumbs Up should also work for Left handedness."""
        lm = _extend_thumb(_base_hand())
        result = recognizer.classify(lm, "Left")
        assert result.name == "Thumbs Up"
