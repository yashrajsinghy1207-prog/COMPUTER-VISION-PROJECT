"""
evaluation.py — Evaluation & Manual Testing Script

Provides two modes:
  1. --synthetic : Tests all gesture classifiers on hand-crafted landmark data
                   (no webcam required; can be run by an examiner on any machine).
  2. --webcam    : Runs a timed webcam session and prints a gesture frequency
                   distribution table.

Usage
-----
  python evaluation.py --synthetic       # no camera needed
  python evaluation.py --webcam --seconds 15
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from typing import List

import cv2

from detector import Landmark
from gesture_recognizer import GestureRecognizer, GestureResult


# ---------------------------------------------------------------------------
# Synthetic Landmark Builders
# ---------------------------------------------------------------------------

def _flat_landmark(x: float, y: float, z: float = 0.0) -> Landmark:
    return Landmark(x=x, y=y, z=z)


def _build_landmarks_open_palm() -> List[Landmark]:
    """
    Construct 21 synthetic landmarks representing an Open Palm.
    All four non-thumb fingers have TIP.y < PIP.y (fingers raised).
    Thumb TIP is far from MCP horizontally.
    """
    lm = [_flat_landmark(0.5, 0.9)] * 21  # default: wrist area

    # Thumb — extended (tip far left of MCP)
    lm[1] = _flat_landmark(0.42, 0.85)  # CMC
    lm[2] = _flat_landmark(0.38, 0.82)  # MCP
    lm[3] = _flat_landmark(0.33, 0.80)  # IP
    lm[4] = _flat_landmark(0.27, 0.77)  # TIP  (far from MCP → extended)

    # Index — extended (TIP.y < PIP.y)
    lm[5] = _flat_landmark(0.50, 0.75)  # MCP
    lm[6] = _flat_landmark(0.50, 0.60)  # PIP
    lm[7] = _flat_landmark(0.50, 0.45)  # DIP
    lm[8] = _flat_landmark(0.50, 0.30)  # TIP

    # Middle — extended
    lm[9]  = _flat_landmark(0.56, 0.75)
    lm[10] = _flat_landmark(0.56, 0.58)
    lm[11] = _flat_landmark(0.56, 0.42)
    lm[12] = _flat_landmark(0.56, 0.26)

    # Ring — extended
    lm[13] = _flat_landmark(0.62, 0.75)
    lm[14] = _flat_landmark(0.62, 0.60)
    lm[15] = _flat_landmark(0.62, 0.46)
    lm[16] = _flat_landmark(0.62, 0.32)

    # Pinky — extended
    lm[17] = _flat_landmark(0.68, 0.77)
    lm[18] = _flat_landmark(0.68, 0.64)
    lm[19] = _flat_landmark(0.68, 0.52)
    lm[20] = _flat_landmark(0.68, 0.40)

    return lm


def _build_landmarks_fist() -> List[Landmark]:
    """All fingers folded: TIP.y > PIP.y for all non-thumb fingers."""
    lm = [_flat_landmark(0.5, 0.9)] * 21

    # Thumb — folded (TIP close to MCP)
    lm[1] = _flat_landmark(0.46, 0.85)
    lm[2] = _flat_landmark(0.44, 0.82)
    lm[3] = _flat_landmark(0.42, 0.80)
    lm[4] = _flat_landmark(0.43, 0.78)  # TIP barely past MCP → not extended

    # Index — folded (TIP.y > PIP.y)
    lm[5] = _flat_landmark(0.50, 0.75)
    lm[6] = _flat_landmark(0.50, 0.68)  # PIP
    lm[7] = _flat_landmark(0.50, 0.74)
    lm[8] = _flat_landmark(0.50, 0.80)  # TIP BELOW PIP → folded

    # Middle — folded
    lm[9]  = _flat_landmark(0.56, 0.75)
    lm[10] = _flat_landmark(0.56, 0.68)
    lm[11] = _flat_landmark(0.56, 0.74)
    lm[12] = _flat_landmark(0.56, 0.80)

    # Ring — folded
    lm[13] = _flat_landmark(0.62, 0.75)
    lm[14] = _flat_landmark(0.62, 0.68)
    lm[15] = _flat_landmark(0.62, 0.74)
    lm[16] = _flat_landmark(0.62, 0.80)

    # Pinky — folded
    lm[17] = _flat_landmark(0.68, 0.77)
    lm[18] = _flat_landmark(0.68, 0.70)
    lm[19] = _flat_landmark(0.68, 0.76)
    lm[20] = _flat_landmark(0.68, 0.82)

    return lm


def _build_landmarks_thumbs_up() -> List[Landmark]:
    """Thumb extended, all other fingers folded."""
    lm = _build_landmarks_fist()

    # Override thumb to be extended
    lm[1] = _flat_landmark(0.42, 0.85)
    lm[2] = _flat_landmark(0.38, 0.82)
    lm[3] = _flat_landmark(0.33, 0.80)
    lm[4] = _flat_landmark(0.27, 0.77)  # far from MCP → extended

    return lm


def _build_landmarks_peace() -> List[Landmark]:
    """Index + middle extended, ring + pinky folded."""
    lm = _build_landmarks_fist()

    # Index — extended
    lm[6] = _flat_landmark(0.50, 0.60)
    lm[8] = _flat_landmark(0.50, 0.30)

    # Middle — extended
    lm[10] = _flat_landmark(0.56, 0.58)
    lm[12] = _flat_landmark(0.56, 0.26)

    return lm


def _build_landmarks_pointing() -> List[Landmark]:
    """Only index extended."""
    lm = _build_landmarks_fist()

    # Index — extended
    lm[6] = _flat_landmark(0.50, 0.60)
    lm[8] = _flat_landmark(0.50, 0.30)

    return lm


# ---------------------------------------------------------------------------
# Synthetic Evaluation
# ---------------------------------------------------------------------------

SYNTHETIC_CASES = [
    ("Open Palm",    _build_landmarks_open_palm),
    ("Fist",         _build_landmarks_fist),
    ("Thumbs Up",    _build_landmarks_thumbs_up),
    ("Peace / V Sign", _build_landmarks_peace),
    ("Pointing",     _build_landmarks_pointing),
]


def run_synthetic_evaluation() -> None:
    """Run all synthetic test cases and print a results table."""
    recognizer = GestureRecognizer()
    print("\n" + "=" * 60)
    print("  SYNTHETIC GESTURE RECOGNITION EVALUATION")
    print("=" * 60)
    print(f"  {'Expected':<20} {'Predicted':<20} {'Conf':>6}  {'Pass?':>6}")
    print("-" * 60)

    passed = 0
    for expected, builder in SYNTHETIC_CASES:
        landmarks = builder()
        result = recognizer.classify(landmarks, handedness="Right")
        ok = "PASS" if result.name == expected else "FAIL"
        if result.name == expected:
            passed += 1
        print(f"  {expected:<20} {result.name:<20} {result.confidence:>5.0%}  {ok:>6}")

    print("-" * 60)
    print(f"  Passed: {passed}/{len(SYNTHETIC_CASES)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Webcam Evaluation
# ---------------------------------------------------------------------------

def run_webcam_evaluation(seconds: int = 15) -> None:
    """
    Run webcam for *seconds* seconds and print gesture frequency distribution.
    Requires a connected camera.
    """
    from detector import HandDetector, WebcamCapture

    recognizer = GestureRecognizer()
    counter: Counter = Counter()
    frames = 0

    print(f"\nStarting {seconds}s webcam evaluation... Press Q to stop early.\n")

    try:
        with WebcamCapture() as cap:
            detector = HandDetector()
            end_time = time.time() + seconds

            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                hands = detector.detect(frame)
                if hands:
                    result = recognizer.classify(hands[0].landmarks, hands[0].handedness)
                    counter[result.name] += 1
                else:
                    counter["(no hand)"] += 1
                frames += 1

                # Show minimal preview
                cv2.putText(frame, "Evaluation mode", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 170), 2)
                cv2.imshow("Evaluation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            detector.close()
            cv2.destroyAllWindows()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return

    print("\n" + "=" * 50)
    print("  WEBCAM EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total frames processed: {frames}")
    print(f"  {'Gesture':<25} {'Count':>6}  {'%':>6}")
    print("-" * 50)
    for gesture, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = count / frames * 100 if frames else 0
        print(f"  {gesture:<25} {count:>6}  {pct:>5.1f}%")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hand Gesture Recognition system."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--synthetic", action="store_true",
        help="Run evaluation on synthetic (pre-built) landmark data. No camera needed."
    )
    mode.add_argument(
        "--webcam", action="store_true",
        help="Run live webcam evaluation for N seconds."
    )
    parser.add_argument(
        "--seconds", type=int, default=15,
        help="Duration for webcam evaluation (default: 15)."
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_evaluation()
    elif args.webcam:
        run_webcam_evaluation(args.seconds)


if __name__ == "__main__":
    main()
