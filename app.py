"""
config.py — Central configuration for all constants.

Changing values here propagates to all modules without touching source logic.
"""

# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0          # 0 = default webcam; change for external camera
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FPS_TARGET: int = 30           # target frames per second

# ---------------------------------------------------------------------------
# MediaPipe Hand Detection Thresholds
# ---------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.5
MAX_NUM_HANDS: int = 1         # track one hand; increase to 2 for multi-hand

# ---------------------------------------------------------------------------
# Gesture Labels (must match GestureRecognizer return values exactly)
# ---------------------------------------------------------------------------
GESTURE_LABELS = [
    "Open Palm",
    "Fist",
    "Thumbs Up",
    "Peace / V Sign",
    "Pointing",
    "Unknown",
]

# ---------------------------------------------------------------------------
# Visualization — Colors (BGR format for OpenCV)
# ---------------------------------------------------------------------------
COLOR_PRIMARY    = (0, 255, 170)    # mint green — landmark dots
COLOR_CONNECTION = (255, 200, 0)    # gold — landmark connections
COLOR_TEXT_BG    = (20, 20, 20)     # near-black — label background
COLOR_TEXT_FG    = (255, 255, 255)  # white — label text
COLOR_FPS        = (0, 220, 255)    # cyan — FPS counter
COLOR_WARNING    = (0, 100, 255)    # orange — "no hand detected"
COLOR_CONFIDENCE = (100, 255, 100)  # light green — confidence bar

# ---------------------------------------------------------------------------
# Visualization — Typography & Layout
# ---------------------------------------------------------------------------
FONT_SCALE_LARGE: float = 1.4
FONT_SCALE_SMALL: float = 0.65
FONT_THICKNESS: int = 2
LANDMARK_RADIUS: int = 6
CONNECTION_THICKNESS: int = 2
LABEL_PADDING: int = 12         # pixels of padding inside label box

# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------
SCREENSHOT_DIR: str = "screenshots"
SCREENSHOT_PREFIX: str = "gesture_"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"         # DEBUG | INFO | WARNING | ERROR
LOG_FILE: str = "gesture_app.log"

# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------
FPS_SMOOTHING_WINDOW: int = 20  # rolling average over N frames

# ---------------------------------------------------------------------------
# Confidence Threshold (below this → label as "Unknown")
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.50  # 50 % minimum to accept a gesture
