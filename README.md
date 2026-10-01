# Real-Time Hand Gesture Recognition and Visualization System

> **VITyarthi Submission — Computer Vision**
> **Student:** YASH RAJ SINGH&nbsp;|&nbsp; **Reg. No.:** 24BAI10569 &nbsp;|&nbsp; **Branch:** B.Tech AI/ML
> Technology: Python · OpenCV · MediaPipe · pytest

---

## Overview

A real-time webcam application that detects a human hand using **MediaPipe**'s pretrained 21-landmark model, classifies it into one of five predefined gestures using a rule-based engine, and overlays the gesture name, a genuine confidence score, and live landmark visualizations on the video feed.

No training data required. No paid APIs. Runs entirely offline on a standard laptop.

---

## Features

| Feature | Details |
|---------|---------|
| 🖐 Hand Detection | MediaPipe 21-landmark model, real-time |
| 🤟 Gesture Recognition | Open Palm, Fist, Thumbs Up, Peace/V, Pointing |
| 📊 Confidence Score | Rule-based ratio — never fabricated |
| 🎨 Live Visualization | Landmarks, connections, colored confidence bar |
| ⌨️ Keyboard Controls | Q=Quit, S=Screenshot, D=Debug landmarks |
| 🚨 Error Handling | Camera missing/busy, no-hand fallback, frame failures |
| 🧪 Unit Tests | pytest suite — runs without any camera |
| 📋 Evaluation CLI | Synthetic + live webcam evaluation modes |

---

## Tech Stack

- **Python 3.9+**
- **OpenCV** (`opencv-python`) — frame capture, display, drawing
- **MediaPipe** (`mediapipe`) — hand landmark detection
- **NumPy** — array operations
- **pytest** — unit testing

---

## Project Structure

```
hand-gesture-cv/
├── app.py                  # Main entry point
├── config.py               # All tunable constants
├── detector.py             # Hand detection (MediaPipe wrapper)
├── gesture_recognizer.py   # Rule-based gesture classification
├── visualizer.py           # Drawing / overlay helpers
├── utils.py                # Logger, FPS counter, screenshot util
├── evaluation.py           # CLI evaluation script
├── requirements.txt
├── README.md
├── statement.md
├── .gitignore
├── tests/
│   └── test_gestures.py    # pytest unit tests (no camera needed)
└── docs/
    ├── architecture.md
    ├── workflow.md
    ├── diagrams.md
    └── PROJECT_REPORT.md
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- A working webcam

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd hand-gesture-cv

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Launch the live webcam app
```bash
python app.py
```

**Keyboard controls while running:**
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit the application |
| `S` | Save a screenshot to `screenshots/` |
| `D` | Toggle debug mode (show landmark indices) |

---

## How to Test

### Run unit tests (no camera required)
```bash
pytest tests/ -v
```

Expected output: all tests pass in a few seconds, no webcam needed.

### Run synthetic evaluation
```bash
python evaluation.py --synthetic
```

### Run live webcam evaluation (15 seconds)
```bash
python evaluation.py --webcam --seconds 15
```

---

## Gesture Reference

| Gesture | Description |
|---------|-------------|
| Open Palm | All five fingers extended |
| Fist | All five fingers folded |
| Thumbs Up | Only thumb extended |
| Peace / V Sign | Index + middle extended, ring + pinky folded |
| Pointing | Only index finger extended |

---

## Screenshots

![alt text](image.png)

## Non-Functional Notes

- **Performance**: Targets 30 FPS; MediaPipe runs in ~20–40 ms on a standard laptop.
- **Reliability**: Graceful handling of camera errors, frame failures, and detection loss.
- **Usability**: On-screen controls legend always visible; clear "No hand detected" feedback.
- **Maintainability**: All constants in `config.py`; fully modular, each file has a single responsibility.
- **Logging**: All events logged to `gesture_app.log` and console (level configurable in `config.py`).

---

## License

MIT License — see source files for full details.
