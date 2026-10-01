# Problem Statement

## Project Title
Real-Time Hand Gesture Recognition and Visualization System

## Domain
Computer Vision / Human-Computer Interaction

## Problem Description

Current laptop and desktop interfaces rely heavily on keyboard and mouse input, limiting accessibility and natural interaction. Hand gestures are an intuitive, efficient, and touchless communication channel between humans and computers — widely applicable in accessibility tools, AR/VR interfaces, presentation controllers, and sign-language translation systems.

However, most gesture recognition systems either require expensive hardware (depth cameras, data gloves) or depend on cloud-based machine-learning APIs that introduce latency, privacy concerns, and subscription costs.

This project addresses the problem:

> **How can we build a real-time, fully offline hand gesture recognition system using only a standard webcam and free open-source libraries?**

## Scope

### In Scope
- Real-time hand landmark detection using MediaPipe (pretrained, offline)
- Rule-based classification of five hand gestures from 21-landmark hand model
- Live video overlay: landmarks, gesture label, confidence score, FPS
- Keyboard-driven interaction (quit, screenshot, debug)
- Unit tests for classification logic (no camera required)
- CLI evaluation script for reproducible testing

### Out of Scope
- Machine-learning model training (MediaPipe's model is used as-is)
- Multi-hand simultaneous classification (single hand processed)
- Custom gesture creation / user-defined training
- Mobile or embedded deployment
- Network/cloud communication

## Target Users

| User | Need |
|------|------|
| Students / Researchers | Learn and experiment with computer vision pipelines |
| Accessibility advocates | Prototype touchless control interfaces |
| Developers | A clean, modular starting point for gesture-based HCI projects |
| VITyarthi examiners | A working, testable, well-documented CV project |

## High-Level Features

1. **Hand Detection & Tracking** — real-time webcam capture + MediaPipe landmark extraction
2. **Gesture Recognition** — rule-based classifier for 5 gestures with genuine confidence scoring
3. **Real-Time Visualization** — overlaid landmarks, gesture label, confidence bar, FPS counter
4. **Evaluation & Testing** — pytest unit tests + CLI evaluation (synthetic and live modes)
