# Finger Tracker

A camera-projector finger tracking application that uses MediaPipe hand landmark detection to track fingertip positions and map them to projector coordinates.

## Prerequisites

- Rust (edition 2021)
- Python 3.10+
- OpenCV
- A webcam

## Setup

### 1. Download the MediaPipe Hand Landmarker Model

```bash
cd models
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 2. Set Up Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe numpy
```

### 3. Build and Run

```bash
cargo build --release
cargo run --release
```

## Usage

1. **Calibration**: Click "Start Calibration" to begin the camera-projector calibration process. A dot will appear on the projector - position your camera to see it and the system will automatically detect the correspondence points.

2. **Tracking**: Once calibrated, the application will track your hand and map the index finger tip position to projector coordinates.

## Files

- `src/main.rs` - Main application with GUI and camera capture
- `src/hand_tracker.rs` - Hand tracking module using MediaPipe via Python subprocess
- `hand_detect.py` - Python script for MediaPipe hand landmark detection
- `models/hand_landmarker.task` - MediaPipe hand landmarker model (download separately)

## Configuration

Configuration files are stored in `~/.config/finger_tracker/`:

- `homography.txt` - Camera-to-projector transformation matrix
- `camera_roi.txt` - Region of interest for hand detection
