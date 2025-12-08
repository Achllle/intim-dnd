# Interactive & Immersive DnD

Attach any USB webcam to a projector to project an interactive Dungeons and Dragons (or similar) game world onto a table-top surface. Interact with the game using intuitive hand gestures. Generate worlds using generative models.

## Features

- **Hand Tracking**: Uses MediaPipe hand landmarks via Python subprocess
- **Pinch Detection**: Detects thumb-index finger pinch gesture and allows moving characters
- **AI Image Generation**: Generate background images using Google Gemini API

## Prerequisites

- Rust (edition 2021)
- Python 3.10+
- OpenCV
- A webcam
- a top-down HDMI projector or table-top monitor

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

### 4. (Optional) Set Up Gemini API for Image Generation

To enable AI background image generation:

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create the config directory and token file:

   ```bash
   mkdir -p ~/.config/finger_tracker
   echo "YOUR_API_KEY_HERE" > ~/.config/finger_tracker/gemini_api_token.txt
   ```

## Usage

1. **Calibration**: Click "Start Calibration" to begin the camera-projector calibration process. A dot will appear on the projector - position your camera to see it and the system will automatically detect the correspondence points.

2. **Tracking**: Once calibrated, the application will track your hand and map the index finger tip position to projector coordinates.

3. **Pinch Gesture**: Bring your thumb and index finger close together to trigger a ripple wave effect at that location.

4. **Image Generation**: Open the Options window (View → Options Window) and use the Image Generation panel to generate AI backgrounds using Gemini.

## Files

- `src/main.rs` - Main application with GUI and camera capture
- `src/hand_tracker.rs` - Hand tracking module using MediaPipe via Python subprocess
- `hand_detect.py` - Python script for MediaPipe hand landmark detection
- `models/hand_landmarker.task` - MediaPipe hand landmarker model (download separately)

## Configuration

Configuration files are stored in `~/.config/finger_tracker/`:

- `homography.txt` - Camera-to-projector transformation matrix
- `camera_roi.txt` - Region of interest for hand detection
- `gemini_api_token.txt` - Gemini API token for image generation (optional)
