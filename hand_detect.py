#!/usr/bin/env python3
"""
Hand landmark detection using MediaPipe Tasks API.
Reads images from stdin (as raw bytes), outputs JSON landmarks to stdout.
"""

import sys
import json
import struct
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_model_path():
    """Get the model path from config directory or fallback to local."""
    # Try config directory first
    config_dir = os.path.expanduser("~/.config/intim-dnd/models")
    config_path = os.path.join(config_dir, "hand_landmarker.task")
    if os.path.exists(config_path):
        return config_path
    # Fallback to local models directory
    local_path = "models/hand_landmarker.task"
    if os.path.exists(local_path):
        return local_path
    # Return config path (will error with helpful message)
    return config_path

def main():
    # Initialize the hand landmarker
    model_path = get_model_path()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Signal ready
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    
    while True:
        try:
            # Read header: width (4 bytes), height (4 bytes), channels (4 bytes)
            header = sys.stdin.buffer.read(12)
            if len(header) < 12:
                break
            
            width, height, channels = struct.unpack('III', header)
            
            # Read image data
            data_size = width * height * channels
            image_data = sys.stdin.buffer.read(data_size)
            if len(image_data) < data_size:
                break
            
            # Convert to numpy array
            image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
            
            # MediaPipe expects RGB, OpenCV gives BGR - convert if needed
            if channels == 3:
                image_array = image_array[:, :, ::-1].copy()  # BGR to RGB
            
            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            
            # Detect hands
            result = detector.detect(mp_image)
            
            # Build output
            output = {
                "hands": []
            }
            
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                hand_data = {
                    "handedness": result.handedness[i][0].category_name if result.handedness else "Unknown",
                    "score": result.handedness[i][0].score if result.handedness else 0.0,
                    "landmarks": []
                }
                
                for landmark in hand_landmarks:
                    hand_data["landmarks"].append({
                        "x": landmark.x,  # Normalized 0-1
                        "y": landmark.y,  # Normalized 0-1
                        "z": landmark.z,  # Depth relative to wrist
                    })
                
                output["hands"].append(hand_data)
            
            # Write JSON response
            response = json.dumps(output) + "\n"
            sys.stdout.write(response)
            sys.stdout.flush()
            
        except Exception as e:
            error_response = json.dumps({"error": str(e)}) + "\n"
            sys.stdout.write(error_response)
            sys.stdout.flush()

if __name__ == "__main__":
    main()
