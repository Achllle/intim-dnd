//! Hand tracking module using MediaPipe via Python subprocess
//!
//! This module provides hand pose estimation using the MediaPipe hand landmarker
//! model through a Python subprocess for reliable hand detection.
//!
//! # Model Setup
//!
//! Download the MediaPipe hand landmarker model:
//! wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
//! Place it at: models/hand_landmarker.task

use anyhow::{Context, Result};
use opencv::{
    core::Mat,
    prelude::*,
};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::path::Path;
use serde::Deserialize;

/// Hand landmark indices (MediaPipe hand landmark model convention)
/// See: https://google.github.io/mediapipe/solutions/hands.html
#[allow(dead_code)]
pub mod landmarks {
    pub const WRIST: usize = 0;
    pub const THUMB_CMC: usize = 1;
    pub const THUMB_MCP: usize = 2;
    pub const THUMB_IP: usize = 3;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_FINGER_MCP: usize = 5;
    pub const INDEX_FINGER_PIP: usize = 6;
    pub const INDEX_FINGER_DIP: usize = 7;
    pub const INDEX_FINGER_TIP: usize = 8;
    pub const MIDDLE_FINGER_MCP: usize = 9;
    pub const MIDDLE_FINGER_PIP: usize = 10;
    pub const MIDDLE_FINGER_DIP: usize = 11;
    pub const MIDDLE_FINGER_TIP: usize = 12;
    pub const RING_FINGER_MCP: usize = 13;
    pub const RING_FINGER_PIP: usize = 14;
    pub const RING_FINGER_DIP: usize = 15;
    pub const RING_FINGER_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_PIP: usize = 18;
    pub const PINKY_DIP: usize = 19;
    pub const PINKY_TIP: usize = 20;
}

/// A single hand landmark with 3D coordinates (x, y, z)
#[derive(Clone, Copy, Debug, Default)]
pub struct Landmark {
    /// X coordinate (0.0 to 1.0, normalized to image width)
    pub x: f32,
    /// Y coordinate (0.0 to 1.0, normalized to image height)
    pub y: f32,
    /// Z coordinate (depth, relative to wrist)
    pub z: f32,
}

/// Hand detection result with all 21 landmarks
#[derive(Clone, Debug)]
pub struct HandLandmarks {
    /// All 21 hand landmarks
    pub landmarks: [Landmark; 21],
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Handedness ("Left" or "Right")
    pub handedness: String,
}

impl HandLandmarks {
    /// Get the index finger tip position in pixel coordinates
    pub fn index_finger_tip(&self, image_width: f32, image_height: f32) -> (f32, f32) {
        let tip = &self.landmarks[landmarks::INDEX_FINGER_TIP];
        (tip.x * image_width, tip.y * image_height)
    }

    /// Get the most extended fingertip (the one farthest from wrist)
    pub fn most_extended_fingertip(&self, image_width: f32, image_height: f32) -> (f32, f32) {
        let wrist = &self.landmarks[landmarks::WRIST];
        let tips = [
            landmarks::THUMB_TIP,
            landmarks::INDEX_FINGER_TIP,
            landmarks::MIDDLE_FINGER_TIP,
            landmarks::RING_FINGER_TIP,
            landmarks::PINKY_TIP,
        ];

        let mut max_dist = 0.0f32;
        let mut best_tip = &self.landmarks[landmarks::INDEX_FINGER_TIP];

        for &tip_idx in &tips {
            let tip = &self.landmarks[tip_idx];
            let dx = tip.x - wrist.x;
            let dy = tip.y - wrist.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > max_dist {
                max_dist = dist;
                best_tip = tip;
            }
        }

        (best_tip.x * image_width, best_tip.y * image_height)
    }

    /// Get all landmarks as pixel coordinates for visualization
    pub fn all_landmarks_pixels(&self, image_width: f32, image_height: f32) -> Vec<(f32, f32)> {
        self.landmarks
            .iter()
            .map(|lm| (lm.x * image_width, lm.y * image_height))
            .collect()
    }
}

/// JSON structures for parsing Python output
#[derive(Deserialize, Debug)]
struct LandmarkJson {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Deserialize, Debug)]
struct HandJson {
    handedness: String,
    score: f32,
    landmarks: Vec<LandmarkJson>,
}

#[derive(Deserialize, Debug)]
struct DetectionResult {
    hands: Vec<HandJson>,
    #[serde(default)]
    error: Option<String>,
}

/// Hand tracker using MediaPipe via Python subprocess
pub struct HandTracker {
    /// Python subprocess
    process: Child,
    /// Buffered reader for stdout
    stdout_reader: BufReader<std::process::ChildStdout>,
    /// Minimum confidence threshold for detection
    confidence_threshold: f32,
}

impl HandTracker {
    /// Create a new hand tracker by starting the Python subprocess
    pub fn new<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
        // Find the Python script and virtual environment
        let script_path = std::env::current_dir()?.join("hand_detect.py");
        let venv_python = std::env::current_dir()?.join(".venv/bin/python");
        
        if !script_path.exists() {
            anyhow::bail!("Python hand detection script not found at {:?}", script_path);
        }
        
        if !venv_python.exists() {
            anyhow::bail!("Python virtual environment not found. Run: python3 -m venv .venv && .venv/bin/pip install mediapipe numpy");
        }

        log::info!("Starting MediaPipe hand detector subprocess...");

        let mut process = Command::new(&venv_python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .current_dir(std::env::current_dir()?)
            .spawn()
            .context("Failed to start Python subprocess")?;

        // Take ownership of stdout for the buffered reader
        let stdout = process.stdout.take().context("Failed to get stdout")?;
        let mut stdout_reader = BufReader::new(stdout);
        
        // Wait for "READY" signal
        let mut ready_line = String::new();
        stdout_reader.read_line(&mut ready_line)?;
        
        if !ready_line.trim().eq("READY") {
            anyhow::bail!("Python subprocess did not signal ready, got: {}", ready_line);
        }

        log::info!("MediaPipe hand detector ready");

        Ok(Self {
            process,
            stdout_reader,
            confidence_threshold: 0.5,
        })
    }

    /// Set the confidence threshold for hand detection
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Detect hand landmarks in an image (OpenCV Mat in BGR format)
    ///
    /// Returns the hand landmarks if a hand is detected with sufficient confidence.
    pub fn detect(&mut self, frame: &Mat) -> Result<Option<HandLandmarks>> {
        if frame.empty() {
            return Ok(None);
        }

        let width = frame.cols() as u32;
        let height = frame.rows() as u32;
        let channels = frame.channels() as u32;

        // Get raw frame data
        let data = frame.data_bytes()?;

        // Send frame to Python: header (width, height, channels) + raw BGR data
        let stdin = self.process.stdin.as_mut().context("Failed to get stdin")?;
        
        // Write header
        stdin.write_all(&width.to_le_bytes())?;
        stdin.write_all(&height.to_le_bytes())?;
        stdin.write_all(&channels.to_le_bytes())?;
        
        // Write image data
        stdin.write_all(data)?;
        stdin.flush()?;

        // Read JSON response
        let mut response = String::new();
        self.stdout_reader.read_line(&mut response)?;

        // Parse JSON
        let result: DetectionResult = serde_json::from_str(&response)
            .context(format!("Failed to parse JSON response: {}", response))?;

        if let Some(error) = result.error {
            log::warn!("Python detector error: {}", error);
            return Ok(None);
        }

        // Return the first hand with sufficient confidence
        for hand in result.hands {
            if hand.score >= self.confidence_threshold {
                if hand.landmarks.len() != 21 {
                    log::warn!("Expected 21 landmarks, got {}", hand.landmarks.len());
                    continue;
                }

                let mut landmarks = [Landmark::default(); 21];
                for (i, lm) in hand.landmarks.iter().enumerate() {
                    landmarks[i] = Landmark {
                        x: lm.x,
                        y: lm.y,
                        z: lm.z,
                    };
                }

                log::debug!(
                    "Hand detected: {} (confidence={:.2}), wrist=({:.3},{:.3}), index_tip=({:.3},{:.3})",
                    hand.handedness, hand.score,
                    landmarks[0].x, landmarks[0].y,
                    landmarks[8].x, landmarks[8].y
                );

                return Ok(Some(HandLandmarks {
                    landmarks,
                    confidence: hand.score,
                    handedness: hand.handedness,
                }));
            }
        }

        Ok(None)
    }

    /// Get the input dimensions expected by the model (not really applicable for MediaPipe)
    pub fn input_size(&self) -> (usize, usize) {
        (224, 224)  // Placeholder, MediaPipe handles its own preprocessing
    }
}

impl Drop for HandTracker {
    fn drop(&mut self) {
        // Kill the Python subprocess when the tracker is dropped
        let _ = self.process.kill();
    }
}

/// Get the default model path relative to the executable
pub fn default_model_path() -> std::path::PathBuf {
    std::path::PathBuf::from("models/hand_landmarker.task")
}

/// Check if a hand landmark model is available
pub fn model_available() -> bool {
    default_model_path().exists() && 
    std::path::PathBuf::from("hand_detect.py").exists() &&
    std::path::PathBuf::from(".venv/bin/python").exists()
}
