mod hand_tracker;

use anyhow::Result;
use eframe::egui;
use nalgebra::{Matrix3, Vector3};
use opencv::{
    calib3d,
    core::{self, Mat, Point, Point2f, Size, Vector},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, CAP_V4L2},
};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Hardcoded homography matrix for camera-to-projector transformation
/// This 3x3 matrix transforms points from camera coordinates to projector coordinates
/// 
/// To calibrate:
/// 1. Display known points on projector (e.g., corners)
/// 2. Find corresponding points in camera image
/// 3. Compute homography using OpenCV's findHomography or manually
/// 
/// The matrix is in row-major order:
/// | h00 h01 h02 |   | scale_x   shear_x   translate_x |
/// | h10 h11 h12 | = | shear_y   scale_y   translate_y |
/// | h20 h21 h22 |   | persp_x   persp_y   scale       |
const HOMOGRAPHY: [[f64; 3]; 3] = [
    [3.0, 0.0, 0.0],    // Scale X by 3 (640 camera -> 1920 projector)
    [0.0, 2.25, 0.0],   // Scale Y by 2.25 (480 camera -> 1080 projector)
    [0.0, 0.0, 1.0],    // No perspective distortion
];

/// Region of interest in camera coordinates (computed from calibration corner points)
#[derive(Clone, Copy, Debug)]
struct CameraRoi {
    /// Top-left corner X
    x: i32,
    /// Top-left corner Y
    y: i32,
    /// Width of ROI
    width: i32,
    /// Height of ROI
    height: i32,
}

/// A ripple effect emanating from a pinch location
#[derive(Clone)]
struct Ripple {
    /// Center position in projector coordinates
    center: (f32, f32),
    /// Time when the ripple was created
    start_time: Instant,
    /// Maximum radius the ripple will expand to
    max_radius: f32,
    /// Duration of the ripple animation in seconds
    duration: f32,
    /// Color of the ripple
    color: egui::Color32,
}

/// Get the path to the homography calibration file
fn get_homography_file_path() -> PathBuf {
    // Store in user's config directory or current directory
    if let Some(config_dir) = dirs::config_dir() {
        let app_dir = config_dir.join("finger_tracker");
        let _ = fs::create_dir_all(&app_dir);
        app_dir.join("homography.txt")
    } else {
        PathBuf::from("homography.txt")
    }
}

/// Get the path to the camera ROI calibration file
fn get_roi_file_path() -> PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        let app_dir = config_dir.join("finger_tracker");
        let _ = fs::create_dir_all(&app_dir);
        app_dir.join("camera_roi.txt")
    } else {
        PathBuf::from("camera_roi.txt")
    }
}

/// Save homography matrix to file
fn save_homography(h: &Matrix3<f64>) -> Result<(), String> {
    let path = get_homography_file_path();
    let mut content = String::new();
    for row in 0..3 {
        for col in 0..3 {
            content.push_str(&format!("{}", h[(row, col)]));
            if col < 2 {
                content.push(' ');
            }
        }
        content.push('\n');
    }
    fs::write(&path, content)
        .map_err(|e| format!("Failed to save homography to {:?}: {}", path, e))?;
    log::info!("Saved homography to {:?}", path);
    Ok(())
}

/// Save camera ROI to file
fn save_camera_roi(roi: &CameraRoi) -> Result<(), String> {
    let path = get_roi_file_path();
    let content = format!("{} {} {} {}\n", roi.x, roi.y, roi.width, roi.height);
    fs::write(&path, content)
        .map_err(|e| format!("Failed to save camera ROI to {:?}: {}", path, e))?;
    log::info!("Saved camera ROI to {:?}", path);
    Ok(())
}

/// Load homography matrix from file
fn load_homography() -> Option<[[f64; 3]; 3]> {
    let path = get_homography_file_path();
    let content = fs::read_to_string(&path).ok()?;
    let mut h = [[0.0f64; 3]; 3];
    
    for (row_idx, line) in content.lines().enumerate() {
        if row_idx >= 3 {
            break;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if values.len() >= 3 {
            h[row_idx][0] = values[0];
            h[row_idx][1] = values[1];
            h[row_idx][2] = values[2];
        } else {
            log::warn!("Invalid homography file format at line {}", row_idx);
            return None;
        }
    }
    
    log::info!("Loaded homography from {:?}", path);
    Some(h)
}

/// Load camera ROI from file
fn load_camera_roi() -> Option<CameraRoi> {
    let path = get_roi_file_path();
    let content = fs::read_to_string(&path).ok()?;
    let values: Vec<i32> = content
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    
    if values.len() >= 4 {
        let roi = CameraRoi {
            x: values[0],
            y: values[1],
            width: values[2],
            height: values[3],
        };
        log::info!("Loaded camera ROI from {:?}: {:?}", path, roi);
        Some(roi)
    } else {
        log::warn!("Invalid camera ROI file format");
        None
    }
}

/// Calibration state machine
#[derive(Clone, PartialEq)]
enum CalibrationState {
    /// Not calibrating
    Idle,
    /// Waiting for user to start (shows instructions)
    WaitingToStart,
    /// Displaying a calibration dot, waiting for it to stabilize
    DisplayingDot { index: usize, start_time: Instant },
    /// Capturing the dot position from camera
    CapturingDot { index: usize, start_time: Instant },
    /// Computing the homography from captured points
    Computing,
    /// Calibration complete
    Complete { success: bool, message: String },
}

/// Calibration data
#[derive(Clone)]
struct CalibrationData {
    /// Current calibration state
    state: CalibrationState,
    /// Projector points (where we display dots)
    projector_points: Vec<(f32, f32)>,
    /// Corresponding camera points (where we detect dots)
    camera_points: Vec<(f32, f32)>,
    /// Number of calibration points (grid)
    grid_cols: usize,
    grid_rows: usize,
    /// Margin from screen edges (as fraction of screen size)
    margin: f32,
    /// Dot display time before capture (seconds)
    dot_display_time: f32,
    /// Dot capture time (seconds)
    dot_capture_time: f32,
    /// Calibration dot radius
    dot_radius: f32,
    /// Detected dot position for current frame
    detected_dot: Option<(f32, f32)>,
    /// Accumulated dot positions for averaging
    accumulated_positions: Vec<(f32, f32)>,
    /// Minimum number of samples required for valid detection
    minimum_samples: usize,
    /// Maximum deviation from median to include in final average (pixels)
    outlier_threshold: f32,
}

/// Shared state between camera thread and UI
struct SharedState {
    /// Current camera frame (RGB format)
    frame: Option<Vec<u8>>,
    frame_width: u32,
    frame_height: u32,
    /// Detected finger tip position in camera coordinates
    finger_tip_camera: Option<(f32, f32)>,
    /// Finger tip position transformed to projector coordinates
    finger_tip_projector: Option<(f32, f32)>,
    /// Homography matrix (camera -> projector)
    homography: Matrix3<f64>,
    /// Camera running flag
    running: bool,
    /// Calibration data
    calibration: CalibrationData,
    /// Debug: thresholded frame for calibration visualization (grayscale as RGB)
    debug_threshold_frame: Option<Vec<u8>>,
    /// Debug: skin-thresholded frame for finger detection visualization (grayscale as RGB)
    debug_skin_threshold_frame: Option<Vec<u8>>,
    /// Dimensions of the skin threshold frame (may differ from full frame when using ROI)
    debug_skin_threshold_width: u32,
    debug_skin_threshold_height: u32,
    /// Brightness threshold for calibration dot detection
    calibration_brightness_threshold: u8,
    /// Camera region of interest (computed from calibration corner points)
    camera_roi: Option<CameraRoi>,
    /// Hand landmarks for visualization (when using ML detection)
    hand_landmarks: Option<Vec<(f32, f32)>>,
    /// Whether a pinch is currently detected
    is_pinching: bool,
    /// Pinch center in camera coordinates (normalized 0-1)
    pinch_center_camera: Option<(f32, f32)>,
    /// Active ripple effects
    ripples: Vec<Ripple>,
    /// Cooldown to prevent multiple ripples from same pinch
    last_pinch_time: Option<Instant>,
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self {
            state: CalibrationState::Idle,
            projector_points: Vec::new(),
            camera_points: Vec::new(),
            grid_cols: 3,
            grid_rows: 3,
            margin: 0.1,
            dot_display_time: 1.0,
            dot_capture_time: 2.0, // Increased capture time for more samples
            dot_radius: 20.0,
            detected_dot: None,
            accumulated_positions: Vec::new(),
            minimum_samples: 5,      // Need at least 5 detections
            outlier_threshold: 30.0, // Reject points > 30px from median
        }
    }
}

/// Compute median of a slice of f32 values
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = values.len();
    if len == 0 {
        return 0.0;
    }
    if len % 2 == 0 {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    }
}

/// Compute robust position from accumulated samples using median and outlier rejection
fn compute_robust_position(positions: &[(f32, f32)], outlier_threshold: f32) -> Option<(f32, f32)> {
    if positions.is_empty() {
        return None;
    }
    
    // Compute medians
    let mut xs: Vec<f32> = positions.iter().map(|p| p.0).collect();
    let mut ys: Vec<f32> = positions.iter().map(|p| p.1).collect();
    
    let median_x = median(&mut xs);
    let median_y = median(&mut ys);
    
    // Filter outliers (points too far from median)
    let filtered: Vec<(f32, f32)> = positions
        .iter()
        .filter(|(x, y)| {
            let dx = x - median_x;
            let dy = y - median_y;
            let dist = (dx * dx + dy * dy).sqrt();
            dist <= outlier_threshold
        })
        .copied()
        .collect();
    
    if filtered.is_empty() {
        // Fall back to median if all points are outliers
        return Some((median_x, median_y));
    }
    
    // Average the inliers for final position
    let sum: (f32, f32) = filtered
        .iter()
        .fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
    let count = filtered.len() as f32;
    
    Some((sum.0 / count, sum.1 / count))
}

impl SharedState {
    fn new() -> Self {
        // Try to load saved homography, fall back to default
        let h = load_homography().unwrap_or(HOMOGRAPHY);
        let homography = Matrix3::from_row_slice(&h.concat());
        // Try to load saved camera ROI
        let camera_roi = load_camera_roi();
        Self {
            frame: None,
            frame_width: 640,
            frame_height: 480,
            finger_tip_camera: None,
            finger_tip_projector: None,
            homography,
            running: true,
            calibration: CalibrationData::default(),
            debug_threshold_frame: None,
            debug_skin_threshold_frame: None,
            debug_skin_threshold_width: 640,
            debug_skin_threshold_height: 480,
            calibration_brightness_threshold: 200,
            camera_roi,
            hand_landmarks: None,
            is_pinching: false,
            pinch_center_camera: None,
            ripples: Vec::new(),
            last_pinch_time: None,
        }
    }

    /// Transform a point from camera coordinates to projector coordinates using homography
    fn transform_point(&self, camera_point: (f32, f32)) -> (f32, f32) {
        let p = Vector3::new(camera_point.0 as f64, camera_point.1 as f64, 1.0);
        let transformed = self.homography * p;
        let w = transformed[2];
        if w.abs() > 1e-10 {
            ((transformed[0] / w) as f32, (transformed[1] / w) as f32)
        } else {
            camera_point
        }
    }

    /// Update the homography matrix
    fn set_homography(&mut self, h: [[f64; 3]; 3]) {
        self.homography = Matrix3::from_row_slice(&h.concat());
    }

    /// Generate calibration grid points in projector coordinates
    fn generate_calibration_points(&mut self, proj_width: f32, proj_height: f32) {
        self.calibration.projector_points.clear();
        self.calibration.camera_points.clear();
        
        let margin_x = proj_width * self.calibration.margin;
        let margin_y = proj_height * self.calibration.margin;
        
        let usable_width = proj_width - 2.0 * margin_x;
        let usable_height = proj_height - 2.0 * margin_y;
        
        for row in 0..self.calibration.grid_rows {
            for col in 0..self.calibration.grid_cols {
                let x = margin_x + (col as f32 / (self.calibration.grid_cols - 1).max(1) as f32) * usable_width;
                let y = margin_y + (row as f32 / (self.calibration.grid_rows - 1).max(1) as f32) * usable_height;
                self.calibration.projector_points.push((x, y));
            }
        }
    }

    /// Compute homography from calibration points
    fn compute_homography(&mut self) -> Result<(), String> {
        if self.calibration.camera_points.len() < 4 {
            return Err("Need at least 4 point pairs for homography".to_string());
        }
        
        if self.calibration.camera_points.len() != self.calibration.projector_points.len() {
            return Err("Mismatched number of camera and projector points".to_string());
        }

        // Convert points to OpenCV format
        let mut src_points = Vector::<Point2f>::new();
        let mut dst_points = Vector::<Point2f>::new();
        
        for (cam, proj) in self.calibration.camera_points.iter()
            .zip(self.calibration.projector_points.iter()) 
        {
            src_points.push(Point2f::new(cam.0, cam.1));
            dst_points.push(Point2f::new(proj.0, proj.1));
        }

        // Compute homography using OpenCV
        let homography_mat = calib3d::find_homography(
            &src_points,
            &dst_points,
            &mut Mat::default(),
            0, // Method: 0 = regular least squares
            3.0, // ransacReprojThreshold (not used with method 0)
        ).map_err(|e| format!("Failed to compute homography: {}", e))?;

        if homography_mat.empty() {
            return Err("Homography computation failed - empty result".to_string());
        }

        // Extract the 3x3 matrix
        let mut h = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                h[row][col] = *homography_mat.at_2d::<f64>(row as i32, col as i32)
                    .map_err(|e| format!("Failed to read homography element: {}", e))?;
            }
        }

        self.set_homography(h);
        
        // Compute camera ROI from the bounding box of calibration camera points
        self.compute_camera_roi();
        
        Ok(())
    }
    
    /// Compute camera ROI from calibration points (bounding box of detected corners)
    fn compute_camera_roi(&mut self) {
        if self.calibration.camera_points.len() < 4 {
            self.camera_roi = None;
            return;
        }
        
        // Find bounding box of all camera calibration points
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        
        for (x, y) in &self.calibration.camera_points {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }
        
        // Add a small margin (5% on each side) to ensure we capture the full area
        let width = max_x - min_x;
        let height = max_y - min_y;
        let margin_x = width * 0.05;
        let margin_y = height * 0.05;
        
        // Clamp to frame bounds
        let x = (min_x - margin_x).max(0.0) as i32;
        let y = (min_y - margin_y).max(0.0) as i32;
        let roi_width = ((max_x - min_x) + 2.0 * margin_x) as i32;
        let roi_height = ((max_y - min_y) + 2.0 * margin_y) as i32;
        
        // Ensure minimum size
        if roi_width > 50 && roi_height > 50 {
            self.camera_roi = Some(CameraRoi {
                x,
                y,
                width: roi_width,
                height: roi_height,
            });
            log::info!("Camera ROI set to: x={}, y={}, w={}, h={}", x, y, roi_width, roi_height);
        } else {
            self.camera_roi = None;
            log::warn!("Computed ROI too small, using full frame");
        }
    }
}

/// Detect a bright calibration dot in the frame
/// Returns the centroid of the brightest blob and optionally the thresholded image
fn detect_calibration_dot(frame: &Mat, min_brightness: u8, return_threshold: bool) -> Result<(Option<(f32, f32)>, Option<Vec<u8>>)> {
    // Convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    
    // Apply Gaussian blur to reduce noise
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    
    // Threshold to find bright regions
    let mut binary = Mat::default();
    imgproc::threshold(
        &blurred,
        &mut binary,
        min_brightness as f64,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    
    // Morphological operations to clean up
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    
    let mut cleaned = Mat::default();
    imgproc::morphology_ex(
        &binary,
        &mut cleaned,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    imgproc::morphology_ex(
        &cleaned,
        &mut binary,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    
    // Convert threshold image to RGB for display if requested
    let threshold_rgb = if return_threshold {
        let mut rgb = Mat::default();
        imgproc::cvt_color(&binary, &mut rgb, imgproc::COLOR_GRAY2RGB, 0)?;
        rgb.data_bytes().ok().map(|d| d.to_vec())
    } else {
        None
    };
    
    // Find contours
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &binary,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    
    // Find the brightest and largest blob
    let mut best_contour_idx: Option<usize> = None;
    let mut max_area = 10.0; // Minimum area to consider (lowered for small dots)
    
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        if area > max_area {
            max_area = area;
            best_contour_idx = Some(i);
        }
    }
    
    if let Some(idx) = best_contour_idx {
        let contour = contours.get(idx)?;
        let moments = imgproc::moments(&contour, false)?;
        if moments.m00 > 0.0 {
            let cx = (moments.m10 / moments.m00) as f32;
            let cy = (moments.m01 / moments.m00) as f32;
            return Ok((Some((cx, cy)), threshold_rgb));
        }
    }
    
    Ok((None, threshold_rgb))
}

/// Camera capture and processing thread
fn camera_thread(state: Arc<Mutex<SharedState>>, device_path: &str) {
    log::info!("Opening camera at {}", device_path);
    
    // Open camera using V4L2
    let mut cap = match VideoCapture::from_file(device_path, CAP_V4L2) {
        Ok(cap) => cap,
        Err(e) => {
            log::error!("Failed to open camera {}: {}", device_path, e);
            return;
        }
    };

    // Check if camera opened successfully
    if !cap.is_opened().unwrap_or(false) {
        log::error!("Camera {} is not opened", device_path);
        return;
    }

    // Set camera properties
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0);
    let _ = cap.set(videoio::CAP_PROP_FPS, 30.0);

    log::info!("Camera opened successfully");
    
    // Try to initialize ML hand tracker (optional)
    let mut ml_tracker: Option<hand_tracker::HandTracker> = {
        let model_path = hand_tracker::default_model_path();
        match hand_tracker::HandTracker::new(&model_path) {
            Ok(tracker) => {
                log::info!("ML hand tracker initialized successfully");
                Some(tracker)
            }
            Err(e) => {
                log::warn!("ML hand tracker not available: {}. Using HSV detection only.", e);
                None
            }
        }
    };

    let mut frame = Mat::default();

    loop {
        // Check if we should stop
        {
            let state = state.lock().unwrap();
            if !state.running {
                break;
            }
        }

        // Capture frame
        match cap.read(&mut frame) {
            Ok(true) if !frame.empty() => {
                // Check if we're in calibration capture mode
                let is_calibrating = {
                    let state = state.lock().unwrap();
                    matches!(state.calibration.state, CalibrationState::CapturingDot { .. })
                };
                
                // Get camera ROI for cropping (if calibrated)
                let camera_roi = {
                    let state = state.lock().unwrap();
                    state.camera_roi
                };

                // Detect finger tip (when not calibrating) using ML hand tracking
                // Returns (finger_tip, debug_frame, frame_width, frame_height, hand_landmarks, pinch_center)
                let (finger_tip, debug_frame, debug_w, debug_h, hand_landmarks, pinch_center) = if !is_calibrating && ml_tracker.is_some() {
                    // Get frame dimensions for bounds checking
                    let frame_size = frame.size().unwrap_or(Size::new(640, 480));
                    
                    // Prepare the detection frame (with optional ROI cropping)
                    let (detect_frame, roi_offset) = if let Some(roi) = camera_roi {
                        // Clamp ROI to frame bounds
                        let roi_x = roi.x.max(0).min(frame_size.width - 1);
                        let roi_y = roi.y.max(0).min(frame_size.height - 1);
                        let roi_w = roi.width.min(frame_size.width - roi_x);
                        let roi_h = roi.height.min(frame_size.height - roi_y);
                        
                        if roi_w > 10 && roi_h > 10 {
                            let roi_rect = core::Rect::new(roi_x, roi_y, roi_w, roi_h);
                            if let Ok(cropped_ref) = Mat::roi(&frame, roi_rect) {
                                let mut cropped = Mat::default();
                                if cropped_ref.copy_to(&mut cropped).is_ok() {
                                    (cropped, Some((roi_x, roi_y, roi_w, roi_h)))
                                } else {
                                    (frame.clone(), None)
                                }
                            } else {
                                (frame.clone(), None)
                            }
                        } else {
                            (frame.clone(), None)
                        }
                    } else {
                        (frame.clone(), None)
                    };
                    
                    let detect_size = detect_frame.size().unwrap_or(Size::new(640, 480));
                    let tracker = ml_tracker.as_mut().unwrap();
                    
                    // Create debug visualization frame (copy of detection frame converted to RGB)
                    let debug_rgb = {
                        let mut rgb = Mat::default();
                        if imgproc::cvt_color(&detect_frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0).is_ok() {
                            rgb.data_bytes().ok().map(|d| d.to_vec())
                        } else {
                            None
                        }
                    };
                    
                    match tracker.detect(&detect_frame) {
                        Ok(Some(landmarks)) => {
                            let (tip_x, tip_y) = landmarks.index_finger_tip(
                                detect_size.width as f32,
                                detect_size.height as f32,
                            );
                            
                            // Get all landmarks for visualization (in ROI-local pixel coordinates)
                            let local_landmarks = landmarks.all_landmarks_pixels(
                                detect_size.width as f32,
                                detect_size.height as f32,
                            );
                            
                            // Detect pinch gesture (threshold is normalized distance)
                            let pinch_threshold = 0.05; // 5% of image dimension
                            let pinch_result = landmarks.detect_pinch(pinch_threshold);
                            
                            // Apply ROI offset to fingertip for full-frame coordinates
                            let final_tip = if let Some((ox, oy, _, _)) = roi_offset {
                                (tip_x + ox as f32, tip_y + oy as f32)
                            } else {
                                (tip_x, tip_y)
                            };
                            
                            // Keep landmarks in local ROI coordinates for debug visualization
                            (Some(final_tip), debug_rgb, detect_size.width as u32, detect_size.height as u32, Some(local_landmarks), pinch_result)
                        }
                        Ok(None) => {
                            // No hand detected
                            (None, debug_rgb, detect_size.width as u32, detect_size.height as u32, None, None)
                        }
                        Err(e) => {
                            log::debug!("ML detection error: {}", e);
                            (None, debug_rgb, detect_size.width as u32, detect_size.height as u32, None, None)
                        }
                    }
                } else if !is_calibrating {
                    // ML tracker not available, show frame but no detection
                    let frame_size = frame.size().unwrap_or(Size::new(640, 480));
                    let debug_rgb = {
                        let mut rgb = Mat::default();
                        if imgproc::cvt_color(&frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0).is_ok() {
                            rgb.data_bytes().ok().map(|d| d.to_vec())
                        } else {
                            None
                        }
                    };
                    (None, debug_rgb, frame_size.width as u32, frame_size.height as u32, None, None)
                } else {
                    (None, None, 640, 480, None, None)
                };
                
                // Get calibration brightness threshold
                let brightness_threshold = {
                    let state = state.lock().unwrap();
                    state.calibration_brightness_threshold
                };
                
                // Detect calibration dot and get threshold image for debug
                let (calibration_dot, threshold_frame) = 
                    detect_calibration_dot(&frame, brightness_threshold, true)
                        .unwrap_or((None, None));

                // Convert frame to RGB for display
                let mut rgb_frame = Mat::default();
                if imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0).is_ok() {
                    // Update shared state
                    let mut state = state.lock().unwrap();
                    
                    // Get frame dimensions
                    if let Ok(size) = rgb_frame.size() {
                        state.frame_width = size.width as u32;
                        state.frame_height = size.height as u32;
                    }
                    
                    // Convert Mat to Vec<u8>
                    if let Ok(data) = rgb_frame.data_bytes() {
                        state.frame = Some(data.to_vec());
                    }

                    // Update finger position (when not calibrating)
                    state.finger_tip_camera = finger_tip;
                    if let Some(camera_pt) = finger_tip {
                        state.finger_tip_projector = Some(state.transform_point(camera_pt));
                    } else {
                        state.finger_tip_projector = None;
                    }
                    
                    // Update calibration dot detection and debug frame
                    state.calibration.detected_dot = calibration_dot;
                    state.debug_threshold_frame = threshold_frame;
                    state.debug_skin_threshold_frame = debug_frame;
                    state.debug_skin_threshold_width = debug_w;
                    state.debug_skin_threshold_height = debug_h;
                    
                    // Update hand landmarks (for ML mode visualization)
                    state.hand_landmarks = hand_landmarks;
                    
                    // Handle pinch detection and ripple creation
                    let was_pinching = state.is_pinching;
                    state.is_pinching = pinch_center.is_some();
                    state.pinch_center_camera = pinch_center;
                    
                    // Create a ripple on pinch start (rising edge)
                    if state.is_pinching && !was_pinching {
                        // Check cooldown to prevent rapid ripples
                        let cooldown_ok = state.last_pinch_time
                            .map(|t| t.elapsed().as_secs_f32() > 0.3)
                            .unwrap_or(true);
                        
                        if cooldown_ok {
                            if let Some((px, py)) = pinch_center {
                                // Convert normalized pinch center to full camera coordinates
                                let camera_roi = state.camera_roi;
                                let (cam_x, cam_y) = if let Some(roi) = camera_roi {
                                    // Pinch is in ROI-local normalized coords, convert to full frame
                                    let full_x = roi.x as f32 + px * roi.width as f32;
                                    let full_y = roi.y as f32 + py * roi.height as f32;
                                    (full_x, full_y)
                                } else {
                                    // Pinch is in full frame normalized coords
                                    (px * state.frame_width as f32, py * state.frame_height as f32)
                                };
                                
                                // Transform to projector coordinates
                                let proj_pos = state.transform_point((cam_x, cam_y));
                                
                                log::info!("Pinch detected! Creating ripple at projector ({:.0}, {:.0})", proj_pos.0, proj_pos.1);
                                
                                state.ripples.push(Ripple {
                                    center: proj_pos,
                                    start_time: Instant::now(),
                                    max_radius: 300.0,
                                    duration: 1.5,
                                    color: egui::Color32::from_rgba_unmultiplied(100, 200, 255, 200),
                                });
                                state.last_pinch_time = Some(Instant::now());
                            }
                        }
                    }
                    
                    // Clean up expired ripples
                    state.ripples.retain(|r| r.start_time.elapsed().as_secs_f32() < r.duration);
                    
                    // Only accumulate positions during capture phase
                    if matches!(state.calibration.state, CalibrationState::CapturingDot { .. }) {
                        if let Some(dot) = calibration_dot {
                            state.calibration.accumulated_positions.push(dot);
                        }
                    }
                }
            }
            Ok(_) => {
                // Empty frame or read failed
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                log::warn!("Failed to read frame: {}", e);
                thread::sleep(Duration::from_millis(100));
            }
        }

        // Small sleep to prevent busy-waiting
        thread::sleep(Duration::from_millis(5));
    }

    log::info!("Camera thread stopped");
}

/// Main application struct for egui
struct FingerTrackerApp {
    state: Arc<Mutex<SharedState>>,
    texture: Option<egui::TextureHandle>,
    /// UI toggle for showing camera feed vs projector output
    show_camera_feed: bool,
    /// Circle radius for finger indicator
    circle_radius: f32,
    /// Circle color
    circle_color: egui::Color32,
    /// Projector output dimensions
    projector_width: f32,
    projector_height: f32,
    /// Homography matrix editor values
    homography_edit: [[f64; 3]; 3],
    /// Whether to show the options window
    show_options_window: bool,
}

impl FingerTrackerApp {
    fn new(state: Arc<Mutex<SharedState>>) -> Self {
        // Get the homography from state (may have been loaded from file)
        let homography_edit = {
            let state_lock = state.lock().unwrap();
            let h = state_lock.homography;
            [
                [h[(0, 0)], h[(0, 1)], h[(0, 2)]],
                [h[(1, 0)], h[(1, 1)], h[(1, 2)]],
                [h[(2, 0)], h[(2, 1)], h[(2, 2)]],
            ]
        };
        Self {
            state,
            texture: None,
            show_camera_feed: true,
            circle_radius: 30.0,
            circle_color: egui::Color32::from_rgb(255, 100, 100),
            projector_width: 1920.0,
            projector_height: 1080.0,
            homography_edit,
            show_options_window: true,
        }
    }
}

fn options_viewport_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("options_viewport")
}

impl eframe::App for FingerTrackerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repaint for real-time updates
        ctx.request_repaint();

        // Handle F11 to toggle fullscreen
        if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            let is_fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
        }

        // Menu bar to toggle options window
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("View", |ui| {
                    if ui.checkbox(&mut self.show_options_window, "Options Window").clicked() {
                        ui.close_menu();
                    }
                });
            });
        });

        // Options window as a separate OS-level viewport (can be moved to different screen)
        if self.show_options_window {
            let state = self.state.clone();
            let show_camera_feed = Arc::new(Mutex::new(self.show_camera_feed));
            let circle_radius = Arc::new(Mutex::new(self.circle_radius));
            let circle_color = Arc::new(Mutex::new(self.circle_color));
            let projector_width = Arc::new(Mutex::new(self.projector_width));
            let projector_height = Arc::new(Mutex::new(self.projector_height));
            let homography_edit = Arc::new(Mutex::new(self.homography_edit));
            let show_options = Arc::new(Mutex::new(true));

            // Clone Arcs for the closure
            let show_camera_feed_c = show_camera_feed.clone();
            let circle_radius_c = circle_radius.clone();
            let circle_color_c = circle_color.clone();
            let projector_width_c = projector_width.clone();
            let projector_height_c = projector_height.clone();
            let homography_edit_c = homography_edit.clone();
            let show_options_c = show_options.clone();

            ctx.show_viewport_immediate(
                options_viewport_id(),
                egui::ViewportBuilder::default()
                    .with_title("Finger Tracker - Options")
                    .with_inner_size([350.0, 650.0]),
                |ctx, _class| {
                    // Handle window close request
                    if ctx.input(|i| i.viewport().close_requested()) {
                        *show_options_c.lock().unwrap() = false;
                    }

                    egui::CentralPanel::default().show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.heading("Finger Tracker Options");
                            ui.separator();

                            {
                                let mut show_cam = show_camera_feed_c.lock().unwrap();
                                ui.checkbox(&mut *show_cam, "Show Camera Feed");
                            }
                            ui.label("(Uncheck for projector output mode)");
                            
                            ui.separator();
                            ui.heading("Circle Settings");
                            {
                                let mut radius = circle_radius_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *radius, 5.0..=100.0).text("Radius"));
                            }
                            ui.horizontal(|ui| {
                                ui.label("Color:");
                                let mut color = circle_color_c.lock().unwrap();
                                egui::color_picker::color_edit_button_srgba(
                                    ui,
                                    &mut *color,
                                    egui::color_picker::Alpha::Opaque,
                                );
                            });

                            ui.separator();
                            ui.heading("Projector Dimensions");
                            {
                                let mut width = projector_width_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *width, 800.0..=3840.0).text("Width"));
                            }
                            {
                                let mut height = projector_height_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *height, 600.0..=2160.0).text("Height"));
                            }

                            ui.separator();
                            ui.heading("ML Hand Tracking");
                            {
                                let ml_available = hand_tracker::model_available();
                                if ml_available {
                                    ui.label("✓ Hand landmark model loaded");
                                } else {
                                    ui.colored_label(egui::Color32::RED, "✗ Model not found");
                                    ui.small("Place hand_landmark.onnx in models/ folder");
                                }
                            }

                            ui.separator();
                            ui.heading("Homography Matrix");
                            ui.label("Camera → Projector transform:");
                            
                            let mut changed = false;
                            {
                                let mut homo = homography_edit_c.lock().unwrap();
                                for row in 0..3 {
                                    ui.horizontal(|ui| {
                                        for col in 0..3 {
                                            let resp = ui.add(
                                                egui::DragValue::new(&mut homo[row][col])
                                                    .speed(0.01)
                                                    .fixed_decimals(3)
                                            );
                                            if resp.changed() {
                                                changed = true;
                                            }
                                        }
                                    });
                                }
                            }
                            
                            if changed {
                                let homo = homography_edit_c.lock().unwrap();
                                let mut state_lock = state.lock().unwrap();
                                state_lock.set_homography(*homo);
                            }

                            ui.separator();
                            ui.heading("Calibration");
                            
                            // Get calibration state for UI
                            let (calib_state, num_points, total_points, detected_dot) = {
                                let state_lock = state.lock().unwrap();
                                (
                                    state_lock.calibration.state.clone(),
                                    state_lock.calibration.camera_points.len(),
                                    state_lock.calibration.projector_points.len(),
                                    state_lock.calibration.detected_dot,
                                )
                            };
                            
                            // Display calibration status
                            match &calib_state {
                                CalibrationState::Idle => {
                                    ui.label("Status: Ready to calibrate");
                                    if ui.button("🎯 Start Calibration").clicked() {
                                        let proj_w = *projector_width_c.lock().unwrap();
                                        let proj_h = *projector_height_c.lock().unwrap();
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.generate_calibration_points(proj_w, proj_h);
                                        state_lock.calibration.state = CalibrationState::WaitingToStart;
                                    }
                                }
                                CalibrationState::WaitingToStart => {
                                    ui.colored_label(egui::Color32::YELLOW, 
                                        "⚠️ Position projector window on projector display!");
                                    ui.label("Switch main window to Projector Mode first.");
                                    if ui.button("▶️ Begin Calibration").clicked() {
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.calibration.state = CalibrationState::DisplayingDot { 
                                            index: 0, 
                                            start_time: Instant::now() 
                                        };
                                    }
                                    if ui.button("❌ Cancel").clicked() {
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.calibration.state = CalibrationState::Idle;
                                    }
                                }
                                CalibrationState::DisplayingDot { index, .. } => {
                                    ui.colored_label(egui::Color32::GREEN, 
                                        format!("Displaying dot {}/{}", index + 1, total_points));
                                    ui.label("Waiting for dot to stabilize...");
                                    if ui.button("❌ Cancel").clicked() {
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.calibration.state = CalibrationState::Idle;
                                        state_lock.calibration.camera_points.clear();
                                    }
                                }
                                CalibrationState::CapturingDot { index, .. } => {
                                    ui.colored_label(egui::Color32::LIGHT_BLUE, 
                                        format!("Capturing dot {}/{}", index + 1, total_points));
                                    
                                    // Show sample count
                                    let (sample_count, min_samples) = {
                                        let state_lock = state.lock().unwrap();
                                        (state_lock.calibration.accumulated_positions.len(),
                                         state_lock.calibration.minimum_samples)
                                    };
                                    ui.label(format!("Samples: {}/{}", sample_count, min_samples));
                                    
                                    if let Some((dx, dy)) = detected_dot {
                                        ui.label(format!("Detected at: ({:.1}, {:.1})", dx, dy));
                                    } else {
                                        ui.colored_label(egui::Color32::RED, "⚠️ Dot not detected!");
                                    }
                                    if ui.button("❌ Cancel").clicked() {
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.calibration.state = CalibrationState::Idle;
                                        state_lock.calibration.camera_points.clear();
                                    }
                                }
                                CalibrationState::Computing => {
                                    ui.colored_label(egui::Color32::YELLOW, "Computing homography...");
                                }
                                CalibrationState::Complete { success, message } => {
                                    if *success {
                                        ui.colored_label(egui::Color32::GREEN, "✅ Calibration complete!");
                                        ui.label(message);
                                    } else {
                                        ui.colored_label(egui::Color32::RED, "❌ Calibration failed!");
                                        ui.label(message);
                                    }
                                    if ui.button("OK").clicked() {
                                        let mut state_lock = state.lock().unwrap();
                                        state_lock.calibration.state = CalibrationState::Idle;
                                        // Update the homography edit display
                                        let h = state_lock.homography;
                                        let mut homo = homography_edit_c.lock().unwrap();
                                        for row in 0..3 {
                                            for col in 0..3 {
                                                homo[row][col] = h[(row, col)];
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Progress bar
                            if total_points > 0 {
                                let progress = num_points as f32 / total_points as f32;
                                ui.add(egui::ProgressBar::new(progress)
                                    .text(format!("{}/{} points", num_points, total_points)));
                            }
                            
                            // Brightness threshold slider
                            ui.add_space(4.0);
                            {
                                let mut state_lock = state.lock().unwrap();
                                ui.add(egui::Slider::new(&mut state_lock.calibration_brightness_threshold, 50..=255)
                                    .text("Brightness Threshold"));
                            }
                            
                            // Debug views - raw camera and threshold side by side
                            ui.add_space(8.0);
                            let (frame_data, threshold_data, skin_threshold_data, frame_w, frame_h, skin_w, skin_h, detected_pt, finger_tip, hand_landmarks) = {
                                let state_lock = state.lock().unwrap();
                                (
                                    state_lock.frame.clone(),
                                    state_lock.debug_threshold_frame.clone(),
                                    state_lock.debug_skin_threshold_frame.clone(),
                                    state_lock.frame_width,
                                    state_lock.frame_height,
                                    state_lock.debug_skin_threshold_width,
                                    state_lock.debug_skin_threshold_height,
                                    state_lock.calibration.detected_dot,
                                    state_lock.finger_tip_camera,
                                    state_lock.hand_landmarks.clone(),
                                )
                            };
                            
                            let width = frame_w as usize;
                            let height = frame_h as usize;
                            let expected_size = width * height * 3;
                            
                            // Display both views side by side
                            let display_width = 160.0;
                            let aspect = height as f32 / width as f32;
                            let display_height = display_width * aspect;
                            
                            ui.horizontal(|ui| {
                                // Raw camera feed
                                ui.vertical(|ui| {
                                    ui.label("Camera Feed:");
                                    if let Some(data) = &frame_data {
                                        if data.len() >= expected_size {
                                            let image = egui::ColorImage::from_rgb(
                                                [width, height],
                                                &data[..expected_size],
                                            );
                                            
                                            let texture: egui::TextureHandle = ctx.load_texture(
                                                "debug_camera",
                                                image,
                                                egui::TextureOptions::LINEAR,
                                            );
                                            
                                            let (rect, _response) = ui.allocate_exact_size(
                                                egui::vec2(display_width, display_height),
                                                egui::Sense::hover(),
                                            );
                                            
                                            ui.painter().image(
                                                texture.id(),
                                                rect,
                                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                egui::Color32::WHITE,
                                            );
                                            
                                            // Draw detected dot position on camera view too
                                            if let Some((dx, dy)) = detected_pt {
                                                let scale_x = display_width / width as f32;
                                                let scale_y = display_height / height as f32;
                                                let screen_x = rect.min.x + dx * scale_x;
                                                let screen_y = rect.min.y + dy * scale_y;
                                                
                                                ui.painter().circle_stroke(
                                                    egui::pos2(screen_x, screen_y),
                                                    6.0,
                                                    egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                );
                                            }
                                        } else {
                                            ui.colored_label(egui::Color32::GRAY, "Size mismatch");
                                        }
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "No data");
                                    }
                                });
                                
                                ui.add_space(8.0);
                                
                                // Threshold debug view
                                ui.vertical(|ui| {
                                    ui.label("Threshold View:");
                                    if let Some(data) = &threshold_data {
                                        if data.len() >= expected_size {
                                            let image = egui::ColorImage::from_rgb(
                                                [width, height],
                                                &data[..expected_size],
                                            );
                                            
                                            let texture: egui::TextureHandle = ctx.load_texture(
                                                "debug_threshold",
                                                image,
                                                egui::TextureOptions::LINEAR,
                                            );
                                            
                                            let (rect, _response) = ui.allocate_exact_size(
                                                egui::vec2(display_width, display_height),
                                                egui::Sense::hover(),
                                            );
                                            
                                            ui.painter().image(
                                                texture.id(),
                                                rect,
                                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                egui::Color32::WHITE,
                                            );
                                            
                                            // Draw detected dot position on threshold view
                                            if let Some((dx, dy)) = detected_pt {
                                                let scale_x = display_width / width as f32;
                                                let scale_y = display_height / height as f32;
                                                let screen_x = rect.min.x + dx * scale_x;
                                                let screen_y = rect.min.y + dy * scale_y;
                                                
                                                // Draw crosshair at detected position
                                                ui.painter().circle_stroke(
                                                    egui::pos2(screen_x, screen_y),
                                                    6.0,
                                                    egui::Stroke::new(2.0, egui::Color32::RED),
                                                );
                                                ui.painter().line_segment(
                                                    [egui::pos2(screen_x - 10.0, screen_y), egui::pos2(screen_x + 10.0, screen_y)],
                                                    egui::Stroke::new(1.0, egui::Color32::RED),
                                                );
                                                ui.painter().line_segment(
                                                    [egui::pos2(screen_x, screen_y - 10.0), egui::pos2(screen_x, screen_y + 10.0)],
                                                    egui::Stroke::new(1.0, egui::Color32::RED),
                                                );
                                            }
                                        } else {
                                            ui.colored_label(egui::Color32::GRAY, "Size mismatch");
                                        }
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "No data");
                                    }
                                });
                                
                                ui.add_space(8.0);
                                
                                // Skin threshold view with fingertip indicator
                                // Note: This may be cropped to the ROI, so use skin_w/skin_h
                                ui.vertical(|ui| {
                                    ui.label("Skin Threshold:");
                                    let skin_width = skin_w as usize;
                                    let skin_height = skin_h as usize;
                                    let skin_expected_size = skin_width * skin_height * 3;
                                    let skin_aspect = skin_height as f32 / skin_width as f32;
                                    let skin_display_height = display_width * skin_aspect;
                                    
                                    if let Some(data) = &skin_threshold_data {
                                        if data.len() >= skin_expected_size {
                                            let image = egui::ColorImage::from_rgb(
                                                [skin_width, skin_height],
                                                &data[..skin_expected_size],
                                            );
                                            
                                            let texture: egui::TextureHandle = ctx.load_texture(
                                                "debug_skin_threshold",
                                                image,
                                                egui::TextureOptions::LINEAR,
                                            );
                                            
                                            let (rect, _response) = ui.allocate_exact_size(
                                                egui::vec2(display_width, skin_display_height),
                                                egui::Sense::hover(),
                                            );
                                            
                                            ui.painter().image(
                                                texture.id(),
                                                rect,
                                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                egui::Color32::WHITE,
                                            );
                                            
                                            // Draw hand skeleton on the debug view
                                            // Hand landmarks are already in ROI-local coordinates
                                            if let Some(landmarks) = &hand_landmarks {
                                                let scale_x = display_width / skin_width as f32;
                                                let scale_y = skin_display_height / skin_height as f32;
                                                
                                                // Helper to convert landmark to screen coordinates
                                                let to_screen = |idx: usize| -> egui::Pos2 {
                                                    if idx < landmarks.len() {
                                                        let (lx, ly) = landmarks[idx];
                                                        egui::pos2(
                                                            rect.min.x + lx * scale_x,
                                                            rect.min.y + ly * scale_y,
                                                        )
                                                    } else {
                                                        egui::pos2(0.0, 0.0)
                                                    }
                                                };
                                                
                                                // Hand skeleton connections (MediaPipe format)
                                                // Thumb: 0-1-2-3-4
                                                // Index: 0-5-6-7-8
                                                // Middle: 0-9-10-11-12
                                                // Ring: 0-13-14-15-16
                                                // Pinky: 0-17-18-19-20
                                                // Palm: 5-9, 9-13, 13-17, 0-5, 0-17
                                                let connections: &[(usize, usize, egui::Color32)] = &[
                                                    // Thumb (orange)
                                                    (0, 1, egui::Color32::from_rgb(255, 165, 0)),
                                                    (1, 2, egui::Color32::from_rgb(255, 165, 0)),
                                                    (2, 3, egui::Color32::from_rgb(255, 165, 0)),
                                                    (3, 4, egui::Color32::from_rgb(255, 165, 0)),
                                                    // Index (green)
                                                    (0, 5, egui::Color32::GREEN),
                                                    (5, 6, egui::Color32::GREEN),
                                                    (6, 7, egui::Color32::GREEN),
                                                    (7, 8, egui::Color32::GREEN),
                                                    // Middle (cyan)
                                                    (0, 9, egui::Color32::LIGHT_BLUE),
                                                    (9, 10, egui::Color32::LIGHT_BLUE),
                                                    (10, 11, egui::Color32::LIGHT_BLUE),
                                                    (11, 12, egui::Color32::LIGHT_BLUE),
                                                    // Ring (magenta)
                                                    (0, 13, egui::Color32::from_rgb(255, 0, 255)),
                                                    (13, 14, egui::Color32::from_rgb(255, 0, 255)),
                                                    (14, 15, egui::Color32::from_rgb(255, 0, 255)),
                                                    (15, 16, egui::Color32::from_rgb(255, 0, 255)),
                                                    // Pinky (red)
                                                    (0, 17, egui::Color32::RED),
                                                    (17, 18, egui::Color32::RED),
                                                    (18, 19, egui::Color32::RED),
                                                    (19, 20, egui::Color32::RED),
                                                    // Palm connections (white)
                                                    (5, 9, egui::Color32::WHITE),
                                                    (9, 13, egui::Color32::WHITE),
                                                    (13, 17, egui::Color32::WHITE),
                                                ];
                                                
                                                // Draw bones (connections)
                                                for &(from, to, color) in connections {
                                                    if from < landmarks.len() && to < landmarks.len() {
                                                        ui.painter().line_segment(
                                                            [to_screen(from), to_screen(to)],
                                                            egui::Stroke::new(2.0, color),
                                                        );
                                                    }
                                                }
                                                
                                                // Draw joints (landmarks)
                                                for (idx, &(lx, ly)) in landmarks.iter().enumerate() {
                                                    let screen_pos = egui::pos2(
                                                        rect.min.x + lx * scale_x,
                                                        rect.min.y + ly * scale_y,
                                                    );
                                                    
                                                    // Different colors for different landmark types
                                                    let (radius, color) = match idx {
                                                        0 => (5.0, egui::Color32::YELLOW),  // Wrist
                                                        4 | 8 | 12 | 16 | 20 => (4.0, egui::Color32::RED),  // Fingertips
                                                        _ => (3.0, egui::Color32::WHITE),  // Other joints
                                                    };
                                                    
                                                    ui.painter().circle_filled(screen_pos, radius, color);
                                                }
                                                
                                                // Highlight index finger tip with large crosshair
                                                if landmarks.len() > 8 {
                                                    let tip_pos = to_screen(8);
                                                    // Large green circle
                                                    ui.painter().circle_stroke(
                                                        tip_pos,
                                                        15.0,
                                                        egui::Stroke::new(3.0, egui::Color32::GREEN),
                                                    );
                                                    // Crosshair lines
                                                    ui.painter().line_segment(
                                                        [egui::pos2(tip_pos.x - 20.0, tip_pos.y), egui::pos2(tip_pos.x + 20.0, tip_pos.y)],
                                                        egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                    );
                                                    ui.painter().line_segment(
                                                        [egui::pos2(tip_pos.x, tip_pos.y - 20.0), egui::pos2(tip_pos.x, tip_pos.y + 20.0)],
                                                        egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                    );
                                                    
                                                    // Also draw wrist with big yellow marker
                                                    let wrist_pos = to_screen(0);
                                                    ui.painter().circle_filled(wrist_pos, 8.0, egui::Color32::YELLOW);
                                                }
                                            } else if let Some((fx, fy)) = finger_tip {
                                                // Fallback: just draw fingertip if no full landmarks
                                                let roi_offset = {
                                                    let state_lock = state.lock().unwrap();
                                                    state_lock.camera_roi.map(|r| (r.x as f32, r.y as f32)).unwrap_or((0.0, 0.0))
                                                };
                                                let local_x = fx - roi_offset.0;
                                                let local_y = fy - roi_offset.1;
                                                
                                                let scale_x = display_width / skin_width as f32;
                                                let scale_y = skin_display_height / skin_height as f32;
                                                let screen_x = rect.min.x + local_x * scale_x;
                                                let screen_y = rect.min.y + local_y * scale_y;
                                                
                                                ui.painter().circle_stroke(
                                                    egui::pos2(screen_x, screen_y),
                                                    8.0,
                                                    egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                );
                                            }
                                        } else {
                                            ui.colored_label(egui::Color32::GRAY, 
                                                format!("Size mismatch: {} vs {}", data.len(), skin_expected_size));
                                        }
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "No data");
                                    }
                                });
                            });

                            ui.separator();
                            ui.heading("Status");
                            let state_lock = state.lock().unwrap();
                            if let Some(camera_pt) = state_lock.finger_tip_camera {
                                ui.label(format!("Camera: ({:.1}, {:.1})", camera_pt.0, camera_pt.1));
                            } else {
                                ui.colored_label(egui::Color32::GRAY, "Camera: No finger detected");
                            }
                            if let Some(proj_pt) = state_lock.finger_tip_projector {
                                ui.label(format!("Projector: ({:.1}, {:.1})", proj_pt.0, proj_pt.1));
                            }
                            
                            // Show camera ROI status
                            ui.add_space(4.0);
                            if let Some(roi) = &state_lock.camera_roi {
                                ui.colored_label(egui::Color32::GREEN, 
                                    format!("ROI: {}x{} at ({}, {})", roi.width, roi.height, roi.x, roi.y));
                            } else {
                                ui.colored_label(egui::Color32::YELLOW, "ROI: Not calibrated (using full frame)");
                            }
                        });
                    });
                },
            );

            // Update main app state from Arc values
            self.show_camera_feed = *show_camera_feed.lock().unwrap();
            self.circle_radius = *circle_radius.lock().unwrap();
            self.circle_color = *circle_color.lock().unwrap();
            self.projector_width = *projector_width.lock().unwrap();
            self.projector_height = *projector_height.lock().unwrap();
            self.homography_edit = *homography_edit.lock().unwrap();
            self.show_options_window = *show_options.lock().unwrap();
        }

        // Main panel for visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.show_camera_feed {
                // Show camera feed with finger overlay
                let (frame_data, frame_width, frame_height, finger_camera) = {
                    let state = self.state.lock().unwrap();
                    (
                        state.frame.clone(),
                        state.frame_width,
                        state.frame_height,
                        state.finger_tip_camera,
                    )
                };

                if let Some(data) = frame_data {
                    let width = frame_width as usize;
                    let height = frame_height as usize;
                    
                    // Ensure data size matches expected dimensions
                    let expected_size = width * height * 3;
                    if data.len() >= expected_size {
                        // Create color image from frame data
                        let image = egui::ColorImage::from_rgb(
                            [width, height],
                            &data[..expected_size],
                        );

                        // Update or create texture
                        let texture = self.texture.get_or_insert_with(|| {
                            ctx.load_texture(
                                "camera_frame",
                                image.clone(),
                                egui::TextureOptions::LINEAR,
                            )
                        });
                        texture.set(image, egui::TextureOptions::LINEAR);

                        // Calculate display size maintaining aspect ratio
                        let available_size = ui.available_size();
                        let aspect = width as f32 / height as f32;
                        let display_size = if available_size.x / available_size.y > aspect {
                            egui::vec2(available_size.y * aspect, available_size.y)
                        } else {
                            egui::vec2(available_size.x, available_size.x / aspect)
                        };

                        // Center the image
                        let offset_x = (available_size.x - display_size.x) / 2.0;
                        let offset_y = (available_size.y - display_size.y) / 2.0;

                        let (rect, _response) = ui.allocate_exact_size(available_size, egui::Sense::hover());
                        let image_rect = egui::Rect::from_min_size(
                            egui::pos2(rect.min.x + offset_x, rect.min.y + offset_y),
                            display_size,
                        );
                        
                        ui.painter().image(
                            texture.id(),
                            image_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );

                        // Draw finger indicator on camera view
                        if let Some((fx, fy)) = finger_camera {
                            let scale_x = display_size.x / width as f32;
                            let scale_y = display_size.y / height as f32;
                            let screen_x = image_rect.min.x + fx * scale_x;
                            let screen_y = image_rect.min.y + fy * scale_y;
                            
                            // Draw circle with outline for visibility
                            ui.painter().circle_filled(
                                egui::pos2(screen_x, screen_y),
                                self.circle_radius * scale_x.min(scale_y),
                                self.circle_color,
                            );
                            ui.painter().circle_stroke(
                                egui::pos2(screen_x, screen_y),
                                self.circle_radius * scale_x.min(scale_y),
                                egui::Stroke::new(2.0, egui::Color32::WHITE),
                            );
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Frame data size mismatch...");
                        });
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Waiting for camera feed...");
                    });
                }
            } else {
                // Projector output mode - black background with finger circle
                // This is what gets displayed on the projector
                let (finger_projector, calib_state, calib_point, dot_radius, ripples, is_pinching) = {
                    let state = self.state.lock().unwrap();
                    let calib_point = match state.calibration.state {
                        CalibrationState::DisplayingDot { index, .. } |
                        CalibrationState::CapturingDot { index, .. } => {
                            state.calibration.projector_points.get(index).copied()
                        }
                        _ => None
                    };
                    (
                        state.finger_tip_projector,
                        state.calibration.state.clone(),
                        calib_point,
                        state.calibration.dot_radius,
                        state.ripples.clone(),
                        state.is_pinching,
                    )
                };

                let available_size = ui.available_size();
                let (rect, _response) = ui.allocate_exact_size(available_size, egui::Sense::hover());
                
                // Black background (projector output)
                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
                
                // Scale factors for projector to screen conversion
                let scale_x = available_size.x / self.projector_width;
                let scale_y = available_size.y / self.projector_height;

                // Handle calibration display and state transitions
                match &calib_state {
                    CalibrationState::DisplayingDot { index, start_time } => {
                        // Draw the calibration dot
                        if let Some((px, py)) = calib_point {
                            let screen_x = rect.min.x + px * scale_x;
                            let screen_y = rect.min.y + py * scale_y;
                            
                            // Bright white dot for detection
                            ui.painter().circle_filled(
                                egui::pos2(screen_x, screen_y),
                                dot_radius * scale_x.min(scale_y),
                                egui::Color32::WHITE,
                            );
                        }
                        
                        // Transition to capture phase after display time
                        let display_time = {
                            let state = self.state.lock().unwrap();
                            state.calibration.dot_display_time
                        };
                        if start_time.elapsed().as_secs_f32() > display_time {
                            let mut state = self.state.lock().unwrap();
                            state.calibration.accumulated_positions.clear();
                            state.calibration.state = CalibrationState::CapturingDot {
                                index: *index,
                                start_time: Instant::now(),
                            };
                        }
                    }
                    CalibrationState::CapturingDot { index, start_time } => {
                        // Continue displaying the dot
                        if let Some((px, py)) = calib_point {
                            let screen_x = rect.min.x + px * scale_x;
                            let screen_y = rect.min.y + py * scale_y;
                            
                            ui.painter().circle_filled(
                                egui::pos2(screen_x, screen_y),
                                dot_radius * scale_x.min(scale_y),
                                egui::Color32::WHITE,
                            );
                        }
                        
                        // Check if capture time elapsed
                        let capture_time = {
                            let state = self.state.lock().unwrap();
                            state.calibration.dot_capture_time
                        };
                        if start_time.elapsed().as_secs_f32() > capture_time {
                            // Compute robust position using median and outlier rejection
                            let mut state = self.state.lock().unwrap();
                            let num_samples = state.calibration.accumulated_positions.len();
                            let min_samples = state.calibration.minimum_samples;
                            let outlier_threshold = state.calibration.outlier_threshold;
                            
                            if num_samples >= min_samples {
                                if let Some(robust_pos) = compute_robust_position(
                                    &state.calibration.accumulated_positions,
                                    outlier_threshold,
                                ) {
                                    state.calibration.camera_points.push(robust_pos);
                                    
                                    let next_index = index + 1;
                                    if next_index < state.calibration.projector_points.len() {
                                        // Move to next point
                                        state.calibration.state = CalibrationState::DisplayingDot {
                                            index: next_index,
                                            start_time: Instant::now(),
                                        };
                                    } else {
                                        // All points captured, compute homography
                                        state.calibration.state = CalibrationState::Computing;
                                    }
                                } else {
                                    // Robust computation failed (shouldn't happen)
                                    state.calibration.state = CalibrationState::Complete {
                                        success: false,
                                        message: format!("Failed to compute position for dot {}", index + 1),
                                    };
                                }
                            } else {
                                // Not enough samples - failed
                                state.calibration.state = CalibrationState::Complete {
                                    success: false,
                                    message: format!(
                                        "Failed to detect dot {} - only {} samples (need {})",
                                        index + 1, num_samples, min_samples
                                    ),
                                };
                            }
                        }
                    }
                    CalibrationState::Computing => {
                        let mut state = self.state.lock().unwrap();
                        match state.compute_homography() {
                            Ok(()) => {
                                // Save the new homography to file
                                if let Err(e) = save_homography(&state.homography) {
                                    log::error!("Failed to save homography: {}", e);
                                }
                                // Save the camera ROI to file
                                if let Some(ref roi) = state.camera_roi {
                                    if let Err(e) = save_camera_roi(roi) {
                                        log::error!("Failed to save camera ROI: {}", e);
                                    }
                                }
                                state.calibration.state = CalibrationState::Complete {
                                    success: true,
                                    message: format!(
                                        "Successfully computed homography from {} points (saved)",
                                        state.calibration.camera_points.len()
                                    ),
                                };
                            }
                            Err(e) => {
                                state.calibration.state = CalibrationState::Complete {
                                    success: false,
                                    message: e,
                                };
                            }
                        }
                    }
                    _ => {
                        // Normal mode or waiting - draw finger if detected
                        if let Some((px, py)) = finger_projector {
                            let screen_x = rect.min.x + px * scale_x;
                            let screen_y = rect.min.y + py * scale_y;

                            // Draw the finger indicator circle
                            // Change color if pinching
                            let finger_color = if is_pinching {
                                egui::Color32::from_rgb(100, 255, 100) // Green when pinching
                            } else {
                                self.circle_color
                            };
                            
                            ui.painter().circle_filled(
                                egui::pos2(screen_x, screen_y),
                                self.circle_radius,
                                finger_color,
                            );
                            ui.painter().circle_stroke(
                                egui::pos2(screen_x, screen_y),
                                self.circle_radius,
                                egui::Stroke::new(3.0, egui::Color32::WHITE),
                            );
                        }
                        
                        // Draw ripple effects
                        for ripple in &ripples {
                            let elapsed = ripple.start_time.elapsed().as_secs_f32();
                            let progress = (elapsed / ripple.duration).min(1.0);
                            
                            // Convert projector coords to screen coords
                            let screen_x = rect.min.x + ripple.center.0 * scale_x;
                            let screen_y = rect.min.y + ripple.center.1 * scale_y;
                            
                            // Draw multiple concentric rings for wave effect
                            let num_rings = 3;
                            for i in 0..num_rings {
                                let ring_offset = (i as f32 / num_rings as f32) * 0.3; // Spread rings
                                let ring_progress = (progress - ring_offset).max(0.0).min(1.0);
                                
                                if ring_progress > 0.0 {
                                    let ring_radius = ring_progress * ripple.max_radius * scale_x.min(scale_y);
                                    let ring_alpha = ((1.0 - ring_progress) * 200.0) as u8;
                                    
                                    let color = egui::Color32::from_rgba_unmultiplied(
                                        ripple.color.r(),
                                        ripple.color.g(),
                                        ripple.color.b(),
                                        ring_alpha,
                                    );
                                    
                                    // Stroke width decreases as ring expands
                                    let stroke_width = (1.0 - ring_progress) * 4.0 + 1.0;
                                    
                                    ui.painter().circle_stroke(
                                        egui::pos2(screen_x, screen_y),
                                        ring_radius,
                                        egui::Stroke::new(stroke_width, color),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Signal camera thread to stop
        let mut state = self.state.lock().unwrap();
        state.running = false;
    }
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Camera device path - change this if your camera is at a different device
    let device_path = "/dev/video5";

    log::info!("Starting Finger Tracker");
    log::info!("Camera device: {}", device_path);

    // Create shared state
    let state = Arc::new(Mutex::new(SharedState::new()));
    
    // Start camera thread
    let camera_state = state.clone();
    let device = device_path.to_string();
    thread::spawn(move || {
        camera_thread(camera_state, &device);
    });

    // Run GUI application
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("Finger Tracker - Camera/Projector")
            .with_maximized(false),
        ..Default::default()
    };

    eframe::run_native(
        "Finger Tracker",
        options,
        Box::new(|_cc| Ok(Box::new(FingerTrackerApp::new(state)))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;

    Ok(())
}

