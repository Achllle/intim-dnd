mod hand_tracker;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use eframe::egui;
use nalgebra::{Matrix3, Vector3};
use opencv::{
    calib3d,
    core::{self, Mat, Point, Point2f, Size, Vector},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, CAP_V4L2},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Load Gemini API token from file
fn load_gemini_api_token() -> Option<String> {
    let paths = [
        PathBuf::from("gemini_api_token.txt"),
        dirs::config_dir().map(|d| d.join("intim-dnd/gemini_api_token.txt")).unwrap_or_default(),
        dirs::home_dir().map(|d| d.join(".gemini_api_token")).unwrap_or_default(),
    ];
    
    for path in &paths {
        if path.exists() {
            if let Ok(token) = fs::read_to_string(path) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    log::info!("Loaded Gemini API token from {:?}", path);
                    return Some(token);
                }
            }
        }
    }
    
    log::warn!("Gemini API token not found. Create gemini_api_token.txt with your API key.");
    None
}

/// Gemini API response structures
#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Deserialize, Debug)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    inline_data: Option<GeminiInlineData>,
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

/// Image generation status
#[derive(Clone, PartialEq)]
enum ImageGenStatus {
    Idle,
    Generating,
    Success(String),
    Error(String),
}

/// Character drag-and-drop state
#[derive(Clone, PartialEq)]
enum DragState {
    /// Not dragging anything
    Idle,
    /// Pinch detected, waiting for initial 300ms to show loading bar
    PinchStarted {
        start_time: Instant,
        position: (f32, f32), // projector coordinates
    },
    /// Loading bar showing (300ms-1300ms), waiting for pickup
    PickingUp {
        start_time: Instant,
        position: (f32, f32),
        character_index: usize,
    },
    /// Character is being dragged
    Dragging {
        character_index: usize,
        original_pos: (u32, u32),
        current_pos: (f32, f32), // projector coordinates
    },
    /// Character is over a cell, waiting for drop confirmation
    DroppingOff {
        start_time: Instant,
        character_index: usize,
        original_pos: (u32, u32),
        target_cell: (u32, u32),
        position: (f32, f32),
    },
}

/// Default drag-and-drop timing values (in milliseconds)
const DEFAULT_PINCH_SHOW_LOADING_MS: u64 = 250;    // Time before showing loading bar
const DEFAULT_PINCH_PICKUP_MS: u64 = 300;          // Additional time to complete pickup
const DEFAULT_PINCH_FLICKER_TOLERANCE_MS: u64 = 445; // Tolerance for tracking flicker
const DEFAULT_DROP_CONFIRM_MS: u64 = 810;           // Time to confirm drop

/// Default hover timing values (in milliseconds)
const DEFAULT_HOVER_SHOW_INFO_MS: u64 = 500;        // Time finger must hover to show info panel
const DEFAULT_HOVER_INFO_DISPLAY_MS: u64 = 5000;    // How long to display the info panel

/// Persistent application settings
#[derive(Serialize, Deserialize, Clone)]
struct AppSettings {
    /// Circle radius for finger indicator
    circle_radius: f32,
    /// Circle color (RGBA)
    circle_color: [u8; 4],
    /// Projector output width
    projector_width: f32,
    /// Projector output height
    projector_height: f32,
    /// Whether to show grid overlay
    show_grid: bool,
    /// Number of rows in the grid
    grid_rows: u32,
    /// Drag timing: time before showing loading bar (ms)
    pinch_show_loading_ms: u64,
    /// Drag timing: additional time to complete pickup (ms)
    pinch_pickup_ms: u64,
    /// Drag timing: tolerance for tracking flicker (ms)
    pinch_flicker_tolerance_ms: u64,
    /// Drag timing: time to confirm drop (ms)
    drop_confirm_ms: u64,
    /// Hover timing: time finger must hover to show info panel (ms)
    hover_show_info_ms: u64,
    /// Hover timing: how long to display the info panel (ms)
    hover_info_display_ms: u64,
    /// Path to the current background image (if any)
    background_image_path: Option<String>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            circle_radius: 14.0,
            circle_color: [100, 225, 255, 255],
            projector_width: 1920.0,
            projector_height: 1080.0,
            show_grid: true,
            grid_rows: 8,
            pinch_show_loading_ms: DEFAULT_PINCH_SHOW_LOADING_MS,
            pinch_pickup_ms: DEFAULT_PINCH_PICKUP_MS,
            pinch_flicker_tolerance_ms: DEFAULT_PINCH_FLICKER_TOLERANCE_MS,
            drop_confirm_ms: DEFAULT_DROP_CONFIRM_MS,
            hover_show_info_ms: DEFAULT_HOVER_SHOW_INFO_MS,
            hover_info_display_ms: DEFAULT_HOVER_INFO_DISPLAY_MS,
            background_image_path: None,
        }
    }
}

/// Get the path to the settings file
fn get_settings_file_path() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("intim-dnd/settings.json"))
        .unwrap_or_else(|| PathBuf::from("settings.json"))
}

/// Load application settings from file
fn load_settings() -> AppSettings {
    let path = get_settings_file_path();
    match fs::read_to_string(&path) {
        Ok(content) => {
            match serde_json::from_str(&content) {
                Ok(settings) => {
                    log::info!("Loaded settings from {:?}", path);
                    settings
                }
                Err(e) => {
                    log::warn!("Failed to parse settings: {}, using defaults", e);
                    AppSettings::default()
                }
            }
        }
        Err(_) => {
            log::info!("No settings file found, using defaults");
            AppSettings::default()
        }
    }
}

/// Save application settings to file
fn save_settings(settings: &AppSettings) {
    let path = get_settings_file_path();
    
    // Ensure config directory exists
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    
    match serde_json::to_string_pretty(settings) {
        Ok(content) => {
            if let Err(e) = fs::write(&path, content) {
                log::error!("Failed to save settings to {:?}: {}", path, e);
            } else {
                log::debug!("Saved settings to {:?}", path);
            }
        }
        Err(e) => {
            log::error!("Failed to serialize settings: {}", e);
        }
    }
}

/// A generated image entry for the gallery
#[derive(Clone)]
struct GeneratedImage {
    /// Path to the image file
    path: PathBuf,
    /// The prompt used to generate this image
    prompt: String,
}

impl GeneratedImage {
    fn new(path: PathBuf, prompt: String) -> Self {
        Self {
            path,
            prompt,
        }
    }
    
    /// Generate a filename from a prompt (sanitized)
    fn filename_from_prompt(prompt: &str) -> String {
        let sanitized: String = prompt
            .chars()
            .take(40)
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
            .collect::<String>()
            .trim()
            .replace(' ', "_")
            .to_lowercase();
        
        // Add timestamp for uniqueness
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        format!("{}_{}.png", sanitized, timestamp)
    }
}

/// Weapon data from YAML
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Weapon {
    name: String,
    damage: String,
    modifier: i32,
}

/// Character data from YAML config
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CharacterConfig {
    name: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    dead: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    race: String,
    class: String,
    armor_class: i32,
    hit_points: i32,
    health: i32,
    token_representation: String,
    weapons: Vec<Weapon>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    current_weapon: String,
}

/// Enemy data from YAML config
#[derive(Serialize, Deserialize, Debug, Clone)]
struct EnemyConfig {
    name: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    dead: bool,
    #[serde(rename = "type")]
    enemy_type: String,
    armor_class: i32,
    hit_points: i32,
    health: i32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    token_representation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weapons: Vec<Weapon>,
}

/// Characters config file structure
#[derive(Serialize, Deserialize, Debug)]
struct CharactersConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    characters: Vec<CharacterConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    enemies: Vec<EnemyConfig>,
}

/// Entity type enum - can be either a character (player) or enemy
#[derive(Clone)]
enum EntityType {
    Character(CharacterConfig),
    Enemy(EnemyConfig),
}

/// Runtime entity state (includes position and visibility)
/// Can represent either a character or an enemy
#[derive(Clone)]
struct Character {
    entity: EntityType,
    /// Grid position (column, row)
    grid_pos: (u32, u32),
    /// Whether the character is visible on the board
    visible: bool,
    /// Loaded token texture
    token_texture: Option<egui::TextureHandle>,
    /// Token image data
    token_image: Option<egui::ColorImage>,
}

impl Character {
    fn from_character(config: CharacterConfig) -> Self {
        let token_path = dirs::config_dir()
            .map(|d| d.join(format!("intim-dnd/token_reps/{}", config.token_representation)))
            .unwrap_or_else(|| PathBuf::from(format!("token_reps/{}", config.token_representation)));
        let token_image = Self::load_token_image(token_path.to_str().unwrap_or(""));
        
        Self {
            entity: EntityType::Character(config),
            grid_pos: (0, 0),
            visible: true,
            token_texture: None,
            token_image,
        }
    }
    
    fn from_enemy(config: EnemyConfig) -> Self {
        let token_path = if !config.token_representation.is_empty() {
            dirs::config_dir()
                .map(|d| d.join(format!("intim-dnd/token_reps/{}", config.token_representation)))
                .unwrap_or_else(|| PathBuf::from(format!("token_reps/{}", config.token_representation)))
        } else {
            PathBuf::new()
        };
        let token_image = Self::load_token_image(token_path.to_str().unwrap_or(""));
        
        Self {
            entity: EntityType::Enemy(config),
            grid_pos: (0, 0),
            visible: true,
            token_texture: None,
            token_image,
        }
    }
    
    fn load_token_image(path: &str) -> Option<egui::ColorImage> {
        let path = std::path::Path::new(path);
        if !path.exists() || path.to_str().map(|s| s.is_empty()).unwrap_or(true) {
            log::warn!("Token image not found at {:?}", path);
            return None;
        }
        
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                log::info!("Loaded token image: {:?}", path);
                Some(egui::ColorImage::from_rgba_unmultiplied(size, &pixels))
            }
            Err(e) => {
                log::error!("Failed to load token image {:?}: {}", path, e);
                None
            }
        }
    }
    
    /// Get the entity's name
    fn name(&self) -> &str {
        match &self.entity {
            EntityType::Character(c) => &c.name,
            EntityType::Enemy(e) => &e.name,
        }
    }
    
    /// Get the entity's hit points
    fn hit_points(&self) -> i32 {
        match &self.entity {
            EntityType::Character(c) => c.hit_points,
            EntityType::Enemy(e) => e.hit_points,
        }
    }
    
    /// Get the entity's current health
    fn health(&self) -> i32 {
        match &self.entity {
            EntityType::Character(c) => c.health,
            EntityType::Enemy(e) => e.health,
        }
    }
    
    /// Check if the entity is dead
    fn is_dead(&self) -> bool {
        match &self.entity {
            EntityType::Character(c) => c.dead,
            EntityType::Enemy(e) => e.dead,
        }
    }
    
    /// Check if this is an enemy
    fn is_enemy(&self) -> bool {
        matches!(self.entity, EntityType::Enemy(_))
    }
    
    /// Health percentage (0.0 - 1.0)
    fn health_percentage(&self) -> f32 {
        let hp = self.hit_points();
        if hp <= 0 {
            return 0.0;
        }
        (self.health() as f32 / hp as f32).clamp(0.0, 1.0)
    }
    
    /// Modify the entity's health by a delta (can be positive or negative)
    fn modify_health(&mut self, delta: i32) {
        match &mut self.entity {
            EntityType::Character(c) => {
                c.health = (c.health + delta).max(0).min(c.hit_points);
            }
            EntityType::Enemy(e) => {
                e.health = (e.health + delta).max(0).min(e.hit_points);
            }
        }
    }
}

/// Load characters and enemies from YAML config file
fn load_characters() -> Vec<Character> {
    let config_path = dirs::config_dir()
        .map(|d| d.join("intim-dnd/characters.yaml"))
        .unwrap_or_else(|| PathBuf::from("characters.yaml"));
    
    let content = match fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("Failed to read characters config: {}", e);
            return Vec::new();
        }
    };
    
    let config: CharactersConfig = match serde_yaml::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to parse characters config: {}", e);
            return Vec::new();
        }
    };
    
    let mut entities: Vec<Character> = Vec::new();
    
    // Load characters first
    for char_config in config.characters {
        entities.push(Character::from_character(char_config));
    }
    
    // Then load enemies
    for enemy_config in config.enemies {
        entities.push(Character::from_enemy(enemy_config));
    }
    
    log::info!("Loaded {} characters and enemies", entities.len());
    entities
}

/// Save characters and enemies to YAML config file
fn save_characters(characters: &[Character]) {
    let config_path = dirs::config_dir()
        .map(|d| d.join("intim-dnd/characters.yaml"))
        .unwrap_or_else(|| PathBuf::from("characters.yaml"));
    
    // Separate characters and enemies
    let mut char_configs: Vec<CharacterConfig> = Vec::new();
    let mut enemy_configs: Vec<EnemyConfig> = Vec::new();
    
    for char in characters {
        match &char.entity {
            EntityType::Character(c) => char_configs.push(c.clone()),
            EntityType::Enemy(e) => enemy_configs.push(e.clone()),
        }
    }
    
    let config = CharactersConfig {
        characters: char_configs,
        enemies: enemy_configs,
    };
    
    match serde_yaml::to_string(&config) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(&config_path, yaml) {
                log::error!("Failed to save characters config: {}", e);
            } else {
                log::info!("Saved characters to {:?}", config_path);
            }
        }
        Err(e) => {
            log::error!("Failed to serialize characters config: {}", e);
        }
    }
}

/// Generate an image using Gemini API
fn generate_image_with_gemini(api_token: &str, prompt: &str) -> Result<Vec<u8>, String> {
    // Use Gemini image generation model
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={}",
        api_token
    );
    
    let request_body = serde_json::json!({
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    });
    
    log::info!("Sending image generation request to Gemini API...");
    
    let response = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&request_body)
        .map_err(|e| format!("API request failed: {}", e))?;
    
    let response_text = response.into_string()
        .map_err(|e| format!("Failed to read response: {}", e))?;
    
    log::debug!("Gemini API response: {}", &response_text[..500.min(response_text.len())]);
    
    // Parse Gemini response
    let gemini_response: GeminiResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse response: {} - {}", e, &response_text[..200.min(response_text.len())]))?;
    
    // Extract image data from response
    if let Some(candidates) = gemini_response.candidates {
        for candidate in candidates {
            if let Some(content) = candidate.content {
                if let Some(parts) = content.parts {
                    for part in parts {
                        if let Some(inline_data) = part.inline_data {
                            if inline_data.mime_type.starts_with("image/") {
                                let image_bytes = BASE64.decode(&inline_data.data)
                                    .map_err(|e| format!("Failed to decode image: {}", e))?;
                                log::info!("Successfully generated image ({} bytes, mime: {})", 
                                    image_bytes.len(), inline_data.mime_type);
                                
                                // Convert to PNG using image crate for consistent format
                                let img = image::load_from_memory(&image_bytes)
                                    .map_err(|e| format!("Failed to decode image data: {}", e))?;
                                let mut png_bytes = Vec::new();
                                img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
                                    .map_err(|e| format!("Failed to encode as PNG: {}", e))?;
                                
                                log::info!("Converted to PNG ({} bytes)", png_bytes.len());
                                return Ok(png_bytes);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Err(format!("No image found in response: {}", &response_text[..300.min(response_text.len())]))
}

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

/// Get the path to the homography calibration file
fn get_homography_file_path() -> PathBuf {
    // Store in user's config directory or current directory
    if let Some(config_dir) = dirs::config_dir() {
        let app_dir = config_dir.join("intim-dnd");
        let _ = fs::create_dir_all(&app_dir);
        app_dir.join("homography.txt")
    } else {
        PathBuf::from("homography.txt")
    }
}

/// Get the path to the camera ROI calibration file
fn get_roi_file_path() -> PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        let app_dir = config_dir.join("intim-dnd");
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
    /// Debug: cropped region frame for hand tracking visualization
    debug_hand_tracking_frame: Option<Vec<u8>>,
    /// Dimensions of the hand tracking cropped frame (may differ from full frame when using ROI)
    debug_hand_tracking_width: u32,
    debug_hand_tracking_height: u32,
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
            debug_hand_tracking_frame: None,
            debug_hand_tracking_width: 640,
            debug_hand_tracking_height: 480,
            calibration_brightness_threshold: 200,
            camera_roi,
            hand_landmarks: None,
            is_pinching: false,
            pinch_center_camera: None,
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
                    state.debug_hand_tracking_frame = debug_frame;
                    state.debug_hand_tracking_width = debug_w;
                    state.debug_hand_tracking_height = debug_h;
                    
                    // Update hand landmarks (for ML mode visualization)
                    state.hand_landmarks = hand_landmarks;
                    
                    // Handle pinch detection
                    state.is_pinching = pinch_center.is_some();
                    state.pinch_center_camera = pinch_center;
                    
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
struct IntImDnDApp {
    state: Arc<Mutex<SharedState>>,
    texture: Option<egui::TextureHandle>,
    /// Background image for projector output
    background_texture: Option<egui::TextureHandle>,
    /// Background image data (loaded once)
    background_image: Option<egui::ColorImage>,
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
    /// Gemini API token
    gemini_api_token: Option<String>,
    /// Image generation prompt
    image_gen_prompt: String,
    /// Image generation status
    image_gen_status: Arc<Mutex<ImageGenStatus>>,
    /// Gallery of generated images (last 5)
    generated_images: Arc<Mutex<Vec<GeneratedImage>>>,
    /// Currently selected image index in gallery
    selected_image_index: Arc<Mutex<Option<usize>>>,
    /// Whether to show grid overlay
    show_grid: bool,
    /// Number of rows in the grid (5-14)
    grid_rows: u32,
    /// Characters on the board
    characters: Arc<Mutex<Vec<Character>>>,
    /// Current drag-and-drop state
    drag_state: DragState,
    /// Last time pinch was detected (for flicker tolerance)
    last_pinch_detected: Option<Instant>,
    /// Last pinch position (for flicker tolerance)
    last_pinch_position: Option<(f32, f32)>,
    /// Drag-and-drop timing: time before showing loading bar (ms)
    pinch_show_loading_ms: u64,
    /// Drag-and-drop timing: additional time to complete pickup (ms)
    pinch_pickup_ms: u64,
    /// Drag-and-drop timing: tolerance for tracking flicker (ms)
    pinch_flicker_tolerance_ms: u64,
    /// Drag-and-drop timing: time to confirm drop (ms)
    drop_confirm_ms: u64,
    /// Current background image path (for saving to settings)
    current_background_path: Option<String>,
    /// Hover timing: time finger must hover to show info panel (ms)
    hover_show_info_ms: u64,
    /// Hover timing: how long to display the info panel (ms)
    hover_info_display_ms: u64,
    /// Hover state: character being hovered and when hover started
    hover_state: Option<(usize, Instant)>,
    /// Info panel state: character to show info for and when to hide
    info_panel_state: Option<(usize, Instant)>,
    /// Whether to show the calibration settings window
    show_calibration_window: bool,
}

impl IntImDnDApp {
    fn new(state: Arc<Mutex<SharedState>>) -> Self {
        // Load saved settings
        let settings = load_settings();
        
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
        
        // Load background image (from saved path or default)
        let background_image = if let Some(ref path) = settings.background_image_path {
            Self::load_background_image(path).or_else(|| Self::load_background_image("assets/default.png"))
        } else {
            Self::load_background_image("assets/default.png")
        };
        
        // Load Gemini API token
        let gemini_api_token = load_gemini_api_token();
        
        // Load existing generated images from assets/generated folder
        let generated_images = Self::load_existing_generated_images();
        
        // Load characters and position them in the middle of the grid
        let mut characters = load_characters();
        let middle_row = settings.grid_rows / 2;
        let middle_col = settings.grid_rows / 2; // Approximate middle column
        for (i, char) in characters.iter_mut().enumerate() {
            // Place characters adjacent to each other in the middle
            char.grid_pos = (middle_col + i as u32, middle_row);
        }
        
        Self {
            state,
            texture: None,
            background_texture: None,
            background_image,
            show_camera_feed: false,
            circle_radius: settings.circle_radius,
            circle_color: egui::Color32::from_rgba_unmultiplied(
                settings.circle_color[0],
                settings.circle_color[1],
                settings.circle_color[2],
                settings.circle_color[3],
            ),
            projector_width: settings.projector_width,
            projector_height: settings.projector_height,
            homography_edit,
            show_options_window: true,
            gemini_api_token,
            image_gen_prompt: String::new(),
            image_gen_status: Arc::new(Mutex::new(ImageGenStatus::Idle)),
            generated_images: Arc::new(Mutex::new(generated_images)),
            selected_image_index: Arc::new(Mutex::new(None)),
            show_grid: settings.show_grid,
            grid_rows: settings.grid_rows,
            characters: Arc::new(Mutex::new(characters)),
            drag_state: DragState::Idle,
            last_pinch_detected: None,
            last_pinch_position: None,
            pinch_show_loading_ms: settings.pinch_show_loading_ms,
            pinch_pickup_ms: settings.pinch_pickup_ms,
            pinch_flicker_tolerance_ms: settings.pinch_flicker_tolerance_ms,
            drop_confirm_ms: settings.drop_confirm_ms,
            current_background_path: settings.background_image_path,
            hover_show_info_ms: settings.hover_show_info_ms,
            hover_info_display_ms: settings.hover_info_display_ms,
            hover_state: None,
            info_panel_state: None,
            show_calibration_window: false,
        }
    }
    
    /// Save current settings to file
    fn save_current_settings(&self) {
        let color = self.circle_color.to_array();
        let settings = AppSettings {
            circle_radius: self.circle_radius,
            circle_color: color,
            projector_width: self.projector_width,
            projector_height: self.projector_height,
            show_grid: self.show_grid,
            grid_rows: self.grid_rows,
            pinch_show_loading_ms: self.pinch_show_loading_ms,
            pinch_pickup_ms: self.pinch_pickup_ms,
            pinch_flicker_tolerance_ms: self.pinch_flicker_tolerance_ms,
            drop_confirm_ms: self.drop_confirm_ms,
            hover_show_info_ms: self.hover_show_info_ms,
            hover_info_display_ms: self.hover_info_display_ms,
            background_image_path: self.current_background_path.clone(),
        };
        save_settings(&settings);
    }
    
    /// Load a background image from file
    fn load_background_image(path: &str) -> Option<egui::ColorImage> {
        let path = std::path::Path::new(path);
        if !path.exists() {
            log::warn!("Background image not found at {:?}", path);
            return None;
        }
        
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                log::info!("Loaded background image: {}x{}", size[0], size[1]);
                Some(egui::ColorImage::from_rgba_unmultiplied(size, &pixels))
            }
            Err(e) => {
                log::error!("Failed to load background image: {}", e);
                None
            }
        }
    }
    
    /// Load default background image
    fn load_default_background(&mut self) {
        self.background_image = Self::load_background_image("assets/cave-map.jpg");
        // Clear the texture so it gets recreated with new image
        self.background_texture = None;
        // Update current path and save settings
        self.current_background_path = Some("assets/cave-map.jpg".to_string());
        self.save_current_settings();
    }
    
    /// Load a specific image as background
    fn load_image_as_background(&mut self, path: &std::path::Path) {
        let path_str = path.to_str().unwrap_or("");
        self.background_image = Self::load_background_image(path_str);
        // Clear the texture so it gets recreated with new image
        self.background_texture = None;
        // Update current path and save settings
        self.current_background_path = Some(path_str.to_string());
        self.save_current_settings();
    }
    
    /// Load existing generated images from the config generated folder
    fn load_existing_generated_images() -> Vec<GeneratedImage> {
        let generated_dir = dirs::config_dir()
            .map(|d| d.join("intim-dnd/generated"))
            .unwrap_or_else(|| PathBuf::from("generated"));
        let mut images = Vec::new();
        
        if generated_dir.exists() {
            if let Ok(entries) = fs::read_dir(&generated_dir) {
                let mut entries: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path().extension()
                            .map(|ext| ext == "png")
                            .unwrap_or(false)
                    })
                    .collect();
                
                // Sort by modification time (newest first)
                entries.sort_by(|a, b| {
                    let a_time = a.metadata().and_then(|m| m.modified()).ok();
                    let b_time = b.metadata().and_then(|m| m.modified()).ok();
                    b_time.cmp(&a_time)
                });
                
                // Take only the last 5
                for entry in entries.into_iter().take(5) {
                    let path = entry.path();
                    // Extract prompt from filename (remove timestamp and extension)
                    let prompt = path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| {
                            // Remove the timestamp suffix (last part after _)
                            if let Some(idx) = s.rfind('_') {
                                s[..idx].replace('_', " ")
                            } else {
                                s.replace('_', " ")
                            }
                        })
                        .unwrap_or_else(|| "Unknown".to_string());
                    
                    images.push(GeneratedImage::new(path, prompt));
                }
            }
        }
        
        log::info!("Loaded {} existing generated images", images.len());
        images
    }
    
    /// Add a new generated image to the gallery
    fn add_generated_image(images: &mut Vec<GeneratedImage>, path: PathBuf, prompt: String) {
        // Add new image at the beginning
        images.insert(0, GeneratedImage::new(path, prompt));
        
        // Keep only the last 5 images
        while images.len() > 5 {
            images.pop();
        }
    }
    
    /// Convert projector coordinates to grid cell (column, row)
    fn projector_to_grid_cell(&self, px: f32, py: f32, rect: &egui::Rect) -> Option<(u32, u32)> {
        let cell_size = rect.height() / self.grid_rows as f32;
        let num_cols = (rect.width() / cell_size).ceil() as u32;
        
        // Convert to relative coordinates within the rect
        let rel_x = px - rect.min.x;
        let rel_y = py - rect.min.y;
        
        if rel_x < 0.0 || rel_y < 0.0 {
            return None;
        }
        
        let col = (rel_x / cell_size) as u32;
        let row = (rel_y / cell_size) as u32;
        
        if col < num_cols && row < self.grid_rows {
            Some((col, row))
        } else {
            None
        }
    }
    
    /// Find character at a given grid cell
    fn find_character_at_cell(&self, cell: (u32, u32)) -> Option<usize> {
        let chars = self.characters.lock().unwrap();
        for (i, char) in chars.iter().enumerate() {
            if char.visible && char.grid_pos == cell {
                return Some(i);
            }
        }
        None
    }
    
    /// Draw a circular loading indicator
    fn draw_loading_circle(
        painter: &egui::Painter,
        center: egui::Pos2,
        radius: f32,
        progress: f32, // 0.0 to 1.0
        color: egui::Color32,
        stroke_width: f32,
    ) {
        use std::f32::consts::PI;
        
        // Background circle (dim)
        painter.circle_stroke(
            center,
            radius,
            egui::Stroke::new(stroke_width, egui::Color32::from_rgba_unmultiplied(
                color.r(), color.g(), color.b(), 60
            )),
        );
        
        // Progress arc
        let start_angle = -PI / 2.0; // Start from top
        let end_angle = start_angle + 2.0 * PI * progress;
        
        // Draw arc as small line segments
        let segments = 32;
        let angle_step = (end_angle - start_angle) / segments as f32;
        
        for i in 0..segments {
            let a1 = start_angle + i as f32 * angle_step;
            let a2 = start_angle + (i + 1) as f32 * angle_step;
            
            if a2 > end_angle {
                break;
            }
            
            let p1 = egui::pos2(
                center.x + radius * a1.cos(),
                center.y + radius * a1.sin(),
            );
            let p2 = egui::pos2(
                center.x + radius * a2.cos(),
                center.y + radius * a2.sin(),
            );
            
            painter.line_segment([p1, p2], egui::Stroke::new(stroke_width, color));
        }
    }
}

fn options_viewport_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("options_viewport")
}

fn calibration_viewport_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("calibration_viewport")
}

impl eframe::App for IntImDnDApp {
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
            let image_gen_prompt = Arc::new(Mutex::new(self.image_gen_prompt.clone()));
            let image_gen_status = self.image_gen_status.clone();
            let gemini_api_token = self.gemini_api_token.clone();
            let use_default_background = Arc::new(Mutex::new(false));
            let generated_images = self.generated_images.clone();
            let selected_image_index = self.selected_image_index.clone();
            let load_selected_image = Arc::new(Mutex::new(false));
            let selected_image_path = Arc::new(Mutex::new(None::<PathBuf>));
            let show_grid = Arc::new(Mutex::new(self.show_grid));
            let grid_rows = Arc::new(Mutex::new(self.grid_rows));
            let characters = self.characters.clone();
            let pinch_show_loading_ms = Arc::new(Mutex::new(self.pinch_show_loading_ms));
            let pinch_pickup_ms = Arc::new(Mutex::new(self.pinch_pickup_ms));
            let pinch_flicker_tolerance_ms = Arc::new(Mutex::new(self.pinch_flicker_tolerance_ms));
            let drop_confirm_ms = Arc::new(Mutex::new(self.drop_confirm_ms));
            let hover_show_info_ms = Arc::new(Mutex::new(self.hover_show_info_ms));
            let hover_info_display_ms = Arc::new(Mutex::new(self.hover_info_display_ms));
            let show_calibration_window = Arc::new(Mutex::new(self.show_calibration_window));

            // Clone Arcs for the closure
            let show_camera_feed_c = show_camera_feed.clone();
            let circle_radius_c = circle_radius.clone();
            let circle_color_c = circle_color.clone();
            let projector_width_c = projector_width.clone();
            let projector_height_c = projector_height.clone();
            let _homography_edit_c = homography_edit.clone();
            let show_options_c = show_options.clone();
            let image_gen_prompt_c = image_gen_prompt.clone();
            let image_gen_status_c = image_gen_status.clone();
            let use_default_background_c = use_default_background.clone();
            let generated_images_c = generated_images.clone();
            let selected_image_index_c = selected_image_index.clone();
            let load_selected_image_c = load_selected_image.clone();
            let selected_image_path_c = selected_image_path.clone();
            let show_grid_c = show_grid.clone();
            let grid_rows_c = grid_rows.clone();
            let characters_c = characters.clone();
            let pinch_show_loading_ms_c = pinch_show_loading_ms.clone();
            let pinch_pickup_ms_c = pinch_pickup_ms.clone();
            let pinch_flicker_tolerance_ms_c = pinch_flicker_tolerance_ms.clone();
            let drop_confirm_ms_c = drop_confirm_ms.clone();
            let hover_show_info_ms_c = hover_show_info_ms.clone();
            let hover_info_display_ms_c = hover_info_display_ms.clone();
            let show_calibration_window_c = show_calibration_window.clone();

            ctx.show_viewport_immediate(
                options_viewport_id(),
                egui::ViewportBuilder::default()
                    .with_title("IntIm-DnD - Options")
                    .with_inner_size([350.0, 650.0]),
                |ctx, _class| {
                    // Handle window close request
                    if ctx.input(|i| i.viewport().close_requested()) {
                        *show_options_c.lock().unwrap() = false;
                    }

                    egui::CentralPanel::default().show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.heading("IntIm-DnD Options");
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
                            ui.heading("Grid Overlay");
                            {
                                let mut show = show_grid_c.lock().unwrap();
                                ui.checkbox(&mut *show, "Show Grid");
                            }
                            {
                                let mut rows = grid_rows_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *rows, 5..=14).text("Grid Rows"));
                            }

                            ui.separator();
                            ui.heading("🎭 Characters");
                            {
                                let mut chars = characters_c.lock().unwrap();
                                let mut save_needed = false;
                                
                                if chars.is_empty() {
                                    ui.colored_label(egui::Color32::GRAY, "No characters loaded");
                                    ui.small("Add characters to ~/.config/intim-dnd/characters.yaml");
                                } else {
                                    for char in chars.iter_mut() {
                                        ui.horizontal(|ui| {
                                            // Visibility checkbox
                                            ui.checkbox(&mut char.visible, "");
                                            
                                            // Token thumbnail
                                            let thumb_size = egui::vec2(32.0, 32.0);
                                            if let Some(ref token_img) = char.token_image {
                                                let texture = ctx.load_texture(
                                                    format!("char_thumb_{}", char.name()),
                                                    token_img.clone(),
                                                    egui::TextureOptions::LINEAR,
                                                );
                                                let (rect, _) = ui.allocate_exact_size(thumb_size, egui::Sense::hover());
                                                ui.painter().image(
                                                    texture.id(),
                                                    rect,
                                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                    egui::Color32::WHITE,
                                                );
                                            } else {
                                                // Placeholder if no token
                                                let (rect, _) = ui.allocate_exact_size(thumb_size, egui::Sense::hover());
                                                let bg_color = if char.is_enemy() {
                                                    egui::Color32::from_rgb(80, 40, 40) // Red tint for enemies
                                                } else {
                                                    egui::Color32::from_gray(60)
                                                };
                                                ui.painter().rect_filled(rect, 4.0, bg_color);
                                                ui.painter().text(
                                                    rect.center(),
                                                    egui::Align2::CENTER_CENTER,
                                                    &char.name().chars().next().unwrap_or('?').to_string(),
                                                    egui::FontId::default(),
                                                    egui::Color32::WHITE,
                                                );
                                            }
                                            
                                            // Character/Enemy info
                                            ui.vertical(|ui| {
                                                ui.label(char.name());
                                                let info_text = match &char.entity {
                                                    EntityType::Character(c) => {
                                                        let race_class = if c.race.is_empty() {
                                                            c.class.clone()
                                                        } else {
                                                            format!("{} {}", c.race, c.class)
                                                        };
                                                        format!("{}", race_class)
                                                    }
                                                    EntityType::Enemy(e) => {
                                                        format!("{}", e.enemy_type)
                                                    }
                                                };
                                                ui.small(info_text);
                                            });
                                            
                                            // Health adjustment buttons
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                // + button
                                                if ui.small_button("+").clicked() {
                                                    char.modify_health(1);
                                                    save_needed = true;
                                                }
                                                // HP display
                                                let hp_text = format!("{}/{}", char.health(), char.hit_points());
                                                ui.label(hp_text);
                                                // - button
                                                if ui.small_button("-").clicked() {
                                                    char.modify_health(-1);
                                                    save_needed = true;
                                                }
                                            });
                                        });
                                        ui.add_space(2.0);
                                    }
                                }
                                
                                // Save after the loop if changes were made
                                if save_needed {
                                    save_characters(&chars);
                                }
                            }

                            ui.separator();
                            ui.heading("🎯 Drag & Drop Timing");
                            ui.small("Adjust timing for character drag-and-drop");
                            {
                                let mut val = pinch_show_loading_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 100..=1000).text("Show loading delay (ms)"));
                            }
                            {
                                let mut val = pinch_pickup_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 200..=3000).text("Pickup duration (ms)"));
                            }
                            {
                                let mut val = pinch_flicker_tolerance_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 50..=500).text("Flicker tolerance (ms)"));
                            }
                            {
                                let mut val = drop_confirm_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 100..=2000).text("Drop confirm (ms)"));
                            }

                            ui.separator();
                            ui.heading("👁️ Character Info Panel");
                            ui.small("Hover finger over character to show info");
                            {
                                let mut val = hover_show_info_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 100..=2000).text("Hover delay (ms)"));
                            }
                            {
                                let mut val = hover_info_display_ms_c.lock().unwrap();
                                ui.add(egui::Slider::new(&mut *val, 1000..=10000).text("Display duration (ms)"));
                            }

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
                            ui.heading("🔧 Calibration");
                            ui.small("Configure camera-to-projector mapping");
                            if ui.button("Open Calibration Settings...").clicked() {
                                *show_calibration_window_c.lock().unwrap() = true;
                            }

                            ui.separator();
                            ui.heading("ML Hand Tracking");
                            {
                                let ml_available = hand_tracker::model_available();
                                if ml_available {
                                    ui.label("✓ Hand landmark model loaded");
                                } else {
                                    ui.colored_label(egui::Color32::RED, "✗ Model not found");
                                    ui.small("Place hand_landmarker.task in ~/.config/intim-dnd/models/");
                                }
                            }
                            
                            // Hand tracking preview (larger view)
                            ui.add_space(8.0);
                            let (hand_frame_data, hand_w, hand_h, finger_tip, hand_landmarks) = {
                                let state_lock = state.lock().unwrap();
                                (
                                    state_lock.debug_hand_tracking_frame.clone(),
                                    state_lock.debug_hand_tracking_width,
                                    state_lock.debug_hand_tracking_height,
                                    state_lock.finger_tip_camera,
                                    state_lock.hand_landmarks.clone(),
                                )
                            };
                            
                            let hand_width = hand_w as usize;
                            let hand_height = hand_h as usize;
                            let hand_expected_size = hand_width * hand_height * 3;
                            let hand_aspect = hand_height as f32 / hand_width as f32;
                            let hand_display_width = 320.0;  // Larger display
                            let hand_display_height = hand_display_width * hand_aspect;
                            
                            ui.vertical(|ui| {
                                ui.label("Hand Tracking View:");
                                if let Some(data) = &hand_frame_data {
                                    if data.len() >= hand_expected_size {
                                        let image = egui::ColorImage::from_rgb(
                                            [hand_width, hand_height],
                                            &data[..hand_expected_size],
                                        );
                                        
                                        let texture: egui::TextureHandle = ctx.load_texture(
                                            "debug_hand_tracking",
                                            image,
                                            egui::TextureOptions::LINEAR,
                                        );
                                        
                                        let (rect, _response) = ui.allocate_exact_size(
                                            egui::vec2(hand_display_width, hand_display_height),
                                            egui::Sense::hover(),
                                        );
                                        
                                        ui.painter().image(
                                            texture.id(),
                                            rect,
                                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                            egui::Color32::WHITE,
                                        );
                                        
                                        // Draw hand skeleton on the view
                                        if let Some(landmarks) = &hand_landmarks {
                                            let scale_x = hand_display_width / hand_width as f32;
                                            let scale_y = hand_display_height / hand_height as f32;
                                            
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
                                            
                                            // Draw bones
                                            for &(from, to, color) in connections {
                                                if from < landmarks.len() && to < landmarks.len() {
                                                    ui.painter().line_segment(
                                                        [to_screen(from), to_screen(to)],
                                                        egui::Stroke::new(2.0, color),
                                                    );
                                                }
                                            }
                                            
                                            // Draw joints
                                            for (idx, &(lx, ly)) in landmarks.iter().enumerate() {
                                                let screen_pos = egui::pos2(
                                                    rect.min.x + lx * scale_x,
                                                    rect.min.y + ly * scale_y,
                                                );
                                                
                                                let (radius, color) = match idx {
                                                    0 => (5.0, egui::Color32::YELLOW),
                                                    4 | 8 | 12 | 16 | 20 => (4.0, egui::Color32::RED),
                                                    _ => (3.0, egui::Color32::WHITE),
                                                };
                                                
                                                ui.painter().circle_filled(screen_pos, radius, color);
                                            }
                                            
                                            // Highlight index finger tip
                                            if landmarks.len() > 8 {
                                                let tip_pos = to_screen(8);
                                                ui.painter().circle_stroke(
                                                    tip_pos,
                                                    15.0,
                                                    egui::Stroke::new(3.0, egui::Color32::GREEN),
                                                );
                                                ui.painter().line_segment(
                                                    [egui::pos2(tip_pos.x - 20.0, tip_pos.y), egui::pos2(tip_pos.x + 20.0, tip_pos.y)],
                                                    egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                );
                                                ui.painter().line_segment(
                                                    [egui::pos2(tip_pos.x, tip_pos.y - 20.0), egui::pos2(tip_pos.x, tip_pos.y + 20.0)],
                                                    egui::Stroke::new(2.0, egui::Color32::GREEN),
                                                );
                                                
                                                let wrist_pos = to_screen(0);
                                                ui.painter().circle_filled(wrist_pos, 8.0, egui::Color32::YELLOW);
                                            }
                                        } else if let Some((fx, fy)) = finger_tip {
                                            // Fallback: just draw fingertip
                                            let roi_offset = {
                                                let state_lock = state.lock().unwrap();
                                                state_lock.camera_roi.map(|r| (r.x as f32, r.y as f32)).unwrap_or((0.0, 0.0))
                                            };
                                            let local_x = fx - roi_offset.0;
                                            let local_y = fy - roi_offset.1;
                                            
                                            let scale_x = hand_display_width / hand_width as f32;
                                            let scale_y = hand_display_height / hand_height as f32;
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
                                            format!("Size mismatch: {} vs {}", data.len(), hand_expected_size));
                                    }
                                } else {
                                    ui.colored_label(egui::Color32::GRAY, "No hand tracking data");
                                }
                            });

                            ui.separator();
                            ui.heading("🎨 Image Generation");
                            
                            // Button to use default background
                            if ui.button("🗺️ Use Default Background").clicked() {
                                *use_default_background_c.lock().unwrap() = true;
                                *selected_image_index_c.lock().unwrap() = None;
                            }
                            
                            ui.add_space(8.0);
                            
                            // Check API token status
                            if gemini_api_token.is_none() {
                                ui.colored_label(egui::Color32::RED, "⚠️ No API token found");
                                ui.small("Create gemini_api_token.txt with your Gemini API key");
                            } else {
                                ui.colored_label(egui::Color32::GREEN, "✓ API token loaded");
                                
                                // Prompt input
                                ui.label("Enter prompt for background image:");
                                {
                                    let mut prompt = image_gen_prompt_c.lock().unwrap();
                                    ui.add(egui::TextEdit::multiline(&mut *prompt)
                                        .desired_rows(3)
                                        .desired_width(f32::INFINITY)
                                        .hint_text("e.g., A dark cave with glowing crystals, top-down view, fantasy map style"));
                                }
                                
                                // Generation status
                                let status = image_gen_status_c.lock().unwrap().clone();
                                match &status {
                                    ImageGenStatus::Idle => {
                                        let prompt = image_gen_prompt_c.lock().unwrap().clone();
                                        let can_generate = !prompt.trim().is_empty();
                                        
                                        if ui.add_enabled(can_generate, egui::Button::new("🖼️ Generate Image")).clicked() {
                                            // Start generation in background thread
                                            let token = gemini_api_token.clone().unwrap();
                                            let status_clone = image_gen_status_c.clone();
                                            let images_clone = generated_images_c.clone();
                                            let selected_clone = selected_image_index_c.clone();
                                            let prompt_for_thread = prompt.clone();
                                            
                                            *status_clone.lock().unwrap() = ImageGenStatus::Generating;
                                            
                                            thread::spawn(move || {
                                                match generate_image_with_gemini(&token, &prompt_for_thread) {
                                                    Ok(image_bytes) => {
                                                        // Create generated folder in config if it doesn't exist
                                                        let generated_dir = dirs::config_dir()
                                                            .map(|d| d.join("intim-dnd/generated"))
                                                            .unwrap_or_else(|| std::path::PathBuf::from("generated"));
                                                        if let Err(e) = fs::create_dir_all(&generated_dir) {
                                                            *status_clone.lock().unwrap() = ImageGenStatus::Error(format!("Failed to create folder: {}", e));
                                                            return;
                                                        }
                                                        
                                                        // Generate unique filename from prompt
                                                        let filename = GeneratedImage::filename_from_prompt(&prompt_for_thread);
                                                        let save_path = generated_dir.join(&filename);
                                                        
                                                        match fs::write(&save_path, &image_bytes) {
                                                            Ok(_) => {
                                                                log::info!("Saved generated image to {:?}", save_path);
                                                                
                                                                // Add to gallery
                                                                let mut images = images_clone.lock().unwrap();
                                                                IntImDnDApp::add_generated_image(
                                                                    &mut images,
                                                                    save_path.clone(),
                                                                    prompt_for_thread.clone()
                                                                );
                                                                
                                                                // Select the newly generated image
                                                                *selected_clone.lock().unwrap() = Some(0);
                                                                
                                                                *status_clone.lock().unwrap() = ImageGenStatus::Success(
                                                                    save_path.to_string_lossy().to_string()
                                                                );
                                                            }
                                                            Err(e) => {
                                                                *status_clone.lock().unwrap() = ImageGenStatus::Error(format!("Failed to save: {}", e));
                                                            }
                                                        }
                                                    }
                                                    Err(e) => {
                                                        *status_clone.lock().unwrap() = ImageGenStatus::Error(e);
                                                    }
                                                }
                                            });
                                        }
                                    }
                                    ImageGenStatus::Generating => {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("Generating image...");
                                        });
                                    }
                                    ImageGenStatus::Success(_path) => {
                                        ui.colored_label(egui::Color32::GREEN, "✓ Image generated!");
                                        if ui.button("Generate Another").clicked() {
                                            *image_gen_status_c.lock().unwrap() = ImageGenStatus::Idle;
                                        }
                                    }
                                    ImageGenStatus::Error(err) => {
                                        ui.colored_label(egui::Color32::RED, format!("✗ Error: {}", err));
                                        if ui.button("Try Again").clicked() {
                                            *image_gen_status_c.lock().unwrap() = ImageGenStatus::Idle;
                                        }
                                    }
                                }
                            }
                            
                            // Image Gallery
                            ui.add_space(8.0);
                            ui.separator();
                            ui.heading("📁 Generated Images");
                            
                            let images = generated_images_c.lock().unwrap();
                            let mut selected_idx = selected_image_index_c.lock().unwrap();
                            
                            if images.is_empty() {
                                ui.colored_label(egui::Color32::GRAY, "No generated images yet");
                            } else {
                                ui.label(format!("Showing last {} images:", images.len()));
                                
                                // Gallery grid
                                ui.horizontal_wrapped(|ui| {
                                    for (idx, img) in images.iter().enumerate() {
                                        let is_selected = *selected_idx == Some(idx);
                                        let thumb_size = egui::vec2(60.0, 60.0);
                                        
                                        // Create a frame for selection highlight
                                        let frame_color = if is_selected {
                                            egui::Color32::from_rgb(100, 200, 100)
                                        } else {
                                            egui::Color32::from_gray(60)
                                        };
                                        
                                        let response = egui::Frame::none()
                                            .stroke(egui::Stroke::new(if is_selected { 3.0 } else { 1.0 }, frame_color))
                                            .inner_margin(2.0)
                                            .show(ui, |ui| {
                                                // Try to load and display thumbnail
                                                if let Ok(img_data) = image::open(&img.path) {
                                                    let thumb = img_data.thumbnail(60, 60);
                                                    let rgba = thumb.to_rgba8();
                                                    let size = [rgba.width() as usize, rgba.height() as usize];
                                                    let pixels = rgba.into_raw();
                                                    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                                                    
                                                    let texture = ctx.load_texture(
                                                        format!("thumb_{}", idx),
                                                        color_image,
                                                        egui::TextureOptions::LINEAR,
                                                    );
                                                    
                                                    let (rect, response) = ui.allocate_exact_size(thumb_size, egui::Sense::click());
                                                    ui.painter().image(
                                                        texture.id(),
                                                        rect,
                                                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                                        egui::Color32::WHITE,
                                                    );
                                                    response
                                                } else {
                                                    // Show placeholder if image can't be loaded
                                                    let (rect, response) = ui.allocate_exact_size(thumb_size, egui::Sense::click());
                                                    ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(40));
                                                    ui.painter().text(
                                                        rect.center(),
                                                        egui::Align2::CENTER_CENTER,
                                                        "?",
                                                        egui::FontId::default(),
                                                        egui::Color32::WHITE,
                                                    );
                                                    response
                                                }
                                            });
                                        
                                        // Handle click to select
                                        if response.inner.clicked() {
                                            *selected_idx = Some(idx);
                                        }
                                        
                                        // Tooltip with prompt
                                        response.response.on_hover_text(&img.prompt);
                                    }
                                });
                                
                                // Show selected image info and actions
                                if let Some(idx) = *selected_idx {
                                    if idx < images.len() {
                                        ui.add_space(4.0);
                                        ui.group(|ui| {
                                            let img = &images[idx];
                                            ui.label(format!("Selected: {}", img.prompt));
                                            
                                            ui.horizontal(|ui| {
                                                if ui.button("🖼️ Display as Background").clicked() {
                                                    *selected_image_path_c.lock().unwrap() = Some(img.path.clone());
                                                    *load_selected_image_c.lock().unwrap() = true;
                                                }
                                            });
                                        });
                                    }
                                }
                            }
                            drop(images);
                            drop(selected_idx);

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
            let old_radius = self.circle_radius;
            let old_color = self.circle_color;
            let old_proj_width = self.projector_width;
            let old_proj_height = self.projector_height;
            let old_show_grid = self.show_grid;
            let old_grid_rows = self.grid_rows;
            let old_pinch_show_loading = self.pinch_show_loading_ms;
            let old_pinch_pickup = self.pinch_pickup_ms;
            let old_flicker_tolerance = self.pinch_flicker_tolerance_ms;
            let old_drop_confirm = self.drop_confirm_ms;
            let old_hover_show_info = self.hover_show_info_ms;
            let old_hover_info_display = self.hover_info_display_ms;
            
            self.show_camera_feed = *show_camera_feed.lock().unwrap();
            self.circle_radius = *circle_radius.lock().unwrap();
            self.circle_color = *circle_color.lock().unwrap();
            self.projector_width = *projector_width.lock().unwrap();
            self.projector_height = *projector_height.lock().unwrap();
            self.homography_edit = *homography_edit.lock().unwrap();
            self.show_options_window = *show_options.lock().unwrap();
            self.image_gen_prompt = image_gen_prompt.lock().unwrap().clone();
            self.show_grid = *show_grid.lock().unwrap();
            self.grid_rows = *grid_rows.lock().unwrap();
            self.pinch_show_loading_ms = *pinch_show_loading_ms.lock().unwrap();
            self.pinch_pickup_ms = *pinch_pickup_ms.lock().unwrap();
            self.pinch_flicker_tolerance_ms = *pinch_flicker_tolerance_ms.lock().unwrap();
            self.drop_confirm_ms = *drop_confirm_ms.lock().unwrap();
            self.hover_show_info_ms = *hover_show_info_ms.lock().unwrap();
            self.hover_info_display_ms = *hover_info_display_ms.lock().unwrap();
            self.show_calibration_window = *show_calibration_window.lock().unwrap();
            
            // Save settings if any changed
            if old_radius != self.circle_radius
                || old_color != self.circle_color
                || old_proj_width != self.projector_width
                || old_proj_height != self.projector_height
                || old_show_grid != self.show_grid
                || old_grid_rows != self.grid_rows
                || old_pinch_show_loading != self.pinch_show_loading_ms
                || old_pinch_pickup != self.pinch_pickup_ms
                || old_flicker_tolerance != self.pinch_flicker_tolerance_ms
                || old_drop_confirm != self.drop_confirm_ms
                || old_hover_show_info != self.hover_show_info_ms
                || old_hover_info_display != self.hover_info_display_ms
            {
                self.save_current_settings();
            }
            
            // Check if we need to use the default background
            if *use_default_background.lock().unwrap() {
                *use_default_background.lock().unwrap() = false;
                self.load_default_background();
            }
            
            // Check if we need to load a selected image from the gallery
            if *load_selected_image.lock().unwrap() {
                *load_selected_image.lock().unwrap() = false;
                if let Some(path) = selected_image_path.lock().unwrap().take() {
                    self.load_image_as_background(&path);
                }
            }
        }

        // Calibration window as a separate OS-level viewport
        if self.show_calibration_window {
            let state = self.state.clone();
            let homography_edit = Arc::new(Mutex::new(self.homography_edit));
            let show_calibration = Arc::new(Mutex::new(true));
            let apply_homography = Arc::new(Mutex::new(false));
            let reset_homography = Arc::new(Mutex::new(false));
            
            let homography_edit_c = homography_edit.clone();
            let show_calibration_c = show_calibration.clone();
            let apply_homography_c = apply_homography.clone();
            let reset_homography_c = reset_homography.clone();
            
            ctx.show_viewport_immediate(
                calibration_viewport_id(),
                egui::ViewportBuilder::default()
                    .with_title("IntIm-DnD - Calibration Settings")
                    .with_inner_size([450.0, 600.0]),
                |ctx, _class| {
                    // Handle window close request
                    if ctx.input(|i| i.viewport().close_requested()) {
                        *show_calibration_c.lock().unwrap() = false;
                    }

                    egui::CentralPanel::default().show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.heading("🔧 Calibration Settings");
                            ui.separator();
                            
                            // Homography Matrix Editor
                            ui.heading("Homography Matrix");
                            ui.small("Camera-to-projector coordinate transformation");
                            ui.add_space(4.0);
                            
                            {
                                let mut h = homography_edit_c.lock().unwrap();
                                egui::Grid::new("homography_grid")
                                    .num_columns(3)
                                    .spacing([8.0, 4.0])
                                    .show(ui, |ui| {
                                        for row in 0..3 {
                                            for col in 0..3 {
                                                ui.add(
                                                    egui::DragValue::new(&mut h[row][col])
                                                        .speed(0.01)
                                                        .fixed_decimals(4)
                                                );
                                            }
                                            ui.end_row();
                                        }
                                    });
                            }
                            
                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                if ui.button("Apply Matrix").clicked() {
                                    *apply_homography_c.lock().unwrap() = true;
                                }
                                
                                if ui.button("Reset to Default").clicked() {
                                    *reset_homography_c.lock().unwrap() = true;
                                }
                            });
                            
                            ui.separator();
                            
                            // Camera Feed Debug View
                            ui.heading("Camera Feed");
                            let (frame_data, frame_w, frame_h) = {
                                let state_lock = state.lock().unwrap();
                                (
                                    state_lock.frame.clone(),
                                    state_lock.frame_width,
                                    state_lock.frame_height,
                                )
                            };
                            
                            let cam_width = frame_w as usize;
                            let cam_height = frame_h as usize;
                            let cam_expected_size = cam_width * cam_height * 3;
                            let cam_aspect = cam_height as f32 / cam_width as f32;
                            let cam_display_width = 400.0;
                            let cam_display_height = cam_display_width * cam_aspect;
                            
                            if let Some(data) = &frame_data {
                                if data.len() >= cam_expected_size {
                                    let image = egui::ColorImage::from_rgb(
                                        [cam_width, cam_height],
                                        &data[..cam_expected_size],
                                    );
                                    
                                    let texture: egui::TextureHandle = ctx.load_texture(
                                        "debug_camera_calib",
                                        image,
                                        egui::TextureOptions::LINEAR,
                                    );
                                    
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(cam_display_width, cam_display_height),
                                        egui::Sense::hover(),
                                    );
                                    
                                    ui.painter().image(
                                        texture.id(),
                                        rect,
                                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                        egui::Color32::WHITE,
                                    );
                                } else {
                                    ui.colored_label(egui::Color32::GRAY, "Size mismatch");
                                }
                            } else {
                                ui.colored_label(egui::Color32::GRAY, "No camera data");
                            }
                            
                            ui.separator();
                            
                            // Cropped Region Preview
                            ui.heading("Cropped Region (Hand Tracking)");
                            let (crop_data, crop_w, crop_h, finger_tip, hand_landmarks) = {
                                let state_lock = state.lock().unwrap();
                                (
                                    state_lock.debug_hand_tracking_frame.clone(),
                                    state_lock.debug_hand_tracking_width,
                                    state_lock.debug_hand_tracking_height,
                                    state_lock.finger_tip_camera,
                                    state_lock.hand_landmarks.clone(),
                                )
                            };
                            
                            let crop_width = crop_w as usize;
                            let crop_height = crop_h as usize;
                            let crop_expected_size = crop_width * crop_height * 3;
                            let crop_aspect = crop_height as f32 / crop_width as f32;
                            let crop_display_width = 400.0;
                            let crop_display_height = crop_display_width * crop_aspect;
                            
                            if let Some(data) = &crop_data {
                                if data.len() >= crop_expected_size {
                                    let image = egui::ColorImage::from_rgb(
                                        [crop_width, crop_height],
                                        &data[..crop_expected_size],
                                    );
                                    
                                    let texture: egui::TextureHandle = ctx.load_texture(
                                        "debug_crop_calib",
                                        image,
                                        egui::TextureOptions::LINEAR,
                                    );
                                    
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(crop_display_width, crop_display_height),
                                        egui::Sense::hover(),
                                    );
                                    
                                    ui.painter().image(
                                        texture.id(),
                                        rect,
                                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                        egui::Color32::WHITE,
                                    );
                                    
                                    // Draw hand skeleton overlay if available
                                    if let Some(landmarks) = &hand_landmarks {
                                        let scale_x = crop_display_width / crop_width as f32;
                                        let scale_y = crop_display_height / crop_height as f32;
                                        
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
                                        
                                        // Draw joints
                                        for (idx, &(lx, ly)) in landmarks.iter().enumerate() {
                                            let screen_pos = egui::pos2(
                                                rect.min.x + lx * scale_x,
                                                rect.min.y + ly * scale_y,
                                            );
                                            
                                            let (radius, color) = match idx {
                                                0 => (5.0, egui::Color32::YELLOW),
                                                4 | 8 | 12 | 16 | 20 => (4.0, egui::Color32::RED),
                                                _ => (3.0, egui::Color32::WHITE),
                                            };
                                            
                                            ui.painter().circle_filled(screen_pos, radius, color);
                                        }
                                        
                                        // Highlight fingertip
                                        if landmarks.len() > 8 {
                                            let tip_pos = to_screen(8);
                                            ui.painter().circle_stroke(
                                                tip_pos,
                                                12.0,
                                                egui::Stroke::new(2.0, egui::Color32::GREEN),
                                            );
                                        }
                                    } else if let Some((fx, fy)) = finger_tip {
                                        // Fallback: draw fingertip point
                                        let roi_offset = {
                                            let state_lock = state.lock().unwrap();
                                            state_lock.camera_roi.map(|r| (r.x as f32, r.y as f32)).unwrap_or((0.0, 0.0))
                                        };
                                        let local_x = fx - roi_offset.0;
                                        let local_y = fy - roi_offset.1;
                                        
                                        let scale_x = crop_display_width / crop_width as f32;
                                        let scale_y = crop_display_height / crop_height as f32;
                                        let screen_x = rect.min.x + local_x * scale_x;
                                        let screen_y = rect.min.y + local_y * scale_y;
                                        
                                        ui.painter().circle_stroke(
                                            egui::pos2(screen_x, screen_y),
                                            8.0,
                                            egui::Stroke::new(2.0, egui::Color32::GREEN),
                                        );
                                    }
                                } else {
                                    ui.colored_label(egui::Color32::GRAY, "Size mismatch");
                                }
                            } else {
                                ui.colored_label(egui::Color32::GRAY, "No cropped region data");
                            }
                            
                            ui.add_space(16.0);
                            
                            if ui.button("Close").clicked() {
                                *show_calibration_c.lock().unwrap() = false;
                            }
                        });
                    });
                },
            );
            
            // Update from Arc values
            self.homography_edit = *homography_edit.lock().unwrap();
            self.show_calibration_window = *show_calibration.lock().unwrap();
            
            // Handle apply/reset actions
            if *apply_homography.lock().unwrap() {
                let h = self.homography_edit;
                self.state.lock().unwrap().set_homography(h);
                // Convert to Matrix3 for saving
                let matrix = Matrix3::from_row_slice(&h.concat());
                let _ = save_homography(&matrix);
                log::info!("Applied and saved homography matrix");
            }
            
            if *reset_homography.lock().unwrap() {
                self.homography_edit = HOMOGRAPHY;
                self.state.lock().unwrap().set_homography(HOMOGRAPHY);
                let matrix = Matrix3::from_row_slice(&HOMOGRAPHY.concat());
                let _ = save_homography(&matrix);
                log::info!("Reset homography to default");
            }
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
                let (finger_projector, calib_state, calib_point, dot_radius, is_pinching) = {
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
                        state.is_pinching,
                    )
                };

                let available_size = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(available_size, egui::Sense::click_and_drag());
                
                // Get mouse state for board interaction
                let mouse_pos_projector: Option<(f32, f32)> = response.hover_pos().map(|pos| {
                    // Convert screen position to projector coordinates
                    let rel_x = (pos.x - rect.min.x) / available_size.x * self.projector_width;
                    let rel_y = (pos.y - rect.min.y) / available_size.y * self.projector_height;
                    (rel_x, rel_y)
                });
                let mouse_clicked = response.dragged_by(egui::PointerButton::Primary) || 
                                   ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
                
                // Combine finger tracking with mouse input
                // Mouse takes precedence when clicking, otherwise use finger tracking
                let effective_position = if mouse_clicked && mouse_pos_projector.is_some() {
                    mouse_pos_projector
                } else if mouse_pos_projector.is_some() && !mouse_clicked {
                    // Mouse hovering but not clicking - use for hover detection only
                    finger_projector
                } else {
                    finger_projector
                };
                
                let effective_pinching = if mouse_clicked && mouse_pos_projector.is_some() {
                    true
                } else {
                    is_pinching
                };
                
                // Draw background image or black if not available
                if let Some(ref bg_image) = self.background_image {
                    // Create/update the background texture if needed
                    let bg_texture = self.background_texture.get_or_insert_with(|| {
                        ctx.load_texture(
                            "background",
                            bg_image.clone(),
                            egui::TextureOptions::LINEAR,
                        )
                    });
                    
                    // Draw background image scaled to fill the available area
                    ui.painter().image(
                        bg_texture.id(),
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                } else {
                    // Fallback to black background
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
                }
                
                // Draw grid overlay if enabled
                if self.show_grid {
                    let grid_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                    let stroke = egui::Stroke::new(1.0, grid_color);
                    
                    // Calculate cell size based on row count (square cells)
                    let cell_size = rect.height() / self.grid_rows as f32;
                    let num_cols = (rect.width() / cell_size).ceil() as u32;
                    
                    // Draw horizontal lines
                    for row in 0..=self.grid_rows {
                        let y = rect.min.y + row as f32 * cell_size;
                        if y <= rect.max.y {
                            ui.painter().line_segment(
                                [egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)],
                                stroke,
                            );
                        }
                    }
                    
                    // Draw vertical lines
                    for col in 0..=num_cols {
                        let x = rect.min.x + col as f32 * cell_size;
                        if x <= rect.max.x {
                            ui.painter().line_segment(
                                [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                                stroke,
                            );
                        }
                    }
                }
                
                // Draw characters on the grid
                let dragged_char_index: Option<usize>;
                let drag_position: Option<(f32, f32)>;
                {
                    let cell_size = rect.height() / self.grid_rows as f32;
                    
                    // Handle drag-and-drop state machine
                    let now = Instant::now();
                    
                    // Check if pinch/click is currently active (with flicker tolerance)
                    // Use effective_pinching which combines mouse click and hand pinch
                    let pinch_active = if effective_pinching {
                        if let Some(fp) = effective_position {
                            self.last_pinch_detected = Some(now);
                            self.last_pinch_position = Some(fp);
                        }
                        true
                    } else if let Some(last_time) = self.last_pinch_detected {
                        // Allow brief flicker - consider pinch still active if within tolerance
                        now.duration_since(last_time).as_millis() < self.pinch_flicker_tolerance_ms as u128
                    } else {
                        false
                    };
                    
                    // Get current pinch/click position (use last known if flickering)
                    let pinch_pos = if effective_pinching {
                        effective_position
                    } else {
                        self.last_pinch_position
                    };
                    
                    // State machine transitions - compute new state first to avoid borrow issues
                    let new_state = match self.drag_state.clone() {
                        DragState::Idle => {
                            if pinch_active {
                                if let Some(pos) = pinch_pos {
                                    Some(DragState::PinchStarted {
                                        start_time: now,
                                        position: pos,
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        DragState::PinchStarted { start_time, position } => {
                            if !pinch_active {
                                self.last_pinch_detected = None;
                                self.last_pinch_position = None;
                                Some(DragState::Idle)
                            } else {
                                let elapsed_ms = now.duration_since(start_time).as_millis() as u64;
                                if elapsed_ms >= self.pinch_show_loading_ms {
                                    // Check if there's a character at this position
                                    let screen_pos = (
                                        rect.min.x + position.0 * (available_size.x / self.projector_width),
                                        rect.min.y + position.1 * (available_size.y / self.projector_height),
                                    );
                                    if let Some(cell) = self.projector_to_grid_cell(screen_pos.0, screen_pos.1, &rect) {
                                        if let Some(char_idx) = self.find_character_at_cell(cell) {
                                            Some(DragState::PickingUp {
                                                start_time,
                                                position: pinch_pos.unwrap_or(position),
                                                character_index: char_idx,
                                            })
                                        } else {
                                            // No character here, reset
                                            Some(DragState::Idle)
                                        }
                                    } else {
                                        Some(DragState::Idle)
                                    }
                                } else {
                                    None
                                }
                            }
                        }
                        DragState::PickingUp { start_time, character_index, .. } => {
                            if !pinch_active {
                                self.last_pinch_detected = None;
                                self.last_pinch_position = None;
                                Some(DragState::Idle)
                            } else {
                                let elapsed_ms = now.duration_since(start_time).as_millis() as u64;
                                let total_pickup_time = self.pinch_show_loading_ms + self.pinch_pickup_ms;
                                if elapsed_ms >= total_pickup_time {
                                    // Pickup complete!
                                    let original_pos = {
                                        let chars = self.characters.lock().unwrap();
                                        chars[character_index].grid_pos
                                    };
                                    log::info!("Picked up character {}", character_index);
                                    Some(DragState::Dragging {
                                        character_index,
                                        original_pos,
                                        current_pos: pinch_pos.unwrap_or((0.0, 0.0)),
                                    })
                                } else {
                                    None
                                }
                            }
                        }
                        DragState::Dragging { character_index, original_pos, current_pos } => {
                            if !pinch_active {
                                // Dropped - place character at last pinch position
                                let screen_pos = (
                                    rect.min.x + current_pos.0 * (available_size.x / self.projector_width),
                                    rect.min.y + current_pos.1 * (available_size.y / self.projector_height),
                                );
                                
                                // Find the cell at the drop position
                                if let Some(target_cell) = self.projector_to_grid_cell(screen_pos.0, screen_pos.1, &rect) {
                                    // Check if cell is not occupied by another character
                                    let cell_occupied = self.find_character_at_cell(target_cell)
                                        .map(|idx| idx != character_index)
                                        .unwrap_or(false);
                                    
                                    if !cell_occupied {
                                        // Place character at new cell
                                        {
                                            let mut chars = self.characters.lock().unwrap();
                                            chars[character_index].grid_pos = target_cell;
                                        }
                                        log::info!("Dropped character {} at {:?}", character_index, target_cell);
                                    } else {
                                        log::info!("Drop location occupied, character stays at original position");
                                    }
                                }
                                self.last_pinch_detected = None;
                                self.last_pinch_position = None;
                                Some(DragState::Idle)
                            } else if let Some(pos) = pinch_pos {
                                // Update position
                                let screen_pos = (
                                    rect.min.x + pos.0 * (available_size.x / self.projector_width),
                                    rect.min.y + pos.1 * (available_size.y / self.projector_height),
                                );
                                
                                // Check if we're over a valid cell
                                if let Some(target_cell) = self.projector_to_grid_cell(screen_pos.0, screen_pos.1, &rect) {
                                    // Check if this cell is different from original and not occupied
                                    let cell_occupied = self.find_character_at_cell(target_cell)
                                        .map(|idx| idx != character_index)
                                        .unwrap_or(false);
                                    
                                    if !cell_occupied && target_cell != original_pos {
                                        // Start drop confirmation
                                        Some(DragState::DroppingOff {
                                            start_time: now,
                                            character_index,
                                            original_pos,
                                            target_cell,
                                            position: pos,
                                        })
                                    } else {
                                        // Update drag position
                                        Some(DragState::Dragging {
                                            character_index,
                                            original_pos,
                                            current_pos: pos,
                                        })
                                    }
                                } else {
                                    // Update drag position
                                    Some(DragState::Dragging {
                                        character_index,
                                        original_pos,
                                        current_pos: pos,
                                    })
                                }
                            } else {
                                None
                            }
                        }
                        DragState::DroppingOff { start_time, character_index, original_pos, target_cell, position } => {
                            if !pinch_active {
                                // Dropped - place character at target cell (even without full confirmation)
                                {
                                    let mut chars = self.characters.lock().unwrap();
                                    chars[character_index].grid_pos = target_cell;
                                }
                                log::info!("Dropped character {} at {:?} (quick drop)", character_index, target_cell);
                                self.last_pinch_detected = None;
                                self.last_pinch_position = None;
                                Some(DragState::Idle)
                            } else {
                                let elapsed_ms = now.duration_since(start_time).as_millis() as u64;
                                
                                // Check if still over the same cell
                                let current_screen_pos = pinch_pos.map(|pos| (
                                    rect.min.x + pos.0 * (available_size.x / self.projector_width),
                                    rect.min.y + pos.1 * (available_size.y / self.projector_height),
                                ));
                                
                                let still_over_target = current_screen_pos
                                    .and_then(|sp| self.projector_to_grid_cell(sp.0, sp.1, &rect))
                                    .map(|cell| cell == target_cell)
                                    .unwrap_or(false);
                                
                                if !still_over_target {
                                    // Moved away from target, go back to dragging
                                    Some(DragState::Dragging {
                                        character_index,
                                        original_pos,
                                        current_pos: pinch_pos.unwrap_or(position),
                                    })
                                } else if elapsed_ms >= self.drop_confirm_ms {
                                    // Drop confirmed!
                                    {
                                        let mut chars = self.characters.lock().unwrap();
                                        chars[character_index].grid_pos = target_cell;
                                    }
                                    log::info!("Dropped character {} at {:?}", character_index, target_cell);
                                    self.last_pinch_detected = None;
                                    self.last_pinch_position = None;
                                    Some(DragState::Idle)
                                } else {
                                    None
                                }
                            }
                        }
                    };
                    
                    // Apply state transition if computed
                    if let Some(state) = new_state {
                        self.drag_state = state;
                    }
                    
                    // Extract drag info for rendering
                    dragged_char_index = match &self.drag_state {
                        DragState::Dragging { character_index, .. } |
                        DragState::DroppingOff { character_index, .. } => Some(*character_index),
                        _ => None,
                    };
                    
                    drag_position = match &self.drag_state {
                        DragState::Dragging { current_pos, .. } => Some(*current_pos),
                        DragState::DroppingOff { position, .. } => Some(*position),
                        _ => None,
                    };
                    
                    // Draw characters
                    let mut chars = self.characters.lock().unwrap();
                    
                    for (char_idx, char) in chars.iter_mut().enumerate() {
                        if !char.visible {
                            continue;
                        }
                        
                        // Check if this character is being dragged
                        let is_dragged = Some(char_idx) == dragged_char_index;
                        
                        let (col, row) = char.grid_pos;
                        
                        // Calculate cell position
                        let cell_x = rect.min.x + col as f32 * cell_size;
                        let cell_y = rect.min.y + row as f32 * cell_size;
                        
                        // Token size (slightly smaller than cell)
                        let token_margin = cell_size * 0.1;
                        let token_size = cell_size - token_margin * 2.0;
                        let token_rect = egui::Rect::from_min_size(
                            egui::pos2(cell_x + token_margin, cell_y + token_margin),
                            egui::vec2(token_size, token_size),
                        );
                        
                        // If dragged, draw a ghost placeholder and skip normal rendering
                        if is_dragged {
                            // Draw ghost (semi-transparent circle indicating original position)
                            ui.painter().circle_filled(
                                token_rect.center(),
                                token_size / 2.0,
                                egui::Color32::from_rgba_unmultiplied(100, 100, 100, 80),
                            );
                            ui.painter().circle_stroke(
                                token_rect.center(),
                                token_size / 2.0,
                                egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(150, 150, 150, 120)),
                            );
                            continue;
                        }
                        
                        // Draw token image or placeholder
                        if let Some(ref token_img) = char.token_image {
                            // Create/update texture
                            let char_name = char.name().to_string();
                            let texture = char.token_texture.get_or_insert_with(|| {
                                ctx.load_texture(
                                    format!("token_{}", char_name),
                                    token_img.clone(),
                                    egui::TextureOptions::LINEAR,
                                )
                            });
                            
                            ui.painter().image(
                                texture.id(),
                                token_rect,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        } else {
                            // Draw placeholder circle with initial
                            let placeholder_color = if char.is_enemy() {
                                egui::Color32::from_rgb(150, 80, 80) // Red tint for enemies
                            } else {
                                egui::Color32::from_rgb(100, 100, 150)
                            };
                            ui.painter().circle_filled(
                                token_rect.center(),
                                token_size / 2.0,
                                placeholder_color,
                            );
                            ui.painter().text(
                                token_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &char.name().chars().next().unwrap_or('?').to_string(),
                                egui::FontId::proportional(token_size * 0.5),
                                egui::Color32::WHITE,
                            );
                        }
                        
                        // Draw health bar on top of the token
                        let health_bar_height = cell_size * 0.08;
                        let health_bar_y = cell_y + token_margin - health_bar_height - 2.0;
                        let health_bar_rect = egui::Rect::from_min_size(
                            egui::pos2(cell_x + token_margin, health_bar_y),
                            egui::vec2(token_size, health_bar_height),
                        );
                        
                        // Health bar background (dark red)
                        ui.painter().rect_filled(
                            health_bar_rect,
                            2.0,
                            egui::Color32::from_rgb(60, 20, 20),
                        );
                        
                        // Health bar fill (green to red based on health)
                        let health_pct = char.health_percentage();
                        let health_color = if health_pct > 0.5 {
                            egui::Color32::from_rgb(50, 200, 50)
                        } else if health_pct > 0.25 {
                            egui::Color32::from_rgb(200, 200, 50)
                        } else {
                            egui::Color32::from_rgb(200, 50, 50)
                        };
                        
                        let health_fill_rect = egui::Rect::from_min_size(
                            health_bar_rect.min,
                            egui::vec2(token_size * health_pct, health_bar_height),
                        );
                        ui.painter().rect_filled(health_fill_rect, 2.0, health_color);
                        
                        // Health bar border
                        ui.painter().rect_stroke(
                            health_bar_rect,
                            2.0,
                            egui::Stroke::new(1.0, egui::Color32::BLACK),
                        );
                    }
                    
                    // Draw dragged character at drag position
                    if let (Some(char_idx), Some(pos)) = (dragged_char_index, drag_position) {
                        let char = &mut chars[char_idx];
                        
                        // Convert projector pos to screen pos
                        let screen_x = rect.min.x + pos.0 * (available_size.x / self.projector_width);
                        let screen_y = rect.min.y + pos.1 * (available_size.y / self.projector_height);
                        
                        // Center the token on the pinch point
                        let token_margin = cell_size * 0.1;
                        let token_size = cell_size - token_margin * 2.0;
                        let token_rect = egui::Rect::from_center_size(
                            egui::pos2(screen_x, screen_y),
                            egui::vec2(token_size, token_size),
                        );
                        
                        // Draw with slight transparency and glow effect
                        let glow_color = egui::Color32::from_rgba_unmultiplied(100, 200, 255, 100);
                        ui.painter().circle_filled(
                            token_rect.center(),
                            token_size * 0.6,
                            glow_color,
                        );
                        
                        if let Some(ref token_img) = char.token_image {
                            let char_name = char.name().to_string();
                            let texture = char.token_texture.get_or_insert_with(|| {
                                ctx.load_texture(
                                    format!("token_{}", char_name),
                                    token_img.clone(),
                                    egui::TextureOptions::LINEAR,
                                )
                            });
                            
                            ui.painter().image(
                                texture.id(),
                                token_rect,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        } else {
                            let placeholder_color = if char.is_enemy() {
                                egui::Color32::from_rgb(150, 80, 80)
                            } else {
                                egui::Color32::from_rgb(100, 100, 150)
                            };
                            ui.painter().circle_filled(
                                token_rect.center(),
                                token_size / 2.0,
                                placeholder_color,
                            );
                            ui.painter().text(
                                token_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &char.name().chars().next().unwrap_or('?').to_string(),
                                egui::FontId::proportional(token_size * 0.5),
                                egui::Color32::WHITE,
                            );
                        }
                    }
                }
                
                // Handle hover detection for character info panel (only when not pinching/clicking/dragging)
                // Combine finger tracking with mouse hover
                let now = Instant::now();
                let hover_position = if mouse_pos_projector.is_some() && !mouse_clicked {
                    // Mouse is hovering (not clicking) - use mouse position
                    mouse_pos_projector
                } else {
                    // Use finger position for hover
                    finger_projector
                };
                
                if !effective_pinching && matches!(self.drag_state, DragState::Idle) {
                    if let Some((px, py)) = hover_position {
                        let _cell_size = rect.height() / self.grid_rows as f32;
                        let screen_x = rect.min.x + px * (available_size.x / self.projector_width);
                        let screen_y = rect.min.y + py * (available_size.y / self.projector_height);
                        
                        // Check which character (if any) the pointer is over
                        let hovered_char = if let Some(cell) = self.projector_to_grid_cell(screen_x, screen_y, &rect) {
                            self.find_character_at_cell(cell)
                        } else {
                            None
                        };
                        
                        if let Some(char_idx) = hovered_char {
                            // Pointer is over a character
                            match self.hover_state {
                                Some((prev_idx, start_time)) if prev_idx == char_idx => {
                                    // Still hovering over the same character
                                    let hover_duration = now.duration_since(start_time).as_millis() as u64;
                                    if hover_duration >= self.hover_show_info_ms {
                                        // Trigger info panel display
                                        if self.info_panel_state.map(|(idx, _)| idx != char_idx).unwrap_or(true) {
                                            self.info_panel_state = Some((char_idx, now));
                                        }
                                    }
                                }
                                _ => {
                                    // Started hovering over a new character
                                    self.hover_state = Some((char_idx, now));
                                }
                            }
                        } else {
                            // Not hovering over any character
                            self.hover_state = None;
                        }
                    } else {
                        // No pointer detected
                        self.hover_state = None;
                    }
                } else {
                    // Pinching/clicking or dragging, reset hover state
                    self.hover_state = None;
                }
                
                // Check if info panel should be hidden
                if let Some((_, show_time)) = self.info_panel_state {
                    let display_duration = now.duration_since(show_time).as_millis() as u64;
                    if display_duration >= self.hover_info_display_ms {
                        self.info_panel_state = None;
                    }
                }
                
                // Draw character/enemy info panel if active
                if let Some((char_idx, _)) = self.info_panel_state {
                    let chars = self.characters.lock().unwrap();
                    if let Some(char) = chars.get(char_idx) {
                        let cell_size = rect.height() / self.grid_rows as f32;
                        
                        // Position the panel near the character
                        let cell_x = rect.min.x + char.grid_pos.0 as f32 * cell_size;
                        let cell_y = rect.min.y + char.grid_pos.1 as f32 * cell_size;
                        
                        // Panel dimensions - smaller for enemies
                        let panel_width = cell_size * 3.5;
                        let panel_height = if char.is_enemy() {
                            cell_size * 1.2 // Smaller panel for enemies
                        } else {
                            cell_size * 2.8
                        };
                        let panel_margin = cell_size * 0.2;
                        
                        // Position panel to the right of the character, or left if it would go off-screen
                        let panel_x = if cell_x + cell_size + panel_width + panel_margin < rect.max.x {
                            cell_x + cell_size + panel_margin
                        } else {
                            cell_x - panel_width - panel_margin
                        };
                        
                        // Keep panel vertically centered with character but within bounds
                        let panel_y = (cell_y + cell_size / 2.0 - panel_height / 2.0)
                            .max(rect.min.y + panel_margin)
                            .min(rect.max.y - panel_height - panel_margin);
                        
                        let panel_rect = egui::Rect::from_min_size(
                            egui::pos2(panel_x, panel_y),
                            egui::vec2(panel_width, panel_height),
                        );
                        
                        // Draw panel background with semi-transparent dark background
                        ui.painter().rect_filled(
                            panel_rect,
                            8.0,
                            egui::Color32::from_rgba_unmultiplied(20, 20, 30, 230),
                        );
                        
                        // Different border color for enemies vs characters
                        let border_color = if char.is_enemy() {
                            egui::Color32::from_rgb(200, 100, 100) // Red for enemies
                        } else {
                            egui::Color32::from_rgb(100, 150, 200) // Blue for characters
                        };
                        ui.painter().rect_stroke(
                            panel_rect,
                            8.0,
                            egui::Stroke::new(2.0, border_color),
                        );
                        
                        // Text styling
                        let padding = cell_size * 0.15;
                        let line_height = cell_size * 0.28;
                        let mut y_offset = panel_rect.min.y + padding;
                        
                        // Name (larger, bold) - different color for enemies
                        let name_color = if char.is_enemy() {
                            egui::Color32::from_rgb(255, 150, 100) // Orange-red for enemies
                        } else {
                            egui::Color32::from_rgb(255, 220, 100) // Gold for characters
                        };
                        ui.painter().text(
                            egui::pos2(panel_rect.min.x + padding, y_offset),
                            egui::Align2::LEFT_TOP,
                            char.name(),
                            egui::FontId::proportional(line_height * 1.2),
                            name_color,
                        );
                        y_offset += line_height * 1.4;
                        
                        // Display content based on entity type
                        match &char.entity {
                            EntityType::Character(config) => {
                                // Race + Class for characters
                                let race_class = if config.race.is_empty() {
                                    config.class.clone()
                                } else {
                                    format!("{} {}", config.race, config.class)
                                };
                                ui.painter().text(
                                    egui::pos2(panel_rect.min.x + padding, y_offset),
                                    egui::Align2::LEFT_TOP,
                                    &race_class,
                                    egui::FontId::proportional(line_height * 0.85),
                                    egui::Color32::from_rgb(180, 180, 200),
                                );
                                y_offset += line_height;
                                
                                // HP with color based on health
                                let health_pct = char.health_percentage();
                                let hp_color = if health_pct > 0.5 {
                                    egui::Color32::from_rgb(100, 255, 100)
                                } else if health_pct > 0.25 {
                                    egui::Color32::from_rgb(255, 255, 100)
                                } else {
                                    egui::Color32::from_rgb(255, 100, 100)
                                };
                                ui.painter().text(
                                    egui::pos2(panel_rect.min.x + padding, y_offset),
                                    egui::Align2::LEFT_TOP,
                                    &format!("HP: {} / {}", config.health, config.hit_points),
                                    egui::FontId::proportional(line_height * 0.85),
                                    hp_color,
                                );
                                y_offset += line_height;
                                
                                // Armor Class
                                ui.painter().text(
                                    egui::pos2(panel_rect.min.x + padding, y_offset),
                                    egui::Align2::LEFT_TOP,
                                    &format!("AC: {}", config.armor_class),
                                    egui::FontId::proportional(line_height * 0.85),
                                    egui::Color32::from_rgb(150, 200, 255),
                                );
                                y_offset += line_height;
                                
                                // Current weapon
                                if !config.current_weapon.is_empty() {
                                    ui.painter().text(
                                        egui::pos2(panel_rect.min.x + padding, y_offset),
                                        egui::Align2::LEFT_TOP,
                                        &format!("Weapon: {}", config.current_weapon),
                                        egui::FontId::proportional(line_height * 0.85),
                                        egui::Color32::from_rgb(255, 180, 150),
                                    );
                                    y_offset += line_height;
                                    
                                    // Find weapon details
                                    if let Some(weapon) = config.weapons.iter().find(|w| w.name == config.current_weapon) {
                                        let modifier_str = if weapon.modifier >= 0 {
                                            format!("+{}", weapon.modifier)
                                        } else {
                                            format!("{}", weapon.modifier)
                                        };
                                        ui.painter().text(
                                            egui::pos2(panel_rect.min.x + padding, y_offset),
                                            egui::Align2::LEFT_TOP,
                                            &format!("  {} ({})", weapon.damage, modifier_str),
                                            egui::FontId::proportional(line_height * 0.75),
                                            egui::Color32::from_rgb(200, 150, 130),
                                        );
                                    }
                                }
                                
                                // Status indicator if dead
                                if config.dead {
                                    ui.painter().text(
                                        egui::pos2(panel_rect.center().x, panel_rect.max.y - padding),
                                        egui::Align2::CENTER_BOTTOM,
                                        "☠️ DEAD",
                                        egui::FontId::proportional(line_height * 0.9),
                                        egui::Color32::from_rgb(255, 50, 50),
                                    );
                                }
                            }
                            EntityType::Enemy(config) => {
                                // Type only for enemies (simpler display)
                                ui.painter().text(
                                    egui::pos2(panel_rect.min.x + padding, y_offset),
                                    egui::Align2::LEFT_TOP,
                                    &config.enemy_type,
                                    egui::FontId::proportional(line_height * 0.85),
                                    egui::Color32::from_rgb(200, 150, 150),
                                );
                                
                                // Status indicator if dead
                                if config.dead {
                                    ui.painter().text(
                                        egui::pos2(panel_rect.center().x, panel_rect.max.y - padding),
                                        egui::Align2::CENTER_BOTTOM,
                                        "☠️ DEAD",
                                        egui::FontId::proportional(line_height * 0.9),
                                        egui::Color32::from_rgb(255, 50, 50),
                                    );
                                }
                            }
                        }
                    }
                }
                
                // Draw pickup/drop loading indicators
                {
                    let cell_size = rect.height() / self.grid_rows as f32;
                    let loading_radius = cell_size * 0.4;
                    
                    match &self.drag_state {
                        DragState::PickingUp { start_time, position, .. } => {
                            let elapsed_ms = Instant::now().duration_since(*start_time).as_millis() as u64;
                            let progress_ms = elapsed_ms.saturating_sub(self.pinch_show_loading_ms);
                            let progress = (progress_ms as f32 / self.pinch_pickup_ms as f32).min(1.0);
                            
                            let screen_x = rect.min.x + position.0 * (available_size.x / self.projector_width);
                            let screen_y = rect.min.y + position.1 * (available_size.y / self.projector_height);
                            
                            Self::draw_loading_circle(
                                ui.painter(),
                                egui::pos2(screen_x, screen_y),
                                loading_radius,
                                progress,
                                egui::Color32::from_rgb(255, 100, 100), // Red for pickup
                                4.0,
                            );
                        }
                        DragState::DroppingOff { start_time, position, .. } => {
                            let elapsed_ms = Instant::now().duration_since(*start_time).as_millis() as u64;
                            let progress = (elapsed_ms as f32 / self.drop_confirm_ms as f32).min(1.0);
                            
                            let screen_x = rect.min.x + position.0 * (available_size.x / self.projector_width);
                            let screen_y = rect.min.y + position.1 * (available_size.y / self.projector_height);
                            
                            Self::draw_loading_circle(
                                ui.painter(),
                                egui::pos2(screen_x, screen_y),
                                loading_radius,
                                progress,
                                egui::Color32::from_rgb(100, 255, 100), // Green for drop
                                4.0,
                            );
                        }
                        _ => {}
                    }
                }
                
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
                        
                        // Also draw mouse cursor indicator when mouse is being used for interaction
                        // (only if not already showing finger, and mouse is clicking or dragging)
                        if finger_projector.is_none() || (mouse_clicked && mouse_pos_projector.is_some()) {
                            if let Some((mx, my)) = mouse_pos_projector {
                                let screen_x = rect.min.x + mx * scale_x;
                                let screen_y = rect.min.y + my * scale_y;
                                
                                // Draw mouse cursor indicator (different style from finger)
                                let mouse_color = if mouse_clicked {
                                    egui::Color32::from_rgb(100, 255, 100) // Green when clicking
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(
                                        self.circle_color.r(),
                                        self.circle_color.g(), 
                                        self.circle_color.b(),
                                        128 // Semi-transparent when just hovering
                                    )
                                };
                                
                                // Only draw if finger isn't at the same position
                                let should_draw = if let Some((fx, fy)) = finger_projector {
                                    let dist = ((mx - fx).powi(2) + (my - fy).powi(2)).sqrt();
                                    dist > 50.0 // Only draw if significantly different position
                                } else {
                                    true
                                };
                                
                                if should_draw {
                                    ui.painter().circle_filled(
                                        egui::pos2(screen_x, screen_y),
                                        self.circle_radius * 0.8, // Slightly smaller than finger
                                        mouse_color,
                                    );
                                    if mouse_clicked {
                                        ui.painter().circle_stroke(
                                            egui::pos2(screen_x, screen_y),
                                            self.circle_radius * 0.8,
                                            egui::Stroke::new(2.0, egui::Color32::WHITE),
                                        );
                                    }
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
    let device_path = "/dev/video4";

    log::info!("Starting IntIm-DnD application");
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
        Box::new(|_cc| Ok(Box::new(IntImDnDApp::new(state)))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;

    Ok(())
}

