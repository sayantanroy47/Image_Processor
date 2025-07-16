//! Core data models for image processing operations

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

/// Unique identifier for processing jobs
pub type JobId = Uuid;

/// Input for image processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInput {
    pub job_id: JobId,
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub operations: Vec<ProcessingOperation>,
    pub options: ProcessingOptions,
    pub file_size: u64,
    pub format: ImageFormat,
}

/// Output from image processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOutput {
    pub job_id: JobId,
    pub output_path: PathBuf,
    pub file_size: u64,
    pub format: ImageFormat,
    pub processing_time: Duration,
    pub operations_applied: Vec<String>,
    pub metadata: Option<ImageMetadata>,
}

/// Available processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingOperation {
    Convert { 
        format: ImageFormat, 
        quality: Option<u8> 
    },
    Resize { 
        width: Option<u32>, 
        height: Option<u32>, 
        algorithm: ResizeAlgorithm 
    },
    Watermark { 
        config: WatermarkConfig 
    },
    RemoveBackground { 
        model: BgRemovalModel 
    },
    ColorCorrect { 
        adjustments: ColorAdjustments 
    },
    Crop { 
        region: CropRegion 
    },
    Rotate { 
        angle: f32 
    },
    AddText { 
        text_config: TextConfig 
    },
    CreateCollage { 
        layout: CollageLayout 
    },
}

/// Processing options that apply to all operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    pub preserve_metadata: bool,
    pub create_backup: bool,
    pub quality_priority: QualityPriority,
    pub hardware_acceleration: bool,
    pub timeout: Option<Duration>,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            preserve_metadata: true,
            create_backup: true,
            quality_priority: QualityPriority::Balanced,
            hardware_acceleration: true,
            timeout: Some(Duration::from_secs(300)), // 5 minutes
        }
    }
}

/// Quality vs speed priority
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QualityPriority {
    Speed,
    Balanced,
    Quality,
}

/// Resize algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ResizeAlgorithm {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
    Mitchell,
}

impl Default for ResizeAlgorithm {
    fn default() -> Self {
        ResizeAlgorithm::Lanczos
    }
}

/// Watermark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkConfig {
    pub watermark_path: PathBuf,
    pub positions: Vec<WatermarkPosition>,
    pub opacity: f32,
    pub scale: f32,
    pub blend_mode: BlendMode,
}

/// Watermark positioning options
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WatermarkPosition {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Custom { x: f32, y: f32 }, // Percentage-based positioning
}

/// Blend modes for watermarks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    HardLight,
}

impl Default for BlendMode {
    fn default() -> Self {
        BlendMode::Normal
    }
}

/// Background removal models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BgRemovalModel {
    U2Net,
    MODNet,
    Custom(PathBuf),
}

impl Default for BgRemovalModel {
    fn default() -> Self {
        BgRemovalModel::U2Net
    }
}

/// Color correction adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAdjustments {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
    pub gamma: f32,
    pub white_balance: Option<WhiteBalance>,
    pub curves: Option<CurveAdjustments>,
}

impl Default for ColorAdjustments {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            hue: 0.0,
            gamma: 1.0,
            white_balance: None,
            curves: None,
        }
    }
}

/// White balance adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteBalance {
    pub temperature: f32,
    pub tint: f32,
}

/// Curve adjustments for highlights, midtones, shadows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveAdjustments {
    pub highlights: f32,
    pub midtones: f32,
    pub shadows: f32,
}

/// Crop region specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Text overlay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub text: String,
    pub font_family: String,
    pub font_size: f32,
    pub color: Color,
    pub position: TextPosition,
    pub effects: TextEffects,
}

/// Text positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextPosition {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Custom { x: f32, y: f32 },
}

/// Text effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEffects {
    pub shadow: Option<Shadow>,
    pub outline: Option<Outline>,
    pub gradient: Option<Gradient>,
}

impl Default for TextEffects {
    fn default() -> Self {
        Self {
            shadow: None,
            outline: None,
            gradient: None,
        }
    }
}

/// Shadow effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadow {
    pub color: Color,
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
}

/// Outline effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outline {
    pub color: Color,
    pub width: f32,
}

/// Gradient effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    pub start_color: Color,
    pub end_color: Color,
    pub direction: GradientDirection,
}

/// Gradient direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GradientDirection {
    Horizontal,
    Vertical,
    Diagonal,
}

/// Color representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    pub fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    pub fn white() -> Self {
        Self::rgb(255, 255, 255)
    }
}

/// Collage layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollageLayout {
    pub template: CollageTemplate,
    pub spacing: u32,
    pub background_color: Color,
    pub border_width: u32,
    pub border_color: Color,
}

/// Collage templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollageTemplate {
    Grid { rows: u32, cols: u32 },
    Mosaic { pattern: MosaicPattern },
    Freeform { positions: Vec<ImagePosition> },
}

/// Mosaic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MosaicPattern {
    Random,
    Spiral,
    Hexagonal,
}

/// Image position in collage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePosition {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
}

/// Image metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub exif: Option<ExifData>,
    pub iptc: Option<IptcData>,
    pub xmp: Option<XmpData>,
    pub custom: HashMap<String, String>,
}

/// EXIF metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExifData {
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens_model: Option<String>,
    pub focal_length: Option<f32>,
    pub aperture: Option<f32>,
    pub shutter_speed: Option<String>,
    pub iso: Option<u32>,
    pub date_taken: Option<DateTime<Utc>>,
    pub gps_coordinates: Option<GpsCoordinates>,
    pub orientation: Option<u8>,
}

/// IPTC metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IptcData {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub author: Option<String>,
    pub copyright: Option<String>,
    pub location: Option<String>,
}

/// XMP metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XmpData {
    pub creator: Option<String>,
    pub rights: Option<String>,
    pub description: Option<String>,
    pub subject: Vec<String>,
    pub custom_fields: HashMap<String, String>,
}

/// GPS coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpsCoordinates {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

/// Processing job representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingJob {
    pub id: JobId,
    pub input: ProcessingInput,
    pub status: JobStatus,
    pub priority: JobPriority,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub progress: f32, // 0.0 to 1.0
}

/// Job status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job priority
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Default for JobPriority {
    fn default() -> Self {
        JobPriority::Normal
    }
}

/// Processing progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingProgress {
    pub job_id: JobId,
    pub current_operation: String,
    pub operations_completed: usize,
    pub total_operations: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub estimated_time_remaining: Option<Duration>,
    pub processing_speed: f64, // MB/s
}

impl ProcessingProgress {
    /// Calculate overall progress as a percentage (0.0 to 1.0)
    pub fn overall_progress(&self) -> f32 {
        if self.total_operations == 0 {
            return 0.0;
        }
        
        let operation_progress = self.operations_completed as f32 / self.total_operations as f32;
        let byte_progress = if self.total_bytes > 0 {
            self.bytes_processed as f32 / self.total_bytes as f32
        } else {
            0.0
        };
        
        // Weight operation progress more heavily than byte progress
        (operation_progress * 0.7 + byte_progress * 0.3).min(1.0)
    }
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch_id: Uuid,
    pub total_jobs: usize,
    pub successful_jobs: usize,
    pub failed_jobs: usize,
    pub total_processing_time: Duration,
    pub results: Vec<JobResult>,
}

/// Individual job result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: JobId,
    pub status: JobStatus,
    pub output: Option<ProcessingOutput>,
    pub error: Option<ProcessingError>,
    pub processing_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_progress_calculation() {
        let progress = ProcessingProgress {
            job_id: Uuid::new_v4(),
            current_operation: "Resize".to_string(),
            operations_completed: 2,
            total_operations: 5,
            bytes_processed: 500,
            total_bytes: 1000,
            estimated_time_remaining: Some(Duration::from_secs(30)),
            processing_speed: 10.0,
        };
        
        let overall = progress.overall_progress();
        assert!(overall > 0.0 && overall <= 1.0);
        
        // Should be weighted combination: (2/5 * 0.7) + (500/1000 * 0.3) = 0.28 + 0.15 = 0.43
        assert!((overall - 0.43).abs() < 0.01);
    }

    #[test]
    fn test_color_creation() {
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
        assert_eq!(color.a, 255);
        
        let black = Color::black();
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Critical > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    #[test]
    fn test_default_values() {
        let options = ProcessingOptions::default();
        assert!(options.preserve_metadata);
        assert!(options.create_backup);
        assert!(matches!(options.quality_priority, QualityPriority::Balanced));
        
        let algorithm = ResizeAlgorithm::default();
        assert!(matches!(algorithm, ResizeAlgorithm::Lanczos));
    }

    #[test]
    fn test_image_format_properties() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert!(ImageFormat::WebP.supports_transparency());
        assert!(ImageFormat::Gif.supports_animation());
    }

    #[test]
    fn test_serialization() {
        let operation = ProcessingOperation::Resize {
            width: Some(800),
            height: Some(600),
            algorithm: ResizeAlgorithm::Lanczos,
        };
        
        let json = serde_json::to_string(&operation).unwrap();
        let deserialized: ProcessingOperation = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            ProcessingOperation::Resize { width, height, algorithm } => {
                assert_eq!(width, Some(800));
                assert_eq!(height, Some(600));
                assert!(matches!(algorithm, ResizeAlgorithm::Lanczos));
            }
            _ => panic!("Deserialization failed"),
        }
    }
}