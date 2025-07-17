//! Watermark processing engine for applying watermarks to images

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use crate::models::{ProcessingInput, ProcessingOutput, ProcessingOperation, WatermarkConfig, WatermarkPosition, BlendMode};
use crate::processing::{
    ImageProcessor, ProcessorCapabilities, ProcessorMetadata, ProcessorOperation as ProcOp,
    PerformanceProfile, MemoryUsage, CpuUsage,
};
use async_trait::async_trait;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use std::path::{Path, PathBuf};
use tokio::task;
use tracing::{debug, info, instrument, warn};

/// Default margin for watermark positioning
const DEFAULT_MARGIN: u32 = 10;

/// Position calculator for watermark placement
#[derive(Debug)]
pub struct PositionCalculator;

impl PositionCalculator {
    /// Calculate the actual pixel position for watermark placement
    pub fn calculate_position(
        position: WatermarkPosition,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
        margin: u32,
    ) -> (u32, u32) {
        match position {
            WatermarkPosition::TopLeft => (margin, margin),
            WatermarkPosition::TopCenter => (
                (image_width.saturating_sub(watermark_width)) / 2,
                margin,
            ),
            WatermarkPosition::TopRight => {
                (image_width.saturating_sub(watermark_width + margin), margin)
            }
            WatermarkPosition::CenterLeft => (
                margin,
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::Center => (
                (image_width.saturating_sub(watermark_width)) / 2,
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::CenterRight => (
                image_width.saturating_sub(watermark_width + margin),
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::BottomLeft => {
                (margin, image_height.saturating_sub(watermark_height + margin))
            }
            WatermarkPosition::BottomCenter => (
                (image_width.saturating_sub(watermark_width)) / 2,
                image_height.saturating_sub(watermark_height + margin),
            ),
            WatermarkPosition::BottomRight => (
                image_width.saturating_sub(watermark_width + margin),
                image_height.saturating_sub(watermark_height + margin),
            ),
            WatermarkPosition::Custom { x, y } => {
                // Custom positions are percentage-based (0.0 to 1.0)
                let x_pos = ((image_width as f32 * x.clamp(0.0, 1.0)) as u32)
                    .min(image_width.saturating_sub(watermark_width));
                let y_pos = ((image_height as f32 * y.clamp(0.0, 1.0)) as u32)
                    .min(image_height.saturating_sub(watermark_height));
                (x_pos, y_pos)
            }
        }
    }
}

/// Watermark processing engine
#[derive(Debug)]
pub struct WatermarkEngine {
    /// Supported input formats
    supported_input_formats: Vec<ImageFormat>,
    /// Supported output formats
    supported_output_formats: Vec<ImageFormat>,
    /// Position calculator
    position_calculator: PositionCalculator,
}

impl WatermarkEngine {
    /// Create a new watermark engine
    pub fn new() -> Self {
        Self {
            supported_input_formats: vec![
                ImageFormat::Jpeg,
                ImageFormat::Png,
                ImageFormat::WebP,
                ImageFormat::Gif,
                ImageFormat::Bmp,
                ImageFormat::Tiff,
            ],
            supported_output_formats: vec![
                ImageFormat::Jpeg,
                ImageFormat::Png,
                ImageFormat::WebP,
                ImageFormat::Gif,
                ImageFormat::Bmp,
                ImageFormat::Tiff,
            ],
            position_calculator: PositionCalculator,
        }
    }

    /// Load and validate a watermark image
    async fn load_watermark(&self, watermark_path: &Path) -> Result<DynamicImage> {
        let path = watermark_path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Loading watermark from: {}", path.display());
            
            if !path.exists() {
                return Err(ProcessingError::FileNotFound { path });
            }
            
            let watermark = image::open(&path)
                .map_err(|e| ProcessingError::ProcessingFailed {
                    message: format!("Failed to load watermark {}: {}", path.display(), e),
                })?;
            
            debug!("Successfully loaded watermark: {}x{}", watermark.width(), watermark.height());
            Ok(watermark)
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }

    /// Scale a watermark image
    fn scale_watermark(&self, watermark: DynamicImage, scale: f32) -> DynamicImage {
        if scale <= 0.0 || scale > 1.0 {
            warn!("Invalid scale factor {}, using 1.0", scale);
            return watermark;
        }
        
        let new_width = (watermark.width() as f32 * scale) as u32;
        let new_height = (watermark.height() as f32 * scale) as u32;
        
        if new_width == 0 || new_height == 0 {
            warn!("Scaled watermark would be too small, using minimum size");
            return watermark.resize(1, 1, image::imageops::FilterType::Lanczos3);
        }
        
        debug!("Scaling watermark from {}x{} to {}x{}", 
               watermark.width(), watermark.height(), new_width, new_height);
        
        watermark.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
    }

    /// Apply alpha blending between two RGBA pixels
    fn blend_pixels(&self, base: Rgba<u8>, overlay: Rgba<u8>, opacity: f32, blend_mode: BlendMode) -> Rgba<u8> {
        let base_alpha = base[3] as f32 / 255.0;
        let overlay_alpha = (overlay[3] as f32 / 255.0) * opacity.clamp(0.0, 1.0);
        
        if overlay_alpha == 0.0 {
            return base;
        }
        
        let final_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha);
        
        if final_alpha == 0.0 {
            return Rgba([0, 0, 0, 0]);
        }
        
        let blend_color = match blend_mode {
            BlendMode::Normal => overlay,
            BlendMode::Multiply => {
                let r = (base[0] as f32 * overlay[0] as f32 / 255.0) as u8;
                let g = (base[1] as f32 * overlay[1] as f32 / 255.0) as u8;
                let b = (base[2] as f32 * overlay[2] as f32 / 255.0) as u8;
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Screen => {
                let r = (255.0 - (255.0 - base[0] as f32) * (255.0 - overlay[0] as f32) / 255.0) as u8;
                let g = (255.0 - (255.0 - base[1] as f32) * (255.0 - overlay[1] as f32) / 255.0) as u8;
                let b = (255.0 - (255.0 - base[2] as f32) * (255.0 - overlay[2] as f32) / 255.0) as u8;
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Overlay => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = if base_f < 0.5 {
                        2.0 * base_f * overlay_f
                    } else {
                        1.0 - 2.0 * (1.0 - base_f) * (1.0 - overlay_f)
                    };
                    (result * 255.0) as u8
                };
                
                let r = blend_channel(base[0], overlay[0]);
                let g = blend_channel(base[1], overlay[1]);
                let b = blend_channel(base[2], overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::SoftLight => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = if overlay_f < 0.5 {
                        2.0 * base_f * overlay_f + base_f * base_f * (1.0 - 2.0 * overlay_f)
                    } else {
                        2.0 * base_f * (1.0 - overlay_f) + (2.0 * overlay_f - 1.0) * base_f.sqrt()
                    };
                    (result * 255.0) as u8
                };
                
                let r = blend_channel(base[0], overlay[0]);
                let g = blend_channel(base[1], overlay[1]);
                let b = blend_channel(base[2], overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::HardLight => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = if overlay_f < 0.5 {
                        2.0 * base_f * overlay_f
                    } else {
                        1.0 - 2.0 * (1.0 - base_f) * (1.0 - overlay_f)
                    };
                    (result * 255.0) as u8
                };
                
                let r = blend_channel(base[0], overlay[0]);
                let g = blend_channel(base[1], overlay[1]);
                let b = blend_channel(base[2], overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
        };
        
        // Alpha compositing
        let inv_alpha = 1.0 - overlay_alpha;
        let r = (blend_color[0] as f32 * overlay_alpha + base[0] as f32 * base_alpha * inv_alpha) / final_alpha;
        let g = (blend_color[1] as f32 * overlay_alpha + base[1] as f32 * base_alpha * inv_alpha) / final_alpha;
        let b = (blend_color[2] as f32 * overlay_alpha + base[2] as f32 * base_alpha * inv_alpha) / final_alpha;
        
        Rgba([
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
            (final_alpha * 255.0) as u8,
        ])
    }

    /// Apply watermark to an image
    async fn apply_watermark_to_image(
        &self,
        base_image: DynamicImage,
        config: &WatermarkConfig,
    ) -> Result<DynamicImage> {
        // Load and prepare watermark
        let watermark = self.load_watermark(&config.watermark_path).await?;
        let scaled_watermark = self.scale_watermark(watermark, config.scale);
        
        // Convert images to RGBA for blending
        let mut base_rgba = base_image.to_rgba8();
        let watermark_rgba = scaled_watermark.to_rgba8();
        
        // Apply watermark at each position (support multiple positions)
        for position in &config.positions {
            // Calculate position
            let (x_pos, y_pos) = PositionCalculator::calculate_position(
                *position,
                base_rgba.width(),
                base_rgba.height(),
                watermark_rgba.width(),
                watermark_rgba.height(),
                DEFAULT_MARGIN,
            );
            
            debug!("Applying watermark at position ({}, {}) for {:?}", x_pos, y_pos, position);
            
            // Apply watermark pixel by pixel
            for (watermark_x, watermark_y, watermark_pixel) in watermark_rgba.enumerate_pixels() {
                let base_x = x_pos + watermark_x;
                let base_y = y_pos + watermark_y;
                
                // Check bounds
                if base_x < base_rgba.width() && base_y < base_rgba.height() {
                    let base_pixel = *base_rgba.get_pixel(base_x, base_y);
                    let blended_pixel = self.blend_pixels(
                        base_pixel,
                        *watermark_pixel,
                        config.opacity,
                        config.blend_mode,
                    );
                    base_rgba.put_pixel(base_x, base_y, blended_pixel);
                }
            }
        }
        
        Ok(DynamicImage::ImageRgba8(base_rgba))
    }

    /// Save the watermarked image
    async fn save_watermarked_image(
        &self,
        image: DynamicImage,
        output_path: &Path,
        format: ImageFormat,
    ) -> Result<u64> {
        let path = output_path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Saving watermarked image to: {}", path.display());
            
            // Ensure output directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| ProcessingError::Io {
                        message: format!("Failed to create output directory: {}", e),
                    })?;
            }
            
            // Save the image
            let output_format = match format {
                ImageFormat::Jpeg => image::ImageFormat::Jpeg,
                ImageFormat::Png => image::ImageFormat::Png,
                ImageFormat::WebP => image::ImageFormat::WebP,
                ImageFormat::Gif => image::ImageFormat::Gif,
                ImageFormat::Bmp => image::ImageFormat::Bmp,
                ImageFormat::Tiff => image::ImageFormat::Tiff,
                _ => return Err(ProcessingError::UnsupportedFormat {
                    format: format!("{:?}", format),
                }),
            };
            
            image.save_with_format(&path, output_format)
                .map_err(|e| ProcessingError::ProcessingFailed {
                    message: format!("Failed to save watermarked image: {}", e),
                })?;
            
            // Get file size
            let metadata = std::fs::metadata(&path)
                .map_err(|e| ProcessingError::Io {
                    message: format!("Failed to get file metadata: {}", e),
                })?;
            
            let file_size = metadata.len();
            debug!("Successfully saved watermarked image: {} bytes", file_size);
            
            Ok(file_size)
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }
}

impl Default for WatermarkEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageProcessor for WatermarkEngine {
    #[instrument(skip(self, input))]
    async fn process(&self, input: &ProcessingInput) -> Result<ProcessingOutput> {
        let start_time = std::time::Instant::now();
        
        // Find the watermark operation
        let watermark_config = input.operations.iter()
            .find_map(|op| match op {
                ProcessingOperation::Watermark { config } => Some(config.clone()),
                _ => None,
            })
            .ok_or_else(|| ProcessingError::ProcessingFailed {
                message: "No watermark operation found".to_string(),
            })?;
        
        // Validate input file exists
        if !input.source_path.exists() {
            return Err(ProcessingError::FileNotFound {
                path: input.source_path.clone(),
            });
        }
        
        info!("Applying watermark from {} to {}", 
              watermark_config.watermark_path.display(), 
              input.source_path.display());
        
        // Load base image
        let base_image = image::open(&input.source_path)
            .map_err(|e| ProcessingError::ProcessingFailed {
                message: format!("Failed to load base image: {}", e),
            })?;
        
        // Apply watermark
        let watermarked_image = self.apply_watermark_to_image(base_image, &watermark_config).await?;
        
        // Save result
        let output_file_size = self.save_watermarked_image(
            watermarked_image,
            &input.output_path,
            input.format,
        ).await?;
        
        let processing_time = start_time.elapsed();
        
        info!("Successfully applied watermark in {}ms, output size: {} bytes", 
              processing_time.as_millis(), output_file_size);
        
        Ok(ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path.clone(),
            file_size: output_file_size,
            format: input.format,
            processing_time,
            operations_applied: vec!["watermark_application".to_string()],
            metadata: None,
        })
    }

    fn supports_format(&self, format: ImageFormat) -> bool {
        self.supported_input_formats.contains(&format)
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            input_formats: self.supported_input_formats.clone(),
            output_formats: self.supported_output_formats.clone(),
            operations: vec![ProcOp::Watermark],
            hardware_acceleration: false, // CPU-based processing
            streaming_support: false, // Requires full image in memory
            max_file_size: Some(100 * 1024 * 1024), // 100MB limit
            batch_optimized: true,
        }
    }

    fn get_metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: "Watermark Engine".to_string(),
            version: "1.0.0".to_string(),
            description: "High-quality watermark application with multiple blend modes and positioning options".to_string(),
            author: "Image Processor Team".to_string(),
            performance_profile: PerformanceProfile {
                speed_factor: 2.0, // Moderate speed due to pixel-level operations
                memory_usage: MemoryUsage::High, // Requires full images in memory
                cpu_usage: CpuUsage::High, // Intensive pixel processing
                parallel_friendly: true, // Can process multiple images in parallel
            },
        }
    }

    fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
        matches!(operation, ProcessingOperation::Watermark { .. })
    }
}