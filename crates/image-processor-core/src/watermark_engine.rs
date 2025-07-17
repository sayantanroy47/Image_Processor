//! Watermark processing engine for applying watermarks to images

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use crate::models::{ProcessingInput, ProcessingOutput, ProcessingOperation, WatermarkConfig, WatermarkPosition, BlendMode, WatermarkScalingOptions, WatermarkAlignment, WatermarkOffset, HorizontalAlign, VerticalAlign, WatermarkVisualEffects, WatermarkShadow, WatermarkOutline, WatermarkGlow, Color};
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

/// Supported watermark formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkFormat {
    Png,
    Jpeg,
    WebP,
    Gif,
    Bmp,
    Tiff,
    Svg,
}

/// Position calculator for watermark placement
#[derive(Debug)]
pub struct PositionCalculator;

impl PositionCalculator {
    /// Calculate the actual pixel position for watermark placement with enhanced alignment and offset support
    pub fn calculate_position(
        position: WatermarkPosition,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
        alignment: WatermarkAlignment,
        offset: WatermarkOffset,
    ) -> (u32, u32) {
        // First calculate base position
        let (base_x, base_y) = Self::calculate_base_position(
            position,
            image_width,
            image_height,
            watermark_width,
            watermark_height,
        );
        
        // Apply alignment adjustments
        let (aligned_x, aligned_y) = Self::apply_alignment(
            base_x,
            base_y,
            image_width,
            image_height,
            watermark_width,
            watermark_height,
            position,
            alignment,
        );
        
        // Apply offset and ensure bounds
        let final_x = (aligned_x as i32 + offset.x)
            .max(0)
            .min((image_width.saturating_sub(watermark_width)) as i32) as u32;
        let final_y = (aligned_y as i32 + offset.y)
            .max(0)
            .min((image_height.saturating_sub(watermark_height)) as i32) as u32;
        
        (final_x, final_y)
    }
    
    /// Calculate base position without alignment or offset
    fn calculate_base_position(
        position: WatermarkPosition,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
    ) -> (u32, u32) {
        match position {
            WatermarkPosition::TopLeft => (0, 0),
            WatermarkPosition::TopCenter => (
                (image_width.saturating_sub(watermark_width)) / 2,
                0,
            ),
            WatermarkPosition::TopRight => (
                image_width.saturating_sub(watermark_width),
                0,
            ),
            WatermarkPosition::CenterLeft => (
                0,
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::Center => (
                (image_width.saturating_sub(watermark_width)) / 2,
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::CenterRight => (
                image_width.saturating_sub(watermark_width),
                (image_height.saturating_sub(watermark_height)) / 2,
            ),
            WatermarkPosition::BottomLeft => (
                0,
                image_height.saturating_sub(watermark_height),
            ),
            WatermarkPosition::BottomCenter => (
                (image_width.saturating_sub(watermark_width)) / 2,
                image_height.saturating_sub(watermark_height),
            ),
            WatermarkPosition::BottomRight => (
                image_width.saturating_sub(watermark_width),
                image_height.saturating_sub(watermark_height),
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
    
    /// Apply alignment adjustments to base position
    fn apply_alignment(
        base_x: u32,
        base_y: u32,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
        position: WatermarkPosition,
        alignment: WatermarkAlignment,
    ) -> (u32, u32) {
        match alignment {
            WatermarkAlignment::Edge => (base_x, base_y),
            WatermarkAlignment::Padded { padding } => {
                Self::apply_padding(base_x, base_y, image_width, image_height, 
                                  watermark_width, watermark_height, position, padding)
            },
            WatermarkAlignment::Centered => {
                // For centered alignment, we keep the base position as it's already centered
                (base_x, base_y)
            },
            WatermarkAlignment::Custom { horizontal, vertical } => {
                Self::apply_custom_alignment(base_x, base_y, image_width, image_height,
                                           watermark_width, watermark_height, position,
                                           horizontal, vertical)
            },
        }
    }
    
    /// Apply padding to position based on watermark location
    fn apply_padding(
        base_x: u32,
        base_y: u32,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
        position: WatermarkPosition,
        padding: u32,
    ) -> (u32, u32) {
        let (x, y) = match position {
            WatermarkPosition::TopLeft => (padding, padding),
            WatermarkPosition::TopCenter => (base_x, padding),
            WatermarkPosition::TopRight => (
                image_width.saturating_sub(watermark_width + padding),
                padding,
            ),
            WatermarkPosition::CenterLeft => (padding, base_y),
            WatermarkPosition::Center => (base_x, base_y),
            WatermarkPosition::CenterRight => (
                image_width.saturating_sub(watermark_width + padding),
                base_y,
            ),
            WatermarkPosition::BottomLeft => (
                padding,
                image_height.saturating_sub(watermark_height + padding),
            ),
            WatermarkPosition::BottomCenter => (
                base_x,
                image_height.saturating_sub(watermark_height + padding),
            ),
            WatermarkPosition::BottomRight => (
                image_width.saturating_sub(watermark_width + padding),
                image_height.saturating_sub(watermark_height + padding),
            ),
            WatermarkPosition::Custom { .. } => (base_x, base_y), // Custom positions ignore padding
        };
        
        (x, y)
    }
    
    /// Apply custom horizontal and vertical alignment
    fn apply_custom_alignment(
        base_x: u32,
        base_y: u32,
        image_width: u32,
        image_height: u32,
        watermark_width: u32,
        watermark_height: u32,
        position: WatermarkPosition,
        horizontal: HorizontalAlign,
        vertical: VerticalAlign,
    ) -> (u32, u32) {
        // For custom positions, use the base position
        if matches!(position, WatermarkPosition::Custom { .. }) {
            return (base_x, base_y);
        }
        
        let x = match horizontal {
            HorizontalAlign::Left => 0,
            HorizontalAlign::Center => (image_width.saturating_sub(watermark_width)) / 2,
            HorizontalAlign::Right => image_width.saturating_sub(watermark_width),
        };
        
        let y = match vertical {
            VerticalAlign::Top => 0,
            VerticalAlign::Middle => (image_height.saturating_sub(watermark_height)) / 2,
            VerticalAlign::Bottom => image_height.saturating_sub(watermark_height),
        };
        
        (x, y)
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

    /// Detect watermark format from file extension and content
    fn detect_watermark_format(&self, path: &Path) -> Result<WatermarkFormat> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        
        match extension.as_str() {
            "svg" => Ok(WatermarkFormat::Svg),
            "png" => Ok(WatermarkFormat::Png),
            "jpg" | "jpeg" => Ok(WatermarkFormat::Jpeg),
            "webp" => Ok(WatermarkFormat::WebP),
            "gif" => Ok(WatermarkFormat::Gif),
            "bmp" => Ok(WatermarkFormat::Bmp),
            "tiff" | "tif" => Ok(WatermarkFormat::Tiff),
            _ => {
                // Try to detect from file content
                let content = std::fs::read(path).map_err(|e| ProcessingError::Io {
                    message: format!("Failed to read watermark file: {}", e),
                })?;
                
                if content.starts_with(b"<svg") || content.starts_with(b"<?xml") {
                    Ok(WatermarkFormat::Svg)
                } else if content.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
                    Ok(WatermarkFormat::Png)
                } else if content.starts_with(&[0xFF, 0xD8, 0xFF]) {
                    Ok(WatermarkFormat::Jpeg)
                } else {
                    Err(ProcessingError::UnsupportedFormat {
                        format: format!("Unknown watermark format for file: {}", path.display()),
                    })
                }
            }
        }
    }

    /// Render SVG to raster image
    async fn render_svg_watermark(&self, svg_path: &Path, target_width: Option<u32>, target_height: Option<u32>) -> Result<DynamicImage> {
        let path = svg_path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Rendering SVG watermark from: {}", path.display());
            
            let svg_content = std::fs::read_to_string(&path)
                .map_err(|e| ProcessingError::ProcessingFailed {
                    message: format!("Failed to read SVG file: {}", e),
                })?;
            
            // Parse SVG dimensions if not provided
            let (width, height) = if let (Some(w), Some(h)) = (target_width, target_height) {
                (w, h)
            } else {
                Self::parse_svg_dimensions(&svg_content).unwrap_or((256, 256))
            };
            
            // Create a simple SVG renderer using basic parsing
            // For production, you'd want to use a proper SVG library like resvg
            let rgba_image = Self::render_svg_to_rgba(&svg_content, width, height)?;
            
            debug!("Successfully rendered SVG watermark: {}x{}", width, height);
            Ok(DynamicImage::ImageRgba8(rgba_image))
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }

    /// Parse SVG dimensions from content
    fn parse_svg_dimensions(svg_content: &str) -> Option<(u32, u32)> {
        // Simple regex-based parsing for width and height attributes
        // Match both quoted and unquoted values
        let width_regex = regex::Regex::new(r#"width\s*=\s*["']?(\d+)["']?"#).ok()?;
        let height_regex = regex::Regex::new(r#"height\s*=\s*["']?(\d+)["']?"#).ok()?;
        
        let width_capture = width_regex.captures(svg_content)?;
        let height_capture = height_regex.captures(svg_content)?;
        
        let width = width_capture.get(1)?
            .as_str()
            .parse::<u32>()
            .ok()?;
        
        let height = height_capture.get(1)?
            .as_str()
            .parse::<u32>()
            .ok()?;
        
        Some((width, height))
    }

    /// Basic SVG to RGBA renderer (simplified implementation)
    fn render_svg_to_rgba(svg_content: &str, width: u32, height: u32) -> Result<RgbaImage> {
        // This is a simplified SVG renderer for basic shapes
        // In production, use a proper SVG library like resvg or usvg
        
        let mut image = RgbaImage::new(width, height);
        
        // Fill with transparent background
        for pixel in image.pixels_mut() {
            *pixel = Rgba([255, 255, 255, 0]);
        }
        
        // Parse basic SVG elements (rect, circle, text)
        if svg_content.contains("<rect") {
            Self::render_svg_rect(&mut image, svg_content)?;
        }
        
        if svg_content.contains("<circle") {
            Self::render_svg_circle(&mut image, svg_content)?;
        }
        
        if svg_content.contains("<text") {
            Self::render_svg_text(&mut image, svg_content)?;
        }
        
        Ok(image)
    }

    /// Render SVG rectangle elements
    fn render_svg_rect(image: &mut RgbaImage, svg_content: &str) -> Result<()> {
        // Basic rect parsing and rendering
        // This is a simplified implementation
        let rect_regex = regex::Regex::new(
            r#"<rect[^>]*x\s*=\s*["']?(\d+)["']?[^>]*y\s*=\s*["']?(\d+)["']?[^>]*width\s*=\s*["']?(\d+)["']?[^>]*height\s*=\s*["']?(\d+)["']?[^>]*>"#
        ).map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to parse SVG rect: {}", e),
        })?;
        
        for captures in rect_regex.captures_iter(svg_content) {
            let x = captures.get(1).unwrap().as_str().parse::<u32>().unwrap_or(0);
            let y = captures.get(2).unwrap().as_str().parse::<u32>().unwrap_or(0);
            let width = captures.get(3).unwrap().as_str().parse::<u32>().unwrap_or(0);
            let height = captures.get(4).unwrap().as_str().parse::<u32>().unwrap_or(0);
            
            // Draw rectangle with black color (simplified)
            for dy in 0..height {
                for dx in 0..width {
                    let px = x + dx;
                    let py = y + dy;
                    if px < image.width() && py < image.height() {
                        image.put_pixel(px, py, Rgba([0, 0, 0, 255]));
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Render SVG circle elements
    fn render_svg_circle(image: &mut RgbaImage, svg_content: &str) -> Result<()> {
        // Basic circle parsing and rendering
        let circle_regex = regex::Regex::new(
            r#"<circle[^>]*cx\s*=\s*["']?(\d+)["']?[^>]*cy\s*=\s*["']?(\d+)["']?[^>]*r\s*=\s*["']?(\d+)["']?[^>]*>"#
        ).map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to parse SVG circle: {}", e),
        })?;
        
        for captures in circle_regex.captures_iter(svg_content) {
            let cx = captures.get(1).unwrap().as_str().parse::<i32>().unwrap_or(0);
            let cy = captures.get(2).unwrap().as_str().parse::<i32>().unwrap_or(0);
            let r = captures.get(3).unwrap().as_str().parse::<i32>().unwrap_or(0);
            
            // Draw circle using basic algorithm
            for y in (cy - r)..=(cy + r) {
                for x in (cx - r)..=(cx + r) {
                    let dx = x - cx;
                    let dy = y - cy;
                    if dx * dx + dy * dy <= r * r && x >= 0 && y >= 0 {
                        let px = x as u32;
                        let py = y as u32;
                        if px < image.width() && py < image.height() {
                            image.put_pixel(px, py, Rgba([0, 0, 0, 255]));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Render SVG text elements (simplified)
    fn render_svg_text(_image: &mut RgbaImage, _svg_content: &str) -> Result<()> {
        // Text rendering would require a font library
        // For now, just return Ok as this is a simplified implementation
        debug!("SVG text rendering not implemented in simplified renderer");
        Ok(())
    }

    /// Load and validate a watermark image with format detection
    async fn load_watermark(&self, watermark_path: &Path) -> Result<DynamicImage> {
        let path = watermark_path.to_path_buf();
        
        if !path.exists() {
            return Err(ProcessingError::FileNotFound { path });
        }
        
        let format = self.detect_watermark_format(&path)?;
        
        match format {
            WatermarkFormat::Svg => {
                debug!("Loading SVG watermark from: {}", path.display());
                self.render_svg_watermark(&path, None, None).await
            }
            _ => {
                task::spawn_blocking(move || {
                    debug!("Loading raster watermark from: {}", path.display());
                    
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
        }
    }

    /// Scale a watermark image with enhanced scaling options and aspect ratio preservation
    fn scale_watermark_advanced(
        &self, 
        watermark: DynamicImage, 
        base_image_width: u32,
        base_image_height: u32,
        scale: f32,
        scaling_options: &WatermarkScalingOptions,
    ) -> DynamicImage {
        let original_width = watermark.width();
        let original_height = watermark.height();
        
        // Calculate initial scaled dimensions
        let mut new_width = (original_width as f32 * scale) as u32;
        let mut new_height = (original_height as f32 * scale) as u32;
        
        // Apply percentage-based constraints
        if let Some(max_width_percent) = scaling_options.max_width_percent {
            let max_width = (base_image_width as f32 * max_width_percent) as u32;
            if new_width > max_width {
                new_width = max_width;
                if scaling_options.preserve_aspect_ratio {
                    let aspect_ratio = original_width as f32 / original_height as f32;
                    new_height = (new_width as f32 / aspect_ratio) as u32;
                }
            }
        }
        
        if let Some(max_height_percent) = scaling_options.max_height_percent {
            let max_height = (base_image_height as f32 * max_height_percent) as u32;
            if new_height > max_height {
                new_height = max_height;
                if scaling_options.preserve_aspect_ratio {
                    let aspect_ratio = original_width as f32 / original_height as f32;
                    new_width = (new_height as f32 * aspect_ratio) as u32;
                }
            }
        }
        
        // Apply minimum size constraints
        if let Some((min_width, min_height)) = scaling_options.min_size {
            if new_width < min_width || new_height < min_height {
                if scaling_options.preserve_aspect_ratio {
                    let width_scale = min_width as f32 / original_width as f32;
                    let height_scale = min_height as f32 / original_height as f32;
                    let scale_factor = width_scale.max(height_scale);
                    
                    new_width = (original_width as f32 * scale_factor) as u32;
                    new_height = (original_height as f32 * scale_factor) as u32;
                } else {
                    new_width = new_width.max(min_width);
                    new_height = new_height.max(min_height);
                }
            }
        }
        
        // Apply maximum size constraints
        if let Some((max_width, max_height)) = scaling_options.max_size {
            if new_width > max_width || new_height > max_height {
                if scaling_options.preserve_aspect_ratio {
                    let width_scale = max_width as f32 / new_width as f32;
                    let height_scale = max_height as f32 / new_height as f32;
                    let scale_factor = width_scale.min(height_scale);
                    
                    new_width = (new_width as f32 * scale_factor) as u32;
                    new_height = (new_height as f32 * scale_factor) as u32;
                } else {
                    new_width = new_width.min(max_width);
                    new_height = new_height.min(max_height);
                }
            }
        }
        
        // Ensure minimum viable size
        if new_width == 0 || new_height == 0 {
            warn!("Calculated watermark size would be zero, using minimum size");
            new_width = new_width.max(1);
            new_height = new_height.max(1);
        }
        
        debug!("Scaling watermark from {}x{} to {}x{} (preserve_aspect_ratio: {})", 
               original_width, original_height, new_width, new_height, 
               scaling_options.preserve_aspect_ratio);
        
        // Use high-quality Lanczos3 filter for scaling
        watermark.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
    }
    
    /// Legacy scale method for backward compatibility
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

    /// Apply alpha blending between two RGBA pixels with enhanced blend modes
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
            BlendMode::ColorDodge => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = if overlay_f >= 1.0 {
                        1.0
                    } else {
                        (base_f / (1.0 - overlay_f)).min(1.0)
                    };
                    (result * 255.0) as u8
                };
                
                let r = blend_channel(base[0], overlay[0]);
                let g = blend_channel(base[1], overlay[1]);
                let b = blend_channel(base[2], overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::ColorBurn => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = if overlay_f <= 0.0 {
                        0.0
                    } else {
                        1.0 - ((1.0 - base_f) / overlay_f).min(1.0)
                    };
                    (result * 255.0) as u8
                };
                
                let r = blend_channel(base[0], overlay[0]);
                let g = blend_channel(base[1], overlay[1]);
                let b = blend_channel(base[2], overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Darken => {
                let r = base[0].min(overlay[0]);
                let g = base[1].min(overlay[1]);
                let b = base[2].min(overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Lighten => {
                let r = base[0].max(overlay[0]);
                let g = base[1].max(overlay[1]);
                let b = base[2].max(overlay[2]);
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Difference => {
                let r = ((base[0] as i32 - overlay[0] as i32).abs()) as u8;
                let g = ((base[1] as i32 - overlay[1] as i32).abs()) as u8;
                let b = ((base[2] as i32 - overlay[2] as i32).abs()) as u8;
                Rgba([r, g, b, overlay[3]])
            }
            BlendMode::Exclusion => {
                let blend_channel = |base: u8, overlay: u8| -> u8 {
                    let base_f = base as f32 / 255.0;
                    let overlay_f = overlay as f32 / 255.0;
                    let result = base_f + overlay_f - 2.0 * base_f * overlay_f;
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

    /// Apply shadow effect to a watermark image
    fn apply_shadow_effect(&self, watermark: &RgbaImage, shadow: &WatermarkShadow) -> RgbaImage {
        let width = watermark.width();
        let height = watermark.height();
        
        // Create shadow buffer with padding for blur
        let blur_padding = (shadow.blur_radius * 2.0) as u32;
        let shadow_width = width + blur_padding * 2;
        let shadow_height = height + blur_padding * 2;
        
        let mut shadow_image = RgbaImage::new(shadow_width, shadow_height);
        
        // Create shadow by copying watermark alpha to shadow color
        for (x, y, pixel) in watermark.enumerate_pixels() {
            let shadow_x = x + blur_padding + shadow.offset_x as u32;
            let shadow_y = y + blur_padding + shadow.offset_y as u32;
            
            if shadow_x < shadow_width && shadow_y < shadow_height {
                let shadow_alpha = (pixel[3] as f32 * shadow.opacity) as u8;
                let shadow_pixel = Rgba([
                    shadow.color.r,
                    shadow.color.g,
                    shadow.color.b,
                    shadow_alpha,
                ]);
                shadow_image.put_pixel(shadow_x, shadow_y, shadow_pixel);
            }
        }
        
        // Apply Gaussian blur to shadow if blur radius > 0
        if shadow.blur_radius > 0.0 {
            shadow_image = self.apply_gaussian_blur(shadow_image, shadow.blur_radius);
        }
        
        // Composite original watermark on top of shadow
        for (x, y, pixel) in watermark.enumerate_pixels() {
            let composite_x = x + blur_padding;
            let composite_y = y + blur_padding;
            
            if composite_x < shadow_width && composite_y < shadow_height {
                let shadow_pixel = *shadow_image.get_pixel(composite_x, composite_y);
                let blended = self.blend_pixels(shadow_pixel, *pixel, 1.0, BlendMode::Normal);
                shadow_image.put_pixel(composite_x, composite_y, blended);
            }
        }
        
        shadow_image
    }
    
    /// Apply outline effect to a watermark image
    fn apply_outline_effect(&self, watermark: &RgbaImage, outline: &WatermarkOutline) -> RgbaImage {
        let width = watermark.width();
        let height = watermark.height();
        
        // Create outline buffer with padding
        let outline_padding = (outline.width * 2.0) as u32;
        let outline_width = width + outline_padding * 2;
        let outline_height = height + outline_padding * 2;
        
        let mut outline_image = RgbaImage::new(outline_width, outline_height);
        
        // Generate outline by dilating the alpha channel
        for (x, y, pixel) in watermark.enumerate_pixels() {
            if pixel[3] > 0 {
                let outline_x = x + outline_padding;
                let outline_y = y + outline_padding;
                
                // Draw outline in a circle around the pixel
                let radius = outline.width as i32;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance <= outline.width {
                            let px = (outline_x as i32 + dx) as u32;
                            let py = (outline_y as i32 + dy) as u32;
                            
                            if px < outline_width && py < outline_height {
                                let outline_alpha = (outline.opacity * 255.0) as u8;
                                let outline_pixel = Rgba([
                                    outline.color.r,
                                    outline.color.g,
                                    outline.color.b,
                                    outline_alpha,
                                ]);
                                
                                let existing = *outline_image.get_pixel(px, py);
                                let blended = self.blend_pixels(existing, outline_pixel, 1.0, BlendMode::Normal);
                                outline_image.put_pixel(px, py, blended);
                            }
                        }
                    }
                }
            }
        }
        
        // Composite original watermark on top of outline
        for (x, y, pixel) in watermark.enumerate_pixels() {
            let composite_x = x + outline_padding;
            let composite_y = y + outline_padding;
            
            if composite_x < outline_width && composite_y < outline_height {
                let outline_pixel = *outline_image.get_pixel(composite_x, composite_y);
                let blended = self.blend_pixels(outline_pixel, *pixel, 1.0, BlendMode::Normal);
                outline_image.put_pixel(composite_x, composite_y, blended);
            }
        }
        
        outline_image
    }
    
    /// Apply glow effect to a watermark image
    fn apply_glow_effect(&self, watermark: &RgbaImage, glow: &WatermarkGlow) -> RgbaImage {
        let width = watermark.width();
        let height = watermark.height();
        
        // Create glow buffer with padding
        let glow_padding = (glow.radius * 2.0) as u32;
        let glow_width = width + glow_padding * 2;
        let glow_height = height + glow_padding * 2;
        
        let mut glow_image = RgbaImage::new(glow_width, glow_height);
        
        // Create glow by copying watermark alpha to glow color
        for (x, y, pixel) in watermark.enumerate_pixels() {
            let glow_x = x + glow_padding;
            let glow_y = y + glow_padding;
            
            if pixel[3] > 0 {
                let glow_alpha = (pixel[3] as f32 * glow.opacity * glow.intensity) as u8;
                let glow_pixel = Rgba([
                    glow.color.r,
                    glow.color.g,
                    glow.color.b,
                    glow_alpha,
                ]);
                glow_image.put_pixel(glow_x, glow_y, glow_pixel);
            }
        }
        
        // Apply Gaussian blur to create glow effect
        if glow.radius > 0.0 {
            glow_image = self.apply_gaussian_blur(glow_image, glow.radius);
        }
        
        // Composite original watermark on top of glow
        for (x, y, pixel) in watermark.enumerate_pixels() {
            let composite_x = x + glow_padding;
            let composite_y = y + glow_padding;
            
            if composite_x < glow_width && composite_y < glow_height {
                let glow_pixel = *glow_image.get_pixel(composite_x, composite_y);
                let blended = self.blend_pixels(glow_pixel, *pixel, 1.0, BlendMode::Normal);
                glow_image.put_pixel(composite_x, composite_y, blended);
            }
        }
        
        glow_image
    }
    
    /// Apply Gaussian blur to an image (simplified implementation)
    fn apply_gaussian_blur(&self, mut image: RgbaImage, radius: f32) -> RgbaImage {
        if radius <= 0.0 {
            return image;
        }
        
        let width = image.width();
        let height = image.height();
        let kernel_size = (radius * 2.0) as usize + 1;
        let sigma = radius / 3.0;
        
        // Generate Gaussian kernel
        let mut kernel = vec![0.0; kernel_size];
        let mut sum = 0.0;
        let center = kernel_size / 2;
        
        for i in 0..kernel_size {
            let x = (i as f32 - center as f32) / sigma;
            kernel[i] = (-0.5 * x * x).exp();
            sum += kernel[i];
        }
        
        // Normalize kernel
        for k in &mut kernel {
            *k /= sum;
        }
        
        // Apply horizontal blur
        let mut temp_image = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let mut r = 0.0;
                let mut g = 0.0;
                let mut b = 0.0;
                let mut a = 0.0;
                
                for (i, &weight) in kernel.iter().enumerate() {
                    let sample_x = (x as i32 + i as i32 - center as i32).max(0).min(width as i32 - 1) as u32;
                    let pixel = *image.get_pixel(sample_x, y);
                    
                    r += pixel[0] as f32 * weight;
                    g += pixel[1] as f32 * weight;
                    b += pixel[2] as f32 * weight;
                    a += pixel[3] as f32 * weight;
                }
                
                temp_image.put_pixel(x, y, Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]));
            }
        }
        
        // Apply vertical blur
        for y in 0..height {
            for x in 0..width {
                let mut r = 0.0;
                let mut g = 0.0;
                let mut b = 0.0;
                let mut a = 0.0;
                
                for (i, &weight) in kernel.iter().enumerate() {
                    let sample_y = (y as i32 + i as i32 - center as i32).max(0).min(height as i32 - 1) as u32;
                    let pixel = *temp_image.get_pixel(x, sample_y);
                    
                    r += pixel[0] as f32 * weight;
                    g += pixel[1] as f32 * weight;
                    b += pixel[2] as f32 * weight;
                    a += pixel[3] as f32 * weight;
                }
                
                image.put_pixel(x, y, Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]));
            }
        }
        
        image
    }
    
    /// Apply visual effects to a watermark image
    fn apply_visual_effects(&self, watermark: RgbaImage, effects: &WatermarkVisualEffects) -> RgbaImage {
        let mut result = watermark;
        
        // Apply shadow effect first (behind the watermark)
        if let Some(shadow) = &effects.shadow {
            result = self.apply_shadow_effect(&result, shadow);
        }
        
        // Apply glow effect (behind the watermark but in front of shadow)
        if let Some(glow) = &effects.glow {
            result = self.apply_glow_effect(&result, glow);
        }
        
        // Apply outline effect (around the watermark)
        if let Some(outline) = &effects.outline {
            result = self.apply_outline_effect(&result, outline);
        }
        
        result
    }

    /// Apply watermark to an image with enhanced positioning and scaling
    async fn apply_watermark_to_image(
        &self,
        base_image: DynamicImage,
        config: &WatermarkConfig,
    ) -> Result<DynamicImage> {
        // Load watermark
        let watermark = self.load_watermark(&config.watermark_path).await?;
        
        // Use enhanced scaling with aspect ratio preservation and constraints
        let scaled_watermark = self.scale_watermark_advanced(
            watermark,
            base_image.width(),
            base_image.height(),
            config.scale,
            &config.scaling_options,
        );
        
        // Convert images to RGBA for blending
        let mut base_rgba = base_image.to_rgba8();
        let mut watermark_rgba = scaled_watermark.to_rgba8();
        
        // Apply visual effects to the watermark
        watermark_rgba = self.apply_visual_effects(watermark_rgba, &config.visual_effects);
        
        debug!("Applying watermark: base={}x{}, watermark={}x{}, positions={}", 
               base_rgba.width(), base_rgba.height(),
               watermark_rgba.width(), watermark_rgba.height(),
               config.positions.len());
        
        // Apply watermark at each position (support multiple positions)
        for position in &config.positions {
            // Calculate position using enhanced positioning with alignment and offset
            let (x_pos, y_pos) = PositionCalculator::calculate_position(
                *position,
                base_rgba.width(),
                base_rgba.height(),
                watermark_rgba.width(),
                watermark_rgba.height(),
                config.alignment,
                config.offset,
            );
            
            debug!("Applying watermark at position ({}, {}) for {:?} with alignment {:?} and offset {:?}", 
                   x_pos, y_pos, position, config.alignment, config.offset);
            
            // Apply watermark pixel by pixel with bounds checking
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

#[cfg(test)]
mod tests;