//! Format conversion engine using libvips for high-performance image processing

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use crate::models::{ProcessingInput, ProcessingOutput, ProcessingOperation};
use crate::processing::{
    ImageProcessor, ProcessorCapabilities, ProcessorMetadata, ProcessorOperation as ProcOp,
    PerformanceProfile, MemoryUsage, CpuUsage,
};
use async_trait::async_trait;
use libvips::{VipsApp, VipsImage};
use std::path::Path;
use std::sync::Once;
use tracing::{debug, error, info, instrument};

static VIPS_INIT: Once = Once::new();

/// Format converter using libvips for high-performance image processing
#[derive(Debug)]
pub struct FormatConverter {
    /// Supported input formats
    supported_input_formats: Vec<ImageFormat>,
    /// Supported output formats
    supported_output_formats: Vec<ImageFormat>,
    /// Whether hardware acceleration is available
    hardware_acceleration: bool,
}

impl FormatConverter {
    /// Create a new format converter
    pub fn new() -> Result<Self> {
        Self::init_vips()?;
        
        let converter = Self {
            supported_input_formats: Self::detect_supported_input_formats(),
            supported_output_formats: Self::detect_supported_output_formats(),
            hardware_acceleration: Self::detect_hardware_acceleration(),
        };
        
        info!("Format converter initialized with {} input formats and {} output formats",
              converter.supported_input_formats.len(),
              converter.supported_output_formats.len());
        
        Ok(converter)
    }

    /// Detect supported input formats based on libvips capabilities
    fn detect_supported_input_formats() -> Vec<ImageFormat> {
        let mut formats = Vec::new();
        
        // Core formats that libvips always supports
        formats.extend_from_slice(&[
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::Bmp,
            ImageFormat::Tiff,
        ]);
        
        // Check for optional format support
        if Self::is_format_supported("webp") {
            formats.push(ImageFormat::WebP);
        }
        
        if Self::is_format_supported("gif") {
            formats.push(ImageFormat::Gif);
        }
        
        // AVIF and HEIC support (newer formats)
        if Self::is_format_supported("avif") {
            formats.push(ImageFormat::Avif);
        }
        
        if Self::is_format_supported("heic") {
            formats.push(ImageFormat::Heic);
        }
        
        debug!("Detected supported input formats: {:?}", formats);
        formats
    }

    /// Detect supported output formats based on libvips capabilities
    fn detect_supported_output_formats() -> Vec<ImageFormat> {
        let mut formats = Vec::new();
        
        // Core formats that libvips always supports
        formats.extend_from_slice(&[
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::Bmp,
            ImageFormat::Tiff,
        ]);
        
        // Check for optional format support
        if Self::is_format_supported("webp") {
            formats.push(ImageFormat::WebP);
        }
        
        if Self::is_format_supported("gif") {
            formats.push(ImageFormat::Gif);
        }
        
        // AVIF and HEIC support (newer formats)
        if Self::is_format_supported("avif") {
            formats.push(ImageFormat::Avif);
        }
        
        if Self::is_format_supported("heic") {
            formats.push(ImageFormat::Heic);
        }
        
        debug!("Detected supported output formats: {:?}", formats);
        formats
    }

    /// Check if a specific format is supported by the current libvips installation
    fn is_format_supported(format: &str) -> bool {
        // This is a simplified check - in a real implementation, you would
        // query libvips for available loaders and savers
        match format {
            "webp" => true,  // Assume WebP is available
            "gif" => true,   // Assume GIF is available
            "avif" => false, // AVIF might not be available in all builds
            "heic" => false, // HEIC might not be available in all builds
            _ => true,
        }
    }

    /// Initialize libvips (thread-safe, called once)
    fn init_vips() -> Result<()> {
        VIPS_INIT.call_once(|| {
            let app = VipsApp::new("image-processor", false)
                .expect("Failed to initialize libvips");
            app.init();
            debug!("libvips initialized successfully");
        });
        Ok(())
    }

    /// Detect if hardware acceleration is available
    fn detect_hardware_acceleration() -> bool {
        // Check if libvips was compiled with hardware acceleration support
        // This is a simplified check - in practice, you'd check for specific features
        true // Assume available for now
    }

    /// Load an image from file using libvips
    #[instrument(skip(self))]
    async fn load_image(&self, path: &Path) -> Result<VipsImage> {
        let path_str = path.to_string_lossy();
        debug!("Loading image from: {}", path_str);
        
        // Load image in a blocking task to avoid blocking the async runtime
        let path_owned = path.to_path_buf();
        let image = tokio::task::spawn_blocking(move || {
            VipsImage::new_from_file(&path_owned.to_string_lossy())
        }).await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to spawn image loading task: {}", e),
        })?
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to load image: {}", e),
        })?;
        
        info!("Image loaded successfully: {}x{} pixels", 
              image.get_width(), image.get_height());
        
        Ok(image)
    }

    /// Save an image to file using libvips with advanced compression options
    #[instrument(skip(self, image))]
    async fn save_image(&self, image: &VipsImage, path: &Path, format: ImageFormat, options: &CompressionOptions) -> Result<()> {
        let path_str = path.to_string_lossy();
        debug!("Saving image to: {} (format: {:?}, options: {:?})", path_str, format, options);
        
        let path_owned = path.to_path_buf();
        let image_clone = image.clone();
        let options_clone = options.clone();
        
        tokio::task::spawn_blocking(move || {
            match format {
                ImageFormat::Jpeg => {
                    let mut saver = image_clone.jpegsave(&path_owned.to_string_lossy());
                    
                    if let Some(quality) = options_clone.quality {
                        saver = saver.quality(quality as i32);
                    }
                    
                    // Advanced JPEG options
                    if options_clone.optimize {
                        saver = saver.optimize_coding(true);
                    }
                    
                    if options_clone.progressive {
                        saver = saver.interlace(true);
                    }
                    
                    saver.call()
                }
                ImageFormat::Png => {
                    let mut saver = image_clone.pngsave(&path_owned.to_string_lossy());
                    
                    // PNG compression level (0-9)
                    let compression = options_clone.compression_level.unwrap_or(6);
                    saver = saver.compression(compression as i32);
                    
                    if options_clone.progressive {
                        saver = saver.interlace(true);
                    }
                    
                    saver.call()
                }
                ImageFormat::WebP => {
                    let mut saver = image_clone.webpsave(&path_owned.to_string_lossy());
                    
                    if let Some(quality) = options_clone.quality {
                        saver = saver.q(quality as i32);
                    }
                    
                    // WebP-specific options
                    if options_clone.lossless {
                        saver = saver.lossless(true);
                    }
                    
                    if let Some(effort) = options_clone.effort {
                        saver = saver.effort(effort as i32);
                    }
                    
                    saver.call()
                }
                ImageFormat::Tiff => {
                    let mut saver = image_clone.tiffsave(&path_owned.to_string_lossy());
                    
                    // TIFF compression options
                    let compression = match options_clone.tiff_compression {
                        Some(TiffCompression::None) => libvips::VipsForeignTiffCompression::None,
                        Some(TiffCompression::Lzw) => libvips::VipsForeignTiffCompression::Lzw,
                        Some(TiffCompression::Deflate) => libvips::VipsForeignTiffCompression::Deflate,
                        Some(TiffCompression::Jpeg) => libvips::VipsForeignTiffCompression::Jpeg,
                        None => libvips::VipsForeignTiffCompression::Lzw, // Default
                    };
                    saver = saver.compression(compression);
                    
                    if let Some(quality) = options_clone.quality {
                        saver = saver.q(quality as i32);
                    }
                    
                    saver.call()
                }
                ImageFormat::Gif => {
                    // For GIF, we need to convert through a different path
                    // This is a simplified implementation
                    image_clone.magicksave(&path_owned.to_string_lossy())
                        .format("GIF")
                        .call()
                }
                ImageFormat::Bmp => {
                    image_clone.magicksave(&path_owned.to_string_lossy())
                        .format("BMP")
                        .call()
                }
                ImageFormat::Avif => {
                    let mut saver = image_clone.heifsave(&path_owned.to_string_lossy());
                    
                    if let Some(quality) = options_clone.quality {
                        saver = saver.q(quality as i32);
                    }
                    
                    if options_clone.lossless {
                        saver = saver.lossless(true);
                    }
                    
                    saver.call()
                }
                ImageFormat::Heic => {
                    let mut saver = image_clone.heifsave(&path_owned.to_string_lossy());
                    
                    if let Some(quality) = options_clone.quality {
                        saver = saver.q(quality as i32);
                    }
                    
                    if options_clone.lossless {
                        saver = saver.lossless(true);
                    }
                    
                    saver.call()
                }
            }
        }).await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to spawn image saving task: {}", e),
        })?
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to save image: {}", e),
        })?;
        
        info!("Image saved successfully to: {}", path_str);
        Ok(())
    }

    /// Convert image format with advanced compression options
    #[instrument(skip(self, input))]
    async fn convert_format(&self, input: &ProcessingInput, target_format: ImageFormat, quality: Option<u8>) -> Result<ProcessingOutput> {
        let start_time = std::time::Instant::now();
        
        // Validate input format is supported
        if !self.supported_input_formats.contains(&input.format) {
            return Err(ProcessingError::UnsupportedFormat {
                format: format!("{:?}", input.format),
            });
        }
        
        // Validate output format is supported
        if !self.supported_output_formats.contains(&target_format) {
            return Err(ProcessingError::UnsupportedFormat {
                format: format!("{:?}", target_format),
            });
        }
        
        // Validate quality if provided
        if let Some(q) = quality {
            Self::validate_quality(target_format, q)?;
        }
        
        // Create compression options based on quality priority
        let compression_options = match input.options.quality_priority {
            crate::models::QualityPriority::Speed => CompressionOptions::fast(target_format),
            crate::models::QualityPriority::Quality => CompressionOptions::high_quality(target_format),
            crate::models::QualityPriority::Balanced => {
                let mut options = CompressionOptions::for_format(target_format, quality);
                // Override with user-specified quality if provided
                if let Some(q) = quality {
                    options.quality = Some(q);
                }
                options
            }
        };
        
        // Load the source image
        let image = self.load_image(&input.source_path).await?;
        
        // Save in the target format with compression options
        self.save_image(&image, &input.output_path, target_format, &compression_options).await?;
        
        // Get output file size
        let output_size = tokio::fs::metadata(&input.output_path).await
            .map_err(|e| ProcessingError::Io { message: e.to_string() })?
            .len();
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path.clone(),
            file_size: output_size,
            format: target_format,
            processing_time,
            operations_applied: vec!["format_conversion".to_string()],
            metadata: None, // TODO: Preserve metadata
        })
    }

    /// Convert image format with custom compression options
    #[instrument(skip(self, input))]
    pub async fn convert_format_with_options(&self, input: &ProcessingInput, target_format: ImageFormat, options: CompressionOptions) -> Result<ProcessingOutput> {
        let start_time = std::time::Instant::now();
        
        // Validate input format is supported
        if !self.supported_input_formats.contains(&input.format) {
            return Err(ProcessingError::UnsupportedFormat {
                format: format!("{:?}", input.format),
            });
        }
        
        // Validate output format is supported
        if !self.supported_output_formats.contains(&target_format) {
            return Err(ProcessingError::UnsupportedFormat {
                format: format!("{:?}", target_format),
            });
        }
        
        // Validate quality if provided
        if let Some(quality) = options.quality {
            Self::validate_quality(target_format, quality)?;
        }
        
        // Load the source image
        let image = self.load_image(&input.source_path).await?;
        
        // Save in the target format with custom options
        self.save_image(&image, &input.output_path, target_format, &options).await?;
        
        // Get output file size
        let output_size = tokio::fs::metadata(&input.output_path).await
            .map_err(|e| ProcessingError::Io { message: e.to_string() })?
            .len();
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path.clone(),
            file_size: output_size,
            format: target_format,
            processing_time,
            operations_applied: vec!["format_conversion_with_options".to_string()],
            metadata: None, // TODO: Preserve metadata
        })
    }

    /// Get image information without loading the full image
    #[instrument(skip(self))]
    pub async fn get_image_info(&self, path: &Path) -> Result<ImageInfo> {
        let path_owned = path.to_path_buf();
        
        let info = tokio::task::spawn_blocking(move || {
            let image = VipsImage::new_from_file(&path_owned.to_string_lossy())?;
            Ok::<ImageInfo, libvips::error::Error>(ImageInfo {
                width: image.get_width(),
                height: image.get_height(),
                bands: image.get_bands(),
                format: Self::detect_format_from_vips(&image),
            })
        }).await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to spawn image info task: {}", e),
        })?
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Failed to get image info: {}", e),
        })?;
        
        Ok(info)
    }

    /// Detect image format from libvips image
    fn detect_format_from_vips(image: &VipsImage) -> Option<ImageFormat> {
        // This is a simplified implementation
        // In practice, you'd check the image's format field or filename
        match image.get_filename() {
            Some(filename) => {
                let filename = filename.to_lowercase();
                if filename.ends_with(".jpg") || filename.ends_with(".jpeg") {
                    Some(ImageFormat::Jpeg)
                } else if filename.ends_with(".png") {
                    Some(ImageFormat::Png)
                } else if filename.ends_with(".webp") {
                    Some(ImageFormat::WebP)
                } else if filename.ends_with(".gif") {
                    Some(ImageFormat::Gif)
                } else if filename.ends_with(".bmp") {
                    Some(ImageFormat::Bmp)
                } else if filename.ends_with(".tiff") || filename.ends_with(".tif") {
                    Some(ImageFormat::Tiff)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Check if a specific format conversion is supported
    pub fn supports_conversion(&self, from: ImageFormat, to: ImageFormat) -> bool {
        self.supported_input_formats.contains(&from) && 
        self.supported_output_formats.contains(&to)
    }

    /// Detect image format from file path
    pub fn detect_format_from_path(path: &Path) -> Option<ImageFormat> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        match extension.as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            "webp" => Some(ImageFormat::WebP),
            "gif" => Some(ImageFormat::Gif),
            "bmp" => Some(ImageFormat::Bmp),
            "tiff" | "tif" => Some(ImageFormat::Tiff),
            "avif" => Some(ImageFormat::Avif),
            "heic" | "heif" => Some(ImageFormat::Heic),
            _ => None,
        }
    }

    /// Validate that a file exists and is readable
    pub async fn validate_input_file(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(ProcessingError::FileNotFound {
                path: path.to_path_buf(),
            });
        }

        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| ProcessingError::Io { message: e.to_string() })?;

        if !metadata.is_file() {
            return Err(ProcessingError::InvalidInput {
                message: format!("Path is not a file: {}", path.display()),
            });
        }

        if metadata.len() == 0 {
            return Err(ProcessingError::InvalidInput {
                message: format!("File is empty: {}", path.display()),
            });
        }

        Ok(())
    }

    /// Get all supported input formats
    pub fn get_supported_input_formats(&self) -> &[ImageFormat] {
        &self.supported_input_formats
    }

    /// Get all supported output formats
    pub fn get_supported_output_formats(&self) -> &[ImageFormat] {
        &self.supported_output_formats
    }

    /// Check if format supports transparency
    pub fn format_supports_transparency(format: ImageFormat) -> bool {
        matches!(format, ImageFormat::Png | ImageFormat::WebP | ImageFormat::Gif)
    }

    /// Check if format supports animation
    pub fn format_supports_animation(format: ImageFormat) -> bool {
        matches!(format, ImageFormat::Gif | ImageFormat::WebP)
    }

    /// Get recommended quality range for a format
    pub fn get_quality_range(format: ImageFormat) -> Option<(u8, u8)> {
        match format {
            ImageFormat::Jpeg => Some((10, 100)),
            ImageFormat::WebP => Some((0, 100)),
            ImageFormat::Png => None, // PNG uses compression levels, not quality
            ImageFormat::Gif => None, // GIF doesn't use quality settings
            ImageFormat::Bmp => None, // BMP is uncompressed
            ImageFormat::Tiff => Some((1, 100)), // TIFF can have quality settings
            ImageFormat::Avif => Some((0, 100)),
            ImageFormat::Heic => Some((0, 100)),
        }
    }

    /// Get default quality for a format
    pub fn get_default_quality(format: ImageFormat) -> Option<u8> {
        match format {
            ImageFormat::Jpeg => Some(85),
            ImageFormat::WebP => Some(80),
            ImageFormat::Tiff => Some(85),
            ImageFormat::Avif => Some(75),
            ImageFormat::Heic => Some(75),
            _ => None,
        }
    }

    /// Validate quality setting for a format
    pub fn validate_quality(format: ImageFormat, quality: u8) -> Result<()> {
        if let Some((min, max)) = Self::get_quality_range(format) {
            if quality < min || quality > max {
                return Err(ProcessingError::InvalidInput {
                    message: format!(
                        "Quality {} is out of range for {:?} format (valid range: {}-{})",
                        quality, format, min, max
                    ),
                });
            }
        }
        Ok(())
    }
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self::new().expect("Failed to create default FormatConverter")
    }
}

#[async_trait]
impl ImageProcessor for FormatConverter {
    #[instrument(skip(self, input))]
    async fn process(&self, input: &ProcessingInput) -> Result<ProcessingOutput> {
        // Find the conversion operation
        for operation in &input.operations {
            if let ProcessingOperation::Convert { format, quality } = operation {
                return self.convert_format(input, *format, *quality).await;
            }
        }
        
        Err(ProcessingError::ProcessingFailed {
            message: "No format conversion operation found".to_string(),
        })
    }

    fn supports_format(&self, format: ImageFormat) -> bool {
        self.supported_input_formats.contains(&format)
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            input_formats: self.supported_input_formats.clone(),
            output_formats: self.supported_output_formats.clone(),
            operations: vec![ProcOp::FormatConversion],
            hardware_acceleration: self.hardware_acceleration,
            streaming_support: true, // libvips supports streaming
            max_file_size: None, // No inherent limit
            batch_optimized: true, // libvips is efficient for batch processing
        }
    }

    fn get_metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: "LibVips Format Converter".to_string(),
            version: "1.0.0".to_string(),
            description: "High-performance format converter using libvips".to_string(),
            author: "Image Processor Core".to_string(),
            performance_profile: PerformanceProfile {
                speed_factor: 2.0, // libvips is fast
                memory_usage: MemoryUsage::Streaming, // Efficient memory usage
                cpu_usage: CpuUsage::Moderate,
                parallel_friendly: true,
            },
        }
    }

    fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
        matches!(operation, ProcessingOperation::Convert { .. })
    }
}

/// Compression options for different image formats
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Quality setting (0-100) for lossy formats
    pub quality: Option<u8>,
    /// Compression level (0-9) for PNG
    pub compression_level: Option<u8>,
    /// Enable progressive/interlaced encoding
    pub progressive: bool,
    /// Enable optimization (for JPEG)
    pub optimize: bool,
    /// Enable lossless mode (for WebP, AVIF, HEIC)
    pub lossless: bool,
    /// Effort level for WebP (0-6)
    pub effort: Option<u8>,
    /// TIFF compression method
    pub tiff_compression: Option<TiffCompression>,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            quality: None,
            compression_level: None,
            progressive: false,
            optimize: true,
            lossless: false,
            effort: None,
            tiff_compression: None,
        }
    }
}

impl CompressionOptions {
    /// Create compression options for a specific format with quality
    pub fn for_format(format: ImageFormat, quality: Option<u8>) -> Self {
        let mut options = Self::default();
        
        // Set format-appropriate defaults
        match format {
            ImageFormat::Jpeg => {
                options.quality = quality.or(Some(85));
                options.optimize = true;
                options.progressive = false;
            }
            ImageFormat::Png => {
                options.compression_level = Some(6); // Good balance of speed/compression
                options.progressive = false;
            }
            ImageFormat::WebP => {
                options.quality = quality.or(Some(80));
                options.effort = Some(4); // Balanced effort
                options.lossless = false;
            }
            ImageFormat::Tiff => {
                options.quality = quality.or(Some(85));
                options.tiff_compression = Some(TiffCompression::Lzw);
            }
            ImageFormat::Avif => {
                options.quality = quality.or(Some(75));
                options.lossless = false;
            }
            ImageFormat::Heic => {
                options.quality = quality.or(Some(75));
                options.lossless = false;
            }
            _ => {
                options.quality = quality;
            }
        }
        
        options
    }

    /// Create high-quality compression options
    pub fn high_quality(format: ImageFormat) -> Self {
        let mut options = Self::for_format(format, Some(95));
        options.optimize = true;
        options.progressive = true;
        options
    }

    /// Create fast compression options (lower quality, faster processing)
    pub fn fast(format: ImageFormat) -> Self {
        let mut options = Self::for_format(format, Some(75));
        options.optimize = false;
        options.progressive = false;
        if format == ImageFormat::WebP {
            options.effort = Some(1); // Fastest effort
        }
        if format == ImageFormat::Png {
            options.compression_level = Some(1); // Fastest compression
        }
        options
    }

    /// Create lossless compression options (where supported)
    pub fn lossless(format: ImageFormat) -> Self {
        let mut options = Self::default();
        match format {
            ImageFormat::WebP | ImageFormat::Avif | ImageFormat::Heic => {
                options.lossless = true;
            }
            ImageFormat::Png => {
                options.compression_level = Some(9); // Maximum compression
            }
            ImageFormat::Tiff => {
                options.tiff_compression = Some(TiffCompression::Lzw);
            }
            _ => {
                // For formats that don't support lossless, use high quality
                options.quality = Some(100);
                options.optimize = true;
            }
        }
        options
    }
}

/// TIFF compression methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    /// No compression
    None,
    /// LZW compression (lossless)
    Lzw,
    /// Deflate compression (lossless)
    Deflate,
    /// JPEG compression (lossy)
    Jpeg,
}

/// Image information structure
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: i32,
    pub height: i32,
    pub bands: i32,
    pub format: Option<ImageFormat>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ProcessingOptions;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn create_test_input(source: PathBuf, output: PathBuf) -> ProcessingInput {
        ProcessingInput {
            job_id: Uuid::new_v4(),
            source_path: source,
            output_path: output,
            operations: vec![ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            }],
            options: ProcessingOptions::default(),
            file_size: 1024,
            format: ImageFormat::Jpeg,
        }
    }

    #[tokio::test]
    async fn test_format_converter_creation() {
        let result = FormatConverter::new();
        // This might fail if libvips is not installed, which is expected in CI
        match result {
            Ok(converter) => {
                assert!(!converter.supported_input_formats.is_empty());
                assert!(!converter.supported_output_formats.is_empty());
            }
            Err(_) => {
                // Expected if libvips is not available
                println!("libvips not available, skipping test");
            }
        }
    }

    #[test]
    fn test_format_converter_capabilities() {
        if let Ok(converter) = FormatConverter::new() {
            let caps = converter.get_capabilities();
            assert!(caps.input_formats.contains(&ImageFormat::Jpeg));
            assert!(caps.output_formats.contains(&ImageFormat::Png));
            assert!(caps.operations.contains(&ProcOp::FormatConversion));
            assert!(caps.streaming_support);
            assert!(caps.batch_optimized);
        }
    }

    #[test]
    fn test_format_converter_metadata() {
        if let Ok(converter) = FormatConverter::new() {
            let metadata = converter.get_metadata();
            assert_eq!(metadata.name, "LibVips Format Converter");
            assert!(!metadata.version.is_empty());
            assert!(!metadata.description.is_empty());
            assert_eq!(metadata.performance_profile.memory_usage, MemoryUsage::Streaming);
        }
    }

    #[test]
    fn test_supports_format() {
        if let Ok(converter) = FormatConverter::new() {
            assert!(converter.supports_format(ImageFormat::Jpeg));
            assert!(converter.supports_format(ImageFormat::Png));
            assert!(converter.supports_format(ImageFormat::WebP));
        }
    }

    #[test]
    fn test_supports_conversion() {
        if let Ok(converter) = FormatConverter::new() {
            assert!(converter.supports_conversion(ImageFormat::Jpeg, ImageFormat::Png));
            assert!(converter.supports_conversion(ImageFormat::Png, ImageFormat::WebP));
            assert!(converter.supports_conversion(ImageFormat::WebP, ImageFormat::Jpeg));
        }
    }

    #[test]
    fn test_can_handle_operation() {
        if let Ok(converter) = FormatConverter::new() {
            let convert_op = ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            };
            assert!(converter.can_handle_operation(&convert_op));
            
            let resize_op = ProcessingOperation::Resize {
                width: Some(800),
                height: Some(600),
                algorithm: crate::models::ResizeAlgorithm::Lanczos,
            };
            assert!(!converter.can_handle_operation(&resize_op));
        }
    }

    #[tokio::test]
    async fn test_process_without_convert_operation() {
        if let Ok(converter) = FormatConverter::new() {
            let temp_dir = tempdir().unwrap();
            let source = temp_dir.path().join("input.jpg");
            let output = temp_dir.path().join("output.png");
            
            let mut input = create_test_input(source, output);
            input.operations = vec![]; // No operations
            
            let result = converter.process(&input).await;
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_format_detection_from_path() {
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.jpg")), Some(ImageFormat::Jpeg));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.jpeg")), Some(ImageFormat::Jpeg));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.png")), Some(ImageFormat::Png));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.webp")), Some(ImageFormat::WebP));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.gif")), Some(ImageFormat::Gif));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.bmp")), Some(ImageFormat::Bmp));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.tiff")), Some(ImageFormat::Tiff));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.tif")), Some(ImageFormat::Tiff));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.avif")), Some(ImageFormat::Avif));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.heic")), Some(ImageFormat::Heic));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.heif")), Some(ImageFormat::Heic));
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test.unknown")), None);
        assert_eq!(FormatConverter::detect_format_from_path(&PathBuf::from("test")), None);
    }

    #[test]
    fn test_format_transparency_support() {
        assert!(FormatConverter::format_supports_transparency(ImageFormat::Png));
        assert!(FormatConverter::format_supports_transparency(ImageFormat::WebP));
        assert!(FormatConverter::format_supports_transparency(ImageFormat::Gif));
        assert!(!FormatConverter::format_supports_transparency(ImageFormat::Jpeg));
        assert!(!FormatConverter::format_supports_transparency(ImageFormat::Bmp));
        assert!(!FormatConverter::format_supports_transparency(ImageFormat::Tiff));
    }

    #[test]
    fn test_format_animation_support() {
        assert!(FormatConverter::format_supports_animation(ImageFormat::Gif));
        assert!(FormatConverter::format_supports_animation(ImageFormat::WebP));
        assert!(!FormatConverter::format_supports_animation(ImageFormat::Jpeg));
        assert!(!FormatConverter::format_supports_animation(ImageFormat::Png));
        assert!(!FormatConverter::format_supports_animation(ImageFormat::Bmp));
        assert!(!FormatConverter::format_supports_animation(ImageFormat::Tiff));
    }

    #[test]
    fn test_quality_ranges() {
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Jpeg), Some((10, 100)));
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::WebP), Some((0, 100)));
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Png), None);
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Gif), None);
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Bmp), None);
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Tiff), Some((1, 100)));
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Avif), Some((0, 100)));
        assert_eq!(FormatConverter::get_quality_range(ImageFormat::Heic), Some((0, 100)));
    }

    #[test]
    fn test_default_quality() {
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Jpeg), Some(85));
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::WebP), Some(80));
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Png), None);
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Gif), None);
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Bmp), None);
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Tiff), Some(85));
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Avif), Some(75));
        assert_eq!(FormatConverter::get_default_quality(ImageFormat::Heic), Some(75));
    }

    #[test]
    fn test_quality_validation() {
        // Valid quality values
        assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 85).is_ok());
        assert!(FormatConverter::validate_quality(ImageFormat::WebP, 80).is_ok());
        assert!(FormatConverter::validate_quality(ImageFormat::Png, 50).is_ok()); // PNG doesn't use quality
        
        // Invalid quality values
        assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 5).is_err()); // Below minimum
        assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 105).is_err()); // Above maximum
        assert!(FormatConverter::validate_quality(ImageFormat::WebP, 101).is_err()); // Above maximum
    }

    #[tokio::test]
    async fn test_validate_input_file_nonexistent() {
        if let Ok(converter) = FormatConverter::new() {
            let nonexistent_path = PathBuf::from("nonexistent_file.jpg");
            let result = converter.validate_input_file(&nonexistent_path).await;
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ProcessingError::FileNotFound { .. }));
        }
    }

    #[tokio::test]
    async fn test_validate_input_file_empty() {
        if let Ok(converter) = FormatConverter::new() {
            let temp_dir = tempdir().unwrap();
            let empty_file = temp_dir.path().join("empty.jpg");
            
            // Create empty file
            tokio::fs::write(&empty_file, b"").await.unwrap();
            
            let result = converter.validate_input_file(&empty_file).await;
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ProcessingError::InvalidInput { .. }));
        }
    }

    #[tokio::test]
    async fn test_validate_input_file_directory() {
        if let Ok(converter) = FormatConverter::new() {
            let temp_dir = tempdir().unwrap();
            
            let result = converter.validate_input_file(temp_dir.path()).await;
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ProcessingError::InvalidInput { .. }));
        }
    }

    #[test]
    fn test_get_supported_formats() {
        if let Ok(converter) = FormatConverter::new() {
            let input_formats = converter.get_supported_input_formats();
            let output_formats = converter.get_supported_output_formats();
            
            assert!(!input_formats.is_empty());
            assert!(!output_formats.is_empty());
            
            // Core formats should always be supported
            assert!(input_formats.contains(&ImageFormat::Jpeg));
            assert!(input_formats.contains(&ImageFormat::Png));
            assert!(output_formats.contains(&ImageFormat::Jpeg));
            assert!(output_formats.contains(&ImageFormat::Png));
        }
    }

    #[test]
    fn test_format_support_detection() {
        // Test the format support detection logic
        assert!(FormatConverter::is_format_supported("webp"));
        assert!(FormatConverter::is_format_supported("gif"));
        assert!(!FormatConverter::is_format_supported("avif")); // Assumed not available
        assert!(!FormatConverter::is_format_supported("heic")); // Assumed not available
        assert!(FormatConverter::is_format_supported("unknown")); // Default to true
    }

    #[test]
    fn test_supported_formats_detection() {
        let input_formats = FormatConverter::detect_supported_input_formats();
        let output_formats = FormatConverter::detect_supported_output_formats();
        
        // Core formats should always be present
        assert!(input_formats.contains(&ImageFormat::Jpeg));
        assert!(input_formats.contains(&ImageFormat::Png));
        assert!(input_formats.contains(&ImageFormat::Bmp));
        assert!(input_formats.contains(&ImageFormat::Tiff));
        
        assert!(output_formats.contains(&ImageFormat::Jpeg));
        assert!(output_formats.contains(&ImageFormat::Png));
        assert!(output_formats.contains(&ImageFormat::Bmp));
        assert!(output_formats.contains(&ImageFormat::Tiff));
        
        // Optional formats should be present based on support detection
        assert!(input_formats.contains(&ImageFormat::WebP)); // Should be supported
        assert!(input_formats.contains(&ImageFormat::Gif)); // Should be supported
        assert!(!input_formats.contains(&ImageFormat::Avif)); // Should not be supported
        assert!(!input_formats.contains(&ImageFormat::Heic)); // Should not be supported
    }

    // Note: Actual file processing tests would require test images and libvips installation
    // These would be integration tests rather than unit tests
}