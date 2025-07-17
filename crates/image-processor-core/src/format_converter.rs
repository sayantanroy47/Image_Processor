//! Format conversion engine with libvips integration for high-performance streaming image processing

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use crate::models::{ProcessingInput, ProcessingOutput, ProcessingOperation};
use crate::processing::{
    ImageProcessor, ProcessorCapabilities, ProcessorMetadata, ProcessorOperation as ProcOp,
    PerformanceProfile, MemoryUsage, CpuUsage, ProcessorType,
};
use async_trait::async_trait;
use image::{DynamicImage, ImageFormat as ImageCrateFormat, io::Reader as ImageReader};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tokio::task;
use tracing::{debug, info, instrument, warn, error};

#[cfg(feature = "libvips")]
use libvips::{VipsApp, VipsImage};

/// Processing backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingBackend {
    /// Use the image crate (default fallback)
    ImageCrate,
    /// Use libvips for streaming processing
    #[cfg(feature = "libvips")]
    LibVips,
}

/// Format converter using the image crate for high-performance image processing
#[derive(Debug)]
pub struct FormatConverter {
    /// Supported input formats
    supported_input_formats: Vec<ImageFormat>,
    /// Supported output formats
    supported_output_formats: Vec<ImageFormat>,
    /// Whether hardware acceleration is available
    hardware_acceleration: bool,
    /// Processing backend to use
    backend: ProcessingBackend,
    /// libvips app instance (if using libvips)
    #[cfg(feature = "libvips")]
    vips_app: Option<VipsApp>,
}

impl FormatConverter {
    /// Create a new format converter
    pub fn new() -> Result<Self> {
        Self::with_backend(Self::detect_best_backend())
    }

    /// Create a new format converter with a specific backend
    pub fn with_backend(backend: ProcessingBackend) -> Result<Self> {
        #[cfg(feature = "libvips")]
        let vips_app = if matches!(backend, ProcessingBackend::LibVips) {
            match VipsApp::new("image-processor", false) {
                Ok(app) => {
                    info!("libvips initialized successfully");
                    Some(app)
                }
                Err(e) => {
                    warn!("Failed to initialize libvips: {}, falling back to image crate", e);
                    None
                }
            }
        } else {
            None
        };

        let actual_backend = match backend {
            #[cfg(feature = "libvips")]
            ProcessingBackend::LibVips if vips_app.is_some() => ProcessingBackend::LibVips,
            _ => ProcessingBackend::ImageCrate,
        };

        let converter = Self {
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
            hardware_acceleration: true,
            backend: actual_backend,
            #[cfg(feature = "libvips")]
            vips_app,
        };
        
        info!("Format converter initialized with {:?} backend, {} input formats and {} output formats",
              converter.backend,
              converter.supported_input_formats.len(),
              converter.supported_output_formats.len());
        
        Ok(converter)
    }

    /// Detect the best available backend
    fn detect_best_backend() -> ProcessingBackend {
        #[cfg(feature = "libvips")]
        {
            // Try to initialize libvips to see if it's available
            match VipsApp::new("image-processor-test", false) {
                Ok(_) => {
                    debug!("libvips is available, using LibVips backend");
                    ProcessingBackend::LibVips
                }
                Err(e) => {
                    debug!("libvips not available ({}), using ImageCrate backend", e);
                    ProcessingBackend::ImageCrate
                }
            }
        }
        #[cfg(not(feature = "libvips"))]
        {
            debug!("libvips feature not enabled, using ImageCrate backend");
            ProcessingBackend::ImageCrate
        }
    }

    /// Get all supported input formats
    pub fn get_supported_input_formats(&self) -> &[ImageFormat] {
        &self.supported_input_formats
    }

    /// Get all supported output formats
    pub fn get_supported_output_formats(&self) -> &[ImageFormat] {
        &self.supported_output_formats
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

    /// Convert our ImageFormat enum to the image crate's ImageFormat
    fn to_image_crate_format(format: ImageFormat) -> Option<ImageCrateFormat> {
        match format {
            ImageFormat::Jpeg => Some(ImageCrateFormat::Jpeg),
            ImageFormat::Png => Some(ImageCrateFormat::Png),
            ImageFormat::WebP => Some(ImageCrateFormat::WebP),
            ImageFormat::Gif => Some(ImageCrateFormat::Gif),
            ImageFormat::Bmp => Some(ImageCrateFormat::Bmp),
            ImageFormat::Tiff => Some(ImageCrateFormat::Tiff),
            ImageFormat::Avif => Some(ImageCrateFormat::Avif),
            ImageFormat::Heic => None, // Not supported by image crate
        }
    }

    /// Load an image from file
    async fn load_image(path: &Path) -> Result<DynamicImage> {
        let path = path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Loading image from: {}", path.display());
            
            let reader = ImageReader::open(&path)
                .map_err(|e| ProcessingError::Io {
                    message: format!("Failed to open image file {}: {}", path.display(), e),
                })?;
            
            let image = reader.decode()
                .map_err(|e| ProcessingError::ProcessingFailed {
                    message: format!("Failed to decode image {}: {}", path.display(), e),
                })?;
            
            debug!("Successfully loaded image: {}x{}", image.width(), image.height());
            Ok(image)
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }

    /// Save an image to file with specified format and quality
    async fn save_image(
        image: DynamicImage,
        path: &Path,
        format: ImageFormat,
        quality: Option<u8>,
    ) -> Result<u64> {
        let path = path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Saving image to: {} (format: {:?})", path.display(), format);
            
            // Ensure output directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| ProcessingError::Io {
                        message: format!("Failed to create output directory: {}", e),
                    })?;
            }
            
            let output_format = Self::to_image_crate_format(format)
                .ok_or_else(|| ProcessingError::UnsupportedFormat {
                    format: format!("{:?}", format),
                })?;
            
            let file = File::create(&path)
                .map_err(|e| ProcessingError::Io {
                    message: format!("Failed to create output file {}: {}", path.display(), e),
                })?;
            
            let mut writer = BufWriter::new(file);
            
            // Handle format-specific encoding options
            match format {
                ImageFormat::Jpeg => {
                    let quality = quality.unwrap_or(85);
                    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality);
                    image.write_with_encoder(encoder)
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to encode JPEG: {}", e),
                        })?;
                    // Flush the buffer to ensure data is written
                    use std::io::Write;
                    writer.flush().map_err(|e| ProcessingError::Io {
                        message: format!("Failed to flush JPEG data: {}", e),
                    })?;
                }
                ImageFormat::WebP => {
                    let quality = quality.unwrap_or(80) as f32;
                    if quality < 100.0 {
                        // For lossy WebP, we need to use a different approach
                        // The image crate doesn't support lossy WebP encoding directly
                        // So we'll save as lossless for now
                        let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut writer);
                        image.write_with_encoder(encoder)
                            .map_err(|e| ProcessingError::ProcessingFailed {
                                message: format!("Failed to encode WebP: {}", e),
                            })?;
                    } else {
                        let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut writer);
                        image.write_with_encoder(encoder)
                            .map_err(|e| ProcessingError::ProcessingFailed {
                                message: format!("Failed to encode WebP: {}", e),
                            })?;
                    }
                    // Flush the buffer to ensure data is written
                    use std::io::Write;
                    writer.flush().map_err(|e| ProcessingError::Io {
                        message: format!("Failed to flush WebP data: {}", e),
                    })?;
                }
                ImageFormat::Png => {
                    let encoder = image::codecs::png::PngEncoder::new(&mut writer);
                    image.write_with_encoder(encoder)
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to encode PNG: {}", e),
                        })?;
                    // Flush the buffer to ensure data is written
                    use std::io::Write;
                    writer.flush().map_err(|e| ProcessingError::Io {
                        message: format!("Failed to flush PNG data: {}", e),
                    })?;
                }
                _ => {
                    // For other formats, use the generic save method
                    image.save_with_format(&path, output_format)
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to save image as {:?}: {}", format, e),
                        })?;
                }
            }
            
            // Ensure the writer is dropped to flush any remaining data
            drop(writer);
            
            // Get file size
            let metadata = std::fs::metadata(&path)
                .map_err(|e| ProcessingError::Io {
                    message: format!("Failed to get file metadata: {}", e),
                })?;
            
            let file_size = metadata.len();
            debug!("Successfully saved image: {} bytes", file_size);
            
            Ok(file_size)
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }

    /// Convert image format using libvips for streaming processing
    #[cfg(feature = "libvips")]
    async fn convert_with_libvips(
        &self,
        input_path: &Path,
        output_path: &Path,
        target_format: ImageFormat,
        quality: Option<u8>,
    ) -> Result<u64> {
        let input_path = input_path.to_path_buf();
        let output_path = output_path.to_path_buf();
        
        task::spawn_blocking(move || {
            debug!("Converting with libvips: {} -> {} ({:?})", 
                   input_path.display(), output_path.display(), target_format);
            
            // Ensure output directory exists
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| ProcessingError::Io {
                        message: format!("Failed to create output directory: {}", e),
                    })?;
            }
            
            // Load image with libvips (streaming)
            let image = VipsImage::new_from_file(&input_path.to_string_lossy())
                .map_err(|e| ProcessingError::ProcessingFailed {
                    message: format!("Failed to load image with libvips: {}", e),
                })?;
            
            debug!("Loaded image: {}x{} pixels", image.get_width(), image.get_height());
            
            // Convert format based on target
            let result = match target_format {
                ImageFormat::Jpeg => {
                    let quality = quality.unwrap_or(85) as i32;
                    image.jpegsave(&output_path.to_string_lossy())
                        .set("Q", quality)
                        .set("optimize_coding", true)
                        .set("interlace", true)
                        .call()
                }
                ImageFormat::Png => {
                    let compression = if let Some(q) = quality {
                        // Convert quality (0-100) to PNG compression (0-9)
                        9 - (q as i32 * 9 / 100)
                    } else {
                        6 // Default compression
                    };
                    image.pngsave(&output_path.to_string_lossy())
                        .set("compression", compression)
                        .set("interlace", true)
                        .call()
                }
                ImageFormat::WebP => {
                    let quality = quality.unwrap_or(80) as i32;
                    image.webpsave(&output_path.to_string_lossy())
                        .set("Q", quality)
                        .set("lossless", quality >= 100)
                        .call()
                }
                ImageFormat::Tiff => {
                    let quality = quality.unwrap_or(85) as i32;
                    image.tiffsave(&output_path.to_string_lossy())
                        .set("Q", quality)
                        .set("compression", "jpeg")
                        .call()
                }
                ImageFormat::Gif => {
                    // GIF doesn't support quality, use default settings
                    image.gifsave(&output_path.to_string_lossy())
                        .call()
                }
                ImageFormat::Bmp => {
                    // Convert to a format libvips can handle, then save as BMP
                    // libvips doesn't have direct BMP save, so we'll use a workaround
                    let temp_path = output_path.with_extension("temp.png");
                    image.pngsave(&temp_path.to_string_lossy())
                        .call()
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to save temporary PNG: {}", e),
                        })?;
                    
                    // Load with image crate and save as BMP
                    let img = image::open(&temp_path)
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to load temporary image: {}", e),
                        })?;
                    
                    img.save_with_format(&output_path, ImageCrateFormat::Bmp)
                        .map_err(|e| ProcessingError::ProcessingFailed {
                            message: format!("Failed to save BMP: {}", e),
                        })?;
                    
                    // Clean up temporary file
                    let _ = std::fs::remove_file(&temp_path);
                    
                    Ok(())
                }
                _ => {
                    return Err(ProcessingError::UnsupportedFormat {
                        format: format!("{:?}", target_format),
                    });
                }
            };
            
            result.map_err(|e| ProcessingError::ProcessingFailed {
                message: format!("Failed to convert image with libvips: {}", e),
            })?;
            
            // Get output file size
            let metadata = std::fs::metadata(&output_path)
                .map_err(|e| ProcessingError::Io {
                    message: format!("Failed to get output file metadata: {}", e),
                })?;
            
            let file_size = metadata.len();
            debug!("Successfully converted image with libvips: {} bytes", file_size);
            
            Ok(file_size)
        })
        .await
        .map_err(|e| ProcessingError::ProcessingFailed {
            message: format!("Task join error: {}", e),
        })?
    }

    /// Process conversion using the appropriate backend
    async fn process_conversion(
        &self,
        input_path: &Path,
        output_path: &Path,
        target_format: ImageFormat,
        quality: Option<u8>,
    ) -> Result<u64> {
        match self.backend {
            #[cfg(feature = "libvips")]
            ProcessingBackend::LibVips => {
                debug!("Using libvips backend for conversion");
                self.convert_with_libvips(input_path, output_path, target_format, quality).await
            }
            ProcessingBackend::ImageCrate => {
                debug!("Using image crate backend for conversion");
                let image = Self::load_image(input_path).await?;
                Self::save_image(image, output_path, target_format, quality).await
            }
        }
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
        let start_time = std::time::Instant::now();
        
        // Find the conversion operation
        let conversion_op = input.operations.iter()
            .find_map(|op| match op {
                ProcessingOperation::Convert { format, quality } => Some((*format, *quality)),
                _ => None,
            })
            .ok_or_else(|| ProcessingError::ProcessingFailed {
                message: "No format conversion operation found".to_string(),
            })?;
        
        let (target_format, quality) = conversion_op;
        
        // Validate input file exists
        if !input.source_path.exists() {
            return Err(ProcessingError::FileNotFound {
                path: input.source_path.clone(),
            });
        }
        
        // Detect input format from file extension if not provided
        let detected_format = Self::detect_format_from_path(&input.source_path);
        if detected_format.is_none() {
            warn!("Could not detect format from file extension: {}", input.source_path.display());
        }
        
        // Validate that we support the target format
        if !self.supported_output_formats.contains(&target_format) {
            return Err(ProcessingError::UnsupportedFormat {
                format: format!("{:?}", target_format),
            });
        }
        
        // Validate quality parameter if provided
        if let Some(q) = quality {
            Self::validate_quality(target_format, q)?;
        }
        
        info!("Converting {} to {:?} format using {:?} backend", 
              input.source_path.display(), target_format, self.backend);
        
        // Process the conversion using the appropriate backend
        let output_file_size = self.process_conversion(
            &input.source_path,
            &input.output_path,
            target_format,
            quality,
        ).await?;
        
        let processing_time = start_time.elapsed();
        
        info!("Successfully converted image in {}ms, output size: {} bytes", 
              processing_time.as_millis(), output_file_size);
        
        Ok(ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path.clone(),
            file_size: output_file_size,
            format: target_format,
            processing_time,
            operations_applied: vec![format!("format_conversion_to_{:?}", target_format)],
            metadata: None, // TODO: Extract and preserve metadata if requested
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
        let (name, description, speed_factor) = match self.backend {
            #[cfg(feature = "libvips")]
            ProcessingBackend::LibVips => (
                "LibVips Format Converter".to_string(),
                "High-performance streaming format converter using libvips".to_string(),
                3.0, // libvips is faster for large files
            ),
            ProcessingBackend::ImageCrate => (
                "Image Crate Format Converter".to_string(),
                "High-performance format converter using the image crate".to_string(),
                2.0, // image crate is fast
            ),
        };

        ProcessorMetadata {
            name,
            version: "1.0.0".to_string(),
            description,
            author: "Image Processor Core".to_string(),
            performance_profile: PerformanceProfile {
                speed_factor,
                memory_usage: MemoryUsage::Streaming, // Both backends support efficient memory usage
                cpu_usage: CpuUsage::Moderate,
                parallel_friendly: true,
            },
        }
    }

    fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
        matches!(operation, ProcessingOperation::Convert { .. })
    }
}