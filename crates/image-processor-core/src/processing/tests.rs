//! Tests for the processing orchestrator and format converter registration

use crate::config::ImageFormat;
use crate::error::Result;
use crate::format_converter::FormatConverter;
use crate::models::{JobId, ProcessingInput, ProcessingOperation, ProcessingOptions, QualityPriority};
use crate::processing::{ImageProcessor, ProcessingOrchestrator, ProcessorType, ProcessorSelectionStrategy};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn test_format_converter_registration() -> Result<()> {
    // Create a new orchestrator
    let mut orchestrator = ProcessingOrchestrator::new();
    
    // Verify no processors are registered initially
    assert_eq!(orchestrator.get_registered_processors().len(), 0);
    
    // Register the FormatConverter
    orchestrator.register_format_converter()?;
    
    // Verify the processor was registered
    let registered_processors = orchestrator.get_registered_processors();
    assert_eq!(registered_processors.len(), 1);
    assert!(registered_processors.contains(&ProcessorType::FormatConverter));
    
    // Verify capabilities
    let capabilities = orchestrator.get_processor_capabilities(&ProcessorType::FormatConverter);
    assert!(capabilities.is_some());
    
    let caps = capabilities.unwrap();
    assert!(!caps.input_formats.is_empty());
    assert!(!caps.output_formats.is_empty());
    assert!(caps.operations.contains(&crate::processing::ProcessorOperation::FormatConversion));
    assert!(caps.streaming_support);
    assert!(caps.batch_optimized);
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_capabilities() -> Result<()> {
    let converter = FormatConverter::new()?;
    let capabilities = converter.get_capabilities();
    
    // Verify basic capabilities
    assert!(!capabilities.input_formats.is_empty());
    assert!(!capabilities.output_formats.is_empty());
    assert!(capabilities.operations.contains(&crate::processing::ProcessorOperation::FormatConversion));
    
    // Verify common formats are supported
    assert!(capabilities.input_formats.contains(&ImageFormat::Jpeg));
    assert!(capabilities.input_formats.contains(&ImageFormat::Png));
    assert!(capabilities.output_formats.contains(&ImageFormat::Jpeg));
    assert!(capabilities.output_formats.contains(&ImageFormat::Png));
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_metadata() -> Result<()> {
    let converter = FormatConverter::new()?;
    let metadata = converter.get_metadata();
    
    // Verify metadata is properly set
    assert_eq!(metadata.name, "Image Crate Format Converter");
    assert_eq!(metadata.version, "1.0.0");
    assert!(!metadata.description.is_empty());
    assert!(!metadata.author.is_empty());
    
    // Verify performance profile
    assert!(metadata.performance_profile.speed_factor > 0.0);
    assert!(metadata.performance_profile.parallel_friendly);
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_format_support() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    // Test format support
    assert!(orchestrator.supports_input_format(ImageFormat::Jpeg));
    assert!(orchestrator.supports_input_format(ImageFormat::Png));
    assert!(orchestrator.supports_input_format(ImageFormat::WebP));
    
    // Test operation support
    let convert_operation = ProcessingOperation::Convert {
        format: ImageFormat::Png,
        quality: Some(85),
    };
    assert!(orchestrator.supports_operation(&convert_operation));
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_processor_selection() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let convert_operation = ProcessingOperation::Convert {
        format: ImageFormat::Png,
        quality: Some(85),
    };
    
    // Test processor selection
    let processor = orchestrator.find_processor_for_operation(&convert_operation);
    assert!(processor.is_some());
    
    // Verify it's the format converter
    let processor = processor.unwrap();
    assert!(processor.can_handle_operation(&convert_operation));
    assert_eq!(processor.get_metadata().name, "Image Crate Format Converter");
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_format_conversion_matrix() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let matrix = orchestrator.get_format_conversion_matrix();
    
    // Verify JPEG can be converted to other formats
    let jpeg_conversions = matrix.get(&ImageFormat::Jpeg);
    assert!(jpeg_conversions.is_some());
    let conversions = jpeg_conversions.unwrap();
    assert!(conversions.contains(&ImageFormat::Png));
    assert!(conversions.contains(&ImageFormat::WebP));
    
    // Verify PNG can be converted to other formats
    let png_conversions = matrix.get(&ImageFormat::Png);
    assert!(png_conversions.is_some());
    let conversions = png_conversions.unwrap();
    assert!(conversions.contains(&ImageFormat::Jpeg));
    assert!(conversions.contains(&ImageFormat::WebP));
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_selection_strategies() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let convert_operation = ProcessingOperation::Convert {
        format: ImageFormat::Png,
        quality: Some(85),
    };
    
    // Test different selection strategies
    orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Fastest);
    let processor = orchestrator.find_processor_for_operation(&convert_operation);
    assert!(processor.is_some());
    
    orchestrator.set_selection_strategy(ProcessorSelectionStrategy::MemoryEfficient);
    let processor = orchestrator.find_processor_for_operation(&convert_operation);
    assert!(processor.is_some());
    
    orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Balanced);
    let processor = orchestrator.find_processor_for_operation(&convert_operation);
    assert!(processor.is_some());
    
    orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Specific("Image Crate Format Converter".to_string()));
    let processor = orchestrator.find_processor_for_operation(&convert_operation);
    assert!(processor.is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_operation_handling() -> Result<()> {
    let converter = FormatConverter::new()?;
    
    // Test operation handling
    let convert_operation = ProcessingOperation::Convert {
        format: ImageFormat::Png,
        quality: Some(85),
    };
    assert!(converter.can_handle_operation(&convert_operation));
    
    // Test unsupported operation
    let resize_operation = ProcessingOperation::Resize {
        width: Some(800),
        height: Some(600),
        algorithm: crate::models::ResizeAlgorithm::Lanczos,
    };
    assert!(!converter.can_handle_operation(&resize_operation));
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_format_support() -> Result<()> {
    let converter = FormatConverter::new()?;
    
    // Test format support
    assert!(converter.supports_format(ImageFormat::Jpeg));
    assert!(converter.supports_format(ImageFormat::Png));
    assert!(converter.supports_format(ImageFormat::WebP));
    assert!(converter.supports_format(ImageFormat::Gif));
    assert!(converter.supports_format(ImageFormat::Bmp));
    assert!(converter.supports_format(ImageFormat::Tiff));
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_conversion_support() -> Result<()> {
    let converter = FormatConverter::new()?;
    
    // Test format conversion support
    assert!(converter.supports_conversion(ImageFormat::Jpeg, ImageFormat::Png));
    assert!(converter.supports_conversion(ImageFormat::Png, ImageFormat::Jpeg));
    assert!(converter.supports_conversion(ImageFormat::WebP, ImageFormat::Png));
    assert!(converter.supports_conversion(ImageFormat::Gif, ImageFormat::Png));
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_auto_discovery() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    
    // Test auto-discovery
    let registered_count = orchestrator.auto_discover_processors().await?;
    assert_eq!(registered_count, 1); // Should find FormatConverter
    
    // Verify the processor was registered
    let registered_processors = orchestrator.get_registered_processors();
    assert!(registered_processors.contains(&ProcessorType::FormatConverter));
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_quality_validation() -> Result<()> {
    // Test valid quality values
    assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 85).is_ok());
    assert!(FormatConverter::validate_quality(ImageFormat::WebP, 80).is_ok());
    
    // Test invalid quality values
    assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 5).is_err());
    assert!(FormatConverter::validate_quality(ImageFormat::Jpeg, 105).is_err());
    
    // Test formats that don't use quality
    assert!(FormatConverter::validate_quality(ImageFormat::Png, 50).is_ok()); // PNG doesn't use quality, so any value is ok
    assert!(FormatConverter::validate_quality(ImageFormat::Gif, 50).is_ok()); // GIF doesn't use quality
    
    Ok(())
}

#[tokio::test]
async fn test_format_converter_utility_functions() -> Result<()> {
    // Test format detection from path
    assert_eq!(
        FormatConverter::detect_format_from_path(&PathBuf::from("test.jpg")),
        Some(ImageFormat::Jpeg)
    );
    assert_eq!(
        FormatConverter::detect_format_from_path(&PathBuf::from("test.png")),
        Some(ImageFormat::Png)
    );
    assert_eq!(
        FormatConverter::detect_format_from_path(&PathBuf::from("test.webp")),
        Some(ImageFormat::WebP)
    );
    assert_eq!(
        FormatConverter::detect_format_from_path(&PathBuf::from("test.unknown")),
        None
    );
    
    // Test quality ranges
    assert_eq!(FormatConverter::get_quality_range(ImageFormat::Jpeg), Some((10, 100)));
    assert_eq!(FormatConverter::get_quality_range(ImageFormat::WebP), Some((0, 100)));
    assert_eq!(FormatConverter::get_quality_range(ImageFormat::Png), None);
    
    // Test default quality
    assert_eq!(FormatConverter::get_default_quality(ImageFormat::Jpeg), Some(85));
    assert_eq!(FormatConverter::get_default_quality(ImageFormat::WebP), Some(80));
    assert_eq!(FormatConverter::get_default_quality(ImageFormat::Png), None);
    
    Ok(())
}

#[tokio::test]
async fn test_compression_settings() -> Result<()> {
    // Test PNG compression range
    assert_eq!(FormatConverter::get_png_compression_range(), (0, 9));
    
    // Test PNG compression validation
    assert!(FormatConverter::validate_png_compression(5).is_ok());
    assert!(FormatConverter::validate_png_compression(0).is_ok());
    assert!(FormatConverter::validate_png_compression(9).is_ok());
    assert!(FormatConverter::validate_png_compression(10).is_err());
    
    // Test compression settings creation
    let jpeg_settings = FormatConverter::create_compression_settings(ImageFormat::Jpeg, Some(90));
    assert_eq!(jpeg_settings.quality, Some(90));
    assert!(jpeg_settings.progressive);
    assert!(jpeg_settings.optimize_coding);
    
    let png_settings = FormatConverter::create_compression_settings(ImageFormat::Png, Some(80));
    assert_eq!(png_settings.png_compression, Some(2)); // 9 - (80 * 9 / 100) = 9 - 7.2 = 1.8 -> rounds to 2
    
    let webp_settings = FormatConverter::create_compression_settings(ImageFormat::WebP, Some(100));
    assert!(webp_settings.webp_lossless);
    
    let webp_lossy_settings = FormatConverter::create_compression_settings(ImageFormat::WebP, Some(80));
    assert!(!webp_lossy_settings.webp_lossless);
    
    // Test compression settings validation
    assert!(FormatConverter::validate_compression_settings(ImageFormat::Jpeg, &jpeg_settings).is_ok());
    assert!(FormatConverter::validate_compression_settings(ImageFormat::Png, &png_settings).is_ok());
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_processor_validation() -> Result<()> {
    let orchestrator = ProcessingOrchestrator::new();
    let converter = FormatConverter::new()?;
    
    // Test processor validation
    let validation_result = orchestrator.validate_processor_compatibility(&converter);
    assert!(validation_result.is_ok());
    
    Ok(())
}

#[tokio::test]
async fn test_real_processing_job() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    // Create a temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.png");
    let output_path = temp_dir.path().join("output.jpg");
    
    // Create a simple test image (1x1 pixel PNG)
    use image::{ImageBuffer, Rgb};
    let img = ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([255u8, 0u8, 0u8]));
    img.save(&input_path).unwrap();
    
    let processing_input = ProcessingInput {
        job_id: Uuid::new_v4(),
        source_path: input_path,
        output_path: output_path.clone(),
        format: ImageFormat::Png,
        file_size: 1024,
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::Jpeg,
            quality: Some(85),
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    // Process the job (this will use the real implementation)
    let result = orchestrator.process_job(processing_input).await;
    assert!(result.is_ok());
    
    let output = result.unwrap();
    assert_eq!(output.format, ImageFormat::Jpeg);
    assert_eq!(output.operations_applied, vec!["format_conversion_to_Jpeg"]);
    assert!(output.processing_time.as_millis() >= 0);
    
    // Verify the output file was created
    assert!(output_path.exists());
    
    // Check actual file size on disk
    let actual_file_size = std::fs::metadata(&output_path).unwrap().len();
    println!("Output file size from processing: {}", output.file_size);
    println!("Actual file size on disk: {}", actual_file_size);
    
    assert!(output.file_size > 0);
    
    Ok(())
}

#[test]
fn test_format_converter_creation() {
    let converter = FormatConverter::new();
    assert!(converter.is_ok());
    
    let converter = converter.unwrap();
    assert!(!converter.get_supported_input_formats().is_empty());
    assert!(!converter.get_supported_output_formats().is_empty());
}

#[test]
fn test_format_converter_default() {
    let converter = FormatConverter::default();
    assert!(!converter.get_supported_input_formats().is_empty());
    assert!(!converter.get_supported_output_formats().is_empty());
}

#[tokio::test]
async fn test_comprehensive_format_conversions() -> Result<()> {
    use image::{ImageBuffer, Rgb};
    
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    // Create a temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    
    // Create a test image (10x10 pixel RGB)
    let test_image = ImageBuffer::from_fn(10, 10, |x, y| {
        Rgb([
            (x * 25) as u8,
            (y * 25) as u8,
            ((x + y) * 12) as u8,
        ])
    });
    
    // Test all supported format conversions
    let formats = vec![
        (ImageFormat::Png, "png"),
        (ImageFormat::Jpeg, "jpg"),
        (ImageFormat::WebP, "webp"),
        (ImageFormat::Gif, "gif"),
        (ImageFormat::Bmp, "bmp"),
        (ImageFormat::Tiff, "tiff"),
    ];
    
    for (source_format, source_ext) in &formats {
        for (target_format, target_ext) in &formats {
            if source_format == target_format {
                continue; // Skip same format conversions
            }
            
            let input_path = temp_dir.path().join(format!("test_input.{}", source_ext));
            let output_path = temp_dir.path().join(format!("test_output_{}_to_{}.{}", source_ext, target_ext, target_ext));
            
            // Save test image in source format
            test_image.save_with_format(&input_path, match source_format {
                ImageFormat::Jpeg => image::ImageFormat::Jpeg,
                ImageFormat::Png => image::ImageFormat::Png,
                ImageFormat::WebP => image::ImageFormat::WebP,
                ImageFormat::Gif => image::ImageFormat::Gif,
                ImageFormat::Bmp => image::ImageFormat::Bmp,
                ImageFormat::Tiff => image::ImageFormat::Tiff,
                _ => continue,
            }).unwrap();
            
            let processing_input = ProcessingInput {
                job_id: uuid::Uuid::new_v4(),
                source_path: input_path.clone(),
                output_path: output_path.clone(),
                format: *source_format,
                file_size: std::fs::metadata(&input_path).unwrap().len(),
                operations: vec![ProcessingOperation::Convert {
                    format: *target_format,
                    quality: Some(85),
                }],
                options: ProcessingOptions {
                    quality_priority: QualityPriority::Balanced,
                    preserve_metadata: true,
                    create_backup: false,
                    hardware_acceleration: true,
                    timeout: None,
                },
            };
            
            // Process the conversion
            let result = orchestrator.process_job(processing_input).await;
            assert!(result.is_ok(), "Failed to convert {} to {}: {:?}", source_ext, target_ext, result.err());
            
            let output = result.unwrap();
            assert_eq!(output.format, *target_format);
            assert!(output_path.exists(), "Output file not created for {} to {} conversion", source_ext, target_ext);
            assert!(output.file_size > 0, "Output file is empty for {} to {} conversion", source_ext, target_ext);
            
            // Verify the output file can be loaded
            let loaded_image = image::open(&output_path);
            assert!(loaded_image.is_ok(), "Cannot load converted {} to {} file", source_ext, target_ext);
            
            let loaded = loaded_image.unwrap();
            assert_eq!(loaded.width(), 10);
            assert_eq!(loaded.height(), 10);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_quality_settings_impact() -> Result<()> {
    use image::{ImageBuffer, Rgb};
    
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a more complex test image for quality testing
    let test_image = ImageBuffer::from_fn(50, 50, |x, y| {
        Rgb([
            ((x * y) % 256) as u8,
            ((x + y * 2) % 256) as u8,
            ((x * 2 + y) % 256) as u8,
        ])
    });
    
    let input_path = temp_dir.path().join("quality_test.png");
    test_image.save(&input_path).unwrap();
    
    // Test different quality levels for JPEG
    let qualities = vec![10, 50, 85, 95];
    let mut file_sizes = Vec::new();
    
    for quality in qualities {
        let output_path = temp_dir.path().join(format!("quality_test_{}.jpg", quality));
        
        let processing_input = ProcessingInput {
            job_id: uuid::Uuid::new_v4(),
            source_path: input_path.clone(),
            output_path: output_path.clone(),
            format: ImageFormat::Png,
            file_size: std::fs::metadata(&input_path).unwrap().len(),
            operations: vec![ProcessingOperation::Convert {
                format: ImageFormat::Jpeg,
                quality: Some(quality),
            }],
            options: ProcessingOptions {
                quality_priority: QualityPriority::Balanced,
                preserve_metadata: true,
                create_backup: false,
                hardware_acceleration: true,
                timeout: None,
            },
        };
        
        let result = orchestrator.process_job(processing_input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        file_sizes.push(output.file_size);
        
        // Verify file exists and can be loaded
        assert!(output_path.exists());
        let loaded = image::open(&output_path).unwrap();
        assert_eq!(loaded.width(), 50);
        assert_eq!(loaded.height(), 50);
    }
    
    // Verify that higher quality generally produces larger files
    // (allowing some tolerance for compression algorithms)
    assert!(file_sizes[0] < file_sizes[2], "Quality 10 should produce smaller file than quality 85");
    assert!(file_sizes[1] < file_sizes[3], "Quality 50 should produce smaller file than quality 95");
    
    Ok(())
}

#[tokio::test]
async fn test_png_compression_levels() -> Result<()> {
    use image::{ImageBuffer, Rgb};
    
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a test image with patterns that compress differently
    let test_image = ImageBuffer::from_fn(100, 100, |x, y| {
        if (x / 10 + y / 10) % 2 == 0 {
            Rgb([255u8, 255u8, 255u8])
        } else {
            Rgb([0u8, 0u8, 0u8])
        }
    });
    
    let input_path = temp_dir.path().join("compression_test.bmp");
    test_image.save_with_format(&input_path, image::ImageFormat::Bmp).unwrap();
    
    // Test different quality levels that should produce different PNG compression levels
    let qualities = vec![10, 50, 90]; // These should map to different compression levels
    let mut file_sizes = Vec::new();
    
    for quality in qualities {
        let output_path = temp_dir.path().join(format!("compression_test_{}.png", quality));
        
        let processing_input = ProcessingInput {
            job_id: uuid::Uuid::new_v4(),
            source_path: input_path.clone(),
            output_path: output_path.clone(),
            format: ImageFormat::Bmp,
            file_size: std::fs::metadata(&input_path).unwrap().len(),
            operations: vec![ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(quality),
            }],
            options: ProcessingOptions {
                quality_priority: QualityPriority::Balanced,
                preserve_metadata: true,
                create_backup: false,
                hardware_acceleration: true,
                timeout: None,
            },
        };
        
        let result = orchestrator.process_job(processing_input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        file_sizes.push(output.file_size);
        
        // Verify file exists and can be loaded
        assert!(output_path.exists());
        let loaded = image::open(&output_path).unwrap();
        assert_eq!(loaded.width(), 100);
        assert_eq!(loaded.height(), 100);
    }
    
    // All files should be valid PNG files (size differences may be minimal for this simple pattern)
    for size in &file_sizes {
        assert!(*size > 0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_large_file_performance() -> Result<()> {
    use image::{ImageBuffer, Rgb};
    use std::time::Instant;
    
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a larger test image (200x200)
    let test_image = ImageBuffer::from_fn(200, 200, |x, y| {
        Rgb([
            (x % 256) as u8,
            (y % 256) as u8,
            ((x + y) % 256) as u8,
        ])
    });
    
    let input_path = temp_dir.path().join("large_test.png");
    test_image.save(&input_path).unwrap();
    
    let output_path = temp_dir.path().join("large_test.jpg");
    
    let processing_input = ProcessingInput {
        job_id: uuid::Uuid::new_v4(),
        source_path: input_path.clone(),
        output_path: output_path.clone(),
        format: ImageFormat::Png,
        file_size: std::fs::metadata(&input_path).unwrap().len(),
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::Jpeg,
            quality: Some(85),
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    let start_time = Instant::now();
    let result = orchestrator.process_job(processing_input).await;
    let processing_time = start_time.elapsed();
    
    assert!(result.is_ok());
    let output = result.unwrap();
    
    // Performance assertions
    assert!(processing_time.as_millis() < 5000, "Processing took too long: {}ms", processing_time.as_millis());
    assert!(output.file_size > 0);
    assert!(output_path.exists());
    
    // Verify the output
    let loaded = image::open(&output_path).unwrap();
    assert_eq!(loaded.width(), 200);
    assert_eq!(loaded.height(), 200);
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_invalid_files() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Test with non-existent file
    let non_existent_path = temp_dir.path().join("does_not_exist.png");
    let output_path = temp_dir.path().join("output.jpg");
    
    let processing_input = ProcessingInput {
        job_id: uuid::Uuid::new_v4(),
        source_path: non_existent_path,
        output_path: output_path.clone(),
        format: ImageFormat::Png,
        file_size: 1024,
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::Jpeg,
            quality: Some(85),
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    let result = orchestrator.process_job(processing_input).await;
    assert!(result.is_err(), "Should fail with non-existent file");
    
    // Test with invalid quality value
    let valid_input_path = temp_dir.path().join("valid_test.png");
    let test_image = image::ImageBuffer::from_fn(10, 10, |_, _| image::Rgb([255u8, 0u8, 0u8]));
    test_image.save(&valid_input_path).unwrap();
    
    let processing_input = ProcessingInput {
        job_id: uuid::Uuid::new_v4(),
        source_path: valid_input_path.clone(),
        output_path: output_path,
        format: ImageFormat::Png,
        file_size: std::fs::metadata(&valid_input_path).unwrap().len(),
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::Jpeg,
            quality: Some(150), // Invalid quality
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    let result = orchestrator.process_job(processing_input).await;
    assert!(result.is_err(), "Should fail with invalid quality value");
    
    Ok(())
}

#[tokio::test]
async fn test_watermark_engine_registration() -> Result<()> {
    // Create a new orchestrator
    let mut orchestrator = ProcessingOrchestrator::new();
    
    // Verify no processors are registered initially
    assert_eq!(orchestrator.get_registered_processors().len(), 0);
    
    // Register the WatermarkEngine
    orchestrator.register_watermark_engine()?;
    
    // Verify the processor was registered
    let registered_processors = orchestrator.get_registered_processors();
    assert_eq!(registered_processors.len(), 1);
    assert!(registered_processors.contains(&ProcessorType::WatermarkEngine));
    
    // Verify capabilities
    let capabilities = orchestrator.get_processor_capabilities(&ProcessorType::WatermarkEngine);
    assert!(capabilities.is_some());
    
    let caps = capabilities.unwrap();
    assert!(!caps.input_formats.is_empty());
    assert!(!caps.output_formats.is_empty());
    assert!(caps.operations.contains(&crate::processing::ProcessorOperation::Watermark));
    assert!(!caps.streaming_support); // Watermark engine doesn't support streaming
    assert!(caps.batch_optimized);
    
    Ok(())
}

#[tokio::test]
async fn test_watermark_engine_capabilities() -> Result<()> {
    let watermark_engine = crate::watermark_engine::WatermarkEngine::new();
    let capabilities = watermark_engine.get_capabilities();
    
    // Verify basic capabilities
    assert!(!capabilities.input_formats.is_empty());
    assert!(!capabilities.output_formats.is_empty());
    assert!(capabilities.operations.contains(&crate::processing::ProcessorOperation::Watermark));
    
    // Verify common formats are supported
    assert!(capabilities.input_formats.contains(&ImageFormat::Jpeg));
    assert!(capabilities.input_formats.contains(&ImageFormat::Png));
    assert!(capabilities.output_formats.contains(&ImageFormat::Jpeg));
    assert!(capabilities.output_formats.contains(&ImageFormat::Png));
    
    Ok(())
}

#[tokio::test]
async fn test_watermark_engine_metadata() -> Result<()> {
    let watermark_engine = crate::watermark_engine::WatermarkEngine::new();
    let metadata = watermark_engine.get_metadata();
    
    // Verify metadata is properly set
    assert_eq!(metadata.name, "Watermark Engine");
    assert_eq!(metadata.version, "1.0.0");
    assert!(!metadata.description.is_empty());
    assert!(!metadata.author.is_empty());
    
    // Verify performance profile
    assert!(metadata.performance_profile.speed_factor > 0.0);
    assert!(metadata.performance_profile.parallel_friendly);
    
    Ok(())
}

#[tokio::test]
async fn test_watermark_operation_handling() -> Result<()> {
    let watermark_engine = crate::watermark_engine::WatermarkEngine::new();
    
    // Test operation handling
    let watermark_operation = ProcessingOperation::Watermark {
        config: crate::models::WatermarkConfig {
            watermark_path: std::path::PathBuf::from("test_watermark.png"),
            positions: vec![crate::models::WatermarkPosition::BottomRight],
            opacity: 0.7,
            scale: 0.1,
            blend_mode: crate::models::BlendMode::Normal,
            scaling_options: crate::models::WatermarkScalingOptions::default(),
            alignment: crate::models::WatermarkAlignment::default(),
            offset: crate::models::WatermarkOffset::default(),
            visual_effects: crate::models::WatermarkVisualEffects::default(),
        },
    };
    assert!(watermark_engine.can_handle_operation(&watermark_operation));
    
    // Test unsupported operation
    let convert_operation = ProcessingOperation::Convert {
        format: ImageFormat::Png,
        quality: Some(85),
    };
    assert!(!watermark_engine.can_handle_operation(&convert_operation));
    
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_auto_discovery_with_watermark() -> Result<()> {
    let mut orchestrator = ProcessingOrchestrator::new();
    
    // Test auto-discovery
    let registered_count = orchestrator.auto_discover_processors().await?;
    assert_eq!(registered_count, 2); // Should find FormatConverter and WatermarkEngine
    
    // Verify both processors were registered
    let registered_processors = orchestrator.get_registered_processors();
    assert!(registered_processors.contains(&ProcessorType::FormatConverter));
    assert!(registered_processors.contains(&ProcessorType::WatermarkEngine));
    
    Ok(())
}

#[tokio::test]
async fn test_webp_lossless_vs_lossy() -> Result<()> {
    use image::{ImageBuffer, Rgb};
    
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a test image
    let test_image = ImageBuffer::from_fn(50, 50, |x, y| {
        Rgb([
            (x * 5) as u8,
            (y * 5) as u8,
            ((x + y) * 2) as u8,
        ])
    });
    
    let input_path = temp_dir.path().join("webp_test.png");
    test_image.save(&input_path).unwrap();
    
    // Test lossless WebP (quality 100)
    let lossless_path = temp_dir.path().join("webp_lossless.webp");
    let lossless_input = ProcessingInput {
        job_id: uuid::Uuid::new_v4(),
        source_path: input_path.clone(),
        output_path: lossless_path.clone(),
        format: ImageFormat::Png,
        file_size: std::fs::metadata(&input_path).unwrap().len(),
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::WebP,
            quality: Some(100),
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    let lossless_result = orchestrator.process_job(lossless_input).await;
    assert!(lossless_result.is_ok());
    
    // Test lossy WebP (quality 80)
    let lossy_path = temp_dir.path().join("webp_lossy.webp");
    let lossy_input = ProcessingInput {
        job_id: uuid::Uuid::new_v4(),
        source_path: input_path.clone(),
        output_path: lossy_path.clone(),
        format: ImageFormat::Png,
        file_size: std::fs::metadata(&input_path).unwrap().len(),
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::WebP,
            quality: Some(80),
        }],
        options: ProcessingOptions {
            quality_priority: QualityPriority::Balanced,
            preserve_metadata: true,
            create_backup: false,
            hardware_acceleration: true,
            timeout: None,
        },
    };
    
    let lossy_result = orchestrator.process_job(lossy_input).await;
    assert!(lossy_result.is_ok());
    
    // Verify both files exist and can be loaded
    assert!(lossless_path.exists());
    assert!(lossy_path.exists());
    
    let lossless_image = image::open(&lossless_path).unwrap();
    let lossy_image = image::open(&lossy_path).unwrap();
    
    assert_eq!(lossless_image.width(), 50);
    assert_eq!(lossless_image.height(), 50);
    assert_eq!(lossy_image.width(), 50);
    assert_eq!(lossy_image.height(), 50);
    
    Ok(())
}