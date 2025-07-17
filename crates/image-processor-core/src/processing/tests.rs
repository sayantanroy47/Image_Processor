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