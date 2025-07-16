//! Integration tests for the image processor core

use image_processor_core::{
    init, version, ConfigManager, ProcessingOrchestrator, MockProcessor, ProcessorType,
    ProcessingInput, ProcessingOperation, ProcessingOptions, ImageFormat, JobId,
};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[tokio::test]
async fn test_core_initialization() {
    let result = init().await;
    assert!(result.is_ok(), "Core initialization should succeed");
}

#[test]
fn test_version_info() {
    let version_str = version();
    assert!(!version_str.is_empty(), "Version should not be empty");
    assert_eq!(version_str, "0.1.0", "Version should match workspace version");
}

#[test]
fn test_config_manager() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("test_config.toml");
    
    let manager = ConfigManager::with_path(config_path);
    assert!(manager.is_ok(), "Config manager should initialize successfully");
    
    let manager = manager.unwrap();
    let config = manager.config();
    
    // Test default values
    assert_eq!(config.processing.default_quality, 85);
    assert!(config.processing.backup_enabled);
    assert_eq!(config.ui.window_width, 1200);
    assert_eq!(config.ui.window_height, 800);
}

#[tokio::test]
async fn test_processing_orchestrator() {
    let mut orchestrator = ProcessingOrchestrator::new(4);
    
    // Register a mock processor
    let processor = Arc::new(MockProcessor::new("test-processor"));
    orchestrator.register_processor(ProcessorType::FormatConverter, processor);
    
    // Verify processor registration
    let retrieved = orchestrator.get_processor(ProcessorType::FormatConverter);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name(), "test-processor");
    
    // Test format support
    assert!(orchestrator.is_format_supported(ImageFormat::Jpeg));
    assert!(orchestrator.is_format_supported(ImageFormat::Png));
    
    // Test capabilities
    let capabilities = orchestrator.get_all_capabilities();
    assert!(!capabilities.is_empty());
    assert!(capabilities.contains_key(&ProcessorType::FormatConverter));
}

#[tokio::test]
async fn test_job_processing() {
    let mut orchestrator = ProcessingOrchestrator::new(4);
    let processor = Arc::new(MockProcessor::new("test-processor"));
    orchestrator.register_processor(ProcessorType::FormatConverter, processor);

    let job_id = Uuid::new_v4();
    let input = ProcessingInput {
        job_id,
        source_path: PathBuf::from("test_input.jpg"),
        output_path: PathBuf::from("test_output.jpg"),
        operations: vec![ProcessingOperation::Convert {
            format: ImageFormat::Jpeg,
            quality: Some(85),
        }],
        options: ProcessingOptions::default(),
        file_size: 1024,
        format: ImageFormat::Jpeg,
    };

    let result = orchestrator.process_job(input).await;
    assert!(result.is_ok(), "Job processing should succeed");
    
    let output = result.unwrap();
    assert_eq!(output.job_id, job_id);
    assert!(!output.operations_applied.is_empty());
    assert!(output.processing_time.as_millis() > 0);
}

#[test]
fn test_image_format_properties() {
    // Test format extensions
    assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::WebP.extension(), "webp");
    
    // Test MIME types
    assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
    assert_eq!(ImageFormat::Png.mime_type(), "image/png");
    assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    
    // Test transparency support
    assert!(!ImageFormat::Jpeg.supports_transparency());
    assert!(ImageFormat::Png.supports_transparency());
    assert!(ImageFormat::WebP.supports_transparency());
    
    // Test animation support
    assert!(!ImageFormat::Jpeg.supports_animation());
    assert!(!ImageFormat::Png.supports_animation());
    assert!(ImageFormat::Gif.supports_animation());
    assert!(ImageFormat::WebP.supports_animation());
}

#[test]
fn test_processing_options_defaults() {
    let options = ProcessingOptions::default();
    
    assert!(options.preserve_metadata);
    assert!(options.create_backup);
    assert!(options.hardware_acceleration);
    assert!(options.timeout.is_some());
    
    let timeout = options.timeout.unwrap();
    assert_eq!(timeout.as_secs(), 300); // 5 minutes
}

#[tokio::test]
async fn test_concurrent_job_limit() {
    let orchestrator = ProcessingOrchestrator::new(1); // Limit to 1 concurrent job
    
    // Since we don't have actual processors registered, 
    // we can't test the actual concurrent limit easily in this mock setup
    // This test verifies the orchestrator can be created with limits
    assert_eq!(orchestrator.get_active_jobs().await.len(), 0);
}

#[test]
fn test_error_handling() {
    use image_processor_core::{ProcessingError, Result};
    
    // Test error categorization
    let error = ProcessingError::UnsupportedFormat {
        format: "xyz".to_string(),
    };
    assert_eq!(error.error_type(), "unsupported_format");
    assert!(!error.is_recoverable());
    
    // Test recoverable errors
    let timeout_error = ProcessingError::Timeout;
    assert!(timeout_error.is_recoverable());
    
    // Test error conversion
    let result: Result<()> = Err(ProcessingError::OutOfMemory);
    assert!(result.is_err());
}

#[test]
fn test_utility_functions() {
    use image_processor_core::utils::{validation, system, performance};
    use std::time::Duration;
    
    // Test validation functions
    assert!(validation::validate_quality(85).is_ok());
    assert!(validation::validate_quality(0).is_err());
    assert!(validation::validate_quality(101).is_err());
    
    assert!(validation::validate_opacity(0.5).is_ok());
    assert!(validation::validate_opacity(-0.1).is_err());
    assert!(validation::validate_opacity(1.1).is_err());
    
    assert!(validation::validate_dimensions(800, 600).is_ok());
    assert!(validation::validate_dimensions(0, 600).is_err());
    
    // Test system functions
    assert!(system::cpu_count() > 0);
    assert!(system::optimal_worker_count() > 0);
    
    // Test performance functions
    let throughput = performance::calculate_throughput(1024 * 1024, Duration::from_secs(1));
    assert!((throughput - 1.0).abs() < 0.01); // Should be ~1 MB/s
}