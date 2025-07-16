//! Benchmarks for image processing operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image_processor_core::{
    ProcessingOrchestrator, MockProcessor, ProcessorType, ProcessingInput, 
    ProcessingOperation, ProcessingOptions, ImageFormat,
};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

fn benchmark_job_processing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("process_single_job", |b| {
        b.to_async(&rt).iter(|| async {
            let mut orchestrator = ProcessingOrchestrator::new(4);
            let processor = Arc::new(MockProcessor::new("benchmark-processor"));
            orchestrator.register_processor(ProcessorType::FormatConverter, processor);

            let input = ProcessingInput {
                job_id: Uuid::new_v4(),
                source_path: PathBuf::from("benchmark_input.jpg"),
                output_path: PathBuf::from("benchmark_output.jpg"),
                operations: vec![ProcessingOperation::Convert {
                    format: ImageFormat::Png,
                    quality: Some(85),
                }],
                options: ProcessingOptions::default(),
                file_size: 1024 * 1024, // 1MB
                format: ImageFormat::Jpeg,
            };

            let result = orchestrator.process_job(black_box(input)).await;
            assert!(result.is_ok());
            result.unwrap()
        });
    });
}

fn benchmark_format_detection(c: &mut Criterion) {
    use image_processor_core::utils::format;
    use std::path::Path;
    
    c.bench_function("detect_format_from_extension", |b| {
        b.iter(|| {
            let path = Path::new("test_image.jpg");
            format::detect_format_from_extension(black_box(path))
        });
    });
}

fn benchmark_validation(c: &mut Criterion) {
    use image_processor_core::utils::validation;
    
    c.bench_function("validate_dimensions", |b| {
        b.iter(|| {
            validation::validate_dimensions(black_box(1920), black_box(1080))
        });
    });
    
    c.bench_function("validate_quality", |b| {
        b.iter(|| {
            validation::validate_quality(black_box(85))
        });
    });
    
    c.bench_function("validate_opacity", |b| {
        b.iter(|| {
            validation::validate_opacity(black_box(0.8))
        });
    });
}

fn benchmark_performance_utils(c: &mut Criterion) {
    use image_processor_core::utils::performance;
    use std::time::Duration;
    
    c.bench_function("calculate_throughput", |b| {
        b.iter(|| {
            performance::calculate_throughput(
                black_box(1024 * 1024), 
                black_box(Duration::from_secs(1))
            )
        });
    });
    
    c.bench_function("performance_timer", |b| {
        b.iter(|| {
            let timer = performance::Timer::new("benchmark");
            std::thread::sleep(Duration::from_nanos(1));
            timer.elapsed_ms()
        });
    });
}

fn benchmark_config_operations(c: &mut Criterion) {
    use image_processor_core::{ConfigManager, AppConfig};
    use tempfile::tempdir;
    
    c.bench_function("config_creation", |b| {
        b.iter(|| {
            let temp_dir = tempdir().unwrap();
            let config_path = temp_dir.path().join("bench_config.toml");
            ConfigManager::with_path(black_box(config_path))
        });
    });
    
    c.bench_function("config_serialization", |b| {
        b.iter(|| {
            let config = AppConfig::default();
            toml::to_string_pretty(&black_box(config))
        });
    });
}

criterion_group!(
    benches,
    benchmark_job_processing,
    benchmark_format_detection,
    benchmark_validation,
    benchmark_performance_utils,
    benchmark_config_operations
);

criterion_main!(benches);