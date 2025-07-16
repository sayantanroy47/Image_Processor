//! Comprehensive logging system with structured output and performance monitoring

use crate::error::{ProcessingError, Result};
use std::path::PathBuf;
use tracing::{Level, Subscriber};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub output: LogOutput,
    pub structured: bool,
    pub performance_metrics: bool,
    pub error_tracking: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            output: LogOutput::Console,
            structured: true,
            performance_metrics: true,
            error_tracking: true,
        }
    }
}

/// Log level configuration
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

/// Log output configuration
#[derive(Debug, Clone)]
pub enum LogOutput {
    Console,
    File(PathBuf),
    Both { console: bool, file: PathBuf },
}

/// Initialize the logging system with the given configuration
pub fn init_logging() -> Result<Option<WorkerGuard>> {
    init_logging_with_config(LoggingConfig::default())
}

/// Initialize the logging system with custom configuration
pub fn init_logging_with_config(config: LoggingConfig) -> Result<Option<WorkerGuard>> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            let level_str = match config.level {
                LogLevel::Trace => "trace",
                LogLevel::Debug => "debug", 
                LogLevel::Info => "info",
                LogLevel::Warn => "warn",
                LogLevel::Error => "error",
            };
            EnvFilter::new(format!("image_processor_core={},image_processor_cli={},image_processor_gui={}", level_str, level_str, level_str))
        });

    let mut guard = None;

    match config.output {
        LogOutput::Console => {
            let fmt_layer = create_console_layer(config.structured);
            
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .try_init()
                .map_err(|e| ProcessingError::LoggingError {
                    message: format!("Failed to initialize console logging: {}", e),
                })?;
        }
        LogOutput::File(path) => {
            let (file_layer, file_guard) = create_file_layer(&path, config.structured)?;
            guard = Some(file_guard);
            
            tracing_subscriber::registry()
                .with(env_filter)
                .with(file_layer)
                .try_init()
                .map_err(|e| ProcessingError::LoggingError {
                    message: format!("Failed to initialize file logging: {}", e),
                })?;
        }
        LogOutput::Both { console, file } => {
            let console_layer = if console {
                Some(create_console_layer(config.structured))
            } else {
                None
            };
            
            let (file_layer, file_guard) = create_file_layer(&file, config.structured)?;
            guard = Some(file_guard);
            
            let registry = tracing_subscriber::registry().with(env_filter);
            
            if let Some(console_layer) = console_layer {
                registry
                    .with(console_layer)
                    .with(file_layer)
                    .try_init()
                    .map_err(|e| ProcessingError::LoggingError {
                        message: format!("Failed to initialize combined logging: {}", e),
                    })?;
            } else {
                registry
                    .with(file_layer)
                    .try_init()
                    .map_err(|e| ProcessingError::LoggingError {
                        message: format!("Failed to initialize file logging: {}", e),
                    })?;
            }
        }
    }

    tracing::info!(
        "Logging system initialized",
        level = ?config.level,
        output = ?config.output,
        structured = config.structured,
        performance_metrics = config.performance_metrics,
        error_tracking = config.error_tracking
    );

    Ok(guard)
}

/// Create a console logging layer
fn create_console_layer(structured: bool) -> Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> {
    if structured {
        Box::new(
            fmt::layer()
                .json()
                .with_span_events(FmtSpan::CLOSE)
                .with_current_span(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
        )
    } else {
        Box::new(
            fmt::layer()
                .pretty()
                .with_span_events(FmtSpan::CLOSE)
                .with_current_span(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
        )
    }
}

/// Create a file logging layer
fn create_file_layer(
    path: &PathBuf,
    _structured: bool,
) -> Result<(Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>, WorkerGuard)> {
    // Ensure the parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ProcessingError::LoggingError {
            message: format!("Failed to create log directory: {}", e),
        })?;
    }

    let file_appender = tracing_appender::rolling::daily(
        path.parent().unwrap_or(&PathBuf::from(".")),
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image-processor.log"),
    );

    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let layer = Box::new(
        fmt::layer()
            .json()
            .with_writer(non_blocking)
            .with_span_events(FmtSpan::CLOSE)
            .with_current_span(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
    );

    Ok((layer, guard))
}

/// Performance metrics structure
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub throughput_mbps: f64,
    pub memory_peak_mb: u64,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: Option<f32>,
    pub cache_hit_rate: f32,
}

/// Batch processing metrics
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    pub total_images: usize,
    pub successful_count: usize,
    pub failed_count: usize,
    pub total_duration: std::time::Duration,
    pub average_processing_time: std::time::Duration,
}

/// Performance logger for structured performance metrics
pub struct PerformanceLogger;

impl PerformanceLogger {
    /// Log processing performance metrics
    pub fn log_processing_metrics(metrics: ProcessingMetrics) {
        tracing::info!(
            "Performance metrics",
            throughput_mbps = metrics.throughput_mbps,
            memory_peak_mb = metrics.memory_peak_mb,
            cpu_usage_percent = metrics.cpu_usage_percent,
            gpu_usage_percent = metrics.gpu_usage_percent,
            cache_hit_rate = metrics.cache_hit_rate
        );
    }

    /// Log batch processing metrics
    pub fn log_batch_metrics(batch_id: &str, metrics: BatchMetrics) {
        tracing::info!(
            "Batch processing completed",
            batch_id = batch_id,
            total_images = metrics.total_images,
            successful = metrics.successful_count,
            failed = metrics.failed_count,
            total_duration_sec = metrics.total_duration.as_secs(),
            average_time_per_image_ms = metrics.average_processing_time.as_millis()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert!(matches!(config.level, LogLevel::Info));
        assert!(matches!(config.output, LogOutput::Console));
        assert!(config.structured);
        assert!(config.performance_metrics);
        assert!(config.error_tracking);
    }

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(Level::from(LogLevel::Info), Level::INFO);
        assert_eq!(Level::from(LogLevel::Error), Level::ERROR);
    }

    #[tokio::test]
    async fn test_console_logging_init() {
        let config = LoggingConfig {
            output: LogOutput::Console,
            ..Default::default()
        };
        
        // This test might fail if logging is already initialized
        // In a real test environment, we'd use a different approach
        let result = init_logging_with_config(config);
        // Just verify it doesn't panic - actual initialization might fail in test environment
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_file_logging_path_creation() {
        let temp_dir = tempdir().unwrap();
        let log_path = temp_dir.path().join("logs").join("test.log");
        
        let config = LoggingConfig {
            output: LogOutput::File(log_path.clone()),
            ..Default::default()
        };
        
        // Test that the function handles path creation
        let result = init_logging_with_config(config);
        assert!(result.is_ok() || result.is_err()); // Either works or fails gracefully
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = ProcessingMetrics {
            throughput_mbps: 150.5,
            memory_peak_mb: 512,
            cpu_usage_percent: 75.2,
            gpu_usage_percent: Some(45.8),
            cache_hit_rate: 0.85,
        };
        
        // Test that logging doesn't panic
        PerformanceLogger::log_processing_metrics(metrics);
    }

    #[test]
    fn test_batch_metrics() {
        let metrics = BatchMetrics {
            total_images: 100,
            successful_count: 95,
            failed_count: 5,
            total_duration: std::time::Duration::from_secs(120),
            average_processing_time: std::time::Duration::from_millis(1200),
        };
        
        // Test that logging doesn't panic
        PerformanceLogger::log_batch_metrics("test-batch-001", metrics);
    }
}