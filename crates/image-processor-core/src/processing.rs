//! Core processing functionality

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use crate::models::{JobId, ProcessingInput, ProcessingOutput, ProcessingOperation};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, error, instrument, debug};

/// Core trait that all image processors must implement
#[async_trait]
pub trait ImageProcessor: Send + Sync {
    /// Process an image with the given input parameters
    async fn process(&self, input: &ProcessingInput) -> Result<ProcessingOutput>;
    
    /// Check if this processor supports the given format
    fn supports_format(&self, format: ImageFormat) -> bool;
    
    /// Get the capabilities of this processor
    fn get_capabilities(&self) -> ProcessorCapabilities;
    
    /// Get metadata about this processor
    fn get_metadata(&self) -> ProcessorMetadata;
    
    /// Check if this processor can handle the given operation
    fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool;
}

/// Capabilities that a processor can provide
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessorCapabilities {
    /// Supported input formats
    pub input_formats: Vec<ImageFormat>,
    /// Supported output formats
    pub output_formats: Vec<ImageFormat>,
    /// Supported operations
    pub operations: Vec<ProcessorOperation>,
    /// Whether hardware acceleration is available
    pub hardware_acceleration: bool,
    /// Whether streaming processing is supported
    pub streaming_support: bool,
    /// Maximum file size that can be processed (in bytes)
    pub max_file_size: Option<u64>,
    /// Whether batch processing is optimized
    pub batch_optimized: bool,
}

/// Types of operations a processor can handle
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorOperation {
    /// Format conversion
    FormatConversion,
    /// Image resizing
    Resize,
    /// Watermarking
    Watermark,
    /// Background removal
    BackgroundRemoval,
    /// Color correction
    ColorCorrection,
    /// Cropping
    Crop,
    /// Rotation
    Rotate,
    /// Text overlay
    TextOverlay,
    /// Collage creation
    Collage,
}

/// Metadata about a processor
#[derive(Debug, Clone)]
pub struct ProcessorMetadata {
    /// Name of the processor
    pub name: String,
    /// Version of the processor
    pub version: String,
    /// Description of what the processor does
    pub description: String,
    /// Author of the processor
    pub author: String,
    /// Performance profile
    pub performance_profile: PerformanceProfile,
}

/// Performance characteristics of a processor
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Speed factor (higher is faster)
    pub speed_factor: f64,
    /// Memory usage pattern
    pub memory_usage: MemoryUsage,
    /// CPU usage pattern
    pub cpu_usage: CpuUsage,
    /// Whether the processor is parallel-friendly
    pub parallel_friendly: bool,
}

/// Memory usage patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryUsage {
    /// Low memory usage
    Low,
    /// Streaming processing (constant memory)
    Streaming,
    /// Moderate memory usage
    Moderate,
    /// High memory usage
    High,
}

/// CPU usage patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuUsage {
    /// Low CPU usage
    Low,
    /// Moderate CPU usage
    Moderate,
    /// High CPU usage
    High,
    /// Very high CPU usage
    VeryHigh,
}

/// Types of processors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    /// Format converter
    FormatConverter,
    /// Watermark engine
    WatermarkEngine,
    /// Resize engine
    ResizeEngine,
    /// Background removal engine
    BackgroundRemovalEngine,
    /// Color correction engine
    ColorCorrectionEngine,
    /// Text overlay engine
    TextOverlayEngine,
    /// Collage engine
    CollageEngine,
}

/// Processor selection strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorSelectionStrategy {
    /// Select the fastest processor
    Fastest,
    /// Select the most memory-efficient processor
    MemoryEfficient,
    /// Select a balanced processor
    Balanced,
    /// Select a specific processor by name
    Specific(String),
}

/// Main orchestrator for coordinating image processing operations
pub struct ProcessingOrchestrator {
    /// Registered processors
    processors: HashMap<ProcessorType, Arc<dyn ImageProcessor>>,
    /// Processor selection strategy
    selection_strategy: ProcessorSelectionStrategy,
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
}

/// Performance monitoring for processors
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Processing times for each processor type
    processing_times: HashMap<ProcessorType, Vec<std::time::Duration>>,
    /// Memory usage for each processor type
    memory_usage: HashMap<ProcessorType, Vec<u64>>,
    /// Success rates for each processor type (successes, total)
    success_rates: HashMap<ProcessorType, (u64, u64)>,
}

impl ProcessingOrchestrator {
    /// Create a new processing orchestrator
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
            selection_strategy: ProcessorSelectionStrategy::Balanced,
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        }
    }

    /// Register a format converter processor
    pub fn register_format_converter(&mut self) -> Result<()> {
        let converter = Arc::new(crate::format_converter::FormatConverter::new()?);
        self.processors.insert(ProcessorType::FormatConverter, converter);
        info!("Registered FormatConverter processor");
        Ok(())
    }

    /// Get all registered processor types
    pub fn get_registered_processors(&self) -> Vec<ProcessorType> {
        self.processors.keys().cloned().collect()
    }

    /// Get capabilities for a specific processor type
    pub fn get_processor_capabilities(&self, processor_type: &ProcessorType) -> Option<ProcessorCapabilities> {
        self.processors.get(processor_type).map(|p| p.get_capabilities())
    }

    /// Check if an input format is supported by any processor
    pub fn supports_input_format(&self, format: ImageFormat) -> bool {
        self.processors.values().any(|p| p.supports_format(format))
    }

    /// Check if an operation is supported by any processor
    pub fn supports_operation(&self, operation: &ProcessingOperation) -> bool {
        self.processors.values().any(|p| p.can_handle_operation(operation))
    }

    /// Find a processor that can handle the given operation
    pub fn find_processor_for_operation(&self, operation: &ProcessingOperation) -> Option<Arc<dyn ImageProcessor>> {
        self.processors.values()
            .find(|p| p.can_handle_operation(operation))
            .cloned()
    }

    /// Get format conversion matrix (what formats can be converted to what)
    pub fn get_format_conversion_matrix(&self) -> HashMap<ImageFormat, Vec<ImageFormat>> {
        let mut matrix = HashMap::new();
        
        if let Some(converter) = self.processors.get(&ProcessorType::FormatConverter) {
            let caps = converter.get_capabilities();
            for input_format in &caps.input_formats {
                matrix.insert(*input_format, caps.output_formats.clone());
            }
        }
        
        matrix
    }

    /// Set the processor selection strategy
    pub fn set_selection_strategy(&mut self, strategy: ProcessorSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Process a job using the appropriate processor
    pub async fn process_job(&self, input: ProcessingInput) -> Result<ProcessingOutput> {
        // Find the appropriate processor for the first operation
        let operation = input.operations.first()
            .ok_or_else(|| ProcessingError::ProcessingFailed {
                message: "No operations specified".to_string(),
            })?;

        let processor = self.find_processor_for_operation(operation)
            .ok_or_else(|| ProcessingError::ProcessingFailed {
                message: format!("No processor found for operation: {:?}", operation),
            })?;

        // Process the job
        let start_time = Instant::now();
        let result = processor.process(&input).await;
        let processing_time = start_time.elapsed();

        // Record performance metrics
        if let Some(processor_type) = self.get_processor_type_for_operation(operation) {
            let mut monitor = Arc::clone(&self.performance_monitor);
            // Note: In a real implementation, we'd need interior mutability here
            // For now, we'll skip the performance recording
        }

        result
    }

    /// Auto-discover and register available processors
    pub async fn auto_discover_processors(&mut self) -> Result<usize> {
        let mut count = 0;
        
        // Try to register FormatConverter
        if self.register_format_converter().is_ok() {
            count += 1;
        }
        
        Ok(count)
    }

    /// Validate processor compatibility
    pub fn validate_processor_compatibility(&self, processor: &dyn ImageProcessor) -> Result<()> {
        let capabilities = processor.get_capabilities();
        
        // Basic validation - ensure processor has at least one supported format
        if capabilities.input_formats.is_empty() {
            return Err(ProcessingError::ProcessingFailed {
                message: "Processor has no supported input formats".to_string(),
            });
        }
        
        if capabilities.output_formats.is_empty() {
            return Err(ProcessingError::ProcessingFailed {
                message: "Processor has no supported output formats".to_string(),
            });
        }
        
        Ok(())
    }

    /// Get processor type for a given operation
    fn get_processor_type_for_operation(&self, operation: &ProcessingOperation) -> Option<ProcessorType> {
        match operation {
            ProcessingOperation::Convert { .. } => Some(ProcessorType::FormatConverter),
            ProcessingOperation::Resize { .. } => Some(ProcessorType::ResizeEngine),
            ProcessingOperation::Watermark { .. } => Some(ProcessorType::WatermarkEngine),
            ProcessingOperation::RemoveBackground { .. } => Some(ProcessorType::BackgroundRemovalEngine),
            ProcessingOperation::ColorCorrect { .. } => Some(ProcessorType::ColorCorrectionEngine),
            ProcessingOperation::Crop { .. } => Some(ProcessorType::ResizeEngine),
            ProcessingOperation::Rotate { .. } => Some(ProcessorType::ResizeEngine),
            ProcessingOperation::AddText { .. } => Some(ProcessorType::TextOverlayEngine),
            ProcessingOperation::CreateCollage { .. } => Some(ProcessorType::CollageEngine),
        }
    }

    /// Validate that a processing pipeline can be executed
    pub fn validate_processing_pipeline(&self, operations: &[ProcessingOperation]) -> Result<()> {
        for operation in operations {
            if !self.supports_operation(operation) {
                return Err(ProcessingError::ProcessingFailed {
                    message: format!("Operation not supported: {:?}", operation),
                });
            }
        }
        Ok(())
    }
}

impl Default for ProcessingOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics for a processor
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Average processing time
    pub avg_processing_time: std::time::Duration,
    /// Total number of operations processed
    pub total_operations: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average memory usage in bytes
    pub avg_memory_usage: u64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            processing_times: HashMap::new(),
            memory_usage: HashMap::new(),
            success_rates: HashMap::new(),
        }
    }

    /// Record processing time for a processor
    pub fn record_processing_time(&mut self, processor_type: ProcessorType, duration: std::time::Duration) {
        self.processing_times
            .entry(processor_type)
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Record memory usage for a processor
    pub fn record_memory_usage(&mut self, processor_type: ProcessorType, memory_bytes: u64) {
        self.memory_usage
            .entry(processor_type)
            .or_insert_with(Vec::new)
            .push(memory_bytes);
    }

    /// Record success or failure for a processor
    pub fn record_result(&mut self, processor_type: ProcessorType, success: bool) {
        let (successes, total) = self.success_rates
            .entry(processor_type)
            .or_insert((0, 0));
        
        if success {
            *successes += 1;
        }
        *total += 1;
    }

    /// Get statistics for all processors
    pub fn get_stats(&self) -> HashMap<ProcessorType, ProcessorStats> {
        let mut stats = HashMap::new();
        let empty_times = Vec::new();
        let empty_memory = Vec::new();
        
        for processor_type in self.processing_times.keys()
            .chain(self.memory_usage.keys())
            .chain(self.success_rates.keys())
        {
            let times = self.processing_times.get(processor_type).unwrap_or(&empty_times);
            let memory = self.memory_usage.get(processor_type).unwrap_or(&empty_memory);
            let (successes, total) = self.success_rates.get(processor_type).unwrap_or(&(0, 0));
            
            let avg_time = if times.is_empty() {
                std::time::Duration::ZERO
            } else {
                let total_time: std::time::Duration = times.iter().sum();
                total_time / times.len() as u32
            };
            
            let avg_memory = if memory.is_empty() {
                0
            } else {
                memory.iter().sum::<u64>() / memory.len() as u64
            };
            
            let success_rate = if *total == 0 {
                1.0
            } else {
                *successes as f64 / *total as f64
            };
            
            stats.insert(processor_type.clone(), ProcessorStats {
                avg_processing_time: avg_time,
                total_operations: *total,
                success_rate,
                avg_memory_usage: avg_memory,
            });
        }
        
        stats
    }

    /// Get statistics for a specific processor
    pub fn get_processor_stats(&self, processor_type: &ProcessorType) -> Option<ProcessorStats> {
        self.get_stats().get(processor_type).cloned()
    }

    /// Clear all statistics
    pub fn clear_stats(&mut self) {
        self.processing_times.clear();
        self.memory_usage.clear();
        self.success_rates.clear();
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

/// Create a new processing orchestrator with FormatConverter registered
pub fn create_orchestrator_with_format_converter() -> Result<ProcessingOrchestrator> {
    let mut orchestrator = ProcessingOrchestrator::new();
    orchestrator.register_format_converter()?;
    Ok(orchestrator)
}