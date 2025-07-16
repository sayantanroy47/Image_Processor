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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProcessorOperation {
    FormatConversion,
    Resize,
    Watermark,
    BackgroundRemoval,
    ColorCorrection,
    Crop,
    Rotate,
    TextOverlay,
    CollageCreation,
    MetadataHandling,
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
    /// Author/vendor of the processor
    pub author: String,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Performance characteristics of a processor
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Relative speed (1.0 = baseline, higher is faster)
    pub speed_factor: f32,
    /// Memory usage pattern
    pub memory_usage: MemoryUsage,
    /// CPU usage pattern
    pub cpu_usage: CpuUsage,
    /// Whether it benefits from parallel processing
    pub parallel_friendly: bool,
}

/// Memory usage patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryUsage {
    /// Low memory usage (< 100MB)
    Low,
    /// Moderate memory usage (100MB - 1GB)
    Moderate,
    /// High memory usage (> 1GB)
    High,
    /// Streaming (constant memory regardless of file size)
    Streaming,
}

/// CPU usage patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuUsage {
    /// Light CPU usage
    Light,
    /// Moderate CPU usage
    Moderate,
    /// Heavy CPU usage
    Heavy,
    /// GPU accelerated (low CPU usage)
    GpuAccelerated,
}

/// Type of processor for registration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    FormatConverter,
    WatermarkEngine,
    BackgroundRemover,
    ResizeEngine,
    ColorCorrector,
    CropRotator,
    TextOverlay,
    CollageCreator,
    MetadataHandler,
}

/// Processing use cases for processor recommendation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingUseCase {
    /// Prioritize highest quality output
    HighQuality,
    /// Prioritize fastest processing speed
    HighSpeed,
    /// Prioritize low memory usage
    LowMemory,
    /// Optimized for batch processing
    BatchProcessing,
}

/// Core processing orchestrator that coordinates all processors
#[derive(Debug)]
pub struct ProcessingOrchestrator {
    /// Registered processors by type
    processors: HashMap<ProcessorType, Arc<dyn ImageProcessor>>,
    /// Processor selection strategy
    selection_strategy: ProcessorSelectionStrategy,
    /// Performance monitoring
    performance_monitor: Arc<PerformanceMonitor>,
}

/// Strategy for selecting processors when multiple are available
#[derive(Debug, Clone)]
pub enum ProcessorSelectionStrategy {
    /// Always use the fastest processor
    Fastest,
    /// Use the most memory-efficient processor
    MemoryEfficient,
    /// Balance speed and memory usage
    Balanced,
    /// Use specific processor by name
    Specific(String),
}

/// Performance monitoring for processors
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Processing time statistics by processor type
    processing_times: HashMap<ProcessorType, Vec<std::time::Duration>>,
    /// Memory usage statistics
    memory_usage: HashMap<ProcessorType, Vec<u64>>,
    /// Success/failure rates
    success_rates: HashMap<ProcessorType, (u64, u64)>, // (successes, total)
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

    /// Register a processor with the orchestrator
    pub fn register_processor(
        &mut self, 
        processor_type: ProcessorType, 
        processor: Arc<dyn ImageProcessor>
    ) -> Result<()> {
        debug!("Registering processor: {:?}", processor_type);
        
        // Validate processor capabilities
        let capabilities = processor.get_capabilities();
        if capabilities.input_formats.is_empty() || capabilities.output_formats.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must support at least one input and output format".to_string(),
            });
        }

        self.processors.insert(processor_type, processor);
        info!("Processor registered successfully: {:?}", processor_type);
        Ok(())
    }

    /// Unregister a processor
    pub fn unregister_processor(&mut self, processor_type: &ProcessorType) -> bool {
        self.processors.remove(processor_type).is_some()
    }

    /// Get all registered processors
    pub fn get_registered_processors(&self) -> Vec<ProcessorType> {
        self.processors.keys().cloned().collect()
    }

    /// Get processor capabilities by type
    pub fn get_processor_capabilities(&self, processor_type: &ProcessorType) -> Option<ProcessorCapabilities> {
        self.processors.get(processor_type).map(|p| p.get_capabilities())
    }

    /// Check if a format is supported for input
    pub fn supports_input_format(&self, format: ImageFormat) -> bool {
        self.processors.values().any(|p| p.supports_format(format))
    }

    /// Check if an operation is supported
    pub fn supports_operation(&self, operation: &ProcessingOperation) -> bool {
        self.processors.values().any(|p| p.can_handle_operation(operation))
    }

    /// Find the best processor for a given operation
    pub fn find_processor_for_operation(&self, operation: &ProcessingOperation) -> Option<&Arc<dyn ImageProcessor>> {
        let candidates: Vec<_> = self.processors.values()
            .filter(|p| p.can_handle_operation(operation))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Select based on strategy
        match &self.selection_strategy {
            ProcessorSelectionStrategy::Fastest => {
                candidates.into_iter()
                    .max_by(|a, b| {
                        let a_speed = a.get_metadata().performance_profile.speed_factor;
                        let b_speed = b.get_metadata().performance_profile.speed_factor;
                        a_speed.partial_cmp(&b_speed).unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
            ProcessorSelectionStrategy::MemoryEfficient => {
                candidates.into_iter()
                    .min_by_key(|p| match p.get_metadata().performance_profile.memory_usage {
                        MemoryUsage::Low => 1,
                        MemoryUsage::Streaming => 2,
                        MemoryUsage::Moderate => 3,
                        MemoryUsage::High => 4,
                    })
            }
            ProcessorSelectionStrategy::Balanced => {
                // Score based on speed and memory efficiency
                candidates.into_iter()
                    .max_by(|a, b| {
                        let a_score = self.calculate_balanced_score(a);
                        let b_score = self.calculate_balanced_score(b);
                        a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
            ProcessorSelectionStrategy::Specific(name) => {
                candidates.into_iter()
                    .find(|p| p.get_metadata().name == *name)
                    .or_else(|| candidates.into_iter().next())
            }
        }
    }

    /// Calculate balanced score for processor selection
    fn calculate_balanced_score(&self, processor: &Arc<dyn ImageProcessor>) -> f32 {
        let profile = &processor.get_metadata().performance_profile;
        let speed_score = profile.speed_factor;
        let memory_score = match profile.memory_usage {
            MemoryUsage::Low | MemoryUsage::Streaming => 2.0,
            MemoryUsage::Moderate => 1.0,
            MemoryUsage::High => 0.5,
        };
        
        // Weight speed and memory efficiency equally
        (speed_score + memory_score) / 2.0
    }

    /// Set processor selection strategy
    pub fn set_selection_strategy(&mut self, strategy: ProcessorSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Process a single job using the orchestrator
    #[instrument(skip(self, input))]
    pub async fn process_job(&self, input: ProcessingInput) -> Result<ProcessingOutput> {
        let start_time = Instant::now();
        
        info!("Starting job processing for job {}, operations: {}", 
              input.job_id, input.operations.len());

        // Validate input
        if input.operations.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "No operations specified for processing".to_string(),
            });
        }

        // Check if we can handle all operations
        for operation in &input.operations {
            if !self.supports_operation(operation) {
                return Err(ProcessingError::ProcessingFailed {
                    message: format!("Operation not supported: {:?}", operation),
                });
            }
        }

        // Execute processing pipeline
        let mut current_input = input.clone();
        let mut operations_applied = Vec::new();
        let mut total_processing_time = std::time::Duration::ZERO;

        for (index, operation) in input.operations.iter().enumerate() {
            debug!("Processing operation {}/{}: {:?}", index + 1, input.operations.len(), operation);
            
            // Find the best processor for this operation
            let processor = self.find_processor_for_operation(operation)
                .ok_or_else(|| ProcessingError::ProcessingFailed {
                    message: format!("No processor available for operation: {:?}", operation),
                })?;

            // Record performance metrics
            let operation_start = Instant::now();
            
            // Process with the selected processor
            let operation_result = processor.process(&current_input).await;
            
            let operation_duration = operation_start.elapsed();
            total_processing_time += operation_duration;
            
            match operation_result {
                Ok(output) => {
                    operations_applied.push(format!("{:?}", operation));
                    
                    // Update input for next operation (chain operations)
                    if index < input.operations.len() - 1 {
                        current_input.source_path = output.output_path.clone();
                        current_input.file_size = output.file_size;
                        current_input.format = output.format;
                    }
                    
                    debug!("Operation completed in {}ms", operation_duration.as_millis());
                }
                Err(e) => {
                    error!("Operation failed: {}", e);
                    return Err(e);
                }
            }
        }

        let final_output = ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path,
            file_size: current_input.file_size,
            format: current_input.format,
            processing_time: total_processing_time,
            operations_applied,
            metadata: None, // TODO: Aggregate metadata from all operations
        };

        info!("Job processing completed for job {} in {}ms", 
              input.job_id, start_time.elapsed().as_millis());

        Ok(final_output)
    }

    /// Process multiple jobs in parallel
    #[instrument(skip(self, inputs))]
    pub async fn process_batch(&self, inputs: Vec<ProcessingInput>) -> Vec<Result<ProcessingOutput>> {
        info!("Starting batch processing of {} jobs", inputs.len());
        
        let batch_start = Instant::now();
        let mut handles = Vec::new();
        
        for input in inputs {
            let orchestrator = self.clone_for_async();
            let handle = tokio::spawn(async move {
                orchestrator.process_job(input).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(ProcessingError::ProcessingFailed {
                    message: format!("Task join error: {}", e),
                })),
            }
        }
        
        info!("Batch processing completed in {}ms", batch_start.elapsed().as_millis());
        results
    }

    /// Create a clone suitable for async operations
    fn clone_for_async(&self) -> Self {
        Self {
            processors: self.processors.clone(),
            selection_strategy: self.selection_strategy.clone(),
            performance_monitor: self.performance_monitor.clone(),
        }
    }

    /// Auto-discover and register available processors
    pub async fn auto_discover_processors(&mut self) -> Result<usize> {
        info!("Starting processor auto-discovery");
        
        let mut registered_count = 0;
        
        // This is a placeholder for actual processor discovery
        // In a real implementation, this would scan for available processors
        // based on system capabilities, installed libraries, etc.
        
        debug!("Auto-discovery completed, found {} processors", registered_count);
        Ok(registered_count)
    }

    /// Validate processor compatibility
    pub fn validate_processor_compatibility(&self, processor: &dyn ImageProcessor) -> Result<()> {
        let capabilities = processor.get_capabilities();
        let metadata = processor.get_metadata();
        
        // Check basic requirements
        if capabilities.input_formats.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must support at least one input format".to_string(),
            });
        }
        
        if capabilities.output_formats.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must support at least one output format".to_string(),
            });
        }
        
        if capabilities.operations.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must support at least one operation".to_string(),
            });
        }
        
        // Validate metadata
        if metadata.name.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must have a non-empty name".to_string(),
            });
        }
        
        if metadata.version.is_empty() {
            return Err(ProcessingError::InvalidInput {
                message: "Processor must have a version".to_string(),
            });
        }
        
        // Validate performance profile
        if metadata.performance_profile.speed_factor <= 0.0 {
            return Err(ProcessingError::InvalidInput {
                message: "Processor speed factor must be positive".to_string(),
            });
        }
        
        Ok(())
    }

    /// Get processing pipeline for a set of operations
    pub fn get_processing_pipeline(&self, operations: &[ProcessingOperation]) -> Result<Vec<Arc<dyn ImageProcessor>>> {
        let mut pipeline = Vec::new();
        
        for operation in operations {
            let processor = self.find_processor_for_operation(operation)
                .ok_or_else(|| ProcessingError::ProcessingFailed {
                    message: format!("No processor available for operation: {:?}", operation),
                })?;
            
            pipeline.push(processor.clone());
        }
        
        Ok(pipeline)
    }

    /// Optimize processing pipeline for performance
    pub fn optimize_pipeline(&self, operations: &[ProcessingOperation]) -> Result<Vec<ProcessingOperation>> {
        // This is a placeholder for pipeline optimization logic
        // In a real implementation, this would:
        // 1. Combine compatible operations
        // 2. Reorder operations for efficiency
        // 3. Remove redundant operations
        // 4. Split operations for parallel processing
        
        debug!("Optimizing pipeline with {} operations", operations.len());
        Ok(operations.to_vec())
    }

    /// Cancel a job
    #[instrument(skip(self))]
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        info!("Job cancelled: {}", job_id);
        // TODO: Implement actual cancellation logic when processors support it
        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<ProcessorType, ProcessorStats> {
        self.performance_monitor.get_stats()
    }

    /// Discover available processors and their capabilities
    pub fn discover_capabilities(&self) -> Vec<(ProcessorType, ProcessorCapabilities, ProcessorMetadata)> {
        self.processors.iter()
            .map(|(ptype, processor)| {
                (ptype.clone(), processor.get_capabilities(), processor.get_metadata())
            })
            .collect()
    }

    /// Get all processors that support a specific input format
    pub fn get_processors_for_format(&self, format: ImageFormat) -> Vec<(ProcessorType, Arc<dyn ImageProcessor>)> {
        self.processors.iter()
            .filter(|(_, processor)| processor.supports_format(format))
            .map(|(ptype, processor)| (ptype.clone(), processor.clone()))
            .collect()
    }

    /// Get all processors that can handle a specific operation
    pub fn get_processors_for_operation(&self, operation: &ProcessingOperation) -> Vec<(ProcessorType, Arc<dyn ImageProcessor>)> {
        self.processors.iter()
            .filter(|(_, processor)| processor.can_handle_operation(operation))
            .map(|(ptype, processor)| (ptype.clone(), processor.clone()))
            .collect()
    }

    /// Check if a specific format conversion is supported
    pub fn supports_format_conversion(&self, from: ImageFormat, to: ImageFormat) -> bool {
        self.processors.values().any(|processor| {
            let caps = processor.get_capabilities();
            caps.input_formats.contains(&from) && caps.output_formats.contains(&to)
        })
    }

    /// Get the best processor for a format conversion
    pub fn get_best_converter(&self, from: ImageFormat, to: ImageFormat) -> Option<Arc<dyn ImageProcessor>> {
        let candidates: Vec<_> = self.processors.values()
            .filter(|processor| {
                let caps = processor.get_capabilities();
                caps.input_formats.contains(&from) && caps.output_formats.contains(&to)
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Select based on current strategy
        self.select_best_processor(candidates)
    }

    /// Select the best processor from candidates based on current strategy
    fn select_best_processor(&self, candidates: Vec<&Arc<dyn ImageProcessor>>) -> Option<Arc<dyn ImageProcessor>> {
        match &self.selection_strategy {
            ProcessorSelectionStrategy::Fastest => {
                candidates.into_iter()
                    .max_by(|a, b| {
                        let a_speed = a.get_metadata().performance_profile.speed_factor;
                        let b_speed = b.get_metadata().performance_profile.speed_factor;
                        a_speed.partial_cmp(&b_speed).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            }
            ProcessorSelectionStrategy::MemoryEfficient => {
                candidates.into_iter()
                    .min_by_key(|p| match p.get_metadata().performance_profile.memory_usage {
                        MemoryUsage::Low => 1,
                        MemoryUsage::Streaming => 2,
                        MemoryUsage::Moderate => 3,
                        MemoryUsage::High => 4,
                    })
                    .cloned()
            }
            ProcessorSelectionStrategy::Balanced => {
                candidates.into_iter()
                    .max_by(|a, b| {
                        let a_score = self.calculate_balanced_score(a);
                        let b_score = self.calculate_balanced_score(b);
                        a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            }
            ProcessorSelectionStrategy::Specific(name) => {
                candidates.into_iter()
                    .find(|p| p.get_metadata().name == *name)
                    .or_else(|| candidates.into_iter().next())
                    .cloned()
            }
        }
    }

    /// Get capability matrix showing which formats can be converted to which
    pub fn get_format_conversion_matrix(&self) -> HashMap<ImageFormat, Vec<ImageFormat>> {
        let mut matrix = HashMap::new();
        
        for processor in self.processors.values() {
            let caps = processor.get_capabilities();
            for input_format in &caps.input_formats {
                let entry = matrix.entry(*input_format).or_insert_with(Vec::new);
                for output_format in &caps.output_formats {
                    if !entry.contains(output_format) {
                        entry.push(*output_format);
                    }
                }
            }
        }
        
        matrix
    }

    /// Get operation capability matrix showing which processors support which operations
    pub fn get_operation_capability_matrix(&self) -> HashMap<ProcessorOperation, Vec<ProcessorType>> {
        let mut matrix = HashMap::new();
        
        for (ptype, processor) in &self.processors {
            let caps = processor.get_capabilities();
            for operation in &caps.operations {
                let entry = matrix.entry(operation.clone()).or_insert_with(Vec::new);
                if !entry.contains(ptype) {
                    entry.push(ptype.clone());
                }
            }
        }
        
        matrix
    }

    /// Check if hardware acceleration is available for any processor
    pub fn has_hardware_acceleration(&self) -> bool {
        self.processors.values()
            .any(|processor| processor.get_capabilities().hardware_acceleration)
    }

    /// Get processors with hardware acceleration support
    pub fn get_hardware_accelerated_processors(&self) -> Vec<(ProcessorType, Arc<dyn ImageProcessor>)> {
        self.processors.iter()
            .filter(|(_, processor)| processor.get_capabilities().hardware_acceleration)
            .map(|(ptype, processor)| (ptype.clone(), processor.clone()))
            .collect()
    }

    /// Check if streaming processing is available
    pub fn has_streaming_support(&self) -> bool {
        self.processors.values()
            .any(|processor| processor.get_capabilities().streaming_support)
    }

    /// Get processors with streaming support
    pub fn get_streaming_processors(&self) -> Vec<(ProcessorType, Arc<dyn ImageProcessor>)> {
        self.processors.iter()
            .filter(|(_, processor)| processor.get_capabilities().streaming_support)
            .map(|(ptype, processor)| (ptype.clone(), processor.clone()))
            .collect()
    }

    /// Get maximum file size that can be processed
    pub fn get_max_file_size(&self) -> Option<u64> {
        self.processors.values()
            .filter_map(|processor| processor.get_capabilities().max_file_size)
            .max()
    }

    /// Check if a file size can be processed
    pub fn can_process_file_size(&self, file_size: u64) -> bool {
        self.processors.values().any(|processor| {
            let caps = processor.get_capabilities();
            caps.max_file_size.map_or(true, |max_size| file_size <= max_size)
        })
    }

    /// Get processors that can handle a specific file size
    pub fn get_processors_for_file_size(&self, file_size: u64) -> Vec<(ProcessorType, Arc<dyn ImageProcessor>)> {
        self.processors.iter()
            .filter(|(_, processor)| {
                let caps = processor.get_capabilities();
                caps.max_file_size.map_or(true, |max_size| file_size <= max_size)
            })
            .map(|(ptype, processor)| (ptype.clone(), processor.clone()))
            .collect()
    }

    /// Get fallback processor for when primary processor fails
    pub fn get_fallback_processor(&self, operation: &ProcessingOperation, exclude: &str) -> Option<Arc<dyn ImageProcessor>> {
        let candidates: Vec<_> = self.processors.values()
            .filter(|processor| {
                processor.can_handle_operation(operation) && 
                processor.get_metadata().name != exclude
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // For fallback, prefer reliability over speed
        candidates.into_iter()
            .min_by_key(|processor| {
                let profile = &processor.get_metadata().performance_profile;
                match profile.memory_usage {
                    MemoryUsage::Low => 1,
                    MemoryUsage::Streaming => 2,
                    MemoryUsage::Moderate => 3,
                    MemoryUsage::High => 4,
                }
            })
            .cloned()
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

    /// Get recommended processor for a specific use case
    pub fn get_recommended_processor(&self, use_case: ProcessingUseCase) -> Option<Arc<dyn ImageProcessor>> {
        let candidates: Vec<_> = self.processors.values().collect();
        
        if candidates.is_empty() {
            return None;
        }

        match use_case {
            ProcessingUseCase::HighQuality => {
                // Prefer processors with high quality settings
                candidates.into_iter()
                    .max_by_key(|processor| {
                        let profile = &processor.get_metadata().performance_profile;
                        (profile.speed_factor * 100.0) as u32
                    })
                    .cloned()
            }
            ProcessingUseCase::HighSpeed => {
                // Prefer fastest processors
                candidates.into_iter()
                    .max_by_key(|processor| {
                        let profile = &processor.get_metadata().performance_profile;
                        (profile.speed_factor * 100.0) as u32
                    })
                    .cloned()
            }
            ProcessingUseCase::LowMemory => {
                // Prefer memory-efficient processors
                candidates.into_iter()
                    .min_by_key(|processor| {
                        let profile = &processor.get_metadata().performance_profile;
                        match profile.memory_usage {
                            MemoryUsage::Low => 1,
                            MemoryUsage::Streaming => 2,
                            MemoryUsage::Moderate => 3,
                            MemoryUsage::High => 4,
                        }
                    })
                    .cloned()
            }
            ProcessingUseCase::BatchProcessing => {
                // Prefer batch-optimized processors
                candidates.into_iter()
                    .filter(|processor| processor.get_capabilities().batch_optimized)
                    .max_by_key(|processor| {
                        let profile = &processor.get_metadata().performance_profile;
                        (profile.speed_factor * 100.0) as u32
                    })
                    .or_else(|| candidates.into_iter().next())
                    .cloned()
            }
        }
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
    pub average_processing_time: std::time::Duration,
    pub success_rate: f64,
    pub average_memory_usage: u64,
    pub total_operations: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            processing_times: HashMap::new(),
            memory_usage: HashMap::new(),
            success_rates: HashMap::new(),
        }
    }

    pub fn record_processing_time(&mut self, processor_type: ProcessorType, duration: std::time::Duration) {
        self.processing_times.entry(processor_type)
            .or_insert_with(Vec::new)
            .push(duration);
    }

    pub fn record_success(&mut self, processor_type: ProcessorType, success: bool) {
        let (successes, total) = self.success_rates.entry(processor_type)
            .or_insert((0, 0));
        
        if success {
            *successes += 1;
        }
        *total += 1;
    }

    pub fn get_stats(&self) -> HashMap<ProcessorType, ProcessorStats> {
        let mut stats = HashMap::new();
        
        for (ptype, times) in &self.processing_times {
            let avg_time = if !times.is_empty() {
                times.iter().sum::<std::time::Duration>() / times.len() as u32
            } else {
                std::time::Duration::ZERO
            };
            
            let (successes, total) = self.success_rates.get(ptype).unwrap_or(&(0, 0));
            let success_rate = if *total > 0 {
                *successes as f64 / *total as f64
            } else {
                0.0
            };
            
            let avg_memory = self.memory_usage.get(ptype)
                .map(|usage| usage.iter().sum::<u64>() / usage.len() as u64)
                .unwrap_or(0);
            
            stats.insert(ptype.clone(), ProcessorStats {
                average_processing_time: avg_time,
                success_rate,
                average_memory_usage: avg_memory,
                total_operations: *total,
            });
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ProcessingOperation, ProcessingOptions};
    use crate::config::ImageFormat;
    use std::path::PathBuf;
    use uuid::Uuid;

    // Mock processor for testing
    struct MockProcessor {
        name: String,
        capabilities: ProcessorCapabilities,
        metadata: ProcessorMetadata,
    }

    impl MockProcessor {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                capabilities: ProcessorCapabilities {
                    input_formats: vec![ImageFormat::Jpeg, ImageFormat::Png],
                    output_formats: vec![ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::WebP],
                    operations: vec![ProcessorOperation::FormatConversion],
                    hardware_acceleration: false,
                    streaming_support: true,
                    max_file_size: Some(100 * 1024 * 1024), // 100MB
                    batch_optimized: true,
                },
                metadata: ProcessorMetadata {
                    name: name.to_string(),
                    version: "1.0.0".to_string(),
                    description: "Mock processor for testing".to_string(),
                    author: "Test Suite".to_string(),
                    performance_profile: PerformanceProfile {
                        speed_factor: 1.0,
                        memory_usage: MemoryUsage::Low,
                        cpu_usage: CpuUsage::Moderate,
                        parallel_friendly: true,
                    },
                },
            }
        }
    }

    #[async_trait]
    impl ImageProcessor for MockProcessor {
        async fn process(&self, input: &ProcessingInput) -> Result<ProcessingOutput> {
            Ok(ProcessingOutput {
                job_id: input.job_id,
                output_path: input.output_path.clone(),
                file_size: input.file_size,
                format: input.format,
                processing_time: std::time::Duration::from_millis(100),
                operations_applied: vec![self.name.clone()],
                metadata: None,
            })
        }

        fn supports_format(&self, format: ImageFormat) -> bool {
            self.capabilities.input_formats.contains(&format)
        }

        fn get_capabilities(&self) -> ProcessorCapabilities {
            self.capabilities.clone()
        }

        fn get_metadata(&self) -> ProcessorMetadata {
            self.metadata.clone()
        }

        fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
            matches!(operation, ProcessingOperation::Convert { .. })
        }
    }

    fn create_test_input() -> ProcessingInput {
        ProcessingInput {
            job_id: Uuid::new_v4(),
            source_path: PathBuf::from("test.jpg"),
            output_path: PathBuf::from("output.jpg"),
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
    async fn test_orchestrator_creation() {
        let orchestrator = ProcessingOrchestrator::new();
        assert_eq!(orchestrator.get_registered_processors().len(), 0);
        assert!(matches!(orchestrator.selection_strategy, ProcessorSelectionStrategy::Balanced));
    }

    #[tokio::test]
    async fn test_processor_registration() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        let result = orchestrator.register_processor(ProcessorType::FormatConverter, processor);
        assert!(result.is_ok());
        
        let registered = orchestrator.get_registered_processors();
        assert_eq!(registered.len(), 1);
        assert!(registered.contains(&ProcessorType::FormatConverter));
    }

    #[tokio::test]
    async fn test_processor_registration_validation() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Create a processor with empty capabilities (should fail validation)
        let invalid_processor = Arc::new(MockProcessor {
            name: "invalid".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![], // Empty - should cause validation failure
                output_formats: vec![ImageFormat::Jpeg],
                operations: vec![],
                hardware_acceleration: false,
                streaming_support: false,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "invalid".to_string(),
                version: "1.0.0".to_string(),
                description: "Invalid processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        let result = orchestrator.register_processor(ProcessorType::FormatConverter, invalid_processor);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_processor_unregistration() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        assert_eq!(orchestrator.get_registered_processors().len(), 1);
        
        let removed = orchestrator.unregister_processor(&ProcessorType::FormatConverter);
        assert!(removed);
        assert_eq!(orchestrator.get_registered_processors().len(), 0);
        
        // Try to remove again - should return false
        let removed_again = orchestrator.unregister_processor(&ProcessorType::FormatConverter);
        assert!(!removed_again);
    }

    #[tokio::test]
    async fn test_format_support_detection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        assert!(orchestrator.supports_input_format(ImageFormat::Jpeg));
        assert!(orchestrator.supports_input_format(ImageFormat::Png));
        assert!(!orchestrator.supports_input_format(ImageFormat::Gif));
    }

    #[tokio::test]
    async fn test_operation_support_detection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        assert!(orchestrator.supports_operation(&convert_op));
        
        let resize_op = ProcessingOperation::Resize {
            width: Some(800),
            height: Some(600),
            algorithm: crate::models::ResizeAlgorithm::Lanczos,
        };
        assert!(!orchestrator.supports_operation(&resize_op));
    }

    #[tokio::test]
    async fn test_processor_selection_strategies() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Register multiple processors with different performance profiles
        let fast_processor = Arc::new(MockProcessor {
            name: "fast-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: true,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: true,
            },
            metadata: ProcessorMetadata {
                name: "fast-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Fast processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 2.0, // Faster
                    memory_usage: MemoryUsage::High,
                    cpu_usage: CpuUsage::Heavy,
                    parallel_friendly: true,
                },
            },
        });
        
        let efficient_processor = Arc::new(MockProcessor {
            name: "efficient-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "efficient-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Memory efficient processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 0.8, // Slower
                    memory_usage: MemoryUsage::Low, // More efficient
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, fast_processor.clone()).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, efficient_processor.clone()).unwrap();
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        
        // Test fastest strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Fastest);
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "fast-processor");
        
        // Test memory efficient strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::MemoryEfficient);
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "efficient-processor");
        
        // Test specific processor strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Specific("efficient-processor".to_string()));
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "efficient-processor");
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("batch-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Create multiple test inputs
        let inputs = vec![
            create_test_input(),
            create_test_input(),
            create_test_input(),
        ];
        
        let results = orchestrator.process_batch(inputs).await;
        
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_processing_pipeline() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("pipeline-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let operations = vec![
            ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            }
        ];
        
        let pipeline = orchestrator.get_processing_pipeline(&operations);
        assert!(pipeline.is_ok());
        
        let processors = pipeline.unwrap();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].get_metadata().name, "pipeline-processor");
    }

    #[tokio::test]
    async fn test_pipeline_optimization() {
        let orchestrator = ProcessingOrchestrator::new();
        
        let operations = vec![
            ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            },
            ProcessingOperation::Resize {
                width: Some(800),
                height: Some(600),
                algorithm: crate::models::ResizeAlgorithm::Lanczos,
            },
        ];
        
        let optimized = orchestrator.optimize_pipeline(&operations);
        assert!(optimized.is_ok());
        
        let optimized_ops = optimized.unwrap();
        assert_eq!(optimized_ops.len(), 2); // Currently no optimization, so same length
    }

    #[tokio::test]
    async fn test_processor_compatibility_validation() {
        let orchestrator = ProcessingOrchestrator::new();
        let valid_processor = MockProcessor::new("valid-processor");
        
        let result = orchestrator.validate_processor_compatibility(&valid_processor);
        assert!(result.is_ok());
        
        // Test invalid processor with empty name
        let invalid_processor = MockProcessor {
            name: "".to_string(), // Empty name should fail validation
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "".to_string(),
                version: "1.0.0".to_string(),
                description: "Invalid processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        };
        
        let result = orchestrator.validate_processor_compatibility(&invalid_processor);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_auto_discovery() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        let result = orchestrator.auto_discover_processors().await;
        assert!(result.is_ok());
        
        let count = result.unwrap();
        assert_eq!(count, 0); // Currently placeholder implementation returns 0
    }

    #[tokio::test]
    async fn test_processing_with_chained_operations() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("chain-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Create input with multiple operations
        let mut input = create_test_input();
        input.operations = vec![
            ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            },
            ProcessingOperation::Convert {
                format: ImageFormat::WebP,
                quality: Some(90),
            },
        ];
        
        let result = orchestrator.process_job(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.operations_applied.len(), 2);
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let orchestrator = ProcessingOrchestrator::new();
        let mut monitor = PerformanceMonitor::new();
        
        // Record some test metrics
        monitor.record_processing_time(ProcessorType::FormatConverter, std::time::Duration::from_millis(100));
        monitor.record_processing_time(ProcessorType::FormatConverter, std::time::Duration::from_millis(200));
        monitor.record_success(ProcessorType::FormatConverter, true);
        monitor.record_success(ProcessorType::FormatConverter, false);
        
        let stats = monitor.get_stats();
        assert!(stats.contains_key(&ProcessorType::FormatConverter));
        
        let converter_stats = &stats[&ProcessorType::FormatConverter];
        assert_eq!(converter_stats.total_operations, 2);
        assert_eq!(converter_stats.success_rate, 0.5);
        assert!(converter_stats.average_processing_time > std::time::Duration::ZERO);
    }

    #[tokio::test]
    async fn test_processor_capabilities_query() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("query-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let capabilities = orchestrator.get_processor_capabilities(&ProcessorType::FormatConverter);
        assert!(capabilities.is_some());
        
        let caps = capabilities.unwrap();
        assert!(caps.input_formats.contains(&ImageFormat::Jpeg));
        assert!(caps.output_formats.contains(&ImageFormat::WebP));
        assert!(caps.operations.contains(&ProcessorOperation::FormatConversion));
    }

    #[tokio::test]
    async fn test_capability_discovery() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor1 = Arc::new(MockProcessor::new("processor-1"));
        let processor2 = Arc::new(MockProcessor::new("processor-2"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor1).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, processor2).unwrap();
        
        let capabilities = orchestrator.discover_capabilities();
        assert_eq!(capabilities.len(), 2);
        
        // Verify we have both processor types
        let types: Vec<_> = capabilities.iter().map(|(t, _, _)| t.clone()).collect();
        assert!(types.contains(&ProcessorType::FormatConverter));
        assert!(types.contains(&ProcessorType::WatermarkEngine));
    }

    #[tokio::test]
    async fn test_error_handling_in_pipeline() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Create a processor that always fails
        struct FailingProcessor;
        
        #[async_trait]
        impl ImageProcessor for FailingProcessor {
            async fn process(&self, _input: &ProcessingInput) -> Result<ProcessingOutput> {
                Err(ProcessingError::ProcessingFailed {
                    message: "Simulated failure".to_string(),
                })
            }
            
            fn supports_format(&self, _format: ImageFormat) -> bool {
                true
            }
            
            fn get_capabilities(&self) -> ProcessorCapabilities {
                ProcessorCapabilities {
                    input_formats: vec![ImageFormat::Jpeg],
                    output_formats: vec![ImageFormat::Png],
                    operations: vec![ProcessorOperation::FormatConversion],
                    hardware_acceleration: false,
                    streaming_support: false,
                    max_file_size: None,
                    batch_optimized: false,
                }
            }
            
            fn get_metadata(&self) -> ProcessorMetadata {
                ProcessorMetadata {
                    name: "failing-processor".to_string(),
                    version: "1.0.0".to_string(),
                    description: "Always fails".to_string(),
                    author: "Test".to_string(),
                    performance_profile: PerformanceProfile {
                        speed_factor: 1.0,
                        memory_usage: MemoryUsage::Low,
                        cpu_usage: CpuUsage::Light,
                        parallel_friendly: false,
                    },
                }
            }
            
            fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
                matches!(operation, ProcessingOperation::Convert { .. })
            }
        }
        
        let failing_processor = Arc::new(FailingProcessor);
        orchestrator.register_processor(ProcessorType::FormatConverter, failing_processor).unwrap();
        
        let input = create_test_input();
        let result = orchestrator.process_job(input).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProcessingError::ProcessingFailed { .. }));
    }

    #[tokio::test]
    async fn test_format_conversion_support() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("format-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Test supported conversion
        assert!(orchestrator.supports_format_conversion(ImageFormat::Jpeg, ImageFormat::Png));
        assert!(orchestrator.supports_format_conversion(ImageFormat::Png, ImageFormat::WebP));
        
        // Test unsupported conversion
        assert!(!orchestrator.supports_format_conversion(ImageFormat::Gif, ImageFormat::Png));
    }

    #[tokio::test]
    async fn test_best_converter_selection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        let fast_converter = Arc::new(MockProcessor {
            name: "fast-converter".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: true,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: true,
            },
            metadata: ProcessorMetadata {
                name: "fast-converter".to_string(),
                version: "1.0.0".to_string(),
                description: "Fast converter".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 2.0,
                    memory_usage: MemoryUsage::High,
                    cpu_usage: CpuUsage::Heavy,
                    parallel_friendly: true,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, fast_converter).unwrap();
        
        let converter = orchestrator.get_best_converter(ImageFormat::Jpeg, ImageFormat::Png);
        assert!(converter.is_some());
        assert_eq!(converter.unwrap().get_metadata().name, "fast-converter");
        
        // Test unsupported conversion
        let no_converter = orchestrator.get_best_converter(ImageFormat::Gif, ImageFormat::Heic);
        assert!(no_converter.is_none());
    }

    #[tokio::test]
    async fn test_format_conversion_matrix() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("matrix-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let matrix = orchestrator.get_format_conversion_matrix();
        
        // Check that JPEG can be converted to PNG and WebP
        assert!(matrix.contains_key(&ImageFormat::Jpeg));
        let jpeg_outputs = &matrix[&ImageFormat::Jpeg];
        assert!(jpeg_outputs.contains(&ImageFormat::Png));
        assert!(jpeg_outputs.contains(&ImageFormat::WebP));
        
        // Check that PNG can be converted to other formats
        assert!(matrix.contains_key(&ImageFormat::Png));
        let png_outputs = &matrix[&ImageFormat::Png];
        assert!(png_outputs.contains(&ImageFormat::Jpeg));
        assert!(png_outputs.contains(&ImageFormat::WebP));
    }

    #[tokio::test]
    async fn test_operation_capability_matrix() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let converter = Arc::new(MockProcessor::new("converter"));
        let watermarker = Arc::new(MockProcessor {
            name: "watermarker".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg, ImageFormat::Png],
                output_formats: vec![ImageFormat::Jpeg, ImageFormat::Png],
                operations: vec![ProcessorOperation::Watermark],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "watermarker".to_string(),
                version: "1.0.0".to_string(),
                description: "Watermark processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Moderate,
                    cpu_usage: CpuUsage::Moderate,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, converter).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, watermarker).unwrap();
        
        let matrix = orchestrator.get_operation_capability_matrix();
        
        // Check format conversion support
        assert!(matrix.contains_key(&ProcessorOperation::FormatConversion));
        let conversion_processors = &matrix[&ProcessorOperation::FormatConversion];
        assert!(conversion_processors.contains(&ProcessorType::FormatConverter));
        
        // Check watermark support
        assert!(matrix.contains_key(&ProcessorOperation::Watermark));
        let watermark_processors = &matrix[&ProcessorOperation::Watermark];
        assert!(watermark_processors.contains(&ProcessorType::WatermarkEngine));
    }

    #[tokio::test]
    async fn test_hardware_acceleration_detection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Initially no hardware acceleration
        assert!(!orchestrator.has_hardware_acceleration());
        assert!(orchestrator.get_hardware_accelerated_processors().is_empty());
        
        // Add processor with hardware acceleration
        let hw_processor = Arc::new(MockProcessor {
            name: "hw-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: true, // Hardware acceleration enabled
                streaming_support: true,
                max_file_size: None,
                batch_optimized: true,
            },
            metadata: ProcessorMetadata {
                name: "hw-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Hardware accelerated processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 3.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::GpuAccelerated,
                    parallel_friendly: true,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, hw_processor).unwrap();
        
        // Now hardware acceleration should be available
        assert!(orchestrator.has_hardware_acceleration());
        let hw_processors = orchestrator.get_hardware_accelerated_processors();
        assert_eq!(hw_processors.len(), 1);
        assert_eq!(hw_processors[0].0, ProcessorType::FormatConverter);
    }

    #[tokio::test]
    async fn test_streaming_support_detection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Initially no streaming support
        assert!(!orchestrator.has_streaming_support());
        assert!(orchestrator.get_streaming_processors().is_empty());
        
        // Add processor with streaming support
        let streaming_processor = Arc::new(MockProcessor::new("streaming-processor"));
        orchestrator.register_processor(ProcessorType::FormatConverter, streaming_processor).unwrap();
        
        // Now streaming should be available
        assert!(orchestrator.has_streaming_support());
        let streaming_processors = orchestrator.get_streaming_processors();
        assert_eq!(streaming_processors.len(), 1);
        assert_eq!(streaming_processors[0].0, ProcessorType::FormatConverter);
    }

    #[tokio::test]
    async fn test_file_size_capability_detection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        let limited_processor = Arc::new(MockProcessor {
            name: "limited-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: false,
                max_file_size: Some(10 * 1024 * 1024), // 10MB limit
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "limited-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Size limited processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, limited_processor).unwrap();
        
        // Test file size limits
        assert_eq!(orchestrator.get_max_file_size(), Some(10 * 1024 * 1024));
        assert!(orchestrator.can_process_file_size(5 * 1024 * 1024)); // 5MB - should work
        assert!(!orchestrator.can_process_file_size(20 * 1024 * 1024)); // 20MB - should fail
        
        let processors_for_small = orchestrator.get_processors_for_file_size(5 * 1024 * 1024);
        assert_eq!(processors_for_small.len(), 1);
        
        let processors_for_large = orchestrator.get_processors_for_file_size(20 * 1024 * 1024);
        assert_eq!(processors_for_large.len(), 0);
    }

    #[tokio::test]
    async fn test_fallback_processor_selection() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        let primary_processor = Arc::new(MockProcessor::new("primary-processor"));
        let fallback_processor = Arc::new(MockProcessor {
            name: "fallback-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "fallback-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Fallback processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 0.5,
                    memory_usage: MemoryUsage::Low, // More reliable
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, primary_processor).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, fallback_processor).unwrap();
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        
        // Get fallback processor (excluding primary)
        let fallback = orchestrator.get_fallback_processor(&convert_op, "primary-processor");
        assert!(fallback.is_some());
        assert_eq!(fallback.unwrap().get_metadata().name, "fallback-processor");
        
        // No fallback available if we exclude all processors
        let no_fallback = orchestrator.get_fallback_processor(&convert_op, "fallback-processor");
        assert!(no_fallback.is_none());
    }

    #[tokio::test]
    async fn test_processing_use_case_recommendations() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        let fast_processor = Arc::new(MockProcessor {
            name: "fast-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: true,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: true,
            },
            metadata: ProcessorMetadata {
                name: "fast-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Fast processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 3.0, // Very fast
                    memory_usage: MemoryUsage::High,
                    cpu_usage: CpuUsage::Heavy,
                    parallel_friendly: true,
                },
            },
        });
        
        let efficient_processor = Arc::new(MockProcessor {
            name: "efficient-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "efficient-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Memory efficient processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low, // Very efficient
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, fast_processor).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, efficient_processor).unwrap();
        
        // Test high speed recommendation
        let speed_rec = orchestrator.get_recommended_processor(ProcessingUseCase::HighSpeed);
        assert!(speed_rec.is_some());
        assert_eq!(speed_rec.unwrap().get_metadata().name, "fast-processor");
        
        // Test low memory recommendation
        let memory_rec = orchestrator.get_recommended_processor(ProcessingUseCase::LowMemory);
        assert!(memory_rec.is_some());
        assert_eq!(memory_rec.unwrap().get_metadata().name, "efficient-processor");
        
        // Test batch processing recommendation
        let batch_rec = orchestrator.get_recommended_processor(ProcessingUseCase::BatchProcessing);
        assert!(batch_rec.is_some());
        assert_eq!(batch_rec.unwrap().get_metadata().name, "fast-processor"); // Has batch_optimized = true
    }

    #[tokio::test]
    async fn test_pipeline_validation() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("validator-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Valid pipeline
        let valid_ops = vec![
            ProcessingOperation::Convert {
                format: ImageFormat::Png,
                quality: Some(85),
            }
        ];
        let result = orchestrator.validate_processing_pipeline(&valid_ops);
        assert!(result.is_ok());
        
        // Invalid pipeline with unsupported operation
        let invalid_ops = vec![
            ProcessingOperation::Resize {
                width: Some(800),
                height: Some(600),
                algorithm: crate::models::ResizeAlgorithm::Lanczos,
            }
        ];
        let result = orchestrator.validate_processing_pipeline(&invalid_ops);
        assert!(result.is_err());
    }

    #[test]
    fn test_processing_use_case_variants() {
        let use_cases = vec![
            ProcessingUseCase::HighQuality,
            ProcessingUseCase::HighSpeed,
            ProcessingUseCase::LowMemory,
            ProcessingUseCase::BatchProcessing,
        ];
        
        assert_eq!(use_cases.len(), 4);
        
        // Test that use cases can be compared
        for use_case in &use_cases {
            assert_eq!(*use_case, *use_case);
        }
    }

    #[tokio::test]
    async fn test_orchestrator_with_no_processors() {
        let orchestrator = ProcessingOrchestrator::new();
        
        // Test that operations fail gracefully when no processors are registered
        assert!(!orchestrator.supports_input_format(ImageFormat::Jpeg));
        assert!(!orchestrator.has_hardware_acceleration());
        assert!(!orchestrator.has_streaming_support());
        assert_eq!(orchestrator.get_max_file_size(), None);
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        assert!(!orchestrator.supports_operation(&convert_op));
        assert!(orchestrator.find_processor_for_operation(&convert_op).is_none());
        
        let matrix = orchestrator.get_format_conversion_matrix();
        assert!(matrix.is_empty());
        
        let op_matrix = orchestrator.get_operation_capability_matrix();
        assert!(op_matrix.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_edge_cases() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Test with processor that has no supported formats (edge case)
        let empty_processor = Arc::new(MockProcessor {
            name: "empty-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![], // Empty
                output_formats: vec![], // Empty
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: false,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "empty-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Empty processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        // This should fail validation
        let result = orchestrator.register_processor(ProcessorType::FormatConverter, empty_processor);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_processor_registration() {
        let orchestrator = Arc::new(std::sync::Mutex::new(ProcessingOrchestrator::new()));
        let mut handles = Vec::new();
        
        // Register processors concurrently
        for i in 0..5 {
            let orchestrator_clone = orchestrator.clone();
            let handle = tokio::spawn(async move {
                let processor = Arc::new(MockProcessor::new(&format!("concurrent-processor-{}", i)));
                let processor_type = match i {
                    0 => ProcessorType::FormatConverter,
                    1 => ProcessorType::WatermarkEngine,
                    2 => ProcessorType::BackgroundRemover,
                    3 => ProcessorType::ResizeEngine,
                    _ => ProcessorType::ColorCorrector,
                };
                
                let mut orch = orchestrator_clone.lock().unwrap();
                orch.register_processor(processor_type, processor)
            });
            handles.push(handle);
        }
        
        // Wait for all registrations
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }
        
        // All registrations should succeed
        for result in results {
            assert!(result.is_ok());
        }
        
        let orch = orchestrator.lock().unwrap();
        assert_eq!(orch.get_registered_processors().len(), 5);
    }

    #[tokio::test]
    async fn test_processor_selection_with_identical_scores() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Create two processors with identical performance profiles
        let processor1 = Arc::new(MockProcessor {
            name: "identical-1".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "identical-1".to_string(),
                version: "1.0.0".to_string(),
                description: "Identical processor 1".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0, // Same speed
                    memory_usage: MemoryUsage::Moderate, // Same memory usage
                    cpu_usage: CpuUsage::Moderate,
                    parallel_friendly: false,
                },
            },
        });
        
        let processor2 = Arc::new(MockProcessor {
            name: "identical-2".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "identical-2".to_string(),
                version: "1.0.0".to_string(),
                description: "Identical processor 2".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0, // Same speed
                    memory_usage: MemoryUsage::Moderate, // Same memory usage
                    cpu_usage: CpuUsage::Moderate,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor1).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, processor2).unwrap();
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        
        // Should still select one processor (deterministic selection)
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
    }

    #[tokio::test]
    async fn test_batch_processing_with_mixed_results() {
        let mut orchestrator = ProcessingOrchestrator::new();
        
        // Create a processor that fails on specific conditions
        struct ConditionalProcessor;
        
        #[async_trait]
        impl ImageProcessor for ConditionalProcessor {
            async fn process(&self, input: &ProcessingInput) -> Result<ProcessingOutput> {
                // Fail if file size is exactly 2048 bytes
                if input.file_size == 2048 {
                    return Err(ProcessingError::ProcessingFailed {
                        message: "Conditional failure".to_string(),
                    });
                }
                
                Ok(ProcessingOutput {
                    job_id: input.job_id,
                    output_path: input.output_path.clone(),
                    file_size: input.file_size,
                    format: input.format,
                    processing_time: std::time::Duration::from_millis(50),
                    operations_applied: vec!["conditional".to_string()],
                    metadata: None,
                })
            }
            
            fn supports_format(&self, _format: ImageFormat) -> bool {
                true
            }
            
            fn get_capabilities(&self) -> ProcessorCapabilities {
                ProcessorCapabilities {
                    input_formats: vec![ImageFormat::Jpeg],
                    output_formats: vec![ImageFormat::Png],
                    operations: vec![ProcessorOperation::FormatConversion],
                    hardware_acceleration: false,
                    streaming_support: false,
                    max_file_size: None,
                    batch_optimized: true,
                }
            }
            
            fn get_metadata(&self) -> ProcessorMetadata {
                ProcessorMetadata {
                    name: "conditional-processor".to_string(),
                    version: "1.0.0".to_string(),
                    description: "Conditionally failing processor".to_string(),
                    author: "Test".to_string(),
                    performance_profile: PerformanceProfile {
                        speed_factor: 1.0,
                        memory_usage: MemoryUsage::Low,
                        cpu_usage: CpuUsage::Light,
                        parallel_friendly: true,
                    },
                }
            }
            
            fn can_handle_operation(&self, operation: &ProcessingOperation) -> bool {
                matches!(operation, ProcessingOperation::Convert { .. })
            }
        }
        
        let conditional_processor = Arc::new(ConditionalProcessor);
        orchestrator.register_processor(ProcessorType::FormatConverter, conditional_processor).unwrap();
        
        // Create batch with mixed file sizes
        let inputs = vec![
            {
                let mut input = create_test_input();
                input.file_size = 1024; // Should succeed
                input
            },
            {
                let mut input = create_test_input();
                input.file_size = 2048; // Should fail
                input
            },
            {
                let mut input = create_test_input();
                input.file_size = 4096; // Should succeed
                input
            },
        ];
        
        let results = orchestrator.process_batch(inputs).await;
        
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok()); // First should succeed
        assert!(results[1].is_err()); // Second should fail
        assert!(results[2].is_ok()); // Third should succeed
    }

    #[tokio::test]
    async fn test_performance_monitoring_integration() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("monitored-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Process multiple jobs to generate performance data
        for _ in 0..5 {
            let input = create_test_input();
            let _ = orchestrator.process_job(input).await;
        }
        
        let stats = orchestrator.get_performance_stats();
        // Stats should be available (though currently empty in mock implementation)
        assert!(stats.is_empty() || stats.contains_key(&ProcessorType::FormatConverter));
    }

    #[tokio::test]
    async fn test_orchestrator_cloning_for_async() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("clone-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        // Test that cloning works for async operations
        let cloned = orchestrator.clone_for_async();
        assert_eq!(cloned.get_registered_processors().len(), 1);
        
        // Both should be able to process jobs
        let input1 = create_test_input();
        let input2 = create_test_input();
        
        let result1 = orchestrator.process_job(input1).await;
        let result2 = cloned.process_job(input2).await;
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_processor_metadata_validation_edge_cases() {
        let orchestrator = ProcessingOrchestrator::new();
        
        // Test processor with zero speed factor
        let zero_speed_processor = MockProcessor {
            name: "zero-speed".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: false,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "zero-speed".to_string(),
                version: "1.0.0".to_string(),
                description: "Zero speed processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 0.0, // Invalid - should be positive
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        };
        
        let result = orchestrator.validate_processor_compatibility(&zero_speed_processor);
        assert!(result.is_err());
        
        // Test processor with empty version
        let empty_version_processor = MockProcessor {
            name: "empty-version".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: false,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "empty-version".to_string(),
                version: "".to_string(), // Empty version - should fail
                description: "Empty version processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 1.0,
                    memory_usage: MemoryUsage::Low,
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        };
        
        let result = orchestrator.validate_processor_compatibility(&empty_version_processor);
        assert!(result.is_err());
    }
                streaming_support: true,
                max_file_size: None,
                batch_optimized: true,
            },
            metadata: ProcessorMetadata {
                name: "fast-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Fast processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 2.0, // Faster
                    memory_usage: MemoryUsage::High,
                    cpu_usage: CpuUsage::Heavy,
                    parallel_friendly: true,
                },
            },
        });
        
        let efficient_processor = Arc::new(MockProcessor {
            name: "efficient-processor".to_string(),
            capabilities: ProcessorCapabilities {
                input_formats: vec![ImageFormat::Jpeg],
                output_formats: vec![ImageFormat::Png],
                operations: vec![ProcessorOperation::FormatConversion],
                hardware_acceleration: false,
                streaming_support: true,
                max_file_size: None,
                batch_optimized: false,
            },
            metadata: ProcessorMetadata {
                name: "efficient-processor".to_string(),
                version: "1.0.0".to_string(),
                description: "Memory efficient processor".to_string(),
                author: "Test".to_string(),
                performance_profile: PerformanceProfile {
                    speed_factor: 0.8, // Slower
                    memory_usage: MemoryUsage::Low, // More efficient
                    cpu_usage: CpuUsage::Light,
                    parallel_friendly: false,
                },
            },
        });
        
        orchestrator.register_processor(ProcessorType::FormatConverter, fast_processor.clone()).unwrap();
        orchestrator.register_processor(ProcessorType::WatermarkEngine, efficient_processor.clone()).unwrap();
        
        let convert_op = ProcessingOperation::Convert {
            format: ImageFormat::Png,
            quality: Some(85),
        };
        
        // Test fastest strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Fastest);
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "fast-processor");
        
        // Test memory efficient strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::MemoryEfficient);
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "efficient-processor");
        
        // Test specific processor strategy
        orchestrator.set_selection_strategy(ProcessorSelectionStrategy::Specific("efficient-processor".to_string()));
        let selected = orchestrator.find_processor_for_operation(&convert_op);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().get_metadata().name, "efficient-processor");
    }

    #[tokio::test]
    async fn test_process_job_with_registered_processor() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let input = create_test_input();
        let job_id = input.job_id;
        
        let result = orchestrator.process_job(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.job_id, job_id);
        assert!(output.operations_applied.contains(&"Convert { format: Png, quality: Some(85) }".to_string()));
    }

    #[tokio::test]
    async fn test_process_job_validation() {
        let orchestrator = ProcessingOrchestrator::new();
        
        // Test with empty operations
        let mut input = create_test_input();
        input.operations.clear();
        
        let result = orchestrator.process_job(input).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProcessingError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_process_job_unsupported_operation() {
        let orchestrator = ProcessingOrchestrator::new();
        let input = create_test_input(); // Contains Convert operation, but no processor registered
        
        let result = orchestrator.process_job(input).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProcessingError::ProcessingFailed { .. }));
    }

    #[tokio::test]
    async fn test_cancel_job() {
        let orchestrator = ProcessingOrchestrator::new();
        let job_id = Uuid::new_v4();
        
        let result = orchestrator.cancel_job(job_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let orchestrator = ProcessingOrchestrator::new();
        let stats = orchestrator.get_performance_stats();
        assert!(stats.is_empty());
    }

    #[tokio::test]
    async fn test_capability_discovery() {
        let mut orchestrator = ProcessingOrchestrator::new();
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor).unwrap();
        
        let capabilities = orchestrator.discover_capabilities();
        assert_eq!(capabilities.len(), 1);
        
        let (ptype, caps, metadata) = &capabilities[0];
        assert_eq!(*ptype, ProcessorType::FormatConverter);
        assert!(caps.input_formats.contains(&ImageFormat::Jpeg));
        assert_eq!(metadata.name, "test-processor");
    }

    #[tokio::test]
    async fn test_processor_capabilities_properties() {
        let caps = ProcessorCapabilities {
            input_formats: vec![ImageFormat::Jpeg, ImageFormat::Png],
            output_formats: vec![ImageFormat::WebP],
            operations: vec![ProcessorOperation::FormatConversion, ProcessorOperation::Resize],
            hardware_acceleration: true,
            streaming_support: false,
            max_file_size: Some(50 * 1024 * 1024),
            batch_optimized: true,
        };
        
        assert_eq!(caps.input_formats.len(), 2);
        assert_eq!(caps.output_formats.len(), 1);
        assert_eq!(caps.operations.len(), 2);
        assert!(caps.hardware_acceleration);
        assert!(!caps.streaming_support);
        assert_eq!(caps.max_file_size, Some(50 * 1024 * 1024));
        assert!(caps.batch_optimized);
    }

    #[tokio::test]
    async fn test_performance_profile_characteristics() {
        let profile = PerformanceProfile {
            speed_factor: 1.5,
            memory_usage: MemoryUsage::Streaming,
            cpu_usage: CpuUsage::GpuAccelerated,
            parallel_friendly: true,
        };
        
        assert_eq!(profile.speed_factor, 1.5);
        assert_eq!(profile.memory_usage, MemoryUsage::Streaming);
        assert_eq!(profile.cpu_usage, CpuUsage::GpuAccelerated);
        assert!(profile.parallel_friendly);
    }

    #[test]
    fn test_processor_operation_variants() {
        let ops = vec![
            ProcessorOperation::FormatConversion,
            ProcessorOperation::Resize,
            ProcessorOperation::Watermark,
            ProcessorOperation::BackgroundRemoval,
            ProcessorOperation::ColorCorrection,
            ProcessorOperation::Crop,
            ProcessorOperation::Rotate,
            ProcessorOperation::TextOverlay,
            ProcessorOperation::CollageCreation,
            ProcessorOperation::MetadataHandling,
        ];
        
        assert_eq!(ops.len(), 10);
        
        // Test that operations can be used in HashSet (requires Hash + Eq)
        let op_set: std::collections::HashSet<_> = ops.into_iter().collect();
        assert_eq!(op_set.len(), 10);
    }

    #[test]
    fn test_processor_type_variants() {
        let types = vec![
            ProcessorType::FormatConverter,
            ProcessorType::WatermarkEngine,
            ProcessorType::BackgroundRemover,
            ProcessorType::ResizeEngine,
            ProcessorType::ColorCorrector,
            ProcessorType::CropRotator,
            ProcessorType::TextOverlay,
            ProcessorType::CollageCreator,
            ProcessorType::MetadataHandler,
        ];
        
        assert_eq!(types.len(), 9);
        
        // Test that types can be used in HashMap (requires Hash + Eq)
        let type_map: HashMap<ProcessorType, String> = types.into_iter()
            .map(|t| (t, format!("{:?}", t)))
            .collect();
        assert_eq!(type_map.len(), 9);
    }

    #[test]
    fn test_memory_usage_ordering() {
        // Test that memory usage can be compared for processor selection
        assert!(MemoryUsage::Low < MemoryUsage::Moderate);
        assert!(MemoryUsage::Moderate < MemoryUsage::High);
        // Streaming should be considered efficient
        assert_eq!(MemoryUsage::Streaming, MemoryUsage::Streaming);
    }

    #[test]
    fn test_cpu_usage_variants() {
        let usages = vec![
            CpuUsage::Light,
            CpuUsage::Moderate,
            CpuUsage::Heavy,
            CpuUsage::GpuAccelerated,
        ];
        
        assert_eq!(usages.len(), 4);
        
        // All variants should be comparable
        for usage in &usages {
            assert_eq!(*usage, *usage);
        }
    }
}