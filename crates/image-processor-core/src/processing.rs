//! Core processing engine and orchestrator

use crate::error::{ProcessingError, Result};
use crate::models::{ProcessingInput, ProcessingOutput, ProcessingJob, JobId, JobStatus};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};

/// Core trait that all image processors must implement
#[async_trait::async_trait]
pub trait ImageProcessor: Send + Sync {
    /// Process an image with the given input parameters
    async fn process(&self, input: ProcessingInput) -> Result<ProcessingOutput>;
    
    /// Check if this processor supports the given format
    fn supports_format(&self, format: crate::config::ImageFormat) -> bool;
    
    /// Get the capabilities of this processor
    fn get_capabilities(&self) -> ProcessorCapabilities;
    
    /// Get the name of this processor
    fn name(&self) -> &'static str;
}

/// Processor capabilities
#[derive(Debug, Clone)]
pub struct ProcessorCapabilities {
    pub supported_formats: Vec<crate::config::ImageFormat>,
    pub supports_streaming: bool,
    pub supports_hardware_acceleration: bool,
    pub max_file_size_mb: Option<u64>,
    pub parallel_processing: bool,
}

/// Type of processor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    FormatConverter,
    WatermarkEngine,
    MetadataHandler,
    ResizeEngine,
    BackgroundRemoval,
    ColorCorrection,
    CropRotate,
    TextOverlay,
    CollageCreator,
}

/// Main processing orchestrator that coordinates all processors
pub struct ProcessingOrchestrator {
    processors: HashMap<ProcessorType, Arc<dyn ImageProcessor>>,
    active_jobs: Arc<RwLock<HashMap<JobId, ProcessingJob>>>,
    max_concurrent_jobs: usize,
}

impl ProcessingOrchestrator {
    /// Create a new processing orchestrator
    pub fn new(max_concurrent_jobs: usize) -> Self {
        Self {
            processors: HashMap::new(),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_jobs,
        }
    }

    /// Register a processor with the orchestrator
    pub fn register_processor(
        &mut self,
        processor_type: ProcessorType,
        processor: Arc<dyn ImageProcessor>,
    ) {
        info!("Registering processor: {:?} ({})", processor_type, processor.name());
        self.processors.insert(processor_type, processor);
    }

    /// Get a processor by type
    pub fn get_processor(&self, processor_type: ProcessorType) -> Option<&Arc<dyn ImageProcessor>> {
        self.processors.get(&processor_type)
    }

    /// Process a single job
    #[instrument(skip(self, input), fields(job_id = ?input.job_id))]
    pub async fn process_job(&self, input: ProcessingInput) -> Result<ProcessingOutput> {
        let job_id = input.job_id;
        
        // Check if we're at capacity
        let active_count = self.active_jobs.read().await.len();
        if active_count >= self.max_concurrent_jobs {
            return Err(ProcessingError::ProcessingFailed {
                message: "Maximum concurrent jobs reached".to_string(),
            });
        }

        // Create and track the job
        let job = ProcessingJob {
            id: job_id,
            input: input.clone(),
            status: JobStatus::Running,
            priority: crate::models::JobPriority::Normal,
            created_at: chrono::Utc::now(),
            started_at: Some(chrono::Utc::now()),
            completed_at: None,
            error: None,
            progress: 0.0,
        };

        self.active_jobs.write().await.insert(job_id, job);

        info!("Starting job processing",
              job_id = ?job_id,
              operations_count = input.operations.len(),
              source_path = ?input.source_path.display());

        let result = self.execute_processing(input).await;

        // Update job status
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            match &result {
                Ok(_) => {
                    job.status = JobStatus::Completed;
                    job.completed_at = Some(chrono::Utc::now());
                    job.progress = 1.0;
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.completed_at = Some(chrono::Utc::now());
                    job.error = Some(e.to_string());
                }
            }
        }

        // Remove completed job from active tracking
        jobs.remove(&job_id);

        match &result {
            Ok(output) => {
                info!("Job completed successfully",
                      job_id = %job_id,
                      output_path = %output.output_path.display(),
                      processing_time_ms = output.processing_time.as_millis());
            }
            Err(e) => {
                error!("Job failed",
                       job_id = %job_id,
                       error = %e);
            }
        }

        result
    }

    /// Execute the actual processing operations
    #[instrument(skip(self, input))]
    async fn execute_processing(&self, input: ProcessingInput) -> Result<ProcessingOutput> {
        let start_time = std::time::Instant::now();
        let mut current_path = input.source_path.clone();
        let mut operations_applied = Vec::new();

        // Process each operation in sequence
        for (i, operation) in input.operations.iter().enumerate() {
            info!("Executing operation {} of {}: {:?}", 
                  i + 1, 
                  input.operations.len(), 
                  std::mem::discriminant(operation));

            // For now, we'll create a placeholder implementation
            // In the actual implementation, this would route to specific processors
            match operation {
                crate::models::ProcessingOperation::Convert { format, quality } => {
                    operations_applied.push(format!("Convert to {:?} (quality: {:?})", format, quality));
                }
                crate::models::ProcessingOperation::Resize { width, height, algorithm } => {
                    operations_applied.push(format!("Resize to {}x{} using {:?}", 
                                                   width.unwrap_or(0), 
                                                   height.unwrap_or(0), 
                                                   algorithm));
                }
                crate::models::ProcessingOperation::Watermark { config } => {
                    operations_applied.push(format!("Apply watermark from {:?}", config.watermark_path));
                }
                crate::models::ProcessingOperation::RemoveBackground { model } => {
                    operations_applied.push(format!("Remove background using {:?}", model));
                }
                crate::models::ProcessingOperation::ColorCorrect { adjustments } => {
                    operations_applied.push(format!("Color correction (brightness: {}, contrast: {})", 
                                                   adjustments.brightness, 
                                                   adjustments.contrast));
                }
                crate::models::ProcessingOperation::Crop { region } => {
                    operations_applied.push(format!("Crop to {}x{} at ({}, {})", 
                                                   region.width, 
                                                   region.height, 
                                                   region.x, 
                                                   region.y));
                }
                crate::models::ProcessingOperation::Rotate { angle } => {
                    operations_applied.push(format!("Rotate by {} degrees", angle));
                }
                crate::models::ProcessingOperation::AddText { text_config } => {
                    operations_applied.push(format!("Add text: '{}'", text_config.text));
                }
                crate::models::ProcessingOperation::CreateCollage { layout } => {
                    operations_applied.push(format!("Create collage with {:?} layout", layout.template));
                }
            }

            // Update current path for chained operations
            // In a real implementation, this would be the output of the previous operation
            current_path = input.output_path.clone();
        }

        let processing_time = start_time.elapsed();

        // Create the output
        let output = ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path.clone(),
            file_size: input.file_size, // Placeholder - would be actual output file size
            format: input.format, // Placeholder - would be actual output format
            processing_time,
            operations_applied,
            metadata: None, // Would be populated with actual metadata
        };

        Ok(output)
    }

    /// Get the status of all active jobs
    pub async fn get_active_jobs(&self) -> Vec<ProcessingJob> {
        self.active_jobs.read().await.values().cloned().collect()
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Cancelled;
            job.completed_at = Some(chrono::Utc::now());
            info!("Job cancelled", job_id = %job_id);
            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found", job_id),
            })
        }
    }

    /// Get processor capabilities
    pub fn get_all_capabilities(&self) -> HashMap<ProcessorType, ProcessorCapabilities> {
        self.processors
            .iter()
            .map(|(processor_type, processor)| {
                (*processor_type, processor.get_capabilities())
            })
            .collect()
    }

    /// Check if a format is supported by any processor
    pub fn is_format_supported(&self, format: crate::config::ImageFormat) -> bool {
        self.processors
            .values()
            .any(|processor| processor.supports_format(format))
    }
}

/// Mock processor for testing and initial setup
pub struct MockProcessor {
    name: &'static str,
    capabilities: ProcessorCapabilities,
}

impl MockProcessor {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            capabilities: ProcessorCapabilities {
                supported_formats: vec![
                    crate::config::ImageFormat::Jpeg,
                    crate::config::ImageFormat::Png,
                    crate::config::ImageFormat::WebP,
                ],
                supports_streaming: true,
                supports_hardware_acceleration: false,
                max_file_size_mb: Some(100),
                parallel_processing: true,
            },
        }
    }
}

#[async_trait::async_trait]
impl ImageProcessor for MockProcessor {
    async fn process(&self, input: ProcessingInput) -> Result<ProcessingOutput> {
        // Simulate processing time
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        info!("Mock processor {} processing job {}", self.name, input.job_id);

        Ok(ProcessingOutput {
            job_id: input.job_id,
            output_path: input.output_path,
            file_size: input.file_size,
            format: input.format,
            processing_time: std::time::Duration::from_millis(100),
            operations_applied: vec![format!("Mock processing by {}", self.name)],
            metadata: None,
        })
    }

    fn supports_format(&self, format: crate::config::ImageFormat) -> bool {
        self.capabilities.supported_formats.contains(&format)
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        self.capabilities.clone()
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = ProcessingOrchestrator::new(4);
        assert_eq!(orchestrator.max_concurrent_jobs, 4);
        assert!(orchestrator.processors.is_empty());
    }

    #[tokio::test]
    async fn test_processor_registration() {
        let mut orchestrator = ProcessingOrchestrator::new(4);
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor.clone());
        
        let retrieved = orchestrator.get_processor(ProcessorType::FormatConverter);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "test-processor");
    }

    #[tokio::test]
    async fn test_format_support_check() {
        let mut orchestrator = ProcessingOrchestrator::new(4);
        let processor = Arc::new(MockProcessor::new("test-processor"));
        
        orchestrator.register_processor(ProcessorType::FormatConverter, processor);
        
        assert!(orchestrator.is_format_supported(crate::config::ImageFormat::Jpeg));
        assert!(orchestrator.is_format_supported(crate::config::ImageFormat::Png));
    }

    #[tokio::test]
    async fn test_mock_processor() {
        let processor = MockProcessor::new("test");
        assert_eq!(processor.name(), "test");
        assert!(processor.supports_format(crate::config::ImageFormat::Jpeg));
        
        let capabilities = processor.get_capabilities();
        assert!(capabilities.supports_streaming);
        assert!(!capabilities.supports_hardware_acceleration);
    }

    #[tokio::test]
    async fn test_job_processing() {
        let mut orchestrator = ProcessingOrchestrator::new(4);
        let processor = Arc::new(MockProcessor::new("test-processor"));
        orchestrator.register_processor(ProcessorType::FormatConverter, processor);

        let input = ProcessingInput {
            job_id: Uuid::new_v4(),
            source_path: std::path::PathBuf::from("test.jpg"),
            output_path: std::path::PathBuf::from("output.jpg"),
            operations: vec![ProcessingOperation::Convert {
                format: crate::config::ImageFormat::Jpeg,
                quality: Some(85),
            }],
            options: ProcessingOptions::default(),
            file_size: 1024,
            format: crate::config::ImageFormat::Jpeg,
        };

        let result = orchestrator.process_job(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.operations_applied.is_empty());
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let orchestrator = ProcessingOrchestrator::new(4);
        let job_id = Uuid::new_v4();
        
        // Try to cancel a non-existent job
        let result = orchestrator.cancel_job(job_id).await;
        assert!(result.is_err());
    }
}