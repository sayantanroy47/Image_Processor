//! Job lifecycle management system for coordinating image processing operations

use crate::error::{ProcessingError, Result};
use crate::models::{JobId, JobPriority, JobStatus, ProcessingInput, ProcessingJob};
use crate::progress::{CompletionStatus, ProgressTracker};
use crate::queue::{JobQueue, JobStatusUpdate};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::time::{interval, sleep, Instant};
use uuid::Uuid;

/// Job manager for coordinating queue operations and job lifecycle
#[derive(Debug)]
pub struct JobManager {
    /// Job queue for managing processing tasks
    job_queue: Arc<JobQueue>,
    /// Progress tracker for monitoring job progress
    progress_tracker: Arc<ProgressTracker>,
    /// Job persistence storage
    persistence: Arc<JobPersistence>,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Active job processors
    active_processors: Arc<RwLock<HashMap<JobId, JobProcessor>>>,
    /// Job status update channel
    status_sender: broadcast::Sender<JobStatusUpdate>,
    /// Shutdown signal
    shutdown_sender: Arc<Mutex<Option<mpsc::Sender<()>>>>,
    /// Manager configuration
    config: JobManagerConfig,
}

/// Job processor for handling individual job execution
#[derive(Debug)]
struct JobProcessor {
    job_id: JobId,
    cancellation_token: tokio_util::sync::CancellationToken,
    retry_count: u32,
    last_attempt: DateTime<Utc>,
}

/// Job persistence for crash recovery
#[derive(Debug)]
pub struct JobPersistence {
    storage_path: PathBuf,
    active_jobs: Arc<RwLock<HashMap<JobId, PersistedJob>>>,
}

/// Persisted job data for crash recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedJob {
    pub job: ProcessingJob,
    pub retry_count: u32,
    pub last_attempt: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub checkpoints: Vec<JobCheckpoint>,
}

/// Job checkpoint for incremental recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCheckpoint {
    pub operation_index: usize,
    pub operation_name: String,
    pub completed_at: DateTime<Utc>,
    pub intermediate_files: Vec<PathBuf>,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub retry_on_errors: Vec<RetryableError>,
}

/// Types of errors that should trigger retries
#[derive(Debug, Clone, PartialEq)]
pub enum RetryableError {
    IoError,
    TemporaryResourceUnavailable,
    NetworkTimeout,
    OutOfMemory,
    HardwareAccelerationFailure,
}

/// Job manager configuration
#[derive(Debug, Clone)]
pub struct JobManagerConfig {
    pub max_concurrent_jobs: usize,
    pub persistence_enabled: bool,
    pub persistence_path: PathBuf,
    pub cleanup_interval: Duration,
    pub job_timeout: Duration,
    pub heartbeat_interval: Duration,
}

impl Default for JobManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 4,
            persistence_enabled: true,
            persistence_path: PathBuf::from("./job_persistence"),
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            job_timeout: Duration::from_secs(3600),     // 1 hour
            heartbeat_interval: Duration::from_secs(30), // 30 seconds
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300), // 5 minutes
            backoff_multiplier: 2.0,
            retry_on_errors: vec![
                RetryableError::IoError,
                RetryableError::TemporaryResourceUnavailable,
                RetryableError::NetworkTimeout,
                RetryableError::HardwareAccelerationFailure,
            ],
        }
    }
}

impl JobManager {
    /// Create a new job manager
    pub async fn new(config: JobManagerConfig) -> Result<Self> {
        let job_queue = Arc::new(JobQueue::new(config.max_concurrent_jobs));
        let progress_tracker = Arc::new(ProgressTracker::new(1000));
        let persistence = Arc::new(JobPersistence::new(&config.persistence_path).await?);
        let retry_config = RetryConfig::default();
        let (status_sender, _) = broadcast::channel(1000);

        let persistence_enabled = config.persistence_enabled;
        
        let manager = Self {
            job_queue,
            progress_tracker,
            persistence,
            retry_config,
            active_processors: Arc::new(RwLock::new(HashMap::new())),
            status_sender,
            shutdown_sender: Arc::new(Mutex::new(None)),
            config,
        };

        // Recover jobs from persistence if enabled
        if persistence_enabled {
            manager.recover_jobs().await?;
        }

        Ok(manager)
    }

    /// Start the job manager background tasks
    pub async fn start(&self) -> Result<()> {
        let (shutdown_tx, _): (broadcast::Sender<()>, broadcast::Receiver<()>) = broadcast::channel(1);
        {
            let mut sender = self.shutdown_sender.lock().await;
            *sender = Some(mpsc::channel(1).0); // Placeholder, we'll use broadcast for actual shutdown
        }

        // Start job processing loop
        let manager_clone = self.clone_arc();
        tokio::spawn(async move {
            manager_clone.job_processing_loop().await;
        });

        // Start cleanup task
        let manager_clone = self.clone_arc();
        let mut cleanup_shutdown_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut cleanup_interval = interval(manager_clone.config.cleanup_interval);
            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        if let Err(e) = manager_clone.cleanup_completed_jobs().await {
                            tracing::error!("Failed to cleanup completed jobs: {}", e);
                        }
                    }
                    _ = cleanup_shutdown_rx.recv() => {
                        tracing::info!("Cleanup task shutting down");
                        break;
                    }
                }
            }
        });

        // Start heartbeat task for job monitoring
        let manager_clone = self.clone_arc();
        let mut heartbeat_shutdown_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut heartbeat_interval = interval(manager_clone.config.heartbeat_interval);
            loop {
                tokio::select! {
                    _ = heartbeat_interval.tick() => {
                        if let Err(e) = manager_clone.check_job_health().await {
                            tracing::error!("Failed to check job health: {}", e);
                        }
                    }
                    _ = heartbeat_shutdown_rx.recv() => {
                        tracing::info!("Heartbeat task shutting down");
                        break;
                    }
                }
            }
        });

        tracing::info!("Job manager started successfully");
        Ok(())
    }

    /// Submit a new job for processing
    pub async fn submit_job(&self, input: ProcessingInput, priority: JobPriority) -> Result<JobId> {
        let job_id = self.job_queue.enqueue_job(input.clone(), priority).await?;

        // Start progress tracking
        let total_operations = input.operations.len();
        self.progress_tracker
            .start_job(job_id, total_operations, input.file_size)
            .await?;

        // Persist job if enabled
        if self.config.persistence_enabled {
            let job = ProcessingJob {
                id: job_id,
                input,
                status: JobStatus::Pending,
                priority,
                created_at: Utc::now(),
                started_at: None,
                completed_at: None,
                error: None,
                progress: 0.0,
            };

            let persisted_job = PersistedJob {
                job,
                retry_count: 0,
                last_attempt: Utc::now(),
                created_at: Utc::now(),
                checkpoints: Vec::new(),
            };

            self.persistence.save_job(&persisted_job).await?;
        }

        tracing::info!("Job {} submitted with priority {:?}", job_id, priority);
        Ok(job_id)
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        // Cancel in queue first
        if let Err(e) = self.job_queue.cancel_job(job_id).await {
            tracing::debug!("Job {} not found in queue: {}", job_id, e);
        }

        // Cancel active processor if running
        {
            let mut processors = self.active_processors.write().await;
            if let Some(processor) = processors.remove(&job_id) {
                processor.cancellation_token.cancel();
                tracing::info!("Cancelled active job processor for job {}", job_id);
            }
        }

        // Update progress tracker
        self.progress_tracker
            .complete_job(job_id, CompletionStatus::Cancelled)
            .await?;

        // Remove from persistence
        if self.config.persistence_enabled {
            self.persistence.remove_job(job_id).await?;
        }

        // Send status update
        let update = JobStatusUpdate {
            job_id,
            old_status: JobStatus::Running, // Could be any status
            new_status: JobStatus::Cancelled,
            timestamp: Utc::now(),
            message: Some("Job cancelled by user".to_string()),
        };

        if let Err(e) = self.status_sender.send(update) {
            tracing::warn!("Failed to send job cancellation update: {}", e);
        }

        tracing::info!("Job {} cancelled successfully", job_id);
        Ok(())
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: JobId) -> Option<JobStatus> {
        self.job_queue.get_job_status(job_id).await
    }

    /// Get job details
    pub async fn get_job(&self, job_id: JobId) -> Option<ProcessingJob> {
        self.job_queue.get_job(job_id).await
    }

    /// Get job progress
    pub async fn get_job_progress(&self, job_id: JobId) -> Option<crate::models::ProcessingProgress> {
        self.progress_tracker.get_progress(job_id).await
    }

    /// Subscribe to job status updates
    pub fn subscribe_to_updates(&self) -> broadcast::Receiver<JobStatusUpdate> {
        self.status_sender.subscribe()
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> crate::queue::QueueStats {
        self.job_queue.get_stats().await
    }

    /// Shutdown the job manager
    pub async fn shutdown(&self) -> Result<()> {
        // Send shutdown signal
        {
            let sender = self.shutdown_sender.lock().await;
            if let Some(tx) = sender.as_ref() {
                let _ = tx.send(()).await;
            }
        }

        // Cancel all active jobs
        {
            let processors = self.active_processors.read().await;
            for processor in processors.values() {
                processor.cancellation_token.cancel();
            }
        }

        // Wait for active jobs to complete or timeout
        let timeout = Duration::from_secs(30);
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            let active_count = {
                let processors = self.active_processors.read().await;
                processors.len()
            };
            
            if active_count == 0 {
                break;
            }
            
            sleep(Duration::from_millis(100)).await;
        }

        tracing::info!("Job manager shutdown completed");
        Ok(())
    }

    /// Main job processing loop
    async fn job_processing_loop(&self) {
        let mut processing_interval = interval(Duration::from_millis(100));
        
        loop {
            processing_interval.tick().await;
            
            // Check if we can process more jobs
            let active_count = {
                let processors = self.active_processors.read().await;
                processors.len()
            };
            
            if active_count >= self.config.max_concurrent_jobs {
                continue;
            }
            
            // Try to get next job from queue
            if let Some(job) = self.job_queue.dequeue_job().await {
                let job_id = job.id;
                
                // Create job processor
                let processor = JobProcessor {
                    job_id,
                    cancellation_token: tokio_util::sync::CancellationToken::new(),
                    retry_count: 0,
                    last_attempt: Utc::now(),
                };
                
                // Add to active processors
                {
                    let mut processors = self.active_processors.write().await;
                    processors.insert(job_id, processor);
                }
                
                // Start job processing
                let manager_clone = self.clone_arc();
                tokio::spawn(async move {
                    manager_clone.process_job(job).await;
                });
            }
        }
    }

    /// Process a single job with retry logic
    async fn process_job(&self, job: ProcessingJob) {
        let job_id = job.id;
        let mut retry_count = 0;
        let mut last_error: Option<ProcessingError> = None;

        loop {
            // Check if job was cancelled
            {
                let processors = self.active_processors.read().await;
                if let Some(processor) = processors.get(&job_id) {
                    if processor.cancellation_token.is_cancelled() {
                        tracing::info!("Job {} was cancelled, stopping processing", job_id);
                        break;
                    }
                } else {
                    tracing::warn!("Job processor {} not found, stopping processing", job_id);
                    break;
                }
            }

            // Update progress tracker
            if let Err(e) = self.progress_tracker
                .update_operation(job_id, format!("Processing (attempt {})", retry_count + 1))
                .await
            {
                tracing::error!("Failed to update progress for job {}: {}", job_id, e);
            }

            // Simulate job processing (in real implementation, this would call the actual processor)
            let result = self.execute_job_operations(&job).await;

            match result {
                Ok(_) => {
                    // Job completed successfully
                    if let Err(e) = self.job_queue.complete_job(job_id, None).await {
                        tracing::error!("Failed to mark job {} as completed: {}", job_id, e);
                    }

                    if let Err(e) = self.progress_tracker
                        .complete_job(job_id, CompletionStatus::Success)
                        .await
                    {
                        tracing::error!("Failed to complete progress tracking for job {}: {}", job_id, e);
                    }

                    // Remove from persistence
                    if self.config.persistence_enabled {
                        if let Err(e) = self.persistence.remove_job(job_id).await {
                            tracing::error!("Failed to remove job {} from persistence: {}", job_id, e);
                        }
                    }

                    tracing::info!("Job {} completed successfully", job_id);
                    break;
                }
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    // Check if error is retryable
                    if self.should_retry(&error) && retry_count < self.retry_config.max_retries {
                        retry_count += 1;
                        let delay = self.calculate_retry_delay(retry_count);
                        
                        tracing::warn!(
                            "Job {} failed (attempt {}), retrying in {:?}: {}",
                            job_id, retry_count, delay, error
                        );

                        // Update retry count in persistence
                        if self.config.persistence_enabled {
                            if let Err(e) = self.persistence.update_retry_count(job_id, retry_count).await {
                                tracing::error!("Failed to update retry count for job {}: {}", job_id, e);
                            }
                        }

                        // Wait before retry
                        sleep(delay).await;
                        continue;
                    } else {
                        // Max retries reached or non-retryable error
                        if let Err(e) = self.job_queue.complete_job(job_id, Some(error.clone())).await {
                            tracing::error!("Failed to mark job {} as failed: {}", job_id, e);
                        }

                        if let Err(e) = self.progress_tracker
                            .complete_job(job_id, CompletionStatus::Failed(error.to_string()))
                            .await
                        {
                            tracing::error!("Failed to complete progress tracking for job {}: {}", job_id, e);
                        }

                        // Remove from persistence
                        if self.config.persistence_enabled {
                            if let Err(e) = self.persistence.remove_job(job_id).await {
                                tracing::error!("Failed to remove job {} from persistence: {}", job_id, e);
                            }
                        }

                        tracing::error!("Job {} failed after {} attempts: {}", job_id, retry_count + 1, error);
                        break;
                    }
                }
            }
        }

        // Remove from active processors
        {
            let mut processors = self.active_processors.write().await;
            processors.remove(&job_id);
        }
    }

    /// Execute job operations (placeholder for actual processing)
    async fn execute_job_operations(&self, job: &ProcessingJob) -> Result<()> {
        // This is a placeholder implementation
        // In the real implementation, this would call the actual image processing engines
        
        let total_operations = job.input.operations.len();
        
        for (index, operation) in job.input.operations.iter().enumerate() {
            // Check for cancellation
            {
                let processors = self.active_processors.read().await;
                if let Some(processor) = processors.get(&job.id) {
                    if processor.cancellation_token.is_cancelled() {
                        return Err(ProcessingError::OperationCancelled {
                            message: "Job was cancelled".to_string(),
                        });
                    }
                }
            }

            // Update progress
            let operation_name = format!("{:?}", operation);
            if let Err(e) = self.progress_tracker
                .update_operation(job.id, operation_name.clone())
                .await
            {
                tracing::error!("Failed to update operation for job {}: {}", job.id, e);
            }

            // Simulate processing time
            sleep(Duration::from_millis(100)).await;

            // Create checkpoint
            if self.config.persistence_enabled {
                let checkpoint = JobCheckpoint {
                    operation_index: index,
                    operation_name,
                    completed_at: Utc::now(),
                    intermediate_files: Vec::new(), // Would contain actual intermediate files
                };

                if let Err(e) = self.persistence.add_checkpoint(job.id, checkpoint).await {
                    tracing::error!("Failed to add checkpoint for job {}: {}", job.id, e);
                }
            }

            // Complete operation
            if let Err(e) = self.progress_tracker.complete_operation(job.id).await {
                tracing::error!("Failed to complete operation for job {}: {}", job.id, e);
            }

            // Update bytes processed (simulated)
            let bytes_per_operation = job.input.file_size / total_operations as u64;
            let bytes_processed = (index + 1) as u64 * bytes_per_operation;
            if let Err(e) = self.progress_tracker
                .update_bytes_processed(job.id, bytes_processed)
                .await
            {
                tracing::error!("Failed to update bytes processed for job {}: {}", job.id, e);
            }
        }

        Ok(())
    }

    /// Check if an error should trigger a retry
    fn should_retry(&self, error: &ProcessingError) -> bool {
        let retryable_error = match error {
            ProcessingError::Io { .. } => Some(RetryableError::IoError),
            ProcessingError::OutOfMemory => Some(RetryableError::OutOfMemory),
            ProcessingError::HardwareAccelerationUnavailable => Some(RetryableError::HardwareAccelerationFailure),
            _ => None,
        };

        if let Some(error_type) = retryable_error {
            self.retry_config.retry_on_errors.contains(&error_type)
        } else {
            false
        }
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, retry_count: u32) -> Duration {
        let delay_secs = self.retry_config.initial_delay.as_secs_f64()
            * self.retry_config.backoff_multiplier.powi(retry_count as i32 - 1);
        
        let delay = Duration::from_secs_f64(delay_secs);
        std::cmp::min(delay, self.retry_config.max_delay)
    }

    /// Recover jobs from persistence after restart
    async fn recover_jobs(&self) -> Result<()> {
        let persisted_jobs = self.persistence.load_all_jobs().await?;
        
        for persisted_job in persisted_jobs {
            let job_id = persisted_job.job.id;
            
            // Re-enqueue the job
            if let Err(e) = self.job_queue
                .enqueue_job(persisted_job.job.input.clone(), persisted_job.job.priority)
                .await
            {
                tracing::error!("Failed to recover job {}: {}", job_id, e);
                continue;
            }

            // Restart progress tracking
            let total_operations = persisted_job.job.input.operations.len();
            if let Err(e) = self.progress_tracker
                .start_job(job_id, total_operations, persisted_job.job.input.file_size)
                .await
            {
                tracing::error!("Failed to restart progress tracking for job {}: {}", job_id, e);
            }

            tracing::info!("Recovered job {} from persistence", job_id);
        }

        tracing::info!("Job recovery completed");
        Ok(())
    }

    /// Cleanup completed jobs from persistence
    async fn cleanup_completed_jobs(&self) -> Result<()> {
        // This would implement cleanup logic for old completed jobs
        // For now, it's a placeholder
        tracing::debug!("Performing job cleanup");
        Ok(())
    }

    /// Check health of active jobs and handle timeouts
    async fn check_job_health(&self) -> Result<()> {
        let now = Utc::now();
        let timeout_duration = chrono::Duration::from_std(self.config.job_timeout)
            .map_err(|e| ProcessingError::InvalidInput {
                message: format!("Invalid timeout duration: {}", e),
            })?;

        let mut timed_out_jobs = Vec::new();

        {
            let processors = self.active_processors.read().await;
            for (job_id, processor) in processors.iter() {
                if now - processor.last_attempt > timeout_duration {
                    timed_out_jobs.push(*job_id);
                }
            }
        }

        for job_id in timed_out_jobs {
            tracing::warn!("Job {} timed out, cancelling", job_id);
            if let Err(e) = self.cancel_job(job_id).await {
                tracing::error!("Failed to cancel timed out job {}: {}", job_id, e);
            }
        }

        Ok(())
    }

    /// Helper method to clone Arc references for async tasks
    fn clone_arc(&self) -> Arc<Self> {
        Arc::new(Self {
            job_queue: self.job_queue.clone(),
            progress_tracker: self.progress_tracker.clone(),
            persistence: self.persistence.clone(),
            retry_config: self.retry_config.clone(),
            active_processors: self.active_processors.clone(),
            status_sender: self.status_sender.clone(),
            shutdown_sender: self.shutdown_sender.clone(),
            config: self.config.clone(),
        })
    }
}

impl JobPersistence {
    /// Create a new job persistence manager
    pub async fn new(storage_path: &Path) -> Result<Self> {
        // Create storage directory if it doesn't exist
        if !storage_path.exists() {
            fs::create_dir_all(storage_path).await.map_err(|e| {
                ProcessingError::Io {
                    message: format!("Failed to create persistence directory: {}", e),
                }
            })?;
        }

        Ok(Self {
            storage_path: storage_path.to_path_buf(),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Save a job to persistence
    pub async fn save_job(&self, job: &PersistedJob) -> Result<()> {
        let job_file = self.storage_path.join(format!("{}.json", job.job.id));
        let json = serde_json::to_string_pretty(job).map_err(|e| {
            ProcessingError::InvalidInput {
                message: format!("Failed to serialize job: {}", e),
            }
        })?;

        fs::write(&job_file, json).await.map_err(|e| {
            ProcessingError::Io {
                message: format!("Failed to write job file: {}", e),
            }
        })?;

        // Update in-memory cache
        {
            let mut active_jobs = self.active_jobs.write().await;
            active_jobs.insert(job.job.id, job.clone());
        }

        Ok(())
    }

    /// Remove a job from persistence
    pub async fn remove_job(&self, job_id: JobId) -> Result<()> {
        let job_file = self.storage_path.join(format!("{}.json", job_id));
        
        if job_file.exists() {
            fs::remove_file(&job_file).await.map_err(|e| {
                ProcessingError::Io {
                    message: format!("Failed to remove job file: {}", e),
                }
            })?;
        }

        // Remove from in-memory cache
        {
            let mut active_jobs = self.active_jobs.write().await;
            active_jobs.remove(&job_id);
        }

        Ok(())
    }

    /// Load all persisted jobs
    pub async fn load_all_jobs(&self) -> Result<Vec<PersistedJob>> {
        let mut jobs = Vec::new();
        let mut dir = fs::read_dir(&self.storage_path).await.map_err(|e| {
            ProcessingError::Io {
                message: format!("Failed to read persistence directory: {}", e),
            }
        })?;

        while let Some(entry) = dir.next_entry().await.map_err(|e| {
            ProcessingError::Io {
                message: format!("Failed to read directory entry: {}", e),
            }
        })? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_job_from_file(&path).await {
                    Ok(job) => jobs.push(job),
                    Err(e) => {
                        tracing::error!("Failed to load job from {}: {}", path.display(), e);
                    }
                }
            }
        }

        // Update in-memory cache
        {
            let mut active_jobs = self.active_jobs.write().await;
            for job in &jobs {
                active_jobs.insert(job.job.id, job.clone());
            }
        }

        Ok(jobs)
    }

    /// Update retry count for a job
    pub async fn update_retry_count(&self, job_id: JobId, retry_count: u32) -> Result<()> {
        let mut job = {
            let active_jobs = self.active_jobs.read().await;
            active_jobs.get(&job_id).cloned()
        };

        if let Some(ref mut job) = job {
            job.retry_count = retry_count;
            job.last_attempt = Utc::now();
            self.save_job(job).await?;
        }

        Ok(())
    }

    /// Add a checkpoint to a job
    pub async fn add_checkpoint(&self, job_id: JobId, checkpoint: JobCheckpoint) -> Result<()> {
        let mut job = {
            let active_jobs = self.active_jobs.read().await;
            active_jobs.get(&job_id).cloned()
        };

        if let Some(ref mut job) = job {
            job.checkpoints.push(checkpoint);
            self.save_job(job).await?;
        }

        Ok(())
    }

    /// Load a job from a file
    async fn load_job_from_file(&self, path: &Path) -> Result<PersistedJob> {
        let content = fs::read_to_string(path).await.map_err(|e| {
            ProcessingError::Io {
                message: format!("Failed to read job file: {}", e),
            }
        })?;

        serde_json::from_str(&content).map_err(|e| {
            ProcessingError::InvalidInput {
                message: format!("Failed to deserialize job: {}", e),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ImageFormat;
    use crate::models::{ProcessingOperation, ProcessingOptions};
    use tempfile::TempDir;

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
    async fn test_job_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = JobManagerConfig {
            persistence_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = JobManager::new(config).await.unwrap();
        let stats = manager.get_queue_stats().await;
        assert_eq!(stats.pending_jobs, 0);
    }

    #[tokio::test]
    async fn test_job_submission() {
        let temp_dir = TempDir::new().unwrap();
        let config = JobManagerConfig {
            persistence_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = JobManager::new(config).await.unwrap();
        let input = create_test_input();
        
        let job_id = manager.submit_job(input, JobPriority::Normal).await.unwrap();
        
        let status = manager.get_job_status(job_id).await;
        assert_eq!(status, Some(JobStatus::Pending));
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let temp_dir = TempDir::new().unwrap();
        let config = JobManagerConfig {
            persistence_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = JobManager::new(config).await.unwrap();
        let input = create_test_input();
        
        let job_id = manager.submit_job(input, JobPriority::Normal).await.unwrap();
        manager.cancel_job(job_id).await.unwrap();
        
        let status = manager.get_job_status(job_id).await;
        assert_eq!(status, Some(JobStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_job_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = JobPersistence::new(temp_dir.path()).await.unwrap();
        
        let job = ProcessingJob {
            id: Uuid::new_v4(),
            input: create_test_input(),
            status: JobStatus::Pending,
            priority: JobPriority::Normal,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            progress: 0.0,
        };

        let persisted_job = PersistedJob {
            job: job.clone(),
            retry_count: 0,
            last_attempt: Utc::now(),
            created_at: Utc::now(),
            checkpoints: Vec::new(),
        };

        persistence.save_job(&persisted_job).await.unwrap();
        
        let loaded_jobs = persistence.load_all_jobs().await.unwrap();
        assert_eq!(loaded_jobs.len(), 1);
        assert_eq!(loaded_jobs[0].job.id, job.id);
    }

    #[tokio::test]
    async fn test_retry_logic() {
        let config = RetryConfig::default();
        
        // Test retry delay calculation
        let delay1 = Duration::from_secs_f64(
            config.initial_delay.as_secs_f64() * config.backoff_multiplier.powi(0)
        );
        let delay2 = Duration::from_secs_f64(
            config.initial_delay.as_secs_f64() * config.backoff_multiplier.powi(1)
        );
        
        assert!(delay2 > delay1);
        assert!(delay2 <= config.max_delay);
    }

    #[tokio::test]
    async fn test_checkpoint_functionality() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = JobPersistence::new(temp_dir.path()).await.unwrap();
        
        let job_id = Uuid::new_v4();
        let job = ProcessingJob {
            id: job_id,
            input: create_test_input(),
            status: JobStatus::Running,
            priority: JobPriority::Normal,
            created_at: Utc::now(),
            started_at: Some(Utc::now()),
            completed_at: None,
            error: None,
            progress: 0.5,
        };

        let persisted_job = PersistedJob {
            job,
            retry_count: 0,
            last_attempt: Utc::now(),
            created_at: Utc::now(),
            checkpoints: Vec::new(),
        };

        persistence.save_job(&persisted_job).await.unwrap();

        let checkpoint = JobCheckpoint {
            operation_index: 0,
            operation_name: "Convert".to_string(),
            completed_at: Utc::now(),
            intermediate_files: vec![PathBuf::from("temp.png")],
        };

        persistence.add_checkpoint(job_id, checkpoint).await.unwrap();
        
        let loaded_jobs = persistence.load_all_jobs().await.unwrap();
        assert_eq!(loaded_jobs[0].checkpoints.len(), 1);
        assert_eq!(loaded_jobs[0].checkpoints[0].operation_name, "Convert");
    }
}