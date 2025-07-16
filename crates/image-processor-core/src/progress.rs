//! Progress tracking system for image processing operations

use crate::error::{ProcessingError, Result};
use crate::models::{JobId, ProcessingProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Progress tracker for managing processing progress across multiple jobs
#[derive(Debug)]
pub struct ProgressTracker {
    /// Active progress tracking for jobs
    active_progress: Arc<RwLock<HashMap<JobId, ProgressState>>>,
    /// Broadcast channel for progress updates
    progress_sender: broadcast::Sender<ProgressUpdate>,
    /// Historical progress data for completed jobs
    history: Arc<RwLock<HashMap<JobId, CompletedProgress>>>,
    /// Maximum number of historical entries to keep
    max_history_entries: usize,
}

/// Internal progress state for a job
#[derive(Debug, Clone)]
struct ProgressState {
    job_id: JobId,
    current_operation: String,
    operations_completed: usize,
    total_operations: usize,
    bytes_processed: u64,
    total_bytes: u64,
    start_time: Instant,
    last_update: Instant,
    processing_speeds: Vec<f64>, // Recent speed measurements for smoothing
    estimated_completion: Option<DateTime<Utc>>,
}

/// Progress update message for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub job_id: JobId,
    pub progress: ProcessingProgress,
    pub timestamp: DateTime<Utc>,
    pub update_type: ProgressUpdateType,
}

/// Types of progress updates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProgressUpdateType {
    Started,
    OperationChanged,
    ProgressIncremented,
    Completed,
    Failed,
    Cancelled,
}

/// Completed progress information for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedProgress {
    pub job_id: JobId,
    pub total_duration: Duration,
    pub average_speed: f64,
    pub completed_at: DateTime<Utc>,
    pub final_status: CompletionStatus,
}

/// Final completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionStatus {
    Success,
    Failed(String),
    Cancelled,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(max_history_entries: usize) -> Self {
        let (progress_sender, _) = broadcast::channel(1000);
        
        Self {
            active_progress: Arc::new(RwLock::new(HashMap::new())),
            progress_sender,
            history: Arc::new(RwLock::new(HashMap::new())),
            max_history_entries,
        }
    }

    /// Start tracking progress for a new job
    pub async fn start_job(
        &self,
        job_id: JobId,
        total_operations: usize,
        total_bytes: u64,
    ) -> Result<()> {
        let now = Instant::now();
        let state = ProgressState {
            job_id,
            current_operation: "Initializing".to_string(),
            operations_completed: 0,
            total_operations,
            bytes_processed: 0,
            total_bytes,
            start_time: now,
            last_update: now,
            processing_speeds: Vec::new(),
            estimated_completion: None,
        };

        {
            let mut active = self.active_progress.write().await;
            active.insert(job_id, state.clone());
        }

        let progress = self.create_processing_progress(&state);
        let update = ProgressUpdate {
            job_id,
            progress,
            timestamp: Utc::now(),
            update_type: ProgressUpdateType::Started,
        };

        if let Err(e) = self.progress_sender.send(update) {
            tracing::warn!("Failed to send progress update: {}", e);
        }

        tracing::info!("Started progress tracking for job {}", job_id);
        Ok(())
    }

    /// Update the current operation for a job
    pub async fn update_operation(&self, job_id: JobId, operation: String) -> Result<()> {
        let mut active = self.active_progress.write().await;
        if let Some(state) = active.get_mut(&job_id) {
            state.current_operation = operation;
            state.last_update = Instant::now();

            let progress = self.create_processing_progress(state);
            let update = ProgressUpdate {
                job_id,
                progress,
                timestamp: Utc::now(),
                update_type: ProgressUpdateType::OperationChanged,
            };

            if let Err(e) = self.progress_sender.send(update) {
                tracing::warn!("Failed to send progress update: {}", e);
            }

            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found in active progress", job_id),
            })
        }
    }

    /// Update bytes processed for a job
    pub async fn update_bytes_processed(&self, job_id: JobId, bytes_processed: u64) -> Result<()> {
        let mut active = self.active_progress.write().await;
        if let Some(state) = active.get_mut(&job_id) {
            let now = Instant::now();
            let time_diff = now.duration_since(state.last_update).as_secs_f64();
            
            if time_diff > 0.0 {
                let bytes_diff = bytes_processed.saturating_sub(state.bytes_processed);
                let speed = (bytes_diff as f64) / (1024.0 * 1024.0) / time_diff; // MB/s
                
                // Keep only recent speed measurements for smoothing
                state.processing_speeds.push(speed);
                if state.processing_speeds.len() > 10 {
                    state.processing_speeds.remove(0);
                }
            }

            state.bytes_processed = bytes_processed;
            state.last_update = now;
            
            // Update estimated completion time
            self.update_estimated_completion(state);

            let progress = self.create_processing_progress(state);
            let update = ProgressUpdate {
                job_id,
                progress,
                timestamp: Utc::now(),
                update_type: ProgressUpdateType::ProgressIncremented,
            };

            if let Err(e) = self.progress_sender.send(update) {
                tracing::warn!("Failed to send progress update: {}", e);
            }

            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found in active progress", job_id),
            })
        }
    }

    /// Mark an operation as completed for a job
    pub async fn complete_operation(&self, job_id: JobId) -> Result<()> {
        let mut active = self.active_progress.write().await;
        if let Some(state) = active.get_mut(&job_id) {
            state.operations_completed += 1;
            state.last_update = Instant::now();
            
            // Update estimated completion time
            self.update_estimated_completion(state);

            let progress = self.create_processing_progress(state);
            let update = ProgressUpdate {
                job_id,
                progress,
                timestamp: Utc::now(),
                update_type: ProgressUpdateType::ProgressIncremented,
            };

            if let Err(e) = self.progress_sender.send(update) {
                tracing::warn!("Failed to send progress update: {}", e);
            }

            tracing::debug!("Operation completed for job {}: {}/{}", 
                          job_id, state.operations_completed, state.total_operations);
            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found in active progress", job_id),
            })
        }
    }

    /// Complete tracking for a job
    pub async fn complete_job(&self, job_id: JobId, status: CompletionStatus) -> Result<()> {
        let state = {
            let mut active = self.active_progress.write().await;
            active.remove(&job_id)
        };

        if let Some(state) = state {
            let total_duration = state.start_time.elapsed();
            let average_speed = if !state.processing_speeds.is_empty() {
                state.processing_speeds.iter().sum::<f64>() / state.processing_speeds.len() as f64
            } else {
                0.0
            };

            let completed_progress = CompletedProgress {
                job_id,
                total_duration,
                average_speed,
                completed_at: Utc::now(),
                final_status: status.clone(),
            };

            // Add to history
            {
                let mut history = self.history.write().await;
                history.insert(job_id, completed_progress);
                
                // Cleanup old entries if we exceed the limit
                if history.len() > self.max_history_entries {
                    let oldest_key = history
                        .iter()
                        .min_by_key(|(_, v)| v.completed_at)
                        .map(|(k, _)| *k);
                    
                    if let Some(key) = oldest_key {
                        history.remove(&key);
                    }
                }
            }

            let progress = ProcessingProgress {
                job_id,
                current_operation: "Completed".to_string(),
                operations_completed: state.operations_completed,
                total_operations: state.total_operations,
                bytes_processed: state.bytes_processed,
                total_bytes: state.total_bytes,
                estimated_time_remaining: None,
                processing_speed: average_speed,
            };

            let update_type = match status {
                CompletionStatus::Success => ProgressUpdateType::Completed,
                CompletionStatus::Failed(_) => ProgressUpdateType::Failed,
                CompletionStatus::Cancelled => ProgressUpdateType::Cancelled,
            };

            let update = ProgressUpdate {
                job_id,
                progress,
                timestamp: Utc::now(),
                update_type,
            };

            if let Err(e) = self.progress_sender.send(update) {
                tracing::warn!("Failed to send progress update: {}", e);
            }

            tracing::info!("Completed progress tracking for job {} with status {:?}", 
                          job_id, status);
            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found in active progress", job_id),
            })
        }
    }

    /// Get current progress for a job
    pub async fn get_progress(&self, job_id: JobId) -> Option<ProcessingProgress> {
        let active = self.active_progress.read().await;
        active.get(&job_id).map(|state| self.create_processing_progress(state))
    }

    /// Get all active progress
    pub async fn get_all_active_progress(&self) -> Vec<ProcessingProgress> {
        let active = self.active_progress.read().await;
        active
            .values()
            .map(|state| self.create_processing_progress(state))
            .collect()
    }

    /// Get historical progress data
    pub async fn get_history(&self, job_id: JobId) -> Option<CompletedProgress> {
        let history = self.history.read().await;
        history.get(&job_id).cloned()
    }

    /// Get all historical progress data
    pub async fn get_all_history(&self) -> Vec<CompletedProgress> {
        let history = self.history.read().await;
        history.values().cloned().collect()
    }

    /// Subscribe to progress updates
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressUpdate> {
        self.progress_sender.subscribe()
    }

    /// Get progress statistics
    pub async fn get_statistics(&self) -> ProgressStatistics {
        let active = self.active_progress.read().await;
        let history = self.history.read().await;

        let active_jobs = active.len();
        let completed_jobs = history.len();
        
        let average_processing_speed = if !history.is_empty() {
            history.values().map(|h| h.average_speed).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };

        let average_completion_time = if !history.is_empty() {
            let total_duration: Duration = history.values().map(|h| h.total_duration).sum();
            total_duration / history.len() as u32
        } else {
            Duration::ZERO
        };

        ProgressStatistics {
            active_jobs,
            completed_jobs,
            average_processing_speed,
            average_completion_time,
        }
    }

    /// Helper method to create ProcessingProgress from ProgressState
    fn create_processing_progress(&self, state: &ProgressState) -> ProcessingProgress {
        let processing_speed = if !state.processing_speeds.is_empty() {
            state.processing_speeds.iter().sum::<f64>() / state.processing_speeds.len() as f64
        } else {
            0.0
        };

        let estimated_time_remaining = if let Some(completion_time) = state.estimated_completion {
            let now = Utc::now();
            if completion_time > now {
                Some((completion_time - now).to_std().unwrap_or(Duration::ZERO))
            } else {
                None
            }
        } else {
            None
        };

        ProcessingProgress {
            job_id: state.job_id,
            current_operation: state.current_operation.clone(),
            operations_completed: state.operations_completed,
            total_operations: state.total_operations,
            bytes_processed: state.bytes_processed,
            total_bytes: state.total_bytes,
            estimated_time_remaining,
            processing_speed,
        }
    }

    /// Update estimated completion time based on current progress
    fn update_estimated_completion(&self, state: &mut ProgressState) {
        if state.total_operations == 0 || state.processing_speeds.is_empty() {
            return;
        }

        let operations_remaining = state.total_operations.saturating_sub(state.operations_completed);
        let bytes_remaining = state.total_bytes.saturating_sub(state.bytes_processed);
        
        if operations_remaining == 0 && bytes_remaining == 0 {
            state.estimated_completion = Some(Utc::now());
            return;
        }

        let average_speed = state.processing_speeds.iter().sum::<f64>() / state.processing_speeds.len() as f64;
        
        if average_speed > 0.0 {
            let estimated_seconds = if bytes_remaining > 0 {
                (bytes_remaining as f64) / (1024.0 * 1024.0) / average_speed
            } else {
                // Estimate based on operations if no byte data
                let elapsed_seconds = state.start_time.elapsed().as_secs_f64();
                let operations_completed = state.operations_completed as f64;
                if operations_completed > 0.0 {
                    (operations_remaining as f64) * (elapsed_seconds / operations_completed)
                } else {
                    60.0 // Default estimate
                }
            };

            let estimated_duration = Duration::from_secs_f64(estimated_seconds);
            state.estimated_completion = Some(Utc::now() + chrono::Duration::from_std(estimated_duration).unwrap_or_default());
        }
    }
}

/// Progress statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressStatistics {
    pub active_jobs: usize,
    pub completed_jobs: usize,
    pub average_processing_speed: f64, // MB/s
    pub average_completion_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(100);
        let stats = tracker.get_statistics().await;
        
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.completed_jobs, 0);
    }

    #[tokio::test]
    async fn test_job_lifecycle() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        // Start job
        tracker.start_job(job_id, 3, 1000).await.unwrap();
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        assert_eq!(progress.job_id, job_id);
        assert_eq!(progress.operations_completed, 0);
        assert_eq!(progress.total_operations, 3);
        
        // Update operation
        tracker.update_operation(job_id, "Processing".to_string()).await.unwrap();
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        assert_eq!(progress.current_operation, "Processing");
        
        // Complete operations
        tracker.complete_operation(job_id).await.unwrap();
        tracker.update_bytes_processed(job_id, 500).await.unwrap();
        tracker.complete_operation(job_id).await.unwrap();
        tracker.update_bytes_processed(job_id, 1000).await.unwrap();
        tracker.complete_operation(job_id).await.unwrap();
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        assert_eq!(progress.operations_completed, 3);
        assert_eq!(progress.bytes_processed, 1000);
        
        // Complete job
        tracker.complete_job(job_id, CompletionStatus::Success).await.unwrap();
        
        // Should no longer be in active progress
        assert!(tracker.get_progress(job_id).await.is_none());
        
        // Should be in history
        let history = tracker.get_history(job_id).await.unwrap();
        assert!(matches!(history.final_status, CompletionStatus::Success));
    }

    #[tokio::test]
    async fn test_progress_updates_broadcast() {
        let tracker = ProgressTracker::new(100);
        let mut receiver = tracker.subscribe();
        let job_id = Uuid::new_v4();
        
        // Start job in background task
        let tracker_clone = Arc::new(tracker);
        let tracker_task = tracker_clone.clone();
        tokio::spawn(async move {
            sleep(TokioDuration::from_millis(10)).await;
            tracker_task.start_job(job_id, 1, 100).await.unwrap();
        });
        
        // Receive the update
        let update = receiver.recv().await.unwrap();
        assert_eq!(update.job_id, job_id);
        assert!(matches!(update.update_type, ProgressUpdateType::Started));
    }

    #[tokio::test]
    async fn test_progress_calculation() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        tracker.start_job(job_id, 4, 2000).await.unwrap();
        tracker.complete_operation(job_id).await.unwrap();
        tracker.update_bytes_processed(job_id, 500).await.unwrap();
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        let overall = progress.overall_progress();
        
        // Should be weighted: (1/4 * 0.7) + (500/2000 * 0.3) = 0.175 + 0.075 = 0.25
        assert!((overall - 0.25).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_history_cleanup() {
        let tracker = ProgressTracker::new(2); // Small limit for testing
        
        // Add 3 jobs to exceed the limit
        for i in 0..3 {
            let job_id = Uuid::new_v4();
            tracker.start_job(job_id, 1, 100).await.unwrap();
            tracker.complete_job(job_id, CompletionStatus::Success).await.unwrap();
            
            // Small delay to ensure different timestamps
            sleep(TokioDuration::from_millis(1)).await;
        }
        
        let stats = tracker.get_statistics().await;
        assert_eq!(stats.completed_jobs, 2); // Should be limited to 2
    }

    #[tokio::test]
    async fn test_error_handling() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        // Try to update non-existent job
        let result = tracker.update_operation(job_id, "Test".to_string()).await;
        assert!(result.is_err());
        
        let result = tracker.update_bytes_processed(job_id, 100).await;
        assert!(result.is_err());
        
        let result = tracker.complete_operation(job_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_progress_timing_accuracy() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        let start_time = std::time::Instant::now();
        tracker.start_job(job_id, 4, 1000).await.unwrap();
        
        // Simulate processing with known timing
        for i in 0..4 {
            sleep(TokioDuration::from_millis(50)).await;
            tracker.update_bytes_processed(job_id, (i + 1) * 250).await.unwrap();
            tracker.complete_operation(job_id).await.unwrap();
        }
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        let elapsed = start_time.elapsed();
        
        // Verify progress is complete
        assert_eq!(progress.operations_completed, 4);
        assert_eq!(progress.bytes_processed, 1000);
        
        // Verify timing is reasonable (should be around 200ms + overhead)
        assert!(elapsed >= Duration::from_millis(180));
        assert!(elapsed <= Duration::from_millis(300));
        
        // Verify processing speed calculation
        assert!(progress.processing_speed > 0.0);
    }

    #[tokio::test]
    async fn test_concurrent_progress_updates() {
        let tracker = Arc::new(ProgressTracker::new(100));
        let mut handles = Vec::new();
        
        // Start multiple jobs concurrently
        for i in 0..5 {
            let tracker_clone = tracker.clone();
            let handle = tokio::spawn(async move {
                let job_id = Uuid::new_v4();
                tracker_clone.start_job(job_id, 3, 300).await.unwrap();
                
                // Update progress concurrently
                for j in 0..3 {
                    sleep(TokioDuration::from_millis(10)).await;
                    tracker_clone.update_bytes_processed(job_id, (j + 1) * 100).await.unwrap();
                    tracker_clone.complete_operation(job_id).await.unwrap();
                }
                
                tracker_clone.complete_job(job_id, CompletionStatus::Success).await.unwrap();
                job_id
            });
            handles.push(handle);
        }
        
        // Wait for all jobs to complete
        let mut job_ids = Vec::new();
        for handle in handles {
            job_ids.push(handle.await.unwrap());
        }
        
        // Verify all jobs completed successfully
        let stats = tracker.get_statistics().await;
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.completed_jobs, 5);
        
        // Verify each job's history
        for job_id in job_ids {
            let history = tracker.get_history(job_id).await.unwrap();
            assert!(matches!(history.final_status, CompletionStatus::Success));
            assert!(history.total_duration > Duration::ZERO);
        }
    }

    #[tokio::test]
    async fn test_progress_estimation_accuracy() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        tracker.start_job(job_id, 10, 1000).await.unwrap();
        
        // Process first half with consistent timing
        for i in 0..5 {
            sleep(TokioDuration::from_millis(20)).await;
            tracker.update_bytes_processed(job_id, (i + 1) * 100).await.unwrap();
            tracker.complete_operation(job_id).await.unwrap();
        }
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        
        // Should have reasonable time estimation
        if let Some(eta) = progress.estimated_time_remaining {
            // Should estimate remaining time based on current speed
            assert!(eta > Duration::from_millis(50));
            assert!(eta < Duration::from_millis(200));
        }
        
        // Verify progress calculation
        let overall = progress.overall_progress();
        assert!(overall >= 0.4 && overall <= 0.6); // Should be around 50%
    }

    #[tokio::test]
    async fn test_progress_broadcast_reliability() {
        let tracker = Arc::new(ProgressTracker::new(100));
        let mut receiver = tracker.subscribe();
        let job_id = Uuid::new_v4();
        
        // Start job and collect updates
        let tracker_clone = tracker.clone();
        let update_handle = tokio::spawn(async move {
            tracker_clone.start_job(job_id, 3, 300).await.unwrap();
            
            for i in 0..3 {
                sleep(TokioDuration::from_millis(10)).await;
                tracker_clone.update_operation(job_id, format!("Operation {}", i + 1)).await.unwrap();
                tracker_clone.update_bytes_processed(job_id, (i + 1) * 100).await.unwrap();
                tracker_clone.complete_operation(job_id).await.unwrap();
            }
            
            tracker_clone.complete_job(job_id, CompletionStatus::Success).await.unwrap();
        });
        
        // Collect all progress updates
        let mut updates = Vec::new();
        let mut update_types = Vec::new();
        
        // Use timeout to avoid hanging if updates don't come
        while updates.len() < 8 { // Expect: start + 3*(operation + progress + complete) + final
            match tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await {
                Ok(Ok(update)) => {
                    update_types.push(update.update_type.clone());
                    updates.push(update);
                }
                _ => break,
            }
        }
        
        update_handle.await.unwrap();
        
        // Verify we received the expected update types
        assert!(!updates.is_empty());
        assert!(update_types.contains(&ProgressUpdateType::Started));
        assert!(update_types.contains(&ProgressUpdateType::Completed));
        
        // Verify all updates are for the correct job
        for update in &updates {
            assert_eq!(update.job_id, job_id);
        }
    }

    #[tokio::test]
    async fn test_progress_memory_management() {
        let tracker = ProgressTracker::new(3); // Small limit for testing
        let mut job_ids = Vec::new();
        
        // Create more jobs than the history limit
        for i in 0..5 {
            let job_id = Uuid::new_v4();
            job_ids.push(job_id);
            
            tracker.start_job(job_id, 1, 100).await.unwrap();
            tracker.complete_operation(job_id).await.unwrap();
            tracker.complete_job(job_id, CompletionStatus::Success).await.unwrap();
            
            // Small delay to ensure different timestamps
            sleep(TokioDuration::from_millis(1)).await;
        }
        
        let stats = tracker.get_statistics().await;
        assert_eq!(stats.completed_jobs, 3); // Should be limited to max_history_entries
        
        // Verify oldest jobs were removed
        let history = tracker.get_all_history().await;
        assert_eq!(history.len(), 3);
        
        // Should contain the 3 most recent jobs
        let recent_job_ids: Vec<_> = job_ids.iter().rev().take(3).collect();
        for recent_id in recent_job_ids {
            assert!(history.iter().any(|h| h.job_id == *recent_id));
        }
    }

    #[tokio::test]
    async fn test_progress_edge_cases() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        // Test job with zero operations
        tracker.start_job(job_id, 0, 0).await.unwrap();
        let progress = tracker.get_progress(job_id).await.unwrap();
        assert_eq!(progress.overall_progress(), 0.0);
        
        // Complete job immediately
        tracker.complete_job(job_id, CompletionStatus::Success).await.unwrap();
        
        // Test job with zero bytes
        let job_id2 = Uuid::new_v4();
        tracker.start_job(job_id2, 5, 0).await.unwrap();
        
        for _ in 0..5 {
            tracker.complete_operation(job_id2).await.unwrap();
        }
        
        let progress = tracker.get_progress(job_id2).await.unwrap();
        assert_eq!(progress.overall_progress(), 1.0); // Should be 100% based on operations
        
        tracker.complete_job(job_id2, CompletionStatus::Success).await.unwrap();
    }

    #[tokio::test]
    async fn test_progress_cancellation_handling() {
        let tracker = ProgressTracker::new(100);
        let job_id = Uuid::new_v4();
        
        tracker.start_job(job_id, 5, 500).await.unwrap();
        
        // Process partially
        tracker.update_bytes_processed(job_id, 200).await.unwrap();
        tracker.complete_operation(job_id).await.unwrap();
        tracker.complete_operation(job_id).await.unwrap();
        
        let progress = tracker.get_progress(job_id).await.unwrap();
        assert_eq!(progress.operations_completed, 2);
        assert_eq!(progress.bytes_processed, 200);
        
        // Cancel the job
        tracker.complete_job(job_id, CompletionStatus::Cancelled).await.unwrap();
        
        // Verify job is no longer active
        assert!(tracker.get_progress(job_id).await.is_none());
        
        // Verify job is in history with cancelled status
        let history = tracker.get_history(job_id).await.unwrap();
        assert!(matches!(history.final_status, CompletionStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_progress_statistics_accuracy() {
        let tracker = Arc::new(ProgressTracker::new(100));
        
        // Start multiple jobs with different characteristics
        let mut job_handles = Vec::new();
        for i in 0..3 {
            let tracker_clone = tracker.clone();
            let handle = tokio::spawn(async move {
                let job_id = Uuid::new_v4();
                tracker_clone.start_job(job_id, 2, 200).await.unwrap();
                
                // Simulate different processing speeds
                let delay = (i + 1) * 10;
                sleep(TokioDuration::from_millis(delay)).await;
                tracker_clone.update_bytes_processed(job_id, 100).await.unwrap();
                tracker_clone.complete_operation(job_id).await.unwrap();
                
                sleep(TokioDuration::from_millis(delay)).await;
                tracker_clone.update_bytes_processed(job_id, 200).await.unwrap();
                tracker_clone.complete_operation(job_id).await.unwrap();
                
                tracker_clone.complete_job(job_id, CompletionStatus::Success).await.unwrap();
                job_id
            });
            job_handles.push(handle);
        }
        
        // Wait for all jobs to complete
        for handle in job_handles {
            handle.await.unwrap();
        }
        
        let stats = tracker.get_statistics().await;
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.completed_jobs, 3);
        assert!(stats.average_processing_speed >= 0.0);
        assert!(stats.average_completion_time > Duration::ZERO);
    }
}