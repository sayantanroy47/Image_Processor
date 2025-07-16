//! Job queue implementation for managing image processing tasks

use crate::error::{ProcessingError, Result};
use crate::models::{JobId, JobPriority, JobStatus, ProcessingInput, ProcessingJob};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;

/// Job queue for managing processing tasks
#[derive(Debug)]
pub struct JobQueue {
    /// Pending jobs ordered by priority
    pending_jobs: Arc<Mutex<BinaryHeap<QueuedJob>>>,
    /// Active jobs being processed
    active_jobs: Arc<RwLock<HashMap<JobId, ProcessingJob>>>,
    /// Completed jobs (kept for history)
    completed_jobs: Arc<RwLock<HashMap<JobId, ProcessingJob>>>,
    /// Channel for job status updates
    status_sender: mpsc::UnboundedSender<JobStatusUpdate>,
    /// Channel receiver for job status updates
    status_receiver: Arc<Mutex<mpsc::UnboundedReceiver<JobStatusUpdate>>>,
    /// Maximum number of concurrent jobs
    max_concurrent_jobs: usize,
    /// Current number of active jobs
    active_job_count: Arc<Mutex<usize>>,
}

/// Wrapper for jobs in the priority queue
#[derive(Debug, Clone)]
struct QueuedJob {
    job: ProcessingJob,
}

impl PartialEq for QueuedJob {
    fn eq(&self, other: &Self) -> bool {
        self.job.priority == other.job.priority && self.job.created_at == other.job.created_at
    }
}

impl Eq for QueuedJob {}

impl PartialOrd for QueuedJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority jobs come first, then earlier created jobs
        self.job.priority.cmp(&other.job.priority)
            .then_with(|| other.job.created_at.cmp(&self.job.created_at))
    }
}

/// Job status update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusUpdate {
    pub job_id: JobId,
    pub old_status: JobStatus,
    pub new_status: JobStatus,
    pub timestamp: DateTime<Utc>,
    pub message: Option<String>,
}

impl JobQueue {
    /// Create a new job queue
    pub fn new(max_concurrent_jobs: usize) -> Self {
        let (status_sender, status_receiver) = mpsc::unbounded_channel();
        
        Self {
            pending_jobs: Arc::new(Mutex::new(BinaryHeap::new())),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            completed_jobs: Arc::new(RwLock::new(HashMap::new())),
            status_sender,
            status_receiver: Arc::new(Mutex::new(status_receiver)),
            max_concurrent_jobs,
            active_job_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Add a new job to the queue
    pub async fn enqueue_job(&self, input: ProcessingInput, priority: JobPriority) -> Result<JobId> {
        let job_id = Uuid::new_v4();
        let now = Utc::now();
        
        let job = ProcessingJob {
            id: job_id,
            input,
            status: JobStatus::Pending,
            priority,
            created_at: now,
            started_at: None,
            completed_at: None,
            error: None,
            progress: 0.0,
        };

        let queued_job = QueuedJob { job };
        
        {
            let mut pending = self.pending_jobs.lock().await;
            pending.push(queued_job);
        }

        // Send status update
        let update = JobStatusUpdate {
            job_id,
            old_status: JobStatus::Pending,
            new_status: JobStatus::Pending,
            timestamp: now,
            message: Some("Job added to queue".to_string()),
        };
        
        if let Err(e) = self.status_sender.send(update) {
            tracing::warn!("Failed to send job status update: {}", e);
        }

        tracing::info!("Job {} enqueued with priority {:?}", job_id, priority);
        Ok(job_id)
    }

    /// Get the next job to process (highest priority, oldest first)
    pub async fn dequeue_job(&self) -> Option<ProcessingJob> {
        let active_count = {
            let count = self.active_job_count.lock().await;
            *count
        };

        if active_count >= self.max_concurrent_jobs {
            return None;
        }

        let mut pending = self.pending_jobs.lock().await;
        if let Some(queued_job) = pending.pop() {
            let mut job = queued_job.job;
            job.status = JobStatus::Running;
            job.started_at = Some(Utc::now());

            // Move to active jobs
            {
                let mut active = self.active_jobs.write().await;
                active.insert(job.id, job.clone());
            }

            // Increment active job count
            {
                let mut count = self.active_job_count.lock().await;
                *count += 1;
            }

            // Send status update
            let update = JobStatusUpdate {
                job_id: job.id,
                old_status: JobStatus::Pending,
                new_status: JobStatus::Running,
                timestamp: job.started_at.unwrap(),
                message: Some("Job started processing".to_string()),
            };
            
            if let Err(e) = self.status_sender.send(update) {
                tracing::warn!("Failed to send job status update: {}", e);
            }

            tracing::info!("Job {} dequeued and started", job.id);
            Some(job)
        } else {
            None
        }
    }

    /// Mark a job as completed
    pub async fn complete_job(&self, job_id: JobId, error: Option<ProcessingError>) -> Result<()> {
        let mut job = {
            let mut active = self.active_jobs.write().await;
            active.remove(&job_id)
                .ok_or_else(|| ProcessingError::InvalidInput {
                    message: format!("Job {} not found in active jobs", job_id),
                })?
        };

        let now = Utc::now();
        job.completed_at = Some(now);
        job.progress = 1.0;

        let (old_status, new_status) = if let Some(err) = error {
            job.status = JobStatus::Failed;
            job.error = Some(err.to_string());
            (JobStatus::Running, JobStatus::Failed)
        } else {
            job.status = JobStatus::Completed;
            (JobStatus::Running, JobStatus::Completed)
        };

        // Move to completed jobs
        {
            let mut completed = self.completed_jobs.write().await;
            completed.insert(job_id, job);
        }

        // Decrement active job count
        {
            let mut count = self.active_job_count.lock().await;
            *count = count.saturating_sub(1);
        }

        // Send status update
        let update = JobStatusUpdate {
            job_id,
            old_status,
            new_status,
            timestamp: now,
            message: if new_status == JobStatus::Failed {
                Some("Job failed".to_string())
            } else {
                Some("Job completed successfully".to_string())
            },
        };
        
        if let Err(e) = self.status_sender.send(update) {
            tracing::warn!("Failed to send job status update: {}", e);
        }

        tracing::info!("Job {} completed with status {:?}", job_id, new_status);
        Ok(())
    }

    /// Cancel a pending or active job
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        // Try to remove from pending jobs first
        {
            let mut pending = self.pending_jobs.lock().await;
            let mut temp_heap = BinaryHeap::new();
            let mut found_job: Option<ProcessingJob> = None;

            while let Some(queued_job) = pending.pop() {
                if queued_job.job.id == job_id {
                    found_job = Some(queued_job.job);
                    break;
                } else {
                    temp_heap.push(queued_job);
                }
            }

            // Restore the heap without the cancelled job
            *pending = temp_heap;

            if let Some(mut cancelled_job) = found_job {
                cancelled_job.status = JobStatus::Cancelled;
                cancelled_job.completed_at = Some(Utc::now());

                // Move to completed jobs
                {
                    let mut completed = self.completed_jobs.write().await;
                    completed.insert(job_id, cancelled_job);
                }

                let update = JobStatusUpdate {
                    job_id,
                    old_status: JobStatus::Pending,
                    new_status: JobStatus::Cancelled,
                    timestamp: Utc::now(),
                    message: Some("Job cancelled while pending".to_string()),
                };
                
                if let Err(e) = self.status_sender.send(update) {
                    tracing::warn!("Failed to send job status update: {}", e);
                }

                tracing::info!("Pending job {} cancelled", job_id);
                return Ok(());
            }
        }

        // Try to cancel active job
        {
            let mut active = self.active_jobs.write().await;
            if let Some(mut job) = active.remove(&job_id) {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(Utc::now());

                // Move to completed jobs
                {
                    let mut completed = self.completed_jobs.write().await;
                    completed.insert(job_id, job);
                }

                // Decrement active job count
                {
                    let mut count = self.active_job_count.lock().await;
                    *count = count.saturating_sub(1);
                }

                let update = JobStatusUpdate {
                    job_id,
                    old_status: JobStatus::Running,
                    new_status: JobStatus::Cancelled,
                    timestamp: Utc::now(),
                    message: Some("Job cancelled while running".to_string()),
                };
                
                if let Err(e) = self.status_sender.send(update) {
                    tracing::warn!("Failed to send job status update: {}", e);
                }

                tracing::info!("Active job {} cancelled", job_id);
                return Ok(());
            }
        }

        Err(ProcessingError::InvalidInput {
            message: format!("Job {} not found", job_id),
        })
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: JobId) -> Option<JobStatus> {
        // Check active jobs
        {
            let active = self.active_jobs.read().await;
            if let Some(job) = active.get(&job_id) {
                return Some(job.status);
            }
        }

        // Check completed jobs
        {
            let completed = self.completed_jobs.read().await;
            if let Some(job) = completed.get(&job_id) {
                return Some(job.status);
            }
        }

        // Check pending jobs
        {
            let pending = self.pending_jobs.lock().await;
            for queued_job in pending.iter() {
                if queued_job.job.id == job_id {
                    return Some(queued_job.job.status);
                }
            }
        }

        None
    }

    /// Get job details
    pub async fn get_job(&self, job_id: JobId) -> Option<ProcessingJob> {
        // Check active jobs
        {
            let active = self.active_jobs.read().await;
            if let Some(job) = active.get(&job_id) {
                return Some(job.clone());
            }
        }

        // Check completed jobs
        {
            let completed = self.completed_jobs.read().await;
            if let Some(job) = completed.get(&job_id) {
                return Some(job.clone());
            }
        }

        // Check pending jobs
        {
            let pending = self.pending_jobs.lock().await;
            for queued_job in pending.iter() {
                if queued_job.job.id == job_id {
                    return Some(queued_job.job.clone());
                }
            }
        }

        None
    }

    /// Get queue statistics
    pub async fn get_stats(&self) -> QueueStats {
        let pending_count = {
            let pending = self.pending_jobs.lock().await;
            pending.len()
        };

        let active_count = {
            let active = self.active_jobs.read().await;
            active.len()
        };

        let completed_count = {
            let completed = self.completed_jobs.read().await;
            completed.len()
        };

        QueueStats {
            pending_jobs: pending_count,
            active_jobs: active_count,
            completed_jobs: completed_count,
            max_concurrent_jobs: self.max_concurrent_jobs,
        }
    }

    /// Get a clone of the status update sender
    pub fn status_sender(&self) -> mpsc::UnboundedSender<JobStatusUpdate> {
        self.status_sender.clone()
    }

    /// Update job progress
    pub async fn update_job_progress(&self, job_id: JobId, progress: f32) -> Result<()> {
        let mut active = self.active_jobs.write().await;
        if let Some(job) = active.get_mut(&job_id) {
            job.progress = progress.clamp(0.0, 1.0);
            
            let update = JobStatusUpdate {
                job_id,
                old_status: JobStatus::Running,
                new_status: JobStatus::Running,
                timestamp: Utc::now(),
                message: Some(format!("Progress: {:.1}%", progress * 100.0)),
            };
            
            if let Err(e) = self.status_sender.send(update) {
                tracing::warn!("Failed to send job progress update: {}", e);
            }
            
            Ok(())
        } else {
            Err(ProcessingError::InvalidInput {
                message: format!("Job {} not found in active jobs", job_id),
            })
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub pending_jobs: usize,
    pub active_jobs: usize,
    pub completed_jobs: usize,
    pub max_concurrent_jobs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ProcessingOperation, ProcessingOptions};
    use std::path::PathBuf;

    fn create_test_input() -> ProcessingInput {
        ProcessingInput {
            job_id: Uuid::new_v4(),
            source_path: PathBuf::from("test.jpg"),
            output_path: PathBuf::from("output.jpg"),
            operations: vec![ProcessingOperation::Convert {
                format: crate::config::ImageFormat::Png,
                quality: Some(85),
            }],
            options: ProcessingOptions::default(),
            file_size: 1024,
            format: crate::config::ImageFormat::Jpeg,
        }
    }

    #[tokio::test]
    async fn test_job_queue_creation() {
        let queue = JobQueue::new(4);
        let stats = queue.get_stats().await;
        
        assert_eq!(stats.pending_jobs, 0);
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.completed_jobs, 0);
        assert_eq!(stats.max_concurrent_jobs, 4);
    }

    #[tokio::test]
    async fn test_enqueue_job() {
        let queue = JobQueue::new(4);
        let input = create_test_input();
        
        let job_id = queue.enqueue_job(input, JobPriority::Normal).await.unwrap();
        
        let stats = queue.get_stats().await;
        assert_eq!(stats.pending_jobs, 1);
        
        let status = queue.get_job_status(job_id).await;
        assert_eq!(status, Some(JobStatus::Pending));
    }

    #[tokio::test]
    async fn test_dequeue_job() {
        let queue = JobQueue::new(4);
        let input = create_test_input();
        
        let job_id = queue.enqueue_job(input, JobPriority::High).await.unwrap();
        let job = queue.dequeue_job().await.unwrap();
        
        assert_eq!(job.id, job_id);
        assert_eq!(job.status, JobStatus::Running);
        assert!(job.started_at.is_some());
        
        let stats = queue.get_stats().await;
        assert_eq!(stats.pending_jobs, 0);
        assert_eq!(stats.active_jobs, 1);
    }

    #[tokio::test]
    async fn test_job_priority_ordering() {
        let queue = JobQueue::new(4);
        
        // Add jobs with different priorities
        let _low_id = queue.enqueue_job(create_test_input(), JobPriority::Low).await.unwrap();
        let high_id = queue.enqueue_job(create_test_input(), JobPriority::High).await.unwrap();
        let _normal_id = queue.enqueue_job(create_test_input(), JobPriority::Normal).await.unwrap();
        
        // High priority job should be dequeued first
        let job = queue.dequeue_job().await.unwrap();
        assert_eq!(job.id, high_id);
        assert_eq!(job.priority, JobPriority::High);
    }

    #[tokio::test]
    async fn test_complete_job() {
        let queue = JobQueue::new(4);
        let input = create_test_input();
        
        let job_id = queue.enqueue_job(input, JobPriority::Normal).await.unwrap();
        let _job = queue.dequeue_job().await.unwrap();
        
        queue.complete_job(job_id, None).await.unwrap();
        
        let stats = queue.get_stats().await;
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.completed_jobs, 1);
        
        let status = queue.get_job_status(job_id).await;
        assert_eq!(status, Some(JobStatus::Completed));
    }

    #[tokio::test]
    async fn test_cancel_pending_job() {
        let queue = JobQueue::new(4);
        let input = create_test_input();
        
        let job_id = queue.enqueue_job(input, JobPriority::Normal).await.unwrap();
        queue.cancel_job(job_id).await.unwrap();
        
        let stats = queue.get_stats().await;
        assert_eq!(stats.pending_jobs, 0);
        assert_eq!(stats.completed_jobs, 1);
        
        let status = queue.get_job_status(job_id).await;
        assert_eq!(status, Some(JobStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_max_concurrent_jobs() {
        let queue = JobQueue::new(2);
        
        // Add 3 jobs
        let _id1 = queue.enqueue_job(create_test_input(), JobPriority::Normal).await.unwrap();
        let _id2 = queue.enqueue_job(create_test_input(), JobPriority::Normal).await.unwrap();
        let _id3 = queue.enqueue_job(create_test_input(), JobPriority::Normal).await.unwrap();
        
        // Should be able to dequeue 2 jobs
        assert!(queue.dequeue_job().await.is_some());
        assert!(queue.dequeue_job().await.is_some());
        
        // Third job should not be dequeued due to max concurrent limit
        assert!(queue.dequeue_job().await.is_none());
        
        let stats = queue.get_stats().await;
        assert_eq!(stats.pending_jobs, 1);
        assert_eq!(stats.active_jobs, 2);
    }

    #[tokio::test]
    async fn test_job_progress_update() {
        let queue = JobQueue::new(4);
        let input = create_test_input();
        
        let job_id = queue.enqueue_job(input, JobPriority::Normal).await.unwrap();
        let _job = queue.dequeue_job().await.unwrap();
        
        queue.update_job_progress(job_id, 0.5).await.unwrap();
        
        let job = queue.get_job(job_id).await.unwrap();
        assert_eq!(job.progress, 0.5);
    }
}