//! Database management for job tracking and metadata storage

use crate::error::{ProcessingError, Result};
use crate::models::{JobId, JobPriority, JobStatus, ProcessingJob, ProcessingInput};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{migrate::MigrateDatabase, Pool, Row, Sqlite, SqlitePool};
use std::path::Path;
use uuid::Uuid;

/// Database manager for job tracking and metadata storage
#[derive(Debug, Clone)]
pub struct DatabaseManager {
    pool: SqlitePool,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub database_url: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub enable_wal_mode: bool,
    pub enable_foreign_keys: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            database_url: "sqlite:./image_processor.db".to_string(),
            max_connections: 10,
            connection_timeout_seconds: 30,
            enable_wal_mode: true,
            enable_foreign_keys: true,
        }
    }
}

/// Job record for database storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub input_data: String, // JSON serialized ProcessingInput
    pub status: String,
    pub priority: i32,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub progress: f32,
    pub retry_count: i32,
}

/// Metadata record for database storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRecord {
    pub id: String,
    pub job_id: String,
    pub key: String,
    pub value: String,
    pub created_at: DateTime<Utc>,
}

/// Processing statistics record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub total_jobs: i64,
    pub completed_jobs: i64,
    pub failed_jobs: i64,
    pub cancelled_jobs: i64,
    pub average_processing_time_seconds: f64,
    pub total_bytes_processed: i64,
}

impl DatabaseManager {
    /// Create a new database manager with the given configuration
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        // Create database if it doesn't exist
        if !Sqlite::database_exists(&config.database_url).await.unwrap_or(false) {
            Sqlite::create_database(&config.database_url)
                .await
                .map_err(|e| ProcessingError::Database {
                    message: format!("Failed to create database: {}", e),
                })?;
        }

        // Create connection pool
        let pool = SqlitePool::connect_with(
            sqlx::sqlite::SqliteConnectOptions::new()
                .filename(&config.database_url.replace("sqlite:", ""))
                .create_if_missing(true)
                .journal_mode(if config.enable_wal_mode {
                    sqlx::sqlite::SqliteJournalMode::Wal
                } else {
                    sqlx::sqlite::SqliteJournalMode::Delete
                })
                .foreign_keys(config.enable_foreign_keys),
        )
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to connect to database: {}", e),
        })?;

        let manager = Self { pool };

        // Run migrations
        manager.run_migrations().await?;

        Ok(manager)
    }

    /// Run database migrations
    pub async fn run_migrations(&self) -> Result<()> {
        // Create jobs table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                input_data TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                progress REAL NOT NULL DEFAULT 0.0,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to create jobs table: {}", e),
        })?;

        // Create metadata table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS job_metadata (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to create job_metadata table: {}", e),
        })?;

        // Create processing_history table for analytics
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS processing_history (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                bytes_processed INTEGER,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to create processing_history table: {}", e),
        })?;

        // Create indexes for better performance
        self.create_indexes().await?;

        tracing::info!("Database migrations completed successfully");
        Ok(())
    }

    /// Create database indexes for better performance
    async fn create_indexes(&self) -> Result<()> {
        let indexes = vec![
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs (priority)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_job_id ON job_metadata (job_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_key ON job_metadata (key)",
            "CREATE INDEX IF NOT EXISTS idx_history_job_id ON processing_history (job_id)",
            "CREATE INDEX IF NOT EXISTS idx_history_operation ON processing_history (operation_name)",
        ];

        for index_sql in indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| ProcessingError::Database {
                    message: format!("Failed to create index: {}", e),
                })?;
        }

        Ok(())
    }

    /// Insert a new job record
    pub async fn insert_job(&self, job: &ProcessingJob) -> Result<()> {
        let input_json = serde_json::to_string(&job.input)
            .map_err(|e| ProcessingError::Serialization {
                message: format!("Failed to serialize job input: {}", e),
            })?;

        sqlx::query(
            r#"
            INSERT INTO jobs (id, input_data, status, priority, created_at, started_at, completed_at, error_message, progress, retry_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
        )
        .bind(job.id.to_string())
        .bind(input_json)
        .bind(job.status.to_string())
        .bind(job.priority as i32)
        .bind(job.created_at.to_rfc3339())
        .bind(job.started_at.map(|dt| dt.to_rfc3339()))
        .bind(job.completed_at.map(|dt| dt.to_rfc3339()))
        .bind(&job.error)
        .bind(job.progress)
        .bind(0) // retry_count
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to insert job: {}", e),
        })?;

        Ok(())
    }

    /// Update job status and progress
    pub async fn update_job_status(
        &self,
        job_id: JobId,
        status: JobStatus,
        progress: f32,
        error_message: Option<String>,
    ) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(
            r#"
            UPDATE jobs 
            SET status = ?1, progress = ?2, error_message = ?3,
                started_at = CASE WHEN ?1 = 'Running' AND started_at IS NULL THEN ?4 ELSE started_at END,
                completed_at = CASE WHEN ?1 IN ('Completed', 'Failed', 'Cancelled') THEN ?4 ELSE completed_at END
            WHERE id = ?5
            "#,
        )
        .bind(status.to_string())
        .bind(progress)
        .bind(error_message)
        .bind(now.to_rfc3339())
        .bind(job_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to update job status: {}", e),
        })?;

        Ok(())
    }

    /// Get job by ID
    pub async fn get_job(&self, job_id: JobId) -> Result<Option<JobRecord>> {
        let row = sqlx::query(
            "SELECT id, input_data, status, priority, created_at, started_at, completed_at, error_message, progress, retry_count FROM jobs WHERE id = ?1"
        )
        .bind(job_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to get job: {}", e),
        })?;

        if let Some(row) = row {
            Ok(Some(JobRecord {
                id: row.get("id"),
                input_data: row.get("input_data"),
                status: row.get("status"),
                priority: row.get("priority"),
                created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse created_at: {}", e),
                    })?
                    .with_timezone(&Utc),
                started_at: row.get::<Option<String>, _>("started_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s))
                    .transpose()
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse started_at: {}", e),
                    })?
                    .map(|dt| dt.with_timezone(&Utc)),
                completed_at: row.get::<Option<String>, _>("completed_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s))
                    .transpose()
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse completed_at: {}", e),
                    })?
                    .map(|dt| dt.with_timezone(&Utc)),
                error_message: row.get("error_message"),
                progress: row.get("progress"),
                retry_count: row.get("retry_count"),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get jobs by status
    pub async fn get_jobs_by_status(&self, status: JobStatus) -> Result<Vec<JobRecord>> {
        let rows = sqlx::query(
            "SELECT id, input_data, status, priority, created_at, started_at, completed_at, error_message, progress, retry_count FROM jobs WHERE status = ?1 ORDER BY created_at DESC"
        )
        .bind(status.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to get jobs by status: {}", e),
        })?;

        let mut jobs = Vec::new();
        for row in rows {
            jobs.push(JobRecord {
                id: row.get("id"),
                input_data: row.get("input_data"),
                status: row.get("status"),
                priority: row.get("priority"),
                created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse created_at: {}", e),
                    })?
                    .with_timezone(&Utc),
                started_at: row.get::<Option<String>, _>("started_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s))
                    .transpose()
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse started_at: {}", e),
                    })?
                    .map(|dt| dt.with_timezone(&Utc)),
                completed_at: row.get::<Option<String>, _>("completed_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s))
                    .transpose()
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse completed_at: {}", e),
                    })?
                    .map(|dt| dt.with_timezone(&Utc)),
                error_message: row.get("error_message"),
                progress: row.get("progress"),
                retry_count: row.get("retry_count"),
            });
        }

        Ok(jobs)
    }

    /// Delete job and related data
    pub async fn delete_job(&self, job_id: JobId) -> Result<()> {
        sqlx::query("DELETE FROM jobs WHERE id = ?1")
            .bind(job_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| ProcessingError::Database {
                message: format!("Failed to delete job: {}", e),
            })?;

        Ok(())
    }

    /// Insert job metadata
    pub async fn insert_metadata(&self, job_id: JobId, key: &str, value: &str) -> Result<()> {
        let metadata_id = Uuid::new_v4();
        
        sqlx::query(
            "INSERT INTO job_metadata (id, job_id, key, value, created_at) VALUES (?1, ?2, ?3, ?4, ?5)"
        )
        .bind(metadata_id.to_string())
        .bind(job_id.to_string())
        .bind(key)
        .bind(value)
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to insert metadata: {}", e),
        })?;

        Ok(())
    }

    /// Get job metadata
    pub async fn get_metadata(&self, job_id: JobId) -> Result<Vec<MetadataRecord>> {
        let rows = sqlx::query(
            "SELECT id, job_id, key, value, created_at FROM job_metadata WHERE job_id = ?1"
        )
        .bind(job_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to get metadata: {}", e),
        })?;

        let mut metadata = Vec::new();
        for row in rows {
            metadata.push(MetadataRecord {
                id: row.get("id"),
                job_id: row.get("job_id"),
                key: row.get("key"),
                value: row.get("value"),
                created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                    .map_err(|e| ProcessingError::Database {
                        message: format!("Failed to parse created_at: {}", e),
                    })?
                    .with_timezone(&Utc),
            });
        }

        Ok(metadata)
    }

    /// Get processing statistics
    pub async fn get_processing_stats(&self) -> Result<ProcessingStats> {
        let row = sqlx::query(
            r#"
            SELECT 
                COUNT(*) as total_jobs,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed_jobs,
                SUM(CASE WHEN status = 'Failed' THEN 1 ELSE 0 END) as failed_jobs,
                SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled_jobs,
                AVG(CASE 
                    WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                    THEN (julianday(completed_at) - julianday(started_at)) * 86400 
                    ELSE NULL 
                END) as avg_processing_time
            FROM jobs
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to get processing stats: {}", e),
        })?;

        // Get total bytes processed from processing history
        let bytes_row = sqlx::query(
            "SELECT COALESCE(SUM(bytes_processed), 0) as total_bytes FROM processing_history WHERE success = 1"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to get bytes processed: {}", e),
        })?;

        Ok(ProcessingStats {
            total_jobs: row.get("total_jobs"),
            completed_jobs: row.get("completed_jobs"),
            failed_jobs: row.get("failed_jobs"),
            cancelled_jobs: row.get("cancelled_jobs"),
            average_processing_time_seconds: row.get::<Option<f64>, _>("avg_processing_time").unwrap_or(0.0),
            total_bytes_processed: bytes_row.get("total_bytes"),
        })
    }

    /// Clean up old completed jobs (older than specified days)
    pub async fn cleanup_old_jobs(&self, days_old: u32) -> Result<u64> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_old as i64);
        
        let result = sqlx::query(
            "DELETE FROM jobs WHERE status IN ('Completed', 'Failed', 'Cancelled') AND completed_at < ?1"
        )
        .bind(cutoff_date.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| ProcessingError::Database {
            message: format!("Failed to cleanup old jobs: {}", e),
        })?;

        Ok(result.rows_affected())
    }

    /// Get database connection pool for advanced operations
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Close database connections
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Pending => write!(f, "Pending"),
            JobStatus::Running => write!(f, "Running"),
            JobStatus::Completed => write!(f, "Completed"),
            JobStatus::Failed => write!(f, "Failed"),
            JobStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ProcessingOperation, ProcessingOptions};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_job() -> ProcessingJob {
        ProcessingJob {
            id: Uuid::new_v4(),
            input: ProcessingInput {
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
            },
            status: JobStatus::Pending,
            priority: JobPriority::Normal,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            progress: 0.0,
        }
    }

    #[tokio::test]
    async fn test_database_creation_and_migrations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();
        
        // Verify tables were created
        let tables: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        .fetch_all(db.pool())
        .await
        .unwrap();

        assert!(tables.contains(&"jobs".to_string()));
        assert!(tables.contains(&"job_metadata".to_string()));
        assert!(tables.contains(&"processing_history".to_string()));
    }

    #[tokio::test]
    async fn test_job_crud_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();
        let job = create_test_job();

        // Insert job
        db.insert_job(&job).await.unwrap();

        // Get job
        let retrieved_job = db.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(retrieved_job.id, job.id.to_string());
        assert_eq!(retrieved_job.status, "Pending");

        // Update job status
        db.update_job_status(job.id, JobStatus::Running, 0.5, None).await.unwrap();
        
        let updated_job = db.get_job(job.id).await.unwrap().unwrap();
        assert_eq!(updated_job.status, "Running");
        assert_eq!(updated_job.progress, 0.5);

        // Delete job
        db.delete_job(job.id).await.unwrap();
        let deleted_job = db.get_job(job.id).await.unwrap();
        assert!(deleted_job.is_none());
    }

    #[tokio::test]
    async fn test_metadata_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();
        let job = create_test_job();

        // Insert job first
        db.insert_job(&job).await.unwrap();

        // Insert metadata
        db.insert_metadata(job.id, "test_key", "test_value").await.unwrap();
        db.insert_metadata(job.id, "another_key", "another_value").await.unwrap();

        // Get metadata
        let metadata = db.get_metadata(job.id).await.unwrap();
        assert_eq!(metadata.len(), 2);
        
        let keys: Vec<&str> = metadata.iter().map(|m| m.key.as_str()).collect();
        assert!(keys.contains(&"test_key"));
        assert!(keys.contains(&"another_key"));
    }

    #[tokio::test]
    async fn test_jobs_by_status() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();

        // Insert jobs with different statuses
        let mut jobs = Vec::new();
        for i in 0..5 {
            let mut job = create_test_job();
            job.status = if i < 2 { JobStatus::Pending } else { JobStatus::Completed };
            db.insert_job(&job).await.unwrap();
            jobs.push(job);
        }

        // Get pending jobs
        let pending_jobs = db.get_jobs_by_status(JobStatus::Pending).await.unwrap();
        assert_eq!(pending_jobs.len(), 2);

        // Get completed jobs
        let completed_jobs = db.get_jobs_by_status(JobStatus::Completed).await.unwrap();
        assert_eq!(completed_jobs.len(), 3);
    }

    #[tokio::test]
    async fn test_processing_stats() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();

        // Insert jobs with different statuses
        for i in 0..10 {
            let mut job = create_test_job();
            job.status = match i % 3 {
                0 => JobStatus::Completed,
                1 => JobStatus::Failed,
                _ => JobStatus::Pending,
            };
            if job.status != JobStatus::Pending {
                job.started_at = Some(Utc::now() - chrono::Duration::minutes(5));
                job.completed_at = Some(Utc::now());
            }
            db.insert_job(&job).await.unwrap();
        }

        let stats = db.get_processing_stats().await.unwrap();
        assert_eq!(stats.total_jobs, 10);
        assert!(stats.completed_jobs > 0);
        assert!(stats.failed_jobs > 0);
    }

    #[tokio::test]
    async fn test_cleanup_old_jobs() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();

        // Insert old completed job
        let mut old_job = create_test_job();
        old_job.status = JobStatus::Completed;
        old_job.completed_at = Some(Utc::now() - chrono::Duration::days(10));
        db.insert_job(&old_job).await.unwrap();

        // Insert recent job
        let recent_job = create_test_job();
        db.insert_job(&recent_job).await.unwrap();

        // Cleanup jobs older than 5 days
        let deleted_count = db.cleanup_old_jobs(5).await.unwrap();
        assert_eq!(deleted_count, 1);

        // Verify recent job still exists
        let remaining_job = db.get_job(recent_job.id).await.unwrap();
        assert!(remaining_job.is_some());

        // Verify old job was deleted
        let deleted_job = db.get_job(old_job.id).await.unwrap();
        assert!(deleted_job.is_none());
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let config = DatabaseConfig {
            database_url: format!("sqlite:{}", db_path.display()),
            max_connections: 5,
            ..Default::default()
        };

        let db = DatabaseManager::new(config).await.unwrap();
        
        // Test concurrent operations
        let mut handles = Vec::new();
        for i in 0..10 {
            let db_clone = db.clone();
            let handle = tokio::spawn(async move {
                let mut job = create_test_job();
                job.input.file_size = i * 100;
                db_clone.insert_job(&job).await.unwrap();
                job.id
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let mut job_ids = Vec::new();
        for handle in handles {
            job_ids.push(handle.await.unwrap());
        }

        // Verify all jobs were inserted
        for job_id in job_ids {
            let job = db.get_job(job_id).await.unwrap();
            assert!(job.is_some());
        }
    }
}