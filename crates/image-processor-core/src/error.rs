//! Error types for the image processing library

use std::path::PathBuf;

/// Main error type for image processing operations
#[derive(Debug, Clone, thiserror::Error, serde::Serialize, serde::Deserialize)]
pub enum ProcessingError {
    #[error("IO error: {message}")]
    Io { message: String },
    
    #[error("Image format not supported: {format}")]
    UnsupportedFormat { format: String },
    
    #[error("Processing failed: {message}")]
    ProcessingFailed { message: String },
    
    #[error("Insufficient memory for operation")]
    OutOfMemory,
    
    #[error("Hardware acceleration not available")]
    HardwareAccelerationUnavailable,
    
    #[error("Background removal model not found: {model}")]
    ModelNotFound { model: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Logging initialization failed: {message}")]
    LoggingError { message: String },
    
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    #[error("Operation cancelled")]
    Cancelled,
    
    #[error("Operation cancelled: {message}")]
    OperationCancelled { message: String },
    
    #[error("Timeout occurred during operation")]
    Timeout,
    
    #[error("Database error: {message}")]
    Database { message: String },
    
    #[error("Serialization error: {message}")]
    Serialization { message: String },
}

impl ProcessingError {
    /// Create an IO error from std::io::Error
    pub fn from_io_error(error: std::io::Error) -> Self {
        ProcessingError::Io {
            message: error.to_string(),
        }
    }

    /// Create a database error from sqlx::Error
    pub fn from_database_error(error: sqlx::Error) -> Self {
        ProcessingError::Database {
            message: error.to_string(),
        }
    }

    /// Create a serialization error from serde_json::Error
    pub fn from_serialization_error(error: serde_json::Error) -> Self {
        ProcessingError::Serialization {
            message: error.to_string(),
        }
    }

    /// Get the error type as a string for categorization
    pub fn error_type(&self) -> &'static str {
        match self {
            ProcessingError::Io { .. } => "io_error",
            ProcessingError::UnsupportedFormat { .. } => "unsupported_format",
            ProcessingError::ProcessingFailed { .. } => "processing_failed",
            ProcessingError::OutOfMemory => "out_of_memory",
            ProcessingError::HardwareAccelerationUnavailable => "hardware_acceleration_unavailable",
            ProcessingError::ModelNotFound { .. } => "model_not_found",
            ProcessingError::ConfigError { .. } => "config_error",
            ProcessingError::LoggingError { .. } => "logging_error",
            ProcessingError::FileNotFound { .. } => "file_not_found",
            ProcessingError::InvalidInput { .. } => "invalid_input",
            ProcessingError::Cancelled => "cancelled",
            ProcessingError::OperationCancelled { .. } => "operation_cancelled",
            ProcessingError::Timeout => "timeout",
            ProcessingError::Database { .. } => "database_error",
            ProcessingError::Serialization { .. } => "serialization_error",
        }
    }
    
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ProcessingError::Timeout
                | ProcessingError::HardwareAccelerationUnavailable
                | ProcessingError::OutOfMemory
        )
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ProcessingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        let error = ProcessingError::UnsupportedFormat {
            format: "xyz".to_string(),
        };
        assert_eq!(error.error_type(), "unsupported_format");
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_recoverable_errors() {
        let error = ProcessingError::Timeout;
        assert!(error.is_recoverable());
        
        let error = ProcessingError::ProcessingFailed {
            message: "test".to_string(),
        };
        assert!(!error.is_recoverable());
    }
}