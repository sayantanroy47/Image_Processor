//! # Image Processor Core
//!
//! High-performance cross-platform image processing library with support for
//! format conversion, watermarking, metadata handling, and batch processing.

pub mod config;
pub mod error;
pub mod logging;
pub mod models;
pub mod processing;
pub mod utils;

// Re-export commonly used types
pub use config::*;
pub use error::*;
pub use logging::*;
pub use models::*;

/// Initialize the image processor core library
pub async fn init() -> Result<(), ProcessingError> {
    // Initialize logging system
    logging::init_logging()?;
    
    tracing::info!("Image Processor Core initialized successfully");
    Ok(())
}

/// Get the version of the image processor core
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init() {
        let result = init().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
    }
}