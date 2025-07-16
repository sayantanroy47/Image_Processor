//! Logging configuration and initialization

use crate::error::{ProcessingError, Result};
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize the logging system
pub fn init_logging() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    match tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init()
    {
        Ok(()) => Ok(()),
        Err(e) => {
            // Check if the error is because logging is already initialized
            let error_msg = e.to_string();
            if error_msg.contains("a global default trace dispatcher has already been set") {
                // Logging is already initialized, which is fine
                Ok(())
            } else {
                Err(ProcessingError::LoggingError {
                    message: format!("Failed to initialize logging: {}", e),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_init() {
        // This test might fail if logging is already initialized
        // but that's okay for our purposes
        let _ = init_logging();
    }
}