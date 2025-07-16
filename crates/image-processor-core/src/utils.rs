//! Utility functions and helpers

use crate::error::{ProcessingError, Result};
use std::path::Path;

/// Check if a file exists and is readable
pub fn validate_file_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(ProcessingError::FileNotFound {
            path: path.to_path_buf(),
        });
    }

    if !path.is_file() {
        return Err(ProcessingError::InvalidInput {
            message: format!("Path is not a file: {}", path.display()),
        });
    }

    Ok(())
}

/// Get file size in bytes
pub fn get_file_size(path: &Path) -> Result<u64> {
    validate_file_path(path)?;
    
    let metadata = std::fs::metadata(path).map_err(ProcessingError::from_io_error)?;
    Ok(metadata.len())
}

/// Format bytes as human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if bytes < THRESHOLD {
        return format!("{} B", bytes);
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Format duration as human readable string
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    
    if total_seconds < 60 {
        format!("{}s", total_seconds)
    } else if total_seconds < 3600 {
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{}m {}s", minutes, seconds)
    } else {
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        format!("{}h {}m {}s", hours, minutes, seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validate_file_path() {
        // Test with non-existent file
        let result = validate_file_path(Path::new("non_existent_file.txt"));
        assert!(result.is_err());
        
        // Test with existing file
        let temp_file = NamedTempFile::new().unwrap();
        let result = validate_file_path(temp_file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_file_size() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), "test content").unwrap();
        
        let size = get_file_size(temp_file.path()).unwrap();
        assert_eq!(size, 12); // "test content" is 12 bytes
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }
}