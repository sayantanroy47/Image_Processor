//! Utility functions and helpers

use crate::config::ImageFormat;
use crate::error::{ProcessingError, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// File utilities
pub mod file {
    use super::*;

    /// Get the file size in bytes
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(path).map_err(|e| ProcessingError::Io(e))?;
        Ok(metadata.len())
    }

    /// Check if a file exists and is readable
    pub fn is_file_accessible(path: &Path) -> bool {
        path.exists() && path.is_file() && std::fs::File::open(path).is_ok()
    }

    /// Create a backup file path
    pub fn create_backup_path(original_path: &Path, backup_dir: &Path) -> Result<PathBuf> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        let file_name = original_path
            .file_name()
            .ok_or_else(|| ProcessingError::InvalidInput {
                message: "Invalid file path".to_string(),
            })?;

        let backup_name = format!("{}_{}", timestamp, file_name.to_string_lossy());
        Ok(backup_dir.join(backup_name))
    }

    /// Ensure a directory exists, creating it if necessary
    pub fn ensure_directory_exists(path: &Path) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| ProcessingError::Io(e))?;
        }
        Ok(())
    }

    /// Get a temporary file path
    pub fn get_temp_path(extension: &str) -> PathBuf {
        let temp_dir = std::env::temp_dir();
        let filename = format!("image_processor_{}_{}.{}", 
                              std::process::id(),
                              uuid::Uuid::new_v4().simple(),
                              extension);
        temp_dir.join(filename)
    }

    /// Clean up temporary files matching a pattern
    pub fn cleanup_temp_files(pattern: &str) -> Result<usize> {
        let temp_dir = std::env::temp_dir();
        let mut cleaned = 0;

        if let Ok(entries) = std::fs::read_dir(&temp_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.contains(pattern) {
                        if std::fs::remove_file(entry.path()).is_ok() {
                            cleaned += 1;
                        }
                    }
                }
            }
        }

        Ok(cleaned)
    }
}

/// Image format detection and utilities
pub mod format {
    use super::*;

    /// Detect image format from file extension
    pub fn detect_format_from_extension(path: &Path) -> Option<ImageFormat> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        
        match extension.as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            "webp" => Some(ImageFormat::WebP),
            "gif" => Some(ImageFormat::Gif),
            "bmp" => Some(ImageFormat::Bmp),
            "tiff" | "tif" => Some(ImageFormat::Tiff),
            "avif" => Some(ImageFormat::Avif),
            "heic" | "heif" => Some(ImageFormat::Heic),
            _ => None,
        }
    }

    /// Detect image format from file header (magic bytes)
    pub fn detect_format_from_header(path: &Path) -> Result<Option<ImageFormat>> {
        let mut file = std::fs::File::open(path).map_err(|e| ProcessingError::Io(e))?;
        let mut buffer = [0u8; 16];
        
        use std::io::Read;
        let bytes_read = file.read(&mut buffer).map_err(|e| ProcessingError::Io(e))?;
        
        if bytes_read < 4 {
            return Ok(None);
        }

        // Check magic bytes for different formats
        match &buffer[..4] {
            [0xFF, 0xD8, 0xFF, _] => Ok(Some(ImageFormat::Jpeg)),
            [0x89, 0x50, 0x4E, 0x47] => Ok(Some(ImageFormat::Png)),
            [0x47, 0x49, 0x46, 0x38] => Ok(Some(ImageFormat::Gif)),
            [0x42, 0x4D, _, _] => Ok(Some(ImageFormat::Bmp)),
            _ => {
                // Check for WebP
                if bytes_read >= 12 && &buffer[0..4] == b"RIFF" && &buffer[8..12] == b"WEBP" {
                    Ok(Some(ImageFormat::WebP))
                }
                // Check for TIFF
                else if &buffer[0..4] == b"II*\0" || &buffer[0..4] == b"MM\0*" {
                    Ok(Some(ImageFormat::Tiff))
                }
                else {
                    Ok(None)
                }
            }
        }
    }

    /// Get the best format for a given use case
    pub fn recommend_format_for_use_case(use_case: UseCase) -> ImageFormat {
        match use_case {
            UseCase::WebOptimized => ImageFormat::WebP,
            UseCase::Photography => ImageFormat::Jpeg,
            UseCase::Graphics => ImageFormat::Png,
            UseCase::Animation => ImageFormat::Gif,
            UseCase::Print => ImageFormat::Tiff,
            UseCase::Archive => ImageFormat::Png,
        }
    }

    /// Use cases for format recommendation
    pub enum UseCase {
        WebOptimized,
        Photography,
        Graphics,
        Animation,
        Print,
        Archive,
    }
}

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Simple performance timer
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }

        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            tracing::debug!("Timer '{}' elapsed: {}ms", self.name, self.elapsed_ms());
        }
    }

    /// Memory usage tracker
    pub struct MemoryTracker {
        initial_usage: Option<u64>,
    }

    impl MemoryTracker {
        pub fn new() -> Self {
            Self {
                initial_usage: Self::get_memory_usage(),
            }
        }

        pub fn current_usage(&self) -> Option<u64> {
            Self::get_memory_usage()
        }

        pub fn usage_delta(&self) -> Option<i64> {
            match (self.initial_usage, self.current_usage()) {
                (Some(initial), Some(current)) => Some(current as i64 - initial as i64),
                _ => None,
            }
        }

        #[cfg(target_os = "linux")]
        fn get_memory_usage() -> Option<u64> {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").ok()?;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u64>().ok().map(|kb| kb * 1024);
                    }
                }
            }
            None
        }

        #[cfg(target_os = "windows")]
        fn get_memory_usage() -> Option<u64> {
            // Windows implementation would use Windows API
            // For now, return None as placeholder
            None
        }

        #[cfg(target_os = "macos")]
        fn get_memory_usage() -> Option<u64> {
            // macOS implementation would use system calls
            // For now, return None as placeholder
            None
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        fn get_memory_usage() -> Option<u64> {
            None
        }
    }

    /// Calculate processing throughput
    pub fn calculate_throughput(bytes_processed: u64, duration: Duration) -> f64 {
        if duration.is_zero() {
            return 0.0;
        }
        
        let seconds = duration.as_secs_f64();
        let mb_processed = bytes_processed as f64 / (1024.0 * 1024.0);
        mb_processed / seconds
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate that a path is safe (no path traversal)
    pub fn is_safe_path(path: &Path) -> bool {
        // Check for path traversal attempts
        let path_str = path.to_string_lossy();
        !path_str.contains("..") && !path_str.contains("~")
    }

    /// Validate image dimensions
    pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
        const MAX_DIMENSION: u32 = 65535; // Common maximum for many formats
        const MIN_DIMENSION: u32 = 1;

        if width < MIN_DIMENSION || width > MAX_DIMENSION {
            return Err(ProcessingError::InvalidInput {
                message: format!("Width {} is out of valid range ({}-{})", 
                               width, MIN_DIMENSION, MAX_DIMENSION),
            });
        }

        if height < MIN_DIMENSION || height > MAX_DIMENSION {
            return Err(ProcessingError::InvalidInput {
                message: format!("Height {} is out of valid range ({}-{})", 
                               height, MIN_DIMENSION, MAX_DIMENSION),
            });
        }

        // Check for potential memory issues with very large images
        let total_pixels = width as u64 * height as u64;
        const MAX_PIXELS: u64 = 500_000_000; // 500 megapixels

        if total_pixels > MAX_PIXELS {
            return Err(ProcessingError::InvalidInput {
                message: format!("Image too large: {} pixels exceeds maximum of {}", 
                               total_pixels, MAX_PIXELS),
            });
        }

        Ok(())
    }

    /// Validate quality setting
    pub fn validate_quality(quality: u8) -> Result<()> {
        if quality == 0 || quality > 100 {
            return Err(ProcessingError::InvalidInput {
                message: format!("Quality {} must be between 1 and 100", quality),
            });
        }
        Ok(())
    }

    /// Validate opacity/transparency value
    pub fn validate_opacity(opacity: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(ProcessingError::InvalidInput {
                message: format!("Opacity {} must be between 0.0 and 1.0", opacity),
            });
        }
        Ok(())
    }
}

/// System information utilities
pub mod system {
    /// Get the number of CPU cores
    pub fn cpu_count() -> usize {
        num_cpus::get()
    }

    /// Get the optimal number of worker threads
    pub fn optimal_worker_count() -> usize {
        // Use 75% of available cores, minimum 1, maximum 16
        (cpu_count() * 3 / 4).max(1).min(16)
    }

    /// Check if hardware acceleration is available
    pub fn has_hardware_acceleration() -> bool {
        // This would check for GPU, specialized codecs, etc.
        // For now, return false as placeholder
        false
    }

    /// Get system memory information
    pub fn get_system_memory_mb() -> Option<u64> {
        // This would query system memory
        // For now, return None as placeholder
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Hello, World!").unwrap();
        
        let size = file::get_file_size(&file_path).unwrap();
        assert_eq!(size, 13); // "Hello, World!" is 13 bytes
    }

    #[test]
    fn test_backup_path_creation() {
        let original = PathBuf::from("/path/to/image.jpg");
        let backup_dir = PathBuf::from("/backups");
        
        let backup_path = file::create_backup_path(&original, &backup_dir).unwrap();
        assert!(backup_path.to_string_lossy().contains("image.jpg"));
        assert!(backup_path.starts_with(&backup_dir));
    }

    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(
            format::detect_format_from_extension(Path::new("test.jpg")),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            format::detect_format_from_extension(Path::new("test.PNG")),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            format::detect_format_from_extension(Path::new("test.unknown")),
            None
        );
    }

    #[test]
    fn test_dimension_validation() {
        assert!(validation::validate_dimensions(800, 600).is_ok());
        assert!(validation::validate_dimensions(0, 600).is_err());
        assert!(validation::validate_dimensions(800, 0).is_err());
        assert!(validation::validate_dimensions(100000, 100000).is_err()); // Too large
    }

    #[test]
    fn test_quality_validation() {
        assert!(validation::validate_quality(85).is_ok());
        assert!(validation::validate_quality(1).is_ok());
        assert!(validation::validate_quality(100).is_ok());
        assert!(validation::validate_quality(0).is_err());
        assert!(validation::validate_quality(101).is_err());
    }

    #[test]
    fn test_opacity_validation() {
        assert!(validation::validate_opacity(0.5).is_ok());
        assert!(validation::validate_opacity(0.0).is_ok());
        assert!(validation::validate_opacity(1.0).is_ok());
        assert!(validation::validate_opacity(-0.1).is_err());
        assert!(validation::validate_opacity(1.1).is_err());
    }

    #[test]
    fn test_safe_path_validation() {
        assert!(validation::is_safe_path(Path::new("image.jpg")));
        assert!(validation::is_safe_path(Path::new("folder/image.jpg")));
        assert!(!validation::is_safe_path(Path::new("../image.jpg")));
        assert!(!validation::is_safe_path(Path::new("~/image.jpg")));
    }

    #[test]
    fn test_performance_timer() {
        let timer = performance::Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_throughput_calculation() {
        let throughput = performance::calculate_throughput(
            1024 * 1024, // 1 MB
            Duration::from_secs(1)
        );
        assert!((throughput - 1.0).abs() < 0.01); // Should be ~1 MB/s
    }

    #[test]
    fn test_system_info() {
        assert!(system::cpu_count() > 0);
        assert!(system::optimal_worker_count() > 0);
    }

    #[test]
    fn test_temp_path_generation() {
        let path1 = file::get_temp_path("jpg");
        let path2 = file::get_temp_path("jpg");
        
        assert_ne!(path1, path2); // Should be unique
        assert!(path1.extension().unwrap() == "jpg");
    }
}