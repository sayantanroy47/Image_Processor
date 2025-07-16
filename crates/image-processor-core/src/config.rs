//! Configuration management for the image processing library

use crate::error::{ProcessingError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub processing: ProcessingConfig,
    pub ui: UiConfig,
    pub storage: StorageConfig,
    pub performance: PerformanceConfig,
    pub logging: LoggingConfigSerde,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            processing: ProcessingConfig::default(),
            ui: UiConfig::default(),
            storage: StorageConfig::default(),
            performance: PerformanceConfig::default(),
            logging: LoggingConfigSerde::default(),
        }
    }
}

/// Processing-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub default_output_format: ImageFormat,
    pub default_quality: u8,
    pub backup_enabled: bool,
    pub backup_directory: PathBuf,
    pub max_concurrent_jobs: usize,
    pub hardware_acceleration: bool,
    pub preserve_metadata: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            default_output_format: ImageFormat::Jpeg,
            default_quality: 85,
            backup_enabled: true,
            backup_directory: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".image-processor")
                .join("backups"),
            max_concurrent_jobs: num_cpus::get(),
            hardware_acceleration: true,
            preserve_metadata: true,
        }
    }
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    pub theme: Theme,
    pub window_width: u32,
    pub window_height: u32,
    pub auto_preview: bool,
    pub show_progress_details: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: Theme::Auto,
            window_width: 1200,
            window_height: 800,
            auto_preview: true,
            show_progress_details: true,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub temp_directory: PathBuf,
    pub cache_directory: PathBuf,
    pub max_cache_size_mb: u64,
    pub cleanup_on_exit: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let base_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("image-processor");
            
        Self {
            temp_directory: base_dir.join("temp"),
            cache_directory: base_dir.join("cache"),
            max_cache_size_mb: 1024, // 1GB
            cleanup_on_exit: true,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub worker_threads: Option<usize>,
    pub memory_limit_mb: Option<u64>,
    pub enable_simd: bool,
    pub enable_gpu_acceleration: bool,
    pub streaming_threshold_mb: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: None, // Use system default
            memory_limit_mb: None, // No limit
            enable_simd: true,
            enable_gpu_acceleration: true,
            streaming_threshold_mb: 100, // Stream files larger than 100MB
        }
    }
}

/// Serializable version of LoggingConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfigSerde {
    pub level: String,
    pub output_type: String,
    pub output_path: Option<PathBuf>,
    pub structured: bool,
    pub performance_metrics: bool,
    pub error_tracking: bool,
}

impl Default for LoggingConfigSerde {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            output_type: "console".to_string(),
            output_path: None,
            structured: true,
            performance_metrics: true,
            error_tracking: true,
        }
    }
}

/// Supported image formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Gif,
    Bmp,
    Tiff,
    Avif,
    Heic,
}

impl ImageFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
            ImageFormat::WebP => "webp",
            ImageFormat::Gif => "gif",
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
            ImageFormat::Avif => "avif",
            ImageFormat::Heic => "heic",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::WebP => "image/webp",
            ImageFormat::Gif => "image/gif",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Tiff => "image/tiff",
            ImageFormat::Avif => "image/avif",
            ImageFormat::Heic => "image/heic",
        }
    }

    /// Check if this format supports transparency
    pub fn supports_transparency(&self) -> bool {
        matches!(self, ImageFormat::Png | ImageFormat::WebP | ImageFormat::Gif)
    }

    /// Check if this format supports animation
    pub fn supports_animation(&self) -> bool {
        matches!(self, ImageFormat::Gif | ImageFormat::WebP)
    }
}

/// UI theme options
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Theme {
    Light,
    Dark,
    Auto,
}

/// Configuration manager
pub struct ConfigManager {
    config_path: PathBuf,
    config: AppConfig,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Result<Self> {
        let config_path = Self::default_config_path()?;
        let config = Self::load_or_create_config(&config_path)?;
        
        Ok(Self {
            config_path,
            config,
        })
    }

    /// Create a configuration manager with a custom path
    pub fn with_path(config_path: PathBuf) -> Result<Self> {
        let config = Self::load_or_create_config(&config_path)?;
        
        Ok(Self {
            config_path,
            config,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: AppConfig) -> Result<()> {
        self.config = config;
        self.save()
    }

    /// Save the current configuration to disk
    pub fn save(&self) -> Result<()> {
        // Ensure the parent directory exists
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ProcessingError::ConfigError {
                message: format!("Failed to create config directory: {}", e),
            })?;
        }

        let config_str = toml::to_string_pretty(&self.config).map_err(|e| {
            ProcessingError::ConfigError {
                message: format!("Failed to serialize config: {}", e),
            }
        })?;

        std::fs::write(&self.config_path, config_str).map_err(|e| {
            ProcessingError::ConfigError {
                message: format!("Failed to write config file: {}", e),
            }
        })?;

        tracing::info!("Configuration saved to {:?}", self.config_path);
        Ok(())
    }

    /// Get the default configuration file path
    fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| ProcessingError::ConfigError {
                message: "Could not determine config directory".to_string(),
            })?
            .join("image-processor");

        Ok(config_dir.join("config.toml"))
    }

    /// Load configuration from file or create default if it doesn't exist
    fn load_or_create_config(path: &PathBuf) -> Result<AppConfig> {
        if path.exists() {
            let config_str = std::fs::read_to_string(path).map_err(|e| {
                ProcessingError::ConfigError {
                    message: format!("Failed to read config file: {}", e),
                }
            })?;

            let config: AppConfig = toml::from_str(&config_str).map_err(|e| {
                ProcessingError::ConfigError {
                    message: format!("Failed to parse config file: {}", e),
                }
            })?;

            tracing::info!("Configuration loaded from {:?}", path);
            Ok(config)
        } else {
            let config = AppConfig::default();
            tracing::info!("Using default configuration");
            Ok(config)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_image_format_properties() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert!(ImageFormat::Png.supports_transparency());
        assert!(!ImageFormat::Jpeg.supports_transparency());
        assert!(ImageFormat::Gif.supports_animation());
        assert!(!ImageFormat::Png.supports_animation());
    }

    #[test]
    fn test_default_configs() {
        let app_config = AppConfig::default();
        assert_eq!(app_config.processing.default_quality, 85);
        assert!(app_config.processing.backup_enabled);
        assert_eq!(app_config.ui.window_width, 1200);
    }

    #[test]
    fn test_config_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let manager = ConfigManager::with_path(config_path.clone());
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert_eq!(manager.config().processing.default_quality, 85);
    }

    #[test]
    fn test_config_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        // Create and save config
        let mut manager = ConfigManager::with_path(config_path.clone()).unwrap();
        let mut config = manager.config().clone();
        config.processing.default_quality = 95;
        
        manager.update_config(config).unwrap();
        
        // Load config again
        let manager2 = ConfigManager::with_path(config_path).unwrap();
        assert_eq!(manager2.config().processing.default_quality, 95);
    }
}