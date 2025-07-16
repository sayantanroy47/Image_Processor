//! Tauri-based GUI for the image processor

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use image_processor_core::{init, version};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{Manager, State};
use tracing::{info, error};

/// Application state
#[derive(Default)]
struct AppState {
    // TODO: Add application state fields
}

/// Response structure for API calls
#[derive(Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }
    
    fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

/// System information structure
#[derive(Serialize, Deserialize)]
struct SystemInfo {
    version: String,
    platform: String,
    architecture: String,
    cpu_cores: usize,
}

/// Get system information
#[tauri::command]
async fn get_system_info() -> ApiResponse<SystemInfo> {
    info!("Getting system information");
    
    let info = SystemInfo {
        version: version().to_string(),
        platform: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        cpu_cores: num_cpus::get(),
    };
    
    ApiResponse::success(info)
}

/// Convert image format
#[tauri::command]
async fn convert_image(
    input_path: String,
    output_path: String,
    format: Option<String>,
    quality: Option<u8>,
) -> ApiResponse<String> {
    info!("Converting image: {} -> {}", input_path, output_path);
    
    // TODO: Implement actual conversion logic
    // For now, return a placeholder response
    ApiResponse::success("Conversion functionality will be implemented in future tasks".to_string())
}

/// Add watermark to image
#[tauri::command]
async fn add_watermark(
    input_path: String,
    output_path: String,
    watermark_path: String,
    position: String,
    opacity: f32,
) -> ApiResponse<String> {
    info!("Adding watermark: {} to {}", watermark_path, input_path);
    
    // TODO: Implement actual watermark logic
    ApiResponse::success("Watermark functionality will be implemented in future tasks".to_string())
}

/// Get processing capabilities
#[tauri::command]
async fn get_capabilities() -> ApiResponse<Vec<String>> {
    info!("Getting processing capabilities");
    
    // TODO: Get actual capabilities from the processing orchestrator
    let capabilities = vec![
        "Format Conversion".to_string(),
        "Watermarking".to_string(),
        "Resizing".to_string(),
        "Color Correction".to_string(),
        "Background Removal".to_string(),
    ];
    
    ApiResponse::success(capabilities)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the core library
    init().await?;
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    info!("Image Processor GUI v{} starting", version());
    
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            convert_image,
            add_watermark,
            get_capabilities
        ])
        .setup(|app| {
            info!("Tauri application setup complete");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
    
    Ok(())
}