# Image Processor

High-performance cross-platform image processing application with support for format conversion, watermarking, metadata handling, and batch processing.

## Project Structure

This is a Rust workspace with three main crates:

### Core Library (`crates/image-processor-core`)
- **Purpose**: Core image processing engine and business logic
- **Key Modules**:
  - `config`: Configuration management and settings
  - `error`: Comprehensive error handling with structured error types
  - `logging`: Advanced logging system with performance monitoring
  - `models`: Data models for processing operations and job management
  - `processing`: Processing orchestrator and trait definitions
  - `utils`: Utility functions for file handling, validation, and system info

### CLI Application (`crates/image-processor-cli`)
- **Purpose**: Command-line interface for image processing
- **Features**:
  - Format conversion commands
  - Watermarking operations
  - Batch processing capabilities
  - System information display
  - Configurable logging levels

### GUI Application (`crates/image-processor-gui`)
- **Purpose**: Tauri-based desktop GUI application
- **Technology Stack**:
  - Rust backend with Tauri framework
  - React + TypeScript frontend
  - Vite for build tooling
- **Features**:
  - Cross-platform desktop application
  - Modern web-based UI
  - Real-time system information display
  - Processing capabilities overview

## Architecture

The application follows a modular architecture:

1. **Core Library**: Contains all business logic and processing capabilities
2. **CLI Interface**: Provides command-line access to core functionality
3. **GUI Interface**: Offers a user-friendly desktop application experience

## Key Features

- **Format Support**: JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF, HEIC
- **Processing Operations**:
  - Format conversion with quality control
  - Image resizing with multiple algorithms
  - Watermarking with positioning and blend modes
  - Background removal using AI models
  - Color correction and adjustments
  - Cropping and rotation
  - Text overlay with effects
  - Collage creation
- **Performance**: 
  - Async processing with Tokio
  - Parallel processing with Rayon
  - Hardware acceleration support
  - Memory-efficient streaming for large files
- **Cross-platform**: Windows, macOS, and Linux support

## Development Setup

### Prerequisites
- Rust 1.70+ with Cargo
- Node.js 18+ and npm (for GUI development)
- System dependencies for image processing libraries

### Building the Project

```bash
# Build all crates
cargo build --workspace

# Build in release mode
cargo build --workspace --release

# Run tests
cargo test --workspace

# Check code without building
cargo check --workspace
```

### Running the Applications

#### CLI Application
```bash
# Run CLI with info command
cargo run --bin image-processor info

# Convert an image
cargo run --bin image-processor convert -i input.jpg -o output.png -f png -q 85

# Add watermark
cargo run --bin image-processor watermark -i input.jpg -o output.jpg -w watermark.png
```

#### GUI Application
```bash
# Install frontend dependencies
cd crates/image-processor-gui
npm install

# Run in development mode
cargo tauri dev

# Build for production
cargo tauri build
```

## Configuration

The application uses TOML configuration files located at:
- **Default**: `~/.config/image-processor/config.toml`
- **Custom**: Specify with `--config` flag

Example configuration:
```toml
[processing]
default_output_format = "Jpeg"
default_quality = 85
backup_enabled = true
max_concurrent_jobs = 8
hardware_acceleration = true

[ui]
theme = "Auto"
window_width = 1200
window_height = 800

[performance]
enable_simd = true
enable_gpu_acceleration = true
streaming_threshold_mb = 100
```

## Logging

The application provides comprehensive logging with:
- Structured JSON output for production
- Pretty-printed output for development
- Performance metrics tracking
- Error categorization and recovery hints
- Configurable log levels and outputs

## Testing

```bash
# Run all tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Run specific test
cargo test --package image-processor-core test_name
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.