//! Tests for watermark engine format support

use super::*;
use crate::models::{ProcessingOperation, ResizeAlgorithm};
use std::fs;
use tempfile::tempdir;

/// Create a simple test SVG content
fn create_test_svg(width: u32, height: u32) -> String {
    format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="50" height="30" fill="black"/>
            <circle cx="100" cy="50" r="20" fill="black"/>
        </svg>"#,
        width, height
    )
}

/// Create a simple test PNG with transparency
fn create_test_png() -> Vec<u8> {
    // Create a simple 32x32 RGBA image with transparency
    let mut img = RgbaImage::new(32, 32);
    
    // Fill with semi-transparent red
    for pixel in img.pixels_mut() {
        *pixel = Rgba([255, 0, 0, 128]); // Semi-transparent red
    }
    
    // Add some fully transparent pixels
    for y in 10..22 {
        for x in 10..22 {
            img.put_pixel(x, y, Rgba([0, 0, 0, 0])); // Fully transparent
        }
    }
    
    let mut buffer = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buffer);
    encoder.encode(
        img.as_raw(),
        32,
        32,
        image::ColorType::Rgba8,
    ).unwrap();
    
    buffer
}

#[tokio::test]
async fn test_watermark_format_detection() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Test SVG detection by extension
    let svg_path = temp_dir.path().join("test.svg");
    fs::write(&svg_path, create_test_svg(100, 100)).unwrap();
    
    let format = engine.detect_watermark_format(&svg_path).unwrap();
    assert_eq!(format, WatermarkFormat::Svg);
    
    // Test PNG detection by extension
    let png_path = temp_dir.path().join("test.png");
    fs::write(&png_path, create_test_png()).unwrap();
    
    let format = engine.detect_watermark_format(&png_path).unwrap();
    assert_eq!(format, WatermarkFormat::Png);
    
    // Test SVG detection by content (no extension)
    let svg_no_ext_path = temp_dir.path().join("test_svg");
    fs::write(&svg_no_ext_path, create_test_svg(100, 100)).unwrap();
    
    let format = engine.detect_watermark_format(&svg_no_ext_path).unwrap();
    assert_eq!(format, WatermarkFormat::Svg);
}

#[tokio::test]
async fn test_svg_dimension_parsing() {
    let svg_content = create_test_svg(200, 150);
    let dimensions = WatermarkEngine::parse_svg_dimensions(&svg_content);
    
    assert_eq!(dimensions, Some((200, 150)));
    
    // Test SVG without dimensions on the svg element
    let svg_no_dims = r#"<svg xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="20"/>
    </svg>"#;
    
    let dimensions = WatermarkEngine::parse_svg_dimensions(svg_no_dims);
    assert_eq!(dimensions, None);
    
    // Test SVG with quoted dimensions
    let svg_quoted = r#"<svg width="300" height="250" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="100" height="100"/>
    </svg>"#;
    
    let dimensions = WatermarkEngine::parse_svg_dimensions(svg_quoted);
    assert_eq!(dimensions, Some((300, 250)));
}

#[tokio::test]
async fn test_svg_rendering() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create test SVG file
    let svg_path = temp_dir.path().join("test.svg");
    fs::write(&svg_path, create_test_svg(100, 100)).unwrap();
    
    // Render SVG to raster image
    let result = engine.render_svg_watermark(&svg_path, Some(100), Some(100)).await;
    assert!(result.is_ok());
    
    let image = result.unwrap();
    assert_eq!(image.width(), 100);
    assert_eq!(image.height(), 100);
}

#[tokio::test]
async fn test_svg_rect_rendering() {
    let mut image = RgbaImage::new(100, 100);
    
    // Fill with transparent background
    for pixel in image.pixels_mut() {
        *pixel = Rgba([255, 255, 255, 0]);
    }
    
    let svg_content = r#"<rect x="10" y="10" width="50" height="30">"#;
    let result = WatermarkEngine::render_svg_rect(&mut image, svg_content);
    assert!(result.is_ok());
    
    // Check that some pixels were drawn (should be black)
    let pixel = image.get_pixel(20, 20);
    assert_eq!(*pixel, Rgba([0, 0, 0, 255]));
    
    // Check that pixels outside the rect are still transparent
    let pixel = image.get_pixel(5, 5);
    assert_eq!(*pixel, Rgba([255, 255, 255, 0]));
}

#[tokio::test]
async fn test_svg_circle_rendering() {
    let mut image = RgbaImage::new(100, 100);
    
    // Fill with transparent background
    for pixel in image.pixels_mut() {
        *pixel = Rgba([255, 255, 255, 0]);
    }
    
    let svg_content = r#"<circle cx="50" cy="50" r="20">"#;
    let result = WatermarkEngine::render_svg_circle(&mut image, svg_content);
    assert!(result.is_ok());
    
    // Check that center pixel was drawn
    let pixel = image.get_pixel(50, 50);
    assert_eq!(*pixel, Rgba([0, 0, 0, 255]));
    
    // Check that pixels outside the circle are still transparent
    let pixel = image.get_pixel(10, 10);
    assert_eq!(*pixel, Rgba([255, 255, 255, 0]));
}

#[tokio::test]
async fn test_png_transparency_loading() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create test PNG with transparency
    let png_path = temp_dir.path().join("transparent.png");
    fs::write(&png_path, create_test_png()).unwrap();
    
    // Load the watermark
    let result = engine.load_watermark(&png_path).await;
    assert!(result.is_ok());
    
    let image = result.unwrap();
    assert_eq!(image.width(), 32);
    assert_eq!(image.height(), 32);
    
    // Convert to RGBA to check transparency
    let rgba_image = image.to_rgba8();
    
    // Check semi-transparent pixel
    let pixel = rgba_image.get_pixel(5, 5);
    assert_eq!(pixel[3], 128); // Alpha should be 128 (semi-transparent)
    
    // Check fully transparent pixel
    let pixel = rgba_image.get_pixel(15, 15);
    assert_eq!(pixel[3], 0); // Alpha should be 0 (fully transparent)
}

#[tokio::test]
async fn test_watermark_format_support() {
    let engine = WatermarkEngine::new();
    
    // Test that engine supports various formats
    assert!(engine.supports_format(ImageFormat::Png));
    assert!(engine.supports_format(ImageFormat::Jpeg));
    assert!(engine.supports_format(ImageFormat::WebP));
    assert!(engine.supports_format(ImageFormat::Gif));
    assert!(engine.supports_format(ImageFormat::Bmp));
    assert!(engine.supports_format(ImageFormat::Tiff));
}

// Tests for visual effects and blending functionality

#[test]
fn test_enhanced_blend_modes() {
    let engine = WatermarkEngine::new();
    
    // Test base and overlay pixels
    let base = Rgba([100, 100, 100, 255]); // Gray
    let overlay = Rgba([200, 50, 50, 255]); // Red
    let opacity = 1.0;
    
    // Test Normal blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Normal);
    assert_eq!(result, overlay); // Normal mode should return overlay
    
    // Test Multiply blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Multiply);
    let expected_r = (100 * 200 / 255) as u8;
    let expected_g = (100 * 50 / 255) as u8;
    let expected_b = (100 * 50 / 255) as u8;
    assert_eq!(result[0], expected_r);
    assert_eq!(result[1], expected_g);
    assert_eq!(result[2], expected_b);
    
    // Test Screen blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Screen);
    let expected_r = (255.0 - (255.0 - 100.0) * (255.0 - 200.0) / 255.0) as u8;
    let expected_g = (255.0 - (255.0 - 100.0) * (255.0 - 50.0) / 255.0) as u8;
    let expected_b = (255.0 - (255.0 - 100.0) * (255.0 - 50.0) / 255.0) as u8;
    assert_eq!(result[0], expected_r);
    assert_eq!(result[1], expected_g);
    assert_eq!(result[2], expected_b);
    
    // Test Darken blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Darken);
    assert_eq!(result[0], 100.min(200)); // Should be 100
    assert_eq!(result[1], 100.min(50));  // Should be 50
    assert_eq!(result[2], 100.min(50));  // Should be 50
    
    // Test Lighten blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Lighten);
    assert_eq!(result[0], 100.max(200)); // Should be 200
    assert_eq!(result[1], 100.max(50));  // Should be 100
    assert_eq!(result[2], 100.max(50));  // Should be 100
    
    // Test Difference blend mode
    let result = engine.blend_pixels(base, overlay, opacity, BlendMode::Difference);
    assert_eq!(result[0], (100i32 - 200i32).abs() as u8); // Should be 100
    assert_eq!(result[1], (100i32 - 50i32).abs() as u8);  // Should be 50
    assert_eq!(result[2], (100i32 - 50i32).abs() as u8);  // Should be 50
}

#[test]
fn test_opacity_controls() {
    let engine = WatermarkEngine::new();
    
    let base = Rgba([100, 100, 100, 255]);
    let overlay = Rgba([200, 200, 200, 255]);
    
    // Test full opacity
    let result = engine.blend_pixels(base, overlay, 1.0, BlendMode::Normal);
    assert_eq!(result, overlay);
    
    // Test half opacity
    let result = engine.blend_pixels(base, overlay, 0.5, BlendMode::Normal);
    // With 50% opacity, result should be blend of base and overlay
    let expected_r = ((200.0 * 0.5 + 100.0 * 0.5) as u8);
    assert!((result[0] as i32 - expected_r as i32).abs() <= 2); // Allow small rounding errors
    
    // Test zero opacity
    let result = engine.blend_pixels(base, overlay, 0.0, BlendMode::Normal);
    assert_eq!(result, base); // Should return base pixel unchanged
}

#[test]
fn test_shadow_effect_creation() {
    let engine = WatermarkEngine::new();
    
    // Create a simple watermark image
    let mut watermark = RgbaImage::new(50, 50);
    for y in 10..40 {
        for x in 10..40 {
            watermark.put_pixel(x, y, Rgba([255, 0, 0, 255])); // Red square
        }
    }
    
    // Create shadow configuration
    let shadow = WatermarkShadow {
        color: Color::new(0, 0, 0, 128),
        offset_x: 5,
        offset_y: 5,
        blur_radius: 2.0,
        opacity: 0.8,
    };
    
    // Apply shadow effect
    let result = engine.apply_shadow_effect(&watermark, &shadow);
    
    // Result should be larger due to padding and offset
    assert!(result.width() > watermark.width());
    assert!(result.height() > watermark.height());
    
    // Check that shadow pixels exist at offset position
    let blur_padding = (shadow.blur_radius * 2.0) as u32;
    let shadow_x = 20 + blur_padding + shadow.offset_x as u32; // Original pixel at (20,20) + padding + offset
    let shadow_y = 20 + blur_padding + shadow.offset_y as u32;
    
    let shadow_pixel = result.get_pixel(shadow_x, shadow_y);
    assert!(shadow_pixel[3] > 0); // Should have some alpha (shadow or original)
}

#[test]
fn test_outline_effect_creation() {
    let engine = WatermarkEngine::new();
    
    // Create a simple watermark image
    let mut watermark = RgbaImage::new(50, 50);
    for y in 20..30 {
        for x in 20..30 {
            watermark.put_pixel(x, y, Rgba([255, 0, 0, 255])); // Red square
        }
    }
    
    // Create outline configuration
    let outline = WatermarkOutline {
        color: Color::new(255, 255, 255, 255), // White outline
        width: 2.0,
        opacity: 1.0,
    };
    
    // Apply outline effect
    let result = engine.apply_outline_effect(&watermark, &outline);
    
    // Result should be larger due to outline padding
    assert!(result.width() > watermark.width());
    assert!(result.height() > watermark.height());
    
    // Check that outline pixels exist around the original shape
    let outline_padding = (outline.width * 2.0) as u32;
    let outline_x = 18 + outline_padding; // Just outside the original red square
    let outline_y = 25 + outline_padding; // Middle of the square height
    
    let outline_pixel = result.get_pixel(outline_x, outline_y);
    assert!(outline_pixel[3] > 0); // Should have alpha from outline
}

#[test]
fn test_glow_effect_creation() {
    let engine = WatermarkEngine::new();
    
    // Create a simple watermark image
    let mut watermark = RgbaImage::new(50, 50);
    for y in 20..30 {
        for x in 20..30 {
            watermark.put_pixel(x, y, Rgba([255, 0, 0, 255])); // Red square
        }
    }
    
    // Create glow configuration
    let glow = WatermarkGlow {
        color: Color::new(255, 255, 0, 255), // Yellow glow
        radius: 3.0,
        intensity: 1.0,
        opacity: 0.8,
    };
    
    // Apply glow effect
    let result = engine.apply_glow_effect(&watermark, &glow);
    
    // Result should be larger due to glow padding
    assert!(result.width() > watermark.width());
    assert!(result.height() > watermark.height());
    
    // Check that glow extends beyond original shape
    let glow_padding = (glow.radius * 2.0) as u32;
    let glow_x = 25 + glow_padding; // Center of original square
    let glow_y = 25 + glow_padding;
    
    let glow_pixel = result.get_pixel(glow_x, glow_y);
    assert!(glow_pixel[3] > 0); // Should have alpha from glow or original
}

#[test]
fn test_gaussian_blur_implementation() {
    let engine = WatermarkEngine::new();
    
    // Create a simple image with a single white pixel in the center
    let mut image = RgbaImage::new(21, 21);
    for pixel in image.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 0]); // Transparent
    }
    image.put_pixel(10, 10, Rgba([255, 255, 255, 255])); // White center pixel
    
    // Apply blur
    let blurred = engine.apply_gaussian_blur(image, 2.0);
    
    // Check that blur spread the white pixel
    let center_pixel = blurred.get_pixel(10, 10);
    let adjacent_pixel = blurred.get_pixel(11, 10);
    
    // Center should still be brightest
    assert!(center_pixel[0] >= adjacent_pixel[0]);
    
    // Adjacent pixel should have some brightness from blur
    assert!(adjacent_pixel[0] > 0);
    
    // Test zero radius (no blur)
    let mut original = RgbaImage::new(10, 10);
    for pixel in original.pixels_mut() {
        *pixel = Rgba([100, 100, 100, 255]);
    }
    let no_blur = engine.apply_gaussian_blur(original.clone(), 0.0);
    
    // Should be identical to original
    for (orig_pixel, blur_pixel) in original.pixels().zip(no_blur.pixels()) {
        assert_eq!(orig_pixel, blur_pixel);
    }
}

#[test]
fn test_visual_effects_combination() {
    let engine = WatermarkEngine::new();
    
    // Create a simple watermark
    let mut watermark = RgbaImage::new(30, 30);
    for y in 10..20 {
        for x in 10..20 {
            watermark.put_pixel(x, y, Rgba([255, 0, 0, 255])); // Red square
        }
    }
    
    // Create visual effects configuration
    let effects = WatermarkVisualEffects {
        shadow: Some(WatermarkShadow {
            color: Color::new(0, 0, 0, 128),
            offset_x: 2,
            offset_y: 2,
            blur_radius: 1.0,
            opacity: 0.5,
        }),
        outline: Some(WatermarkOutline {
            color: Color::new(255, 255, 255, 255),
            width: 1.0,
            opacity: 1.0,
        }),
        glow: Some(WatermarkGlow {
            color: Color::new(255, 255, 0, 255),
            radius: 2.0,
            intensity: 0.8,
            opacity: 0.6,
        }),
    };
    
    // Apply all effects
    let result = engine.apply_visual_effects(watermark, &effects);
    
    // Result should be significantly larger due to all effects
    assert!(result.width() > 30);
    assert!(result.height() > 30);
    
    // Should have non-transparent pixels from effects
    let mut has_non_transparent = false;
    for pixel in result.pixels() {
        if pixel[3] > 0 {
            has_non_transparent = true;
            break;
        }
    }
    assert!(has_non_transparent);
}

#[test]
fn test_color_struct_functionality() {
    // Test Color creation and conversion
    let color = Color::new(255, 128, 64, 200);
    assert_eq!(color.r, 255);
    assert_eq!(color.g, 128);
    assert_eq!(color.b, 64);
    assert_eq!(color.a, 200);
    
    // Test predefined colors
    let white = Color::white();
    assert_eq!(white.r, 255);
    assert_eq!(white.g, 255);
    assert_eq!(white.b, 255);
    assert_eq!(white.a, 255);
    
    let black = Color::black();
    assert_eq!(black.r, 0);
    assert_eq!(black.g, 0);
    assert_eq!(black.b, 0);
    assert_eq!(black.a, 255);
    
    let transparent = Color::transparent();
    assert_eq!(transparent.a, 0);
    
    // Test conversion to RGBA
    let rgba = color.to_rgba();
    assert_eq!(rgba[0], 255);
    assert_eq!(rgba[1], 128);
    assert_eq!(rgba[2], 64);
    assert_eq!(rgba[3], 200);
}

#[test]
fn test_visual_effects_default_values() {
    // Test default visual effects
    let effects = WatermarkVisualEffects::default();
    assert!(effects.shadow.is_none());
    assert!(effects.outline.is_none());
    assert!(effects.glow.is_none());
    
    // Test default shadow
    let shadow = WatermarkShadow::default();
    assert_eq!(shadow.offset_x, 2);
    assert_eq!(shadow.offset_y, 2);
    assert_eq!(shadow.blur_radius, 4.0);
    assert_eq!(shadow.opacity, 0.5);
    
    // Test default outline
    let outline = WatermarkOutline::default();
    assert_eq!(outline.width, 1.0);
    assert_eq!(outline.opacity, 1.0);
    
    // Test default glow
    let glow = WatermarkGlow::default();
    assert_eq!(glow.radius, 5.0);
    assert_eq!(glow.intensity, 1.0);
    assert_eq!(glow.opacity, 0.8);
}

#[tokio::test]
async fn test_watermark_config_with_visual_effects() {
    let temp_dir = tempdir().unwrap();
    
    // Create a test watermark file
    let watermark_path = temp_dir.path().join("test.png");
    fs::write(&watermark_path, create_test_png()).unwrap();
    
    // Create watermark config with visual effects
    let config = WatermarkConfig {
        watermark_path: watermark_path.clone(),
        positions: vec![WatermarkPosition::Center],
        opacity: 0.8,
        scale: 0.5,
        blend_mode: BlendMode::Overlay,
        scaling_options: WatermarkScalingOptions::default(),
        alignment: WatermarkAlignment::default(),
        offset: WatermarkOffset::default(),
        visual_effects: WatermarkVisualEffects {
            shadow: Some(WatermarkShadow::default()),
            outline: Some(WatermarkOutline::default()),
            glow: None,
        },
    };
    
    // Verify config serialization/deserialization
    let serialized = serde_json::to_string(&config).unwrap();
    let deserialized: WatermarkConfig = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(config.opacity, deserialized.opacity);
    assert_eq!(config.blend_mode as u8, deserialized.blend_mode as u8);
    assert!(deserialized.visual_effects.shadow.is_some());
    assert!(deserialized.visual_effects.outline.is_some());
    assert!(deserialized.visual_effects.glow.is_none());
}

#[tokio::test]
async fn test_watermark_capabilities() {
    let engine = WatermarkEngine::new();
    let capabilities = engine.get_capabilities();
    
    // Check that watermark operation is supported
    assert!(capabilities.operations.contains(&ProcOp::Watermark));
    
    // Check supported formats
    assert!(capabilities.input_formats.contains(&ImageFormat::Png));
    assert!(capabilities.input_formats.contains(&ImageFormat::Jpeg));
    assert!(capabilities.output_formats.contains(&ImageFormat::Png));
    assert!(capabilities.output_formats.contains(&ImageFormat::Jpeg));
}

#[tokio::test]
async fn test_invalid_watermark_format() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create file with unknown format
    let unknown_path = temp_dir.path().join("test.unknown");
    fs::write(&unknown_path, b"invalid content").unwrap();
    
    let result = engine.detect_watermark_format(&unknown_path);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_missing_watermark_file() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Try to load non-existent file
    let missing_path = temp_dir.path().join("missing.png");
    let result = engine.load_watermark(&missing_path).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        ProcessingError::FileNotFound { .. } => {}, // Expected
        _ => panic!("Expected FileNotFound error"),
    }
}

#[tokio::test]
async fn test_svg_rendering_with_custom_dimensions() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create test SVG file
    let svg_path = temp_dir.path().join("test.svg");
    fs::write(&svg_path, create_test_svg(50, 50)).unwrap();
    
    // Render with custom dimensions
    let result = engine.render_svg_watermark(&svg_path, Some(200), Some(150)).await;
    assert!(result.is_ok());
    
    let image = result.unwrap();
    assert_eq!(image.width(), 200);
    assert_eq!(image.height(), 150);
}

#[tokio::test]
async fn test_watermark_engine_metadata() {
    let engine = WatermarkEngine::new();
    let metadata = engine.get_metadata();
    
    assert_eq!(metadata.name, "Watermark Engine");
    assert_eq!(metadata.version, "1.0.0");
    assert!(metadata.description.contains("watermark"));
    assert!(metadata.performance_profile.parallel_friendly);
}

#[tokio::test]
async fn test_can_handle_watermark_operation() {
    let engine = WatermarkEngine::new();
    
    let watermark_config = WatermarkConfig {
        watermark_path: PathBuf::from("test.png"),
        positions: vec![WatermarkPosition::Center],
        opacity: 0.5,
        scale: 1.0,
        blend_mode: BlendMode::Normal,
        scaling_options: WatermarkScalingOptions::default(),
        alignment: WatermarkAlignment::default(),
        offset: WatermarkOffset::default(),
        visual_effects: WatermarkVisualEffects::default(),
    };
    
    let watermark_op = ProcessingOperation::Watermark { 
        config: watermark_config 
    };
    
    assert!(engine.can_handle_operation(&watermark_op));
    
    // Test with non-watermark operation
    let resize_op = ProcessingOperation::Resize {
        width: Some(100),
        height: Some(100),
        algorithm: ResizeAlgorithm::Lanczos,
    };
    
    assert!(!engine.can_handle_operation(&resize_op));
}

// Tests for enhanced positioning and scaling functionality

#[tokio::test]
async fn test_position_calculator_basic_positions() {
    let image_width = 800;
    let image_height = 600;
    let watermark_width = 100;
    let watermark_height = 80;
    let alignment = WatermarkAlignment::Edge;
    let offset = WatermarkOffset::default();
    
    // Test all basic positions
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopLeft,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (0, 0));
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopCenter,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (350, 0)); // (800-100)/2 = 350
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopRight,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (700, 0)); // 800-100 = 700
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::Center,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (350, 260)); // (800-100)/2, (600-80)/2
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::BottomRight,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (700, 520)); // 800-100, 600-80
}

#[tokio::test]
async fn test_position_calculator_custom_position() {
    let image_width = 800;
    let image_height = 600;
    let watermark_width = 100;
    let watermark_height = 80;
    let alignment = WatermarkAlignment::Edge;
    let offset = WatermarkOffset::default();
    
    // Test custom position (percentage-based)
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::Custom { x: 0.25, y: 0.75 },
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (200, 450)); // 800*0.25=200, 600*0.75=450
    
    // Test custom position with clamping
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::Custom { x: 1.5, y: -0.5 }, // Out of bounds
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (700, 0)); // Clamped to valid range
}

#[tokio::test]
async fn test_position_calculator_with_padding() {
    let image_width = 800;
    let image_height = 600;
    let watermark_width = 100;
    let watermark_height = 80;
    let padding = 20;
    let alignment = WatermarkAlignment::Padded { padding };
    let offset = WatermarkOffset::default();
    
    // Test padded positions
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopLeft,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (20, 20)); // Padding applied
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::BottomRight,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (680, 500)); // 800-100-20, 600-80-20
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopCenter,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (350, 20)); // Center horizontally, padded vertically
}

#[tokio::test]
async fn test_position_calculator_with_offset() {
    let image_width = 800;
    let image_height = 600;
    let watermark_width = 100;
    let watermark_height = 80;
    let alignment = WatermarkAlignment::Edge;
    let offset = WatermarkOffset { x: 50, y: -30 };
    
    // Test position with offset
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopLeft,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (50, 0)); // x offset applied, y clamped to 0
    
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::Center,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (400, 230)); // 350+50, 260-30
}

#[tokio::test]
async fn test_position_calculator_custom_alignment() {
    let image_width = 800;
    let image_height = 600;
    let watermark_width = 100;
    let watermark_height = 80;
    let alignment = WatermarkAlignment::Custom {
        horizontal: HorizontalAlign::Right,
        vertical: VerticalAlign::Middle,
    };
    let offset = WatermarkOffset::default();
    
    // Test custom alignment
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopLeft, // This should be overridden by custom alignment
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (700, 260)); // Right + Middle alignment
}

#[tokio::test]
async fn test_watermark_scaling_basic() {
    let engine = WatermarkEngine::new();
    
    // Create a test watermark image
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(200, 150));
    let base_width = 800;
    let base_height = 600;
    let scale = 0.5;
    let scaling_options = WatermarkScalingOptions::default();
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    assert_eq!(scaled.width(), 100); // 200 * 0.5
    assert_eq!(scaled.height(), 75);  // 150 * 0.5
}

#[tokio::test]
async fn test_watermark_scaling_with_percentage_constraints() {
    let engine = WatermarkEngine::new();
    
    // Create a large watermark that would exceed percentage limits
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(500, 400));
    let base_width = 800;
    let base_height = 600;
    let scale = 1.0; // No scaling reduction
    
    let scaling_options = WatermarkScalingOptions {
        preserve_aspect_ratio: true,
        max_width_percent: Some(0.3), // 30% of 800 = 240px max
        max_height_percent: Some(0.3), // 30% of 600 = 180px max
        min_size: None,
        max_size: None,
    };
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    // Should be constrained by height (180px max)
    // Aspect ratio: 500/400 = 1.25
    // Height constrained to 180, so width = 180 * 1.25 = 225
    assert_eq!(scaled.width(), 225);
    assert_eq!(scaled.height(), 180);
}

#[tokio::test]
async fn test_watermark_scaling_with_min_size() {
    let engine = WatermarkEngine::new();
    
    // Create a small watermark
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(20, 15));
    let base_width = 800;
    let base_height = 600;
    let scale = 0.5; // Would make it even smaller
    
    let scaling_options = WatermarkScalingOptions {
        preserve_aspect_ratio: true,
        max_width_percent: None,
        max_height_percent: None,
        min_size: Some((50, 40)),
        max_size: None,
    };
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    // Should be scaled up to meet minimum size
    // Aspect ratio: 20/15 = 1.33
    // Need at least 50x40, so scale by max(50/20, 40/15) = max(2.5, 2.67) = 2.67
    // Final size: 20*2.67=53.4≈53, 15*2.67=40.05≈40
    assert!(scaled.width() >= 50);
    assert!(scaled.height() >= 40);
}

#[tokio::test]
async fn test_watermark_scaling_with_max_size() {
    let engine = WatermarkEngine::new();
    
    // Create a large watermark
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(1000, 800));
    let base_width = 800;
    let base_height = 600;
    let scale = 1.0;
    
    let scaling_options = WatermarkScalingOptions {
        preserve_aspect_ratio: true,
        max_width_percent: None,
        max_height_percent: None,
        min_size: None,
        max_size: Some((300, 200)),
    };
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    // Should be constrained by max size
    // Aspect ratio: 1000/800 = 1.25
    // Max size 300x200, so scale by min(300/1000, 200/800) = min(0.3, 0.25) = 0.25
    // Final size: 1000*0.25=250, 800*0.25=200
    assert_eq!(scaled.width(), 250);
    assert_eq!(scaled.height(), 200);
}

#[tokio::test]
async fn test_watermark_scaling_without_aspect_ratio_preservation() {
    let engine = WatermarkEngine::new();
    
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(200, 100));
    let base_width = 800;
    let base_height = 600;
    let scale = 1.0;
    
    let scaling_options = WatermarkScalingOptions {
        preserve_aspect_ratio: false,
        max_width_percent: Some(0.5), // 400px max width
        max_height_percent: Some(0.5), // 300px max height
        min_size: None,
        max_size: None,
    };
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    // Without aspect ratio preservation, should fit within both constraints
    assert!(scaled.width() <= 400);
    assert!(scaled.height() <= 300);
}

#[tokio::test]
async fn test_watermark_config_defaults() {
    let config = WatermarkConfig::default();
    
    assert_eq!(config.positions, vec![WatermarkPosition::BottomRight]);
    assert_eq!(config.opacity, 0.8);
    assert_eq!(config.scale, 0.2);
    assert!(matches!(config.blend_mode, BlendMode::Normal));
    assert!(config.scaling_options.preserve_aspect_ratio);
    assert_eq!(config.scaling_options.max_width_percent, Some(0.3));
    assert_eq!(config.scaling_options.max_height_percent, Some(0.3));
    assert!(matches!(config.alignment, WatermarkAlignment::Padded { padding: 10 }));
    assert_eq!(config.offset.x, 0);
    assert_eq!(config.offset.y, 0);
}

#[tokio::test]
async fn test_watermark_scaling_options_defaults() {
    let options = WatermarkScalingOptions::default();
    
    assert!(options.preserve_aspect_ratio);
    assert_eq!(options.max_width_percent, Some(0.3));
    assert_eq!(options.max_height_percent, Some(0.3));
    assert_eq!(options.min_size, Some((10, 10)));
    assert_eq!(options.max_size, None);
}

#[tokio::test]
async fn test_watermark_alignment_defaults() {
    let alignment = WatermarkAlignment::default();
    
    assert!(matches!(alignment, WatermarkAlignment::Padded { padding: 10 }));
}

#[tokio::test]
async fn test_watermark_offset_defaults() {
    let offset = WatermarkOffset::default();
    
    assert_eq!(offset.x, 0);
    assert_eq!(offset.y, 0);
}

#[tokio::test]
async fn test_position_bounds_checking() {
    let image_width = 100;
    let image_height = 100;
    let watermark_width = 50;
    let watermark_height = 50;
    let alignment = WatermarkAlignment::Edge;
    
    // Test large positive offset
    let offset = WatermarkOffset { x: 1000, y: 1000 };
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::TopLeft,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (50, 50)); // Clamped to max valid position
    
    // Test large negative offset
    let offset = WatermarkOffset { x: -1000, y: -1000 };
    let (x, y) = PositionCalculator::calculate_position(
        WatermarkPosition::BottomRight,
        image_width, image_height,
        watermark_width, watermark_height,
        alignment, offset,
    );
    assert_eq!((x, y), (0, 0)); // Clamped to minimum position
}

#[tokio::test]
async fn test_watermark_scaling_edge_cases() {
    let engine = WatermarkEngine::new();
    
    // Test zero-sized watermark (should be handled gracefully)
    let watermark = DynamicImage::ImageRgba8(RgbaImage::new(1, 1));
    let base_width = 800;
    let base_height = 600;
    let scale = 0.0001; // Very small scale
    
    let scaling_options = WatermarkScalingOptions {
        preserve_aspect_ratio: true,
        max_width_percent: None,
        max_height_percent: None,
        min_size: Some((1, 1)),
        max_size: None,
    };
    
    let scaled = engine.scale_watermark_advanced(
        watermark,
        base_width,
        base_height,
        scale,
        &scaling_options,
    );
    
    // Should maintain minimum viable size
    assert!(scaled.width() >= 1);
    assert!(scaled.height() >= 1);
}

#[tokio::test]
async fn test_multiple_watermark_positions() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create test images
    let base_image_path = temp_dir.path().join("base.png");
    let watermark_path = temp_dir.path().join("watermark.png");
    
    // Create base image
    let base_image = DynamicImage::ImageRgba8(RgbaImage::new(400, 300));
    base_image.save(&base_image_path).unwrap();
    
    // Create watermark
    fs::write(&watermark_path, create_test_png()).unwrap();
    
    // Create config with multiple positions
    let config = WatermarkConfig {
        watermark_path: watermark_path.clone(),
        positions: vec![
            WatermarkPosition::TopLeft,
            WatermarkPosition::TopRight,
            WatermarkPosition::BottomLeft,
            WatermarkPosition::BottomRight,
        ],
        opacity: 0.8,
        scale: 0.1,
        blend_mode: BlendMode::Normal,
        scaling_options: WatermarkScalingOptions::default(),
        alignment: WatermarkAlignment::Padded { padding: 10 },
        offset: WatermarkOffset::default(),
        visual_effects: WatermarkVisualEffects::default(),
    };
    
    // Apply watermarks
    let result = engine.apply_watermark_to_image(base_image, &config).await;
    assert!(result.is_ok());
    
    let watermarked = result.unwrap();
    assert_eq!(watermarked.width(), 400);
    assert_eq!(watermarked.height(), 300);
}

#[tokio::test]
async fn test_watermark_with_custom_alignment_and_offset() {
    let engine = WatermarkEngine::new();
    let temp_dir = tempdir().unwrap();
    
    // Create test images
    let base_image_path = temp_dir.path().join("base.png");
    let watermark_path = temp_dir.path().join("watermark.png");
    
    // Create base image
    let base_image = DynamicImage::ImageRgba8(RgbaImage::new(400, 300));
    base_image.save(&base_image_path).unwrap();
    
    // Create watermark
    fs::write(&watermark_path, create_test_png()).unwrap();
    
    // Create config with custom alignment and offset
    let config = WatermarkConfig {
        watermark_path: watermark_path.clone(),
        positions: vec![WatermarkPosition::Center],
        opacity: 0.8,
        scale: 0.2,
        blend_mode: BlendMode::Normal,
        scaling_options: WatermarkScalingOptions {
            preserve_aspect_ratio: true,
            max_width_percent: Some(0.5),
            max_height_percent: Some(0.5),
            min_size: Some((20, 20)),
            max_size: None,
        },
        alignment: WatermarkAlignment::Custom {
            horizontal: HorizontalAlign::Right,
            vertical: VerticalAlign::Bottom,
        },
        offset: WatermarkOffset { x: -20, y: -15 },
        visual_effects: WatermarkVisualEffects::default(),
    };
    
    // Apply watermark
    let result = engine.apply_watermark_to_image(base_image, &config).await;
    assert!(result.is_ok());
    
    let watermarked = result.unwrap();
    assert_eq!(watermarked.width(), 400);
    assert_eq!(watermarked.height(), 300);
}