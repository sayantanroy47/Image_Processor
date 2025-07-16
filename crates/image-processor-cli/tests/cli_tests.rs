//! CLI integration tests

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_info_command() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Image Processor"))
        .stdout(predicate::str::contains("System Information"))
        .stdout(predicate::str::contains("CPU cores"));
}

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("High-performance cross-platform image processing tool"))
        .stdout(predicate::str::contains("convert"))
        .stdout(predicate::str::contains("watermark"))
        .stdout(predicate::str::contains("batch"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("0.1.0"));
}

#[test]
fn test_convert_command_placeholder() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.args(&["convert", "-i", "input.jpg", "-o", "output.png", "-f", "png"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Conversion functionality will be implemented"));
}

#[test]
fn test_watermark_command_placeholder() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.args(&["watermark", "-i", "input.jpg", "-o", "output.jpg", "-w", "watermark.png"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Watermark functionality will be implemented"));
}

#[test]
fn test_batch_command_placeholder() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.args(&["batch", "-i", "input_dir", "-o", "output_dir", "-c", "config.toml"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Batch processing functionality will be implemented"));
}

#[test]
fn test_invalid_command() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.arg("invalid-command")
        .assert()
        .failure();
}

#[test]
fn test_verbose_flag() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.args(&["--verbose", "info"])
        .assert()
        .success();
}

#[test]
fn test_config_flag() {
    let mut cmd = Command::cargo_bin("image-processor").unwrap();
    cmd.args(&["--config", "custom_config.toml", "info"])
        .assert()
        .success();
}