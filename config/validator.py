"""Configuration validator with cross-validation capabilities.

This module provides validation functions for system configuration,
including directory creation, model verification, performance warnings,
and cross-field consistency checks.
"""

import logging
from pathlib import Path

from ultralytics import YOLO

from core.constants import MAX_RECOMMENDED_FPS, MINIMUM_BUFFER_MEMORY_MB
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_config(config) -> list[str]:
    """Validates the system configuration.

    This function performs comprehensive validation of the configuration object,
    including:
    - Creating required output directories
    - Verifying model file existence (or downloading if available)
    - Checking performance implications of settings
    - Validating cross-field consistency
    - Warning about potential performance issues

    Args:
        config: Pydantic configuration object to validate.

    Returns:
        List[str]: List of warning messages (empty if all checks pass).

    Raises:
        ConfigurationError: If critical configuration errors are detected.

    Example:
        >>> config = Config()
        >>> warnings = validate_config(config)
        >>> if warnings:
        ...     for warning in warnings:
        ...         print(f"Warning: {warning}")
    """
    warnings = []

    output_dirs = [
        ("screenshots", config.output.screenshots_dir),
        ("exports", config.output.export_dir),
        ("logs", config.output.logs_dir),
    ]

    for name, path in output_dirs:
        dir_path = Path(path)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created: {path}")
            except OSError as e:
                raise ConfigurationError(
                    f"Could not create directory '{name}': {e}",
                    details={"path": str(path), "error": str(e)},
                ) from e

    model_path = Path(config.model.model_path)
    if not model_path.exists():
        if model_path.suffix == ".pt" and model_path.stem.startswith("yolo"):
            try:
                logger.info(f"Downloading model {model_path.name}...")
                YOLO(model_path.name)
                logger.info(f"Model {model_path.name} downloaded successfully")
            except Exception as e:
                raise ConfigurationError(
                    f"Model not found and could not be downloaded: {model_path}",
                    details={"model_path": str(model_path), "error": str(e)},
                ) from e
        else:
            raise ConfigurationError(
                f"Model file not found: {model_path}",
                details={"model_path": str(model_path)},
            )

    if config.model.device == "cpu" and config.tracker.enable_reidentification:
        warnings.append(
            "Re-identification with features on CPU may be very slow. "
            "Consider disabling 'enable_reidentification' or using GPU."
        )

    memory_per_frame = config.camera.width * config.camera.height * 3
    buffer_memory_mb = (memory_per_frame * config.camera.buffer_size) / (1024 * 1024)

    if buffer_memory_mb > MINIMUM_BUFFER_MEMORY_MB:
        warnings.append(
            f"Buffer of {config.camera.buffer_size} frames uses "
            f"~{buffer_memory_mb:.0f} MB of RAM. Consider reducing it."
        )

    if config.model.imgsz != config.camera.width:
        warnings.append(
            f"Model expects {config.model.imgsz}x{config.model.imgsz} but "
            f"camera provides {config.camera.width}x{config.camera.height}. "
            "It will be resized automatically, but this may affect performance."
        )

    if not config.counting_lines:
        warnings.append("No counting lines configured. The system will not count vehicles.")

    if config.camera.fps and config.camera.fps > MAX_RECOMMENDED_FPS:
        warnings.append(
            f"Configured FPS ({config.camera.fps}) is very high. "
            "Consider reducing it for better performance."
        )

    return warnings


def validate_config_required_fields(config) -> None:
    """Validates that all required fields are present in the configuration.

    This function checks for the presence of essential configuration fields
    that are critical for system operation. Missing fields will raise an
    error with detailed information about what's missing.

    Args:
        config: Pydantic configuration object to validate.

    Raises:
        ConfigurationError: If any required field is missing.

    Example:
        >>> try:
        ...     validate_config_required_fields(config)
        ... except ConfigurationError as e:
        ...     print(f"Missing required fields: {e}")
    """
    required_fields = [
        ("model", "model_path"),
        ("camera", "source"),
        ("tracker", "type"),
    ]

    missing = []
    for section, field in required_fields:
        obj = getattr(config, section, None)
        if obj is None or not hasattr(obj, field) or getattr(obj, field) is None:
            missing.append(f"{section}.{field}")

    if missing:
        raise ConfigurationError(
            f"Missing required fields: {', '.join(missing)}",
            details={"missing_fields": missing},
        )
