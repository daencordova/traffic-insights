"""
General utilities module.

This module provides various utility functions and classes used
throughout the system, including geometry calculations, logging,
color management, and file system helpers.

Submodules:
    - utils.geometry: Geometry calculations (centroid, IoU, distance, crossing detection)
    - utils.helpers: File system helpers (directory creation, timestamp filenames)
    - utils.logger: Logging utilities (structured logging, mixins, configuration)
    - utils.color_manager: Color management for visualization

Features:
    - Geometry utilities for tracking and detection
    - File system operations for data management
    - Structured logging with context and mixins
    - Consistent color management for visualization
    - Helper functions for common operations

Example:
    >>> from utils import calculate_centroid, calculate_iou, setup_logging, get_color_manager
    >>>
    >>> # Calculate centroid
    >>> centroid = calculate_centroid(10, 20, 50, 60)
    >>> print(f"Centroid: {centroid}")  # (30, 40)
    >>>
    >>> # Calculate IoU between two bounding boxes
    >>> iou = calculate_iou(bbox1, bbox2)
    >>> print(f"IoU: {iou:.2f}")
    >>>
    >>> # Setup logging
    >>> setup_logging(log_level="INFO")
    >>>
    >>> # Get color for visualization
    >>> color_manager = get_color_manager()
    >>> color = color_manager.get_color("car")
    >>>
    >>> # Create directory
    >>> ensure_directory_exists("data/output/")
    >>>
    >>> # Generate timestamped filename
    >>> filename = get_timestamp_filename("output")
    >>> print(filename)  # output_20240101_120000.json
"""

from utils.color_manager import ColorManager, get_color, get_color_manager
from utils.geometry import (
    calculate_centroid,
    calculate_iou,
    check_crossing,
    euclidean_distance,
    point_in_bbox,
)
from utils.helpers import (
    ensure_directory_exists,
    format_time,
    get_timestamp_filename,
)
from utils.logger import (
    LoggerMixin,
    StructuredLogger,
    config,
    get_default_logger,
    get_logger,
    get_logger_for_class,
    get_logging_status,
    log_context,
    set_module_level,
    setup_logging,
    temporary_log_level,
)

__all__ = [
    "calculate_centroid",
    "calculate_iou",
    "check_crossing",
    "euclidean_distance",
    "point_in_bbox",
    "ensure_directory_exists",
    "get_timestamp_filename",
    "format_time",
    "setup_logging",
    "get_logger",
    "get_logger_for_class",
    "get_default_logger",
    "StructuredLogger",
    "LoggerMixin",
    "config",
    "get_logging_status",
    "set_module_level",
    "temporary_log_level",
    "log_context",
    "ColorManager",
    "get_color_manager",
    "get_color",
]
