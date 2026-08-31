"""
Centralized validators module for the system.

Provides reusable validation functions for frames, detections,
bounding boxes, and centroids throughout the system.

This module consolidates all validation logic into a single location
to ensure consistency across the codebase.

Submodules:
    - core.validators.frame_validator: Frame validation utilities
    - core.validators.bbox_validator: Bounding box validation utilities
    - core.validators.detection_validator: Detection validation utilities

Features:
    - Frame validation (size, shape, color/grayscale detection)
    - Bounding box validation (format, area, normalization)
    - Detection validation (required fields, filtering, statistics)
    - Centroid validation
    - Type conversion utilities (bbox <-> numpy arrays)

Example:
    >>> from core.validators import (
    ...     validate_frame,
    ...     validate_bbox,
    ...     validate_detection,
    ...     DetectionValidationResult,
    ... )
    >>>
    >>> # Validate a frame
    >>> if validate_frame(frame, min_width=100, min_height=100):
    ...     print("Frame is valid")
    >>>
    >>> # Validate a bounding box
    >>> bbox = [100, 200, 300, 400]  # x1, y1, x2, y2
    >>> if validate_bbox(bbox):
    ...     print("BBox is valid")
    >>>
    >>> # Validate a detection
    >>> detection = {
    ...     "box": [100, 200, 300, 400],
    ...     "centroid": [200, 300],
    ...     "confidence": 0.95,
    ...     "class_id": 2,
    ... }
    >>> result = validate_detection(detection)
    >>> if result.is_valid:
    ...     print("Detection is valid")
"""

from core.validators.bbox_validator import (
    bbox_to_numpy,
    get_bbox_area,
    is_bbox_valid,
    normalize_bbox,
    numpy_to_bbox,
    validate_bbox,
    validate_bbox_list,
    validate_centroid,
)
from core.validators.detection_validator import (
    DetectionValidationResult,
    filter_valid_detections,
    get_detection_stats,
    validate_detection,
    validate_detection_list,
    validate_detection_required_fields,
)
from core.validators.frame_validator import (
    create_default_frame,
    ensure_valid_frame,
    get_frame_dimensions,
    is_color,
    is_grayscale,
    validate_frame,
    validate_frame_shape,
)

__all__ = [
    "validate_frame",
    "validate_frame_shape",
    "ensure_valid_frame",
    "get_frame_dimensions",
    "is_grayscale",
    "is_color",
    "create_default_frame",
    "validate_bbox",
    "validate_centroid",
    "normalize_bbox",
    "validate_bbox_list",
    "is_bbox_valid",
    "bbox_to_numpy",
    "numpy_to_bbox",
    "get_bbox_area",
    "validate_detection",
    "validate_detection_list",
    "validate_detection_required_fields",
    "filter_valid_detections",
    "DetectionValidationResult",
    "get_detection_stats",
]
