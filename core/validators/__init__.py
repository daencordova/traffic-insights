"""
Módulo de validadores centralizados para el sistema.

Proporciona funciones reutilizables para validar frames, detecciones,
bounding boxes y centroides en todo el sistema.
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
