"""Object detection validator.

Provides functions for validating individual detections and lists,
ensuring they meet system requirements and constraints.

This module handles:
    - Single detection validation (required fields, confidence, bbox, centroid)
    - Detection list validation and filtering
    - Required fields checking
    - Detection statistics computation
    - Confidence and area validation

Example:
    >>> from core.validators import (
    ...     validate_detection,
    ...     validate_detection_list,
    ...     filter_valid_detections,
    ...     get_detection_stats,
    ... )
    >>>
    >>> # Validate a single detection
    >>> detection = {
    ...     "box": [100, 200, 300, 400],
    ...     "centroid": [200, 300],
    ...     "confidence": 0.95,
    ...     "class_id": 2,
    ...     "label": "car",
    ... }
    >>> result = validate_detection(detection)
    >>> if result.is_valid:
    ...     print("Detection is valid")
    >>> else:
    ...     print(f"Errors: {result.errors}")
    >>>
    >>> # Filter valid detections
    >>> valid = filter_valid_detections(detections, min_confidence=0.5)
    >>>
    >>> # Get statistics
    >>> stats = get_detection_stats(valid)
    >>> print(f"Average confidence: {stats['avg_confidence']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.constants import (
    MAX_DETECTION_CONFIDENCE,
    MIN_DETECTION_AREA,
    MIN_DETECTION_CONFIDENCE,
)
from core.validators.bbox_validator import validate_bbox, validate_centroid


@dataclass(slots=True)
class DetectionValidationResult:
    """Result of a detection validation.

    Attributes:
        is_valid: Whether the detection passed validation.
        errors: List of error messages (critical issues).
        warnings: List of warning messages (non-critical issues).
        score: Validation score (0.0 to 1.0).

    Example:
        >>> result = DetectionValidationResult(
        ...     is_valid=True, errors=[], warnings=["Low confidence: 0.45"], score=0.85
        ... )
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    score: float


def validate_detection(
    detection: dict[str, Any],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    max_confidence: float = MAX_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA,
    require_all_fields: bool = True,
) -> DetectionValidationResult:
    """Validates a complete detection.

    This function performs comprehensive validation of a detection:
        - Required fields presence (box, centroid, confidence)
        - Bounding box validity (format, size, boundaries)
        - Centroid validity (format, boundaries)
        - Confidence range validation
        - Area validation
        - Class ID validation
        - Label validation

    Args:
        detection: Detection dictionary to validate.
        min_confidence: Minimum allowed confidence (default: MIN_DETECTION_CONFIDENCE).
        max_confidence: Maximum allowed confidence (default: MAX_DETECTION_CONFIDENCE).
        min_area: Minimum allowed area in pixels (default: MIN_DETECTION_AREA).
        require_all_fields: Whether all required fields must exist.

    Returns:
        DetectionValidationResult: Validation result with errors, warnings, and score.

    Example:
        >>> detection = {
        ...     "box": [100, 200, 300, 400],
        ...     "centroid": [200, 300],
        ...     "confidence": 0.95,
        ...     "class_id": 2,
        ... }
        >>> result = validate_detection(detection)
        >>> if result.is_valid:
        ...     print(f"Score: {result.score:.2f}")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """
    errors = []
    warnings = []
    score = 1.0

    if not isinstance(detection, dict):
        errors.append("Detection must be a dictionary")
        return DetectionValidationResult(False, errors, warnings, 0.0)

    required_fields = ["box", "centroid", "confidence"]
    if require_all_fields:
        missing = [f for f in required_fields if f not in detection]
        if missing:
            errors.append(f"Missing required fields: {missing}")
            score *= 0.5

    box = detection.get("box")
    if box is not None:
        if not validate_bbox(box):
            errors.append(f"Invalid bounding box: {box}")
            score *= 0.3
    elif require_all_fields:
        errors.append("Missing 'box' field")
        score *= 0.3

    centroid = detection.get("centroid")
    if centroid is not None:
        if not validate_centroid(centroid):
            errors.append(f"Invalid centroid: {centroid}")
            score *= 0.3
    elif require_all_fields:
        errors.append("Missing 'centroid' field")
        score *= 0.3

    confidence = detection.get("confidence")
    if confidence is not None:
        try:
            conf = float(confidence)
            if conf < min_confidence or conf > max_confidence:
                warnings.append(
                    f"Confidence out of range [{min_confidence}, {max_confidence}]: {conf}"
                )
                score *= 0.7
        except (TypeError, ValueError):
            errors.append(f"Invalid confidence: {confidence}")
            score *= 0.5
    elif require_all_fields:
        errors.append("Missing 'confidence' field")
        score *= 0.5

    if box and validate_bbox(box):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            warnings.append(f"Area too small: {area} < {min_area}")
            score *= 0.7

    class_id = detection.get("class_id", 0)
    if class_id is not None and not isinstance(class_id, int) or class_id < 0:
        errors.append(f"Invalid class_id: {class_id}")
        score *= 0.5

    label = detection.get("label")
    if label is not None and not isinstance(label, str):
        errors.append(f"Invalid label: {label}")
        score *= 0.5

    is_valid = len(errors) == 0
    return DetectionValidationResult(is_valid, errors, warnings, max(0.0, min(1.0, score)))


def validate_detection_list(
    detections: list[dict[str, Any]],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    max_confidence: float = MAX_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA,
) -> tuple[list[dict[str, Any]], list[DetectionValidationResult]]:
    """Validates a list of detections.

    This function validates each detection in a list and returns
    both the valid detections and individual validation results.

    Args:
        detections: List of detections to validate.
        min_confidence: Minimum allowed confidence.
        max_confidence: Maximum allowed confidence.
        min_area: Minimum allowed area in pixels.

    Returns:
        Tuple[List[Dict], List[DetectionValidationResult]]:
            List of valid detections and list of validation results.

    Example:
        >>> detections = [det1, det2, det3]
        >>> valid, results = validate_detection_list(detections)
        >>> print(f"Found {len(valid)} valid detections")
        >>> for i, result in enumerate(results):
        ...     if not result.is_valid:
        ...         print(f"Detection {i} invalid: {result.errors}")
    """
    valid_detections = []
    results = []

    for detection in detections:
        result = validate_detection(
            detection, min_confidence, max_confidence, min_area, require_all_fields=True
        )
        results.append(result)
        if result.is_valid:
            valid_detections.append(detection)

    return valid_detections, results


def validate_detection_required_fields(detection: dict[str, Any]) -> bool:
    """Quick check if a detection has all required fields.

    Args:
        detection: Detection to check.

    Returns:
        bool: True if all required fields are present.

    Example:
        >>> if validate_detection_required_fields(detection):
        ...     print("All required fields present")
    """
    required = ["box", "centroid", "confidence"]
    return all(field in detection for field in required)


def filter_valid_detections(
    detections: list[dict[str, Any]],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA,
) -> list[dict[str, Any]]:
    """Filters valid detections based on confidence and area.

    This function provides a fast filtering mechanism for detections,
    checking confidence threshold and area requirements without performing
    a full validation.

    Args:
        detections: List of detections.
        min_confidence: Minimum confidence threshold.
        min_area: Minimum area in pixels.

    Returns:
        List[Dict[str, Any]]: List of valid detections.

    Example:
        >>> # Keep only high-confidence detections with reasonable area
        >>> valid = filter_valid_detections(detections, min_confidence=0.6, min_area=100)
        >>> print(f"Filtered {len(valid)} detections")
    """
    valid = []

    for det in detections:
        confidence = det.get("confidence", 0.0)
        if confidence < min_confidence:
            continue

        box = det.get("box")
        if not box or not validate_bbox(box):
            continue

        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        valid.append(det)

    return valid


def get_detection_stats(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Gets statistics for a list of detections.

    This function computes aggregate statistics including confidence
    distribution, class distribution, and area statistics.

    Args:
        detections: List of detections.

    Returns:
        Dict[str, Any]: Detection statistics including:
            - count: Total number of detections
            - avg_confidence: Average confidence score
            - min_confidence: Minimum confidence score
            - max_confidence: Maximum confidence score
            - class_distribution: Count per class ID
            - avg_area: Average bounding box area

    Example:
        >>> stats = get_detection_stats(detections)
        >>> print(f"Total detections: {stats['count']}")
        >>> print(f"Average confidence: {stats['avg_confidence']:.2f}")
        >>> print(f"Class distribution: {stats['class_distribution']}")
    """
    if not detections:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "class_distribution": {},
            "avg_area": 0.0,
        }

    confidences = []
    class_counts = {}
    areas = []

    for det in detections:
        conf = det.get("confidence", 0.0)
        confidences.append(conf)

        class_id = det.get("class_id", -1)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

        box = det.get("box")
        if box and len(box) == 4:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

    return {
        "count": len(detections),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "class_distribution": class_counts,
        "avg_area": sum(areas) / len(areas) if areas else 0.0,
    }
