"""Bounding box and centroid validator.

Provides functions for validating bounding boxes and centroids,
ensuring they meet system requirements and constraints.

This module handles:
    - Bounding box format validation (x1, y1, x2, y2)
    - Centroid format validation (x, y)
    - Size constraints (minimum and maximum dimensions)
    - Image boundary validation
    - Bounding box normalization and conversion utilities

Example:
    >>> from core.validators import validate_bbox, validate_centroid, normalize_bbox
    >>>
    >>> # Validate a bounding box
    >>> bbox = [100, 200, 300, 400]
    >>> if validate_bbox(bbox, image_shape=(720, 1280)):
    ...     print("BBox is valid")
    >>>
    >>> # Validate a centroid
    >>> centroid = (200, 300)
    >>> if validate_centroid(centroid, image_shape=(720, 1280)):
    ...     print("Centroid is valid")
    >>>
    >>> # Normalize a bounding box
    >>> normalized = normalize_bbox(bbox, image_shape=(720, 1280))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import MAX_BBOX_SIZE, MIN_BBOX_SIZE

if TYPE_CHECKING:
    from core.types import BoundingBox


def validate_bbox(
    bbox: Any,
    min_size: int = MIN_BBOX_SIZE,
    max_size: int = MAX_BBOX_SIZE,
    image_shape: tuple[int, int] | None = None,
) -> bool:
    """Validates a bounding box.

    This function checks if a bounding box meets all requirements:
        - Correct format (x1, y1, x2, y2)
        - Valid coordinate types (int or float)
        - Non-negative coordinates
        - x1 < x2 and y1 < y2
        - Minimum size constraints
        - Maximum size constraints
        - Within image boundaries (if provided)

    Args:
        bbox: Bounding box to validate (x1, y1, x2, y2).
        min_size: Minimum allowed size (default: MIN_BBOX_SIZE).
        max_size: Maximum allowed size (default: MAX_BBOX_SIZE).
        image_shape: Image dimensions (height, width) for boundary validation.

    Returns:
        bool: True if the bbox is valid.

    Example:
        >>> bbox = [100, 200, 300, 400]
        >>> validate_bbox(bbox)
        True
        >>>
        >>> # Invalid bbox (negative coordinates)
        >>> bbox = [-10, 200, 300, 400]
        >>> validate_bbox(bbox)
        False
        >>>
        >>> # With image boundary validation
        >>> validate_bbox(bbox, image_shape=(480, 640))
        True
    """
    if not isinstance(bbox, (tuple, list)):
        return False

    if len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = bbox

        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            return False

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False

        if x1 >= x2 or y1 >= y2:
            return False

        width = x2 - x1
        height = y2 - y1

        if width < min_size or height < min_size:
            return False

        if width > max_size or height > max_size:
            return False

        if image_shape is not None:
            h, w = image_shape[:2]
            if x1 >= w or y1 >= h or x2 > w or y2 > h:
                return False

        return True

    except (TypeError, ValueError):
        return False


def validate_centroid(centroid: Any, image_shape: tuple[int, int] | None = None) -> bool:
    """Validates a centroid.

    This function checks if a centroid meets all requirements:
        - Correct format (x, y)
        - Valid coordinate types (int or float)
        - Non-negative coordinates
        - Within image boundaries (if provided)

    Args:
        centroid: Centroid to validate (x, y).
        image_shape: Image dimensions (height, width) for boundary validation.

    Returns:
        bool: True if the centroid is valid.

    Example:
        >>> centroid = (200, 300)
        >>> validate_centroid(centroid)
        True
        >>>
        >>> # Invalid centroid (negative coordinates)
        >>> centroid = (-10, 300)
        >>> validate_centroid(centroid)
        False
        >>>
        >>> # With image boundary validation
        >>> validate_centroid(centroid, image_shape=(480, 640))
        True
    """
    if not isinstance(centroid, (tuple, list)):
        return False

    if len(centroid) != 2:
        return False

    try:
        x, y = centroid

        if not all(isinstance(v, (int, float)) for v in [x, y]):
            return False

        if x < 0 or y < 0:
            return False

        if image_shape is not None:
            h, w = image_shape[:2]
            if x >= w or y >= h:
                return False

        return True

    except (TypeError, ValueError):
        return False


def normalize_bbox(bbox: BoundingBox, image_shape: tuple[int, int]) -> BoundingBox:
    """Normalizes a bounding box to fit within image boundaries.

    This function clips the bounding box coordinates to ensure they
    stay within the image boundaries while maintaining a minimum size.

    Args:
        bbox: Bounding box to normalize (x1, y1, x2, y2).
        image_shape: Image dimensions (height, width).

    Returns:
        BoundingBox: Normalized bounding box.

    Example:
        >>> bbox = (-10, 200, 650, 500)  # Outside image bounds
        >>> normalized = normalize_bbox(bbox, image_shape=(480, 640))
        >>> print(normalized)
        (0, 200, 640, 480)
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return (x1, y1, x2, y2)


def validate_bbox_list(bboxes: list[Any]) -> list[bool]:
    """Validates a list of bounding boxes.

    Args:
        bboxes: List of bounding boxes to validate.

    Returns:
        List[bool]: List of validation results.

    Example:
        >>> bboxes = [[100, 200, 300, 400], [-10, 200, 300, 400]]
        >>> results = validate_bbox_list(bboxes)
        >>> print(results)  # [True, False]
    """
    return [validate_bbox(bbox) for bbox in bboxes]


def is_bbox_valid(bbox: Any) -> bool:
    """Quick validity check for a bounding box.

    This is a convenience alias for validate_bbox().

    Args:
        bbox: Bounding box to validate.

    Returns:
        bool: True if valid.

    Example:
        >>> if is_bbox_valid(bbox):
        ...     process_bbox(bbox)
    """
    return validate_bbox(bbox)


def bbox_to_numpy(bbox: BoundingBox) -> np.ndarray:
    """Converts a bounding box to a numpy array.

    Args:
        bbox: Bounding box to convert.

    Returns:
        np.ndarray: Array of 4 elements (float32).

    Example:
        >>> bbox = [100, 200, 300, 400]
        >>> arr = bbox_to_numpy(bbox)
        >>> print(arr.dtype)  # float32
    """
    return np.array(bbox, dtype=np.float32)


def numpy_to_bbox(arr: np.ndarray) -> BoundingBox:
    """Converts a numpy array to a bounding box.

    Args:
        arr: Array of 4 elements.

    Returns:
        BoundingBox: Bounding box as a tuple.

    Example:
        >>> arr = np.array([100, 200, 300, 400], dtype=np.float32)
        >>> bbox = numpy_to_bbox(arr)
        >>> print(bbox)  # (100, 200, 300, 400)
    """
    return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))


def get_bbox_area(bbox: BoundingBox) -> int:
    """Calculates the area of a bounding box.

    Args:
        bbox: Bounding box (x1, y1, x2, y2).

    Returns:
        int: Area of the bbox in pixels.

    Example:
        >>> bbox = [100, 200, 300, 400]
        >>> area = get_bbox_area(bbox)  # 200 * 200 = 40000
    """
    if not validate_bbox(bbox):
        return 0
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)
