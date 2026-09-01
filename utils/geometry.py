"""Utility functions for geometry operations.

This module provides common geometry calculations used throughout
the system for tracking, detection, and counting operations.

Features:
    - Centroid calculation from bounding boxes
    - Line crossing detection
    - Euclidean distance between points
    - Point-in-bounding-box checking
    - IoU (Intersection over Union) calculation
    - Color palette for visualization

Example:
    >>> from utils.geometry import (
    ...     calculate_centroid,
    ...     calculate_iou,
    ...     check_crossing,
    ...     euclidean_distance,
    ...     point_in_bbox,
    ...     get_color,
    ... )
    >>>
    >>> # Calculate centroid
    >>> centroid = calculate_centroid(10, 20, 50, 60)
    >>> print(f"Centroid: {centroid}")  # (30, 40)
    >>>
    >>> # Calculate IoU
    >>> bbox1 = (10, 10, 50, 50)
    >>> bbox2 = (20, 20, 60, 60)
    >>> iou = calculate_iou(bbox1, bbox2)
    >>> print(f"IoU: {iou:.2f}")
    >>>
    >>> # Check line crossing
    >>> crossed = check_crossing(prev_y=100, current_y=150, line_y=120)
    >>> print(f"Crossed: {crossed}")  # True
    >>>
    >>> # Get distance between points
    >>> dist = euclidean_distance((0, 0), (3, 4))
    >>> print(f"Distance: {dist}")  # 5.0
"""

import numpy as np

from core.types import BoundingBox, FloatPoint, Point


def calculate_centroid(x1: int, y1: int, x2: int, y2: int) -> Point:
    """Calculates the centroid of a bounding box.

    Args:
        x1: X coordinate of the top-left corner.
        y1: Y coordinate of the top-left corner.
        x2: X coordinate of the bottom-right corner.
        y2: Y coordinate of the bottom-right corner.

    Returns:
        Point: Tuple (cx, cy) with the centroid coordinates.

    Example:
        >>> centroid = calculate_centroid(10, 20, 50, 60)
        >>> print(centroid)  # (30, 40)
    """
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def check_crossing(prev_y: int, current_y: int, line_y: int, direction: str = "down") -> bool:
    """Checks if an object has crossed a line.

    This function determines if an object has crossed a virtual line
    based on its previous and current Y positions and the direction
    of movement.

    Args:
        prev_y: Previous Y position of the object.
        current_y: Current Y position of the object.
        line_y: Y position of the counting line.
        direction: Crossing direction ('down' or 'up').

    Returns:
        bool: True if the object crossed the line, False otherwise.

    Example:
        >>> # Crossing downward
        >>> check_crossing(100, 150, 120, "down")
        True
        >>>
        >>> # Crossing upward
        >>> check_crossing(150, 100, 120, "up")
        True
        >>>
        >>> # Not crossing
        >>> check_crossing(100, 110, 120, "down")
        False
    """
    if direction.lower() == "down":
        return prev_y < line_y and current_y >= line_y
    if direction.lower() == "up":
        return prev_y > line_y and current_y <= line_y
    return False


def euclidean_distance(p1: Point | FloatPoint, p2: Point | FloatPoint) -> float:
    """Calculates the Euclidean distance between two points.

    Args:
        p1: First point as tuple (x, y).
        p2: Second point as tuple (x, y).

    Returns:
        float: Euclidean distance between the points.

    Example:
        >>> distance = euclidean_distance((0, 0), (3, 4))
        >>> print(distance)  # 5.0
        >>>
        >>> distance = euclidean_distance((10.5, 20.3), (15.7, 25.1))
        >>> print(f"{distance:.2f}")  # 7.43
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_in_bbox(point: Point, bbox: BoundingBox) -> bool:
    """Checks if a point is inside a bounding box.

    Args:
        point: Point to check as tuple (x, y).
        bbox: Bounding box as tuple (x1, y1, x2, y2).

    Returns:
        bool: True if the point is inside the bounding box.

    Example:
        >>> point_in_bbox((30, 40), (10, 20, 50, 60))
        True
        >>>
        >>> point_in_bbox((0, 0), (10, 20, 50, 60))
        False
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes.

    IoU is a metric used to measure the overlap between two bounding boxes.
    It ranges from 0 (no overlap) to 1 (perfect overlap).

    Args:
        bbox1: First bounding box as tuple (x1, y1, x2, y2).
        bbox2: Second bounding box as tuple (x1, y1, x2, y2).

    Returns:
        float: IoU value between 0 and 1.

    Example:
        >>> bbox1 = (10, 10, 50, 50)
        >>> bbox2 = (20, 20, 60, 60)
        >>> iou = calculate_iou(bbox1, bbox2)
        >>> print(f"IoU: {iou:.2f}")  # IoU: 0.14
        >>>
        >>> # Identical boxes
        >>> calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
        1.0
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_color(index: int) -> tuple:
    """Gets a color from the palette for identifying elements.

    This function provides a consistent color palette for visualization
    elements like tracks, detections, and annotations.

    Args:
        index: Index to select a color from the palette.

    Returns:
        tuple: Color as (B, G, R) tuple.

    Example:
        >>> color = get_color(0)
        >>> print(color)  # (0, 255, 0) - Green
        >>>
        >>> color = get_color(1)
        >>> print(color)  # (255, 165, 0) - Orange
        >>>
        >>> # Colors cycle through the palette
        >>> color1 = get_color(0)
        >>> color2 = get_color(8)  # Same as index 0
    """
    colors = [
        (0, 255, 0),
        (255, 165, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (128, 0, 255),
    ]
    return colors[index % len(colors)]
