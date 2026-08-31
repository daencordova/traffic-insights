"""Geometric types for the system.

This module defines type aliases for all geometric data structures
used throughout the system, including points, bounding boxes, velocity,
acceleration, and color types.

Example:
    >>> from core.types import Point, BoundingBox, Velocity
    >>>
    >>> # Create a point
    >>> centroid: Point = (100, 200)
    >>>
    >>> # Create a bounding box
    >>> bbox: BoundingBox = (50, 50, 150, 150)
    >>>
    >>> # Create a velocity vector
    >>> velocity: Velocity = (5.2, -3.1)
"""

from typing import TypeAlias

Point: TypeAlias = tuple[int, int]
"""A 2D point in integer coordinates (x, y).

Example:
    >>> position: Point = (100, 200)
"""

FloatPoint: TypeAlias = tuple[float, float]
"""A 2D point in floating-point coordinates (x, y).

Example:
    >>> position: FloatPoint = (100.5, 200.3)
"""

BoundingBox: TypeAlias = tuple[int, int, int, int]
"""A bounding box in integer format (x1, y1, x2, y2).

Example:
    >>> bbox: BoundingBox = (10, 20, 30, 40)  # x1, y1, x2, y2
"""

FloatBoundingBox: TypeAlias = tuple[float, float, float, float]
"""A bounding box in floating-point format (x1, y1, x2, y2).

Example:
    >>> bbox: FloatBoundingBox = (10.5, 20.3, 30.7, 40.1)
"""

Velocity: TypeAlias = tuple[float, float]
"""Velocity vector (vx, vy) in pixels per frame.

Example:
    >>> velocity: Velocity = (5.2, -3.1)  # Moving right and down
"""

Acceleration: TypeAlias = tuple[float, float]
"""Acceleration vector (ax, ay) in pixels per frame².

Example:
    >>> acceleration: Acceleration = (0.5, -0.2)
"""

Color: TypeAlias = tuple[int, int, int]
"""Color in BGR format (blue, green, red), each value 0-255.

Example:
    >>> # Blue color
    >>> blue: Color = (255, 0, 0)
    >>> # Green color
    >>> green: Color = (0, 255, 0)
    >>> # Red color
    >>> red: Color = (0, 0, 255)
"""

ColorWithAlpha: TypeAlias = tuple[int, int, int, float]
"""Color in BGRA format (blue, green, red, alpha).
Alpha values range from 0.0 (transparent) to 1.0 (opaque).

Example:
    >>> # Semi-transparent blue
    >>> blue_alpha: ColorWithAlpha = (255, 0, 0, 0.5)
    >>> # Fully opaque green
    >>> green_alpha: ColorWithAlpha = (0, 255, 0, 1.0)
"""

Centroid = Point
"""Alias for Point representing a centroid (cx, cy).

Example:
    >>> centroid: Centroid = (100, 200)
"""

BBoxHistory = list[BoundingBox]
"""List of bounding boxes representing track history.

Example:
    >>> history: BBoxHistory = [
    ...     (10, 20, 30, 40),
    ...     (12, 22, 32, 42),
    ...     (14, 24, 34, 44)
    ... ]
"""


__all__ = [
    "Point",
    "FloatPoint",
    "BoundingBox",
    "FloatBoundingBox",
    "Velocity",
    "Acceleration",
    "Color",
    "ColorWithAlpha",
    "Centroid",
    "BBoxHistory",
]
