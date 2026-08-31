"""Types for object detections.

This module defines type definitions and data structures used
for representing object detections throughout the system.
"""

from typing import TypeAlias, TypedDict

from core.types.geometry import BoundingBox, FloatPoint, Point


class DetectionDict(TypedDict, total=False):
    """Structure of an object detection.

    This TypedDict defines the complete structure of a detection,
    including both required and optional fields.

    Attributes:
        box: Bounding box in (x1, y1, x2, y2) format.
        centroid: Centroid in (cx, cy) format.
        confidence: Detection confidence score (0-1).
        class_id: Detected class ID.
        label: Class name/label.
        area: Bounding box area in pixels.
        features: Visual feature vector (optional).
        metadata: Additional metadata (optional).

    Example:
        >>> detection: DetectionDict = {
        ...     "box": (100, 200, 300, 400),
        ...     "centroid": (200, 300),
        ...     "confidence": 0.95,
        ...     "class_id": 2,
        ...     "label": "car",
        ...     "area": 40000,
        ...     "features": None,
        ...     "metadata": {"frame": 42},
        ... }
    """

    box: BoundingBox
    centroid: Point
    confidence: float
    class_id: int
    label: str
    area: int
    features: FloatPoint | None
    metadata: dict[str, any]


DetectionList: TypeAlias = list[DetectionDict]
"""List of detections. Type alias for a list of DetectionDict objects.

Example:
    >>> detections: DetectionList = [
    ...     {"box": (10, 20, 30, 40), "centroid": (20, 30), "confidence": 0.9},
    ...     {"box": (50, 60, 70, 80), "centroid": (60, 70), "confidence": 0.85}
    ... ]
"""


class DetectionValidationResult:
    """Result of detection validation.

    This class encapsulates the outcome of validating a detection,
    including validity status, errors, warnings, and a quality score.

    Attributes:
        is_valid: Whether the detection is valid.
        errors: List of errors found (critical issues).
        warnings: List of warnings (non-critical issues).
        score: Quality score (0.0 to 1.0).

    Example:
        >>> result = DetectionValidationResult(
        ...     is_valid=True, errors=[], warnings=["Low confidence: 0.45"], score=0.85
        ... )
        >>> if result.is_valid:
        ...     print("Detection passed validation")
        ... else:
        ...     print(f"Validation failed: {result.errors}")
    """

    __slots__ = ("is_valid", "errors", "warnings", "score")

    def __init__(
        self,
        *,
        is_valid: bool = True,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        score: float = 1.0,
    ):
        """Initializes the validation result.

        Args:
            is_valid: Whether the detection is valid.
            errors: List of errors found (critical issues).
            warnings: List of warnings (non-critical issues).
            score: Quality score (0.0 to 1.0).

        Example:
            >>> result = DetectionValidationResult(
            ...     is_valid=False,
            ...     errors=["Missing centroid field"],
            ...     warnings=["Low confidence"],
            ...     score=0.3,
            ... )
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.score = score

    def __repr__(self) -> str:
        """Human-readable representation of the validation result."""
        return (
            f"DetectionValidationResult(valid={self.is_valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )

    def to_dict(self) -> dict[str, any]:
        """Converts the validation result to a dictionary.

        Returns:
            dict[str, any]: Dictionary representation with all fields.

        Example:
            >>> result.to_dict()
            {"is_valid": True, "errors": [], "warnings": ["Low confidence"], "score": 0.85}
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "score": self.score,
        }


__all__ = [
    "DetectionDict",
    "DetectionList",
    "DetectionValidationResult",
]
