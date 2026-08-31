"""
Centralized types for the traffic tracking system.

This module exports all types used throughout the system.
Maintains compatibility with existing code that imports from core.types.

This module serves as the central type registry, ensuring consistent
type definitions across all system components.

Submodules:
    - core.types.detection: Detection-related types
    - core.types.geometry: Geometry and spatial types
    - core.types.matching: Matching and association types
    - core.types.prediction: Prediction and trajectory types
    - core.types.stats: Statistics and counting types
    - core.types.track: Tracking and state types

Example:
    >>> from core.types import Point, BoundingBox, TrackState
    >>>
    >>> # Create a point
    >>> centroid = Point(100, 200)
    >>>
    >>> # Create a bounding box
    >>> bbox = BoundingBox(50, 50, 150, 150)
    >>>
    >>> # Create a track state
    >>> track = create_track_state(track_id=1, centroid=centroid, bbox=bbox, confidence=0.9)
"""

from core.types.detection import (
    DetectionDict,
    DetectionList,
    DetectionValidationResult,
)
from core.types.geometry import (
    Acceleration,
    BBoxHistory,
    BoundingBox,
    Centroid,
    Color,
    ColorWithAlpha,
    FloatBoundingBox,
    FloatPoint,
    Point,
    Velocity,
)
from core.types.matching import (
    MatchLevel,
    MatchResult,
    ScoreMatrix,
)
from core.types.prediction import (
    PredictionData,
    TrajectoryPrediction,
    TrajectoryState,
)
from core.types.stats import (
    CountingLineDict,
    EventsList,
    LineList,
    StatsDict,
)
from core.types.track import (
    TrackDataDict,
    TrackInfoDict,
    TracksDict,
    TrackState,
    TrackStateDict,
    TrackStatus,
)

ConfigDict = dict[str, any]
"""Generic configuration dictionary type."""

MetadataDict = dict[str, any]
"""Generic metadata dictionary type."""


def is_valid_detection(detection: DetectionDict) -> bool:
    """Checks if a detection has all required fields.

    Args:
        detection: Detection dictionary to validate.

    Returns:
        bool: True if the detection has all required fields.

    Example:
        >>> detection = {"box": [0, 0, 10, 10], "centroid": (5, 5), "confidence": 0.9}
        >>> is_valid_detection(detection)
        True
        >>>
        >>> invalid = {"box": [0, 0, 10, 10]}
        >>> is_valid_detection(invalid)
        False
    """
    required = ["box", "centroid", "confidence"]
    return all(field in detection for field in required)


def detection_to_dict(detection: DetectionDict) -> dict[str, any]:
    """Converts a detection to a dictionary, filtering out None values.

    Args:
        detection: Detection dictionary to convert.

    Returns:
        dict[str, any]: Dictionary with all non-None key-value pairs.

    Example:
        >>> detection = {"box": [0, 0, 10, 10], "label": "car", "metadata": None}
        >>> detection_to_dict(detection)
        {"box": [0, 0, 10, 10], "label": "car"}
    """
    return {k: v for k, v in detection.items() if v is not None}


def track_data_to_dict(track: TrackDataDict) -> dict[str, any]:
    """Converts track data to a dictionary, filtering out None values.

    Args:
        track: Track data dictionary to convert.

    Returns:
        dict[str, any]: Dictionary with all non-None key-value pairs.

    Example:
        >>> track = {"track_id": 1, "centroid": (100, 200), "velocity": None}
        >>> track_data_to_dict(track)
        {"track_id": 1, "centroid": (100, 200)}
    """
    return {k: v for k, v in track.items() if v is not None}


def create_track_state(track_id: int, centroid: Point, bbox: BoundingBox, **kwargs) -> TrackState:
    """Creates a TrackState with basic validation.

    This helper function simplifies the creation of TrackState objects
    with required fields and optional additional attributes.

    Args:
        track_id: Track ID.
        centroid: Centroid (x, y) position.
        bbox: Bounding box (x1, y1, x2, y2).
        **kwargs: Additional arguments for TrackState.

    Returns:
        TrackState: Created track state.

    Example:
        >>> track = create_track_state(
        ...     track_id=1,
        ...     centroid=(100, 100),
        ...     bbox=(50, 50, 150, 150),
        ...     confidence=0.9,
        ...     label="car",
        ...     velocity=(5.0, 3.0),
        ... )
        >>> print(f"Track {track.track_id} at {track.centroid}")
    """
    return TrackState(track_id=track_id, centroid=centroid, bbox=bbox, **kwargs)


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
    "DetectionDict",
    "DetectionList",
    "DetectionValidationResult",
    "TrackDataDict",
    "TrackState",
    "TrackStatus",
    "TrackInfoDict",
    "TrackStateDict",
    "TracksDict",
    "MatchLevel",
    "MatchResult",
    "ScoreMatrix",
    "StatsDict",
    "CountingLineDict",
    "LineList",
    "EventsList",
    "TrajectoryState",
    "TrajectoryPrediction",
    "PredictionData",
    "ConfigDict",
    "MetadataDict",
    "is_valid_detection",
    "detection_to_dict",
    "track_data_to_dict",
    "create_track_state",
]
