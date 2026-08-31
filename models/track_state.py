"""Track state for the tracking system.

This module defines the complete state of a track, including its position,
history, motion metrics, and tracking status. The implementation is
optimized for memory-intensive use with __slots__.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import (
    MAX_BBOX_HISTORY_DISPLAY,
    MAX_BBOX_HISTORY_STORAGE,
    MAX_FRAMES_MISSED,
    MAX_TRACK_HISTORY,
    MIN_HISTORY_FOR_ACCELERATION,
    MIN_HISTORY_FOR_VELOCITY,
    MIN_HITS_TO_CONFIRM,
)
from core.types import (
    Acceleration,
    BBoxHistory,
    BoundingBox,
    Point,
    Velocity,
)
from core.validators import validate_bbox, validate_centroid
from models.enums import TrackStatus

if TYPE_CHECKING:
    from models.kalman import EnhancedKalmanFilter


class TrackState:
    """Complete track state with memory optimization.

    This class represents the state of a tracked object, including its
    position, history, motion metrics, and tracking status.

    Optimization features:
        - __slots__ for reduced memory usage (~40% less than without slots)
        - Efficient parameter validation
        - Native types for fast operations
        - Deque history for efficient queue operations

    Attributes:
        track_id: Unique track identifier.
        bbox: Current bounding box (x1, y1, x2, y2).
        centroid: Current centroid (x, y).
        features: Visual features for re-identification (optional).
        confidence: Track confidence (0-1).
        class_id: Object class ID.
        label: Class name.
        status: Current track status (TrackStatus).
        age: Track age in frames.
        hits: Number of associated detections.
        no_losses: Consecutive frames without loss.
        history: Position history.
        velocity: Current velocity (vx, vy) in pixels/frame.
        acceleration: Current acceleration (ax, ay) in pixels/frame².
        predicted_centroid: Kalman-predicted centroid.
        kalman_filter: Kalman filter for prediction (optional).
        metadata: Dictionary for additional metadata.
        bbox_history: Bounding box history.

    Example:
        >>> track = TrackState(
        ...     track_id=1, bbox=(10, 20, 50, 60), centroid=(30, 40), confidence=0.85
        ... )
        >>> track.update(new_detection)
        >>> track.predict_position()
        (32, 42)
        >>> print(track.status)
        TrackStatus.TENTATIVE
    """

    __slots__ = (
        "track_id",
        "class_id",
        "age",
        "hits",
        "no_losses",
        "bbox",
        "centroid",
        "predicted_centroid",
        "velocity",
        "acceleration",
        "status",
        "confidence",
        "label",
        "history",
        "bbox_history",
        "features",
        "metadata",
        "kalman_filter",
        "_history_deque",
    )

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        centroid: Point,
        *,
        features: np.ndarray | None = None,
        confidence: float = 0.5,
        class_id: int = -1,
        label: str = "unknown",
    ) -> None:
        """Initializes a new track.

        Args:
            track_id: Unique track identifier (must be >= 0).
            bbox: Bounding box (x1, y1, x2, y2) with valid dimensions.
            centroid: Object centroid (x, y).
            features: Visual features for re-identification (optional).
            confidence: Detection confidence (0-1).
            class_id: Object class ID (>= -1).
            label: Class name (non-empty string).

        Raises:
            ValueError: If any parameter is invalid.

        Example:
            >>> track = TrackState(
            ...     track_id=42,
            ...     bbox=(100, 100, 200, 200),
            ...     centroid=(150, 150),
            ...     confidence=0.9,
            ...     class_id=2,
            ...     label="car",
            ... )
        """
        self._validate_track_id(track_id)
        self._validate_bbox(bbox)
        self._validate_centroid(centroid)
        self._validate_confidence(confidence)

        self.track_id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.features = features
        self.confidence = confidence
        self.class_id = class_id
        self.label = label

        self.status = TrackStatus.TENTATIVE
        self.age = 0
        self.hits = 1
        self.no_losses = 0

        self.history = deque(maxlen=MAX_TRACK_HISTORY)
        self.history.append(centroid)

        self.bbox_history: BBoxHistory = []
        self.bbox_history.append(bbox)

        self.velocity: Velocity = (0.0, 0.0)
        self.acceleration: Acceleration = (0.0, 0.0)
        self.predicted_centroid: Point = centroid

        self.kalman_filter: EnhancedKalmanFilter | None = None

        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _validate_track_id(track_id: int) -> None:
        """Validates the track ID.

        Args:
            track_id: ID to validate.

        Raises:
            ValueError: If the ID is invalid (negative or not integer).
        """
        if not isinstance(track_id, int) or track_id < 0:
            raise ValueError(f"Invalid track_id: {track_id}")

    @staticmethod
    def _validate_bbox(bbox: Any) -> None:
        """Validates a bounding box using the central validator."""
        if not validate_bbox(bbox):
            raise ValueError(f"Invalid bbox: {bbox}")

    @staticmethod
    def _validate_centroid(centroid: Any) -> None:
        """Validates a centroid using the central validator."""
        if not validate_centroid(centroid):
            raise ValueError(f"Invalid centroid: {centroid}")

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Validates track confidence."""
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise ValueError(f"Invalid confidence: {confidence}")

    def update(self, detection: dict[str, Any], features: np.ndarray | None = None) -> None:
        """Updates the track with a new detection.

        This is the main update point for the track, processing a new
        detection and updating all metrics.

        Args:
            detection: Detection dictionary with 'box', 'centroid', 'confidence', etc.
            features: New visual features (optional).

        Note:
            Update includes:
            1. Position (bbox and centroid)
            2. Confidence
            3. Class and label
            4. History
            5. Motion metrics
            6. Track status
            7. Kalman filter

        Example:
            >>> track.update(
            ...     {
            ...         "box": (95, 95, 205, 205),
            ...         "centroid": (150, 150),
            ...         "confidence": 0.92,
            ...         "class_id": 2,
            ...         "label": "car",
            ...     }
            ... )
        """
        if not isinstance(detection, dict):
            return

        new_bbox = detection.get("box")
        if new_bbox is not None and validate_bbox(new_bbox):
            self.bbox = new_bbox
            self.bbox_history.append(new_bbox)

            if len(self.bbox_history) > MAX_BBOX_HISTORY_STORAGE:
                self.bbox_history = self.bbox_history[-MAX_BBOX_HISTORY_STORAGE:]

        new_centroid = detection.get("centroid")
        if new_centroid is not None and validate_centroid(new_centroid):
            self.centroid = new_centroid
            self.history.append(new_centroid)

        new_confidence = detection.get("confidence")
        if isinstance(new_confidence, (int, float)):
            self.confidence = max(0.0, min(1.0, new_confidence))

        new_class_id = detection.get("class_id")
        if isinstance(new_class_id, int) and new_class_id >= 0:
            self.class_id = new_class_id

        new_label = detection.get("label")
        if isinstance(new_label, str) and new_label:
            self.label = new_label

        if features is not None:
            self.features = features

        self.hits += 1
        self.no_losses = 0
        self.age += 1

        self._update_motion()

        self._update_status()

        if self.kalman_filter:
            self._update_kalman()

    def predict_position(self) -> Point:
        """Predicts the next position using the Kalman filter.

        If a Kalman filter is available, uses its prediction.
        Otherwise, returns the current position.

        Returns:
            Point: Predicted position (x, y).

        Example:
            >>> track.predict_position()
            (152, 148)  # Prediction using Kalman
        """
        if self.kalman_filter:
            try:
                pred = self.kalman_filter.predict()

                self.predicted_centroid = (
                    max(0, int(pred[0])),
                    max(0, int(pred[1])),
                )
                return self.predicted_centroid
            except Exception:
                self.kalman_filter = None

        self.predicted_centroid = self.centroid
        return self.centroid

    def mark_lost(self) -> None:
        """Marks the track as lost.

        Increments the loss counter and updates the status.
        Called when a track has no associated detection.
        """
        self.no_losses += 1
        self.age += 1
        self._update_status()

    def reset(self) -> None:
        """Resets the track to its initial state.

        Useful when recovering a lost track.
        """
        self.status = TrackStatus.TENTATIVE
        self.hits = 0
        self.no_losses = 0
        self.age = 0
        self.kalman_filter = None
        self.history.clear()
        self.bbox_history.clear()
        self.metadata.clear()

    def _update_motion(self) -> None:
        """Updates motion estimates (velocity and acceleration)."""
        history_len = len(self.history)

        if history_len >= MIN_HISTORY_FOR_VELOCITY:
            prev = self.history[-2]
            curr = self.history[-1]
            self.velocity = (
                float(curr[0] - prev[0]),
                float(curr[1] - prev[1]),
            )

            if history_len >= MIN_HISTORY_FOR_ACCELERATION:
                p1 = self.history[-3]
                p2 = self.history[-2]
                prev_vel = (
                    float(p2[0] - p1[0]),
                    float(p2[1] - p1[1]),
                )
                self.acceleration = (
                    self.velocity[0] - prev_vel[0],
                    self.velocity[1] - prev_vel[1],
                )

    def _update_status(self) -> None:
        """Updates track status based on hits and losses.

        Possible transitions:
        - TENTATIVE -> CONFIRMED: hits >= MIN_HITS_TO_CONFIRM
        - CONFIRMED -> LOST: no_losses > MAX_FRAMES_MISSED // 2
        - CONFIRMED -> DEAD: no_losses > MAX_FRAMES_MISSED
        - LOST -> CONFIRMED: hits >= MIN_HITS_TO_CONFIRM and no_losses == 0
        - LOST -> DEAD: no_losses > MAX_FRAMES_MISSED
        """
        if self.status == TrackStatus.DEAD:
            return

        if self.status == TrackStatus.TENTATIVE:
            if self.hits >= MIN_HITS_TO_CONFIRM:
                self.status = TrackStatus.CONFIRMED

        elif self.status == TrackStatus.CONFIRMED:
            if self.no_losses > MAX_FRAMES_MISSED:
                self.status = TrackStatus.DEAD
            elif self.no_losses > MAX_FRAMES_MISSED // 2:
                self.status = TrackStatus.LOST

        elif self.status == TrackStatus.LOST:
            if self.no_losses > MAX_FRAMES_MISSED:
                self.status = TrackStatus.DEAD
            elif self.hits >= MIN_HITS_TO_CONFIRM and self.no_losses == 0:
                self.status = TrackStatus.CONFIRMED

    def _update_kalman(self) -> None:
        """Updates the Kalman filter with the current measurement."""
        if self.kalman_filter:
            try:
                measurement = np.array([self.centroid[0], self.centroid[1]], dtype=np.float32)
                self.kalman_filter.correct(measurement)
            except Exception:
                self.kalman_filter = None

    def is_active(self) -> bool:
        """Checks if the track is active (not dead).

        Returns:
            bool: True if track is in TENTATIVE, CONFIRMED, or LOST state.

        Example:
            >>> if track.is_active():
            ...     print(f"Track {track.track_id} is active")
        """
        return self.status in (TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.LOST)

    def is_confirmed(self) -> bool:
        """Checks if the track is confirmed.

        Returns:
            bool: True if track is in CONFIRMED state.

        Example:
            >>> if track.is_confirmed():
            ...     print(f"Track {track.track_id} confirmed")
        """
        return self.status == TrackStatus.CONFIRMED

    def is_lost(self) -> bool:
        """Checks if the track is lost.

        Returns:
            bool: True if track is in LOST state.

        Example:
            >>> if track.is_lost():
            ...     print(f"Track {track.track_id} lost")
        """
        return self.status == TrackStatus.LOST

    def is_dead(self) -> bool:
        """Checks if the track is dead.

        Returns:
            bool: True if track is in DEAD state.

        Example:
            >>> if track.is_dead():
            ...     print(f"Track {track.track_id} dead")
        """
        return self.status == TrackStatus.DEAD

    def get_speed(self) -> float:
        """Gets the current speed of the track.

        Returns:
            float: Speed magnitude in pixels/frame.

        Example:
            >>> speed = track.get_speed()
            >>> if speed > 10:
            ...     print("Fast movement")
        """
        return float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))

    def get_movement_direction(self) -> float:
        """Gets the movement direction in radians.

        Returns:
            float: Movement angle (0 = right, π/2 = down).

        Example:
            >>> import math
            >>> angle = track.get_movement_direction()
            >>> degrees = math.degrees(angle)
            >>> print(f"Direction: {degrees:.1f}°")
        """
        return float(np.arctan2(self.velocity[1], self.velocity[0]))

    def to_dict(self) -> dict[str, Any]:
        """Converts the track to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary with all track data.

        Example:
            >>> track_data = track.to_dict()
            >>> import json
            >>> json.dump(track_data, file)
        """
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "status": self.status.value,
            "age": self.age,
            "hits": self.hits,
            "no_losses": self.no_losses,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "class_id": self.class_id,
            "label": self.label,
            "history": list(self.history),
            "bbox_history": self.bbox_history[-MAX_BBOX_HISTORY_DISPLAY:],
            "metadata": self.metadata,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """Converts the track to a compact dictionary (essential data only).

        Useful for real-time data transmission.

        Returns:
            dict[str, Any]: Dictionary with essential track data.

        Example:
            >>> compact = track.to_compact_dict()
            >>> # Send over WebSocket or save to database
        """
        return {
            "id": self.track_id,
            "position": self.centroid,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class": self.label,
            "status": self.status.value,
            "speed": self.get_speed(),
        }

    def __repr__(self) -> str:
        """Human-readable representation of the track."""
        return (
            f"TrackState(id={self.track_id}, status={self.status.value}, "
            f"pos={self.centroid}, conf={self.confidence:.2f})"
        )

    def __str__(self) -> str:
        """User-friendly representation for logging."""
        return f"Track {self.track_id} [{self.status.value}] at {self.centroid}"

    def __len__(self) -> int:
        """Returns the length of the history."""
        return len(self.history)

    def __contains__(self, pos: Point) -> bool:
        """Checks if a position is in the history."""
        return pos in self.history


def create_track_from_detection(
    track_id: int,
    detection: dict[str, Any],
    features: np.ndarray | None = None,
) -> TrackState:
    """Creates a TrackState from a detection.

    This helper function provides consistent track creation
    across the entire system.

    Args:
        track_id: Track ID.
        detection: Detection dictionary.
        features: Visual features (optional).

    Returns:
        TrackState: Created track.

    Example:
        >>> track = create_track_from_detection(1, detection)
    """
    return TrackState(
        track_id=track_id,
        bbox=detection.get("box", (0, 0, 0, 0)),
        centroid=detection.get("centroid", (0, 0)),
        features=features,
        confidence=detection.get("confidence", 0.5),
        class_id=detection.get("class_id", -1),
        label=detection.get("label", "unknown"),
    )


def merge_tracks(track1: TrackState, track2: TrackState) -> TrackState:
    """Merges two tracks into one.

    Useful when two tracks represent the same object.

    Args:
        track1: First track (primary).
        track2: Second track (secondary).

    Returns:
        TrackState: Merged track.

    Example:
        >>> merged = merge_tracks(track_a, track_b)
    """
    if track2.confidence > track1.confidence:
        track1, track2 = track2, track1

    for pos in track2.history:
        if pos not in track1.history:
            track1.history.append(pos)

    track1.hits += track2.hits
    track1.no_losses = min(track1.no_losses, track2.no_losses)
    track1.confidence = max(track1.confidence, track2.confidence)

    track1._update_motion()

    return track1


__all__ = [
    "TrackState",
    "Point",
    "BoundingBox",
    "Velocity",
    "Acceleration",
    "create_track_from_detection",
    "merge_tracks",
]
