"""Types for the tracking system.

This module defines type definitions and data structures used
for object tracking throughout the system.

Example:
    >>> from core.types import TrackState, TrackStatus
    >>>
    >>> # Create a track state
    >>> track = TrackState(
    ...     track_id=42,
    ...     centroid=(100, 200),
    ...     bbox=(50, 50, 150, 150),
    ...     confidence=0.95,
    ...     label="car",
    ...     class_id=2,
    ... )
    >>>
    >>> print(f"Track {track.track_id} at {track.centroid}")
    >>> print(f"Status: {track.status}")
    >>> print(f"Active: {track.is_active()}")
    >>> print(f"Speed: {track.get_speed():.2f} px/frame")
"""

from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict

from core.types.geometry import Acceleration, BoundingBox, Point, Velocity


class TrackStatus:
    """Possible states of a track.

    This class defines the lifecycle states of a track and provides
    utility methods for checking track status.

    Attributes:
        TENTATIVE: Initial state, track is being confirmed.
        CONFIRMED: Track is confirmed and reliable.
        LOST: Track was lost but may be recovered.
        DEAD: Track is dead and should be removed.

    Example:
        >>> status = TrackStatus.CONFIRMED
        >>> if TrackStatus.is_active(status):
        ...     print("Track is active")
        >>>
        >>> display_name = TrackStatus.get_display_name(status)
        >>> print(f"Status name: {display_name}")
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DEAD = "dead"

    ACTIVE = (TENTATIVE, CONFIRMED, LOST)
    TERMINAL = (DEAD,)

    @classmethod
    def is_active(cls, status: str) -> bool:
        """Checks if the status is active.

        Args:
            status: Status string to check.

        Returns:
            bool: True if status is active (tentative, confirmed, or lost).

        Example:
            >>> TrackStatus.is_active("confirmed")
            True
            >>> TrackStatus.is_active("dead")
            False
        """
        return status in cls.ACTIVE

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Checks if the status is terminal.

        Args:
            status: Status string to check.

        Returns:
            bool: True if status is terminal (dead).

        Example:
            >>> TrackStatus.is_terminal("dead")
            True
            >>> TrackStatus.is_terminal("confirmed")
            False
        """
        return status in cls.TERMINAL

    @classmethod
    def get_display_name(cls, status: str) -> str:
        """Gets a human-readable name for the status.

        Args:
            status: Status string.

        Returns:
            str: Display name for the status.

        Example:
            >>> TrackStatus.get_display_name("confirmed")
            "Confirmed"
            >>> TrackStatus.get_display_name("lost")
            "Lost"
        """
        names = {
            cls.TENTATIVE: "Tentative",
            cls.CONFIRMED: "Confirmed",
            cls.LOST: "Lost",
            cls.DEAD: "Dead",
        }
        return names.get(status, status)


class TrackDataDict(TypedDict, total=False):
    """Data structure for an active track.

    This TypedDict defines the complete structure of track data
    used throughout the system.

    Attributes:
        centroid: Current centroid (x, y).
        bbox: Current bounding box (x1, y1, x2, y2).
        status: Track status (tentative, confirmed, lost, dead).
        age: Age in frames.
        hits: Number of associated detections.
        no_losses: Consecutive frames without loss.
        confidence: Current confidence (0-1).
        velocity: Velocity vector (vx, vy).
        acceleration: Acceleration vector (ax, ay).
        label: Class name.
        class_id: Class ID.
        history: Position history.
        predicted_centroid: Kalman-predicted position.
        features: Visual features for re-identification.
        metadata: Additional metadata.

    Example:
        >>> track: TrackDataDict = {
        ...     "centroid": (100, 200),
        ...     "bbox": (50, 50, 150, 150),
        ...     "status": "confirmed",
        ...     "age": 42,
        ...     "hits": 35,
        ...     "no_losses": 10,
        ...     "confidence": 0.95,
        ...     "velocity": (5.0, 3.0),
        ...     "acceleration": (0.0, 0.0),
        ...     "label": "car",
        ...     "class_id": 2,
        ...     "history": [(100, 200), (105, 205)],
        ...     "predicted_centroid": (102, 202),
        ...     "metadata": {"lane": 1},
        ... }
    """

    centroid: Point
    bbox: BoundingBox
    status: str
    age: int
    hits: int
    no_losses: int
    confidence: float
    velocity: Velocity
    acceleration: Acceleration
    label: str
    class_id: int
    history: list[Point]
    predicted_centroid: Point
    features: list[float] | None
    metadata: dict[str, any]


@dataclass(slots=True)
class TrackState:
    """Complete state of a track with memory optimization.

    This class represents the full state of a track, including all
    tracking-related data and history.

    Attributes:
        track_id: Unique track identifier.
        centroid: Current centroid (x, y).
        bbox: Current bounding box (x1, y1, x2, y2).
        status: Track status.
        confidence: Track confidence (0-1).
        age: Age in frames.
        hits: Number of associated detections.
        no_losses: Consecutive frames without loss.
        velocity: Velocity (vx, vy).
        acceleration: Acceleration (ax, ay).
        label: Class name.
        class_id: Class ID.
        predicted_centroid: Kalman-predicted centroid.
        history: Position history.
        bbox_history: Bounding box history.
        features: Visual features for re-identification.
        metadata: Additional metadata.

    Example:
        >>> track = TrackState(
        ...     track_id=1,
        ...     centroid=(100, 200),
        ...     bbox=(50, 50, 150, 150),
        ...     status=TrackStatus.CONFIRMED,
        ...     confidence=0.9,
        ...     age=30,
        ...     hits=25,
        ...     no_losses=5,
        ...     velocity=(5.0, 3.0),
        ...     acceleration=(0.5, 0.2),
        ...     label="car",
        ...     class_id=2,
        ...     predicted_centroid=(105, 205),
        ...     history=[(100, 200), (105, 205)],
        ...     bbox_history=[(50, 50, 150, 150), (55, 55, 155, 155)],
        ... )
    """

    track_id: int
    centroid: Point
    bbox: BoundingBox
    status: str = TrackStatus.TENTATIVE
    confidence: float = 0.5
    age: int = 0
    hits: int = 1
    no_losses: int = 0
    velocity: Velocity = (0.0, 0.0)
    acceleration: Acceleration = (0.0, 0.0)
    label: str = "unknown"
    class_id: int = -1
    predicted_centroid: Point = (0, 0)
    history: list[Point] = field(default_factory=list)
    bbox_history: list[BoundingBox] = field(default_factory=list)
    features: list[float] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initializes empty collections if None."""
        if self.history is None:
            self.history = []
        if self.bbox_history is None:
            self.bbox_history = []
        if self.metadata is None:
            self.metadata = {}

    def is_active(self) -> bool:
        """Checks if the track is active.

        Returns:
            bool: True if the track is in an active state.

        Example:
            >>> if track.is_active():
            ...     process_track(track)
        """
        return TrackStatus.is_active(self.status)

    def is_confirmed(self) -> bool:
        """Checks if the track is confirmed.

        Returns:
            bool: True if the track is confirmed.

        Example:
            >>> if track.is_confirmed():
            ...     print("Track is reliable")
        """
        return self.status == TrackStatus.CONFIRMED

    def is_lost(self) -> bool:
        """Checks if the track is lost.

        Returns:
            bool: True if the track is lost.

        Example:
            >>> if track.is_lost():
            ...     attempt_recovery(track)
        """
        return self.status == TrackStatus.LOST

    def is_dead(self) -> bool:
        """Checks if the track is dead.

        Returns:
            bool: True if the track is dead.

        Example:
            >>> if track.is_dead():
            ...     remove_track(track)
        """
        return self.status == TrackStatus.DEAD

    def get_speed(self) -> float:
        """Gets the current speed of the track.

        Returns:
            float: Speed in pixels per frame.

        Example:
            >>> speed = track.get_speed()
            >>> print(f"Track speed: {speed:.2f} px/frame")
        """
        import math

        return math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

    def get_direction(self) -> float:
        """Gets the direction of movement in radians.

        Returns:
            float: Direction angle in radians.

        Example:
            >>> import math
            >>> direction = track.get_direction()
            >>> degrees = math.degrees(direction)
            >>> print(f"Direction: {degrees:.1f} degrees")
        """
        import math

        return math.atan2(self.velocity[1], self.velocity[0])

    def get_age_seconds(self, fps: float = 30.0) -> float:
        """Gets the track age in seconds.

        Args:
            fps: Frames per second (default: 30.0).

        Returns:
            float: Age in seconds.

        Example:
            >>> age_seconds = track.get_age_seconds(fps=25.0)
            >>> print(f"Track is {age_seconds:.1f} seconds old")
        """
        return self.age / max(1.0, fps)

    def to_dict(self) -> TrackDataDict:
        """Converts the track to a dictionary.

        Returns:
            TrackDataDict: Dictionary representation of the track.

        Example:
            >>> data = track.to_dict()
            >>> print(f"Track data: {data['centroid']}")
        """
        return {
            "centroid": self.centroid,
            "bbox": self.bbox,
            "status": self.status,
            "age": self.age,
            "hits": self.hits,
            "no_losses": self.no_losses,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "label": self.label,
            "class_id": self.class_id,
            "history": self.history,
            "predicted_centroid": self.predicted_centroid,
            "metadata": self.metadata,
        }

    def to_compact_dict(self) -> dict[str, any]:
        """Converts to a compact dictionary (essential data only).

        Returns:
            dict[str, any]: Compact dictionary representation.

        Example:
            >>> compact = track.to_compact_dict()
            >>> print(f"Track {compact['id']}: {compact['position']}")
        """
        return {
            "id": self.track_id,
            "position": self.centroid,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class": self.label,
            "status": self.status,
            "speed": self.get_speed(),
        }


TrackStateDict: TypeAlias = dict[int, TrackState]
"""Dictionary of track states: track_id -> TrackState.

Example:
    >>> tracks: TrackStateDict = {1: track1, 2: track2, 3: track3}
"""

TrackInfoDict: TypeAlias = dict[int, TrackDataDict]
"""Dictionary of track information: track_id -> TrackDataDict.

Example:
    >>> track_info: TrackInfoDict = {1: track_data1, 2: track_data2}
"""

TracksDict: TypeAlias = TrackInfoDict
"""Alias for TrackInfoDict for backward compatibility.

Example:
    >>> tracks: TracksDict = {1: track_data1, 2: track_data2}
"""


__all__ = [
    "TrackStatus",
    "TrackDataDict",
    "TrackState",
    "TrackStateDict",
    "TrackInfoDict",
    "TracksDict",
]
