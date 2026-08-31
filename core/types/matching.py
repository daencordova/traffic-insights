"""Types for the matching system.

This module defines type definitions and data structures used
for matching detections to tracks in the tracking system.

Example:
    >>> from core.types import MatchLevel, MatchResult
    >>>
    >>> # Create a match result
    >>> result = MatchResult(
    ...     matches=[(0, 1), (2, 3)],
    ...     unmatched_detections=[1],
    ...     unmatched_tracks=[0],
    ...     level_used=MatchLevel.IOU,
    ... )
    >>>
    >>> # Access properties
    >>> print(f"Match rate: {result.match_rate:.2f}")
    >>> print(f"Matches: {result.match_count}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class MatchLevel(Enum):
    """Matching levels in priority order.

    Attributes:
        IOU: Matching by IoU (most accurate).
        FEATURE: Matching by visual features.
        MOTION: Matching by motion prediction.
        SHAPE: Matching by shape (aspect ratio).
        SPATIAL: Matching by spatial distance (fallback).

    Example:
        >>> level = MatchLevel.IOU
        >>> priority = MatchLevel.get_priority(level)
        >>> print(f"Priority: {priority}")  # 0 (highest)
        >>>
        >>> level = MatchLevel.SPATIAL
        >>> priority = MatchLevel.get_priority(level)  # 4 (lowest)
    """

    IOU = "iou"
    FEATURE = "feature"
    MOTION = "motion"
    SHAPE = "shape"
    SPATIAL = "spatial"

    @classmethod
    def get_priority(cls, level: "MatchLevel") -> int:
        """Gets the priority of the level (lower = higher priority).

        Args:
            level: MatchLevel to get priority for.

        Returns:
            int: Priority index (0 = highest, 4 = lowest).

        Example:
            >>> priority = MatchLevel.get_priority(MatchLevel.FEATURE)
            >>> print(priority)  # 1
        """
        order = [cls.IOU, cls.FEATURE, cls.MOTION, cls.SHAPE, cls.SPATIAL]
        return order.index(level)


@dataclass(slots=True)
class MatchResult:
    """Result of a matching operation.

    This class encapsulates the complete result of matching detections
    to tracks, including all matches, unmatched items, and metadata.

    Attributes:
        matches: List of (detection_idx, track_idx) matched pairs.
        unmatched_detections: Indices of unmatched detections.
        unmatched_tracks: Indices of unmatched tracks.
        match_scores: Dictionary of scores for each match pair.
        level_used: Matching level that was used.
        time_ms: Execution time in milliseconds.
        metadata: Additional matching metadata.

    Example:
        >>> result = MatchResult(
        ...     matches=[(0, 1), (2, 3), (4, 5)],
        ...     unmatched_detections=[3],
        ...     unmatched_tracks=[2],
        ...     match_scores={(0, 1): 0.95, (2, 3): 0.87, (4, 5): 0.76},
        ...     level_used=MatchLevel.FEATURE,
        ...     time_ms=1.5,
        ...     metadata={"method": "hungarian"},
        ... )
        >>>
        >>> print(f"Match count: {result.match_count}")
        >>> print(f"Match rate: {result.match_rate:.2%}")
        >>> print(f"Level used: {result.level_used.value}")
    """

    matches: list[tuple[int, int]]
    unmatched_detections: list[int]
    unmatched_tracks: list[int]
    match_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    level_used: MatchLevel = MatchLevel.IOU
    time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def match_count(self) -> int:
        """Number of matches found.

        Returns:
            int: Total number of matched pairs.

        Example:
            >>> count = result.match_count
            >>> print(f"Found {count} matches")
        """
        return len(self.matches)

    @property
    def total_detections(self) -> int:
        """Total number of detections.

        Returns:
            int: Sum of matched and unmatched detections.

        Example:
            >>> total = result.total_detections
            >>> print(f"Total detections: {total}")
        """
        return len(self.matches) + len(self.unmatched_detections)

    @property
    def total_tracks(self) -> int:
        """Total number of tracks.

        Returns:
            int: Sum of matched and unmatched tracks.

        Example:
            >>> total = result.total_tracks
            >>> print(f"Total tracks: {total}")
        """
        return len(self.matches) + len(self.unmatched_tracks)

    @property
    def match_rate(self) -> float:
        """Match rate (matches / detections).

        Returns:
            float: Ratio of matched detections to total detections (0-1).

        Example:
            >>> rate = result.match_rate
            >>> print(f"Match rate: {rate:.2%}")
        """
        return self.match_count / max(1, self.total_detections)

    def to_dict(self) -> dict[str, Any]:
        """Converts to dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation with all fields.

        Example:
            >>> data = result.to_dict()
            >>> print(data["match_count"])
            >>> print(data["level_used"])
        """
        return {
            "match_count": self.match_count,
            "unmatched_detections": len(self.unmatched_detections),
            "unmatched_tracks": len(self.unmatched_tracks),
            "match_rate": self.match_rate,
            "time_ms": self.time_ms,
            "level_used": self.level_used.value if self.level_used else None,
            "metadata": self.metadata,
        }


ScoreMatrix: TypeAlias = list[list[float]]
"""Score matrix for matching operations.

This type alias represents a 2D matrix of matching scores where
rows correspond to detections and columns correspond to tracks.

Example:
    >>> scores: ScoreMatrix = [
    ...     [0.95, 0.32, 0.12],  # Detection 0 scores
    ...     [0.45, 0.87, 0.23],  # Detection 1 scores
    ...     [0.11, 0.34, 0.76]   # Detection 2 scores
    ... ]
"""


__all__ = [
    "MatchLevel",
    "MatchResult",
    "ScoreMatrix",
]
