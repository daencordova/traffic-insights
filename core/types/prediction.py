"""Types for trajectory prediction.

This module defines type definitions and data structures used
for predicting object trajectories in the tracking system.

Example:
    >>> from core.types import TrajectoryState, TrajectoryPrediction
    >>>
    >>> # Create a trajectory prediction
    >>> prediction = TrajectoryPrediction(
    ...     track_id=42,
    ...     positions=[(100, 200), (105, 205), (110, 210)],
    ...     confidences=[0.95, 0.90, 0.85],
    ...     timestamps=[1.0, 1.1, 1.2],
    ...     horizon_seconds=2.0,
    ...     state=TrajectoryState.MOVING,
    ...     motion_model="constant_velocity",
    ...     predicted_velocity=(5.0, 5.0),
    ...     predicted_acceleration=(0.0, 0.0),
    ...     uncertainty=0.15,
    ...     collision_risk=0.05,
    ...     trajectory_type="linear",
    ... )
    >>>
    >>> print(f"Last position: {prediction.get_last_position()}")
    >>> print(f"Average confidence: {prediction.get_average_confidence():.2f}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from core.types.geometry import FloatPoint, Velocity


class TrajectoryState(Enum):
    """Possible states of an object's trajectory.

    Attributes:
        MOVING: Object is in motion.
        STOPPED: Object is stationary.
        ACCELERATING: Object is accelerating.
        DECELERATING: Object is decelerating.
        TURNING: Object is turning/changing direction.
        ERRATIC: Object is moving erratically.
        UNKNOWN: State is unknown.

    Example:
        >>> state = TrajectoryState.MOVING
        >>> color = TrajectoryState.get_color(state)
        >>> print(f"State: {state.value}, Color: {color}")
    """

    MOVING = "moving"
    STOPPED = "stopped"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    TURNING = "turning"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"

    @classmethod
    def get_color(cls, state: "TrajectoryState") -> tuple[int, int, int]:
        """Gets the associated color for the state.

        Args:
            state: TrajectoryState to get color for.

        Returns:
            tuple[int, int, int]: Color in BGR format.

        Example:
            >>> color = TrajectoryState.get_color(TrajectoryState.MOVING)
            >>> print(f"Color: {color}")  # (255, 255, 0) - Yellow
        """
        colors = {
            cls.MOVING: (255, 255, 0),
            cls.STOPPED: (0, 0, 255),
            cls.ACCELERATING: (0, 255, 255),
            cls.DECELERATING: (0, 165, 255),
            cls.TURNING: (255, 0, 255),
            cls.ERRATIC: (255, 0, 0),
            cls.UNKNOWN: (255, 255, 0),
        }
        return colors.get(state, (255, 255, 0))


@dataclass(slots=True)
class TrajectoryPrediction:
    """Represents a trajectory prediction.

    This class encapsulates all data related to predicting an object's
    future trajectory, including positions, confidences, and metadata.

    Attributes:
        track_id: Track ID.
        positions: List of predicted positions (x, y).
        confidences: Confidence for each prediction (0-1).
        timestamps: Timestamps for each prediction.
        horizon_seconds: Prediction horizon in seconds.
        state: Trajectory state.
        motion_model: Motion model used for prediction.
        predicted_velocity: Predicted velocity (vx, vy).
        predicted_acceleration: Predicted acceleration (ax, ay).
        uncertainty: Prediction uncertainty (0-1).
        collision_risk: Collision risk (0-1).
        trajectory_type: Type of trajectory.
        metadata: Additional metadata.

    Example:
        >>> prediction = TrajectoryPrediction(
        ...     track_id=1,
        ...     positions=[(100, 200), (120, 220), (140, 240)],
        ...     confidences=[0.9, 0.85, 0.8],
        ...     timestamps=[1.0, 1.1, 1.2],
        ...     horizon_seconds=1.5,
        ...     state=TrajectoryState.ACCELERATING,
        ...     motion_model="constant_acceleration",
        ...     predicted_velocity=(20.0, 20.0),
        ...     predicted_acceleration=(5.0, 5.0),
        ...     uncertainty=0.2,
        ...     collision_risk=0.1,
        ...     trajectory_type="curved",
        ... )
    """

    track_id: int
    positions: list[FloatPoint]
    confidences: list[float]
    timestamps: list[float]
    horizon_seconds: float
    state: TrajectoryState
    motion_model: str
    predicted_velocity: Velocity
    predicted_acceleration: Velocity
    uncertainty: float
    collision_risk: float
    trajectory_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_last_position(self) -> FloatPoint | None:
        """Gets the last predicted position.

        Returns:
            FloatPoint | None: Last predicted position or None if no positions.

        Example:
            >>> last = prediction.get_last_position()
            >>> if last:
            ...     print(f"Last position: {last}")
        """
        return self.positions[-1] if self.positions else None

    def get_average_confidence(self) -> float:
        """Gets the average confidence across all predictions.

        Returns:
            float: Average confidence (0-1).

        Example:
            >>> avg_conf = prediction.get_average_confidence()
            >>> print(f"Average confidence: {avg_conf:.2f}")
        """
        return sum(self.confidences) / max(1, len(self.confidences))


PredictionData: TypeAlias = dict[str, Any]
"""Trajectory prediction data in dictionary format.

This type alias represents prediction data stored as a dictionary
for flexible serialization and deserialization.

Example:
    >>> data: PredictionData = {
    ...     "track_id": 42,
    ...     "positions": [(100, 200), (120, 220)],
    ...     "state": "moving",
    ...     "confidence": 0.85
    ... }
"""


__all__ = [
    "TrajectoryState",
    "TrajectoryPrediction",
    "PredictionData",
]
