"""Types for system statistics.

This module defines type definitions and data structures used
for statistics, counting, and events throughout the system.

Example:
    >>> from core.types import StatsDict, CountingLineDict
    >>>
    >>> # Create statistics dictionary
    >>> stats: StatsDict = {
    ...     "total": 42,
    ...     "line_counts": {"entrance": 25, "exit": 17},
    ...     "class_counts": {"car": 30, "truck": 12},
    ...     "avg_speed": 45.5,
    ...     "max_speed": 80.0,
    ...     "min_speed": 10.2,
    ...     "avg_per_minute": 14.0,
    ...     "count_rate": 0.7,
    ...     "total_events": 42,
    ...     "runtime_seconds": 60.0,
    ...     "active_objects": 15,
    ...     "frame_counter": 1800,
    ...     "timestamp": 1645123456.0,
    ...     "line_count": 2,
    ...     "class_count": 3,
    ... }
    >>>
    >>> # Create counting line configuration
    >>> line: CountingLineDict = {
    ...     "id": "entrance",
    ...     "name": "Main Entrance",
    ...     "points": [(100, 300), (500, 300)],
    ...     "color": (0, 255, 0),
    ...     "direction": "down",
    ...     "y_position": 300,
    ...     "enabled": True,
    ...     "metadata": {"description": "Main entrance counting line"},
    ... }
"""

from typing import TypeAlias, TypedDict


class StatsDict(TypedDict, total=False):
    """System statistics dictionary.

    This TypedDict defines the complete structure of system statistics,
    including counting, speed, and performance metrics.

    Attributes:
        total: Total vehicles counted.
        line_counts: Counts per line.
        class_counts: Counts per class.
        avg_speed: Average speed in pixels per frame.
        max_speed: Maximum speed.
        min_speed: Minimum speed.
        avg_per_minute: Average count per minute.
        count_rate: Counting rate (vehicles per second).
        total_events: Total events recorded.
        runtime_seconds: System runtime in seconds.
        active_objects: Currently active objects.
        frame_counter: Current frame number.
        timestamp: Current timestamp.
        line_count: Number of active lines.
        class_count: Number of detected classes.

    Example:
        >>> stats: StatsDict = {
        ...     "total": 100,
        ...     "line_counts": {"line_1": 60, "line_2": 40},
        ...     "class_counts": {"car": 70, "truck": 30},
        ...     "avg_speed": 52.3,
        ...     "max_speed": 85.0,
        ...     "min_speed": 5.2,
        ...     "avg_per_minute": 10.0,
        ...     "count_rate": 1.67,
        ...     "total_events": 100,
        ...     "runtime_seconds": 60.0,
        ...     "active_objects": 8,
        ...     "frame_counter": 1800,
        ...     "timestamp": 1645123456.0,
        ...     "line_count": 2,
        ...     "class_count": 2,
        ... }
    """

    total: int
    line_counts: dict[str, int]
    class_counts: dict[str, int]
    avg_speed: float
    max_speed: float
    min_speed: float
    avg_per_minute: float
    count_rate: float
    total_events: int
    runtime_seconds: float
    active_objects: int
    frame_counter: int
    timestamp: float
    line_count: int
    class_count: int


class CountingLineDict(TypedDict, total=False):
    """Configuration of a counting line.

    This TypedDict defines the structure for configuring a virtual
    counting line in the system.

    Attributes:
        id: Unique identifier.
        name: Descriptive name.
        points: List of points defining the line.
        color: Color in BGR format.
        direction: Counting direction ('up' or 'down').
        y_position: Y position of the line.
        enabled: Whether the line is active.
        metadata: Additional metadata.

    Example:
        >>> line: CountingLineDict = {
        ...     "id": "exit",
        ...     "name": "Main Exit",
        ...     "points": [(100, 500), (500, 500)],
        ...     "color": (0, 0, 255),
        ...     "direction": "up",
        ...     "y_position": 500,
        ...     "enabled": True,
        ...     "metadata": {"lane": 2},
        ... }
    """

    id: str
    name: str
    points: list[tuple[int, int]]
    color: tuple[int, int, int]
    direction: str
    y_position: int
    enabled: bool
    metadata: dict[str, any]


LineList: TypeAlias = list[CountingLineDict]
"""List of counting lines.

Example:
    >>> lines: LineList = [
    ...     {"id": "line_1", "name": "Entrance", "points": [(100, 300), (500, 300)]},
    ...     {"id": "line_2", "name": "Exit", "points": [(100, 500), (500, 500)]}
    ... ]
"""

EventsList: TypeAlias = list[dict[str, any]]
"""List of counting events.

Example:
    >>> events: EventsList = [
    ...     {"object_id": 1, "line": "entrance", "timestamp": "14:30:25"},
    ...     {"object_id": 2, "line": "exit", "timestamp": "14:30:27"}
    ... ]
"""


__all__ = [
    "StatsDict",
    "CountingLineDict",
    "LineList",
    "EventsList",
]
