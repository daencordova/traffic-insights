"""
Vehicle counting module.

Provides components for counting vehicles crossing virtual lines.

This module implements a complete vehicle counting system with:
    - Virtual line-based counting
    - Direction detection (entering/exiting)
    - Duplicate crossing prevention
    - Statistics collection and aggregation
    - Class-based counting (car, truck, bus, etc.)

Components:
    - VehicleCounter: Main orchestrator for the counting system
      └── Coordinates LineManager, CrossingDetector, and StatisticsCollector

    - LineManager: Management of counting lines
      └── Validation, access, and state management of lines

    - CountingLine: Virtual line definition
      └── Line geometry, direction, and configuration

    - CrossingDetector: Detection of line crossings
      └── Direction detection, duplicate prevention, crossing events

    - StatisticsCollector: Statistics collection and aggregation
      └── Per-line counts, class-based counts, speeds, events

    - VehicleEvent: Event representation for vehicle crossings
      └── Vehicle ID, line, direction, timestamp, metadata

Example:
    >>> from core.counter import VehicleCounter
    >>>
    >>> # Initialize counter
    >>> counter = VehicleCounter()
    >>>
    >>> # Process tracks from the tracker
    >>> stats = counter.process(tracks, frame)
    >>>
    >>> # Access statistics
    >>> print(f"Total vehicles: {stats['total']}")
    >>> print(f"Cars: {stats['by_class'][2]}")
    >>> print(f"Trucks: {stats['by_class'][5]}")
    >>>
    >>> # Get detailed statistics
    >>> detailed_stats = counter.get_stats()
    >>> for line_name, count in detailed_stats["by_line"].items():
    ...     print(f"Line {line_name}: {count} vehicles")

Example with custom lines:
    >>> from core.counter import LineManager, CountingLine
    >>>
    >>> # Define counting lines
    >>> lines = [
    ...     CountingLine(name="Entrance", start=(100, 200), end=(500, 200), direction="vertical"),
    ...     CountingLine(name="Exit", start=(100, 400), end=(500, 400), direction="vertical"),
    ... ]
    >>>
    >>> manager = LineManager(lines)
    >>> counter.set_line_manager(manager)
    >>>
    >>> # Process counting
    >>> stats = counter.process(tracks, frame)
"""

from core.counter.base import VehicleCounter
from core.counter.crossing_detector import CrossingDetector
from core.counter.line_manager import CountingLine, LineManager
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent

__all__ = [
    "VehicleCounter",
    "LineManager",
    "CountingLine",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
]
