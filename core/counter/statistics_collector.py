"""Counting statistics collector.

Handles collection and calculation of vehicle statistics.
"""

from __future__ import annotations

from collections import defaultdict
import time
from typing import TYPE_CHECKING, Any

from core.constants import HISTORY_MAX_SIZE, SECONDS_PER_MINUTE, VELOCITY_DIMENSION

if TYPE_CHECKING:
    from core.types import Point, Velocity


class VehicleEvent:
    """Vehicle counting event.

    This class represents a single vehicle crossing event with
    all associated metadata.

    Attributes:
        timestamp: Event timestamp (HH:MM:SS format).
        object_id: Vehicle object ID.
        line_id: Counting line ID.
        line_name: Counting line name.
        label: Vehicle class label.
        class_id: Vehicle class ID.
        centroid: Vehicle centroid position (x, y).
        velocity: Vehicle velocity (vx, vy).
        confidence: Detection confidence.
        metadata: Additional event metadata.

    Example:
        >>> event = VehicleEvent(
        ...     timestamp="14:30:25",
        ...     object_id=42,
        ...     line_id="entrance",
        ...     line_name="Main Entrance",
        ...     label="car",
        ...     class_id=2,
        ...     centroid=(320, 250),
        ...     velocity=(5.2, -3.1),
        ...     confidence=0.92,
        ...     metadata={"lane": 1},
        ... )
    """

    __slots__ = (
        "timestamp",
        "object_id",
        "line_id",
        "line_name",
        "label",
        "class_id",
        "centroid",
        "velocity",
        "confidence",
        "metadata",
    )

    def __init__(
        self,
        timestamp: str,
        object_id: int,
        line_id: str,
        line_name: str,
        label: str,
        class_id: int,
        centroid: Point,
        velocity: Velocity,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        """Initializes a vehicle event.

        Args:
            timestamp: Event timestamp.
            object_id: Vehicle object ID.
            line_id: Counting line ID.
            line_name: Counting line name.
            label: Vehicle class label.
            class_id: Vehicle class ID.
            centroid: Vehicle centroid position.
            velocity: Vehicle velocity.
            confidence: Detection confidence.
            metadata: Additional event metadata.
        """
        self.timestamp = timestamp
        self.object_id = object_id
        self.line_id = line_id
        self.line_name = line_name
        self.label = label
        self.class_id = class_id
        self.centroid = centroid
        self.velocity = velocity
        self.confidence = confidence
        self.metadata = metadata or {}


class StatisticsCollector:
    """Collector of counting statistics.

    This class manages all counting statistics including:
        - Per-line counts
        - Class-based counts
        - Speed statistics (average, min, max, percentiles)
        - Event history
        - Per-minute counting rates
        - Count rates

    Responsibilities:
        - Maintain line counts
        - Collect class statistics
        - Calculate average speeds
        - Maintain event history
        - Provide statistical summaries

    Attributes:
        line_counts: Counts per line.
        class_counts: Counts per class.
        speed_history: Speed history per object.
        events: Event history.

    Example:
        >>> collector = StatisticsCollector(max_history=500)
        >>>
        >>> # Record a crossing
        >>> collector.record_crossing(
        ...     object_id=42,
        ...     line_id="entrance",
        ...     line_name="Main Entrance",
        ...     track_data={"label": "car", "class_id": 2, "confidence": 0.92},
        ...     centroid=(320, 250),
        ... )
        >>>
        >>> # Record speed
        >>> collector.record_speed(42, (5.2, -3.1))
        >>>
        >>> # Get statistics
        >>> stats = collector.get_stats()
        >>> print(f"Total vehicles: {stats['total']}")
        >>> print(f"Average speed: {stats['avg_speed']:.1f} px/frame")
        >>>
        >>> # Get recent events
        >>> events = collector.get_recent_events(10)
        >>> for event in events:
        ...     print(f"{event['label']} crossed {event['line']}")
    """

    def __init__(self, max_history: int = 1000):
        """Initializes the statistics collector.

        Args:
            max_history: Maximum number of events in history.

        Example:
            >>> collector = StatisticsCollector(max_history=100)
            >>> # Only keep last 100 events
        """
        self.line_counts: dict[str, int] = defaultdict(int)
        self.class_counts: dict[str, int] = defaultdict(int)
        self.speed_history: dict[int, list[float]] = defaultdict(list)
        self.events: list[VehicleEvent] = []
        self.max_history = max_history

        self._start_time = time.time()
        self._counts_per_minute: list[int] = []
        self._last_count_time = time.time()
        self._last_stats_time = time.time()
        self._stats_window = 60.0

    def record_crossing(
        self,
        object_id: int,
        line_id: str,
        line_name: str,
        track_data: dict[str, Any],
        centroid: tuple,
    ) -> None:
        """Records a line crossing.

        This method updates counts and creates an event for a vehicle
        crossing a counting line.

        Args:
            object_id: Object ID.
            line_id: Line ID.
            line_name: Line name.
            track_data: Track data containing labels and metadata.
            centroid: Object position.

        Example:
            >>> collector.record_crossing(
            ...     object_id=42,
            ...     line_id="entrance",
            ...     line_name="Main Entrance",
            ...     track_data={"label": "truck", "class_id": 5},
            ...     centroid=(320, 250),
            ... )
        """
        self.line_counts[line_id] += 1

        label = track_data.get("label", "vehicle")
        self.class_counts[label] += 1

        event = VehicleEvent(
            timestamp=time.strftime("%H:%M:%S"),
            object_id=object_id,
            line_id=line_id,
            line_name=line_name,
            label=label,
            class_id=track_data.get("class_id", -1),
            centroid=centroid,
            velocity=track_data.get("velocity", (0, 0)),
            confidence=track_data.get("confidence", 0.0),
            metadata=track_data.get("metadata", {}),
        )
        self.events.append(event)

        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history :]

    def record_speed(self, object_id: int, velocity: tuple) -> None:
        """Records the speed of an object.

        Args:
            object_id: Object ID.
            velocity: Velocity (vx, vy) tuple.

        Example:
            >>> collector.record_speed(42, (5.2, -3.1))
            >>> # Speed is automatically calculated from velocity
        """
        if not isinstance(velocity, (tuple, list)) or len(velocity) != VELOCITY_DIMENSION:
            return

        speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        self.speed_history[object_id].append(speed)

        if len(self.speed_history[object_id]) > HISTORY_MAX_SIZE:
            self.speed_history[object_id] = self.speed_history[object_id][-HISTORY_MAX_SIZE:]

    def update_minute_counts(self, total: int) -> None:
        """Updates per-minute counts.

        Args:
            total: Current total count.

        Example:
            >>> collector.update_minute_counts(collector.get_total_count())
        """
        current_time = time.time()
        if current_time - self._last_count_time >= SECONDS_PER_MINUTE:
            self._counts_per_minute.append(total)
            self._last_count_time = current_time

            if len(self._counts_per_minute) > SECONDS_PER_MINUTE:
                self._counts_per_minute = self._counts_per_minute[-SECONDS_PER_MINUTE:]

    def get_average_speed(self) -> float:
        """Calculates the average speed of all objects.

        Returns:
            float: Average speed in pixels per frame.

        Example:
            >>> avg_speed = collector.get_average_speed()
            >>> print(f"Average speed: {avg_speed:.1f} px/frame")
        """
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        try:
            return float(sum(all_speeds) / len(all_speeds))
        except Exception:
            return 0.0

    def get_max_speed(self) -> float:
        """Gets the maximum recorded speed.

        Returns:
            float: Maximum speed in pixels per frame.

        Example:
            >>> max_speed = collector.get_max_speed()
            >>> print(f"Max speed: {max_speed:.1f} px/frame")
        """
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        return float(max(all_speeds))

    def get_min_speed(self) -> float:
        """Gets the minimum recorded speed.

        Returns:
            float: Minimum speed in pixels per frame.

        Example:
            >>> min_speed = collector.get_min_speed()
            >>> print(f"Min speed: {min_speed:.1f} px/frame")
        """
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        return float(min(all_speeds))

    def get_speed_percentile(self, percentile: float = 50.0) -> float:
        """Gets a percentile of speeds.

        Args:
            percentile: Percentile to calculate (0-100).

        Returns:
            float: Speed at the specified percentile.

        Example:
            >>> median_speed = collector.get_speed_percentile(50.0)
            >>> p95_speed = collector.get_speed_percentile(95.0)
            >>> print(f"Median: {median_speed:.1f}, P95: {p95_speed:.1f}")
        """
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        sorted_speeds = sorted(all_speeds)
        index = int(len(sorted_speeds) * percentile / 100.0)
        return float(sorted_speeds[min(index, len(sorted_speeds) - 1)])

    def get_average_per_minute(self) -> float:
        """Gets the average count per minute.

        Returns:
            float: Average vehicles per minute.

        Example:
            >>> avg_minute = collector.get_average_per_minute()
            >>> print(f"Average: {avg_minute:.1f} vehicles/min")
        """
        if not self._counts_per_minute:
            return 0.0
        return float(sum(self._counts_per_minute) / len(self._counts_per_minute))

    def get_total_count(self) -> int:
        """Gets the total count across all lines.

        Returns:
            int: Total vehicles counted.

        Example:
            >>> total = collector.get_total_count()
            >>> print(f"Total vehicles: {total}")
        """
        return sum(self.line_counts.values())

    def get_count_rate(self) -> float:
        """Gets the counting rate per second.

        Returns:
            float: Vehicles per second.

        Example:
            >>> rate = collector.get_count_rate()
            >>> print(f"Rate: {rate:.2f} vehicles/sec")
        """
        runtime = time.time() - self._start_time
        if runtime <= 0:
            return 0.0
        return self.get_total_count() / runtime

    def get_stats(self) -> dict[str, Any]:
        """Gets all statistics.

        Returns:
            dict[str, Any]: Complete statistics including:
                - total: Total vehicles counted
                - line_counts: Counts per line
                - class_counts: Counts per class
                - avg_speed: Average speed
                - max_speed: Maximum speed
                - min_speed: Minimum speed
                - avg_per_minute: Average per minute
                - count_rate: Counting rate
                - total_events: Total events
                - runtime_seconds: Runtime in seconds
                - active_objects: Active objects
                - timestamp: Current timestamp
                - line_count: Number of lines
                - class_count: Number of classes

        Example:
            >>> stats = collector.get_stats()
            >>> print(f"Runtime: {stats['runtime_seconds']:.1f}s")
            >>> print(f"Active objects: {stats['active_objects']}")
        """
        return {
            "total": self.get_total_count(),
            "line_counts": dict(self.line_counts),
            "class_counts": dict(self.class_counts),
            "avg_speed": self.get_average_speed(),
            "max_speed": self.get_max_speed(),
            "min_speed": self.get_min_speed(),
            "avg_per_minute": self.get_average_per_minute(),
            "count_rate": self.get_count_rate(),
            "total_events": len(self.events),
            "runtime_seconds": time.time() - self._start_time,
            "active_objects": len(self.speed_history),
            "timestamp": time.time(),
            "line_count": len(self.line_counts),
            "class_count": len(self.class_counts),
        }

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Gets recent events.

        Args:
            limit: Maximum number of events.

        Returns:
            List[Dict[str, Any]]: Recent events as dictionaries.

        Example:
            >>> events = collector.get_recent_events(10)
            >>> for event in events:
            ...     print(f"{event['label']} at {event['timestamp']}")
        """
        recent = self.events[-limit:] if self.events else []
        return [
            {
                "timestamp": e.timestamp,
                "object_id": e.object_id,
                "line": e.line_name,
                "label": e.label,
                "centroid": e.centroid,
                "confidence": e.confidence,
            }
            for e in recent
        ]

    def get_line_count(self, line_id: str) -> int:
        """Gets the count for a specific line.

        Args:
            line_id: Line ID.

        Returns:
            int: Count for the line.

        Example:
            >>> entrance_count = collector.get_line_count("entrance")
            >>> print(f"Entrance: {entrance_count}")
        """
        return self.line_counts.get(line_id, 0)

    def get_class_count(self, class_name: str) -> int:
        """Gets the count for a specific class.

        Args:
            class_name: Class name.

        Returns:
            int: Count for the class.

        Example:
            >>> car_count = collector.get_class_count("car")
            >>> truck_count = collector.get_class_count("truck")
            >>> print(f"Cars: {car_count}, Trucks: {truck_count}")
        """
        return self.class_counts.get(class_name, 0)

    def reset(self) -> None:
        """Resets all statistics.

        This method clears all counts, history, and resets timers.

        Example:
            >>> collector.reset()
            >>> # All statistics are reset to zero
        """
        self.line_counts.clear()
        self.class_counts.clear()
        self.speed_history.clear()
        self.events.clear()
        self._counts_per_minute.clear()
        self._start_time = time.time()
        self._last_count_time = time.time()

    def merge(self, other: StatisticsCollector) -> None:
        """Merges statistics from another collector.

        This method combines counts and history from another
        statistics collector into this one.

        Args:
            other: Another statistics collector.

        Example:
            >>> collector1.merge(collector2)
            >>> # Statistics from collector2 are now included in collector1
        """
        for line_id, count in other.line_counts.items():
            self.line_counts[line_id] += count

        for class_name, count in other.class_counts.items():
            self.class_counts[class_name] += count

        self.events.extend(other.events)
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history :]

        for obj_id, speeds in other.speed_history.items():
            if obj_id in self.speed_history:
                self.speed_history[obj_id].extend(speeds)
                if len(self.speed_history[obj_id]) > HISTORY_MAX_SIZE:
                    self.speed_history[obj_id] = self.speed_history[obj_id][-HISTORY_MAX_SIZE:]
            else:
                self.speed_history[obj_id] = speeds.copy()
