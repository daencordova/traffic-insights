"""Intelligent vehicle counter.

This module implements the main counter system that orchestrates
line management, crossing detection, and statistics collection
for vehicle counting.

The counter is responsible for:
    - Managing virtual counting lines
    - Detecting vehicle crossings across lines
    - Collecting counting statistics
    - Maintaining event history
    - Calculating traffic metrics

Main components:
    - LineManager: Management of counting lines
    - CrossingDetector: Detection of line crossings
    - StatisticsCollector: Collection of statistics

Example:
    >>> counter = VehicleCounter()
    >>> tracks = tracker.update(detections, frame)
    >>> stats = counter.process(tracks, frame)
    >>> print(f"Total vehicles: {stats['total']}")
    >>> print(f"Line counts: {stats['line_counts']}")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from core.constants import (
    DETECTION_POINT_DIMENSION,
    DETECTION_VELOCITY_DIMENSION,
)
from core.counter.crossing_detector import CrossingDetector
from core.counter.line_manager import CountingLine, LineManager
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent
from core.interfaces import ICounter
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    import numpy as np

    from core.types import StatsDict, TrackInfoDict


class VehicleCounter(ICounter, LoggerMixin):
    """Vehicle counter with support for multiple lines.

    This class orchestrates the line management, crossing detection,
    and statistics collection components to provide a complete vehicle
    counting system.

    Features:
        - Multiple virtual counting lines
        - Direction detection (entering/exiting)
        - Per-line and per-class statistics
        - Event history with timestamps
        - Average speed calculation
        - Per-minute counting
        - Automatic line configuration from settings

    Attributes:
        line_manager: Manager of counting lines.
        crossing_detector: Detector of line crossings.
        stats_collector: Collector of statistics.
        config: System configuration.
        _frame_counter: Counter of processed frames.
        _last_process_time: Time of the last update.

    Example:
        >>> counter = VehicleCounter()
        >>>
        >>> # Process tracks from tracker
        >>> tracks = tracker.update(detections, frame)
        >>> stats = counter.process(tracks, frame)
        >>>
        >>> # Access statistics
        >>> print(f"Total vehicles: {stats['total']}")
        >>> print(f"Counts by line: {stats['line_counts']}")
        >>> print(f"Counts by class: {stats['class_counts']}")
        >>>
        >>> # Get recent events
        >>> events = counter.get_recent_log(10)
        >>> for event in events:
        ...     print(f"Vehicle {event['object_id']} crossed {event['line_name']}")
    """

    def __init__(self, config=None) -> None:
        """Initializes the vehicle counter.

        Args:
            config: System configuration. If None, uses global configuration.

        Note:
            If no counting lines are configured, the counter will
            operate without performing counts (only basic statistics).

        Example:
            >>> # Use global configuration
            >>> counter = VehicleCounter()
            >>>
            >>> # Use custom configuration
            >>> counter = VehicleCounter(custom_config)
        """
        from config.manager import config_manager

        self.config = config or config_manager.config
        self.logger.info("Initializing VehicleCounter")

        self.line_manager = LineManager(self.config.counting_lines)
        self.crossing_detector = CrossingDetector()
        self.stats_collector = StatisticsCollector()

        line_count = self.line_manager.get_line_count()
        self.logger.info(
            "Counter initialized",
            lines=line_count,
            line_ids=[line.id for line in self.line_manager.get_all_lines()],
        )

        if line_count == 0:
            self.logger.warning("No counting lines configured")

        self._frame_counter = 0
        self._last_process_time = 0.0
        self._processed_tracks_count = 0

    def process(self, tracks: TrackInfoDict, frame: np.ndarray) -> dict[str, Any]:
        """Processes tracks and updates counts.

        This method analyzes each active track and detects crossings
        across counting lines, updating statistics accordingly.

        Args:
            tracks: Dictionary of active tracks from the tracker.
            frame: Current frame (needed for dimensions and context).

        Returns:
            Dict[str, Any]: Updated statistics including:
                - total: Total vehicles counted
                - line_counts: Counts per line
                - class_counts: Counts per class
                - avg_speed: Average speed
                - max_speed: Maximum speed
                - min_speed: Minimum speed
                - active_objects: Number of active objects
                - frame_counter: Processed frame number

        Example:
            >>> tracks = tracker.update(detections, frame)
            >>> stats = counter.process(tracks, frame)
            >>> if stats["total"] > 100:
            ...     print("High traffic volume detected")
            >>>
            >>> # Check specific line count
            >>> if stats["line_counts"].get("entrance", 0) > 50:
            ...     print("Many vehicles entering")

        Note:
            Processing is performed for each active track, checking
            whether it has crossed any counting line.
        """
        start_time = time.perf_counter()
        self._frame_counter += 1
        self._processed_tracks_count = 0

        if not self._is_valid_input(frame, tracks):
            return self.get_stats()

        if not self.line_manager.has_active_lines():
            return self.get_stats()

        height = frame.shape[0]
        self._process_all_tracks(tracks, height)

        self._update_minute_stats()

        self._log_performance(start_time)

        return self.get_stats()

    def _is_valid_input(self, frame: np.ndarray, tracks: dict[int, dict[str, Any]]) -> bool:
        """Validates input to the process method.

        Args:
            frame: Frame to validate.
            tracks: Tracks to validate.

        Returns:
            bool: True if input is valid.

        Example:
            >>> if not self._is_valid_input(frame, tracks):
            ...     return self.get_stats()
        """
        if frame is None or frame.size == 0:
            self.logger.debug("Invalid frame received")
            return False

        if not isinstance(tracks, dict):
            self.logger.debug("Invalid tracks (not a dictionary)")
            return False

        return True

    def _process_all_tracks(self, tracks: dict[int, dict[str, Any]], height: int) -> None:
        """Processes all tracks for crossing detection.

        Args:
            tracks: Dictionary of active tracks.
            height: Frame height.
        """
        for object_id, track_data in tracks.items():
            try:
                if self._process_single_track(object_id, track_data, height):
                    self._processed_tracks_count += 1
            except Exception as e:
                self.logger.debug("Error processing track", object_id=object_id, error=str(e))
                continue

    def _process_single_track(
        self, object_id: int, track_data: dict[str, Any], height: int
    ) -> bool:
        """Processes a single track for crossing detection.

        Args:
            object_id: Object ID.
            track_data: Track data.
            height: Frame height (for coordinates).

        Returns:
            bool: True if processed successfully (crossed any line).
        """
        if not self._validate_track_data(track_data):
            return False

        centroid = track_data["centroid"]
        crossed_any = False

        for line in self.line_manager.get_all_lines():
            if self._check_line_crossing(object_id, centroid, line, height):
                crossed_any = True

        self._record_track_velocity(object_id, track_data)

        return crossed_any

    def _validate_track_data(self, track_data: dict[str, Any]) -> bool:
        """Validates track data.

        Args:
            track_data: Track data to validate.

        Returns:
            bool: True if data is valid.

        Example:
            >>> if not self._validate_track_data(track_data):
            ...     # Skip this track
            ...     continue
        """
        if not isinstance(track_data, dict):
            return False

        centroid = track_data.get("centroid")
        if centroid is None:
            return False

        if not isinstance(centroid, (tuple, list)) or len(centroid) != DETECTION_POINT_DIMENSION:
            return False

        x, y = centroid
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False

        return not (x < 0 or y < 0)

    def _check_line_crossing(
        self, object_id: int, centroid: tuple[int, int], line: CountingLine, height: int
    ) -> bool:
        """Checks if an object has crossed a specific line.

        Args:
            object_id: Object ID.
            centroid: Current centroid.
            line: Counting line.
            height: Frame height.

        Returns:
            bool: True if the object crossed the line.
        """
        crossed = self.crossing_detector.detect_crossing(
            object_id=object_id, current_position=centroid, line=line, height=height
        )

        if crossed:
            self._record_crossing(object_id, line)
            return True

        return False

    def _record_crossing(self, object_id: int, line: CountingLine) -> None:
        """Records a crossing in the statistics collector.

        Args:
            object_id: Object ID.
            line: Counting line.

        Note:
            This method is called when a vehicle crosses a counting line.
            It updates the statistics collector with the crossing event.
        """
        self.stats_collector.record_crossing(
            object_id=object_id,
            line_id=line.id,
            line_name=line.name,
        )

        self.logger.debug(
            "Vehicle counted",
            object_id=object_id,
            line=line.id,
            total=self.stats_collector.get_total_count(),
        )

    def _record_track_velocity(self, object_id: int, track_data: dict[str, Any]) -> None:
        """Records track velocity.

        Args:
            object_id: Object ID.
            track_data: Track data containing velocity.
        """
        velocity = track_data.get("velocity", (0, 0))
        if isinstance(velocity, (tuple, list)) and len(velocity) == DETECTION_VELOCITY_DIMENSION:
            self.stats_collector.record_speed(object_id, velocity)

    def _update_minute_stats(self) -> None:
        """Updates per-minute counts."""
        total = self.stats_collector.get_total_count()
        self.stats_collector.update_minute_counts(total)

    def _log_performance(self, start_time: float) -> None:
        """Logs performance metrics.

        Args:
            start_time: Timestamp of processing start.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._last_process_time = elapsed_ms

        if self._frame_counter % 100 == 0 and self._processed_tracks_count > 0:
            self.logger.debug(
                "Counting processing",
                frames=self._frame_counter,
                tracks_processed=self._processed_tracks_count,
                total=self.stats_collector.get_total_count(),
                time_ms=f"{elapsed_ms:.2f}",
            )

    def get_stats(self) -> StatsDict:
        """Returns detailed counting statistics.

        Returns:
            Dict[str, Any]: Current statistics including:
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
                - frame_counter: Frame number
                - last_process_time_ms: Processing time
                - processed_tracks: Number of processed tracks

        Example:
            >>> stats = counter.get_stats()
            >>> print(f"Total: {stats['total']}")
            >>> print(f"Average speed: {stats['avg_speed']:.1f} px/frame")
            >>> print(f"Count rate: {stats['count_rate']:.2f} vehicles/min")
        """
        stats = self.stats_collector.get_stats()
        stats["active_objects"] = self.crossing_detector.get_active_objects()
        stats["frame_counter"] = self._frame_counter
        stats["last_process_time_ms"] = self._last_process_time
        stats["processed_tracks"] = self._processed_tracks_count

        return stats

    def get_recent_log(self, limit: int = 20) -> list[dict[str, Any]]:
        """Gets recent counting events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List[Dict[str, Any]]: Recent events as dictionaries.

        Example:
            >>> events = counter.get_recent_log(10)
            >>> for event in events:
            ...     print(f"Vehicle {event['object_id']} at {event['timestamp']}")
        """
        return self.stats_collector.get_recent_events(limit)

    def get_line_count(self, line_id: str) -> int:
        """Gets the count for a specific line.

        Args:
            line_id: Line ID.

        Returns:
            int: Line count.

        Example:
            >>> entrance_count = counter.get_line_count("entrance")
            >>> exit_count = counter.get_line_count("exit")
        """
        return self.stats_collector.get_line_count(line_id)

    def get_class_count(self, class_name: str) -> int:
        """Gets the count for a specific class.

        Args:
            class_name: Class name (e.g., 'car', 'truck').

        Returns:
            int: Class count.

        Example:
            >>> cars = counter.get_class_count("car")
            >>> trucks = counter.get_class_count("truck")
        """
        return self.stats_collector.get_class_count(class_name)

    def get_crossed_lines(self, object_id: int) -> set[str]:
        """Gets the lines an object has crossed.

        Args:
            object_id: Object ID.

        Returns:
            Set[str]: Set of line IDs crossed.

        Example:
            >>> lines = counter.get_crossed_lines(42)
            >>> if "entrance" in lines:
            ...     print("Vehicle entered the area")
        """
        return self.crossing_detector.get_crossed_lines(object_id)

    def get_line_manager(self) -> LineManager:
        """Gets the line manager."""
        return self.line_manager

    def get_crossing_detector(self) -> CrossingDetector:
        """Gets the crossing detector."""
        return self.crossing_detector

    def get_statistics_collector(self) -> StatisticsCollector:
        """Gets the statistics collector."""
        return self.stats_collector

    def get_events(self, limit: int = 100) -> list[VehicleEvent]:
        """Gets counting events.

        Args:
            limit: Maximum number of events.

        Returns:
            List[VehicleEvent]: Counting events.

        Example:
            >>> events = counter.get_events(10)
            >>> for event in events:
            ...     print(f"{event.object_id} crossed {event.line_name}")
        """
        return self.stats_collector.events[-limit:] if self.stats_collector.events else []

    def reset_line(self, line_id: str) -> None:
        """Resets the count for a specific line.

        Args:
            line_id: ID of the line to reset.

        Example:
            >>> counter.reset_line("entrance")
            >>> # The count for the entrance line is reset to 0
        """
        self.crossing_detector.reset_line(line_id)
        self.stats_collector.line_counts[line_id] = 0
        self.logger.info(f"Line {line_id} reset")

    def reset(self) -> None:
        """Resets all counters and statistics.

        Example:
            >>> counter.reset()
            >>> # All counts are reset to 0
        """
        self.logger.info(
            "Resetting counter",
            total=self.stats_collector.get_total_count(),
            lines=self.line_manager.get_total_lines(),
        )

        self.crossing_detector.clear()
        self.stats_collector.reset()
        self._frame_counter = 0
        self._last_process_time = 0.0
        self._processed_tracks_count = 0

        self.logger.info("Counter reset")

    def __len__(self) -> int:
        """Returns the total number of counts."""
        return self.stats_collector.get_total_count()

    def __str__(self) -> str:
        """String representation of the counter."""
        return (
            f"VehicleCounter(total={self.stats_collector.get_total_count()}, "
            f"lines={self.line_manager.get_line_count()})"
        )
