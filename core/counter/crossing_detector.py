"""Line crossing detector.

Handles detection of when an object crosses a counting line.
"""

from collections import defaultdict
import time

from core.counter.line_manager import CountingLine
from utils.geometry import check_crossing


class CrossingDetector:
    """Detector of line crossings.

    This class tracks object positions and detects when objects cross
    virtual counting lines. It prevents duplicate counting and maintains
    history of crossings per object.

    Responsibilities:
        - Detect line crossings
        - Maintain crossing history per object
        - Prevent duplicate counting
        - Track object positions over time
        - Provide crossing statistics

    Attributes:
        _crossed_lines: History of crossed lines per object.
        _previous_positions: Previous positions per object.
        _cross_timestamps: Timestamps of crossings per object.
        _stats: Detection statistics.

    Example:
        >>> detector = CrossingDetector()
        >>>
        >>> # Define a counting line
        >>> line = CountingLine(
        ...     id="entrance", name="Entrance", start=(100, 300), end=(500, 300), direction="down"
        ... )
        >>>
        >>> # Check crossing for an object
        >>> object_id = 42
        >>> current_pos = (320, 310)  # (x, y)
        >>> previous_pos = (320, 290)
        >>>
        >>> # Store previous position
        >>> detector._previous_positions[object_id] = previous_pos
        >>>
        >>> # Detect crossing
        >>> crossed = detector.detect_crossing(object_id, current_pos, line, height=720)
        >>> if crossed:
        ...     print("Vehicle crossed the entrance line!")
    """

    def __init__(self):
        """Initializes the crossing detector."""
        self._crossed_lines: dict[int, set[str]] = defaultdict(set)
        self._previous_positions: dict[int, tuple[int, int]] = {}
        self._cross_timestamps: dict[int, float] = {}

        self._stats = {
            "total_crossings": 0,
            "unique_objects": 0,
            "active_objects": 0,
        }

    def detect_crossing(
        self, object_id: int, current_position: tuple[int, int], line: CountingLine, height: int
    ) -> bool:
        """Detects if an object has crossed a line.

        This method checks if an object has crossed a counting line by
        comparing its previous and current positions. It prevents duplicate
        crossings for the same object-line pair.

        Args:
            object_id: Object ID.
            current_position: Current position (x, y) of the object.
            line: Counting line to check.
            height: Frame height (used for line Y position if not specified).

        Returns:
            bool: True if the object crossed the line.

        Example:
            >>> crossed = detector.detect_crossing(
            ...     object_id=42, current_position=(100, 200), line=entrance_line, height=720
            ... )
            >>> if crossed:
            ...     print(f"Object {object_id} crossed the line!")
        """
        if line.id in self._crossed_lines[object_id]:
            return False

        prev_pos = self._previous_positions.get(object_id)
        if prev_pos is None:
            self._previous_positions[object_id] = current_position
            return False

        prev_y, current_y = prev_pos[1], current_position[1]
        line_y = line.y_position or height // 2

        crossed = check_crossing(prev_y, current_y, line_y, line.direction)

        if crossed:
            self._crossed_lines[object_id].add(line.id)
            self._cross_timestamps[object_id] = time.time()
            self._stats["total_crossings"] += 1

            if len(self._crossed_lines[object_id]) == 1:
                self._stats["unique_objects"] += 1

        self._previous_positions[object_id] = current_position

        return crossed

    def has_crossed_line(self, object_id: int, line_id: str) -> bool:
        """Checks if an object has already crossed a specific line.

        Args:
            object_id: Object ID.
            line_id: Line ID.

        Returns:
            bool: True if the object has crossed the line.

        Example:
            >>> if detector.has_crossed_line(42, "entrance"):
            ...     print("Vehicle already counted")
        """
        return line_id in self._crossed_lines.get(object_id, set())

    def get_crossed_lines(self, object_id: int) -> set[str]:
        """Gets the lines that an object has crossed.

        Args:
            object_id: Object ID.

        Returns:
            Set[str]: Set of crossed line IDs.

        Example:
            >>> crossed = detector.get_crossed_lines(42)
            >>> for line_id in crossed:
            ...     print(f"Vehicle crossed line: {line_id}")
        """
        return self._crossed_lines.get(object_id, set())

    def get_cross_timestamp(self, object_id: int) -> float | None:
        """Gets the timestamp of the last crossing for an object.

        Args:
            object_id: Object ID.

        Returns:
            Optional[float]: Timestamp of the last crossing or None.

        Example:
            >>> timestamp = detector.get_cross_timestamp(42)
            >>> if timestamp:
            ...     print(f"Last crossed at: {timestamp}")
        """
        return self._cross_timestamps.get(object_id)

    def reset_object(self, object_id: int) -> None:
        """Resets the history for a specific object.

        This method removes all tracking data for an object, allowing
        it to be counted again if it reappears.

        Args:
            object_id: Object ID to reset.

        Example:
            >>> detector.reset_object(42)
            >>> # Object 42 can now be counted again
        """
        self._crossed_lines.pop(object_id, None)
        self._previous_positions.pop(object_id, None)
        self._cross_timestamps.pop(object_id, None)

    def reset_line(self, line_id: str) -> None:
        """Resets the history for a specific line.

        This method removes all crossings for a specific line,
        allowing objects to be counted again for that line.

        Args:
            line_id: ID of the line to reset.

        Example:
            >>> detector.reset_line("entrance")
            >>> # All objects can now be counted for the entrance line
        """
        for object_id in list(self._crossed_lines.keys()):
            if line_id in self._crossed_lines[object_id]:
                self._crossed_lines[object_id].remove(line_id)
                self._stats["total_crossings"] = max(0, self._stats["total_crossings"] - 1)

    def clear(self) -> None:
        """Clears all history.

        This method resets the detector to its initial state, removing
        all tracking data and statistics.

        Example:
            >>> detector.clear()
            >>> # Detector is now reset to initial state
        """
        self._crossed_lines.clear()
        self._previous_positions.clear()
        self._cross_timestamps.clear()
        self._stats = {
            "total_crossings": 0,
            "unique_objects": 0,
            "active_objects": 0,
        }

    def get_stats(self) -> dict[str, int]:
        """Gets detector statistics.

        Returns:
            dict[str, int]: Statistics including:
                - total_crossings: Total crossing events
                - unique_objects: Number of unique objects
                - active_objects: Currently active objects

        Example:
            >>> stats = detector.get_stats()
            >>> print(f"Total crossings: {stats['total_crossings']}")
            >>> print(f"Unique objects: {stats['unique_objects']}")
        """
        self._stats["active_objects"] = len(self._crossed_lines)
        return self._stats.copy()

    def get_active_objects(self) -> int:
        """Gets the number of active objects.

        Returns:
            int: Number of objects with registered crossings.

        Example:
            >>> active = detector.get_active_objects()
            >>> print(f"Active objects: {active}")
        """
        return len(self._crossed_lines)

    def __len__(self) -> int:
        """Returns the number of objects with registered crossings."""
        return len(self._crossed_lines)
