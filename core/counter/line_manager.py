"""Line manager for counting lines.

Handles configuration, validation, and access to counting lines.
"""

from dataclasses import dataclass
from typing import Any

from core.constants import POINT_DIMENSION


@dataclass(slots=True)
class CountingLine:
    """Represents a counting line.

    A counting line is a virtual line in the frame that vehicles cross
    to be counted. It defines the geometry, direction, and behavior
    of the counting operation.

    Attributes:
        id: Unique identifier for the line.
        name: Descriptive name for the line.
        points: Points defining the line.
        color: Color in BGR format.
        direction: Counting direction ('up' or 'down').
        y_position: Y position of the line.
        enabled: Whether the line is active.
        metadata: Additional metadata.

    Example:
        >>> line = CountingLine(
        ...     id="entrance",
        ...     name="Main Entrance",
        ...     points=[(100, 300), (500, 300)],
        ...     color=(0, 255, 0),
        ...     direction="down",
        ...     y_position=300,
        ...     enabled=True,
        ... )
        >>>
        >>> # Convert to dictionary
        >>> line_dict = line.to_dict()
        >>> print(line_dict["name"])
    """

    id: str
    name: str
    points: list[tuple[int, int]]
    color: tuple[int, int, int]
    direction: str
    y_position: int
    enabled: bool = True
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initializes metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Converts the line to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the line.

        Example:
            >>> line_dict = line.to_dict()
            >>> # Use as JSON or configuration
        """
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "color": self.color,
            "direction": self.direction,
            "y_position": self.y_position,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


class LineManager:
    """Manager for counting lines.

    This class handles all operations related to counting lines
    including validation, access, and state management.

    Responsibilities:
        - Validate line configurations
        - Provide access to lines by ID or index
        - Maintain line state (enabled/disabled)
        - Add and remove lines
        - Toggle line activation

    Attributes:
        lines: List of counting lines.

    Example:
        >>> # Initialize from configuration
        >>> config = [
        ...     {
        ...         "id": "entrance",
        ...         "name": "Entrance",
        ...         "points": [(100, 300), (500, 300)],
        ...         "color": [0, 255, 0],
        ...         "direction": "down",
        ...         "enabled": True,
        ...     },
        ...     {
        ...         "id": "exit",
        ...         "name": "Exit",
        ...         "points": [(100, 500), (500, 500)],
        ...         "color": [255, 0, 0],
        ...         "direction": "up",
        ...         "enabled": True,
        ...     },
        ... ]
        >>>
        >>> manager = LineManager(config)
        >>>
        >>> # Access lines
        >>> entrance = manager.get_line("entrance")
        >>> all_lines = manager.get_all_lines()
        >>>
        >>> # Toggle a line
        >>> manager.toggle_line("exit")
        >>>
        >>> # Get statistics
        >>> stats = manager.get_stats()
        >>> print(f"Active lines: {stats['active_lines']}")
    """

    def __init__(self, config_lines: list[dict[str, Any]]):
        """Initializes the line manager.

        Args:
            config_lines: List of line configurations.

        Example:
            >>> manager = LineManager(config_lines)
            >>> print(f"Loaded {manager.get_total_lines()} lines")
        """
        self.lines: list[CountingLine] = []
        self._initialize_lines(config_lines)

    def _initialize_lines(self, config_lines: list[dict[str, Any]]) -> None:
        """Initializes lines from configuration.

        Args:
            config_lines: List of line configurations.

        Note:
            Invalid configurations are skipped with a warning.
        """
        for idx, line_config in enumerate(config_lines):
            if not self._validate_line_config(line_config):
                continue

            points = line_config.get("points", [])
            first_point = points[0] if points else (0, 0)

            line = CountingLine(
                id=line_config.get("id", f"line_{idx}"),
                name=line_config.get("name", f"Line {idx + 1}"),
                points=points,
                color=tuple(line_config.get("color", (0, 255, 0))),
                direction=line_config.get("direction", "down"),
                y_position=first_point[1] if points else 0,
                enabled=line_config.get("enabled", True),
                metadata=line_config.get("metadata", {}),
            )
            self.lines.append(line)

    def _validate_line_config(self, config: dict[str, Any]) -> bool:
        """Validates a line configuration.

        Args:
            config: Configuration to validate.

        Returns:
            bool: True if the configuration is valid.

        Example:
            >>> if not manager._validate_line_config(config):
            ...     print("Invalid line configuration")
        """
        if not isinstance(config, dict):
            return False

        points = config.get("points", [])
        if not points or len(points) < 1:
            return False

        first_point = points[0]
        if not isinstance(first_point, (list, tuple)) or len(first_point) != POINT_DIMENSION:
            return False

        x, y = first_point
        return isinstance(x, (int, float)) or not isinstance(y, (int, float))

    def get_line(self, line_id: str) -> CountingLine | None:
        """Gets a line by its ID.

        Args:
            line_id: Line ID.

        Returns:
            Optional[CountingLine]: Found line or None.

        Example:
            >>> line = manager.get_line("entrance")
            >>> if line:
            ...     print(f"Found line: {line.name}")
        """
        for line in self.lines:
            if line.id == line_id:
                return line
        return None

    def get_line_by_index(self, index: int) -> CountingLine | None:
        """Gets a line by its index.

        Args:
            index: Line index.

        Returns:
            Optional[CountingLine]: Found line or None.

        Example:
            >>> line = manager.get_line_by_index(0)
            >>> if line:
            ...     print(f"First line: {line.name}")
        """
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return None

    def get_all_lines(self) -> list[CountingLine]:
        """Gets all active lines.

        Returns:
            list[CountingLine]: List of enabled lines.

        Example:
            >>> active_lines = manager.get_all_lines()
            >>> for line in active_lines:
            ...     print(f"Active line: {line.name}")
        """
        return [line for line in self.lines if line.enabled]

    def get_line_count(self) -> int:
        """Gets the number of active lines.

        Returns:
            int: Number of enabled lines.

        Example:
            >>> count = manager.get_line_count()
            >>> print(f"Active lines: {count}")
        """
        return len(self.get_all_lines())

    def get_total_lines(self) -> int:
        """Gets the total number of lines (including disabled).

        Returns:
            int: Total number of lines.

        Example:
            >>> total = manager.get_total_lines()
            >>> active = manager.get_line_count()
            >>> print(f"{active} active out of {total} total")
        """
        return len(self.lines)

    def is_empty(self) -> bool:
        """Checks if no lines are configured.

        Returns:
            bool: True if there are no lines.

        Example:
            >>> if manager.is_empty():
            ...     print("No counting lines configured")
        """
        return len(self.lines) == 0

    def has_active_lines(self) -> bool:
        """Checks if there are active lines.

        Returns:
            bool: True if there are active lines.

        Example:
            >>> if manager.has_active_lines():
            ...     print("Counting lines available")
        """
        return self.get_line_count() > 0

    def add_line(self, line: CountingLine) -> None:
        """Adds a new line.

        Args:
            line: Line to add.

        Example:
            >>> new_line = CountingLine(...)
            >>> manager.add_line(new_line)
            >>> print(f"Added line: {new_line.id}")
        """
        self.lines.append(line)

    def remove_line(self, line_id: str) -> bool:
        """Removes a line by its ID.

        Args:
            line_id: ID of the line to remove.

        Returns:
            bool: True if removed successfully.

        Example:
            >>> if manager.remove_line("entrance"):
            ...     print("Line removed")
        """
        for i, line in enumerate(self.lines):
            if line.id == line_id:
                self.lines.pop(i)
                return True
        return False

    def toggle_line(self, line_id: str) -> bool:
        """Toggles a line enabled/disabled.

        Args:
            line_id: Line ID.

        Returns:
            bool: True if the line was found.

        Example:
            >>> manager.toggle_line("exit")
            >>> # Line is now toggled
        """
        line = self.get_line(line_id)
        if line:
            line.enabled = not line.enabled
            return True
        return False

    def to_dict(self) -> list[dict[str, Any]]:
        """Converts all lines to a dictionary.

        Returns:
            list[dict[str, Any]]: List of line dictionaries.

        Example:
            >>> lines_dict = manager.to_dict()
            >>> # Save to JSON or configuration file
        """
        return [line.to_dict() for line in self.lines]

    def get_stats(self) -> dict[str, Any]:
        """Gets manager statistics.

        Returns:
            dict[str, Any]: Statistics including:
                - total_lines: Total number of lines
                - active_lines: Number of active lines
                - disabled_lines: Number of disabled lines
                - line_ids: List of line IDs
                - line_names: List of line names

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Total: {stats['total_lines']}")
            >>> print(f"Active: {stats['active_lines']}")
            >>> print(f"Line IDs: {stats['line_ids']}")
        """
        return {
            "total_lines": self.get_total_lines(),
            "active_lines": self.get_line_count(),
            "disabled_lines": self.get_total_lines() - self.get_line_count(),
            "line_ids": [line.id for line in self.lines],
            "line_names": [line.name for line in self.lines],
        }

    def __len__(self) -> int:
        """Returns the total number of lines."""
        return len(self.lines)

    def __iter__(self):
        """Iterator over the lines."""
        return iter(self.lines)
