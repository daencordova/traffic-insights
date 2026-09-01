"""Color manager for tracks and visualizations.

Provides consistent and efficient color assignment for visual
elements in the system.

This module handles color management for:
    - Track identification colors
    - Visualization elements
    - Consistent color palettes
    - Thread-safe color assignment
    - Color generation from IDs

Example:
    >>> from utils import get_color_manager
    >>>
    >>> # Get global color manager
    >>> color_manager = get_color_manager()
    >>>
    >>> # Get color for a track
    >>> color = color_manager.get_color(42)
    >>> print(f"Track color: {color}")  # (255, 0, 0)
    >>>
    >>> # Get colors for multiple tracks
    >>> tracks = [1, 5, 10, 15]
    >>> colors = color_manager.get_colors_for_tracks(tracks)
    >>> for track_id, color in colors.items():
    ...     print(f"Track {track_id}: {color}")
    >>>
    >>> # Get brighter version
    >>> brighter = color_manager.get_brighter_color(42, factor=1.5)
    >>>
    >>> # Get color with transparency
    >>> color_alpha = color_manager.get_color_with_alpha(42, alpha=0.7)
    >>>
    >>> # Get color palette
    >>> palette = color_manager.get_color_palette(10)
"""

import logging
import threading
from typing import Any

from core.constants import (
    COLOR_CHANNEL_MAX,
    HUE_CYCLE,
    HUE_SEGMENTS,
    SATURATION,
    VALUE,
)

logger = logging.getLogger(__name__)

DEFAULT_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
    (255, 0, 128),
    (128, 255, 0),
    (0, 128, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 0, 255),
]

HIGH_CONTRAST_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


class ColorManager:
    """Color manager with cache for tracks.

    This class provides consistent color assignment for tracks and
    visual elements with thread-safe operations and caching.

    Features:
        - Consistent color assignment by ID
        - Predefined color pool
        - Thread-safe for async pipelines
        - Support for custom colors
        - Usage statistics
        - Color variants (bright, dark, alpha)

    Attributes:
        _color_pool: List of predefined colors in BGR format.
        _color_cache: Cache of assigned colors by ID.
        _lock: Lock for thread-safe operations.
        _stats: Usage statistics of the manager.

    Example:
        >>> manager = ColorManager(colors=HIGH_CONTRAST_COLORS)
        >>> color = manager.get_color(42)
        >>> print(color)  # (255, 0, 0)
    """

    def __init__(self, colors: list[tuple[int, int, int]] | None = None):
        """Initializes the color manager.

        Args:
            colors: Custom list of colors (optional).
                   If not provided, uses DEFAULT_COLORS.

        Example:
            >>> # Use default colors
            >>> manager = ColorManager()
            >>>
            >>> # Use high contrast colors
            >>> manager = ColorManager(colors=HIGH_CONTRAST_COLORS)
            >>>
            >>> # Use custom color palette
            >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            >>> manager = ColorManager(colors=custom_colors)
        """
        self._color_pool = colors or DEFAULT_COLORS
        self._color_cache: dict[int, tuple[int, int, int]] = {}
        self._lock = threading.RLock()
        self._next_color_index = 0

        self._stats = {
            "total_assignments": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "active_colors": 0,
            "pool_size": len(self._color_pool),
        }

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        """Gets a color for a track.

        Args:
            track_id: Track ID.

        Returns:
            Tuple[int, int, int]: Color in BGR format.

        Example:
            >>> manager = ColorManager()
            >>> color = manager.get_color(42)
            >>> print(color)  # (255, 0, 0)
        """
        with self._lock:
            if track_id in self._color_cache:
                self._stats["cache_hits"] += 1
                return self._color_cache[track_id]

            self._stats["cache_misses"] += 1

            color = self._assign_color(track_id)
            self._color_cache[track_id] = color
            self._stats["total_assignments"] += 1
            self._stats["active_colors"] = len(self._color_cache)

            return color

    def _assign_color(self, track_id: int) -> tuple[int, int, int]:
        """Assigns a color to a track.

        Args:
            track_id: Track ID.

        Returns:
            Tuple[int, int, int]: Assigned color in BGR format.

        Note:
            Uses a combination of predefined colors and generated
            colors based on the track ID.
        """
        index = self._next_color_index % len(self._color_pool)
        self._next_color_index += 1

        if self._next_color_index >= len(self._color_pool) * 2:
            return self._generate_color_from_id(track_id)

        return self._color_pool[index]

    def _generate_color_from_id(self, track_id: int) -> tuple[int, int, int]:
        """Generates a color from the track ID.

        Args:
            track_id: Track ID.

        Returns:
            Tuple[int, int, int]: Generated color in BGR format.

        Note:
            Uses the HSV color model to generate colors distributed
            uniformly across the spectrum.
        """
        hue = (track_id * 137) % HUE_CYCLE
        saturation = SATURATION
        value = VALUE

        h = hue / 60.0
        c = value * saturation / COLOR_CHANNEL_MAX
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        segment = int(h) % HUE_SEGMENTS

        if segment == 0:
            r, g, b = c, x, 0
        elif segment == 1:
            r, g, b = x, c, 0
        elif segment == 2:
            r, g, b = 0, c, x
        elif segment == 3:
            r, g, b = 0, x, c
        elif segment == 4:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        b = int((b + m) * COLOR_CHANNEL_MAX)
        g = int((g + m) * COLOR_CHANNEL_MAX)
        r = int((r + m) * COLOR_CHANNEL_MAX)

        return (b, g, r)

    def get_color_palette(self, count: int) -> list[tuple[int, int, int]]:
        """Gets a color palette for multiple elements.

        Args:
            count: Number of colors to get.

        Returns:
            List[Tuple[int, int, int]]: List of colors in BGR format.

        Example:
            >>> colors = manager.get_color_palette(5)
            >>> for i, color in enumerate(colors):
            ...     print(f"Element {i}: {color}")
        """
        with self._lock:
            palette = []
            for i in range(count):
                temp_id = i
                palette.append(self.get_color(temp_id))
            return palette

    def get_colors_for_tracks(self, track_ids: list[int]) -> dict[int, tuple[int, int, int]]:
        """Gets colors for a list of tracks efficiently.

        Args:
            track_ids: List of track IDs.

        Returns:
            Dict[int, Tuple[int, int, int]]: Mapping of ID to color.

        Example:
            >>> active_tracks = [1, 5, 10, 15]
            >>> colors = manager.get_colors_for_tracks(active_tracks)
            >>> for track_id, color in colors.items():
            ...     print(f"Track {track_id}: {color}")
        """
        result = {}
        for track_id in track_ids:
            result[track_id] = self.get_color(track_id)
        return result

    def get_color_with_alpha(
        self, track_id: int, alpha: float = 0.5
    ) -> tuple[int, int, int, float]:
        """Gets a color with transparency for a track.

        Args:
            track_id: Track ID.
            alpha: Transparency level (0-1).

        Returns:
            Tuple[int, int, int, float]: Color in BGRA format.

        Example:
            >>> color = manager.get_color_with_alpha(42, 0.7)
            >>> # Useful for semi-transparent overlays
            >>> print(color)  # (255, 0, 0, 0.7)
        """
        b, g, r = self.get_color(track_id)
        return (b, g, r, alpha)

    def get_brighter_color(self, track_id: int, factor: float = 1.3) -> tuple[int, int, int]:
        """Gets a brighter version of a track's color.

        Args:
            track_id: Track ID.
            factor: Brightness factor (>1 for brighter).

        Returns:
            Tuple[int, int, int]: Brighter color in BGR format.

        Example:
            >>> brighter = manager.get_brighter_color(42, factor=1.5)
            >>> print(brighter)  # Brighter version of the color
        """
        b, g, r = self.get_color(track_id)

        b = min(COLOR_CHANNEL_MAX, int(b * factor))
        g = min(COLOR_CHANNEL_MAX, int(g * factor))
        r = min(COLOR_CHANNEL_MAX, int(r * factor))

        return (b, g, r)

    def get_darker_color(self, track_id: int, factor: float = 0.7) -> tuple[int, int, int]:
        """Gets a darker version of a track's color.

        Args:
            track_id: Track ID.
            factor: Darkness factor (0-1).

        Returns:
            Tuple[int, int, int]: Darker color in BGR format.

        Example:
            >>> darker = manager.get_darker_color(42, factor=0.5)
            >>> print(darker)  # Darker version of the color
        """
        b, g, r = self.get_color(track_id)

        b = max(0, int(b * factor))
        g = max(0, int(g * factor))
        r = max(0, int(r * factor))

        return (b, g, r)

    def clear_cache(self) -> None:
        """Clears the color cache.

        Example:
            >>> manager.clear_cache()
            >>> # All assigned colors are removed from cache
        """
        with self._lock:
            count = len(self._color_cache)
            self._color_cache.clear()
            self._next_color_index = 0
            self._stats["active_colors"] = 0
            self._stats["cache_hits"] = 0
            self._stats["cache_misses"] = 0
            self._stats["total_assignments"] = 0

            logger.info(f"Color cache cleared: {count} colors removed")

    def get_stats(self) -> dict[str, Any]:
        """Gets color manager statistics.

        Returns:
            Dict[str, any]: Usage statistics including:
                - total_assignments: Total color assignments
                - cache_hits: Cache hits
                - cache_misses: Cache misses
                - active_colors: Currently cached colors
                - pool_size: Size of color pool
                - hit_rate: Cache hit rate (0-1)
                - miss_rate: Cache miss rate (0-1)

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Active colors: {stats['active_colors']}")
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]

            return {
                **self._stats,
                "hit_rate": self._stats["cache_hits"] / max(1, total_requests),
                "miss_rate": self._stats["cache_misses"] / max(1, total_requests),
                "color_pool_size": len(self._color_pool),
            }

    def __len__(self) -> int:
        """Returns the number of colors in the cache."""
        with self._lock:
            return len(self._color_cache)


_default_color_manager: ColorManager | None = None


def get_color_manager() -> ColorManager:
    """Gets the global color manager instance.

    This function provides a singleton instance of the color manager
    for consistent color assignment across the system.

    Returns:
        ColorManager: Global instance of the color manager.

    Example:
        >>> color_manager = get_color_manager()
        >>> color = color_manager.get_color(42)
    """
    global _default_color_manager
    if _default_color_manager is None:
        _default_color_manager = ColorManager()
    return _default_color_manager


def get_color(index: int) -> tuple[int, int, int]:
    """Compatibility function for existing code.

    This function maintains compatibility with the previous
    get_color function in utils/geometry.py.

    Args:
        index: Index or ID to get color for.

    Returns:
        Tuple[int, int, int]: Color in BGR format.

    Example:
        >>> color = get_color(42)
        >>> print(color)  # (255, 0, 0)

    Note:
        This is a convenience wrapper around get_color_manager().get_color().
    """
    return get_color_manager().get_color(index)
