"""General utility functions.

This module provides various utility functions for file system operations,
memory management, garbage collection, and time formatting.

Features:
    - Directory creation and management
    - Timestamp-based filename generation
    - Time formatting (HH:MM:SS)
    - Memory usage monitoring (via psutil)
    - Garbage collection control
    - Memory tracking snapshots

Example:
    >>> from utils.helpers import (
    ...     ensure_directory_exists,
    ...     get_timestamp_filename,
    ...     format_time,
    ...     get_memory_usage,
    ...     force_garbage_collection,
    ...     MemoryTracker,
    ... )
    >>>
    >>> # Create directory
    >>> ensure_directory_exists("data/output/")
    >>>
    >>> # Generate timestamped filename
    >>> filename = get_timestamp_filename("capture", "jpg")
    >>> print(filename)  # capture_20240101_120000.jpg
    >>>
    >>> # Format time
    >>> formatted = format_time(3661)
    >>> print(formatted)  # 01:01:01
    >>>
    >>> # Monitor memory
    >>> tracker = MemoryTracker("app")
    >>> tracker.snapshot("Start")
    >>> # ... do work ...
    >>> tracker.snapshot("After processing")
    >>> stats = tracker.get_stats()
    >>> print(f"Peak memory: {stats['peak_mb']:.2f} MB")
"""

import gc
from pathlib import Path
import time
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def ensure_directory_exists(path: str) -> None:
    """Ensures a directory exists, creating it if necessary.

    This function creates all parent directories if they don't exist,
    similar to `mkdir -p` in Unix.

    Args:
        path: Directory path.

    Example:
        >>> ensure_directory_exists("data/screenshots/")
        >>> ensure_directory_exists("output/videos/processed/")
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp_filename(prefix: str = "", extension: str = "jpg") -> Path:
    """Generates a filename with a timestamp.

    Creates a filename in the format: [prefix]_YYYYMMDD_HHMMSS.extension

    Args:
        prefix: Prefix for the filename (optional).
        extension: File extension (without dot).

    Returns:
        Path: Path object with timestamped filename.

    Example:
        >>> get_timestamp_filename("capture", "jpg")
        Path("capture_20240101_120000.jpg")
        >>> get_timestamp_filename("log", "txt")
        Path("log_20240101_120000.txt")
        >>> get_timestamp_filename(extension="png")
        Path("20240101_120000.png")
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}.{extension}" if prefix else f"{timestamp}.{extension}"
    return Path(name)


def format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS format.

    Args:
        seconds: Seconds to format.

    Returns:
        str: Formatted string in HH:MM:SS or MM:SS format.

    Example:
        >>> format_time(3661)
        "01:01:01"
        >>> format_time(125)
        "02:05"
        >>> format_time(0)
        "00:00"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_memory_usage() -> dict[str, float]:
    """Gets memory usage information for the current process.

    Returns:
        Dict[str, float]: Memory information including:
            - rss_mb: Resident Set Size in MB
            - vms_mb: Virtual Memory Size in MB
            - percent: Process memory percentage
            - system_percent: System memory percentage
            - system_available_mb: Available system memory in MB

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"RSS: {mem['rss_mb']:.2f} MB")
        >>> print(f"System available: {mem['system_available_mb']:.2f} MB")
        >>> print(f"System usage: {mem['system_percent']:.1f}%")
    """
    if not PSUTIL_AVAILABLE:
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
            "system_percent": 0.0,
            "system_available_mb": 0.0,
        }

    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "system_percent": psutil.virtual_memory().percent,
            "system_available_mb": psutil.virtual_memory().available / (1024 * 1024),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
            "system_percent": 0.0,
            "system_available_mb": 0.0,
        }


def force_garbage_collection() -> dict[str, int | bool]:
    """Forces garbage collection and returns statistics.

    This function triggers garbage collection and provides statistics
    about collected objects.

    Returns:
        Dict[str, Union[int, bool]]: GC statistics including:
            - collected_objects: Number of objects collected
            - gc_enabled: Whether GC is enabled
            - garbage_count: Number of objects in garbage

    Example:
        >>> stats = force_garbage_collection()
        >>> print(f"Collected: {stats['collected_objects']} objects")
        >>> print(f"Garbage remaining: {stats['garbage_count']}")
    """
    collected = gc.collect()
    return {
        "collected_objects": collected,
        "gc_enabled": gc.isenabled(),
        "garbage_count": len(gc.garbage),
    }


class MemoryTracker:
    """Simple memory usage tracker.

    This class provides functionality to track memory usage over time
    by taking snapshots and calculating statistics.

    Attributes:
        name: Identifier for the tracker.

    Example:
        >>> tracker = MemoryTracker("application")
        >>>
        >>> # Take initial snapshot
        >>> tracker.snapshot("Start")
        >>>
        >>> # Perform operations...
        >>> large_data = load_data()
        >>> tracker.snapshot("After loading")
        >>>
        >>> # Process data...
        >>> result = process(large_data)
        >>> tracker.snapshot("After processing")
        >>>
        >>> # Get statistics
        >>> stats = tracker.get_stats()
        >>> print(f"Peak: {stats['peak_mb']:.2f} MB")
        >>> print(f"Delta: {stats['delta_mb']:.2f} MB")
        >>>
        >>> # Clear snapshots
        >>> tracker.clear()
    """

    def __init__(self, name: str = "memory_tracker") -> None:
        """Initializes the memory tracker.

        Args:
            name: Identifying name for the tracker.

        Example:
            >>> tracker = MemoryTracker("video_processing")
        """
        self.name: str = name
        self._snapshots: list[dict[str, Any]] = []
        self._max_snapshots: int = 100
        self._start_memory: float | None = None

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Takes a snapshot of memory usage.

        Args:
            label: Label to identify the snapshot.

        Returns:
            dict: Memory information including timestamp, label, and delta.

        Example:
            >>> snapshot = tracker.snapshot("Loading model")
            >>> print(f"Memory: {snapshot['rss_mb']:.2f} MB")
            >>> print(f"Delta from start: {snapshot['delta_mb']:.2f} MB")
        """
        memory = get_memory_usage()
        memory["timestamp"] = time.time()
        memory["label"] = label

        if self._start_memory is None:
            self._start_memory = memory["rss_mb"]

        memory["delta_mb"] = memory["rss_mb"] - self._start_memory

        self._snapshots.append(memory)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

        return memory

    def get_stats(self) -> dict[str, float]:
        """Gets statistics from the snapshots.

        Returns:
            dict: Statistics including:
                - count: Number of snapshots
                - current_mb: Current memory usage in MB
                - peak_mb: Peak memory usage in MB
                - delta_mb: Delta from start in MB
                - start_mb: Starting memory in MB

        Example:
            >>> stats = tracker.get_stats()
            >>> print(f"Count: {stats['count']:.0f} snapshots")
            >>> print(f"Current: {stats['current_mb']:.2f} MB")
            >>> print(f"Peak: {stats['peak_mb']:.2f} MB")
            >>> print(f"Delta: {stats['delta_mb']:.2f} MB")
        """
        if not self._snapshots:
            return {"count": 0.0}

        current = self._snapshots[-1]
        peak = max(s["rss_mb"] for s in self._snapshots)

        return {
            "count": float(len(self._snapshots)),
            "current_mb": current["rss_mb"],
            "peak_mb": peak,
            "delta_mb": current["delta_mb"],
            "start_mb": self._start_memory or 0.0,
        }

    def clear(self) -> None:
        """Clears all stored snapshots.

        Example:
            >>> tracker.clear()
            >>> # All snapshots are removed
        """
        self._snapshots.clear()
        self._start_memory = None
