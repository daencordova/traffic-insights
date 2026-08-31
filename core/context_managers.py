"""Specialized context managers for resource management.

This module provides a collection of context managers for common operations:
- Performance measurement and timing
- Memory usage tracking
- Video capture with automatic cleanup
- Thread locking with timeout support
- File handling with automatic directory creation
- Garbage collection control
- Resource pooling for expensive objects
- Reconnection handling for video sources

All context managers ensure proper resource cleanup even when exceptions occur.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
import gc
from pathlib import Path
import threading
import time
from typing import Any, TypeVar

import cv2
import numpy as np

from utils.helpers import get_memory_usage
from utils.logger import LoggerMixin

T = TypeVar("T")


@contextmanager
def timer_context(name: str = "operation") -> Generator[float, None, None]:
    """Context manager for measuring execution time.

    This context manager measures the time elapsed within the context block
    and provides the duration in seconds.

    Args:
        name: Name of the operation being measured (maintained for API compatibility).

    Yields:
        float: Elapsed time in seconds (returns 0.0 during execution,
               actual value is calculated in finally block).

    Example:
        >>> with timer_context("inference") as elapsed:
        ...     result = model.predict(frame)
        ...     # elapsed is 0.0 during execution
        >>> print(f"Inference took {elapsed:.3f}s")  # Actual elapsed time
    """
    start_time = time.perf_counter()
    try:
        yield 0.0
    finally:
        elapsed = time.perf_counter() - start_time


@contextmanager
def memory_tracker_context(name: str = "memory") -> Generator[dict[str, float], None, None]:
    """Context manager for monitoring memory usage.

    This context manager tracks memory usage before and after the context block,
    providing detailed memory statistics.

    Args:
        name: Name of the context for identification.

    Yields:
        dict: Initial memory statistics including:
            - start_memory_mb: Initial memory usage in MB
            - start_time: Start timestamp
            - name: Context name

    Example:
        >>> with memory_tracker_context("data_loading") as stats:
        ...     data = load_large_dataset()
        >>> print(f"Memory delta: {stats['memory_delta_mb']:.2f} MB")
    """
    start_memory = get_memory_usage()
    start_time = time.time()

    yield {
        "start_memory_mb": start_memory.get("rss_mb", 0),
        "start_time": start_time,
        "name": name,
    }

    end_memory = get_memory_usage()
    end_time = time.time()

    _stats = {
        "name": name,
        "duration_seconds": end_time - start_time,
        "start_memory_mb": start_memory.get("rss_mb", 0),
        "end_memory_mb": end_memory.get("rss_mb", 0),
        "memory_delta_mb": end_memory.get("rss_mb", 0) - start_memory.get("rss_mb", 0),
        "system_percent": end_memory.get("system_percent", 0),
    }


@contextmanager
def video_capture_context(source: str | int) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager for video capture with automatic resource cleanup.

    This context manager handles video capture initialization and ensures
    the capture is properly released when the context exits.

    Args:
        source: Video source (device number as string/int or file path).

    Yields:
        cv2.VideoCapture: Configured video capture object.

    Raises:
        RuntimeError: If the source cannot be opened.

    Example:
        >>> with video_capture_context("0") as cap:
        ...     ret, frame = cap.read()
        ...     process_frame(frame)
        >>> # Capture is automatically released
    """
    cap = None
    try:
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        yield cap
    finally:
        if cap is not None:
            cap.release()


@contextmanager
def image_window_context(window_name: str) -> Generator[None, None, None]:
    """Context manager for image windows with automatic cleanup.

    This context manager ensures that image windows are properly destroyed
    when the context exits, even if an exception occurs.

    Args:
        window_name: Name of the window to manage.

    Example:
        >>> with image_window_context("output"):
        ...     cv2.imshow("output", frame)
        ...     cv2.waitKey(1)
        >>> # Window is automatically destroyed
    """
    try:
        yield
    finally:
        with suppress(cv2.error):
            cv2.destroyWindow(window_name)


@contextmanager
def lock_context(lock: threading.Lock, timeout: float | None = None) -> Generator[bool, None, None]:
    """Context manager for thread locks with optional timeout.

    This context manager safely acquires and releases a lock, handling
    timeout scenarios gracefully.

    Args:
        lock: Lock object to acquire.
        timeout: Timeout in seconds (optional). If None, waits indefinitely.

    Yields:
        bool: True if the lock was acquired, False if timeout occurred.

    Example:
        >>> with lock_context(my_lock, timeout=5.0) as acquired:
        ...     if acquired:
        ...         # Critical section
        ...         shared_data.update()
        ...     else:
        ...         print("Could not acquire lock within timeout")
    """
    acquired = False
    try:
        if timeout is not None:
            acquired = lock.acquire(timeout=timeout)
        else:
            lock.acquire()
            acquired = True

        yield acquired
    finally:
        if acquired:
            lock.release()


@contextmanager
def file_context(
    filepath: str, mode: str = "r", encoding: str = "utf-8"
) -> Generator[Any, None, None]:
    """Context manager for file handling with automatic directory creation.

    This context manager handles file opening and ensures proper closing.
    It automatically creates parent directories when writing or appending.

    Args:
        filepath: Path to the file.
        mode: File opening mode ('r', 'w', 'a', etc.).
        encoding: File encoding (default: utf-8).

    Yields:
        File object: Open file handle.

    Example:
        >>> with file_context("data/output.json", "w") as f:
        ...     json.dump(data, f)
        >>> # File is automatically closed, directory created if needed
    """
    path = Path(filepath)
    if "w" in mode or "a" in mode:
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, mode, encoding=encoding) as f:
        yield f


@contextmanager
def gc_context(aggressive: bool = False) -> Generator[dict[str, int], None, None]:
    """Context manager for garbage collection control.

    This context manager temporarily disables garbage collection and
    optionally performs aggressive cleanup on exit.

    Args:
        aggressive: If True, performs multiple GC passes for thorough cleanup.

    Yields:
        dict: GC statistics before cleanup including:
            - garbage_count: Number of objects in garbage
            - gc_enabled: Whether GC was enabled

    Example:
        >>> with gc_context(aggressive=True) as stats:
        ...     # GC is disabled here for performance
        ...     process_large_data()
        >>> # GC is re-enabled and aggressive cleanup performed
    """
    gc.disable()
    stats_before = {
        "garbage_count": len(gc.garbage),
        "gc_enabled": gc.isenabled(),
    }

    try:
        yield stats_before
    finally:
        gc.enable()
        collected = gc.collect()
        if aggressive:
            for _ in range(3):
                gc.collect()


@contextmanager
def performance_context(name: str = "operation") -> Generator[dict[str, Any], None, None]:
    """Context manager for comprehensive performance measurement.

    This context manager measures both execution time and memory usage
    simultaneously, providing complete performance statistics.

    Args:
        name: Name of the operation being measured.

    Yields:
        dict: Performance statistics updated on exit including:
            - name: Operation name
            - start_time: Start timestamp
            - start_memory_mb: Initial memory usage
            - duration_ms: Duration in milliseconds
            - duration_seconds: Duration in seconds
            - memory_delta_mb: Memory change during execution
            - end_memory_mb: Final memory usage

    Example:
        >>> with performance_context("model_inference") as stats:
        ...     result = model.predict(frame)
        >>> print(f"Took {stats['duration_ms']:.2f}ms, used {stats['memory_delta_mb']:.2f}MB")
    """
    start_time = time.perf_counter()
    start_memory = get_memory_usage()

    stats = {
        "name": name,
        "start_time": start_time,
        "start_memory_mb": start_memory.get("rss_mb", 0),
    }

    yield stats

    end_time = time.perf_counter()
    end_memory = get_memory_usage()

    stats.update(
        {
            "duration_ms": (end_time - start_time) * 1000,
            "duration_seconds": end_time - start_time,
            "memory_delta_mb": end_memory.get("rss_mb", 0) - start_memory.get("rss_mb", 0),
            "end_memory_mb": end_memory.get("rss_mb", 0),
            "start_memory_mb": start_memory.get("rss_mb", 0),
        }
    )


class VideoCaptureContext(LoggerMixin):
    """Advanced context manager for video capture with automatic reconnection.

    This class provides robust video capture with:
        - Automatic reconnection on failure
        - Configurable resolution
        - Retry logic with delays
        - Proper resource cleanup
        - Detailed logging

    Attributes:
        source: Video source (device number or file path).
        width: Desired capture width (optional).
        height: Desired capture height (optional).
        reconnect_attempts: Number of reconnection attempts.
        reconnect_delay: Delay between attempts in seconds.
        cap: VideoCapture object (internal).
        _is_open: Boolean indicating if capture is open.

    Example:
        >>> with VideoCaptureContext("0", width=640, height=480) as cap:
        ...     ret, frame = cap.read()
        ...     while ret:
        ...         process_frame(frame)
        ...         ret, frame = cap.read()
        >>> # Capture is automatically closed
    """

    def __init__(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ) -> None:
        """Initializes the video capture context.

        Args:
            source: Video source (device number or file path).
            width: Desired capture width (optional).
            height: Desired capture height (optional).
            reconnect_attempts: Number of reconnection attempts.
            reconnect_delay: Delay between attempts in seconds.
        """
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap: cv2.VideoCapture | None = None
        self._is_open = False

        self.logger.info(
            "Initializing VideoCaptureContext", source=source, width=width, height=height
        )

    def __enter__(self) -> "VideoCaptureContext":
        """Opens the video capture."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the video capture."""
        self.close()

    def _open(self) -> None:
        """Opens the capture with retry logic."""
        for attempt in range(self.reconnect_attempts):
            try:
                if isinstance(self.source, str) and self.source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.source))
                else:
                    self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    self._configure_capture()
                    self._is_open = True
                    self.logger.info("Capture opened successfully", attempt=attempt + 1)
                    return

                self.logger.warning("Open attempt failed", attempt=attempt + 1)
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

            except Exception as e:
                self.logger.warning("Error opening capture", attempt=attempt + 1, error=str(e))
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

        raise RuntimeError(f"Could not open source after {self.reconnect_attempts} attempts")

    def _configure_capture(self) -> None:
        """Configures the capture with specified parameters."""
        if self.cap is None:
            return

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple:
        """Reads a frame from the capture.

        Returns:
            tuple: (ret, frame) where ret is a boolean indicating success,
                   and frame is the captured frame (None if failed).
        """
        if not self._is_open or self.cap is None:
            self.logger.warning("Attempted to read from closed capture")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.debug("Could not read frame")
            return ret, frame
        except Exception as e:
            self.logger.error("Error reading frame", error=str(e))
            return False, None

    def get_fps(self) -> float:
        """Gets the FPS of the capture.

        Returns:
            float: Frames per second, or 0.0 if capture is not open.
        """
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_size(self) -> tuple:
        """Gets the frame size of the capture.

        Returns:
            tuple: (width, height) in pixels, or (0, 0) if capture is not open.
        """
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        """Checks if the capture is currently open.

        Returns:
            bool: True if capture is open and functional, False otherwise.
        """
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def close(self) -> None:
        """Closes the capture and releases resources."""
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.debug("Capture released")
            except Exception as e:
                self.logger.warning("Error releasing capture", error=str(e))
            finally:
                self.cap = None
                self._is_open = False

    def __del__(self) -> None:
        """Cleanup when the object is destroyed."""
        self.close()


class ResourcePool(LoggerMixin):
    """Resource pool for reusing expensive objects.

    This class implements a pool pattern for managing reusable resources
    such as video captures, arrays, or other expensive objects.

    Features:
        - Thread-safe operations with locking
        - Configurable maximum pool size
        - Timeout on resource acquisition
        - Automatic validation of resources
        - Cleanup of invalid resources
        - Statistics and monitoring

    Attributes:
        max_size: Maximum number of resources per type.
        timeout: Timeout in seconds for lock acquisition.

    Example:
        >>> pool = ResourcePool(max_size=5)
        >>> # Get a resource
        >>> cap = pool.get("video", lambda: cv2.VideoCapture(0))
        >>> if cap:
        ...     use_resource(cap)
        ...     pool.release("video", cap)
    """

    def __init__(self, max_size: int = 10, timeout: float = 30.0) -> None:
        """Initializes the resource pool.

        Args:
            max_size: Maximum number of resources to keep per type.
            timeout: Timeout in seconds for lock acquisition.
        """
        self.max_size = max_size
        self.timeout = timeout
        self._pool: dict[str, list] = {}
        self._lock = threading.Lock()

        self.logger.info("ResourcePool initialized", max_size=max_size, timeout=timeout)

    def get(self, resource_type: str, creator: Callable[[], T]) -> T | None:
        """Gets a resource from the pool or creates a new one.

        This method attempts to reuse an existing resource from the pool.
        If no resource is available, it creates a new one using the creator function.

        Args:
            resource_type: Type identifier for the resource.
            creator: Function that creates a new resource.

        Returns:
            T | None: Resource object if successful, None on failure.

        Example:
            >>> capture = pool.get("camera", lambda: cv2.VideoCapture(0))
        """
        with lock_context(self._lock, timeout=self.timeout) as acquired:
            if not acquired:
                self.logger.warning("Timeout acquiring lock", resource_type=resource_type)
                return None

            if resource_type not in self._pool:
                self._pool[resource_type] = []

            pool = self._pool[resource_type]

            while pool:
                resource = pool.pop()
                if self._is_valid(resource):
                    self.logger.debug("Resource reused", resource_type=resource_type)
                    return resource

            try:
                resource = creator()
                self.logger.debug("Resource created", resource_type=resource_type)
                return resource
            except Exception as e:
                self.logger.error(
                    "Error creating resource", resource_type=resource_type, error=str(e)
                )
                return None

    def release(self, resource_type: str, resource: T) -> bool:
        """Releases a resource back to the pool.

        The resource is added back to the pool for future reuse if it's valid
        and the pool hasn't reached its maximum size.

        Args:
            resource_type: Type identifier for the resource.
            resource: Resource object to release.

        Returns:
            bool: True if the resource was successfully released, False otherwise.
        """
        with lock_context(self._lock, timeout=self.timeout) as acquired:
            if not acquired:
                self.logger.warning(
                    "Timeout acquiring lock for release", resource_type=resource_type
                )
                return False

            if resource_type not in self._pool:
                self._pool[resource_type] = []

            pool = self._pool[resource_type]

            if len(pool) < self.max_size and self._is_valid(resource):
                pool.append(resource)
                self.logger.debug("Resource released to pool", resource_type=resource_type)
                return True

            self._cleanup_resource(resource)
            return False

    def _is_valid(self, resource: Any) -> bool:
        """Checks if a resource is valid for reuse.

        Args:
            resource: Resource to validate.

        Returns:
            bool: True if the resource is valid, False otherwise.
        """
        if resource is None:
            return False

        if isinstance(resource, cv2.VideoCapture):
            return resource.isOpened()

        if isinstance(resource, np.ndarray):
            return resource.size > 0

        return True

    def _cleanup_resource(self, resource: Any) -> None:
        """Cleans up a resource before discarding it.

        Args:
            resource: Resource to clean up.
        """
        try:
            if isinstance(resource, cv2.VideoCapture):
                resource.release()
            elif isinstance(resource, np.ndarray):
                pass
        except Exception as e:
            self.logger.debug("Error cleaning up resource", error=str(e))

    def clear(self, resource_type: str | None = None) -> None:
        """Clears the pool of resources.

        Args:
            resource_type: Specific type to clear, or None to clear all.
        """
        with lock_context(self._lock) as acquired:
            if not acquired:
                return

            if resource_type is None:
                for rt in list(self._pool.keys()):
                    self._clear_pool(rt)
            elif resource_type in self._pool:
                self._clear_pool(resource_type)

    def _clear_pool(self, resource_type: str) -> None:
        """Clears a specific resource pool."""
        pool = self._pool.get(resource_type, [])
        for resource in pool:
            self._cleanup_resource(resource)
        pool.clear()
        self.logger.debug("Pool cleared", resource_type=resource_type)

    def stats(self) -> dict[str, Any]:
        """Gets statistics for the resource pool.

        Returns:
            dict: Statistics including:
                - total_types: Number of resource types
                - total_resources: Total resources in pool
                - resources_by_type: Resources by type
                - max_size: Maximum pool size

        Example:
            >>> stats = pool.stats()
            >>> print(f"Total resources: {stats['total_resources']}")
        """
        with lock_context(self._lock) as acquired:
            if not acquired:
                return {}

            return {
                "total_types": len(self._pool),
                "total_resources": sum(len(pool) for pool in self._pool.values()),
                "resources_by_type": {rt: len(pool) for rt, pool in self._pool.items()},
                "max_size": self.max_size,
            }
