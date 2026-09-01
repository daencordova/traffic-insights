"""Resource management with context managers and pool.

This module provides generic resource management with lifecycle
control, cleanup, and context manager support.

Features:
    - Generic resource manager with configurable lifetime
    - Automatic cleanup of expired resources
    - Thread-safe operations with locking
    - Context manager for managed resources
    - Resource statistics and monitoring

Example:
    >>> from utils.resource_manager import ResourceManager, managed_resource
    >>>
    >>> # Create a resource manager with 30-second lifetime
    >>> manager = ResourceManager(max_lifetime=30.0)
    >>>
    >>> # Register a resource
    >>> manager.register("camera_1", camera_object)
    >>>
    >>> # Get a resource
    >>> camera = manager.get("camera_1")
    >>> if camera:
    ...     frame = camera.read()
    >>>
    >>> # Use context manager for automatic cleanup
    >>> with managed_resource(manager, "model", model_object) as model:
    ...     result = model.predict(frame)
    >>> # Resource is automatically cleaned up
    >>>
    >>> # Check statistics
    >>> stats = manager.stats()
    >>> print(f"Total resources: {stats['total_resources']}")
    >>> print(f"Keys: {stats['keys']}")
"""

from collections.abc import Generator
from contextlib import contextmanager
import threading
import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ResourceManager(Generic[T]):
    """Generic resource manager with lifetime limit.

    This class manages resources with configurable maximum lifetime,
    automatic cleanup, and thread-safe operations.

    Features:
        - Configurable resource lifetime
        - Automatic cleanup of expired resources
        - Thread-safe with locking
        - Resource registration and retrieval
        - Statistics and monitoring

    Attributes:
        max_lifetime: Maximum lifetime of a resource in seconds.
        cleanup_interval: Interval between cleanups in seconds.

    Example:
        >>> manager = ResourceManager(max_lifetime=60.0, cleanup_interval=10.0)
        >>>
        >>> # Register a video capture resource
        >>> manager.register("camera_0", cv2.VideoCapture(0))
        >>>
        >>> # Get the resource
        >>> cap = manager.get("camera_0")
        >>> if cap and cap.isOpened():
        ...     ret, frame = cap.read()
        >>>
        >>> # Resources are automatically cleaned up
        >>> # after max_lifetime seconds
    """

    def __init__(self, max_lifetime: float = 60.0, cleanup_interval: float = 10.0):
        """Initializes the resource manager.

        Args:
            max_lifetime: Maximum lifetime of a resource in seconds.
            cleanup_interval: Interval between cleanups in seconds.

        Example:
            >>> # Short-lived resources
            >>> manager = ResourceManager(max_lifetime=10.0, cleanup_interval=5.0)
            >>>
            >>> # Long-lived resources
            >>> manager = ResourceManager(max_lifetime=300.0, cleanup_interval=60.0)
        """
        self.max_lifetime = max_lifetime
        self.cleanup_interval = cleanup_interval
        self._resources: dict[str, T] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def register(self, key: str, resource: T) -> None:
        """Registers a resource with its timestamp.

        Args:
            key: Unique key for the resource.
            resource: Resource to register.

        Example:
            >>> manager.register("model_1", model)
            >>> manager.register("camera", cv2.VideoCapture(0))
        """
        with self._lock:
            self._resources[key] = resource
            self._timestamps[key] = time.time()

    def get(self, key: str) -> T | None:
        """Gets a resource by its key.

        Args:
            key: Resource key.

        Returns:
            Resource or None if not found or expired.

        Example:
            >>> model = manager.get("model_1")
            >>> if model:
            ...     result = model.predict(data)
            ... else:
            ...     print("Resource not available")
        """
        with self._lock:
            if key not in self._resources:
                return None

            if self._is_expired(key):
                self._remove(key)
                return None

            return self._resources[key]

    def remove(self, key: str) -> bool:
        """Removes a resource.

        Args:
            key: Resource key.

        Returns:
            bool: True if removed successfully.

        Example:
            >>> if manager.remove("camera"):
            ...     print("Camera resource removed")
        """
        with self._lock:
            return self._remove(key)

    def _remove(self, key: str) -> bool:
        """Internal method to remove a resource."""
        if key in self._resources:
            resource = self._resources.pop(key)
            self._timestamps.pop(key, None)
            self._cleanup_resource(resource)
            return True
        return False

    def _is_expired(self, key: str) -> bool:
        """Checks if a resource has expired."""
        if key not in self._timestamps:
            return True

        age = time.time() - self._timestamps[key]
        return age > self.max_lifetime

    def _cleanup_resource(self, resource: T) -> None:
        """Cleans up a specific resource."""
        try:
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "release"):
                resource.release()
            elif hasattr(resource, "clear"):
                resource.clear()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Cleans up expired resources.

        This method is called automatically but can also be called
        manually to force cleanup.

        Example:
            >>> manager.cleanup()  # Force cleanup of expired resources
        """
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            expired_keys = [key for key in self._resources if self._is_expired(key)]

            for key in expired_keys:
                self._remove(key)

            self._last_cleanup = current_time

    def stats(self) -> dict[str, Any]:
        """Gets manager statistics.

        Returns:
            dict: Statistics including:
                - total_resources: Number of active resources
                - keys: List of resource keys
                - max_lifetime_seconds: Maximum lifetime setting

        Example:
            >>> stats = manager.stats()
            >>> print(f"Active resources: {stats['total_resources']}")
            >>> print(f"Resource keys: {stats['keys']}")
        """
        with self._lock:
            return {
                "total_resources": len(self._resources),
                "keys": list(self._resources.keys()),
                "max_lifetime_seconds": self.max_lifetime,
            }


@contextmanager
def managed_resource(manager: ResourceManager, key: str, resource: T) -> Generator[T, None, None]:
    """Context manager for managed resources.

    This context manager automatically registers a resource and
    ensures it is removed when the context exits.

    Args:
        manager: Resource manager instance.
        key: Resource key.
        resource: Resource to manage.

    Yields:
        T: Managed resource.

    Example:
        >>> with managed_resource(manager, "connection", db_connection) as conn:
        ...     data = conn.query("SELECT * FROM table")
        >>> # Resource is automatically removed and cleaned up
        >>>
        >>> # With custom cleanup
        >>> class MyResource:
        ...     def close(self):
        ...         print("Cleaning up...")
        >>>
        >>> with managed_resource(manager, "my_res", MyResource()) as res:
        ...     res.process()
        >>> # close() is called automatically
    """
    try:
        manager.register(key, resource)
        yield resource
    finally:
        manager.remove(key)
