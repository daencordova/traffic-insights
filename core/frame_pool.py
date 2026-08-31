"""Frame pool for optimized memory preallocation.

This module provides a pool of preallocated frames to avoid repeated
memory allocations in real-time video processing pipelines.

Features:
    - Preallocated memory to avoid reallocations
    - Thread-safe for use in async pipelines
    - Usage statistics
    - Automatic cleanup
    - Immediate memory release
    - Dynamic resizing capability
"""

import gc
import threading

import numpy as np

from utils.logger import LoggerMixin


class FramePool(LoggerMixin):
    """Frame pool for optimized memory reuse.

    This class manages a pool of preallocated numpy arrays (frames) to
    avoid the overhead of repeated memory allocation in real-time
    video processing pipelines.

    Features:
        - Preallocated memory to avoid reallocations
        - Thread-safe for use in async pipelines
        - Usage statistics for monitoring
        - Automatic cleanup on release
        - Immediate memory release
        - Dynamic resizing

    Attributes:
        pool_size: Number of frames in the pool.
        frame_shape: Shape of frames (height, width, channels).
        dtype: Data type of frames.

    Example:
        >>> pool = FramePool(pool_size=5, frame_shape=(480, 640, 3), dtype=np.uint8)
        >>>
        >>> # Acquire a frame
        >>> frame = pool.acquire()
        >>>
        >>> # Use the frame
        >>> process_frame(frame)
        >>>
        >>> # Release back to pool
        >>> pool.release(frame)
        >>>
        >>> # Check statistics
        >>> stats = pool.get_stats()
        >>> print(f"Pool hits: {stats['pool_hits']}")
    """

    def __init__(
        self,
        pool_size: int = 3,
        frame_shape: tuple[int, int, int] = (480, 640, 3),
        dtype: np.dtype = np.uint8,
    ) -> None:
        """Initializes the frame pool.

        Args:
            pool_size: Number of frames in the pool (kept small to save memory).
            frame_shape: Shape of frames (height, width, channels).
            dtype: Data type of frames.

        Example:
            >>> pool = FramePool(pool_size=3, frame_shape=(720, 1280, 3), dtype=np.float32)
        """
        self.pool_size = pool_size
        self.frame_shape = frame_shape
        self.dtype = dtype

        self._pool: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._idx = 0

        self._stats = {
            "total_allocated": pool_size,
            "total_acquired": 0,
            "total_released": 0,
            "current_used": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "memory_used_mb": 0,
        }

        self._preallocate()

        self.logger.info(
            "FramePool initialized", pool_size=pool_size, frame_shape=frame_shape, dtype=str(dtype)
        )

    def _preallocate(self) -> None:
        """Preallocates all frames in the pool."""
        for _ in range(self.pool_size):
            frame = np.zeros(self.frame_shape, dtype=self.dtype)
            self._pool.append(frame)
            self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)

    def acquire(self) -> np.ndarray:
        """Acquires a frame from the pool.

        This method retrieves a frame from the pool. If the pool is empty,
        it creates a new frame on demand.

        Returns:
            np.ndarray: Frame from the pool (zeroed and ready for use).

        Example:
            >>> frame = pool.acquire()
            >>> # Frame is now ready for use
            >>> frame[:, :, :] = some_data
        """
        with self._lock:
            if not self._pool:
                self.logger.warning("Pool empty, creating new frame")
                frame = np.zeros(self.frame_shape, dtype=self.dtype)
                self._stats["total_allocated"] += 1
                self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)
                self._stats["pool_misses"] += 1
                return frame

            frame = self._pool.pop(0)
            self._stats["total_acquired"] += 1
            self._stats["current_used"] += 1
            self._stats["pool_hits"] += 1

            return frame

    def release(self, frame: np.ndarray) -> bool:
        """Releases a frame back to the pool.

        This method returns a frame to the pool for reuse. The frame is
        zeroed before being stored to prevent data leakage.

        Args:
            frame: Frame to release.

        Returns:
            bool: True if the frame was successfully released, False otherwise.

        Example:
            >>> if pool.release(frame):
            ...     print("Frame returned to pool")
            ... else:
            ...     print("Frame was discarded (pool full)")
        """
        with self._lock:
            if frame is None or frame.size == 0:
                return False

            if len(self._pool) >= self.pool_size:
                self.logger.debug("Pool full, discarding frame")
                frame.fill(0)
                return False

            frame.fill(0)
            self._pool.append(frame)
            self._stats["total_released"] += 1
            self._stats["current_used"] -= 1

            return True

    def get_stats(self) -> dict:
        """Gets statistics for the frame pool.

        Returns:
            dict: Dictionary with pool statistics including:
                - total_allocated: Total frames allocated
                - total_acquired: Total frames acquired
                - total_released: Total frames released
                - current_used: Currently in use
                - pool_hits: Successful pool acquisitions
                - pool_misses: Acquisitions that required new allocation
                - memory_used_mb: Current memory usage in MB
                - pool_size: Maximum pool size
                - available: Available frames in pool
                - frame_shape: Shape of frames

        Example:
            >>> stats = pool.get_stats()
            >>> print(f"Hit rate: {stats['pool_hits'] / (stats['total_acquired'] or 1):.2%}")
            >>> print(f"Memory used: {stats['memory_used_mb']:.1f} MB")
        """
        with self._lock:
            return {
                **self._stats,
                "pool_size": self.pool_size,
                "available": len(self._pool),
                "frame_shape": self.frame_shape,
                "memory_used_mb": self._stats["memory_used_mb"],
            }

    def clear(self) -> None:
        """Clears the pool and releases memory.

        This method removes all frames from the pool and forces
        garbage collection to reclaim memory.

        Example:
            >>> pool.clear()
            >>> # All pool memory has been freed
        """
        with self._lock:
            for frame in self._pool:
                frame.fill(0)
            self._pool.clear()
            self._stats["total_allocated"] = 0
            self._stats["current_used"] = 0
            self._stats["memory_used_mb"] = 0

        gc.collect()
        self.logger.info("FramePool cleared and memory freed")

    def resize(self, new_size: int) -> None:
        """Resizes the pool to a new size.

        This method changes the pool size, either adding new frames
        or removing existing ones.

        Args:
            new_size: New pool size (must be >= 0).

        Example:
            >>> pool.resize(10)  # Increase pool to 10 frames
            >>> pool.resize(2)  # Reduce pool to 2 frames
        """
        with self._lock:
            current_size = len(self._pool)

            if new_size > current_size:
                for _ in range(new_size - current_size):
                    frame = np.zeros(self.frame_shape, dtype=self.dtype)
                    self._pool.append(frame)
                    self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)
            elif new_size < current_size:
                for _ in range(current_size - new_size):
                    if self._pool:
                        frame = self._pool.pop()
                        frame.fill(0)
                        self._stats["memory_used_mb"] -= frame.nbytes / (1024 * 1024)

            self.pool_size = new_size
            self._stats["total_allocated"] = new_size

        gc.collect()
        self.logger.info(
            "FramePool resized", new_size=new_size, current_used=self._stats["current_used"]
        )

    def __len__(self) -> int:
        """Returns the number of available frames in the pool.

        Returns:
            int: Number of frames currently available.

        Example:
            >>> available = len(pool)
            >>> if available > 0:
            ...     frame = pool.acquire()
        """
        with self._lock:
            return len(self._pool)
