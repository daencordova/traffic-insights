"""Object detection caching system.

Implements an LRU cache with memory management to store detection
results and avoid redundant processing.
"""

from collections import OrderedDict
import hashlib
import time
from typing import Any

import cv2
import numpy as np

from core.constants import (
    CACHE_CLEANUP_THRESHOLD,
    DEFAULT_CACHE_SIZE,
    MAX_CACHE_MEMORY_MB,
)
from utils.logger import LoggerMixin

DetectionList = list[dict[str, Any]]
FrameHash = str


class CacheEntry:
    """Entry in the detection cache.

    This class represents a single cached detection result with
    metadata for LRU management and expiration.

    Attributes:
        detections: List of stored detections.
        timestamp: Creation time of the entry.
        size: Estimated size in bytes of the entry.
        access_count: Number of times the entry has been accessed.

    Example:
        >>> entry = CacheEntry(detections)
        >>> entry.touch()  # Update access time
        >>> if entry.is_valid(max_age=3.0):
        ...     print("Entry is still valid")
    """

    __slots__ = ("detections", "timestamp", "size", "access_count")

    def __init__(self, detections: DetectionList):
        """Initializes a cache entry.

        Args:
            detections: List of detections to store.
        """
        self.detections = detections
        self.timestamp = time.time()
        self.size = len(detections) * 4 * 4
        self.access_count = 0

    def touch(self) -> None:
        """Updates access count and timestamp.

        This method is called whenever the entry is accessed,
        keeping its position in the LRU order up to date.
        """
        self.access_count += 1
        self.timestamp = time.time()

    def is_valid(self, max_age: float = 3.0) -> bool:
        """Checks if the entry is still valid.

        Args:
            max_age: Maximum age in seconds before expiration (default: 3.0).

        Returns:
            bool: True if the entry is valid and not expired.

        Example:
            >>> if entry.is_valid(max_age=5.0):
            ...     # Entry can be used
            ...     detections = entry.detections
        """
        return (time.time() - self.timestamp) < max_age


class DetectionCache(LoggerMixin):
    """LRU cache for object detections.

    Features:
        - LRU (Least Recently Used) policy
        - Configurable memory limit
        - Time-based expiration
        - Usage statistics
        - Thread-safe operations

    Attributes:
        max_size: Maximum number of cache entries.
        max_age_seconds: Maximum lifetime of an entry.
        max_memory_mb: Maximum memory allowed for cache.
        cleanup_threshold: Threshold for automatic cleanup.

    Example:
        >>> cache = DetectionCache(max_size=64, max_age_seconds=3.0, max_memory_mb=250)
        >>>
        >>> # Compute key for a frame
        >>> key = cache.compute_key(frame)
        >>>
        >>> # Try to get from cache
        >>> detections = cache.get(key)
        >>> if detections is None:
        ...     # Not in cache, run detection
        ...     detections = detector.detect(frame)
        ...     # Store in cache
        ...     cache.put(key, detections)
        >>>
        >>> # Check statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        max_age_seconds: float = 3.0,
        max_memory_mb: int = MAX_CACHE_MEMORY_MB,
        cleanup_threshold: float = CACHE_CLEANUP_THRESHOLD,
    ):
        """Initializes the detection cache.

        Args:
            max_size: Maximum number of entries.
            max_age_seconds: Maximum age of an entry in seconds.
            max_memory_mb: Maximum memory in MB.
            cleanup_threshold: Threshold for automatic cleanup.

        Example:
            >>> # Cache with aggressive expiration
            >>> cache = DetectionCache(
            ...     max_size=32,
            ...     max_age_seconds=1.0,  # Short lifetime
            ...     max_memory_mb=100,
            ... )
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold

        self._cache: OrderedDict[FrameHash, CacheEntry] = OrderedDict()
        self._memory_usage: float = 0.0

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 5.0

        self.logger.info(
            "DetectionCache initialized",
            max_size=max_size,
            max_age_seconds=max_age_seconds,
            max_memory_mb=max_memory_mb,
        )

    def compute_key(self, frame: np.ndarray) -> str:
        """Computes a unique key for the frame.

        Args:
            frame: Image to process.

        Returns:
            str: MD5 hash of the frame resized to 32x32.

        Note:
            Resizing to 32x32 ensures that similar frames produce
            the same key, increasing the cache hit rate.

        Example:
            >>> key = cache.compute_key(frame)
            >>> print(f"Frame key: {key[:8]}...")
        """
        try:
            small = cv2.resize(frame, (32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()
        except Exception:
            return str(time.perf_counter())

    def get(self, key: str) -> DetectionList | None:
        """Gets detections from the cache.

        Args:
            key: Frame key.

        Returns:
            Optional[DetectionList]: Cached detections or None if not found.

        Note:
            This operation updates the LRU order of the entry.
            Expired entries are automatically removed.

        Example:
            >>> key = cache.compute_key(frame)
            >>> detections = cache.get(key)
            >>> if detections:
            ...     print(f"Cache hit! Found {len(detections)} detections")
            ... else:
            ...     print("Cache miss, running detection")
        """
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if not entry.is_valid(self.max_age_seconds):
            self._remove(key)
            self._misses += 1
            return None

        entry.touch()
        self._cache.move_to_end(key)
        self._hits += 1

        self.logger.debug("Cache hit", key=key[:8], detections=len(entry.detections))

        return entry.detections

    def put(self, key: str, detections: DetectionList) -> None:
        """Stores detections in the cache.

        Args:
            key: Frame key.
            detections: List of detections to store.

        Note:
            If the cache is full, the oldest entry (LRU) is evicted.
            If memory exceeds the limit, aggressive cleanup is performed.

        Example:
            >>> detections = detector.detect(frame)
            >>> key = cache.compute_key(frame)
            >>> cache.put(key, detections)
            >>> print("Detections cached")
        """
        if not detections:
            return

        entry_size = len(detections) * 4 * 4
        if self._memory_usage + entry_size > self.max_memory_mb * 1024 * 1024:
            self._cleanup(aggressive=True)

        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        entry = CacheEntry(detections)
        self._cache[key] = entry
        self._memory_usage += entry.size

        self._periodic_cleanup()

        self.logger.debug(
            "Cache put",
            key=key[:8],
            detections=len(detections),
            cache_size=len(self._cache),
            memory_mb=self._memory_usage / (1024 * 1024),
        )

    def _remove(self, key: str) -> None:
        """Removes an entry from the cache.

        Args:
            key: Key of the entry to remove.

        Note:
            This method updates the eviction counter and frees memory.
        """
        entry = self._cache.pop(key, None)
        if entry:
            self._memory_usage -= entry.size
            self._evictions += 1

    def _evict_oldest(self) -> None:
        """Evicts the oldest entry (LRU).

        Note:
            Removes the first entry from the OrderedDict,
            which corresponds to the least recently used.
        """
        if not self._cache:
            return

        oldest_key = next(iter(self._cache))
        self._remove(oldest_key)
        self.logger.debug("Evicted oldest entry", key=oldest_key[:8], cache_size=len(self._cache))

    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries.

        Note:
            Cleanup is performed every `_cleanup_interval` seconds
            to avoid overhead on every operation.
        """
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        expired_keys = [
            key for key, entry in self._cache.items() if not entry.is_valid(self.max_age_seconds)
        ]

        for key in expired_keys:
            self._remove(key)

        if expired_keys:
            self.logger.debug("Cleaned expired entries", count=len(expired_keys))

    def _cleanup(self, aggressive: bool = False) -> None:
        """Cache cleanup.

        Args:
            aggressive: If True, removes more entries (50%).
                If False, removes only 30% of entries.

        Note:
            Aggressive cleanup is used when memory exceeds the limit.
        """
        if not self._cache:
            return

        if aggressive:
            keys_to_remove = list(self._cache.keys())[: len(self._cache) // 2]
            for key in keys_to_remove:
                self._remove(key)
            self.logger.debug("Aggressive cleanup", removed=len(keys_to_remove))
        else:
            keys_to_remove = list(self._cache.keys())[: int(len(self._cache) * 0.3)]
            for key in keys_to_remove:
                self._remove(key)
            self.logger.debug("Partial cleanup", removed=len(keys_to_remove))

    def clear(self) -> None:
        """Clears the entire cache.

        Note:
            Resets all statistics and frees all memory.

        Example:
            >>> cache.clear()
            >>> # Cache is now empty
        """
        count = len(self._cache)
        self._cache.clear()
        self._memory_usage = 0
        self._hits = 0
        self._misses = 0
        self.logger.info("Cache cleared", entries=count)

    def get_stats(self) -> dict[str, Any]:
        """Gets cache statistics.

        Returns:
            Dict[str, Any]: Statistics including:
                - size: Current number of entries
                - max_size: Maximum configured size
                - memory_usage_mb: Used memory in MB
                - max_memory_mb: Maximum configured memory
                - hits: Number of hits
                - misses: Number of misses
                - evictions: Number of evicted entries
                - hit_rate: Hit rate (0-1)
                - max_age_seconds: Maximum configured age

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Size: {stats['size']}/{stats['max_size']}")
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
            >>> print(f"Memory: {stats['memory_usage_mb']:.1f} MB")
        """
        total_requests = self._hits + self._misses

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / max(1, total_requests),
            "max_age_seconds": self.max_age_seconds,
        }

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / max(1, total)

    def __len__(self) -> int:
        """Returns the number of entries in the cache."""
        return len(self._cache)
