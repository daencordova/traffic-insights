"""Cache for extracted features.

Implements an LRU cache with memory management to store
features and avoid redundant extractions.
"""

from collections import OrderedDict
import hashlib
import time
from typing import Any

import cv2
import numpy as np

from core.constants import MIN_CACHE_QUALITY
from utils.logger import LoggerMixin


class FeatureCacheEntry:
    """Optimized feature cache entry.

    This class represents a single entry in the feature cache,
    storing the feature vector along with metadata for LRU management.

    Attributes:
        features: Extracted feature vector.
        timestamp: Creation timestamp.
        confidence: Detection confidence.
        quality: Region quality score (0-1).
        access_count: Number of times accessed.

    Example:
        >>> entry = FeatureCacheEntry(features, confidence=0.9, quality=0.8)
        >>> entry.touch()  # Update access time
        >>> if entry.is_valid(max_age=3.0):
        ...     features = entry.features
        >>> score = entry.get_score()  # For eviction decisions
    """

    __slots__ = ("features", "timestamp", "confidence", "quality", "access_count")

    def __init__(
        self,
        features: np.ndarray,
        confidence: float,
        quality: float,
    ) -> None:
        """Initializes a cache entry.

        Args:
            features: Feature vector to store.
            confidence: Detection confidence (0-1).
            quality: Region quality score (0-1).
        """
        self.features = features.copy()
        self.timestamp = time.time()
        self.confidence = confidence
        self.quality = quality
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
            max_age: Maximum age in seconds before expiration.

        Returns:
            bool: True if the entry is valid and not expired.

        Example:
            >>> if entry.is_valid(max_age=5.0):
            ...     features = entry.features
        """
        return (time.time() - self.timestamp) < max_age

    def get_score(self) -> float:
        """Calculates a score for eviction decisions.

        The score combines access frequency, quality, and age
        to determine which entries are least useful.

        Returns:
            float: Score where higher = more useful.

        Note:
            Score components:
            - Access frequency (20% weight)
            - Quality score (40% weight)
            - Age recency (40% weight)
        """
        access_score = min(1.0, self.access_count / 10.0)
        quality_score = min(1.0, self.quality)
        age_score = min(1.0, (time.time() - self.timestamp) / 30.0)

        return 0.2 * access_score + 0.4 * quality_score + 0.4 * age_score


class FeatureCache(LoggerMixin):
    """LRU cache for extracted features.

    Features:
        - LRU (Least Recently Used) policy
        - Time-based expiration
        - Size limit
        - Usage statistics
        - Smart eviction based on quality and access

    Attributes:
        max_size: Maximum number of entries.
        max_age_seconds: Maximum lifetime of an entry.

    Example:
        >>> cache = FeatureCache(max_size=500, max_age_seconds=3.0)
        >>>
        >>> # Compute key for a region
        >>> key = cache.compute_key(image, bbox)
        >>>
        >>> # Try to get from cache
        >>> features = cache.get(key)
        >>> if features is None:
        ...     # Not in cache, extract features
        ...     features = extract_features(region)
        ...     # Store in cache
        ...     cache.put(key, features, confidence=0.9, quality=0.8)
        >>>
        >>> # Check statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        >>> print(f"Cache size: {stats['size']}/{stats['max_size']}")
    """

    __slots__ = (
        "max_size",
        "max_age_seconds",
        "_cache",
        "_hits",
        "_misses",
        "_evictions",
        "_last_cleanup",
        "_cleanup_interval",
    )

    def __init__(
        self,
        max_size: int = 500,
        max_age_seconds: float = 3.0,
    ) -> None:
        """Initializes the feature cache.

        Args:
            max_size: Maximum number of entries.
            max_age_seconds: Maximum age in seconds.

        Example:
            >>> # Large cache for long-term storage
            >>> cache = FeatureCache(max_size=1000, max_age_seconds=5.0)
            >>>
            >>> # Small cache for short-term storage
            >>> cache = FeatureCache(max_size=100, max_age_seconds=1.0)
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds

        self._cache: OrderedDict[str, FeatureCacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        self._last_cleanup = time.time()
        self._cleanup_interval = 5.0

        self.logger.info(
            "FeatureCache initialized",
            max_size=max_size,
            max_age_seconds=max_age_seconds,
        )

    def compute_key(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> str:
        """Computes a unique key for the region.

        Args:
            image: Full image.
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            str: MD5 hash of the resized region.

        Example:
            >>> key = cache.compute_key(frame, (100, 100, 200, 200))
            >>> print(f"Cache key: {key[:8]}...")
        """
        try:
            x1, y1, x2, y2 = bbox
            region = image[y1:y2, x1:x2]

            if region.size > 0:
                small = cv2.resize(region, (32, 32))
                return hashlib.md5(small.tobytes()).hexdigest()

        except Exception:
            pass

        return f"{int(time.time() * 1000)}"

    def get(self, key: str) -> np.ndarray | None:
        """Gets features from the cache.

        Args:
            key: Region key.

        Returns:
            Optional[np.ndarray]: Feature vector or None if not found.

        Example:
            >>> features = cache.get(key)
            >>> if features is not None:
            ...     print("Cache hit!")
            ... else:
            ...     print("Cache miss")
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

        return entry.features

    def put(
        self,
        key: str,
        features: np.ndarray,
        confidence: float,
        quality: float,
    ) -> None:
        """Stores features in the cache.

        Args:
            key: Region key.
            features: Feature vector.
            confidence: Detection confidence (0-1).
            quality: Region quality score (0-1).

        Example:
            >>> cache.put(key, features, confidence=0.95, quality=0.85)
            >>> print("Features cached")
        """
        if features is None or quality < MIN_CACHE_QUALITY:
            return

        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        entry = FeatureCacheEntry(features, confidence, quality)
        self._cache[key] = entry

        self._periodic_cleanup()

    def _remove(self, key: str) -> None:
        """Removes an entry from the cache."""
        if key in self._cache:
            del self._cache[key]
            self._evictions += 1

    def _evict_oldest(self) -> None:
        """Evicts the least useful entry based on score."""
        if not self._cache:
            return

        scores = {}
        for key, entry in self._cache.items():
            scores[key] = entry.get_score()

        worst_key = min(scores, key=scores.get)
        self._remove(worst_key)

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
            self.logger.debug(
                "Cleaned expired entries",
                count=len(expired_keys),
            )

    def clear(self) -> None:
        """Clears the entire cache.

        Example:
            >>> cache.clear()
            >>> # Cache is now empty
        """
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self.logger.info("Cache cleared", entries=count)

    def get_stats(self) -> dict[str, Any]:
        """Gets cache statistics.

        Returns:
            dict[str, Any]: Statistics including:
                - size: Current number of entries
                - max_size: Maximum size
                - hits: Number of cache hits
                - misses: Number of cache misses
                - evictions: Number of evictions
                - hit_rate: Cache hit rate (0-1)
                - max_age_seconds: Maximum age

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
            >>> print(f"Size: {stats['size']}/{stats['max_size']}")
            >>> print(f"Evictions: {stats['evictions']}")
        """
        total_requests = self._hits + self._misses

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
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
        """Cache hit rate (0-1)."""
        total = self._hits + self._misses
        return self._hits / max(1, total)

    def __len__(self) -> int:
        """Returns the number of entries in the cache."""
        return len(self._cache)
