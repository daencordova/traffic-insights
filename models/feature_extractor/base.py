"""Main feature extractor.

Coordinates different backends and provides a unified interface
for feature extraction.
"""

from __future__ import annotations

from collections import deque
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import MIN_REGION_QUALITY
from models.feature_extractor.cache import FeatureCache
from models.feature_extractor.validator import FeatureValidator
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    from models.feature_extractor.backends.base import FeatureBackend


class FeatureExtractor(LoggerMixin):
    """Feature extractor for re-identification.

    Coordinates the backend, cache, and validator to extract
    features from image regions.

    This class provides a unified interface for feature extraction
    with caching, validation, and performance metrics.

    Attributes:
        backend: Feature extraction backend.
        cache: Feature cache.
        validator: Quality validator.
        feature_dim: Dimension of the feature vector.

    Example:
        >>> from models.feature_extractor import FeatureExtractor
        >>> from models.feature_extractor.backends import ResNetBackend
        >>>
        >>> backend = ResNetBackend(device="cuda")
        >>> extractor = FeatureExtractor(backend=backend, cache_size=500, feature_dim=2048)
        >>>
        >>> # Extract features from a region
        >>> features = extractor.extract_features(image, bbox, confidence=0.9)
        >>>
        >>> if features is not None:
        ...     print(f"Extracted {len(features)} features")
        >>>
        >>> # Compare two feature vectors
        >>> similarity = extractor.compare_features(features1, features2)
        >>> print(f"Similarity: {similarity:.3f}")
        >>>
        >>> # Get performance metrics
        >>> metrics = extractor.get_metrics()
        >>> print(f"Success rate: {metrics['success_rate']:.2%}")
        >>>
        >>> # Clear cache when done
        >>> extractor.clear_cache()
    """

    __slots__ = (
        "backend",
        "feature_dim",
        "cache",
        "validator",
        "_metrics",
    )

    def __init__(
        self,
        backend: FeatureBackend,
        cache_size: int = 500,
        feature_dim: int = 2048,
        max_age_seconds: float = 3.0,
    ) -> None:
        """Initializes the feature extractor.

        Args:
            backend: Feature extraction backend.
            cache_size: Cache size for features.
            feature_dim: Dimension of the feature vector.
            max_age_seconds: Maximum age of cache entries.

        Example:
            >>> # With Histogram backend
            >>> backend = HistogramBackend()
            >>> extractor = FeatureExtractor(backend=backend, cache_size=100, feature_dim=128)
        """
        self.backend = backend
        self.feature_dim = feature_dim
        self.cache = FeatureCache(
            max_size=cache_size,
            max_age_seconds=max_age_seconds,
        )
        self.validator = FeatureValidator()

        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cached_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "extraction_times": deque(maxlen=100),
        }

        try:
            self.backend.warmup()
        except Exception as e:
            self.logger.warning(f"Warmup error: {e}")

        self.logger.info(
            "FeatureExtractor initialized",
            backend=backend.name,
            backend_available=backend.is_available,
            feature_dim=feature_dim,
            cache_size=cache_size,
        )

    def extract_features(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float = 0.5,
        *,
        force: bool = False,
    ) -> np.ndarray | None:
        """Extracts features from an image region.

        This method handles the complete feature extraction pipeline:
            1. Input validation
            2. Cache lookup
            3. Region extraction
            4. Quality validation
            5. Feature extraction via backend
            6. Cache storage

        Args:
            image: Full image.
            bbox: Bounding box (x1, y1, x2, y2).
            confidence: Detection confidence (0-1).
            force: Force extraction even if quality is low.

        Returns:
            Optional[np.ndarray]: Feature vector or None if extraction fails.

        Example:
            >>> # Normal extraction with quality check
            >>> features = extractor.extract_features(image, bbox, confidence=0.8)
            >>>
            >>> # Force extraction regardless of quality
            >>> features = extractor.extract_features(image, bbox, confidence=0.5, force=True)
        """
        start_time = time.perf_counter()
        self._metrics["total_extractions"] += 1

        if not self._validate_input(image, bbox):
            self._metrics["failed_extractions"] += 1
            return None

        cached_result = self._check_cache(image, bbox, force)
        if cached_result is not None:
            return cached_result

        region = self._extract_region(image, bbox)
        quality_score = self.validator.validate_region(region)

        if not self._is_quality_sufficient(quality_score, force):
            self._metrics["failed_extractions"] += 1
            self.logger.debug(
                "Low quality region",
                quality=f"{quality_score:.2f}",
                bbox=bbox,
            )
            return None

        features = self.backend.extract(region)

        if features is None:
            self._metrics["failed_extractions"] += 1
            return None

        self._cache_if_valid(bbox, features, confidence, quality_score, force)

        self._update_extraction_metrics(start_time)

        return features

    def _validate_input(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Validates input for feature extraction.

        Args:
            image: Image to validate.
            bbox: Bounding box to validate.

        Returns:
            bool: True if input is valid.
        """
        if image is None or image.size == 0:
            return False

        return self.validator.validate_bbox(bbox, image.shape)

    def _check_cache(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        force: bool,
    ) -> np.ndarray | None:
        """Checks if features are in cache.

        Args:
            image: Full image.
            bbox: Bounding box.
            force: Whether force extraction is enabled.

        Returns:
            Optional[np.ndarray]: Cached features or None.
        """
        if force:
            return None

        cache_key = self.cache.compute_key(image, bbox)
        cached = self.cache.get(cache_key)

        if cached is not None:
            self._metrics["cached_extractions"] += 1
            self._metrics["successful_extractions"] += 1
            return cached

        return None

    def _extract_region(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extracts the region from the image based on bounding box."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def _is_quality_sufficient(self, quality_score: float, force: bool) -> bool:
        """Checks if region quality is sufficient.

        Args:
            quality_score: Quality score (0-1).
            force: Whether force extraction is enabled.

        Returns:
            bool: True if quality is sufficient or force is enabled.
        """
        return force or quality_score >= MIN_REGION_QUALITY

    def _cache_if_valid(
        self,
        bbox: tuple[int, int, int, int],
        features: np.ndarray,
        confidence: float,
        quality_score: float,
        force: bool,
    ) -> None:
        """Stores in cache if conditions are favorable.

        Args:
            bbox: Bounding box.
            features: Extracted features.
            confidence: Detection confidence.
            quality_score: Quality score.
            force: Whether force extraction was used.
        """
        if force or quality_score < MIN_REGION_QUALITY:
            return

        cache_key = self.cache.compute_key(bbox, features)
        self.cache.put(cache_key, features, confidence, quality_score)

    def _update_extraction_metrics(self, start_time: float) -> None:
        """Updates extraction metrics."""
        self._metrics["successful_extractions"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._metrics["extraction_times"].append(elapsed_ms)
        self._metrics["avg_extraction_time_ms"] = sum(self._metrics["extraction_times"]) / len(
            self._metrics["extraction_times"]
        )

    def compare_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        method: str = "cosine",
    ) -> float:
        """Compares two feature vectors.

        Args:
            features1: First feature vector.
            features2: Second feature vector.
            method: Comparison method ('cosine', 'euclidean', 'dot').

        Returns:
            float: Similarity score (0-1 for cosine, distances for others).

        Example:
            >>> # Cosine similarity (recommended)
            >>> similarity = extractor.compare_features(f1, f2)
            >>>
            >>> # Euclidean distance
            >>> similarity = extractor.compare_features(f1, f2, "euclidean")
            >>>
            >>> # Dot product
            >>> similarity = extractor.compare_features(f1, f2, "dot")
        """
        if features1 is None or features2 is None:
            return 0.0

        try:
            if method == "cosine":
                return self._cosine_similarity(features1, features2)
            if method == "euclidean":
                return self._euclidean_similarity(features1, features2)
            if method == "dot":
                return float(np.dot(features1, features2))
            self.logger.warning(f"Unsupported method: {method}")
            return self.compare_features(features1, features2, "cosine")

        except Exception as e:
            self.logger.debug(f"Error comparing features: {e}")
            return 0.0

    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(f1, f2) / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))

    def _euclidean_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculates similarity based on Euclidean distance."""
        dist = np.linalg.norm(f1 - f2)
        return 1.0 / (1.0 + dist)

    def clear_cache(self) -> None:
        """Clears the feature cache."""
        self.cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Gets cache statistics."""
        return self.cache.get_stats()

    def get_metrics(self) -> dict[str, Any]:
        """Gets performance metrics.

        Returns:
            dict[str, Any]: Metrics including:
                - total_extractions: Total extraction attempts
                - successful_extractions: Successful extractions
                - failed_extractions: Failed extractions
                - cached_extractions: Cache hits
                - avg_extraction_time_ms: Average extraction time
                - success_rate: Success rate
                - backend: Backend name
                - backend_available: Backend availability
                - cache: Cache statistics
                - validator: Validator statistics
                - feature_dim: Feature dimension

        Example:
            >>> metrics = extractor.get_metrics()
            >>> print(f"Success rate: {metrics['success_rate']:.2%}")
            >>> print(f"Avg time: {metrics['avg_extraction_time_ms']:.2f}ms")
            >>> print(f"Cache hits: {metrics['cached_extractions']}")
        """
        total = self._metrics["total_extractions"]
        success = self._metrics["successful_extractions"]

        return {
            **self._metrics,
            "success_rate": success / max(1, total),
            "backend": self.backend.name,
            "backend_available": self.backend.is_available,
            "cache": self.cache.get_stats(),
            "validator": self.validator.get_stats(),
            "feature_dim": self.feature_dim,
        }

    def reset_metrics(self) -> None:
        """Resets all metrics.

        Example:
            >>> extractor.reset_metrics()
            >>> # All metrics are reset to zero
        """
        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cached_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "extraction_times": deque(maxlen=100),
        }
        self.validator.reset_stats()

    @property
    def is_available(self) -> bool:
        """Checks if the extractor is available."""
        return self.backend.is_available

    @property
    def feature_dimension(self) -> int:
        """Dimension of the feature vector."""
        return self.feature_dim

    def __enter__(self) -> FeatureExtractor:
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the context manager and clears cache."""
        self.clear_cache()
