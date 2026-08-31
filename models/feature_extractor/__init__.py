"""
Feature extraction module for re-identification.

Provides feature extractors with different backends and cache
management for vehicle re-identification.

This module enables robust feature extraction for tracking and
re-identification tasks, supporting multiple backend implementations
ranging from traditional computer vision to deep learning approaches.

Features:
    - Multiple feature extraction backends (SIFT, Histogram, ResNet)
    - Cache management for efficient feature storage
    - Feature validation and quality assessment
    - Factory pattern for easy backend selection
    - Consistent interface across all backends

Example:
    >>> from models.feature_extractor import FeatureExtractor, FeatureExtractorFactory
    >>>
    >>> # Create extractor with ResNet backend
    >>> extractor = FeatureExtractorFactory.create(backend_type="resnet", device="cuda")
    >>>
    >>> # Extract features from a vehicle region
    >>> region = frame[y1:y2, x1:x2]
    >>> features = extractor.extract(region)
    >>>
    >>> # Use cache for efficiency
    >>> cached_features = extractor.get_cached(region)
    >>> if cached_features is not None:
    ...     print("Using cached features")
    ... else:
    ...     features = extractor.extract(region)
    ...     extractor.cache_features(region, features)
    >>>
    >>> # Compare features for re-identification
    >>> similarity = extractor.compare_features(features1, features2)
    >>> print(f"Similarity score: {similarity:.3f}")
"""

from models.feature_extractor.backends import (
    FeatureBackend,
    HistogramBackend,
    ResNetBackend,
    SIFTBackend,
)
from models.feature_extractor.base import FeatureExtractor
from models.feature_extractor.cache import FeatureCache, FeatureCacheEntry
from models.feature_extractor.factory import FeatureExtractorFactory
from models.feature_extractor.validator import FeatureValidator

__all__ = [
    "FeatureExtractor",
    "FeatureCache",
    "FeatureCacheEntry",
    "FeatureValidator",
    "FeatureExtractorFactory",
    "FeatureBackend",
    "ResNetBackend",
    "HistogramBackend",
    "SIFTBackend",
]
