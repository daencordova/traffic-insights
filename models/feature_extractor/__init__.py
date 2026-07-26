"""
Módulo de extracción de features para re-identificación.

Proporciona extractores de features con diferentes backends
y gestión de caché para re-identificación de vehículos.
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
