"""
Módulo de extracción de features para re-identificación.

Proporciona extractores de features con diferentes backends
y gestión de caché para re-identificación de vehículos.
"""

from .base import FeatureExtractor
from .cache import FeatureCache, FeatureCacheEntry
from .validator import FeatureValidator
from .factory import FeatureExtractorFactory
from .backends import (
    FeatureBackend,
    ResNetBackend,
    HistogramBackend,
    SIFTBackend,
)

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
