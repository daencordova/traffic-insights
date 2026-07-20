"""
Backends para extracción de features.
"""

from models.feature_extractor.backends.base import FeatureBackend
from models.feature_extractor.backends.resnet import ResNetBackend
from models.feature_extractor.backends.histogram import HistogramBackend
from models.feature_extractor.backends.sift import SIFTBackend

__all__ = [
    "FeatureBackend",
    "ResNetBackend",
    "HistogramBackend",
    "SIFTBackend",
]
