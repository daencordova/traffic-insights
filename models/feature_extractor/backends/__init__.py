"""
Backends para extracción de features.
"""

from .base import FeatureBackend
from .resnet import ResNetBackend
from .histogram import HistogramBackend
from .sift import SIFTBackend

__all__ = [
    "FeatureBackend",
    "ResNetBackend",
    "HistogramBackend",
    "SIFTBackend",
]
