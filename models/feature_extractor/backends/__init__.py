"""
Backends for feature extraction.

This module provides various feature extraction backends for
visual feature extraction used in re-identification and tracking.

Available backends:
    - FeatureBackend: Base interface for all feature backends
    - ResNetBackend: Deep learning-based features using ResNet
    - HistogramBackend: Traditional histogram-based features
    - SIFTBackend: SIFT (Scale-Invariant Feature Transform) features

Example:
    >>> from models.feature_extractor.backends import ResNetBackend, SIFTBackend
    >>>
    >>> # Create a ResNet backend
    >>> resnet = ResNetBackend(model_name="resnet18", device="cpu")
    >>> features = resnet.extract(frame)
    >>>
    >>> # Create a SIFT backend
    >>> sift = SIFTBackend(n_features=500)
    >>> features = sift.extract(frame)
    >>>
    >>> # Switch backends
    >>> backend = FeatureBackend.create("resnet", device="cuda")
"""

from models.feature_extractor.backends.base import FeatureBackend
from models.feature_extractor.backends.histogram import HistogramBackend
from models.feature_extractor.backends.resnet import ResNetBackend
from models.feature_extractor.backends.sift import SIFTBackend

__all__ = [
    "FeatureBackend",
    "ResNetBackend",
    "HistogramBackend",
    "SIFTBackend",
]
