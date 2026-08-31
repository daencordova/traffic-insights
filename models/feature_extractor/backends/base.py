"""Abstract interface for feature extraction backends.

This module defines the base interface that all feature extraction
backends must implement to be used by the FeatureExtractor.

Example:
    >>> from models.feature_extractor.backends import FeatureBackend
    >>>
    >>> class MyCustomBackend(FeatureBackend):
    ...     def extract(self, region: np.ndarray) -> np.ndarray | None:
    ...         # Custom feature extraction logic
    ...         return features
    ...
    ...     def warmup(self) -> None:
    ...         # Warmup logic
    ...         pass
    ...
    ...     @property
    ...     def feature_dim(self) -> int:
    ...         return 128
    ...
    ...     @property
    ...     def is_available(self) -> bool:
    ...         return True
"""

from abc import ABC, abstractmethod

import numpy as np


class FeatureBackend(ABC):
    """Abstract interface for feature extraction backends.

    All backends must implement this interface to be used by the
    FeatureExtractor. This ensures consistent feature extraction
    across different implementations.

    Attributes:
        feature_dim: Dimension of the feature vector.
        is_available: Whether the backend is available.

    Example:
        >>> backend = ResNetBackend()
        >>> if backend.is_available:
        ...     features = backend.extract(cropped_region)
        ...     print(f"Extracted features: {features.shape}")
        >>>
        >>> # Warmup for faster first inference
        >>> backend.warmup()
    """

    @abstractmethod
    def extract(self, region: np.ndarray) -> np.ndarray | None:
        """Extracts features from an image region.

        Args:
            region: Image region (cropped object patch).

        Returns:
            Optional[np.ndarray]: Feature vector or None if extraction fails.

        Example:
            >>> region = frame[100:200, 50:150]  # Cropped object
            >>> features = backend.extract(region)
            >>> if features is not None:
            ...     print(f"Extracted {len(features)} features")
        """

    @abstractmethod
    def warmup(self) -> None:
        """Warms up the backend to reduce initial latency.

        This method performs any necessary initialization or
        dummy inference to load models into memory.

        Example:
            >>> backend.warmup()  # Perform warmup
            >>> # Subsequent calls will be faster
        """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the feature vector.

        Returns:
            int: Number of dimensions in the feature vector.

        Example:
            >>> dim = backend.feature_dim
            >>> print(f"Feature dimension: {dim}")
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Checks whether the backend is available.

        Returns:
            bool: True if the backend is ready for use.

        Example:
            >>> if backend.is_available:
            ...     features = backend.extract(region)
            ... else:
            ...     print("Backend not available")
        """

    @property
    def name(self) -> str:
        """Name of the backend.

        Returns:
            str: Backend name derived from class name.

        Example:
            >>> print(backend.name)  # "resnet" for ResNetBackend
        """
        return self.__class__.__name__.replace("Backend", "").lower()
