"""SIFT backend for feature extraction.

Uses SIFT (Scale-Invariant Feature Transform) to extract local
features from the image.

Features:
    - Scale and rotation invariant local features
    - Good for textured objects
    - No GPU required
    - Robust to viewpoint changes

Example:
    >>> from models.feature_extractor.backends import SIFTBackend
    >>>
    >>> backend = SIFTBackend(n_features=128)
    >>>
    >>> # Extract features from a cropped object
    >>> region = frame[100:200, 50:150]
    >>> features = backend.extract(region)
    >>>
    >>> if features is not None:
    ...     print(f"Extracted {len(features)} features")
    ...     print(f"Feature dimension: {backend.feature_dim}")
    >>>
    >>> # Warmup for consistent performance
    >>> backend.warmup()
"""

import cv2
import numpy as np

from models.feature_extractor.backends.base import FeatureBackend
from utils.logger import LoggerMixin


class SIFTBackend(FeatureBackend, LoggerMixin):
    """SIFT backend for feature extraction.

    Uses SIFT (Scale-Invariant Feature Transform) to extract local
    features that are invariant to scale, rotation, and illumination.

    Features:
        - Scale and rotation invariant local features
        - Good for textured objects
        - No GPU required
        - Robust to viewpoint changes
        - Fast extraction for small regions

    Attributes:
        feature_dim: Dimension of the feature vector (128).
        is_available: Whether the backend is available.
        n_features: Maximum number of features to extract.

    Example:
        >>> backend = SIFTBackend(n_features=256)
        >>> features = backend.extract(cropped_region)
        >>> print(features.shape)  # (128,)
    """

    FEATURE_DIM = 128

    def __init__(self, n_features: int = 128):
        """Initializes the SIFT backend.

        Args:
            n_features: Maximum number of features to extract.

        Example:
            >>> # Standard SIFT
            >>> backend = SIFTBackend(n_features=128)
            >>>
            >>> # More features for better discrimination
            >>> backend = SIFTBackend(n_features=256)
        """
        self.n_features = n_features
        self._sift = None
        self._available = False
        self._warmed_up = False

        self._initialize()

        self.logger.info(
            "SIFTBackend initialized", available=self._available, n_features=n_features
        )

    def _initialize(self) -> None:
        """Initializes the SIFT extractor."""
        try:
            self._sift = cv2.SIFT_create(nfeatures=self.n_features)
            self._available = True
            self.logger.info("SIFT initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SIFT: {e}")
            self._available = False

    def extract(self, region: np.ndarray) -> np.ndarray | None:
        """Extracts features using SIFT.

        This method detects keypoints and computes descriptors for
        the region. If no keypoints are found, returns a fallback
        histogram-based feature vector.

        Args:
            region: Image region (cropped object patch).

        Returns:
            Optional[np.ndarray]: Feature vector of dimension 128 or None.

        Example:
            >>> region = frame[100:200, 50:150]
            >>> features = backend.extract(region)
            >>> if features is not None:
            ...     print(f"Extracted {len(features)} features")
        """
        if not self._available or self._sift is None:
            return None

        if region is None or region.size == 0:
            return None

        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            keypoints, descriptors = self._sift.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                return self._fallback_features(region)

            features = np.mean(descriptors, axis=0)

            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features.astype(np.float32)

        except Exception as e:
            self.logger.debug(f"SIFT extraction error: {e}")
            return self._fallback_features(region)

    def _fallback_features(self, region: np.ndarray) -> np.ndarray | None:
        """Fallback features using simple histogram.

        This method provides a fallback when SIFT fails to find
        keypoints in the region.

        Args:
            region: Image region.

        Returns:
            Optional[np.ndarray]: Feature vector of dimension 128.

        Note:
            This fallback uses a 32-bin histogram of grayscale values.
        """
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if len(hist) < self.FEATURE_DIM:
                hist = np.pad(hist, (0, self.FEATURE_DIM - len(hist)))

            return hist[: self.FEATURE_DIM].astype(np.float32)

        except Exception:
            return None

    def warmup(self) -> None:
        """Warms up the backend.

        Performs a dummy extraction to initialize any internal
        state and ensure consistent performance.

        Example:
            >>> backend.warmup()
            >>> # First extraction is now faster
        """
        if self._warmed_up or not self._available:
            return

        self.logger.info("Warming up SIFT...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.extract(dummy)
            self._warmed_up = True
            self.logger.info("SIFT warmed up")
        except Exception as e:
            self.logger.warning(f"Warmup error: {e}")

    @property
    def feature_dim(self) -> int:
        """Dimension of the feature vector."""
        return self.FEATURE_DIM

    @property
    def is_available(self) -> bool:
        """Whether the backend is available."""
        return self._available
