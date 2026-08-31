"""Histogram-based backend for feature extraction.

This backend is fast and does not require GPU, ideal for CPU.
Combines color histograms, texture, and moments for robust feature extraction.

Features:
    - HSV and LAB color histograms
    - Gradient histograms
    - Hu moments
    - Basic statistics (mean, std, median, min, max)
    - No GPU required
    - Fast and lightweight

Example:
    >>> from models.feature_extractor.backends import HistogramBackend
    >>>
    >>> backend = HistogramBackend()
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


class HistogramBackend(FeatureBackend, LoggerMixin):
    """Histogram-based backend for feature extraction.

    This backend extracts features using a combination of:
        - HSV and LAB color histograms
        - Gradient magnitude and angle histograms
        - Hu moments (shape descriptors)
        - Basic image statistics (mean, std, median, min, max)
        - Area ratio and aspect ratio

    Features:
        - No GPU required (CPU-only)
        - Fast extraction (suitable for real-time)
        - Robust to lighting variations
        - Dimensionality: 2048 features

    Attributes:
        feature_dim: Dimension of the feature vector (2048).
        is_available: Always True (no external dependencies).

    Example:
        >>> backend = HistogramBackend()
        >>> features = backend.extract(cropped_region)
        >>> print(features.shape)  # (2048,)
    """

    FEATURE_DIM = 2048

    def __init__(self):
        """Initializes the histogram backend."""
        self._available = True
        self._warmed_up = False

        self.logger.info("HistogramBackend initialized", feature_dim=self.FEATURE_DIM)

    def extract(self, region: np.ndarray) -> np.ndarray | None:
        """Extracts features using histograms.

        This method combines multiple feature types:
            1. HSV color histogram (8x8)
            2. LAB color histogram (4x4x4)
            3. Gradient magnitude histogram (16 bins)
            4. Gradient angle histogram (16 bins)
            5. Hu moments (first 4)
            6. Image statistics (mean, std, median, min, max)
            7. Area ratio and aspect ratio

        Args:
            region: Image region (cropped object patch).

        Returns:
            Optional[np.ndarray]: Feature vector of dimension 2048 or None.

        Example:
            >>> region = frame[y1:y2, x1:x2]
            >>> features = backend.extract(region)
            >>> if features is not None:
            ...     print(f"Extracted features: {len(features)}")
        """
        if region is None or region.size == 0:
            return None

        try:
            features = []

            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hist_hsv = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            features.extend(hist_hsv[:64])

            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            features.extend(hist_lab[:32])

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobel_x, sobel_y)

            hist_mag, _ = np.histogram(magnitude.flatten(), bins=16, range=(0, 255))
            hist_mag = cv2.normalize(
                hist_mag.astype(np.float32), hist_mag.astype(np.float32)
            ).flatten()
            features.extend(hist_mag[:16])

            angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)
            hist_angle, _ = np.histogram(angle.flatten(), bins=16, range=(0, 360))
            hist_angle = cv2.normalize(
                hist_angle.astype(np.float32), hist_angle.astype(np.float32)
            ).flatten()
            features.extend(hist_angle[:8])

            moments = cv2.HuMoments(cv2.moments(gray)).flatten()
            features.extend(moments[:4])

            stats = [
                float(np.mean(gray)) / 255.0,
                float(np.std(gray)) / 255.0,
                float(np.median(gray)) / 255.0,
                float(np.min(gray)) / 255.0,
                float(np.max(gray)) / 255.0,
            ]
            features.extend(stats)

            h, w = region.shape[:2]
            area_ratio = (h * w) / (region.size / 3)
            aspect_ratio = w / h if h > 0 else 1.0
            features.extend([area_ratio, min(aspect_ratio, 5.0) / 5.0])

            features_array = np.array(features, dtype=np.float32)

            if len(features_array) > self.FEATURE_DIM:
                features_array = features_array[: self.FEATURE_DIM]
            elif len(features_array) < self.FEATURE_DIM:
                padding = self.FEATURE_DIM - len(features_array)
                features_array = np.pad(features_array, (0, padding))

            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm

            return features_array

        except Exception as e:
            self.logger.debug(f"Histogram extraction error: {e}")
            return None

    def warmup(self) -> None:
        """Warms up the backend.

        Performs a dummy extraction to initialize any internal state
        and ensure consistent performance.

        Example:
            >>> backend.warmup()
            >>> # First extraction is now faster
        """
        if self._warmed_up:
            return

        self.logger.info("Warming up HistogramBackend...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.extract(dummy)
            self._warmed_up = True
            self.logger.info("HistogramBackend warmed up")
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
