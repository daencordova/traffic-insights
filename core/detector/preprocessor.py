"""Image preprocessing for improved detection.

Provides functions to enhance image quality before object detection,
including noise reduction, histogram equalization, and contrast enhancement.
"""

import time
from typing import Any

import cv2
import numpy as np

from core.constants import PROCESSING_TIMES_MAX
from utils.logger import LoggerMixin


class ImagePreprocessor(LoggerMixin):
    """Image preprocessor for improved detection.

    This class applies various image enhancement techniques to improve
    object detection quality and accuracy.

    Features:
        - Noise reduction (fastNlMeansDenoising)
        - Histogram equalization (CLAHE)
        - Contrast enhancement
        - Normalization
        - Automatic quality assessment

    Attributes:
        enabled: Whether preprocessing is enabled.
        denoise_strength: Noise reduction filter strength (1-10).
        equalize_histogram: Whether to apply histogram equalization.
        enhance_contrast: Whether to enhance contrast.

    Example:
        >>> preprocessor = ImagePreprocessor(
        ...     enabled=True, denoise_strength=5, equalize_histogram=True
        ... )
        >>>
        >>> # Process a frame
        >>> enhanced = preprocessor.process(frame)
        >>>
        >>> # Check if processing is needed
        >>> if preprocessor.should_process(frame):
        ...     enhanced = preprocessor.process(frame)
        >>>
        >>> # Get statistics
        >>> stats = preprocessor.get_stats()
        >>> print(f"Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")
        >>>
        >>> # Toggle preprocessing
        >>> preprocessor.set_enabled(False)
    """

    def __init__(
        self,
        enabled: bool = False,
        denoise_strength: int = 5,
        equalize_histogram: bool = True,
        enhance_contrast: bool = True,
    ):
        """Initializes the image preprocessor.

        Args:
            enabled: Whether preprocessing is enabled.
            denoise_strength: Noise reduction filter strength (1-10).
                Higher values apply stronger filtering.
            equalize_histogram: Whether to apply histogram equalization.
            enhance_contrast: Whether to enhance contrast using CLAHE.

        Example:
            >>> # Aggressive preprocessing
            >>> preprocessor = ImagePreprocessor(
            ...     enabled=True, denoise_strength=8, equalize_histogram=True, enhance_contrast=True
            ... )
            >>>
            >>> # Light preprocessing
            >>> preprocessor = ImagePreprocessor(
            ...     enabled=True,
            ...     denoise_strength=3,
            ...     equalize_histogram=False,
            ...     enhance_contrast=False,
            ... )
        """
        self.enabled = enabled
        self.denoise_strength = denoise_strength
        self.equalize_histogram = equalize_histogram
        self.enhance_contrast = enhance_contrast

        self._stats = {
            "processed_frames": 0,
            "avg_processing_time_ms": 0.0,
            "processing_times": [],
        }

        self.logger.info(
            "ImagePreprocessor initialized",
            enabled=enabled,
            denoise_strength=denoise_strength,
            equalize_histogram=equalize_histogram,
        )

    def should_process(self, frame: np.ndarray) -> bool:
        """Determines if the frame needs preprocessing based on quality.

        This method analyzes the frame quality (brightness and contrast)
        and decides whether preprocessing is necessary.

        Args:
            frame: Image to analyze.

        Returns:
            bool: True if preprocessing should be applied.

        Example:
            >>> if preprocessor.should_process(frame):
            ...     frame = preprocessor.process(frame)
            ... else:
            ...     # Frame is already good quality
            ...     pass
        """
        if not self.enabled:
            return False

        if frame is None or frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        if mean_brightness > 30 and mean_brightness < 220 and std_brightness > 20:
            return False

        return True

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processes an image applying configured enhancements.

        The enhancements are applied in this order:
            1. Noise reduction (bilateral filter)
            2. Histogram equalization
            3. Contrast enhancement (CLAHE)

        Args:
            frame: Image to process in BGR format.

        Returns:
            np.ndarray: Processed image or copy of original if disabled.

        Note:
            If preprocessing is disabled or the frame is invalid,
            the original frame is returned unchanged.

        Example:
            >>> enhanced = preprocessor.process(frame)
            >>> cv2.imshow("Original", frame)
            >>> cv2.imshow("Enhanced", enhanced)
        """
        if not self.should_process(frame):
            return frame

        if not self.enabled or frame is None:
            return frame

        start_time = time.perf_counter()

        try:
            result = frame.copy()

            if self.denoise_strength > 0:
                d = max(1, min(15, self.denoise_strength * 3))
                sigma_color = 75
                sigma_space = 75
                result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)

            if self.equalize_histogram:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if self.enhance_contrast:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(elapsed_ms)

            return result

        except Exception as e:
            self.logger.warning(f"Preprocessing error: {e}")
            return frame

    def _update_stats(self, time_ms: float) -> None:
        """Updates processing statistics.

        Args:
            time_ms: Processing time in milliseconds.
        """
        self._stats["processed_frames"] += 1
        self._stats["processing_times"].append(time_ms)

        if len(self._stats["processing_times"]) > PROCESSING_TIMES_MAX:
            self._stats["processing_times"] = self._stats["processing_times"][
                -PROCESSING_TIMES_MAX:
            ]

        self._stats["avg_processing_time_ms"] = sum(self._stats["processing_times"]) / len(
            self._stats["processing_times"]
        )

    def get_stats(self) -> dict[str, Any]:
        """Gets preprocessor statistics.

        Returns:
            Dict[str, Any]: Statistics including:
                - processed_frames: Number of processed frames
                - avg_processing_time_ms: Average processing time
                - enabled: Whether preprocessing is enabled
                - denoise_strength: Noise reduction strength
                - equalize_histogram: Whether equalization is enabled

        Example:
            >>> stats = preprocessor.get_stats()
            >>> print(f"Frames processed: {stats['processed_frames']}")
            >>> print(f"Avg time: {stats['avg_processing_time_ms']:.2f}ms")
        """
        return {
            **self._stats,
            "enabled": self.enabled,
            "denoise_strength": self.denoise_strength,
            "equalize_histogram": self.equalize_histogram,
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables preprocessing.

        Args:
            enabled: True to enable, False to disable.

        Example:
            >>> preprocessor.set_enabled(True)
            >>> # Preprocessing is now active
            >>>
            >>> preprocessor.set_enabled(False)
            >>> # Preprocessing is now disabled
        """
        self.enabled = enabled
        self.logger.info(f"Preprocessing {'enabled' if enabled else 'disabled'}")

    def set_denoise_strength(self, strength: int) -> None:
        """Adjusts the noise reduction filter strength.

        Args:
            strength: Filter strength (0-10). 0 disables the filter.

        Example:
            >>> preprocessor.set_denoise_strength(7)
            >>> # Stronger noise reduction
        """
        self.denoise_strength = max(0, min(10, strength))
        self.logger.info(f"Noise reduction strength: {self.denoise_strength}")
