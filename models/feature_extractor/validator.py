"""Quality validator for image regions.

Verifies that extracted regions have sufficient quality
for feature extraction.
"""

from typing import Any

import cv2
import numpy as np

from core.constants import MAX_BBOX_DIMENSION, MAX_BRIGHTNESS, MIN_BBOX_SIZE, MIN_VALID_SCORE
from utils.logger import LoggerMixin


class FeatureValidator(LoggerMixin):
    """Quality validator for image regions.

    This class validates image regions for feature extraction by checking:
        - Minimum size
        - Sufficient brightness
        - Adequate contrast
        - Non-empty regions

    Attributes:
        min_area: Minimum region area.
        min_brightness: Minimum allowed brightness.
        min_contrast: Minimum allowed contrast.

    Example:
        >>> validator = FeatureValidator(min_area=100, min_brightness=10, min_contrast=5)
        >>>
        >>> # Validate bounding box
        >>> if validator.validate_bbox(bbox, image_shape):
        ...     region = image[y1:y2, x1:x2]
        ...     quality_score = validator.validate_region(region)
        ...     if quality_score > 0.5:
        ...         features = extract_features(region)
        >>>
        >>> # Get validation statistics
        >>> stats = validator.get_stats()
        >>> print(f"Valid rate: {stats['valid_rate']:.2%}")
    """

    __slots__ = (
        "min_area",
        "min_brightness",
        "min_contrast",
        "_stats",
    )

    def __init__(
        self,
        min_area: int = 100,
        min_brightness: int = 10,
        min_contrast: int = 5,
    ) -> None:
        """Initializes the validator.

        Args:
            min_area: Minimum region area in pixels.
            min_brightness: Minimum allowed brightness (0-255).
            min_contrast: Minimum allowed contrast (standard deviation).

        Example:
            >>> # Strict validation
            >>> validator = FeatureValidator(min_area=200, min_brightness=20, min_contrast=10)
            >>>
            >>> # Lenient validation
            >>> validator = FeatureValidator(min_area=50, min_brightness=5, min_contrast=3)
        """
        self.min_area = min_area
        self.min_brightness = min_brightness
        self.min_contrast = min_contrast

        self._stats = {
            "valid": 0,
            "invalid": 0,
            "too_small": 0,
            "too_dark": 0,
            "too_bright": 0,
            "low_contrast": 0,
            "empty_region": 0,
        }

        self.logger.info(
            "FeatureValidator initialized",
            min_area=min_area,
            min_brightness=min_brightness,
            min_contrast=min_contrast,
        )

    def validate_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
    ) -> bool:
        """Validates that the bounding box is valid.

        This method checks:
            - Correct structure (4 elements)
            - Coordinates within image boundaries
            - Minimum dimensions

        Args:
            bbox: Bounding box (x1, y1, x2, y2).
            image_shape: Image dimensions (height, width).

        Returns:
            bool: True if the bbox is valid.

        Example:
            >>> if validator.validate_bbox(bbox, (480, 640)):
            ...     region = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        """
        if not self._is_valid_bbox_structure(bbox):
            return False

        try:
            x1, y1, x2, y2 = bbox
            h, w = image_shape[:2]

            if not self._are_coordinates_valid(x1, y1, x2, y2, w, h):
                self._stats["invalid"] += 1
                return False

            if not self._are_dimensions_valid(x1, y1, x2, y2):
                self._stats["invalid"] += 1
                return False

            return True

        except (TypeError, ValueError):
            return False

    def _is_valid_bbox_structure(self, bbox: Any) -> bool:
        """Checks that the bbox has the correct structure."""
        return isinstance(bbox, (tuple, list)) and len(bbox) == MAX_BBOX_DIMENSION

    def _are_coordinates_valid(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
    ) -> bool:
        """Checks that coordinates are within bounds."""
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False

        return not (x1 >= w or y1 >= h or x2 > w or y2 > h)

    def _are_dimensions_valid(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Checks that bbox dimensions are valid."""
        width = x2 - x1
        height = y2 - y1

        if width < MIN_BBOX_SIZE or height < MIN_BBOX_SIZE:
            self._stats["too_small"] += 1
            return False

        area = width * height
        if area < self.min_area:
            self._stats["too_small"] += 1
            return False

        return True

    def validate_region(self, region: np.ndarray) -> float:
        """Validates the quality of an image region.

        This method evaluates region quality based on:
            - Size/area
            - Brightness (mean intensity)
            - Contrast (standard deviation)

        Args:
            region: Image region.

        Returns:
            float: Quality score (0-1), where 0 is invalid.

        Example:
            >>> score = validator.validate_region(region)
            >>> if score > 0.6:
            ...     print("Region quality is good")
            ...     features = extract_features(region)
            ... else:
            ...     print("Region quality is poor")
        """
        if not self._is_valid_region(region):
            self._stats["empty_region"] += 1
            self._stats["invalid"] += 1
            return 0.0

        try:
            h, w = region.shape[:2]
            area = h * w

            if area < self.min_area:
                self._stats["too_small"] += 1
                self._stats["invalid"] += 1
                return 0.0

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            brightness_score = self._validate_brightness(gray)
            if brightness_score <= 0.0:
                return 0.0

            contrast_score = self._validate_contrast(gray)
            if contrast_score <= 0.0:
                return 0.0

            area_score = min(1.0, area / 2000.0)

            score = 0.3 * brightness_score + 0.4 * contrast_score + 0.3 * area_score

            score = min(1.0, score)
            self._update_stats(score)

            return score

        except Exception as e:
            self.logger.debug(f"Error validating region: {e}")
            self._stats["invalid"] += 1
            return 0.0

    def _is_valid_region(self, region: np.ndarray) -> bool:
        """Checks that the region is not None or empty."""
        return region is not None and region.size > 0

    def _validate_brightness(self, gray: np.ndarray) -> float:
        """Validates region brightness.

        Returns:
            float: Brightness score (0-1), 0 if invalid.
        """
        mean_brightness = np.mean(gray)

        if mean_brightness < self.min_brightness:
            self._stats["too_dark"] += 1
            self._stats["invalid"] += 1
            return 0.0

        if mean_brightness > MAX_BRIGHTNESS:
            self._stats["too_bright"] += 1
            self._stats["invalid"] += 1
            return 0.0

        return 1.0 - abs(mean_brightness - 128) / 128.0

    def _validate_contrast(self, gray: np.ndarray) -> float:
        """Validates region contrast.

        Returns:
            float: Contrast score (0-1), 0 if invalid.
        """
        std_brightness = np.std(gray)

        if std_brightness < self.min_contrast:
            self._stats["low_contrast"] += 1
            self._stats["invalid"] += 1
            return 0.0

        return min(1.0, std_brightness / 50.0)

    def _update_stats(self, score: float) -> None:
        """Updates statistics based on score."""
        if score >= MIN_VALID_SCORE:
            self._stats["valid"] += 1
        else:
            self._stats["invalid"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Gets validator statistics.

        Returns:
            dict[str, Any]: Statistics including:
                - valid: Number of valid regions
                - invalid: Number of invalid regions
                - too_small: Regions too small
                - too_dark: Regions too dark
                - too_bright: Regions too bright
                - low_contrast: Regions with low contrast
                - empty_region: Empty regions
                - total_validations: Total validations
                - valid_rate: Validation success rate

        Example:
            >>> stats = validator.get_stats()
            >>> print(f"Valid: {stats['valid']}")
            >>> print(f"Invalid: {stats['invalid']}")
            >>> print(f"Success rate: {stats['valid_rate']:.2%}")
        """
        total = self._stats["valid"] + self._stats["invalid"]

        return {
            **self._stats,
            "total_validations": total,
            "valid_rate": self._stats["valid"] / max(1, total),
        }

    def reset_stats(self) -> None:
        """Resets all statistics.

        Example:
            >>> validator.reset_stats()
            >>> # All statistics are reset to zero
        """
        self._stats = {
            "valid": 0,
            "invalid": 0,
            "too_small": 0,
            "too_dark": 0,
            "too_bright": 0,
            "low_contrast": 0,
            "empty_region": 0,
        }
