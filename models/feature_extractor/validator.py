"""
Validador de calidad para regiones de imagen.

Verifica que las regiones extraídas tengan calidad suficiente
para la extracción de features.
"""

from typing import Tuple, Any, Dict

import cv2
import numpy as np


from utils.logger import LoggerMixin


class FeatureValidator(LoggerMixin):
    """
    Validador de calidad para regiones de imagen.

    Verifica:
    - Tamaño mínimo
    - Brillo suficiente
    - Contraste adecuado
    - Sin regiones vacías

    Attributes:
        min_area: Área mínima de la región
        min_brightness: Brillo mínimo permitido
        min_contrast: Contraste mínimo permitido
    """

    def __init__(
        self,
        min_area: int = 100,
        min_brightness: int = 10,
        min_contrast: int = 5
    ):
        """
        Inicializa el validador.

        Args:
            min_area: Área mínima de la región
            min_brightness: Brillo mínimo permitido
            min_contrast: Contraste mínimo permitido
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
            "FeatureValidator inicializado",
            min_area=min_area,
            min_brightness=min_brightness,
            min_contrast=min_contrast
        )

    def validate_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int]
    ) -> bool:
        """
        Valida que el bounding box sea válido.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            image_shape: Dimensiones de la imagen (height, width)

        Returns:
            bool: True si el bbox es válido
        """
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return False

        try:
            x1, y1, x2, y2 = bbox

            h, w = image_shape[:2]
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return False
            if x1 >= w or y1 >= h or x2 > w or y2 > h:
                return False

            width = x2 - x1
            height = y2 - y1
            if width < 10 or height < 10:
                self._stats["too_small"] += 1
                self._stats["invalid"] += 1
                return False

            area = width * height
            if area < self.min_area:
                self._stats["too_small"] += 1
                self._stats["invalid"] += 1
                return False

            return True

        except (TypeError, ValueError):
            return False

    def validate_region(self, region: np.ndarray) -> float:
        """
        Valida la calidad de una región de imagen.

        Args:
            region: Región de imagen

        Returns:
            float: Puntuación de calidad (0-1)
        """
        if region is None or region.size == 0:
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

            mean_brightness = np.mean(gray)

            if mean_brightness < self.min_brightness:
                self._stats["too_dark"] += 1
                self._stats["invalid"] += 1
                return 0.1

            if mean_brightness > 240:
                self._stats["too_bright"] += 1
                self._stats["invalid"] += 1
                return 0.2

            std_brightness = np.std(gray)

            if std_brightness < self.min_contrast:
                self._stats["low_contrast"] += 1
                self._stats["invalid"] += 1
                return 0.2

            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            brightness_score = max(0, brightness_score)

            contrast_score = min(1.0, std_brightness / 50.0)

            area_score = min(1.0, area / 2000.0)

            score = (
                0.3 * brightness_score +
                0.4 * contrast_score +
                0.3 * area_score
            )

            score = min(1.0, score)

            if score >= 0.3:
                self._stats["valid"] += 1
            else:
                self._stats["invalid"] += 1

            return score

        except Exception as e:
            self.logger.debug(f"Error validando región: {e}")
            self._stats["invalid"] += 1
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del validador."""
        total = self._stats["valid"] + self._stats["invalid"]

        return {
            **self._stats,
            "total_validations": total,
            "valid_rate": self._stats["valid"] / max(1, total),
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas."""
        self._stats = {
            "valid": 0,
            "invalid": 0,
            "too_small": 0,
            "too_dark": 0,
            "too_bright": 0,
            "low_contrast": 0,
            "empty_region": 0,
        }
