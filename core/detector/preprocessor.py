"""
Preprocesamiento de imágenes para mejorar detección.

Proporciona funciones para mejorar la calidad de las imágenes
antes de la detección de objetos.
"""

from typing import Dict, Any

import cv2
import numpy as np

from utils.logger import LoggerMixin


class ImagePreprocessor(LoggerMixin):
    """
    Preprocesador de imágenes para mejorar detección.

    Características:
    - Reducción de ruido
    - Ecualización de histograma
    - Mejora de contraste
    - Normalización

    Attributes:
        enabled: Si el preprocesamiento está activado
        denoise_strength: Fuerza del filtro de reducción de ruido
        equalize_histogram: Si aplicar ecualización de histograma
    """

    def __init__(
        self,
        enabled: bool = False,
        denoise_strength: int = 5,
        equalize_histogram: bool = True,
        enhance_contrast: bool = True
    ):
        """
        Inicializa el preprocesador.

        Args:
            enabled: Si el preprocesamiento está activado
            denoise_strength: Fuerza del filtro de reducción de ruido (1-10)
            equalize_histogram: Si aplicar ecualización de histograma
            enhance_contrast: Si mejorar el contraste
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
            "ImagePreprocessor inicializado",
            enabled=enabled,
            denoise_strength=denoise_strength,
            equalize_histogram=equalize_histogram
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa una imagen aplicando las mejoras configuradas.

        Args:
            frame: Imagen a procesar

        Returns:
            np.ndarray: Imagen procesada
        """
        if not self.enabled or frame is None:
            return frame

        import time
        start_time = time.perf_counter()

        try:
            result = frame.copy()

            if self.denoise_strength > 0:
                h = max(1, min(10, self.denoise_strength))
                result = cv2.fastNlMeansDenoisingColored(
                    result,
                    None,
                    h,
                    h + 5,
                    7,
                    21
                )

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
            self.logger.warning(f"Error en preprocesamiento: {e}")
            return frame

    def _update_stats(self, time_ms: float) -> None:
        """Actualiza estadísticas de procesamiento."""
        self._stats["processed_frames"] += 1
        self._stats["processing_times"].append(time_ms)

        if len(self._stats["processing_times"]) > 100:
            self._stats["processing_times"] = self._stats["processing_times"][-100:]

        self._stats["avg_processing_time_ms"] = (
            sum(self._stats["processing_times"]) / len(self._stats["processing_times"])
        )

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del preprocesador."""
        return {
            **self._stats,
            "enabled": self.enabled,
            "denoise_strength": self.denoise_strength,
            "equalize_histogram": self.equalize_histogram,
        }

    def set_enabled(self, enabled: bool) -> None:
        """Activa o desactiva el preprocesamiento."""
        self.enabled = enabled
        self.logger.info(f"Preprocesamiento {'activado' if enabled else 'desactivado'}")

    def set_denoise_strength(self, strength: int) -> None:
        """Ajusta la fuerza del filtro de reducción de ruido."""
        self.denoise_strength = max(0, min(10, strength))
        self.logger.info(f"Fuerza de reducción de ruido: {self.denoise_strength}")
