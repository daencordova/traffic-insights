"""Preprocesamiento de imágenes para mejorar detección.

Proporciona funciones para mejorar la calidad de las imágenes
antes de la detección de objetos, incluyendo reducción de ruido,
ecualización de histograma y mejora de contraste.
"""

import time
from typing import Any

import cv2
import numpy as np

from core.constants.values import PROCESSING_TIMES_MAX
from utils.logger import LoggerMixin


class ImagePreprocessor(LoggerMixin):
    """Preprocesador de imágenes para mejorar detección.

    Características:
        - Reducción de ruido (fastNlMeansDenoising)
        - Ecualización de histograma (CLAHE)
        - Mejora de contraste
        - Normalización

    Attributes:
        enabled: Si el preprocesamiento está activado.
        denoise_strength: Fuerza del filtro de reducción de ruido (1-10).
        equalize_histogram: Si aplicar ecualización de histograma.
        enhance_contrast: Si mejorar el contraste.

    Example:
        >>> preprocessor = ImagePreprocessor(enabled=True, denoise_strength=5)
        >>> enhanced = preprocessor.process(frame)
    """

    def __init__(
        self,
        enabled: bool = False,
        denoise_strength: int = 5,
        equalize_histogram: bool = True,
        enhance_contrast: bool = True,
    ):
        """Inicializa el preprocesador.

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
            equalize_histogram=equalize_histogram,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Procesa una imagen aplicando las mejoras configuradas.

        Args:
            frame: Imagen a procesar en formato BGR.

        Returns:
            np.ndarray: Imagen procesada o copia del original si está desactivado.

        Note:
            Las mejoras se aplican en orden:
            1. Reducción de ruido
            2. Ecualización de histograma
            3. Mejora de contraste (CLAHE)
        """
        if not self.enabled or frame is None:
            return frame

        start_time = time.perf_counter()

        try:
            result = frame.copy()

            if self.denoise_strength > 0:
                h = max(1, min(10, self.denoise_strength))
                result = cv2.fastNlMeansDenoisingColored(result, None, h, h + 5, 7, 21)

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
        """Actualiza estadísticas de procesamiento.

        Args:
            time_ms: Tiempo de procesamiento en milisegundos.
        """
        self._stats["processed_frames"] += 1
        self._stats["processing_times"].append(time_ms)

        if len(self._stats["processing_times"]) > PROCESSING_TIMES_MAX:
            self._stats["processing_times"] = self._stats["processing_times"][-PROCESSING_TIMES_MAX:]

        self._stats["avg_processing_time_ms"] = sum(self._stats["processing_times"]) / len(
            self._stats["processing_times"]
        )

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del preprocesador.

        Returns:
            Dict[str, Any]: Estadísticas incluyendo:
                - processed_frames: Frames procesados
                - avg_processing_time_ms: Tiempo promedio de procesamiento
                - enabled: Si está activado
                - denoise_strength: Fuerza de reducción de ruido
                - equalize_histogram: Si ecualización está activada

        Example:
            >>> stats = preprocessor.get_stats()
            >>> print(f"Avg time: {stats['avg_processing_time_ms']:.2f}ms")
        """
        return {
            **self._stats,
            "enabled": self.enabled,
            "denoise_strength": self.denoise_strength,
            "equalize_histogram": self.equalize_histogram,
        }

    def set_enabled(self, enabled: bool) -> None:
        """Activa o desactiva el preprocesamiento.

        Args:
            enabled: True para activar, False para desactivar.
        """
        self.enabled = enabled
        self.logger.info(f"Preprocesamiento {'activado' if enabled else 'desactivado'}")

    def set_denoise_strength(self, strength: int) -> None:
        """Ajusta la fuerza del filtro de reducción de ruido.

        Args:
            strength: Fuerza del filtro (0-10). 0 desactiva el filtro.
        """
        self.denoise_strength = max(0, min(10, strength))
        self.logger.info(f"Fuerza de reducción de ruido: {self.denoise_strength}")
