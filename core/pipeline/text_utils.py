"""
Utilidades para el renderizado de texto.
"""

from typing import Dict, TypedDict

import cv2


class TextMetrics(TypedDict):
    """Métricas de texto calculadas."""
    text: str
    width: int
    height: int
    baseline: int


class TextMetricsCache:
    """
    Caché para métricas de texto calculadas.
    Reduce llamadas a cv2.getTextSize.
    """

    __slots__ = ("_cache", "_max_size", "_font", "_scale", "_thickness")

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 0.5,
        thickness: int = 1,
        max_size: int = 100,
    ):
        self._cache: Dict[str, TextMetrics] = {}
        self._max_size = max_size
        self._font = font
        self._scale = scale
        self._thickness = thickness

    def get(self, text: str) -> TextMetrics:
        """Obtiene métricas de texto, calculando si no están en caché."""
        if text in self._cache:
            return self._cache[text]

        (width, height), baseline = cv2.getTextSize(
            text,
            self._font,
            self._scale,
            self._thickness,
        )

        metrics: TextMetrics = {
            "text": text,
            "width": width,
            "height": height,
            "baseline": baseline,
        }

        if len(self._cache) >= self._max_size:
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[text] = metrics
        return metrics

    def clear(self) -> None:
        """Limpia el caché."""
        self._cache.clear()
