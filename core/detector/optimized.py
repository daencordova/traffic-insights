"""Detector YOLO optimizado para CPU con ONNX Runtime y Numba (Fachada).

Este módulo proporciona una fachada para el sistema de detección optimizado,
delegando toda la funcionalidad al DetectorOrchestrator.

La clase OptimizedYOLODetector actúa como una fachada para mantener
compatibilidad con el código existente.
"""

from typing import Any

import numpy as np

from core.detector.base import YOLODetector
from core.detector.config import DetectorConfig
from core.detector.orchestrator import DetectorOrchestrator
from core.exceptions import ModelLoadError
from core.types import DetectionList


class OptimizedYOLODetector(YOLODetector):
    """Detector YOLO optimizado para CPU (Fachada).

    Esta clase actúa como fachada para el DetectorOrchestrator,
    manteniendo la misma interfaz que la implementación anterior
    para garantizar compatibilidad con el código existente.

    Características:
        - ONNX Runtime para inferencia rápida en CPU
        - Numba para NMS optimizado
        - Caché LRU para detecciones
        - Warmup automático
        - Fallback a PyTorch si ONNX no está disponible

    Example:
        >>> detector = OptimizedYOLODetector()
        >>> frame = cv2.imread("image.jpg")
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"Objeto: {det['label']} confianza: {det['confidence']:.2f}")
    """

    __slots__ = ("_orchestrator", "_config")

    def __init__(self, config: DetectorConfig | None = None):
        """Inicializa el detector optimizado (fachada).

        Args:
            config: Configuración del detector. Si es None, se usa
                la configuración global del sistema.

        Raises:
            ModelLoadError: Si no se puede cargar ningún modelo.
        """
        self._config = config or DetectorConfig.from_global_config()
        self.logger.info("Inicializando OptimizedYOLODetector (fachada)")

        try:
            self._orchestrator = DetectorOrchestrator(self._config)
            self.logger.info(
                "OptimizedYOLODetector inicializado",
                onnx_available=self._orchestrator.is_onnx_available,
                numba_available=self._orchestrator.is_numba_available,
                warmed_up=self._orchestrator._warmed_up,
            )
        except Exception as err:
            self.logger.error(f"Error inicializando detector optimizado: {err}")
            raise ModelLoadError(f"No se pudo inicializar el detector: {err}") from err

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detecta objetos en un frame.

        Args:
            frame: Imagen a procesar en formato numpy array (H, W, C) BGR.

        Returns:
            DetectionList: Lista de detecciones validadas.
        """
        return self._orchestrator.detect(frame)

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionList]:
        """Detecta objetos en múltiples frames (batch inference).

        Args:
            frames: Lista de imágenes a procesar.

        Returns:
            List[DetectionList]: Lista de listas de detecciones.
        """
        return self._orchestrator.detect_batch(frames)

    def get_classes(self) -> list[int]:
        """Retorna las clases que detecta el modelo."""
        return self._orchestrator.get_classes()

    def get_performance_stats(self) -> dict[str, Any]:
        """Retorna estadísticas de rendimiento del detector."""
        return self._orchestrator.get_performance_stats()

    def clear_cache(self) -> None:
        """Limpia el caché de detecciones."""
        self._orchestrator.clear_cache()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Activa o desactiva el preprocesamiento de imágenes.

        Args:
            enable: True para activar, False para desactivar.
        """
        self._orchestrator.enable_enhancement(enable)

    @property
    def config(self) -> DetectorConfig:
        """Retorna la configuración del detector."""
        return self._config

    @property
    def device(self) -> str:
        """Retorna el dispositivo de inferencia."""
        return self._orchestrator.device

    @property
    def model(self):
        """Retorna el modelo activo (para compatibilidad)."""
        return self._orchestrator.model

    @property
    def cache(self):
        """Retorna el caché de detecciones (para compatibilidad)."""
        return self._orchestrator.cache

    @property
    def preprocessor(self):
        """Retorna el preprocesador (para compatibilidad)."""
        return self._orchestrator.preprocessor

    @property
    def post_processor(self):
        """Retorna el post-procesador (para compatibilidad)."""
        return self._orchestrator.post_processor

    @property
    def onnx_engine(self):
        """Retorna el motor ONNX (para compatibilidad)."""
        return self._orchestrator.onnx_engine

    @property
    def pytorch_engine(self):
        """Retorna el motor PyTorch (para compatibilidad)."""
        return self._orchestrator.pytorch_engine

    @property
    def model_manager(self):
        """Retorna el gestor de modelos (para compatibilidad)."""
        return self._orchestrator.model_manager

    @property
    def model_exporter(self):
        """Retorna el exportador de modelos (para compatibilidad)."""
        return self._orchestrator.model_exporter
