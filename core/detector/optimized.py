"""Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

Orquesta los componentes de gestión de modelos, inferencia,
post-procesamiento y caché.
"""

import os
import time
from typing import Any

import numpy as np

from core.constants.tracking import (
    CACHE_DEFAULT_SIZE,
    CACHE_MAX_SIZE,
    CACHE_MIN_SIZE,
    FEATURE_CACHE_MAX_AGE,
)
from core.constants.values import INFERENCE_TIMES_MAX
from core.detector.base import YOLODetector
from core.detector.cache import DetectionCache
from core.detector.config import DetectorConfig
from core.detector.inference_engine import InferenceEngine, InferenceEngineFactory
from core.detector.model_exporter import ModelExporter
from core.detector.model_manager import ModelLoadError, ModelManager
from core.detector.post_processor import PostProcessor
from core.detector.preprocessor import ImagePreprocessor
from utils.helpers import get_memory_usage


class OptimizedYOLODetector(YOLODetector):
    """Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

    Orquesta los componentes de gestión de modelos, inferencia,
    post-procesamiento y caché para máxima eficiencia en CPU.

    Características:
        - ONNX Runtime para inferencia rápida en CPU
        - Numba para NMS optimizado
        - Caché LRU para detecciones
        - Warmup automático
        - Fallback a PyTorch si ONNX no está disponible

    Example:
        >>> detector = OptimizedYOLODetector()
        >>> frame = cv2.imread("image.jpg")
        >>> detections = detector.detect(frame)  # Usa ONNX si está disponible
    """

    __slots__ = (
        "confidence_threshold",
        "iou_threshold",
        "vehicle_classes",
        "imgsz",
        "max_det",
        "model_manager",
        "model_exporter",
        "_pytorch_engine",
        "_onnx_engine",
        "post_processor",
        "_warmed_up",
        "_onnx_available",
        "_numba_available",
    )

    def __init__(self, config: DetectorConfig | None = None):
        """Inicializa el detector optimizado.

        Args:
            config: Configuración del detector. Si es None, se usa
                la configuración global del sistema.

        Raises:
            ModelLoadError: Si no se puede cargar ningún modelo.
        """
        self.config = config or DetectorConfig.from_global_config()
        self.logger.info("Inicializando OptimizedYOLODetector")

        self.config.device = "cpu"
        self.device = "cpu"

        self.confidence_threshold = self.config.confidence_threshold
        self.iou_threshold = self.config.iou_threshold
        self.vehicle_classes = self.config.vehicle_classes
        self.imgsz = self.config.imgsz
        self.max_det = self.config.max_det

        self._using_optimized = True

        self.logger.info(
            "Configuración del detector",
            confidence=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.vehicle_classes,
        )

        self._init_components()
        self._check_availability()
        self._warmup()

        self.logger.info(
            "OptimizedYOLODetector inicializado",
            onnx_available=self._onnx_available,
            numba_available=self._numba_available,
            warmed_up=self._warmed_up,
        )

    def _init_components(self) -> None:
        """Inicializa todos los componentes del detector optimizado.

        Este método configura:
            - ModelManager: Gestión de modelos (PyTorch/ONNX)
            - ModelExporter: Exportación a ONNX
            - Inference Engines: PyTorch y ONNX
            - PostProcessor: Procesamiento de resultados
            - Cache: Caché LRU
            - Preprocessor: Preprocesamiento de imágenes
        """
        self.model_manager = ModelManager(
            model_path=self.config.model_path,
            device=self.device,
            use_half_precision=self.config.use_half_precision,
            imgsz=self.imgsz,
            vehicle_classes=self.vehicle_classes,
        )

        self.model_exporter = ModelExporter(
            model_path=self.config.model_path,
            imgsz=self.imgsz,
        )

        self._pytorch_engine: InferenceEngine | None = None
        self._onnx_engine: InferenceEngine | None = None

        self.post_processor = PostProcessor(
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            vehicle_classes=self.vehicle_classes,
            imgsz=self.imgsz,
        )

        self.cache = DetectionCache(
            max_size=self._calculate_cache_size(), max_age_seconds=FEATURE_CACHE_MAX_AGE
        )

        self.preprocessor = ImagePreprocessor(enabled=False)

        self._warmed_up = False
        self._onnx_available = False
        self._numba_available = self._check_numba_availability()
        self._inference_times: list[float] = []
        self._total_detections = 0

        self.logger.info("Componentes inicializados")

    def _check_onnx_availability(self) -> bool:
        """Verifica disponibilidad de ONNX Runtime.

        Returns:
            bool: True si ONNX Runtime está disponible.
        """
        try:
            import onnxruntime as ort

            return True
        except ImportError:
            return False

    def _check_numba_availability(self) -> bool:
        """Verifica disponibilidad de Numba.

        Returns:
            bool: True si Numba está disponible.
        """
        try:
            import numba

            return True
        except ImportError:
            return False

    def _check_availability(self) -> None:
        """Verifica la disponibilidad de los motores de inferencia.

        Prioriza ONNX sobre PyTorch para mejor rendimiento en CPU.

        Raises:
            ModelLoadError: Si no se puede cargar ningún modelo.
        """
        if self.config.use_onnx:
            onnx_path = self.config.model_path.replace(".pt", ".onnx")

            if not os.path.exists(onnx_path):
                self.logger.info("Exportando modelo a ONNX...")
                onnx_path = self.model_exporter.export()

            if onnx_path and self.model_manager.load_onnx(onnx_path):
                self._onnx_engine = InferenceEngineFactory.create_onnx(
                    session=self.model_manager.get_onnx_session(),
                    input_name=self.model_manager.get_onnx_input_name(),
                    output_names=self.model_manager.get_onnx_output_names(),
                    imgsz=self.imgsz,
                )
                self._onnx_available = True
                self.logger.info("✅ ONNX disponible")

        if not self._onnx_available:
            self.logger.info("Cargando PyTorch como fallback...")
            if self.model_manager.load_pytorch():
                self._pytorch_engine = InferenceEngineFactory.create_pytorch(
                    model=self.model_manager.get_pytorch_model(),
                    imgsz=self.imgsz,
                    vehicle_classes=self.vehicle_classes,
                    device=self.device,
                    max_det=self.max_det,
                )
                self.logger.info("✅ PyTorch disponible")
            else:
                self.logger.error("❌ No se pudo cargar ningún modelo")
                raise ModelLoadError(f"No se pudo cargar el modelo: {self.config.model_path}")

    def _warmup(self) -> None:
        """Calienta los motores de inferencia para reducir latencia inicial.

        Note:
            Ejecuta inferencias dummy en ambos motores para asegurar
            que todas las optimizaciones estén activas.
        """
        if self._warmed_up:
            return

        self.logger.info("🔥 Ejecutando warmup...")

        try:
            if self._onnx_engine:
                self._onnx_engine.warmup()

            if self._pytorch_engine:
                self._pytorch_engine.warmup()

            self._warmed_up = True
            self.logger.info("✅ Warmup completado")

        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detecta objetos en un frame usando el motor optimizado.

        Args:
            frame: Imagen a procesar en formato numpy array (H, W, C) BGR.

        Returns:
            List[Dict[str, Any]]: Lista de detecciones validadas.

        Note:
            - Usa ONNX si está disponible, de lo contrario PyTorch
            - Aplica caché LRU para detecciones repetidas
            - Realiza preprocesamiento automático
        """
        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()

        if self.config.use_onnx:
            cached = self._check_cache(frame)
            if cached is not None:
                return cached

        processed = self._preprocess(frame)

        detections = self._infer(processed, frame.shape)

        if self.config.use_onnx and detections:
            self._cache_detections(frame, detections)

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        if len(self._inference_times) > INFERENCE_TIMES_MAX:
            self._inference_times = self._inference_times[-INFERENCE_TIMES_MAX:]

        self._total_detections += len(detections)

        return detections

    def _check_cache(self, frame: np.ndarray) -> list[dict[str, Any]] | None:
        """Verifica el caché de detecciones.

        Args:
            frame: Frame para buscar en caché.

        Returns:
            Optional[List[Dict[str, Any]]]: Detecciones cacheadas o None.
        """
        try:
            key = self.cache.compute_key(frame)
            cached = self.cache.get(key)
            if cached is not None:
                self.logger.debug(f"Cache hit: {len(cached)} detecciones")
                return cached
        except Exception as e:
            self.logger.debug(f"Error en caché: {e}")
        return None

    def _cache_detections(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
        """Almacena detecciones en caché.

        Args:
            frame: Frame original para calcular la clave.
            detections: Detecciones a almacenar.
        """
        try:
            key = self.cache.compute_key(frame)
            self.cache.put(key, detections)
        except Exception as e:
            self.logger.debug(f"Error guardando en caché: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa el frame para mejorar la detección.

        Args:
            frame: Frame original.

        Returns:
            np.ndarray: Frame preprocesado.
        """
        try:
            return self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning(f"Error en preprocesamiento: {e}")
            return frame

    def _infer(self, frame: np.ndarray, original_shape: tuple) -> list[dict[str, Any]]:
        """Realiza inferencia usando el motor disponible.

        Args:
            frame: Frame preprocesado.
            original_shape: Shape original de la imagen (height, width).

        Returns:
            List[Dict[str, Any]]: Detecciones procesadas.
        """
        if self._onnx_engine and self._onnx_engine.is_available:
            try:
                output = self._onnx_engine.infer(frame)
                if output is not None and len(output) > 0:
                    return self.post_processor.process_onnx_output(output, original_shape[:2])
            except Exception as e:
                self.logger.warning(f"Error en ONNX: {e}, usando PyTorch")

        if self._pytorch_engine and self._pytorch_engine.is_available:
            try:
                results = self._pytorch_engine.infer(frame)
                if results is not None:
                    return self.post_processor.process_pytorch_results(results, original_shape[:2])
            except Exception as e:
                self.logger.error(f"Error en PyTorch: {e}")

        return []

    def _calculate_cache_size(self) -> int:
        """Calcula el tamaño óptimo del caché."""
        try:
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(CACHE_MIN_SIZE, min(CACHE_MAX_SIZE, size))
        except Exception:
            return CACHE_DEFAULT_SIZE

    def get_performance_stats(self) -> dict[str, Any]:
        """Retorna estadísticas de rendimiento del detector optimizado.

        Returns:
            Dict[str, Any]: Estadísticas incluyendo:
                - total_detections: Total de detecciones
                - avg_inference_time_ms: Tiempo promedio de inferencia
                - samples: Número de muestras
                - device: Dispositivo utilizado
                - onnx_available: Si ONNX está disponible
                - numba_available: Si Numba está disponible
                - warmed_up: Si el warmup se completó
                - cache: Estadísticas del caché
                - preprocessor: Estadísticas del preprocesador
                - post_processor: Estadísticas del post-procesador
        """
        avg_time = np.mean(self._inference_times) if self._inference_times else 0

        return {
            "total_detections": self._total_detections,
            "avg_inference_time_ms": avg_time,
            "samples": len(self._inference_times),
            "device": self.device,
            "onnx_available": self._onnx_available,
            "numba_available": self._numba_available,
            "warmed_up": self._warmed_up,
            "cache": self.cache.get_stats(),
            "preprocessor": self.preprocessor.get_stats(),
            "post_processor": self.post_processor.get_stats(),
        }

    def clear_cache(self) -> None:
        """Limpia el caché de detecciones."""
        self.cache.clear()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Activa/desactiva el preprocesamiento."""
        self.preprocessor.set_enabled(enable)
