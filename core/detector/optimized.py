"""
Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

Orquesta los componentes de gestión de modelos, inferencia,
post-procesamiento y caché.
"""

import os
import time
from typing import Optional, List, Dict, Any

import numpy as np

from core.detector.base import YOLODetector
from core.detector.model_manager import ModelManager, ModelLoadError
from core.detector.model_exporter import ModelExporter
from core.detector.inference_engine import (
    InferenceEngine,
    InferenceEngineFactory
)
from core.detector.post_processor import PostProcessor
from core.detector.cache import DetectionCache
from core.detector.preprocessor import ImagePreprocessor
from core.detector.config import DetectorConfig


class OptimizedYOLODetector(YOLODetector):
    """
    Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

    Orquesta los componentes de gestión de modelos, inferencia,
    post-procesamiento y caché para máxima eficiencia en CPU.

    Características:
    - ONNX Runtime para inferencia rápida en CPU
    - Numba para NMS optimizado
    - Caché LRU para detecciones
    - Warmup automático
    - Fallback a PyTorch si ONNX no está disponible
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Inicializa el detector optimizado."""
        self.config = config or DetectorConfig.from_global_config()
        self.logger.info("Inicializando OptimizedYOLODetector")

        self.config.device = "cpu"
        self.device = "cpu"

        self.confidence_threshold = self.config.confidence_threshold
        self.iou_threshold = self.config.iou_threshold
        self.vehicle_classes = self.config.vehicle_classes
        self.imgsz = self.config.imgsz
        self.max_det = self.config.max_det

        self.logger.info(
            "Configuración del detector",
            confidence=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.vehicle_classes
        )

        self._init_components()
        self._check_availability()
        self._warmup()

        self.logger.info(
            "OptimizedYOLODetector inicializado",
            onnx_available=self._onnx_available,
            numba_available=self._numba_available,
            warmed_up=self._warmed_up
        )

    def _init_components(self) -> None:
        """Inicializa todos los componentes."""
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

        self._pytorch_engine: Optional[InferenceEngine] = None
        self._onnx_engine: Optional[InferenceEngine] = None

        self.post_processor = PostProcessor(
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            vehicle_classes=self.vehicle_classes,
            imgsz=self.imgsz,
        )

        self.cache = DetectionCache(
            max_size=self._calculate_cache_size(),
            max_age_seconds=3.0
        )

        self.preprocessor = ImagePreprocessor(enabled=False)

        self._warmed_up = False
        self._onnx_available = False
        self._numba_available = self._check_numba()
        self._inference_times: List[float] = []
        self._total_detections = 0

        self.logger.info("Componentes inicializados")

    def _check_numba(self) -> bool:
        """Verifica si Numba está disponible."""
        try:
            import numba
            return True
        except ImportError:
            return False

    def _check_availability(self) -> None:
        """Verifica la disponibilidad de los motores de inferencia."""
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
        """Calienta los motores de inferencia."""
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

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta objetos en un frame.

        Args:
            frame: Imagen a procesar

        Returns:
            List[Dict[str, Any]]: Detecciones
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
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]

        self._total_detections += len(detections)

        return detections

    def _check_cache(self, frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """Verifica el caché de detecciones."""
        try:
            key = self.cache.compute_key(frame)
            cached = self.cache.get(key)
            if cached is not None:
                self.logger.debug(f"Cache hit: {len(cached)} detecciones")
                return cached
        except Exception as e:
            self.logger.debug(f"Error en caché: {e}")
        return None

    def _cache_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Almacena detecciones en caché."""
        try:
            key = self.cache.compute_key(frame)
            self.cache.put(key, detections)
        except Exception as e:
            self.logger.debug(f"Error guardando en caché: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa el frame."""
        try:
            return self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning(f"Error en preprocesamiento: {e}")
            return frame

    def _infer(self, frame: np.ndarray, original_shape: tuple) -> List[Dict[str, Any]]:
        """
        Realiza inferencia usando el motor disponible.

        Args:
            frame: Frame preprocesado
            original_shape: Shape original de la imagen

        Returns:
            List[Dict[str, Any]]: Detecciones
        """
        if self._onnx_engine and self._onnx_engine.is_available:
            try:
                output = self._onnx_engine.infer(frame)
                if output is not None and len(output) > 0:
                    return self.post_processor.process_onnx_output(
                        output,
                        original_shape[:2]
                    )
            except Exception as e:
                self.logger.warning(f"Error en ONNX: {e}, usando PyTorch")

        if self._pytorch_engine and self._pytorch_engine.is_available:
            try:
                results = self._pytorch_engine.infer(frame)
                if results is not None:
                    return self.post_processor.process_pytorch_results(
                        results,
                        original_shape[:2]
                    )
            except Exception as e:
                self.logger.error(f"Error en PyTorch: {e}")

        return []

    def _calculate_cache_size(self) -> int:
        """Calcula el tamaño óptimo del caché."""
        try:
            from utils.helpers import get_memory_usage
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(4, min(64, size))
        except Exception:
            return 16

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de rendimiento."""
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
