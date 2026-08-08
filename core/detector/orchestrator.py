"""Orquestador del sistema de detección.

Este módulo coordina todos los componentes del sistema de detección:
- Gestión de modelos (PyTorch/ONNX)
- Motores de inferencia
- Preprocesamiento y post-procesamiento
- Caché de detecciones
- Estadísticas de rendimiento

El orquestador sigue el patrón de diseño "Facade", proporcionando
una interfaz simplificada para el sistema de detección.
"""

import os
import time
from typing import Any

import numpy as np

from core.constants.pipeline import MEMORY_CHECK_INTERVAL
from core.constants.system import MEMORY_WARNING_THRESHOLD
from core.constants.tracking import CACHE_DEFAULT_SIZE, CACHE_MAX_SIZE, CACHE_MIN_SIZE
from core.constants.values import BATCH_TIMES_MAX, INFERENCE_TIMES_MAX
from core.detector.cache import DetectionCache
from core.detector.config import DetectorConfig
from core.detector.inference_engine import (
    InferenceEngineFactory,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
)
from core.detector.model_exporter import ModelExporter
from core.detector.model_manager import ModelManager
from core.detector.post_processor import PostProcessor
from core.detector.preprocessor import ImagePreprocessor
from core.exceptions import DetectionError, ModelLoadError
from core.types import DetectionList
from utils.helpers import get_memory_usage
from utils.logger import LoggerMixin


class DetectorOrchestrator(LoggerMixin):
    """Orquestador del sistema de detección.

    Esta clase coordina todos los componentes del sistema de detección,
    proporcionando una interfaz unificada para el procesamiento de frames.

    Características:
        - Gestión de modelos (PyTorch/ONNX)
        - Múltiples motores de inferencia
        - Caché LRU para detecciones
        - Preprocesamiento y post-procesamiento
        - Estadísticas de rendimiento
        - Gestión automática de memoria

    Attributes:
        config: Configuración del detector.
        device: Dispositivo de inferencia.
        model_manager: Gestor de modelos.
        pytorch_engine: Motor de inferencia PyTorch.
        onnx_engine: Motor de inferencia ONNX.
        post_processor: Post-procesador de detecciones.
        cache: Caché de detecciones.
        preprocessor: Preprocesador de imágenes.

    Example:
        >>> orchestrator = DetectorOrchestrator(config)
        >>> frame = cv2.imread("image.jpg")
        >>> detections = orchestrator.detect(frame)
        >>> for det in detections:
        ...     print(f"Objeto: {det['label']} confianza: {det['confidence']:.2f}")
    """

    def __init__(self, config: DetectorConfig):
        """Inicializa el orquestador de detección.

        Args:
            config: Configuración del detector.

        Raises:
            ModelLoadError: Si no se puede cargar ningún modelo.
        """
        self.config = config
        self.logger.info("Inicializando DetectorOrchestrator")

        self.device = self._get_device()

        self._init_model_manager()
        self._init_engines()
        self._init_post_processor()
        self._init_cache()
        self._init_preprocessor()

        self._inference_times: list[float] = []
        self._batch_inference_times: list[float] = []
        self._total_detections: int = 0
        self._total_batches: int = 0
        self._last_memory_check: float = time.time()
        self._warmed_up: bool = False

        self._warmup()

        self.logger.info(
            "DetectorOrchestrator inicializado",
            device=self.device,
            cache_size=self.cache.max_size,
            onnx_available=self._onnx_available,
            numba_available=self._check_numba_availability(),
        )

    def _get_device(self) -> str:
        """Obtiene el dispositivo óptimo para inferencia.

        Returns:
            str: Dispositivo seleccionado ('cpu', 'cuda', 'mps').
        """
        device = self.config.device

        if hasattr(device, "value"):
            device = device.value

        if device == "auto" or device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        if device not in ["cpu", "cuda", "mps"]:
            self.logger.warning(f"Dispositivo desconocido: {device}, usando CPU")
            return "cpu"

        return str(device)

    def _init_model_manager(self) -> None:
        """Inicializa el gestor de modelos."""
        device_str = self.device if isinstance(self.device, str) else str(self.device)

        self.model_manager = ModelManager(
            model_path=self.config.model_path,
            device=device_str,
            use_half_precision=self.config.use_half_precision,
            imgsz=self.config.imgsz,
            vehicle_classes=self.config.vehicle_classes,
        )

        self.model_exporter = ModelExporter(
            model_path=self.config.model_path,
            imgsz=self.config.imgsz,
        )

    def _init_engines(self) -> None:
        """Inicializa los motores de inferencia."""
        self.pytorch_engine: PyTorchInferenceEngine | None = None
        self.onnx_engine: ONNXInferenceEngine | None = None
        self._onnx_available = False

        if self.config.use_onnx:
            onnx_path = self.config.model_path.replace(".pt", ".onnx")

            if not os.path.exists(onnx_path):
                self.logger.info("Exportando modelo a ONNX...")
                onnx_path = self.model_exporter.export()

            if onnx_path and self.model_manager.load_onnx(onnx_path):
                self.onnx_engine = InferenceEngineFactory.create_onnx(
                    session=self.model_manager.get_onnx_session(),
                    input_name=self.model_manager.get_onnx_input_name(),
                    output_names=self.model_manager.get_onnx_output_names(),
                    imgsz=self.config.imgsz,
                )
                self._onnx_available = True
                self.logger.info("✅ ONNX disponible")

        if not self._onnx_available:
            self.logger.info("Cargando PyTorch como fallback...")
            if self.model_manager.load_pytorch():
                self.pytorch_engine = InferenceEngineFactory.create_pytorch(
                    model=self.model_manager.get_pytorch_model(),
                    imgsz=self.config.imgsz,
                    vehicle_classes=self.config.vehicle_classes,
                    device=self.device,
                    max_det=self.config.max_det,
                )
                self.logger.info("✅ PyTorch disponible")
            else:
                self.logger.error("❌ No se pudo cargar ningún modelo")
                raise ModelLoadError(f"No se pudo cargar el modelo: {self.config.model_path}")

    def _init_post_processor(self) -> None:
        """Inicializa el post-procesador."""
        self.post_processor = PostProcessor(
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            vehicle_classes=self.config.vehicle_classes,
            imgsz=self.config.imgsz,
        )

    def _init_cache(self) -> None:
        """Inicializa el caché de detecciones."""
        cache_size = self._calculate_cache_size()
        self.cache = DetectionCache(
            max_size=cache_size,
            max_age_seconds=3.0,
        )

    def _init_preprocessor(self) -> None:
        """Inicializa el preprocesador de imágenes."""
        self.preprocessor = ImagePreprocessor(enabled=False)

    def _check_numba_availability(self) -> bool:
        """Verifica disponibilidad de Numba."""
        try:
            import numba

            return True
        except ImportError:
            return False

    def _calculate_cache_size(self) -> int:
        """Calcula el tamaño óptimo del caché basado en memoria disponible."""
        try:
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(CACHE_MIN_SIZE, min(CACHE_MAX_SIZE, size))
        except Exception:
            return CACHE_DEFAULT_SIZE

    def _warmup(self) -> None:
        """Calienta los motores de inferencia para reducir latencia inicial."""
        if self._warmed_up:
            return

        self.logger.info("🔥 Ejecutando warmup...")

        try:
            if self.onnx_engine:
                self.onnx_engine.warmup()

            if self.pytorch_engine:
                self.pytorch_engine.warmup()

            self._warmed_up = True
            self.logger.info("✅ Warmup completado")

        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detecta objetos en un frame.

        Args:
            frame: Imagen a procesar en formato numpy array (H, W, C) BGR.

        Returns:
            DetectionList: Lista de detecciones validadas.

        Raises:
            DetectionError: Si ocurre un error durante la inferencia.
        """
        if not self._validate_frame(frame):
            return []

        self._check_memory()
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
        self._update_stats(inference_time, len(detections))

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionList]:
        """Detecta objetos en múltiples frames (batch inference).

        Args:
            frames: Lista de imágenes a procesar.

        Returns:
            List[DetectionList]: Lista de listas de detecciones.
        """
        if not frames:
            return []

        valid_indices = []
        valid_frames = []
        for i, frame in enumerate(frames):
            if self._validate_frame(frame):
                valid_indices.append(i)
                valid_frames.append(frame)

        if not valid_frames:
            return [[] for _ in frames]

        start_time = time.perf_counter()

        try:
            processed_frames = []
            for frame in valid_frames:
                try:
                    processed = self.preprocessor.process(frame)
                    processed_frames.append(processed)
                except Exception as e:
                    self.logger.warning("Error en preprocesamiento", error=str(e))
                    processed_frames.append(frame)

            if self.onnx_engine:
                all_detections = []
                for processed in processed_frames:
                    detections = self._infer(processed, frame.shape)
                    all_detections.append(detections)
            else:
                results = self.pytorch_engine.model(
                    processed_frames,
                    classes=self.config.vehicle_classes,
                    verbose=False,
                    augment=False,
                    imgsz=self.config.imgsz,
                    device=self.device,
                    max_det=self.config.max_det,
                )

                all_detections = []
                for result in results:
                    detections = self.post_processor.process_pytorch_results(
                        result, processed_frames[0].shape[:2]
                    )
                    all_detections.append(detections)

            result_list = [[] for _ in frames]
            for idx, detections in zip(valid_indices, all_detections):
                result_list[idx] = detections

            batch_time = (time.perf_counter() - start_time) * 1000
            self._batch_inference_times.append(batch_time)
            self._total_batches += 1
            self._total_detections += sum(len(d) for d in all_detections)

            if len(self._batch_inference_times) > BATCH_TIMES_MAX:
                self._batch_inference_times = self._batch_inference_times[-BATCH_TIMES_MAX:]

            return result_list

        except Exception as e:
            self.logger.error(f"Error en batch inference: {e}", exc_info=True)
            return [[] for _ in frames]

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida que el frame sea válido para procesamiento."""
        if frame is None or frame.size == 0:
            return False

        if len(frame.shape) not in (2, 3):
            return False

        h, w = frame.shape[:2]
        return not (h < 10 or w < 10)

    def _check_cache(self, frame: np.ndarray) -> DetectionList | None:
        """Verifica si el frame está en caché."""
        try:
            key = self.cache.compute_key(frame)
            cached = self.cache.get(key)
            if cached is not None:
                self.logger.debug(f"Cache hit: {len(cached)} detecciones")
                return cached
        except Exception as e:
            self.logger.debug(f"Error en caché: {e}")
        return None

    def _cache_detections(self, frame: np.ndarray, detections: DetectionList) -> None:
        """Almacena detecciones en caché."""
        try:
            key = self.cache.compute_key(frame)
            self.cache.put(key, detections)
        except Exception as e:
            self.logger.debug(f"Error guardando en caché: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa el frame para mejorar la detección."""
        try:
            return self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning(f"Error en preprocesamiento: {e}")
            return frame

    def _infer(self, frame: np.ndarray, original_shape: tuple[int, ...]) -> DetectionList:
        """Realiza inferencia usando el motor disponible."""
        if self.onnx_engine and self.onnx_engine.is_available:
            try:
                output = self.onnx_engine.infer(frame)
                if output is not None and len(output) > 0:
                    return self.post_processor.process_onnx_output(output, original_shape[:2])
            except Exception as e:
                self.logger.warning(f"Error en ONNX: {e}, usando PyTorch")

        if self.pytorch_engine and self.pytorch_engine.is_available:
            try:
                results = self.pytorch_engine.infer(frame)
                if results is not None:
                    return self.post_processor.process_pytorch_results(results, original_shape[:2])
            except Exception as e:
                self.logger.error(f"Error en PyTorch: {e}")
                raise DetectionError(
                    "Error en inferencia del modelo",
                    details={"error": str(e), "frame_shape": original_shape},
                ) from e

        return []

    def _update_stats(self, inference_time: float, num_detections: int) -> None:
        """Actualiza estadísticas de rendimiento."""
        self._inference_times.append(inference_time)
        self._total_detections += num_detections

        if len(self._inference_times) > INFERENCE_TIMES_MAX:
            self._inference_times = self._inference_times[-INFERENCE_TIMES_MAX:]

    def get_performance_stats(self) -> dict[str, Any]:
        """Retorna estadísticas de rendimiento del detector."""
        import numpy as np

        avg_time = np.mean(self._inference_times) if self._inference_times else 0
        avg_batch_time = np.mean(self._batch_inference_times) if self._batch_inference_times else 0

        return {
            "total_detections": self._total_detections,
            "avg_inference_time_ms": avg_time,
            "avg_batch_time_ms": avg_batch_time,
            "total_batches": self._total_batches,
            "samples": len(self._inference_times),
            "device": self.device,
            "onnx_available": self._onnx_available,
            "numba_available": self._check_numba_availability(),
            "warmed_up": self._warmed_up,
            "cache": self.cache.get_stats(),
            "preprocessor": self.preprocessor.get_stats(),
            "post_processor": self.post_processor.get_stats(),
        }

    def get_classes(self) -> list[int]:
        """Retorna las clases que detecta el modelo."""
        return self.config.vehicle_classes

    def _check_memory(self) -> None:
        """Verifica uso de memoria y limpia caché si es necesario."""
        current_time = time.time()
        if current_time - self._last_memory_check < MEMORY_CHECK_INTERVAL:
            return

        self._last_memory_check = current_time

        try:
            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > MEMORY_WARNING_THRESHOLD:
                self.logger.warning(
                    "Memoria alta, limpiando caché",
                    memory_percent=f"{mem_percent:.1f}",
                    cache_size=len(self.cache),
                )
                self.cache.clear()
        except Exception as e:
            self.logger.debug(f"Error verificando memoria: {e}")

    def clear_cache(self) -> None:
        """Limpia el caché de detecciones."""
        self.cache.clear()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Activa o desactiva el preprocesamiento de imágenes."""
        self.preprocessor.set_enabled(enable)

    @property
    def model(self):
        """Retorna el modelo activo (para compatibilidad)."""
        if self.pytorch_engine:
            return self.pytorch_engine.model
        return None

    @property
    def is_onnx_available(self) -> bool:
        """Retorna si ONNX está disponible."""
        return self._onnx_available

    @property
    def is_numba_available(self) -> bool:
        """Retorna si Numba está disponible."""
        return self._check_numba_availability()
