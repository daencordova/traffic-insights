"""
Detector de objetos YOLO con optimizaciones.

Implementa el detector base con caché, preprocesamiento y estadísticas.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from core.detector.cache import DetectionCache
from core.detector.preprocessor import ImagePreprocessor
from core.detector.config import DetectorConfig
from core.exceptions import DetectionError, ModelLoadError
from utils.geometry import calculate_centroid
from utils.helpers import get_memory_usage
from utils.logger import LoggerMixin
from core.validators import validate_frame, validate_bbox, validate_centroid
from core.constants import (
    MIN_BOX_SIZE,
    MAX_BOX_SIZE,
    MIN_DETECTION_AREA,
    MAX_DETECTION_AREA,
    MIN_DETECTION_CONFIDENCE,
    MAX_DETECTION_CONFIDENCE,
    MEMORY_CHECK_INTERVAL,
    MEMORY_WARNING_THRESHOLD
)
from core.interfaces import IDetector

Detection = Dict[str, Any]
DetectionList = List[Detection]
BoundingBox = Tuple[int, int, int, int]


class YOLODetector(IDetector, LoggerMixin):
    """
    Detector YOLO con caché y preprocesamiento.

    Características:
    - Detección de objetos con YOLO
    - Caché LRU para detecciones
    - Preprocesamiento de imágenes
    - Estadísticas de rendimiento
    - Soporte para batch inference

    Attributes:
        config: Configuración del detector
        device: Dispositivo de inferencia
        model: Modelo YOLO
        cache: Caché de detecciones
        preprocessor: Preprocesador de imágenes
    """

    MIN_BOX_SIZE: int = MIN_BOX_SIZE
    MAX_BOX_SIZE: int = MAX_BOX_SIZE
    MIN_CONFIDENCE: float = MIN_DETECTION_CONFIDENCE
    MAX_CONFIDENCE: float = MAX_DETECTION_CONFIDENCE
    MIN_AREA: int = MIN_DETECTION_AREA
    MAX_AREA: int = MAX_DETECTION_AREA

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Inicializa el detector YOLO.

        Args:
            config: Configuración del detector (opcional)
        """
        self.config = config or DetectorConfig.from_global_config()
        self.logger.info("Inicializando YOLODetector", model=self.config.model_path)

        self.device: str = self._get_device()
        self.model: YOLO = self._load_model()

        self.model.conf = self.config.confidence_threshold
        self.model.iou = self.config.iou_threshold
        self.model.classes = self.config.vehicle_classes

        self.cache = DetectionCache(
            max_size=self._calculate_cache_size(),
            max_age_seconds=3.0
        )

        self.preprocessor = ImagePreprocessor(enabled=False)

        self._inference_times: List[float] = []
        self._batch_inference_times: List[float] = []
        self._total_detections: int = 0
        self._total_batches: int = 0
        self._last_memory_check: float = time.time()

        self._print_startup_info()
        self.logger.info(
            "YOLODetector inicializado",
            device=self.device,
            cache_size=self.cache.max_size
        )

    def _get_device(self) -> str:
        """Obtiene el dispositivo óptimo."""
        device = self.config.device

        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        return str(device)

    def _load_model(self) -> YOLO:
        """Carga el modelo YOLO."""
        self.logger.info("Cargando modelo", path=self.config.model_path)

        try:
            model = YOLO(self.config.model_path)
        except FileNotFoundError as e:
            raise ModelLoadError(f"Modelo no encontrado: {self.config.model_path}", {
                "model_path": self.config.model_path,
                "error": str(e)
            })
        except Exception as e:
            raise ModelLoadError(f"Error cargando modelo {self.config.model_path}", {
                "model_path": self.config.model_path,
                "error": str(e)
            })

        if self.device != "cpu":
            try:
                model.to(self.device)
                self.logger.debug("Modelo movido a dispositivo", device=self.device)
            except Exception as e:
                self.logger.warning(
                    "No se pudo mover a dispositivo, continuando en CPU",
                    device=self.device,
                    error=str(e)
                )
                self.device = "cpu"

        if self.config.use_half_precision and self.device != "cpu":
            try:
                model.model.half()
                self.logger.info("Half precision activado")
            except Exception as e:
                self.logger.warning("No se pudo activar half precision", error=str(e))

        if self.config.use_onnx:
            try:
                onnx_path = self._export_to_onnx(model)
                if onnx_path:
                    model = YOLO(onnx_path)
                    self.logger.info("ONNX cargado", path=onnx_path)
            except Exception as e:
                self.logger.warning("Error en ONNX", error=str(e))

        return model

    def _export_to_onnx(self, model: YOLO) -> Optional[str]:
        """Exporta modelo a ONNX."""
        import os

        onnx_path = self.config.model_path.replace(".pt", ".onnx")

        if not os.path.exists(onnx_path):
            try:
                model.export(
                    format="onnx",
                    imgsz=self.config.imgsz,
                    optimize=True,
                    opset=12,
                    simplify=True,
                )
                self.logger.info("Modelo exportado a ONNX", path=onnx_path)
            except Exception as e:
                self.logger.warning("Error exportando a ONNX", error=str(e))
                return None

        return onnx_path

    def _calculate_cache_size(self) -> int:
        """Calcula el tamaño óptimo del caché."""
        try:
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(4, min(64, size))
        except Exception:
            return 16

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida que el frame sea válido."""
        return validate_frame(frame, min_width=10, min_height=10)

    def _validate_box(self, box: Any) -> bool:
        """Valida un bounding box."""
        return validate_bbox(box, min_size=self.MIN_BOX_SIZE, max_size=self.MAX_BOX_SIZE)

    def _validate_centroid(self, centroid: Any) -> bool:
        """Valida un centroide."""
        return validate_centroid(centroid)

    def _validate_confidence(self, confidence: Any) -> bool:
        """Valida un valor de confianza."""
        if not isinstance(confidence, (int, float)):
            return False
        return self.MIN_CONFIDENCE <= confidence <= self.MAX_CONFIDENCE

    def _validate_detection(self, detection: Dict[str, Any]) -> bool:
        """Valida una detección completa."""
        from core.validators import validate_detection
        result = validate_detection(detection, min_confidence=0.0)
        return result.is_valid

    def _filter_valid_detections(self, detections: DetectionList) -> DetectionList:
        """Filtra detecciones válidas."""
        valid = [d for d in detections if self._validate_detection(d)]
        if len(valid) != len(detections):
            self.logger.debug(
                "Detecciones filtradas",
                total=len(detections),
                valid=len(valid),
                invalid=len(detections) - len(valid)
            )
        return valid

    def _parse_results(self, result) -> DetectionList:
        """Parsea resultados de YOLO."""
        detections = []

        if result is None or result.boxes is None:
            return detections

        for box in result.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)
                if area < self.MIN_AREA:
                    continue

                centroid = calculate_centroid(x1, y1, x2, y2)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "centroid": centroid,
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": self.model.names[class_id],
                    "area": area,
                })
            except Exception as e:
                self.logger.debug("Error parseando box", error=str(e))
                continue

        return detections


    def _check_memory(self) -> None:
        """Verifica uso de memoria."""
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
                    cache_size=len(self.cache)
                )
                self.cache.clear()
        except Exception as e:
            self.logger.debug("Error verificando memoria", error=str(e))

    def detect(self, frame: np.ndarray) -> DetectionList:
        """
        Detecta objetos en un frame.

        Args:
            frame: Imagen a procesar

        Returns:
            Lista de detecciones validadas
        """
        if not self._validate_frame(frame):
            return []

        self._check_memory()
        start_time = time.perf_counter()

        if self.config.use_onnx:
            try:
                key = self.cache.compute_key(frame)
                cached = self.cache.get(key)
                if cached is not None:
                    return self._filter_valid_detections(cached)
            except Exception as e:
                self.logger.warning("Error en caché", error=str(e))

        try:
            processed = self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning("Error en preprocesamiento", error=str(e))
            processed = frame

        try:
            results = self.model(
                processed,
                classes=self.config.vehicle_classes,
                verbose=False,
                augment=False,
                imgsz=self.config.imgsz,
                device=self.device,
                max_det=self.config.max_det,
            )
        except Exception as e:
            raise DetectionError("Error en inferencia del modelo", {
                "frame_shape": frame.shape,
                "error": str(e)
            })

        try:
            detections = self._parse_results(results[0])
        except Exception as e:
            self.logger.warning("Error parseando resultados", error=str(e))
            detections = []

        valid_detections = self._filter_valid_detections(detections)

        if self.config.use_onnx:
            try:
                self.cache.put(key, valid_detections)
            except Exception as e:
                self.logger.warning("Error guardando en caché", error=str(e))

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        self._total_detections += len(valid_detections)

        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]

        if valid_detections:
            self.logger.debug(
                "Detecciones completadas",
                count=len(valid_detections),
                time_ms=f"{inference_time:.1f}"
            )

        return valid_detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionList]:
        """
        Detecta objetos en múltiples frames.

        Args:
            frames: Lista de imágenes

        Returns:
            Lista de listas de detecciones
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

            results = self.model(
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
                detections = self._parse_results(result)
                valid_detections = self._filter_valid_detections(detections)
                all_detections.append(valid_detections)

            result_list = [[] for _ in frames]
            for idx, detections in zip(valid_indices, all_detections):
                result_list[idx] = detections

            batch_time = (time.perf_counter() - start_time) * 1000
            self._batch_inference_times.append(batch_time)
            self._total_batches += 1
            self._total_detections += sum(len(d) for d in all_detections)

            if len(self._batch_inference_times) > 50:
                self._batch_inference_times = self._batch_inference_times[-50:]

            self.logger.debug(
                "Batch inference completado",
                batch_size=len(valid_frames),
                total_detections=sum(len(d) for d in all_detections),
                time_ms=f"{batch_time:.1f}"
            )

            return result_list

        except Exception as e:
            self.logger.error(f"Error en batch inference: {e}", exc_info=True)
            return [[] for _ in frames]

    def get_classes(self) -> List[int]:
        """Retorna las clases que detecta."""
        return self.config.vehicle_classes

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de rendimiento."""
        avg_time = np.mean(self._inference_times) if self._inference_times else 0
        avg_batch_time = np.mean(self._batch_inference_times) if self._batch_inference_times else 0

        return {
            "total_detections": self._total_detections,
            "avg_inference_time_ms": avg_time,
            "avg_batch_time_ms": avg_batch_time,
            "total_batches": self._total_batches,
            "samples": len(self._inference_times),
            "device": self.device,
            "cache": self.cache.get_stats(),
            "preprocessor": self.preprocessor.get_stats(),
        }

    def clear_cache(self) -> None:
        """Limpia el caché de detecciones."""
        self.cache.clear()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Activa/desactiva el preprocesamiento."""
        self.preprocessor.set_enabled(enable)

    def _print_startup_info(self) -> None:
        """Imprime información de inicio usando logger."""
        self.logger.info("=" * 60)
        self.logger.info("🤖 DETECTOR YOLO")
        self.logger.info("=" * 60)
        self.logger.info(f"📁 Modelo: {self.config.model_path}")
        self.logger.info(f"🎯 Dispositivo: {self.device}")
        self.logger.info(f"⚡ Half precision: {'✅' if self.config.use_half_precision else '❌'}")
        self.logger.info(f"💾 Caché: {'✅' if self.config.use_onnx else '❌'}")
        self.logger.info(f"📐 IMG Size: {self.config.imgsz}")
        self.logger.info(f"🧠 Memoria caché: {self.cache.max_size} entradas")
        self.logger.info("=" * 60)
