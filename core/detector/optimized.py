"""
Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

Versión optimizada del detector para ejecución en CPU con soporte
para ONNX Runtime y aceleración con Numba.
"""

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from core.detector.base import YOLODetector
from core.detector.cache import DetectionCache
from core.detector.preprocessor import ImagePreprocessor
from core.detector.config import DetectorConfig
from core.detector.base import DetectionList
from utils.geometry import calculate_centroid

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


@jit(nopython=True, cache=True)
def _nms_fast(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """NMS optimizado con Numba."""
    if len(detections) == 0:
        return np.array([], dtype=np.int64)

    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]

    return np.array(keep, dtype=np.int64)


class OptimizedYOLODetector(YOLODetector):
    """
    Detector YOLO optimizado para CPU con ONNX Runtime y Numba.

    Características adicionales:
    - Inferencia con ONNX Runtime (más rápido en CPU)
    - NMS optimizado con Numba
    - Fallback automático a PyTorch
    - Perfilamiento detallado
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Inicializa el detector optimizado."""
        self.config = config or DetectorConfig.from_global_config()
        self.logger.info("Inicializando OptimizedYOLODetector para CPU")

        self.config.device = "cpu"
        self.device = "cpu"

        self.confidence_threshold = self.config.confidence_threshold
        self.iou_threshold = self.config.iou_threshold
        self.vehicle_classes = self.config.vehicle_classes
        self.imgsz = self.config.imgsz
        self.max_det = self.config.max_det

        self.logger.info(f"🎯 Usando confianza: {self.confidence_threshold}")
        self.logger.info(f"📊 IOU: {self.iou_threshold}")
        self.logger.info(f"📐 IMG Size: {self.imgsz}")

        self.cache = DetectionCache(
            max_size=self._calculate_cache_size(),
            max_age_seconds=3.0
        )
        self.preprocessor = ImagePreprocessor(enabled=False)

        self.model: Optional[YOLO] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self._onnx_available = False
        self._yolo_available = False
        self._warmed_up = False

        self._diagnostic_stats = {
            "frames_processed": 0,
            "frames_with_detections": 0,
            "total_boxes_before_nms": 0,
            "total_boxes_after_nms": 0,
            "onnx_used": False,
            "yolo_used": False,
            "onnx_export_attempted": False,
            "onnx_export_success": False,
        }
        self._inference_times: List[float] = []
        self._total_detections: int = 0
        self._last_memory_check: float = time.time()

        self._load_model()

        self.logger.info(
            "OptimizedYOLODetector inicializado",
            onnx=self._onnx_available,
            numba=NUMBA_AVAILABLE,
            warmed_up=self._warmed_up
        )

    def _load_model(self) -> None:
        """Carga el modelo YOLO optimizado para CPU."""
        model_path = self.config.model_path
        onnx_path = model_path.replace(".pt", ".onnx")

        self.logger.info("=" * 60)
        self.logger.info(f"📁 Cargando modelo desde: {model_path}")
        self.logger.info(f"🎯 Dispositivo: {self.device}")
        self.logger.info("=" * 60)

        if not os.path.exists(model_path):
            self.logger.error(f"❌ Modelo no encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        onnx_loaded = False
        if self.config.use_onnx and ONNX_AVAILABLE:
            self.logger.info("📦 Intentando cargar modelo ONNX...")

            if os.path.exists(onnx_path):
                if self._load_onnx_model(onnx_path):
                    self._onnx_available = True
                    self._diagnostic_stats["onnx_used"] = True
                    onnx_loaded = True
                    self.logger.info("✅ Modelo ONNX cargado correctamente")
            else:
                self.logger.info(f"   ⚠️ Archivo ONNX no encontrado: {onnx_path}")
                self.logger.info("   🔄 Exportando modelo a ONNX...")
                self._diagnostic_stats["onnx_export_attempted"] = True

                if self._export_to_onnx(model_path):
                    self._diagnostic_stats["onnx_export_success"] = True
                    if self._load_onnx_model(onnx_path):
                        self._onnx_available = True
                        self._diagnostic_stats["onnx_used"] = True
                        onnx_loaded = True
                        self.logger.info("✅ Modelo ONNX exportado y cargado")

            if onnx_loaded:
                self._warmup_onnx()
                self._print_startup_info()
                return

        self.logger.info("🔄 Cargando modelo YOLO (PyTorch) como fallback...")
        self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, model_path: str) -> None:
        """Carga modelo con PyTorch."""
        try:
            self.model = YOLO(model_path)
            self.model.to("cpu")
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.classes = self.vehicle_classes

            test_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model(
                test_frame,
                classes=self.vehicle_classes,
                verbose=False,
                imgsz=self.imgsz,
                device="cpu",
                max_det=1,
            )

            self._yolo_available = True
            self._diagnostic_stats["yolo_used"] = True
            self.logger.info("✅ Modelo YOLO (PyTorch) cargado correctamente")

            self._warmup_pytorch()
            self._print_startup_info()

        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error("❌ ERROR CRÍTICO: No se pudo cargar el modelo")
            self.logger.error("=" * 60)
            self.logger.error(f"   Modelo: {model_path}")
            self.logger.error(f"   Error: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo desde {model_path}: {e}")

    def _load_onnx_model(self, onnx_path: str) -> bool:
        """Carga modelo ONNX."""
        if not ONNX_AVAILABLE or ort is None:
            return False

        try:
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

            self.onnx_session = ort.InferenceSession(
                onnx_path,
                providers=providers,
                sess_options=sess_options
            )

            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_names = [o.name for o in self.onnx_session.get_outputs()]

            self.logger.info(f"   ✅ ONNX cargado con providers: {self.onnx_session.get_providers()}")
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Error cargando ONNX: {e}")
            self.onnx_session = None
            return False

    def _export_to_onnx(self, model_path: str) -> bool:
        """Exporta modelo a ONNX."""
        if not ONNX_AVAILABLE:
            return False

        onnx_path = model_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_path):
            return True

        try:
            if self.model is None:
                self.model = YOLO(model_path)

            self.model.export(
                format="onnx",
                imgsz=self.imgsz,
                optimize=True,
                opset=12,
                simplify=True,
                dynamic=False,
                verbose=False
            )

            return os.path.exists(onnx_path)

        except Exception as e:
            self.logger.warning(f"   ❌ Error exportando a ONNX: {e}")
            return False

    def _warmup_onnx(self) -> None:
        """Warmup para ONNX Runtime."""
        if not self._onnx_available or self.onnx_session is None:
            return

        self.logger.info("🔥 Ejecutando warmup para ONNX...")
        try:
            dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
            for _ in range(3):
                inputs = {self.input_name: dummy}
                self.onnx_session.run(self.output_names, inputs)
            self._warmed_up = True
            self.logger.info("✅ Warmup ONNX completado")
        except Exception as e:
            self.logger.warning(f"⚠️ Warmup ONNX falló: {e}")

    def _warmup_pytorch(self) -> None:
        """Warmup para PyTorch."""
        if not self._yolo_available or self.model is None:
            return

        self.logger.info("🔥 Ejecutando warmup para PyTorch...")
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.model(
                    dummy,
                    classes=self.vehicle_classes,
                    verbose=False,
                    imgsz=self.imgsz,
                    device="cpu",
                    max_det=1,
                )
            self._warmed_up = True
            self.logger.info("✅ Warmup PyTorch completado")
        except Exception as e:
            self.logger.warning(f"⚠️ Warmup PyTorch falló: {e}")

    def _preprocess_onnx(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa frame para ONNX."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        if h != self.imgsz or w != self.imgsz:
            frame = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        if len(frame.shape) == 3:
            frame = np.transpose(frame, (2, 0, 1))
            frame = np.expand_dims(frame, axis=0)

        return frame

    def _infer_onnx(self, frame: np.ndarray) -> np.ndarray:
        """Inferencia con ONNX Runtime."""
        if self.onnx_session is None:
            return np.array([])

        try:
            if len(frame.shape) == 3:
                frame = np.expand_dims(frame, axis=0)

            inputs = {self.input_name: frame}
            outputs = self.onnx_session.run(self.output_names, inputs)

            output = outputs[0]

            if len(output.shape) == 3:
                output = output[0].T
            elif len(output.shape) == 2:
                output = output.T

            if output.shape[1] < 6:
                return np.array([])

            if output.shape[1] == 6:
                boxes = output[:, :4]
                scores = output[:, 4]
                class_ids = output[:, 5].astype(np.int64)
            else:
                x_center = output[:, 0]
                y_center = output[:, 1]
                width = output[:, 2]
                height = output[:, 3]
                conf = output[:, 4]
                class_scores = output[:, 5:]

                max_scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
                scores = conf * max_scores

                x1 = (x_center - width / 2) * self.imgsz
                y1 = (y_center - height / 2) * self.imgsz
                x2 = (x_center + width / 2) * self.imgsz
                y2 = (y_center + height / 2) * self.imgsz
                boxes = np.column_stack([x1, y1, x2, y2])

            mask = scores >= self.confidence_threshold
            filtered = np.column_stack([boxes[mask], scores[mask], class_ids[mask]])

            class_mask = np.isin(filtered[:, 5].astype(np.int64), self.vehicle_classes)
            filtered = filtered[class_mask]

            if len(filtered) > 0:
                keep = _nms_fast(filtered, self.iou_threshold)
                filtered = filtered[keep]

            return filtered

        except Exception as e:
            self.logger.warning(f"Error en inferencia ONNX: {e}")
            return np.array([])

    def _parse_onnx_output(self, output: np.ndarray) -> DetectionList:
        """Parsea salida de ONNX."""
        detections = []

        if output is None or len(output) == 0:
            return detections

        for box_data in output:
            try:
                x1, y1, x2, y2 = map(int, box_data[:4])
                confidence = float(box_data[4])
                class_id = int(box_data[5])

                if class_id not in self.vehicle_classes:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < self.MIN_AREA or area > self.MAX_AREA:
                    continue

                centroid = calculate_centroid(x1, y1, x2, y2)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "centroid": centroid,
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": f"class_{class_id}",
                    "area": area,
                })
            except Exception as e:
                self.logger.debug(f"Error parseando box ONNX: {e}")
                continue

        return detections

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detecta objetos en un frame."""
        self._diagnostic_stats["frames_processed"] += 1

        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()

        if self.config.use_onnx:
            try:
                key = self.cache.compute_key(frame)
                cached = self.cache.get(key)
                if cached is not None:
                    self.logger.debug(f"Cache hit: {len(cached)} detecciones")
                    return cached
            except Exception as e:
                self.logger.debug(f"Error en caché: {e}")

        if self._onnx_available and self.onnx_session is not None:
            processed = self._preprocess_onnx(frame)
            output = self._infer_onnx(processed)
            detections = self._parse_onnx_output(output)
        else:
            detections = super().detect(frame)

        if self.config.use_onnx:
            try:
                self.cache.put(key, detections)
            except Exception as e:
                self.logger.debug(f"Error guardando en caché: {e}")

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]

        self._total_detections += len(detections)
        if detections:
            self._diagnostic_stats["frames_with_detections"] += 1

        return detections

    def _print_startup_info(self) -> None:
        """Imprime información de inicio usando logger."""
        self.logger.info("=" * 60)
        self.logger.info("🤖 DETECTOR YOLO OPTIMIZADO (CPU)")
        self.logger.info("=" * 60)
        self.logger.info(f"📁 Modelo: {self.config.model_path}")
        self.logger.info(f"🎯 Dispositivo: {self.device}")
        self.logger.info(f"⚡ ONNX Runtime: {'✅' if self._onnx_available else '❌'}")
        self.logger.info(f"⚡ Numba: {'✅' if NUMBA_AVAILABLE else '❌'}")
        self.logger.info(f"💾 Caché: {'✅' if self.config.use_onnx else '❌'}")
        self.logger.info(f"📐 IMG Size: {self.imgsz}")
        self.logger.info(f"🎯 Confianza: {self.confidence_threshold}")
        self.logger.info(f"📊 IOU: {self.iou_threshold}")
        self.logger.info(f"📦 Clases vehículo: {self.vehicle_classes}")
        self.logger.info(f"🧠 Memoria caché: {self.cache.max_size} entradas")
        self.logger.info(f"🔥 Warmup: {'✅' if self._warmed_up else '❌'}")
        self.logger.info("=" * 60)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de rendimiento."""
        stats = super().get_performance_stats()
        stats.update({
            "onnx_available": ONNX_AVAILABLE,
            "numba_available": NUMBA_AVAILABLE,
            "onnx_used": self._diagnostic_stats["onnx_used"],
            "yolo_used": self._diagnostic_stats["yolo_used"],
            "warmed_up": self._warmed_up,
            "onnx_export_success": self._diagnostic_stats["onnx_export_success"],
        })
        return stats
