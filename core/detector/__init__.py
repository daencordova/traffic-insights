"""
Módulo de detección de objetos.

Proporciona detectores de objetos optimizados para diferentes escenarios:

Componentes principales:
- YOLODetector: Detector base con YOLO (Ultralytics)
  └── Caché LRU, preprocesamiento, estadísticas

- OptimizedYOLODetector: Detector optimizado para CPU
  └── ONNX Runtime, Numba NMS, inferencia rápida

- DetectorFactory: Fábrica de detectores
  └── Creación automática del mejor detector disponible

Subsistemas:
- InferenceEngine: Motores de inferencia (PyTorch, ONNX)
- PostProcessor: Procesamiento de resultados y NMS
- ModelManager: Gestión de modelos y exportación

Ejemplo de uso:
    >>> detector = DetectorFactory.create_best_available()
    >>> detections = detector.detect(frame)
"""

from core.detector.base import YOLODetector
from core.detector.cache import DetectionCache
from core.detector.config import DetectorConfig
from core.detector.factory import DetectorFactory
from core.detector.inference_engine import (
    InferenceEngine,
    InferenceEngineFactory,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
)
from core.detector.model_exporter import ModelExporter
from core.detector.model_manager import ModelManager
from core.detector.optimized import OptimizedYOLODetector
from core.detector.post_processor import PostProcessor
from core.detector.preprocessor import ImagePreprocessor

__all__ = [
    "YOLODetector",
    "OptimizedYOLODetector",
    "DetectorFactory",
    "DetectionCache",
    "ImagePreprocessor",
    "DetectorConfig",
    "ModelManager",
    "ModelExporter",
    "InferenceEngine",
    "PyTorchInferenceEngine",
    "ONNXInferenceEngine",
    "InferenceEngineFactory",
    "PostProcessor",
]
