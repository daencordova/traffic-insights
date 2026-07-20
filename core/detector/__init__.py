"""
Módulo de detección de objetos.
"""

from core.detector.base import YOLODetector
from core.detector.optimized import OptimizedYOLODetector
from core.detector.factory import DetectorFactory
from core.detector.cache import DetectionCache
from core.detector.preprocessor import ImagePreprocessor
from core.detector.config import DetectorConfig
from core.detector.model_manager import ModelManager
from core.detector.model_exporter import ModelExporter
from core.detector.inference_engine import (
    InferenceEngine,
    PyTorchInferenceEngine,
    ONNXInferenceEngine,
    InferenceEngineFactory,
)
from core.detector.post_processor import PostProcessor

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
