"""
Módulo de detección de objetos.
"""

from .base import YOLODetector
from .optimized import OptimizedYOLODetector
from .factory import DetectorFactory
from .cache import DetectionCache
from .preprocessor import ImagePreprocessor
from .config import DetectorConfig

__all__ = [
    "YOLODetector",
    "OptimizedYOLODetector",
    "DetectorFactory",
    "DetectionCache",
    "ImagePreprocessor",
    "DetectorConfig",
]
