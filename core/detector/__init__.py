"""
Object detection module.

Provides optimized object detectors for different scenarios and hardware
configurations. This module implements a flexible detection system with
multiple inference engines and post-processing capabilities.

Main Components:
    - YOLODetector: Base detector using YOLO (Ultralytics)
    - OptimizedYOLODetector: CPU-optimized detector (Facade pattern)
    - DetectorOrchestrator: Detection system orchestrator
    - DetectorFactory: Factory for creating detector instances

Subsystems:
    - InferenceEngine: Inference engines (PyTorch, ONNX)
    - PostProcessor: Result processing and NMS
    - ModelManager: Model management and export
    - DetectionCache: Caching for detection results
    - ImagePreprocessor: Image preprocessing utilities
    - DetectorConfig: Detector configuration management

Features:
    - Multiple inference backends (PyTorch, ONNX)
    - CPU and GPU support
    - Model caching and management
    - Configurable preprocessing
    - NMS and post-processing
    - Export to ONNX format
    - Performance optimization for CPU

Example:
    >>> from core.detector import DetectorFactory, YOLODetector
    >>> from core.detector.config import DetectorConfig
    >>>
    >>> # Create configuration
    >>> config = DetectorConfig(
    ...     model_path="yolov8n.pt", confidence_threshold=0.5, iou_threshold=0.45, device="cpu"
    ... )
    >>>
    >>> # Create detector using factory
    >>> detector = DetectorFactory.create_detector(detector_type="yolo", config=config)
    >>>
    >>> # Perform detection
    >>> detections = detector.detect(frame)
    >>> for det in detections:
    ...     print(f"Class: {det['class_id']}, Confidence: {det['confidence']}")
    >>>
    >>> # Use optimized detector for CPU
    >>> optimizer = OptimizedYOLODetector(config)
    >>> detections = optimizer.detect(frame)

Example with inference engine:
    >>> from core.detector import InferenceEngineFactory
    >>>
    >>> # Create ONNX inference engine
    >>> engine = InferenceEngineFactory.create_engine(
    ...     engine_type="onnx", model_path="model.onnx", device="cpu"
    ... )
    >>>
    >>> # Run inference
    >>> results = engine.infer(frame)
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
from core.detector.orchestrator import DetectorOrchestrator
from core.detector.post_processor import PostProcessor
from core.detector.preprocessor import ImagePreprocessor

__all__ = [
    "YOLODetector",
    "OptimizedYOLODetector",
    "DetectorOrchestrator",
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
