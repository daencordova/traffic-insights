"""CPU-optimized YOLO detector with ONNX Runtime and Numba (Facade).

This module provides a facade for the optimized detection system,
delegating all functionality to the DetectorOrchestrator.

The OptimizedYOLODetector class acts as a facade to maintain
compatibility with existing code while providing optimized
performance through ONNX Runtime and Numba.
"""

from typing import Any

import numpy as np

from core.detector.base import YOLODetector
from core.detector.config import DetectorConfig
from core.detector.orchestrator import DetectorOrchestrator
from core.exceptions import ModelLoadError
from core.types import DetectionList


class OptimizedYOLODetector(YOLODetector):
    """CPU-optimized YOLO detector (Facade).

    This class acts as a facade for the DetectorOrchestrator,
    maintaining the same interface as the previous implementation
    to ensure compatibility with existing code.

    Features:
        - ONNX Runtime for fast CPU inference
        - Numba for optimized NMS
        - LRU cache for detections
        - Automatic warmup
        - Fallback to PyTorch if ONNX is not available

    Example:
        >>> detector = OptimizedYOLODetector()
        >>> frame = cv2.imread("image.jpg")
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"Object: {det['label']} confidence: {det['confidence']:.2f}")
        >>>
        >>> # Batch processing
        >>> frames = [frame1, frame2, frame3]
        >>> results = detector.detect_batch(frames)
        >>> for i, detections in enumerate(results):
        ...     print(f"Frame {i}: {len(detections)} detections")
        >>>
        >>> # Clear cache
        >>> detector.clear_cache()
        >>>
        >>> # Get performance stats
        >>> stats = detector.get_performance_stats()
        >>> print(f"Average inference: {stats['avg_inference_time_ms']:.1f}ms")
    """

    __slots__ = ("_orchestrator", "_config")

    def __init__(self, config: DetectorConfig | None = None):
        """Initializes the optimized detector (facade).

        Args:
            config: Detector configuration. If None, uses global configuration.

        Raises:
            ModelLoadError: If no model can be loaded.

        Example:
            >>> # Use default configuration
            >>> detector = OptimizedYOLODetector()
            >>>
            >>> # Use custom configuration
            >>> config = DetectorConfig(
            ...     model_path="yolov8n.pt", confidence_threshold=0.6, imgsz=640
            ... )
            >>> detector = OptimizedYOLODetector(config)
        """
        self._config = config or DetectorConfig.from_global_config()
        self.logger.info("Initializing OptimizedYOLODetector (facade)")

        try:
            self._orchestrator = DetectorOrchestrator(self._config)
            self.logger.info(
                "OptimizedYOLODetector initialized",
                onnx_available=self._orchestrator.is_onnx_available,
                numba_available=self._orchestrator.is_numba_available,
                warmed_up=self._orchestrator._warmed_up,
            )
        except Exception as err:
            self.logger.error(f"Error initializing optimized detector: {err}")
            raise ModelLoadError(f"Could not initialize detector: {err}") from err

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detects objects in a frame.

        Args:
            frame: Image to process as numpy array (H, W, C) BGR.

        Returns:
            DetectionList: List of validated detections.

        Example:
            >>> detections = detector.detect(frame)
            >>> print(f"Found {len(detections)} objects")
        """
        return self._orchestrator.detect(frame)

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionList]:
        """Detects objects in multiple frames (batch inference).

        Args:
            frames: List of images to process.

        Returns:
            List[DetectionList]: List of detection lists, one per input frame.

        Example:
            >>> frames = [frame1, frame2, frame3]
            >>> results = detector.detect_batch(frames)
            >>> for i, detections in enumerate(results):
            ...     print(f"Frame {i}: {len(detections)} detections")
        """
        return self._orchestrator.detect_batch(frames)

    def get_classes(self) -> list[int]:
        """Returns the classes detected by the model.

        Returns:
            list[int]: List of class IDs configured for detection.
        """
        return self._orchestrator.get_classes()

    def get_performance_stats(self) -> dict[str, Any]:
        """Returns performance statistics for the detector.

        Returns:
            dict[str, Any]: Performance statistics including:
                - total_detections: Total detections performed
                - avg_inference_time_ms: Average inference time
                - avg_batch_time_ms: Average batch inference time
                - total_batches: Number of batches processed
                - samples: Number of inference samples
                - device: Device used
                - cache: Cache statistics
                - preprocessor: Preprocessor statistics
        """
        return self._orchestrator.get_performance_stats()

    def clear_cache(self) -> None:
        """Clears the detection cache."""
        self._orchestrator.clear_cache()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Enables or disables image preprocessing.

        Args:
            enable: True to enable, False to disable.

        Example:
            >>> detector.enable_enhancement(True)
            >>> # Image preprocessing is now enabled
        """
        self._orchestrator.enable_enhancement(enable)

    @property
    def config(self) -> DetectorConfig:
        """Returns the detector configuration."""
        return self._config

    @property
    def device(self) -> str:
        """Returns the inference device."""
        return self._orchestrator.device

    @property
    def model(self):
        """Returns the active model (for compatibility)."""
        return self._orchestrator.model

    @property
    def cache(self):
        """Returns the detection cache (for compatibility)."""
        return self._orchestrator.cache

    @property
    def preprocessor(self):
        """Returns the preprocessor (for compatibility)."""
        return self._orchestrator.preprocessor

    @property
    def post_processor(self):
        """Returns the post-processor (for compatibility)."""
        return self._orchestrator.post_processor

    @property
    def onnx_engine(self):
        """Returns the ONNX engine (for compatibility)."""
        return self._orchestrator.onnx_engine

    @property
    def pytorch_engine(self):
        """Returns the PyTorch engine (for compatibility)."""
        return self._orchestrator.pytorch_engine

    @property
    def model_manager(self):
        """Returns the model manager (for compatibility)."""
        return self._orchestrator.model_manager

    @property
    def model_exporter(self):
        """Returns the model exporter (for compatibility)."""
        return self._orchestrator.model_exporter
