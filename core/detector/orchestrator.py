"""Detection system orchestrator.

This module coordinates all components of the detection system:
    - Model management (PyTorch/ONNX)
    - Inference engines
    - Preprocessing and post-processing
    - Detection caching
    - Performance statistics

The orchestrator follows the Facade design pattern, providing
a simplified interface for the detection system.
"""

from pathlib import Path
import time
from typing import Any

import numpy as np

from core.constants import (
    BATCH_TIMES_MAX,
    CACHE_DEFAULT_SIZE,
    CACHE_MAX_SIZE,
    CACHE_MIN_SIZE,
    INFERENCE_TIMES_MAX,
    MEMORY_CHECK_INTERVAL,
    MEMORY_WARNING_THRESHOLD,
)
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
    """Detection system orchestrator.

    This class coordinates all components of the detection system,
    providing a unified interface for frame processing.

    Features:
        - Model management (PyTorch/ONNX)
        - Multiple inference engines
        - LRU cache for detections
        - Preprocessing and post-processing
        - Performance statistics
        - Automatic memory management

    Attributes:
        config: Detector configuration.
        device: Inference device.
        model_manager: Model manager.
        pytorch_engine: PyTorch inference engine.
        onnx_engine: ONNX inference engine.
        post_processor: Detection post-processor.
        cache: Detection cache.
        preprocessor: Image preprocessor.

    Example:
        >>> orchestrator = DetectorOrchestrator(config)
        >>> frame = cv2.imread("image.jpg")
        >>> detections = orchestrator.detect(frame)
        >>> for det in detections:
        ...     print(f"Object: {det['label']} confidence: {det['confidence']:.2f}")
        >>>
        >>> # Batch processing
        >>> frames = [frame1, frame2, frame3]
        >>> results = orchestrator.detect_batch(frames)
        >>>
        >>> # Clear cache
        >>> orchestrator.clear_cache()
        >>>
        >>> # Get performance stats
        >>> stats = orchestrator.get_performance_stats()
        >>> print(f"Average inference: {stats['avg_inference_time_ms']:.1f}ms")
    """

    def __init__(self, config: DetectorConfig):
        """Initializes the detection orchestrator.

        Args:
            config: Detector configuration.

        Raises:
            ModelLoadError: If no model can be loaded.
        """
        self.config = config
        self.logger.info("Initializing DetectorOrchestrator")

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
            "DetectorOrchestrator initialized",
            device=self.device,
            cache_size=self.cache.max_size,
            onnx_available=self._onnx_available,
            numba_available=self._check_numba_availability(),
        )

    def _get_device(self) -> str:
        """Gets the optimal device for inference.

        Returns:
            str: Selected device ('cpu', 'cuda', 'mps').
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
            self.logger.warning(f"Unknown device: {device}, using CPU")
            return "cpu"

        return str(device)

    def _init_model_manager(self) -> None:
        """Initializes the model manager."""
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
        """Initializes the inference engines."""
        self.pytorch_engine: PyTorchInferenceEngine | None = None
        self.onnx_engine: ONNXInferenceEngine | None = None
        self._onnx_available = False

        if self.config.use_onnx:
            onnx_path = Path(self.config.model_path).with_suffix(".onnx")

            if not onnx_path.exists():
                self.logger.info("Exporting model to ONNX...")
                onnx_path = self.model_exporter.export()

            if onnx_path and self.model_manager.load_onnx(onnx_path):
                self.onnx_engine = InferenceEngineFactory.create_onnx(
                    session=self.model_manager.get_onnx_session(),
                    input_name=self.model_manager.get_onnx_input_name(),
                    output_names=self.model_manager.get_onnx_output_names(),
                    imgsz=self.config.imgsz,
                )
                self._onnx_available = True
                self.logger.info("ONNX available")

        if not self._onnx_available:
            self.logger.info("Loading PyTorch as fallback...")
            if self.model_manager.load_pytorch():
                self.pytorch_engine = InferenceEngineFactory.create_pytorch(
                    model=self.model_manager.get_pytorch_model(),
                    imgsz=self.config.imgsz,
                    vehicle_classes=self.config.vehicle_classes,
                    device=self.device,
                    max_det=self.config.max_det,
                )
                self.logger.info("PyTorch available")
            else:
                self.logger.error("Could not load any model")
                raise ModelLoadError(f"Could not load model: {self.config.model_path}")

    def _init_post_processor(self) -> None:
        """Initializes the post-processor."""
        self.post_processor = PostProcessor(
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            vehicle_classes=self.config.vehicle_classes,
            imgsz=self.config.imgsz,
        )

    def _init_cache(self) -> None:
        """Initializes the detection cache."""
        cache_size = self._calculate_cache_size()
        self.cache = DetectionCache(
            max_size=cache_size,
            max_age_seconds=3.0,
        )

    def _init_preprocessor(self) -> None:
        """Initializes the image preprocessor."""
        self.preprocessor = ImagePreprocessor(enabled=False)

    def _check_numba_availability(self) -> bool:
        """Checks Numba availability."""
        try:
            import numba

            return True
        except ImportError:
            return False

    def _calculate_cache_size(self) -> int:
        """Calculates optimal cache size based on available memory."""
        try:
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(CACHE_MIN_SIZE, min(CACHE_MAX_SIZE, size))
        except Exception:
            return CACHE_DEFAULT_SIZE

    def _warmup(self) -> None:
        """Warms up inference engines to reduce initial latency."""
        if self._warmed_up:
            return

        self.logger.info("Performing warmup...")

        try:
            if self.onnx_engine:
                self.onnx_engine.warmup()

            if self.pytorch_engine:
                self.pytorch_engine.warmup()

            self._warmed_up = True
            self.logger.info("Warmup completed")

        except Exception as e:
            self.logger.warning(f"Warmup error: {e}")

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detects objects in a frame.

        Args:
            frame: Image to process as numpy array (H, W, C) BGR.

        Returns:
            DetectionList: List of validated detections.

        Raises:
            DetectionError: If an error occurs during inference.

        Example:
            >>> detections = orchestrator.detect(frame)
            >>> print(f"Found {len(detections)} objects")
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
        """Detects objects in multiple frames (batch inference).

        Args:
            frames: List of images to process.

        Returns:
            List[DetectionList]: List of detection lists, one per input frame.

        Example:
            >>> frames = [frame1, frame2, frame3]
            >>> results = orchestrator.detect_batch(frames)
            >>> for i, detections in enumerate(results):
            ...     print(f"Frame {i}: {len(detections)} detections")
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
                    self.logger.warning("Preprocessing error", error=str(e))
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
            self.logger.error(f"Batch inference error: {e}", exc_info=True)
            return [[] for _ in frames]

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validates that the frame is valid for processing."""
        if frame is None or frame.size == 0:
            return False

        if len(frame.shape) not in (2, 3):
            return False

        h, w = frame.shape[:2]
        return not (h < 10 or w < 10)

    def _check_cache(self, frame: np.ndarray) -> DetectionList | None:
        """Checks if the frame is in the cache."""
        try:
            key = self.cache.compute_key(frame)
            cached = self.cache.get(key)
            if cached is not None:
                self.logger.debug(f"Cache hit: {len(cached)} detections")
                return cached
        except Exception as e:
            self.logger.debug(f"Cache error: {e}")
        return None

    def _cache_detections(self, frame: np.ndarray, detections: DetectionList) -> None:
        """Stores detections in the cache."""
        try:
            key = self.cache.compute_key(frame)
            self.cache.put(key, detections)
        except Exception as e:
            self.logger.debug(f"Error saving to cache: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses the frame to improve detection."""
        try:
            return self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning(f"Preprocessing error: {e}")
            return frame

    def _infer(self, frame: np.ndarray, original_shape: tuple[int, ...]) -> DetectionList:
        """Performs inference using the available engine."""
        if self.onnx_engine and self.onnx_engine.is_available:
            try:
                output = self.onnx_engine.infer(frame)
                if output is not None and len(output) > 0:
                    return self.post_processor.process_onnx_output(output, original_shape[:2])
            except Exception as e:
                self.logger.warning(f"ONNX error: {e}, using PyTorch")

        if self.pytorch_engine and self.pytorch_engine.is_available:
            try:
                results = self.pytorch_engine.infer(frame)
                if results is not None:
                    return self.post_processor.process_pytorch_results(results, original_shape[:2])
            except Exception as e:
                self.logger.error(f"PyTorch error: {e}")
                raise DetectionError(
                    "Model inference error",
                    details={"error": str(e), "frame_shape": original_shape},
                ) from e

        return []

    def _update_stats(self, inference_time: float, num_detections: int) -> None:
        """Updates performance statistics."""
        self._inference_times.append(inference_time)
        self._total_detections += num_detections

        if len(self._inference_times) > INFERENCE_TIMES_MAX:
            self._inference_times = self._inference_times[-INFERENCE_TIMES_MAX:]

    def get_performance_stats(self) -> dict[str, Any]:
        """Returns performance statistics for the detector.

        Returns:
            dict[str, Any]: Statistics including:
                - total_detections: Total detections performed
                - avg_inference_time_ms: Average inference time
                - avg_batch_time_ms: Average batch inference time
                - total_batches: Number of batches processed
                - samples: Number of inference samples
                - device: Device used
                - onnx_available: Whether ONNX is available
                - numba_available: Whether Numba is available
                - warmed_up: Whether warmup has been performed
                - cache: Cache statistics
                - preprocessor: Preprocessor statistics
                - post_processor: Post-processor statistics
        """
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
        """Returns the classes detected by the model.

        Returns:
            list[int]: List of class IDs configured for detection.
        """
        return self.config.vehicle_classes

    def _check_memory(self) -> None:
        """Checks memory usage and clears cache if necessary."""
        current_time = time.time()
        if current_time - self._last_memory_check < MEMORY_CHECK_INTERVAL:
            return

        self._last_memory_check = current_time

        try:
            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > MEMORY_WARNING_THRESHOLD:
                self.logger.warning(
                    "High memory usage, clearing cache",
                    memory_percent=f"{mem_percent:.1f}",
                    cache_size=len(self.cache),
                )
                self.cache.clear()
        except Exception as e:
            self.logger.debug(f"Error checking memory: {e}")

    def clear_cache(self) -> None:
        """Clears the detection cache."""
        self.cache.clear()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Enables or disables image preprocessing.

        Args:
            enable: True to enable, False to disable.
        """
        self.preprocessor.set_enabled(enable)

    @property
    def model(self):
        """Returns the active model (for compatibility)."""
        if self.pytorch_engine:
            return self.pytorch_engine.model
        return None

    @property
    def is_onnx_available(self) -> bool:
        """Returns whether ONNX is available."""
        return self._onnx_available

    @property
    def is_numba_available(self) -> bool:
        """Returns whether Numba is available."""
        return self._check_numba_availability()
