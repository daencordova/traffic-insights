"""YOLO object detector with optimizations.

This module implements the base detector of the system using YOLO
with caching, preprocessing, and performance statistics.

The detector supports:
    - Object detection with YOLO (Ultralytics)
    - LRU cache for detections
    - Image preprocessing
    - Performance statistics
    - Batch inference support
    - Automatic export to ONNX
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from ultralytics import YOLO

from core.constants import (
    BATCH_TIMES_MAX,
    CACHE_DEFAULT_SIZE,
    CACHE_MAX_SIZE,
    CACHE_MIN_SIZE,
    FEATURE_CACHE_MAX_AGE,
    INFERENCE_TIMES_MAX,
    MAX_BBOX_SIZE,
    MAX_DETECTION_CONFIDENCE,
    MEMORY_CHECK_INTERVAL,
    MEMORY_WARNING_THRESHOLD,
    MIN_BBOX_SIZE,
    MIN_DETECTION_AREA,
    MIN_DETECTION_CONFIDENCE,
)
from core.detector.cache import DetectionCache
from core.detector.config import DetectorConfig
from core.detector.preprocessor import ImagePreprocessor
from core.exceptions import DetectionError, ModelLoadError
from core.interfaces import IDetector
from core.types import DetectionList
from core.validators import validate_bbox, validate_centroid, validate_detection, validate_frame
from utils.geometry import calculate_centroid
from utils.helpers import get_memory_usage
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    from core.types import DetectionList


class YOLODetector(IDetector, LoggerMixin):
    """YOLO detector with cache and preprocessing.

    This detector uses YOLO from Ultralytics with optimizations
    to improve performance on CPU and GPU.

    Features:
        - Object detection with YOLO
        - LRU cache for detections (reduces redundant processing)
        - Image preprocessing (improves quality)
        - Performance statistics
        - Batch inference support
        - Automatic export to ONNX for optimized inference

    Attributes:
        config: Detector configuration (thresholds, classes, etc.)
        device: Inference device ('cpu', 'cuda', 'mps')
        model: Loaded YOLO model
        cache: LRU detection cache
        preprocessor: Image preprocessor

    Example:
        >>> detector = YOLODetector()
        >>> frame = cv2.imread("image.jpg")
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"Object: {det['label']} confidence: {det['confidence']:.2f}")
    """

    __slots__ = (
        "config",
        "device",
        "model",
        "cache",
        "preprocessor",
        "_inference_times",
        "_batch_inference_times",
        "_total_detections",
        "_total_batches",
        "_last_memory_check",
        "_using_optimized",
    )

    def __init__(self, config: DetectorConfig | None = None):
        """Initializes the YOLO detector.

        Args:
            config: Detector configuration. If None, uses global configuration.

        Raises:
            ModelLoadError: If the model cannot be loaded.
            ConfigurationError: If the configuration is invalid.

        Example:
            >>> # Use default configuration
            >>> detector = YOLODetector()
            >>>
            >>> # Use custom configuration
            >>> config = DetectorConfig(model_path="yolov8n.pt", confidence_threshold=0.7)
            >>> detector = YOLODetector(config)
        """
        self.config = config or DetectorConfig.from_global_config()
        self.logger.info("Initializing YOLODetector", model=self.config.model_path)

        self._using_optimized = False

        self.device: str = self._get_device()
        self.model: YOLO = self._load_model()

        self.model.conf = self.config.confidence_threshold
        self.model.iou = self.config.iou_threshold
        self.model.classes = self.config.vehicle_classes

        self.cache = DetectionCache(
            max_size=self._calculate_cache_size(), max_age_seconds=FEATURE_CACHE_MAX_AGE
        )

        self.preprocessor = ImagePreprocessor(enabled=False)

        self._inference_times: list[float] = []
        self._batch_inference_times: list[float] = []
        self._total_detections: int = 0
        self._total_batches: int = 0
        self._last_memory_check: float = time.time()

        self._print_startup_info()
        self.logger.info(
            "YOLODetector initialized", device=self.device, cache_size=self.cache.max_size
        )

    def _get_device(self) -> str:
        """Gets the optimal device for inference.

        Returns:
            str: Selected device ('cpu', 'cuda', 'mps').

        Note:
            If device is 'auto', automatically selects the best available
            (GPU > MPS > CPU).
        """
        device = self.config.device

        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        return str(device)

    def _load_model(self) -> YOLO:
        """Loads the YOLO model from the configured path.

        Returns:
            YOLO: Loaded YOLO model.

        Raises:
            ModelLoadError: If the model is not found or cannot be loaded.

        Note:
            If the model doesn't exist but is a standard YOLO model,
            it is automatically downloaded from Ultralytics.
        """
        self.logger.info("Loading model", path=self.config.model_path)

        try:
            model = YOLO(self.config.model_path)
        except FileNotFoundError as error:
            raise ModelLoadError(
                f"Model not found: {self.config.model_path}",
                {"model_path": self.config.model_path, "error": str(error)},
            ) from error
        except Exception as error:
            raise ModelLoadError(
                f"Error loading model {self.config.model_path}",
                {"model_path": self.config.model_path, "error": str(error)},
            ) from error

        if self.device != "cpu":
            try:
                model.to(self.device)
                self.logger.debug("Model moved to device", device=self.device)
            except Exception as e:
                self.logger.warning(
                    "Could not move to device, continuing on CPU",
                    device=self.device,
                    error=str(e),
                )
                self.device = "cpu"

        if self.config.use_half_precision and self.device != "cpu":
            try:
                model.model.half()
                self.logger.info("Half precision enabled")
            except Exception as e:
                self.logger.warning("Could not enable half precision", error=str(e))

        if self.config.use_onnx:
            try:
                onnx_path = self._export_to_onnx(model)
                if onnx_path:
                    model = YOLO(onnx_path)
                    self.logger.info("ONNX loaded", path=onnx_path)
            except Exception as e:
                self.logger.warning("ONNX error", error=str(e))

        return model

    def _export_to_onnx(self, model: YOLO) -> str | None:
        """Exports the model to ONNX format for optimized inference.

        Args:
            model: YOLO model to export.

        Returns:
            Optional[str]: Path to the ONNX file or None if failed.

        Note:
            ONNX Runtime offers better performance on CPU than PyTorch.
        """
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
                self.logger.info("Model exported to ONNX", path=onnx_path)
            except Exception as e:
                self.logger.warning("Error exporting to ONNX", error=str(e))
                return None

        return onnx_path

    def _calculate_cache_size(self) -> int:
        """Calculates optimal cache size based on available memory.

        Returns:
            int: Cache size between 4 and 64 entries.

        Note:
            Size is calculated as 10% of available memory,
            limited to 250MB maximum.
        """
        try:
            mem = get_memory_usage()
            available_mb = mem.get("system_available_mb", 4096)
            max_mb = min(available_mb * 0.1, 250)
            size = int(max_mb * 64)
            return max(CACHE_MIN_SIZE, min(CACHE_MAX_SIZE, size))
        except Exception:
            return CACHE_DEFAULT_SIZE

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validates that the frame is valid for processing.

        Args:
            frame: Frame to validate.

        Returns:
            bool: True if the frame is valid.

        Note:
            Checks that the frame is not None, has size > 0,
            and is a numpy array with minimum dimensions.
        """
        return validate_frame(frame, min_width=10, min_height=10)

    def _validate_box(self, box: Any) -> bool:
        """Validates a bounding box.

        Args:
            box: Bounding box to validate (x1, y1, x2, y2).

        Returns:
            bool: True if the box is valid.

        Note:
            Checks that the box has minimum and maximum configured sizes.
        """
        return validate_bbox(box, min_size=MIN_BBOX_SIZE, max_size=MAX_BBOX_SIZE)

    def _validate_centroid(self, centroid: Any) -> bool:
        """Validates a centroid.

        Args:
            centroid: Centroid to validate (x, y).

        Returns:
            bool: True if the centroid is valid.
        """
        return validate_centroid(centroid)

    def _validate_confidence(self, confidence: Any) -> bool:
        """Validates a confidence value.

        Args:
            confidence: Confidence to validate (0-1).

        Returns:
            bool: True if confidence is in valid range.
        """
        if not isinstance(confidence, (int, float)):
            return False
        return MIN_DETECTION_CONFIDENCE <= confidence <= MAX_DETECTION_CONFIDENCE

    def _validate_detection(self, detection: dict[str, Any]) -> bool:
        """Validates a complete detection.

        Args:
            detection: Detection dictionary.

        Returns:
            bool: True if the detection is valid.

        Note:
            Checks that all required fields are present and valid.
        """
        result = validate_detection(detection, min_confidence=0.0)
        return result.is_valid

    def _filter_valid_detections(self, detections: DetectionList) -> DetectionList:
        """Filters valid detections using list comprehension.

        Args:
            detections: List of detections to filter.

        Returns:
            DetectionList: List of valid detections.
        """
        if not detections:
            return []

        valid = [
            d
            for d in detections
            if d.get("box") and d.get("centroid") and d.get("confidence", 0) > 0
        ]

        if len(valid) != len(detections):
            self.logger.debug(
                "Detections filtered",
                total=len(detections),
                valid=len(valid),
                invalid=len(detections) - len(valid),
            )
        return valid

    def _parse_results(self, result) -> DetectionList:
        """Parses YOLO results to standard format.

        Args:
            result: YOLO result (ultralytics.Results).

        Returns:
            DetectionList: List of parsed detections.

        Note:
            Converts YOLO results to a consistent dictionary format
            used throughout the system.
        """
        detections: DetectionList = []

        if result is None or result.boxes is None:
            return detections

        for box in result.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_DETECTION_AREA:
                    continue

                centroid = calculate_centroid(x1, y1, x2, y2)

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "centroid": centroid,
                        "confidence": confidence,
                        "class_id": class_id,
                        "label": self.model.names[class_id],
                        "area": area,
                    }
                )
            except Exception as e:
                self.logger.debug("Error parsing box", error=str(e))
                continue

        return detections

    def _check_memory(self) -> None:
        """Checks memory usage and clears cache if necessary.

        Note:
            If memory exceeds the warning threshold (75%),
            the cache is automatically cleared.
        """
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
            self.logger.debug("Error checking memory", error=str(e))

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detects objects in a frame.

        Args:
            frame: Image to process as numpy array (H, W, C) BGR.

        Returns:
            DetectionList: List of validated detections.

        Raises:
            DetectionError: If an error occurs during inference.

        Example:
            >>> detector = YOLODetector()
            >>> frame = cv2.imread("traffic.jpg")
            >>> detections = detector.detect(frame)
            >>> print(f"Found {len(detections)} vehicles")
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
                self.logger.warning("Cache error", error=str(e))

        try:
            processed = self.preprocessor.process(frame)
        except Exception as e:
            self.logger.warning("Preprocessing error", error=str(e))
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
        except Exception as error:
            raise DetectionError(
                "Model inference error", {"frame_shape": frame.shape, "error": str(error)}
            ) from error

        try:
            detections = self._parse_results(results[0])
        except Exception as e:
            self.logger.warning("Error parsing results", error=str(e))
            detections = []

        filtered_detections = []

        for det in detections:
            box = det.get("box")
            if box:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 1000:
                    continue

            if det.get("confidence", 0) < 0.6:
                continue

            filtered_detections.append(det)

        valid_detections = self._filter_valid_detections(filtered_detections)

        if self.config.use_onnx:
            try:
                self.cache.put(key, valid_detections)
            except Exception as e:
                self.logger.warning("Error saving to cache", error=str(e))

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        self._total_detections += len(valid_detections)

        if len(self._inference_times) > INFERENCE_TIMES_MAX:
            self._inference_times = self._inference_times[-INFERENCE_TIMES_MAX:]

        if valid_detections:
            self.logger.debug(
                "Detections completed",
                count=len(valid_detections),
                time_ms=f"{inference_time:.1f}",
            )

        return valid_detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionList]:
        """Detects objects in multiple frames (batch inference).

        Args:
            frames: List of images to process.

        Returns:
            List[DetectionList]: List of detection lists, one per input frame.

        Note:
            Batch inference is more efficient than processing frames
            individually when multiple frames are available.
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

            if len(self._batch_inference_times) > BATCH_TIMES_MAX:
                self._batch_inference_times = self._batch_inference_times[-BATCH_TIMES_MAX:]

            self.logger.debug(
                "Batch inference completed",
                batch_size=len(valid_frames),
                total_detections=sum(len(d) for d in all_detections),
                time_ms=f"{batch_time:.1f}",
            )

            return result_list

        except Exception as e:
            self.logger.error(f"Batch inference error: {e}", exc_info=True)
            return [[] for _ in frames]

    def get_classes(self) -> list[int]:
        """Returns the classes detected by the model.

        Returns:
            List[int]: List of class IDs configured for detection.
        """
        return self.config.vehicle_classes

    def get_performance_stats(self) -> dict[str, Any]:
        """Returns performance statistics of the detector.

        Returns:
            Dict[str, Any]: Dictionary with statistics including:
                - total_detections: Total detections performed
                - avg_inference_time_ms: Average inference time
                - avg_batch_time_ms: Average batch inference time
                - total_batches: Number of batches processed
                - samples: Number of inference samples
                - device: Device used
                - cache: Cache statistics
                - preprocessor: Preprocessor statistics
        """
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
        """Clears the detection cache."""
        self.cache.clear()

    def enable_enhancement(self, enable: bool = True) -> None:
        """Enables or disables image preprocessing.

        Args:
            enable: True to enable, False to disable.
        """
        self.preprocessor.set_enabled(enable)

    def _print_startup_info(self) -> None:
        """Prints startup information using the logger."""
        self.logger.info("=" * 60)
        self.logger.info("YOLO DETECTOR")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.config.model_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Half precision: {self.config.use_half_precision}")
        self.logger.info(f"Cache: {self.config.use_onnx}")
        self.logger.info(f"IMG Size: {self.config.imgsz}")
        self.logger.info(f"Cache memory: {self.cache.max_size} entries")
        self.logger.info("=" * 60)
