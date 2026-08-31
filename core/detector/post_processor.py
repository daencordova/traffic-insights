"""Post-processor for YOLO detections.

Handles parsing of results, NMS, and detection validation.
Includes Numba optimizations for fast NMS.
"""

from typing import Any

import numpy as np

from core.constants import IMAGE_CHANNELS_GRAY, IMAGE_CHANNELS_RGB
from utils.geometry import calculate_centroid
from utils.logger import LoggerMixin

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
def nms_fast(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Fast NMS optimized with Numba.

    This function performs Non-Maximum Suppression on detections
    using Numba JIT compilation for maximum performance.

    Args:
        detections: Array of detections [N, 6] (x1, y1, x2, y2, score, class_id).
        iou_threshold: IoU threshold for suppression.

    Returns:
        np.ndarray: Indices of detections to keep.

    Example:
        >>> detections = np.array([[10, 10, 50, 50, 0.9, 2], ...])
        >>> keep = nms_fast(detections, 0.45)
        >>> filtered = detections[keep]
    """
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


class PostProcessor(LoggerMixin):
    """Post-processor for YOLO detections.

    This class handles the processing of raw inference results,
    applying NMS, validation, and conversion to standard format.

    Responsibilities:
        - Parse inference results
        - Apply NMS (Non-Maximum Suppression)
        - Validate detections (confidence, area, etc.)
        - Convert to standard format

    Attributes:
        confidence_threshold: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.
        vehicle_classes: Classes to keep.
        min_area: Minimum detection area.
        max_area: Maximum detection area.
        imgsz: Image size for scaling coordinates.

    Example:
        >>> processor = PostProcessor(
        ...     confidence_threshold=0.5, iou_threshold=0.45, vehicle_classes=[2, 3, 5, 7]
        ... )
        >>>
        >>> # Process ONNX output
        >>> detections = processor.process_onnx_output(output, (480, 640))
        >>>
        >>> # Process PyTorch results
        >>> detections = processor.process_pytorch_results(results, (480, 640))
        >>>
        >>> # Get statistics
        >>> stats = processor.get_stats()
        >>> print(f"Detections after NMS: {stats['detections_after_nms']}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        vehicle_classes: list | None = None,
        min_area: int = 500,
        max_area: int = 100000,
        imgsz: int = 320,
    ):
        """Initializes the post-processor.

        Args:
            confidence_threshold: Minimum confidence threshold (0-1).
            iou_threshold: IoU threshold for NMS (0-1).
            vehicle_classes: List of class IDs to keep.
            min_area: Minimum detection area in pixels.
            max_area: Maximum detection area in pixels.
            imgsz: Image size for scaling coordinates.

        Example:
            >>> processor = PostProcessor(confidence_threshold=0.6, iou_threshold=0.5, min_area=300)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]
        self.min_area = min_area
        self.max_area = max_area
        self.imgsz = imgsz

        self._stats = {
            "total_detections": 0,
            "filtered_low_confidence": 0,
            "filtered_wrong_class": 0,
            "filtered_small_area": 0,
            "filtered_large_area": 0,
            "detections_after_nms": 0,
        }

        self.logger.info(
            "PostProcessor initialized",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            vehicle_classes=vehicle_classes,
        )

    def process_onnx_output(
        self, output: np.ndarray, original_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Processes ONNX Runtime output.

        Args:
            output: ONNX model output.
            original_shape: Original image shape (height, width).

        Returns:
            List[Dict[str, Any]]: Processed and validated detections.

        Note:
            Supports two output formats:
            1. [N, 6] with (x1, y1, x2, y2, score, class_id)
            2. [N, 85] with (x_center, y_center, w, h, conf, class_scores...)

        Example:
            >>> detections = processor.process_onnx_output(output, (480, 640))
            >>> for det in detections:
            ...     print(f"Class: {det['class_id']}, Conf: {det['confidence']:.2f}")
        """
        if output is None or len(output) == 0:
            return []

        try:
            if len(output.shape) == IMAGE_CHANNELS_RGB:
                output = output[0].T
            elif len(output.shape) == IMAGE_CHANNELS_GRAY:
                output = output.T

            if output.shape[1] < 6:
                return []

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
                keep = nms_fast(filtered, self.iou_threshold)
                filtered = filtered[keep]

            return self._parse_detections(filtered, original_shape)

        except Exception as e:
            self.logger.error(f"Error processing ONNX output: {e}")
            return []

    def process_pytorch_results(
        self, results, original_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Processes PyTorch (YOLO) results.

        Args:
            results: YOLO results (ultralytics.Results).
            original_shape: Original image shape (height, width).

        Returns:
            List[Dict[str, Any]]: Processed and validated detections.

        Example:
            >>> results = model(frame)
            >>> detections = processor.process_pytorch_results(results, (480, 640))
            >>> for det in detections:
            ...     print(f"Class: {det['class_id']}, Conf: {det['confidence']:.2f}")
        """
        if results is None or results.boxes is None:
            return []

        detections = []

        try:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)
                if area < self.min_area or area > self.max_area:
                    self._stats["filtered_small_area"] += 1
                    continue

                if class_id not in self.vehicle_classes:
                    self._stats["filtered_wrong_class"] += 1
                    continue

                if confidence < self.confidence_threshold:
                    self._stats["filtered_low_confidence"] += 1
                    continue

                centroid = calculate_centroid(x1, y1, x2, y2)

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "centroid": centroid,
                        "confidence": confidence,
                        "class_id": class_id,
                        "area": area,
                    }
                )

                self._stats["detections_after_nms"] += 1

        except Exception as e:
            self.logger.error(f"Error processing PyTorch results: {e}")

        return detections

    def _parse_detections(
        self, detections: np.ndarray, original_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Parses detections to standard format.

        Args:
            detections: Array of detections [N, 6].
            original_shape: Original shape (height, width).

        Returns:
            List[Dict[str, Any]]: Detections in standard format.

        Note:
            Converts normalized coordinates to pixels and applies
            area and confidence validations.
        """
        if detections is None or len(detections) == 0:
            return []

        parsed = []
        h_orig, w_orig = original_shape

        scale_x = w_orig / self.imgsz
        scale_y = h_orig / self.imgsz

        for box_data in detections:
            try:
                x1, y1, x2, y2 = map(int, box_data[:4])
                confidence = float(box_data[4])
                class_id = int(box_data[5])

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                area = (x2 - x1) * (y2 - y1)

                if area < self.min_area or area > self.max_area:
                    continue

                centroid = calculate_centroid(x1, y1, x2, y2)

                parsed.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "centroid": centroid,
                        "confidence": confidence,
                        "class_id": class_id,
                        "area": area,
                    }
                )

                self._stats["detections_after_nms"] += 1

            except Exception as e:
                self.logger.debug(f"Error parsing detection: {e}")
                continue

        return parsed

    def get_stats(self) -> dict:
        """Gets post-processor statistics.

        Returns:
            Dict: Statistics including:
                - total_detections: Total processed
                - filtered_low_confidence: Filtered by confidence
                - filtered_wrong_class: Filtered by class
                - filtered_small_area: Filtered by minimum area
                - filtered_large_area: Filtered by maximum area
                - detections_after_nms: Detections after NMS

        Example:
            >>> stats = processor.get_stats()
            >>> print(f"Filtered by confidence: {stats['filtered_low_confidence']}")
            >>> print(f"Detections after NMS: {stats['detections_after_nms']}")
        """
        return self._stats

    def reset_stats(self) -> None:
        """Resets post-processor statistics.

        Example:
            >>> processor.reset_stats()
            >>> # All statistics are reset to zero
        """
        self._stats = {
            "total_detections": 0,
            "filtered_low_confidence": 0,
            "filtered_wrong_class": 0,
            "filtered_small_area": 0,
            "filtered_large_area": 0,
            "detections_after_nms": 0,
        }
