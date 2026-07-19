"""
Post-procesador para detecciones YOLO.

Maneja el parsing de resultados, NMS y validación de detecciones.
"""

from typing import List, Dict, Any, Tuple

import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

from utils.logger import LoggerMixin
from utils.geometry import calculate_centroid


@jit(nopython=True, cache=True)
def nms_fast(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    NMS optimizado con Numba.

    Args:
        detections: Array de detecciones [N, 6] (x1, y1, x2, y2, score, class_id)
        iou_threshold: Umbral de IoU

    Returns:
        np.ndarray: Índices de detecciones a mantener
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
    """
    Post-procesador para detecciones YOLO.

    Responsabilidades:
    - Parsear resultados de inferencia
    - Aplicar NMS
    - Validar detecciones (confianza, área, etc.)
    - Convertir a formato estándar

    Attributes:
        confidence_threshold: Umbral de confianza
        iou_threshold: Umbral de IoU para NMS
        vehicle_classes: Clases a mantener
        min_area: Área mínima de detección
        max_area: Área máxima de detección
        imgsz: Tamaño de imagen para escalar coordenadas
    """

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        vehicle_classes: list = None,
        min_area: int = 500,
        max_area: int = 100000,
        imgsz: int = 320,
    ):
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
            "PostProcessor inicializado",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            vehicle_classes=vehicle_classes
        )

    def process_onnx_output(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Procesa la salida de ONNX.

        Args:
            output: Salida del modelo ONNX
            original_shape: Shape original de la imagen (height, width)

        Returns:
            List[Dict[str, Any]]: Detecciones procesadas
        """
        if output is None or len(output) == 0:
            return []

        try:
            if len(output.shape) == 3:
                output = output[0].T
            elif len(output.shape) == 2:
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
            self.logger.error(f"Error procesando salida ONNX: {e}")
            return []

    def process_pytorch_results(
        self,
        results,
        original_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Procesa los resultados de PyTorch.

        Args:
            results: Resultados de YOLO
            original_shape: Shape original de la imagen (height, width)

        Returns:
            List[Dict[str, Any]]: Detecciones procesadas
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

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "centroid": centroid,
                    "confidence": confidence,
                    "class_id": class_id,
                    "area": area,
                })

                self._stats["detections_after_nms"] += 1

        except Exception as e:
            self.logger.error(f"Error procesando resultados PyTorch: {e}")

        return detections

    def _parse_detections(
        self,
        detections: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Parsea detecciones al formato estándar.

        Args:
            detections: Array de detecciones
            original_shape: Shape original (height, width)

        Returns:
            List[Dict[str, Any]]: Detecciones en formato estándar
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

                parsed.append({
                    "box": (x1, y1, x2, y2),
                    "centroid": centroid,
                    "confidence": confidence,
                    "class_id": class_id,
                    "area": area,
                })

                self._stats["detections_after_nms"] += 1

            except Exception as e:
                self.logger.debug(f"Error parseando detección: {e}")
                continue

        return parsed

    def get_stats(self) -> dict:
        """Obtiene estadísticas del post-procesador."""
        return self._stats

    def reset_stats(self) -> None:
        """Reinicia las estadísticas."""
        self._stats = {
            "total_detections": 0,
            "filtered_low_confidence": 0,
            "filtered_wrong_class": 0,
            "filtered_small_area": 0,
            "filtered_large_area": 0,
            "detections_after_nms": 0,
        }
