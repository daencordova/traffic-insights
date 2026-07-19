"""
Estado de un track para el sistema de tracking
"""

from typing import Optional, Tuple, Dict, Any, Deque
from collections import deque

import numpy as np

from models.enums import TrackStatus
from core.constants import (
    MIN_HITS_TO_CONFIRM, MAX_FRAMES_MISSED,
    MIN_BOX_SIZE, MAX_BOX_SIZE, MAX_TRACK_HISTORY
)
from .kalman import EnhancedKalmanFilter
from core.validators import validate_bbox, validate_centroid

Point = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]
Velocity = Tuple[float, float]
Acceleration = Tuple[float, float]
TrackHistory = Deque[Point]


class TrackState:
    """Estado completo de un track con optimización de memoria."""

    __slots__ = (
        'track_id', 'bbox', 'centroid', 'features', 'confidence',
        'class_id', 'label', 'status', 'age', 'hits', 'no_losses',
        'history', 'velocity', 'acceleration', 'predicted_centroid',
        'kalman_filter', 'metadata', 'bbox_history', '_history_deque'
    )

    MIN_HITS_TO_CONFIRM: int = MIN_HITS_TO_CONFIRM
    MAX_LOST_FRAMES: int = MAX_FRAMES_MISSED
    MIN_BOX_SIZE: int = MIN_BOX_SIZE
    MAX_BOX_SIZE: int = MAX_BOX_SIZE
    MAX_HISTORY_LENGTH: int = MAX_TRACK_HISTORY

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        centroid: Point,
        features: Optional[np.ndarray] = None,
        confidence: float = 0.5,
        class_id: int = -1,
        label: str = "unknown",
    ) -> None:
        if not isinstance(track_id, int) or track_id < 0:
            raise ValueError(f"track_id inválido: {track_id}")

        if not self._validate_bbox(bbox):
            raise ValueError(f"bbox inválido: {bbox}")

        if not self._validate_centroid(centroid):
            raise ValueError(f"centroid inválido: {centroid}")

        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise ValueError(f"confidence inválido: {confidence}")

        self.track_id: int = track_id
        self.bbox: BoundingBox = bbox
        self.centroid: Point = centroid
        self.features: Optional[np.ndarray] = features
        self.confidence: float = confidence
        self.class_id: int = class_id
        self.label: str = label

        self.status: TrackStatus = TrackStatus.TENTATIVE
        self.age: int = 0
        self.hits: int = 1
        self.no_losses: int = 0

        self.history: TrackHistory = deque(maxlen=self.MAX_HISTORY_LENGTH)
        self.history.append(centroid)

        self.velocity: Velocity = (0.0, 0.0)
        self.acceleration: Acceleration = (0.0, 0.0)
        self.predicted_centroid: Point = centroid

        self.kalman_filter: Optional[EnhancedKalmanFilter] = None

        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def _validate_bbox(bbox: Any) -> bool:
        """Valida un bounding box usando el validador central."""
        return validate_bbox(bbox)

    @staticmethod
    def _validate_centroid(centroid: Any) -> bool:
        """Valida un centroide usando el validador central."""
        return validate_centroid(centroid)

    def update(self, detection: Dict[str, Any], features: Optional[np.ndarray] = None) -> None:
        """Actualiza el track con una nueva detección"""
        if not isinstance(detection, dict):
            return

        new_bbox = detection.get("box")
        if new_bbox is not None and self._validate_bbox(new_bbox):
            self.bbox = new_bbox

        new_centroid = detection.get("centroid")
        if new_centroid is not None and self._validate_centroid(new_centroid):
            self.centroid = new_centroid

        new_confidence = detection.get("confidence")
        if isinstance(new_confidence, (int, float)) and 0 <= new_confidence <= 1:
            self.confidence = new_confidence

        new_class_id = detection.get("class_id")
        if isinstance(new_class_id, int) and new_class_id >= 0:
            self.class_id = new_class_id

        new_label = detection.get("label")
        if isinstance(new_label, str) and new_label:
            self.label = new_label

        if features is not None:
            self.features = features

        self.hits += 1
        self.no_losses = 0
        self.age += 1

        self.history.append(self.centroid)
        self._update_motion()
        self._update_status()

        if self.kalman_filter:
            self._update_kalman()

    def predict_position(self) -> Point:
        """Predice la siguiente posición usando Kalman"""
        if self.kalman_filter:
            try:
                pred = self.kalman_filter.predict()
                self.predicted_centroid = (int(pred[0]), int(pred[1]))
                return self.predicted_centroid
            except Exception:
                pass

        self.predicted_centroid = self.centroid
        return self.centroid

    def mark_lost(self) -> None:
        """Marca el track como perdido"""
        self.no_losses += 1
        self.age += 1
        self._update_status()

    def _update_motion(self) -> None:
        """Actualiza estimaciones de movimiento"""
        if len(self.history) >= 2:
            prev = self.history[-2]
            curr = self.history[-1]

            self.velocity = (curr[0] - prev[0], curr[1] - prev[1])

            if len(self.history) >= 3:
                prev_vel = self._get_previous_velocity()
                if prev_vel:
                    self.acceleration = (
                        self.velocity[0] - prev_vel[0],
                        self.velocity[1] - prev_vel[1],
                    )

    def _get_previous_velocity(self) -> Optional[Velocity]:
        """Obtiene la velocidad anterior del historial"""
        if len(self.history) >= 3:
            p1 = self.history[-3]
            p2 = self.history[-2]
            return (p2[0] - p1[0], p2[1] - p1[1])
        return None

    def _update_status(self) -> None:
        """Actualiza el estado del track"""
        if self.status == TrackStatus.DEAD:
            return

        if self.status == TrackStatus.TENTATIVE:
            if self.hits >= self.MIN_HITS_TO_CONFIRM:
                self.status = TrackStatus.CONFIRMED

        elif self.status == TrackStatus.CONFIRMED:
            if self.no_losses > self.MAX_LOST_FRAMES:
                self.status = TrackStatus.DEAD
            elif self.no_losses > self.MAX_LOST_FRAMES // 2:
                self.status = TrackStatus.LOST

        elif self.status == TrackStatus.LOST:
            if self.no_losses > self.MAX_LOST_FRAMES:
                self.status = TrackStatus.DEAD
            elif self.hits >= self.MIN_HITS_TO_CONFIRM and self.no_losses == 0:
                self.status = TrackStatus.CONFIRMED

    def _update_kalman(self) -> None:
        """Actualiza el filtro de Kalman con la medición"""
        if self.kalman_filter:
            try:
                measurement = np.array([self.centroid[0], self.centroid[1]])
                self.kalman_filter.correct(measurement)
            except Exception:
                self.kalman_filter = None

    def is_active(self) -> bool:
        """Verifica si el track está activo"""
        return self.status in [TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.LOST]

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "status": self.status.value,
            "age": self.age,
            "hits": self.hits,
            "no_losses": self.no_losses,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "class_id": self.class_id,
            "label": self.label,
            "history": list(self.history),
            "metadata": self.metadata,
        }
