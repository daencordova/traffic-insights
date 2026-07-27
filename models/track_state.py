"""Estado de un track para el sistema de tracking.

Este módulo define el estado completo de un track, incluyendo su posición,
historial, métricas de movimiento y estado de seguimiento. La implementación
está optimizada para uso intensivo de memoria con __slots__.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import (
    MAX_BOX_SIZE,
    MAX_FRAMES_MISSED as MAX_LOST_FRAMES,
    MAX_TRACK_HISTORY as MAX_HISTORY_LENGTH,
    MIN_BOX_SIZE,
    MIN_HITS_TO_CONFIRM,
)
from core.validators import validate_bbox, validate_centroid
from models.enums import TrackStatus

if TYPE_CHECKING:
    from models.kalman import EnhancedKalmanFilter

MIN_HISTORY_FOR_VELOCITY: int = 2
"""Mínimo de puntos en historial para calcular velocidad."""
MIN_HISTORY_FOR_ACCELERATION: int = 3
"""Mínimo de puntos en historial para calcular aceleración."""

Point = tuple[int, int]
BoundingBox = tuple[int, int, int, int]
Velocity = tuple[float, float]
Acceleration = tuple[float, float]
TrackHistory = deque[Point]
BBoxHistory = list[BoundingBox]


class TrackState:
    """Estado completo de un track con optimización de memoria.

    Esta clase representa el estado de un objeto en seguimiento, incluyendo
    su posición, historial, métricas de movimiento y estado de seguimiento.

    Características de optimización:
        - __slots__ para reducir uso de memoria (~40% menos que sin slots)
        - Validación eficiente de parámetros
        - Uso de tipos nativos para operaciones rápidas
        - Historial con deque para eficiencia en operaciones de cola

    Attributes:
        track_id: Identificador único del track.
        bbox: Bounding box actual (x1, y1, x2, y2).
        centroid: Centroide actual (x, y).
        features: Features visuales para re-identificación (opcional).
        confidence: Confianza del track (0-1).
        class_id: ID de la clase del objeto.
        label: Nombre de la clase.
        status: Estado actual del track (TrackStatus).
        age: Edad del track en frames.
        hits: Número de detecciones asociadas.
        no_losses: Frames consecutivos sin pérdida.
        history: Historial de posiciones.
        velocity: Velocidad actual (vx, vy) en píxeles/frame.
        acceleration: Aceleración actual (ax, ay) en píxeles/frame².
        predicted_centroid: Centroide predicho por Kalman.
        kalman_filter: Filtro de Kalman para predicción (opcional).
        metadata: Diccionario para metadatos adicionales.
        bbox_history: Historial de bounding boxes.

    Example:
        >>> track = TrackState(
        ...     track_id=1, bbox=(10, 20, 50, 60), centroid=(30, 40), confidence=0.85
        ... )
        >>> track.update(new_detection)
        >>> track.predict_position()
        (32, 42)
        >>> print(track.status)
        TrackStatus.TENTATIVE
    """

    __slots__ = (
        "track_id",
        "class_id",
        "age",
        "hits",
        "no_losses",
        "bbox",
        "centroid",
        "predicted_centroid",
        "velocity",
        "acceleration",
        "status",
        "confidence",
        "label",
        "history",
        "bbox_history",
        "features",
        "metadata",
        "kalman_filter",
        "_history_deque",
    )

    MIN_HITS_TO_CONFIRM: int = MIN_HITS_TO_CONFIRM
    MAX_LOST_FRAMES: int = MAX_LOST_FRAMES
    MIN_BOX_SIZE: int = MIN_BOX_SIZE
    MAX_BOX_SIZE: int = MAX_BOX_SIZE
    MAX_HISTORY_LENGTH: int = MAX_HISTORY_LENGTH

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        centroid: Point,
        *,
        features: np.ndarray | None = None,
        confidence: float = 0.5,
        class_id: int = -1,
        label: str = "unknown",
    ) -> None:
        """Inicializa un nuevo track.

        Args:
            track_id: Identificador único del track (debe ser >= 0).
            bbox: Bounding box (x1, y1, x2, y2) con dimensiones válidas.
            centroid: Centroide del objeto (x, y).
            features: Features visuales para re-identificación (opcional).
            confidence: Confianza de la detección (0-1).
            class_id: ID de la clase del objeto (>= -1).
            label: Nombre de la clase (string no vacío).

        Raises:
            ValueError: Si algún parámetro es inválido.

        Example:
            >>> track = TrackState(
            ...     track_id=42,
            ...     bbox=(100, 100, 200, 200),
            ...     centroid=(150, 150),
            ...     confidence=0.9,
            ...     class_id=2,
            ...     label="car",
            ... )
        """
        self._validate_track_id(track_id)
        self._validate_bbox(bbox)
        self._validate_centroid(centroid)
        self._validate_confidence(confidence)

        self.track_id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.features = features
        self.confidence = confidence
        self.class_id = class_id
        self.label = label

        self.status = TrackStatus.TENTATIVE
        self.age = 0
        self.hits = 1
        self.no_losses = 0

        self.history = deque(maxlen=self.MAX_HISTORY_LENGTH)
        self.history.append(centroid)

        self.bbox_history: BBoxHistory = []
        self.bbox_history.append(bbox)

        self.velocity: Velocity = (0.0, 0.0)
        self.acceleration: Acceleration = (0.0, 0.0)
        self.predicted_centroid: Point = centroid

        self.kalman_filter: EnhancedKalmanFilter | None = None

        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _validate_track_id(track_id: int) -> None:
        """Valida el ID del track."""
        if not isinstance(track_id, int) or track_id < 0:
            raise ValueError(f"track_id inválido: {track_id}")

    @staticmethod
    def _validate_bbox(bbox: Any) -> None:
        """Valida un bounding box usando el validador central."""
        if not validate_bbox(bbox):
            raise ValueError(f"bbox inválido: {bbox}")

    @staticmethod
    def _validate_centroid(centroid: Any) -> None:
        """Valida un centroide usando el validador central."""
        if not validate_centroid(centroid):
            raise ValueError(f"centroid inválido: {centroid}")

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Valida la confianza del track."""
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise ValueError(f"confidence inválido: {confidence}")

    def update(self, detection: dict[str, Any], features: np.ndarray | None = None) -> None:
        """Actualiza el track con una nueva detección.

        Este método es el punto principal de actualización del track,
        procesando una nueva detección y actualizando todas las métricas.

        Args:
            detection: Diccionario de detección con 'box', 'centroid', 'confidence', etc.
            features: Nuevos features visuales (opcional).

        Note:
            La actualización incluye:
            1. Posición (bbox y centroid)
            2. Confianza
            3. Clase y etiqueta
            4. Historial
            5. Métricas de movimiento
            6. Estado del track
            7. Filtro de Kalman

        Example:
            >>> track.update(
            ...     {
            ...         "box": (95, 95, 205, 205),
            ...         "centroid": (150, 150),
            ...         "confidence": 0.92,
            ...         "class_id": 2,
            ...         "label": "car",
            ...     }
            ... )
        """
        if not isinstance(detection, dict):
            return

        new_bbox = detection.get("box")
        if new_bbox is not None and validate_bbox(new_bbox):
            self.bbox = new_bbox
            self.bbox_history.append(new_bbox)

            if len(self.bbox_history) > 30:
                self.bbox_history = self.bbox_history[-30:]

        new_centroid = detection.get("centroid")
        if new_centroid is not None and validate_centroid(new_centroid):
            self.centroid = new_centroid
            self.history.append(new_centroid)

        new_confidence = detection.get("confidence")
        if isinstance(new_confidence, (int, float)):
            self.confidence = max(0.0, min(1.0, new_confidence))

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

        self._update_motion()

        self._update_status()

        if self.kalman_filter:
            self._update_kalman()

    def predict_position(self) -> Point:
        """Predice la siguiente posición usando el filtro de Kalman.

        Si el filtro de Kalman está disponible, usa su predicción.
        En caso contrario, retorna la posición actual.

        Returns:
            Point: Posición predicha (x, y).

        Example:
            >>> track.predict_position()
            (152, 148)  # Predicción usando Kalman
        """
        if self.kalman_filter:
            try:
                pred = self.kalman_filter.predict()

                self.predicted_centroid = (
                    max(0, int(pred[0])),
                    max(0, int(pred[1])),
                )
                return self.predicted_centroid
            except Exception:
                self.kalman_filter = None

        self.predicted_centroid = self.centroid
        return self.centroid

    def mark_lost(self) -> None:
        """Marca el track como perdido.

        Incrementa el contador de pérdidas y actualiza el estado.
        Este método se llama cuando un track no tiene detección asociada.
        """
        self.no_losses += 1
        self.age += 1
        self._update_status()

    def reset(self) -> None:
        """Reinicia el track a su estado inicial.

        Útil cuando se recupera un track perdido.
        """
        self.status = TrackStatus.TENTATIVE
        self.hits = 0
        self.no_losses = 0
        self.age = 0
        self.kalman_filter = None
        self.history.clear()
        self.bbox_history.clear()
        self.metadata.clear()

    def _update_motion(self) -> None:
        """Actualiza estimaciones de movimiento (velocidad y aceleración)."""
        history_len = len(self.history)

        if history_len >= MIN_HISTORY_FOR_VELOCITY:
            prev = self.history[-2]
            curr = self.history[-1]
            self.velocity = (
                float(curr[0] - prev[0]),
                float(curr[1] - prev[1]),
            )

            if history_len >= MIN_HISTORY_FOR_ACCELERATION:
                p1 = self.history[-3]
                p2 = self.history[-2]
                prev_vel = (
                    float(p2[0] - p1[0]),
                    float(p2[1] - p1[1]),
                )
                self.acceleration = (
                    self.velocity[0] - prev_vel[0],
                    self.velocity[1] - prev_vel[1],
                )

    def _update_status(self) -> None:
        """Actualiza el estado del track según hits y pérdidas.

        Transiciones posibles:
        - TENTATIVE -> CONFIRMED: hits >= MIN_HITS_TO_CONFIRM
        - CONFIRMED -> LOST: no_losses > MAX_LOST_FRAMES // 2
        - CONFIRMED -> DEAD: no_losses > MAX_LOST_FRAMES
        - LOST -> CONFIRMED: hits >= MIN_HITS_TO_CONFIRM y no_losses == 0
        - LOST -> DEAD: no_losses > MAX_LOST_FRAMES
        """
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
        """Actualiza el filtro de Kalman con la medición actual."""
        if self.kalman_filter:
            try:
                measurement = np.array([self.centroid[0], self.centroid[1]], dtype=np.float32)
                self.kalman_filter.correct(measurement)
            except Exception:
                self.kalman_filter = None

    def is_active(self) -> bool:
        """Verifica si el track está activo (no muerto).

        Returns:
            bool: True si el track está en estado TENTATIVE, CONFIRMED o LOST.

        Example:
            >>> if track.is_active():
            ...     print(f"Track {track.track_id} está activo")
        """
        return self.status in (TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.LOST)

    def is_confirmed(self) -> bool:
        """Verifica si el track está confirmado.

        Returns:
            bool: True si el track está en estado CONFIRMED.

        Example:
            >>> if track.is_confirmed():
            ...     print(f"Track {track.track_id} confirmado")
        """
        return self.status == TrackStatus.CONFIRMED

    def is_lost(self) -> bool:
        """Verifica si el track está perdido.

        Returns:
            bool: True si el track está en estado LOST.

        Example:
            >>> if track.is_lost():
            ...     print(f"Track {track.track_id} perdido")
        """
        return self.status == TrackStatus.LOST

    def is_dead(self) -> bool:
        """Verifica si el track está muerto.

        Returns:
            bool: True si el track está en estado DEAD.

        Example:
            >>> if track.is_dead():
            ...     print(f"Track {track.track_id} muerto")
        """
        return self.status == TrackStatus.DEAD

    def get_speed(self) -> float:
        """Obtiene la velocidad actual del track.

        Returns:
            float: Magnitud de la velocidad en píxeles/frame.

        Example:
            >>> speed = track.get_speed()
            >>> if speed > 10:
            ...     print("Movimiento rápido")
        """
        return float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))

    def get_movement_direction(self) -> float:
        """Obtiene la dirección del movimiento en radianes.

        Returns:
            float: Ángulo del movimiento (0 = derecha, π/2 = abajo).

        Example:
            >>> import math
            >>> angle = track.get_movement_direction()
            >>> degrees = math.degrees(angle)
            >>> print(f"Dirección: {degrees:.1f}°")
        """
        return float(np.arctan2(self.velocity[1], self.velocity[0]))

    def to_dict(self) -> dict[str, Any]:
        """Convierte el track a diccionario para serialización.

        Returns:
            dict[str, Any]: Diccionario con todos los datos del track.

        Example:
            >>> track_data = track.to_dict()
            >>> import json
            >>> json.dump(track_data, file)
        """
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
            "bbox_history": self.bbox_history[-10:],
            "metadata": self.metadata,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """Convierte el track a diccionario compacto (solo datos esenciales).

        Útil para transmisión de datos en tiempo real.

        Returns:
            dict[str, Any]: Diccionario con datos esenciales del track.

        Example:
            >>> compact = track.to_compact_dict()
            >>> # Enviar por WebSocket o guardar en base de datos
        """
        return {
            "id": self.track_id,
            "position": self.centroid,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class": self.label,
            "status": self.status.value,
            "speed": self.get_speed(),
        }

    def __repr__(self) -> str:
        """Representación legible del track."""
        return (
            f"TrackState(id={self.track_id}, status={self.status.value}, "
            f"pos={self.centroid}, conf={self.confidence:.2f})"
        )

    def __str__(self) -> str:
        """Representación amigable para logs."""
        return f"Track {self.track_id} [{self.status.value}] at {self.centroid}"

    def __len__(self) -> int:
        """Retorna la longitud del historial."""
        return len(self.history)

    def __contains__(self, pos: Point) -> bool:
        """Verifica si una posición está en el historial."""
        return pos in self.history


def create_track_from_detection(
    track_id: int,
    detection: dict[str, Any],
    features: np.ndarray | None = None,
) -> TrackState:
    """Crea un TrackState a partir de una detección.

    Esta función es un helper para crear tracks consistentemente
    en todo el sistema.

    Args:
        track_id: ID del track.
        detection: Diccionario de detección.
        features: Features visuales (opcional).

    Returns:
        TrackState: Track creado.

    Example:
        >>> track = create_track_from_detection(1, detection)
    """
    return TrackState(
        track_id=track_id,
        bbox=detection.get("box", (0, 0, 0, 0)),
        centroid=detection.get("centroid", (0, 0)),
        features=features,
        confidence=detection.get("confidence", 0.5),
        class_id=detection.get("class_id", -1),
        label=detection.get("label", "unknown"),
    )


def merge_tracks(track1: TrackState, track2: TrackState) -> TrackState:
    """Fusiona dos tracks en uno solo.

    Útil cuando dos tracks representan el mismo objeto.

    Args:
        track1: Primer track (principal).
        track2: Segundo track (secundario).

    Returns:
        TrackState: Track fusionado.

    Example:
        >>> merged = merge_tracks(track_a, track_b)
    """
    if track2.confidence > track1.confidence:
        track1, track2 = track2, track1

    for pos in track2.history:
        if pos not in track1.history:
            track1.history.append(pos)

    track1.hits += track2.hits
    track1.no_losses = min(track1.no_losses, track2.no_losses)
    track1.confidence = max(track1.confidence, track2.confidence)

    track1._update_motion()

    return track1


__all__ = [
    "TrackState",
    "Point",
    "BoundingBox",
    "Velocity",
    "Acceleration",
    "TrackHistory",
    "BBoxHistory",
    "create_track_from_detection",
    "merge_tracks",
]
