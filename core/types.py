"""Tipos compartidos y estructuras de datos para todo el sistema.

Este módulo centraliza todos los tipos que son utilizados por múltiples
módulos para evitar imports circulares.

Contiene:
    - TypedDict para estructuras de datos (detecciones, tracks, etc.)
    - Protocol para interfaces
    - TypeAlias para tipos complejos
    - Clases base para evitar dependencias circulares
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from models.track_state import TrackState


class DetectionDict(TypedDict, total=False):
    """Estructura de una detección de objeto.

    Attributes:
        box: Bounding box en formato (x1, y1, x2, y2)
        centroid: Centroide en formato (cx, cy)
        confidence: Confianza de la detección (0-1)
        class_id: ID de la clase detectada
        label: Nombre de la clase
        area: Área del bounding box en píxeles
        features: Vector de features visuales (opcional)
        metadata: Metadatos adicionales
    """

    box: BoundingBox
    centroid: Point
    confidence: float
    class_id: int
    label: str
    area: int
    features: FloatNDArray | None
    metadata: dict[str, Any]


class TrackDataDict(TypedDict, total=False):
    """Estructura de datos de un track activo.

    Attributes:
        centroid: Centroide actual
        bbox: Bounding box actual
        status: Estado del track (tentative, confirmed, lost, dead)
        age: Edad en frames
        hits: Número de detecciones asociadas
        no_losses: Frames consecutivos sin pérdida
        confidence: Confianza actual
        velocity: Velocidad (vx, vy)
        acceleration: Aceleración (ax, ay)
        label: Nombre de la clase
        class_id: ID de la clase
        history: Historial de posiciones
        predicted_centroid: Posición predicha por Kalman
        features: Features visuales
        metadata: Metadatos adicionales
    """

    centroid: Point
    bbox: BoundingBox
    status: str
    age: int
    hits: int
    no_losses: int
    confidence: float
    velocity: Velocity
    acceleration: Acceleration
    label: str
    class_id: int
    history: list[Point]
    predicted_centroid: Point
    features: FloatNDArray | None
    metadata: dict[str, Any]


class StatsDict(TypedDict, total=False):
    """Estadísticas del sistema.

    Attributes:
        total: Total de vehículos contados
        line_counts: Conteo por línea
        class_counts: Conteo por clase
        avg_speed: Velocidad promedio
        max_speed: Velocidad máxima
        min_speed: Velocidad mínima
        avg_per_minute: Promedio por minuto
        count_rate: Tasa de conteo
        total_events: Total de eventos
        runtime_seconds: Tiempo de ejecución
        active_objects: Objetos activos
        frame_counter: Número de frame
        timestamp: Timestamp actual
    """

    total: int
    line_counts: dict[str, int]
    class_counts: dict[str, int]
    avg_speed: float
    max_speed: float
    min_speed: float
    avg_per_minute: float
    count_rate: float
    total_events: int
    runtime_seconds: float
    active_objects: int
    frame_counter: int
    timestamp: float


class CountingLineDict(TypedDict, total=False):
    """Configuración de una línea de conteo.

    Attributes:
        id: Identificador único
        name: Nombre descriptivo
        points: Lista de puntos que definen la línea
        color: Color en formato BGR
        direction: Dirección de conteo ('up' or 'down')
        y_position: Posición Y de la línea
        enabled: Si la línea está activa
        metadata: Metadatos adicionales
    """

    id: str
    name: str
    points: list[Point]
    color: Color
    direction: Literal["up", "down"]
    y_position: int
    enabled: bool
    metadata: dict[str, Any]


class PipelineStatsDict(TypedDict, total=False):
    """Estadísticas del pipeline.

    Attributes:
        fps: FPS actual
        frame_count: Frames procesados
        processing_time_ms: Tiempo de procesamiento en ms
        buffer_usage: Uso del buffer (0-1)
        active_tracks: Tracks activos
        total_detections: Detecciones totales
        state: Estado del pipeline
    """

    fps: float
    frame_count: int
    processing_time_ms: float
    buffer_usage: float
    active_tracks: int
    total_detections: int
    state: str


class HypothesisData(TypedDict, total=False):
    """Datos de una hipótesis MHT.

    Attributes:
        track_id: ID del track
        positions: Historial de posiciones
        confidence: Confianza de la hipótesis
        probability: Probabilidad de la hipótesis
        last_update: Timestamp de última actualización
        active: Si la hipótesis está activa
        status: Estado de la hipótesis
        velocity: Velocidad estimada
    """

    track_id: int
    positions: list[Point]
    confidence: float
    probability: float
    last_update: float
    active: bool
    status: str
    velocity: Velocity


class PredictionData(TypedDict, total=False):
    """Datos de predicción de trayectoria.

    Attributes:
        positions: Posiciones predichas
        confidences: Confianza por predicción
        timestamps: Timestamps de predicción
        horizon_seconds: Horizonte de predicción
        state: Estado de la trayectoria
        motion_model: Modelo de movimiento usado
        uncertainty: Incertidumbre de la predicción
        collision_risk: Riesgo de colisión
    """

    positions: list[FloatPoint]
    confidences: list[float]
    timestamps: list[float]
    horizon_seconds: float
    state: str
    motion_model: str
    uncertainty: float
    collision_risk: float


class TrackStateBase:
    """Clase base para TrackState (evita imports circulares con tracker)."""

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
    )

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.class_id: int = -1
        self.age: int = 0
        self.hits: int = 0
        self.no_losses: int = 0
        self.bbox: BoundingBox = (0, 0, 0, 0)
        self.centroid: Point = (0, 0)
        self.predicted_centroid: Point = (0, 0)
        self.velocity: Velocity = (0.0, 0.0)
        self.acceleration: Acceleration = (0.0, 0.0)
        self.status: str = "tentative"
        self.confidence: float = 0.0
        self.label: str = "unknown"
        self.history: list[Point] = []
        self.bbox_history: list[BoundingBox] = []
        self.features: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.kalman_filter: Any = None


@dataclass(slots=True)
class TrackData:
    """Datos de un track para transferencia entre componentes."""

    centroid: Point
    bbox: BoundingBox
    status: str
    age: int
    hits: int
    no_losses: int
    confidence: float
    velocity: Velocity
    label: str
    class_id: int
    history: list[Point]
    predicted_centroid: Point
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario para compatibilidad."""
        return {
            "centroid": self.centroid,
            "bbox": self.bbox,
            "status": self.status,
            "age": self.age,
            "hits": self.hits,
            "no_losses": self.no_losses,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "label": self.label,
            "class_id": self.class_id,
            "history": self.history,
            "predicted_centroid": self.predicted_centroid,
            "metadata": self.metadata,
        }

    @classmethod
    def from_track_state(cls, track_state: TrackState) -> TrackData:
        """Crea TrackData desde un TrackState."""
        return cls(
            centroid=track_state.centroid,
            bbox=track_state.bbox,
            status=track_state.status.value,
            age=track_state.age,
            hits=track_state.hits,
            no_losses=track_state.no_losses,
            confidence=track_state.confidence,
            velocity=track_state.velocity,
            label=track_state.label,
            class_id=track_state.class_id,
            history=list(track_state.history),
            predicted_centroid=track_state.predicted_centroid,
            metadata=track_state.metadata.copy(),
        )


@runtime_checkable
class IDetector(Protocol):
    """Interfaz para detectores de objetos."""

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detecta objetos en un frame."""
        ...

    def get_classes(self) -> list[int]:
        """Retorna las clases que detecta el modelo."""
        ...

    def get_performance_stats(self) -> dict[str, Any]:
        """Retorna estadísticas de rendimiento."""
        ...

    def clear_cache(self) -> None:
        """Limpia el caché de detecciones."""
        ...


@runtime_checkable
class ITracker(Protocol):
    """Interfaz para trackers de objetos."""

    def update(self, detections: DetectionList, frame: np.ndarray) -> TracksDict:
        """Actualiza el tracker con nuevas detecciones."""
        ...

    def get_tracking_info(self) -> TracksDict:
        """Retorna información de tracking actual."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas del tracker."""
        ...

    def reset(self) -> None:
        """Reinicia el tracker completamente."""
        ...


@runtime_checkable
class ICounter(Protocol):
    """Interfaz para contadores de objetos."""

    def process(self, tracks: TracksDict, frame: np.ndarray) -> StatsDict:
        """Procesa los tracks y actualiza los conteos."""
        ...

    def get_stats(self) -> StatsDict:
        """Retorna estadísticas actuales."""
        ...

    def reset(self) -> None:
        """Reinicia los contadores."""
        ...


@runtime_checkable
class IPipeline(Protocol):
    """Interfaz para el pipeline principal."""

    def run(self, source: str | None = None) -> None:
        """Ejecuta el pipeline principal."""
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame individual."""
        ...

    def pause(self) -> None:
        """Pausa la ejecución del pipeline."""
        ...

    def resume(self) -> None:
        """Reanuda la ejecución del pipeline."""
        ...

    def stop(self) -> None:
        """Detiene la ejecución del pipeline."""
        ...


@runtime_checkable
class IFeatureExtractor(Protocol):
    """Interfaz para extractores de features."""

    def extract_features(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        confidence: float = 0.5,
        *,
        force: bool = False,
    ) -> np.ndarray | None:
        """Extrae features de una región de imagen."""
        ...

    def compare_features(
        self, features1: np.ndarray, features2: np.ndarray, method: str = "cosine"
    ) -> float:
        """Compara dos vectores de features."""
        ...

    def clear_cache(self) -> None:
        """Limpia el caché de features."""
        ...

    @property
    def is_available(self) -> bool:
        """Verifica si el extractor está disponible."""
        ...


@runtime_checkable
class IMotionModel(Protocol):
    """Interfaz para modelos de movimiento."""

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int,
    ) -> tuple[list[FloatPoint], list[float]]:
        """Predice posiciones futuras."""
        ...

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """Evalúa la precisión del modelo."""
        ...

    @property
    def name(self) -> str:
        """Nombre del modelo."""
        ...


Point: TypeAlias = tuple[int, int]
"""Un punto en coordenadas 2D (x, y)."""

FloatPoint: TypeAlias = tuple[float, float]
"""Un punto en coordenadas 2D con precisión flotante."""

BoundingBox: TypeAlias = tuple[int, int, int, int]
"""Un bounding box en formato (x1, y1, x2, y2)."""

FloatBoundingBox: TypeAlias = tuple[float, float, float, float]
"""Un bounding box con precisión flotante."""

Velocity: TypeAlias = tuple[float, float]
"""Vector de velocidad (vx, vy)."""

Acceleration: TypeAlias = tuple[float, float]
"""Vector de aceleración (ax, ay)."""

Color: TypeAlias = tuple[int, int, int]
"""Color en formato BGR (blue, green, red)."""

ColorWithAlpha: TypeAlias = tuple[int, int, int, float]
"""Color en formato BGRA (blue, green, red, alpha)."""

FloatNDArray: TypeAlias = NDArray[np.float32]
"""Array NumPy de floats 32 bits."""

UInt8NDArray: TypeAlias = NDArray[np.uint8]
"""Array NumPy de uint8."""

Frame: TypeAlias = UInt8NDArray
"""Frame de video como array NumPy."""

TracksDict: TypeAlias = dict[int, TrackDataDict]
"""Diccionario de tracks: track_id -> TrackDataDict."""

TracksStateDict: TypeAlias = dict[int, Any]
"""Diccionario de estados de tracks: track_id -> TrackState."""

DetectionList: TypeAlias = list[DetectionDict]
"""Lista de detecciones."""

LineList: TypeAlias = list[CountingLineDict]
"""Lista de líneas de conteo."""

EventsList: TypeAlias = list[dict[str, Any]]
"""Lista de eventos de conteo."""

BBoxHistory: TypeAlias = list[BoundingBox]

Centroid: TypeAlias = tuple[int, int]


def is_valid_detection(detection: DetectionDict) -> bool:
    """Verifica si una detección tiene todos los campos requeridos."""
    required = ["box", "centroid", "confidence"]
    return all(field in detection for field in required)


def detection_to_dict(detection: DetectionDict) -> dict[str, Any]:
    """Convierte una detección a diccionario."""
    return {k: v for k, v in detection.items() if v is not None}


def track_data_to_dict(track: TrackDataDict) -> dict[str, Any]:
    """Convierte datos de track a diccionario."""
    return {k: v for k, v in track.items() if v is not None}


__all__ = [
    "Point",
    "FloatPoint",
    "BoundingBox",
    "FloatBoundingBox",
    "Velocity",
    "Acceleration",
    "Color",
    "ColorWithAlpha",
    "DetectionDict",
    "TrackDataDict",
    "StatsDict",
    "CountingLineDict",
    "HypothesisData",
    "PredictionData",
    "PipelineStatsDict",
    "TracksDict",
    "TracksStateDict",
    "DetectionList",
    "LineList",
    "EventsList",
    "IDetector",
    "ITracker",
    "ICounter",
    "IPipeline",
    "IFeatureExtractor",
    "IMotionModel",
    "TrackStateBase",
    "is_valid_detection",
    "detection_to_dict",
    "track_data_to_dict",
]
