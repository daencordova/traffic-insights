"""Constantes de tracking: Kalman, re-identificación, MHT, matching."""

from typing import Final

# TRACKING GENERAL
MAX_ACTIVE_TRACKS: Final[int] = 50
"""Número máximo de tracks activos simultáneamente."""

MAX_LOST_TRACKS: Final[int] = 50
"""Número máximo de tracks perdidos almacenados."""

MAX_TRACK_HISTORY: Final[int] = 15
"""Longitud máxima del historial de posiciones por track."""

MIN_HITS_TO_CONFIRM: Final[int] = 3
"""Número mínimo de detecciones para confirmar un track."""

MAX_FRAMES_MISSED: Final[int] = 30
"""Número máximo de frames perdidos antes de eliminar un track."""

IOU_THRESHOLD: Final[float] = 0.3
"""Umbral de IoU para matching entre detecciones y tracks."""

FEATURE_THRESHOLD: Final[float] = 0.5
"""Umbral de similitud de features para re-identificación."""

MAX_MATCH_DISTANCE: Final[float] = 50.0
"""Distancia máxima para matching espacial."""

MIN_MOTION_DISTANCE: Final[float] = 5.0
"""Distancia mínima para considerar movimiento."""

# MATCHING
MAX_DETECTIONS_PER_FRAME: Final[int] = 50
"""Máximo de detecciones por frame para matching."""

MIN_MATCH_SCORE: Final[float] = 0.1
"""Puntuación mínima para considerar un match."""

MAX_MATCH_RADIUS: Final[float] = 150.0
"""Radio máximo para búsqueda de matches."""

# VALIDACIÓN DE TRACKS
TRACK_VALIDATION_MIN_CONFIDENCE: Final[float] = 0.3
"""Confianza mínima para validación de tracks."""

TRACK_VALIDATION_MAX_SPEED_CHANGE: Final[float] = 50.0
"""Cambio máximo de velocidad para validación."""

TRACK_VALIDATION_IOU_THRESHOLD: Final[float] = 0.3
"""Umbral de IoU para validación."""

TRACK_VALIDATION_FEATURE_THRESHOLD: Final[float] = 0.6
"""Umbral de features para validación."""

TRACK_VALIDATION_MOTION_THRESHOLD: Final[float] = 0.7
"""Umbral de movimiento para validación."""

TRACK_VALIDATION_SHAPE_THRESHOLD: Final[float] = 0.5
"""Umbral de forma para validación."""

# KALMAN FILTER
KALMAN_DT: Final[float] = 1.0
"""Delta de tiempo para Kalman."""

KALMAN_PROCESS_NOISE: Final[float] = 0.03
"""Ruido de proceso para Kalman."""

KALMAN_MEASUREMENT_NOISE: Final[float] = 0.1
"""Ruido de medición para Kalman."""

# MHT (Multi-Hypothesis Tracking)
MHT_MAX_DEPTH: Final[int] = 5
"""Profundidad máxima del árbol MHT."""

MHT_PRUNING_THRESHOLD: Final[float] = 0.01
"""Umbral de poda de hipótesis MHT."""

MHT_MAX_HYPOTHESES: Final[int] = 3
"""Máximo de hipótesis por track."""

MIN_MHT_DEPTH: Final[int] = 2
"""Profundidad mínima de MHT."""

# RE-IDENTIFICACIÓN
REID_SIMILARITY_THRESHOLD: Final[float] = 0.6
"""Umbral de similitud para re-identificación."""

REID_SPATIAL_THRESHOLD: Final[float] = 100.0
"""Umbral espacial para re-identificación."""

REID_MAX_AGE_SECONDS: Final[float] = 30.0
"""Edad máxima para re-identificación."""

REID_CACHE_SIZE: Final[int] = 1000
"""Tamaño de caché de re-identificación."""

REID_MIN_FEATURES: Final[int] = 3
"""Mínimo de features para re-identificación."""

MIN_REID_CACHE_SIZE: Final[int] = 100
"""Tamaño mínimo de caché de re-identificación."""

# SENSOR FUSION
SENSOR_FUSION_VISUAL_WEIGHT: Final[float] = 0.7
"""Peso del sensor visual."""

SENSOR_FUSION_DEPTH_WEIGHT: Final[float] = 0.5
"""Peso del sensor de profundidad."""

SENSOR_FUSION_THERMAL_WEIGHT: Final[float] = 0.4
"""Peso del sensor térmico."""

SENSOR_FUSION_MOTION_WEIGHT: Final[float] = 0.3
"""Peso del sensor de movimiento."""

SENSOR_FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión."""

SENSOR_FUSION_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de fusión."""

SENSOR_FUSION_PARTICLE_COUNT: Final[int] = 500
"""Número de partículas para filtro de partículas."""

MIN_PARTICLE_COUNT: Final[int] = 100
"""Mínimo de partículas para filtro de partículas."""

FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión."""

FUSION_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de fusión."""

FUSION_PARTICLE_COUNT: Final[int] = 500
"""Número de partículas para fusión."""

# PATH PREDICTION
PATH_PREDICTION_HISTORY_LENGTH: Final[int] = 30
"""Longitud de histórico para predicción."""

PATH_PREDICTION_HORIZON: Final[float] = 2.0
"""Horizonte de predicción en segundos."""

PATH_PREDICTION_STEPS: Final[int] = 20
"""Número de pasos de predicción."""

PATH_PREDICTION_MIN_SAMPLES: Final[int] = 5
"""Mínimo de muestras para predicción."""

PATH_PREDICTION_UNCERTAINTY_THRESHOLD: Final[float] = 0.7
"""Umbral de incertidumbre para predicción."""

MIN_PREDICTION_HORIZON: Final[float] = 0.5
"""Horizonte mínimo de predicción en segundos."""

# HISTORIAL
MAX_FEATURE_HISTORY: Final[int] = 20
"""Máximo de features en historial."""

MAX_BBOX_HISTORY: Final[int] = 30
"""Máximo de bboxes en historial."""

MAX_EVENT_HISTORY: Final[int] = 1000
"""Máximo de eventos en historial."""

MAX_TRANSITION_HISTORY: Final[int] = 1000
"""Máximo de transiciones en historial."""

# FEATURES
FEATURE_EXTRACTOR_DIM: Final[int] = 2048
"""Dimensión por defecto de features."""

FEATURE_CACHE_MAX_SIZE: Final[int] = 500
"""Tamaño máximo de caché de features."""

FEATURE_CACHE_MAX_AGE: Final[float] = 3.0
"""Edad máxima de features en caché."""

SIFT_FEATURE_COUNT: Final[int] = 128
"""Número de features SIFT."""

# APRENDIZAJE EN LÍNEA
ONLINE_LEARNING_DEFAULT_LR: Final[float] = 0.05
"""Tasa de aprendizaje por defecto."""

ONLINE_LEARNING_MIN_SAMPLES: Final[int] = 5
"""Mínimo de muestras para aprendizaje."""

ONLINE_LEARNING_DRIFT_THRESHOLD: Final[float] = 0.35
"""Umbral de drift de concepto."""

ONLINE_LEARNING_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de aprendizaje."""

# PREDICCIÓN DE MOVIMIENTO
MIN_HISTORY_FOR_VELOCITY: Final[int] = 2
"""Mínimo de puntos en historial para calcular velocidad."""

MIN_HISTORY_FOR_ACCELERATION: Final[int] = 3
"""Mínimo de puntos en historial para calcular aceleración."""

MIN_HISTORY_FOR_PREDICTION: Final[int] = 3
"""Mínimo de puntos en historial para predicción."""

MAX_ANGLE_CHANGE: Final[float] = 1.0
"""Cambio máximo de ángulo para considerar movimiento errático."""

MIN_SPEED_FOR_MOVEMENT: Final[float] = 0.5
"""Velocidad mínima para considerar movimiento."""

# BENCHMARK
BENCHMARK_FRAMES: Final[int] = 50
"""Número de frames para benchmark."""

BENCHMARK_ITERATIONS: Final[int] = 50
"""Número de iteraciones para benchmark."""

# ESTADOS Y COLORES
STATUS_COLORS: Final[dict[str, tuple[tuple[int, int, int], str, str]]] = {
    "confirmed": ((0, 255, 0), "✅", "OK"),
    "lost": ((0, 255, 255), "⚠️", "Lost"),
    "tentative": ((255, 255, 0), "⏳", "New"),
    "dead": ((128, 128, 128), "💀", "Dead"),
}
"""Colores, iconos y textos para estados de tracks."""

PREDICTION_STATE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "stopped": (0, 0, 255),
    "accelerating": (0, 255, 255),
    "decelerating": (0, 165, 255),
    "turning": (255, 0, 255),
    "erratic": (255, 0, 0),
    "moving": (255, 255, 0),
    "unknown": (255, 255, 0),
}
"""Colores para estados de predicción."""

# CACHÉ Y MEMORIA
CACHE_MIN_SIZE: Final[int] = 4
"""Tamaño mínimo del caché de detecciones."""
CACHE_MAX_SIZE: Final[int] = 64
"""Tamaño máximo del caché de detecciones."""
CACHE_DEFAULT_SIZE: Final[int] = 16
"""Tamaño por defecto del caché de detecciones."""

# HISTORIAL
MAX_BBOX_HISTORY_DISPLAY: Final[int] = 10
"""Número máximo de bounding boxes a mostrar en el historial."""
MAX_BBOX_HISTORY_STORAGE: Final[int] = 30
"""Número máximo de bounding boxes a almacenar en el historial."""

# DIMENSIONES
DETECTION_POINT_DIMENSION: Final[int] = 2
"""Dimensión de un punto de detección (x, y)."""
DETECTION_VELOCITY_DIMENSION: Final[int] = 2
"""Dimensión de un vector de velocidad (vx, vy)."""
DETECTION_BBOX_DIMENSION: Final[int] = 4
"""Dimensión de un bounding box (x1, y1, x2, y2)."""

# VALIDACIÓN
SPATIAL_THRESHOLD_NORMALIZED: Final[float] = 100.0
"""Umbral de distancia espacial normalizado para matching."""

# ESTADOS Y HISTORIAL
STATE_HISTORY_MAX: Final[int] = 50
"""Tamaño máximo del historial de estados por track."""

CLEANUP_INTERVAL_FEATURES: Final[float] = 10.0
"""Intervalo de limpieza para caché de features en segundos."""

# MATCHING
MAX_UNMATCHED_TRACKS: Final[int] = 100
"""Número máximo de tracks no asociados permitidos."""

MAX_UNMATCHED_DETECTIONS: Final[int] = 100
"""Número máximo de detecciones no asociadas permitidas."""

REID_MATCH_TIMEOUT: Final[float] = 2.0
"""Timeout para matching de re-identificación en segundos."""
