"""Constantes para reemplazar magic numbers en el código.

Este módulo centraliza todos los valores numéricos que aparecen como
magic numbers en el código, proporcionando nombres semánticos que
mejoran la legibilidad y mantenibilidad.
"""

from typing import Final

# DIMENSIONES Y TAMAÑOS

POINT_DIMENSION: Final[int] = 2
"""Dimensión de un punto (x, y)."""

BBOX_DIMENSION: Final[int] = 4
"""Dimensión de un bounding box (x1, y1, x2, y2)."""

VELOCITY_DIMENSION: Final[int] = 2
"""Dimensión de un vector de velocidad (vx, vy)."""

IMAGE_CHANNELS_RGB: Final[int] = 3
"""Número de canales en una imagen RGB/BGR."""

IMAGE_CHANNELS_GRAY: Final[int] = 1
"""Número de canales en una imagen en escala de grises."""

# LÍMITES DE HISTORIAL Y CACHÉ

HISTORY_MAX_SIZE: Final[int] = 30
"""Tamaño máximo de historiales de tracks y bboxes."""

FEATURE_HISTORY_MAX_SIZE: Final[int] = 20
"""Tamaño máximo de historial de features."""

EVENT_HISTORY_MAX_SIZE: Final[int] = 1000
"""Tamaño máximo de historial de eventos."""

TRANSITION_HISTORY_MAX_SIZE: Final[int] = 1000
"""Tamaño máximo de historial de transiciones."""

INFERENCE_TIMES_MAX: Final[int] = 100
"""Número máximo de tiempos de inferencia almacenados."""

BATCH_TIMES_MAX: Final[int] = 50
"""Número máximo de tiempos de batch almacenados."""

PROCESSING_TIMES_MAX: Final[int] = 100
"""Número máximo de tiempos de procesamiento almacenados."""

HEALTH_ISSUES_MAX: Final[int] = 100
"""Número máximo de issues de salud almacenados."""

HEALTH_ISSUES_TRIM: Final[int] = 50
"""Número de issues a mantener después de poda."""

# UMBRALES DE CONFIANZA Y CALIDAD

CONFIDENCE_HIGH: Final[float] = 0.7
"""Confianza alta (> 70%)."""

CONFIDENCE_MEDIUM: Final[float] = 0.5
"""Confianza media (> 50%)."""

CONFIDENCE_LOW: Final[float] = 0.3
"""Confianza baja (> 30%)."""

CONFIDENCE_MINIMUM: Final[float] = 0.3
"""Confianza mínima aceptable."""

CONFIDENCE_FOR_REID: Final[float] = 0.3
"""Confianza mínima para re-identificación."""

CONFIDENCE_FOR_MATCHING: Final[float] = 0.5
"""Confianza mínima para matching."""

SIMILARITY_HIGH: Final[float] = 0.85
"""Similitud alta para clasificación de trayectorias."""

SIMILARITY_MEDIUM: Final[float] = 0.5
"""Similitud media para clasificación de trayectorias."""

# UMBRALES DE BUFFER

BUFFER_USAGE_FULL: Final[float] = 0.7
"""Uso de buffer considerado como FULL."""

BUFFER_USAGE_OVERFLOW: Final[float] = 0.9
"""Uso de buffer considerado como OVERFLOW."""

BUFFER_USAGE_RECOVERY: Final[float] = 0.6
"""Uso de buffer para recuperación."""

# UMBRALES DE MEMORIA

MEMORY_WARNING_PERCENT: Final[int] = 75
"""Porcentaje de memoria para advertencia."""

MEMORY_CRITICAL_PERCENT: Final[int] = 85
"""Porcentaje de memoria para estado crítico."""

MEMORY_HIGH_PERCENT: Final[int] = 80
"""Porcentaje de memoria considerado alto."""

MEMORY_SAFE_PERCENT: Final[int] = 70
"""Porcentaje de memoria considerado seguro."""

# UMBRALES DE FPS

FPS_CRITICAL: Final[float] = 5.0
"""FPS considerado crítico."""

FPS_MINIMUM: Final[float] = 5.0
"""FPS mínimo aceptable."""

FPS_LOW: Final[float] = 15.0
"""FPS considerado bajo (para comparaciones)."""

FPS_TARGET: Final[float] = 30.0
"""FPS objetivo."""

# LÍMITES NUMÉRICOS

MAX_ERRORS: Final[int] = 5
"""Número máximo de errores consecutivos."""

MAX_RECENT_ERRORS: Final[int] = 10
"""Número máximo de errores en ventana de tiempo."""

MAX_RECONNECT_ATTEMPTS: Final[int] = 3
"""Número máximo de intentos de reconexión."""

MAX_TRACKS_IN_MHT: Final[int] = 30
"""Número máximo de tracks para considerar en MHT."""

MAX_HYPOTHESES_TEMP: Final[int] = 10
"""Número máximo de hipótesis temporales."""

MAX_RECOVERIES_PER_FRAME: Final[int] = 2
"""Número máximo de recuperaciones por frame."""

MIN_COORDINATE_VALUE: Final[float] = 0.0
"""Valor mínimo de coordenada."""

EPSILON: Final[float] = 1e-8
"""Valor epsilon para comparaciones numéricas."""

# UMBRALES DE RIESGO

RISK_LOW_THRESHOLD: Final[float] = 0.3
"""Umbral bajo de riesgo de colisión."""

RISK_MEDIUM_THRESHOLD: Final[float] = 0.5
"""Umbral medio de riesgo de colisión."""

RISK_HIGH_THRESHOLD: Final[float] = 0.7
"""Umbral alto de riesgo de colisión."""

RISK_CRITICAL_THRESHOLD: Final[float] = 0.6
"""Umbral crítico de riesgo de colisión."""

RISK_WARNING: Final[float] = 0.5
"""Umbral de advertencia de riesgo."""

# INTERVALOS DE TIEMPO

RECONNECT_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout de reconexión en segundos."""

STATUS_RECOVERY_TIMEOUT: Final[float] = 5.0
"""Timeout para recuperación de estado."""

BUFFER_RECOVERY_TIMEOUT: Final[float] = 60.0
"""Timeout para recuperación de buffer."""

MHT_PRUNE_TIMEOUT: Final[float] = 15.0
"""Timeout entre podas MHT."""

COOLDOWN_SECONDS: Final[float] = 2.0
"""Tiempo de cooldown estándar."""

# UMBRALES DE MOVIMIENTO

SPEED_MINIMUM: Final[float] = 0.5
"""Velocidad mínima para considerar movimiento."""

SPEED_CHANGE_MAX: Final[float] = 5.0
"""Cambio máximo de velocidad permitido."""

ANGLE_CHANGE_MAX: Final[float] = 1.0
"""Cambio máximo de ángulo permitido."""

MOTION_DELTA_MIN: Final[float] = 0.01
"""Delta mínimo de movimiento para considerar cambio."""

# UMBRALES DE DETECCIÓN

DETECTION_IOU_HIGH: Final[float] = 0.6
"""IoU considerada alta (para matching)."""

DETECTION_IOU_LOW: Final[float] = 0.3
"""IoU considerada baja."""

AREA_MINIMUM: Final[int] = 100
"""Área mínima para consideración."""

SIZE_SMALL: Final[int] = 20
"""Tamaño pequeño para dimensiones."""

SIZE_LARGE: Final[int] = 300
"""Tamaño grande para dimensiones."""

LARGE_TRACK_ID: Final[int] = 10000
"""ID de track considerado grande (para límites MHT)."""

# CONSTANTES DE VISUALIZACIÓN

MAX_LINES_IN_DASHBOARD: Final[int] = 4
"""Número máximo de líneas a mostrar en dashboard."""

MIN_ARROW_DELTA: Final[int] = 2
"""Delta mínimo para dibujar flecha de dirección."""

MIN_CONFIDENCE_FOR_DISPLAY: Final[float] = 0.3
"""Confianza mínima para mostrar información."""

MAX_PREDICTIONS_TO_DISPLAY: Final[int] = 5
"""Número máximo de predicciones a mostrar."""

MAX_RISK_TRACKS_TO_DISPLAY: Final[int] = 5
"""Número máximo de tracks con riesgo a mostrar."""

# CONSTANTES DE SENSOR FUSION

MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión."""

MAX_OBSERVATIONS_PER_TRACK: Final[int] = 50
"""Máximo de observaciones por track."""

MIN_FUSION_WEIGHT: Final[float] = 0.01
"""Peso mínimo para fusión."""

MIN_ESTIMATE_CONFIDENCE: Final[float] = 0.1
"""Confianza mínima para estimación fusionada."""

# CONSTANTES DE RE-IDENTIFICACIÓN

MIN_FEATURES_FOR_REID: Final[int] = 3
"""Mínimo de features para re-identificación."""

MAX_SPATIAL_DISTANCE: Final[float] = 80.0
"""Distancia espacial máxima para re-identificación."""

MHT_RECOVERY_THRESHOLD: Final[float] = 0.35
"""Umbral de recuperación MHT."""

MHT_PROBABILITY_THRESHOLD: Final[float] = 0.3
"""Umbral de probabilidad MHT."""

MHT_HYPOTHESIS_PROBABILITY: Final[float] = 0.1
"""Umbral de probabilidad de hipótesis MHT."""

TEMP_HYPOTHESIS_COUNT: Final[int] = 2
"""Número de hipótesis temporales a crear."""

# CONSTANTES DE APRENDIZAJE

MIN_SAMPLES_FOR_LEARNING: Final[int] = 5
"""Mínimo de muestras para aprendizaje."""

MIN_SAMPLES_FOR_STRATEGY: Final[int] = 5
"""Mínimo de muestras para estrategia de aprendizaje."""

MAX_HISTORY_FOR_STRATEGY: Final[int] = 10
"""Máximo histórico para estrategias."""

# CONSTANTES DE TRACKING

MIN_HISTORY_FOR_VELOCITY: Final[int] = 2
"""Mínimo de puntos para calcular velocidad."""

MIN_HISTORY_FOR_ACCELERATION: Final[int] = 3
"""Mínimo de puntos para calcular aceleración."""

MIN_HISTORY_FOR_PREDICTION: Final[int] = 3
"""Mínimo de puntos para predicción."""

MIN_HISTORY_FOR_MHT: Final[int] = 3
"""Mínimo de puntos para MHT."""

MIN_POSITIONS_FOR_MHT: Final[int] = 3
"""Mínimo de posiciones para hipótesis MHT."""

TRACK_AGE_FOR_RECOVERY: Final[int] = 3
"""Edad mínima de track para recuperación."""

MAX_TRACKS_FOR_MHT_PRUNING: Final[int] = 30
"""Máximo de tracks para considerar en poda MHT."""

MHT_PREDICTION_SAMPLES: Final[int] = 5
"""Mínimo de muestras para predicción MHT."""

MHT_MAX_TEMP_HYPOTHESES: Final[int] = 10
"""Máximo de hipótesis temporales MHT."""

# CONSTANTES DE VALIDACIÓN

VALIDATION_MIN_SCORE: Final[float] = 0.3
"""Puntuación mínima para validación."""

VALIDATION_MIN_HITS: Final[int] = 3
"""Mínimo de hits para validación."""

MAX_VIOLATIONS: Final[int] = 2
"""Máximo de violaciones permitidas."""

MIN_SAMPLES_FOR_VALIDATION: Final[int] = 3
"""Mínimo de muestras para validación."""

MIN_POINTS_FOR_SMOOTHNESS: Final[int] = 5
"""Mínimo de puntos para verificar suavidad."""

MIN_BBOX_HISTORY: Final[int] = 3
"""Mínimo de bboxes en historial."""

MIN_ASPECT_RATIOS: Final[int] = 3
"""Mínimo de aspect ratios para análisis."""

# CONSTANTES DE PERFORMANCE

PERFORMANCE_ALERT_MS: Final[float] = 10.0
"""Umbral de alerta en milisegundos."""

MAX_MATRIX_SIZE: Final[int] = 50
"""Tamaño máximo de matriz para asignación greedy."""

INFINITE_COST_THRESHOLD: Final[float] = 1000.0
"""Umbral para considerar costo infinito."""

# EXPORTACIÓN

__all__ = [
    "POINT_DIMENSION",
    "BBOX_DIMENSION",
    "VELOCITY_DIMENSION",
    "IMAGE_CHANNELS_RGB",
    "IMAGE_CHANNELS_GRAY",
    "HISTORY_MAX_SIZE",
    "FEATURE_HISTORY_MAX_SIZE",
    "EVENT_HISTORY_MAX_SIZE",
    "TRANSITION_HISTORY_MAX_SIZE",
    "INFERENCE_TIMES_MAX",
    "BATCH_TIMES_MAX",
    "PROCESSING_TIMES_MAX",
    "HEALTH_ISSUES_MAX",
    "HEALTH_ISSUES_TRIM",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "CONFIDENCE_MINIMUM",
    "CONFIDENCE_FOR_REID",
    "CONFIDENCE_FOR_MATCHING",
    "SIMILARITY_HIGH",
    "SIMILARITY_MEDIUM",
    "BUFFER_USAGE_FULL",
    "BUFFER_USAGE_OVERFLOW",
    "BUFFER_USAGE_RECOVERY",
    "MEMORY_WARNING_PERCENT",
    "MEMORY_CRITICAL_PERCENT",
    "MEMORY_HIGH_PERCENT",
    "MEMORY_SAFE_PERCENT",
    "FPS_CRITICAL",
    "FPS_MINIMUM",
    "FPS_LOW",
    "FPS_TARGET",
    "MAX_ERRORS",
    "MAX_RECENT_ERRORS",
    "MAX_RECONNECT_ATTEMPTS",
    "MAX_TRACKS_IN_MHT",
    "MAX_HYPOTHESES_TEMP",
    "MAX_RECOVERIES_PER_FRAME",
    "MIN_COORDINATE_VALUE",
    "EPSILON",
    "RISK_LOW_THRESHOLD",
    "RISK_MEDIUM_THRESHOLD",
    "RISK_HIGH_THRESHOLD",
    "RISK_CRITICAL_THRESHOLD",
    "RISK_WARNING",
    "RECONNECT_TIMEOUT_SECONDS",
    "STATUS_RECOVERY_TIMEOUT",
    "BUFFER_RECOVERY_TIMEOUT",
    "MHT_PRUNE_TIMEOUT",
    "COOLDOWN_SECONDS",
    "SPEED_MINIMUM",
    "SPEED_CHANGE_MAX",
    "ANGLE_CHANGE_MAX",
    "MOTION_DELTA_MIN",
    "DETECTION_IOU_HIGH",
    "DETECTION_IOU_LOW",
    "AREA_MINIMUM",
    "SIZE_SMALL",
    "SIZE_LARGE",
    "LARGE_TRACK_ID",
    "MAX_LINES_IN_DASHBOARD",
    "MIN_ARROW_DELTA",
    "MIN_CONFIDENCE_FOR_DISPLAY",
    "MAX_PREDICTIONS_TO_DISPLAY",
    "MAX_RISK_TRACKS_TO_DISPLAY",
    "MIN_OBSERVATIONS",
    "MAX_OBSERVATIONS_PER_TRACK",
    "MIN_FUSION_WEIGHT",
    "MIN_ESTIMATE_CONFIDENCE",
    "MIN_FEATURES_FOR_REID",
    "MAX_SPATIAL_DISTANCE",
    "MHT_RECOVERY_THRESHOLD",
    "MHT_PROBABILITY_THRESHOLD",
    "MHT_HYPOTHESIS_PROBABILITY",
    "TEMP_HYPOTHESIS_COUNT",
    "MIN_SAMPLES_FOR_LEARNING",
    "MIN_SAMPLES_FOR_STRATEGY",
    "MAX_HISTORY_FOR_STRATEGY",
    "MIN_HISTORY_FOR_VELOCITY",
    "MIN_HISTORY_FOR_ACCELERATION",
    "MIN_HISTORY_FOR_PREDICTION",
    "MIN_HISTORY_FOR_MHT",
    "MIN_POSITIONS_FOR_MHT",
    "TRACK_AGE_FOR_RECOVERY",
    "MAX_TRACKS_FOR_MHT_PRUNING",
    "MHT_PREDICTION_SAMPLES",
    "MHT_MAX_TEMP_HYPOTHESES",
    "VALIDATION_MIN_SCORE",
    "VALIDATION_MIN_HITS",
    "MAX_VIOLATIONS",
    "MIN_SAMPLES_FOR_VALIDATION",
    "MIN_POINTS_FOR_SMOOTHNESS",
    "MIN_BBOX_HISTORY",
    "MIN_ASPECT_RATIOS",
    "PERFORMANCE_ALERT_MS",
    "MAX_MATRIX_SIZE",
    "INFINITE_COST_THRESHOLD",
]
