"""Constantes numéricas centralizadas (valores, límites, umbrales).

Este módulo contiene todos los valores numéricos que aparecen en el código,
organizados por dominio de aplicación para facilitar su mantenimiento.

Principios:
    - Sin números mágicos en el código
    - Nombres descriptivos en UPPER_SNAKE_CASE
    - Comentarios que explican el propósito y contexto
    - Valores agrupados por categoría funcional
"""

from typing import Final

# DIMENSIONES Y TAMAÑOS

# Coordenadas y geometría
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

# Tamaños de imagen
DEFAULT_IMAGE_WIDTH: Final[int] = 640
"""Ancho de imagen por defecto."""

DEFAULT_IMAGE_HEIGHT: Final[int] = 480
"""Alto de imagen por defecto."""

MIN_FRAME_DIMENSION: Final[int] = 10
"""Dimensión mínima de un frame en píxeles."""

MIN_FRAME_WIDTH: Final[int] = 10
"""Ancho mínimo de un frame en píxeles."""

MIN_FRAME_HEIGHT: Final[int] = 10
"""Alto mínimo de un frame en píxeles."""

# Ventanas
MIN_WINDOW_WIDTH: Final[int] = 320
"""Ancho mínimo de la ventana de visualización."""

MIN_WINDOW_HEIGHT: Final[int] = 240
"""Alto mínimo de la ventana de visualización."""

MAX_WINDOW_WIDTH: Final[int] = 1920
"""Ancho máximo de la ventana de visualización."""

MAX_WINDOW_HEIGHT: Final[int] = 1080
"""Alto máximo de la ventana de visualización."""

DEFAULT_WINDOW_WIDTH: Final[int] = 1280
"""Ancho de ventana por defecto."""

DEFAULT_WINDOW_HEIGHT: Final[int] = 720
"""Alto de ventana por defecto."""

# Bounding boxes
MIN_BOX_SIZE: Final[int] = 10
"""Tamaño mínimo de un bounding box en píxeles."""

MAX_BOX_SIZE: Final[int] = 10000
"""Tamaño máximo de un bounding box en píxeles."""

MIN_DETECTION_AREA: Final[int] = 500
"""Área mínima de una detección en píxeles cuadrados."""

MAX_DETECTION_AREA: Final[int] = 100000
"""Área máxima de una detección en píxeles cuadrados."""

HEALTH_ISSUES_MAX: Final[int] = 100
"""Número máximo de issues de salud almacenados."""

HEALTH_ISSUES_TRIM: Final[int] = 50
"""Número de issues a mantener después de poda."""

# CONFIANZA Y CALIDAD

MIN_DETECTION_CONFIDENCE: Final[float] = 0.0
"""Confianza mínima permitida para una detección."""

MAX_DETECTION_CONFIDENCE: Final[float] = 1.0
"""Confianza máxima permitida para una detección."""

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.35
"""Umbral de confianza por defecto para detecciones."""

DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
"""Umbral de IoU por defecto para NMS."""

CONFIDENCE_HIGH: Final[float] = 0.7
"""Confianza considerada alta (> 70%)."""

CONFIDENCE_MEDIUM: Final[float] = 0.5
"""Confianza considerada media (> 50%)."""

CONFIDENCE_LOW: Final[float] = 0.3
"""Confianza considerada baja (> 30%)."""

CONFIDENCE_MINIMUM: Final[float] = 0.3
"""Confianza mínima aceptable para cualquier operación."""

CONFIDENCE_FOR_REID: Final[float] = 0.3
"""Confianza mínima para re-identificación."""

CONFIDENCE_FOR_MATCHING: Final[float] = 0.5
"""Confianza mínima para matching entre detecciones y tracks."""

SIMILARITY_HIGH: Final[float] = 0.85
"""Similitud considerada alta para clasificación de trayectorias."""

SIMILARITY_MEDIUM: Final[float] = 0.5
"""Similitud considerada media para clasificación de trayectorias."""

TRACK_VALIDATION_MIN_CONFIDENCE: Final[float] = 0.3
"""Confianza mínima para validación de tracks."""

MIN_VALID_SCORE: Final[float] = 0.3
"""Puntuación mínima para considerar una validación exitosa."""

VALIDATION_MIN_SCORE: Final[float] = 0.3
"""Puntuación mínima para validación de tracks."""

MIN_REGION_QUALITY: Final[float] = 0.3
"""Calidad mínima requerida para una región de imagen."""

MIN_CACHE_QUALITY: Final[float] = 0.3
"""Calidad mínima para almacenar en caché."""

# CONGESTIÓN Y ANÁLISIS

CONGESTION_LOW: Final[float] = 0.3
"""Umbral bajo de congestión (0-1)."""

CONGESTION_MEDIUM: Final[float] = 0.6
"""Umbral medio de congestión (0-1)."""

CONGESTION_HIGH: Final[float] = 0.8
"""Umbral alto de congestión (0-1)."""

ANALYSIS_WINDOW_SECONDS: Final[int] = 60
"""Ventana de análisis en segundos."""

PREDICTION_HORIZON_SECONDS: Final[int] = 300
"""Horizonte de predicción en segundos."""

PREDICTION_SAMPLES: Final[int] = 100
"""Número de muestras para predicción."""

# TIEMPO Y FPS

TARGET_FPS: Final[int] = 30
"""FPS objetivo del sistema."""

MIN_ACCEPTABLE_FPS: Final[int] = 15
"""FPS mínimo aceptable para rendimiento aceptable."""

CRITICAL_FPS: Final[int] = 5
"""FPS crítico por debajo del cual el sistema es inestable."""

FPS_CRITICAL: Final[float] = 5.0
"""FPS considerado crítico (alias de CRITICAL_FPS)."""

FPS_MINIMUM: Final[float] = 5.0
"""FPS mínimo aceptable (alias de MIN_ACCEPTABLE_FPS)."""

FPS_LOW: Final[float] = 15.0
"""FPS considerado bajo (para comparaciones)."""

MAX_RECOMMENDED_FPS: Final[int] = 60
"""FPS máximo recomendado."""

VALID_FPS_RANGE: Final[tuple[float, float]] = (1.0, 120.0)
"""Rango válido de FPS."""

# MEMORIA

MEMORY_WARNING_THRESHOLD: Final[float] = 70.0
"""Umbral de memoria para advertencia (%)."""

MEMORY_CRITICAL_THRESHOLD: Final[float] = 80.0
"""Umbral de memoria para estado crítico (%)."""

MEMORY_LIMIT_MB: Final[int] = 2048
"""Límite de memoria del sistema en MB."""

MEMORY_MINIMUM_AVAILABLE_MB: Final[int] = 500
"""Memoria mínima disponible en MB para operar."""

MINIMUM_BUFFER_MEMORY_MB: Final[int] = 500
"""Memoria mínima para buffer en MB."""

MAX_CACHE_MEMORY_MB: Final[int] = 250
"""Memoria máxima del caché en MB."""

MONITOR_MEMORY_CRITICAL_MB: Final[int] = 2000
"""Memoria crítica en MB para forzar GC."""

MONITOR_ALERT_THRESHOLD_MB: Final[float] = 100.0
"""Umbral de alerta de memoria en MB."""

MEMORY_WARNING_PERCENT: Final[int] = 75
"""Porcentaje de memoria para advertencia."""

MEMORY_CRITICAL_PERCENT: Final[int] = 85
"""Porcentaje de memoria para estado crítico."""

MEMORY_HIGH_PERCENT: Final[int] = 80
"""Porcentaje de memoria considerado alto."""

MEMORY_SAFE_PERCENT: Final[int] = 70
"""Porcentaje de memoria considerado seguro."""

# BUFFER Y COLA

BUFFER_DROP_THRESHOLD: Final[float] = 0.8
"""Umbral de ocupación para comenzar a descartar frames (0-1)."""

BUFFER_RECOVERY_THRESHOLD: Final[float] = 0.3
"""Umbral de ocupación para recuperar frames (0-1)."""

BUFFER_SKIP_MAX: Final[int] = 2
"""Máximo de frames a saltar."""

BUFFER_SKIP_CONSECUTIVE_LIMIT: Final[int] = 5
"""Límite de saltos consecutivos."""

BUFFER_USAGE_FULL: Final[float] = 0.7
"""Uso de buffer considerado como FULL (0-1)."""

BUFFER_USAGE_OVERFLOW: Final[float] = 0.9
"""Uso de buffer considerado como OVERFLOW (0-1)."""

BUFFER_USAGE_RECOVERY: Final[float] = 0.6
"""Uso de buffer para recuperación (0-1)."""

HEALTH_BUFFER_CRITICAL: Final[float] = 0.85
"""Umbral crítico de ocupación de buffer."""

HEALTH_BUFFER_WARNING: Final[float] = 0.7
"""Umbral de advertencia de ocupación de buffer."""

HEALTH_QUEUE_CRITICAL: Final[int] = 30
"""Tamaño crítico de cola."""

HEALTH_QUEUE_WARNING: Final[int] = 15
"""Tamaño de advertencia de cola."""

# Tamaños de buffer por modo
BUFFER_SIZE_CPU: Final[int] = 20
"""Tamaño de buffer recomendado en modo CPU."""

BUFFER_SIZE_GPU: Final[int] = 30
"""Tamaño de buffer recomendado en modo GPU."""

MAX_BUFFER_SIZE_CPU: Final[int] = 20
"""Tamaño máximo del buffer en modo CPU."""

MAX_BUFFER_SIZE_GPU: Final[int] = 30
"""Tamaño máximo del buffer en modo GPU."""

# WORKERS Y PARALELISMO

MAX_WORKERS_CPU: Final[int] = 4
"""Máximo de workers en modo CPU."""

MAX_WORKERS_GPU: Final[int] = 8
"""Máximo de workers en modo GPU."""

MIN_WORKERS_CPU: Final[int] = 2
"""Mínimo de workers en modo CPU."""

THREAD_POOL_MIN_WORKERS: Final[int] = 2
"""Mínimo de workers en thread pool."""

THREAD_POOL_MAX_WORKERS: Final[int] = 8
"""Máximo de workers en thread pool."""

# BATCH PROCESSING

DEFAULT_BATCH_SIZE: Final[int] = 4
"""Tamaño de lote por defecto."""

MAX_BATCH_SIZE: Final[int] = 8
"""Tamaño máximo de lote."""

MIN_BATCH_SIZE: Final[int] = 2
"""Tamaño mínimo de lote."""

BATCH_TIMEOUT: Final[float] = 0.01
"""Timeout para procesamiento por lotes en segundos."""

BATCH_TIMES_MAX: Final[int] = 50
"""Número máximo de tiempos de batch almacenados."""

# TRACKING

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
"""Distancia máxima para matching espacial en píxeles."""

MIN_MOTION_DISTANCE: Final[float] = 5.0
"""Distancia mínima para considerar movimiento en píxeles."""

MAX_DETECTIONS_PER_FRAME: Final[int] = 50
"""Máximo de detecciones por frame para matching."""

MIN_MATCH_SCORE: Final[float] = 0.1
"""Puntuación mínima para considerar un match."""

MAX_MATCH_RADIUS: Final[float] = 150.0
"""Radio máximo para búsqueda de matches en píxeles."""

MAX_SEARCH_RADIUS: Final[float] = 150.0
"""Radio máximo para búsqueda de tracks cercanos (alias)."""

# Validación de tracks
TRACK_VALIDATION_MAX_SPEED_CHANGE: Final[float] = 50.0
"""Cambio máximo de velocidad para validación en píxeles/frame."""

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
"""Umbral espacial para re-identificación en píxeles."""

REID_MAX_AGE_SECONDS: Final[float] = 30.0
"""Edad máxima en segundos para re-identificación."""

REID_CACHE_SIZE: Final[int] = 1000
"""Tamaño de caché de re-identificación."""

REID_MIN_FEATURES: Final[int] = 3
"""Mínimo de features para re-identificación."""

MIN_REID_CACHE_SIZE: Final[int] = 100
"""Tamaño mínimo de caché de re-identificación."""

MIN_FEATURES_FOR_REID: Final[int] = 3
"""Mínimo de features para re-identificación (alias)."""

MAX_SPATIAL_DISTANCE: Final[float] = 80.0
"""Distancia espacial máxima para re-identificación en píxeles."""

# SENSOR FUSION

SENSOR_FUSION_VISUAL_WEIGHT: Final[float] = 0.7
"""Peso del sensor visual en fusión."""

SENSOR_FUSION_DEPTH_WEIGHT: Final[float] = 0.5
"""Peso del sensor de profundidad en fusión."""

SENSOR_FUSION_THERMAL_WEIGHT: Final[float] = 0.4
"""Peso del sensor térmico en fusión."""

SENSOR_FUSION_MOTION_WEIGHT: Final[float] = 0.3
"""Peso del sensor de movimiento en fusión."""

SENSOR_FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión."""

SENSOR_FUSION_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de fusión."""

SENSOR_FUSION_PARTICLE_COUNT: Final[int] = 500
"""Número de partículas para filtro de partículas."""

MIN_PARTICLE_COUNT: Final[int] = 100
"""Mínimo de partículas para filtro de partículas."""

FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión (alias)."""

FUSION_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de fusión (alias)."""

FUSION_PARTICLE_COUNT: Final[int] = 500
"""Número de partículas para fusión (alias)."""

MIN_FUSION_WEIGHT: Final[float] = 0.01
"""Peso mínimo para fusión."""

MIN_ESTIMATE_CONFIDENCE: Final[float] = 0.1
"""Confianza mínima para estimación fusionada."""

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

# HISTORIAL Y CACHÉ

HISTORY_MAX_SIZE: Final[int] = 30
"""Tamaño máximo de historiales de tracks y bboxes."""

FEATURE_HISTORY_MAX_SIZE: Final[int] = 20
"""Tamaño máximo de historial de features."""

MAX_FEATURE_HISTORY: Final[int] = 20
"""Máximo de features en historial (alias)."""

MAX_BBOX_HISTORY: Final[int] = 30
"""Máximo de bboxes en historial (alias)."""

MAX_BBOX_HISTORY_DISPLAY: Final[int] = 10
"""Número máximo de bboxes a mostrar en el historial."""

MAX_BBOX_HISTORY_STORAGE: Final[int] = 30
"""Número máximo de bboxes a almacenar en el historial."""

MAX_EVENT_HISTORY: Final[int] = 1000
"""Máximo de eventos en historial."""

MAX_TRANSITION_HISTORY: Final[int] = 1000
"""Máximo de transiciones en historial."""

EVENT_HISTORY_MAX_SIZE: Final[int] = 1000
"""Tamaño máximo de historial de eventos (alias)."""

TRANSITION_HISTORY_MAX_SIZE: Final[int] = 1000
"""Tamaño máximo de historial de transiciones (alias)."""

# Tamaños de caché
CACHE_MIN_SIZE: Final[int] = 4
"""Tamaño mínimo del caché de detecciones."""

CACHE_MAX_SIZE: Final[int] = 64
"""Tamaño máximo del caché de detecciones."""

CACHE_DEFAULT_SIZE: Final[int] = 16
"""Tamaño por defecto del caché de detecciones."""

DEFAULT_CACHE_SIZE: Final[int] = 16
"""Tamaño por defecto del caché (alias)."""

MAX_CACHE_SIZE: Final[int] = 64
"""Tamaño máximo del caché (alias)."""

MIN_CACHE_SIZE: Final[int] = 4
"""Tamaño mínimo del caché (alias)."""

FEATURE_CACHE_MAX_SIZE: Final[int] = 500
"""Tamaño máximo de caché de features."""

FEATURE_CACHE_MAX_AGE: Final[float] = 3.0
"""Edad máxima de features en caché en segundos."""

CACHE_ENTRY_SIZE_ESTIMATE: Final[int] = 16
"""Tamaño estimado de una entrada de caché en bytes."""

# APRENDIZAJE EN LÍNEA

ONLINE_LEARNING_DEFAULT_LR: Final[float] = 0.05
"""Tasa de aprendizaje por defecto."""

ONLINE_LEARNING_MIN_SAMPLES: Final[int] = 5
"""Mínimo de muestras para aprendizaje."""

ONLINE_LEARNING_DRIFT_THRESHOLD: Final[float] = 0.35
"""Umbral de drift de concepto."""

ONLINE_LEARNING_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de aprendizaje."""

MIN_SAMPLES_FOR_LEARNING: Final[int] = 5
"""Mínimo de muestras para aprendizaje (alias)."""

MIN_SAMPLES_FOR_STRATEGY: Final[int] = 5
"""Mínimo de muestras para estrategia de aprendizaje."""

MAX_HISTORY_FOR_STRATEGY: Final[int] = 10
"""Máximo histórico para estrategias."""

# MOVIMIENTO

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

MAX_ANGLE_CHANGE: Final[float] = 1.0
"""Cambio máximo de ángulo para considerar movimiento errático en radianes."""

MIN_SPEED_FOR_MOVEMENT: Final[float] = 0.5
"""Velocidad mínima para considerar movimiento en píxeles/frame."""

SPEED_MINIMUM: Final[float] = 0.5
"""Velocidad mínima para considerar movimiento (alias)."""

SPEED_CHANGE_MAX: Final[float] = 5.0
"""Cambio máximo de velocidad permitido."""

MOTION_DELTA_MIN: Final[float] = 0.01
"""Delta mínimo de movimiento para considerar cambio."""

# FEATURES

FEATURE_EXTRACTOR_DIM: Final[int] = 2048
"""Dimensión por defecto de features."""

SIFT_FEATURE_COUNT: Final[int] = 128
"""Número de features SIFT."""

# BENCHMARK

BENCHMARK_FRAMES: Final[int] = 50
"""Número de frames para benchmark."""

BENCHMARK_ITERATIONS: Final[int] = 50
"""Número de iteraciones para benchmark."""

# RENDIMIENTO

INFERENCE_TIMES_MAX: Final[int] = 100
"""Número máximo de tiempos de inferencia almacenados."""

PROCESSING_TIMES_MAX: Final[int] = 100
"""Número máximo de tiempos de procesamiento almacenados."""

MAX_RENDER_TIMES: Final[int] = 100
"""Máximo de tiempos de renderizado almacenados."""

PERFORMANCE_ALERT_MS: Final[float] = 10.0
"""Umbral de alerta en milisegundos."""

MONITOR_SAMPLE_INTERVAL: Final[float] = 5.0
"""Intervalo de muestreo para monitoreo en segundos."""

MONITOR_MAX_SAMPLES: Final[int] = 60
"""Máximo de muestras de monitoreo."""

# RIESGO

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

# TIEMPOS DE ESPERA Y TIMEOUTS

DEFAULT_SLEEP_SHORT: Final[float] = 0.001
"""Sleep corto para loops de alta frecuencia (1ms)."""

DEFAULT_SLEEP_MEDIUM: Final[float] = 0.01
"""Sleep medio para loops de frecuencia media (10ms)."""

DEFAULT_SLEEP_LONG: Final[float] = 0.1
"""Sleep largo para operaciones de baja frecuencia (100ms)."""

DEFAULT_TIMEOUT_SHORT: Final[float] = 0.05
"""Timeout corto (50ms)."""

DEFAULT_TIMEOUT_MEDIUM: Final[float] = 0.5
"""Timeout medio (500ms)."""

DEFAULT_TIMEOUT_LONG: Final[float] = 5.0
"""Timeout largo (5s)."""

DEFAULT_TIMEOUT_VERY_LONG: Final[float] = 30.0
"""Timeout muy largo (30s)."""

THREAD_POOL_IDLE_TIMEOUT: Final[float] = 30.0
"""Timeout para workers inactivos en segundos."""

THREAD_POOL_MAX_QUEUE_SIZE: Final[int] = 100
"""Tamaño máximo de cola de thread pool."""

THREAD_POOL_MAX_HISTORY: Final[int] = 1000
"""Máximo histórico de tareas en thread pool."""

# RECONEXIÓN Y ERRORES

MAX_RECONNECT_ATTEMPTS: Final[int] = 3
"""Número máximo de intentos de reconexión."""

CAPTURE_RECONNECT_ATTEMPTS: Final[int] = 5
"""Número de intentos de reconexión de captura."""

CAPTURE_RECONNECT_DELAY: Final[float] = 1.0
"""Delay de reconexión en segundos."""

CAPTURE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos antes de reconectar."""

MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos antes de acción."""

MAX_ERRORS_IN_WINDOW: Final[int] = 10
"""Máximo de errores en ventana de tiempo."""

ERROR_RECOVERY_COOLDOWN: Final[float] = 1.0
"""Cooldown para recuperación de errores en segundos."""

ERROR_WINDOW_SECONDS: Final[float] = 60.0
"""Ventana de tiempo para conteo de errores en segundos."""

RECONNECT_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout de reconexión en segundos."""

# COOLDOWNS

COOLDOWN_SECONDS: Final[float] = 2.0
"""Tiempo de cooldown estándar en segundos."""

COOLDOWN_RECOVERY: Final[float] = 3.0
"""Cooldown para recuperación de tracks en segundos."""

COOLDOWN_REIDENTIFICATION: Final[float] = 2.0
"""Cooldown para re-identificación en segundos."""

COOLDOWN_RENDER_ERROR: Final[float] = 1.0
"""Cooldown para errores de renderizado en segundos."""

MHT_PRUNE_TIMEOUT: Final[float] = 15.0
"""Timeout entre podas MHT en segundos."""

BUFFER_RECOVERY_TIMEOUT: Final[float] = 60.0
"""Timeout para recuperación de buffer en segundos."""

STATUS_RECOVERY_TIMEOUT: Final[float] = 5.0
"""Timeout para recuperación de estado en segundos."""

# MISCELÁNEO

MAX_LINES_IN_DASHBOARD: Final[int] = 4
"""Número máximo de líneas a mostrar en dashboard."""

MIN_ARROW_DELTA: Final[int] = 2
"""Delta mínimo para dibujar flecha de dirección en píxeles."""

MIN_CONFIDENCE_FOR_DISPLAY: Final[float] = 0.3
"""Confianza mínima para mostrar información."""

MAX_PREDICTIONS_TO_DISPLAY: Final[int] = 5
"""Número máximo de predicciones a mostrar."""

MAX_RISK_TRACKS_TO_DISPLAY: Final[int] = 5
"""Número máximo de tracks con riesgo a mostrar."""

MAX_MATRIX_SIZE: Final[int] = 50
"""Tamaño máximo de matriz para asignación greedy."""

INFINITE_COST_THRESHOLD: Final[float] = 1000.0
"""Umbral para considerar costo infinito."""

LARGE_TRACK_ID: Final[int] = 10000
"""ID de track considerado grande (para límites MHT)."""

AREA_MINIMUM: Final[int] = 100
"""Área mínima para consideración en píxeles cuadrados."""

SIZE_SMALL: Final[int] = 20
"""Tamaño pequeño para dimensiones en píxeles."""

SIZE_LARGE: Final[int] = 300
"""Tamaño grande para dimensiones en píxeles."""

EPSILON: Final[float] = 1e-8
"""Valor epsilon para comparaciones numéricas."""

MIN_COORDINATE_VALUE: Final[float] = 0.0
"""Valor mínimo de coordenada."""

# EXPORTACIÓN

__all__ = [
    # Dimensiones y tamaños
    "POINT_DIMENSION",
    "BBOX_DIMENSION",
    "VELOCITY_DIMENSION",
    "IMAGE_CHANNELS_RGB",
    "IMAGE_CHANNELS_GRAY",
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_IMAGE_HEIGHT",
    "MIN_FRAME_DIMENSION",
    "MIN_FRAME_WIDTH",
    "MIN_FRAME_HEIGHT",
    "MIN_WINDOW_WIDTH",
    "MIN_WINDOW_HEIGHT",
    "MAX_WINDOW_WIDTH",
    "MAX_WINDOW_HEIGHT",
    "DEFAULT_WINDOW_WIDTH",
    "DEFAULT_WINDOW_HEIGHT",
    "MIN_BOX_SIZE",
    "MAX_BOX_SIZE",
    "MIN_DETECTION_AREA",
    "MAX_DETECTION_AREA",
    # Confianza y calidad
    "MIN_DETECTION_CONFIDENCE",
    "MAX_DETECTION_CONFIDENCE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "CONFIDENCE_MINIMUM",
    "CONFIDENCE_FOR_REID",
    "CONFIDENCE_FOR_MATCHING",
    "SIMILARITY_HIGH",
    "SIMILARITY_MEDIUM",
    "TRACK_VALIDATION_MIN_CONFIDENCE",
    "MIN_VALID_SCORE",
    "VALIDATION_MIN_SCORE",
    "MIN_REGION_QUALITY",
    "MIN_CACHE_QUALITY",
    # Congestión
    "CONGESTION_LOW",
    "CONGESTION_MEDIUM",
    "CONGESTION_HIGH",
    "ANALYSIS_WINDOW_SECONDS",
    "PREDICTION_HORIZON_SECONDS",
    "PREDICTION_SAMPLES",
    # Tiempo y FPS
    "TARGET_FPS",
    "MIN_ACCEPTABLE_FPS",
    "CRITICAL_FPS",
    "FPS_CRITICAL",
    "FPS_MINIMUM",
    "FPS_LOW",
    "MAX_RECOMMENDED_FPS",
    "VALID_FPS_RANGE",
    # Memoria
    "MEMORY_WARNING_THRESHOLD",
    "MEMORY_CRITICAL_THRESHOLD",
    "MEMORY_LIMIT_MB",
    "MEMORY_MINIMUM_AVAILABLE_MB",
    "MINIMUM_BUFFER_MEMORY_MB",
    "MAX_CACHE_MEMORY_MB",
    "MONITOR_MEMORY_CRITICAL_MB",
    "MONITOR_ALERT_THRESHOLD_MB",
    "MEMORY_WARNING_PERCENT",
    "MEMORY_CRITICAL_PERCENT",
    "MEMORY_HIGH_PERCENT",
    "MEMORY_SAFE_PERCENT",
    # Buffer
    "BUFFER_DROP_THRESHOLD",
    "BUFFER_RECOVERY_THRESHOLD",
    "BUFFER_SKIP_MAX",
    "BUFFER_SKIP_CONSECUTIVE_LIMIT",
    "BUFFER_USAGE_FULL",
    "BUFFER_USAGE_OVERFLOW",
    "BUFFER_USAGE_RECOVERY",
    "HEALTH_BUFFER_CRITICAL",
    "HEALTH_BUFFER_WARNING",
    "HEALTH_QUEUE_CRITICAL",
    "HEALTH_QUEUE_WARNING",
    "BUFFER_SIZE_CPU",
    "BUFFER_SIZE_GPU",
    "MAX_BUFFER_SIZE_CPU",
    "MAX_BUFFER_SIZE_GPU",
    # Workers
    "MAX_WORKERS_CPU",
    "MAX_WORKERS_GPU",
    "MIN_WORKERS_CPU",
    "THREAD_POOL_MIN_WORKERS",
    "THREAD_POOL_MAX_WORKERS",
    # Batch
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "MIN_BATCH_SIZE",
    "BATCH_TIMEOUT",
    "BATCH_TIMES_MAX",
    # Tracking
    "MAX_ACTIVE_TRACKS",
    "MAX_LOST_TRACKS",
    "MAX_TRACK_HISTORY",
    "MIN_HITS_TO_CONFIRM",
    "MAX_FRAMES_MISSED",
    "IOU_THRESHOLD",
    "FEATURE_THRESHOLD",
    "MAX_MATCH_DISTANCE",
    "MIN_MOTION_DISTANCE",
    "MAX_DETECTIONS_PER_FRAME",
    "MIN_MATCH_SCORE",
    "MAX_MATCH_RADIUS",
    "MAX_SEARCH_RADIUS",
    "TRACK_VALIDATION_MAX_SPEED_CHANGE",
    "TRACK_VALIDATION_IOU_THRESHOLD",
    "TRACK_VALIDATION_FEATURE_THRESHOLD",
    "TRACK_VALIDATION_MOTION_THRESHOLD",
    "TRACK_VALIDATION_SHAPE_THRESHOLD",
    # Kalman
    "KALMAN_DT",
    "KALMAN_PROCESS_NOISE",
    "KALMAN_MEASUREMENT_NOISE",
    # MHT
    "MHT_MAX_DEPTH",
    "MHT_PRUNING_THRESHOLD",
    "MHT_MAX_HYPOTHESES",
    "MIN_MHT_DEPTH",
    # Re-identificación
    "REID_SIMILARITY_THRESHOLD",
    "REID_SPATIAL_THRESHOLD",
    "REID_MAX_AGE_SECONDS",
    "REID_CACHE_SIZE",
    "REID_MIN_FEATURES",
    "MIN_REID_CACHE_SIZE",
    "MIN_FEATURES_FOR_REID",
    "MAX_SPATIAL_DISTANCE",
    # Sensor Fusion
    "SENSOR_FUSION_VISUAL_WEIGHT",
    "SENSOR_FUSION_DEPTH_WEIGHT",
    "SENSOR_FUSION_THERMAL_WEIGHT",
    "SENSOR_FUSION_MOTION_WEIGHT",
    "SENSOR_FUSION_MIN_OBSERVATIONS",
    "SENSOR_FUSION_MAX_HISTORY",
    "SENSOR_FUSION_PARTICLE_COUNT",
    "MIN_PARTICLE_COUNT",
    "FUSION_MIN_OBSERVATIONS",
    "FUSION_MAX_HISTORY",
    "FUSION_PARTICLE_COUNT",
    "MIN_FUSION_WEIGHT",
    "MIN_ESTIMATE_CONFIDENCE",
    # Path Prediction
    "PATH_PREDICTION_HISTORY_LENGTH",
    "PATH_PREDICTION_HORIZON",
    "PATH_PREDICTION_STEPS",
    "PATH_PREDICTION_MIN_SAMPLES",
    "PATH_PREDICTION_UNCERTAINTY_THRESHOLD",
    "MIN_PREDICTION_HORIZON",
    # Historial y caché
    "HISTORY_MAX_SIZE",
    "FEATURE_HISTORY_MAX_SIZE",
    "MAX_FEATURE_HISTORY",
    "MAX_BBOX_HISTORY",
    "MAX_BBOX_HISTORY_DISPLAY",
    "MAX_BBOX_HISTORY_STORAGE",
    "MAX_EVENT_HISTORY",
    "MAX_TRANSITION_HISTORY",
    "EVENT_HISTORY_MAX_SIZE",
    "TRANSITION_HISTORY_MAX_SIZE",
    "CACHE_MIN_SIZE",
    "CACHE_MAX_SIZE",
    "CACHE_DEFAULT_SIZE",
    "DEFAULT_CACHE_SIZE",
    "MAX_CACHE_SIZE",
    "MIN_CACHE_SIZE",
    "FEATURE_CACHE_MAX_SIZE",
    "FEATURE_CACHE_MAX_AGE",
    "CACHE_ENTRY_SIZE_ESTIMATE",
    # Aprendizaje
    "ONLINE_LEARNING_DEFAULT_LR",
    "ONLINE_LEARNING_MIN_SAMPLES",
    "ONLINE_LEARNING_DRIFT_THRESHOLD",
    "ONLINE_LEARNING_MAX_HISTORY",
    "MIN_SAMPLES_FOR_LEARNING",
    "MIN_SAMPLES_FOR_STRATEGY",
    "MAX_HISTORY_FOR_STRATEGY",
    # Movimiento
    "MIN_HISTORY_FOR_VELOCITY",
    "MIN_HISTORY_FOR_ACCELERATION",
    "MIN_HISTORY_FOR_PREDICTION",
    "MIN_HISTORY_FOR_MHT",
    "MIN_POSITIONS_FOR_MHT",
    "MAX_ANGLE_CHANGE",
    "MIN_SPEED_FOR_MOVEMENT",
    "SPEED_MINIMUM",
    "SPEED_CHANGE_MAX",
    "MOTION_DELTA_MIN",
    # Features
    "FEATURE_EXTRACTOR_DIM",
    "SIFT_FEATURE_COUNT",
    # Benchmark
    "BENCHMARK_FRAMES",
    "BENCHMARK_ITERATIONS",
    # Rendimiento
    "INFERENCE_TIMES_MAX",
    "PROCESSING_TIMES_MAX",
    "MAX_RENDER_TIMES",
    "PERFORMANCE_ALERT_MS",
    "MONITOR_SAMPLE_INTERVAL",
    "MONITOR_MAX_SAMPLES",
    # Riesgo
    "RISK_LOW_THRESHOLD",
    "RISK_MEDIUM_THRESHOLD",
    "RISK_HIGH_THRESHOLD",
    "RISK_CRITICAL_THRESHOLD",
    "RISK_WARNING",
    # Timeouts
    "DEFAULT_SLEEP_SHORT",
    "DEFAULT_SLEEP_MEDIUM",
    "DEFAULT_SLEEP_LONG",
    "DEFAULT_TIMEOUT_SHORT",
    "DEFAULT_TIMEOUT_MEDIUM",
    "DEFAULT_TIMEOUT_LONG",
    "DEFAULT_TIMEOUT_VERY_LONG",
    "THREAD_POOL_IDLE_TIMEOUT",
    "THREAD_POOL_MAX_QUEUE_SIZE",
    "THREAD_POOL_MAX_HISTORY",
    # Reconexión y errores
    "MAX_RECONNECT_ATTEMPTS",
    "CAPTURE_RECONNECT_ATTEMPTS",
    "CAPTURE_RECONNECT_DELAY",
    "CAPTURE_MAX_CONSECUTIVE_ERRORS",
    "MAX_CONSECUTIVE_ERRORS",
    "MAX_ERRORS_IN_WINDOW",
    "ERROR_RECOVERY_COOLDOWN",
    "ERROR_WINDOW_SECONDS",
    "RECONNECT_TIMEOUT_SECONDS",
    # Cooldowns
    "COOLDOWN_SECONDS",
    "COOLDOWN_RECOVERY",
    "COOLDOWN_REIDENTIFICATION",
    "COOLDOWN_RENDER_ERROR",
    "MHT_PRUNE_TIMEOUT",
    "BUFFER_RECOVERY_TIMEOUT",
    "STATUS_RECOVERY_TIMEOUT",
    # Misceláneo
    "MAX_LINES_IN_DASHBOARD",
    "MIN_ARROW_DELTA",
    "MIN_CONFIDENCE_FOR_DISPLAY",
    "MAX_PREDICTIONS_TO_DISPLAY",
    "MAX_RISK_TRACKS_TO_DISPLAY",
    "MAX_MATRIX_SIZE",
    "INFINITE_COST_THRESHOLD",
    "LARGE_TRACK_ID",
    "AREA_MINIMUM",
    "SIZE_SMALL",
    "SIZE_LARGE",
    "EPSILON",
    "MIN_COORDINATE_VALUE",
]
