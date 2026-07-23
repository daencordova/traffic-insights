"""
Configuración centralizada con Pydantic para validación robusta
"""

from typing import List, Optional, Tuple, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class DeviceType(str, Enum):
    """Tipos de dispositivos soportados"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class TrackerType(str, Enum):
    """Tipos de tracker disponibles"""
    CENTROID = "centroid"
    DEEPSORT = "deepsort"
    HYBRID = "hybrid"


Point = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]
Color = Tuple[int, int, int]
LinePoints = List[Point]
DetectionDict = Dict[str, Any]
TrackDict = Dict[int, Dict[str, Any]]
StatsDict = Dict[str, Any]


class ModelConfig(BaseModel):
    """Configuración del modelo de detección"""
    model_config = ConfigDict(extra="ignore")

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.1, le=0.9)
    vehicle_classes: List[int] = Field(default=[2, 3, 5, 7])
    device: DeviceType = DeviceType.AUTO
    use_half_precision: bool = False
    use_onnx: bool = False
    imgsz: int = 640
    max_det: int = 100

    @field_validator('imgsz')
    def validate_imgsz(cls, v: int) -> int:
        valid_sizes = [320, 416, 512, 640, 768, 832, 1024]
        if v not in valid_sizes:
            raise ValueError(f'imgsz debe ser múltiplo de 32: {v}')
        return v


class CameraConfig(BaseModel):
    """Configuración de la cámara"""
    model_config = ConfigDict(extra="ignore")

    source: str = "0"
    width: int = 640
    height: int = 480
    fps: Optional[float] = None
    buffer_size: int = 30
    capture_buffer_size: int = Field(1, ge=1, le=10, description="Tamaño del buffer de captura de OpenCV (CV_CAP_PROP_BUFFERSIZE)")
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0


class TrackerConfig(BaseModel):
    """Configuración del tracker - VERSIÓN COMPLETA con todos los parámetros"""
    model_config = ConfigDict(extra="ignore")

    type: TrackerType = TrackerType.HYBRID
    max_distance: float = Field(50.0, ge=1.0, le=500.0)
    max_frames_missed: int = Field(30, ge=1, le=100)
    min_hits_to_confirm: int = Field(3, ge=1, le=20)
    max_active_tracks: int = Field(100, ge=1, le=500)
    feature_model_path: Optional[str] = None
    use_kalman: bool = True
    motion_model: Literal["constant_velocity", "constant_acceleration"] = "constant_velocity"
    min_motion_distance: float = Field(5.0, ge=0.0, le=100.0)
    motion_history_size: int = Field(10, ge=1, le=50)

    enable_reidentification: bool = True
    reid_similarity_threshold: float = Field(0.6, ge=0.1, le=0.95)
    reid_spatial_threshold: float = Field(100.0, ge=10.0, le=500.0)
    reid_max_age_seconds: float = Field(30.0, ge=1.0, le=120.0)
    reid_cache_size: int = Field(1000, ge=100, le=5000)
    reid_min_features: int = Field(3, ge=1, le=10)

    enable_hierarchical_matching: bool = True
    iou_threshold: float = Field(0.3, ge=0.1, le=0.7)
    feature_threshold: float = Field(0.6, ge=0.1, le=0.95)
    motion_threshold: float = Field(0.7, ge=0.1, le=0.95)
    shape_threshold: float = Field(0.5, ge=0.1, le=0.9)
    enable_adaptive_thresholds: bool = True

    enable_mht: bool = Field(False, description="Habilitar sistema MHT")
    mht_max_depth: int = Field(5, ge=1, le=20, description="Profundidad máxima del árbol MHT")
    mht_pruning_threshold: float = Field(0.01, ge=0.0, le=0.5, description="Umbral de poda MHT")
    mht_max_hypotheses: int = Field(3, ge=1, le=10, description="Máximo de hipótesis por track")

    enable_sensor_fusion: bool = Field(False, description="Habilitar fusión de sensores")
    fusion_method: Literal["weighted_average", "particle_filter", "bayesian"] = Field(
        "weighted_average", description="Método de fusión"
    )
    fusion_min_observations: int = Field(2, ge=1, le=10, description="Mínimo de observaciones para fusionar")
    fusion_max_history: int = Field(50, ge=10, le=200, description="Máximo historial de fusiones")
    fusion_particle_count: int = Field(500, ge=100, le=1000, description="Número de partículas (particle filter)")

    fusion_visual_weight: float = Field(0.7, ge=0.0, le=1.0, description="Peso del sensor visual")
    fusion_depth_weight: float = Field(0.5, ge=0.0, le=1.0, description="Peso del sensor de profundidad")
    fusion_thermal_weight: float = Field(0.4, ge=0.0, le=1.0, description="Peso del sensor térmico")
    fusion_motion_weight: float = Field(0.3, ge=0.0, le=1.0, description="Peso del sensor de movimiento")

    enable_path_prediction: bool = Field(True, description="Habilitar predicción de trayectoria")
    prediction_history_length: int = Field(30, ge=5, le=100, description="Longitud del historial para predicción")
    prediction_horizon: float = Field(2.0, ge=0.5, le=10.0, description="Horizonte de predicción en segundos")
    prediction_steps: int = Field(20, ge=5, le=50, description="Número de pasos de predicción")
    prediction_min_samples: int = Field(5, ge=2, le=20, description="Mínimo de muestras para predicción")
    prediction_motion_model: Literal["linear", "adaptive", "curved", "cyclic"] = Field(
        "adaptive", description="Modelo de movimiento para predicción"
    )
    prediction_uncertainty_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Umbral de incertidumbre")

    max_search_radius: float = Field(
        150.0,
        ge=30.0,
        le=500.0,
        description="Radio máximo para búsqueda de tracks cercanos en píxeles. "
                    "Valores más bajos = más rápido pero puede perder matches."
    )
    tree_update_interval: float = Field(
        0.5,
        ge=0.1,
        le=2.0,
        description="Intervalo de actualización del KD-Tree en segundos."
    )

    @field_validator('mht_max_depth')
    def validate_mht_depth(cls, v, info):
        """Valida que mht_max_depth sea >= 2 si MHT está activo."""
        enable_mht = info.data.get('enable_mht', False)
        if enable_mht and v < 2:
            raise ValueError(f"mht_max_depth debe ser al menos 2 si MHT está activo: {v}")
        return v

    @field_validator('fusion_particle_count')
    def validate_particle_count(cls, v, info):
        """Valida que fusion_particle_count sea >= 100 para particle filter."""
        enable_sensor_fusion = info.data.get('enable_sensor_fusion', False)
        fusion_method = info.data.get('fusion_method', 'weighted_average')
        if enable_sensor_fusion and fusion_method == 'particle_filter' and v < 100:
            raise ValueError(f"fusion_particle_count debe ser al menos 100 para particle filter: {v}")
        return v

    @field_validator('prediction_horizon')
    def validate_prediction_horizon(cls, v, info):
        """Valida que prediction_horizon sea >= 0.5s si path_prediction está activo."""
        enable_path_prediction = info.data.get('enable_path_prediction', False)
        if enable_path_prediction and v < 0.5:
            raise ValueError(f"prediction_horizon debe ser al menos 0.5s si path_prediction está activo: {v}")
        return v

    @field_validator('reid_cache_size')
    def validate_reid_cache(cls, v, info):
        """Valida que reid_cache_size sea >= 100 si re-identificación está activa."""
        enable_reidentification = info.data.get('enable_reidentification', False)
        if enable_reidentification and v < 100:
            raise ValueError(f"reid_cache_size debe ser al menos 100 si re-identificación está activa: {v}")
        return v


class LaneConfig(BaseModel):
    """Configuración de carriles"""
    model_config = ConfigDict(extra="ignore")

    enable_detection: bool = False
    enable_auto_calibration: bool = True
    calibration_interval: int = 300
    min_lane_confidence: float = Field(0.6, ge=0.1, le=1.0)
    max_lanes: int = 6
    lane_width: int = 40
    use_bird_eye_view: bool = True
    min_samples_for_calibration: int = 50
    calibration_confidence_threshold: float = 0.7


class AnalyticsConfig(BaseModel):
    """Configuración de análisis"""
    model_config = ConfigDict(extra="ignore")

    enable_real_time: bool = True
    analysis_window: int = 60
    update_interval: float = 1.0
    congestion_low: float = 0.3
    congestion_medium: float = 0.6
    congestion_high: float = 0.8
    enable_predictions: bool = True
    prediction_horizon: int = 300
    prediction_samples: int = 100


class VisualizationConfig(BaseModel):
    """Configuración de visualización"""
    model_config = ConfigDict(extra="ignore")

    show_detections: bool = True
    show_tracks: bool = True
    show_trails: bool = True
    show_velocity_vectors: bool = True
    show_occupancy: bool = True
    show_heatmap: bool = False
    show_system_info: bool = True
    show_dashboard: bool = True
    show_controls_help: bool = True
    show_track_ids: bool = Field(
        default=True,
        description="Mostrar IDs numéricos de los tracks en la visualización"
    )
    dashboard_position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left"
    trail_length: int = 30
    font_scale: float = 0.5
    line_thickness: int = 2

    show_track_arrows: bool = Field(
        True,
        description="Mostrar flechas de dirección en los tracks"
    )
    show_track_speed: bool = Field(
        True,
        description="Mostrar velocidad en tiempo real"
    )
    show_track_confidence: bool = Field(
        True,
        description="Mostrar confianza del track"
    )
    track_circle_style: Literal["solid", "outline", "pulse"] = Field(
        "solid",
        description="Estilo de círculo para tracks: solid, outline, pulse"
    )


class OptimizationConfig(BaseModel):
    """Configuración de optimización"""
    model_config = ConfigDict(extra="ignore")

    enable_batch_processing: bool = False
    batch_size: int = 4
    max_batch_size: int = 8
    min_batch_size: int = 2
    batch_timeout: float = 0.01

    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_async_processing: bool = False

    memory_limit_mb: int = 4096
    enable_memory_optimization: bool = True
    memory_gc_threshold: float = 70.0

    enable_performance_monitoring: bool = True
    monitor_interval: int = 60

    use_optimized_detector: bool = True
    use_optimized_kalman: bool = True
    use_optimized_geometry: bool = True
    enable_frame_pool: bool = True
    preallocate_memory: bool = True


class OutputConfig(BaseModel):
    """Configuración de salida"""
    model_config = ConfigDict(extra="ignore")

    screenshots_dir: str = "data/screenshots/"
    export_dir: str = "data/exports/"
    logs_dir: str = "data/logs/"
    export_stats: bool = True
    stats_export_format: Literal["json", "csv", "both"] = "json"
    auto_save_interval: int = 300


ConfigDictType = Dict[str, Any]


class Config(BaseModel):
    """Configuración principal del sistema"""
    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    lanes: LaneConfig = Field(default_factory=LaneConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    counting_lines: List[Dict[str, Any]] = Field(default_factory=list)

    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
