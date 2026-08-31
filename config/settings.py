"""Centralized configuration system with Pydantic for robust validation.

This module provides a comprehensive configuration system for computer vision
applications using Pydantic models. It includes configurations for:
- Object detection models (YOLO, etc.)
- Camera input sources
- Multi-object tracking (with re-identification, MHT, sensor fusion)
- Lane detection
- Analytics and statistics
- Visualization and rendering
- Performance optimization
- Output and data storage

All configuration classes inherit from Pydantic's BaseModel, providing:
- Automatic type validation
- Field constraints (min/max values, length limits)
- Custom validators for complex business rules
- Serialization/deserialization to/from JSON/YAML
- Environment variable override support

Example:
    >>> from config import Config
    >>> config = Config()
    >>> config.model.confidence_threshold = 0.6
    >>> config.tracker.enable_reidentification = True
    >>> print(config.model.dict())
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.constants import (
    MIN_MHT_DEPTH,
    MIN_PARTICLE_COUNT,
    MIN_PREDICTION_HORIZON,
    MIN_REID_CACHE_SIZE,
)
from models.enums import DeviceType, TrackerType


class ModelConfig(BaseModel):
    """Configuration for the detection model.

    This class defines all parameters related to the object detection model,
    including model path, confidence thresholds, and inference settings.

    Attributes:
        model_path: Path to the model file or pretrained model name.
        confidence_threshold: Minimum confidence score for detections (0.1-1.0).
        iou_threshold: Intersection over Union threshold for NMS (0.1-0.9).
        vehicle_classes: List of class IDs to detect as vehicles.
        device: Device type for inference (auto, cpu, cuda).
        use_half_precision: Enable half precision (FP16) inference.
        use_onnx: Use ONNX runtime for inference.
        imgsz: Input image size for the model (must be multiple of 32).
        max_det: Maximum number of detections per frame.
    """

    model_config = ConfigDict(extra="ignore")

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.1, le=0.9)
    vehicle_classes: list[int] = Field(default=[2, 3, 5, 7])
    device: DeviceType = DeviceType.AUTO
    use_half_precision: bool = False
    use_onnx: bool = False
    imgsz: int = 640
    max_det: int = 100

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, v: int) -> int:
        """Validates that imgsz is a multiple of 32.

        Args:
            v: The image size value to validate.

        Returns:
            int: The validated image size.

        Raises:
            ValueError: If the image size is not a valid multiple of 32.
        """
        valid_sizes = [320, 416, 512, 640, 768, 832, 1024]
        if v not in valid_sizes:
            raise ValueError(f"imgsz must be a multiple of 32: {v}")
        return v


class CameraConfig(BaseModel):
    """Configuration for the camera input source.

    This class defines all parameters related to video capture from cameras
    or video files, including resolution, FPS, and buffering settings.

    Attributes:
        source: Camera source (device index, file path, or URL).
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second (None for automatic).
        buffer_size: Number of frames to buffer.
        capture_buffer_size: OpenCV capture buffer size (1-10).
        reconnect_attempts: Number of reconnection attempts on failure.
        reconnect_delay: Delay between reconnection attempts in seconds.
    """

    model_config = ConfigDict(extra="ignore")

    source: str = "0"
    width: int = 640
    height: int = 480
    fps: float | None = None
    buffer_size: int = 30
    capture_buffer_size: int = Field(
        1,
        ge=1,
        le=10,
        description="OpenCV capture buffer size (CV_CAP_PROP_BUFFERSIZE)",
    )
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0


class TrackerConfig(BaseModel):
    """Tracker configuration - FULL VERSION with all parameters.

    This class defines comprehensive tracking parameters including:
    - Basic tracking (max distance, frames missed, confirmation)
    - Re-identification (features, similarity thresholds)
    - Hierarchical matching (IOU, feature, motion, shape)
    - Multiple Hypothesis Tracking (MHT)
    - Sensor fusion (visual, depth, thermal, motion)
    - Path prediction (motion models, horizon, uncertainty)

    Attributes:
        type: Type of tracker to use.
        max_distance: Maximum distance for associating detections to tracks.
        max_frames_missed: Maximum frames a track can be missed before deletion.
        min_hits_to_confirm: Minimum detections to confirm a new track.
        max_active_tracks: Maximum number of active tracks.
        feature_model_path: Path to the feature extraction model.
        use_kalman: Enable Kalman filter for motion prediction.
        motion_model: Motion model type for prediction.
        min_motion_distance: Minimum distance for motion detection.
        motion_history_size: Number of frames to keep in motion history.
        enable_reidentification: Enable re-identification of tracks.
        reid_similarity_threshold: Threshold for re-identification similarity.
        reid_spatial_threshold: Spatial threshold for re-identification.
        reid_max_age_seconds: Maximum age of re-identification features.
        reid_cache_size: Size of the re-identification feature cache.
        reid_min_features: Minimum features needed for re-identification.
        enable_hierarchical_matching: Enable hierarchical matching strategy.
        iou_threshold: IOU threshold for hierarchical matching.
        feature_threshold: Feature similarity threshold.
        motion_threshold: Motion consistency threshold.
        shape_threshold: Shape similarity threshold.
        enable_adaptive_thresholds: Enable adaptive thresholding.
        enable_mht: Enable Multiple Hypothesis Tracking system.
        mht_max_depth: Maximum depth of MHT tree.
        mht_pruning_threshold: Pruning threshold for MHT hypotheses.
        mht_max_hypotheses: Maximum hypotheses per track.
        enable_sensor_fusion: Enable sensor fusion for tracking.
        fusion_method: Method for sensor fusion (weighted_average, particle_filter, bayesian).
        fusion_min_observations: Minimum observations for fusion.
        fusion_max_history: Maximum history size for fusion.
        fusion_particle_count: Number of particles for particle filter.
        fusion_visual_weight: Weight for visual sensor.
        fusion_depth_weight: Weight for depth sensor.
        fusion_thermal_weight: Weight for thermal sensor.
        fusion_motion_weight: Weight for motion sensor.
        enable_path_prediction: Enable path prediction.
        prediction_history_length: History length for prediction.
        prediction_horizon: Prediction horizon in seconds.
        prediction_steps: Number of prediction steps.
        prediction_min_samples: Minimum samples for prediction.
        prediction_motion_model: Motion model for prediction.
        prediction_uncertainty_threshold: Uncertainty threshold for prediction.
        max_search_radius: Maximum search radius for nearby tracks in pixels.
        tree_update_interval: KD-Tree update interval in seconds.
    """

    model_config = ConfigDict(extra="ignore")

    type: TrackerType = TrackerType.HYBRID
    max_distance: float = Field(50.0, ge=1.0, le=500.0)
    max_frames_missed: int = Field(30, ge=1, le=100)
    min_hits_to_confirm: int = Field(3, ge=1, le=20)
    max_active_tracks: int = Field(100, ge=1, le=500)
    feature_model_path: str | None = None
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

    enable_mht: bool = Field(
        default=False, description="Enable Multiple Hypothesis Tracking system"
    )
    mht_max_depth: int = Field(5, ge=1, le=20, description="Maximum depth of MHT tree")
    mht_pruning_threshold: float = Field(0.01, ge=0.0, le=0.5, description="MHT pruning threshold")
    mht_max_hypotheses: int = Field(3, ge=1, le=10, description="Maximum hypotheses per track")

    enable_sensor_fusion: bool = Field(default=False, description="Enable sensor fusion")
    fusion_method: Literal["weighted_average", "particle_filter", "bayesian"] = Field(
        "weighted_average", description="Fusion method"
    )
    fusion_min_observations: int = Field(
        2, ge=1, le=10, description="Minimum observations for fusion"
    )
    fusion_max_history: int = Field(50, ge=10, le=200, description="Maximum fusion history")
    fusion_particle_count: int = Field(
        500, ge=100, le=1000, description="Number of particles for particle filter"
    )

    fusion_visual_weight: float = Field(0.7, ge=0.0, le=1.0, description="Visual sensor weight")
    fusion_depth_weight: float = Field(0.5, ge=0.0, le=1.0, description="Depth sensor weight")
    fusion_thermal_weight: float = Field(0.4, ge=0.0, le=1.0, description="Thermal sensor weight")
    fusion_motion_weight: float = Field(0.3, ge=0.0, le=1.0, description="Motion sensor weight")

    enable_path_prediction: bool = Field(default=True, description="Enable path prediction")
    prediction_history_length: int = Field(
        30, ge=5, le=100, description="History length for prediction"
    )
    prediction_horizon: float = Field(
        2.0, ge=0.5, le=10.0, description="Prediction horizon in seconds"
    )
    prediction_steps: int = Field(20, ge=5, le=50, description="Number of prediction steps")
    prediction_min_samples: int = Field(
        5, ge=2, le=20, description="Minimum samples for prediction"
    )
    prediction_motion_model: Literal["linear", "adaptive", "curved", "cyclic"] = Field(
        "adaptive", description="Motion model for prediction"
    )
    prediction_uncertainty_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Uncertainty threshold"
    )

    max_search_radius: float = Field(
        150.0,
        ge=30.0,
        le=500.0,
        description="Maximum search radius for nearby tracks in pixels. "
        "Lower values = faster but may miss matches.",
    )
    tree_update_interval: float = Field(
        0.5, ge=0.1, le=2.0, description="KD-Tree update interval in seconds."
    )

    @field_validator("mht_max_depth")
    @classmethod
    def validate_mht_depth(cls, v: int, info) -> int:
        """Validates that mht_max_depth meets minimum requirement if MHT is enabled.

        Args:
            v: The MHT max depth value.
            info: Validation context containing other field values.

        Returns:
            int: The validated MHT max depth.

        Raises:
            ValueError: If MHT is enabled and depth is below minimum.
        """
        enable_mht = info.data.get("enable_mht", False)
        if enable_mht and v < MIN_MHT_DEPTH:
            raise ValueError(
                f"mht_max_depth must be at least {MIN_MHT_DEPTH} if MHT is enabled: {v}"
            )
        return v

    @field_validator("fusion_particle_count")
    @classmethod
    def validate_particle_count(cls, v: int, info) -> int:
        """Validates particle count meets minimum for particle filter method.

        Args:
            v: The particle count value.
            info: Validation context containing other field values.

        Returns:
            int: The validated particle count.

        Raises:
            ValueError: If particle filter is enabled and count is below minimum.
        """
        enable_sensor_fusion = info.data.get("enable_sensor_fusion", False)
        fusion_method = info.data.get("fusion_method", "weighted_average")
        if enable_sensor_fusion and fusion_method == "particle_filter" and v < MIN_PARTICLE_COUNT:
            raise ValueError(
                f"fusion_particle_count must be at least {MIN_PARTICLE_COUNT} "
                f"for particle filter: {v}"
            )
        return v

    @field_validator("prediction_horizon")
    @classmethod
    def validate_prediction_horizon(cls, v: float, info) -> float:
        """Validates prediction horizon meets minimum if path prediction is enabled.

        Args:
            v: The prediction horizon value in seconds.
            info: Validation context containing other field values.

        Returns:
            float: The validated prediction horizon.

        Raises:
            ValueError: If path prediction is enabled and horizon is below minimum.
        """
        enable_path_prediction = info.data.get("enable_path_prediction", False)
        if enable_path_prediction and v < MIN_PREDICTION_HORIZON:
            raise ValueError(
                f"prediction_horizon must be at least {MIN_PREDICTION_HORIZON}s "
                f"if path prediction is enabled: {v}"
            )
        return v

    @field_validator("reid_cache_size")
    @classmethod
    def validate_reid_cache(cls, v: int, info) -> int:
        """Validates re-identification cache size meets minimum if re-ID is enabled.

        Args:
            v: The cache size value.
            info: Validation context containing other field values.

        Returns:
            int: The validated cache size.

        Raises:
            ValueError: If re-identification is enabled and cache size is below minimum.
        """
        enable_reidentification = info.data.get("enable_reidentification", False)
        if enable_reidentification and v < MIN_REID_CACHE_SIZE:
            raise ValueError(
                f"reid_cache_size must be at least {MIN_REID_CACHE_SIZE} "
                f"if re-identification is enabled: {v}"
            )
        return v


class LaneConfig(BaseModel):
    """Configuration for lane detection.

    This class defines parameters for lane detection and calibration,
    including automatic calibration, confidence thresholds, and lane properties.

    Attributes:
        enable_detection: Enable lane detection.
        enable_auto_calibration: Enable automatic lane calibration.
        calibration_interval: Number of frames between calibrations.
        min_lane_confidence: Minimum confidence for lane detection.
        max_lanes: Maximum number of lanes to detect.
        lane_width: Width of detected lanes in pixels.
        use_bird_eye_view: Use bird's eye view transformation.
        min_samples_for_calibration: Minimum samples needed for calibration.
        calibration_confidence_threshold: Confidence threshold for calibration.
    """

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
    """Configuration for analytics and statistics.

    This class defines parameters for real-time analytics,
    congestion detection, and traffic predictions.

    Attributes:
        enable_real_time: Enable real-time analytics.
        analysis_window: Analysis window size in seconds.
        update_interval: Update interval for analytics in seconds.
        congestion_low: Lower threshold for low congestion.
        congestion_medium: Medium congestion threshold.
        congestion_high: High congestion threshold.
        enable_predictions: Enable traffic predictions.
        prediction_horizon: Prediction horizon in seconds.
        prediction_samples: Number of samples for prediction.
    """

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
    """Configuration for visualization settings.

    This class defines parameters for rendering and displaying
    detection results, tracks, overlays, and dashboards.

    Attributes:
        show_detections: Show detection bounding boxes.
        show_tracks: Show track visualizations.
        show_trails: Show track trails.
        show_velocity_vectors: Show velocity vectors.
        show_occupancy: Show occupancy grid.
        show_heatmap: Show heatmap overlay.
        show_system_info: Show system information overlay.
        show_dashboard: Show analytics dashboard.
        show_controls_help: Show controls help overlay.
        show_track_ids: Show numeric track IDs.
        dashboard_position: Position of the dashboard on screen.
        trail_length: Length of track trails in frames.
        font_scale: Font scale for text overlays.
        line_thickness: Thickness of drawn lines.
        show_track_arrows: Show direction arrows on tracks.
        show_track_speed: Show real-time speed on tracks.
        show_track_confidence: Show track confidence.
        track_circle_style: Style for track circles (solid, outline, pulse).
    """

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
        default=True, description="Show numeric track IDs in visualization"
    )
    dashboard_position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left"
    trail_length: int = 30
    font_scale: float = 0.5
    line_thickness: int = 2

    show_track_arrows: bool = Field(default=True, description="Show direction arrows on tracks")
    show_track_speed: bool = Field(default=True, description="Show real-time speed")
    show_track_confidence: bool = Field(default=True, description="Show track confidence")
    track_circle_style: Literal["solid", "outline", "pulse"] = Field(
        "solid", description="Track circle style: solid, outline, pulse"
    )


class OptimizationConfig(BaseModel):
    """Configuration for optimization settings.

    This class defines performance optimization parameters including
    batch processing, parallel processing, memory management, and
    caching strategies.

    Attributes:
        enable_batch_processing: Enable batch processing of frames.
        batch_size: Number of frames per batch.
        max_batch_size: Maximum batch size.
        min_batch_size: Minimum batch size.
        batch_timeout: Timeout for batch collection in seconds.
        enable_parallel_processing: Enable parallel processing.
        max_workers: Maximum number of worker threads.
        enable_async_processing: Enable asynchronous processing.
        memory_limit_mb: Memory limit in MB.
        enable_memory_optimization: Enable memory optimization.
        memory_gc_threshold: Garbage collection threshold percentage.
        enable_performance_monitoring: Enable performance monitoring.
        monitor_interval: Monitoring interval in seconds.
        use_optimized_detector: Use optimized detector.
        use_optimized_kalman: Use optimized Kalman filter.
        use_optimized_geometry: Use optimized geometry calculations.
        enable_frame_pool: Enable frame pool for reuse.
        preallocate_memory: Preallocate memory for performance.
        use_sparse_matching: Use sparse matching for efficiency.
        sparse_matching_threshold: Threshold for sparse matching.
        max_search_radius: Maximum search radius for matching.
        feature_search_radius: Search radius for feature matching.
        use_batch_prediction: Enable batch prediction.
        prediction_batch_size: Batch size for predictions.
        prediction_workers: Number of workers for prediction.
        use_fast_feature_cache: Enable fast feature caching.
        feature_cache_nn_neighbors: Number of neighbors for feature cache.
        feature_cache_update_interval: Update interval for feature cache.
    """

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

    use_sparse_matching: bool = True
    sparse_matching_threshold: int = 50
    max_search_radius: float = 150.0
    feature_search_radius: float = 100.0

    use_batch_prediction: bool = True
    prediction_batch_size: int = 10
    prediction_workers: int = 4

    use_fast_feature_cache: bool = True
    feature_cache_nn_neighbors: int = 10
    feature_cache_update_interval: float = 0.5


class OutputConfig(BaseModel):
    """Configuration for output and data storage.

    This class defines parameters for exporting results,
    saving screenshots, and managing output directories.

    Attributes:
        screenshots_dir: Directory for saving screenshots.
        export_dir: Directory for exporting data.
        logs_dir: Directory for log files.
        export_stats: Enable statistics export.
        stats_export_format: Format for statistics export (json, csv, both).
        auto_save_interval: Auto-save interval in seconds.
    """

    model_config = ConfigDict(extra="ignore")

    screenshots_dir: str = "data/screenshots/"
    export_dir: str = "data/exports/"
    logs_dir: str = "data/logs/"
    export_stats: bool = True
    stats_export_format: Literal["json", "csv", "both"] = "json"
    auto_save_interval: int = 300


class Config(BaseModel):
    """Main system configuration.

    This is the root configuration class that aggregates all
    subsystem configurations into a single, validated object.

    Attributes:
        model: Model configuration for object detection.
        camera: Camera configuration for video capture.
        tracker: Tracker configuration for object tracking.
        lanes: Lane detection configuration.
        analytics: Analytics and statistics configuration.
        visualization: Visualization and rendering configuration.
        optimization: Performance optimization configuration.
        output: Output and data storage configuration.
        counting_lines: List of counting line definitions.
        debug: Enable debug mode.
        log_level: Logging level for the application.
    """

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    lanes: LaneConfig = Field(default_factory=LaneConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    counting_lines: list[dict[str, Any]] = Field(default_factory=list)

    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
