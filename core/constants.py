"""Centralized constants for the Traffic Insights system.

This module contains all system-wide constants organized by domain.
Following the principle of single source of truth to avoid duplication.

Organization:
    - BASE: Fundamental constants (dimensions, geometry)
    - CONFIG: Configuration limits and validation ranges
    - PIPELINE: Pipeline, buffer, and processing constants
    - TRACKING: Tracking, matching, Kalman, and MHT constants
    - VISION: Vision, detection, and image processing constants
    - VISUALIZATION: UI, dashboard, and display constants
    - ANALYTICS: Analysis, congestion, and statistics constants
    - SYSTEM: System, logging, and timeout constants
    - MISC: Miscellaneous constants
"""

from typing import Final

# BASE - Fundamental Constants

# Dimensions and Geometry
POINT_DIMENSION: Final[int] = 2
"""Dimension of a point (x, y)."""

BBOX_DIMENSION: Final[int] = 4
"""Dimension of a bounding box (x1, y1, x2, y2)."""

VELOCITY_DIMENSION: Final[int] = 2
"""Dimension of a velocity vector (vx, vy)."""

IMAGE_CHANNELS_RGB: Final[int] = 3
"""Number of channels in an RGB/BGR image."""

IMAGE_CHANNELS_GRAY: Final[int] = 1
"""Number of channels in a grayscale image."""

# Image Dimensions
DEFAULT_IMAGE_WIDTH: Final[int] = 640
"""Default image width in pixels."""

DEFAULT_IMAGE_HEIGHT: Final[int] = 480
"""Default image height in pixels."""

MIN_FRAME_DIMENSION: Final[int] = 10
"""Minimum frame dimension in pixels."""

MIN_FRAME_WIDTH: Final[int] = 10
"""Minimum frame width in pixels."""

MIN_FRAME_HEIGHT: Final[int] = 10
"""Minimum frame height in pixels."""

# Window Dimensions
MIN_WINDOW_WIDTH: Final[int] = 320
"""Minimum window width in pixels."""

MIN_WINDOW_HEIGHT: Final[int] = 240
"""Minimum window height in pixels."""

MAX_WINDOW_WIDTH: Final[int] = 1920
"""Maximum window width in pixels."""

MAX_WINDOW_HEIGHT: Final[int] = 1080
"""Maximum window height in pixels."""

DEFAULT_WINDOW_WIDTH: Final[int] = 1280
"""Default window width in pixels."""

DEFAULT_WINDOW_HEIGHT: Final[int] = 720
"""Default window height in pixels."""

# Bounding Box Limits
MIN_BBOX_SIZE: Final[int] = 10
"""Minimum bounding box size in pixels."""

MAX_BBOX_SIZE: Final[int] = 10000
"""Maximum bounding box size in pixels."""

MIN_DETECTION_AREA: Final[int] = 500
"""Minimum detection area in square pixels."""

MAX_DETECTION_AREA: Final[int] = 100000
"""Maximum detection area in square pixels."""

# Numeric Constants
EPSILON: Final[float] = 1e-8
"""Epsilon value for numerical comparisons."""

INFINITE_COST_THRESHOLD: Final[float] = 1000.0
"""Threshold to consider a cost as infinite."""

LARGE_TRACK_ID: Final[int] = 10000
"""Track ID considered large (for MHT limits)."""

AREA_MINIMUM: Final[int] = 100
"""Minimum area for consideration in square pixels."""

SIZE_SMALL: Final[int] = 20
"""Small size for dimensions in pixels."""

SIZE_LARGE: Final[int] = 300
"""Large size for dimensions in pixels."""

MIN_COORDINATE_VALUE: Final[float] = 0.0
"""Minimum allowed coordinate value."""

# CONFIG - Configuration Limits and Validation

# Validation Ranges
VALID_IMGSZ: Final[list[int]] = [320, 416, 512, 640, 768, 832, 1024]
"""Valid image sizes for the model (multiples of 32)."""

VALID_CONFIDENCE_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
"""Valid confidence range (0-1)."""

VALID_IOU_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
"""Valid IoU range (0-1)."""

VALID_FPS_RANGE: Final[tuple[float, float]] = (1.0, 120.0)
"""Valid FPS range (1-120)."""

MAX_RECOMMENDED_FPS: Final[int] = 60
"""Maximum recommended FPS for the system."""

# Memory Limits
MINIMUM_BUFFER_MEMORY_MB: Final[int] = 500
"""Minimum recommended buffer memory in MB."""

MAX_CACHE_MEMORY_MB: Final[int] = 250
"""Maximum cache memory in MB."""

MEMORY_LIMIT_MB: Final[int] = 2048
"""System memory limit in MB."""

MEMORY_MINIMUM_AVAILABLE_MB: Final[int] = 500
"""Minimum available memory in MB to operate."""

MEMORY_WARNING_PERCENT: Final[int] = 75
"""Memory percentage for warning."""

MEMORY_CRITICAL_PERCENT: Final[int] = 85
"""Memory percentage for critical state."""

MEMORY_SAFE_PERCENT: Final[int] = 70
"""Memory percentage considered safe."""

MEMORY_HIGH_PERCENT: Final[int] = 80
"""Memory percentage considered high."""

MONITOR_MEMORY_CRITICAL_MB: Final[int] = 2000
"""Critical memory in MB to force GC."""

MONITOR_ALERT_THRESHOLD_MB: Final[float] = 100.0
"""Memory alert threshold in MB."""

# Performance Limits
MAX_DETECTIONS_PER_FRAME: Final[int] = 50
"""Maximum detections per frame."""

MAX_MATRIX_SIZE_FOR_HUNGARIAN: Final[int] = 50
"""Maximum matrix size for Hungarian algorithm."""

MAX_MATRIX_SIZE: Final[int] = 50
"""Alias of MAX_MATRIX_SIZE_FOR_HUNGARIAN for compatibility."""

CACHE_ENTRY_SIZE_ESTIMATE: Final[int] = 16
"""Estimated cache entry size in bytes."""

MONITOR_SAMPLE_INTERVAL: Final[float] = 5.0
"""Monitoring sampling interval in seconds."""

MONITOR_MAX_SAMPLES: Final[int] = 60
"""Maximum number of monitoring samples."""

PERFORMANCE_ALERT_MS: Final[float] = 10.0
"""Performance alert threshold in milliseconds."""

# History Limits
HISTORY_MAX_SIZE: Final[int] = 30
"""Maximum history size for tracks and bboxes."""

FEATURE_HISTORY_MAX_SIZE: Final[int] = 20
"""Maximum feature history size."""

MAX_TRACK_HISTORY: Final[int] = 15
"""Alias of HISTORY_MAX_SIZE for compatibility."""

MAX_FEATURE_HISTORY: Final[int] = 20
"""Alias of FEATURE_HISTORY_MAX_SIZE for compatibility."""

MAX_EVENT_HISTORY: Final[int] = 1000
"""Maximum event history size."""

MAX_TRANSITION_HISTORY: Final[int] = 1000
"""Maximum transition history size."""

STATS_HISTORY_MAX_SIZE: Final[int] = 60
"""Maximum number of records in statistics history."""

HEALTH_ISSUES_MAX: Final[int] = 50
"""Maximum number of health issues to keep."""

HEALTH_ISSUES_TRIM: Final[int] = 25
"""Number of issues to keep after trimming."""

# Cache Limits
CACHE_MIN_SIZE: Final[int] = 4
"""Minimum detection cache size."""

CACHE_MAX_SIZE: Final[int] = 64
"""Maximum detection cache size."""

CACHE_DEFAULT_SIZE: Final[int] = 16
"""Default detection cache size."""

DEFAULT_CACHE_SIZE: Final[int] = 16
"""Default cache size (alias)."""

MAX_CACHE_SIZE: Final[int] = 64
"""Maximum cache size (alias)."""

MIN_CACHE_SIZE: Final[int] = 4
"""Minimum cache size (alias)."""

FEATURE_CACHE_MAX_SIZE: Final[int] = 500
"""Maximum feature cache size."""

FEATURE_CACHE_MAX_AGE: Final[float] = 3.0
"""Maximum age of features in cache in seconds."""

CACHE_CLEANUP_THRESHOLD: Final[float] = 0.6
"""Cache occupancy threshold for cleanup."""

DEFAULT_FEATURE_CACHE_SIZE: Final[int] = 500
"""Default feature cache size."""

DEFAULT_FEATURE_DIM: Final[int] = 2048
"""Default feature vector dimension."""

# Queue Limits
HEALTH_QUEUE_CRITICAL: Final[int] = 30
"""Critical queue size."""

HEALTH_QUEUE_WARNING: Final[int] = 15
"""Warning queue size."""

THREAD_POOL_MAX_QUEUE_SIZE: Final[int] = 100
"""Maximum thread pool queue size."""

THREAD_POOL_MAX_HISTORY: Final[int] = 1000
"""Maximum thread pool task history."""

# Supported Formats
SUPPORTED_IMAGE_FORMATS: Final[list[str]] = ["jpg", "png", "bmp", "tiff"]
"""Supported image formats."""

SUPPORTED_EXPORT_FORMATS: Final[list[str]] = ["json", "csv", "both"]
"""Supported export formats."""

# PIPELINE - Pipeline and Processing Constants

# Performance Targets
TARGET_FPS: Final[int] = 30
"""Target system FPS."""

MIN_ACCEPTABLE_FPS: Final[int] = 15
"""Minimum acceptable FPS for acceptable performance."""

CRITICAL_FPS: Final[int] = 5
"""Critical FPS below which the system is unstable."""

MEMORY_CHECK_INTERVAL: Final[int] = 30
"""Memory check interval in seconds."""

GC_INTERVAL: Final[int] = 60
"""Garbage collection interval in seconds."""

CLEANUP_INTERVAL: Final[int] = 50
"""Cleanup interval in frames."""

PERFORMANCE_MONITOR_INTERVAL: Final[int] = 60
"""Performance monitoring interval in seconds."""

# Frame Processing
MAX_FRAME_SKIP: Final[int] = 2
"""Maximum number of frames to skip in flow control."""

MIN_FRAME_SKIP: Final[int] = 1
"""Minimum number of frames to skip."""

PROCESS_EVERY_N_FRAMES: Final[int] = 1
"""Process every N frames (1 = all)."""

# Buffer Configuration
BUFFER_SIZE_CPU: Final[int] = 20
"""Buffer size in CPU mode."""

BUFFER_SIZE_GPU: Final[int] = 30
"""Buffer size in GPU mode."""

MAX_WORKERS_CPU: Final[int] = 4
"""Maximum workers in CPU mode."""

MAX_WORKERS_GPU: Final[int] = 8
"""Maximum workers in GPU mode."""

MIN_WORKERS_CPU: Final[int] = 2
"""Minimum workers in CPU mode."""

MAX_BUFFER_SIZE_CPU: Final[int] = 20
"""Maximum buffer size in CPU mode."""

MAX_BUFFER_SIZE_GPU: Final[int] = 30
"""Maximum buffer size in GPU mode."""

BUFFER_DROP_THRESHOLD: Final[float] = 0.8
"""Buffer occupancy threshold to start dropping frames."""

BUFFER_RECOVERY_THRESHOLD: Final[float] = 0.3
"""Buffer occupancy threshold to recover frames."""

BUFFER_SKIP_MAX: Final[int] = 2
"""Maximum frames to skip."""

BUFFER_SKIP_CONSECUTIVE_LIMIT: Final[int] = 5
"""Consecutive skip limit."""

# Capture Configuration
CAPTURE_MIN_FPS_CPU: Final[float] = 5.0
"""Minimum capture FPS in CPU mode."""

CAPTURE_MAX_FPS_CPU: Final[float] = 15.0
"""Maximum capture FPS in CPU mode."""

CAPTURE_TARGET_FPS_CPU: Final[float] = 8.0
"""Target capture FPS in CPU mode."""

CAPTURE_TARGET_FPS_GPU: Final[float] = 30.0
"""Target capture FPS in GPU mode."""

CAPTURE_DEFAULT_INTERVAL_CPU: Final[float] = 1.0 / 8.0
"""Default capture interval in CPU mode."""

CAPTURE_DEFAULT_INTERVAL_GPU: Final[float] = 1.0 / 30.0
"""Default capture interval in GPU mode."""

CAPTURE_BUFFER_MIN_SIZE: Final[int] = 1
"""Minimum capture buffer size."""

CAPTURE_BUFFER_MAX_SIZE: Final[int] = 10
"""Maximum capture buffer size."""

CAPTURE_BUFFER_DEFAULT_SIZE: Final[int] = 1
"""Default capture buffer size."""

CAPTURE_CV2_BUFFER_SIZE: Final[int] = 1
"""OpenCV buffer size (CV_CAP_PROP_BUFFERSIZE)."""

CAPTURE_FOURCC_MJPG: Final[int] = 0x47504A4D
"""FOURCC code for MJPG format."""

CAPTURE_RECONNECT_DELAY: Final[float] = 1.0
"""Reconnection delay in seconds."""

CAPTURE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Maximum consecutive errors before reconnecting."""

CAPTURE_RECONNECT_ATTEMPTS: Final[int] = 5
"""Number of reconnection attempts."""

# Pipeline Configuration
PIPELINE_MAX_RECONNECT_ATTEMPTS: Final[int] = 3
"""Pipeline maximum reconnection attempts."""

PIPELINE_RECONNECT_DELAY: Final[float] = 0.1
"""Pipeline reconnection delay."""

PIPELINE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Pipeline maximum consecutive errors."""

PIPELINE_DEFAULT_FRAME_TIMEOUT: Final[int] = 100
"""Default frame timeout."""

PIPELINE_MAX_RENDER_QUEUE_RATIO: Final[float] = 0.33
"""Maximum render queue ratio."""

PIPELINE_STATS_INTERVAL_DEFAULT: Final[float] = 5.0
"""Default pipeline statistics interval."""

# Health Checks
HEALTH_CHECK_INTERVAL: Final[float] = 10.0
"""Health check interval in seconds."""

HEALTH_BUFFER_CRITICAL: Final[float] = 0.85
"""Critical buffer occupancy threshold."""

HEALTH_BUFFER_WARNING: Final[float] = 0.7
"""Warning buffer occupancy threshold."""

HEALTH_FPS_CRITICAL: Final[float] = 3.0
"""Critical FPS for health check."""

HEALTH_FPS_WARNING: Final[float] = 8.0
"""Warning FPS for health check."""

HEALTH_DROP_RATE_CRITICAL: Final[float] = 0.3
"""Critical drop rate."""

HEALTH_DROP_RATE_WARNING: Final[float] = 0.1
"""Warning drop rate."""

# Batch Processing
DEFAULT_BATCH_SIZE: Final[int] = 4
"""Default batch size."""

MAX_BATCH_SIZE: Final[int] = 8
"""Maximum batch size."""

MIN_BATCH_SIZE: Final[int] = 2
"""Minimum batch size."""

BATCH_TIMEOUT: Final[float] = 0.01
"""Batch processing timeout."""

BATCH_TIMES_MAX: Final[int] = 50
"""Maximum number of batch times stored."""

# Thread Pool
THREAD_POOL_MIN_WORKERS: Final[int] = 2
"""Minimum workers in thread pool."""

THREAD_POOL_MAX_WORKERS: Final[int] = 8
"""Maximum workers in thread pool."""

THREAD_POOL_IDLE_TIMEOUT: Final[float] = 30.0
"""Idle timeout for workers."""

# Rendering
RENDER_ERROR_COOLDOWN: Final[float] = 1.0
"""Render error cooldown in seconds."""

MAX_RENDER_TIMES: Final[int] = 100
"""Maximum render times stored."""

MAX_INFERENCE_TIMES: Final[int] = 100
"""Maximum inference times stored."""

MAX_PROCESSING_TIMES: Final[int] = 100
"""Maximum processing times stored."""

# TRACKING - Tracking and Matching Constants

# General Tracking
MAX_ACTIVE_TRACKS: Final[int] = 50
"""Maximum number of simultaneously active tracks."""

MAX_LOST_TRACKS: Final[int] = 50
"""Maximum number of lost tracks stored."""

MAX_FRAMES_MISSED: Final[int] = 30
"""Maximum frames missed before deleting a track."""

MIN_HITS_TO_CONFIRM: Final[int] = 3
"""Minimum detections to confirm a track."""

IOU_THRESHOLD: Final[float] = 0.3
"""IoU threshold for detection-track matching."""

FEATURE_THRESHOLD: Final[float] = 0.5
"""Feature similarity threshold for re-identification."""

MAX_MATCH_DISTANCE: Final[float] = 50.0
"""Maximum distance for spatial matching."""

MIN_MOTION_DISTANCE: Final[float] = 5.0
"""Minimum distance to consider movement."""

# Matching
MIN_MATCH_SCORE: Final[float] = 0.1
"""Minimum score to consider a match."""

MAX_MATCH_RADIUS: Final[float] = 150.0
"""Maximum radius for match search."""

MAX_SEARCH_RADIUS: Final[float] = 150.0
"""Maximum radius for nearby track search."""

MAX_SPATIAL_DISTANCE: Final[float] = 80.0
"""Maximum spatial distance for re-identification."""

MAX_UNMATCHED_TRACKS: Final[int] = 100
"""Maximum allowed unmatched tracks."""

MAX_UNMATCHED_DETECTIONS: Final[int] = 100
"""Maximum allowed unmatched detections."""

# Track Validation
TRACK_VALIDATION_MIN_CONFIDENCE: Final[float] = 0.3
"""Minimum confidence for track validation."""

TRACK_VALIDATION_MAX_SPEED_CHANGE: Final[float] = 50.0
"""Maximum speed change for validation."""

TRACK_VALIDATION_IOU_THRESHOLD: Final[float] = 0.3
"""IoU threshold for validation."""

TRACK_VALIDATION_FEATURE_THRESHOLD: Final[float] = 0.6
"""Feature threshold for validation."""

TRACK_VALIDATION_MOTION_THRESHOLD: Final[float] = 0.7
"""Motion threshold for validation."""

TRACK_VALIDATION_SHAPE_THRESHOLD: Final[float] = 0.5
"""Shape threshold for validation."""

# Kalman Filter
KALMAN_DT: Final[float] = 1.0
"""Kalman delta time."""

KALMAN_PROCESS_NOISE: Final[float] = 0.03
"""Kalman process noise."""

KALMAN_MEASUREMENT_NOISE: Final[float] = 0.1
"""Kalman measurement noise."""

# MHT (Multi-Hypothesis Tracking)
MHT_MAX_DEPTH: Final[int] = 5
"""Maximum depth of MHT tree."""

MHT_PRUNING_THRESHOLD: Final[float] = 0.01
"""MHT hypothesis pruning threshold."""

MHT_MAX_HYPOTHESES: Final[int] = 3
"""Maximum hypotheses per track."""

MIN_MHT_DEPTH: Final[int] = 2
"""Minimum MHT depth."""

# Re-Identification
REID_SIMILARITY_THRESHOLD: Final[float] = 0.6
"""Re-identification similarity threshold."""

REID_SPATIAL_THRESHOLD: Final[float] = 100.0
"""Re-identification spatial threshold."""

REID_MAX_AGE_SECONDS: Final[float] = 30.0
"""Maximum age for re-identification in seconds."""

REID_CACHE_SIZE: Final[int] = 1000
"""Re-identification cache size."""

REID_MIN_FEATURES: Final[int] = 3
"""Minimum features for re-identification."""

MIN_REID_CACHE_SIZE: Final[int] = 100
"""Minimum re-identification cache size."""

MIN_FEATURES_FOR_REID: Final[int] = 3
"""Alias of REID_MIN_FEATURES for compatibility."""

REID_MATCH_TIMEOUT: Final[float] = 2.0
"""Re-identification match timeout."""

# Sensor Fusion
SENSOR_FUSION_VISUAL_WEIGHT: Final[float] = 0.7
"""Visual sensor weight in fusion."""

SENSOR_FUSION_DEPTH_WEIGHT: Final[float] = 0.5
"""Depth sensor weight in fusion."""

SENSOR_FUSION_THERMAL_WEIGHT: Final[float] = 0.4
"""Thermal sensor weight in fusion."""

SENSOR_FUSION_MOTION_WEIGHT: Final[float] = 0.3
"""Motion sensor weight in fusion."""

SENSOR_FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Minimum observations for fusion."""

SENSOR_FUSION_MAX_HISTORY: Final[int] = 50
"""Maximum fusion history."""

SENSOR_FUSION_PARTICLE_COUNT: Final[int] = 500
"""Particle count for particle filter."""

MIN_PARTICLE_COUNT: Final[int] = 100
"""Minimum particles for particle filter."""

FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Alias of SENSOR_FUSION_MIN_OBSERVATIONS."""

FUSION_MAX_HISTORY: Final[int] = 50
"""Alias of SENSOR_FUSION_MAX_HISTORY."""

FUSION_PARTICLE_COUNT: Final[int] = 500
"""Alias of SENSOR_FUSION_PARTICLE_COUNT."""

MIN_FUSION_WEIGHT: Final[float] = 0.01
"""Minimum weight for fusion."""

MIN_ESTIMATE_CONFIDENCE: Final[float] = 0.1
"""Minimum confidence for fused estimate."""

# Path Prediction
PATH_PREDICTION_HISTORY_LENGTH: Final[int] = 30
"""History length for prediction."""

PATH_PREDICTION_HORIZON: Final[float] = 2.0
"""Prediction horizon in seconds."""

PATH_PREDICTION_STEPS: Final[int] = 20
"""Number of prediction steps."""

PATH_PREDICTION_MIN_SAMPLES: Final[int] = 5
"""Minimum samples for prediction."""

PATH_PREDICTION_UNCERTAINTY_THRESHOLD: Final[float] = 0.7
"""Prediction uncertainty threshold."""

MIN_PREDICTION_HORIZON: Final[float] = 0.5
"""Minimum prediction horizon in seconds."""

# Features
FEATURE_EXTRACTOR_DIM: Final[int] = 2048
"""Default feature extractor dimension."""

SIFT_FEATURE_COUNT: Final[int] = 128
"""Number of SIFT features."""

SPATIAL_THRESHOLD_NORMALIZED: Final[float] = 100.0
"""Normalized spatial distance threshold for matching."""

# History
MAX_BBOX_HISTORY: Final[int] = 30
"""Maximum bboxes in history."""

MAX_BBOX_HISTORY_DISPLAY: Final[int] = 10
"""Maximum bboxes to display in history."""

MAX_BBOX_HISTORY_STORAGE: Final[int] = 30
"""Maximum bboxes to store in history."""

STATE_HISTORY_MAX: Final[int] = 50
"""Maximum state history size per track."""

# Detection Dimensions
DETECTION_POINT_DIMENSION: Final[int] = 2
"""Detection point dimension (x, y)."""

DETECTION_VELOCITY_DIMENSION: Final[int] = 2
"""Detection velocity dimension (vx, vy)."""

DETECTION_BBOX_DIMENSION: Final[int] = 4
"""Detection bounding box dimension (x1, y1, x2, y2)."""

# Online Learning
ONLINE_LEARNING_DEFAULT_LR: Final[float] = 0.05
"""Default online learning rate."""

ONLINE_LEARNING_MIN_SAMPLES: Final[int] = 5
"""Minimum samples for online learning."""

ONLINE_LEARNING_DRIFT_THRESHOLD: Final[float] = 0.35
"""Concept drift threshold."""

ONLINE_LEARNING_MAX_HISTORY: Final[int] = 50
"""Maximum online learning history."""

# Benchmark
BENCHMARK_FRAMES: Final[int] = 50
"""Number of frames for benchmark."""

BENCHMARK_ITERATIONS: Final[int] = 50
"""Number of iterations for benchmark."""

# States and Colors
STATUS_COLORS: Final[dict[str, tuple[tuple[int, int, int], str, str]]] = {
    "confirmed": ((0, 255, 0), "✅", "OK"),
    "lost": ((0, 255, 255), "⚠️", "Lost"),
    "tentative": ((255, 255, 0), "⏳", "New"),
    "dead": ((128, 128, 128), "💀", "Dead"),
}
"""Track status colors, icons, and text labels."""

PREDICTION_STATE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "stopped": (0, 0, 255),
    "accelerating": (0, 255, 255),
    "decelerating": (0, 165, 255),
    "turning": (255, 0, 255),
    "erratic": (255, 0, 0),
    "moving": (255, 255, 0),
    "unknown": (255, 255, 0),
}
"""Colors for prediction states."""

# VISION - Vision and Image Processing Constants

# Detection Parameters
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.35
"""Default confidence threshold for detections."""

DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
"""Default IoU threshold for NMS."""

MIN_DETECTION_CONFIDENCE: Final[float] = 0.0
"""Minimum allowed detection confidence."""

MAX_DETECTION_CONFIDENCE: Final[float] = 1.0
"""Maximum allowed detection confidence."""

# Colors
COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "CYAN": (255, 255, 0),
    "MAGENTA": (255, 0, 255),
    "ORANGE": (0, 165, 255),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (128, 128, 128),
    "DARK_GRAY": (64, 64, 64),
    "LIGHT_GRAY": (192, 192, 192),
}
"""Predefined colors in BGR format."""

DETECTION_COLORS: Final[list[tuple[int, int, int]]] = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
]
"""Color palette for detections and tracks."""

# Image Processing
DEFAULT_LANE_WIDTH: Final[int] = 40
"""Default lane width in pixels."""

DEFAULT_BUFFER_ZONE: Final[int] = 15
"""Default buffer zone in pixels."""

# Frame Dimensions
DEFAULT_FRAME_WIDTH: Final[int] = 640
"""Default frame width in pixels."""

DEFAULT_FRAME_HEIGHT: Final[int] = 480
"""Default frame height in pixels."""

DEFAULT_FRAME_CHANNELS: Final[int] = 3
"""Default frame channels."""

# Rendering
DEFAULT_RENDER_WIDTH: Final[int] = 640
"""Default render width in pixels."""

DEFAULT_RENDER_HEIGHT: Final[int] = 480
"""Default render height in pixels."""

DEFAULT_RENDER_CHANNELS: Final[int] = 3
"""Default render channels."""

DEFAULT_FONT: Final[int] = 0
"""Default font (OpenCV)."""

DEFAULT_FONT_SCALE: Final[float] = 0.5
"""Default font scale."""

DEFAULT_FONT_THICKNESS: Final[int] = 2
"""Default font thickness."""

DEFAULT_LINE_THICKNESS: Final[int] = 2
"""Default line thickness."""

# Image Preprocessing
IMAGE_RESIZE_DEFAULT: Final[tuple[int, int]] = (32, 32)
"""Default size for image resizing."""

IMAGE_PREPROCESS_DENOISE_STRENGTH: Final[int] = 3
"""Denoise strength for preprocessing."""

IMAGE_EQUALIZE_HISTOGRAM: Final[bool] = True
"""Whether to apply histogram equalization."""

IMAGE_ENHANCE_CONTRAST: Final[bool] = True
"""Whether to enhance contrast."""

# Window
WINDOW_NAME: Final[str] = "Tracking Vehicle System"
"""Main window name."""

# BBox Validation
MAX_BBOX_DIMENSION: Final[int] = 4
"""Maximum elements in a bounding box."""

MAX_BRIGHTNESS: Final[int] = 240
"""Maximum allowed brightness before considering overexposed."""

# VISUALIZATION - UI and Dashboard Constants

# Dashboard
DASHBOARD_WIDTH: Final[int] = 220
"""Dashboard width in pixels."""

DASHBOARD_HEIGHT: Final[int] = 120
"""Dashboard height in pixels."""

DASHBOARD_ALPHA: Final[float] = 0.7
"""Dashboard opacity (0 = transparent, 1 = opaque)."""

# Font and UI
FONT_SCALE: Final[float] = 0.5
"""Font scale for UI text."""

LINE_THICKNESS: Final[int] = 2
"""Line thickness for UI elements."""

POINT_RADIUS: Final[int] = 4
"""Point radius for UI elements."""

TRAIL_POINTS: Final[int] = 15
"""Number of points in trajectory trail."""

# Track Visualization
TRACK_ARROW_LENGTH_MIN: Final[int] = 10
"""Minimum track arrow length."""

TRACK_ARROW_LENGTH_MAX: Final[int] = 30
"""Maximum track arrow length."""

TRACK_CIRCLE_RADIUS: Final[int] = 6
"""Track circle radius."""

TRACK_CONFIDENCE_RADIUS_MIN: Final[int] = 2
"""Minimum track confidence radius."""

TRACK_CONFIDENCE_RADIUS_MAX: Final[int] = 6
"""Maximum track confidence radius."""

TRACK_TRAIL_THICKNESS_MIN: Final[int] = 1
"""Minimum track trail thickness."""

TRACK_TRAIL_THICKNESS_MAX: Final[int] = 2
"""Maximum track trail thickness."""

TRACK_BBOX_THICKNESS_MIN: Final[int] = 1
"""Minimum track bbox thickness."""

TRACK_BBOX_THICKNESS_MAX: Final[int] = 2
"""Maximum track bbox thickness."""

PREDICTION_POINT_RADIUS_MIN: Final[int] = 2
"""Minimum prediction point radius."""

PREDICTION_POINT_RADIUS_MAX: Final[int] = 5
"""Maximum prediction point radius."""

# User Controls
CONTROL_KEY_QUIT: Final[int] = ord("q")
"""Key to quit."""

CONTROL_KEY_ESCAPE: Final[int] = 27
"""ESC key to quit."""

CONTROL_KEY_PAUSE: Final[int] = ord(" ")
"""SPACE key to pause/resume."""

CONTROL_KEY_SCREENSHOT: Final[int] = ord("s")
"""S key for screenshot."""

CONTROL_KEY_RESET: Final[int] = ord("r")
"""R key to reset."""

CONTROL_KEY_HELP: Final[int] = ord("h")
"""H key for help."""

# Colors and Visualization
HUE_SEGMENTS: Final[int] = 6
"""Number of segments for HSV color wheel."""

HUE_CYCLE: Final[int] = 360
"""Full HSV color wheel cycle."""

SATURATION: Final[int] = 200
"""Default saturation for generated colors (0-255)."""

VALUE: Final[int] = 200
"""Default value/brightness for generated colors (0-255)."""

MAX_COLOR_INDEX: Final[int] = 255
"""Maximum color index value."""

COLOR_CHANNEL_MAX: Final[int] = 255
"""Maximum color channel value."""

# ANALYTICS - Analysis and Statistics Constants

# Congestion Levels
CONGESTION_LOW: Final[float] = 0.3
"""Low congestion threshold."""

CONGESTION_MEDIUM: Final[float] = 0.6
"""Medium congestion threshold."""

CONGESTION_HIGH: Final[float] = 0.8
"""High congestion threshold."""

ANALYSIS_WINDOW_SECONDS: Final[int] = 60
"""Analysis window in seconds."""

PREDICTION_HORIZON_SECONDS: Final[int] = 300
"""Prediction horizon in seconds."""

PREDICTION_SAMPLES: Final[int] = 100
"""Number of samples for prediction."""

# Export
AUTO_SAVE_INTERVAL_SECONDS: Final[int] = 300
"""Auto-save interval in seconds."""

# SYSTEM - System and Logging Constants

# Encoding and Files
DEFAULT_ENCODING: Final[str] = "utf-8"
"""Default file encoding."""

DEFAULT_CONFIG_FILENAME: Final[str] = "config.yaml"
"""Default configuration filename."""

DEFAULT_LOG_FILENAME: Final[str] = "system.log"
"""Default log filename."""

DEFAULT_CONFIG_PATH: Final[str] = "config.yaml"
"""Default configuration file path."""

DEFAULT_DATA_DIR: Final[str] = "data/"
"""Default data directory."""

SCREENSHOTS_DIR: Final[str] = "data/screenshots/"
"""Screenshots directory."""

EXPORTS_DIR: Final[str] = "data/exports/"
"""Exports directory."""

LOGS_DIR: Final[str] = "data/logs/"
"""Logs directory."""

# System Memory
BYTES_PER_KB: Final[int] = 1024
"""Bytes per kilobyte."""

BYTES_PER_MB: Final[int] = 1024 * 1024
"""Bytes per megabyte."""

# Devices
PREFERRED_DEVICE_ORDER: Final[list[str]] = ["cuda", "mps", "cpu"]
"""Preferred inference device order."""

# Logging
LOG_LEVELS: Final[dict[str, int]] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
"""Mapping of log levels to numeric values."""

LOG_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Log timestamp format."""

# Time
SECONDS_PER_MINUTE: Final[int] = 60
"""Seconds per minute."""

SECONDS_PER_HOUR: Final[int] = 3600
"""Seconds per hour."""

MILLISECONDS_PER_SECOND: Final[int] = 1000
"""Milliseconds per second."""

# Sleep and Timeouts
DEFAULT_SLEEP_SHORT: Final[float] = 0.001
"""Short sleep for high-frequency loops (1ms)."""

DEFAULT_SLEEP_MEDIUM: Final[float] = 0.01
"""Medium sleep for medium-frequency loops (10ms)."""

DEFAULT_SLEEP_LONG: Final[float] = 0.1
"""Long sleep for low-frequency operations (100ms)."""

DEFAULT_TIMEOUT_SHORT: Final[float] = 0.05
"""Short timeout (50ms)."""

DEFAULT_TIMEOUT_MEDIUM: Final[float] = 0.5
"""Medium timeout (500ms)."""

DEFAULT_TIMEOUT_LONG: Final[float] = 5.0
"""Long timeout (5s)."""

DEFAULT_TIMEOUT_VERY_LONG: Final[float] = 30.0
"""Very long timeout (30s)."""

# Errors and Recovery
MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Maximum consecutive errors before action."""

ERROR_RECOVERY_COOLDOWN: Final[float] = 1.0
"""Error recovery cooldown."""

ERROR_WINDOW_SECONDS: Final[float] = 60.0
"""Time window for error counting."""

MAX_ERRORS_IN_WINDOW: Final[int] = 10
"""Maximum errors in time window."""

COOLDOWN_RECOVERY: Final[float] = 3.0
"""Cooldown for track recovery."""

COOLDOWN_REIDENTIFICATION: Final[float] = 2.0
"""Cooldown for re-identification."""

COOLDOWN_RENDER_ERROR: Final[float] = 1.0
"""Cooldown for render errors."""

# MISC - Miscellaneous Constants

# Validation and Quality
MIN_VALIDATION_SCORE: Final[float] = 0.4
"""Minimum score to pass track validation (0-1)."""

MAX_VALIDATION_VIOLATIONS: Final[int] = 2
"""Maximum allowed validation violations for a track."""

MIN_REGION_QUALITY: Final[float] = 0.3
"""Minimum region quality for feature extraction (0-1)."""

MIN_REGION_AREA: Final[int] = 100
"""Minimum region area in square pixels for feature extraction."""

MIN_REGION_BRIGHTNESS: Final[int] = 10
"""Minimum region brightness (0-255) for feature extraction."""

MIN_REGION_CONTRAST: Final[int] = 5
"""Minimum region contrast (std dev) for feature extraction."""

# Tracking and Re-Identification
REID_COOLDOWN_SECONDS: Final[float] = 2.0
"""Cooldown to prevent repeated re-identification of the same track."""

MIN_MOVEMENT_FOR_ARROW: Final[int] = 2
"""Minimum movement in pixels to draw direction arrow."""

ARROW_LENGTH_SCALE: Final[float] = 1.5
"""Scale factor for direction arrow length based on velocity."""

CENTROID_DOT_RADIUS: Final[int] = 2
"""Centroid dot radius in pixels."""

HIGH_RISK_THRESHOLD: Final[float] = 0.6
"""Threshold for high collision risk (0-1)."""

MEDIUM_RISK_THRESHOLD: Final[float] = 0.3
"""Threshold for medium collision risk (0-1)."""

# Additional Sleep and Timeouts
SLEEP_RECONNECT_ATTEMPT: Final[float] = 0.5
"""Sleep between reconnection attempts in seconds."""

SLEEP_PAUSE_CHECK: Final[float] = 0.01
"""Sleep to check pause state in loops."""

SLEEP_MAIN_LOOP: Final[float] = 0.001
"""Sleep for high-frequency main loops."""

SLEEP_ERROR_RECOVERY: Final[float] = 0.1
"""Sleep between error recovery attempts."""

# Error Recovery
MAX_CONSECUTIVE_READ_ERRORS: Final[int] = 5
"""Maximum consecutive read errors before reconnecting."""

# Track Visualization
TRACK_CIRCLE_PULSE_MIN: Final[float] = 0.8
"""Minimum factor for track circle pulse (pulse style)."""

TRACK_CIRCLE_PULSE_MAX: Final[float] = 1.2
"""Maximum factor for track circle pulse (pulse style)."""

# Path Prediction
PREDICTION_LINE_ALPHA_MIN: Final[float] = 0.1
"""Minimum alpha for prediction lines (transparency)."""

PREDICTION_POINT_UNCERTAINTY_MIN: Final[float] = 0.1
"""Minimum uncertainty factor for prediction points."""

PREDICTION_POINT_UNCERTAINTY_MAX: Final[float] = 0.5
"""Maximum uncertainty factor for prediction points."""

# Collision
COLLISION_RISK_RADIUS: Final[int] = 25
"""Base radius for collision alert visualization in pixels."""

COLLISION_RISK_PULSE_AMPLITUDE: Final[int] = 10
"""Pulse amplitude for collision alerts in pixels."""

COLLISION_RISK_DISPLAY_LIMIT: Final[int] = 5
"""Maximum number of tracks with collision risk to display."""

# Frame Processing
FRAME_SKIP_RECOVERY_STEP: Final[int] = 2
"""Steps to reduce consecutive skip counter during recovery."""

FRAME_SKIP_INITIAL: Final[int] = 0
"""Initial frame skip counter value."""

# Text Rendering
TEXT_CACHE_DEFAULT_SIZE: Final[int] = 100
"""Default text metrics cache size."""

TEXT_PADDING: Final[int] = 2
"""Padding in pixels for text backgrounds."""

TEXT_CACHE_MAX_SIZE: Final[int] = 1000
"""Maximum text metrics cache size."""

# Performance Storage
RENDER_TIMES_MAX: Final[int] = 100
"""Maximum stored render times."""

INFERENCE_TIMES_MAX: Final[int] = 100
"""Maximum stored inference times."""

PROCESSING_TIMES_MAX: Final[int] = 100
"""Maximum stored processing times."""

# Cleanup Intervals
CLEANUP_INTERVAL_FEATURES: Final[float] = 10.0
"""Feature cache cleanup interval in seconds."""

CLEANUP_INTERVAL_MEMORY: Final[float] = 30.0
"""Memory cleanup interval in seconds."""

CLEANUP_INTERVAL_GC: Final[float] = 60.0
"""Garbage collection interval in seconds."""

# Recovery Cooldowns
RECOVERY_COOLDOWN: Final[float] = 3.0
"""Cooldown for track recovery in seconds."""

RECOVERY_REIDENTIFICATION_COOLDOWN: Final[float] = 2.0
"""Cooldown for re-identification in seconds."""

RECOVERY_RENDER_ERROR_COOLDOWN: Final[float] = 1.0
"""Cooldown for render errors in seconds."""

BUFFER_USAGE_RECOVERY_BOUNDARY = 0.6
"""Threshold for determining that the buffer is recovering."""

MAX_HEALTH_ISSUES = 50
"""Maximum number of health issues to keep in the history."""

HEALTH_ISSUES_TRIM_SIZE = 25
"""Number of issues to retain after pruning the history"""

MEMORY_WARNING_THRESHOLD: Final[float] = 70.0
"""Memory threshold for a warning (%)."""

BUFFER_USAGE_FULL: Final[float] = 0.7
"""Buffer usage is considered FULL (0-1)."""

BUFFER_USAGE_OVERFLOW: Final[float] = 0.9
"""Buffer usage classified as OVERFLOW (0-1)."""

BUFFER_USAGE_RECOVERY: Final[float] = 0.6
"""Use of a buffer for recovery (0-1)."""

MAX_LINES_IN_DASHBOARD: Final[int] = 4
"""Maximum number of rows to display on the dashboard."""

FPS_LOW: Final[float] = 15.0
"""FPS considered low (for comparison purposes)."""

MIN_CACHE_QUALITY: Final[float] = 0.3
"""Minimum quality for caching."""

MIN_HISTORY_FOR_VELOCITY: Final[int] = 2
"""Minimum number of points required to calculate speed."""

MIN_HISTORY_FOR_ACCELERATION: Final[int] = 3
"""Minimum number of points required to calculate acceleration."""

MIN_VALID_SCORE: Final[float] = 0.3
"""Minimum score required for validation to be considered successful."""

__all__ = [
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
    "MIN_BBOX_SIZE",
    "MAX_BBOX_SIZE",
    "MIN_DETECTION_AREA",
    "MAX_DETECTION_AREA",
    "EPSILON",
    "INFINITE_COST_THRESHOLD",
    "LARGE_TRACK_ID",
    "AREA_MINIMUM",
    "SIZE_SMALL",
    "SIZE_LARGE",
    "MIN_COORDINATE_VALUE",
    "VALID_IMGSZ",
    "VALID_CONFIDENCE_RANGE",
    "VALID_IOU_RANGE",
    "VALID_FPS_RANGE",
    "MAX_RECOMMENDED_FPS",
    "MINIMUM_BUFFER_MEMORY_MB",
    "MAX_CACHE_MEMORY_MB",
    "MEMORY_LIMIT_MB",
    "MEMORY_MINIMUM_AVAILABLE_MB",
    "MEMORY_WARNING_PERCENT",
    "MEMORY_CRITICAL_PERCENT",
    "MEMORY_SAFE_PERCENT",
    "MEMORY_HIGH_PERCENT",
    "MONITOR_MEMORY_CRITICAL_MB",
    "MONITOR_ALERT_THRESHOLD_MB",
    "MAX_DETECTIONS_PER_FRAME",
    "MAX_MATRIX_SIZE_FOR_HUNGARIAN",
    "MAX_MATRIX_SIZE",
    "CACHE_ENTRY_SIZE_ESTIMATE",
    "MONITOR_SAMPLE_INTERVAL",
    "MONITOR_MAX_SAMPLES",
    "PERFORMANCE_ALERT_MS",
    "HISTORY_MAX_SIZE",
    "FEATURE_HISTORY_MAX_SIZE",
    "MAX_TRACK_HISTORY",
    "MAX_FEATURE_HISTORY",
    "MAX_EVENT_HISTORY",
    "MAX_TRANSITION_HISTORY",
    "STATS_HISTORY_MAX_SIZE",
    "HEALTH_ISSUES_MAX",
    "HEALTH_ISSUES_TRIM",
    "CACHE_MIN_SIZE",
    "CACHE_MAX_SIZE",
    "CACHE_DEFAULT_SIZE",
    "DEFAULT_CACHE_SIZE",
    "MAX_CACHE_SIZE",
    "MIN_CACHE_SIZE",
    "FEATURE_CACHE_MAX_SIZE",
    "FEATURE_CACHE_MAX_AGE",
    "CACHE_CLEANUP_THRESHOLD",
    "DEFAULT_FEATURE_CACHE_SIZE",
    "DEFAULT_FEATURE_DIM",
    "HEALTH_QUEUE_CRITICAL",
    "HEALTH_QUEUE_WARNING",
    "THREAD_POOL_MAX_QUEUE_SIZE",
    "THREAD_POOL_MAX_HISTORY",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_EXPORT_FORMATS",
    "TARGET_FPS",
    "MIN_ACCEPTABLE_FPS",
    "CRITICAL_FPS",
    "MEMORY_CHECK_INTERVAL",
    "GC_INTERVAL",
    "CLEANUP_INTERVAL",
    "PERFORMANCE_MONITOR_INTERVAL",
    "MAX_FRAME_SKIP",
    "MIN_FRAME_SKIP",
    "PROCESS_EVERY_N_FRAMES",
    "BUFFER_SIZE_CPU",
    "BUFFER_SIZE_GPU",
    "MAX_WORKERS_CPU",
    "MAX_WORKERS_GPU",
    "MIN_WORKERS_CPU",
    "MAX_BUFFER_SIZE_CPU",
    "MAX_BUFFER_SIZE_GPU",
    "BUFFER_DROP_THRESHOLD",
    "BUFFER_RECOVERY_THRESHOLD",
    "BUFFER_SKIP_MAX",
    "BUFFER_SKIP_CONSECUTIVE_LIMIT",
    "CAPTURE_MIN_FPS_CPU",
    "CAPTURE_MAX_FPS_CPU",
    "CAPTURE_TARGET_FPS_CPU",
    "CAPTURE_TARGET_FPS_GPU",
    "CAPTURE_DEFAULT_INTERVAL_CPU",
    "CAPTURE_DEFAULT_INTERVAL_GPU",
    "CAPTURE_BUFFER_MIN_SIZE",
    "CAPTURE_BUFFER_MAX_SIZE",
    "CAPTURE_BUFFER_DEFAULT_SIZE",
    "CAPTURE_CV2_BUFFER_SIZE",
    "CAPTURE_FOURCC_MJPG",
    "CAPTURE_RECONNECT_DELAY",
    "CAPTURE_MAX_CONSECUTIVE_ERRORS",
    "CAPTURE_RECONNECT_ATTEMPTS",
    "PIPELINE_MAX_RECONNECT_ATTEMPTS",
    "PIPELINE_RECONNECT_DELAY",
    "PIPELINE_MAX_CONSECUTIVE_ERRORS",
    "PIPELINE_DEFAULT_FRAME_TIMEOUT",
    "PIPELINE_MAX_RENDER_QUEUE_RATIO",
    "PIPELINE_STATS_INTERVAL_DEFAULT",
    "HEALTH_CHECK_INTERVAL",
    "HEALTH_BUFFER_CRITICAL",
    "HEALTH_BUFFER_WARNING",
    "HEALTH_FPS_CRITICAL",
    "HEALTH_FPS_WARNING",
    "HEALTH_DROP_RATE_CRITICAL",
    "HEALTH_DROP_RATE_WARNING",
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "MIN_BATCH_SIZE",
    "BATCH_TIMEOUT",
    "BATCH_TIMES_MAX",
    "THREAD_POOL_MIN_WORKERS",
    "THREAD_POOL_MAX_WORKERS",
    "THREAD_POOL_IDLE_TIMEOUT",
    "RENDER_ERROR_COOLDOWN",
    "MAX_RENDER_TIMES",
    "MAX_INFERENCE_TIMES",
    "MAX_PROCESSING_TIMES",
    "MAX_ACTIVE_TRACKS",
    "MAX_LOST_TRACKS",
    "MAX_FRAMES_MISSED",
    "MIN_HITS_TO_CONFIRM",
    "IOU_THRESHOLD",
    "FEATURE_THRESHOLD",
    "MAX_MATCH_DISTANCE",
    "MIN_MOTION_DISTANCE",
    "MIN_MATCH_SCORE",
    "MAX_MATCH_RADIUS",
    "MAX_SEARCH_RADIUS",
    "MAX_SPATIAL_DISTANCE",
    "MAX_UNMATCHED_TRACKS",
    "MAX_UNMATCHED_DETECTIONS",
    "TRACK_VALIDATION_MIN_CONFIDENCE",
    "TRACK_VALIDATION_MAX_SPEED_CHANGE",
    "TRACK_VALIDATION_IOU_THRESHOLD",
    "TRACK_VALIDATION_FEATURE_THRESHOLD",
    "TRACK_VALIDATION_MOTION_THRESHOLD",
    "TRACK_VALIDATION_SHAPE_THRESHOLD",
    "KALMAN_DT",
    "KALMAN_PROCESS_NOISE",
    "KALMAN_MEASUREMENT_NOISE",
    "MHT_MAX_DEPTH",
    "MHT_PRUNING_THRESHOLD",
    "MHT_MAX_HYPOTHESES",
    "MIN_MHT_DEPTH",
    "REID_SIMILARITY_THRESHOLD",
    "REID_SPATIAL_THRESHOLD",
    "REID_MAX_AGE_SECONDS",
    "REID_CACHE_SIZE",
    "REID_MIN_FEATURES",
    "MIN_REID_CACHE_SIZE",
    "MIN_FEATURES_FOR_REID",
    "REID_MATCH_TIMEOUT",
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
    "PATH_PREDICTION_HISTORY_LENGTH",
    "PATH_PREDICTION_HORIZON",
    "PATH_PREDICTION_STEPS",
    "PATH_PREDICTION_MIN_SAMPLES",
    "PATH_PREDICTION_UNCERTAINTY_THRESHOLD",
    "MIN_PREDICTION_HORIZON",
    "FEATURE_EXTRACTOR_DIM",
    "SIFT_FEATURE_COUNT",
    "SPATIAL_THRESHOLD_NORMALIZED",
    "MAX_BBOX_HISTORY",
    "MAX_BBOX_HISTORY_DISPLAY",
    "MAX_BBOX_HISTORY_STORAGE",
    "STATE_HISTORY_MAX",
    "DETECTION_POINT_DIMENSION",
    "DETECTION_VELOCITY_DIMENSION",
    "DETECTION_BBOX_DIMENSION",
    "ONLINE_LEARNING_DEFAULT_LR",
    "ONLINE_LEARNING_MIN_SAMPLES",
    "ONLINE_LEARNING_DRIFT_THRESHOLD",
    "ONLINE_LEARNING_MAX_HISTORY",
    "BENCHMARK_FRAMES",
    "BENCHMARK_ITERATIONS",
    "STATUS_COLORS",
    "PREDICTION_STATE_COLORS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "MIN_DETECTION_CONFIDENCE",
    "MAX_DETECTION_CONFIDENCE",
    "COLORS",
    "DETECTION_COLORS",
    "DEFAULT_LANE_WIDTH",
    "DEFAULT_BUFFER_ZONE",
    "DEFAULT_FRAME_WIDTH",
    "DEFAULT_FRAME_HEIGHT",
    "DEFAULT_FRAME_CHANNELS",
    "DEFAULT_RENDER_WIDTH",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_CHANNELS",
    "DEFAULT_FONT",
    "DEFAULT_FONT_SCALE",
    "DEFAULT_FONT_THICKNESS",
    "DEFAULT_LINE_THICKNESS",
    "IMAGE_RESIZE_DEFAULT",
    "IMAGE_PREPROCESS_DENOISE_STRENGTH",
    "IMAGE_EQUALIZE_HISTOGRAM",
    "IMAGE_ENHANCE_CONTRAST",
    "WINDOW_NAME",
    "MAX_BBOX_DIMENSION",
    "MAX_BRIGHTNESS",
    "DASHBOARD_WIDTH",
    "DASHBOARD_HEIGHT",
    "DASHBOARD_ALPHA",
    "FONT_SCALE",
    "LINE_THICKNESS",
    "POINT_RADIUS",
    "TRAIL_POINTS",
    "TRACK_ARROW_LENGTH_MIN",
    "TRACK_ARROW_LENGTH_MAX",
    "TRACK_CIRCLE_RADIUS",
    "TRACK_CONFIDENCE_RADIUS_MIN",
    "TRACK_CONFIDENCE_RADIUS_MAX",
    "TRACK_TRAIL_THICKNESS_MIN",
    "TRACK_TRAIL_THICKNESS_MAX",
    "TRACK_BBOX_THICKNESS_MIN",
    "TRACK_BBOX_THICKNESS_MAX",
    "PREDICTION_POINT_RADIUS_MIN",
    "PREDICTION_POINT_RADIUS_MAX",
    "CONTROL_KEY_QUIT",
    "CONTROL_KEY_ESCAPE",
    "CONTROL_KEY_PAUSE",
    "CONTROL_KEY_SCREENSHOT",
    "CONTROL_KEY_RESET",
    "CONTROL_KEY_HELP",
    "HUE_SEGMENTS",
    "HUE_CYCLE",
    "SATURATION",
    "VALUE",
    "MAX_COLOR_INDEX",
    "COLOR_CHANNEL_MAX",
    "CONGESTION_LOW",
    "CONGESTION_MEDIUM",
    "CONGESTION_HIGH",
    "ANALYSIS_WINDOW_SECONDS",
    "PREDICTION_HORIZON_SECONDS",
    "PREDICTION_SAMPLES",
    "AUTO_SAVE_INTERVAL_SECONDS",
    "DEFAULT_ENCODING",
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_LOG_FILENAME",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "SCREENSHOTS_DIR",
    "EXPORTS_DIR",
    "LOGS_DIR",
    "BYTES_PER_KB",
    "BYTES_PER_MB",
    "PREFERRED_DEVICE_ORDER",
    "LOG_LEVELS",
    "LOG_TIMESTAMP_FORMAT",
    "SECONDS_PER_MINUTE",
    "SECONDS_PER_HOUR",
    "MILLISECONDS_PER_SECOND",
    "DEFAULT_SLEEP_SHORT",
    "DEFAULT_SLEEP_MEDIUM",
    "DEFAULT_SLEEP_LONG",
    "DEFAULT_TIMEOUT_SHORT",
    "DEFAULT_TIMEOUT_MEDIUM",
    "DEFAULT_TIMEOUT_LONG",
    "DEFAULT_TIMEOUT_VERY_LONG",
    "MAX_CONSECUTIVE_ERRORS",
    "ERROR_RECOVERY_COOLDOWN",
    "ERROR_WINDOW_SECONDS",
    "MAX_ERRORS_IN_WINDOW",
    "COOLDOWN_RECOVERY",
    "COOLDOWN_REIDENTIFICATION",
    "COOLDOWN_RENDER_ERROR",
    "MIN_VALIDATION_SCORE",
    "MAX_VALIDATION_VIOLATIONS",
    "MIN_REGION_QUALITY",
    "MIN_REGION_AREA",
    "MIN_REGION_BRIGHTNESS",
    "MIN_REGION_CONTRAST",
    "REID_COOLDOWN_SECONDS",
    "MIN_MOVEMENT_FOR_ARROW",
    "ARROW_LENGTH_SCALE",
    "CENTROID_DOT_RADIUS",
    "HIGH_RISK_THRESHOLD",
    "MEDIUM_RISK_THRESHOLD",
    "SLEEP_RECONNECT_ATTEMPT",
    "SLEEP_PAUSE_CHECK",
    "SLEEP_MAIN_LOOP",
    "SLEEP_ERROR_RECOVERY",
    "MAX_CONSECUTIVE_READ_ERRORS",
    "TRACK_CIRCLE_PULSE_MIN",
    "TRACK_CIRCLE_PULSE_MAX",
    "PREDICTION_LINE_ALPHA_MIN",
    "PREDICTION_POINT_UNCERTAINTY_MIN",
    "PREDICTION_POINT_UNCERTAINTY_MAX",
    "COLLISION_RISK_RADIUS",
    "COLLISION_RISK_PULSE_AMPLITUDE",
    "COLLISION_RISK_DISPLAY_LIMIT",
    "FRAME_SKIP_RECOVERY_STEP",
    "FRAME_SKIP_INITIAL",
    "TEXT_CACHE_DEFAULT_SIZE",
    "TEXT_PADDING",
    "TEXT_CACHE_MAX_SIZE",
    "RENDER_TIMES_MAX",
    "INFERENCE_TIMES_MAX",
    "PROCESSING_TIMES_MAX",
    "CLEANUP_INTERVAL_FEATURES",
    "CLEANUP_INTERVAL_MEMORY",
    "CLEANUP_INTERVAL_GC",
    "RECOVERY_COOLDOWN",
    "RECOVERY_REIDENTIFICATION_COOLDOWN",
    "RECOVERY_RENDER_ERROR_COOLDOWN",
]
