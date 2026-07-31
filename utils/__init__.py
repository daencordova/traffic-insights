"""
Módulo de utilidades generales
"""

from utils.color_manager import ColorManager, get_color, get_color_manager
from utils.geometry import (
    calculate_centroid,
    calculate_iou,
    check_crossing,
    euclidean_distance,
    point_in_bbox,
)
from utils.helpers import (
    ensure_directory_exists,
    format_time,
    get_timestamp_filename,
)
from utils.logger import (
    LoggerMixin,
    StructuredLogger,
    config,
    get_default_logger,
    get_logger,
    get_logger_for_class,
    get_logging_status,
    log_context,
    set_module_level,
    setup_logging,
    temporary_log_level,
)

__all__ = [
    "calculate_centroid",
    "calculate_iou",
    "check_crossing",
    "euclidean_distance",
    "point_in_bbox",
    "ensure_directory_exists",
    "get_timestamp_filename",
    "format_time",
    "setup_logging",
    "get_logger",
    "get_logger_for_class",
    "get_default_logger",
    "StructuredLogger",
    "LoggerMixin",
    "config",
    "get_logging_status",
    "set_module_level",
    "temporary_log_level",
    "log_context",
    "ColorManager",
    "get_color_manager",
    "get_color",
]
