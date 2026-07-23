"""
Módulo de utilidades generales
"""

from utils.logger import setup_logger, LoggerMixin
from utils.color_manager import ColorManager, get_color_manager, get_color
from utils.geometry import (
    calculate_centroid,
    calculate_iou,
    check_crossing,
    euclidean_distance,
    point_in_bbox,
)
from utils.helpers import (
    ensure_directory_exists,
    get_timestamp_filename,
    format_time,
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
    "setup_logger",
    "LoggerMixin",
    "ColorManager",
    "get_color_manager",
    "get_color",
]
