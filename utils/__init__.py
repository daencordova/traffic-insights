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
from utils.logger import LoggerMixin, setup_logger

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
