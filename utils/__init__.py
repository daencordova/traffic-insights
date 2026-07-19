"""
Módulo de utilidades generales
"""

from .geometry import (
    calculate_centroid,
    calculate_iou,
    check_crossing,
    euclidean_distance,
    point_in_bbox,
)
from .helpers import (
    ensure_directory_exists,
    get_timestamp_filename,
    format_time,
)
from .logger import setup_logger, LoggerMixin

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
]
