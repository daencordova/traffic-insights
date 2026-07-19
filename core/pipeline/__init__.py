"""
Módulo de pipeline del sistema de seguimiento de tráfico.
"""

from .sync_pipeline import VehicleCountingPipeline as SyncPipeline
from .async_pipeline import AsyncVehicleCountingPipeline
from .renderer import FrameRenderer
from .renderer_config import RendererConfig
from .render_pipeline import RenderPipeline, RenderLayer
from .text_utils import TextMetricsCache
from .system_info_renderer import SystemInfoRenderer
from .dashboard import DashboardRenderer
from .overlay import OverlayRenderer
from .controls import ControlHandler
from .context import VideoCaptureContext
from .system_info import (
    SystemInfo,
    SystemInfoCollector,
    get_system_info,
    set_system_status,
    get_system_status,
    get_system_info_collector,
)

__all__ = [
    "SyncPipeline",
    "VehicleCountingPipeline",
    "AsyncVehicleCountingPipeline",
    "FrameRenderer",
    "RendererConfig",
    "RenderPipeline",
    "RenderLayer",
    "TextMetricsCache",
    "SystemInfoRenderer",
    "DashboardRenderer",
    "OverlayRenderer",
    "ControlHandler",
    "VideoCaptureContext",
    "SystemInfo",
    "SystemInfoCollector",
    "get_system_info",
    "set_system_status",
    "get_system_status",
    "get_system_info_collector",
]

VehicleCountingPipeline = SyncPipeline
