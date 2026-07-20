"""
Módulo de pipeline del sistema de seguimiento de tráfico.
"""

from core.pipeline.sync_pipeline import VehicleCountingPipeline as SyncPipeline
from core.pipeline.async_pipeline import AsyncVehicleCountingPipeline
from core.pipeline.renderer import FrameRenderer
from core.pipeline.renderer_config import RendererConfig
from core.pipeline.render_pipeline import RenderPipeline, RenderLayer
from core.pipeline.text_utils import TextMetricsCache
from core.pipeline.system_info_renderer import SystemInfoRenderer
from core.pipeline.dashboard import DashboardRenderer
from core.pipeline.overlay import OverlayRenderer
from core.pipeline.controls import ControlHandler
from core.pipeline.context import VideoCaptureContext
from core.pipeline.system_info import (
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
