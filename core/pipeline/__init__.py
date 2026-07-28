"""
Módulo de pipeline del sistema de seguimiento de tráfico.
"""

from core.pipeline.async_pipeline import AsyncPipeline
from core.pipeline.context import VideoCaptureContext
from core.pipeline.controls import ControlHandler
from core.pipeline.dashboard import DashboardRenderer
from core.pipeline.overlay import OverlayRenderer
from core.pipeline.render_pipeline import RenderLayer, RenderPipeline
from core.pipeline.renderer import FrameRenderer
from core.pipeline.renderer_config import RendererConfig
from core.pipeline.sync_pipeline import SyncPipeline
from core.pipeline.system_info import (
    SystemInfo,
    SystemInfoCollector,
    get_system_info,
    get_system_info_collector,
    get_system_status,
    set_system_status,
)
from core.pipeline.system_info_renderer import SystemInfoRenderer
from core.pipeline.text_utils import TextMetricsCache

__all__ = [
    "SyncPipeline",
    "AsyncPipeline",
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
