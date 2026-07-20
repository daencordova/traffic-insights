"""
Servicios especializados del pipeline.

Cada servicio tiene una responsabilidad única y bien definida:
- CaptureService: Captura de video y gestión de frames
- ProcessingService: Detección, tracking y conteo
- RenderService: Visualización y UI
- ControlService: Manejo de eventos de teclado
- MonitoringService: Métricas y salud del sistema
"""

from .capture_service import CaptureService
from .processing_service import ProcessingService
from .render_service import RenderService
from .control_service import ControlService
from .monitoring_service import MonitoringService

__all__ = [
    "CaptureService",
    "ProcessingService",
    "RenderService",
    "ControlService",
    "MonitoringService",
]
