"""
Servicios especializados del pipeline.

Cada servicio tiene una responsabilidad única y bien definida siguiendo
el principio de responsabilidad única (SRP):

- CaptureService: Captura de video y gestión de frames
  └── Buffer circular, reconexión automática, control de flujo

- ProcessingService: Detección, tracking y conteo
  └── Procesamiento de frames, batch processing, gestión de resultados

- RenderService: Visualización y UI
  └── Renderizado de overlays, manejo de ventanas, eventos de teclado

- ControlService: Manejo de eventos de usuario
  └── Pausa, captura de pantalla, reinicio, ayuda

- MonitoringService: Métricas y salud del sistema
  └── Recolección de métricas, detección de problemas, alertas
"""

from core.pipeline.services.capture_service import CaptureService
from core.pipeline.services.control_service import ControlService
from core.pipeline.services.monitoring_service import MonitoringService
from core.pipeline.services.processing_service import ProcessingService
from core.pipeline.services.render_service import RenderService

__all__ = [
    "CaptureService",
    "ProcessingService",
    "RenderService",
    "ControlService",
    "MonitoringService",
]
