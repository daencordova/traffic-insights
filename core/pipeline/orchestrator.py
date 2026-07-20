"""
Orquestador principal del pipeline.
Responsable de coordinar los servicios y gestionar el estado global.
"""

import time
from typing import Optional, Dict, Any

from core.pipeline.services.capture_service import CaptureService
from core.pipeline.services.processing_service import ProcessingService
from core.pipeline.services.render_service import RenderService
from core.pipeline.services.control_service import ControlService
from core.pipeline.services.monitoring_service import MonitoringService
from core.pipeline.state import PipelineState, PipelineStatus
from utils.logger import LoggerMixin


class PipelineOrchestrator(LoggerMixin):
    """
    Orquestador del pipeline.

    Responsabilidades:
    - Inicializar y conectar los servicios
    - Gestionar el ciclo de vida del pipeline
    - Coordinar el flujo de datos entre servicios
    - Gestionar eventos y estados
    - No contiene lógica de negocio específica
    """

    def __init__(
        self,
        config,
        detector=None,
        tracker=None,
        counter=None,
        renderer=None,
        controls=None
    ):
        self.config = config
        self._state = PipelineState()
        self._services = {}
        self._event_handlers = {}
        self._is_running = False

        self.logger.info("Inicializando PipelineOrchestrator")

        self._init_services(detector, tracker, counter, renderer, controls)
        self._setup_event_handlers()

        self.logger.info("PipelineOrchestrator inicializado")

    def _init_services(self, detector, tracker, counter, renderer, controls):
        """Inicializa todos los servicios."""
        self._services['capture'] = CaptureService(
            config=self.config,
            on_frame_captured=self._on_frame_captured,
            on_frame_dropped=self._on_frame_dropped
        )

        self._services['processing'] = ProcessingService(
            config=self.config,
            detector=detector,
            tracker=tracker,
            counter=counter,
            on_frame_processed=self._on_frame_processed
        )

        self._services['render'] = RenderService(
            config=self.config,
            renderer=renderer,
            on_key_pressed=self._on_key_pressed
        )

        self._services['control'] = ControlService(
            config=self.config,
            controls=controls
        )

        self._services['monitoring'] = MonitoringService(
            config=self.config,
            interval=5.0
        )

    def _setup_event_handlers(self):
        """Configura los manejadores de eventos entre servicios."""
        self._event_handlers['frame_captured'] = self._services['processing'].enqueue_frame
        self._event_handlers['frame_processed'] = self._services['render'].enqueue_frame
        self._event_handlers['key_pressed'] = self._services['control'].handle_key

    def start(self, source: Optional[str] = None) -> None:
        """
        Inicia el pipeline orquestando todos los servicios.

        Args:
            source: Fuente de video (opcional)
        """
        if self._is_running:
            self.logger.warning("Pipeline ya está en ejecución")
            return

        self.logger.info("Iniciando pipeline...")
        self._state.set_status(PipelineStatus.RUNNING)
        self._is_running = True

        try:
            self._services['capture'].start(source)
            self._services['processing'].start()
            self._services['render'].start()
            self._services['monitoring'].start()

            self.logger.info("Pipeline iniciado exitosamente")
            self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Error iniciando pipeline: {e}")
            self.stop()
            raise

    def _run_main_loop(self) -> None:
        """
        Bucle principal de monitoreo.
        No bloquea, solo verifica el estado y aplica control de flujo.
        """
        while self._is_running:
            try:
                if self._state.get_status() == PipelineStatus.ERROR:
                    self._handle_error()

                self._apply_flow_control()

                self._services['monitoring'].update()

                time.sleep(0.01)

            except KeyboardInterrupt:
                self.logger.info("Interrupción recibida")
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Error en bucle principal: {e}")
                self._state.set_status(PipelineStatus.ERROR)

    def _apply_flow_control(self) -> None:
        """Aplica control de flujo basado en el estado del sistema."""
        if self._state.is_paused():
            self._services['capture'].pause()
            self._services['processing'].pause()
            self._services['render'].pause()
        else:
            self._services['capture'].resume()
            self._services['processing'].resume()
            self._services['render'].resume()

    def _handle_error(self) -> None:
        """Maneja errores del sistema."""
        self.logger.warning("Estado de error detectado, intentando recuperación...")

        if self._state.can_recover():
            self.logger.info("Intentando recuperación automática...")
            self._services['capture'].reconnect()
            self._services['processing'].reset()
            self._state.set_status(PipelineStatus.RUNNING)
        else:
            self.logger.error("Recuperación automática fallida, deteniendo pipeline")
            self.stop()

    def pause(self) -> None:
        """Pausa la ejecución del pipeline."""
        self._state.set_status(PipelineStatus.PAUSED)
        self.logger.info("Pipeline pausado")

    def resume(self) -> None:
        """Reanuda la ejecución del pipeline."""
        self._state.set_status(PipelineStatus.RUNNING)
        self.logger.info("Pipeline reanudado")

    def stop(self) -> None:
        """Detiene la ejecución del pipeline."""
        self.logger.info("Deteniendo pipeline...")
        self._is_running = False
        self._state.set_status(PipelineStatus.STOPPED)

        for service_name in ['monitoring', 'render', 'processing', 'capture']:
            try:
                if service_name in self._services:
                    self._services[service_name].stop()
            except Exception as e:
                self.logger.warning(f"Error deteniendo {service_name}: {e}")

        self.logger.info("Pipeline detenido")

    def _on_frame_captured(self, frame, metadata):
        """Maneja un frame capturado."""
        if self._state.is_running():
            self._services['processing'].process_frame(frame, metadata)

    def _on_frame_processed(self, result):
        """Maneja un frame procesado."""
        if self._state.is_running():
            self._services['render'].render_frame(result)

    def _on_frame_dropped(self, frame_number):
        """Maneja un frame descartado."""
        self._services['monitoring'].record_dropped_frame()

    def _on_key_pressed(self, key):
        """Maneja una tecla presionada."""
        self._services['control'].handle_key(key)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas combinadas de todos los servicios."""
        stats = {
            'state': self._state.get_status().value,
            'is_paused': self._state.is_paused(),
            'is_running': self._is_running,
            'uptime_seconds': self._state.get_uptime(),
        }

        for name, service in self._services.items():
            try:
                if hasattr(service, 'get_stats'):
                    stats[name] = service.get_stats()
            except Exception:
                pass

        return stats

    @property
    def is_running(self) -> bool:
        return self._is_running and self._state.is_running()

    @property
    def is_paused(self) -> bool:
        return self._state.is_paused()
